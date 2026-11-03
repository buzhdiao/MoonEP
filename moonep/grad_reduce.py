"""Remote expert grad reduce kernel.

Accumulates fp32 gradients from per-rank prefetch reduce buffers back into the
owner rank's local expert gradient buffer.

Warp-specialized, tile-partitioned design:
  - 5 warps/CTA: 1 load warp (G2S TMA), 4 fp32 ACC warps (acc + store).
  - The TMA pipeline streams only the remote reduce tiles. The acc warps
    seed their accumulator with the local grad tile through a vectorized
    16B autovec_copy whose latency hides under the first remote-stage wait,
    then store the result back with 16B stores. The seed/store MUST go
    through autovec_copy over views with *static* strides: per-element
    fragment assignments from gmem (or a dynamic-stride view) demote the
    128-reg accumulator to a stack alloca or fall back to scalar LDG/STG.
  - Work is split by 128x128 tile across all SMs over a prescan-compacted
    list of active local experts (experts with no remote slot cost nothing).
    The prescan itself is cooperative (smem stage + cta-atomic count +
    match_any warp fill): a single-thread gmem scan of the R*EPN plan is
    bottlenecked by serial load latency.
  - Accumulation never writes the reduce buffers. After all tiles are summed
    a cross-rank barrier fences all peers, then each rank clears only its own
    consumed slots locally with grid-strided 16B vector stores. Remote-write
    clears (scalar or bulk S2G) were tried and rejected: simultaneous remote
    reads + remote writes share a single NVLink budget per GPU (each
    direction gets roughly half), so riding the "idle" write
    direction stretches phase 1 far more than the local clear costs.
"""

import functools  # 提供 lru_cache：缓存设备属性查询与编译产物

import torch  # PyTorch：张量校验与 CUDA 流获取
import cuda.bindings.driver as cuda  # CUDA driver 绑定：CUstream 句柄类型

import cutlass  # CUTLASS Python DSL：constexpr 常量包装
import cutlass.cute as cute  # CuTe 核心：layout/tensor 抽象与 jit/kernel 装饰器
import cutlass.pipeline as pipeline  # CUTLASS 流水线原语：mbarrier 生产者-消费者同步
import cutlass.utils as utils  # CuTe 工具集：SmemAllocator
import cutlass.cute.nvgpu.cpasync as cpasync  # cp.async / TMA 拷贝原子与分区工具
from cutlass import Float32, Int32, Int64, Uint32  # DSL 数值类型（fp32/int32/int64/uint32）
from cutlass.cute.runtime import make_ptr  # 宿主侧构造裸指针实参

from moonep._common import cross_rank_barrier  # 跨 rank 屏障：phase2 前同步所有 peer
from moonep.planning import match_any_b32, st_global_v4_s32  # warp 内 match_any 原语；128-bit 全局向量零存储


# 冗余专家梯度归约 kernel：把各 peer rank 预取到 reduce buffer 的 fp32 梯度
# 按"归属 rank"归约回本 rank 拥有的专家梯度缓冲（详细设计见类 docstring）
class GradReduceKernel:
    """Persistent fp32 tile reducer for remote expert grads."""

    M_BLOCK = 128  # tile 的行方向（H 维）分块大小
    N_BLOCK = 128  # tile 的列方向（H' 维）分块大小
    STAGES = 3                 # load-pipeline depth
    ACC_THREADS = 128          # warps 1..4
    NUM_THREADS = 160          # 5 warps: 1 load + 4 fp32 acc (acc also stores)

    # 构造函数：保存形状配置，并校验单个 128×128 fp32 累加 tile 能否放进 smem 预算
    def __init__(
        self,
        EPN: int,  # 每 rank 的专家（槽位）数
        H: int,  # 专家梯度第 1 维大小
        Hp: int,  # 专家梯度第 2 维大小（H'，可填充）
        R: int,  # EP 组内 rank 数
        rank_stride: int,  # reduce buffer 中相邻 rank 切片的元素步长（允许填充）
        meta_stride: int,  # meta buffer 中每 rank 元数据块步长
        num_sms: int,  # 常驻 SM 数（持久化网格大小）
        smem_budget: int,  # 每 block 可用共享内存上限（字节）
    ):
        self.EPN = EPN  # 记录每 rank 专家数
        self.H = H  # 记录梯度第 1 维
        self.Hp = Hp  # 记录梯度第 2 维
        self.R = R  # 记录 rank 数
        self.rank_stride = rank_stride  # 记录 rank 切片步长
        self.meta_stride = meta_stride  # 记录 meta 块步长
        self.num_sms = num_sms  # 记录网格大小
        need = self._smem_bytes()  # 计算实际 smem 需求
        if need > smem_budget:  # 超预算则报错
            raise RuntimeError(
                "grad_reduce: not enough per-block shared memory for one "
                f"{self.M_BLOCK}x{self.N_BLOCK} fp32 acc tile: need {need} B, "
                f"budget {smem_budget} B"
            )

    # 计算每 block 所需动态共享内存：TMA 级缓冲 + mbarrier + 预扫描工作区 + 余量
    def _smem_bytes(self) -> int:
        def _round_up(n: int, a: int) -> int:  # 向上取整到 a 的倍数
            return (n + a - 1) // a * a

        tile_bytes = self.M_BLOCK * self.N_BLOCK * 4  # 单个 128×128 fp32 tile 的字节数
        stage = _round_up(self.STAGES * tile_bytes, 128)  # TMA 级缓冲：3 级 fp32 tile，128B 对齐
        mbar = _round_up(self.STAGES * 2 * 8, 16)  # mbarrier：每级 full/empty 各 1 个（8B），16B 对齐
        # off[EPN+1] + alist[EPN] + acnt[1] + slist[R*EPN] + sexp[R*EPN]
        # + cur[EPN+1] + clist[EPN] + ccnt[1]
        scan = _round_up(  # 预扫描工作区（全部 int32，见 kernel 内各分配处注释）
            (4 * self.EPN + 4 + 2 * self.R * self.EPN) * 4, 128)
        return stage + mbar + scan + 256  # 三部分相加再加 256B cutlass 内部余量

    # 宿主侧 JIT 入口：构建各全局内存视图与 TMA 原子，cooperative 发射 kernel
    @cute.jit
    def __call__(
        self,
        expert_grad_ptr: cute.Pointer,    # fp32 [EPN, H, H']
        reduce_buf_ptr: cute.Pointer,     # fp32 [R, EPN, H, H']
        experts_ptr: cute.Pointer,        # int32 [R, EPN]
        meta_ptr: cute.Pointer,           # int32 [R*meta_stride] (barrier)
        bar_ptr: cute.Pointer,            # int32 [1] grid barrier counter
        rank: Int32,  # 当前 rank 编号
        barrier_off: Int32,  # 屏障槽位在 meta 块内的偏移
        stream: cuda.CUstream,  # 目标 CUDA 流
    ):
        EPN = cutlass.const_expr(self.EPN)  # 每 rank 专家数
        H = cutlass.const_expr(self.H)  # 梯度第 1 维
        Hp = cutlass.const_expr(self.Hp)  # 梯度第 2 维
        R = cutlass.const_expr(self.R)  # rank 数
        meta_stride = cutlass.const_expr(self.meta_stride)  # meta 块步长
        M_BLOCK = cutlass.const_expr(self.M_BLOCK)  # tile 行块大小
        N_BLOCK = cutlass.const_expr(self.N_BLOCK)  # tile 列块大小
        Hp64 = cutlass.Int64(Hp)  # 行距的 int64 形式，防止大张量行偏移溢出

        expert_grad = cute.make_tensor(  # 本地专家梯度视图：[EPN*H, Hp] 行主序（把前两维展平）
            expert_grad_ptr,
            cute.make_layout((EPN * H, Hp), stride=(Hp64, cutlass.Int64(1))),
        )
        reduce_buf = cute.make_tensor(  # reduce buffer 视图：[EPN*H, Hp, R]，第 3 维以 rank_stride 为步长
            reduce_buf_ptr,
            cute.make_layout(
                (EPN * H, Hp, R),
                stride=(Hp64, cutlass.Int64(1), cutlass.Int64(self.rank_stride)),
            ),
        )
        experts = cute.make_tensor(experts_ptr, cute.make_layout((R * EPN,)))  # 专家归属表：长度 R*EPN 的一维 int32
        meta = cute.make_tensor(meta_ptr, cute.make_layout((R * meta_stride,)))  # meta 缓冲：跨 rank 屏障槽位所在
        bar = cute.make_tensor(bar_ptr, cute.make_layout((1,)))  # 网格屏障计数器

        # 2D TMA atom for the remote reduce tiles; the local grad tile is
        # loaded by the acc warps with vectorized 16B reads.
        smem_tile_layout = cute.make_ordered_layout((M_BLOCK, N_BLOCK), order=(1, 0))  # smem tile 布局：N 最快（行内连续）
        cta_tiler = (M_BLOCK, N_BLOCK)  # 每 CTA 处理的 tile 形状
        tma_g2s_red, tma_red = cpasync.make_tiled_tma_atom(  # 构造针对 reduce_buf 的 2D TMA 原子及其分区视图
            cpasync.CopyBulkTensorTileG2SOp(),
            reduce_buf,
            smem_tile_layout,
            cta_tiler,
        )

        self.kernel(expert_grad, reduce_buf,  # 以实际视图/原子为实参实例化并发射 kernel
                    tma_g2s_red, tma_red, experts, meta, bar,
                    rank, barrier_off).launch(
            grid=(self.num_sms, 1, 1),  # 网格 = 常驻 SM 数（持久化 CTA）
            block=(self.NUM_THREADS, 1, 1),  # 每 block 5 个 warp
            smem=self._smem_bytes(),  # 动态共享内存字节数
            stream=stream,  # 目标 CUDA 流
            cooperative=True,  # cooperative 启动（跨 rank 屏障需要全体 block 常驻）
        )

    # 设备侧 kernel 主体：先协作预扫描专家归属表，phase1 按 tile 归约远端梯度，
    # phase2 跨 rank 屏障后本地清零自己消费过的槽位
    @cute.kernel
    def kernel(
        self,
        expert_grad: cute.Tensor,  # 本地专家梯度视图
        reduce_buf: cute.Tensor,  # reduce buffer 视图
        tma_g2s_red: cute.CopyAtom,  # G2S TMA 拷贝原子
        tma_red: cute.Tensor,  # TMA 分区后的全局 tile 视图
        experts: cute.Tensor,  # 专家归属表 [R*EPN]
        meta: cute.Tensor,  # meta 缓冲
        bar: cute.Tensor,  # 网格屏障计数器
        rank: Int32,  # 当前 rank
        barrier_off: Int32,  # 屏障槽位偏移
    ):
        EPN = cutlass.const_expr(self.EPN)  # 每 rank 专家数
        H = cutlass.const_expr(self.H)  # 梯度第 1 维
        Hp = cutlass.const_expr(self.Hp)  # 梯度第 2 维
        R = cutlass.const_expr(self.R)  # rank 数
        M_BLOCK = cutlass.const_expr(self.M_BLOCK)  # tile 行块大小
        N_BLOCK = cutlass.const_expr(self.N_BLOCK)  # tile 列块大小
        meta_stride = cutlass.const_expr(self.meta_stride)  # meta 块步长
        num_ranks = cutlass.const_expr(self.R)  # rank 数
        TILE_ELEMS = cutlass.const_expr(M_BLOCK * N_BLOCK)  # 单 tile 元素数
        MTILES = cutlass.const_expr(H // M_BLOCK)  # H 维 tile 数
        NTILES = cutlass.const_expr(Hp // N_BLOCK)  # H' 维 tile 数
        TILES_PER_EXPERT = cutlass.const_expr(MTILES * NTILES)  # 每个专家的 tile 总数

        STAGES = cutlass.const_expr(self.STAGES)  # 装载流水线级数
        ACC_THREADS = cutlass.const_expr(self.ACC_THREADS)  # 累加线程数 128
        num_threads = cutlass.const_expr(self.NUM_THREADS)  # 每 CTA 线程数 160
        TILE_BYTES = cutlass.const_expr(TILE_ELEMS * 4)  # 单 tile 字节数（fp32）

        tidx, _, _ = cute.arch.thread_idx()  # 线程编号
        bidx, _, _ = cute.arch.block_idx()  # block 编号
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 统一化 warp 编号

        smem = utils.SmemAllocator()  # 动态共享内存分配器
        load_mbar = smem.allocate_array(Int64, num_elems=2 * STAGES)  # TMA 装载流水线 mbarrier：每级 2 个
        stage_smem = smem.allocate_tensor(  # TMA 级缓冲：[M_BLOCK, N_BLOCK, STAGES]，N 最快 → 每行连续
            Float32,
            cute.make_ordered_layout((M_BLOCK, N_BLOCK, STAGES), order=(1, 0, 2)),
            byte_alignment=128,  # 128B 对齐满足 TMA 要求
        )
        # flat per-stage view of the same smem for the acc warps.
        stage_flat = cute.make_tensor(  # 同一缓冲的一维扁平视图：[TILE_ELEMS, STAGES]，便于累加寻址
            stage_smem.iterator,
            cute.make_ordered_layout((TILE_ELEMS, STAGES), order=(0, 1)),
        )
        # prescan: cnt[le] and a per-expert-grouped slot list. cnt[EPN] holds
        # running prefix; slist[off[le] .. off[le]+cnt[le]) are the matching rb.
        off = smem.allocate_tensor(Int32, cute.make_layout((EPN + 1,)), byte_alignment=16)  # 每本地专家的槽位计数 → 前缀和偏移表
        # compacted list of active local experts (off[le+1] > off[le]).
        alist = smem.allocate_tensor(Int32, cute.make_layout((EPN,)), byte_alignment=16)  # 紧凑的"有远端槽位"本地专家列表
        acnt = smem.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=16)  # 活跃本地专家数
        slist = smem.allocate_tensor(Int32, cute.make_layout((R * EPN,)), byte_alignment=16)  # 按专家分组的槽位（rb 下标）列表
        # smem stage of experts[] (pre-biased to local ids) for the prescan.
        sexp = smem.allocate_tensor(Int32, cute.make_layout((R * EPN,)), byte_alignment=16)  # experts 表的 smem 暂存（预减 rank*EPN 得本地 id）
        # slist fill cursors; cur[EPN] is a dummy bucket for invalid lanes.
        cur = smem.allocate_tensor(Int32, cute.make_layout((EPN + 1,)), byte_alignment=16)  # slist 各桶的填充游标（EPN 号为无效 lane 的哑桶）
        # this rank's own consumed slots (experts[rank,b] >= 0) for clearing.
        clist = smem.allocate_tensor(Int32, cute.make_layout((EPN,)), byte_alignment=16)  # 本 rank 消费过的槽位列表（phase2 清零用）
        ccnt = smem.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=16)  # 本 rank 消费槽位数

        rank_epn = rank * EPN  # 当前 rank 在全局专家 id 空间的基址

        # ---- prescan experts once (parallel): bucket slots per local expert.
        # A single-thread gmem scan of all R*EPN entries is bottlenecked by a
        # serial load latency chain, so it must be cooperative: stage the
        # table to smem with all threads, count with cta atomics, then have
        # warp 0 fill slist chunk-by-chunk with match_any so the final order
        # stays rb-ascending (bitwise-identical to the serial scan).
        for i in cutlass.range(tidx, R * EPN, self.NUM_THREADS, unroll=1):  # 全线程并行把归属表搬进 smem
            sexp[i] = experts[i] - rank_epn  # 预减 rank*EPN：>=0 且 <EPN 即归本 rank 所有
        for i in cutlass.range(tidx, EPN + 1, self.NUM_THREADS, unroll=1):  # 清零偏移计数表
            off[i] = Int32(0)
        if tidx == 0:  # 仅线程 0 初始化清零计数
            ccnt[0] = Int32(0)
        cute.arch.sync_threads()  # 等待上述初始化完成
        for i in cutlass.range(tidx, R * EPN, self.NUM_THREADS, unroll=1):  # 统计每个本地专家的远端槽位数
            e = sexp[i]  # 该槽位归属的本地专家 id
            if e >= Int32(0) and e < Int32(EPN):  # 仅统计归本 rank 的槽位
                cute.arch.atomic_add(off.iterator + (e + Int32(1)), Int32(1),  # CTA 级原子计数（+1 偏移为前缀和留位）
                                     scope="cta")
        # this rank's own consumed slots (experts[rank,b] >= 0), unordered.
        for b in cutlass.range(tidx, EPN, self.NUM_THREADS, unroll=1):  # 记录本 rank 自己消费过的槽位 b
            if sexp[rank * EPN + b] >= Int32(0) - rank_epn:  # 原表 experts[rank,b] >= 0（非 0xFFFFFFFF 哨兵）
                pos = cute.arch.atomic_add(ccnt.iterator, Int32(1), scope="cta")  # 原子领取写入位置
                clist[pos] = Int32(b)  # 记录槽位号（顺序无保证）
        cute.arch.sync_threads()  # 等待计数/收集完成
        # serial prefix + active-expert compaction over EPN (smem-only).
        if tidx == 0:  # 单线程串行做前缀和与活跃专家压缩（仅 smem，代价极小）
            run = Int32(0)  # 前缀累加器
            ac = Int32(0)  # 活跃专家计数
            for le in cutlass.range(EPN, unroll=1):  # 逐本地专家
                c = off[le + 1]  # 该专家的远端槽位数
                if c > Int32(0):  # 有远端槽位 → 记入活跃列表
                    alist[ac] = Int32(le)
                    ac = ac + Int32(1)
                run = run + c  # 累计前缀
                off[le + 1] = run  # 写回前缀和（off 变为互斥前缀和）
            acnt[0] = ac  # 保存活跃专家数
        cute.arch.sync_threads()  # 等待前缀和完成
        for i in cutlass.range(tidx, EPN + 1, self.NUM_THREADS, unroll=1):  # 把前缀和复制为 slist 填充游标
            cur[i] = off[i]
        cute.arch.sync_threads()  # 等待游标就绪
        if warp_idx == 0:  # warp0 用 match_any 逐块填充 slist（保持 rb 升序）
            lane = cute.arch.lane_idx()  # lane 号
            lanes_lt = (Uint32(1) << lane) - Uint32(1)  # 掩码：低于本 lane 的所有位
            NCHUNK = cutlass.const_expr((R * EPN + 31) // 32)  # 按 32 lane 分块数
            for c0 in cutlass.range(NCHUNK, unroll=1):  # 逐块处理
                rb = c0 * 32 + lane  # 本 lane 负责的槽位下标
                e = Int32(EPN)  # 默认归属哑桶 EPN（无效 lane 用）
                if rb < Int32(R * EPN):  # 尾块越界保护
                    ee = sexp[rb]  # 读该槽位归属的本地专家
                    if ee >= Int32(0) and ee < Int32(EPN):  # 仅归本 rank 的槽位
                        e = ee
                peers = match_any_b32(e)  # warp 内同值掩码：与本 lane 同归属专家的所有 lane
                cell = cur.iterator + e  # 该专家桶的游标单元
                base = cute.arch.load(cell, Int32)  # 读当前桶基址
                pos = base + Int32(cute.arch.popc(peers & lanes_lt))  # 本 lane 在桶内的写入位（按低于自己的同组 lane 数排位）
                cute.arch.sync_warp()  # 确保所有 lane 都读完游标后再更新
                if (peers & lanes_lt) == Uint32(0):  # 每桶编号最小的 lane 负责推进游标
                    cute.arch.store(cell, base + Int32(cute.arch.popc(peers)))
                cute.arch.sync_warp()  # 游标更新对下一块可见
                if e < Int32(EPN):  # 有效槽位才写入 slist
                    slist[pos] = Int32(rb)  # 记录槽位下标（同块内保持 rb 升序）
        cute.arch.sync_threads()  # 预扫描全部完成

        cta_tiler = (M_BLOCK, N_BLOCK)  # tile 形状
        smem_for_tma = cute.group_modes(stage_smem, 0, 2)  # 把 smem tile 的前两维折叠为一个模式供 TMA 分区
        red_tiles = cute.zipped_divide(tma_red, cta_tiler)  # 把 reduce_buf 全局视图按 tile 形状划分
        tRsR, tRgR = cpasync.tma_partition(  # TMA 分区：tRsR 为 smem 侧切片，tRgR 为 gmem 侧 tile 切片
            tma_g2s_red, 0, cute.make_layout(1), smem_for_tma, red_tiles,
        )
        load_pipe = pipeline.PipelineTmaAsync.create(  # TMA 装载流水线：warp0 生产、4 个 ACC warp 消费
            barrier_storage=load_mbar, num_stages=STAGES,  # mbarrier 存储与级数
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),  # 生产者单线程计数
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),  # 消费者按 4 计数
            tx_count=TILE_BYTES,  # 每级期望事务字节数 = 一个 fp32 tile
        )
        ld_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)  # 装载生产者状态（warp0）
        cd_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)  # 装载消费者状态（ACC warps）
        acc_tid = tidx - 32  # ACC 组内线程号（0..127）
        H_PER = cutlass.const_expr(TILE_ELEMS // ACC_THREADS)  # 每线程持有的 fp32 寄存器数
        acc_reg = cute.make_rmem_tensor((H_PER,), Float32)  # 寄存器累加器

        # ============ phase 1: warp-specialized accumulate, no clears ========
        # phase1：按 tile 在所有 SM 间分摊归约工作（不写 reduce buffer）
        total_work = acnt[0] * TILES_PER_EXPERT  # 总工作量 = 活跃专家数 × 每专家 tile 数
        for work_idx in cutlass.range(bidx, total_work, self.num_sms, unroll=1):  # 网格跨步分派 work 项
            local_expert = alist[work_idx // TILES_PER_EXPERT]  # 该 work 项对应的活跃本地专家
            rem = work_idx % TILES_PER_EXPERT  # 专家内的 tile 序号
            mt = rem // NTILES  # tile 的行块号
            nt = rem % NTILES  # tile 的列块号
            base_row = mt * M_BLOCK  # tile 起始行（H 维）
            base_col = nt * N_BLOCK  # tile 起始列（H' 维）
            beg = off[local_expert]  # 该专家的槽位列表起始偏移
            nslot = off[local_expert + 1] - beg  # 该专家的远端槽位数
            # load warp (0): stream the nslot remote reduce tiles.
            if warp_idx == 0:  # warp0：把该 tile 的全部远端槽位 tile 流式装载进流水线
                for s in cutlass.range(nslot, unroll=1):  # 逐槽位
                    load_pipe.producer_acquire(ld_state)  # 等待当前级空闲
                    rb = slist[beg + s]  # 槽位在 experts/reduce_buf 中的全局下标
                    src_rank = rb // EPN  # 源 rank 编号
                    src_mt = (rb % EPN) * MTILES + mt  # 源槽位内的行块号（槽位行基址 + 本 tile 行偏移）
                    cute.copy(  # 用 TMA 原子拷贝一个 (mt, nt, src_rank) tile 到 smem 当前级
                        tma_g2s_red,
                        tRgR[(None, (src_mt, nt, src_rank))],
                        tRsR[(None, ld_state.index)],
                        tma_bar_ptr=load_pipe.producer_get_barrier(ld_state),
                    )
                    ld_state.advance()  # 推进生产者状态
            # acc warps (1..4): seed the accumulator with the local grad tile
            # (16B vector loads, hidden under the first remote-stage wait),
            # sum the remote stages, then write back with 16B stores. Each
            # thread owns groups of 4 consecutive elements:
            #   e = (j//4)*4*ACC_THREADS + acc_tid*4 + j%4.
            elif warp_idx >= 1:  # warps1..4：以本地梯度 tile 播种累加器，叠加远端 tile 后写回
                grad0 = local_expert * H + base_row  # 本 tile 在 expert_grad 中的起始行号
                row_off = acc_tid // Int32(32)  # 本线程在 tile 内的行偏移（0..3）
                col0 = (acc_tid % Int32(32)) * Int32(4)  # 本线程负责的列起点（每线程 4 个连续列）
                base_t = row_off * N_BLOCK + col0  # 本线程在扁平 tile 内的元素基偏移
                # per-thread (32 groups x 4 consecutive elems) view of the
                # grad tile. The strides must stay static and the offset 16B
                # aligned for autovec_copy to emit 128-bit accesses. The row
                # offset must be widened to i64 BEFORE multiplying by Hp so
                # large local gradient layouts cannot overflow i32 arithmetic.
                goff = (cutlass.Int64(grad0 + row_off) * cutlass.Int64(Hp)  # 行偏移先扩为 i64 再乘 Hp，防 i32 溢出
                        + cutlass.Int64(base_col + col0))
                gview = cute.make_tensor(  # 本线程的梯度切片视图：[H_PER/4, 4]，每组 4 个连续元素
                    expert_grad.iterator + cute.assume(goff, divby=4),  # 声明 16B 对齐假设（divby=4 个 fp32）
                    cute.make_layout(
                        (H_PER // 4, 4),
                        stride=(4 * self.Hp, 1),  # 组间跨一行中的 4 列（相邻组在 tile 内下移一行）
                    ),
                )
                acc_view = cute.make_tensor(  # 累加寄存器的匹配视图：[H_PER/4, 4] 连续布局
                    acc_reg.iterator,
                    cute.make_layout((H_PER // 4, 4), stride=(4, 1)),
                )
                cute.autovec_copy(gview, acc_view)  # 向量化播种：本地梯度 tile → fp32 寄存器（16B 访问）
                for _ in cutlass.range(nslot, unroll=1):  # 逐远端槽位累加
                    load_pipe.consumer_wait(cd_state)  # 等待当前级 TMA tile 到齐
                    for j in cutlass.range_constexpr(H_PER):  # 遍历本线程寄存器
                        acc_reg[j] = acc_reg[j] + stage_flat[  # 从扁平 smem 视图读对应元素并 FADD
                            (j // 4) * (self.ACC_THREADS * 4) + (j % 4) + base_t,
                            cd_state.index,
                        ]
                    # Stage may be overwritten immediately after release.
                    # consumer_release signals from each warp's lane 0 only
                    # (PipelineTmaAsync signalling thread; consumer_group=4),
                    # so sync_warp makes that arrive cover all 32 lanes' reads;
                    # cross-warp completion is gated by the 4-count empty mbarrier.
                    cute.arch.fence_view_async_shared()  # 使本线程的共享内存读可见性有序
                    cute.arch.sync_warp()  # warp 内同步：本 warp 32 线程读完该级
                    load_pipe.consumer_release(cd_state)  # 释放该级供生产者覆盖
                    cd_state.advance()  # 推进消费者状态
                cute.autovec_copy(acc_view, gview)  # 向量化写回：累加结果 → 本地专家梯度（16B 存储）

        # ============ phase 2: cross-rank barrier then local clear ============
        # phase2：跨 rank 屏障确保所有 peer 完成读取后，本地清零自己消费过的槽位
        cross_rank_barrier(  # 全体 rank 的 combine/reduce 都越过此屏障后才清零
            meta, meta_stride, barrier_off, rank, num_ranks,
            bar.iterator, Int32(self.num_sms), num_threads, tidx,
        )
        # Each rank zeroes only its own consumed slots (experts[rank,b] >= 0).
        # A slot is one contiguous [H*Hp] fp32 region; clear it with
        # grid-strided 16B vector stores from every thread. This is bound by
        # the SM<->L2 write throughput, which vector stores already saturate
        # (scalar 4B stores fall short, and wider machinery such as TMA bulk
        # S2G zeros buys nothing).
        cons = ccnt[0]  # 本 rank 消费过的槽位数
        VECS = cutlass.const_expr(H * Hp // 4)  # 每槽位的 128-bit 向量数（4 个 fp32/向量）
        for k in cutlass.range(cons, unroll=1):  # 逐本 rank 消费的槽位
            sbase = (cutlass.Int64(rank) * cutlass.Int64(self.rank_stride)  # 槽位在 reduce_buf 中的元素基址（含 rank 切片步长）
                     + cutlass.Int64(clist[k]) * cutlass.Int64(H) * cutlass.Int64(Hp))
            for v in cutlass.range(bidx * self.NUM_THREADS + tidx, VECS,  # 网格跨步遍历全部向量
                                   self.num_sms * self.NUM_THREADS, unroll=1):
                addr = (reduce_buf.iterator  # 第 v 个向量的字节地址
                        + (sbase + cutlass.Int64(v) * 4)).toint()
                st_global_v4_s32(addr.ir_value(),  # 单条 128-bit 存储写 4 个 fp32 零
                                 Int32(0), Int32(0), Int32(0), Int32(0))


# 查询设备每 block 可选入的最大动态共享内存，按设备号缓存
@functools.lru_cache(maxsize=None)
def _max_smem_per_block_optin(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


# 按 kernel 形状参数缓存 cute.compile 的编译结果（同参数只编译一次）
@functools.lru_cache(maxsize=None)
def _get_compiled(
    epn: int,  # 每 rank 专家数
    H: int,  # 梯度第 1 维
    Hp: int,  # 梯度第 2 维
    R: int,  # rank 数
    rank_stride: int,  # rank 切片步长
    meta_stride: int,  # meta 块步长
    num_sms: int,  # 网格大小
    device_index: int,  # 目标 GPU 编号
):
    smem_budget = _max_smem_per_block_optin(device_index) - 1024  # 预留 1KB 余量
    kernel = GradReduceKernel(  # 用真实形状参数构造 kernel 对象
        EPN=epn,
        H=H,
        Hp=Hp,
        R=R,
        rank_stride=rank_stride,
        meta_stride=meta_stride,
        num_sms=num_sms,
        smem_budget=smem_budget,
    )

    f32_ptr = make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16)  # 代表性 fp32 空指针
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)  # 代表性 int32 空指针
    stream_arg = cuda.CUstream(0)  # 代表性默认流

    return cute.compile(  # 以代表实参完成 DSL 编译
        kernel,
        f32_ptr,
        f32_ptr,
        i32_ptr,
        i32_ptr,
        i32_ptr,
        Int32(0),
        Int32(0),
        stream_arg,
    )


# launch_grad_reduce 的 Python 入口：校验入参形状/设备并绑定真实地址发射 kernel
def launch_grad_reduce(
    local_expert_grads: torch.Tensor,  # [epn, H, H'] 连续 fp32 本地专家梯度
    reduce_buffers: torch.Tensor,  # [R, epn, H, H'] fp32 预取归约缓冲
    experts_to_copy: torch.Tensor,  # [R, epn] int32 专家归属表
    rank: int,  # 当前 EP rank
    num_sms: int,  # 持久化 CTA 数
    meta_buf: torch.Tensor,  # int32 NVL 分布式 meta 缓冲（屏障槽位所在）
    meta_stride: int,  # 每 rank 的 meta 块步长
    barrier_off: int,  # 屏障槽位在块内偏移
    grid_sync_bar: torch.Tensor,  # int32 [1] 网格屏障计数器
) -> None:
    """Launch remote expert grad reduction.

    Args:
        local_expert_grads: contiguous fp32 tensor shaped [epn, H, H'].
            Global expert ids owned by this rank are mapped to local rows by
            subtracting ``rank * epn``.
        reduce_buffers: fp32 tensor shaped [R, epn, H, H'], contiguous
            within each rank with optional padding between ranks.
            Slots whose ``experts_to_copy[r, b]`` belongs to this rank's owner
            range are accumulated into ``local_expert_grads``; afterwards a
            cross-rank barrier fences peers and each rank clears its own
            consumed slots locally.
        experts_to_copy: contiguous int32 tensor shaped [R, epn].
        rank: current EP rank.
        num_sms: number of persistent CTAs to launch.
        meta_buf: int32 NVL-distributed meta buffer holding the barrier slots.
        meta_stride: per-rank stride (meta_chunk_padded) into ``meta_buf``.
        barrier_off: offset of the barrier slots within each rank's chunk.
        grid_sync_bar: int32 [1] grid-barrier counter.

    The caller must ensure all ranks' writes to ``reduce_buffers`` are
    visible to phase-1 TMA reads and prior launches' clears have completed
    before reusing the buffers. All ranks must launch collectively (or all
    skip via the empty-input early return).
    """
    if reduce_buffers.numel() == 0 or experts_to_copy.numel() == 0:  # 空输入提前返回（须全体 rank 一致跳过）
        return

    assert local_expert_grads.dtype == torch.float32 and local_expert_grads.is_contiguous(), \
        "local_expert_grads must be contiguous fp32 [epn, H, H']"  # 本地梯度必须是连续 fp32
    assert reduce_buffers.dtype == torch.float32, \
        "reduce_buffers must be fp32 [R, epn, H, H']"  # 归约缓冲必须是 fp32
    assert experts_to_copy.dtype == torch.int32 and experts_to_copy.is_contiguous(), \
        "experts_to_copy must be contiguous int32 [R, epn]"  # 归属表必须是连续 int32
    assert local_expert_grads.ndim == 3, \
        f"local_expert_grads must have rank 3, got shape={tuple(local_expert_grads.shape)}"  # 维度数校验
    assert reduce_buffers.ndim == 4, \
        f"reduce_buffers must have rank 4, got shape={tuple(reduce_buffers.shape)}"  # 维度数校验
    assert experts_to_copy.ndim == 2, \
        f"experts_to_copy must have rank 2, got shape={tuple(experts_to_copy.shape)}"  # 维度数校验

    rank_stride = int(reduce_buffers.stride(0))  # 取 rank 切片步长（允许 rank 间填充）
    epn, H, Hp = (int(x) for x in local_expert_grads.shape)  # 解出 epn/H/Hp
    R, _, buf_H, buf_Hp = (int(x) for x in reduce_buffers.shape)  # 解出 R 与缓冲的 H/Hp
    assert tuple(reduce_buffers.shape[:2]) == (R, epn), \
        f"reduce_buffers must have leading shape [R, epn]=[{R}, {epn}]"  # 前两维必须匹配
    assert tuple(experts_to_copy.shape) == (R, epn), \
        f"experts_to_copy must be [R, epn]=[{R}, {epn}]"  # 归属表形状必须匹配
    assert buf_H == H and buf_Hp == Hp, (
        f"reduce_buffers shape {tuple(reduce_buffers.shape)} "
        f"incompatible with local_expert_grads {tuple(local_expert_grads.shape)}"
    )  # 缓冲的 H/Hp 必须与本地梯度一致
    assert 0 <= int(rank) < R, f"rank must be in [0, {R}), got {rank}"  # rank 范围校验
    assert H % GradReduceKernel.M_BLOCK == 0 and Hp % GradReduceKernel.N_BLOCK == 0, \
        f"H and H' must be multiples of ({GradReduceKernel.M_BLOCK}, {GradReduceKernel.N_BLOCK}), got ({H}, {Hp})"  # H/H' 须为 tile 大小倍数
    assert isinstance(num_sms, int) and num_sms > 0, \
        f"num_sms must be a positive int, got {num_sms}"  # num_sms 必须为正整数

    device_index = local_expert_grads.device.index  # 目标 GPU 编号
    assert device_index is not None, "local_expert_grads must be a CUDA tensor"  # 必须是 CUDA 张量
    assert reduce_buffers.device.index == device_index, \
        "reduce_buffers must be on the same CUDA device as local_expert_grads"  # 归约缓冲须同设备
    assert experts_to_copy.device.index == device_index, \
        "experts_to_copy must be on the same CUDA device as local_expert_grads"  # 归属表须同设备

    compiled = _get_compiled(epn, H, Hp, R, rank_stride, int(meta_stride), int(num_sms),  # 取（或触发）编译
                             int(device_index))

    grad_ptr = make_ptr(  # 本地专家梯度指针
        Float32,
        local_expert_grads.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    reduce_ptr = make_ptr(  # 归约缓冲指针
        Float32,
        reduce_buffers.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    experts_ptr = make_ptr(  # 专家归属表指针
        Int32,
        experts_to_copy.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    meta_ptr = make_ptr(  # meta 缓冲指针
        Int32, meta_buf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16,
    )
    bar_ptr = make_ptr(  # 网格屏障计数器指针
        Int32, grid_sync_bar.data_ptr(), cute.AddressSpace.gmem, assumed_align=16,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)  # 绑定当前 PyTorch CUDA 流

    compiled(grad_ptr, reduce_ptr, experts_ptr, meta_ptr, bar_ptr,  # 调用编译好的 kernel：绑定地址并发射
             Int32(int(rank)), Int32(int(barrier_off)), stream)
