"""
Remote expert prefetch kernel.

Every rank pushes its own local expert weights into the prefetch slots of the
peer ranks that need them. The three weight projections and, when present,
their three scale tensors share one persistent kernel launch, one compacted
owner-slot table, and one trailing cross-rank completion barrier.

Copy engine: persistent, warp-specialized 2D TMA pipeline:

  - warp 0: GMEM -> SMEM 2D TMA load
  - warps 1..6: two SMEM -> GMEM 2D TMA store warps per projection

Each tensor keeps its own TMA descriptor and tile count. A compile-time tensor
loop selects the descriptor while reusing the same producer/consumer loop
body. Tiles default to 128 x 128 elements and widen to 128 x 256 for unscaled
weights when every trailing extent permits it. Scale tensors are re-tiled to
``[128, nbytes // 128]`` per expert before entering the kernel.
"""

import functools  # 标准库：lru_cache 缓存编译产物

import torch  # PyTorch：张量校验、重排视图与 CUDA 流
import cuda.bindings.driver as cuda  # CUDA driver 绑定：CUstream 类型

import cutlass  # CUTLASS DSL 根模块（const_expr 等编译期工具）
import cutlass.cute as cute  # CuTe：布局/张量/内核 DSL
import cutlass.pipeline as pipeline  # CuTe 流水线：TMA mbarrier 同步
import cutlass.utils as utils  # CuTe 工具：SmemAllocator
import cutlass.cute.nvgpu.cpasync as cpasync  # cpasync：2D TMA tiled copy 原子与分区
from cutlass import BFloat16, Int8, Int32, Int64, Uint8, Uint32  # DSL 标量类型
from cutlass.cute.runtime import make_ptr  # 由裸指针构造 DSL Pointer

from moonep._common import cross_rank_barrier  # 自复位跨 rank 屏障（收尾发布）
from moonep.planning import match_any_b32  # PTX match.any.sync：warp 内同值 peer 掩码


# torch dtype -> (DSL 元素类型, 每元素字节数) 的映射表。
_ELEM_TYPES = {
    torch.bfloat16: (BFloat16, 2),  # bf16：2 字节
    torch.int8: (Int8, 1),  # int8：1 字节
    torch.uint8: (Uint8, 1),  # uint8：1 字节
}


# 远端专家权重预取内核：persistent、warp 特化、多张量共享一次启动。
class PrefetchKernel:
    """Persistent multi-tensor 2D TMA local-expert prefetch push."""

    NUM_THREADS = 224  # 每 CTA 线程数：1 生产 warp + 6 存储 warp = 7 × 32
    PRODUCER_WARP = 0  # warp 0：G2S 2D TMA 生产者
    FIRST_STORE_WARP = 1  # 存储 warp 起始编号
    STORE_WARPS_PER_PROJECTION = 2  # 每个投影（gate/up/down）分配 2 个存储 warp
    STORE_WARPS = 6  # 存储 warp 总数：3 投影 × 2
    LAST_STORE_WARP = FIRST_STORE_WARP + STORE_WARPS  # 存储 warp 结束编号（不含）
    M_BLOCK = 128  # 2D tile 的行块大小（H 维）
    N_BLOCK = 128  # 2D tile 的列块大小（H' 维），无 scale 时可放宽到 256

    # 构造函数：保存形状/步长参数，选择 N_BLOCK 并挑选流水线级数。
    def __init__(
        self,
        EPN: int,
        R: int,
        tensor_shapes: tuple[tuple[int, int], ...],
        rank_strides: tuple[int, ...],
        has_scales: bool,
        meta_stride: int,
        num_sms: int,
        smem_budget: int,
        elem_ty=BFloat16,
        elem_bytes: int = 2,
    ):
        self.EPN = EPN  # 每 rank 本地专家数
        self.R = R  # EP 组内 rank 数
        self.tensor_shapes = tensor_shapes  # 各张量的 (H, H') 形状（按张量序）
        self.rank_strides = rank_strides  # 各张量在目的池中的 rank 步长（允许 rank 间有 padding）
        self.has_scales = has_scales  # 是否附带 scale 张量
        self.num_tensors = 6 if has_scales else 3  # 张量总数：3 权重（+3 scale）
        self.meta_stride = meta_stride  # meta_buf 每 rank 槽位步长
        self.num_sms = num_sms  # 网格 CTA 数
        self.elem_ty = elem_ty  # 拷贝元素类型
        self.elem_bytes = elem_bytes  # 每元素字节数
        if not has_scales and all(Hp % 256 == 0 for _, Hp in tensor_shapes):  # 无 scale 且所有 H' 都能被 256 整除时
            self.N_BLOCK = 256  # 列块放宽到 256，减少 tile 数
        self.stages = self._pick_stages(smem_budget)  # 按共享内存预算挑流水线级数
        if self.stages == 0:  # 放不进一个 tile 即失败
            raise RuntimeError(  # 快速失败并报告预算
                "prefetch: not enough per-block shared memory for one "
                f"{self.M_BLOCK}x{self.N_BLOCK} x {elem_bytes}B tile under "
                f"budget {smem_budget} B"
            )

    def _smem_bytes(self, stages: int) -> int:  # 计算给定级数下每 CTA 的共享内存用量
        def _round_up(n: int, a: int) -> int:  # 内部工具：向上取整到 a 的倍数
            return (n + a - 1) // a * a  # 标准向上取整公式

        tile_bytes = self.M_BLOCK * self.N_BLOCK * self.elem_bytes  # 单个 2D tile 的字节数
        # sexp[R*EPN] + off[EPN+1] + alist[EPN] + acnt[1]
        # + slist[R*EPN] + cur[EPN+1]
        scan = _round_up(  # 预扫描结构（专家分桶）所需共享内存，128B 对齐
            (2 * self.R * self.EPN + 3 * self.EPN + 3) * 4, 128)
        return (  # 各部分按对齐求和再加余量
            _round_up(stages * tile_bytes, 128)  # stage_smem：stages 级 × 单 tile，128B 对齐
            + _round_up(stages * 2 * 8, 16)  # 每级 full+empty 两个 mbarrier，16B 对齐
            + scan  # 预扫描区（sexp/off/alist/acnt/slist/cur）
            + 256  # cutlass 内部静态共享内存余量
        )

    def _pick_stages(self, smem_budget: int) -> int:  # 在候选级数中选预算内最深流水线
        for stages in (6, 5, 4, 3, 2):  # 从深到浅尝试
            if self._smem_bytes(stages) <= smem_budget:  # 第一个满足预算的级数
                return stages  # 返回选中级数
        return 0  # 全部超预算返回 0

    @cute.jit  # 标记为 CuTe JIT 编译的辅助函数
    def _make_tma_atoms(  # 为单个张量构造 G2S/S2G 2D TMA 原子及源/目的张量
        self,
        local_ptr: cute.Pointer,
        pool_ptr: cute.Pointer,
        H: cutlass.Constexpr[int],
        Hp: cutlass.Constexpr[int],
        rank_stride: cutlass.Constexpr[int],
    ):
        EPN = cutlass.const_expr(self.EPN)  # 编译期常量：每 rank 专家数
        R = cutlass.const_expr(self.R)  # 编译期常量：rank 数
        M_BLOCK = cutlass.const_expr(self.M_BLOCK)  # 编译期常量：tile 行块
        N_BLOCK = cutlass.const_expr(self.N_BLOCK)  # 编译期常量：tile 列块

        Hp64 = cutlass.Int64(Hp)  # H' 维步长的 int64 形式
        gmem_src = cute.make_tensor(  # 源张量：本地权重 [EPN*H, H'] 行主序视图
            local_ptr,
            cute.make_layout(
                (EPN * H, Hp),
                stride=(Hp64, cutlass.Int64(1)),
            ),
        )
        gmem_dst = cute.make_tensor(  # 目的张量：预取池 [EPN*H, H', R]，第 3 维跨 rank 步进
            pool_ptr,
            cute.make_layout(
                (EPN * H, Hp, R),
                stride=(Hp64, cutlass.Int64(1), cutlass.Int64(rank_stride)),
            ),
        )

        smem_tile_layout = cute.make_ordered_layout(  # smem 中 2D tile 的布局（N 连续、M 跨行）
            (M_BLOCK, N_BLOCK),
            order=(1, 0),
        )
        cta_tiler = (M_BLOCK, N_BLOCK)  # CTA 每次拷贝的 tile 形状
        tma_g2s, tma_src = cpasync.make_tiled_tma_atom(  # 构造 G2S 2D TMA 原子及分区后的源张量
            cpasync.CopyBulkTensorTileG2SOp(),
            gmem_src,
            smem_tile_layout,
            cta_tiler,
        )
        tma_s2g, tma_dst = cpasync.make_tiled_tma_atom(  # 构造 S2G 2D TMA 原子及分区后的目的张量
            cpasync.CopyBulkTensorTileS2GOp(),
            gmem_dst,
            smem_tile_layout,
            cta_tiler,
        )
        return tma_g2s, tma_src, tma_s2g, tma_dst  # 返回 (G2S 原子, 源, S2G 原子, 目的)

    @cute.jit  # 标记为 CuTe JIT 编译的 host 侧入口
    def __call__(  # 内核调用入口：为每个张量建 TMA 描述符并启动内核
        self,
        local_ptrs: tuple[cute.Pointer, ...],
        pool_ptrs: tuple[cute.Pointer, ...],
        experts_ptr: cute.Pointer,          # int32 [R, EPN]
        meta_ptr: cute.Pointer,             # int32 [R * meta_stride]
        bar_ptr: cute.Pointer,              # int32 [1]
        rank: Int32,
        barrier_off: Int32,
        stream: cuda.CUstream,
    ):
        tma_g2s_atoms = []  # 各张量的 G2S TMA 原子列表
        tma_sources = []  # 各张量分区后的源张量列表
        tma_s2g_atoms = []  # 各张量的 S2G TMA 原子列表
        tma_destinations = []  # 各张量分区后的目的张量列表
        for tensor_id in cutlass.range_constexpr(self.num_tensors):  # 编译期循环：每个张量独立建描述符
            (
                tma_g2s,
                tma_src,
                tma_s2g,
                tma_dst,
            ) = self._make_tma_atoms(  # 用该张量的指针/形状/步长构造原子
                local_ptrs[tensor_id],
                pool_ptrs[tensor_id],
                self.tensor_shapes[tensor_id][0],
                self.tensor_shapes[tensor_id][1],
                self.rank_strides[tensor_id],
            )
            tma_g2s_atoms.append(tma_g2s)  # 收集 G2S 原子
            tma_sources.append(tma_src)  # 收集源张量
            tma_s2g_atoms.append(tma_s2g)  # 收集 S2G 原子
            tma_destinations.append(tma_dst)  # 收集目的张量

        R = cutlass.const_expr(self.R)  # 编译期常量：rank 数
        EPN = cutlass.const_expr(self.EPN)  # 编译期常量：每 rank 专家数
        meta_stride = cutlass.const_expr(self.meta_stride)  # 编译期常量：meta 步长
        experts = cute.make_tensor(  # 全局专家表一维视图 [R*EPN]
            experts_ptr,
            cute.make_layout((R * EPN,)),
        )
        meta = cute.make_tensor(  # meta 缓冲一维视图（屏障槽所在）
            meta_ptr,
            cute.make_layout((R * meta_stride,)),
        )
        bar = cute.make_tensor(bar_ptr, cute.make_layout((1,)))  # 网格屏障计数器

        smem_bytes = self._smem_bytes(self.stages)  # 最终动态共享内存用量
        self.kernel(  # 以参数实例化设备内核
            tma_g2s_atoms,
            tma_sources,
            tma_s2g_atoms,
            tma_destinations,
            experts,
            meta,
            bar,
            rank,
            barrier_off,
        ).launch(  # 启动内核（persistent 协作网格）
            grid=(self.num_sms, 1, 1),  # 网格 = num_sms 个 CTA
            block=(self.NUM_THREADS, 1, 1),  # 每 CTA 224 线程（7 个专职 warp）
            smem=smem_bytes,  # 动态共享内存字节数
            stream=stream,  # 在调用方指定的 CUDA 流上启动
            cooperative=True,  # 协作启动：支持网格级同步
        )

    @cute.kernel  # 标记为 CuTe 设备内核
    def kernel(  # 设备侧主内核：预扫描专家表 + 2D TMA 生产/消费推送
        self,
        tma_g2s_atoms: list[cute.CopyAtom],
        tma_sources: list[cute.Tensor],
        tma_s2g_atoms: list[cute.CopyAtom],
        tma_destinations: list[cute.Tensor],
        experts: cute.Tensor,
        meta: cute.Tensor,
        bar: cute.Tensor,
        rank: Int32,
        barrier_off: Int32,
    ):
        EPN = cutlass.const_expr(self.EPN)  # 编译期常量：每 rank 专家数
        R = cutlass.const_expr(self.R)  # 编译期常量：rank 数
        stages = cutlass.const_expr(self.stages)  # 编译期常量：流水线级数
        num_sms = cutlass.const_expr(self.num_sms)  # 编译期常量：CTA 数
        NUM_THREADS = cutlass.const_expr(self.NUM_THREADS)  # 编译期常量：每 CTA 线程数
        meta_stride = cutlass.const_expr(self.meta_stride)  # 编译期常量：meta 步长
        M_BLOCK = cutlass.const_expr(self.M_BLOCK)  # 编译期常量：tile 行块
        N_BLOCK = cutlass.const_expr(self.N_BLOCK)  # 编译期常量：tile 列块
        TILE_ELEMS = cutlass.const_expr(M_BLOCK * N_BLOCK)  # 单 tile 元素数
        TILE_BYTES = cutlass.const_expr(TILE_ELEMS * self.elem_bytes)  # 单 tile 字节数

        bidx, _, _ = cute.arch.block_idx()  # 当前 CTA 的网格编号
        tidx, _, _ = cute.arch.thread_idx()  # CTA 内线程编号（预扫描协作用）
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 归一化 warp 编号

        smem = utils.SmemAllocator()  # 动态共享内存分配器
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages)  # 每级 full+empty 两个 mbarrier 存储
        sexp = smem.allocate_tensor(Int32, cute.make_layout((R * EPN,)), byte_alignment=4)  # 专家表减去本 rank 基址后的相对槽
        off = smem.allocate_tensor(Int32, cute.make_layout((EPN + 1,)), byte_alignment=4)  # 每本地专家槽位区间的排他前缀和
        alist = smem.allocate_tensor(Int32, cute.make_layout((EPN,)), byte_alignment=4)  # 有槽位的活跃本地专家列表
        acnt = smem.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=4)  # 活跃专家数
        slist = smem.allocate_tensor(Int32, cute.make_layout((R * EPN,)), byte_alignment=4)  # 槽位按所属专家分桶的 rb 下标表
        cur = smem.allocate_tensor(Int32, cute.make_layout((EPN + 1,)), byte_alignment=4)  # 分桶写入游标（off 的副本）
        stage_smem = smem.allocate_tensor(  # 多级 2D tile 缓冲 [M, N, stages]，128B 对齐
            self.elem_ty,
            cute.make_ordered_layout(
                (M_BLOCK, N_BLOCK, stages),
                order=(1, 0, 2),
            ),
            byte_alignment=128,
        )

        cta_tiler = (M_BLOCK, N_BLOCK)  # tile 形状
        smem_for_tma = cute.group_modes(stage_smem, 0, 2)  # 把 tile 的前两维折叠为 TMA 的 smem 盒
        tSsS_list = []  # 各张量的 smem 侧 tile 视图（S2G 源）
        tSgS_list = []  # 各张量的 gmem 源 tile 视图（G2S 源）
        tDsD_list = []  # 各张量的 smem 侧 tile 视图（S2G 目）
        tDgD_list = []  # 各张量的 gmem 目的 tile 视图（S2G 目标）
        for tensor_id in cutlass.range_constexpr(self.num_tensors):  # 编译期循环：逐张量做 TMA 分区
            src_tiles = cute.zipped_divide(tma_sources[tensor_id], cta_tiler)  # 源张量按 tile 划分（余维打包）
            dst_tiles = cute.zipped_divide(tma_destinations[tensor_id], cta_tiler)  # 目的张量按 tile 划分
            tSsS, tSgS = cpasync.tma_partition(  # 分区出 smem 盒视图与 gmem 源视图
                tma_g2s_atoms[tensor_id],
                0,
                cute.make_layout(1),
                smem_for_tma,
                src_tiles,
            )
            tDsD, tDgD = cpasync.tma_partition(  # 分区出 smem 盒视图与 gmem 目的视图
                tma_s2g_atoms[tensor_id],
                0,
                cute.make_layout(1),
                smem_for_tma,
                dst_tiles,
            )
            tSsS_list.append(tSsS)  # 收集 smem 盒视图
            tSgS_list.append(tSgS)  # 收集 gmem 源视图
            tDsD_list.append(tDsD)  # 收集 S2G smem 侧
            tDgD_list.append(tDgD)  # 收集 S2G gmem 侧

        # All six store warps share one stage ring. The empty barrier expects
        # one arrival from each store warp, so the producer cannot overwrite a
        # stage until every consumer has observed and released that generation.
        # Keeping one ring preserves the full stage budget across the
        # tensor-major gate/up/down loop, while projection-dedicated store-warp
        # queues let remote write tails overlap across tensor boundaries.
        load_pipe = pipeline.PipelineTmaAsync.create(  # 创建共享环形 TMA 流水线（6 个消费者 warp）
            barrier_storage=load_mbar,
            num_stages=stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),  # 生产者组：1 个有效线程
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.STORE_WARPS),  # 消费者组：6 个存储 warp
            tx_count=TILE_BYTES,  # 每级期望传输字节数 = 单 tile
        )

        rank_epn = rank * EPN  # 本 rank 在专家表一维布局中的基下标

        # ---- prescan experts once (parallel): bucket slots per local expert.
        # A single-thread gmem scan of all R*EPN entries is bottlenecked by a
        # serial load latency chain, so it must be cooperative: stage the
        # table to smem with all threads, count with cta atomics, then have
        # warp 0 fill slist chunk-by-chunk with match_any so the final order
        # stays rb-ascending (bitwise-identical to the serial scan).
        for i in cutlass.range(tidx, R * EPN, self.NUM_THREADS, unroll=1):  # 全体线程跨步把专家表搬入 smem
            sexp[i] = experts[i] - rank_epn  # 减去本 rank 基址：>=0 且 <EPN 的槽属于本 rank 所辖专家
        for i in cutlass.range(tidx, EPN + 1, self.NUM_THREADS, unroll=1):  # 清零前缀计数数组
            off[i] = Int32(0)
        cute.arch.sync_threads()  # CTA 内同步：sexp 就绪
        for i in cutlass.range(tidx, R * EPN, self.NUM_THREADS, unroll=1):  # 统计每个本地专家的槽位数
            e = sexp[i]  # 当前槽的相对专家号
            if e >= Int32(0) and e < Int32(EPN):  # 仅本 rank 拥有的槽计入
                cute.arch.atomic_add(off.iterator + (e + Int32(1)), Int32(1), scope="cta")  # CTA 原子加：off[e+1] 计数
        cute.arch.sync_threads()  # 同步：计数完成
        if tidx == Int32(0):  # 单线程串行计算排他前缀和与活跃专家表
            running = Int32(0)  # 累计槽位数
            active_count = Int32(0)  # 活跃专家计数
            for local_expert in cutlass.range(EPN, unroll=1):  # 遍历每个本地专家
                cur_running = off[local_expert + Int32(1)]  # 该专家的槽位数
                if cur_running > Int32(0):  # 有槽位才进入活跃表
                    alist[active_count] = Int32(local_expert)  # 记录活跃专家号
                    active_count += Int32(1)  # 活跃数递增
                running += cur_running  # 累加槽位数
                off[local_expert + Int32(1)] = running  # 写回排他前缀和（区间终点）
            acnt[0] = active_count  # 保存活跃专家总数
        cute.arch.sync_threads()  # 同步：off/alist/acnt 就绪
        for i in cutlass.range(tidx, EPN + 1, self.NUM_THREADS, unroll=1):  # 复制 off 到 cur 作为分桶写入游标
            cur[i] = off[i]
        cute.arch.sync_threads()  # 同步：游标就绪
        if warp_idx == Int32(0):  # warp 0 负责按专家分桶填充 slist
            lane = cute.arch.lane_idx()  # lane 编号
            lanes_lt = (Uint32(1) << lane) - Uint32(1)  # 低于本 lane 的 peer 掩码（前缀掩码）
            NCHUNK = cutlass.const_expr((R * EPN + 31) // 32)  # warp 需处理的 32 槽块数
            for chunk in cutlass.range(NCHUNK, unroll=1):  # 逐块处理
                rb = chunk * Int32(32) + lane  # 本 lane 负责的槽位一维下标
                local_expert = Int32(EPN)  # 默认哨兵：不属于本 rank
                if rb < Int32(R * EPN):  # 越界保护
                    e = sexp[rb]  # 读相对专家号
                    if e >= Int32(0) and e < Int32(EPN):  # 属于本 rank 才记录
                        local_expert = e
                peers = match_any_b32(local_expert)  # warp 内同专家号 lane 的互斥掩码
                cell = cur.iterator + local_expert  # 该专家的写入游标地址
                base = cute.arch.load(cell, Int32)  # 读当前游标基址
                pos = base + Int32(cute.arch.popc(peers & lanes_lt))  # 本 lane 在同组内的写偏移（保持 rb 升序）
                cute.arch.sync_warp()  # 同步：所有 lane 都已读完 base
                if (peers & lanes_lt) == Uint32(0):  # 组内最低 lane 代表整组推进游标
                    cute.arch.store(cell, base + Int32(cute.arch.popc(peers)))  # 游标 += 组大小
                cute.arch.sync_warp()  # 同步：游标推进完成
                if local_expert < Int32(EPN):  # 有效槽才写 slist
                    slist[pos] = Int32(rb)  # 槽位下标写入其专家的桶
        cute.arch.sync_threads()  # 同步：slist 全部就绪
        n_active = acnt[0]  # 活跃专家数（决定 tile 总数）

        if warp_idx == self.PRODUCER_WARP:  # warp 0：G2S 生产者
            load_state = pipeline.make_pipeline_state(  # 初始化生产者流水线状态
                pipeline.PipelineUserType.Producer, stages
            )
            for tensor_id in cutlass.range_constexpr(self.num_tensors):  # 编译期循环：张量序遍历 gate/up/down（及 scale）
                H, Hp = self.tensor_shapes[tensor_id]  # 该张量的 (H, H')
                MTILES = cutlass.const_expr(H // M_BLOCK)  # H 维 tile 数
                NTILES = cutlass.const_expr(Hp // N_BLOCK)  # H' 维 tile 数
                TILES_PER_EXPERT = cutlass.const_expr(MTILES * NTILES)  # 每个专家的 tile 数
                total_tiles = n_active * TILES_PER_EXPERT  # 该张量的 tile 总数

                for tile in cutlass.range(bidx, total_tiles, num_sms, unroll=1):  # CTA 间跨步认领 tile
                    i = tile // TILES_PER_EXPERT  # tile 所属的活跃专家序号
                    rem = tile % TILES_PER_EXPERT  # 专家内的 tile 余数
                    mt = rem // NTILES  # 行方向 tile 索引
                    nt = rem % NTILES  # 列方向 tile 索引

                    load_pipe.producer_acquire(load_state)  # 等待当前级空出
                    src_mt = alist[i] * MTILES + mt  # 源 gmem 中该 tile 的行块索引（按本地专家定位）
                    cute.copy(  # 发起 2D TMA 拷贝：gmem 源 tile -> smem 当前级
                        tma_g2s_atoms[tensor_id],
                        tSgS_list[tensor_id][(None, (src_mt, nt))],
                        tSsS_list[tensor_id][(None, load_state.index)],
                        tma_bar_ptr=load_pipe.producer_get_barrier(load_state),  # 完成后到达当前级 full mbarrier
                    )
                    load_state.advance()  # 推进生产者状态

        elif warp_idx >= self.FIRST_STORE_WARP and warp_idx < self.LAST_STORE_WARP:  # warp 1..6：S2G 存储
            store_warp_idx = warp_idx - self.FIRST_STORE_WARP  # 本 warp 在存储 warp 中的序号（0..5）
            # Every store warp advances through every producer tile, including
            # tiles owned by another projection. This keeps all consumer
            # states on the same ring index/phase and avoids phase-skip ABA
            # after the physical stages wrap around.
            use_state = pipeline.make_pipeline_state(  # 初始化消费者流水线状态（全员同环）
                pipeline.PipelineUserType.Consumer, stages
            )
            for tensor_id in cutlass.range_constexpr(self.num_tensors):  # 与生产者相同的张量序遍历
                H, Hp = self.tensor_shapes[tensor_id]  # 该张量形状
                MTILES = cutlass.const_expr(H // M_BLOCK)  # H 维 tile 数
                NTILES = cutlass.const_expr(Hp // N_BLOCK)  # H' 维 tile 数
                TILES_PER_EXPERT = cutlass.const_expr(MTILES * NTILES)  # 每专家 tile 数
                total_tiles = n_active * TILES_PER_EXPERT  # tile 总数
                projection = tensor_id % 3  # 所属投影：0=gate, 1=up, 2=down（scale 跟随其权重）

                tensor_work = Int32(0)  # 本张量内已处理的 tile 计数（用于分配属主）
                for tile in cutlass.range(bidx, total_tiles, num_sms, unroll=1):  # CTA 间跨步遍历全部 tile
                    # Each projection owns two independent store-warp queues.
                    # Alternate CTA-local tiles between them; using the local
                    # ordinal matters because global tile ids advance by
                    # num_sms and may otherwise keep the same parity.
                    owner = (  # 本 tile 的属主存储 warp：投影基址 + 局部序号奇偶交替
                        projection * Int32(self.STORE_WARPS_PER_PROJECTION)
                        + tensor_work % Int32(self.STORE_WARPS_PER_PROJECTION)
                    )
                    is_owner = owner == store_warp_idx  # 本 warp 是否该 tile 的属主

                    load_pipe.consumer_wait(use_state)  # 等待该 tile 数据就绪（即使非属主也要同步）
                    if is_owner:  # 仅属主发出扇出写
                        i = tile // TILES_PER_EXPERT  # 活跃专家序号
                        rem = tile % TILES_PER_EXPERT  # 专家内余数
                        mt = rem // NTILES  # 行方向 tile 索引
                        nt = rem % NTILES  # 列方向 tile 索引
                        local_expert = alist[i]  # 还原本地专家号
                        beg = off[local_expert]  # 该专家槽位桶的起点
                        nslot = off[local_expert + Int32(1)] - beg  # 该专家的槽位（目的 rank）数

                        # The unique tile owner issues the complete fan-out,
                        # reusing one local G2S tile for every destination slot.
                        for s in cutlass.range(nslot, unroll=1):  # 对每个目的槽写同一份 tile
                            rb = slist[beg + s]  # 目的槽的一维下标（编码目的 rank）
                            dst_rank = rb // EPN  # 解码目的 rank
                            dst_mt = (rb % EPN) * MTILES + mt  # 目的 gmem 中该 tile 的行块索引
                            cute.copy(  # 2D TMA 拷贝：smem tile -> 目的池 tile
                                tma_s2g_atoms[tensor_id],
                                tDsD_list[tensor_id][(None, use_state.index)],
                                tDgD_list[tensor_id][(None, (dst_mt, nt, dst_rank))],
                            )
                        cute.arch.cp_async_bulk_commit_group()  # 本 tile 的扇出提交为一个 bulk_group

                    # `.read` waits only until this warp's S2G groups stop
                    # reading the shared source; remote destination writes may
                    # remain in flight. All six consumers then arrive on the
                    # shared ring's empty barrier. Non-owners have no payload
                    # group for this tile, but must still arrive and advance;
                    # the owner's arrival is ordered after its source-read
                    # completion, making producer reuse of the stage safe.
                    cute.arch.cp_async_bulk_wait_group(0, read=True)  # 仅等待源读取完成（远端写可仍在途）
                    load_pipe.consumer_release(use_state)  # 到达共享环的 empty 屏障，允许生产者复用该级
                    use_state.advance()  # 推进消费状态
                    tensor_work += Int32(1)  # 更新本张量内 tile 计数

            # Unlike the per-stage read wait above, this full wait drains each
            # store warp's remote destination writes before the cross-rank
            # barrier publishes them to the receiving ranks.
            cute.arch.cp_async_bulk_wait_group(0)  # 完整排空本 warp 的远端目的写

        cross_rank_barrier(  # 收尾跨 rank 屏障：把全部预取写发布给接收 rank
            meta, meta_stride, barrier_off, rank, R,
            bar.iterator, Int32(num_sms), NUM_THREADS, tidx,
        )


@functools.lru_cache(maxsize=None)  # 按设备缓存设备属性查询
def _max_smem_per_block_optin(device_index: int) -> int:  # 查询每 CTA 可选用的最大动态共享内存
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin  # 读取 CUDA 设备属性


@functools.lru_cache(maxsize=None)  # 按 (形状, 选项) 缓存编译产物
def _get_compiled(  # 编译并缓存预取内核
    EPN: int,
    R: int,
    tensor_shapes: tuple[tuple[int, int], ...],
    rank_strides: tuple[int, ...],
    has_scales: bool,
    meta_stride: int,
    num_sms: int,
    device_index: int,
    torch_dtype: torch.dtype,
):
    elem_ty, elem_bytes = _ELEM_TYPES[torch_dtype]  # 由 torch dtype 查 DSL 元素类型与字节数
    smem_budget = _max_smem_per_block_optin(device_index) - 1024  # 预留 1KB 余量后的共享内存预算
    kernel = PrefetchKernel(  # 构造内核对象（内含级数/N_BLOCK 挑选）
        EPN=EPN,
        R=R,
        tensor_shapes=tensor_shapes,
        rank_strides=rank_strides,
        has_scales=has_scales,
        meta_stride=meta_stride,
        num_sms=num_sms,
        smem_budget=smem_budget,
        elem_ty=elem_ty,
        elem_bytes=elem_bytes,
    )

    data_ptr = make_ptr(elem_ty, 0, cute.AddressSpace.gmem, assumed_align=16)  # 占位数据指针（定类型/对齐）
    data_ptrs = tuple(data_ptr for _ in range(6 if has_scales else 3))  # 按张量数复制占位指针元组
    i32_experts_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=4)  # 占位专家表指针（4B 对齐）
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)  # 占位 meta/屏障指针
    stream_arg = cuda.CUstream(0)  # 占位流参数

    return cute.compile(  # 按 Buffer 类型与形状特化编译
        kernel,
        data_ptrs,
        data_ptrs,
        i32_experts_ptr,
        i32_ptr,
        i32_ptr,
        Int32(0),
        Int32(0),
        stream_arg,
    )


# 计算预取重排所需的每专家字节数（向上取整到 tile 大小）。
def prefetch_retile_nbytes(per_expert_nbytes: int) -> int:
    """Round a per-expert byte count up to what ``retile_for_prefetch`` needs.
    """
    tile = PrefetchKernel.M_BLOCK * PrefetchKernel.N_BLOCK  # 单 tile 元素数（也即字节倍数要求）
    return (per_expert_nbytes + tile - 1) // tile * tile  # 向上取整到 tile 的整数倍


# 把连续的 [N, ...] 专家张量改视图为 [N, 128, X] 的 uint8 张量（不拷贝数据）。
def retile_for_prefetch(t: torch.Tensor) -> torch.Tensor:
    """View a contiguous ``[N, ...]`` expert tensor as ``[N, 128, X]`` uint8.
    """
    assert t.is_contiguous(), "retile_for_prefetch requires a contiguous tensor"  # 必须连续才能安全改视图
    n = int(t.shape[0])  # 专家数 N
    per_expert = t.nbytes // n if n else 0  # 每专家字节数
    tile = PrefetchKernel.M_BLOCK * PrefetchKernel.N_BLOCK  # tile 元素数（= 内核的 2D tile 尺寸）
    assert per_expert % tile == 0, (  # 每专家字节数必须是 tile 的整数倍
        f"retile_for_prefetch: per-expert extent {per_expert} bytes is not a "
        f"multiple of {tile}; allocate {prefetch_retile_nbytes(per_expert)} "
        f"bytes per expert instead"
    )
    return t.view(torch.uint8).reshape(n, PrefetchKernel.M_BLOCK, -1)  # 按 uint8 重排为 [N, M_BLOCK, X]


# 校验一对 (本地张量, 预取池) 的 dtype/形状/设备/分块约束。
def _validate_prefetch_pair(
    local: torch.Tensor,
    pool: torch.Tensor,
    EPN: int,
    R: int,
    dtype: torch.dtype,
    device_index: int,
) -> None:
    # 本地张量：指定 dtype 且连续
    assert local.dtype == dtype and local.is_contiguous(), \
        f"local tensors must be contiguous with dtype {dtype}"
    # 形状 [EPN, H, H']
    assert local.ndim == 3 and int(local.shape[0]) == EPN, \
        f"local tensors must be [EPN={EPN}, H, H']"
    assert pool.dtype == dtype, f"prefetch pools must have dtype {dtype}"  # 预取池 dtype 一致
    # 预取池前两维 [R, EPN]
    assert pool.ndim == 4 and tuple(pool.shape[:2]) == (R, EPN), \
        f"prefetch pools must have leading shape [R, EPN]=[{R}, {EPN}]"
    # 尾部形状与本地张量一致
    assert tuple(pool.shape[2:]) == tuple(local.shape[1:]), \
        "local tensor and prefetch pool trailing shapes must match"
    # 同一 CUDA 设备
    assert local.device.index == device_index and pool.device.index == device_index, \
        "all prefetch tensors must be on the same CUDA device"
    H, Hp = (int(x) for x in local.shape[1:])  # 取 (H, H')
    # 维度须为 tile 大小整数倍
    assert H % PrefetchKernel.M_BLOCK == 0 and Hp % PrefetchKernel.N_BLOCK == 0, \
        "prefetch tensor trailing dimensions must be multiples of " \
        f"({PrefetchKernel.M_BLOCK}, {PrefetchKernel.N_BLOCK}), got ({H}, {Hp})"


# 校验预取启动所需的控制张量与参数。
def _validate_prefetch_launch(
    experts_to_copy: torch.Tensor,
    meta_buf: torch.Tensor,
    grid_sync_bar: torch.Tensor,
    R: int,
    EPN: int,
    rank: int,
    num_sms: int,
    device_index: int,
) -> None:
    # 专家表：连续 int32
    assert experts_to_copy.dtype == torch.int32 and experts_to_copy.is_contiguous(), \
        "experts_to_copy must be contiguous int32 [R, EPN]"
    # 形状 [R, EPN]
    assert tuple(experts_to_copy.shape) == (R, EPN), \
        f"experts_to_copy must be [R, EPN]=[{R}, {EPN}]"
    # 与预取池同设备
    assert experts_to_copy.device.index == device_index, \
        "experts_to_copy must be on the same CUDA device as prefetch pools"
    assert 0 <= int(rank) < R, f"rank must be in [0, {R}), got {rank}"  # rank 合法
    # num_sms 为正整数
    assert isinstance(num_sms, int) and num_sms > 0, \
        f"num_sms must be a positive int, got {num_sms}"
    assert meta_buf.dtype == torch.int32 and meta_buf.is_contiguous()  # meta 缓冲：连续 int32
    assert grid_sync_bar.dtype == torch.int32 and grid_sync_bar.is_contiguous() and grid_sync_bar.numel() == 1  # 屏障计数器：单元素 int32
    # 同设备
    assert meta_buf.device.index == device_index and grid_sync_bar.device.index == device_index, \
        "barrier tensors must be on the same CUDA device as prefetch pools"


# 预处理 scale 张量：校验后重排为内核兼容的 tile 视图。
def _prepare_prefetch_scales(
    scales: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
    EPN: int,
    R: int,
    device_index: int,
    weight_dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    if scales is None:  # 未提供 scale
        return (), ()  # 返回空元组

    # 必须提供 3 对 (本地, 池)
    assert len(scales) == 3, \
        "prefetch scales must contain three local/pool pairs"
    # 权重与 scale 必须共用 uint8 拷贝通道
    assert weight_dtype == torch.uint8, \
        "weight and re-tiled scale tensors must share uint8 copy dtype"
    scale_locals = []  # 重排后的本地 scale 列表
    scale_pools = []  # 重排后的池 scale 列表
    for local_scale, scale_pool in scales:  # 逐对处理
        assert local_scale.is_contiguous() and int(local_scale.shape[0]) == EPN  # 本地 scale：连续且首维 EPN
        assert tuple(scale_pool.shape[:2]) == (R, EPN)  # 池前两维 [R, EPN]
        assert scale_pool.dtype == local_scale.dtype  # dtype 一致
        assert tuple(scale_pool.shape[2:]) == tuple(local_scale.shape[1:])  # 尾部形状一致
        assert local_scale.device.index == device_index and scale_pool.device.index == device_index  # 同设备

        local_tiled = retile_for_prefetch(local_scale)  # 本地 scale 重排为 tile 视图
        # Preserve the writable pool alias and its rank gaps while re-tiling bytes.
        scale_bytes = scale_pool.view(torch.uint8)  # 池按 uint8 视图（保留原存储）
        pool_tiled = scale_bytes.as_strided(  # 用 as_strided 重排视图，保留 rank 间隙（stride(0) 不变）
            (R, EPN, *local_tiled.shape[1:]),
            (int(scale_pool.stride(0)) * scale_pool.element_size(), *local_tiled.stride()),
        )
        scale_locals.append(local_tiled)  # 收集本地视图
        scale_pools.append(pool_tiled)  # 收集池视图

    return tuple(scale_locals), tuple(scale_pools)  # 返回重排后的元组


# 预取的 host 侧启动入口：校验输入、合并权重与 scale、取编译缓存并启动。
def launch_prefetch(
    local_weights: tuple[torch.Tensor, ...],
    prefetch_buffers: tuple[torch.Tensor, ...],
    experts_to_copy: torch.Tensor,
    rank: int,
    num_sms: int,
    meta_buf: torch.Tensor,
    meta_stride: int,
    barrier_off: int,
    grid_sync_bar: torch.Tensor,
    scales: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
):
    """Launch one fused owner-push prefetch for all weight/scale tensors.

    ``local_weights`` contains gate/up/down tensors shaped ``[EPN, H, H']``;
    ``prefetch_buffers`` contains their all-rank ``[R, EPN, H, H']`` pools,
    contiguous within each rank with optional padding between ranks.
    ``experts_to_copy[R, EPN]`` stores global expert ids. When ``scales`` is
    present it contains three ``(local_scale, all_rank_scale_pool)`` pairs;
    they are re-tiled internally and participate in the same kernel launch.
    """
    # 必须 3 组权重与 3 个预取池
    assert len(local_weights) == 3 and len(prefetch_buffers) == 3, \
        "prefetch requires three local weights and three prefetch buffers"

    dtype = local_weights[0].dtype  # 拷贝 dtype 以第一个权重为准
    assert dtype in _ELEM_TYPES, (  # dtype 必须在支持列表内
        f"prefetch: unsupported dtype {dtype}; "
        f"supported: {sorted(str(d) for d in _ELEM_TYPES)}"
    )
    assert local_weights[0].ndim == 3 and prefetch_buffers[0].ndim == 4  # 维度约定：3D 本地 / 4D 池
    EPN = int(local_weights[0].shape[0])  # 每 rank 专家数
    R = int(prefetch_buffers[0].shape[0])  # rank 数
    device_index = prefetch_buffers[0].device.index  # 设备编号
    assert device_index is not None, "prefetch buffers must be CUDA tensors"  # 必须是 CUDA 张量

    for local_weight, prefetch_buffer in zip(  # 逐对校验权重/池
        local_weights, prefetch_buffers, strict=True
    ):
        _validate_prefetch_pair(
            local_weight, prefetch_buffer, EPN, R, dtype, device_index
        )

    _validate_prefetch_launch(  # 校验控制张量与参数
        experts_to_copy, meta_buf, grid_sync_bar,
        R, EPN, rank, num_sms, device_index,
    )

    has_scales = scales is not None  # 是否附带 scale
    scale_locals, scale_pools = _prepare_prefetch_scales(  # 预处理 scale（校验+重排）
        scales, EPN, R, device_index, dtype
    )

    all_locals = (*local_weights, *scale_locals)  # 合并：权重在前、scale 在后
    all_pools = (*prefetch_buffers, *scale_pools)  # 池同样合并
    tensor_shapes = tuple(  # 收集每个张量的 (H, H')
        (int(local.shape[1]), int(local.shape[2])) for local in all_locals
    )
    rank_strides = tuple(int(pool.stride(0)) for pool in all_pools)  # 收集每个池的 rank 步长（允许 rank 间 padding）

    compiled = _get_compiled(  # 取（或编译）本形状特化的内核
        EPN, R, tensor_shapes, rank_strides, has_scales,
        int(meta_stride), int(num_sms), int(device_index), dtype,
    )

    elem_ty, _ = _ELEM_TYPES[dtype]  # 拷贝元素类型
    local_ptrs = tuple(  # 各本地张量的 gmem 指针
        make_ptr(
            elem_ty,
            tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        for tensor in all_locals
    )
    pool_ptrs = tuple(  # 各预取池的 gmem 指针
        make_ptr(
            elem_ty,
            tensor.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        for tensor in all_pools
    )
    experts_ptr = make_ptr(  # 专家表指针（4B 对齐）
        Int32,
        experts_to_copy.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    meta_ptr = make_ptr(  # meta 缓冲指针
        Int32,
        meta_buf.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    bar_ptr = make_ptr(  # 网格屏障计数器指针
        Int32,
        grid_sync_bar.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)  # 当前 PyTorch 流封装为 CUstream

    compiled(  # 调用编译产物：填充指针并启动内核
        local_ptrs, pool_ptrs, experts_ptr, meta_ptr, bar_ptr,
        Int32(int(rank)), Int32(int(barrier_off)), stream,
    )
