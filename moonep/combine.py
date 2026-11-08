"""
MoonEP Combine — CuTe DSL implementation.

3-stage warp-specialized G2S / fp32 ACC / S2G with cp.async.bulk per-row TMA,
expressed entirely in the CUTLASS Python DSL. Hidden size H is a JIT constexpr;
the load-pipeline depth (kStagesL) is computed automatically inside the
host-side jit function from H plus the device's per-block opt-in dynamic smem
cap, so callers do not need to dispatch on stage count manually.
``launch_combine`` pulls back only the deduped payload rows (``dst >= 0``);
duplicate entries are pre-reduced into their primary slot by the combine
prologue on the destination rank (``moonep/combine_prologue.py``).
The dedup path runs a cross-rank publish barrier at combine entry before
reading peer ranks' staged NVL rows.
"""

import functools  # 提供 lru_cache：按参数缓存设备属性与编译好的 kernel

import torch  # PyTorch：入参张量的 dtype/contiguous 校验与 CUDA 流获取
import cuda.bindings.driver as cuda  # CUDA driver 绑定：CUstream 句柄类型

import cutlass  # CUTLASS Python DSL：constexpr 常量包装与数值类型
import cutlass.cute as cute  # CuTe 核心：layout/tensor 抽象、jit/kernel 装饰器
import cutlass.pipeline as pipeline  # CUTLASS 流水线原语：基于 mbarrier 的生产者-消费者同步
import cutlass.utils as utils  # CuTe 工具集：SmemAllocator 共享内存分配器
from cutlass import BFloat16, Float32, Int32, Int64  # DSL 数值类型：bf16/fp32/int32/int64
from cutlass.cute.runtime import make_ptr  # 宿主侧构造带 dtype+对齐假设的裸指针实参

from moonep._common import (  # 项目内通用辅助函数
    cp_async_bulk_g2s as _cp_async_bulk_g2s,  # 全局→共享内存的 cp.async.bulk 内联汇编封装
    cp_async_bulk_s2g as _cp_async_bulk_s2g,  # 共享→全局内存的 cp.async.bulk 内联汇编封装
    cross_rank_barrier,  # combine 入口处的跨 rank 发布屏障（dedup 路径需要）
    pdl_wait_predecessor,  # PDL（程序化依赖启动）模式下等待前序 kernel 完成
)


# ============================================================================
# Combine kernel
# ============================================================================
# combine 主 kernel：反向 gather 阶段。把 dispatch 阶段暂存在各 rank
# （hidden_buf，按 [rank, 本地槽位, H] 布局）的隐层行，按 dst 路由表拉回，
# 并对同一 token 的 K 个 top-k 副本在 fp32 寄存器中累加，写出每个 token
# 的聚合输出 [S, H]（可选地顺便 gather 每个 top-k 槽位的路由权重）。

# 三级 warp 专用化 combine kernel（详细布局与 dedup 契约见类 docstring）
class CombineKernel:
    """3-stage warp-specialized combine.

    Layout:
      - 6 warps per CTA: 1 G2S producer (warp 0), 4 fp32 ACC consumers
        (warps 1-4), 1 S2G consumer (warp 5).
      - Load pipeline (PipelineTmaAsync, kStagesL stages) hands one bf16 row
        at a time to the ACC warps. kStagesL is auto-picked at JIT time.
      - Output pipeline (PipelineAsync, 2 stages, fixed) hands the bf16 output
        row to warp 5, which TMA-stores it back to gmem.
      - The 4 ACC warps share a NamedBarrier (id=0) for sub-block sync without
        stalling G2S/S2G.

    Dedup contract: non-negative ``dst`` entries pull one NVL row and
    accumulate it. Negative entries encode the same raw destination as
    ``-raw_dst - 1`` for duplicate top-k entries: their contribution was
    pre-reduced into the primary slot by the combine prologue on the
    destination rank, so the hidden path skips them entirely; the weight
    gather decodes the raw slot and reads it as usual (route weights are
    per-topk values, never deduplicated). The cross-rank publish barrier runs
    at combine entry before remote rows are consumed.
    """

    ACC_THREADS = 128             # warps 1..4  累加线程组：warp1..4 共 4×32=128 线程
    STAGES_O = 2  # 输出流水线（ACC warps → warp5）固定 2 级

    # 构造函数：保存 kernel 的形状与配置参数；加载流水线级数 kStagesL
    # 依据 H 与每 block 可用 smem 预算在 JIT 前自动选取（见 _pick_stages_l）
    def __init__(
        self,
        H: int,  # 隐藏维大小（JIT constexpr）
        R: int,  # EP 组内 rank 数
        S: int,  # 本 rank 收到的 token 数（combine 输出行数）
        K: int,  # 每 token 的 top-k 路由数
        NvS: int,  # 每 rank 实际暂存的 payload 行数（recv 量）
        NvS_padded: int,  # 每 rank 在 hidden_buf 中对齐填充后的行数（行距 stride）
        meta_stride: int,  # meta_buf 中每个 rank 元数据块的填充步长
        num_sms: int,  # 启动网格的 block 数（=常驻 SM 数，cooperative 启动）
        with_weights: bool,  # 是否同时 gather 路由权重（决定是否启用 warp6）
        smem_budget: int,  # 每 block 可用的共享内存上限（字节）
        pdl_launch: bool,  # 是否以 PDL（程序化依赖启动）方式发射
    ):
        self.H = H  # 记录隐藏维
        self.R = R  # 记录 rank 数
        self.S = S  # 记录输出 token 数
        self.K = K  # 记录 top-k 数
        self.NvS = NvS  # 记录每 rank 实际行数（用于 dst 编解码取模）
        self.NvS_padded = NvS_padded  # 记录填充后行数（用于行地址计算）
        self.meta_stride = meta_stride  # 记录 meta 块步长
        self.num_sms = num_sms  # 记录网格 block 数
        self.with_weights = with_weights  # 记录是否需要权重 gather
        self.pdl_launch = pdl_launch  # 记录是否启用 PDL
        self.stages_l = self._pick_stages_l(H, smem_budget)  # 自动选择加载流水线深度
        self.num_warps = 6 if not with_weights else 7  # warp6 仅在需要权重 gather 时存在
        self.num_threads = 32 * self.num_warps  # 每 CTA 总线程数（每 warp 32 线程）
        if self.stages_l == 0:  # 连最小 2 级都放不下说明 H 超出 smem 预算
            raise RuntimeError(
                f"combine: H={H} too large for per-block smem budget "
                f"{smem_budget} B (need at least "
                f"{self._smem_bytes(H, 2)} B)"
            )

    # 计算 kernel 需要的每 block 动态共享内存字节数（静态方法，宿主侧调用）
    @classmethod
    def _smem_bytes(cls, H: int, stages_l: int) -> int:
        # stage_smem (bf16, kStagesL deep) + out_smem (bf16, STAGES_O deep)
        # + 2 mbarriers per stage for the load pipe and the out pipe.
        # The fp32 accumulator lives entirely in registers — no smem cost.
        # Pad each tensor to 128 B for the byte_alignment we ask SmemAllocator
        # for, plus 256 B headroom for any internal cutlass-allocated state.
        def _round_up(n: int, a: int) -> int:  # 把 n 向上取整到 a 的倍数
            return (n + a - 1) // a * a
        return (  # 加载 smem（stages_l 行 bf16）+ 输出 smem（2 行 bf16）
            _round_up(stages_l * H * 2, 128)  # 加载级缓冲：每级一行 H 个 bf16（2 字节），按 128B 对齐
            + _round_up(cls.STAGES_O * H * 2, 128)  # 输出级缓冲：2 级 × H 个 bf16，按 128B 对齐
            + _round_up(stages_l * 2 * 8, 16)  # 加载流水线 mbarrier：每级 full/empty 各 1 个（8B/个），16B 对齐
            + _round_up(cls.STAGES_O * 2 * 8, 16)  # 输出流水线 mbarrier：每级 2 个，16B 对齐
            + 256  # 预留 256B 余量给 cutlass 内部状态
        )

    # 从大到小尝试候选级数，返回第一个能放进 smem 预算的加载流水线深度；都不行返回 0
    @classmethod
    def _pick_stages_l(cls, H: int, smem_budget: int) -> int:
        for s in (16, 14, 12, 10, 8, 6, 4, 2):  # 候选级数（偶数，2~16）
            if cls._smem_bytes(H, s) <= smem_budget:  # 该级数下 smem 占用不超预算即采用
                return s
        return 0  # 无候选级数满足预算，交由构造函数报错

    # ------------------------------------------------------------------ host

    # 宿主侧 JIT 入口：构建全局内存 layout 并以 cooperative 方式发射 kernel
    @cute.jit
    def __call__(
        self,
        # Tensors / pointers
        output_ptr: cute.Pointer,        # bf16 [S, H]
        output_sk_ptr: cute.Pointer,     # int32 view of fp32 [S, K] (or placeholder)
        hidden_ptr: cute.Pointer,        # bf16 [R*NvS_padded, H]
        meta_ptr: cute.Pointer,          # int32 [R*meta_chunk_padded]
        dst_ptr: cute.Pointer,           # int32 [N=S*K]
        bar_ptr: cute.Pointer,           # int32 [1] grid barrier counter
        # Scalars
        rank: Int32,  # 当前 rank 编号
        weights_off: Int32,  # 权重区在 meta 块内的偏移（元素数）
        barrier_off: Int32,  # 屏障计数区在 meta 块内的偏移（元素数）
        stream: cuda.CUstream,  # 目标 CUDA 流
    ):
        H = cutlass.const_expr(self.H)  # 隐藏维（编译期常量）
        stages_l = cutlass.const_expr(self.stages_l)  # 加载流水线级数（编译期常量）
        S = cutlass.const_expr(self.S)  # 输出 token 数
        K = cutlass.const_expr(self.K)  # top-k 数
        NvS_padded = cutlass.const_expr(self.NvS_padded)  # 填充后行距
        meta_stride = cutlass.const_expr(self.meta_stride)  # meta 块步长
        R = cutlass.const_expr(self.R)  # rank 数

        # 2-D row-major views with int64 stride. R*NvS_padded*H can exceed 2^31
        # in production, so the row-offset arithmetic `srow * H` must be
        # evaluated as int64. The C++ port does the same with
        # `make_shape((int64_t)R * NvS_padded, H64)`.
        in_rows = R * NvS_padded  # 输入缓冲总行数（所有 rank 的暂存区之和）
        H64 = cutlass.Int64(H)  # 行距的 int64 形式，保证行偏移乘法按 64 位进行
        gmem_in = cute.make_tensor(  # 输入视图：hidden_buf 视作 [in_rows, H] 行主序
            hidden_ptr,
            cute.make_layout((in_rows, H), stride=(H64, cutlass.Int64(1))),  # 行距 H（int64），列距 1
        )
        gmem_out = cute.make_tensor(  # 输出视图：output_sh 视作 [S, H] 行主序
            output_ptr,
            cute.make_layout((S, H), stride=(H64, cutlass.Int64(1))),  # 行距 H（int64），列距 1
        )

        # No atoms needed: producer/consumer issue cp.async.bulk via inline asm.

        # dst / meta stay as plain global tensors (read with normal loads).
        dst_tensor = cute.make_tensor(  # dst 路由表：长度 S*K 的一维 int32 张量
            dst_ptr, cute.make_layout((S * K,))
        )
        meta_tensor = cute.make_tensor(  # meta 元数据：所有 rank 的 meta 块拼成的一维 int32 张量
            meta_ptr, cute.make_layout((R * meta_stride,))
        )
        # output_sk: when with_weights=False the kernel never reads it; the
        # caller passes a placeholder. We only need a sized tensor when we do.
        if cutlass.const_expr(self.with_weights):  # 需要权重时按真实大小建视图
            sk_tensor = cute.make_tensor(  # 权重输出视图：长度 S*K 的一维 int32 张量
                output_sk_ptr, cute.make_layout((S * K,))
            )
        else:  # 不需要权重时只建长度 1 的占位视图（kernel 不解引用）
            sk_tensor = cute.make_tensor(
                output_sk_ptr, cute.make_layout((1,))
            )

        bar_tensor = cute.make_tensor(bar_ptr, cute.make_layout((1,)))  # 网格级屏障计数器（单元素）

        smem_bytes = self._smem_bytes(H, stages_l)  # 本次发射所需的动态共享内存大小

        self.kernel(  # 以实际张量/标量为实参实例化并发射 kernel
            gmem_in,  # 输入隐层行视图
            gmem_out,  # 输出聚合结果视图
            dst_tensor,  # dst 路由表
            meta_tensor,  # meta 元数据
            sk_tensor,  # 权重输出视图
            bar_tensor,  # 网格屏障计数器
            rank,  # 当前 rank
            weights_off,  # 权重区偏移
            barrier_off,  # 屏障区偏移
        ).launch(  # 发射配置
            grid=(self.num_sms, 1, 1),  # 网格 = 常驻 SM 数个 block
            block=(self.num_threads, 1, 1),  # 每 block 6/7 个 warp 的线程
            smem=smem_bytes,  # 动态共享内存字节数
            stream=stream,  # 目标 CUDA 流
            cooperative=True,  # cooperative 启动：保证所有 block 同时常驻（网格级屏障需要）
            use_pdl=self.pdl_launch,  # 是否启用 PDL 依赖启动
        )

    # ----------------------------------------------------------------- kernel

    # 设备侧 kernel 主体：按 warp 分工实现 G2S 加载 → fp32 累加 → S2G 写回，
    # warp6 可选地并行执行权重 gather
    @cute.kernel
    def kernel(
        self,
        gmem_in: cute.Tensor,  # 输入隐层行 [R*NvS_padded, H]
        gmem_out: cute.Tensor,  # 输出聚合结果 [S, H]
        dst_tensor: cute.Tensor,  # dst 路由表 [S*K]
        meta_tensor: cute.Tensor,  # meta 元数据 [R*meta_stride]
        sk_tensor: cute.Tensor,  # 权重输出 [S*K]（或占位）
        bar_tensor: cute.Tensor,  # 网格屏障计数器 [1]
        rank: Int32,  # 当前 rank 编号
        weights_off: Int32,  # 权重区偏移
        barrier_off: Int32,  # 屏障区偏移
    ):
        H = cutlass.const_expr(self.H)  # 隐藏维
        S = cutlass.const_expr(self.S)  # 输出 token 数
        K = cutlass.const_expr(self.K)  # top-k 数
        NvS = cutlass.const_expr(self.NvS)  # 每 rank 实际暂存行数（解码用）
        NvS_padded = cutlass.const_expr(self.NvS_padded)  # 填充后行距（寻址用）
        meta_stride = cutlass.const_expr(self.meta_stride)  # meta 块步长
        stages_l = cutlass.const_expr(self.stages_l)  # 加载流水线级数
        STAGES_O = cutlass.const_expr(self.STAGES_O)  # 输出流水线级数
        num_ranks = cutlass.const_expr(self.R)  # rank 数
        num_threads = cutlass.const_expr(self.num_threads)  # 每 CTA 线程数

        if cutlass.const_expr(self.pdl_launch):  # PDL 模式下先等待前序 kernel 的依赖部分完成
            pdl_wait_predecessor()

        # ----- entry barrier: every peer rank's combine has reached us。
        tidx0 = cute.arch.thread_idx()[0]  # 线程在 block 内的一维编号
        cross_rank_barrier(  # 入口跨 rank 发布屏障：确保所有 peer rank 已把待读取的 NVL 行发布完毕
            meta_tensor, meta_stride, barrier_off, rank, num_ranks,
            bar_tensor.iterator, Int32(self.num_sms), num_threads, tidx0,
        )

        tidx, _, _ = cute.arch.thread_idx()  # 线程编号
        bidx, _, _ = cute.arch.block_idx()  # block 编号（决定本 block 负责的 token 区间）
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 统一化 warp 编号（避免分支发散）

        # ----- shared memory layout
        smem = utils.SmemAllocator()  # 动态共享内存分配器

        # Mbarrier blocks first (16-byte alignment is plenty).
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages_l)  # 加载流水线 mbarrier：每级 full/empty 各 1 个（8B）
        out_mbar = smem.allocate_array(Int64, num_elems=2 * STAGES_O)  # 输出流水线 mbarrier：每级 2 个

        stage_smem = smem.allocate_tensor(  # 加载级缓冲：[H, stages_l]，order=(0,1) 表示第 0 维最快 → 行距 1
            BFloat16,
            cute.make_ordered_layout((H, stages_l), order=(0, 1)),  # 元素 (h, s) 位于 h + s*H：每级是连续的 H 个 bf16
            byte_alignment=128,  # 128B 对齐，满足 cp.async.bulk 的地址对齐要求
        )
        out_smem = smem.allocate_tensor(  # 输出级缓冲：[H, STAGES_O]，同样的行连续布局
            BFloat16,
            cute.make_ordered_layout((H, STAGES_O), order=(0, 1)),  # 元素 (h, s) 位于 h + s*H
            byte_alignment=128,  # 128B 对齐以满足 S2G bulk 拷贝要求
        )

        # ----- pipelines
        # load pipe (TMA G2S):
        #   Producer = warp 0 (32 threads). Producer_acquire's full-mbar arrive
        #   uses elect_one internally → producer_group size = 1.
        #   Consumer = warps 1..4 (128 threads). PipelineTmaAsync.consumer_release
        #   guards arrive with `is_signalling_thread` (= lane 0 of each warp in
        #   the non-cluster case), so 4 arrivals → consumer_group size = 4.
        load_pipe = pipeline.PipelineTmaAsync.create(  # 加载流水线：warp0 生产、4 个 ACC warp 消费
            barrier_storage=load_mbar,  # mbarrier 存储区
            num_stages=stages_l,  # 级数
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),  # 生产者按单线程计数（elect_one）
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),  # 消费者按 4 计数（每 ACC warp 的 lane0 各到达一次）
            tx_count=H * 2,  # 每级的期望字节数：一行 H 个 bf16
        )

        # out pipe (PipelineAsync, plain mbar handshake):
        #   Producer = warps 1..4 (128 threads). All 128 call producer_commit
        #   (count-based arrive_mbarrier) → size = 128.
        #   Consumer = warp 5 (32 threads). All 32 call consumer_release
        #   (count-based arrive) → size = 32.
        out_pipe = pipeline.PipelineAsync.create(  # 输出流水线：ACC warps 生产、warp5 消费
            barrier_storage=out_mbar,  # mbarrier 存储区
            num_stages=STAGES_O,  # 固定 2 级
            producer_group=pipeline.CooperativeGroup(  # 生产者按 128 线程计数
                pipeline.Agent.Thread, self.ACC_THREADS
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),  # 消费者按 32 线程计数
        )

        # User NamedBarrier id=0 maps to HW id 8 (id 0..7 are reserved by the
        # CUTLASS pipelines / __syncthreads). 4 ACC warps = 128 threads.
        acc_bar = pipeline.NamedBarrier(barrier_id=8, num_threads=self.ACC_THREADS)  # 仅 4 个 ACC warp 参与的具名屏障（硬件 id 8）

        # ----- per-block token range
        tpb = (S + self.num_sms - 1) // self.num_sms  # 每 block 负责的 token 数（向上均分）
        s_beg = bidx * tpb  # 本 block 起始 token 下标
        s_end = cutlass.min(s_beg + tpb, S)  # 本 block 结束 token 下标（不超过 S）
        n_tok = s_end - s_beg  # 本 block 实际处理的 token 数

        # ============================================
        # Warp 0 — G2S producer
        # ============================================
        # warp0：加载生产者。逐 (token, k) 读取 dst，对有效项从 peer rank 的
        # 暂存行发起 cp.async.bulk 拷贝到共享内存的对应加载级
        if warp_idx == 0:
            load_state = pipeline.make_pipeline_state(  # 初始化生产者流水线状态（相位+槽位下标）
                pipeline.PipelineUserType.Producer, stages_l
            )
            H_BYTES = cutlass.const_expr(H * 2)  # 一行 bf16 的字节数（编译期常量）
            for s_local in cutlass.range(n_tok, unroll=1):  # 逐个处理本 block 的 token（禁用展开以省寄存器）
                s = s_beg + s_local  # 全局 token 下标
                sK = s * K  # 该 token 在 dst 表中的起始下标
                for k in cutlass.range(K, unroll=1):  # 逐个 top-k 槽位
                    dst_val = dst_tensor[sK + k]  # 读取路由目标（编码 dest_rank*NvS+loff，负值表示重复项）
                    # Negative dst marks a duplicate entry: its contribution
                    # was pre-reduced into the primary slot on the destination
                    # rank, so no payload row is loaded for it (the raw slot
                    # is only ever decoded by the warp-6 weight gather).
                    load_token = dst_val >= Int32(0)  # 负值=重复 top-k 项，载荷已并入主槽位，跳过加载

                    if load_token:  # 仅对有效项发起 TMA 拷贝
                        drank = dst_val // NvS  # 解码出源 rank 编号
                        loff = dst_val % NvS  # 解码出该 rank 内的本地行号
                        srow = drank * NvS_padded + loff  # 在 hidden_buf 中的全局行号（含填充行距）

                        load_pipe.producer_acquire(load_state)  # 等待当前加载级空闲（empty 屏障到达）

                        # Issue cp.async.bulk via inline asm — see _cp_async_bulk_g2s.
                        # cp.async.bulk is a single-thread instruction; only lane 0
                        # of warp 0 must execute it (otherwise 32 lanes each start a
                        # copy and the mbar transaction-arrive over-decrements).
                        if cute.arch.lane_idx() == 0:  # cp.async.bulk 是单线程指令，仅 lane0 发起
                            g_row_int = (  # 计算源行（全局内存）的字节地址
                                gmem_in.iterator + cutlass.Int64(srow) * cutlass.Int64(H)
                            ).toint()
                            s_row_int = (  # 计算目的行（共享内存当前级）的字节地址
                                stage_smem.iterator + load_state.index * H
                            ).toint()
                            mbar_int = load_pipe.producer_get_barrier(load_state).toint()  # 该级的 full mbarrier 地址
                            _cp_async_bulk_g2s(  # 发起异步批量拷贝：完成时事务计数自动到达 mbarrier
                                s_row_int.ir_value(),
                                g_row_int.ir_value(),
                                Int32(H_BYTES).ir_value(),
                                mbar_int.ir_value(),
                            )

                        load_state.advance()  # 推进生产者状态到下一级

        # ============================================
        # Warps 1..4 — fp32 ACC consumers (accumulate in registers)
        # ============================================
        # warps1..4：累加消费者。从共享内存逐行读入并以 fp32 寄存器累加，
        # 每处理完一个 token 后转成 bf16 写入输出级缓冲并通知 warp5
        elif warp_idx >= 1 and warp_idx <= 4:
            acc_tid = tidx - 32  # ACC 组内线程号（0..127）

            # Each ACC thread owns H / ACC_THREADS fp32 registers; the j-th
            # register covers element idx = j * ACC_THREADS + acc_tid. The
            # host-side assert (H % ACC_THREADS == 0) lets us drop the
            # `idx < H` predicate, which otherwise blocks the codegen from
            # software-pipelining LDS→SHF→FADD across j iterations.
            H_PER_THR = cutlass.const_expr(H // self.ACC_THREADS)  # 每线程持有的 fp32 累加寄存器个数
            acc_reg = cute.make_rmem_tensor(  # 寄存器累加器：H_PER_THR 个 fp32
                cute.make_layout((H_PER_THR,)), Float32
            )

            load_state = pipeline.make_pipeline_state(  # 加载流水线的消费者状态
                pipeline.PipelineUserType.Consumer, stages_l
            )
            out_state = pipeline.make_pipeline_state(  # 输出流水线的生产者状态
                pipeline.PipelineUserType.Producer, STAGES_O
            )

            # Two-level loop over (s_local, k): the K-cycle prologue
            # (acquire+zero) and epilogue (convert+commit) hoist outside,
            # letting the codegen drop the `if k == 0` / `if k == K-1`
            # branches and the `s_local = li // K` lowering.
            for s_local in cutlass.range(n_tok, unroll=1):  # 外层逐 token
                s = s_beg + s_local  # 全局 token 下标
                sK = s * K  # 该 token 的 dst 表起始下标
                stg_o = s_local % STAGES_O  # 本 token 使用的输出级槽位（循环复用）

                # Prologue: acquire next output slot + zero accumulator regs.
                # Thread-local — no cross-warp barrier needed.
                out_pipe.producer_acquire(out_state)  # 等待输出槽位空闲（warp5 已消费完毕）
                for j in cutlass.range_constexpr(H_PER_THR):  # 清零本线程的全部累加寄存器
                    acc_reg[j] = Float32(0.0)

                for k in cutlass.range(K, unroll=1):  # 内层逐 top-k 槽位累加
                    dst_val = dst_tensor[sK + k]  # 读取路由目标
                    # Same skip predicate as the producer (both read the same
                    # dst value), so wait/release stay stage-aligned with the
                    # issued loads: per token both sides step exactly once per
                    # non-negative entry. The predicate is uniform across all
                    # 128 ACC threads, keeping the acc_bar counts consistent.
                    accumulate_token = dst_val >= Int32(0)  # 与生产者相同的跳过判定，保证流水线状态步进对齐

                    if accumulate_token:  # 仅对有效项等待并累加
                        # Wait for this stage's TMA load to complete.
                        load_pipe.consumer_wait(load_state)  # 等待该级的 TMA 拷贝事务完成

                        # fp32 accumulate: acc_reg[j] += bf16-as-f32(stage[idx])
                        for j in cutlass.range_constexpr(H_PER_THR):  # 遍历本线程负责的元素
                            idx = j * self.ACC_THREADS + acc_tid  # 该寄存器覆盖的隐藏维下标（跨线程交织分布）
                            v = Float32(stage_smem[idx, load_state.index])  # 读当前级共享内存元素并转 fp32
                            acc_reg[j] = acc_reg[j] + v  # 累加到寄存器

                        # Weight gather lives in standalone warp6 (see below):
                        # keeping it on a single ACC thread serialized one global
                        # LDG per K-iter onto the critical path and capped c_bwd
                        # at a small fraction of the weights-free path.

                        # ACC warps must finish reading stage_smem before
                        # consumer_release lets the producer overwrite it.
                        # consumer_release signals from each warp's lane 0 only
                        # (PipelineTmaAsync signalling thread; consumer_group=4),
                        # so a warp-level sync is enough to make that arrive cover
                        # all 32 lanes' reads — cross-warp completion is already
                        # gated by the 4-count empty mbarrier.
                        cute.arch.fence_view_async_shared()  # 使本线程对共享内存的异步读取可见性有序
                        cute.arch.sync_warp()  # warp 内同步：确保本 warp 全部 32 线程读完该级
                        load_pipe.consumer_release(load_state)  # 释放该级（每 warp 由 lane0 代表到达，4 warp 共 4 次计数）
                        load_state.advance()  # 推进消费者状态到下一级

                # Epilogue: convert fp32 acc → bf16 out and notify warp 5.
                for j in cutlass.range_constexpr(H_PER_THR):  # 把 fp32 累加结果转 bf16 写入输出级缓冲
                    idx = j * self.ACC_THREADS + acc_tid  # 对应的隐藏维下标
                    out_smem[idx, stg_o] = BFloat16(acc_reg[j])  # 写入输出共享内存的当前槽位
                # All 128 ACC threads must finish writing out_smem before
                # warp 5's TMA store proxy can read it.
                acc_bar.arrive_and_wait()  # 4 个 ACC warp 在具名屏障汇合：确保 out_smem 写完
                cute.arch.fence_view_async_shared()  # 使共享内存写入对 TMA 存储代理可见
                out_pipe.producer_commit(out_state)  # 通知 warp5 该输出槽位就绪（128 线程计数）
                out_state.advance()  # 推进输出生产者状态

        # ============================================
        # Warp 5 — S2G consumer
        # ============================================
        # warp5：输出消费者。把就绪的输出级缓冲行用 cp.async.bulk 写回全局内存，
        # 并通过等待/释放维持输出流水线的空槽供应
        elif warp_idx == 5:
            use_state = pipeline.make_pipeline_state(  # 用于 consumer_wait 的状态
                pipeline.PipelineUserType.Consumer, STAGES_O
            )
            rel_state = pipeline.make_pipeline_state(  # 用于延迟 consumer_release 的状态（与 S2G 完成对齐）
                pipeline.PipelineUserType.Consumer, STAGES_O
            )
            H_BYTES = cutlass.const_expr(H * 2)  # 一行 bf16 的字节数
            for s_local in cutlass.range(n_tok, unroll=1):  # 逐 token 写回
                s = s_beg + s_local  # 全局 token 下标

                out_pipe.consumer_wait(use_state)  # 等待对应输出槽位被 ACC warps 写好

                # cp.async.bulk is single-thread — only lane 0 of warp 5 issues it.
                if cute.arch.lane_idx() == 0:  # 仅 lane0 发起单线程的 bulk 拷贝
                    g_row_int = (  # 目的行（全局内存 output_sh 第 s 行）字节地址
                        gmem_out.iterator + cutlass.Int64(s) * cutlass.Int64(H)
                    ).toint()
                    s_row_int = (  # 源行（共享内存当前输出级）字节地址
                        out_smem.iterator + use_state.index * H
                    ).toint()
                    _cp_async_bulk_s2g(  # 发起共享→全局的异步批量写回
                        s_row_int.ir_value(),
                        g_row_int.ir_value(),
                        Int32(H_BYTES).ir_value(),
                    )
                    cute.arch.cp_async_bulk_commit_group()  # 提交到 bulk 拷贝组，便于按组跟踪完成情况

                use_state.advance()  # 推进等待用状态

                if s_local >= Int32(STAGES_O - 1):  # 攒够 STAGES_O-1 个未完成拷贝后开始回收最旧的槽位
                    cute.arch.cp_async_bulk_wait_group(STAGES_O - 1)  # 等到至多剩 STAGES_O-1 个组未完成
                    out_pipe.consumer_release(rel_state)  # 释放对应槽位供 ACC warps 复用
                    rel_state.advance()  # 推进释放用状态

            cute.arch.cp_async_bulk_wait_group(0)  # 收尾：等待全部未完成的 S2G 拷贝完成
            # Drain remaining (up to STAGES_O-1) outstanding releases.
            for _ in cutlass.range(STAGES_O - 1, unroll=1):  # 补齐尾部未释放的槽位（≤STAGES_O-1 个）
                if rel_state.count < use_state.count:  # 仍有未释放的槽位则继续释放
                    out_pipe.consumer_release(rel_state)
                    rel_state.advance()

        # ============================================
        # Warp 6 — weight-gather epilogue
        # ============================================
        # warp6：权重 gather 尾声（仅反向且 with_weights 时存在）。与数据流水线
        # 无依赖，独立并行执行，避免串进 warp5 的关键路径
        elif warp_idx == 6:
            # Independent route for weight gather (bwd only).
            # The gather has no data dependency on the load/out pipelines, so it runs
            # concurrently in its own warp instead of serializing in warp 5's prologue,
            # which was throttled by the 2-stage out_pipe.
            if cutlass.const_expr(self.with_weights):  # 编译期裁剪：不需要权重时整段不生成
                lane = cute.arch.lane_idx()  # 本 warp 内 lane 号（0..31）
                sk_base = s_beg * K  # 本 block 起始 token 在 sk 表中的下标
                sk_total = n_tok * K  # 本 block 负责的 (token,k) 槽位总数
                n_chunks = (sk_total + 31) // 32  # 按 warp 宽度 32 分块数
                for sk_chunk in cutlass.range(n_chunks, unroll=1):  # 逐块处理
                    sk_local = sk_chunk * 32 + lane  # 本 lane 负责的槽位在块内的全局偏移
                    if sk_local < sk_total:  # 尾块越界的 lane 不参与
                        sk_global = sk_base + sk_local  # 在 sk 表中的全局下标
                        dst_val = dst_tensor[sk_global]  # 读取同一位置的路由目标
                        # Duplicate entries (-raw_dst - 1) still own their
                        # per-topk weight slot: decode and gather as usual,
                        # mirroring the dispatch-side weight scatter.
                        raw_dst = dst_val  # 原始编码（可能是负的重复项编码）
                        if raw_dst < 0:  # 重复项编码为 -raw_dst-1，还原真实目标
                            raw_dst = - raw_dst - Int32(1)
                        drank = raw_dst // NvS  # 解码源 rank
                        loff = raw_dst % NvS  # 解码本地行号
                        wb = meta_tensor[  # 从源 rank 的 meta 块权重区读取该行的路由权重
                            drank * meta_stride + weights_off + loff
                        ]
                        sk_tensor[sk_global] = wb  # 写入该 top-k 槽位的权重输出

# ============================================================================
# Host launcher with per-(H, R, with_weights) compile cache
# ============================================================================
# 宿主侧发射器：按 (H, R, with_weights) 等参数缓存编译产物，避免重复编译

# 查询设备每 block 可选入（opt-in）的最大动态共享内存，按设备号缓存
@functools.lru_cache(maxsize=None)
def _max_smem_per_block_optin(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


# 按 kernel 形状参数缓存 cute.compile 的编译结果（同参数只编译一次）
@functools.lru_cache(maxsize=None)
def _get_compiled(
    H: int,  # 隐藏维
    R: int,  # rank 数
    S: int,  # token 数
    K: int,  # top-k 数
    NvS: int,  # 每 rank 实际行数
    NvS_padded: int,  # 填充后行距
    meta_stride: int,  # meta 块步长
    num_sms: int,  # 网格 block 数
    with_weights: bool,  # 是否 gather 权重
    device_index: int,  # 目标 GPU 编号
    pdl_launch: bool,  # 是否启用 PDL
):
    smem_budget = _max_smem_per_block_optin(device_index) - 1024  # match C++ headroom
    kernel = CombineKernel(  # 用真实形状参数构造 kernel 对象
        H=H, R=R, S=S, K=K, NvS=NvS, NvS_padded=NvS_padded,
        meta_stride=meta_stride, num_sms=num_sms,
        with_weights=with_weights, smem_budget=smem_budget,
        pdl_launch=pdl_launch,
    )

    # Representative arguments for cute.compile. Pointers carry only their
    # dtype + assumed alignment — actual addresses are bound at call time.
    bf16_ptr = make_ptr(BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)  # 代表性 bf16 空指针（仅携带 dtype+对齐信息）
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)  # 代表性 int32 空指针
    stream_arg = cuda.CUstream(0)  # 代表性默认流

    return cute.compile(  # 以代表实参完成 DSL 编译，得到可反复调用的可执行对象
        kernel,
        bf16_ptr,  # output
        i32_ptr,   # output_sk (int32 view)
        bf16_ptr,  # hidden
        i32_ptr,   # meta
        i32_ptr,   # dst
        i32_ptr,   # bar (grid barrier counter)
        Int32(0),  # rank
        Int32(0),  # weights_off
        Int32(0),  # barrier_off
        stream_arg,
    )


# launch_combine 的 Python 入口：校验入参、取编译缓存并绑定真实地址发射 kernel
def launch_combine(
    ctx: dict,  # 会话上下文：hidden_buf/meta_buf/维度/偏移/屏障缓冲等
    output_sh,  # [S, H] bf16 输出缓冲，接收逐 token 累加结果
    dst,  # [N=S*K] int32 路由表（须与填充 hidden_buf 的 dispatch 一致）
    output_sk=None,  # [S, K] fp32 权重输出缓冲；None 表示跳过权重 gather
    *,
    pdl_launch: bool = False,  # 是否以 PDL 方式发射
):
    """Launch the combine kernel.

    Args:
        output_sh: [S, H] bf16 output buffer to receive the per-token
            accumulated result.
        dst: [N=S*K] int32 routing offsets (must match the dispatch that
            populated hidden_buf / weights_buf). Non-negative entries encode
            ``dest_rank * NvS + local_offset`` and are pulled and accumulated.
            Negative entries encode the same raw destination as
            ``-raw_dst - 1`` (duplicate top-k entries, pre-reduced into the
            primary slot by the combine prologue): the hidden path skips
            them, the weights gather decodes and reads them as usual.
        output_sk: [S, K] fp32 buffer to receive gathered route weights, or
            None to skip the weights gather (placeholder tensor is passed to
            satisfy the non-null pointer constraint; kernel ignores it when
            with_weights=False).
    """
    with_weights = output_sk is not None  # 提供了权重缓冲即启用权重 gather
    if not with_weights:  # 未启用时用 dst 顶替作为占位指针
        # Placeholder; kernel will not dereference.
        output_sk = dst

    assert output_sh.dtype == torch.bfloat16 and output_sh.is_contiguous(), \
        "output_sh must be contiguous bf16"  # 输出必须是连续 bf16
    assert dst.dtype == torch.int32 and dst.is_contiguous(), \
        "dst must be contiguous int32"  # 路由表必须是连续 int32
    assert ctx['hidden_buf'].dtype == torch.bfloat16 and ctx['hidden_buf'].is_contiguous()  # 隐层缓冲必须连续 bf16
    assert ctx['meta_buf'].dtype == torch.int32 and ctx['meta_buf'].is_contiguous()  # 元数据缓冲必须连续 int32
    if with_weights:  # 启用权重时校验权重缓冲
        assert output_sk.dtype == torch.float32 and output_sk.is_contiguous(), \
            "output_sk must be contiguous fp32"  # 权重输出必须是连续 fp32
    assert ctx['H'] % CombineKernel.ACC_THREADS == 0, \
        f"H must be a multiple of ACC_THREADS={CombineKernel.ACC_THREADS} " \
        "(also covers the 16-B bulk-copy alignment requirement)"  # H 须为 128 的倍数（兼顾寄存器划分与 bulk 拷贝对齐）

    H = int(ctx['H'])  # 取隐藏维
    R = int(ctx['R'])  # 取 rank 数
    S = int(ctx['S'])  # 取 token 数
    K = int(ctx['K'])  # 取 top-k 数
    NvS = int(ctx['NvS'])  # 取每 rank 实际行数
    NvS_padded = int(ctx['NvS_padded'])  # 取填充后行距
    meta_stride = int(ctx['meta_chunk_padded'])  # 取 meta 块填充步长
    num_sms = int(ctx['num_sms'])  # 取网格 block 数
    device_index = output_sh.device.index  # 取输出所在 GPU 编号

    compiled = _get_compiled(  # 按形状参数取（或触发）编译
        H, R, S, K, NvS, NvS_padded, meta_stride,
        num_sms, with_weights, device_index,
        bool(pdl_launch),
    )

    # int32 view onto fp32 output_sk so the kernel always sees an int32 ptr
    # (matches the C++ approach: a 4-byte gather, no fp32 arithmetic).
    sk_int = output_sk.view(torch.int32) if with_weights else output_sk  # 把 fp32 权重缓冲重解释为 int32 视图

    out_ptr = make_ptr(BFloat16, output_sh.data_ptr(), cute.AddressSpace.gmem,  # 输出缓冲指针（16B 对齐假设）
                       assumed_align=16)
    sk_ptr = make_ptr(Int32, sk_int.data_ptr(), cute.AddressSpace.gmem,  # 权重输出（int32 视图）指针
                      assumed_align=16)
    hid_ptr = make_ptr(BFloat16, ctx['hidden_buf'].data_ptr(),  # 隐层缓冲指针
                       cute.AddressSpace.gmem, assumed_align=16)
    meta_ptr = make_ptr(Int32, ctx['meta_buf'].data_ptr(),  # 元数据缓冲指针
                        cute.AddressSpace.gmem, assumed_align=16)
    dst_ptr = make_ptr(Int32, dst.data_ptr(), cute.AddressSpace.gmem,  # 路由表指针
                       assumed_align=16)
    bar_ptr = make_ptr(Int32, ctx['grid_sync_bar'].data_ptr(),  # 网格屏障计数器指针
                       cute.AddressSpace.gmem, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)  # 绑定当前 PyTorch CUDA 流

    compiled(  # 调用编译好的 kernel：绑定真实地址与标量并发射
        out_ptr,  # 输出 [S, H]
        sk_ptr,  # 权重输出（int32 视图）
        hid_ptr,  # 隐层缓冲
        meta_ptr,  # 元数据
        dst_ptr,  # 路由表
        bar_ptr,  # 网格屏障计数器
        Int32(int(ctx['rank'])),  # 当前 rank
        Int32(int(ctx['WEIGHTS_OFF'])),  # 权重区偏移
        Int32(int(ctx['BARRIER_OFF'])),  # 屏障区偏移
        stream,  # CUDA 流
    )
