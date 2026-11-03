"""
MoonEP Dispatch — CuTe DSL implementation.

Warp-specialized G2S/S2G ring buffer with cp.async.bulk per-row TMA, expressed
entirely in the CUTLASS Python DSL. ``launch_dispatch`` writes representative
payload rows (``dst >= 0``) into the NVL buffer and scatters the per-topk
route weights, zero-fills the segment-padding rows of the local NVL shard
(warp 2, driven by ``plan.zero_fill_ranges``), and on fresh planning paths
materializes the plan-owned dedup structures (``dup_groups`` / ``dup_loffs``
/ ``dup_counts``) with builder warps; plan-reuse paths keep those tensors
untouched. The in-place duplicate expansion on the NVL shard lives in
``moonep/dispatch_epilogue.py`` (``launch_dispatch_epilogue``).
"""

import functools  # 标准库：lru_cache 用于按形状缓存编译产物

import torch  # PyTorch：张量属性校验与 CUDA 流
import cuda.bindings.driver as cuda  # CUDA driver 绑定：CUstream 类型

import cutlass  # CUTLASS Python DSL 根模块（const_expr 等编译期工具）
import cutlass.cute as cute  # CuTe：布局/张量/内核 DSL
import cutlass.pipeline as pipeline  # CuTe 流水线：TMA mbarrier 生产者-消费者同步
import cutlass.utils as utils  # CuTe 工具：SmemAllocator 共享内存分配器
from cutlass import BFloat16, Int32, Int64, Uint32  # DSL 标量类型
from cutlass.cute.runtime import make_ptr  # 由裸指针构造 DSL Pointer 运行期参数

from moonep._common import (  # 共享底层原语：bulk 拷贝、跨 rank 屏障、原子操作
    cp_async_bulk_g2s,  # PTX cp.async.bulk：全局内存 -> 共享内存整行拷贝
    cp_async_bulk_s2g,  # PTX cp.async.bulk：共享内存 -> 全局内存整行拷贝
    cross_rank_barrier,  # 自复位跨 rank 屏障（grid_sync + 系统级 release/acquire）
    cross_warp_sync,  # 跨 CTA 多 warp 参与的自复位屏障（原子计数实现）
    atom_add_relaxed_gpu_s32,  # gpu 作用域 relaxed 32 位原子加
    atom_min_relaxed_gpu_s32,  # gpu 作用域 relaxed 32 位原子取最小值
    atom_or_relaxed_gpu_b32,  # gpu 作用域 relaxed 32 位原子按位或
    popc_b32,  # PTX popc.b32：32 位按位 1 计数
    ctz_b32,  # PTX 尾部零计数：最低置位 bit 的位置
    pdl_trigger_dependents,  # PDL：发布写入并触发依赖网格提前启动
)
from moonep.constants import (  # 项目级常量
    KIDX_BITS,  # 打包编码中 top-k 索引 kidx 字段的位宽
    DEDUP_BUILDER_WARPS,  # 每个 dispatch CTA 分配的去重构建 warp 数
)
from moonep.planning import MoonEPCommPlan, warp_inclusive_scan  # 通信计划类型与 warp 内前缀和


# ============================================================================
# Dispatch kernel
# ============================================================================

# dispatch 主内核：warp 特化的 G2S/S2G 环形缓冲，零拷贝直写远端 NVL shard。
class DispatchKernel:
    """Warp-specialized dispatch.

    Layout:
      - Warp 0 G2S producer / warp 1 S2G consumer (data path, single thread
        of each drives the TMA copies; pipeline depth auto-picked at JIT).
      - Warp 2 zero warp: per-expert segment-padding zeroing of the local NVL
        shard (both hidden rows and the matching int32 weight slots when
        ``with_weights`` is set), driven by ``plan.zero_fill_ranges``. Runs on
        reuse paths too.
      - Warps 3..: DEDUP_BUILDER_WARPS dedup builder warps, fresh-planning
        paths only (build_dedup_map=True). They materialize the plan-owned
        ``dup_groups`` / ``dup_loffs`` / ``dup_counts`` from the slot_info
        provenance published by planning.
      - Exit barrier is cross_rank_barrier: dispatch writes to NVL hidden_buf /
        weights, then publishes them through grid_sync plus system-scope
        release/acquire atomics before peer ranks consume the rows.

    Dedup contract: non-negative ``dst`` entries copy the payload row to NVL.
    Negative entries encode the same raw destination for duplicate top-k
    entries; they skip the payload copy but still scatter the per-topk weight.
    Duplicate rows are expanded in place on the NVL shard afterwards by
    ``launch_dispatch_epilogue``.
    """

    # 3 fixed warps (producer + consumer + zero) + DEDUP_BUILDER_WARPS
    # dedup builder warps on fresh-planning paths.
    num_threads = 96 + 32 * DEDUP_BUILDER_WARPS  # 默认每 CTA 线程数：3 个固定 warp + 构建 warp 数 × 32
    PRODUCER_WARP = 0  # warp 0：G2S 生产者（TMA 装载源 token 行）
    CONSUMER_WARP = 1  # warp 1：S2G 消费者（TMA 写出 + 权重散射）
    ZERO_WARP = 2  # warp 2：段填充清零 warp
    DEDUP_BUILDER_WARP = 3  # warp 3 起：去重结构构建 warp 的起始编号

    # 构造函数：保存编译期形状参数并挑选流水线级数。
    def __init__(
        self,
        H: int,
        R: int,
        S: int,
        K: int,
        zero_groups: int,       # zero_fill_ranges entries = 2 * epn
        NvS: int,
        NvS_padded: int,
        SRC_INFO_OFF: int,
        meta_stride: int,
        num_sms: int,
        with_weights: bool,
        build_dedup_map: bool,
        smem_budget: int,
        pdl_trigger: bool,
    ):
        self.H = H  # 隐藏维度（每 token 行的 bf16 元素数）
        self.R = R  # EP 组内 rank 数
        self.S = S  # 每 rank 的 token 数
        self.K = K  # top-k 路由值
        self.zero_groups = zero_groups  # zero_fill_ranges 条目数（= 2*epn）
        self.NvS = NvS  # 每 rank NVL shard 的有效行数
        self.NvS_padded = NvS_padded  # 每 rank NVL shard 的填充后行数
        self.meta_stride = meta_stride  # meta_buf 中每 rank 槽位的步长
        self.SRC_INFO_OFF = SRC_INFO_OFF  # meta 内 src_info 划痕区的槽偏移
        self.num_sms = num_sms  # 启动网格的 CTA（SM）数
        self.with_weights = with_weights  # 是否散射 per-topk 路由权重
        self.pdl_trigger = pdl_trigger  # 结束时是否触发 PDL 依赖网格
        self.build_dedup_map = build_dedup_map  # 是否为新鲜规划路径重建去重结构
        self.num_threads = (  # 按是否构建去重图决定线程总数（复用路径无构建 warp）
            96 + 32 * DEDUP_BUILDER_WARPS if build_dedup_map else 96
        )
        self.stages = self._pick_stages(H, smem_budget)  # 按共享内存预算挑选流水线级数
        if self.stages == 0:  # 级数为 0 说明一行都放不进预算
            raise RuntimeError(  # 快速失败并报告所需最小共享内存
                f"dispatch: H={H} too large for per-block smem budget "
                f"{smem_budget} B (need at least "
                f"{self._smem_bytes(H, 2)} B)"
            )

    @classmethod  # 类方法：不依赖实例状态
    def _smem_bytes(cls, H: int, stages: int) -> int:  # 计算给定级数下每 CTA 所需动态共享内存字节数
        # stage_smem (bf16, kStages deep) + 2 mbar i64 per stage (full+empty).
        # Pad stage_smem to 128 B for byte_alignment, mbar block
        # to 16 B, plus 256 B headroom for any cutlass-internal static smem.
        def _round_up(n: int, a: int) -> int:  # 内部工具：向上取整到 a 的倍数
            return (n + a - 1) // a * a  # 标准向上取整公式
        return (  # 各部分按对齐要求求和再加余量
            _round_up(stages * H * 2, 128)  # stage_smem：stages 级 × 每行 H×2 字节（bf16），128B 对齐
            + _round_up(H * 2, 128)  # zero_smem：单行清零模板缓冲，128B 对齐
            + _round_up(stages * 2 * 8, 16)  # 每级 full+empty 两个 mbarrier（各 8 字节），16B 对齐
            + 256  # 为 cutlass 内部静态共享内存预留的余量
        )

    @classmethod  # 类方法
    def _pick_stages(cls, H: int, smem_budget: int) -> int:  # 在候选级数中选出预算内最深流水线
        for s in (16, 14, 12, 10, 8, 6, 4, 2):  # 从深到浅尝试候选级数
            if cls._smem_bytes(H, s) <= smem_budget:  # 第一个满足预算的级数即为最优
                return s  # 返回选中的级数
        return 0  # 全部超预算则返回 0（调用方据此报错）

    # ------------------------------------------------------------------ host

    @cute.jit  # 标记为 CuTe JIT 编译的 host 侧入口
    def __call__(  # 内核调用入口：构造张量视图并启动设备内核
        self,
        hidden_sh_ptr: cute.Pointer,          # bf16 [S, H]
        hidden_buf_ptr: cute.Pointer,         # bf16 [R*NvS_padded, H]
        weights_ptr: cute.Pointer,            # int32 view of fp32 [S, K] (or placeholder)
        dst_ptr: cute.Pointer,                # int32 [N=S*K]
        meta_ptr: cute.Pointer,               # int32 [R*meta_stride]
        zero_fill_ranges_ptr: cute.Pointer,   # int32 [2*epn, 2] (col0=pad_start, col1=n_pad)
        bar_ptr: cute.Pointer,                # int32 [1] grid barrier counter
        primary_packed_ptr: cute.Pointer,
        kmask_ptr: cute.Pointer,
        kidx_to_loff_ptr: cute.Pointer,
        dup_groups_ptr: cute.Pointer,
        dup_loffs_ptr: cute.Pointer,
        dup_counts_ptr: cute.Pointer,
        builder_bar_ptr: cute.Pointer,
        rank: Int32,
        weights_off: Int32,
        barrier_off: Int32,
        stream: cuda.CUstream,
    ):
        H = cutlass.const_expr(self.H)  # 提升为编译期常量：隐藏维度
        S = cutlass.const_expr(self.S)  # 编译期常量：每 rank token 数
        K = cutlass.const_expr(self.K)  # 编译期常量：top-k 值
        NvS = cutlass.const_expr(self.NvS)  # 编译期常量：每 rank 有效行数
        NvS_padded = cutlass.const_expr(self.NvS_padded)  # 编译期常量：填充后行数
        meta_stride = cutlass.const_expr(self.meta_stride)  # 编译期常量：meta 步长
        R = cutlass.const_expr(self.R)  # 编译期常量：rank 数
        stages = cutlass.const_expr(self.stages)  # 编译期常量：流水线级数

        # 2-D row-major views with int64 stride. R*NvS_padded*H can exceed
        # 2^31 in production, so the row-offset arithmetic must be evaluated
        # as int64.
        out_rows = R * NvS_padded  # 目的缓冲总行数（所有 rank 的 shard 拼接）
        H64 = cutlass.Int64(H)  # 行步长的 int64 形式，防止行偏移溢出 int32
        gmem_src = cute.make_tensor(  # 源 hidden [S, H] 的行主序张量视图
            hidden_sh_ptr,
            cute.make_layout((S, H), stride=(H64, cutlass.Int64(1))),
        )
        gmem_dst = cute.make_tensor(  # 目的 NVL 缓冲 [R*NvS_padded, H] 的行主序视图
            hidden_buf_ptr,
            cute.make_layout((out_rows, H), stride=(H64, cutlass.Int64(1))),
        )

        dst_tensor = cute.make_tensor(  # 目的地编码数组 [S*K]：正数=rank*NvS+loff，负数=重复条目
            dst_ptr, cute.make_layout((S * K,))
        )
        meta_tensor = cute.make_tensor(  # meta 缓冲的一维 int32 视图 [R*meta_stride]
            meta_ptr, cute.make_layout((R * meta_stride,))
        )
        # weights: int32 view of fp32 [S, K] when with_weights, else placeholder.
        if cutlass.const_expr(self.with_weights):  # 编译期分支：带权重路径
            w_tensor = cute.make_tensor(  # 路由权重的 int32 位视图 [S*K]
                weights_ptr, cute.make_layout((S * K,))
            )
        else:  # 无权重时占位，避免空指针
            w_tensor = cute.make_tensor(  # 长度 1 的占位张量（内核不会解引用）
                weights_ptr, cute.make_layout((1,))
            )
        # zero_fill_ranges: int32 [2*epn, 2] — col 0 pad_start_loff, col 1 n_pad_rows.
        # Linearized as length 4*epn so warp 2's per-group read pulls both
        # values via a single contiguous int2 load.
        zero_fill_ranges_tensor = cute.make_tensor(  # 清零区间表线性化为一维 [4*epn]
            zero_fill_ranges_ptr,
            cute.make_layout((2 * cutlass.const_expr(self.zero_groups),)),
        )
        bar_tensor = cute.make_tensor(bar_ptr, cute.make_layout((1,)))  # 网格级屏障计数器（长度 1）

        primary_packed_tensor = cute.make_tensor(  # 每 (源rank,token) 主选槽打包值的划痕区 [R*S]
            primary_packed_ptr, cute.make_layout((R * S,))
        )
        kmask_tensor = cute.make_tensor(  # 每 (源rank,token) 的 kidx 占用位掩码划痕区 [R*S]
            kmask_ptr, cute.make_layout((R * S,))
        )
        kidx_to_loff_tensor = cute.make_tensor(  # kidx -> 本地槽偏移映射 [R*S*K]
            kidx_to_loff_ptr, cute.make_layout((R * S * K,))
        )
        dup_groups_tensor = cute.make_tensor(  # 去重组头表 [NvS,3] 展平视图
            dup_groups_ptr, cute.make_layout((NvS * 3,))
        )
        dup_loffs_tensor = cute.make_tensor(  # 紧凑重复槽偏移表 [NvS]
            dup_loffs_ptr, cute.make_layout((NvS,))
        )
        dup_counts_tensor = cute.make_tensor(  # 去重组数/重复槽数输出计数器 [2]
            dup_counts_ptr, cute.make_layout((2,))
        )
        builder_bar_tensor = cute.make_tensor(  # 构建器跨 CTA 全局同步计数器
            builder_bar_ptr, cute.make_layout((1,))
        )

        smem_bytes = self._smem_bytes(H, stages)  # 最终动态共享内存用量

        self.kernel(  # 以张量/标量参数实例化设备内核
            gmem_src,
            gmem_dst,
            dst_tensor,
            meta_tensor,
            w_tensor,
            zero_fill_ranges_tensor,
            bar_tensor,
            primary_packed_tensor,
            kmask_tensor,
            kidx_to_loff_tensor,
            dup_groups_tensor,
            dup_loffs_tensor,
            dup_counts_tensor,
            builder_bar_tensor,
            rank,
            weights_off,
            barrier_off,
        ).launch(  # 启动内核（persistent 协作网格）
            grid=(self.num_sms, 1, 1),  # 网格 = num_sms 个 CTA
            block=(self.num_threads, 1, 1),  # 每 CTA 线程数（warp 特化布局）
            smem=smem_bytes,  # 动态共享内存字节数
            stream=stream,  # 在调用方指定的 CUDA 流上启动
            cooperative=True,  # 协作启动：内核内可做网格级同步
        )

    # ----------------------------------------------------------------- kernel

    @cute.kernel  # 标记为 CuTe 设备内核
    def kernel(  # 设备侧主内核：warp 特化的 dispatch 数据路径
        self,
        gmem_src: cute.Tensor,
        gmem_dst: cute.Tensor,
        dst_tensor: cute.Tensor,
        meta_tensor: cute.Tensor,
        w_tensor: cute.Tensor,
        zero_fill_ranges_tensor: cute.Tensor,
        bar_tensor: cute.Tensor,
        primary_packed_tensor: cute.Tensor,
        kmask_tensor: cute.Tensor,
        kidx_to_loff_tensor: cute.Tensor,
        dup_groups_tensor: cute.Tensor,
        dup_loffs_tensor: cute.Tensor,
        dup_counts_tensor: cute.Tensor,
        builder_bar_tensor: cute.Tensor,
        rank: Int32,
        weights_off: Int32,
        barrier_off: Int32,
    ):
        R = cutlass.const_expr(self.R)  # 编译期常量：rank 数
        H = cutlass.const_expr(self.H)  # 编译期常量：隐藏维度
        S = cutlass.const_expr(self.S)  # 编译期常量：token 数
        K = cutlass.const_expr(self.K)  # 编译期常量：top-k 值
        zero_groups = cutlass.const_expr(self.zero_groups)  # 编译期常量：清零区间组数
        NvS = cutlass.const_expr(self.NvS)  # 编译期常量：每 rank 有效行数
        NvS_padded = cutlass.const_expr(self.NvS_padded)  # 编译期常量：填充后行数
        meta_stride = cutlass.const_expr(self.meta_stride)  # 编译期常量：meta 步长
        stages = cutlass.const_expr(self.stages)  # 编译期常量：流水线级数
        num_sms = cutlass.const_expr(self.num_sms)  # 编译期常量：CTA 数
        num_threads = cutlass.const_expr(self.num_threads)  # 编译期常量：每 CTA 线程数
        num_ranks = cutlass.const_expr(self.R)  # 编译期常量：退出屏障参与的 rank 数
        H_BYTES = cutlass.const_expr(H * 2)  # 一行 hidden 的字节数（bf16 每元素 2 字节）
        # Offset of the per-rank src_info scratch written by planning.
        SRC_INFO_OFF = cutlass.const_expr(self.SRC_INFO_OFF)  # src_info 划痕区在 meta 槽内的偏移
        # src_info mirrors dst's rank-stride encoding:
        # src_rank * NvS + offv, with -1 for empty slots.
        NvS_BITS = cutlass.const_expr(32 - 1 - KIDX_BITS)  # 打包值中槽偏移字段的位宽（1 位符号 + KIDX_BITS 位 kidx）
        NvS_MASK = cutlass.const_expr((1 << NvS_BITS) - 1)  # 提取低 NvS_BITS 位槽偏移的掩码
        INT32_MAX = cutlass.const_expr(0x7FFFFFFF)  # int32 最大值，作为"未选中"哨兵

        bidx, _, _ = cute.arch.block_idx()  # 当前 CTA 的网格编号
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 归一化 warp 编号（全 warp 一致，便于分支）
        tidx, _, _ = cute.arch.thread_idx()  # CTA 内线程编号（共享内存清零与屏障使用）

        # ----- shared memory layout
        smem = utils.SmemAllocator()  # 动态共享内存分配器
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages)  # 每级 full+empty 两个 mbarrier 的存储
        stage_smem = smem.allocate_tensor(  # 多级行缓冲 [H, stages]：G2S 暂存区
            BFloat16,
            cute.make_ordered_layout((H, stages), order=(0, 1)),  # 行内连续、级间跨行（order=(0,1)）
            byte_alignment=128,  # cp.async.bulk 要求 128 字节对齐
        )
        zero_smem = smem.allocate_tensor(  # 单行清零模板缓冲 [H]（warp 2 复用）
            BFloat16,
            cute.make_layout((H,)),
            byte_alignment=128,
        )

        # Load pipeline (TMA G2S):
        #   Producer = warp 0 (1 effective thread, lane 0).
        #   Consumer = warp 1 (1 effective thread, lane 0).
        load_pipe = pipeline.PipelineTmaAsync.create(  # 创建 TMA 异步 G2S 流水线
            barrier_storage=load_mbar,  # mbarrier 存储指向已分配的共享内存
            num_stages=stages,  # 流水线深度
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),  # 生产者组：1 个有效线程
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),  # 消费者组：1 个有效线程
            tx_count=H * 2,  # 每级期望传输字节数 = 一行 hidden
        )

        # ----- zero_smem one-shot init (reused by every padding s2g from
        # warp 2). All NUM_THREADS cooperatively zero H bf16 elements, then
        # a single fence publishes the cta-scoped smem writes to the
        # cp.async.bulk async proxy for the rest of the kernel.
        H_PER_THR_ZERO = cutlass.const_expr((H + num_threads - 1) // num_threads)  # 每线程需清零的元素数（向上取整）
        for j in cutlass.range_constexpr(H_PER_THR_ZERO):  # 静态循环遍历本线程负责的分片
            idx = j * num_threads + tidx  # 本线程第 j 个元素的下标（跨线程步幅分布）
            if idx < Int32(H):  # H 非整除时防越界
                zero_smem[idx] = BFloat16(0)  # 写入 bf16 零
        cute.arch.fence_view_async_shared()  # 把普通 smem 写发布到异步代理（供 cp.async.bulk 读取）
        cute.arch.barrier()  # 全 CTA 同步：确保零模板就绪

        # ----- per-block token range
        tpb = (S + self.num_sms - 1) // self.num_sms  # 每 CTA 分摊的 token 数（向上取整均分）
        s_beg = bidx * tpb  # 本 CTA 的起始 token 下标
        s_end = cutlass.min(s_beg + tpb, S)  # 结束下标（不超过 S）
        n_tok = s_end - s_beg  # 本 CTA 处理的 token 数

        # ============================================
        # Warp 0 — G2S producer
        # ============================================
        if warp_idx == self.PRODUCER_WARP:  # warp 0：G2S 生产者
            load_state = pipeline.make_pipeline_state(  # 初始化生产者流水线状态（级索引+相位）
                pipeline.PipelineUserType.Producer, stages
            )
            for li in cutlass.range(n_tok, unroll=1):  # 逐 token 装载（unroll=1 控制代码体积）
                s = s_beg + li  # 当前全局 token 下标
                load_pipe.producer_acquire(load_state)  # 等待当前级缓冲已被消费者释放
                # cp.async.bulk is single-thread; only lane 0 issues it.
                if cute.arch.lane_idx() == 0:  # bulk 拷贝是单线程指令，仅 lane 0 发出
                    g_row_int = (  # 源 token 行在全局内存的起始地址（int64 行偏移防溢出）
                        gmem_src.iterator + cutlass.Int64(s) * cutlass.Int64(H)
                    ).toint()
                    s_row_int = (  # 当前级共享内存行起始地址
                        stage_smem.iterator + load_state.index * H
                    ).toint()
                    mbar_int = load_pipe.producer_get_barrier(load_state).toint()  # 当前级 full mbarrier 地址
                    cp_async_bulk_g2s(  # 发起 TMA 整行拷贝：gmem -> smem，完成后到达 mbarrier
                        s_row_int.ir_value(),
                        g_row_int.ir_value(),
                        Int32(H_BYTES).ir_value(),  # 拷贝字节数 = 一行 hidden
                        mbar_int.ir_value(),
                    )
                load_state.advance()  # 推进生产者状态到下一级

        # ============================================
        # Warp 1 — S2G consumer (K stores per token + optional weight scatter)
        # ============================================
        elif warp_idx == self.CONSUMER_WARP:  # warp 1：S2G 消费者（每 token K 次存储 + 可选权重散射）
            use_state = pipeline.make_pipeline_state(  # 消费状态：等待 full 信号
                pipeline.PipelineUserType.Consumer, stages
            )
            rel_state = pipeline.make_pipeline_state(  # 释放状态：滞后 stages-1 拍用于限流
                pipeline.PipelineUserType.Consumer, stages
            )
            for li in cutlass.range(n_tok, unroll=1):  # 逐 token 消费
                s = s_beg + li  # 当前全局 token 下标
                sK = s * K  # 该 token 在 [S,K] 布局中的基下标

                load_pipe.consumer_wait(use_state)  # 等待当前级 smem 行数据就绪

                # All K stores + commit + optional weight scatter run on lane 0;
                # 32 lanes would over-decrement the bulk_group / re-scatter.
                if cute.arch.lane_idx() == 0:  # bulk_group 提交计数必须单线程，仅 lane 0 执行
                    s_row_int = (  # 当前级共享内存暂存行起始地址
                        stage_smem.iterator + use_state.index * H
                    ).toint()
                    for k in cutlass.range(K, unroll=1):  # 对该 token 的 K 个 top-k 目的地逐一处理
                        dst_val = dst_tensor[sK + k]  # 读取第 k 个目的地编码
                        # Negative dst keeps the raw destination for the
                        # weight scatter, but marks the payload as duplicate.
                        store_token = dst_val >= Int32(0)  # 非负=需拷贝 payload；负=重复 topk 条目
                        raw_dst = dst_val  # 暂存原始编码
                        if not store_token:  # 重复条目：还原真实目的地 = -dst - 1
                            raw_dst = - dst_val - Int32(1)
                        drank = raw_dst // NvS  # 解码目的 rank 编号
                        loff = raw_dst % NvS  # 解码目的 rank 内槽偏移

                        if store_token:  # 非重复条目：整行直写远端最终位置（零拷贝）
                            drow = drank * NvS_padded + loff  # 目的行全局行号（按填充后步长索引）
                            g_row_int = (  # 目的行全局内存起始地址
                                gmem_dst.iterator
                                + cutlass.Int64(drow) * cutlass.Int64(H)
                            ).toint()
                            cp_async_bulk_s2g(  # TMA 整行拷贝：smem -> 远端 NVL 缓冲
                                s_row_int.ir_value(),
                                g_row_int.ir_value(),
                                Int32(H_BYTES).ir_value(),
                            )

                        if cutlass.const_expr(self.with_weights):  # 编译期开关：是否散射路由权重
                            wb = w_tensor[sK + k]  # 读取该 topk 位置的权重（int32 位视图）
                            meta_tensor[  # 直写目的 rank 的权重槽（fp32 位模式按 int32 写入）
                                drank * meta_stride + weights_off + loff
                            ] = wb
                    cute.arch.cp_async_bulk_commit_group()  # 将本 token 的全部 bulk 存储提交为一个 bulk_group

                use_state.advance()  # 推进消费状态到下一级

                # Throttle in-flight bulk_groups to <= STAGES-1.
                if li >= Int32(stages - 1):  # 起步阶段之后开始限流
                    cute.arch.cp_async_bulk_wait_group(stages - 1)  # 等待在途 bulk_group 数降到 stages-1
                    load_pipe.consumer_release(rel_state)  # 释放对应级缓冲供生产者复用
                    rel_state.advance()  # 推进释放状态

            # Drain the trailing in-flight stores. consumer_release for the
            # last (kStages-1) stages isn't strictly needed since the kernel
            # exits below, but we keep it symmetric so the empty mbars are
            # in a known state if this kernel is replayed.
            cute.arch.cp_async_bulk_wait_group(0)  # 排空全部在途存储

        # ============================================
        # Warp 2 — per-group zero-fill loop (runs concurrently with
        # producer/consumer dispatch). For each group e in this rank's compact
        # layout, planning wrote zero_fill_ranges[e] = (pad_start_loff, n_pad_rows)
        # for the segment-padding rows DeepGEMM will read but the dispatch
        # consumer does not write. Each CTA strides through zero_groups entries; for
        # each non-empty range it issues n_pad_rows back-to-back
        # cp.async.bulk_s2g zeros from zero_smem and, when with_weights is
        # set, a paired int32 zero store into the matching slot of meta_buf
        # (fp32 0.0 shares the all-zero bit pattern, so int32 0 suffices).
        #
        # Each row is written by at most one source (dispatch consumer OR
        # this warp), so the local rank's zero writes do not race with peer
        # NVL writes. cross_rank_barrier below uses grid_sync plus
        # system-scope release/acquire atomics to publish both kinds of writes
        # before peer ranks consume the rows.
        # ============================================
        if warp_idx == self.ZERO_WARP:  # warp 2：按组清零填充行（与生产/消费路径并行）
            issued = Int32(0)  # 本 warp 已提交的 bulk_group 计数（限流用）
            e = Int32(bidx)  # 起始组下标：各 CTA 交错认领（跨 CTA 步进遍历）
            while e < Int32(zero_groups):  # 遍历分配到本 CTA 的清零组
                pad_start = zero_fill_ranges_tensor[e * Int32(2) + Int32(0)]  # 该组填充起始槽偏移
                n_pad = zero_fill_ranges_tensor[e * Int32(2) + Int32(1)]  # 该组需清零的行数
                if n_pad > Int32(0):  # 空区间跳过，不发起拷贝
                    for j in cutlass.range(n_pad, unroll=1):  # 逐行写零
                        loff = pad_start + Int32(j)  # 当前填充行的槽偏移
                        local_row = Int32(rank) * Int32(NvS_padded) + loff  # 本 rank shard 区间内的全局行号
                        if cute.arch.lane_idx() == 0:  # bulk 拷贝仅 lane 0 发出
                            g_row_int = (  # 清零目的行的全局内存地址
                                gmem_dst.iterator
                                + cutlass.Int64(local_row) * cutlass.Int64(H)
                            ).toint()
                            z_smem_int = zero_smem.iterator.toint()  # 零模板行（smem）地址
                            cp_async_bulk_s2g(  # 从零模板 bulk 写出一整行零
                                z_smem_int.ir_value(),
                                g_row_int.ir_value(),
                                Int32(H_BYTES).ir_value(),
                            )
                            cute.arch.cp_async_bulk_commit_group()  # 每行单独提交为一个 bulk_group
                            if cutlass.const_expr(self.with_weights):  # 带权重时成对清零对应权重槽
                                meta_tensor[  # fp32 0.0 与 int32 0 位模式相同，int32 写入即可
                                    Int32(rank) * meta_stride + weights_off + loff
                                ] = Int32(0)
                        if issued >= Int32(stages - 1):  # 限流：在途 bulk_group 数不超过 stages-1
                            cute.arch.cp_async_bulk_wait_group(stages - 1)  # 等待最旧一组完成
                        issued += Int32(1)  # 更新已提交组计数
                e += Int32(self.num_sms)  # 跳到下一个属于本 CTA 的组
            cute.arch.cp_async_bulk_wait_group(0)  # 排空全部在途清零写

        # ============================================
        # Warps 3.. — dedup structure builder (fresh planning only)
        # ============================================
        elif warp_idx >= self.DEDUP_BUILDER_WARP:  # warp 3..：去重结构构建器
            if cutlass.const_expr(self.build_dedup_map):  # 编译期开关：仅新鲜规划路径构建
                lane = cute.arch.lane_idx()  # warp 内 lane 编号
                # Global builder index: DEDUP_BUILDER_WARPS warps per CTA each
                # own one chunk, dividing the latency-bound scan wall time.
                n_builders = cutlass.const_expr(  # 全局构建器 warp 总数 = CTA 数 × 每 CTA 构建 warp 数
                    self.num_sms * DEDUP_BUILDER_WARPS
                )
                gb = (  # 本 warp 的全局构建器编号（0 起，连续编号）
                    bidx * Int32(DEDUP_BUILDER_WARPS)
                    + (warp_idx - self.DEDUP_BUILDER_WARP)
                )
                is_leader = Int32(0)  # 是否为全局屏障的 leader（贡献自复位大增量）
                if gb == 0:  # 仅全局 0 号构建器 warp（CTA0 的 warp3）充当 leader
                    is_leader = Int32(1)
                # Hierarchical phase barrier: builder warps rendezvous inside
                # the CTA on a named barrier, then only the first builder warp
                # of each CTA joins the global release/acquire barrier. This
                # keeps the global participant count at num_sms regardless of
                # DEDUP_BUILDER_WARPS (a flat num_sms*W-warp global barrier
                # measurably ate the multi-warp speedup).
                # Use HW named barrier 8; id 0..7 are reserved by the
                # CUTLASS pipelines / __syncthreads.
                cta_bar = pipeline.NamedBarrier(8, 32 * DEDUP_BUILDER_WARPS)  # CTA 内构建器 warp 的硬件命名屏障

                # Initialize the per-token primary-election scratch every
                # fresh dispatch (a tail-clear scheme was measured neutral and
                # rejected: callers typically reuse one Buffer across many
                # iterations and cannot guarantee the previous dispatch fully
                # completed). The leader also clears the dup_counts output
                # counters that the warp-aggregated atomicAdds below allocate
                # from; phase 1 publishes both.
                seg_pk = cute.ceil_div(R * S, n_builders)  # 每个构建器分摊的 (rank,token) 键数
                sbeg_pk = seg_pk * gb  # 本构建器负责键区间的起点
                send_pk = cutlass.min(sbeg_pk + seg_pk, R * S)  # 键区间终点（不越界）
                for base_pk in cutlass.range(sbeg_pk + lane, send_pk, Int32(32)):  # lane 间跨步 32 遍历
                    primary_packed_tensor[base_pk] = INT32_MAX  # 主选值初始化为"未选中"哨兵
                    kmask_tensor[base_pk] = Uint32(0)  # kidx 掩码清零
                if gb == 0:  # 全局 0 号构建器额外负责清零输出计数器
                    if lane < 2:  # lane0/1 分别清零两个计数器
                        dup_counts_tensor[lane] = Int32(0)

                # Phase 1: init published.
                cta_bar.arrive_and_wait()  # CTA 内：等待初始化写全部完成
                if warp_idx == self.DEDUP_BUILDER_WARP:  # 仅每 CTA 的第一个构建器 warp 参与全局屏障
                    cross_warp_sync(  # 跨 CTA 自复位屏障：num_sms 个参与者，leader 贡献大增量
                        builder_bar_tensor.iterator, num_sms, is_leader
                    )
                cta_bar.arrive_and_wait()  # CTA 内：把全局屏障结果广播给其余构建器 warp

                # Pass 1: elect one primary local slot per (source rank, token)
                # and record the kidx -> loff mapping for duplicate expansion.
                seg_src = cute.ceil_div(NvS, n_builders)  # 每个构建器分摊的本地槽偏移数
                sbeg_src = gb * seg_src  # 本构建器槽区间起点
                send_src = cutlass.min(sbeg_src + seg_src, NvS)  # 槽区间终点
                for loff in cutlass.range(sbeg_src + lane, send_src, Int32(32)):  # lane 跨步遍历本构建器的槽
                    info = meta_tensor[rank * meta_stride + SRC_INFO_OFF + loff]  # 读该槽的来源信息（-1 为空）
                    if info >= 0:  # 空槽跳过
                        src_rank = info // NvS  # 解码来源 rank
                        offv = info - src_rank * NvS  # 解码来源侧线性偏移
                        token = offv // K  # 还原来源 token 下标
                        kidx = offv - token * K  # 还原 top-k 索引
                        key = src_rank * S + token  # (源rank, token) 的线性键
                        packed = (kidx << NvS_BITS) | loff  # 打包 (kidx, 主选槽偏移) 为一个 int32
                        primary_packed_ptr = (  # 主选值槽地址
                            primary_packed_tensor.iterator + key
                        ).toint()
                        atom_min_relaxed_gpu_s32(primary_packed_ptr.ir_value(), packed)  # 原子 min 选主：最小打包值当选
                        kmask_ptr = (  # 掩码槽地址
                            kmask_tensor.iterator + key
                        ).toint()
                        atom_or_relaxed_gpu_b32(  # 原子或：登记该 kidx 已出现
                            kmask_ptr.ir_value(), Uint32(1) << kidx
                        )
                        kidx_to_loff_tensor[key * K + kidx] = loff  # 记录 kidx -> 本地槽偏移映射

                # Phase 2: primary election published.
                cta_bar.arrive_and_wait()  # CTA 内：等待 Pass 1 的原子写完成
                if warp_idx == self.DEDUP_BUILDER_WARP:  # 仅第一个构建器 warp 参与全局屏障
                    cross_warp_sync(  # 跨 CTA 屏障：主选结果全局可见
                        builder_bar_tensor.iterator, num_sms, is_leader
                    )
                cta_bar.arrive_and_wait()  # CTA 内广播

                # Pass 2a (per-lane, straight-line): classify this chunk's
                # slots from the election results and tally each lane's
                # primary-with-duplicates group / duplicate totals in
                # registers. Keeping this pass free of warp-collective ops
                # lets the compiler pipeline the dependent gmem loads (info
                # -> election) across iterations; keep the classification in
                # sync with Pass 2b below.
                lane_grp_n = Int32(0)  # 本 lane 名下的去重组计数
                lane_dup_n = Int32(0)  # 本 lane 名下的重复槽总数
                for loff in cutlass.range(sbeg_src + lane, send_src, Int32(32)):  # 与 Pass 1 相同的槽遍历
                    info = meta_tensor[rank * meta_stride + SRC_INFO_OFF + loff]  # 读来源信息
                    if info >= 0:  # 空槽跳过
                        src_rank = info // NvS  # 解码来源 rank
                        offv = info - src_rank * NvS  # 解码来源线性偏移
                        token = offv // K  # 还原 token 下标
                        key = src_rank * S + token  # (源rank, token) 线性键
                        packed = primary_packed_tensor[key]  # 读主选结果（打包值）
                        mask = Uint32(kmask_tensor[key])  # 读该键的 kidx 占用掩码
                        primary_loff = packed & NvS_MASK  # 解出主选槽偏移
                        dup_count = popc_b32(mask) - 1  # 重复数 = 掩码中 1 的个数 - 1（主选自身）

                        if loff == primary_loff:  # 仅主选槽负责登记其去重组
                            if dup_count > Int32(0):  # 无重复的主选不产生组
                                lane_grp_n += Int32(1)  # 本 lane 累计一个组
                                lane_dup_n += dup_count  # 累计该组的重复槽数

                # Per-lane exclusive offsets within this warp's output range,
                # then one warp-aggregated atomicAdd per output array reserves
                # the warp's compact prefix range (2 atomics per warp,
                # 2*num_sms*DEDUP_BUILDER_WARPS total — a per-record atomicAdd
                # scheme serialized and dominated dispatch latency in the
                # pieces-era builder).
                incl_grp = warp_inclusive_scan(lane_grp_n, lane)  # warp 内组数前缀和（含本 lane）
                incl_dup = warp_inclusive_scan(lane_dup_n, lane)  # warp 内重复数前缀和
                grp_total = cute.arch.shuffle_sync(incl_grp, Int32(31))  # lane31 的前缀和 = 全 warp 组总数
                dup_total = cute.arch.shuffle_sync(incl_dup, Int32(31))  # 全 warp 重复槽总数
                lane_grp_off = incl_grp - lane_grp_n  # 本 lane 在 warp 输出区间内的排他偏移（组）
                lane_dup_off = incl_dup - lane_dup_n  # 本 lane 的排他偏移（重复槽）

                grp_base_w = Int32(0)  # 本 warp 在 dup_groups 中的保留区间基址
                dup_base_w = Int32(0)  # 本 warp 在 dup_loffs 中的保留区间基址
                if lane == 0:  # 每 warp 仅 lane 0 发起聚合原子加
                    grp_count_ptr = (  # dup_counts[0]：组计数器地址
                        dup_counts_tensor.iterator + 0
                    ).toint()
                    grp_base_w = atom_add_relaxed_gpu_s32(  # 原子加预留本 warp 的组前缀区间
                        grp_count_ptr.ir_value(), grp_total
                    )
                    dup_count_ptr = (  # dup_counts[1]：重复槽计数器地址
                        dup_counts_tensor.iterator + 1
                    ).toint()
                    dup_base_w = atom_add_relaxed_gpu_s32(  # 原子加预留本 warp 的重复槽区间
                        dup_count_ptr.ir_value(), dup_total
                    )
                grp_base_w = cute.arch.shuffle_sync(grp_base_w, Int32(0))  # lane0 的预留基址广播到全 warp
                dup_base_w = cute.arch.shuffle_sync(dup_base_w, Int32(0))  # 同上（重复槽基址）

                # Pass 2b (per-lane, straight-line): re-classify and emit
                # dup_groups / dup_loffs at the reserved positions. The
                # compact-prefix order across warps follows atomicAdd arrival
                # order and is NOT stable run-to-run; consumers iterate by
                # index and tests must compare group sets, not element order.
                my_grp = grp_base_w + lane_grp_off  # 本 lane 下一个组在 dup_groups 中的写入位置
                my_dup = dup_base_w + lane_dup_off  # 本 lane 下一个重复槽在 dup_loffs 中的写入位置
                for loff in cutlass.range(sbeg_src + lane, send_src, Int32(32)):  # 再次遍历本构建器的槽
                    info = meta_tensor[rank * meta_stride + SRC_INFO_OFF + loff]  # 读来源信息
                    if info >= 0:  # 空槽跳过
                        src_rank = info // NvS  # 解码来源 rank
                        offv = info - src_rank * NvS  # 解码来源线性偏移
                        token = offv // K  # 还原 token
                        key = src_rank * S + token  # (源rank, token) 线性键
                        packed = primary_packed_tensor[key]  # 读主选结果
                        mask = Uint32(kmask_tensor[key])  # 读 kidx 掩码
                        primary_kidx = packed >> NvS_BITS  # 解出主选 kidx（高字段）
                        primary_loff = packed & NvS_MASK  # 解出主选槽偏移（低字段）
                        dup_count = popc_b32(mask) - 1  # 重复槽数

                        if loff == primary_loff:  # 主选槽负责输出该组
                            if dup_count > Int32(0):  # 有重复才输出
                                dup_group_key = my_grp * 3  # 该组在展平 dup_groups 中的三元组基下标
                                dup_groups_tensor[dup_group_key] = loff  # 写组头：主选槽偏移
                                dup_groups_tensor[dup_group_key + 1] = my_dup  # 写组头：重复槽在 dup_loffs 的起始
                                dup_groups_tensor[dup_group_key + 2] = dup_count  # 写组头：重复槽数
                                key_row = key * K  # 该键的 kidx_to_loff 行基下标
                                dup_mask = mask & (~(Uint32(1) << primary_kidx))  # 掩码去掉主选位，仅剩重复位
                                pos = Int32(0)  # 组内写入游标
                                while dup_mask != Uint32(0):  # 逐位展开重复槽
                                    cur_dup_kidx = ctz_b32(dup_mask)  # 最低置位位 = 下一个重复 kidx
                                    cur_dup_loff = kidx_to_loff_tensor[  # 查映射得到重复槽的本地偏移
                                        key_row + cur_dup_kidx
                                    ]
                                    dup_loffs_tensor[my_dup + pos] = cur_dup_loff  # 写入紧凑重复槽表
                                    pos += Int32(1)  # 前进游标
                                    dup_mask = dup_mask & (dup_mask - Uint32(1))  # 清除最低置位位（b->b-1 技巧）
                                my_grp += Int32(1)  # 前进组写入位置
                                my_dup += dup_count  # 前进重复槽写入位置

        # ----- exit barrier: dispatch wrote NVL hidden_buf / weight slots.
        # cross_rank_barrier uses grid_sync plus system-scope release/acquire
        # atomics to publish the whole grid's writes to peer ranks before the
        # local epilogue reads those rows.
        cross_rank_barrier(  # 退出屏障：把本网格对 NVL 数据/权重槽的写发布给对端 rank
            meta_tensor, meta_stride, barrier_off, rank, num_ranks,
            bar_tensor.iterator, Int32(self.num_sms), num_threads, tidx,
        )
        if cutlass.const_expr(self.pdl_trigger):  # 编译期开关：PDL 触发依赖网格
            pdl_trigger_dependents(tidx)  # 发布写入并允许依赖内核提前启动


# ============================================================================
# Host launcher with per-shape compile cache
# ============================================================================

@functools.lru_cache(maxsize=None)  # 按设备缓存查询结果（设备属性不变）
def _max_smem_per_block_optin(device_index: int) -> int:  # 查询每 CTA 可选用的最大动态共享内存
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin  # 读取 CUDA 设备属性


@functools.lru_cache(maxsize=None)  # 按 (形状, 选项) 缓存编译后的内核
def _get_compiled(  # 编译并缓存 dispatch 内核（相同参数只编译一次）
    H: int,
    R: int,
    S: int,
    K: int,
    zero_groups: int,
    NvS: int,
    NvS_padded: int,
    meta_stride: int,
    SRC_INFO_OFF: int,
    num_sms: int,
    with_weights: bool,
    build_dedup_map: bool,
    device_index: int,
    pdl_trigger: bool,
):
    smem_budget = _max_smem_per_block_optin(device_index) - 1024  # 预留 1KB 余量后的共享内存预算
    kernel = DispatchKernel(  # 构造内核对象（内含级数挑选）
        H=H, R=R, S=S, K=K, zero_groups=zero_groups,
        NvS=NvS, NvS_padded=NvS_padded,
        SRC_INFO_OFF=SRC_INFO_OFF, meta_stride=meta_stride,
        num_sms=num_sms,
        with_weights=with_weights,
        build_dedup_map=build_dedup_map,
        smem_budget=smem_budget,
        pdl_trigger=pdl_trigger,
    )

    bf16_ptr = make_ptr(BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)  # 占位 bf16 指针（仅定类型/对齐）
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)  # 占位 int32 指针
    u32_ptr = make_ptr(Uint32, 0, cute.AddressSpace.gmem, assumed_align=16)  # 占位 uint32 指针
    stream_arg = cuda.CUstream(0)  # 占位流参数（编译期只看类型）

    return cute.compile(  # 按 Buffer 类型和形状特化编译内核
        kernel,
        bf16_ptr,  # hidden_sh
        bf16_ptr,  # hidden_buf
        i32_ptr,   # weights (int32 view of fp32)
        i32_ptr,   # dst
        i32_ptr,   # meta
        i32_ptr,   # zero_fill_ranges
        i32_ptr,   # bar (grid barrier counter)
        i32_ptr,   # primary_packed
        u32_ptr,   # kmask
        i32_ptr,   # kidx_to_loff
        i32_ptr,   # dup_groups
        i32_ptr,   # dup_loffs
        i32_ptr,   # dup_counts
        i32_ptr,   # builder_bar
        Int32(0),  # rank
        Int32(0),  # weights_off
        Int32(0),  # barrier_off
        stream_arg,
    )


# 校验去重构建器的位宽/容量约束（打包编码不溢出）。
def _check_dedup_builder_bounds(ctx: dict) -> None:
    R = int(ctx['R'])  # rank 数
    S = int(ctx['S'])  # token 数
    K = int(ctx['K'])  # top-k 值
    NvS = int(ctx['NvS'])  # 每 rank 有效槽位
    NvS_BITS = 32 - 1 - KIDX_BITS  # 打包值中槽偏移字段的位宽
    N = S * K  # 每 rank 的 topk 槽总数
    int32_max = 2**31 - 1  # int32 上限
    assert N <= NvS, (  # src_info 的 NvS 步长编码要求 S*K 不超过 NvS
        f"src_info NvS-stride encoding requires S*K <= NvS, got S*K={N}, NvS={NvS}"
    )
    assert R * NvS <= int32_max, (  # src_info 线性编码要求 R*NvS 不溢出 int32
        "src_info linear encoding requires R*NvS <= int32_max: "
        f"R={R}, NvS={NvS}, R*NvS={R * NvS}, int32_max={int32_max}"
    )
    assert K <= (1 << KIDX_BITS) - 1, (  # packed 打包要求 K 放得进 kidx 字段
        f"primary_packed encoding requires K <= "
        f"{(1 << KIDX_BITS) - 1}, got {K}"
    )
    assert NvS <= (1 << NvS_BITS) - 1, (  # packed 打包要求 NvS 放得进槽偏移字段
        f"primary_packed encoding requires NvS <= "
        f"{(1 << NvS_BITS) - 1}, got {NvS}"
    )
    assert K <= 32, f"kmask bitmask requires K <= 32, got {K}"  # kmask 是单个 32 位字，K 不能超过 32


# 校验 plan 携带的 dispatch 相关张量的 dtype/形状/设备一致性。
def _check_dispatch_plan(ctx: dict, hidden_sh: torch.Tensor, plan: MoonEPCommPlan) -> None:
    S = int(ctx['S'])  # token 数
    K = int(ctx['K'])  # top-k 值
    N = S * K  # 每 rank topk 槽数
    R = int(ctx['R'])  # rank 数
    E = int(ctx['E'])  # 专家总数
    epn = E // R  # 每 rank 专家数
    NvS = int(ctx['NvS'])  # 每 rank 有效槽位
    dev = hidden_sh.device  # 以输入 hidden 的设备为基准

    assert plan.N == N, f"plan.N must be S*K={N}, got {plan.N}"  # 计划的 N 必须等于 S*K
    assert plan.R == R, f"plan.R must match ctx R={R}, got {plan.R}"  # rank 数一致
    assert plan.K == K, f"plan.K must match ctx K={K}, got {plan.K}"  # top-k 一致
    assert plan.NvS == NvS, f"plan.NvS must match ctx NvS={NvS}, got {plan.NvS}"  # 槽位一致

    def _check_tensor(t: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:  # 通用张量检查工具
        # 必须是连续 int32
        assert t.dtype == torch.int32 and t.is_contiguous(), \
            f"{name} must be contiguous int32"
        # 形状必须精确匹配
        assert tuple(t.shape) == shape, \
            f"{name} must be shape {shape}, got {tuple(t.shape)}"
        # 必须与 hidden 同设备
        assert t.device == dev, \
            f"{name} must be on {dev}, got {t.device}"

    _check_tensor(plan.dst, "dst", (N,))  # 校验目的地编码数组
    _check_tensor(plan.zero_fill_ranges, "zero_fill_ranges", (2 * epn, 2))  # 校验清零区间表
    _check_tensor(plan.dup_groups, "dup_groups", (NvS, 3))  # 校验去重组头表
    _check_tensor(plan.dup_loffs, "dup_loffs", (NvS,))  # 校验重复槽表
    _check_tensor(plan.dup_counts, "dup_counts", (2,))  # 校验去重计数器


# 校验构建器划痕张量存在于 ctx 且 dtype/大小/设备正确。
def _check_dedup_builder_tensors(ctx: dict, dev: torch.device) -> None:
    R = int(ctx['R'])  # rank 数
    S = int(ctx['S'])  # token 数
    K = int(ctx['K'])  # top-k 值

    def _check_scratch(name: str, numel: int) -> None:  # 通用划痕区检查工具
        assert name in ctx, f"ctx missing {name}"  # 必须存在
        t = ctx[name]  # 取出张量
        # 连续 int32
        assert t.dtype == torch.int32 and t.is_contiguous(), \
            f"{name} must be contiguous int32"
        # 元素数精确匹配
        assert t.numel() == numel, \
            f"{name} must have {numel} elements, got {t.numel()}"
        # 设备一致
        assert t.device == dev, \
            f"{name} must be on {dev}, got {t.device}"

    _check_scratch("primary_packed", R * S)  # 主选打包值划痕区
    _check_scratch("kmask", R * S)  # kidx 掩码划痕区
    _check_scratch("kidx_to_loff", R * S * K)  # kidx->偏移映射划痕区
    _check_scratch("builder_bar", 1)  # 构建器屏障计数器


# dispatch 的 host 侧启动入口：校验输入、取编译缓存并填充指针启动内核。
def launch_dispatch(
    ctx: dict,
    hidden_sh,
    route_weights_sk,
    plan,
    *,
    build_dedup_map: bool = True,
    pdl_trigger: bool = False,
):
    """Launch the dispatch kernel.

    Args:
        hidden_sh: [S, H] bf16 source hidden states.
        route_weights_sk: [S, K] fp32 route weights, or None to skip the
            weights scatter (placeholder tensor is passed to satisfy the
            non-null pointer constraint; the kernel ignores it when
            with_weights=False).
        plan: communication plan carrying ``dst``, ``zero_fill_ranges`` and
            the plan-owned dedup structures. Non-negative ``dst`` entries
            encode ``dest_rank * NvS + local_offset`` and copy the payload.
            Negative entries encode ``-raw_dst - 1`` and only scatter the
            corresponding weight.
        build_dedup_map: true only immediately after fresh planning. Reuse and
            backward paths pass false so the saved dedup structures
            (``dup_groups`` / ``dup_loffs`` / ``dup_counts``) are not rebuilt
            from stale ``src_info`` scratch. The zero warp runs on both paths.

    The in-place duplicate expansion on the NVL shard is a separate kernel —
    call ``moonep.dispatch_epilogue.launch_dispatch_epilogue`` afterwards on the same
    stream.
    """
    assert isinstance(plan, MoonEPCommPlan)  # plan 必须是通信计划类型
    with_weights = route_weights_sk is not None  # 传入权重则启用权重散射
    if not with_weights:  # 无权重时
        # Placeholder; kernel will not dereference.
        route_weights_sk = plan.dst  # 用 dst 张量充当占位指针（内核不解引用）

    H = int(ctx['H'])  # 隐藏维度
    R = int(ctx['R'])  # rank 数
    S = int(ctx['S'])  # token 数
    K = int(ctx['K'])  # top-k 值
    E = int(ctx['E'])  # 专家总数
    epn = E // R  # 每 rank 专家数

    # 输入必须是连续 bf16
    assert hidden_sh.dtype == torch.bfloat16 and hidden_sh.is_contiguous(), \
        "hidden_sh must be contiguous bf16"
    assert hidden_sh.is_cuda, "hidden_sh must be a CUDA tensor"  # 必须在 CUDA 上
    # 形状必须为 [S, H]
    assert tuple(hidden_sh.shape) == (S, H), \
        f"hidden_sh must be shape [S={S}, H={H}], got {tuple(hidden_sh.shape)}"
    _check_dispatch_plan(ctx, hidden_sh, plan)  # 校验 plan 张量
    assert ctx['hidden_buf'].dtype == torch.bfloat16 and ctx['hidden_buf'].is_contiguous()  # NVL 数据缓冲：连续 bf16
    assert ctx['hidden_buf'].device == hidden_sh.device  # 与输入同设备
    assert ctx['meta_buf'].dtype == torch.int32 and ctx['meta_buf'].is_contiguous()  # meta 缓冲：连续 int32
    assert ctx['meta_buf'].device == hidden_sh.device  # 同设备
    assert ctx['grid_sync_bar'].dtype == torch.int32 and ctx['grid_sync_bar'].is_contiguous()  # 网格屏障计数器：连续 int32
    assert ctx['grid_sync_bar'].numel() == 1  # 恰好一个元素
    assert ctx['grid_sync_bar'].device == hidden_sh.device  # 同设备
    if with_weights:  # 带权重路径的额外校验
        # 必须是连续 fp32
        assert route_weights_sk.dtype == torch.float32 and route_weights_sk.is_contiguous(), \
            "route_weights_sk must be contiguous fp32"
        # 形状必须为 [S, K]
        assert tuple(route_weights_sk.shape) == (S, K), \
            f"route_weights_sk must be shape [S={S}, K={K}], got {tuple(route_weights_sk.shape)}"
        assert route_weights_sk.device == hidden_sh.device  # 同设备
    assert ctx['H'] % 8 == 0, "H must be multiple of 8 for 16-B bulk-copy alignment"  # H 须为 8 的倍数保证 16B 对齐
    if build_dedup_map:  # 新鲜规划路径才需要构建器资源
        _check_dedup_builder_bounds(ctx)  # 位宽/容量约束
        _check_dedup_builder_tensors(ctx, hidden_sh.device)  # 划痕张量校验

    NvS = int(ctx['NvS'])  # 每 rank 有效槽位
    NvS_padded = int(ctx['NvS_padded'])  # 填充后槽位
    meta_stride = int(ctx['meta_chunk_padded'])  # meta 每 rank 步长
    SRC_INFO_OFF = int(ctx['SRC_INFO_OFF'])  # src_info 划痕区偏移
    num_sms = int(ctx['num_sms'])  # 网格 CTA 数
    device_index = hidden_sh.device.index  # CUDA 设备编号（编译缓存键之一）

    dispatch_compiled = _get_compiled(  # 取（或编译）本形状特化的内核
        H, R, S, K, 2 * epn, NvS, NvS_padded, meta_stride, SRC_INFO_OFF,
        num_sms, with_weights, build_dedup_map, device_index,
        bool(pdl_trigger),
    )

    # int32 view onto fp32 weights so the kernel always sees an int32 ptr
    # (matches the C++ approach: a 4-byte gather, no fp32 arithmetic).
    w_int = (  # 把 fp32 权重视为 int32，内核按 4 字节整数搬运
        route_weights_sk.view(torch.int32) if with_weights else route_weights_sk
    )

    hsh_ptr = make_ptr(BFloat16, hidden_sh.data_ptr(), cute.AddressSpace.gmem,  # 源 hidden 指针（16B 对齐假设）
                       assumed_align=16)
    hbf_ptr = make_ptr(BFloat16, ctx['hidden_buf'].data_ptr(),  # 目的 NVL 数据缓冲指针
                       cute.AddressSpace.gmem, assumed_align=16)
    w_ptr = make_ptr(Int32, w_int.data_ptr(), cute.AddressSpace.gmem,  # 权重（int32 视图）指针
                     assumed_align=16)
    dst_ptr = make_ptr(Int32, plan.dst.data_ptr(), cute.AddressSpace.gmem,  # 目的地编码数组指针
                      assumed_align=16)
    meta_ptr = make_ptr(Int32, ctx['meta_buf'].data_ptr(),  # meta 缓冲指针
                        cute.AddressSpace.gmem, assumed_align=16)
    zfr_ptr = make_ptr(Int32, plan.zero_fill_ranges.data_ptr(),  # 清零区间表指针（8B 对齐足够）
                       cute.AddressSpace.gmem, assumed_align=8)
    bar_ptr = make_ptr(Int32, ctx['grid_sync_bar'].data_ptr(),  # 网格屏障计数器指针
                       cute.AddressSpace.gmem, assumed_align=16)

    # Builder scratch is only dereferenced on fresh-planning launches; the
    # reuse path passes a harmless placeholder to satisfy the pointer args.
    primary_packed = ctx['primary_packed'] if build_dedup_map else plan.dst  # 复用路径用 dst 充当占位划痕区
    kmask = ctx['kmask'] if build_dedup_map else plan.dst  # 同上（掩码划痕区）
    kidx_to_loff = ctx['kidx_to_loff'] if build_dedup_map else plan.dst  # 同上（映射划痕区）
    builder_bar = ctx['builder_bar'] if build_dedup_map else plan.dst  # 同上（屏障计数器）
    primary_packed_ptr = make_ptr(Int32, primary_packed.data_ptr(),  # 主选划痕区指针
                                  cute.AddressSpace.gmem, assumed_align=16)
    kmask_ptr = make_ptr(Uint32, kmask.data_ptr(),  # 掩码划痕区指针（uint32 类型）
                         cute.AddressSpace.gmem, assumed_align=16)
    kidx_to_loff_ptr = make_ptr(Int32, kidx_to_loff.data_ptr(),  # kidx->偏移映射指针
                                cute.AddressSpace.gmem, assumed_align=16)
    builder_bar_ptr = make_ptr(Int32, builder_bar.data_ptr(),  # 构建器屏障计数器指针
                               cute.AddressSpace.gmem, assumed_align=16)

    dup_groups_ptr = make_ptr(Int32, plan.dup_groups.data_ptr(),  # 去重组头表指针
                              cute.AddressSpace.gmem, assumed_align=16)
    dup_loffs_ptr = make_ptr(Int32, plan.dup_loffs.data_ptr(),  # 重复槽表指针
                             cute.AddressSpace.gmem, assumed_align=16)
    dup_counts_ptr = make_ptr(Int32, plan.dup_counts.data_ptr(),  # 去重计数器指针
                              cute.AddressSpace.gmem, assumed_align=8)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)  # 取当前 PyTorch 流封装为 CUstream

    dispatch_compiled(  # 调用编译产物：填充真实指针并启动内核
        hsh_ptr,
        hbf_ptr,
        w_ptr,
        dst_ptr,
        meta_ptr,
        zfr_ptr,
        bar_ptr,
        primary_packed_ptr,
        kmask_ptr,
        kidx_to_loff_ptr,
        dup_groups_ptr,
        dup_loffs_ptr,
        dup_counts_ptr,
        builder_bar_ptr,
        Int32(int(ctx['rank'])),  # 本 rank 编号
        Int32(int(ctx['WEIGHTS_OFF'])),  # 权重槽在 meta 内的偏移
        Int32(int(ctx['BARRIER_OFF'])),  # 屏障槽在 meta 内的偏移
        stream,
    )
