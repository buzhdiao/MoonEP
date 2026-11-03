"""
MoonEP Dispatch Epilogue — CuTe DSL implementation.

In-place duplicate expansion on the local NVL shard after dispatch: for each
duplicate group, the primary row (the only copy dispatch sent over NVLink) is
staged once through smem and stored to every duplicate slot of the same
shard. No user tensor is involved — the ``zero_copy=False`` boundary copies
the whole shard afterwards with a plain ``tensor.copy_`` (master-style), and
``zero_copy=True`` returns the shard view directly. Padding rows are already
zero-filled by the dispatch zero warp (``plan.zero_fill_ranges``), and the
per-topk route weights are already scattered by the dispatch consumer.

Runs entirely on local memory — no cross-rank communication; it must be
launched after the dispatch kernel on the same stream (dispatch's exit
cross_rank_barrier publishes the remote NVL writes it reads). Its outputs are
consumed locally (GEMM) or published later by combine's entry barrier, so it
needs no cross-rank barrier of its own.
"""

import functools  # 标准库：lru_cache 缓存编译产物

import torch  # PyTorch：张量校验与 CUDA 流
import cuda.bindings.driver as cuda  # CUDA driver 绑定：CUstream 类型

import cutlass  # CUTLASS DSL 根模块（const_expr 等编译期工具）
import cutlass.cute as cute  # CuTe：布局/张量/内核 DSL
import cutlass.pipeline as pipeline  # CuTe 流水线：TMA mbarrier 同步
import cutlass.utils as utils  # CuTe 工具：SmemAllocator
from cutlass import BFloat16, Int32, Int64  # DSL 标量类型
from cutlass.cute.runtime import make_ptr  # 由裸指针构造 DSL Pointer

from moonep._common import (  # 共享底层原语
    cp_async_bulk_g2s,  # PTX cp.async.bulk：全局 -> 共享整行拷贝
    cp_async_bulk_s2g,  # PTX cp.async.bulk：共享 -> 全局整行拷贝
    pdl_wait_predecessor,  # PDL：等待前驱网格完成并刷写
)
from moonep.planning import MoonEPCommPlan  # 通信计划类型（携带去重结构）


# ============================================================================
# Dispatch epilogue kernel
# ============================================================================

# dispatch 尾处理内核：在本地 NVL shard 上原位展开重复行（纯本地，无跨 rank 通信）。
class DispatchEpilogueKernel:
    """Round-robin duplicate-group expansion on the local NVL shard.

    Two warps per CTA mirror the dispatch data path: warp 0 stages a batch of
    ``B`` primary rows G2S into each smem pipeline stage (one mbarrier
    handshake per batch), warp 1 stores every staged row S2G to all duplicate
    slots of its group (the primary row is read from gmem exactly once
    however many copies it fans out to).

    The plan-owned dedup structures drive all work: ``dup_groups`` lists the
    ``(primary_loff, dup_start, dup_n)`` headers, ``dup_loffs`` the compact
    flat duplicate slots, and ``dup_counts[0]`` the valid ``dup_groups``
    prefix length (read on device — no host synchronization).

    Blocks take group batches round-robin (batch ``i`` = groups
    ``[i*B, i*B+B)`` -> block ``i % num_sms``), so the launch grid must
    equal ``num_sms``.
    """

    num_threads = 64  # 每 CTA 线程数：2 个专职 warp
    PRODUCER_WARP = 0  # warp 0：G2S 生产者（成批暂存主选行）
    CONSUMER_WARP = 1  # warp 1：S2G 消费者（扇出到全部重复槽）

    # Handshake amortization dominates and pipeline depth beyond 2 is
    # measurably irrelevant, so pick the largest per-stage group batch B
    # that still double-buffers.
    _B_CANDIDATES = (32, 16, 8, 4, 2, 1)  # 每级成批装载的组数候选（从大到小尝试）
    _MIN_STAGES = 2  # 最少需要 2 级以支持双缓冲

    # 构造函数：保存形状参数并挑选 (批大小 B, 流水线级数) 几何参数。
    def __init__(
        self,
        H: int,
        NvS: int,
        num_sms: int,
        smem_budget: int,
        pdl_launch: bool,
    ):
        self.H = H  # 隐藏维度（每行 bf16 元素数）
        self.NvS = NvS  # 每 rank NVL shard 有效行数
        self.num_sms = num_sms  # 网格 CTA 数（round-robin 分批的前提）
        self.pdl_launch = pdl_launch  # 是否以 PDL 方式启动（等待前驱）
        self.B, self.stages = self._pick_geometry(H, smem_budget)  # 选出批大小与级数
        # Consumer reads the B batch headers with one lane each + shuffle.
        assert self.B <= 32, "batch size B must fit in one warp"  # 每组头由一个 lane 读取，B 不能超过 32
        if self.stages == 0:  # 级数为 0 表示放不进预算
            raise RuntimeError(  # 快速失败并报告所需最小共享内存
                f"dispatch epilogue: H={H} too large for per-block smem budget "
                f"{smem_budget} B (need at least {self._smem_bytes(H, 2, 1)} B)"
            )

    @staticmethod  # 静态方法：纯计算
    def _smem_bytes(H: int, stages: int, B: int) -> int:  # 计算给定 (级数, 批大小) 的共享内存用量
        # stage_smem (bf16, B rows x kStages deep) + 2 mbar i64 per stage
        # (full+empty), padded like the dispatch kernel, plus 256 B cutlass
        # headroom.
        def _round_up(n: int, a: int) -> int:  # 内部工具：向上取整到 a 的倍数
            return (n + a - 1) // a * a  # 标准向上取整公式
        return (  # 各部分按对齐求和再加余量
            _round_up(stages * B * H * 2, 128)  # stage_smem：每级 B 行 × H×2 字节（bf16），128B 对齐
            + _round_up(stages * 2 * 8, 16)  # 每级 full+empty 两个 mbarrier，16B 对齐
            + 256  # cutlass 内部静态共享内存余量
        )

    @classmethod  # 类方法
    def _pick_stages(cls, H: int, smem_budget: int, B: int) -> int:  # 在给定批大小下选预算内最深级数
        for s in (16, 14, 12, 10, 8, 6, 4, 2):  # 从深到浅尝试候选级数
            if cls._smem_bytes(H, s, B) <= smem_budget:  # 第一个满足预算的级数
                return s  # 返回选中级数
        return 0  # 全部超预算返回 0

    @classmethod  # 类方法
    def _pick_geometry(cls, H: int, smem_budget: int) -> tuple[int, int]:  # 选 (B, stages) 组合：优先大 B
        for b in cls._B_CANDIDATES:  # 从最大批大小开始
            s = cls._pick_stages(H, smem_budget, b)  # 该批大小下可行的级数
            if s >= cls._MIN_STAGES:  # 至少能双缓冲即采纳
                return b, s  # 返回 (批大小, 级数)
        return 1, cls._pick_stages(H, smem_budget, 1)  # 退化：批大小 1（可能级数仍为 0，由调用方报错）

    # ------------------------------------------------------------------ host

    @cute.jit  # 标记为 CuTe JIT 编译的 host 侧入口
    def __call__(  # 内核调用入口：构造张量视图并启动内核
        self,
        hidden_buf_local_ptr: cute.Pointer,  # bf16 [NvS, H] local NVL shard
        dup_groups_ptr: cute.Pointer,        # int32 [NvS, 3]
        dup_loffs_ptr: cute.Pointer,         # int32 [NvS,]
        dup_counts_ptr: cute.Pointer,        # int32 [2,]
        stream: cuda.CUstream,
    ):
        H = cutlass.const_expr(self.H)  # 编译期常量：隐藏维度
        NvS = cutlass.const_expr(self.NvS)  # 编译期常量：每 rank 有效行数

        hidden = cute.make_tensor(  # 本地 NVL shard 的行主序张量视图 [NvS, H]
            hidden_buf_local_ptr,
            cute.make_layout((NvS, H), stride=(Int64(H), Int64(1))),
        )
        dup_groups_tensor = cute.make_tensor(  # 去重组头表展平视图 [NvS*3]
            dup_groups_ptr, cute.make_layout((NvS * 3,))
        )
        dup_loffs_tensor = cute.make_tensor(  # 紧凑重复槽偏移表视图 [NvS]
            dup_loffs_ptr, cute.make_layout((NvS,))
        )
        dup_counts_tensor = cute.make_tensor(  # 去重计数器视图 [2]
            dup_counts_ptr, cute.make_layout((2,))
        )

        smem_bytes = self._smem_bytes(H, self.stages, self.B)  # 最终动态共享内存用量

        self.kernel(  # 以张量参数实例化设备内核
            hidden,
            dup_groups_tensor,
            dup_loffs_tensor,
            dup_counts_tensor,
        ).launch(  # 启动内核
            grid=(self.num_sms, 1, 1),  # 网格 = num_sms 个 CTA（round-robin 分批依赖此值）
            block=(self.num_threads, 1, 1),  # 每 CTA 两个专职 warp
            smem=smem_bytes,  # 动态共享内存字节数
            stream=stream,  # 在调用方指定的 CUDA 流上启动
            use_pdl=self.pdl_launch,  # 是否启用 PDL（与 dispatch 重叠启动）
        )

    # ----------------------------------------------------------------- kernel

    @cute.kernel  # 标记为 CuTe 设备内核
    def kernel(  # 设备侧内核：warp 0 批量暂存主选行，warp 1 扇出到重复槽
        self,
        hidden: cute.Tensor,
        dup_groups_tensor: cute.Tensor,
        dup_loffs_tensor: cute.Tensor,
        dup_counts_tensor: cute.Tensor,
    ):
        H = cutlass.const_expr(self.H)  # 编译期常量：隐藏维度
        H_BYTES = cutlass.const_expr(H * 2)  # 一行 hidden 的字节数（bf16）
        stages = cutlass.const_expr(self.stages)  # 编译期常量：流水线级数
        num_sms = cutlass.const_expr(self.num_sms)  # 编译期常量：CTA 数
        B = cutlass.const_expr(self.B)  # 编译期常量：每级成批装载的组数

        if cutlass.const_expr(self.pdl_launch):  # 编译期开关：PDL 启动时
            pdl_wait_predecessor()  # 等待前驱网格（dispatch）完成并刷写其写操作

        bidx, _, _ = cute.arch.block_idx()  # 当前 CTA 的网格编号
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 归一化 warp 编号（全 warp 一致）

        # ----- shared memory layout
        smem = utils.SmemAllocator()  # 动态共享内存分配器
        load_mbar = smem.allocate_array(Int64, num_elems=2 * stages)  # 每级 full+empty 两个 mbarrier 存储
        stage_smem = smem.allocate_tensor(  # 多级批缓冲 [B, H, stages]（bf16，128B 对齐）
            BFloat16,
            cute.make_ordered_layout((B, H, stages), order=(1, 0, 2)),  # 行内连续、批间跨行、级间再跨批
            byte_alignment=128,
        )

        # Load pipeline (TMA G2S):
        #   Producer = warp 0 (1 effective thread, lane 0).
        #   Consumer = warp 1 (1 effective thread, lane 0).
        load_pipe = pipeline.PipelineTmaAsync.create(  # 创建 TMA 异步 G2S 流水线
            barrier_storage=load_mbar,  # mbarrier 存储指向已分配共享内存
            num_stages=stages,  # 流水线深度
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),  # 生产者组：1 个有效线程
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),  # 消费者组：1 个有效线程
            tx_count=0,  # 由生产者按批动态 arrive_and_expect_tx，这里置 0
        )

        n_groups = dup_counts_tensor[0]  # 有效去重组数（设备侧读取，避免 host 同步）

        # ============================================
        # Warp 0 — G2S producer: one primary row per pipeline stage
        # ============================================
        if warp_idx == self.PRODUCER_WARP:  # warp 0：G2S 生产者
            load_state = pipeline.make_pipeline_state(  # 初始化生产者流水线状态
                pipeline.PipelineUserType.Producer, stages
            )
            gi = Int32(bidx * B)  # 本 CTA 首个批次的起始组号（round-robin）
            lane = cute.arch.lane_idx()  # warp 内 lane 编号
            while gi < n_groups:  # 逐批处理分配到本 CTA 的组
                n = Int32(B)  # 本批默认 B 个组
                if gi + n > n_groups:  # 末批不足 B 时
                    n = n_groups - gi  # 截断到剩余组数

                load_pipe.sync_object_empty.wait(  # 等待当前级缓冲被消费者释放（empty 信号）
                    load_state.index, load_state.phase
                )
                # cp.async.bulk is single-thread; only lane 0 issues it.
                if lane == 0:  # mbarrier 与 bulk 指令均单线程发出
                    mbar = load_pipe.producer_get_barrier(load_state)  # 当前级 full mbarrier
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, n * H_BYTES)  # 预告本批期望到达的字节数

                    for idx in cutlass.range(n, unroll=1):  # 逐组发起装载
                        loff = dup_groups_tensor[(gi + idx) * 3]  # 该组主选行的槽偏移（组头第 0 列）
                        g_row_int = (  # 主选行在全局内存的起始地址
                            hidden.iterator + Int64(loff) * Int64(H)
                        ).toint()
                        s_row_int = (  # 本级 smem 批内第 idx 行的起始地址
                            stage_smem.iterator + load_state.index * B * H + idx * H
                        ).toint()
                        cp_async_bulk_g2s(  # TMA 整行拷贝：主选行 gmem -> smem
                            s_row_int.ir_value(),
                            g_row_int.ir_value(),
                            Int32(H_BYTES).ir_value(),
                            mbar.toint().ir_value(),
                        )
                load_state.advance()  # 推进生产者状态到下一级
                gi += Int32(num_sms * B)  # 跳到下一个属于本 CTA 的批次（跨 CTA 步进）

        # ============================================
        # Warp 1 — S2G consumer: staged primary row -> every duplicate slot
        # ============================================
        elif warp_idx == self.CONSUMER_WARP:  # warp 1：S2G 消费者
            use_state = pipeline.make_pipeline_state(  # 消费状态：等待 full
                pipeline.PipelineUserType.Consumer, stages
            )
            rel_state = pipeline.make_pipeline_state(  # 释放状态：滞后若干拍用于限流
                pipeline.PipelineUserType.Consumer, stages
            )

            issued = Int32(0)  # 已消费批计数（限流用）
            gi = Int32(bidx * B)  # 本 CTA 首个批次的起始组号
            lane = cute.arch.lane_idx()  # lane 编号
            while gi < n_groups:  # 逐批处理
                n = Int32(B)  # 本批默认组数
                if gi + n > n_groups:  # 末批截断
                    n = n_groups - gi

                dup_start = Int32(0)  # 本 lane 对应组在 dup_loffs 的起始（无效时为 0）
                dup_n = Int32(0)  # 本 lane 对应组的重复槽数（无效时为 0）
                if lane < n:  # 每 lane 认领批内一个组（lane >= n 的组不存在）
                    dup_start = dup_groups_tensor[(gi + lane) * 3 + 1]  # 组头第 1 列：重复槽起始
                    dup_n = dup_groups_tensor[(gi + lane) * 3 + 2]  # 组头第 2 列：重复槽数

                load_pipe.consumer_wait(use_state)  # 等待本批 B 行数据就绪
                for idx in cutlass.range(n, unroll=1):  # 逐组扇出
                    cur_dup_start = cute.arch.shuffle_sync(dup_start, Int32(idx))  # 从 lane idx 广播该组重复槽起始
                    cur_dup_n = cute.arch.shuffle_sync(dup_n, Int32(idx))  # 广播该组重复槽数
                    if lane == 0:  # bulk 存储仅 lane 0 发出
                        s_row_int = (  # 本级 smem 批内第 idx 行（主选行）地址
                            stage_smem.iterator + use_state.index * B * H + idx * H
                        ).toint()
                        for k in cutlass.range(cur_dup_n, unroll=1):  # 逐重复槽写出
                            slot = dup_loffs_tensor[cur_dup_start + k]  # 第 k 个重复槽的偏移
                            g_dup_int = (  # 重复槽在全局内存的行地址
                                hidden.iterator + Int64(slot) * Int64(H)
                            ).toint()
                            cp_async_bulk_s2g(  # TMA 整行拷贝：smem 主选行 -> 重复槽
                                s_row_int.ir_value(),
                                g_dup_int.ir_value(),
                                Int32(H_BYTES).ir_value(),
                            )
                if lane == 0:  # 本批全部存储提交为一个 bulk_group
                    cute.arch.cp_async_bulk_commit_group()
                use_state.advance()  # 推进消费状态到下一级

                # Throttle in-flight bulk_groups to <= STAGES-1.
                if issued >= Int32(stages - 1):  # 起步阶段之后开始限流
                    cute.arch.cp_async_bulk_wait_group(stages - 1)  # 等待在途组数降到 stages-1
                    load_pipe.consumer_release(rel_state)  # 释放对应级缓冲供生产者复用
                    rel_state.advance()  # 推进释放状态
                issued += Int32(1)  # 更新已消费批计数
                gi += Int32(num_sms * B)  # 跳到下一个本 CTA 批次

            # Drain the trailing in-flight stores.
            cute.arch.cp_async_bulk_wait_group(0)  # 排空全部在途存储


# ============================================================================
# Host launcher with per-shape compile cache
# ============================================================================

@functools.lru_cache(maxsize=None)  # 按设备缓存设备属性查询
def _max_smem_per_block_optin(device_index: int) -> int:  # 查询每 CTA 可选用的最大动态共享内存
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin  # 读取 CUDA 设备属性


# 校验 plan 携带的去重结构与 ctx 的形状/设备一致。
def _check_epilogue_plan(
    ctx: dict,
    plan: MoonEPCommPlan,
    dev: torch.device,
) -> None:
    NvS = int(ctx['NvS'])  # 每 rank 有效槽位
    assert isinstance(plan, MoonEPCommPlan)  # plan 类型检查
    assert plan.NvS == NvS, f"plan.NvS must match ctx NvS={NvS}, got {plan.NvS}"  # 槽位数一致

    def _check_tensor(t: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:  # 通用张量检查工具
        # 连续 int32
        assert t.dtype == torch.int32 and t.is_contiguous(), \
            f"{name} must be contiguous int32"
        # 形状精确匹配
        assert tuple(t.shape) == shape, \
            f"{name} must be shape {shape}, got {tuple(t.shape)}"
        # 设备一致
        assert t.device == dev, \
            f"{name} must be on {dev}, got {t.device}"

    _check_tensor(plan.dup_groups, "dup_groups", (NvS, 3))  # 去重组头表
    _check_tensor(plan.dup_loffs, "dup_loffs", (NvS,))  # 重复槽表
    _check_tensor(plan.dup_counts, "dup_counts", (2,))  # 去重计数器


@functools.lru_cache(maxsize=None)  # 按 (形状, 选项) 缓存编译产物
def _get_compiled(  # 编译并缓存尾处理内核
    H: int,
    NvS: int,
    num_sms: int,
    device_index: int,
    pdl_launch: bool,
):
    smem_budget = _max_smem_per_block_optin(device_index) - 1024  # 预留 1KB 余量后的共享内存预算
    kernel = DispatchEpilogueKernel(  # 构造内核对象（内含几何挑选）
        H=H, NvS=NvS, num_sms=num_sms, smem_budget=smem_budget,
        pdl_launch=pdl_launch,
    )

    bf16_ptr = make_ptr(BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)  # 占位 bf16 指针
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)  # 占位 int32 指针
    stream_arg = cuda.CUstream(0)  # 占位流参数

    return cute.compile(  # 按 Buffer 类型与形状特化编译
        kernel,
        bf16_ptr,  # hidden_buf_local
        i32_ptr,   # dup_groups
        i32_ptr,   # dup_loffs
        i32_ptr,   # dup_counts
        stream_arg,
    )


# 尾处理的 host 侧启动入口：校验输入、取编译缓存并填充指针启动内核。
def launch_dispatch_epilogue(
    ctx: dict,
    plan: MoonEPCommPlan,
    *,
    pdl_launch: bool = False,
):
    """Expand duplicate rows in place on the local NVL shard.

    Must run after ``launch_dispatch`` on the same stream: dispatch's exit
    cross_rank_barrier publishes the remote NVL writes this kernel reads.
    Consumes the plan-owned ``dup_groups`` / ``dup_loffs`` / ``dup_counts``
    written by a fresh dispatch builder (reuse paths pass the saved tensors
    unchanged). Purely local — no cross-rank communication.
    """
    H = int(ctx['H'])  # 隐藏维度
    NvS = int(ctx['NvS'])  # 每 rank 有效槽位
    num_sms = int(ctx['num_sms_dedup'])  # 尾处理专用网格 CTA 数
    dev = ctx['hidden_buf_local'].device  # 本地 shard 所在设备
    device_index = dev.index  # 设备编号（编译缓存键之一）

    assert ctx['H'] % 8 == 0, "H must be multiple of 8 for 16-B bulk-copy alignment"  # H 须为 8 的倍数保证 16B 对齐
    # 本地 shard：连续 bf16
    assert ctx['hidden_buf_local'].dtype == torch.bfloat16 and \
        ctx['hidden_buf_local'].is_contiguous()
    assert ctx['hidden_buf_local'].is_cuda, "hidden_buf_local must be a CUDA tensor"  # 必须在 CUDA 上
    # 形状必须为 [NvS, H]
    assert tuple(ctx['hidden_buf_local'].shape) == (NvS, H), \
        f"hidden_buf_local must be shape ({NvS}, {H}), " \
        f"got {tuple(ctx['hidden_buf_local'].shape)}"
    assert device_index is not None, "dispatch epilogue requires a CUDA device"  # 需要具体 CUDA 设备
    _check_epilogue_plan(ctx, plan, dev)  # 校验 plan 去重结构

    epilogue_compiled = _get_compiled(H, NvS, num_sms, device_index, bool(pdl_launch))  # 取（或编译）内核

    hbf_local_ptr = make_ptr(BFloat16, ctx['hidden_buf_local'].data_ptr(),  # 本地 NVL shard 指针
                             cute.AddressSpace.gmem, assumed_align=16)
    dup_groups_ptr = make_ptr(Int32, plan.dup_groups.data_ptr(),  # 去重组头表指针
                              cute.AddressSpace.gmem, assumed_align=16)
    dup_loffs_ptr = make_ptr(Int32, plan.dup_loffs.data_ptr(),  # 重复槽表指针
                             cute.AddressSpace.gmem, assumed_align=16)
    dup_counts_ptr = make_ptr(Int32, plan.dup_counts.data_ptr(),  # 去重计数器指针
                              cute.AddressSpace.gmem, assumed_align=8)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)  # 取当前 PyTorch 流封装为 CUstream

    epilogue_compiled(  # 调用编译产物：填充指针并启动内核
        hbf_local_ptr,
        dup_groups_ptr,
        dup_loffs_ptr,
        dup_counts_ptr,
        stream,
    )
