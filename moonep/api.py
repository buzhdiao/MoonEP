"""
MoonEP Top-level API.

Notation: S = input tokens per rank, K = routed top-k per token, E = total routed
experts in the EP group, R = number of EP ranks (EP comm size), epn = E/R local
experts and prefetch slots per rank, NvS = dispatched token slots per rank (S*K
real tokens plus per-VM-group padding), H = hidden size, H' = expert FFN
intermediate size.

Usage:
    buffer = Buffer(S, H, K, E, num_ep_ranks, num_sms=None, token_padding=128)

    # dispatch fwd
    hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
        hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
    )
    buffer.prefetch_weight(
        plan=plan,
        local_gate_weight=local_gate_weight,
        local_up_weight=local_up_weight,
        local_down_weight=local_down_weight,
        gate_prefetch_buffer=gate_prefetch_buffer,
        up_prefetch_buffer=up_prefetch_buffer,
        down_prefetch_buffer=down_prefetch_buffer,
    )

    # combine fwd
    output_sh, gathered_route_weights_sk, _ = buffer.combine(
        plan=plan,
        hidden_nvsh=expert_output_nvsh,
        route_weights_nvs=route_weights_nvs,
    )

    # combine bwd: re-dispatch the output grad with the saved plan
    grad_expert_output_nvsh, _, _, _ = buffer.dispatch(grad_output_sh, plan=plan)

    # dispatch bwd: combine the hidden grad back to token-major, and reduce
    # duplicated experts' weight grads to their home ranks
    grad_hidden_sh, _, _ = buffer.combine(plan=plan, hidden_nvsh=grad_hidden_nvsh)
    buffer.reduce_grad(
        plan=plan,
        local_gate_grad=local_gate_grad,
        local_up_grad=local_up_grad,
        local_down_grad=local_down_grad,
        gate_reduce_buffer=gate_reduce_buffer,
        up_reduce_buffer=up_reduce_buffer,
        down_reduce_buffer=down_reduce_buffer,
    )

    buffer.destroy()
"""

import logging  # 日志：缓冲区大小等信息输出
import os  # 环境变量读取（MOONEP_NUM_SMS_DEDUP）
import warnings  # 析构未显式销毁时的 ResourceWarning

import torch  # PyTorch：张量、流、事件与设备接口
import torch.distributed as dist  # 进程组与跨 rank 同步原语

from .buffer import (  # 底层缓冲设施：NVLink 共享/多播张量创建与对齐工具
    create_nvl_dist_tensor,  # 创建跨 rank 的 NVLink 共享分布张量
    create_nvl_dist_multicast_tensor,  # 创建带 multicast 能力的分布张量
    pad_dim0_for_alignment,  # 按对齐要求补齐 dim0
    get_vmm_granularity,  # 查询 VMM 映射粒度
    get_multicast_granularity,  # 查询 multicast 绑定粒度
)
from .constants import DEDUP_BUILDER_WARPS  # 去重 builder 的 warp 数常量
from .planning import MoonEPCommPlan, allocate_planning_outputs, launch_planning  # 规划：计划容器、输出分配与 kernel 发射
from .inter_rank_sync import launch_inter_rank_sync  # 跨 rank 预同步 kernel
from .dispatch import launch_dispatch  # dispatch 主 kernel
from .dispatch_epilogue import launch_dispatch_epilogue  # dispatch 后处理（重复项展开）
from .combine import launch_combine  # combine 主 kernel
from .combine_prologue import launch_combine_prologue  # combine 前处理（重复行累加）
from .prefetch import _ELEM_TYPES, launch_prefetch  # 权重预取：支持 dtype 与发射函数
from .grad_reduce import launch_grad_reduce  # 梯度归约 kernel

logger = logging.getLogger(__name__)  # 模块级 logger


def _align_up(x: int, alignment: int) -> int:  # 整数向上对齐到 alignment 的倍数
    """Round x up to a multiple of alignment."""
    return ((x + alignment - 1) // alignment) * alignment  # 先上取整再乘回 alignment


def _validate_rank_strided_pool(pool: torch.Tensor) -> int:  # 校验按 rank 分块的池张量布局，返回 rank 步长
    """Validate a CUDA pool with contiguous rank payloads and optional gaps."""
    assert pool.is_cuda and pool.ndim >= 2 and pool.shape[0] > 0  # 必须是 CUDA 上的非空 >=2 维张量
    assert pool[0].is_contiguous(), "pools must be contiguous within each rank"  # 每个 rank 的负载必须连续
    rank_numel = pool[0].numel()  # 单个 rank 的元素数
    rank_stride = int(pool.stride(0))  # rank 维步长（允许 rank 间有 padding 间隙）
    assert rank_numel > 0 and rank_stride >= rank_numel, "rank payloads must not overlap"  # 步长不小于负载，保证不重叠
    stride_bytes = rank_stride * pool.element_size()  # 步长对应字节数
    assert stride_bytes % 16 == 0 and 0 < stride_bytes < (1 << 40)  # 步长须 16B 对齐且规模合理
    assert pool.data_ptr() % 16 == 0, "pools must be 16-byte aligned"  # 基址必须 16B 对齐（TMA/向量访问要求）
    end = pool.storage_offset() + (pool.shape[0] - 1) * rank_stride + rank_numel  # 最后一个 rank 的末元素位置
    assert end * pool.element_size() <= pool.untyped_storage().nbytes()  # 不得越过底层存储
    return rank_stride  # 返回 rank 步长供后续使用


def _num_sms_dedup_from_env(max_sms: int) -> int:  # 从环境变量解析 epilogue/prologue 的 SM 数
    """Resolve the local epilogue/prologue SM count.

    ``MOONEP_NUM_SMS_DEDUP`` is intentionally an environment override rather
    than a Buffer argument so benchmark jobs can sweep it without touching the
    public API surface.
    """
    raw = os.environ.get("MOONEP_NUM_SMS_DEDUP")  # 读取环境变量
    if raw is None or raw == "":  # 未设置则用默认值
        return max_sms
    try:  # 解析为整数
        value = int(raw)
    except ValueError as exc:  # 非整数直接报错
        raise ValueError(
            f"MOONEP_NUM_SMS_DEDUP must be an integer in [1, {max_sms}], got {raw!r}"
        ) from exc
    if not (1 <= value <= max_sms):  # 范围校验
        raise ValueError(
            f"MOONEP_NUM_SMS_DEDUP must be in [1, {max_sms}], got {value}"
        )
    return value  # 返回合法的 SM 数


def _log_context_buffer_size(ctx: dict) -> None:  # rank0 打印各通信缓冲区大小
    def tensor_nbytes(tensor: torch.Tensor) -> int:  # 计算张量字节数
        return int(tensor.numel()) * int(tensor.element_size())

    def format_nbytes(nbytes: int) -> str:  # 字节数格式化为人类可读单位
        if nbytes >= 1 << 30:  # GiB 量级
            return f"{nbytes / (1 << 30):.2f} GiB"
        if nbytes >= 1 << 20:  # MiB 量级
            return f"{nbytes / (1 << 20):.2f} MiB"
        if nbytes >= 1 << 10:  # KiB 量级
            return f"{nbytes / (1 << 10):.2f} KiB"
        return f"{nbytes} B"  # 小于 1KiB 直接显示字节数

    if int(ctx['rank']) != 0:  # 只有 rank0 打印
        return

    R = int(ctx['R'])  # rank 数
    hidden_mapped_bytes = tensor_nbytes(ctx['hidden_buf'])  # hidden 共享缓冲总映射字节数
    meta_mapped_bytes = tensor_nbytes(ctx['meta_buf'])  # meta 共享缓冲总映射字节数
    hidden_chunk_bytes = hidden_mapped_bytes // R  # 单 rank 的 hidden chunk
    meta_chunk_bytes = meta_mapped_bytes // R  # 单 rank 的 meta chunk
    local_temp_bytes = sum(  # 累加所有本地临时缓冲的大小
        tensor_nbytes(ctx[name])
        for name in (
            'alloc',  # 规划 alloc 临时
            'group_tokens',  # 组 token 计数
            'z',  # 跨组转移矩阵
            'local_hist',  # 本地直方图
            'grid_sync_bar',  # 软件 grid 屏障
            'primary_packed',  # 去重 builder 的主键打包暂存
            'kmask',  # top-k 位图暂存
            'kidx_to_loff',  # topk 索引到槽位映射
            'builder_bar',  # builder 自复位屏障
        )
    )

    total_buffer_bytes = hidden_chunk_bytes + meta_chunk_bytes + local_temp_bytes  # 单 rank 总缓冲量

    log_lines = [  # 组装多行日志
        "MoonEP comm buffer sizes:",  # 标题
        "  config                 : "  # 配置行
        f"R={ctx['R']}, S={ctx['S']}, H={ctx['H']}, E={ctx['E']}, "
        f"K={ctx['K']}, NvS={ctx['NvS']}, NvS_padded={ctx['NvS_padded']}",
        "  sm_count               : "  # SM 数行
        f"num_sms={ctx['num_sms']}, num_sms_dedup={ctx['num_sms_dedup']}",
        f"  hidden_buffer_size     : {format_nbytes(hidden_chunk_bytes)}",  # 单 rank hidden 大小
        f"  meta_buffer_size       : {format_nbytes(meta_chunk_bytes)}",  # 单 rank meta 大小
        f"  local_temp_buffer_size : {format_nbytes(local_temp_bytes)}",  # 本地临时缓冲大小
        f"  total_size             : {format_nbytes(total_buffer_bytes)}",  # 单 rank 总计
    ]
    logger.info("\n".join(log_lines))  # 输出日志


def _launch_grad_reduces(  # 对 gate/up/down 三组权重梯度依次发射归约 kernel
    ctx,
    experts_to_copy: torch.Tensor,
    local_grads: tuple[torch.Tensor, ...],
    reduce_buffers: tuple[torch.Tensor, ...],
) -> None:
    rank = int(ctx['rank'])  # 当前 rank 号
    num_sms = int(ctx['num_sms'])  # 通信 kernel 的 SM 数
    for local_grad, reduce_buffer in zip(  # 逐对（本地梯度, 归约缓冲）发射
        local_grads, reduce_buffers, strict=True,
    ):
        launch_grad_reduce(  # 发射单个权重梯度的跨 rank 归约
            local_grad,  # 本地参数梯度 [epn, H, H']
            reduce_buffer,  # 全 rank 归约缓冲池
            experts_to_copy,  # 规划产出的远端专家拷贝表
            rank=rank,  # 当前 rank
            num_sms=num_sms,  # SM 数
            meta_buf=ctx['meta_buf'],  # NVLink meta 缓冲（屏障用）
            meta_stride=int(ctx['meta_chunk_padded']),  # meta chunk 步长
            barrier_off=int(ctx['BARRIER_OFF']),  # 屏障区偏移
            grid_sync_bar=ctx['grid_sync_bar'],  # 软件 grid 屏障计数器
        )


def _create_context(  # 预分配全部 NVLink 共享缓冲与本地临时缓冲，构造上下文
    S: int,
    H: int,
    K: int,
    E: int,
    num_ep_ranks: int,
    num_sms: int | None = None,
    token_padding: int = 128,
    group: "dist.ProcessGroup | None" = None,
) -> dict:
    """Pre-allocate all NVLink shared buffers and local temp buffers.

    The extra logical NvS slots from token_padding are bounded by
    (token_padding - 1) * 2 * E / num_ep_ranks. The logical capacity is then
    rounded up to token_padding, while the physical allocation is separately
    aligned to VMM granularity.
    NVLink shared buffer layout:
      - hidden_buf: [NvS_padded, H] bf16 — separate allocation, padded for VMM alignment
      - meta_buf:   [meta_chunk_padded] int32 — merged allocation containing:
          [0, NvS):                weights (fp32, alias as int32)
          [NvS, NvS + R*E):       tpe gather (only rank 0 is the read/write target; symmetric reserve)
          [NvS + R*E, ...):       planning internal scratch/staging
          [..., +N4):             topk0 offload scratch
          [..., +N4):             ORDER, c1 order consumed by c2/passB
          [..., +N4):             ORDER0, rank1 rank0-c1 scratch
          [..., +3):              cross-rank phase barrier
          [..., +NvS):            src_info scratch published by planning and
                                  consumed by the fresh dispatch dedup builder;
                                  non-negative entries encode src_rank * NvS + offv,
                                  -1 marks empty slots
    """
    if num_sms is not None:  # 显式指定 SM 数时做类型与取值校验
        if not isinstance(num_sms, int) or isinstance(num_sms, bool):
            raise TypeError(
                f"num_sms must be a positive int, got "
                f"{type(num_sms).__name__}"
            )
        if num_sms <= 0:
            raise ValueError(f"num_sms must be a positive int, got {num_sms}")

    device = torch.cuda.current_device()  # 当前 CUDA 设备号
    max_sms = torch.cuda.get_device_properties(device).multi_processor_count  # 设备的 SM 总数
    if num_sms is None:  # 默认取 32 与设备 SM 数的较小者
        num_sms = min(32, max_sms)
    if num_sms > max_sms:  # 不得超过设备 SM 数
        raise ValueError(
            f"num_sms ({num_sms}) must not exceed the device SM count "
            f"({max_sms})"
        )

    rank = dist.get_rank(group=group)  # 进程组内当前 rank
    R = num_ep_ranks  # EP 组大小
    assert R == dist.get_world_size(group=group), (  # 必须与进程组 world size 一致
        f"num_ep_ranks ({R}) must equal group world size "
        f"({dist.get_world_size(group=group)})"
    )
    N = S * K  # top-k 平铺总数
    dev = f"cuda:{device}"  # 本地临时缓冲的设备字符串
    num_sms_dedup = _num_sms_dedup_from_env(max_sms)  # epilogue/prologue 的 SM 数

    epn = E // R  # 每 rank 本地专家数
    assert E % R == 0, f"E ({E}) must be divisible by R ({R})"  # E 必须被 R 整除
    assert isinstance(token_padding, int) and token_padding > 0, \
        f"token_padding must be a positive int, got {token_padding}"  # padding 粒度必须为正整数

    NvS_capacity = S * K  # 真实 token 槽位硬容量

    # The current constructive planner lets each destination rank receive
    # tokens from at most one remote home group. Each home group has E/R
    # experts, so each rank has at most E/R remote expert segments, plus at
    # most E/R local expert segments. Each non-empty segment is padded up
    # from at least 1 real token slot to a multiple of token_padding, so the
    # extra logical NvS slot bound is (token_padding - 1) * 2 * E/R.
    # This tight bound is not necessarily a multiple of token_padding.
    # Dispatch returns the full logical [NvS, H] view. Aligning NvS to
    # token_padding lets callers launch tiled operators with fixed shapes
    # without reading GPU-side cu_seqlens or dynamically padding the output.
    # The physical allocation is aligned separately to VMM granularity below.
    # ↑ 中文补充：当前构造式规划器使每个目标 rank 至多从一个远端 home 组接收 token；
    # 每组 E/R 个专家，故远端/本地段各至多 E/R 个。每个非空段至少从 1 个真实槽位
    # padding 到 token_padding 的倍数，故额外逻辑槽位上界为 (token_padding-1)*2*E/R。
    # 该紧界不一定是 token_padding 的倍数；把 NvS 对齐到 token_padding 后，
    # 调用方可以按固定形状启动 tiled 算子，无需读取 cu_seqlens 或动态 padding。
    token_padding_extra = (token_padding - 1) * 2 * epn  # padding 带来的额外逻辑槽位上界
    NvS = _align_up(NvS_capacity + token_padding_extra, token_padding)  # 对齐后的逻辑槽位数

    # ================================================================
    # Planning constexpr computation and int32 range guards
    # （规划编译期常量计算与 int32 范围守卫）
    # ================================================================
    BLOCK_SIZE_P2 = 2048  # 规划 kernel 的分片大小（与 planning.py 保持一致）

    int32_max = 2**31 - 1  # int32 上限
    assert 0 < N < int32_max, (  # S*K 必须落在 int32 正数范围内
        "planning requires 0 < S*K < int32_max: "
        f"S={S}, K={K}, S*K={N}, int32_max={int32_max}"
    )
    num_vblocks = (N + BLOCK_SIZE_P2 - 1) // BLOCK_SIZE_P2  # 直方图分块数 = ceil(N / BLOCK_SIZE_P2)

    # ================================================================
    # Compute padded dimensions for VMM alignment（计算 VMM 对齐的补齐维度）
    # ================================================================
    NvS_padded = pad_dim0_for_alignment([NvS, H], torch.bfloat16)  # hidden 缓冲 dim0 按 VMM 粒度补齐

    # Meta offsets are int32 elements. Vectorized regions are 16B-aligned.
    # ↑ 中文补充：meta 偏移以 int32 元素计；向量化区域 16B（4 元素）对齐。
    WEIGHTS_OFF = 0  # 权重（fp32 别名 int32）区偏移
    TPE_OFF = _align_up(NvS, 4)  # tpe gather 区偏移（16B 对齐）
    PLAN_OFF = _align_up(TPE_OFF + R * E, 4)  # 规划输出区偏移（16B 对齐）
    broadcast_elems = 3 * E * R  # 规划中需多播广播的元素数
    assert broadcast_elems % 4 == 0, (  # 多播走 v4，必须能被 4 整除
        f"broadcast_elems ({broadcast_elems}) must be divisible by 4"
    )
    planning_out_elems = (  # 规划输出区总元素数
        broadcast_elems  # alloc 前缀 + tpe 前缀 + expert offsets
        + R * (2 * epn)  # cu_seqlens 区
        + 2 * R * (2 * epn)  # 零填充区间（start/count 两列）
        + epn * R  # experts_to_copy 区
        + 2 * R  # remote_stats 区
    )

    # C1 scratch segments are padded so bulk copies can stay vectorized.
    # ↑ 中文补充：C1 暂存段做 4 元素对齐补齐，保证 bulk 拷贝保持向量化。
    N4 = _align_up(N, 4)  # N 补齐到 4 的倍数
    TOPK0_OFF = _align_up(PLAN_OFF + planning_out_elems, 4)  # topk0 暂存区偏移（16B 对齐）
    ORDER_OFF = TOPK0_OFF + N4  # order 区偏移
    ORDER0_OFF = ORDER_OFF + N4  # rank0 order 暂存区偏移
    BARRIER_OFF = ORDER0_OFF + N4  # 跨 rank 屏障区偏移
    # cross_rank_barrier uses 3 int32 per rank: 2 phase signals + 1 phase/sign
    # counter (independent of rank count, self-resetting double buffer).
    # ↑ 中文补充：cross_rank_barrier 每 rank 用 3 个 int32：2 个相位信号 +
    # 1 个相位/符号计数器（与 rank 数无关，自复位双缓冲）。
    BARRIER_SLOTS = 3  # 屏障槽位数
    SRC_INFO_OFF = BARRIER_OFF + BARRIER_SLOTS  # src_info 暂存区偏移
    meta_chunk_logical = SRC_INFO_OFF + NvS  # meta chunk 的逻辑长度（含 src_info 切片）

    # Chunk size must satisfy both VMM mapping and multicast binding.
    # ↑ 中文补充：chunk 大小须同时满足 VMM 映射与 multicast 绑定两种粒度。
    gran = get_vmm_granularity()  # VMM 映射粒度
    mc_gran = get_multicast_granularity(R)  # multicast 绑定粒度
    chunk_align_bytes = max(gran, mc_gran)  # 取两者较大者作为对齐要求
    meta_chunk_bytes = meta_chunk_logical * 4  # meta chunk 字节数（int32*4）
    meta_chunk_padded = _align_up(meta_chunk_bytes, chunk_align_bytes) // 4  # 对齐后的 chunk 长度（int32 元素）

    # Some CuTe DSL address expressions multiply a runtime Int32 rank/drank by
    # these constexpr strides, so guard the largest reachable nonnegative index.
    # ↑ 中文补充：部分 CuTe DSL 地址表达式会把运行期 Int32 rank 乘以这些编译期步长，
    # 因此守卫可达的最大非负下标，防止 Int32 溢出。
    max_meta_index = (R - 1) * meta_chunk_padded + meta_chunk_logical - 1  # meta 缓冲最大下标
    assert max_meta_index <= int32_max, (  # meta 索引不得溢出 Int32
        "meta_buf rank-stride indexing would overflow CuTe Int32 arithmetic: "
        f"max_index={max_meta_index}, R={R}, S={S}, K={K}, N={N}, "
        f"NvS={NvS}, meta_chunk_logical={meta_chunk_logical}, "
        f"meta_chunk_padded={meta_chunk_padded}, int32_max={int32_max}"
    )
    max_hidden_index = (R - 1) * NvS_padded + NvS - 1  # hidden 缓冲最大下标
    assert max_hidden_index <= int32_max, (  # hidden 索引不得溢出 Int32
        "hidden_buf/dst rank-stride indexing would overflow CuTe Int32 arithmetic: "
        f"max_index={max_hidden_index}, R={R}, S={S}, K={K}, N={N}, "
        f"NvS={NvS}, NvS_padded={NvS_padded}, int32_max={int32_max}"
    )

    # ================================================================
    # Allocate NVLink shared buffers（分配 NVLink 共享缓冲）
    # ================================================================
    hidden_buf = create_nvl_dist_tensor([NvS_padded, H], torch.bfloat16, rank, R, group=group)  # 全 rank 的 hidden 共享缓冲
    meta_buf, meta_mc = create_nvl_dist_multicast_tensor(  # meta 缓冲：普通视图 + multicast 视图
        [meta_chunk_padded], torch.int32, rank, R, group=group,
    )

    # Zero the barrier region on all ranks（把所有 rank 的屏障区清零）
    for r in range(R):  # 逐 rank 清零屏障槽位
        meta_buf[r * meta_chunk_padded + BARRIER_OFF:
                 r * meta_chunk_padded + BARRIER_OFF + BARRIER_SLOTS].zero_()
    dist.barrier(group=group)  # 进程组屏障：清零完成后全体可见


    # ================================================================
    # Local buffers（本地临时缓冲）
    # ================================================================
    alloc = torch.empty(E * R, dtype=torch.int32, device=dev)  # 规划 alloc 临时缓冲 [E*R]
    group_tokens = torch.empty(R, dtype=torch.int32, device=dev)  # 各目标组 token 计数 [R]
    z = torch.empty(R * R, dtype=torch.int32, device=dev)  # 跨组转移矩阵 [R*R]
    local_hist = torch.empty(E * num_vblocks, dtype=torch.int32, device=dev)  # 本地 vblock 直方图 [num_vblocks, E]
    # Global arrive counter (1 int32) for the software grid barrier: the
    # equivalent of cudaLaunchCooperative's grid workspace. Self-resetting
    # (sentinel-bit flip), allocated once and shared by planning/dispatch/
    # combine/inter_rank, with no per-launch zeroing needed.
    # ↑ 中文补充：软件 grid 屏障的全局 arrive 计数器（1 个 int32），等价于
    # cudaLaunchCooperative 的 grid workspace；经哨兵位翻转自复位，仅分配一次，
    # 由 planning/dispatch/combine/inter_rank 共享，无需每次发射前清零。
    grid_sync_bar = torch.zeros(1, dtype=torch.int32, device=dev)
    # Dispatch builder scratch for fresh planning. ``kidx_to_loff`` maps each
    # (source rank, token, topk index) key to the local NvS slot used later when
    # expanding duplicate groups. ``builder_bar`` is a self-resetting TAG
    # counter, so it is initialized once like grid_sync_bar.
    # ↑ 中文补充：新鲜规划的 dispatch builder 暂存。kidx_to_loff 把每个
    # (源 rank, token, topk 索引) 键映射到本地 NvS 槽位，供后续展开重复组使用；
    # builder_bar 是自复位 TAG 计数器，与 grid_sync_bar 一样只初始化一次。
    primary_packed = torch.empty(R * S, dtype=torch.int32, device=dev)  # 主键打包暂存 [R*S]
    kmask = torch.empty(R * S, dtype=torch.int32, device=dev)  # top-k 位图暂存 [R*S]
    kidx_to_loff = torch.empty(R * S * K, dtype=torch.int32, device=dev)  # 键到槽位映射 [R*S*K]
    builder_bar = torch.zeros(1, dtype=torch.int32, device=dev)  # builder 自复位屏障

    ctx = {  # 组装上下文字典
        'rank': rank,  # 当前 rank 号
        'group': group,  # 进程组
        'R': R, 'E': E, 'S': S, 'K': K, 'H': H,  # 维度配置
        'N': N, 'NvS': NvS,  # 派生规模：top-k 总数 / 逻辑槽位数
        'NvS_capacity': NvS_capacity,  # 真实槽位硬容量
        'NvS_padded': NvS_padded,  # VMM 对齐后的槽位数
        'num_sms': num_sms,  # 通信 kernel SM 数
        'num_sms_dedup': num_sms_dedup,  # epilogue/prologue SM 数
        'token_padding': token_padding,  # VM 组 padding 粒度
        'device': device,  # CUDA 设备号
        'token_padding_extra': token_padding_extra,  # padding 额外槽位上界
        # NVLink shared（NVLink 共享缓冲）
        'hidden_buf': hidden_buf,  # 全 rank hidden 共享缓冲
        'meta_buf': meta_buf,  # meta 缓冲（普通视图）
        'meta_mc': meta_mc,  # meta 缓冲（multicast 视图）
        # Meta layout（meta 布局）
        'meta_chunk_padded': meta_chunk_padded,  # 对齐后的 chunk 步长
        'WEIGHTS_OFF': WEIGHTS_OFF,  # 权重区偏移
        'TPE_OFF': TPE_OFF,  # tpe gather 区偏移
        'PLAN_OFF': PLAN_OFF,  # 规划输出区偏移
        'TOPK0_OFF': TOPK0_OFF,  # topk0 暂存区偏移
        'ORDER_OFF': ORDER_OFF,  # order 区偏移
        'ORDER0_OFF': ORDER0_OFF,  # order0 暂存区偏移
        'BARRIER_OFF': BARRIER_OFF,  # 屏障区偏移
        'SRC_INFO_OFF': SRC_INFO_OFF,  # src_info 区偏移
        # Local temps（本地临时缓冲）
        'alloc': alloc, 'group_tokens': group_tokens,  # 规划 alloc / 组 token 计数
        'z': z,  # 跨组转移矩阵
        'local_hist': local_hist,  # 本地直方图
        'grid_sync_bar': grid_sync_bar,  # 软件 grid 屏障
        'primary_packed': primary_packed,  # 主键打包暂存
        'kmask': kmask,  # top-k 位图
        'kidx_to_loff': kidx_to_loff,  # 键到槽位映射
        'builder_bar': builder_bar,  # builder 屏障
        'num_vblocks': num_vblocks,  # 直方图分块数
        # Local views（本地视图）
        'hidden_buf_local': hidden_buf[rank * NvS_padded: rank * NvS_padded + NvS],  # 本 rank 的 [NvS, H] hidden 切片
        'weights_buf_local': meta_buf[rank * meta_chunk_padded + WEIGHTS_OFF:  # 本 rank 的权重区切片
                                      rank * meta_chunk_padded + WEIGHTS_OFF + NvS],
    }
    _log_context_buffer_size(ctx)  # rank0 打印缓冲区大小
    return ctx  # 返回上下文


class Buffer:  # MoonEP 通信缓冲的持有者与管理入口
    """MoonEP communication buffer owner.

    A Buffer owns the NVLink/VMM communication buffers, local scratch tensors,
    and the async communication stream used by dispatch/prefetch/combine/reduce. Call
    ``destroy()`` before tearing down the process group.
    """

    def __init__(
        self,
        S: int,
        H: int,
        K: int,
        E: int,
        num_ep_ranks: int,
        num_sms: int | None = None,
        token_padding: int = 128,
        group: "dist.ProcessGroup | None" = None,
        comm_stream_priority: int = -1,
        enable_pdl: bool = True,
        explicitly_destroy: bool = False,
    ):  # 构造：分配并持有全部通信缓冲
        """Allocate and hold all communication buffers.

        Args:
            S: input tokens per rank.
            H: hidden size.
            K: routed top-k per token.
            E: total routed experts in the EP group; must be divisible by
                ``num_ep_ranks``.
            num_ep_ranks: EP group size (R); must equal the group world size.
            num_sms: SM count for the comm kernels; must not exceed the
                current CUDA device's SM count. None defaults to the smaller
                of 32 and the device SM count.
            token_padding: each non-empty VM group is padded up to a multiple
                of this token count.
            group: torch.distributed process group; None uses the default
                group. Must be called after ``init_process_group``.
            comm_stream_priority: priority of the async comm stream; the
                default -1 gives it higher priority than the main stream.
            enable_pdl: launch kernels with programmatic dependent launch;
                False falls back to plain same-stream serial launches.
            explicitly_destroy: if True, warn (instead of auto-destroying)
                when the Buffer is garbage-collected without ``destroy()``.
        """
        assert isinstance(comm_stream_priority, int), (  # 流优先级必须为 int
            f"comm_stream_priority must be an int, got "
            f"{type(comm_stream_priority).__name__}"
        )
        assert isinstance(enable_pdl, bool), (  # PDL 开关必须为 bool
            f"enable_pdl must be a bool, got "
            f"{type(enable_pdl).__name__}"
        )
        self.explicitly_destroy = explicitly_destroy  # 是否要求显式销毁（GC 时仅告警）
        self.comm_stream_priority = comm_stream_priority  # 通信流优先级
        self.enable_pdl = enable_pdl  # 是否启用程序化依赖发射（PDL）
        self._comm_stream: torch.cuda.Stream | None = None  # 异步通信流（稍后创建）
        self._destroyed = False  # 销毁标志
        self._ctx = _create_context(  # 预分配全部共享与本地缓冲
            S, H, K, E, num_ep_ranks,
            num_sms=num_sms,
            token_padding=token_padding,
            group=group,
        )
        self._comm_stream = torch.cuda.Stream(  # 创建异步通信流
            device=int(self._ctx['device']),
            priority=self.comm_stream_priority,
        )

    @property
    def destroyed(self) -> bool:  # 是否已销毁
        return self._destroyed

    @property
    def hidden_nvsh_buffer_view(self) -> torch.Tensor:  # 本 rank 的 [NvS, H] bf16 通信缓冲视图
        """The local rank's [NvS, H] bf16 communication buffer.

        Exposed for zero-copy integration: callers may write expert outputs
        into this view and hand it back to ``combine(zero_copy=True)``. The
        view aliases persistent comm state that every dispatch/combine on
        this Buffer overwrites — never let it (or any tensor sharing its
        storage) cross into autograd-saved state.
        """
        return self._require_ctx()['hidden_buf_local']

    @property
    def router_weight_buffer_view(self) -> torch.Tensor:  # 本 rank 的 [NvS] fp32 路由权重缓冲视图
        """fp32 view of the local rank's [NvS] route-weights comm buffer.

        Same aliasing/lifetime rules as ``hidden_nvsh_buffer_view``.
        """
        return self._require_ctx()['weights_buf_local'].view(torch.float32)  # 把 int32 存储重解释为 fp32

    def _require_ctx(self) -> dict:  # 取上下文并校验生命周期
        assert not self._destroyed, "MoonEP Buffer has been destroyed"  # 已销毁则拒绝
        assert self._ctx is not None, "MoonEP Buffer is not initialized"  # 未初始化则拒绝
        return self._ctx

    def destroy(self) -> None:  # 同步并释放本 Buffer 持有的全部 VMM/NVLink 资源
        """Synchronize and release all VMM/NVLink resources held by this Buffer.

        Must be called before tearing down the process group. Idempotent.
        """
        if self._destroyed:  # 幂等：已销毁直接返回
            return
        ctx = self._ctx  # 取上下文
        if ctx is None:  # 从未初始化完成
            self._destroyed = True
            return

        if self._comm_stream is not None:  # 先同步通信流上的在途 kernel
            self._comm_stream.synchronize()
        torch.cuda.synchronize()  # 全设备同步
        group = ctx.get('group')  # 取进程组
        if dist.is_initialized():  # 进程组仍存活时做跨 rank 屏障
            dist.barrier(group=group)

        # Drop views before owners. Tensor deleters release VMM mappings and
        # multicast handles once the Python references disappear.
        # ↑ 中文补充：先丢弃视图再释放持有者；Python 引用消失后张量删除器
        # 会释放 VMM 映射与 multicast 句柄。
        for key in (  # 逐个弹出缓冲引用，触发底层释放
            'hidden_buf_local',  # 本地 hidden 视图
            'weights_buf_local',  # 本地权重视图
            'meta_mc',  # multicast meta 视图
            'meta_buf',  # meta 缓冲
            'hidden_buf',  # hidden 共享缓冲
            'alloc',  # 规划 alloc
            'group_tokens',  # 组 token 计数
            'z',  # 转移矩阵
            'local_hist',  # 本地直方图
            'grid_sync_bar',  # grid 屏障
            'primary_packed',  # 主键打包
            'kmask',  # top-k 位图
            'kidx_to_loff',  # 键到槽位映射
            'builder_bar',  # builder 屏障
        ):
            ctx.pop(key, None)
        ctx.clear()  # 清空上下文
        self._ctx = None  # 置空上下文引用
        self._comm_stream = None  # 释放通信流引用
        self._destroyed = True  # 标记已销毁

    def __del__(self):  # 析构：视配置自动销毁或告警
        if getattr(self, "_destroyed", True):  # 已销毁或未完成初始化则跳过
            return
        if getattr(self, "explicitly_destroy", False):  # 要求显式销毁时仅告警
            warnings.warn(
                "MoonEP Buffer was not destroyed explicitly; resources may leak.",
                ResourceWarning,
            )
            return
        try:
            self.destroy()  # 自动销毁释放资源
        except Exception as exc:  # 析构失败转为告警，避免解释器期异常
            warnings.warn(
                f"MoonEP Buffer destructor failed to destroy resources: {exc!r}",
                ResourceWarning,
            )

    @staticmethod
    def _record_streams(tensors, stream: torch.cuda.Stream) -> None:  # 为跨流使用的张量登记流，防止缓存分配器过早复用
        for t in tensors:  # 逐张量登记
            if t is not None:  # 跳过空槽位
                t.record_stream(stream)

    @staticmethod
    def _plan_runtime_tensors(plan: MoonEPCommPlan):  # 收集计划的全部运行期张量（跨流登记用）
        return (
            plan.dst,  # dst 输出
            plan.experts_to_copy,  # 专家拷贝表
            plan.zero_fill_ranges,  # 零填充区间
            plan.remote_stats,  # 远端统计
            plan.dup_groups,  # 去重组
            plan.dup_loffs,  # 重复 loff
            plan.dup_counts,  # 去重计数
        )

    def _run_dispatch_on_current_stream(  # 在当前流上执行完整 dispatch 流水
        self,
        ctx: dict,
        hidden_sh: torch.Tensor,
        route_weights_sk: torch.Tensor | None,
        planning_args,
        plan: MoonEPCommPlan,
        hidden_nvsh: torch.Tensor,
        route_weights_nvs: torch.Tensor | None,
        *,
        inter_rank_sync: bool,
        zero_copy: bool,
        route_weights_zero_copy: bool,
    ) -> None:
        if inter_rank_sync:  # 可选：规划前做跨 rank 预同步
            launch_inter_rank_sync(ctx)

        if planning_args is not None:  # 新鲜规划路径：先跑规划 kernel
            topk_flat, tokens_per_expert, cu_seqlens = planning_args  # 取规划输入
            launch_planning(ctx, topk_flat, tokens_per_expert, cu_seqlens, plan)  # 发射规划

        # Fresh planning publishes dst/src_info immediately before dispatch, so
        # dispatch can safely materialize the plan-owned dedup structures from
        # those temporaries. Plan-reuse paths must keep the saved structures
        # untouched. Dispatch also zero-fills the segment-padding rows of the
        # local shard (zero warp, plan.zero_fill_ranges).
        # ↑ 中文补充：新鲜规划在 dispatch 前立即发布 dst/src_info，dispatch 可安全地
        # 从这些临时量物化计划持有的去重结构；计划复用路径不得改动已保存结构。
        # dispatch 同时按 zero_fill_ranges 清零本地 shard 的 padding 行。
        launch_dispatch(  # 发射 dispatch 主 kernel
            ctx,
            hidden_sh,  # 输入 token
            route_weights_sk,  # 路由权重输入（可空）
            plan,  # 规划计划
            build_dedup_map=planning_args is not None,  # 仅新鲜规划时重建去重映射
            pdl_trigger=self.enable_pdl,  # PDL 触发开关
        )
        # In-place duplicate expansion on the NVL shard: after this the shard
        # holds the full user-visible [NvS, H] layout.
        # ↑ 中文补充：在 NVL shard 上就地展开重复项；此后 shard 即为完整的
        # 用户可见 [NvS, H] 布局。
        launch_dispatch_epilogue(ctx, plan, pdl_launch=self.enable_pdl)
        # master-style boundary copies (same stream, plain SM copies), gated
        # independently per output tensor.
        # ↑ 中文补充：master 风格的边界拷贝（同流、普通 SM 拷贝），每个输出张量独立开关。
        if not zero_copy:  # 非零拷贝：把 shard 拷出到独立输出张量
            hidden_nvsh.copy_(ctx['hidden_buf_local'])
        if route_weights_nvs is not None and not route_weights_zero_copy:  # 权重同理，独立开关
            route_weights_nvs.copy_(
                ctx['weights_buf_local'].view(torch.float32)
            )

    def _run_combine_on_current_stream(  # 在当前流上执行完整 combine 流水
        self,
        ctx: dict,
        hidden_sh: torch.Tensor,
        plan: MoonEPCommPlan,
        hidden_nvsh: torch.Tensor,
        route_weights_nvs: torch.Tensor | None,
        route_weights_sk: torch.Tensor | None,
        *,
        inter_rank_sync: bool,
        zero_copy: bool,
        router_weights_zero_copy: bool,
    ) -> None:
        # Pre-staging sync keeps its master-era position; combine's own entry
        # cross_rank_barrier publishes the staged + accumulated NVL writes.
        # ↑ 中文补充：预暂存同步保留 master 时代的位置；combine 自己入口的
        # cross_rank_barrier 负责发布暂存 + 累加后的 NVL 写入。
        if inter_rank_sync:  # 可选：预同步
            launch_inter_rank_sync(ctx)
        # master-style boundary copies into the shard (same stream), gated
        # independently per input tensor.
        # ↑ 中文补充：master 风格的入 shard 边界拷贝（同流），每个输入张量独立开关。
        if not zero_copy:  # 非零拷贝：先把输入拷入 shard
            ctx['hidden_buf_local'].copy_(hidden_nvsh)
        if route_weights_nvs is not None and not router_weights_zero_copy:  # 权重同理
            ctx['weights_buf_local'].copy_(
                route_weights_nvs.view(torch.int32)
            )
        # In-place fp32 accumulation of duplicate rows into their primary.
        # ↑ 中文补充：把重复行按 fp32 就地累加到其主行。
        launch_combine_prologue(ctx, plan, pdl_trigger=self.enable_pdl)
        launch_combine(  # combine 主 kernel：按 dst 反向 gather 并 K 求和
            ctx,
            hidden_sh,  # token-major 输出
            plan.dst,  # 规划的 dst 编码
            output_sk=route_weights_sk,  # 可选：gather 回的 [S, K] 权重
            pdl_launch=self.enable_pdl,
        )

    def _run_reduce_grad_on_current_stream(  # 在当前流上执行梯度归约
        self,
        ctx: dict,
        experts_to_copy: torch.Tensor,
        grad_reduce_args,
    ) -> None:
        _launch_grad_reduces(ctx, experts_to_copy, *grad_reduce_args)  # 展开（本地梯度, 归约缓冲）逐对发射

    def _run_prefetch_weight_on_current_stream(  # 在当前流上执行权重预取
        self,
        ctx: dict,
        experts_to_copy: torch.Tensor,
        weight_prefetch_args,
        scale_prefetch_args=None,
    ) -> None:
        launch_prefetch(  # 把本 rank 专家权重推到各 rank 的预取槽位
            *weight_prefetch_args,  # (本地权重三元组, 预取缓冲三元组)
            experts_to_copy,  # 远端专家拷贝表
            scales=scale_prefetch_args,  # 可选的 scale 预取参数
            rank=int(ctx['rank']),  # 当前 rank
            num_sms=int(ctx['num_sms']),  # SM 数
            meta_buf=ctx['meta_buf'],  # meta 缓冲（屏障用）
            meta_stride=int(ctx['meta_chunk_padded']),  # meta chunk 步长
            barrier_off=int(ctx['BARRIER_OFF']),  # 屏障区偏移
            grid_sync_bar=ctx['grid_sync_bar'],  # 软件 grid 屏障
        )

    def dispatch(  # dispatch 前向：规划（或复用计划）并把 token 按专家分组 scatter 到远端
        self,
        hidden_sh: torch.Tensor,
        route_weights_sk: torch.Tensor | None = None,
        topk_experts_sk: torch.Tensor | None = None,
        tokens_per_expert: torch.Tensor | None = None,
        plan: MoonEPCommPlan | None = None,
        async_finish: bool = False,
        *,
        inter_rank_sync: bool = True,
        zero_copy: bool = False,
        router_weights_zero_copy: bool = False,
    ):
        """dispatch fwd: run planning (unless reusing a plan) and scatter tokens
        to their expert-grouped positions on remote ranks.

        Args:
            hidden_sh: [S, H] bf16 input tokens.
            route_weights_sk: [S, K] fp32 routing weights; None skips the
                weights buffer entirely.
            topk_experts_sk: [S, K] int32 expert ids; required when ``plan``
                is None.
            tokens_per_expert: [E] int32 local token count per expert;
                required when ``plan`` is None.
            plan: saved MoonEPCommPlan to reuse — planning is skipped and
                ``topk_experts_sk`` / ``tokens_per_expert`` are ignored. This
                is the combine bwd path: re-dispatching ``grad_output_sh``
                with the fwd plan scatters the output grad back to VM group
                order.
            async_finish: run on the comm stream and append a CUDA event to
                the return value.
            inter_rank_sync: run a CuTe DSL rank sync before planning
                (default True).
            zero_copy: return a view of the communication buffer
                (``hidden_buf_local``) instead of a fresh tensor. The view
                aliases state that the next dispatch/combine on this Buffer
                overwrites, so callers must not keep it across communication
                calls (in particular autograd must not save it for backward
                — that is exactly the case that requires ``zero_copy=False``).
                Row content is only defined within the ``cu_seqlens``-covered
                padded segments.
            router_weights_zero_copy: like ``zero_copy`` but for
                ``route_weights_nvs`` (the fp32 view of ``weights_buf_local``).
                Defaults to False even when ``zero_copy=True``: the weights
                view is the dangerous special case (tiny tensor, commonly
                saved into autograd state by training frameworks), so callers
                that only consume it before the next dispatch — e.g.
                inference — opt in explicitly.

        Returns:
            ``(hidden_nvsh, route_weights_nvs, cu_seqlens, plan)``, plus a
            comm-stream CUDA event when ``async_finish=True``:

            - hidden_nvsh: [NvS, H] bf16 dispatched tokens in physical VM
              group order.
            - route_weights_nvs: [NvS] fp32, or None when ``route_weights_sk``
              is None.
            - cu_seqlens: [2*epn] int32 padded token end offset per VM group
              row, where epn=E/R; None on the plan-reuse path.
            - plan: MoonEPCommPlan; save it for prefetch/combine and both
              backward passes.
        """
        ctx = self._require_ctx()  # 取上下文（校验未销毁）

        if plan is None:  # 新鲜规划：校验并分配输出
            assert topk_experts_sk is not None and tokens_per_expert is not None  # 新鲜规划必须提供路由输入
            topk_flat = topk_experts_sk.reshape(-1)  # topk 展平为 [N]
            assert topk_flat.dtype == torch.int32 and topk_flat.numel() == int(ctx['N'])  # 类型与长度校验
            assert tokens_per_expert.dtype == torch.int32  # tpe 必须 int32
            assert tokens_per_expert.numel() == int(ctx['E']) and tokens_per_expert.is_contiguous()  # 长度 E 且连续
            plan, cu_seqlens = allocate_planning_outputs(ctx)  # 分配计划与 cu_seqlens
            planning_args = (topk_flat, tokens_per_expert, cu_seqlens)  # 打包规划输入
        else:
            cu_seqlens = None  # 复用路径不返回 cu_seqlens
            planning_args = None  # 复用路径不重跑规划
            assert isinstance(plan, MoonEPCommPlan)  # plan 类型校验

        if zero_copy:  # 零拷贝：直接返回 shard 视图
            hidden_nvsh = ctx['hidden_buf_local']
        else:  # 常规：分配独立输出张量
            hidden_nvsh = torch.empty_like(ctx['hidden_buf_local'])
        if route_weights_sk is None:  # 未提供权重则整路跳过
            route_weights_nvs = None
        elif router_weights_zero_copy:  # 权重零拷贝：返回 shard 的 fp32 视图
            route_weights_nvs = ctx['weights_buf_local'].view(torch.float32)
        else:  # 常规：分配独立权重输出
            route_weights_nvs = torch.empty(
                ctx['NvS'], dtype=torch.float32, device=ctx['meta_buf'].device
            )

        if not async_finish:  # 同步模式：直接在当前流执行并返回
            self._run_dispatch_on_current_stream(
                ctx,
                hidden_sh,
                route_weights_sk,
                planning_args,
                plan,
                hidden_nvsh,
                route_weights_nvs,
                inter_rank_sync=inter_rank_sync,
                zero_copy=zero_copy,
                route_weights_zero_copy=router_weights_zero_copy,
            )
            return hidden_nvsh, route_weights_nvs, cu_seqlens, plan

        main_stream = torch.cuda.current_stream()  # 主流（输入产出流）
        comm = self._comm_stream  # 异步通信流
        assert comm is not None, "MoonEP Buffer communication stream is not initialized"  # 通信流必须已创建

        tensors_to_record = [  # 汇总需登记到通信流的张量
            hidden_sh,  # 输入 token
            route_weights_sk,  # 输入权重（可空）
            hidden_nvsh,  # 输出 token
            route_weights_nvs,  # 输出权重（可空）
            *self._plan_runtime_tensors(plan),  # 计划的全部运行期张量
        ]
        if planning_args is not None:  # 新鲜规划：登记规划输入/输出
            tensors_to_record.extend(planning_args)
        self._record_streams(tensors_to_record, comm)  # 执行跨流登记

        input_ready = main_stream.record_event()  # 主流记录事件：输入就绪
        comm.wait_event(input_ready)  # 通信流等待输入就绪

        with torch.cuda.stream(comm):  # 切到通信流执行 dispatch
            self._run_dispatch_on_current_stream(
                ctx,
                hidden_sh,
                route_weights_sk,
                planning_args,
                plan,
                hidden_nvsh,
                route_weights_nvs,
                inter_rank_sync=inter_rank_sync,
                zero_copy=zero_copy,
                route_weights_zero_copy=router_weights_zero_copy,
            )
            done = comm.record_event()  # 通信流记录完成事件

        return hidden_nvsh, route_weights_nvs, cu_seqlens, plan, done  # 异步模式多返回完成事件

    def prefetch_weight(  # 按 plan 把本 rank 专家权重推到所有 rank 的预取槽位
        self,
        plan: MoonEPCommPlan | None = None,
        async_finish: bool = False,
        *,
        local_gate_weight: torch.Tensor | None = None,
        local_up_weight: torch.Tensor | None = None,
        local_down_weight: torch.Tensor | None = None,
        gate_prefetch_buffer: torch.Tensor | None = None,
        up_prefetch_buffer: torch.Tensor | None = None,
        down_prefetch_buffer: torch.Tensor | None = None,
        local_gate_scale: torch.Tensor | None = None,
        local_up_scale: torch.Tensor | None = None,
        local_down_scale: torch.Tensor | None = None,
        gate_scale_prefetch_buffer: torch.Tensor | None = None,
        up_scale_prefetch_buffer: torch.Tensor | None = None,
        down_scale_prefetch_buffer: torch.Tensor | None = None,
    ):
        """Push local expert weights selected by ``plan`` into all ranks'
        prefetch slots (dispatch fwd, weight side).

        Args:
            plan: MoonEPCommPlan returned by ``dispatch``.
            async_finish: run on the comm stream and return a CUDA event.
            local_gate_weight / local_up_weight / local_down_weight:
                [epn, H, H'] contiguous local expert weights, where epn=E/R.
                bf16 for unquantized experts, uint8 for packed MXFP4.
            gate_prefetch_buffer / up_prefetch_buffer /
            down_prefetch_buffer:
                [R, epn, H, H'] all-rank prefetch-buffer views, contiguous
                within each rank with optional padding between ranks.
            local_gate_scale / local_up_scale / local_down_scale:
                optional [epn, ...] contiguous local block-scale tensors.
            gate_scale_prefetch_buffer / up_scale_prefetch_buffer /
            down_scale_prefetch_buffer:
                optional [R, epn, ...] all-rank scale-buffer views with the
                same rank-contiguous layout (each pool has its own stride).

        Returns:
            None in synchronous mode, or the comm-stream CUDA event when
            ``async_finish=True``.

        Kept separate from ``dispatch`` so the plan-reuse path (combine bwd)
        can skip re-prefetching. Calling this after
        ``dispatch(async_finish=True)`` preserves stream ordering, and the
        returned event also covers the queued dispatch.
        """
        ctx = self._require_ctx()  # 取上下文

        assert isinstance(plan, MoonEPCommPlan), "Buffer.prefetch_weight: plan is required"  # 必须提供计划
        R = int(ctx['R'])  # rank 数
        epn = int(ctx['E']) // R  # 每 rank 专家数
        local_weights = (local_gate_weight, local_up_weight, local_down_weight)  # 本地权重三元组
        prefetch_buffers = (  # 预取缓冲三元组
            gate_prefetch_buffer,
            up_prefetch_buffer,
            down_prefetch_buffer,
        )
        assert all(t is not None for t in (*local_weights, *prefetch_buffers)), \
            "prefetch_weight tensors must be provided together"  # 权重与缓冲必须成套提供
        for local, pool in zip(local_weights, prefetch_buffers, strict=True):  # 逐对校验布局
            assert local.dtype in _ELEM_TYPES, \
                f"prefetch_weight: unsupported weight dtype {local.dtype}"  # dtype 必须受支持
            assert local.is_contiguous()  # 本地权重必须连续
            assert local.ndim == 3 and int(local.shape[0]) == epn  # 形状 [epn, H, H']
            assert pool.dtype == local.dtype and _validate_rank_strided_pool(pool)  # 缓冲 dtype 一致且布局合法
            assert pool.ndim == 4 and tuple(pool.shape[:2]) == (R, epn)  # 缓冲形如 [R, epn, ...]
            assert tuple(pool.shape[2:]) == tuple(local.shape[1:])  # 后两维与本地权重匹配
            assert pool.device == local.device  # 同设备
        weight_prefetch_args = (local_weights, prefetch_buffers)  # 打包权重预取参数

        local_scales = (local_gate_scale, local_up_scale, local_down_scale)  # 本地 scale 三元组
        scale_prefetch_buffers = (  # scale 预取缓冲三元组
            gate_scale_prefetch_buffer,
            up_scale_prefetch_buffer,
            down_scale_prefetch_buffer,
        )
        if any(t is not None for t in (*local_scales, *scale_prefetch_buffers)):  # 提供了 scale 则必须成套
            assert all(t is not None for t in (*local_scales, *scale_prefetch_buffers)), \
                "prefetch_weight scales must be provided together"
            for local, pool in zip(  # 逐对校验 scale 布局
                local_scales, scale_prefetch_buffers, strict=True,
            ):
                assert local.is_contiguous()  # 本地 scale 必须连续
                assert local.ndim >= 2 and int(local.shape[0]) == epn  # 首维为 epn
                assert pool.dtype == local.dtype and _validate_rank_strided_pool(pool)  # 缓冲布局合法
                assert pool.ndim == local.ndim + 1  # 缓冲多一个 rank 维
                assert tuple(pool.shape[:2]) == (R, epn)  # 缓冲形如 [R, epn, ...]
                assert tuple(pool.shape[2:]) == tuple(local.shape[1:])  # 尾维匹配
                assert pool.device == local.device  # 同设备
            scale_prefetch_args = tuple(zip(  # 打包 scale 预取参数
                local_scales, scale_prefetch_buffers, strict=True,
            ))
        else:
            scale_prefetch_args = None  # 未提供 scale

        if not async_finish:  # 同步模式：当前流执行
            self._run_prefetch_weight_on_current_stream(
                ctx,
                plan.experts_to_copy,  # 远端专家拷贝表
                weight_prefetch_args,
                scale_prefetch_args,
            )
            return None

        main_stream = torch.cuda.current_stream()  # 主流
        comm = self._comm_stream  # 通信流
        assert comm is not None, "MoonEP Buffer communication stream is not initialized"  # 通信流必须存在

        self._record_streams(  # 登记计划张量与全部权重/缓冲
            (
                plan.experts_to_copy,
                *local_weights,
                *prefetch_buffers,
                *(local_scales if scale_prefetch_args else ()),  # 有 scale 才登记
                *(scale_prefetch_buffers if scale_prefetch_args else ()),
            ),
            comm,
        )
        input_ready = main_stream.record_event()  # 主流记录事件
        comm.wait_event(input_ready)  # 通信流等待

        with torch.cuda.stream(comm):  # 通信流执行预取
            self._run_prefetch_weight_on_current_stream(
                ctx,
                plan.experts_to_copy,
                weight_prefetch_args,
                scale_prefetch_args,
            )
            done = comm.record_event()  # 记录完成事件

        return done  # 返回完成事件

    def combine(  # combine 前向：从 NVL 缓冲 gather 专家输出并 K 求和回 token-major
        self,
        plan: MoonEPCommPlan | None = None,
        hidden_nvsh: torch.Tensor | None = None,
        route_weights_nvs: torch.Tensor | None = None,
        async_finish: bool = False,
        inter_rank_sync: bool = True,
        *,
        zero_copy: bool = False,
        router_weights_zero_copy: bool = False,
    ):
        """combine fwd: gather expert outputs from the NVL buffer and K-sum
        back to token-major [S, H].

        Also serves as dispatch bwd: combining ``grad_hidden_nvsh`` sums each
        token's K dispatched grad copies back to its token-major grad.

        Args:
            plan: MoonEPCommPlan returned by ``dispatch``.
            hidden_nvsh: [NvS, H] bf16 expert outputs in physical VM group
                order.
            route_weights_nvs: [NvS] fp32, optional; pass the
                dispatch-returned weights to also gather them back to
                token-major.
            async_finish: run on the comm stream and return a CUDA event.
            inter_rank_sync: run a CuTe DSL rank sync before staging
                (default True).
            zero_copy: ``hidden_nvsh`` must be exactly the view returned by a
                ``zero_copy=True`` dispatch — the caller's FFN writes its
                output in place on the shard and no boundary copy is performed
                (asserted via ``data_ptr()``). With ``zero_copy=False`` the
                input is an ordinary tensor that is first copied into the
                shard.
            router_weights_zero_copy: same contract for ``route_weights_nvs``
                (the fp32 view of ``weights_buf_local``); default False, i.e.
                an ordinary tensor that is copied into the shard first.

        Returns:
            ``(hidden_sh, route_weights_sk, event)``:

            - hidden_sh: [S, H] bf16 combined token-major output, allocated
              by MoonEP.
            - route_weights_sk: [S, K] fp32 gathered routing weights, or None
              when ``route_weights_nvs`` is None.
            - event: comm-stream CUDA event when ``async_finish=True``, else
              None.
        """
        ctx = self._require_ctx()  # 取上下文

        assert isinstance(plan, MoonEPCommPlan), "Buffer.combine: plan is required"  # 必须提供计划

        assert hidden_nvsh is not None  # 必须提供专家输出
        assert hidden_nvsh.dtype == torch.bfloat16  # 必须 bf16
        assert hidden_nvsh.is_contiguous()  # 必须连续
        assert tuple(hidden_nvsh.shape) == (int(ctx['NvS']), int(ctx['H']))  # 形状 [NvS, H]
        if route_weights_nvs is not None:  # 提供权重时校验布局
            assert route_weights_nvs.dtype == torch.float32  # 必须 fp32
            assert route_weights_nvs.is_contiguous()  # 必须连续
            assert tuple(route_weights_nvs.shape) == (int(ctx['NvS']),)  # 形状 [NvS]
        if zero_copy:  # 零拷贝：输入必须精确别名 shard 视图
            assert hidden_nvsh.data_ptr() == ctx['hidden_buf_local'].data_ptr(), (
                "combine(zero_copy=True): hidden_nvsh must alias the NVL shard "
                "view returned by dispatch(zero_copy=True)"
            )
        if router_weights_zero_copy and route_weights_nvs is not None:  # 权重零拷贝同理
            assert route_weights_nvs.data_ptr() == \
                ctx['weights_buf_local'].data_ptr(), (
                "combine(router_weights_zero_copy=True): route_weights_nvs must "
                "alias the NVL weights view returned by "
                "dispatch(router_weights_zero_copy=True)"
            )

        hidden_sh = torch.empty(  # 分配 token-major 输出
            int(ctx['S']),
            int(ctx['H']),
            dtype=hidden_nvsh.dtype,
            device=hidden_nvsh.device,
        )
        route_weights_sk = (  # 可选：分配 [S, K] 权重输出
            torch.empty(
                int(ctx['S']),
                int(ctx['K']),
                dtype=torch.float32,
                device=hidden_nvsh.device,
            )
            if route_weights_nvs is not None else None
        )

        if not async_finish:  # 同步模式：当前流执行
            self._run_combine_on_current_stream(
                ctx,
                hidden_sh,
                plan,
                hidden_nvsh,
                route_weights_nvs,
                route_weights_sk,
                inter_rank_sync=inter_rank_sync,
                zero_copy=zero_copy,
                router_weights_zero_copy=router_weights_zero_copy,
            )
            return hidden_sh, route_weights_sk, None

        main_stream = torch.cuda.current_stream()  # 主流
        comm = self._comm_stream  # 通信流
        assert comm is not None, "MoonEP Buffer communication stream is not initialized"  # 通信流必须存在

        self._record_streams(  # 登记输出/计划/输入张量
            (
                hidden_sh,
                *self._plan_runtime_tensors(plan),
                hidden_nvsh,
                route_weights_nvs,
                route_weights_sk,
            ),
            comm,
        )
        input_ready = main_stream.record_event()  # 主流记录事件
        comm.wait_event(input_ready)  # 通信流等待

        with torch.cuda.stream(comm):  # 通信流执行 combine
            self._run_combine_on_current_stream(
                ctx,
                hidden_sh,
                plan,
                hidden_nvsh,
                route_weights_nvs,
                route_weights_sk,
                inter_rank_sync=inter_rank_sync,
                zero_copy=zero_copy,
                router_weights_zero_copy=router_weights_zero_copy,
            )
            done = comm.record_event()  # 记录完成事件

        return hidden_sh, route_weights_sk, done  # 返回输出与事件

    def reduce_grad(  # 把重复专家的权重梯度归约回其 home rank
        self,
        plan: MoonEPCommPlan | None = None,
        async_finish: bool = False,
        *,
        local_gate_grad: torch.Tensor | None = None,
        local_up_grad: torch.Tensor | None = None,
        local_down_grad: torch.Tensor | None = None,
        gate_reduce_buffer: torch.Tensor | None = None,
        up_reduce_buffer: torch.Tensor | None = None,
        down_reduce_buffer: torch.Tensor | None = None,
    ):
        """Reduce duplicated experts' weight grads back to their home ranks
        (dispatch bwd, weight side).

        Each rank remote-reads the prefetch-slot grads of its own experts
        from every rank's reduce buffer, accumulates them into its local
        grad rows, then zeros its own consumed slots for the next
        microbatch.

        Args:
            plan: MoonEPCommPlan returned by ``dispatch``.
            async_finish: run on the comm stream and return a CUDA event.
            local_gate_grad / local_up_grad / local_down_grad:
                [epn, H, H'] fp32 contiguous local parameter grads.
            gate_reduce_buffer / up_reduce_buffer / down_reduce_buffer:
                [R, epn, H, H'] fp32 — all R ranks' reduce buffers mapped as
                one view, contiguous within each rank with optional rank padding.

        Returns:
            None in synchronous mode, or the comm-stream CUDA event when
            ``async_finish=True``.

        Kept separate from ``combine``; running on the shared comm stream
        keeps the Buffer's barrier/meta resources serialized. The caller
        must ensure all ranks' reduce-buffer writes are visible to remote
        TMA reads and prior reductions' clears have completed before buffer
        reuse.
        """
        ctx = self._require_ctx()  # 取上下文

        R = int(ctx['R'])  # rank 数
        epn = int(ctx['E']) // R  # 每 rank 专家数
        local_grads = (local_gate_grad, local_up_grad, local_down_grad)  # 本地梯度三元组
        reduce_buffers = (  # 归约缓冲三元组
            gate_reduce_buffer,
            up_reduce_buffer,
            down_reduce_buffer,
        )
        assert all(t is not None for t in (*local_grads, *reduce_buffers)), \
            "reduce_grad tensors must be provided together"  # 梯度与缓冲必须成套提供
        assert isinstance(plan, MoonEPCommPlan), "Buffer.reduce_grad: plan is required"  # 必须提供计划
        for local, pool in zip(local_grads, reduce_buffers, strict=True):  # 逐对校验布局
            assert local.dtype == torch.float32 and local.is_contiguous()  # 梯度必须 fp32 连续
            assert local.ndim == 3 and int(local.shape[0]) == epn  # 形状 [epn, H, H']
            assert pool.dtype == torch.float32 and _validate_rank_strided_pool(pool)  # 缓冲 fp32 且布局合法
            assert pool.ndim == 4 and tuple(pool.shape[:2]) == (R, epn)  # 缓冲形如 [R, epn, ...]
            assert tuple(pool.shape[2:]) == tuple(local.shape[1:])  # 尾维匹配
            assert pool.device == local.device  # 同设备
        grad_reduce_args = (local_grads, reduce_buffers)  # 打包归约参数

        if not async_finish:  # 同步模式：当前流执行
            self._run_reduce_grad_on_current_stream(
                ctx,
                plan.experts_to_copy,  # 远端专家拷贝表
                grad_reduce_args,
            )
            return None

        main_stream = torch.cuda.current_stream()  # 主流
        comm = self._comm_stream  # 通信流
        assert comm is not None, "MoonEP Buffer communication stream is not initialized"  # 通信流必须存在

        self._record_streams(  # 登记计划张量与梯度/缓冲
            (plan.experts_to_copy, *local_grads, *reduce_buffers), comm,
        )
        input_ready = main_stream.record_event()  # 主流记录事件
        comm.wait_event(input_ready)  # 通信流等待

        with torch.cuda.stream(comm):  # 通信流执行归约
            self._run_reduce_grad_on_current_stream(
                ctx,
                plan.experts_to_copy,
                grad_reduce_args,
            )
            done = comm.record_event()  # 记录完成事件

        return done  # 返回完成事件
