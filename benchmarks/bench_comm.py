"""
MoonEP intranode dispatch/combine operator benchmark.

Sweeps:
  - hidden_size: 3584, 7168
  - topk:        8, 16
  - ep_size:     4, 8  (sub-groups carved out of the global world)
  - seq_length:  8192
  - bias_ratio:  0.1, 1, 5
                 sigma of the lognormal expert-logit distribution (mirrors
                 the unbalance forcing used in training). 0.1 -> near
                 balanced; 1 -> typical dropless-MoE skew; 5 -> near-degenerate.

E is fixed at 896 total experts; experts-per-rank (epn = E / ep) shrinks as
the EP group grows.

Reports the MoonEP communication operators:
  - dispatch_fwd: dispatch kernel with route-weight scatter (dedup: only
    representative ``dst >= 0`` rows are transferred).
  - dispatch_bwd: hidden-only dispatch kernel with the saved plan.
  - epilogue_fwd: in-place duplicate expansion on the NVL shard
    (dispatch_epilogue kernel), timed after dispatch_fwd.
  - epilogue_bwd: same expansion timed after the saved-plan dispatch_bwd.
  - combine_prologue_fwd: in-place fp32 accumulation of duplicate rows into
    their primary (combine_prologue kernel).
  - combine_prologue_bwd: same accumulation timed on the backward staging
    state.
  - combine_fwd:  dedup-aware combine kernel after the prologue.
  - combine_bwd:  dedup-aware combine kernel with droute_weights_sk gather
                  after the prologue.
  - prefetch:     remote expert-weight prefetch driven by the planner's
                  experts_to_copy. Local owners push into the all-rank pool;
                  the kernel includes collective completion.
  - grad_reduce:  remote expert-grad reduction (cross-rank barrier built in).

Expert weights/grads use compact local [epn, H, Hp] tensors and NVL-distributed
communication pools [R, epn, H, Hp].
max_recv is the max over ranks of prefetched expert count (slots with
experts_to_copy >= 0); max_send is the max over owner ranks of those slots.
Both owner-push prefetch and local-destination grad_reduce bandwidth use
max_send * H * Hp as the critical byte count (bf16 for prefetch, fp32 for
grad_reduce). mx/mean is the global
expert-load imbalance
(max over experts of routed token count, divided by the mean).

The epilogue/prologue bandwidth columns use total HBM traffic (read + write
bytes divided by time; route weights are negligible and ignored):
  epilogue = (dup_group_count + dup_loff_count) * H * 2  (read primaries,
             write duplicates)
  prologue = (2 * dup_group_count + dup_loff_count) * H * 2  (read primary +
             duplicates, write primary)

Planning is run once per config to populate `dst` and src_info. Fresh forward
dispatch materializes the plan-owned dedup structures with its builder warps
(time included in dispatch_fwd) and zero-fills padding with the zero warp.
Saved-plan backward dispatch reuses the existing dedup structures. The
zero_copy=False boundary ``tensor.copy_`` calls are plain torch ops, not
MoonEP kernels, and are not benchmarked here (with zero_copy=True they
disappear entirely).

Set ``MOONEP_NUM_SMS_DEDUP=<n>`` before launching to override the SM count used
by the local dispatch_epilogue and combine_prologue kernels. By default they
use all SMs on the current device.
"""

# 中文说明：MoonEP 节点内 dispatch/combine 算子基准。
# 扫描维度：hidden 3584/7168，topk 8/16，ep 4/8（从全局进程组切出子组），
#   序列长 8192，bias_ratio 0.1/1/5（lognormal 专家打分的 sigma，模拟训练中
#   的不均衡：0.1≈均衡，1≈典型 dropless-MoE 偏斜，5≈接近退化）。
# 专家总数固定 896，ep 越大每 rank 专家数（epn=E/ep）越少。
# 报告的算子（详见各类注释）：dispatch_fwd（带路由权重 scatter、去重后仅传
#   代表行）、dispatch_bwd（saved-plan 仅 hidden）、epilogue_fwd/bwd（NVL
#   shard 上的原位重复展开）、combine_prologue_fwd/bwd（fp32 原位累加重复行）、
#   combine_fwd/bwd（去重感知的 combine，bwd 另带 droute_weights gather）、
#   prefetch（按 experts_to_copy 的属主推送前取）、grad_reduce（内置跨 rank
#   屏障的梯度归约）。
# 字节口径：专家权重/梯度用紧凑本地 [epn, H, Hp] 张量与 NVL 分布式池
#   [R, epn, H, Hp]；max_recv/max_send 分别是接收/发送最重 rank 的槽位数；
#   prefetch 与 grad_reduce 带宽按 max_send*H*Hp 关键字节折算（bf16/fp32）；
#   mx/mean 为全局专家负载不均衡度（最热专家路由 token 数 / 平均）。
#   epilogue/prologue 带宽按总 HBM 流量（读+写字节 / 时间；路由权重忽略）：
#   epilogue=(组数+跨rank行数)*H*2，prologue=(2*组数+跨rank行数)*H*2。
# 每个配置先跑一次规划填充 dst 与 src_info；首次前向 dispatch 由 builder warp
#   物化去重结构（时间计入 dispatch_fwd）并用 zero warp 清零填充行；
#   saved-plan 反向 dispatch 复用已有去重结构。zero_copy=False 的边界
#   tensor.copy_ 是普通 torch op，不属于 MoonEP 内核，不在本基准范围内
#   （zero_copy=True 时该拷贝完全消失）。
# 启动前可设 MOONEP_NUM_SMS_DEDUP=<n> 覆盖本地 dispatch_epilogue 与
#   combine_prologue 内核的 SM 数；默认用当前设备的全部 SM。

# argparse：解析命令行参数
import argparse
# csv：写出结果表格
import csv
# gc：bench_one 结束后强制回收，配合清空 CUDA 缓存
import gc
# math：fmt4 中的 log10
import math
# os：读取环境变量与创建输出目录
import os
# time：统计各阶段墙钟耗时
import time
# traceback：某配置失败时打印完整堆栈
import traceback

# torch：张量、CUDA 事件与 graph；dist：进程组与集合通信
import torch
import torch.distributed as dist

# 共享的路由生成器（跨 rank 共享种子控制专家热度）
from tests.generate_topk_routing import generate_topk_routing

# MoonEP 的 Buffer 与各被测内核的启动封装
from moonep import Buffer
from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment
from moonep.dispatch import launch_dispatch
from moonep.dispatch_epilogue import launch_dispatch_epilogue
from moonep.combine import launch_combine
from moonep.combine_prologue import launch_combine_prologue
from moonep.grad_reduce import launch_grad_reduce
from moonep.planning import allocate_planning_outputs, launch_planning
from moonep.prefetch import launch_prefetch


def fmt4(x: float) -> str:
    """Format float to exactly 4 significant digits, keeping trailing zeros.

    Examples: 181.0 -> '181.0', 1486.65 -> '1487', 856.71 -> '856.7', 0.5 -> '0.5000'.
    """
    # 中文：把浮点数格式化为恰好 4 位有效数字并保留末尾零。
    if x == 0:
        # 零值特殊处理
        return "0.000"
    # 小数点前的位数 = floor(log10(|x|)) + 1
    d = math.floor(math.log10(abs(x))) + 1  # digits before the decimal point
    # 小数位数 = max(0, 4 - 整数位数)，保证 4 位有效数字
    return f"{x:.{max(0, 4 - d)}f}"


# -------------------------------------------------------------------------
# Distributed setup
# 分布式初始化
# -------------------------------------------------------------------------

def setup():
    # 初始化：绑定 GPU、建 NCCL 进程组，返回 (全局 rank, 世界大小, 本地 rank)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    return world_rank, world_size, local_rank


# -------------------------------------------------------------------------
# Benchmark a single config (called only on ranks in the EP subgroup)
# 单配置基准（仅在 EP 子组内的 rank 上执行）
# -------------------------------------------------------------------------

def time_gpu_op(launch_fn, warmup, iters, group, cudagraph=True):
    """Run launch_fn() warmup+iters times; return cross-rank mean per-iter us.

    Pure GPU-side timing: no host syncs inside the loop. The kernel's own
    cross-rank barrier serializes successive calls. Each rank in `group`
    measures its own elapsed time; we all_gather the per-rank means and return
    the average across the subgroup.

    cudagraph=True (default): capture `iters` launches and time the replay,
    removing the per-launch Python overhead (cute JIT orchestration +
    make_ptr) so the timing reflects the kernels alone. cooperative launch +
    the cross-rank NVLink barrier are capturable; output tensors are freshly
    allocated each round and the replay reuses the captured addresses.
    cudagraph=False: eager loop (includes host launch overhead).
    """
    # 中文：把 launch_fn 预热 iters 次后计时，返回子组平均的单次微秒数。
    # 纯 GPU 侧计时：循环内无 host 同步；内核自带的跨 rank 屏障会把相邻
    # 调用串行化。子组内各 rank 各自计时，all_gather 各自均值后取平均。
    # cudagraph=True（默认）：捕获 iters 次 launch 再回放计时，剔除每次
    # launch 的 Python 开销（cute JIT 编排 + make_ptr），使计时只反映内核。
    # 协作 launch 与跨 rank NVLink 屏障均可捕获；输出张量每轮新分配、回放
    # 复用捕获时地址。cudagraph=False：eager 循环（含 host launch 开销）。
    # 预热（兼做 JIT 编译）
    for _ in range(warmup):
        launch_fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # 两个 CUDA Event 夹住计时区间
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if cudagraph:
        # 捕获 iters 次调用到 graph，同步后回放计时
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                launch_fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        start.record()
        g.replay()
        end.record()
    else:
        # eager 模式：逐次 launch
        start.record()
        for _ in range(iters):
            launch_fn()
        end.record()
    end.synchronize()
    # 本 rank 平均单次时延（ms -> us）
    local_us = start.elapsed_time(end) / iters * 1e3  # ms -> us

    # 子组内 all_gather 各 rank 均值并取平均
    world_size = dist.get_world_size(group=group)
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    t = torch.tensor([local_us], dtype=torch.float64, device=dev)
    outs = [torch.empty(1, dtype=torch.float64, device=dev) for _ in range(world_size)]
    dist.all_gather(outs, t, group=group)
    return torch.cat(outs).mean().item()


def bench_one(group, group_rank, R, S, K, E, H, bias_ratio, Hp,
              num_sms=32, warmup=5, iters=20,
              cudagraph: bool = True):
    """Run one (R, S, K, E, H, Hp, bias_ratio) configuration on the given subgroup."""
    # 中文：在给定子组上运行一个 (R, S, K, E, H, Hp, bias_ratio) 配置，
    # 依次计时全部 MoonEP 通信算子并返回结果字典。
    # 本子组的 CUDA 设备
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    # 创建该配置的 Buffer（含跨 rank 屏障与 meta_buf 等共享上下文）
    buffer = Buffer(
        S, H, K, E, R, num_sms=num_sms, group=group,
        explicitly_destroy=True,
    )
    ctx = buffer._require_ctx()

    # Shared seed -> rank-shared expert popularity (the gate is global in
    # real training); rank -> independent per-token draws.
    # 共享种子 -> 各 rank 一致的专家热度（真实训练中门控是全局的）；
    # rank 种子 -> 各 token 的独立抽样。
    topk, tpe = generate_topk_routing(S, K, E, R, bias_ratio, dev, 1234,
                                      rank=group_rank)

    # Global expert-load imbalance:
    # per-expert routed token count summed over ranks, max / mean.
    # 全局专家负载不均衡度：跨 rank 求和的每专家路由 token 数，max/mean。
    global_tpe = tpe.to(torch.int64)
    dist.all_reduce(global_tpe, group=group)
    max_load = int(global_tpe.max().item())
    mean_load = float(global_tpe.sum().item()) / E
    load_max_mean = max_load / mean_load if mean_load else 0.0

    # 准备输入：hidden/路由权重/输出缓冲与拉平的 topk 索引
    hidden = torch.randn(S, H, dtype=torch.bfloat16, device=dev)
    weights = torch.rand(S, K, dtype=torch.float32, device=dev)
    output = torch.empty(S, H, dtype=torch.bfloat16, device=dev)
    topk_flat = topk.reshape(-1).contiguous()

    # 分配规划输出（计划表 + cu_seqlens）
    plan, cu_seqlens = allocate_planning_outputs(ctx)

    def _plan_call():
        # 单次规划调用：按路由填充计划表
        launch_planning(ctx, topk_flat, tpe, cu_seqlens, plan)

    # —— 算子 1：planning ——
    planning_us = time_gpu_op(_plan_call, warmup, iters, group, cudagraph=cudagraph)

    # Capture deduped `dst` plus src_info. The first forward dispatch below
    # materializes the plan-owned dedup structures (dup_groups / dup_loffs /
    # dup_counts), which then drive the dispatch epilogue / combine prologue.
    # The plan carries the full [R, epn] experts_to_copy table needed by
    # prefetch/grad_reduce.
    # 再跑一次规划以捕获去重后的 dst 与 src_info。下面的首次前向 dispatch
    # 会物化计划自有的去重结构（dup_groups/dup_loffs/dup_counts），供
    # dispatch 尾核 / combine 前核使用；计划还带 prefetch/grad_reduce 所需
    # 的完整 [R, epn] experts_to_copy 表。
    launch_planning(ctx, topk_flat, tpe, cu_seqlens, plan)
    dst = plan.dst
    experts_to_copy = plan.experts_to_copy
    # NVL shard 的 token 容量（combine 输入 staging 的大小）
    NvS = int(ctx['NvS'])
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # ---- dispatch_fwd: forward dispatch writes hidden + route weights
    # (representative rows only under dedup), zero-fills the padding rows
    # (zero warp) and materializes the dedup structures (builder warps).
    # —— dispatch_fwd：前向 dispatch 写入 hidden + 路由权重（去重下仅传
    # 代表行），用 zero warp 清零填充行，并由 builder warp 物化去重结构。
    def _dispatch_fwd_call():
        launch_dispatch(ctx, hidden, weights, plan, build_dedup_map=True)

    dispatch_fwd_us = time_gpu_op(
        _dispatch_fwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- epilogue_fwd: in-place duplicate expansion on the NVL shard
    # (load primary once, store to every duplicate slot). Timed after
    # dispatch_fwd so it reads the freshly written NVL state, matching the
    # real dispatch -> epilogue order. The zero_copy=False boundary
    # ``tensor.copy_`` is a plain torch op, not a MoonEP kernel, and is
    # deliberately not benchmarked here.
    # —— epilogue_fwd：NVL shard 上的原位重复展开（主行读一次、写入每个
    # 重复槽位）。放在 dispatch_fwd 之后计时以读到刚写入的 NVL 状态，
    # 与真实 dispatch->epilogue 顺序一致。zero_copy=False 的边界
    # tensor.copy_ 是普通 torch op，刻意不在本基准内。
    def _epilogue_fwd_call():
        launch_dispatch_epilogue(ctx, plan)

    epilogue_fwd_us = time_gpu_op(
        _epilogue_fwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- dispatch_bwd: hidden-only dispatch with the saved plan.
    # —— dispatch_bwd：复用 saved plan 的仅 hidden dispatch（反向梯度分发）。
    grad_output = torch.randn(S, H, dtype=torch.bfloat16, device=dev)

    def _dispatch_bwd_call():
        launch_dispatch(ctx, grad_output, None, plan, build_dedup_map=False)

    dispatch_bwd_us = time_gpu_op(
        _dispatch_bwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- epilogue_bwd: same in-place expansion timed after dispatch_bwd
    # (saved-plan backward state).
    # —— epilogue_bwd：同样的原位展开，在 saved-plan 反向 dispatch 之后计时。
    def _epilogue_bwd_call():
        launch_dispatch_epilogue(ctx, plan)

    epilogue_bwd_us = time_gpu_op(
        _epilogue_bwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- Combine-side paths. The combine input is staged into the NVL
    # shard once as untimed setup (zero_copy=True callers write it in place;
    # the zero_copy=False boundary copy is a plain torch op). The prologue
    # then reduces duplicate rows into their primary in place; repeated timed
    # iterations keep the exact same memory work regardless of the
    # accumulated values.
    # —— combine 侧路径。combine 输入先一次性 stage 进 NVL shard（不计时的
    # 准备步骤；zero_copy=True 的调用者本就原位写入，zero_copy=False 的边界
    # 拷贝是普通 torch op）。前核随后把重复行原位归约进主行；重复的计时迭代
    # 保证内存工作量恒定（与累加出的数值无关）。
    hidden_nvsh = torch.randn(NvS, H, dtype=torch.bfloat16, device=dev)
    droute_weights_sk = torch.empty(S, K, dtype=torch.float32, device=dev)
    ctx['hidden_buf_local'].copy_(hidden_nvsh)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # ---- combine_prologue_fwd: in-place fp32 duplicate accumulation.
    # —— combine_prologue_fwd：fp32 原位重复行累加。
    def _combine_prologue_fwd_call():
        launch_combine_prologue(ctx, plan)

    combine_prologue_fwd_us = time_gpu_op(
        _combine_prologue_fwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- combine_fwd: forward combine gathers/sums K rows back to [S, H].
    # —— combine_fwd：前向 combine 把 K 行 gather/求和回 [S, H]。
    def _combine_fwd_call():
        launch_combine(ctx, output, dst)

    combine_fwd_us = time_gpu_op(
        _combine_fwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- combine_prologue_bwd: same kernel on the backward staging state.
    # —— combine_prologue_bwd：同一前核内核，作用于反向 staging 状态。
    def _combine_prologue_bwd_call():
        launch_combine_prologue(ctx, plan)

    combine_prologue_bwd_us = time_gpu_op(
        _combine_prologue_bwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- combine_bwd: combine kernel with droute_weights_sk gather.
    # —— combine_bwd：带 droute_weights_sk gather 的 combine（反向路由权重回收）。
    def _combine_bwd_call():
        launch_combine(ctx, output, dst, output_sk=droute_weights_sk)

    combine_bwd_us = time_gpu_op(
        _combine_bwd_call, warmup, iters, group, cudagraph=cudagraph
    )

    # ---- prefetch / grad_reduce: remote expert weight movement driven by the
    # planner's experts_to_copy ([R, epn] global expert ids, -1 = idle slot).
    # —— prefetch / grad_reduce：由规划器 experts_to_copy（[R, epn] 全局专家
    # id，-1=空闲槽位）驱动的远程专家权重/梯度搬运。
    epn = E // R
    etc_cpu = experts_to_copy.cpu()
    valid = etc_cpu >= 0
    # max_recv: worst rank's prefetched (received) expert count.
    # max_recv：最重接收 rank 的前取专家数。
    max_recv = int(valid.sum(dim=1).max().item())
    # max_send: worst owner rank's slot count across all ranks — these slots'
    # grads flow back to (are pulled by) the owner in grad_reduce.
    # max_send：最重属主 rank 的跨 rank 槽位数——这些槽位的梯度在
    # grad_reduce 中回流（被拉取）到属主。
    if bool(valid.any()):
        owners = etc_cpu[valid].long() // epn
        max_send = int(torch.bincount(owners, minlength=R).max().item())
    else:
        max_send = 0

    # 校验前取池（bf16）的固定 epn VMM 对齐，然后创建 3 份本地权重与 NVL 池
    assert pad_dim0_for_alignment([epn, H, Hp], torch.bfloat16) == epn
    weights_local = tuple(
        torch.randn(epn, H, Hp, dtype=torch.bfloat16, device=dev)
        for _ in range(3)
    )
    prefetch_full = tuple(
        create_nvl_dist_tensor(
            [epn, H, Hp], torch.bfloat16, group_rank, R, group=group,
        )
        for _ in range(3)
    )
    prefetch_buffers = tuple(
        tensor.view(R, epn, H, Hp) for tensor in prefetch_full
    )

    # 校验梯度池（fp32）对齐，创建梯度张量与 NVL reduce 池并预填槽位
    assert pad_dim0_for_alignment([epn, H, Hp], torch.float32) == epn
    grads_local = torch.zeros(epn, H, Hp, dtype=torch.float32, device=dev)
    reduce_full = create_nvl_dist_tensor(
        [epn, H, Hp], torch.float32, group_rank, R, group=group)
    reduce_buffers = reduce_full.view(R, epn, H, Hp)
    local_reduce_slots = torch.randn_like(reduce_buffers[group_rank])
    reduce_buffers[group_rank].copy_(local_reduce_slots)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    def _prefetch_call():
        # 单次前取：属主按 experts_to_copy 推送 3 份专家权重并集体完成
        launch_prefetch(
            weights_local,
            prefetch_buffers,
            experts_to_copy,
            rank=group_rank,
            num_sms=num_sms,
            meta_buf=ctx['meta_buf'],
            meta_stride=int(ctx['meta_chunk_padded']),
            barrier_off=int(ctx['BARRIER_OFF']),
            grid_sync_bar=ctx['grid_sync_bar'],
        )

    prefetch_us = time_gpu_op(_prefetch_call, warmup, iters, group, cudagraph=cudagraph)

    def _grad_reduce_call():
        # 单次梯度归约：远端槽位读到属主并原位累加（内置跨 rank 屏障）
        launch_grad_reduce(
            grads_local, reduce_buffers, experts_to_copy,
            rank=group_rank, num_sms=num_sms,
            meta_buf=ctx['meta_buf'],
            meta_stride=int(ctx['meta_chunk_padded']),
            barrier_off=int(ctx['BARRIER_OFF']),
            grid_sync_bar=ctx['grid_sync_bar'],
        )

    grad_reduce_us = time_gpu_op(_grad_reduce_call, warmup, iters, group, cudagraph=cudagraph)

    # Byte accounting follows the timed operator work above.
    # 字节统计与上面计时的算子工作一致：四个通信算子均为 S*K*H*2（bf16）。
    dispatch_fwd_bytes = S * K * H * 2
    dispatch_bwd_bytes = S * K * H * 2
    combine_fwd_bytes = S * K * H * 2
    combine_bwd_bytes = S * K * H * 2

    # Local-op bandwidth accounting (route weights ignored: [NvS] fp32 is
    # negligible next to [NvS, H] bf16).
    # 本地算子带宽统计（忽略路由权重：[NvS] 的 fp32 相比 [NvS, H] 的 bf16
    # 可忽略不计）。
    # dst 总行数
    n_entries = dst.numel()
    group_size = dist.get_world_size(group=group)
    # 去重结构计数：组数与跨 rank 行数（在 CPU 上读取）
    dup_counts_cpu = plan.dup_counts.cpu()
    dup_group_count = int(dup_counts_cpu[0].item())
    dup_loff_count = int(dup_counts_cpu[1].item())
    # 零填充行总数
    zero_pad_rows = int(plan.zero_fill_ranges[:, 1].sum().item())
    # 统计向量：去重比例 / 组数 / 跨 rank 行数，跨子组 all_gather 取均值
    stats_t = torch.tensor(
        [
            float((dst < 0).sum().item()) / n_entries,
            float(dup_group_count),
            float(dup_loff_count),
        ],
        dtype=torch.float64,
        device=dev,
    )
    stats_outs = [
        torch.empty(3, dtype=torch.float64, device=dev) for _ in range(group_size)
    ]
    dist.all_gather(stats_outs, stats_t, group=group)
    stats_mean = torch.stack(stats_outs).mean(dim=0)
    dedup_ratio = stats_mean[0].item()
    mean_groups = stats_mean[1].item()
    mean_dups = stats_mean[2].item()

    # epilogue: read each group's primary once, write every duplicate slot.
    # epilogue 字节：每组主行读一次 + 每个重复槽位写一次。
    epilogue_total_bytes = (mean_groups + mean_dups) * H * 2
    # prologue: read primary + duplicates, write primary back.
    # prologue 字节：主行读 + 重复行读 + 主行写回。
    combine_prologue_total_bytes = (2 * mean_groups + mean_dups) * H * 2

    # 折算各本地算子的总流量带宽（GB/s）
    epilogue_fwd_total_gbps = epilogue_total_bytes / epilogue_fwd_us * 1e6 / 1e9
    epilogue_bwd_total_gbps = epilogue_total_bytes / epilogue_bwd_us * 1e6 / 1e9
    combine_prologue_fwd_total_gbps = (
        combine_prologue_total_bytes / combine_prologue_fwd_us * 1e6 / 1e9
    )
    combine_prologue_bwd_total_gbps = (
        combine_prologue_total_bytes / combine_prologue_bwd_us * 1e6 / 1e9
    )

    # Both owner-push prefetch and owner-local grad reduction are limited by
    # the busiest owner rank. max_recv remains reported as a workload shape.
    # 属主推送式前取与属主本地梯度归约都受最繁忙属主 rank 限制；
    # max_recv 仅作为负载形状报告。
    prefetch_bytes = 3 * max_send * H * Hp * 2
    grad_reduce_bytes = max_send * H * Hp * 4

    # 汇总本配置的全部结果
    result = {
        'planning_us': planning_us,
        'dispatch_fwd_us': dispatch_fwd_us,
        'dispatch_bwd_us': dispatch_bwd_us,
        'epilogue_fwd_us': epilogue_fwd_us,
        'epilogue_bwd_us': epilogue_bwd_us,
        'combine_prologue_fwd_us': combine_prologue_fwd_us,
        'combine_prologue_bwd_us': combine_prologue_bwd_us,
        'combine_fwd_us': combine_fwd_us,
        'combine_bwd_us': combine_bwd_us,
        'epn': epn,
        'NvS': NvS,
        'dedup_ratio': dedup_ratio,
        'dup_group_count': dup_group_count,
        'dup_loff_count': dup_loff_count,
        'zero_pad_rows': zero_pad_rows,
        'prefetch_us': prefetch_us,
        'grad_reduce_us': grad_reduce_us,
        'max_recv': max_recv,
        'max_send': max_send,
        'load_max_mean': load_max_mean,
        'dispatch_fwd_GBps': dispatch_fwd_bytes / dispatch_fwd_us * 1e6 / 1e9,
        'dispatch_bwd_GBps': dispatch_bwd_bytes / dispatch_bwd_us * 1e6 / 1e9,
        'combine_fwd_GBps': combine_fwd_bytes / combine_fwd_us * 1e6 / 1e9,
        'combine_bwd_GBps': combine_bwd_bytes / combine_bwd_us * 1e6 / 1e9,
        'epilogue_fwd_total_GBps': epilogue_fwd_total_gbps,
        'epilogue_bwd_total_GBps': epilogue_bwd_total_gbps,
        'combine_prologue_fwd_total_GBps': combine_prologue_fwd_total_gbps,
        'combine_prologue_bwd_total_GBps': combine_prologue_bwd_total_gbps,
        'prefetch_GBps': prefetch_bytes / prefetch_us * 1e6 / 1e9,
        'grad_reduce_GBps': grad_reduce_bytes / grad_reduce_us * 1e6 / 1e9,
    }
    # 销毁 Buffer 释放资源后返回
    buffer.destroy()
    return result


def cleanup_ctx():
    # 配置间的清理：强制 Python GC + 清空 CUDA 缓存，避免显存碎片累积
    gc.collect()
    torch.cuda.empty_cache()


# -------------------------------------------------------------------------
# Driver
# 驱动流程
# -------------------------------------------------------------------------

# 扫描的 EP 组大小列表
EP_SIZES = [4, 8, 16, 32, 64]


def build_configs():
    # 生成全部扫描配置（ep x H x K x S x bias_ratio 的笛卡尔积）
    HIDDENS = [3584, 7168]
    TOPKS = [8, 16]
    SEQLENS = [8192]
    # bias_ratio sweep — sigma of the lognormal expert-logit distribution.
    # Three regimes: near-balanced, typical dropless-MoE skew, pathological.
    # bias_ratio 扫描——lognormal 专家打分的 sigma。三档：近似均衡、
    # 典型 dropless-MoE 偏斜、病态。
    RATIOS = [0.1, 1.0, 5.0]
    # Total expert count is fixed; experts-per-rank (epn = E / ep)
    # shrinks as the EP group grows (e.g. 224 at ep=4, 112 at ep=8).
    # 专家总数固定；每 rank 专家数（epn=E/ep）随 EP 组增大而减少
    # （ep=4 时 224，ep=8 时 112）。
    TOTAL_EXPERTS = 896

    configs = []
    for ep in EP_SIZES:
        # 每个 ep 下的全局专家数
        E = TOTAL_EXPERTS
        assert E % ep == 0, f"E ({E}) must be divisible by ep ({ep})"
        for H in HIDDENS:
            for K in TOPKS:
                for S in SEQLENS:
                    for ratio in RATIOS:
                        # topk 超过专家数的组合无意义，跳过
                        if K > E:
                            continue
                        configs.append(dict(
                            ep=ep, H=H, K=K, S=S, E=E,
                            unbalance_ratio=ratio,
                        ))
    return configs


def main():
    # 入口：解析参数 -> 初始化分布式 -> 预建子组 -> 逐配置运行并输出
    ap = argparse.ArgumentParser()
    # CSV 输出路径（仅 rank 0 写）
    ap.add_argument("--out", default=None,
                    help="Output CSV path (rank 0 only). Defaults to stdout-only.")
    # 预热与计时迭代次数
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    # 每内核 SM 数
    ap.add_argument("--num-sms", type=int, default=32)
    # Expert weight inner dim H' for compact [epn, H, Hp] proxy tensors.
    # 专家权重内维 H'（紧凑 [epn, H, Hp] 代理张量用）。
    ap.add_argument("--hp", type=int, default=3072,
                    help="Expert weight inner dim H' (multiple of 128).")
    # CUDA-graph timing is on by default, removing per-launch overhead;
    # --no-cudagraph falls back to the eager loop.
    # CUDA graph 计时默认开启（剔除每次 launch 的开销）；
    # --no-cudagraph 退回 eager 循环。
    ap.add_argument("--no-cudagraph", dest="cudagraph", action="store_false",
                    help="Disable CUDA-graph timing (eager launch loop instead).")
    ap.set_defaults(cudagraph=True)
    # Per-dimension filters. If set, restrict the sweep to matching configs.
    # Useful for profiling (ncu/nsys) a single point.
    # 按维度过滤：设置后只扫描匹配的配置，便于 ncu/nsys 分析单点。
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--unbalance-ratio", type=float, default=None)
    # Override total expert count E on the selected config(s). Useful for
    # profiling unusual (ep, E) combos that the default ep*14 sweep doesn't
    # produce; the planning CuTe DSL path must have a matching specialization.
    # 覆盖所选配置的专家总数 E：用于分析默认扫描不覆盖的 (ep, E) 组合；
    # 规划的 CuTe DSL 路径必须有对应的特化编译。
    ap.add_argument("--experts", type=int, default=None)
    args = ap.parse_args()
    # H' 必须是 128 的倍数
    assert args.hp % 128 == 0, f"--hp must be a multiple of 128, got {args.hp}"

    # 初始化分布式
    world_rank, world_size, local_rank = setup()

    if world_rank == 0:
        # 打印基准开始信息与节点名
        print(f"=== bench_comm starting: world_size={world_size} ===", flush=True)
        print(f"hostname={os.uname().nodename} local_rank={local_rank}", flush=True)

    # Pre-create one subgroup per ep_size we'll visit. Subgroups must be
    # created collectively by all ranks (those not in `ranks=` get NULL).
    # 为每个要访问的 ep_size 预创建子组。子组必须由所有 rank 集体创建
    # （不在 ranks= 列表内的 rank 得到 NULL 组）。
    ep_groups = {}
    for ep in EP_SIZES:
        if ep > world_size:
            continue
        g = dist.new_group(ranks=list(range(ep)))
        ep_groups[ep] = g

    # 生成并过滤配置列表
    configs = build_configs()
    configs = [c for c in configs if (
        (args.ep is None or c['ep'] == args.ep)
        and (args.hidden is None or c['H'] == args.hidden)
        and (args.topk is None or c['K'] == args.topk)
        and (args.unbalance_ratio is None or c['unbalance_ratio'] == args.unbalance_ratio)
    )]
    if args.experts is not None:
        # 覆盖专家总数并剔除 K>E 的组合
        for c in configs:
            c['E'] = args.experts
        configs = [c for c in configs if c['K'] <= c['E']]
    if world_rank == 0:
        print(f"Sweeping {len(configs)} configs...", flush=True)

    # 结果收集与 CSV 写手
    results = []
    csv_writer = None
    csv_file = None
    # CSV 表头列
    csv_header = [
        "ep", "H", "K", "S", "E", "epn", "Hp", "unbalance_ratio",
        "max_recv", "max_send", "load_max_mean",
        "planning_us",
        "dispatch_fwd_us", "dispatch_bwd_us",
        "epilogue_fwd_us", "epilogue_bwd_us",
        "combine_prologue_fwd_us", "combine_prologue_bwd_us",
        "combine_fwd_us", "combine_bwd_us",
        "NvS", "dedup_ratio",
        "dup_group_count", "dup_loff_count", "zero_pad_rows",
        "prefetch_us", "grad_reduce_us",
        "dispatch_fwd_GBps", "dispatch_bwd_GBps",
        "combine_fwd_GBps", "combine_bwd_GBps",
        "epilogue_fwd_total_GBps", "epilogue_bwd_total_GBps",
        "combine_prologue_fwd_total_GBps", "combine_prologue_bwd_total_GBps",
        "prefetch_GBps", "grad_reduce_GBps",
    ]
    if world_rank == 0 and args.out:
        # 打开 CSV 并写表头（目录不存在则创建）
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        csv_file = open(args.out, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)

    if world_rank == 0:
        # 终端表头（各列宽与含义）
        header = (f"{'ep':>3} {'H':>5} {'K':>3} {'S':>5} {'E':>4} {'Hp':>5} "
                  f"{'unb_r':>5} {'mx_rcv':>6} {'mx_snd':>6} {'mx/mean':>7} "
                  f"{'plan_us':>9} "
                  f"{'d_fwd':>9} {'d_bwd':>9} "
                  f"{'e_fwd':>9} {'e_bwd':>9} "
                  f"{'cp_fwd':>9} {'cp_bwd':>9} "
                  f"{'c_fwd':>9} {'c_bwd':>9} "
                  f"{'NvS':>7} {'dedup%':>7} "
                  f"{'grps':>7} {'dups':>7} {'zpad':>7} "
                  f"{'pf':>9} {'gr':>9} "
                  f"{'DfGB/s':>8} {'DbGB/s':>8} "
                  f"{'CfGB/s':>8} {'CbGB/s':>8} "
                  f"{'EfTotGB':>8} {'EbTotGB':>8} "
                  f"{'CpFTot':>8} {'CpBTot':>8} "
                  f"{'PfGB/s':>8} {'GrGB/s':>8}")
        print(header, flush=True)
        print("-" * len(header), flush=True)

    for cfg in configs:
        # 逐配置运行：只有 world_rank < ep 的 rank 参与该子组
        ep = cfg['ep']
        if ep > world_size:
            continue
        group = ep_groups.get(ep)
        in_group = (world_rank < ep)

        res = None
        if in_group:
            try:
                # 子组 rank 0..ep-1 恰为全局 rank 0..ep-1
                group_rank = world_rank  # subgroup ranks 0..ep-1 == world ranks 0..ep-1
                res = bench_one(
                    group=group, group_rank=group_rank,
                    R=ep, S=cfg['S'], K=cfg['K'], E=cfg['E'], H=cfg['H'],
                    bias_ratio=cfg['unbalance_ratio'], Hp=args.hp,
                    num_sms=args.num_sms,
                    warmup=args.warmup, iters=args.iters,
                    cudagraph=args.cudagraph,
                )
            except Exception:
                # 单配置失败只打印堆栈，不中断整体扫描
                if world_rank == 0:
                    traceback.print_exc()
            finally:
                # 无论成败都清理显存
                cleanup_ctx()

        # Hard global sync so idle ranks don't race ahead and so failures
        # in one group don't leave the next config wedged.
        # 全局硬同步：防止空闲 rank 抢跑，也防止某组的失败卡住下一配置。
        dist.barrier()

        if world_rank == 0:
            if res is not None:
                # 打印本配置结果行
                line = (f"{ep:>3} {cfg['H']:>5} {cfg['K']:>3} {cfg['S']:>5} {cfg['E']:>4} "
                        f"{args.hp:>5} "
                        f"{cfg['unbalance_ratio']:>5.2f} "
                        f"{res['max_recv']:>6d} "
                        f"{res['max_send']:>6d} "
                        f"{res['load_max_mean']:>7.3f} "
                        f"{fmt4(res['planning_us']):>9} "
                        f"{fmt4(res['dispatch_fwd_us']):>9} "
                        f"{fmt4(res['dispatch_bwd_us']):>9} "
                        f"{fmt4(res['epilogue_fwd_us']):>9} "
                        f"{fmt4(res['epilogue_bwd_us']):>9} "
                        f"{fmt4(res['combine_prologue_fwd_us']):>9} "
                        f"{fmt4(res['combine_prologue_bwd_us']):>9} "
                        f"{fmt4(res['combine_fwd_us']):>9} "
                        f"{fmt4(res['combine_bwd_us']):>9} "
                        f"{res['NvS']:>7d} "
                        f"{res['dedup_ratio'] * 100:>6.2f}% "
                        f"{res['dup_group_count']:>7d} "
                        f"{res['dup_loff_count']:>7d} "
                        f"{res['zero_pad_rows']:>7d} "
                        f"{fmt4(res['prefetch_us']):>9} "
                        f"{fmt4(res['grad_reduce_us']):>9} "
                        f"{fmt4(res['dispatch_fwd_GBps']):>8} "
                        f"{fmt4(res['dispatch_bwd_GBps']):>8} "
                        f"{fmt4(res['combine_fwd_GBps']):>8} "
                        f"{fmt4(res['combine_bwd_GBps']):>8} "
                        f"{fmt4(res['epilogue_fwd_total_GBps']):>8} "
                        f"{fmt4(res['epilogue_bwd_total_GBps']):>8} "
                        f"{fmt4(res['combine_prologue_fwd_total_GBps']):>8} "
                        f"{fmt4(res['combine_prologue_bwd_total_GBps']):>8} "
                        f"{fmt4(res['prefetch_GBps']):>8} "
                        f"{fmt4(res['grad_reduce_GBps']):>8}")
                print(line, flush=True)
                if csv_writer:
                    # 同步写一行 CSV（与时延列一一对应）
                    csv_writer.writerow([
                        ep, cfg['H'], cfg['K'], cfg['S'], cfg['E'],
                        res['epn'], args.hp,
                        f"{cfg['unbalance_ratio']:.2f}",
                        res['max_recv'],
                        res['max_send'],
                        f"{res['load_max_mean']:.3f}",
                        fmt4(res['planning_us']),
                        fmt4(res['dispatch_fwd_us']),
                        fmt4(res['dispatch_bwd_us']),
                        fmt4(res['epilogue_fwd_us']),
                        fmt4(res['epilogue_bwd_us']),
                        fmt4(res['combine_prologue_fwd_us']),
                        fmt4(res['combine_prologue_bwd_us']),
                        fmt4(res['combine_fwd_us']),
                        fmt4(res['combine_bwd_us']),
                        res['NvS'],
                        f"{res['dedup_ratio']:.4f}",
                        res['dup_group_count'],
                        res['dup_loff_count'],
                        res['zero_pad_rows'],
                        fmt4(res['prefetch_us']),
                        fmt4(res['grad_reduce_us']),
                        fmt4(res['dispatch_fwd_GBps']),
                        fmt4(res['dispatch_bwd_GBps']),
                        fmt4(res['combine_fwd_GBps']),
                        fmt4(res['combine_bwd_GBps']),
                        fmt4(res['epilogue_fwd_total_GBps']),
                        fmt4(res['epilogue_bwd_total_GBps']),
                        fmt4(res['combine_prologue_fwd_total_GBps']),
                        fmt4(res['combine_prologue_bwd_total_GBps']),
                        fmt4(res['prefetch_GBps']),
                        fmt4(res['grad_reduce_GBps']),
                    ])
                    csv_file.flush()
            else:
                # 失败配置：打印 FAILED 行并写空 CSV 行
                line = (f"{ep:>3} {cfg['H']:>5} {cfg['K']:>3} {cfg['S']:>5} {cfg['E']:>4} "
                        f"{args.hp:>5} "
                        f"{cfg['unbalance_ratio']:>5.2f} FAILED")
                print(line, flush=True)
                if csv_writer:
                    row = [
                        ep, cfg['H'], cfg['K'], cfg['S'], cfg['E'], args.hp,
                        f"{cfg['unbalance_ratio']:.2f}",
                        "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                    ]
                    row.extend([""] * (len(csv_header) - len(row)))
                    csv_writer.writerow(row)
                    csv_file.flush()
            # 收集结果
            results.append((cfg, res))

    if world_rank == 0:
        # 收尾：打印完成统计并关闭 CSV
        print(f"\n=== bench_comm done: {len(results)} configs ===", flush=True)
        if csv_file:
            csv_file.close()
            print(f"Wrote CSV to {args.out}", flush=True)

    # 结束：销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
