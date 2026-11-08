"""Aligned dispatch/combine benchmark: MoonEP vs DeepEP v2 (elastic).

Fused-permute edition. MoonEP's dispatch epilogue fuses the expert permute
into the comm path (and combine_prologue fuses the unpermute reduction), so
the only apples-to-apples DeepEP comparison is the v2 elastic path with
``do_expand=True`` (expanded dispatch) and the reduced combine. (DeepEP v1's
recv_x is grouped by source rank — the expert permute/unpermute is a separate
unfused step outside the library — so it is not compared here.)

Everything shared across both libraries:
  - identical routing matrix (tests/generate_topk_routing.py, seed 1234 shared
    across ranks for expert popularity, rank seed for per-token draws)
  - identical input hidden/weights tensors (fixed generator seeds)
  - identical timing harness (eager CUDA-event timing over back-to-back
    iterations, cross-rank mean), warmup=20, iters=50
  - identical SM budget (32) and expert_alignment/token_padding (128)

Reported ops (MoonEP runs the public Buffer API end-to-end; local weights are
[epn, H, H'] and the communication pools are [R, epn, H, H']):
  - plan: MoonEP launch_planning (exact, separate)  |  v2 layout is fused
          into dispatch_impl and cannot be timed in isolation; estimated as
          full-dispatch minus cached-dispatch (comes out ~= 0)
  - d_f:  buffer.dispatch(zero_copy=True) = inter_rank_sync + planning +
          dispatch + epilogue (fused permute), then buffer.prefetch_weight
          |  v2 expanded dispatch (with topk_weights)
  - d_b:  buffer.dispatch(saved plan, zero_copy=True) + prefetch_weight
          |  v2 cached expanded dispatch (hidden-only)
  - c_f:  buffer.combine(hidden only)  |  v2 reduced combine (no topk_weights)
  - c_b:  buffer.combine(with droute_weights gather)  |  v2 reduced combine
          (with topk_weights)
  - pf:   buffer.prefetch_weight separately (for its share of d_f/d_b)
  grad_reduce is NOT included: it is overlappable with subsequent compute
  and not on the MoE critical path.

Route weights are transferred in both directions on both libraries (dispatch
scatters them, combine gathers them back).

Bandwidth = S*K*H*2 / t (logical payload bytes, same formula for both).

Single node only (ep == world size). Launch:
  torchrun --nproc_per_node=8 benchmarks/bench_vs_deepep.py [--out r.csv] [--plot r.png]
"""

# 中文说明：对齐口径的 dispatch/combine 基准——MoonEP 对比 DeepEP v2（elastic）。
# 融合 permute 版：MoonEP 在 dispatch 尾核中融合了专家 permute（combine 前核
# 融合了反 permute 归约），因此唯一对等的对比对象是 DeepEP v2 elastic 路径的
# ``do_expand=True``（展开 dispatch）+ reduced combine 组合；DeepEP v1 的
# recv_x 按源 rank 分组、permute/unpermute 在库外单独执行，不具可比性，故不测。
# 两库共享的实验条件：完全相同的路由矩阵（tests/generate_topk_routing.py，
#   种子 1234 跨 rank 共享控制专家热度、rank 种子控制逐 token 抽样）、相同的
#   hidden/weights 输入（固定生成器种子）、相同的计时框架（背靠背迭代的 eager
#   CUDA 事件计时、跨 rank 取均值；warmup=20、iters=50）、相同的 SM 预算（32）
#   与 expert_alignment/token_padding（128）。
# 报告的算子（MoonEP 走公开 Buffer API；本地权重 [epn, H, H']，通信池
#   [R, epn, H, H']；"|" 左为 MoonEP、右为 v2）：
#   plan：launch_planning（精确、独立计时）| v2 的 layout 融合在 dispatch_impl
#         内无法单独计时，用"完整 dispatch 减缓存 dispatch"估计（结果≈0）
#   d_f ：dispatch(zero_copy=True) = 跨rank同步+规划+dispatch+尾核（融合
#         permute），再 prefetch_weight | v2 展开式 dispatch（带 topk_weights）
#   d_b ：dispatch(复用已存 plan, zero_copy=True) + prefetch_weight |
#         v2 缓存式展开 dispatch（仅 hidden）
#   c_f ：combine（仅 hidden）| v2 reduced combine（无 topk_weights）
#   c_b ：combine（带 droute_weights gather）| v2 reduced combine（带 topk_weights）
#   pf  ：单独计时 prefetch_weight（对应 d_f/d_b 中它的份额）
#   grad_reduce 不计入：它可与后续计算重叠，不在 MoE 关键路径上。
# 两库都在双向传输路由权重（dispatch scatter、combine gather）。
# 带宽 = S*K*H*2 / t（逻辑负载字节，两库同一公式）。仅单机（ep == world size）。

# argparse：解析命令行参数
import argparse
# csv：写出结果表格
import csv
# os：读取 LOCAL_RANK 等环境变量
import os
# time：统计建 Buffer/求解 sigma 等阶段的墙钟耗时
import time

# torch：张量、CUDA 事件；dist：进程组与集合通信
import torch
import torch.distributed as dist

# 共享的路由生成器：保证两库拿到完全相同的路由矩阵
from tests.generate_topk_routing import generate_topk_routing


# ---------------------------------------------------------------------------
# Shared timing harness — the ONLY timing method used for all three libraries
# 共享计时框架——所有库唯一使用的计时方法
# ---------------------------------------------------------------------------

def time_op(fn, group, warmup=20, iters=50):
    """Eager CUDA-event timing: back-to-back launches, cross-rank mean us."""
    # 中文：eager CUDA 事件计时——背靠背连续 launch，跨 rank 取平均微秒。
    # 预热（兼做 JIT 编译）
    for _ in range(warmup):
        fn()
    # 同步并跨 rank 屏障，保证计时起点对齐
    torch.cuda.synchronize()
    dist.barrier(group=group)
    # 两个 CUDA Event 夹住 iters 次连续调用
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    # 本 rank 平均单次时延（ms -> us）
    local_us = start.elapsed_time(end) / iters * 1e3
    # 把各 rank 的均值 all_gather 后取平均，作为该算子的报告时延
    t = torch.tensor([local_us], dtype=torch.float64, device='cuda')
    outs = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(outs, t, group=group)
    return torch.cat(outs).mean().item()


def gbps(S, K, H, us):
    # 带宽换算：S*K*H*2 字节（bf16）除以时延，单位 GB/s（两库统一口径）
    return S * K * H * 2 / (us * 1e-6) / 1e9


def _maxvio_for_sigma(sigma, S, K, E, R, dev):
    """Realized MaxVio of the lognormal routing at the given sigma, using the
    benchmark's fixed seeds (1234 shared, per-rank local) — deterministic."""
    # 中文：给定 sigma 时 lognormal 路由的实际 MaxVio。用基准的固定种子
    # （1234 跨 rank 共享 + 各 rank 本地种子），结果完全确定。
    # 累加各 rank 的 per-expert 路由 token 数
    total = torch.zeros(E, dtype=torch.int64, device=dev)
    for r in range(R):
        # 用 rank r 的本地种子生成路由并累加专家负载
        _, tpe = generate_topk_routing(S, K, E, R, sigma, dev, 1234, rank=r)
        total += tpe.to(torch.int64)
    # MaxVio = 最热专家负载 / 平均负载 - 1
    return total.max().item() / (total.sum().item() / E) - 1.0


def solve_sigma_for_maxvio(target, S, K, E, R, dev):
    """Log-scale bisection for the lognormal sigma whose realized MaxVio best
    matches `target`. Returns (sigma, realized_maxvio). If the target is below
    the uniform-sampling noise floor, returns the smallest sigma tried."""
    # 中文：在对数尺度上二分求解 lognormal 的 sigma，使实际 MaxVio 最接近
    # 目标值；返回 (sigma, 实际 MaxVio)。若目标低于均匀抽样的噪声下限，
    # 返回尝试过的最小 sigma。
    # 搜索区间 [1e-5, 10]
    lo, hi = 1e-5, 10.0
    # 检查下界处的 MaxVio：若已不低于目标，直接返回下界
    mv_lo = _maxvio_for_sigma(lo, S, K, E, R, dev)
    if mv_lo >= target:
        return lo, mv_lo
    # 记录当前最接近目标的 (sigma, MaxVio)
    best_sigma, best_mv = lo, mv_lo
    for _ in range(60):
        # 对数尺度取中点
        mid = (lo * hi) ** 0.5
        # 评估中点的实际 MaxVio
        mv = _maxvio_for_sigma(mid, S, K, E, R, dev)
        # 更新历史最优
        if abs(mv - target) < abs(best_mv - target):
            best_sigma, best_mv = mid, mv
        # 达到精度要求（相对 2% 或绝对 0.005）则提前返回
        if abs(mv - target) <= max(0.02 * target, 0.005):
            return mid, mv
        # 标准二分收缩区间
        if mv < target:
            lo = mid
        else:
            hi = mid
    # 迭代用尽：返回历史最优点
    return best_sigma, best_mv


# ---------------------------------------------------------------------------
# MoonEP — public Buffer API for dispatch/combine; kernel-level launches for
# the separate planning/prefetch measurements (prefetch uses the same
# launch_prefetch the public prefetch_weight wraps, over a sharded NVL pool)
# MoonEP——dispatch/combine 走公开 Buffer API；规划/前取的独立计时走内核级
# 启动（prefetch 用的就是公开 prefetch_weight 内部封装的 launch_prefetch，
# 作用于分片的 NVL 池）
# ---------------------------------------------------------------------------

class MoonEPRunner:
    # MoonEP 的被测运行器（封装 Buffer 与各算子的调用方式）
    NAME = 'moonep'

    def __init__(self, group, R, S, K, E, H, num_sms, hp=2048):
        # 延迟导入 Buffer 与规划/缓冲工具
        from moonep import Buffer
        from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment
        from moonep.planning import allocate_planning_outputs, launch_planning
        # 保存规划内核启动函数供后续单独计时
        self._launch_planning = launch_planning
        self.num_sms = num_sms
        # 本 rank 在该进程组中的编号
        self.rank = dist.get_rank(group)
        self.R = R
        # 创建 MoonEP Buffer（含跨 rank 屏障、meta_buf 等共享上下文）
        self.buffer = Buffer(S, H, K, E, R, num_sms=num_sms, group=group,
                             explicitly_destroy=True)
        self.ctx = self.buffer._require_ctx()
        # 分配规划输出（计划表与 cu_seqlens），供独立的规划计时使用
        self.plan_scratch, self.cu_seqlens = allocate_planning_outputs(self.ctx)

        # 每 rank 专家数与前取权重的内维 H'
        self.epn = E // R
        self.hp = hp
        assert pad_dim0_for_alignment(
            [self.epn, H, hp], torch.bfloat16
        ) == self.epn, "prefetch pool chunk must be VMM aligned"

        # grad_reduce is not on the
        # MoE critical path (overlappable with subsequent compute) and is not
        # included in this benchmark.
        # grad_reduce 不在 MoE 关键路径上（可与后续计算重叠），不计入本基准。
        # 创建 3 份本地专家权重（gate/up/down）与对应的 NVL 前取池
        self.local_weights = []
        self.prefetch_buffers = []
        for _ in range(3):
            self.local_weights.append(
                torch.randn(
                    self.epn, H, hp, dtype=torch.bfloat16, device='cuda'
                )
            )
            # 创建 VMM 分布式池并 view 成 [R, epn, H, hp]
            pool = create_nvl_dist_tensor(
                [self.epn, H, hp], torch.bfloat16,
                self.rank, R, group=group,
            )
            self.prefetch_buffers.append(pool.view(R, self.epn, H, hp))
        # 同步确保所有 rank 的池/缓冲就绪
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def prepare(self, topk, tpe, hidden, weights):
        # 保存本轮的输入与路由，并做一次不计时的完整 dispatch 以物化计划
        self.hidden, self.weights = hidden, weights
        self.topk, self.tpe = topk, tpe
        # 拉平的 topk 索引（规划内核入参）
        self._topk_flat = topk.reshape(-1).contiguous()
        # Untimed full dispatch (public API): materializes the plan + dedup
        # structures used by saved-plan dispatch/combine below.
        # 不计时的完整 dispatch（公开 API）：物化后续 saved-plan
        # dispatch/combine 所需的计划表与去重结构。
        _, _, _, self.plan = self.buffer.dispatch(
            hidden, weights, topk, tpe, zero_copy=True, router_weights_zero_copy=True)
        # Scratch plan for the separate (exact) planning measurement.
        # 写一份临时计划，供"独立精确计时规划"的算子使用。
        self._launch_planning(self.ctx, self._topk_flat, tpe,
                              self.cu_seqlens, self.plan_scratch)
        # Byte accounting from the plan's slot table (global expert ids).
        # 用计划表的槽位表（全局专家 id）做字节量统计。
        etc_cpu = self.plan.experts_to_copy.cpu()   # [R, epn] int32, -1 = idle
        # 有效槽位（非空闲）
        valid = etc_cpu >= 0
        # max_recv：最重接收 rank 的槽位数
        self.max_recv = int(valid.sum(dim=1).max().item())
        # 各槽位的目的 rank 编号矩阵
        dst = torch.arange(self.R).view(self.R, 1).expand_as(etc_cpu)
        # 由全局专家 id 反推属主 rank
        owners = torch.div(
            etc_cpu.clamp_min(0), self.epn, rounding_mode='floor'
        )
        # 跨 rank 槽位（属主 != 目的地）才是远端发送
        remote = valid & (owners != dst)
        # max_send：最重发送属主的跨 rank 槽位数
        self.max_send = int(
            torch.bincount(owners[remote], minlength=self.R).max().item()
        ) if bool(remote.any()) else 0
        # combine 的零拷贝输入视图：NVL shard 的 hidden 与 fp32 权重
        self.shard_view = self.ctx['hidden_buf_local']
        self.weights_view = self.ctx['weights_buf_local'].view(torch.float32)
        # 同步确保准备阶段完成
        torch.cuda.synchronize()
        dist.barrier()

    def planning(self):
        # 独立计时的规划算子：重复对同一份路由做规划，写入临时计划
        self._launch_planning(self.ctx, self._topk_flat, self.tpe,
                              self.cu_seqlens, self.plan_scratch)

    def _prefetch(self):
        # 按已存计划把本地三层专家权重推送到 all-rank 前取池
        self.buffer.prefetch_weight(
            plan=self.plan,
            local_gate_weight=self.local_weights[0],
            local_up_weight=self.local_weights[1],
            local_down_weight=self.local_weights[2],
            gate_prefetch_buffer=self.prefetch_buffers[0],
            up_prefetch_buffer=self.prefetch_buffers[1],
            down_prefetch_buffer=self.prefetch_buffers[2])

    def prefetch(self):
        # 单独计时的前取算子
        self._prefetch()

    def dispatch_fwd(self):
        # full fwd (public API): inter_rank_sync + planning + dispatch +
        # epilogue (zero_copy, no boundary copy), then prefetch weight.
        # 前向 dispatch（公开 API 全流程）：跨 rank 同步 + 规划 + dispatch +
        # 尾核（zero_copy，无边界拷贝），随后前取权重。
        _, _, _, self.plan = self.buffer.dispatch(
            self.hidden, self.weights, self.topk, self.tpe,
            zero_copy=True, router_weights_zero_copy=True)
        self._prefetch()

    def dispatch_bwd(self):
        # bwd: saved plan (no planning, no dedup builder), zero_copy; the
        # backward expert GEMMs need the same weights, so prefetch again.
        # 反向 dispatch：复用已存计划（不再规划、不再构建去重结构），zero_copy；
        # 反向的专家 GEMM 需要同样的权重，因此再前取一次。
        self.buffer.dispatch(self.hidden, plan=self.plan, zero_copy=True)
        self._prefetch()

    def combine_fwd(self):
        # c_f: hidden only
        # 前向 combine：只聚合 hidden
        self.buffer.combine(plan=self.plan, hidden_nvsh=self.shard_view,
                            zero_copy=True)

    def combine_bwd(self):
        # c_b: with droute_weights gather. grad_reduce is not on the critical
        # path and is not included in this benchmark.
        # 反向 combine：额外 gather droute_weights。
        # grad_reduce 不在关键路径上，不计入本基准。
        self.buffer.combine(plan=self.plan, hidden_nvsh=self.shard_view,
                            route_weights_nvs=self.weights_view,
                            zero_copy=True, router_weights_zero_copy=True)

    def destroy(self):
        # 销毁 Buffer，释放 VMM/通信资源
        self.buffer.destroy()


# ---------------------------------------------------------------------------
# DeepEP v2 (elastic) — expanded dispatch / reduced combine (fused permute)
# DeepEP v2（elastic）——展开式 dispatch / reduced combine（融合 permute）
# ---------------------------------------------------------------------------

class DeepEPV2Runner:
    # DeepEP v2 的被测运行器（封装 ElasticBuffer 的调用方式）
    NAME = 'deepep_v2'

    def __init__(self, group, R, S, K, E, H, num_sms):
        # 延迟导入 deep_ep，避免 MoonEP-only 环境报错
        import deep_ep
        self.deep_ep = deep_ep
        self.S, self.E, self.num_sms = S, E, num_sms
        # 创建 elastic 模式的 DeepEP v2 缓冲区（允许混合/多次归约等配置
        # 与 reduced combine 路径对齐）
        self.buffer = deep_ep.ElasticBuffer(
            group, num_max_tokens_per_rank=S, hidden=H,
            deterministic=False, allow_hybrid_mode=True,
            allow_multiple_reduction=True, prefer_overlap_with_compute=False,
            sl_idx=0, num_allocated_qps=0, explicitly_destroy=True,
            num_gpu_timeout_secs=100, num_cpu_timeout_secs=100)
        # 依 SM 数计算理论 QP 数
        self.num_qps = self.buffer.get_theoretical_num_qps(num_sms)

    def _expanded_args(self):
        # 展开式 dispatch 的公共参数：对齐 128、同步执行、do_expand 开启
        return dict(num_sms=self.num_sms, num_qps=self.num_qps,
                    num_max_tokens_per_rank=self.S, num_experts=self.E,
                    expert_alignment=128, async_with_compute_stream=0,
                    allocate_on_comm_stream=0, do_handle_copy=1, do_cpu_sync=0,
                    do_expand=True, use_tma_aligned_col_major_sf=True)

    def prepare(self, topk, tpe, hidden, weights):
        # 保存输入；类型转为 DeepEP 要求的 topk 索引类型
        self.topk_idx = topk.to(self.deep_ep.topk_idx_t)
        self.hidden, self.weights = hidden, weights
        # 不计时的完整 dispatch：拿到输出与 handle（缓存式路径需要）
        recv = self.buffer.dispatch(x=hidden, topk_idx=self.topk_idx,
                                    topk_weights=weights, **self._expanded_args())
        # 保存展开后的接收张量、路由权重与 handle
        self.recv_x, self.recv_topk_weights, self.handle = recv[0], recv[2], recv[3]
        torch.cuda.synchronize()
        dist.barrier()

    def dispatch_fwd(self):
        # 前向 dispatch：完整展开（含 topk_weights scatter）
        self.buffer.dispatch(x=self.hidden, topk_idx=self.topk_idx,
                             topk_weights=self.weights, **self._expanded_args())

    def dispatch_bwd(self):
        # 反向 dispatch：handle 缓存式（不再路由计算），仅 hidden
        self.buffer.dispatch(x=self.hidden, num_sms=self.num_sms,
                             num_qps=self.num_qps, async_with_compute_stream=0,
                             allocate_on_comm_stream=0,
                             do_expand=True, use_tma_aligned_col_major_sf=True,
                             do_zero_padding=True, handle=self.handle)

    def combine_fwd(self):
        # c_f: hidden only
        # 前向 combine：不带 topk_weights 归约
        self.buffer.combine(x=self.recv_x, handle=self.handle,
                            num_sms=self.num_sms, num_qps=self.num_qps,
                            async_with_compute_stream=0,
                            allocate_on_comm_stream=0)

    def combine_bwd(self):
        # c_b: with topk_weights reduction
        # 反向 combine：带 topk_weights 归约
        self.buffer.combine(x=self.recv_x, topk_weights=self.recv_topk_weights,
                            handle=self.handle, num_sms=self.num_sms,
                            num_qps=self.num_qps, async_with_compute_stream=0,
                            allocate_on_comm_stream=0)

    def destroy(self):
        # 销毁 DeepEP 缓冲区
        self.buffer.destroy()


# ---------------------------------------------------------------------------
# Driver
# 驱动流程
# ---------------------------------------------------------------------------

# 被测库注册表：命令行 --libs 里的名字 -> 运行器类
RUNNERS = {
    'moonep': MoonEPRunner,
    'v2': DeepEPV2Runner,
}


def main():
    # 入口：解析参数 -> 初始化分布式 -> 建 Buffer -> 逐 MaxVio 点计时 -> 输出
    ap = argparse.ArgumentParser()
    # 要对比的库（可多选）
    ap.add_argument('--libs', nargs='+', default=['moonep', 'v2'],
                    choices=list(RUNNERS))
    # 结果 CSV 输出路径（rank 0）
    ap.add_argument('--out', default=None)
    # 若设置，把堆叠柱状对比图保存到该路径（仅 rank 0）
    ap.add_argument('--plot', default=None,
                    help='If set, save the stacked-bar comparison figure to '
                         'this path (rank 0 only).')
    # 目标 MaxVio 列表：每个值反向求解一个 lognormal sigma
    ap.add_argument('--maxvios', default='0.2,1,10,20',
                    help='Comma-separated target MaxVio values; the lognormal '
                         'sigma is reverse-solved for each (default: %(default)s)')
    # 工作负载形状：专家总数、隐藏维、topk
    ap.add_argument('--experts', type=int, default=384)
    ap.add_argument('--hidden', type=int, default=7168)
    ap.add_argument('--topk', type=int, default=8)
    # 专家权重内维 H'（仅影响 MoonEP 的 prefetch 算子）
    ap.add_argument('--hp', type=int, default=2048,
                    help='Expert weight inner dim H\' (MoE intermediate size) '
                         'for the MoonEP prefetch op (default: %(default)s)')
    # SM 预算与计时迭代参数（两库一致）
    ap.add_argument('--num-sms', type=int, default=32)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=50)
    args = ap.parse_args()

    # 绑定本进程 GPU 并初始化 NCCL
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    R = dist.get_world_size()
    # 本基准只支持单机 ep=8
    assert R == 8, f'ep=8 only: world_size must be 8, got {R}'
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{local_rank}')

    # 序列长度固定 8192/rank；专家数与隐藏维来自命令行
    S, E = 8192, args.experts
    H, K = args.hidden, args.topk
    # 目标 MaxVio 列表
    targets = [float(x) for x in args.maxvios.split(',')]

    def make_runner(lib):
        # 按库名构造对应运行器（MoonEP 额外传 hp）
        if lib == 'moonep':
            return MoonEPRunner(group, R, S, K, E, H, args.num_sms, hp=args.hp)
        return RUNNERS[lib](group, R, S, K, E, H, args.num_sms)

    # Buffers depend only on (S, H, K, E, num_sms) — not on the routing —
    # so each library's buffer is created ONCE and reused across MaxVio points.
    # Buffer 只依赖 (S, H, K, E, num_sms)——与路由无关——因此每个库只建一次，
    # 在所有 MaxVio 点间复用。
    t0 = time.perf_counter()
    runners = {lib: make_runner(lib) for lib in args.libs}
    t_create = time.perf_counter() - t0
    if rank == 0:
        print(f'[phase] create buffers (once): {t_create:.2f}s', flush=True)

    # 收集所有 (库, MaxVio) 的结果行
    rows = []
    for target in targets:
        # ---- shared inputs: identical routing + data for all libraries ----
        # ——共享输入：所有库使用完全相同的路由与数据——
        # Rank 0 reverse-solves the lognormal sigma for the target MaxVio
        # (deterministic given the fixed seeds), then broadcasts it.
        # rank0 为目标 MaxVio 反解 lognormal 的 sigma（固定种子下确定），
        # 再广播给所有 rank。
        t0 = time.perf_counter()
        if rank == 0:
            sigma, realized = solve_sigma_for_maxvio(target, S, K, E, R, dev)
        else:
            sigma, realized = None, None
        # 用对象广播把 (sigma, realized) 发到所有 rank
        obj = [sigma, realized]
        dist.broadcast_object_list(obj, src=0)
        sigma, realized = obj
        t_solve = time.perf_counter() - t0
        # 各 rank 用共享种子+本地种子生成相同热度的路由矩阵
        topk, tpe = generate_topk_routing(S, K, E, R, sigma, dev, 1234, rank=rank)
        # MaxVio = (max_i Load_i - mean_i Load_i) / mean_i Load_i over the
        # global per-expert routed-token counts: 0 = perfectly balanced,
        # k = the hottest expert carries (k+1)x the balanced load.
        # MaxVio 定义：对全局 per-expert 路由 token 数，(最大负载-平均负载)/
        # 平均负载。0=完美均衡；k=最热专家承载 (k+1) 倍的均衡负载。
        global_tpe = tpe.to(torch.int64)
        dist.all_reduce(global_tpe, group=group)
        maxvio = global_tpe.max().item() / (global_tpe.sum().item() / E) - 1.0
        if rank == 0:
            # 实际值与目标偏差过大时给出提示
            note = '' if abs(maxvio - target) <= max(0.05 * target, 0.01) \
                else f' (target {target} unreachable, floored)'
            print(f'--- target MaxVio={target}: sigma={sigma:.5f}, '
                  f'realized MaxVio={maxvio:.3f}{note} '
                  f'[solve {t_solve:.2f}s]', flush=True)
        # 用每 rank 独立种子生成 hidden 与路由权重（两库一致）
        g = torch.Generator(device=dev).manual_seed(7777 + rank)
        hidden = torch.randn(S, H, dtype=torch.bfloat16, device=dev, generator=g)
        weights = torch.rand(S, K, dtype=torch.float32, device=dev, generator=g)

        for lib in args.libs:
            # 逐库 prepare + 计时四个算子（d_f/d_b/c_f/c_b）
            runner = runners[lib]
            t0 = time.perf_counter()
            runner.prepare(topk, tpe, hidden, weights)
            t_prepare = time.perf_counter() - t0
            t0 = time.perf_counter()
            # 四个通信算子依次用共享计时框架测量
            d_f = time_op(runner.dispatch_fwd, group, args.warmup, args.iters)
            d_b = time_op(runner.dispatch_bwd, group, args.warmup, args.iters)
            c_f = time_op(runner.combine_fwd, group, args.warmup, args.iters)
            c_b = time_op(runner.combine_bwd, group, args.warmup, args.iters)
            t_ops = time.perf_counter() - t0
            if hasattr(runner, 'planning'):
                # MoonEP: planning measured exactly and separately.
                # MoonEP：规划精确且独立计时；前取同样单独计时。
                plan_us = time_op(runner.planning, group, args.warmup, args.iters)
                pf = time_op(runner.prefetch, group, args.warmup, args.iters)
                # 前取带宽：3 份权重 * max_send 个槽位 * H*H' * 2 字节
                pf_gbps = (3 * runner.max_send * H * runner.hp * 2
                           / (pf * 1e-6) / 1e9) if pf > 0 else 0.0
            else:
                # v2: layout is FUSED into dispatch_impl — the public API has
                # no separate layout call (unlike v1's get_dispatch_layout),
                # so it cannot be timed in isolation. The best available
                # estimate is full-dispatch minus cached-dispatch, which comes
                # out ~= 0 — i.e. v2's layout cost is negligible. No prefetch op.
                # v2：layout 融合进 dispatch_impl——公开 API 没有独立的 layout
                # 调用（不同于 v1 的 get_dispatch_layout），无法单独计时；
                # 最好估计是完整 dispatch 减缓存 dispatch，结果≈0，即 v2 的
                # layout 开销可忽略。v2 无前取算子。
                plan_us = max(d_f - d_b, 0.0)
                pf, pf_gbps = 0.0, 0.0
            # 记录一行结果（时延、带宽、字节统计等）
            rows.append(dict(lib=lib, ep=R, E=E, H=H, K=K, sigma=sigma,
                             target_maxvio=target, maxvio=maxvio,
                             plan_us=plan_us, pf_us=pf, pf_gbps=pf_gbps,
                             max_send=getattr(runner, 'max_send', 0),
                             max_recv=getattr(runner, 'max_recv', 0),
                             d_f_us=d_f, d_b_us=d_b, c_f_us=c_f, c_b_us=c_b,
                             d_f_gbps=gbps(S, K, H, d_f),
                             d_b_gbps=gbps(S, K, H, d_b),
                             c_f_gbps=gbps(S, K, H, c_f),
                             c_b_gbps=gbps(S, K, H, c_b)))
            if rank == 0:
                # 打印该库在该 MaxVio 点的各算子时延与带宽
                print(f'{lib:>8} ep={R} E={E} H={H} K={K} sigma={sigma:.5f}: '
                      f'maxvio={maxvio:6.2f} | '
                      f'plan {plan_us:6.1f} us ({plan_us / d_f * 100:4.1f}%) | '
                      f'd_f {d_f:7.1f} us ({gbps(S, K, H, d_f):5.0f} GB/s) | '
                      f'd_b {d_b:7.1f} us ({gbps(S, K, H, d_b):5.0f} GB/s) | '
                      f'c_f {c_f:7.1f} us ({gbps(S, K, H, c_f):5.0f} GB/s) | '
                      f'c_b {c_b:7.1f} us ({gbps(S, K, H, c_b):5.0f} GB/s) | '
                      f'pf {pf:6.1f} us ({pf_gbps:4.0f} GB/s, '
                      f'{pf / d_f * 100:4.1f}%) '
                      f'[prepare {t_prepare:.2f}s ops {t_ops:.2f}s]',
                      flush=True)
        # 释放本轮输入张量
        del hidden, weights

    # 全部测完：销毁各库 Buffer
    t0 = time.perf_counter()
    for runner in runners.values():
        runner.destroy()
    if rank == 0:
        print(f'[phase] destroy buffers: {time.perf_counter() - t0:.2f}s',
              flush=True)

    if rank == 0 and args.out:
        # 写出 CSV（表头取自第一行的键）
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {args.out}')

    if rank == 0 and args.plot:
        # 生成堆叠柱状对比图
        plot_rows(rows, args.plot)
        print(f'wrote {args.plot}')

    # 结束：全局屏障并销毁进程组
    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Plotting (inline): stacked-bar comparison figure
# 绘图（内联）：堆叠柱状对比图
# ---------------------------------------------------------------------------

# 各分段/库的配色
_COLORS = {'comm': '#2563eb', 'planning': '#22c55e', 'prefetch': '#a855f7',
           'v2': '#f59e0b'}


def plot_rows(rows, path):
    # 绘制 2x2 面板：dispatch fwd / dispatch bwd / combine fwd / combine bwd，
    # 每面板把 MoonEP 与 v2 的时延按 comm+planning+prefetch 分段堆叠对比。
    # 使用 Agg 后端（无显示环境）
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np  # noqa: F401 (used by bar positions)

    # 按 (实际 MaxVio, 目标 MaxVio) 排序去重，得到横轴刻度
    pairs = sorted({(r['maxvio'], r['target_maxvio']) for r in rows})
    xs = [m for m, _ in pairs]
    labels = [f'{t:g}' for _, t in pairs]
    C = _COLORS

    def get(lib, op, x):
        # 取某库某算子在给定实际 MaxVio 处的时延
        return next(r[f'{op}_us'] for r in rows
                    if r['lib'] == lib and r['maxvio'] == x)

    # 2x2 面板共享横轴
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True)
    width = 0.35
    xpos = np.arange(len(xs))
    legend_patches = {}

    def stacked(ax, comm_fn, segs, title, total_fn):
        # 通用堆叠柱绘制：底部为 comm 段，其上叠加 segs 中的各分段，
        # 柱顶标注总时延。comm_fn/lib->comm 时延；segs 为 (名称, 取值函数)。
        # 收集各柱底部值用于设置 y 轴下限
        ymin_candidates = []
        for i, lib in enumerate(['moonep', 'v2']):
            # comm 段（dispatch 或 combine 本体）
            comm_vals = [comm_fn(lib, x) for x in xs]
            ymin_candidates += comm_vals
            # MoonEP 与 v2 用不同颜色
            color = C['comm'] if lib == 'moonep' else C['v2']
            ax.bar(xpos + (i - 0.5) * width, comm_vals, width, color=color)
            legend_patches.setdefault(
                'MoonEP comm' if lib == 'moonep' else 'DeepEP v2',
                plt.Rectangle((0, 0), 1, 1, fc=color))
            # 叠加各分段（planning / prefetch）
            bottoms = comm_vals[:]
            for sname, fn in segs:
                vals = [fn(lib, x) for x in xs]
                ax.bar(xpos + (i - 0.5) * width, vals, width, bottom=bottoms,
                       color=C[sname])
                bottoms = [b + v for b, v in zip(bottoms, vals)]
                if lib == 'moonep':
                    legend_patches.setdefault(
                        f'MoonEP {sname}',
                        plt.Rectangle((0, 0), 1, 1, fc=C[sname]))
            # 柱顶标注总时延
            for x0, t in zip(xpos + (i - 0.5) * width,
                             [total_fn(lib, x) for x in xs]):
                ax.text(x0, t + 12, f'{t:.0f}', ha='center', va='bottom',
                        fontsize=7.5)
        # 面板标题/刻度/网格/轴标签
        ax.set_title(title, fontsize=11)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylabel('latency (us)')
        # y 轴范围：下限取最小 comm 值的 85%，上限取全局最大总时延的 112%
        ymax = max(total_fn(l, x) for l in ('moonep', 'v2') for x in xs)
        ax.set_ylim(min(ymin_candidates) * 0.85, ymax * 1.12)

    # 面板 1：dispatch fwd = comm + planning + prefetch
    stacked(axes.flat[0],
            lambda lib, x: get(lib, 'd_f', x) - get(lib, 'plan', x) - get(lib, 'pf', x),
            [('planning', lambda lib, x: get(lib, 'plan', x)),
             ('prefetch', lambda lib, x: get(lib, 'pf', x))],
            'dispatch fwd', lambda lib, x: get(lib, 'd_f', x))

    # 面板 2：dispatch bwd = comm + prefetch
    stacked(axes.flat[1],
            lambda lib, x: get(lib, 'd_b', x) - get(lib, 'pf', x),
            [('prefetch', lambda lib, x: get(lib, 'pf', x))],
            'dispatch bwd', lambda lib, x: get(lib, 'd_b', x))

    # 面板 3：combine fwd（简单分组柱，柱顶标数值）
    ax = axes.flat[2]
    for i, lib in enumerate(['moonep', 'v2']):
        ys = [get(lib, 'c_f', x) for x in xs]
        bars = ax.bar(xpos + (i - 0.5) * width, ys, width,
                      color=C['comm'] if lib == 'moonep' else C['v2'])
        ax.bar_label(bars, fmt='%.0f', fontsize=7.5, padding=2)
    ax.set_title('combine fwd', fontsize=11)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylabel('latency (us)')
    all_y = [r['c_f_us'] for r in rows]
    ax.set_ylim(min(all_y) * 0.92, max(all_y) * 1.06)

    # 面板 4：combine bwd（同面板 3 样式）
    ax = axes.flat[3]
    for i, lib in enumerate(['moonep', 'v2']):
        ys = [get(lib, 'c_b', x) for x in xs]
        bars = ax.bar(xpos + (i - 0.5) * width, ys, width,
                      color=C['comm'] if lib == 'moonep' else C['v2'])
        ax.bar_label(bars, fmt='%.0f', fontsize=7.5, padding=2)
    ax.set_title('combine bwd', fontsize=11)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylabel('latency (us)')
    all_y = [r['c_b_us'] for r in rows]
    ax.set_ylim(min(all_y) * 0.92, max(all_y) * 1.06)

    # 底行面板加横轴说明
    for ax in axes[-1]:
        ax.set_xlabel('MaxVio (0 = balanced, k = hottest expert at (k+1)x load)')

    # 统一图例顺序
    order = ['MoonEP comm', 'MoonEP planning', 'MoonEP prefetch', 'DeepEP v2']
    fig.legend([legend_patches[k] for k in order], order,
               loc='lower center', ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.028))
    # 脚注：grad_reduce 不在关键路径、未计入
    fig.text(0.5, 0.005,
             '* grad_reduce is not on the MoE critical path '
             '(overlappable with subsequent compute) and is not included '
             'in this benchmark.',
             ha='center', fontsize=8.5, style='italic', color='#444444')
    # 总标题：实验配置摘要
    E = rows[0]['E']
    H = rows[0]['H']
    K = rows[0]['K']
    fig.suptitle(f'MoonEP vs DeepEP v2 — ep=8, E={E}, H={H}, K={K}, '
                 f'S=8192/rank, 32 SMs', fontsize=11)
    # 保存图片
    fig.tight_layout(rect=[0, 0.055, 1, 0.96])
    fig.savefig(path, dpi=150)


if __name__ == '__main__':
    main()
