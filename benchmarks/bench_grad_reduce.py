"""
Remote expert grad reduce benchmark.

Run with:
    torchrun --nproc_per_node=8 benchmarks/bench_grad_reduce.py
"""

# 中文说明：远程专家梯度归约（grad reduce）基准测试。
# 测量对象：launch_grad_reduce 内核——反向传播后，把各 rank 上重复专家的
#   fp32 梯度经 NVLink 写入属主 rank 的 reduce 池，并在属主侧原位累加。
# 测量方法：warmup 后把 iters 次迭代捕获进单个 CUDA graph 回放计时；
#   由于本基准所有专家都归到 rank0，直接取 rank0 的平均时延（它就是最慢
#   的关键路径 rank，其余 rank 只在屏障处等待）；带宽按关键路径字节折算。
# 运行方式：torchrun --nproc_per_node=8 benchmarks/bench_grad_reduce.py

# argparse：解析命令行参数
import argparse
# os：读取 LOCAL_RANK 等环境变量
import os
# sys：检查原始 argv 以判断是否显式指定了单配置参数
import sys

# torch：张量、随机数、CUDA 事件与 graph
import torch
# torch.distributed：多卡进程组、barrier
import torch.distributed as dist

# MoonEP 的核心 Buffer（提供跨 rank 屏障槽位等共享上下文）
from moonep import Buffer
# VMM 分布式通信池创建函数与 epn 维对齐检查函数
from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment
# 被测内核的启动封装
from moonep.grad_reduce import launch_grad_reduce


# Expert shape fixed at 3584x3072 (3583 padded up to a 128 multiple).
# 专家梯度形状固定为 3584x3072（3583 向上补齐到 128 的倍数）。
EXP_H = 3584
EXP_HP = 3072

# All ranks launch collectively with rank0 as the busiest owner. Each local
# expert is reduced
# a *different* number of times in 0..3 (R-1=3 is the max), via `counts[e]`.
# 0 means that expert is skipped. Default shape is 3584x3072; later cases vary
# the expert shape and epn.
# 所有 rank 集体启动，rank0 是负载最重的属主。每个本地专家被归约的次数
# 由 counts[e] 指定（0..3，R-1=3 为上限；0 表示跳过该专家）。
# 默认形状 3584x3072；后续用例改变专家形状与 epn。
NUM_SMS = 32
# 默认用例套件：从"只有少数槽位活跃"到"全部饱和"逐步加压
DEFAULT_CASES = [
    # small-slot cases: only a few experts reduced at all.
    # 小槽位用例：只有少数专家被归约。
    {"label": "slots_1",  "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [1, 0, 0, 0, 0, 0, 0, 0]},
    {"label": "slots_2",  "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [1, 1, 0, 0, 0, 0, 0, 0]},
    {"label": "slots_3",  "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [2, 1, 0, 0, 0, 0, 0, 0]},
    {"label": "slots_5",  "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [3, 2, 0, 0, 0, 0, 0, 0]},
    # counts 从 0 到 3 逐列爬坡
    {"label": "ramp_0_3", "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [0, 1, 2, 3, 0, 1, 2, 3]},
    # 交错出现满/空/中/低负载
    {"label": "mixed",    "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    # 大部分槽位高负载
    {"label": "heavy",    "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [3, 3, 2, 3, 1, 0, 2, 3]},
    # saturated: every expert contributed by every remote rank (24 slots).
    # 饱和用例：每个专家都被所有远端 rank 贡献（共 24 个槽位）。
    {"label": "full_3x8", "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [3, 3, 3, 3, 3, 3, 3, 3]},
    # dense slot columns: every slot is active.
    # 密集槽位列用例：每个槽位都活跃。
    {"label": "dense_epn8", "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [2, 1, 3, 1, 2, 3, 1, 2]},
    # fewer / more experts per rank at the same shape.
    # 相同形状下每 rank 专家数更少 / 更多。
    {"label": "epn4_full",  "epn": 4, "H": EXP_H, "Hp": EXP_HP,
     "counts": [3, 3, 3, 3]},
    {"label": "epn16_mixed", "epn": 16, "H": EXP_H, "Hp": EXP_HP,
     "counts": [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]},
    # different expert shapes (same mixed/heavy count patterns).
    # 不同专家形状（沿用 mixed/heavy 的 counts 模式）。
    {"label": "thin_7168x128", "epn": 8, "H": 7168, "Hp": 128,
     "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    {"label": "tall_1024x3072", "epn": 8, "H": 1024, "Hp": 3072,
     "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    {"label": "tiny_512x512", "epn": 8, "H": 512, "Hp": 512,
     "counts": [3, 3, 2, 3, 1, 0, 2, 3]},
]


def setup():
    # 初始化 NCCL 进程组并绑定本进程的 GPU（LOCAL_RANK）
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return rank, dist.get_world_size()


def expert_plan(R, epn, counts, dev):
    """rank0 owns experts 0..epn-1; expert e is reduced ``counts[e]`` times.

    For expert e the contributing remote ranks are 1..counts[e], each placing
    its grad in slot column e. All other slots stay idle (-1). counts[e] must
    be < R (only R-1 remote ranks exist).
    """
    # 中文：rank0 拥有专家 0..epn-1；专家 e 被归约 counts[e] 次。
    # 对专家 e，贡献方是远端 rank 1..counts[e]，各自把梯度放到槽位列 e；
    # 其余槽位保持空闲（-1）。counts[e] 必须 < R（只有 R-1 个远端 rank）。
    # 计划表形状 [R, epn]，-1 表示该槽位空闲
    plan = torch.full((R, epn), -1, dtype=torch.int32, device=dev)
    for e in range(epn):
        for src_rank in range(1, counts[e] + 1):
            # 源 rank src_rank 在自己的槽位列 e 写入属主本地专家编号 e
            plan[src_rank, e] = e  # owner rank0, local expert e
    return plan


def bench_case(case, args, rank, R):
    # 运行单个用例：建张量与 Buffer -> warmup -> （可选）CUDA graph 捕获
    # -> 重置槽位数据 -> 计时 -> 折算带宽 -> 销毁 Buffer，返回结果元组。
    dev = "cuda"
    # 每 rank 专家数
    epn = int(case["epn"])
    # 全局专家总数
    E = R * epn
    # 专家梯度的两个维度
    H = int(case["H"])
    Hp = int(case["Hp"])
    # counts 未给出时按统一的 nremote 展开
    counts = case.get("counts") or [int(case.get("nremote", 1))] * epn
    num_sms = NUM_SMS

    # 校验：H/Hp 必须是 128 的倍数
    assert H % 128 == 0 and Hp % 128 == 0, \
        f"{case['label']}: H and Hp must be multiples of 128"
    # counts 数量必须等于 epn，且每个计数落在 [0, R)
    assert len(counts) == epn, f"{case['label']}: counts must have epn={epn} entries"
    assert all(0 <= c < R for c in counts), f"{case['label']}: each count must be in [0, R)"
    # 校验 fp32 梯度池的固定 epn VMM chunk 对齐成立
    assert pad_dim0_for_alignment([epn, H, Hp], torch.float32) == epn, \
        f"{case['label']}: fixed-epn VMM chunk must be aligned"

    # 创建跨 rank 的 fp32 reduce 池并 view 成 [R, epn, H, Hp]
    reduce_full = create_nvl_dist_tensor([epn, H, Hp], torch.float32, rank, R)
    reduce_buffers = reduce_full.view(R, epn, H, Hp)
    # 生成归约计划表（属主固定为 rank0）
    experts_to_copy = expert_plan(R, epn, counts, dev)

    # grad_reduce needs the cross-rank barrier slots from Buffer.
    # grad_reduce 需要使用 Buffer 里的跨 rank 屏障槽位。
    buffer = Buffer(
        S=128, H=H, K=1, E=E, num_ep_ranks=R,
        num_sms=num_sms,
    )
    # 取内部上下文（meta_buf、grid_sync_bar 等）
    ctx = buffer._require_ctx()

    # 生成本 rank 的本地专家梯度（各 rank 独立种子）
    gen = torch.Generator(device=dev).manual_seed(321 + rank)
    local_expert_grads = torch.randn(
        epn, H, Hp, dtype=torch.float32, device=dev, generator=gen
    )
    # 预填本 rank 的 reduce 槽位（模拟上游已写入的重复梯度）
    local_slots = torch.randn_like(reduce_buffers[rank])
    reduce_buffers[rank].copy_(local_slots)
    # 全局同步后再开始 warmup，确保所有 rank 的池就绪
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    def reduce_once():
        # 单次归约：远端梯度槽位经 NVLink 读到属主并原位累加进本地梯度
        launch_grad_reduce(
            local_expert_grads,
            reduce_buffers,
            experts_to_copy,
            rank=rank,
            num_sms=num_sms,
            meta_buf=ctx['meta_buf'],
            meta_stride=ctx['meta_chunk_padded'],
            barrier_off=ctx['BARRIER_OFF'],
            grid_sync_bar=ctx['grid_sync_bar'],
        )

    # Warmup (also JIT-compiles + primes the cross-rank barrier) then capture
    # the iters loop into a single CUDA graph to strip launch/python overhead.
    # 预热（同时触发内核 JIT 编译并"预热"跨 rank 屏障），随后把 iters 次迭代
    # 捕获进单个 CUDA graph，以剔除 launch/Python 开销。
    for _ in range(args.warmup):
        reduce_once()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    # --no-graph keeps plain stream launches so tools like NCU can intercept
    # each kernel (graph capture/replay hides launches from kernel filters).
    # --no-graph 保持普通流式 launch，便于 NCU 等工具拦截每个内核
    # （graph 捕获/回放会让内核对过滤器不可见）。
    graph = None
    if not args.no_graph:
        # 捕获 iters 次迭代到一个 graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.iters):
                reduce_once()
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])

    # 计时前把属主槽位重置回初始值：归约会修改 reduce 池内容，
    # 重置保证每次计时的内存写入路径一致（fp32 累加结果不同不影响时延）
    reduce_buffers[rank].copy_(local_slots)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    # 用两个 CUDA Event 夹住计时区间
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if graph is not None:
        # 回放 graph（内含 iters 次迭代）
        graph.replay()
    else:
        # 无 graph 时逐次 eager launch
        for _ in range(args.iters):
            reduce_once()
    end.record()
    end.synchronize()

    # Time rank0 only — it owns every expert here, so it is the worst rank and
    # the idle ranks just wait at the barrier.
    # 只取 rank0 的时延——本基准中它拥有全部专家，必然是最慢的关键路径
    # rank，其余 rank 只在屏障处等待。
    worst_us = start.elapsed_time(end) * 1e3 / args.iters if rank == 0 else 0.0

    # Critical-path traffic for the owner rank: read every consumed reduce
    # slot (4B/elem), then read+write the grad of each active expert (8B/elem).
    # Slot clears happen locally on each source rank in parallel, off the
    # owner's critical path, so they are not counted.
    # 属主 rank 的关键路径流量：先读每个被消费的槽位（4B/元素），
    # 再对每个活跃专家的梯度读+写（8B/元素）。槽位清零在各源 rank 本地
    # 并行执行、不在属主关键路径上，因此不计入。
    # 总槽位数
    slots = int(sum(counts))
    # 活跃专家数（counts>0 的列数）
    active = int(sum(1 for c in counts if c > 0))
    # 单个专家梯度 tile 的字节数（fp32）
    tile = H * Hp * 4
    # 属主关键路径字节 = 槽位读 + 活跃专家梯度的读+写
    bytes_per_rank = slots * tile + active * tile * 2
    # 有效带宽（GB/s）
    bw_gbs = bytes_per_rank / worst_us * 1e6 / 1e9 if worst_us > 0 else 0.0
    # pure NVLink read traffic: only the remote reduce-slot reads (grad rw and
    # clears are local HBM, off the NVLink path).
    # 纯 NVLink 读流量口径：只计远端槽位读取（梯度读写与清零都是本地
    # HBM 操作，不占 NVLink）。
    comm_gbs = slots * tile / worst_us * 1e6 / 1e9 if worst_us > 0 else 0.0

    # 同步后销毁 Buffer，释放资源
    dist.barrier(device_ids=[torch.cuda.current_device()])
    buffer.destroy()
    # 返回 (最差时延, 数据量MB, 带宽, 通信带宽, 专家数, 槽位数)
    return worst_us, bytes_per_rank / 1e6, bw_gbs, comm_gbs, E, slots


def explicit_single_config_requested(argv):
    # 判断用户是否显式给出了形状类参数（--epn/--H/--Hp/--nremote），
    # 若有则自动进入"单配置"模式
    shape_flags = ("--epn", "--H", "--Hp", "--nremote")
    for arg in argv:
        for flag in shape_flags:
            if arg == flag or arg.startswith(flag + "="):
                return True
    return False


def parse_args():
    # 命令行参数定义
    parser = argparse.ArgumentParser()
    # --suite：跑默认多用例套件
    parser.add_argument("--suite", action="store_true",
                        help="Run the default multi-case benchmark suite")
    # --single：只跑 --epn/--H/--Hp/--nremote 描述的单一配置
    parser.add_argument("--single", action="store_true",
                        help="Run exactly the config described by --epn/--H/--Hp/--nremote")
    # 形状参数：每 rank 专家数、专家梯度两维、每专家远端贡献次数
    parser.add_argument("--epn", type=int, default=8)
    parser.add_argument("--H", type=int, default=3584)
    parser.add_argument("--Hp", type=int, default=3072)
    parser.add_argument("--nremote", type=int, default=2)
    # 预热与计时迭代次数
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    # 关闭 CUDA graph 计时（供 NCU 使用）
    parser.add_argument("--no-graph", action="store_true",
                        help="Time plain launches instead of a CUDA graph (for NCU)")
    return parser.parse_args()


def main():
    # 入口：解析参数、初始化分布式、选择用例并逐个计时打印
    args = parse_args()
    rank, R = setup()
    # 归约至少需要 2 张卡（存在远端贡献者才有意义）
    assert R >= 2, "bench_grad_reduce requires at least 2 GPUs"

    # --single 与 --suite 互斥
    if args.single and args.suite:
        raise ValueError("Use only one of --single or --suite")

    # 单配置模式：显式 --single，或未指定 --suite 但命令行带了形状参数
    run_single = args.single or (
        not args.suite and explicit_single_config_requested(sys.argv[1:])
    )
    if run_single:
        # 由命令行参数合成单用例
        cases = [{
            "label": "custom",
            "epn": args.epn,
            "H": args.H,
            "Hp": args.Hp,
            "nremote": args.nremote,
        }]
    else:
        # 否则跑默认套件
        cases = DEFAULT_CASES

    if rank == 0:
        # 表头：说明口径与列含义
        print(
            f"MoonEP Grad Reduce Benchmark (R={R}, warmup={args.warmup}, "
            f"iters={args.iters})"
        )
        print(
            f"{'Config':<20} {'E':>5} {'epn':>4} {'H':>7} {'Hp':>6} "
            f"{'SMs':>5} {'Slots':>6} {'Data(MB)':>10} {'Worst(us)':>10} {'BW(GB/s)':>9} {'CommBW':>8}"
        )
        print("-" * 97)

    for case in cases:
        # 逐用例测量并打印（仅 rank 0 输出）
        worst_us, mb, bw_gbs, comm_gbs, E, slots = bench_case(
            case, args, rank, R
        )
        if rank == 0:
            print(
                f"{case['label']:<20} {E:>5} {case['epn']:>4} "
                f"{case['H']:>7} {case['Hp']:>6} {NUM_SMS:>5} {slots:>6} "
                f"{mb:>10.2f} {worst_us:>10.2f} {bw_gbs:>9.2f} {comm_gbs:>8.2f}"
            )

    # 结束：销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
