"""
Remote expert prefetch benchmark.

Run with:
    torchrun --nproc_per_node=8 benchmarks/bench_prefetch.py
"""

# 中文说明：远程专家权重前取（prefetch）基准测试。
# 测量对象：launch_prefetch 内核——由专家属主（owner）rank 把本地专家权重
#   经 NVLink 推送到各目标 rank 的 all-rank prefetch 池（owner-push 模式）。
# 测量方法：warmup 后把 iters 次迭代捕获进单个 CUDA graph 再回放计时（消除
#   Python/launch 开销），用 CUDA Event 记录端到端时间；跨 rank 取最差
#   （MAX）平均时延作为报告时延；带宽按逻辑负载字节数 / 时延折算。
# 运行方式：torchrun --nproc_per_node=8 benchmarks/bench_prefetch.py

# argparse：解析命令行参数
import argparse
# os：读取 LOCAL_RANK 等环境变量
import os
# sys：检查原始 argv 以判断是否显式指定了单配置参数
import sys

# torch：张量、随机数、CUDA 事件与 graph
import torch
# torch.distributed：多卡进程组、barrier、all_reduce
import torch.distributed as dist

# MoonEP 的核心 Buffer（提供跨 rank 屏障槽位等共享上下文）
from moonep import Buffer
# VMM 分布式通信池创建函数与 epn 维对齐检查函数
from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment
# 被测内核的启动封装
from moonep.prefetch import launch_prefetch


# Expert shape fixed at 3584x3072 (3583 padded up to a 128 multiple).
# 专家权重形状固定为 3584x3072（3583 向上补齐到 128 的倍数）。
EXP_H = 3584
EXP_HP = 3072

# 三层 FFN 权重的 bf16 布局：gate/up 转置存成 (I, H)，down 为 (H, I)
K3_BF16_WEIGHTS = [
    {"transposed": True},                 # gate  -> (I, H)
    {"transposed": True},                 # up    -> (I, H)
    {},                                   # down  -> (H, I)
]
# 三层 FFN 权重的 MXFP4 布局：2 个 fp4 打进 1 字节，等效维数减半
K3_MXFP4_WEIGHTS = [
    {"transposed": True, "pack": 2},      # gate packed -> (I, H/2)
    {"transposed": True, "pack": 2},      # up packed   -> (I, H/2)
    {"pack": 2},                          # down packed -> (H, I/2)
]
# MXFP4 对应的 ue8m0 缩放因子布局
K3_MXFP4_SCALES = [
    {"scale": True},                      # gate scale  -> (128, 2688)
    {"scale": True},                      # up scale
    {"scale": True},                      # down scale
]

# torch dtype -> 展示用短标签
_DT_LABEL = {
    torch.bfloat16: "bf16",
    torch.int8: "i8",
    torch.uint8: "u8",
}
# 展示标签 -> torch dtype 的反向映射（供 --dtype 参数使用）
_DT_LABEL_TO_TORCH = {v: k for k, v in _DT_LABEL.items()}

# All cases use one selected owner and num_sms == 32. Each local expert is pushed
# a *different* number of times in 0..3 (R-1=3 is the max), via `counts[e]`.
# 0 means that expert is skipped. Default shape is 3584x3072; later cases vary
# the expert shape and epn.
# 所有用例共用一个属主 rank 且 num_sms == 32。每个本地专家被推送的次数
# 由 counts[e] 指定（0..3，R-1=3 为上限；0 表示跳过该专家）。
# 默认形状 3584x3072；后续用例改变专家形状与 epn。
NUM_SMS = 32
# 默认用例套件：从"只有少数槽位活跃"到"全部饱和"逐步加压
DEFAULT_CASES = [
    # small-slot cases: only a few experts prefetched at all.
    # 小槽位用例：只有少数专家被前取。
    {"label": "slots_1",  "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [1, 0, 0, 0, 0, 0, 0, 0]},
    {"label": "slots_2",  "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [1, 1, 0, 0, 0, 0, 0, 0]},
    {"label": "slots_3",  "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [2, 1, 0, 0, 0, 0, 0, 0]},
    {"label": "slots_5",  "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [3, 2, 0, 0, 0, 0, 0, 0]},
    # counts 从 0 到 3 逐列爬坡，模拟温和的负载
    {"label": "ramp_0_3", "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [0, 1, 2, 3, 0, 1, 2, 3]},
    # 交错出现满/空/中/低负载
    {"label": "mixed",    "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    # 大部分槽位高负载
    {"label": "heavy",    "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "counts": [3, 3, 2, 3, 1, 0, 2, 3]},
    # saturated: every expert contributed by every remote rank (24 slots).
    # 饱和用例：每个专家都被所有远端 rank 推送（共 24 个槽位）。
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
    # --- Quantized experts -------------------------------------------------
    # —— 量化专家用例：u8 存储 = 打包的 MXFP4 或其 ue8m0 scale ——
    {"label": "mxfp4_gate_up", "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "transposed": True, "pack": 2,
     "dtype": torch.uint8, "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    {"label": "mxfp4_down", "epn": 8, "H": EXP_H, "Hp": EXP_HP, "pack": 2,
     "dtype": torch.uint8, "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    # 整层 3 个权重张量按 bf16 布局前取
    {"label": "k3_layer_bf16", "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "weight_parts": K3_BF16_WEIGHTS, "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
    # 整层 3 个权重张量按 MXFP4 布局 + scale 张量一起前取
    {"label": "k3_layer_mxfp4", "epn": 8, "H": EXP_H, "Hp": EXP_HP,
     "dtype": torch.uint8,
     "weight_parts": K3_MXFP4_WEIGHTS,
     "scale_parts": K3_MXFP4_SCALES,
     "counts": [3, 0, 2, 1, 3, 0, 2, 1]},
]


def derive_extents(H, Hp, part):
    # 由专家逻辑形状 (H, Hp) 和部件描述推导该张量的实际存储形状 (th, thp)。
    if part.get("scale"):
        # One ue8m0 byte per 32 values, re-cut into whole 128x128 tiles and
        # padded in 4096-column units so the default epn=8 VMM chunk is aligned.
        # scale 张量：每 32 个值对应 1 字节 ue8m0；重新切成完整的 128x128 块，
        # 并按 4096 列为单位补齐，使默认 epn=8 的 VMM chunk 保持对齐。
        nbytes = H * Hp // 32
        return 128, ((nbytes // 128 + 4095) // 4096) * 4096
    # 权重张量：转置则形状互换为 (Hp, H)；pack=2 表示 2 值/字节，收缩维减半
    out, contracted = (Hp, H) if part.get("transposed") else (H, Hp)
    return out, contracted // int(part.get("pack", 1))


def resolve_parts(case):
    # 把用例的部件描述解析成 [(th, thp, dtype), ...]：权重 3 份 + 可选 scale 3 份。
    H, Hp = int(case["H"]), int(case["Hp"])
    # 未显式指定 dtype 时默认 bf16
    default_dt = case.get("dtype", torch.bfloat16)
    # weight_parts 未给时三份权重共用用例本身的描述
    weight_parts = case.get("weight_parts") or [case] * 3
    scale_parts = case.get("scale_parts") or []

    def resolve(parts):
        # 逐部件推导存储形状与 dtype
        resolved = []
        for part in parts:
            th, thp = derive_extents(H, Hp, part)
            resolved.append((th, thp, part.get("dtype", default_dt)))
        return resolved

    weights = resolve(weight_parts)
    scales = resolve(scale_parts)
    # 必须恰好 3 份权重，scale 要么 0 份要么 3 份
    assert len(weights) == 3 and len(scales) in (0, 3)
    return weights, scales


def fill_random(shape, dtype, gen, dev="cuda"):
    # 用给定生成器填充随机数据：浮点用 randn，整数用 randint（避开 max 边界）
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=dev, generator=gen)
    info = torch.iinfo(dtype)
    return torch.randint(info.min, info.max, shape, dtype=dtype, device=dev,
                         generator=gen)


def setup():
    # 初始化 NCCL 进程组并绑定本进程的 GPU（LOCAL_RANK）
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return rank, dist.get_world_size()


def expert_plan(R, epn, counts, owner_rank, dev):
    """Owner rank pushes expert column e to ``counts[e]`` destinations.

    Each destination stores the global owner expert id in slot column e.
    All other slots stay idle.
    """
    # 中文：属主 rank 把专家列 e 推送到 counts[e] 个目的地。
    # 每个目的地在槽位列 e 处写入属主专家的全局 id；其余槽位保持空闲（-1）。
    # 计划表形状 [R, epn]，-1 表示该槽位空闲
    plan = torch.full((R, epn), -1, dtype=torch.int32, device=dev)
    for e in range(epn):
        for offset in range(1, counts[e] + 1):
            # 目的地沿 rank 环向后取 offset 个 rank，避开属主自身
            dst_rank = (owner_rank + offset) % R
            # 在目的地槽位列 e 写入属主的全局专家 id
            plan[dst_rank, e] = owner_rank * epn + e
    return plan


def bench_case(case, args, rank, R):
    # 运行单个用例：建张量与 Buffer -> warmup -> （可选）CUDA graph 捕获 -> 计时
    # -> 换算带宽 -> 销毁 Buffer，返回结果元组。
    dev = "cuda"
    # 每 rank 专家数
    epn = int(case["epn"])
    # 全局专家总数
    E = R * epn
    # 解析权重（与可选 scale）的实际形状与 dtype
    weight_parts, scale_parts = resolve_parts(case)
    parts = (*weight_parts, *scale_parts)
    # 取第一份权重的形状/dtype（Buffer 构造与标签用）
    th0, _thp0, dt0 = weight_parts[0]
    # counts 未给出时按统一的 nremote 展开
    counts = case.get("counts") or [int(case.get("nremote", 1))] * epn
    num_sms = NUM_SMS

    # 校验：所有部件形状必须是 128 的倍数（VMM chunk 对齐的前提）
    for th, thp, _dt in parts:
        assert th % 128 == 0 and thp % 128 == 0, \
            f"{case['label']}: derived extents must be multiples of 128, " \
            f"got ({th}, {thp})"
        # 校验固定 epn 的 VMM chunk 尺寸对齐成立
        assert pad_dim0_for_alignment([epn, th, thp], _dt) == epn, \
            f"{case['label']}: fixed-epn VMM chunk must be aligned"
    # counts 数量必须等于 epn，且每个计数落在 [0, R)
    assert len(counts) == epn, f"{case['label']}: counts must have epn={epn} entries"
    assert all(0 <= c < R for c in counts), f"{case['label']}: each count must be in [0, R)"

    # 属主 rank 编号（由命令行指定）
    owner_rank = int(args.owner_rank)

    def make_tensors(resolved_parts, tensor_offset):
        # 为一组部件创建：本地随机数据张量 + all-rank NVL 通信池视图
        locals_ = []
        pools = []
        for i, (th, thp, dt) in enumerate(resolved_parts):
            # 每个张量用独立且确定的种子（本地数据各 rank 不同，模拟真实分布）
            gen = torch.Generator(device=dev).manual_seed(
                321 + rank + tensor_offset + i
            )
            locals_.append(fill_random((epn, th, thp), dt, gen, dev))
            # 创建 VMM 分布式池并 view 成 [R, epn, th, thp]
            pool = create_nvl_dist_tensor([epn, th, thp], dt, rank, R)
            pools.append(pool.view(R, epn, th, thp))
        return tuple(locals_), tuple(pools)

    # 本地权重 + 前取目标池（权重与 scale 分别创建，种子偏移 3 避免重叠）
    local_weights, prefetch_buffers = make_tensors(weight_parts, 0)
    local_scales, scale_prefetch_buffers = make_tensors(scale_parts, 3)
    # 生成专家推送计划表
    experts_to_copy = expert_plan(R, epn, counts, owner_rank, dev)
    # scale 的 (本地, 池) 配对；无 scale 时置 None
    scales = tuple(zip(
        local_scales, scale_prefetch_buffers, strict=True,
    )) if local_scales else None

    # 创建 Buffer（仅为了取其共享的 meta_buf / 屏障槽位等上下文）
    buffer = Buffer(
        S=1,
        H=th0,
        K=1,
        E=E,
        num_ep_ranks=R,
        num_sms=num_sms,
        token_padding=1,
        explicitly_destroy=True,
    )
    # 取内部上下文（meta_buf、grid_sync_bar 等）
    ctx = buffer._require_ctx()
    # 全局同步后再开始 warmup，确保所有 rank 的池/缓冲就绪
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    def prefetch_once():
        # 单次前取：由 launch_prefetch 按计划表把属主专家推入各 rank 的池
        launch_prefetch(
            local_weights,
            prefetch_buffers,
            experts_to_copy,
            rank=rank,
            num_sms=num_sms,
            meta_buf=ctx["meta_buf"],
            meta_stride=int(ctx["meta_chunk_padded"]),
            barrier_off=int(ctx["BARRIER_OFF"]),
            grid_sync_bar=ctx["grid_sync_bar"],
            scales=scales,
        )

    # Warmup (also JIT-compiles the kernel) then capture the iters loop into a
    # single CUDA graph to strip launch/python overhead.
    # 预热（同时触发内核 JIT 编译），随后把 iters 次迭代捕获进单个 CUDA graph，
    # 以剔除 launch/Python 开销。
    for _ in range(args.warmup):
        prefetch_once()
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
                prefetch_once()
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
            prefetch_once()
    end.record()
    end.synchronize()

    # 本 rank 平均单次时延（us）
    local_us = start.elapsed_time(end) * 1e3 / args.iters
    # 跨 rank all_reduce 取最大值：前取是集体操作，整体时延由最慢 rank 决定
    worst = torch.tensor([local_us], dtype=torch.float64, device=dev)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    worst_us = float(worst.item())

    # This benchmark has one sending owner. Count each active expert's
    # payload once for local reads, then once per destination slot for writes.
    # These are logical payload bytes, not measured physical HBM traffic.
    # 本基准只有一个发送属主。字节量口径：每个活跃专家的负载先被本地读一次，
    # 再为每个目的槽位写一次。这是逻辑负载字节，不是实测 HBM 物理流量。
    # 活跃专家数（counts>0 的列数）
    active_experts = sum(c > 0 for c in counts)
    # 总槽位数（所有 counts 之和）
    slots = int(sum(counts))
    # 单槽位负载字节（所有部件之和）
    per_slot = sum(th * thp * dt.itemsize for th, thp, dt in parts)
    # 属主侧总字节数 = 读 + 写
    owner_payload_bytes = (active_experts + slots) * per_slot
    # 属主侧带宽（GB/s）
    bw_gbs = owner_payload_bytes / worst_us * 1e6 / 1e9 if worst_us > 0 else 0.0
    # With one owner, total sent bytes >= any rank's received bytes, so this
    # also equals max(max_send_bytes, max_recv_bytes).
    # 单属主时总发送字节 >= 任一 rank 的接收字节，故该口径也等于
    # max(max_send_bytes, max_recv_bytes)。
    comm_gbs = slots * per_slot / worst_us * 1e6 / 1e9 if worst_us > 0 else 0.0

    # 同步后销毁 Buffer，释放 VMM/屏障资源
    dist.barrier(device_ids=[torch.cuda.current_device()])
    buffer.destroy()
    # dtype 标签：所有部件一致时用其标签，混合时标 "mix"
    dt_label = (_DT_LABEL.get(dt0, str(dt0))
                if len({dt for _, _, dt in parts}) == 1 else "mix")
    # 返回 (最差时延, 数据量MB, 属主带宽, 通信带宽, 专家数, 槽位数, 张量数, dtype标签, H, Hp)
    return (worst_us, owner_payload_bytes / 1e6, bw_gbs, comm_gbs, E,
            slots, len(parts), dt_label, int(case["H"]), int(case["Hp"]))


def explicit_single_config_requested(argv):
    # 判断用户是否显式给出了形状类参数（--epn/--H/--Hp/--nremote/--dtype），
    # 若有则自动进入"单配置"模式
    shape_flags = ("--epn", "--H", "--Hp", "--nremote", "--dtype")
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
    # 形状参数：每 rank 专家数、专家两维、每专家远端推送次数
    parser.add_argument("--epn", type=int, default=8)
    parser.add_argument("--H", type=int, default=3584)
    parser.add_argument("--Hp", type=int, default=3072)
    parser.add_argument("--nremote", type=int, default=2)
    # 前取张量的元素类型（u8 = 打包 MXFP4 或其 ue8m0 scale）
    parser.add_argument("--dtype", choices=sorted(_DT_LABEL_TO_TORCH),
                        default="bf16",
                        help="Element type of the prefetched tensor "
                             "(u8 = packed MXFP4 or its ue8m0 scales)")
    # 预热与计时迭代次数
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    # 属主 rank：其所在 NUMA/NVLink 位置决定测得的是哪种链路带宽
    parser.add_argument("--owner-rank", type=int, default=1,
                        help="Rank that owns and pushes the selected experts. "
                             "Choose its placement to measure "
                             "NVLink.")
    # 关闭 CUDA graph 计时（供 NCU 使用）
    parser.add_argument("--no-graph", action="store_true",
                        help="Time plain launches instead of a CUDA graph (for NCU)")
    return parser.parse_args()


def main():
    # 入口：解析参数、初始化分布式、选择用例并逐个计时打印
    args = parse_args()
    rank, R = setup()
    # 前取至少需要 2 张卡（存在远端才有意义）
    assert R >= 2, "bench_prefetch requires at least 2 GPUs"

    # --single 与 --suite 互斥
    if args.single and args.suite:
        raise ValueError("Use only one of --single or --suite")
    # 属主 rank 必须在有效范围内
    if not 0 <= args.owner_rank < R:
        raise ValueError(
            f"--owner-rank must be in [0, {R}), got {args.owner_rank}"
        )

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
            "dtype": _DT_LABEL_TO_TORCH[args.dtype],
            "nremote": args.nremote,
        }]
    else:
        # 否则跑默认套件
        cases = DEFAULT_CASES

    if rank == 0:
        # 表头：说明口径与列含义
        print(
            f"MoonEP Prefetch Benchmark (R={R}, warmup={args.warmup}, "
            f"iters={args.iters}, owner=rank{args.owner_rank}, "
            f"num_sms={NUM_SMS})"
        )
        print(
            "Data/BW: owner logical reads + remote writes; "
            "CommBW: remote-write payload GB/s; "
            "both bandwidths use the maximum per-rank average latency."
        )
        print(
            f"{'Config':<20} {'E':>5} {'epn':>4} {'dt':>5} {'N':>3} {'H':>7} {'Hp':>6} "
            f"{'SMs':>5} {'Slots':>6} {'Data(MB)':>10} {'Worst(us)':>10} {'BW(GB/s)':>9} {'CommBW':>8}"
        )
        print("-" * 107)

    for case in cases:
        # 逐用例测量并打印（仅 rank 0 输出）
        (worst_us, mb, bw_gbs, comm_gbs, E, slots,
         ntensor, dt_label, H, Hp) = bench_case(case, args, rank, R)
        if rank == 0:
            print(
                f"{case['label']:<20} {E:>5} {case['epn']:>4} {dt_label:>5} {ntensor:>3} "
                f"{H:>7} {Hp:>6} {NUM_SMS:>5} {slots:>6} "
                f"{mb:>10.2f} {worst_us:>10.2f} {bw_gbs:>9.2f} {comm_gbs:>8.2f}"
            )

    # 结束：销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
