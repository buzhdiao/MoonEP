"""
MoonEP inter-rank sync - a minimal CuTe DSL cross-rank barrier kernel.

This is used to align EP ranks immediately before launching the planning
kernel, so planning/communication timings are less affected by CPU-side or
upstream stream skew.
"""
# 中文说明：MoonEP 的跨 rank 同步——一个极简的 CuTe DSL 跨 rank 屏障内核。
# 用途：在启动规划（planning）内核之前先把所有 EP rank 对齐，使规划/通信的
# 计时尽量不受 CPU 侧延迟或上游流（stream）相位差的影响。

# functools：提供 lru_cache，用于按参数缓存编译好的内核
import functools

# torch：读取张量的 dtype/设备/数据指针，以及获取当前 CUDA 流
import torch
# cuda.bindings.driver：CUDA 驱动 API 绑定，这里主要用 CUstream 类型
import cuda.bindings.driver as cuda

# cutlass / cute：NVIDIA CuTe DSL，用于以 Python 定义并 JIT 编译 GPU 内核
import cutlass
import cutlass.cute as cute
# Int32：CuTe DSL 的 32 位整数标量类型（内核参数用）
from cutlass import Int32
# make_ptr：把裸指针地址包装成带类型/地址空间/对齐信息的 CuTe 指针
from cutlass.cute.runtime import make_ptr

# cross_rank_barrier：_common 中实现的多 rank 屏障原语（本内核的核心逻辑）
from moonep._common import cross_rank_barrier


class InterRankSyncKernel:
    """Single-CTA inter-rank barrier over the shared meta_buf barrier slots."""
    # 中文：单 CTA 的跨 rank 屏障，复用共享 meta_buf 中的屏障槽位。
    # 所有 rank 各 launch 一个单 CTA 内核，通过全局内存计数器互相等待，实现栅栏。

    def __init__(self, R: int, meta_stride: int, num_threads: int):
        # 仅保存编译期常量：rank 数、meta 槽位步长、每 CTA 线程数
        self.R = R
        self.meta_stride = meta_stride
        self.num_threads = num_threads

    # @cute.jit：把 __call__ 编译为可执行的 JIT 入口（host 侧包装）
    @cute.jit
    def __call__(
        self,
        meta_ptr: cute.Pointer,      # int32 [R * meta_stride]
        bar_ptr: cute.Pointer,       # int32 [1] grid barrier counter
        rank: Int32,
        barrier_off: Int32,
        stream: cuda.CUstream,
    ):
        # 把 Python 侧配置转成编译期常量，让内核按 (R, meta_stride) 特化
        R = cutlass.const_expr(self.R)
        meta_stride = cutlass.const_expr(self.meta_stride)

        # 用 meta_ptr 构造一维 int32 张量视图（长度 R*meta_stride，覆盖所有 rank 的 meta 槽）
        meta_tensor = cute.make_tensor(
            meta_ptr,
            cute.make_layout((R * meta_stride,)),
        )
        # 用 bar_ptr 构造长度为 1 的张量视图（全局屏障计数器）
        bar_tensor = cute.make_tensor(bar_ptr, cute.make_layout((1,)))

        # 启动设备侧 kernel：单 CTA、num_threads 线程、协作式 launch
        # cooperative=True 允许 CTA 内做 grid 级同步原语；smem=0 表示不用共享内存
        self.kernel(
            meta_tensor,
            bar_tensor,
            rank,
            barrier_off,
        ).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            smem=0,
            stream=stream,
            cooperative=True,
        )

    # @cute.kernel：标记为设备侧内核（在 GPU 上执行）
    @cute.kernel
    def kernel(
        self,
        meta_tensor: cute.Tensor,
        bar_tensor: cute.Tensor,
        rank: Int32,
        barrier_off: Int32,
    ):
        # 再次把 meta 步长 / rank 数固定为编译期常量
        meta_stride = cutlass.const_expr(self.meta_stride)
        num_ranks = cutlass.const_expr(self.R)
        # 取当前线程在其 CTA 内的线程号（单 CTA 内核，0..num_threads-1）
        tid = cute.arch.thread_idx()[0]

        # 调用通用跨 rank 屏障：各线程参与对 meta_buf 屏障槽位的
        # 到达/释放计数，直到所有 rank 都到达后才继续往下走。
        cross_rank_barrier(
            meta_tensor,
            meta_stride,
            barrier_off,
            rank,
            num_ranks,
            bar_tensor.iterator,
            Int32(1),
            cutlass.const_expr(self.num_threads),
            tid,
        )


def _num_threads_for_ranks(R: int) -> int:
    # 根据 rank 数选择内核线程数：向上取整到 32 的倍数，且至少一个 warp（32）
    assert R > 0, f"R must be positive, got {R}"
    assert R <= 1024, f"inter_rank_sync supports at most 1024 ranks, got {R}"
    return max(32, ((R + 31) // 32) * 32)


# @functools.lru_cache：按 (R, meta_stride, num_threads, device_index) 缓存编译产物，
# 避免每次 launch 都重新 JIT 编译
@functools.lru_cache(maxsize=None)
def _get_compiled(
    R: int,
    meta_stride: int,
    num_threads: int,
    device_index: int,
):
    # device_index 仅用于区分缓存键（CuTe 编译绑定当前设备），函数体内不使用
    _ = device_index
    # 构造内核对象（携带编译期配置）
    kernel = InterRankSyncKernel(
        R=R,
        meta_stride=meta_stride,
        num_threads=num_threads,
    )

    # 用占位指针（地址 0、按 16 字节对齐）与占位流做编译期签名——
    # cute.compile 只需要参数类型/形状，不需要真实地址
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    stream_arg = cuda.CUstream(0)

    # 执行 JIT 编译并返回可反复调用的可执行对象；注释标注各占位参数含义
    return cute.compile(
        kernel,
        i32_ptr,       # meta_buf
        i32_ptr,       # bar (grid barrier counter)
        Int32(0),      # rank
        Int32(0),      # barrier_off
        stream_arg,
    )


def launch_inter_rank_sync(ctx: dict) -> None:
    """Launch the pre-planning inter-rank sync kernel on the current stream."""
    # 中文：在当前流上启动"规划前"的跨 rank 同步内核。
    # meta_buf 必须是连续的 int32 张量（内核按 int32 槽位直接读写）
    assert ctx['meta_buf'].dtype == torch.int32 and ctx['meta_buf'].is_contiguous()

    # 从共享上下文取出 rank 数与 meta 槽位步长
    R = int(ctx['R'])
    meta_stride = int(ctx['meta_chunk_padded'])
    # 依 rank 数确定内核线程数
    num_threads = _num_threads_for_ranks(R)
    # 取 meta_buf 所在 GPU 的设备号（仅作编译缓存键）
    device_index = ctx['meta_buf'].device.index

    # 取（或触发编译）该配置对应的可执行内核
    compiled = _get_compiled(
        R,
        meta_stride,
        num_threads,
        device_index,
    )

    # 把 meta_buf 的数据地址包装成 CuTe 全局内存指针（假设 16 字节对齐）
    meta_ptr = make_ptr(
        Int32,
        ctx['meta_buf'].data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    # 同样包装 grid 级屏障计数器 buffer 的地址
    bar_ptr = make_ptr(
        Int32,
        ctx['grid_sync_bar'].data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    # 取 PyTorch 当前流并包装成 CUstream，使内核与后续 planning 内核同流串行
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # 真正 launch：传入本 rank 号与屏障槽位偏移，阻塞式对齐所有 EP rank
    compiled(
        meta_ptr,
        bar_ptr,
        Int32(int(ctx['rank'])),
        Int32(int(ctx['BARRIER_OFF'])),
        stream,
    )
