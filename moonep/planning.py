"""CuTe DSL implementation of the MoonEP planning kernel.

A single cooperative grid launch (num_sms blocks) runs Phase A/B/C/D in one
kernel and produces the canonical ``dst`` (with negative encoding for
duplicates), cu_seqlens, experts_to_copy, zero_fill_ranges and remote_stats.
Fresh dispatch then builds the plan-owned dedup structures from ``dst`` and
the ``src_info`` scratch; plan reuse reuses these structures directly.
Inter-block sync uses a software grid barrier (cooperative launch keeps all
blocks resident); cross-rank sync uses a system-scope atomic self-resetting
barrier on the NVLink meta_buf.
"""

import functools  # 提供 lru_cache：按配置缓存已编译的规划 kernel
from dataclasses import dataclass  # 声明式数据类：定义规划结果容器

import torch  # PyTorch：张量、流与设备接口
import cuda.bindings.driver as cuda  # CUDA driver 绑定：CUstream 等类型

import cutlass  # CuTe DSL 主入口
import cutlass.cute as cute  # CuTe 张量/布局原语与 kernel 构造接口
import cutlass.utils as utils  # CuTe 工具集（共享内存分配器等）
from cutlass import Int32, Int64, Uint32  # DSL 定宽整数类型
from cutlass._mlir.dialects import llvm  # MLIR LLVM 方言：内联 PTX 汇编需要
from cutlass.cutlass_dsl import T, dsl_user_op  # T：PTX 类型构造器；dsl_user_op：用户自定义 DSL 操作装饰器
from cutlass.cute.runtime import make_ptr  # 由裸地址构造运行期指针实参

from moonep._common import cp_async_bulk_g2s, cross_rank_barrier, grid_sync  # 公共原语：bulk G2S 拷贝、跨 rank 系统级屏障、软件 grid 屏障
from moonep.constants import KIDX_BITS  # top-k 索引位宽常量（去重编码上限校验用）


# ============================================================
# 规划结果容器：单个协同启动的规划 kernel 产出的全部输出张量
# ============================================================
@dataclass(frozen=True, slots=True)  # 不可变、带槽位的数据类
class MoonEPCommPlan:
    dst: torch.Tensor  # [N] int32 规范化目标槽位 dst = dest_rank*NvS + offv；重复项编码为 -raw_dst - 1
    experts_to_copy: torch.Tensor  # [R, epn] int32 每个目标 rank 需拷贝权重的远端专家 id（-1 表示空）
    zero_fill_ranges: torch.Tensor  # [2*epn, 2] int32 每个 VM 组的 (零填充起始, 零填充长度)
    remote_stats: torch.Tensor  # [2] int32 本 rank 的远端规划统计（远端专家总数、各来源触发计数）
    N: int  # 本 rank 的 top-k 平铺元素总数 S*K
    R: int  # EP 组内 rank 数
    E: int  # 全组路由专家总数
    NvS: int  # 每 rank 的 dispatch token 槽位数（含 padding）
    K: int  # 每 token 的 top-k

    # Dedup structures written by the dispatch builder and consumed by the
    # dispatch epilogue / combine prologue. dup_counts = [n_groups, n_dup_loffs];
    # only the compact prefix of dup_groups/dup_loffs is valid, and the ordering
    # is decided by the builder's atomicAdd — not guaranteed to be stable.
    # ↑ 中文补充：去重结构由 dispatch builder 写入、dispatch epilogue / combine prologue 消费；
    # dup_counts = [组数, 重复 loff 数]；仅前缀有效，顺序由 builder 的 atomicAdd 决定，不保证稳定。
    dup_groups: torch.Tensor  # [NvS, 3] int32 去重组记录三元组（目标槽位、源 rank/topk 信息、组内计数）
    dup_loffs: torch.Tensor  # [NvS] int32 重复项在目标 rank 上的局部槽位偏移列表
    dup_counts: torch.Tensor  # [2] int32 去重结构的实际写入计数 [组数, 重复项数]

    def __post_init__(self) -> None:  # 构造后校验各输出张量的 dtype/形状/连续性
        N = int(self.N)  # 本 rank top-k 平铺总数 S*K
        R = int(self.R)  # EP rank 数
        E = int(self.E)  # 全组专家总数
        epn = E // R  # 每 rank 本地专家数（= VM 组数）
        NvS = int(self.NvS)  # 每 rank dispatch 槽位数
        assert self.dst.dtype == torch.int32 and self.dst.is_contiguous()  # dst 必须为连续 int32
        assert self.dst.numel() == N  # dst 长度恰为 N=S*K
        assert self.experts_to_copy.dtype == torch.int32 and self.experts_to_copy.is_contiguous()  # 拷贝表必须为连续 int32
        assert tuple(self.experts_to_copy.shape) == (R, epn)  # 拷贝表形状 [R, epn]
        assert self.remote_stats.dtype == torch.int32 and self.remote_stats.is_contiguous()  # 统计必须为连续 int32
        assert tuple(self.remote_stats.shape) == (2,)  # 统计形状 [2]
        assert self.zero_fill_ranges.dtype == torch.int32 and self.zero_fill_ranges.is_contiguous()  # 零填充区间必须为连续 int32
        assert tuple(self.zero_fill_ranges.shape) == (2 * epn, 2)  # 零填充区间形状 [2*epn, 2]
        assert self.dup_groups.dtype == torch.int32 and self.dup_groups.is_contiguous()  # 去重组必须为连续 int32
        assert tuple(self.dup_groups.shape) == (NvS, 3)  # 去重组容量 [NvS, 3]
        assert self.dup_loffs.dtype == torch.int32 and self.dup_loffs.is_contiguous()  # 重复 loff 必须为连续 int32
        assert tuple(self.dup_loffs.shape) == (NvS,)  # 重复 loff 容量 [NvS]
        assert self.dup_counts.dtype == torch.int32 and self.dup_counts.is_contiguous()  # 去重计数必须为连续 int32
        assert tuple(self.dup_counts.shape) == (2,)  # 去重计数形状 [2]

    def clone(self) -> "MoonEPCommPlan":  # 深拷贝全部设备张量，共享标量配置，得到一份独立计划
        return type(self)(  # 以同类型逐字段构造
            dst=self.dst.clone(),  # 克隆 dst
            experts_to_copy=self.experts_to_copy.clone(),  # 克隆专家拷贝表
            zero_fill_ranges=self.zero_fill_ranges.clone(),  # 克隆零填充区间
            remote_stats=self.remote_stats.clone(),  # 克隆远端统计
            dup_groups=self.dup_groups.clone(),  # 克隆去重组
            dup_loffs=self.dup_loffs.clone(),  # 克隆重复 loff
            dup_counts=self.dup_counts.clone(),  # 克隆去重计数
            N=self.N,  # 保留配置 N
            R=self.R,  # 保留配置 R
            E=self.E,  # 保留配置 E
            NvS=self.NvS,  # 保留配置 NvS
            K=self.K,  # 保留配置 K
        )


# ============================================================
# Compile-time constants（编译期常量）
# ============================================================
BLOCK_SIZE_P2 = 2048  # run_c1 每轮处理的 token 分片大小（2 的幂）
BLOCK_DIM_P2 = 512  # 规划 kernel 每 CTA 的线程数（= 16 个 warp）
ITEMS_PER_THREAD_P2 = BLOCK_SIZE_P2 // BLOCK_DIM_P2  # 4  # 每线程在分片内持有的元素数


def ceil_div(x, y):  # 整数上取整除法（编译期 tiling 用）
    """Return ceil(x / y), for compile-time integer tiling."""
    return (x + y - 1) // y  # 经典 ceil(x/y) 公式


def align_up(x, alignment):  # 向上对齐到 alignment 的倍数
    """Round x up to a multiple of alignment."""
    return ceil_div(x, alignment) * alignment  # 先上取整再乘回 alignment


def ceil_pow2(x):  # 返回 >= x 的最小 2 的幂（共享内存/布局 padding 用）
    """Return the smallest power of 2 >= x, for shared-memory/layout padding."""
    return 1 << max(x - 1, 0).bit_length()  # (x-1) 的位长即目标幂次


def log2_r(R):  # 固定次数 rank 二分查找所需的循环步数
    """Return max(ceil(log2(R + 1)), 1), for fixed-trip-count rank binary search."""
    return max(R.bit_length(), 1)  # R 的位长即 ceil(log2(R+1))，且至少为 1


# ============================================================
# Low-level inline-PTX helpers（底层内联 PTX 辅助函数）
# ============================================================


@dsl_user_op  # 声明为用户自定义 DSL 内联汇编操作
def match_any_b32(val, *, loc=None, ip=None) -> Uint32:  # warp 内匹配任意：返回与 val 相同值的活跃 lane 掩码
    """__match_any_sync(0xffffffff, val) -> peer mask."""
    return Uint32(llvm.inline_asm(  # 内联 PTX：match.any.sync.b32
        T.i32(), [Uint32(val).ir_value(loc=loc, ip=ip),  # 输入值 val
                  Uint32(0xFFFFFFFF).ir_value(loc=loc, ip=ip)],  # 全 warp 活跃掩码 0xffffffff
        "match.any.sync.b32 $0, $1, $2;", "=r,r,r",  # PTX 模板与约束：$0 输出、$1/$2 输入
        has_side_effects=True, is_align_stack=False,  # 汇编属性
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))  # AT&T 语法


@dsl_user_op  # 声明为用户自定义 DSL 内联汇编操作
def st_global_v4_s32(addr_i64, x, y, z, w, *, loc=None, ip=None) -> None:  # 一次 128bit 向量化全局存储
    """st.global.v4.s32 [addr], {x,y,z,w}: one 128bit store, NVLink transactions /4."""
    llvm.inline_asm(  # 内联 PTX：v4.s32 一次写 4 个 int32
        None,  # 无返回值
        [addr_i64,  # 64bit 目标地址
         Int32(x).ir_value(loc=loc, ip=ip), Int32(y).ir_value(loc=loc, ip=ip),  # 待写的第 1、2 个 int32
         Int32(z).ir_value(loc=loc, ip=ip), Int32(w).ir_value(loc=loc, ip=ip)],  # 待写的第 3、4 个 int32
        "st.global.v4.s32 [$0], {$1, $2, $3, $4};",  # PTX 向量存储指令
        "l,r,r,r,r", has_side_effects=True, is_align_stack=False,  # 约束：64bit 地址 + 4 个寄存器
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)  # AT&T 语法


# multimem.st：多播存储，一条指令把同一 16B 数据写入 NVLink multicast 缓冲（全 rank 同时可见）
@dsl_user_op  # 声明为用户自定义 DSL 内联汇编操作
def multimem_st_v4(addr_i64, x, y, z, w, *, loc=None, ip=None) -> None:  # 向 multicast 地址做 v4 多播写
    llvm.inline_asm(  # 内联 PTX
        None,  # 无返回值
        [addr_i64,  # multicast 目标地址（64bit）
         Uint32(x).ir_value(loc=loc, ip=ip), Uint32(y).ir_value(loc=loc, ip=ip),  # 多播数据的 4 个 int32 分量
         Uint32(z).ir_value(loc=loc, ip=ip), Uint32(w).ir_value(loc=loc, ip=ip)],  # 共 16 字节
        "{\n\t.reg .u64 g;\n\t cvta.to.global.u64 g, $0;\n\t"  # 临时寄存器：把通用地址转成 global 空间地址
        "multimem.st.relaxed.sys.global.v4.f32 [g], {$1, $2, $3, $4};\n\t}",  # relaxed.sys 语义的多播向量存储
        "l,r,r,r,r", has_side_effects=True, is_align_stack=False,  # 约束与汇编属性
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)  # AT&T 语法


@cute.jit  # 编译为 CuTe JIT 设备函数
def warp_inclusive_scan(v, lane):  # warp 级 Hillis-Steele 含入前缀和（shuffle 实现）
    offset = 1  # 当前轮的比较跨距
    for scan_step in cutlass.range_constexpr(5):  # log2(32)=5 轮覆盖整个 warp
        y = cute.arch.shuffle_sync(v, lane - offset)  # 从 lane-offset 同步取部分和
        if lane >= offset:  # 只有跨距之上的 lane 参与累加
            v += y  # 累加左侧隔 offset 的部分和
        offset <<= 1  # 跨距翻倍
    return v  # 返回 warp 内含入前缀和（所有 lane 结果一致）


@cute.jit  # 编译为 CuTe JIT 设备函数
def warp_exclusive_scan_e(s_hist, E: cutlass.Constexpr, tid):  # 把 smem 中 E 个计数就地转换为排他前缀和
    CHUNK = cutlass.const_expr(ceil_div(E, 32))  # 每个 lane 负责的分段长度
    if tid < 32:  # 只用第一个 warp 执行 scan
        lane = tid  # 本线程的 lane 号
        csum = 0  # lane 内串行累计
        for j in cutlass.range_constexpr(CHUNK):  # 第一步：lane 内串行前缀
            idx = lane * CHUNK + j  # 该 lane 第 j 个元素的全局下标
            v = 0  # 读出的原值（越界记 0）
            if idx < E:  # 越界保护
                v = s_hist[idx]; s_hist[idx] = csum  # 读旧值并写入 lane 内排他前缀
            csum += v  # 累加到 lane 局部总和
        x = warp_inclusive_scan(csum, lane)  # 对各 lane 的局部总和做 warp 含入 scan
        offset = x - csum  # 减去自身即得跨 lane 的排他偏移
        for j in cutlass.range_constexpr(CHUNK):  # 第二步：把跨 lane 偏移加回 smem
            idx = lane * CHUNK + j  # 元素下标
            if idx < E: s_hist[idx] += offset  # 就地得到全局排他前缀和
    cute.arch.barrier()  # CTA 级屏障：scan 结果对所有线程可见


@cute.jit  # 编译为 CuTe JIT 设备函数
def elem_ptr(tensor, coord):  # 计算张量在 coord 处的元素地址（供原子操作取指针）
    return tensor.iterator + tensor.layout(coord)  # 基址迭代器 + 布局映射出的偏移


# warp argmax/argmin: instead of packing value/idx into one int32, run two
# redux passes over value and tie-break idx separately. value takes max/min;
# on ties idx takes the min (sentinel s32_max) or the max (sentinel -1).
# The returned (best_val, best_idx) is already broadcast across the warp.
# idx is >= 0 by convention, so the sentinel never wins.
# ↑ 中文补充：warp 求极值不把 value/idx 打包进一个 int32，而是两次 redux——
# 先对 value 取 max/min，平局时再对 idx 取 min（哨兵 s32_max）或 max（哨兵 -1）；
# 返回的 (best_val, best_idx) 已在 warp 内广播；约定 idx>=0，哨兵不会胜出。
@cute.jit  # 编译为 CuTe JIT 设备函数
def warp_argmax_min_idx(v, i):  # warp 内取最大值，平局取最小下标
    m = cute.arch.warp_redux_sync(v, "max")  # 归约求最大值（广播到全 warp）
    j = cute.arch.warp_redux_sync(i if v == m else 2147483647, "min")  # 仅取到最大值的 lane 竞逐最小下标
    return m, j  # 返回 (最大值, 其最小下标)

@cute.jit  # 编译为 CuTe JIT 设备函数
def warp_argmin_min_idx(v, i):  # warp 内取最小值，平局取最小下标
    m = cute.arch.warp_redux_sync(v, "min")  # 归约求最小值
    j = cute.arch.warp_redux_sync(i if v == m else 2147483647, "min")  # 平局时取最小下标
    return m, j  # 返回 (最小值, 其最小下标)

@cute.jit  # 编译为 CuTe JIT 设备函数
def warp_argmin_max_idx(v, i):  # warp 内取最小值，平局取最大下标
    m = cute.arch.warp_redux_sync(v, "min")  # 归约求最小值
    j = cute.arch.warp_redux_sync(i if v == m else -1, "max")  # 平局时取最大下标（哨兵 -1 不会干扰）
    return m, j  # 返回 (最小值, 其最大下标)

# Scan version: one warp strides over s[0:n] taking the extremum; n is
# arbitrary (no longer limited to 32/64).
# ↑ 中文补充：扫描版——一个 warp 以 32 为步长扫过 s[0:n] 求极值；n 任意，
# 不再受 32/64 限制。
@cute.jit  # 编译为 CuTe JIT 设备函数
def warp_scan_argmax_min_idx(s, n, lane):  # warp 扫描版：取最大值 + 最小下标
    bv = -2147483648; bi = 2147483647  # lane 局部最优值/下标初始化为哨兵
    for k in cutlass.range(lane, n, 32):  # 从 lane 号起、步长 32 的跨步扫描
        x = s[k]  # 读取共享内存元素
        if x > bv: bv = x; bi = k  # 严格大于才更新（保证平局取最小下标）
    return warp_argmax_min_idx(bv, bi)  # warp 归约出全局最优

@cute.jit  # 编译为 CuTe JIT 设备函数
def warp_scan_argmin_min_idx(s, n, lane):  # warp 扫描版：取最小值 + 最小下标
    bv = 2147483647; bi = 2147483647  # 局部最优哨兵初始化
    for k in cutlass.range(lane, n, 32):  # 跨步扫描
        x = s[k]  # 读取共享内存元素
        if x < bv: bv = x; bi = k  # 严格小于才更新（平局取最小下标）
    return warp_argmin_min_idx(bv, bi)  # warp 归约出全局最优


# Pure-register scan version: one warp splits [0:N) into CHUNK=ceil(N/32)
# segments, lane holds reg[j]=s[lane+j*32]; the whole segment stays in
# registers (CHUNK constexpr, fully unrolled), never touching smem. N is
# arbitrary, bounded by register count rather than 32/64.
# ↑ 中文补充：纯寄存器版——一个 warp 把 [0:N) 分成 CHUNK=ceil(N/32) 段，
# lane 持有 reg[j]=s[lane+j*32]；整段常驻寄存器（CHUNK 为编译期常量、完全展开），
# 不经过共享内存；N 只受寄存器数量约束。
@cute.jit  # 编译为 CuTe JIT 设备函数
def reg_scan_argmax_min_idx(reg, N: cutlass.Constexpr, lane):  # 寄存器扫描版：取最大值 + 最小下标
    bv = -2147483648; bi = 2147483647  # 局部最优哨兵初始化
    for j in cutlass.range_constexpr(ceil_div(N, 32)):  # 编译期展开各寄存器段
        k = lane + j * 32  # 该寄存器槽对应的元素下标
        if k < N and reg[j] > bv: bv = reg[j]; bi = k  # 严格大于才更新
    return warp_argmax_min_idx(bv, bi)  # warp 归约出全局最优

@cute.jit  # 编译为 CuTe JIT 设备函数
def reg_scan_argmin_min_idx(reg, N: cutlass.Constexpr, lane):  # 寄存器扫描版：取最小值 + 最小下标
    bv = 2147483647; bi = 2147483647  # 局部最优哨兵初始化
    for j in cutlass.range_constexpr(ceil_div(N, 32)):  # 编译期展开各寄存器段
        k = lane + j * 32  # 该寄存器槽对应的元素下标
        if k < N and reg[j] < bv: bv = reg[j]; bi = k  # 严格小于才更新
    return warp_argmin_min_idx(bv, bi)  # warp 归约出全局最优


@cute.jit  # 编译为 CuTe JIT 设备函数
def copy_v4_remote(dst, dst_off, src, n: cutlass.Constexpr,
                   pid, tid, nth, num_sms):  # 远端可见的向量化拷贝：src[0:n] -> dst[dst_off:dst_off+n]
    # src[n] -> dst[dst_off:]: scalar head pads to 16B, int4 body does one
    # 128bit store (transactions /4), scalar tail.
    # dst_off may be arbitrarily aligned (dst.iterator is 16B aligned); no src
    # padding needed.
    # ↑ 中文补充：标量头部补齐到 16B、主体用 v4 一次 128bit 存储（NVLink 事务数 /4）、标量尾部收尾；
    # dst_off 可任意对齐（dst 基址本身 16B 对齐），src 无需 padding。
    head = (-dst_off) & 3                      # make dst_off+head ≡0 mod4 so v4 addresses are 16B aligned  # 补齐量：使 dst_off+head 为 4 的倍数，v4 地址即 16B 对齐
    nv = (n - head) >> 2                       # number of vectorizable 4-tuples  # 可向量化的 4 元组个数

    for h in cutlass.range(pid * nth + tid, head, num_sms * nth):  # grid-stride 遍历标量头部
        dst[dst_off + h] = src[h]  # 逐元素补齐到对齐边界

    for i in cutlass.range(pid * nth + tid, nv, num_sms * nth):  # grid-stride 遍历向量主体
        j = head + i * 4  # 该 4 元组在 src 中的起始下标
        st_global_v4_s32((dst.iterator + (dst_off + j)).toint().ir_value(),  # 一次 128bit 写入 4 个 int32
                         src[j], src[j + 1], src[j + 2], src[j + 3])  # 连续 4 个源元素

    for off in cutlass.range(head + nv * 4 + pid * nth + tid, n, num_sms * nth):  # grid-stride 遍历标量尾部
        dst[dst_off + off] = src[off]  # 尾部逐元素拷贝


@cute.jit  # 编译为 CuTe JIT 设备函数
def _pd_cta_slice(total: cutlass.Constexpr, pid, num_sms, tile: cutlass.Constexpr):  # 把 total 个元素按 CTA 切成对齐的连续段
    # Phase D outputs are small: slice contiguous segments per CTA to keep
    # gmem writeback coalesced; the copy_begin clamp only handles empty CTAs
    # whose pid falls past the valid segments, so copy_count never goes negative.
    # ↑ 中文补充：Phase D 输出很小，按 CTA 切连续段以保持 gmem 写回的访存合并；
    # copy_begin 的截断只为处理落在有效段之后的空 CTA，保证 copy_count 不为负。
    per_cta = cute.round_up(cute.ceil_div(total, num_sms), tile)  # 每 CTA 段长（按 tile 对齐）
    begin = pid * per_cta  # 本 CTA 段起始（可能越过 total）
    end = cutlass.min(begin + per_cta, total)  # 本 CTA 段结束（截断到 total）
    copy_begin = cutlass.min(begin, total)  # 拷贝起始同样截断
    copy_count = end - copy_begin  # 实际拷贝元素数（空 CTA 为 0）
    return begin, end, copy_begin, copy_count  # 返回切分四元组


@cute.jit  # 编译为 CuTe JIT 设备函数
def _pd_stage_bias(src_begin):  # 计算 16B 包络内逻辑起始的偏移（int32 单位）
    # src_begin is in int32 units; its low 2 bits are the offset of the logical
    # slice inside the 16B envelope, used to realign the SMEM stage to the
    # logical start on Phase D writeback.
    # ↑ 中文补充：src_begin 的低 2 位是逻辑切片在 16B 包络内的偏移，
    # Phase D 写回时用它把 SMEM 暂存重新对齐到逻辑起点。
    return src_begin & 3  # 低 2 位即包络内偏移


@cute.jit  # 编译为 CuTe JIT 设备函数
def _pd_aligned_ints(src_begin, logical_count):  # 计算对齐包络需要搬运的 int32 总数
    # cp.async.bulk requires 16B-aligned src/dst and a size that is a multiple
    # of 16B. Copy from floor(src/16)*16 here; the extra ints only land in the
    # staging padding.
    # ↑ 中文补充：cp.async.bulk 要求 src/dst 16B 对齐且大小为 16B 的倍数；
    # 这里从 floor(src/16)*16 开始拷，多出的整数只落在暂存区的 padding 里。
    aligned_ints = 0  # 空段不需要搬运
    if logical_count > 0:  # 仅对非空段计算
        aligned_ints = cute.round_up(_pd_stage_bias(src_begin) + logical_count, 4)  # 向上补到 4 个 int32（16B）
    return aligned_ints  # 返回包络内的 int32 数


@cute.jit  # 编译为 CuTe JIT 设备函数
def _pd_issue_g2s(meta, smem_stage, src_begin, logical_count, mbar):  # 发起 Phase D 的 bulk G2S 暂存拷贝
    if logical_count > 0:  # 空段直接跳过
        # Bulk-copy from the 16B-aligned gmem address into the 16B-aligned
        # SMEM stage. The extra head/tail ints stay in the stage padding and
        # are not part of the final writeback.
        # ↑ 中文补充：从 16B 对齐的 gmem 地址整块拷入 16B 对齐的 SMEM 暂存区；
        # 多出的头/尾整数留在 padding，不参与最终写回。
        src_aligned = src_begin - _pd_stage_bias(src_begin)  # 源地址向下取到 16B 对齐
        aligned_ints = _pd_aligned_ints(src_begin, logical_count)  # 包络内的 int32 数
        cp_async_bulk_g2s(  # 发起异步 bulk 拷贝
            smem_stage.iterator.toint().ir_value(),  # SMEM 目的地址
            (meta.iterator + src_aligned).toint().ir_value(),  # gmem 源地址（已对齐）
            Int32(aligned_ints * 4).ir_value(),  # 拷贝字节数（int32 数 * 4）
            mbar.toint().ir_value(),  # 完成通知所用的 mbarrier
        )


class PlanningKernel:  # 规划 kernel 封装类：持有编译期配置并负责发射
    def __init__(self, R, E, S, K, NvS_capacity, NvS, num_vblocks, meta_stride,
                 TPE_OFF, PLAN_OFF, BARRIER_OFF, TOPK0_OFF, ORDER_OFF, ORDER0_OFF,
                 token_padding, num_sms):  # 保存全部编译期常量
        self.R, self.E, self.S, self.K = R, E, S, K  # EP 维度配置：rank 数/专家总数/每 rank token 数/top-k
        self.N = self.S * self.K  # 本 rank 的 top-k 平铺总数 S*K
        self.NvS_capacity, self.NvS, self.num_vblocks = NvS_capacity, NvS, num_vblocks  # 槽位硬容量/逻辑槽位数/直方图分块数
        self.meta_stride = meta_stride  # meta_buf 中每个 rank 的 chunk 步长
        self.TPE_OFF, self.PLAN_OFF, self.BARRIER_OFF = TPE_OFF, PLAN_OFF, BARRIER_OFF  # meta 内偏移：tpe gather/规划输出/跨 rank 屏障
        self.TOPK0_OFF, self.ORDER_OFF, self.ORDER0_OFF = TOPK0_OFF, ORDER_OFF, ORDER0_OFF  # meta 内偏移：topk0 暂存/c1 order/rank0 order 暂存
        self.token_padding, self.num_sms = token_padding, num_sms  # VM 组 padding 粒度与 grid 大小（SM 数）

    @cute.jit
    def __call__(self, tpe, topk, meta, mc, dst, cu_seqlens,
                 experts_to_copy, zero_fill, remote_stats, alloc, group_tokens, z,
                 local_hist, bar,
                 rank: Int32, stream: cuda.CUstream):  # 主机侧入口：包装张量并以协同启动方式发射 kernel
        R = cutlass.const_expr(self.R)  # EP rank 数（编译期常量）
        ms = cutlass.const_expr(self.meta_stride)  # 每 rank 的 meta chunk 步长
        N = cutlass.const_expr(self.N)  # 本 rank top-k 平铺总数 S*K
        epn = cutlass.const_expr(self.E // self.R)  # 每 rank 本地专家数
        num_sms = cutlass.const_expr(self.num_sms)  # grid 中的 CTA 数（= SM 数）
        meta_t = cute.make_tensor(meta, cute.make_layout((R * ms,)))  # 全体 rank 的 meta 缓冲视图
        mc_t = cute.make_tensor(mc, cute.make_layout((R * ms,)))  # multicast meta 缓冲视图
        tpe_t = cute.make_tensor(tpe, cute.make_layout((self.E,)))  # 每专家 token 数输入视图
        topk_t = cute.make_tensor(topk, cute.make_layout((N,)))  # top-k 专家 id 平铺视图
        dst_t = cute.make_tensor(dst, cute.make_layout((N,)))  # dst 输出视图
        cu_t = cute.make_tensor(cu_seqlens, cute.make_layout((2 * epn,)))  # cu_seqlens 输出视图
        etc_t = cute.make_tensor(experts_to_copy, cute.make_layout((self.R * epn,)))  # experts_to_copy 输出视图
        zfr_t = cute.make_tensor(zero_fill, cute.make_layout((4 * epn,)))  # 零填充区间输出视图（2*epn x 2 展平）
        stats_t = cute.make_tensor(remote_stats, cute.make_layout((2,)))  # remote_stats 输出视图
        alloc_t = cute.make_tensor(alloc, cute.make_layout((R * self.E,)))  # 本地 alloc 临时缓冲视图
        gt_t = cute.make_tensor(group_tokens, cute.make_layout((R,)))  # 各目标组 token 计数临时缓冲视图
        z_t = cute.make_tensor(z, cute.make_layout((R * R,)))  # 跨组转移矩阵临时缓冲视图
        lh_t = cute.make_tensor(local_hist, cute.make_layout((self.num_vblocks * self.E,)))  # 本地 vblock 直方图临时缓冲视图
        bar_t = cute.make_tensor(bar, cute.make_layout((1,)))  # 软件 grid 屏障计数器视图

        self.kernel(tpe_t, topk_t, meta_t, mc_t, dst_t, cu_t, etc_t,  # 发射规划 kernel
                    zfr_t, stats_t, alloc_t, gt_t, z_t, lh_t, bar_t,  # 传入全部张量与 rank
                    rank).launch(  # 并指定发射参数
            grid=(num_sms, 1, 1), block=(BLOCK_DIM_P2, 1, 1),  # grid=num_sms 个 CTA，每 CTA 512 线程
            stream=stream, cooperative=True)  # 协同启动：所有 CTA 常驻以支持软件 grid 屏障

    # =========================================================
    # run_c1: 1a histogram / 1b vblock prefix / expoff / passA scatter
    # ↑ 中文补充：c1 阶段——直方图统计、vblock 前缀和、专家排他偏移与 passA 稳定重排
    # =========================================================
    @cute.jit
    def run_c1(self, topk_src, order_dst, tpe_src, local_hist, s_hist, s_bp, s_wcount,
               bar_ptr, num_sms, pid, tid):  # 对给定 topk/tpe 计算稳定重排顺序并写入 order_dst
        R = cutlass.const_expr(self.R)  # EP rank 数
        E = cutlass.const_expr(self.E)  # 专家总数
        NUM_WARPS = cutlass.const_expr(BLOCK_DIM_P2 // 32)  # 每 CTA warp 数
        WST = cutlass.const_expr(NUM_WARPS + 1)  # warp 计数表的第二维步长（多 1 列作边界）
        IPT = cutlass.const_expr(ITEMS_PER_THREAD_P2)  # 每线程持有元素数
        N = cutlass.const_expr(self.N)  # top-k 平铺总数 S*K
        num_vblocks = cutlass.const_expr(self.num_vblocks)  # 直方图分块数
        num_threads = BLOCK_DIM_P2  # 每 CTA 线程数
        warp = tid >> 5  # 本线程所在 warp 编号
        lane = tid & 31  # 本线程在 warp 内的 lane 号

        topk_in = cute.make_tensor(topk_src.iterator, cute.make_layout((N,)))  # 输入 topk 视图
        order_out = cute.make_tensor(order_dst.iterator, cute.make_layout((N,)))  # 输出 order 视图
        tpe_counts = cute.make_tensor(tpe_src.iterator, cute.make_layout((E,)))  # 每专家 token 数视图
        vblocks_histogram = cute.make_tensor(  # gmem 中 [num_vblocks, E] 的直方图视图
            local_hist.iterator,
            cute.make_layout((num_vblocks, E), stride=(E, 1)),
        )
        s_histogram = cute.make_tensor(s_hist.iterator, cute.make_layout((E,)))  # smem：专家直方图
        s_block_prefix = cute.make_tensor(s_bp.iterator, cute.make_layout((E,)))  # smem：block 内专家前缀
        s_warp_counts = cute.make_tensor(  # smem：[E+1, WST] 的每 warp 计数表
            s_wcount.iterator,
            cute.make_layout((E + 1, WST), stride=(WST, 1)),
        )

        # 1a —— 统计每个 vblock 的专家直方图
        for vb in cutlass.range(pid, num_vblocks, num_sms):  # 各 CTA 轮流处理 vblock
            for e in cutlass.range(tid, E, num_threads):  # 清零 smem 直方图
                s_histogram[e] = 0

            cute.arch.barrier()  # CTA 屏障：清零完成

            chunk = vb * BLOCK_SIZE_P2  # 该 vblock 覆盖的 topk 区间起点
            for p in cutlass.range(tid, BLOCK_SIZE_P2, num_threads):  # 线程遍历分片内元素
                off = chunk + p  # 全局 topk 下标
                if off < N:  # 尾块越界保护
                    expert = topk_in[off]  # 读出专家 id
                    cute.arch.atomic_add(elem_ptr(s_histogram, expert), 1, scope="cta")  # CTA 内原子累加直方图

            cute.arch.barrier()  # CTA 屏障：直方图统计完成

            for e in cutlass.range(tid, E, num_threads):  # 把本 vblock 的直方图写回 gmem
                vblocks_histogram[vb, e] = s_histogram[e]
            cute.arch.barrier()  # CTA 屏障：避免下一轮清零竞态

        grid_sync(bar_ptr, num_sms, tid)  # 软件 grid 屏障：所有 vblock 直方图就绪
        E_SEG = 32  # 专家维度按 32 对齐切分
        seg_raw = cute.ceil_div(E, num_sms)  # 每 CTA 原始专家数
        experts_per_block = cute.round_up(seg_raw, E_SEG)  # 对齐后的每 CTA 专家数
        e_lo = pid * experts_per_block  # 本 CTA 负责的专家区间起点
        e_hi = cutlass.min(e_lo + experts_per_block, E)  # 终点（截断到 E）
        for e in cutlass.range(e_lo + tid, e_hi, num_threads):  # 线程并行处理各自专家列
            cumsum = 0  # vblock 维度的累计值
            for vb in cutlass.range_constexpr(num_vblocks):  # 就地把直方图改为 vblock 排他前缀
                v = vblocks_histogram[vb, e]  # 读原计数
                vblocks_histogram[vb, e] = cumsum  # 写排他前缀
                cumsum += v  # 累加

        grid_sync(bar_ptr, num_sms, tid)  # grid 屏障：前缀和就绪
        for e in cutlass.range(tid, E, num_threads):  # 用全局 tpe 计数初始化 smem 直方图
            s_histogram[e] = tpe_counts[e]

        cute.arch.barrier()  # CTA 屏障
        warp_exclusive_scan_e(s_histogram, E, tid)  # 就地求专家维度的全局排他前缀（= 该专家之前所有 token 数）
        lanes_lt = (Uint32(1) << lane) - Uint32(1)  # 掩码：低于本 lane 的那些位
        for vb in cutlass.range(pid, num_vblocks, num_sms):  # 各 CTA 轮流做 passA 重排
            chunk = vb * BLOCK_SIZE_P2  # 本 vblock 的 topk 起点
            my_e = []; my_p = []  # 每线程寄存器缓存：专家 id 与分片内位置
            for i in cutlass.range_constexpr(IPT):  # 预取本线程的 IPT 个元素
                p = warp * (32 * IPT) + i * 32 + lane  # 线程到分片位置的映射（warp 内跨步）
                off = chunk + p  # 全局 topk 下标
                my_p.append(p)  # 记录分片内位置
                ev = E  # 越界哨兵（不参与写回）
                if off < N:  # 尾块保护
                    ev = topk_in[off]  # 读出专家 id
                my_e.append(ev)  # 记录专家 id

            for idx in cutlass.range(tid, (E + 1) * WST, num_threads):  # 清零 warp 计数表
                expert_idx = idx // WST  # 表行：专家
                warp_slot = idx - expert_idx * WST  # 表列：warp 槽位
                s_warp_counts[expert_idx, warp_slot] = 0

            cute.arch.barrier()  # CTA 屏障：清零完成

            for e in cutlass.range(tid, E, num_threads):  # 取本 vblock 的 block 内专家前缀
                s_block_prefix[e] = vblocks_histogram[vb, e]

            ww = []  # 每元素在 warp 内的次序（相对其他 warp 的基址）
            for i in cutlass.range_constexpr(IPT):  # 逐元素 warp 内去重计数
                peers = match_any_b32(my_e[i])  # 找出 warp 内同专家的 lane 集合
                cell = elem_ptr(s_warp_counts, (my_e[i], warp))  # 本 (专家, warp) 槽位地址
                base = cute.arch.load(cell, Int32)  # 读该槽位当前基址
                ww.append(base + Int32(cute.arch.popc(peers & lanes_lt)))  # 基址 + 低于本 lane 的同专家数量
                cute.arch.sync_warp()  # warp 同步
                if (peers & lanes_lt) == Uint32(0):  # 组内最低 lane 负责把整组人数累加进槽位
                    cute.arch.store(cell, base + Int32(cute.arch.popc(peers)))
                cute.arch.sync_warp()  # warp 同步
            cute.arch.barrier()  # CTA 屏障：warp 计数完成

            for e in cutlass.range(tid, E, num_threads):  # 对各专家的 warp 计数做排他前缀
                cumsum = 0
                for w in cutlass.range_constexpr(NUM_WARPS):  # 逐 warp 累计
                    c = s_warp_counts[e, w]  # 读该 warp 的人数
                    s_warp_counts[e, w] = cumsum  # 写排他前缀
                    cumsum += c  # 累加

            cute.arch.barrier()  # CTA 屏障

            for i in cutlass.range_constexpr(IPT):  # passA：把每个元素写到目标 order 位置
                ei = my_e[i]  # 该元素专家 id
                if ei < E:  # 越界哨兵不写
                    within = s_warp_counts[ei, warp] + ww[i]  # block 内目标偏移（warp 前缀 + warp 内次序）
                    sp = s_histogram[ei] + s_block_prefix[ei] + within  # 全局目标 = 全局专家排他前缀 + block 前缀 + block 内偏移
                    order_out[sp] = chunk + my_p[i]  # 写入原 topk 下标
            cute.arch.barrier()  # CTA 屏障：进入下一 vblock

        grid_sync(bar_ptr, num_sms, tid)  # grid 屏障：run_c1 的 order 全部就绪

    @cute.kernel
    def kernel(self, tpe, topk, meta, mc, dst, cu_seqlens,
               experts_to_copy, zfr, remote_stats, alloc, group_tokens, z, lh, bar,
               rank: Int32):  # 规划 kernel 主体：Phase A（tpe 广播与 rank0 规划）→ Phase C（order 重排）→ C2（dst 计算）→ Phase D（结果写回）
        R = cutlass.const_expr(self.R)  # EP rank 数（编译期常量）
        E = cutlass.const_expr(self.E)  # 专家总数
        S = cutlass.const_expr(self.S)  # 每 rank 输入 token 数
        K = cutlass.const_expr(self.K)  # 每 token top-k
        epn = cutlass.const_expr(E // R)  # 每 rank 本地专家数
        LOG2_R = cutlass.const_expr(log2_r(R))  # rank 二分查找的固定循环次数
        EB_PAD = cutlass.const_expr(ceil_pow2(2 * epn))  # expert-offsets 区块向上补到 2 的幂
        IPT_EB = cutlass.const_expr(ceil_div(EB_PAD, BLOCK_DIM_P2))  # expert-offsets 每线程元素数
        ms = cutlass.const_expr(self.meta_stride)  # 每 rank 的 meta chunk 步长
        N = cutlass.const_expr(self.N)  # top-k 平铺总数 S*K
        NvS = cutlass.const_expr(self.NvS)  # 每 rank dispatch 槽位数
        CAP = cutlass.const_expr(self.NvS_capacity)  # 槽位硬容量（S*K，不含 padding）
        tp = cutlass.const_expr(self.token_padding)  # VM 组 token padding 粒度
        num_sms = cutlass.const_expr(self.num_sms)  # grid 中 CTA 数
        TPE_OFF = cutlass.const_expr(self.TPE_OFF)  # meta 内 tpe gather 区偏移
        PLAN_OFF = cutlass.const_expr(self.PLAN_OFF)  # meta 内规划输出区偏移
        BARRIER_OFF = cutlass.const_expr(self.BARRIER_OFF)  # meta 内跨 rank 屏障区偏移
        TOPK0_OFF = cutlass.const_expr(self.TOPK0_OFF)  # meta 内 topk0 暂存区偏移
        ORDER_OFF = cutlass.const_expr(self.ORDER_OFF)  # meta 内 order 区偏移
        ORDER0_OFF = cutlass.const_expr(self.ORDER0_OFF)  # meta 内 rank0 order 暂存区偏移
        BARRIER_SLOTS = 3  # 每 rank 的跨 rank 屏障占用 3 个 int32
        SRC_INFO_OFF = cutlass.const_expr(self.BARRIER_OFF + BARRIER_SLOTS)  # 屏障区之后的 src_info 暂存区偏移
        ALLOC_SUB = 0  # 规划输出区内 alloc 前缀和的子偏移
        TPE_SUB = E * R  # 规划输出区内 tpe 前缀和的子偏移
        EOFF_SUB = 2 * E * R  # 规划输出区内 expert-offset 区的子偏移
        CU_SUB = 3 * E * R  # 规划输出区内 cu_seqlens 区的子偏移
        ZFR_SUB = CU_SUB + R * (2 * epn)  # 零填充区间区的子偏移
        ETC_SUB = ZFR_SUB + 2 * R * (2 * epn)  # experts_to_copy 区的子偏移
        STATS_SUB = ETC_SUB + R * epn  # remote_stats 区的子偏移
        PB = PLAN_OFF  # 简写：规划输出区基址
        num_threads = BLOCK_DIM_P2  # 每 CTA 线程数
        NUM_WARPS = cutlass.const_expr(BLOCK_DIM_P2 // 32)  # 每 CTA warp 数
        S1_TILE = 32  # tpe 前缀和阶段每轮处理的专家列数
        S1_COLS = cutlass.const_expr(min(  # 每 CTA 负责的专家列数（按 32 对齐并受线程数上限约束）
            align_up((E + num_sms - 1) // num_sms, S1_TILE),
            BLOCK_DIM_P2,
        ))
        pid = cute.arch.block_idx()[0]  # 当前 CTA 编号
        tid = cute.arch.thread_idx()[0]  # 当前线程编号

        smem = utils.SmemAllocator()  # 共享内存分配器
        def sa(n):  # 分配 n 个 int32 的 16B 对齐共享内存
            align_elems = 16  # 以 16 个 int32（64B）为对齐粒度
            aligned_n = align_up(n, align_elems)  # 元素数向上取整
            return smem.allocate_tensor(Int32, cute.make_layout((aligned_n,)), byte_alignment=16)  # 分配并返回张量视图
        PHASE_D_TILE = 32  # Phase D 各输出按 32 元素 tile 对齐切分
        PHASE_D_GROUPS_PER_CTA = cutlass.const_expr(  # 每 CTA 负责的 VM 组数（对齐到 tile）
            align_up((2 * epn + num_sms - 1) // num_sms, PHASE_D_TILE)
        )
        PHASE_D_ETC_PER_CTA = cutlass.const_expr(  # 每 CTA 负责的 experts_to_copy 条数（对齐到 tile）
            align_up((R * epn + num_sms - 1) // num_sms, PHASE_D_TILE)
        )
        # Each CTA only stages the Phase D slice it owns. Each segment gets an
        # extra +4 ints to hold the head/tail elements the 16B-aligned envelope
        # may pull in.
        # ↑ 中文补充：每个 CTA 只暂存自己负责的 Phase D 分片；每段额外 +4 个 int32，
        # 容纳 16B 对齐包络可能多带入的头/尾元素。
        PD_CU_OFF = 0  # cu_seqlens 暂存区在 scratch 内的偏移
        PD_CU_LEN = cutlass.const_expr(align_up(PHASE_D_GROUPS_PER_CTA + 4, 4))  # cu 暂存区长度（含余量、对齐到 4）
        PD_ZFR_OFF = cutlass.const_expr(PD_CU_OFF + PD_CU_LEN)  # 零填充区间暂存区偏移
        PD_ZFR_LEN = cutlass.const_expr(align_up(2 * PHASE_D_GROUPS_PER_CTA + 4, 4))  # 零填充区间暂存区长度
        PD_ETC_OFF = cutlass.const_expr(PD_ZFR_OFF + PD_ZFR_LEN)  # experts_to_copy 暂存区偏移
        PD_ETC_LEN = cutlass.const_expr(align_up(PHASE_D_ETC_PER_CTA + 4, 4))  # experts_to_copy 暂存区长度
        PD_SCRATCH_INTS = cutlass.const_expr(PD_ETC_OFF + PD_ETC_LEN)  # Phase D 暂存总大小
        scratch_ints = cutlass.const_expr(max(  # 统一 scratch 取各用途的最大需求
            R * S1_COLS,  # 用途一：tpe 列暂存 [R, S1_COLS]
            E + E // R + R,  # 用途二：s_hist + s_bp + s_chosen 合计
            (E + 1) * (BLOCK_DIM_P2 // 32 + 1),  # 用途三：run_c1 的 s_warp_counts
            PD_SCRATCH_INTS,  # 用途四：Phase D 暂存
        ))
        scratch = sa(scratch_ints)  # 分配统一的 int32 scratch 共享内存
        pd_mbar = smem.allocate_array(Int64, num_elems=1)  # Phase D 暂存用的 mbarrier（1 个 64bit）
        s_hist = sa(E)  # smem：专家直方图 / 排他前缀（C2 用作 expoff）
        s_bp = sa(E)  # smem：run_c1 的 block 内专家前缀
        s_col = sa(E)  # smem：规划阶段专家计数列暂存
        s_chosen = sa(epn)  # smem：被选中的远端专家列表
        s_wmax = sa(64)  # smem：warp 级 scan 前缀暂存
        bar_p = bar.iterator  # 软件 grid 屏障计数器指针
        # Phase A
        # tpe gather -> rank0 chunk (helper handles head/tail alignment itself)
        # ↑ 中文补充：Phase A——把 tpe 拷入本 rank 的 meta chunk（头尾对齐由 helper 自理）
        copy_v4_remote(meta, TPE_OFF + rank * E, tpe, E, pid, tid, num_threads, num_sms)  # tpe 写入本 rank chunk 的 TPE 区
        if cutlass.const_expr(R > 1):  # 多 rank 时由 rank0 额外收集全局 topk/tpe
            if rank == 0:  # 仅 rank0 执行
                # The TOPK0/TPE region of alloc is guaranteed 16B aligned: topk/tpe push goes v4 (tail padded internally).
                # ↑ 中文补充：meta 的 TOPK0/TPE 区保证 16B 对齐；topk/tpe 推送走 v4（尾部内部补齐）。
                copy_v4_remote(meta, ms + TOPK0_OFF, topk, N, pid, tid, num_threads, num_sms)  # topk 拷入 rank0 chunk 的 TOPK0 区
                copy_v4_remote(meta, ms + TPE_OFF, tpe, E, pid, tid, num_threads, num_sms)  # tpe 拷入 rank0 chunk 的 TPE 区
        cross_rank_barrier(meta, ms, BARRIER_OFF, rank, R, bar_p, num_sms, num_threads, tid)  # 系统 scope 屏障：所有 rank 的 tpe/topk 写入可见
        if rank == 0:  # 规划仅由 rank0 计算
            z_tensor = cute.make_tensor(  # 跨组槽位转移矩阵 z[R][R] 视图
                z.iterator,
                cute.make_layout((R, R), stride=(R, 1)),
            )
            for i in cutlass.range(  # grid-stride 清零 z 矩阵
                pid * num_threads + tid, R * R, num_sms * num_threads
            ):
                z[i] = 0
            for i in cutlass.range(  # grid-stride 清零各组 token 计数
                pid * num_threads + tid, R, num_sms * num_threads
            ):
                group_tokens[i] = 0
            grid_sync(bar_p, num_sms, tid)  # 软件 grid 屏障：清零完成
            # Split the expert dimension across blocks; S1_TILE alignment keeps
            # each round processing a fixed number of columns.
            # ↑ 中文补充：把专家维度切分到各 CTA；S1_TILE 对齐保证每轮处理固定列数。
            seg_raw = cute.ceil_div(E, num_sms)  # 每 CTA 原始专家数
            experts_per_block = cute.round_up(seg_raw, S1_TILE)  # 对齐后的每 CTA 专家数
            start_idx = pid * experts_per_block; end_idx = cutlass.min(start_idx + experts_per_block, E)  # 本 CTA 的专家区间
            tpe_gather = cute.make_tensor(  # 全体 rank 的 tpe gather 视图 [R, E]
                meta.iterator + TPE_OFF,
                cute.make_layout((R, E), stride=(E, 1)),
            )
            tpe_cumsum = cute.make_tensor(  # tpe 的 rank 维前缀和输出视图 [R, E]
                meta.iterator + (PB + TPE_SUB),
                cute.make_layout((R, E), stride=(E, 1)),
            )
            s_tpe = cute.make_tensor(  # smem：当前专家 tile 的暂存 [R, S1_COLS]
                scratch.iterator,
                cute.make_layout((R, S1_COLS), stride=(S1_COLS, 1)),
            )
            for e0 in cutlass.range(start_idx, end_idx, S1_COLS):  # 按列 tile 迭代专家
                # All threads copy the current expert tile from gmem into s_tpe.
                # ↑ 中文补充：所有线程协作把当前专家 tile 从 gmem 拷入 s_tpe。
                for idx in cutlass.range(tid, R * S1_COLS, num_threads):
                    r = idx // S1_COLS; col = idx - r * S1_COLS; expert_idx = e0 + col  # 展开一维下标到 (rank, 列, 专家)
                    v = 0  # 越界补 0
                    if expert_idx < end_idx:  # 越界保护
                        v = tpe_gather[r, expert_idx]  # 读该 rank 该专家的 token 数
                    s_tpe[r, col] = v  # 存入 smem

                cute.arch.barrier()  # CTA 屏障：tile 就绪

                if tid < S1_COLS:  # 每列由一个线程做 rank 维前缀和
                    expert_idx = e0 + tid  # 该列对应的专家
                    if expert_idx < end_idx:  # 越界保护
                        # Rank prefix sum within the column; the final run accumulates
                        # the total tokens of the target rank group.
                        # ↑ 中文补充：列内按 rank 前缀求和；最终 run 累计该专家（目标组）的全部 token 数。
                        run = 0
                        for r in cutlass.range_constexpr(R):  # 串行累加各 rank
                            run += s_tpe[r, tid]; s_tpe[r, tid] = run  # 就地写含入前缀
                        cute.arch.atomic_add(group_tokens.iterator + expert_idx // epn, run, scope="gpu")  # 把该专家 token 数原子累加到目标组的计数
                cute.arch.barrier()  # CTA 屏障

                # Write the prefix sums back to meta; step3 queries them by rank/expert.
                # ↑ 中文补充：把前缀和写回 meta；C2 步按 rank/专家查询。
                for idx in cutlass.range(tid, R * S1_COLS, num_threads):
                    r = idx // S1_COLS; col = idx - r * S1_COLS; expert_idx = e0 + col  # 展开下标
                    if expert_idx < end_idx:  # 越界保护
                        tpe_cumsum[r, expert_idx] = s_tpe[r, col]  # 写前缀和到规划输出区

                cute.arch.barrier()  # CTA 屏障：进入下一 tile
            grid_sync(bar_p, num_sms, tid)  # grid 屏障：tpe 前缀和与组计数就绪
            if pid == 0:  # 组间平衡只由 CTA0 的一个 warp 完成
                if tid < 32:
                    lane = tid  # lane 号
                    # balance stays in registers throughout: lane holds
                    # bal[j]=group_tokens[lane+j*32]-CAP, CHUNK=ceil(R/32).
                    # ↑ 中文补充：balance 全程驻留寄存器：lane 持有
                    # bal[j]=group_tokens[lane+j*32]-CAP，CHUNK=ceil(R/32)。
                    CHUNK = cutlass.const_expr(ceil_div(R, 32))  # 每 lane 的寄存器槽数
                    balance = cute.make_rmem_tensor(CHUNK, Int32)  # 各组超出容量 CAP 的余量
                    for j in cutlass.range_constexpr(CHUNK):  # 初始化 balance
                        k = lane + j * 32  # 该槽对应的组号
                        balance[j] = 0  # 越界槽置 0
                        if k < R: balance[j] = group_tokens[k] - CAP  # 余量 = 组 token 数 - 容量
                    keep_balancing = True  # 平衡循环开关
                    while keep_balancing:  # 迭代转移直至无可转移余量
                        # surplus takes max (larger balance first, smaller rank
                        # on ties); deficit takes min (larger shortfall first,
                        # smaller rank on ties).
                        # ↑ 中文补充：盈余取 max（余量最大者优先，平局取小组号）；
                        # 缺口取 min（缺口最大者优先，平局取小组号）。
                        surplus, surplus_rank = reg_scan_argmax_min_idx(balance, R, lane)  # 找最大盈余组
                        deficit, deficit_rank = reg_scan_argmin_min_idx(balance, R, lane)  # 找最大缺口组
                        if surplus <= 0 or deficit >= 0:  # 无盈余或无缺口则结束
                            keep_balancing = False
                        else:
                            # The move amount is limited by the receiver's
                            # shortfall; refill deficit_rank back to CAP in one shot.
                            # ↑ 中文补充：转移量受接收方缺口限制；一次性把 deficit_rank 补回 CAP。
                            move_tokens = -deficit  # 转移量 = 缺口大小
                            for j in cutlass.range_constexpr(CHUNK):  # 更新两端的 balance
                                k = lane + j * 32  # 组号
                                if k == surplus_rank: balance[j] -= move_tokens  # 盈余组减去转出量
                                elif k == deficit_rank: balance[j] = 0  # 缺口组归零
                            if lane == 0:  # lane0 记录转移矩阵
                                z_tensor[surplus_rank, deficit_rank] = move_tokens  # z[源, 目标] = 转移槽位数
                            cute.arch.sync_warp()  # warp 同步
            grid_sync(bar_p, num_sms, tid)  # grid 屏障：组间平衡完成
            # alloc_by_rank[rank, expert] feeds step4; alloc_prefix[expert, rank]
            # stores the cumulative counts consumed by C2 binary search.
            # ↑ 中文补充：alloc[rank, expert] 供第 4 步使用；alloc 前缀 [expert, rank]
            # 存储 C2 二分查找所需的累计计数。
            alloc_cumsum = cute.make_tensor(  # alloc 前缀和视图 [E, R]
                meta.iterator + (PB + ALLOC_SUB),
                cute.make_layout((E, R), stride=(R, 1)),
            )
            alloc_tensor = cute.make_tensor(  # 本地 alloc 张量视图 [R, E]
                alloc.iterator,
                cute.make_layout((R, E), stride=(E, 1)),
            )
            s_alloc = cute.make_tensor(  # smem：当前 owner 组的配额表 [R, epn]
                scratch.iterator,
                cute.make_layout((R, epn), stride=(epn, 1)),
            )
            for owner_rank in cutlass.range(pid, R, num_sms):  # 各 CTA 轮流处理一个 owner rank
                expert_base = owner_rank * epn  # 该 owner 的本地专家区间起点

                for idx in cutlass.range(tid, epn * R, num_threads):  # 初始化配额表：只有 owner 自己保留全额
                    local_expert_id = idx // R  # 本地专家号
                    rank_idx = idx - local_expert_id * R  # 目标 rank 号
                    global_expert = expert_base + local_expert_id  # 全局专家号
                    s_alloc[rank_idx, local_expert_id] = (  # owner 保留原始 token 数，其余为 0
                        tpe_cumsum[R - 1, global_expert] if rank_idx == owner_rank else 0
                    )

                cute.arch.barrier()  # CTA 屏障

                if tid < 32:  # 一个 warp 贪心分配 z 矩阵带来的配额
                    lane = tid  # lane 号
                    R_CHUNK = cutlass.const_expr(ceil_div(R, 32))  # rank 维寄存器槽数
                    EPN_CHUNK = cutlass.const_expr(ceil_div(epn, 32))  # 专家维寄存器槽数
                    quotas = cute.make_rmem_tensor(R_CHUNK, Int32)  # 本 CTA 从各 donor 组拿到的配额
                    owner_remaining = cute.make_rmem_tensor(EPN_CHUNK, Int32)  # owner 各专家仍未放置的 token 数
                    for j in cutlass.range_constexpr(R_CHUNK):  # 读入各 donor 的配额
                        rank_idx = lane + j * 32  # donor 组号
                        quotas[j] = 0  # 越界槽置 0
                        if rank_idx < R: quotas[j] = z_tensor[owner_rank, rank_idx]  # z[owner, donor] 即 donor 给 owner 的槽位数
                    for j in cutlass.range_constexpr(EPN_CHUNK):  # 读入 owner 各专家的剩余量
                        local_expert_id = lane + j * 32  # 本地专家号
                        owner_remaining[j] = 0  # 越界槽置 0
                        if local_expert_id < epn:
                            owner_remaining[j] = s_alloc[owner_rank, local_expert_id]  # owner 原始保留量

                    keep_balancing = cutlass.Boolean(True)  # 贪心循环开关
                    while keep_balancing:  # 逐一消费剩余配额
                        max_quota, target_rank = reg_scan_argmax_min_idx(quotas, R, lane)  # 剩余最多的 donor 配额
                        if max_quota <= 0:  # 配额用尽则结束
                            keep_balancing = cutlass.Boolean(False)
                        else:
                            max_remaining, selected_expert_id = reg_scan_argmax_min_idx(  # owner 中剩余 token 最多的专家
                                owner_remaining, epn, lane)
                            if max_remaining <= 0:  # owner 无剩余 token 则结束
                                keep_balancing = cutlass.Boolean(False)
                            else:
                                take = cutlass.min(max_remaining, max_quota)  # 本次转移量 = 两者取小
                                for j in cutlass.range_constexpr(R_CHUNK):  # 扣减 donor 配额
                                    rank_idx = lane + j * 32  # donor 组号
                                    if rank_idx == target_rank: quotas[j] = max_quota - take
                                for j in cutlass.range_constexpr(EPN_CHUNK):  # 扣减专家剩余量
                                    local_expert_id = lane + j * 32  # 本地专家号
                                    if local_expert_id == selected_expert_id:
                                        owner_remaining[j] = max_remaining - take
                                if tid == 0:  # lane0 写回配额表
                                    s_alloc[target_rank, selected_expert_id] += take  # 接收方（本 rank）加上 take
                                    s_alloc[owner_rank, selected_expert_id] = max_remaining - take  # owner 剩余量更新
                                cute.arch.sync_warp()  # warp 同步
                cute.arch.barrier()  # CTA 屏障：贪心分配完成

                for idx in cutlass.range(tid, epn * R, num_threads):  # 把配额表写回全局 alloc 张量
                    rank_idx = idx // epn  # 接收 rank
                    local_expert_id = idx - rank_idx * epn  # 本地专家号
                    global_expert = expert_base + local_expert_id  # 全局专家号
                    alloc_tensor[rank_idx, global_expert] = (  # 写回 [R, E] 视图
                        s_alloc[rank_idx, local_expert_id]
                    )

                cute.arch.barrier()  # CTA 屏障

                for local_expert_id in cutlass.range(tid, epn, num_threads):  # 对每个本地专家做 rank 维含入前缀
                    cum = 0
                    for rank_idx in cutlass.range_constexpr(R):  # 逐 rank 累计
                        cum += s_alloc[rank_idx, local_expert_id]  # 累加
                        s_alloc[rank_idx, local_expert_id] = cum  # 就地写含入前缀

                cute.arch.barrier()  # CTA 屏障

                for idx in cutlass.range(tid, epn * R, num_threads):  # 把前缀和写入 alloc_cumsum 供 C2 二分
                    local_expert_id = idx // R  # 本地专家号
                    rank_idx = idx - local_expert_id * R  # rank 号
                    global_expert = expert_base + local_expert_id  # 全局专家号
                    alloc_cumsum[global_expert, rank_idx] = (  # 写入 [E, R] 视图
                        s_alloc[rank_idx, local_expert_id]
                    )

                cute.arch.barrier()  # CTA 屏障：进入下一个 owner
            grid_sync(bar_p, num_sms, tid)  # grid 屏障：alloc 全部就绪
            expert_offsets = cute.make_tensor(  # 每个 (目标 rank, 专家) 的起始偏移视图 [R, E]
                meta.iterator + (PB + EOFF_SUB),
                cute.make_layout((R, E), stride=(E, 1)),
            )
            all_cu_seqlens = cute.make_tensor(  # 每个 (目标 rank, VM 组) 的行尾偏移视图 [R, 2*epn]
                meta.iterator + (PB + CU_SUB),
                cute.make_layout((R, 2 * epn), stride=(2 * epn, 1)),
            )
            zero_fill_start = cute.make_tensor(  # 零填充起始视图 [R, 2*epn]（步长 2 的交错布局）
                meta.iterator + (PB + ZFR_SUB),
                cute.make_layout((R, 2 * epn), stride=(4 * epn, 2)),
            )
            zero_fill_count = cute.make_tensor(  # 零填充长度视图（与 start 交错）
                meta.iterator + (PB + ZFR_SUB + 1),
                cute.make_layout((R, 2 * epn), stride=(4 * epn, 2)),
            )
            all_experts_to_copy = cute.make_tensor(  # 各目标 rank 的远端专家拷贝表视图 [R, epn]
                meta.iterator + (PB + ETC_SUB),
                cute.make_layout((R, epn), stride=(epn, 1)),
            )
            all_remote_stats = cute.make_tensor(  # 各 rank 的远端统计视图 [R, 2]
                meta.iterator + (PB + STATS_SUB),
                cute.make_layout((R, 2), stride=(2, 1)),
            )
            s_expert_counts = cute.make_tensor(s_col.iterator, cute.make_layout((E,)))  # smem：各专家的接收 token 数
            s_selected_experts = cute.make_tensor(s_chosen.iterator, cute.make_layout((epn,)))  # smem：压缩后的远端专家列表
            s_scan_warp_prefix = cute.make_tensor(s_wmax.iterator, cute.make_layout((NUM_WARPS,)))  # smem：warp scan 的前缀
            for idx in cutlass.range(  # grid-stride 清零全部 rank 的远端统计
                pid * num_threads + tid, R * 2, num_sms * num_threads
            ):
                stat_rank = idx // 2  # rank 行
                stat_idx = idx - stat_rank * 2  # 统计列
                all_remote_stats[stat_rank, stat_idx] = 0
            grid_sync(bar_p, num_sms, tid)  # grid 屏障：清零完成
            for dest_rank in cutlass.range(pid, R, num_sms):  # 各 CTA 轮流处理一个目标 rank
                local_start = dest_rank * epn  # 该 rank 本地专家区间起点
                local_end = local_start + epn  # 终点
                for expert_idx in cutlass.range(tid, E, num_threads):  # 读入各专家的接收 token 数
                    cnt = alloc_tensor[dest_rank, expert_idx]  # 该 rank 从该专家分到的 token 数
                    s_expert_counts[expert_idx] = cnt
                for slot in cutlass.range(tid, epn, num_threads):  # 初始化拷贝表为 -1（空）
                    s_selected_experts[slot] = -1
                    all_experts_to_copy[dest_rank, slot] = -1
                cute.arch.barrier()  # CTA 屏障

                # With epn slots every remote expert fits. Compact all positive
                # remote experts once in global-expert order.
                # ↑ 中文补充：每 rank 至多 epn 个远端专家，槽位必然够用；
                # 按全局专家序一次性压缩所有正数的远端专家。
                if tid < 32:  # 一个 warp 完成远端专家压缩
                    lane = tid  # lane 号
                    E_CHUNK = cutlass.const_expr(ceil_div(E, 32))  # 每 lane 的专家寄存器槽数
                    lanes_lt = (Uint32(1) << lane) - Uint32(1)  # 低于本 lane 的位掩码
                    remote_expert_count = Int32(0)  # 已压缩的远端专家计数
                    for chunk in cutlass.range(E_CHUNK, unroll=1):  # 逐块扫描全部专家
                        expert_idx = chunk * Int32(32) + lane  # 该 lane 负责的专家号
                        selected = cutlass.Boolean(False)  # 是否为该 rank 的远端专家
                        if expert_idx < E:  # 越界保护
                            cnt = s_expert_counts[expert_idx]  # 该专家的接收 token 数
                            selected = (cnt > 0) & (  # 有 token 且不属于本 rank 本地专家
                                (expert_idx < local_start) | (expert_idx >= local_end)
                            )
                        peers = Uint32(cute.arch.vote_ballot_sync(selected))  # warp 内被选中的 lane 掩码
                        if selected:  # 被选中的专家按 lane 序压缩写入
                            slot = remote_expert_count + Int32(  # 目标槽位 = 基数 + 组内秩
                                cute.arch.popc(peers & lanes_lt)
                            )
                            s_selected_experts[slot] = expert_idx  # 写入 smem 列表
                            all_experts_to_copy[dest_rank, slot] = expert_idx  # 写入全局拷贝表
                            owner_rank = expert_idx // epn  # 该专家的 home rank
                            cute.arch.atomic_add(  # 原子累加 owner 的“被他人拷贝”计数
                                elem_ptr(all_remote_stats, (owner_rank, 1)),
                                1,
                                scope="gpu",
                            )
                        remote_expert_count += Int32(cute.arch.popc(peers))  # 累加本块选中的专家数
                    if lane == 0:  # lane0 记录该 rank 的远端专家总数
                        all_remote_stats[dest_rank, 0] = remote_expert_count
                cute.arch.barrier()  # CTA 屏障

                count_values = []  # 每线程寄存器：各 VM 组的真实 token 数
                expert_values = []  # 对应专家 id（-1 表示空组）
                padded_values = []  # padding 后的 token 数
                for i in cutlass.range_constexpr(IPT_EB):  # 每线程负责 IPT_EB 个组槽位
                    group_idx = tid * IPT_EB + i  # 组槽位下标（前 epn 个本地、后 epn 个远端）
                    token_count = 0  # 初始化：空组
                    expert_id = -1  # 空组无专家
                    if group_idx < 2 * epn:  # 越界保护
                        if group_idx < epn:  # 前半：本地专家组
                            expert_id = local_start + group_idx  # 本地专家号
                            token_count = s_expert_counts[expert_id]  # 该组 token 数
                        else:  # 后半：远端专家组
                            selected_expert = s_selected_experts[group_idx - epn]  # 压缩列表中的远端专家
                            if selected_expert >= 0:  # 有效远端专家
                                token_count = s_expert_counts[selected_expert]  # 其 token 数
                                expert_id = selected_expert
                    padded_count = 0  # 空/零 token 组不需要 padding
                    if token_count > 0:  # 非空组才做 padding
                        if cutlass.const_expr(tp > 1):  # token_padding 生效时向上取整
                            padded_count = cute.round_up(token_count, tp)
                        else:
                            padded_count = token_count  # padding 关闭则原样
                    count_values.append(token_count)  # 存寄存器
                    expert_values.append(expert_id)  # 存寄存器
                    padded_values.append(padded_count)  # 存寄存器

                total_padded = 0  # 本线程所有组的 padding 后总和
                for i in cutlass.range_constexpr(IPT_EB):  # 累加
                    total_padded += padded_values[i]
                # Block-wide exclusive prefix (cub BlockScan equivalent):
                # intra-warp shfl scan + warp-segment combine.
                # ↑ 中文补充：block 级排他前缀（等价 cub BlockScan）：
                # warp 内 shuffle scan + warp 段间合并。
                lane = tid & 31  # lane 号
                warp_id = tid >> 5  # warp 号
                inclusive = warp_inclusive_scan(total_padded, lane)  # warp 内含入前缀
                if lane == 31:  # 每 warp 的末 lane 记录 warp 总和
                    s_scan_warp_prefix[warp_id] = inclusive
                cute.arch.barrier()  # CTA 屏障
                if tid == 0:  # 线程 0 对各 warp 总和做排他前缀
                    acc = 0
                    for warp_idx in cutlass.range_constexpr(NUM_WARPS):  # 逐 warp
                        warp_total = s_scan_warp_prefix[warp_idx]  # 读 warp 总和
                        s_scan_warp_prefix[warp_idx] = acc  # 写排他前缀
                        acc += warp_total  # 累加
                cute.arch.barrier()  # CTA 屏障
                base = s_scan_warp_prefix[warp_id] + inclusive - total_padded  # 本线程的 block 级排他起始
                for i in cutlass.range_constexpr(IPT_EB):  # 逐组写出偏移信息
                    group_idx = tid * IPT_EB + i  # 组槽位下标
                    if group_idx < 2 * epn:  # 越界保护
                        padded_end = base + padded_values[i]  # 该组 padding 后的行尾偏移
                        token_count = count_values[i]  # 真实 token 数
                        expert_id = expert_values[i]  # 专家 id
                        if token_count > 0:  # 非空组才记录专家起始偏移
                            expert_offsets[dest_rank, expert_id] = base
                        all_cu_seqlens[dest_rank, group_idx] = padded_end  # 写行尾偏移
                        pad_start = 0  # 零填充区间默认为空
                        pad_count = 0
                        if token_count > 0:  # 非空组计算 padding 区间
                            pad_extra = padded_values[i] - token_count  # 需要填充的行数
                            if pad_extra > 0:  # 有 padding 才记录
                                pad_start = base + token_count  # 从真实数据之后开始
                                pad_count = pad_extra
                        zero_fill_start[dest_rank, group_idx] = pad_start  # 写零填充起始
                        zero_fill_count[dest_rank, group_idx] = pad_count  # 写零填充长度
                        base += padded_values[i]  # 推进到下一组的起始
                cute.arch.barrier()  # CTA 屏障：进入下一目标 rank
            grid_sync(bar_p, num_sms, tid)  # grid 屏障：全部规划输出就绪
            nb = 3 * E * R; nvec = nb // 4  # 需要广播的规划输出元素数与其 v4 单元数
            for i in cutlass.range(pid * num_threads + tid, nvec, num_sms * num_threads):  # grid-stride 遍历 v4 单元
                a0 = meta[PB + i * 4 + 0]; a1 = meta[PB + i * 4 + 1]  # 读出 4 个 int32 分量
                a2 = meta[PB + i * 4 + 2]; a3 = meta[PB + i * 4 + 3]
                addr = (mc.iterator + (PLAN_OFF + i * 4)).toint()  # multicast 缓冲中对应地址
                multimem_st_v4(addr.ir_value(), a0, a1, a2, a3)  # 多播写入，全部 rank 同时收到规划结果
        order = cute.make_tensor(meta.iterator + (rank * ms + ORDER_OFF), cute.make_layout((N,)))  # 本 rank 的 order 输出视图
        if cutlass.const_expr(R > 1):  # 多 rank：非 0 rank 自己算 c1，rank1 顺带代算 rank0
            if rank != 0:  # rank0 已由规划流程直接产出，无需 c1
                self.run_c1(topk, order, tpe, lh, s_hist, s_bp, scratch, bar_p, num_sms, pid, tid)  # 对本 rank 数据跑 c1
                if rank == 1:  # rank1 额外用 rank0 的暂存数据跑 c1
                    tk0 = cute.make_tensor(meta.iterator + (rank * ms + TOPK0_OFF), cute.make_layout((N,)))  # rank0 的 topk 暂存视图
                    tp0 = cute.make_tensor(meta.iterator + (rank * ms + TPE_OFF), cute.make_layout((E,)))  # rank0 的 tpe 暂存视图
                    order0 = cute.make_tensor(meta.iterator + (rank * ms + ORDER0_OFF), cute.make_layout((N,)))  # rank0 的 order 暂存视图
                    self.run_c1(tk0, order0, tp0, lh, s_hist, s_bp, scratch, bar_p, num_sms, pid, tid)  # 代算 rank0 的 c1
                    copy_v4_remote(meta, ORDER_OFF, order0, N, pid, tid, num_threads, num_sms)  # 把 rank0 的 order 写回其 ORDER 区
        else:
            self.run_c1(topk, order, tpe, lh, s_hist, s_bp, scratch, bar_p, num_sms, pid, tid)  # 单 rank：直接对本 rank 跑 c1
        # Clear this rank's src_info slice before all ranks publish fresh slot
        # provenance into destination-rank slices below. src_info mirrors dst's
        # rank-stride encoding: src_rank * NvS + offv; -1 is the empty-slot
        # sentinel. offv is always in [0, N), and NvS >= N.
        # ↑ 中文补充：先清空本 rank 的 src_info 切片，随后各 rank 再把新的槽位来源
        # 写入目标 rank 的切片。src_info 复用 dst 的 rank 步长编码：src_rank * NvS + offv；
        # -1 为空槽哨兵；offv 恒在 [0, N)，且 NvS >= N。
        for idx in cutlass.range(pid * num_threads + tid, NvS, num_sms * num_threads):  # grid-stride 清空本 rank 的 src_info
            meta[rank * ms + SRC_INFO_OFF + idx] = Int32(-1)
        cross_rank_barrier(meta, ms, BARRIER_OFF, rank, R, bar_p, num_sms, num_threads, tid)  # 系统 scope 屏障：清空对全体可见
        s_expoff = cute.make_tensor(s_hist.iterator, cute.make_layout((E,)))  # smem：本 rank 视角的专家排他偏移
        for e in cutlass.range(tid, E, num_threads):  # 载入本 rank 的 tpe 计数
            s_expoff[e] = tpe[e]

        cute.arch.barrier()  # CTA 屏障
        warp_exclusive_scan_e(s_expoff, E, tid)  # 求专家排他前缀（该专家之前本 rank 的 token 数）
        plo = rank * ms + PLAN_OFF  # 本 rank 收到的规划输出区基址
        order_in = cute.make_tensor(meta.iterator + (rank * ms + ORDER_OFF), cute.make_layout((N,)))  # 本 rank 的 order 输入视图
        topk_by_off = cute.make_tensor(topk.iterator, cute.make_layout((N,)))  # topk 视图（按 order 下标取专家）
        dst_out = cute.make_tensor(dst.iterator, cute.make_layout((N,)))  # dst 输出视图
        tpe_cumsum_view = cute.make_tensor(  # 规划输出中的 tpe 前缀和视图 [R, E]
            meta.iterator + (plo + TPE_SUB),
            cute.make_layout((R, E), stride=(E, 1)),
        )
        alloc_cumsum_view = cute.make_tensor(  # 规划输出中的 alloc 前缀和视图 [E, R]
            meta.iterator + (plo + ALLOC_SUB),
            cute.make_layout((E, R), stride=(R, 1)),
        )
        expert_off_view = cute.make_tensor(  # 规划输出中的专家起始偏移视图 [R, E]
            meta.iterator + (plo + EOFF_SUB),
            cute.make_layout((R, E), stride=(E, 1)),
        )

        pd_group_count = cutlass.const_expr(2 * epn)  # Phase D 的 VM 组总数
        group_begin, group_end, group_copy_begin, group_copy_count = _pd_cta_slice(  # 本 CTA 的组区间切分
            pd_group_count, pid, num_sms, PHASE_D_TILE
        )
        etc_begin, etc_end, etc_copy_begin, etc_copy_count = _pd_cta_slice(  # 本 CTA 的 experts_to_copy 区间切分
            R * epn, pid, num_sms, PHASE_D_TILE
        )
        # Index 0 of the stage tensors is the aligned envelope start; the
        # logical start is pointed inside the envelope by *_stage_bias, so the
        # Phase D writeback still reads by group/ETC-local indices.
        # ↑ 中文补充：暂存张量的下标 0 是对齐包络的起点；逻辑起点由 *_stage_bias
        # 指向包络内部，Phase D 写回仍按组/拷贝表局部下标读取。
        s_pd_cu_stage = cute.make_tensor(  # smem：cu_seqlens 暂存
            scratch.iterator + PD_CU_OFF,
            cute.make_layout((PD_CU_LEN,)),
        )
        s_pd_zfr_stage = cute.make_tensor(  # smem：零填充区间暂存
            scratch.iterator + PD_ZFR_OFF,
            cute.make_layout((PD_ZFR_LEN,)),
        )
        s_pd_etc_stage = cute.make_tensor(  # smem：experts_to_copy 暂存
            scratch.iterator + PD_ETC_OFF,
            cute.make_layout((PD_ETC_LEN,)),
        )
        cu_src_begin = PB + CU_SUB + rank * pd_group_count + group_copy_begin  # 本 CTA 的 cu 源起始（meta 内）
        zfr_src_begin = PB + ZFR_SUB + (rank * pd_group_count + group_copy_begin) * 2  # 零填充区间的源起始（交错布局乘 2）
        etc_src_begin = PB + ETC_SUB + etc_copy_begin  # experts_to_copy 的源起始
        zfr_copy_count = group_copy_count * 2  # 零填充区间每组分 start/count 两列

        if tid == 0:  # 线程 0 初始化 mbarrier
            cute.arch.mbarrier_init(pd_mbar, 1)  # 期望 1 次 arrive
        cute.arch.mbarrier_init_fence()  # 初始化可见性栅栏
        # The scratch smem was previously written through the generic proxy
        # (Phase B s_tpe / run_c1 s_wcount). Bridge those writes into the
        # async proxy before the Phase-D bulk G2S copies overwrite the same
        # storage (every thread fences its own prior writes before the
        # rendezvous).
        # ↑ 中文补充：scratch 之前经由通用代理写入（Phase B 的 s_tpe / run_c1 的 s_wcount）；
        # 在 Phase D 的 bulk G2S 覆写同一块存储前，把这些写桥接进异步代理
        # （每个线程在会合前对自己的旧写做栅栏）。
        cute.arch.fence_view_async_shared()  # 通用代理 -> 异步代理的共享内存栅栏
        cute.arch.barrier()  # CTA 屏障：栅栏完成
        if tid == 0:
            # Kick off the three G2S bulk copies before C2 starts; the
            # mbarrier's expected bytes must cover all actual bulk envelope
            # bytes.
            # ↑ 中文补充：在 C2 开始前发起三路 G2S bulk 拷贝；mbarrier 的期望字节数
            # 必须覆盖所有实际 bulk 包络字节。
            pd_stage_bytes = (  # 三段暂存的对齐字节数总和
                _pd_aligned_ints(cu_src_begin, group_copy_count)
                + _pd_aligned_ints(zfr_src_begin, zfr_copy_count)
                + _pd_aligned_ints(etc_src_begin, etc_copy_count)
            ) * 4
            cute.arch.mbarrier_arrive_and_expect_tx(pd_mbar, pd_stage_bytes)  # arrive 并设置期望传输字节数
            _pd_issue_g2s(meta, s_pd_cu_stage, cu_src_begin, group_copy_count, pd_mbar)  # 异步暂存 cu_seqlens
            _pd_issue_g2s(meta, s_pd_zfr_stage, zfr_src_begin, zfr_copy_count, pd_mbar)  # 异步暂存零填充区间
            _pd_issue_g2s(meta, s_pd_etc_stage, etc_src_begin, etc_copy_count, pd_mbar)  # 异步暂存 experts_to_copy
        cute.arch.barrier()  # CTA 屏障：发射完成
        seg = cute.ceil_div(N, num_sms)  # C2 每处理元素数
        sbeg = pid * seg; send = cutlass.min(sbeg + seg, N)  # 本 CTA 的元素区间
        for base in cutlass.range(sbeg + tid, send, num_threads * ITEMS_PER_THREAD_P2):  # 每线程每轮处理 IPT 个（跨线程块步进）
            for i in cutlass.range_constexpr(ITEMS_PER_THREAD_P2):  # 展开各元素
                idx = base + i * BLOCK_DIM_P2  # 该元素的全局下标（列优先跨步布局）
                if idx < send:  # 越界保护
                    offv = order_in[idx]  # 按 order 取出原 topk 平铺下标
                    expert_idx = topk_by_off[offv]  # 该位置的专家 id
                    prev = 0  # 前序 rank 的累计 token 数
                    if rank > 0:  # rank0 无前缀
                        prev = tpe_cumsum_view[rank - 1, expert_idx]
                    global_rank = prev + (idx - s_expoff[expert_idx])  # 该 (专家, token) 对的全局序号
                    lo = 0; hi = R; pc = 0  # 二分查找接收 rank 与其前缀
                    for bin_step in cutlass.range_constexpr(LOG2_R):  # 固定次数的二分（无分支发散）
                        mid = (lo + hi) >> 1  # 中点
                        ac = alloc_cumsum_view[expert_idx, mid]  # 该专家前 mid 个 rank 的累计分配
                        if ac > global_rank: hi = mid  # 目标落在左半
                        else: lo = mid + 1; pc = ac  # 落在右半，记录已跨过的累计
                    bo = expert_off_view[lo, expert_idx]  # 该 rank 上该专家的起始槽位
                    dst_out[offv] = lo * NvS + bo + (global_rank - pc)  # 规范化 dst = rank*NvS + 专家内偏移
                    # Publish source provenance for the dispatch builder. This
                    # mirrors dst's rank-stride encoding, but points back to the
                    # source rank and source flat top-k offset.
                    # ↑ 中文补充：为 dispatch builder 发布源信息：编码同 dst 的 rank 步长格式，
                    # 但回指源 rank 与源 topk 平铺下标。
                    src_val = rank * NvS + offv  # 源编码 = 源 rank * NvS + 源槽位
                    loff = bo + (global_rank - pc)  # 目标 rank 上的槽位偏移
                    meta[lo * ms + SRC_INFO_OFF + loff] = src_val  # 写入目标 rank 的 src_info 切片
        # Dispatch may start independently on each rank after planning
        # returns. This barrier makes all peer src_info writes visible before
        # any rank's fresh dispatch builder reads its local src_info slice.
        # ↑ 中文补充：规划返回后各 rank 可独立开始 dispatch；此屏障保证所有 peer 的
        # src_info 写入在任何 rank 的 dispatch builder 读取本地切片之前可见。
        cross_rank_barrier(meta, ms, BARRIER_OFF, rank, R, bar_p, num_sms, num_threads, tid)  # 系统 scope 屏障

        # Canonicalize dst duplicate entries. The first top-k entry per
        # destination rank stays non-negative and copies the payload; later
        # entries encode -raw_dst - 1 and only carry weights. Fresh dispatch
        # materializes the dedup structures from src_info.
        # ↑ 中文补充：规范化 dst 的重复项：每个目标 rank 的首个 top-k 项保持非负并拷贝负载；
        # 后续项编码为 -raw_dst - 1，仅携带权重。新鲜 dispatch 从 src_info 物化去重结构。
        seg_dst = cute.ceil_div(S, num_sms)  # 按源 token 维切分
        sbeg_dst = pid * seg_dst; send_dst = cutlass.min(sbeg_dst + seg_dst, S)  # 本 CTA 的 token 区间
        for base in cutlass.range(sbeg_dst + tid, send_dst, num_threads):  # 线程遍历各自 token
            s = base  # 源 token 号
            base_idx = s * K  # 该 token 在 dst 中的起始下标
            dst_vals = []  # 寄存器缓存 K 个原始 dst 值
            dests = []  # 对应的目标 rank
            for k in cutlass.range_constexpr(K):  # 逐 top-k 读入
                v = dst_out[base_idx + k]  # 原始 dst 编码
                d = v // NvS  # 解出目标 rank
                dst_vals.append(v)
                dests.append(d)
            # Detect duplicate destination ranks in registers. Two i64 masks
            # cover up to 128 ranks; host-side bounds guard larger configs.
            # ↑ 中文补充：在寄存器中检测重复的目标 rank：两个 i64 位图最多覆盖
            # 128 个 rank；更大配置由主机侧断言保护。
            seen_lo = Int64(0)  # rank 0-63 的已见位图
            seen_hi = Int64(0)  # rank 64-127 的已见位图
            for k in cutlass.range_constexpr(K):  # 逐 top-k 判重
                d = dests[k]  # 目标 rank
                dup = cutlass.Boolean(False)  # 是否重复
                if d < Int32(64):  # 低半段位图
                    shift = Int64(d)  # 位偏移
                    bit = Int64(1) << shift  # 对应位
                    dup = (seen_lo & bit) != Int64(0)  # 已见过则重复
                    seen_lo = seen_lo | bit  # 标记已见
                else:  # 高半段位图
                    shift = Int64(d - Int32(64))  # 位偏移
                    bit = Int64(1) << shift  # 对应位
                    dup = (seen_hi & bit) != Int64(0)  # 已见过则重复
                    seen_hi = seen_hi | bit  # 标记已见
                if dup:  # 重复项改写为负编码
                    dst_out[base_idx + k] = -(dst_vals[k]) - 1  # -raw_dst - 1，仅携带权重
        cute.arch.mbarrier_wait(pd_mbar, 0)  # 等待 Phase D 的三路 G2S 暂存完成
        cu_stage_bias = _pd_stage_bias(cu_src_begin)  # cu 暂存的包络内偏移
        zfr_stage_bias = _pd_stage_bias(zfr_src_begin)  # 零填充暂存的包络内偏移
        etc_stage_bias = _pd_stage_bias(etc_src_begin)  # 拷贝表暂存的包络内偏移
        if pid == 0:  # 仅 CTA0 读回远端统计
            for i in cutlass.range(tid, 2, num_threads):  # 两个统计槽位
                remote_stats[i] = meta[PB + STATS_SUB + rank * 2 + i]  # 从规划输出区拷到输出张量
        for group_idx in cutlass.range(group_begin + tid, group_end, num_threads):  # 线程写回本 CTA 的组区间
            local_group = group_idx - group_begin  # CTA 内局部组号
            cu_seqlens[group_idx] = s_pd_cu_stage[cu_stage_bias + local_group]  # 写回 cu_seqlens
            zfr[group_idx * 2] = s_pd_zfr_stage[zfr_stage_bias + local_group * 2]  # 写回零填充起始
            zfr[group_idx * 2 + 1] = s_pd_zfr_stage[zfr_stage_bias + local_group * 2 + 1]  # 写回零填充长度

        for etc_idx in cutlass.range(etc_begin + tid, etc_end, num_threads):  # 线程写回拷贝表区间
            experts_to_copy[etc_idx] = s_pd_etc_stage[  # 从暂存按包络偏移读取
                etc_stage_bias + (etc_idx - etc_begin)
            ]
        # grid_sync self-resets: the counter's low 31 bits always return to 0
        # and the top bit alternates between two phases, so no cleanup zeroing
        # is needed.
        # ↑ 中文补充：grid_sync 自复位：计数器低 31 位总会归零、最高位在两个相位间翻转，
        # 因此无需清理清零。

# ============================================================
# Host side: compile cache + launch（主机侧：编译缓存与发射）
# ============================================================
@functools.lru_cache(maxsize=None)  # 以全部 constexpr 配置为键缓存编译产物
def _get_compiled(R, E, S, K, NvS_capacity, NvS, num_vblocks, meta_stride,
                  TPE_OFF, PLAN_OFF, BARRIER_OFF, TOPK0_OFF, ORDER_OFF, ORDER0_OFF,
                  token_padding, num_sms):  # 按配置编译（或取缓存）规划 kernel
    k = PlanningKernel(R, E, S, K, NvS_capacity, NvS, num_vblocks, meta_stride,  # 构造 kernel 封装对象
                       TPE_OFF, PLAN_OFF, BARRIER_OFF, TOPK0_OFF, ORDER_OFF, ORDER0_OFF,  # 传入全部 meta 偏移
                       token_padding, num_sms)  # 传入 padding 粒度与 SM 数
    i32 = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)  # 占位指针（编译期只关心类型与对齐）
    return cute.compile(k, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,  # 编译：14 个张量参数用占位指针
                        i32, i32, i32, i32, Int32(0), cuda.CUstream(0))  # rank 用 Int32 占位、流占位


def _launch_planning_kernel(ctx, topk, tpe, dst, cu_seqlens,
                            experts_to_copy, zero_fill_ranges, remote_stats):  # 取编译产物并传入真实指针发射
    comp = _get_compiled(  # 获取（或触发编译）规划 kernel
        int(ctx['R']),  # EP rank 数
        int(ctx['E']),  # 专家总数
        int(ctx['S']),  # 每 rank token 数
        int(ctx['K']),  # top-k
        int(ctx['NvS_capacity']),  # 槽位硬容量
        int(ctx['NvS']),  # 逻辑槽位数
        int(ctx['num_vblocks']),  # 直方图分块数
        int(ctx['meta_chunk_padded']),  # meta chunk 步长
        int(ctx['TPE_OFF']),  # tpe 区偏移
        int(ctx['PLAN_OFF']),  # 规划输出区偏移
        int(ctx['BARRIER_OFF']),  # 屏障区偏移
        int(ctx['TOPK0_OFF']),  # topk0 暂存区偏移
        int(ctx['ORDER_OFF']),  # order 区偏移
        int(ctx['ORDER0_OFF']),  # order0 暂存区偏移
        int(ctx['token_padding']),  # padding 粒度
        int(ctx['num_sms']),  # SM 数
    )

    def p16(t):  # large buffers 16B aligned -> allows coalesced/vectorized access  # 大缓冲按 16B 对齐声明，允许合并/向量化访问
        return make_ptr(Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    def p4(t):  # odd-length outputs; avoids out-of-bounds vector writes  # 奇数长度输出按 4B 对齐声明，避免向量写越界
        return make_ptr(Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)  # 包装当前 CUDA 流
    comp(  # 发射编译好的 kernel
        p16(tpe),  # 每 expert token 数（16B 对齐）
        p16(topk),  # topk 平铺输入（16B 对齐）
        p16(ctx['meta_buf']),  # NVLink meta 缓冲
        p16(ctx['meta_mc']),  # multicast meta 缓冲
        p16(dst),  # dst 输出
        p4(cu_seqlens),  # cu_seqlens 输出（4B 对齐）
        p4(experts_to_copy),  # 拷贝表输出（4B 对齐）
        p4(zero_fill_ranges),  # 零填充区间输出（4B 对齐）
        p4(remote_stats),  # 远端统计输出（4B 对齐）
        p16(ctx['alloc']),  # 本地 alloc 临时缓冲
        p16(ctx['group_tokens']),  # 组 token 计数临时缓冲
        p16(ctx['z']),  # 转移矩阵临时缓冲
        p16(ctx['local_hist']),  # 本地直方图临时缓冲
        p16(ctx['grid_sync_bar']),  # 软件 grid 屏障计数器
        Int32(int(ctx['rank'])),  # 当前 rank 号
        stream,  # 目标 CUDA 流
    )


def _round4(n):  # 向上取整到 4 的倍数（16B 对齐，防尾部向量写越界）
    return (n + 3) & ~3  # 位运算实现


def allocate_planning_outputs(ctx: dict):  # 在当前流上分配规划输出与计划对象
    """Allocate a ``(MoonEPCommPlan, cu_seqlens)`` pair on the current stream.

    The plan-owned dedup tensors are allocated here so the returned plan is
    complete, but fresh planning leaves their contents for the dispatch builder
    to materialize.
    """
    E = ctx['E']  # 专家总数
    epn = E // ctx['R']  # 每 rank 本地专家数
    S = ctx['S']  # 每 rank token 数
    K = ctx['K']  # top-k
    N = S * K  # top-k 平铺总数
    NvS = ctx['NvS']  # 逻辑槽位数
    dev = ctx['meta_buf'].device  # 输出所在的 CUDA 设备

    # CuTe DSL kernel grid-stride writes get vectorized (up to 4×int32 = 16B).
    # Over-allocate output tensors to a multiple of 4 and then slice, avoiding
    # an out-of-bounds full-vector write at the tail (returned shapes unchanged).
    # ↑ 中文补充：CuTe DSL 的 grid-stride 写会向量化（最多 4×int32=16B）；
    # 输出张量先过量分配到 4 的倍数再切片，避免尾部整向量写越界（返回形状不变）。
    dst = torch.empty(_round4(N), dtype=torch.int32, device=dev)[:N]  # dst 输出 [N]
    cu_seqlens = torch.empty(_round4(2 * epn), dtype=torch.int32, device=dev)[:2 * epn]  # cu_seqlens 输出 [2*epn]
    experts_to_copy = torch.empty(  # 拷贝表输出 [R, epn]
        _round4(ctx['R'] * epn), dtype=torch.int32, device=dev
    )[:ctx['R'] * epn].view(ctx['R'], epn)
    zero_fill_ranges = torch.empty(  # 零填充区间输出 [2*epn, 2]
        _round4(4 * epn), dtype=torch.int32, device=dev
    )[:4 * epn].view(2 * epn, 2)
    remote_stats = torch.empty(_round4(2), dtype=torch.int32, device=dev)[:2]  # 远端统计输出 [2]
    dup_groups = torch.empty(  # 去重组记录 [NvS, 3]
        _round4(NvS * 3), dtype=torch.int32, device=dev
    )[:NvS * 3].view(NvS, 3)
    dup_loffs = torch.empty(_round4(NvS), dtype=torch.int32, device=dev)[:NvS]  # 重复 loff 列表 [NvS]
    dup_counts = torch.empty(_round4(2), dtype=torch.int32, device=dev)[:2]  # 去重计数 [2]

    plan = MoonEPCommPlan(  # 组装计划对象
        dst=dst,  # dst 输出
        experts_to_copy=experts_to_copy,  # 拷贝表
        zero_fill_ranges=zero_fill_ranges,  # 零填充区间
        remote_stats=remote_stats,  # 远端统计
        dup_groups=dup_groups,  # 去重组
        dup_loffs=dup_loffs,  # 重复 loff
        dup_counts=dup_counts,  # 去重计数
        N=N,  # 标量配置 N
        R=ctx['R'],  # 标量配置 R
        E=E,  # 标量配置 E
        NvS=NvS,  # 标量配置 NvS
        K=K,  # 标量配置 K
    )
    return plan, cu_seqlens  # 返回计划与 cu_seqlens


def _check_planning_outputs(ctx: dict, cu_seqlens, plan) -> None:  # 校验调用方传入的输出对象
    assert isinstance(plan, MoonEPCommPlan)  # plan 必须是 MoonEPCommPlan
    E = ctx['E']  # 专家总数
    epn = E // ctx['R']  # 每 rank 专家数
    assert plan.N == ctx['S'] * ctx['K']  # N 配置一致
    assert plan.R == ctx['R']  # R 配置一致
    assert plan.E == E  # E 配置一致
    assert plan.NvS == ctx['NvS']  # NvS 配置一致
    assert plan.K == ctx['K']  # K 配置一致
    assert cu_seqlens.dtype == torch.int32 and cu_seqlens.is_contiguous()  # cu_seqlens 必须为连续 int32
    assert tuple(cu_seqlens.shape) == (2 * epn,)  # 形状 [2*epn]


def _check_dedup_encoding_bounds(ctx: dict) -> None:  # 校验去重编码的 int32 位宽上限
    R = int(ctx['R'])  # rank 数
    S = int(ctx['S'])  # 每 rank token 数
    K = int(ctx['K'])  # top-k
    NvS = int(ctx['NvS'])  # 槽位数
    NvS_BITS = 32 - 1 - KIDX_BITS  # packed 编码中留给 NvS 的位宽
    N = S * K  # top-k 平铺总数
    int32_max = 2**31 - 1  # int32 上限
    assert N <= NvS, (  # src_info 的 NvS 步长编码要求 S*K <= NvS
        f"src_info NvS-stride encoding requires S*K <= NvS, got S*K={N}, NvS={NvS}"
    )
    assert R * NvS <= int32_max, (  # src_info 线性编码要求 R*NvS 不溢出 int32
        "src_info linear encoding requires R*NvS <= int32_max: "
        f"R={R}, NvS={NvS}, R*NvS={R * NvS}, int32_max={int32_max}"
    )
    assert R <= 128, (  # dst 判重位图最多支持 128 rank
        f"dst duplicate canonicalization rank bitset requires R <= 128, got {R}"
    )
    assert K <= (1 << KIDX_BITS) - 1, (  # packed 编码要求 K 不超出其位宽
        f"primary_packed encoding requires K <= {(1 << KIDX_BITS) - 1}, got {K}"
    )
    assert NvS <= (1 << NvS_BITS) - 1, (  # packed 编码要求 NvS 不超出其位宽
        f"primary_packed encoding requires NvS <= {(1 << NvS_BITS) - 1}, got {NvS}"
    )
    assert K <= 32, f"kmask bitmask requires K <= 32, got {K}"  # kmask 位图要求 K <= 32


def launch_planning(  # 对外入口：运行规划 kernel，输出原地写入调用方对象
    ctx: dict,
    topk_experts_flat,
    tokens_per_expert,
    cu_seqlens,
    plan,
) -> None:
    """Run the planning kernel; may reuse caller-allocated output objects.

    Results are written in-place into ``plan.dst``, ``plan.experts_to_copy``,
    ``plan.zero_fill_ranges``, ``plan.remote_stats`` and ``cu_seqlens``.
    The plan-owned dedup structures are materialized in the fresh dispatch
    builder; the reuse path with an existing plan reuses these tensors
    directly.
    """
    _check_planning_outputs(ctx, cu_seqlens, plan)  # 校验输出对象
    _check_dedup_encoding_bounds(ctx)  # 校验编码位宽上限

    _launch_planning_kernel(  # 发射规划 kernel
        ctx, topk_experts_flat, tokens_per_expert,  # 上下文与输入
        plan.dst, cu_seqlens, plan.experts_to_copy,  # 输出：dst / cu_seqlens / 拷贝表
        plan.zero_fill_ranges, plan.remote_stats,  # 输出：零填充区间 / 远端统计
    )
