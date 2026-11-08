"""
MoonEP shared CuTe DSL helpers (common to dispatch + combine).

The core works around two cp.async.bulk lowering limitations of
nvidia-cutlass-dsl 4.4.2:
1) the TMA descriptor of ``CopyBulkTensorTileG2SOp`` caps a single-dim box at
   256 elements, so one H=7168 bf16 row is split into 28 UTMA instructions;
2) the plain form of ``CopyBulkG2SOp`` force-inserts mapa.shared::cluster,
   which mismatches the mbar that PipelineTmaAsync allocates in shared::cta,
   causing a runtime hang.

We issue cp.async.bulk PTX directly via ``llvm.inline_asm``, matching the
semantics of CUTLASS C++ ``cute::SM90_BULK_COPY_{G2S,S2G}::copy``.

Note: cp.async.bulk is a single-thread instruction; callers must wrap the
inline asm in ``if cute.arch.lane_idx() == 0:``.
"""  # 模块文档字符串：dispatch 与 combine 共用的 CuTe DSL 辅助工具；绕过 nvidia-cutlass-dsl 4.4.2 对 cp.async.bulk 的两个 lowering 限制，直接用 llvm.inline_asm 发射 cp.async.bulk PTX

import cutlass  # 导入 cutlass 主模块（用于 Boolean 等）
from cutlass import Int32, Int64, Uint32  # 导入 DSL 整数类型构造器
from cutlass._mlir.dialects import llvm  # 导入 MLIR llvm 方言，用于内联 PTX 汇编
from cutlass.cutlass_dsl import T, dsl_user_op  # 导入类型标记 T 和用户算子装饰器
import cutlass.cute as cute  # 导入 CuTe DSL 主命名空间


# ============================================================================
# Cross-rank barrier constants and helpers
# ============================================================================
# ============================================================================
# 跨 rank 栅栏常量与辅助函数
# ============================================================================

# grid_sync sentinel: per-round whole-grid increments sum to 0x80000000; a flipped top bit means all arrived.
# grid_sync 哨兵值：每轮全 grid 的增量之和恰为 0x80000000；最高位翻转即表示全部到达。
GRID_SYNC_TAG = 0x80000000  # 全 grid 栅栏的自复位哨兵常量

# Same value as GRID_SYNC_TAG
# 与 GRID_SYNC_TAG 相同的值
WARP_SYNC_TAG = 0x80000000  # 跨 warp 栅栏的自复位哨兵常量（取值同上）

# spin-wait timeout. Fail-fast on timeout to avoid further polluting the barrier state.
# 自旋等待超时上限。超时后快速失败（trap），避免继续污染栅栏状态。
BARRIER_TIMEOUT_CYCLES = 100 * 2_000_000_000  # 超时周期数：约 100 秒（按 2GHz 时钟估算）


@dsl_user_op  # 装饰器：声明为 DSL 用户算子（允许在 jit 上下文中发射内联汇编）
def clock64(*, loc=None, ip=None) -> Int64:
    """Corresponds to PTX ``mov.u64 ret, %clock64;``."""  # 文档字符串：对应 PTX 指令 mov.u64 ret, %clock64
    # 读取全局时钟周期计数器 %clock64
    return Int64(  # 把内联汇编结果包装为 Int64 返回
        llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
            T.i64(),  # 返回值类型：64 位整数
            [],  # 无输入操作数
            "mov.u64 $0, %clock64;",  # PTX 汇编：把时钟计数读入输出寄存器
            "=l",  # 约束：$0 为 64 位输出寄存器
            has_side_effects=True,  # 标记有副作用，防止被优化删除
            is_align_stack=False,  # 不要求栈对齐
            asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
            loc=loc,  # MLIR 位置信息
            ip=ip,  # MLIR 插入点
        )
    )


@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def device_trap(*, loc=None, ip=None) -> None:
    """Corresponds to PTX ``trap;``; used for device-side fail-fast."""  # 文档字符串：对应 PTX 指令 trap，用于设备端快速失败
    # 触发设备端 trap（快速失败/中止）
    llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
        None,  # 无返回值
        [],  # 无输入操作数
        "trap;",  # PTX 汇编：触发 trap，使整个 grid 异常终止
        "",  # 无寄存器约束
        has_side_effects=True,  # 标记有副作用，防止被优化删除
        is_align_stack=False,  # 不要求栈对齐
        asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
        loc=loc,  # MLIR 位置信息
        ip=ip,  # MLIR 插入点
    )


@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def atom_add_release_gpu(ptr_i64, val, *, loc=None, ip=None) -> Int32:
    # 执行带 release 语义（gpu 作用域）的 32 位原子加，用于栅栏到达信号
    return Int32(  # 把内联汇编结果包装为 Int32 返回
        llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
            T.i32(),  # 返回值类型：32 位整数（原子加前的旧值）
            [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],  # 操作数：目标地址指针与增量
            "atom.add.release.gpu.global.s32 $0, [$1], $2;",  # PTX：release 语义的 gpu 作用域原子加
            "=r,l,r",  # 约束：$0 输出寄存器，$1 64 位地址，$2 输入寄存器
            has_side_effects=True,  # 标记有副作用，防止被优化删除
            is_align_stack=False,  # 不要求栈对齐
            asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
            loc=loc,  # MLIR 位置信息
            ip=ip,  # MLIR 插入点
        )
    )


@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def atom_add_relaxed_gpu_s32(ptr_i64, val, *, loc=None, ip=None) -> Int32:
    # 执行 relaxed 语义（gpu 作用域）的 32 位原子加
    return Int32(  # 把内联汇编结果包装为 Int32 返回
        llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
            T.i32(),  # 返回值类型：32 位整数（原子加前的旧值）
            [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],  # 操作数：目标地址指针与增量
            "atom.add.relaxed.gpu.global.s32 $0, [$1], $2;",  # PTX：relaxed 语义的 gpu 作用域原子加
            "=r,l,r",  # 约束：$0 输出寄存器，$1 64 位地址，$2 输入寄存器
            has_side_effects=True,  # 标记有副作用，防止被优化删除
            is_align_stack=False,  # 不要求栈对齐
            asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
            loc=loc,  # MLIR 位置信息
            ip=ip,  # MLIR 插入点
        )
    )


@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def atom_min_relaxed_gpu_s32(ptr_i64, val, *, loc=None, ip=None) -> Int32:
    # 执行 relaxed 语义（gpu 作用域）的 32 位原子最小值操作
    return Int32(  # 把内联汇编结果包装为 Int32 返回
        llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
            T.i32(),  # 返回值类型：32 位整数（原子操作前的旧值）
            [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],  # 操作数：目标地址指针与比较值
            "atom.min.relaxed.gpu.global.s32 $0, [$1], $2;",  # PTX：relaxed 语义的 gpu 作用域原子取最小值
            "=r,l,r",  # 约束：$0 输出寄存器，$1 64 位地址，$2 输入寄存器
            has_side_effects=True,  # 标记有副作用，防止被优化删除
            is_align_stack=False,  # 不要求栈对齐
            asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
            loc=loc,  # MLIR 位置信息
            ip=ip,  # MLIR 插入点
        )
    )

@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def atom_or_relaxed_gpu_s32(ptr_i64, val, *, loc=None, ip=None) -> Int32:
    # 执行 relaxed 语义（gpu 作用域）的 32 位原子按位或（有符号版）
    return Int32(  # 把内联汇编结果包装为 Int32 返回
        llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
            T.i32(),  # 返回值类型：32 位整数（原子操作前的旧值）
            [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],  # 操作数：目标地址指针与或值
            "atom.or.relaxed.gpu.global.b32 $0, [$1], $2;",  # PTX：relaxed 语义的 gpu 作用域原子按位或（b32）
            "=r,l,r",  # 约束：$0 输出寄存器，$1 64 位地址，$2 输入寄存器
            has_side_effects=True,  # 标记有副作用，防止被优化删除
            is_align_stack=False,  # 不要求栈对齐
            asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
            loc=loc,  # MLIR 位置信息
            ip=ip,  # MLIR 插入点
        )
    )


@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def atom_or_relaxed_gpu_b32(ptr_i64, val, *, loc=None, ip=None) -> Uint32:
    # 执行 relaxed 语义（gpu 作用域）的 32 位原子按位或（无符号版）
    return Uint32(  # 把内联汇编结果包装为 Uint32 返回
        llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
            T.i32(),  # 返回值类型：32 位整数（原子操作前的旧值）
            [ptr_i64, Uint32(val).ir_value(loc=loc, ip=ip)],  # 操作数：目标地址指针与或值
            "atom.or.relaxed.gpu.global.b32 $0, [$1], $2;",  # PTX：relaxed 语义的 gpu 作用域原子按位或（b32）
            "=r,l,r",  # 约束：$0 输出寄存器，$1 64 位地址，$2 输入寄存器
            has_side_effects=True,  # 标记有副作用，防止被优化删除
            is_align_stack=False,  # 不要求栈对齐
            asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
            loc=loc,  # MLIR 位置信息
            ip=ip,  # MLIR 插入点
        )
    )


@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def popc_b32(val, *, loc=None, ip=None) -> Int32:
    # 统计 32 位值中置位（1）的个数
    return Int32(  # 把内联汇编结果包装为 Int32 返回
        llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
            T.i32(),  # 返回值类型：32 位整数（置位个数）
            [Uint32(val).ir_value(loc=loc, ip=ip)],  # 操作数：待统计的 32 位值
            "popc.b32 $0, $1;",  # PTX：统计 32 位值的置位个数
            "=r,r",  # 约束：$0 与 $1 均为 32 位寄存器
            has_side_effects=False,  # 纯计算，无副作用
            is_align_stack=False,  # 不要求栈对齐
            asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
            loc=loc,  # MLIR 位置信息
            ip=ip,  # MLIR 插入点
        )
    )


@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def ctz_b32(val, *, loc=None, ip=None) -> Int32:
    """Count trailing zeros (bit position of lowest set bit).

    PTX ``brev.b32`` + ``clz.b32``: bit-reverse first, then count leading zeros
    = count trailing zeros. Returns 32 for input 0; the caller must guarantee
    non-zero input or handle that boundary.
    """  # 文档字符串：统计末尾零的个数（最低置位位的位号）；输入为 0 时返回 32，调用者需保证输入非零
    # 统计末尾零的个数：先位反转再统计前导零
    return Int32(  # 把内联汇编结果包装为 Int32 返回
        llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
            T.i32(),  # 返回值类型：32 位整数（末尾零个数）
            [Uint32(val).ir_value(loc=loc, ip=ip)],  # 操作数：待统计的 32 位值
            "brev.b32 $0, $1; clz.b32 $0, $0;",  # PTX：先位反转（brev）再统计前导零（clz），等价于统计末尾零
            "=r,r",  # 约束：$0 与 $1 均为 32 位寄存器
            has_side_effects=False,  # 纯计算，无副作用
            is_align_stack=False,  # 不要求栈对齐
            asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
            loc=loc,  # MLIR 位置信息
            ip=ip,  # MLIR 插入点
        )
    )


@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def ld_acquire_gpu_s32(ptr_i64, *, loc=None, ip=None) -> Int32:
    # 以 acquire 语义（gpu 作用域）加载 32 位整数
    return Int32(  # 把内联汇编结果包装为 Int32 返回
        llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
            T.i32(),  # 返回值类型：32 位整数（加载值）
            [ptr_i64],  # 操作数：目标地址指针
            "ld.acquire.gpu.global.s32 $0, [$1];",  # PTX：acquire 语义的 gpu 作用域 32 位加载
            "=r,l",  # 约束：$0 输出寄存器，$1 64 位地址
            has_side_effects=True,  # 标记有副作用，防止被优化删除
            is_align_stack=False,  # 不要求栈对齐
            asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
            loc=loc,  # MLIR 位置信息
            ip=ip,  # MLIR 插入点
        )
    )


@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def red_add_release_sys(ptr_i64, val, *, loc=None, ip=None) -> None:
    """Corresponds to PTX ``red.release.sys.global.add.s32 [ptr], val;`` (no return value)."""  # 文档字符串：对应 PTX red.release.sys.global.add.s32，无返回值
    # 执行 release 语义（sys 全系统作用域）的原子加，无返回值
    llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
        None,  # 无返回值
        [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],  # 操作数：目标地址指针与增量
        "red.release.sys.global.add.s32 [$0], $1;",  # PTX：release 语义的 sys 作用域原子加（reduction，不返回旧值）
        "l,r",  # 约束：$0 64 位地址，$1 输入寄存器
        has_side_effects=True,  # 标记有副作用，防止被优化删除
        is_align_stack=False,  # 不要求栈对齐
        asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
        loc=loc,  # MLIR 位置信息
        ip=ip,  # MLIR 插入点
    )


@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def ld_acquire_sys_s32(ptr_i64, *, loc=None, ip=None) -> Int32:
    """Corresponds to PTX ``ld.acquire.sys.s32 ret, [ptr];``."""  # 文档字符串：对应 PTX ld.acquire.sys.s32
    # 以 acquire 语义（sys 全系统作用域）加载 32 位整数
    return Int32(  # 把内联汇编结果包装为 Int32 返回
        llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
            T.i32(),  # 返回值类型：32 位整数（加载值）
            [ptr_i64],  # 操作数：目标地址指针
            "ld.acquire.sys.global.s32 $0, [$1];",  # PTX：acquire 语义的 sys 作用域 32 位加载
            "=r,l",  # 约束：$0 输出寄存器，$1 64 位地址
            has_side_effects=True,  # 标记有副作用，防止被优化删除
            is_align_stack=False,  # 不要求栈对齐
            asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
            loc=loc,  # MLIR 位置信息
            ip=ip,  # MLIR 插入点
        )
    )


@cute.jit  # 装饰器：把函数编译为 CuTe jit 核函数
def grid_sync(bar_ptr, nsm: Int32, tid: Int32):
    """Self-resetting cooperative grid barrier (modeled on DeepGEMM ``grid_sync``).

    ``bar_ptr[0]`` is the arrive counter: each round SM0 adds ``TAG-(nsm-1)``
    and every other SM adds 1, totaling exactly +0x80000000, so the low 31 bits
    stay 0 and no reset is needed; a flipped top bit means all have arrived.
    The fence is just ``bar.sync`` (=sync_threads); cross-SM ordering is
    guaranteed by release/acquire atomics.
    """  # 文档字符串：自复位协作式全 grid 栅栏（仿 DeepGEMM grid_sync）：每轮所有 SM 的增量之和恰为 TAG，最高位翻转即全部到达
    # 自复位的全 grid 协作栅栏（仿 DeepGEMM grid_sync）
    cute.arch.sync_threads()  # 先做 CTA 内线程同步，保证本 CTA 先前写入可见
    if tid == 0:  # 仅每个 CTA 的 0 号线程参与到达与自旋
        b0 = bar_ptr.toint().ir_value()  # 取栅栏计数器地址的整数值
        pid = cute.arch.block_idx()[0]  # 取本 CTA 的 block 编号
        inc = Int32(1)  # 默认增量为 +1（非 SM0）
        if pid == 0:  # SM0（block 0）特殊
            inc = Int32(GRID_SYNC_TAG) - (nsm - 1)  # SM0 增量为 TAG-(nsm-1)，使每轮总增量恰为 TAG（自复位）
        old = atom_add_release_gpu(b0, inc)  # 以 release 语义原子加并记录旧值（作为本轮快照）
        done = cutlass.Boolean(False)  # 完成标志，初始为 False
        while not done:  # 自旋等待直到最高位相对快照翻转
            new = ld_acquire_gpu_s32(b0)  # 以 acquire 语义重新读取计数器
            done = ((new ^ old) & GRID_SYNC_TAG) != 0  # 与旧值异或后最高位为 1 表示全员到达
    cute.arch.sync_threads()  # CTA 内同步：0 号线程的观测结果广播给全部线程


@cute.jit  # 装饰器：把函数编译为 CuTe jit 核函数
def cross_rank_barrier(
    meta_buf: cute.Tensor,        # int32 view of the merged meta_buf
    meta_stride: Int32,           # meta_chunk_padded
    barrier_off: Int32,           # BARRIER_OFF
    rank: Int32,
    num_ranks: cutlass.Constexpr[int],
    bar_ptr,
    nsm: Int32,
    num_threads: cutlass.Constexpr[int],
    tid: Int32,
):
    """Self-resetting cross-rank barrier (modeled on DeepGEMM ``nvlink_barrier``).

    grid_sync -> only block0 sends the cross-rank +/-1 signal -> grid_sync.
    Each rank reuses 3 slots in the ``barrier_off`` region: ``+0/+1`` are the
    two phase signals (sent by peers), ``+2`` is the local phase/sign state.
    A single atomic with alternating phase/sign self-resets, so an all-zero
    initial state is already correct. The fence is always ``bar.sync``.

    Proxy-bridge contract (all threads execute both fences):
    - Entry ``fence.proxy.alias``: writer-side bridge for the VMM alias proxy.
      A thread's prior ``multimem`` accesses (mc VA) are made coherent with
      unicast-VA accesses before the barrier release publishes them, so peers
      may read the same physical memory through the unicast alias afterwards.
    - Exit ``fence.proxy.async.global``: consumer-side bridge for the async
      proxy. Data acquired into the generic proxy by this barrier becomes
      visible to subsequent TMA (``cp.async.bulk``) accesses issued by the
      executing thread.
    """  # 文档字符串：自复位跨 rank 栅栏（仿 DeepGEMM nvlink_barrier）；入口/出口的 proxy fence 分别桥接组播写与后续 TMA 访问
    # 自复位的跨 rank 栅栏（仿 DeepGEMM nvlink_barrier）
    assert num_threads >= num_ranks, (  # 断言块内线程数不少于 rank 数（每个线程负责向一个 peer 发信号）
        "cross_rank_barrier requires blockDim.x >= num_ranks: "  # 断言失败信息第一行
        f"num_threads={num_threads}, num_ranks={num_ranks}"  # 断言失败信息第二行（含具体数值）
    )
    # Writer side of the alias bridge: must precede the barrier release so the
    # multicast stores are published in a state coherent for unicast readers.
    # 别名代理桥的写者侧：必须先于栅栏 release 执行，使组播写对之后经单播别名读取的 peer 保持一致。
    cute.arch.fence_proxy("alias")  # 执行 alias 代理 fence，使 mc VA 写与单播 VA 访问一致
    grid_sync(bar_ptr, nsm, tid)  # 第一次全 grid 栅栏：等待本 rank 所有 CTA 到齐
    if cute.arch.block_idx()[0] == 0:  # 仅 block 0 负责跨 rank 信号交换
        status = meta_buf[rank * meta_stride + barrier_off + 2] & 3  # 读取本地相位/符号状态槽的低 2 位
        phase = status & 1  # 低 1 位为当前相位（0/1，交替）
        sign = status >> 1  # 高 1 位为符号标志（决定本轮信号方向）
        # Each rank sends +/-1 to the peer's current phase slot; sign picks the direction for self-reset.
        # 每个 rank 向 peer 的当前相位槽发送 +/-1；符号位决定方向以实现自复位。
        if tid < num_ranks:  # 前 num_ranks 个线程各负责向一个 peer 发信号
            peer = (meta_buf.iterator + (tid * meta_stride + barrier_off + phase)).toint()  # 计算 peer 当前相位槽的地址
            delta = Int32(1)  # 默认发送 +1
            if sign != 0:  # 若符号标志置位
                delta = Int32(-1)  # 发送 -1（使下一轮相位槽回到基态）
            red_add_release_sys(peer.ir_value(), delta)  # 以 release 语义（sys 作用域）原子加到 peer 的相位槽
        cute.arch.sync_threads()  # CTA 内同步，保证信号发射完成
        if tid == 0:  # 仅 0 号线程负责等待收齐信号并推进相位
            cute.arch.atomic_add(meta_buf.iterator + (rank * meta_stride + barrier_off + 2), Int32(1), scope="gpu")  # 本地状态槽 +1，翻转相位
            target = Int32(num_ranks)  # 期望收到的信号值：+1 * num_ranks
            if sign != 0:  # 若本轮是符号轮
                target = Int32(0)  # 期望收到 -1 * num_ranks，使相位槽归零（自复位）
            own = (meta_buf.iterator + (rank * meta_stride + barrier_off + phase)).toint()  # 本 rank 当前相位槽的地址
            done = cutlass.Boolean(False)  # 完成标志，初始为 False
            start = clock64()  # 记录自旋起始时钟，用于超时检测
            while not done:  # 自旋等待收齐所有 peer 的信号
                cur = ld_acquire_sys_s32(own.ir_value())  # 以 acquire 语义（sys 作用域）读取相位槽当前值
                done = cur == target  # 达到目标值即收齐信号
                if (clock64() - start) >= Int64(BARRIER_TIMEOUT_CYCLES):  # 检查是否超过超时周期数
                    cute.printf(  # 打印诊断信息便于定位死锁
                        "MoonEP cross_rank_barrier timeout (100s), trapping: "  # 超时提示信息第一行
                        "rank=%d barrier_off=%d phase=%d sign=%d signal=%d target=%d\n",  # 格式化输出各状态量
                        rank, barrier_off, phase, sign, cur, target,  # 传入 rank、偏移、相位、符号、当前值、目标值
                    )
                    device_trap()  # 触发设备端 trap 快速失败
    grid_sync(bar_ptr, nsm, tid)  # 第二次全 grid 栅栏：把跨 rank 完成状态广播给本 rank 所有 CTA
    # Consumer side of the async bridge: must follow the barrier acquire so
    # peer stores now visible in the generic proxy are bridged to this
    # thread's subsequent TMA (cp.async.bulk) accesses.
    # 异步代理桥的消费者侧：必须跟在栅栏 acquire 之后，把 generic 代理中已可见的 peer 写桥接到本线程后续的 TMA 访问。
    cute.arch.fence_proxy("async.global")  # 执行 async 代理 fence，使后续 cp.async.bulk（TMA）能看到数据


@cute.jit  # 装饰器：把函数编译为 CuTe jit 核函数
def cross_warp_sync(
    bar_ptr,
    nparticipants: cutlass.Constexpr[int],
    leader: Int32,
):
    """All-participating-warp barrier: lane 0 of each participating warp
    arrives once and spin-waits until all have arrived. leader must be non-zero
    on exactly one participating warp (it contributes the large increment that
    makes the barrier self-resetting). Participants may span multiple SMs x
    multiple builder warps."""  # 文档字符串：全体参与 warp 的栅栏：每个参与 warp 的 lane 0 到达并自旋等待；恰好一个 warp 是 leader（贡献大增量实现自复位）
    # 全体参与 warp 的同步栅栏
    assert nparticipants > 0, (  # 断言至少有一个参与 warp
        "cross_warp_sync requires at least one participating warp"  # 断言失败信息
    )
    assert nparticipants < WARP_SYNC_TAG, (  # 断言参与数小于哨兵值（否则自复位数学不成立）
        "cross_warp_sync participant count must be smaller than WARP_SYNC_TAG: "  # 断言失败信息第一行
        f"nparticipants={nparticipants}, tag={WARP_SYNC_TAG}"  # 断言失败信息第二行（含具体数值）
    )
    # 1. Intra-warp sync: writes by the 32 lanes of this warp become visible to each other
    # 1. warp 内同步：本 warp 32 个 lane 的写入对彼此可见
    cute.arch.sync_warp()  # 执行 warp 内同步

    b0 = bar_ptr.toint().ir_value()  # 取栅栏计数器地址的整数值
    lane = cute.arch.lane_idx()  # 取本线程在 warp 内的 lane 编号

    # 2. Release arrive
    # 2. 以 release 语义到达栅栏
    inc = Int32(1)  # 普通参与 warp 的增量为 +1
    if leader != 0:  # 若本 warp 是 leader warp
        inc = Int32(WARP_SYNC_TAG) - Int32(nparticipants - 1)   # self-reset: total increment = TAG
        # leader 增量为 TAG-(n-1)，使总增量恰为 TAG（自复位）

    # 3. ALL builder warps acquire-wait (not just bidx == 0)
    # 3. 所有参与 warp 都做 acquire 等待（不只 block 0）
    old = Int32(0)  # 快照变量，仅 lane 0 使用
    if lane == 0:  # 仅每个参与 warp 的 lane 0 到达并自旋
        old = atom_add_release_gpu(b0, inc)  # 以 release 语义原子加并记录旧值快照
        done = cutlass.Boolean(False)  # 完成标志，初始为 False
        start = clock64()  # 记录自旋起始时钟，用于超时检测
        while not done:  # 自旋等待直到最高位相对快照翻转
            new = ld_acquire_gpu_s32(b0)  # 以 acquire 语义重新读取计数器
            done = ((new ^ old) & WARP_SYNC_TAG) != 0  # 与旧值异或后最高位为 1 表示全部到达
            if (clock64() - start) >= Int64(BARRIER_TIMEOUT_CYCLES):  # 检查是否超过超时周期数
                cute.printf("MoonEP builder barrier timeout, trapping\n")  # 打印超时提示
                device_trap()  # 触发设备端 trap 快速失败
    cute.arch.sync_warp()      # broadcast completion to all 32 lanes
    # warp 内同步：把完成状态广播给全部 32 个 lane


@cute.jit  # 装饰器：把函数编译为 CuTe jit 核函数
def pdl_trigger_dependents(tid: Int32):
    """Publish this CTA's writes, then trigger programmatic dependent launch."""  # 文档字符串：发布本 CTA 的写入，然后触发程序化依赖启动（PDL）
    # 发布本 CTA 的写入并触发程序化依赖启动
    cute.arch.sync_threads()  # CTA 内同步
    cute.arch.fence_acq_rel_sys()  # sys 作用域的 acq_rel fence，确保写入对全系统可见
    cute.arch.sync_threads()  # 再次 CTA 内同步
    if tid == 0:  # 仅 0 号线程触发
        cute.arch.griddepcontrol_launch_dependents()  # 触发依赖 grid 的启动（PDL launch_dependents）


@cute.jit  # 装饰器：把函数编译为 CuTe jit 核函数
def pdl_wait_predecessor():
    """Wait until the predecessor grid has completed and flushed its writes."""  # 文档字符串：等待前驱 grid 完成并刷出其写入
    # 等待前驱 grid 完成并刷出写入
    cute.arch.griddepcontrol_wait()  # 执行 PDL 的 griddepcontrol.wait


# ============================================================================
# cp.async.bulk inline-PTX wrappers
# ============================================================================
# ============================================================================
# cp.async.bulk 内联 PTX 封装
# ============================================================================


@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def cp_async_bulk_g2s(
    dst_smem_i32, src_gmem_i64, size_i32, mbar_smem_i32,
    *, loc=None, ip=None,
) -> None:
    """Corresponds to PTX ``cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes``.

    On completion the copy automatically transaction-arrives ``size_i32``
    bytes on the mbar. Single-thread instruction; must be called by the
    elected lane.
    """  # 文档字符串：对应 PTX cp.async.bulk G2S（mbarrier complete_tx）；完成时自动在 mbar 上事务到达 size 字节；单线程指令
    # 全局显存到共享显存的 bulk 异步拷贝（mbarrier 完成跟踪）
    llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
        None,  # 无返回值
        [dst_smem_i32, src_gmem_i64, size_i32, mbar_smem_i32],  # 操作数：共享内存目的地址、全局内存源地址、字节数、mbar 地址
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "  # PTX 指令第一行
        "[$0], [$1], $2, [$3];",  # PTX 指令第二行：目的、源、大小、mbar
        "r,l,r,r",  # 约束：两个 32 位地址、一个 64 位地址、一个 32 位大小
        has_side_effects=True,  # 标记有副作用，防止被优化删除
        is_align_stack=False,  # 不要求栈对齐
        asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
        loc=loc, ip=ip,  # MLIR 位置信息与插入点
    )


@dsl_user_op  # 装饰器：声明为 DSL 用户算子
def cp_async_bulk_s2g(
    src_smem_i32, dst_gmem_i64, size_i32,
    *, loc=None, ip=None,
) -> None:
    """Corresponds to PTX ``cp.async.bulk.global.shared::cta.bulk_group [dst], [src], size;``.

    bulk_group form -- completion is tracked via ``cp_async_bulk_commit_group``
    / ``cp_async_bulk_wait_group<N>``; no mbar. Also a single-thread
    instruction.
    """  # 文档字符串：对应 PTX cp.async.bulk S2G（bulk_group 形式）；用 commit/wait_group 跟踪完成，无需 mbar；单线程指令
    # 共享显存到全局显存的 bulk 异步拷贝（bulk_group 完成跟踪）
    llvm.inline_asm(  # 通过 MLIR llvm.inline_asm 发射 PTX
        None,  # 无返回值
        [dst_gmem_i64, src_smem_i32, size_i32],  # 操作数：全局内存目的地址、共享内存源地址、字节数
        "cp.async.bulk.global.shared::cta.bulk_group "  # PTX 指令第一行
        "[$0], [$1], $2;",  # PTX 指令第二行：目的、源、大小
        "l,r,r",  # 约束：64 位目的地址、32 位源地址与大小
        has_side_effects=True,  # 标记有副作用，防止被优化删除
        is_align_stack=False,  # 不要求栈对齐
        asm_dialect=llvm.AsmDialect.AD_ATT,  # 使用 AT&T 汇编方言
        loc=loc, ip=ip,  # MLIR 位置信息与插入点
    )
