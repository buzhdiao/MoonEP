"""
MoonEP Combine Prologue — CuTe DSL implementation.

In-place duplicate accumulation on the local NVL shard before combine: for
each duplicate group the primary row and its duplicate rows are streamed
through a smem pipeline, accumulated in fp32 registers, and the partial sum
is stored back to the primary row. The dispatch epilogue fans a primary out
to its duplicates; this kernel is the reverse reduction for the combine
direction.

The caller stages the combine input into the NVL shard first
(``zero_copy=False`` copies the user tensor in with ``tensor.copy_``;
``zero_copy=True`` requires the FFN output to already alias the shard).

This kernel only stages the local rank's combine inputs. Launch it
immediately before ``launch_combine`` on the same stream; the combine kernel
performs the cross-rank publish barrier at entry before consuming peer
ranks' staged rows.
"""

import functools  # 提供 lru_cache：缓存设备属性查询与编译产物

import torch  # PyTorch：张量校验与 CUDA 流获取
import cuda.bindings.driver as cuda  # CUDA driver 绑定：CUstream 句柄类型

import cutlass  # CUTLASS Python DSL：constexpr 常量包装
import cutlass.cute as cute  # CuTe 核心：layout/tensor 抽象与 jit/kernel 装饰器
import cutlass.pipeline as pipeline  # CUTLASS 流水线原语：mbarrier 生产者-消费者同步
import cutlass.utils as utils  # CuTe 工具集：SmemAllocator
from cutlass import BFloat16, Float32, Int32, Int64  # DSL 数值类型
from cutlass.cute.runtime import make_ptr  # 宿主侧构造裸指针实参

from moonep._common import cp_async_bulk_g2s, pdl_trigger_dependents  # G2S bulk 拷贝封装；PDL 触发后继 kernel
from moonep.planning import MoonEPCommPlan  # 通信规划对象：携带 dedup 元数据张量


# ============================================================================
# Combine prologue kernel
# ============================================================================
# combine 前置 kernel：在本地 NVL 分片上对重复 top-k 项做原地归约。
# dispatch 尾声把主行（primary）扇出到各重复行；本 kernel 是其逆操作——
# 把每组重复行累加回主行，使后续 combine 只需拉取去重后的主行。

# 轮转（round-robin）处理重复组的 fp32 归约 kernel（详细契约见类 docstring）
class CombinePrologueKernel:
    """Round-robin fp32 accumulation of duplicate groups on the NVL shard.

    Plan-owned dedup structures drive all work: ``dup_groups`` lists the
    ``(primary_loff, dup_start, dup_n)`` headers, ``dup_loffs`` the compact
    flat duplicate slots, and ``dup_counts[0]`` the valid ``dup_groups``
    prefix length (read on device — no host synchronization).

    Each CTA walks its round-robin groups (group ``i`` -> block
    ``i % num_sms``) as one flat row stream (primary first, then the
    duplicates of each group) and packs ``B`` rows per smem pipeline stage,
    so the mbarrier handshake is amortized over ``B`` rows. Producer and
    consumers count rows with the same ``cnt`` counter, so their stage
    boundaries mirror exactly; a stage may span a group boundary. The
    producer prescans its own group headers once (lane-parallel + butterfly
    reduction) to learn the CTA's total row count, so every stage —
    including the final partial one — announces the exact
    ``expect_tx = min(B, rows_left) * H_BYTES`` before issuing its copies;
    no padding copies and no host-side counts are needed, and the consumers
    simply stop after their last real row.

    Warp layout (5 warps, 160 threads):
      - Warp 0: G2S producer — ``B`` rows per pipeline stage, one
        ``expect_tx`` handshake per stage.
      - Warps 1-4: fp32 ACC consumers — each thread owns
        ``H / ACC_THREADS`` fp32 registers. ``VEC >= 2`` shapes use
        vectorized LDS/STG over ``CHUNKS`` vectors of contiguous elements;
        ``VEC == 1`` falls back to scalar bf16 stores because CuTe DSL
        narrow-precision vector stores require at least 32-bit alignment.

    In-place safety: duplicate rows are only read; a primary row is staged
    strictly before its own consumers write it (the write happens after the
    full-barrier of the stage holding the group's last row, by which time
    every g2s of the group — including the primary — has completed), and no
    other group touches it. The launch grid must equal ``num_sms``.

    This kernel has no cross-rank barrier. Its local pipeline syncs only
    order the in-kernel staging work; ``combine`` publishes the local NVL
    shard with an entry ``cross_rank_barrier`` before reading remote rows.
    """

    ACC_THREADS = 128             # warps 1..4  累加线程组：warp1..4 共 4×32=128 线程
    num_threads = 32 + ACC_THREADS  # 每 CTA 总线程数：warp0（生产者）+ 4 个累加 warp = 160

    # Handshake amortization vs pipeline depth: prefer the largest per-stage
    # row batch B that still leaves a >= _MIN_STAGES-deep pipeline; fall
    # back to the deepest available geometry when even B=1 cannot reach
    # _MIN_STAGES. (Geometry no longer depends on R — the pre-batching
    # (4,3,2)/(2,) split came from the 05-step study of the un-batched
    # kernel.)
    _B_CANDIDATES = (16, 8, 4, 2, 1)  # 每级打包行数 B 的候选（从大到小优先摊销握手开销）
    _MIN_STAGES = 4  # 流水线最小深度要求

    # LDS/STG vector width (elements): largest of (8, 4, 2, 1) dividing
    # H / ACC_THREADS, i.e. 16/8/4/2-byte accesses.
    _VEC_CANDIDATES = (8, 4, 2, 1)  # 访存向量化宽度候选：对应 16/8/4/2 字节访问

    # 构造函数：保存形状配置，自动选取 (B, 流水线级数) 几何参数与访存向量化宽度
    def __init__(
        self,
        H: int,  # 隐藏维大小
        R: int,  # EP 组内 rank 数（当前几何不再依赖 R，仅为缓存键保留）
        NvS: int,  # 本地 NVL 分片的行数
        num_sms: int,  # 网格 block 数（=常驻 SM 数）
        smem_budget: int,  # 每 block 可用共享内存上限（字节）
        pdl_trigger: bool,  # 是否在收尾时用 PDL 触发后继 kernel
    ):
        self.H = H  # 记录隐藏维
        self.R = R  # 记录 rank 数
        self.NvS = NvS  # 记录分片行数
        self.num_sms = num_sms  # 记录网格 block 数
        self.pdl_trigger = pdl_trigger  # 记录是否启用 PDL 触发
        self.B, self.stages_acc = self._pick_geometry(H, smem_budget)  # 自动选每级行数与流水线深度
        if self.stages_acc == 0:  # 连 B=1 都放不下说明 H 超出 smem 预算
            raise RuntimeError(
                f"combine prologue: H={H} too large for per-block smem budget "
                f"{smem_budget} B (need at least {self._smem_bytes(H, 2, 1)} B)"
            )
        assert H % self.ACC_THREADS == 0  # H 须被 128 整除（寄存器划分 + bulk 拷贝对齐）
        self.VEC = 1  # 向量化宽度默认 1（标量路径）
        for v in self._VEC_CANDIDATES:  # 从大到小找能整除 H/128 的最大向量宽度
            if H % (self.ACC_THREADS * v) == 0:
                self.VEC = v
                break

    # 计算 kernel 需要的每 block 动态共享内存字节数（静态方法）
    @staticmethod
    def _smem_bytes(H: int, stages_acc: int, B: int) -> int:
        # acc_smem (bf16, B rows x stages_acc deep) + 2 mbar i64 per stage,
        # padded to 128 B / 16 B, plus 256 B headroom for CUTLASS state.
        def _round_up(n: int, a: int) -> int:  # 向上取整到 a 的倍数
            return (n + a - 1) // a * a
        return (  # 行缓冲 + mbarrier + 预留
            _round_up(stages_acc * B * H * 2, 128)  # 累加级缓冲：stages_acc 级 × 每级 B 行 × H bf16，128B 对齐
            + _round_up(stages_acc * 2 * 8, 16)  # mbarrier：每级 full/empty 各 1 个（8B），16B 对齐
            + 256  # 预留 256B 给 cutlass 内部状态
        )

    # 对给定 B，从大到小尝试候选级数，返回第一个满足 smem 预算的深度；无则 0
    @classmethod
    def _pick_stages(cls, H: int, smem_budget: int, B: int) -> int:
        for s in (16, 14, 12, 10, 8, 6, 4, 2):  # 候选级数（偶数）
            if cls._smem_bytes(H, s, B) <= smem_budget:  # 占用不超预算即采用
                return s
        return 0  # 无满足者

    # 选择 (B, stages) 几何：优先取仍能保证 ≥ _MIN_STAGES 级深度的最大 B；
    # 若 B=1 也达不到最小深度，则退回 B=1 下能取得的最大级数
    @classmethod
    def _pick_geometry(cls, H: int, smem_budget: int) -> tuple[int, int]:
        for b in cls._B_CANDIDATES:  # B 从大到小尝试
            s = cls._pick_stages(H, smem_budget, b)  # 该 B 下的最大级数
            if s >= cls._MIN_STAGES:  # 同时满足最小深度即采用
                return b, s
        return 1, cls._pick_stages(H, smem_budget, 1)  # 兜底：B=1 取最深可用级数

    # ------------------------------------------------------------------ host

    # 宿主侧 JIT 入口：构建本地分片与 dedup 元数据的视图并发射 kernel
    @cute.jit
    def __call__(
        self,
        hidden_buf_local_ptr: cute.Pointer,  # bf16 [NvS, H] local NVL shard
        dup_groups_ptr: cute.Pointer,        # int32 [NvS, 3]
        dup_loffs_ptr: cute.Pointer,         # int32 [NvS,]
        dup_counts_ptr: cute.Pointer,        # int32 [2,]
        stream: cuda.CUstream,  # 目标 CUDA 流
    ):
        H = cutlass.const_expr(self.H)  # 隐藏维（编译期常量）
        NvS = cutlass.const_expr(self.NvS)  # 分片行数

        hidden = cute.make_tensor(  # 本地 NVL 分片视图：[NvS, H] 行主序，行距用 int64 防溢出
            hidden_buf_local_ptr,
            cute.make_layout((NvS, H), stride=(Int64(H), Int64(1))),
        )
        dup_groups_tensor = cute.make_tensor(  # 重复组头表：每组 (primary_loff, dup_start, dup_n) 三个 int32
            dup_groups_ptr, cute.make_layout((NvS * 3,))
        )
        dup_loffs_tensor = cute.make_tensor(  # 紧凑的重复行槽位表：各组的重复行号连续存放
            dup_loffs_ptr, cute.make_layout((NvS,))
        )
        dup_counts_tensor = cute.make_tensor(  # 计数区：[0] 为有效组数（设备端读取，避免主机同步）
            dup_counts_ptr, cute.make_layout((2,))
        )

        smem_bytes = self._smem_bytes(H, self.stages_acc, self.B)  # 计算所需动态共享内存

        self.kernel(  # 以实际张量为实参实例化并发射 kernel
            hidden,  # 本地分片视图
            dup_groups_tensor,  # 组头表
            dup_loffs_tensor,  # 重复行槽位表
            dup_counts_tensor,  # 计数区
        ).launch(  # 发射配置
            grid=(self.num_sms, 1, 1),  # 网格 = 常驻 SM 数（组按 i % num_sms 轮转分配）
            block=(self.num_threads, 1, 1),  # 每 block 5 个 warp
            smem=smem_bytes,  # 动态共享内存字节数
            stream=stream,  # 目标 CUDA 流
        )

    # ----------------------------------------------------------------- kernel

    # 设备侧 kernel 主体：warp0 逐行流式装载重复组，warps1..4 寄存器累加，
    # 组处理完后把 fp32 部分和写回该组主行（原地归约）
    @cute.kernel
    def kernel(
        self,
        hidden: cute.Tensor,  # 本地 NVL 分片 [NvS, H]
        dup_groups_tensor: cute.Tensor,  # 重复组头表
        dup_loffs_tensor: cute.Tensor,  # 重复行槽位表
        dup_counts_tensor: cute.Tensor,  # 计数区
    ):
        H = cutlass.const_expr(self.H)  # 隐藏维
        H_BYTES = cutlass.const_expr(H * 2)  # 一行 bf16 的字节数
        stages_acc = cutlass.const_expr(self.stages_acc)  # 流水线级数
        num_sms = cutlass.const_expr(self.num_sms)  # 网格 block 数（轮转步长）
        acc_threads = cutlass.const_expr(self.ACC_THREADS)  # 累加线程数 128
        B = cutlass.const_expr(self.B)  # 每级打包行数
        VEC = cutlass.const_expr(self.VEC)  # 访存向量化宽度（元素数）
        VEC_BYTES = cutlass.const_expr(VEC * 2)  # 向量宽度对应字节数（bf16×VEC）
        CHUNKS = cutlass.const_expr(H // (self.ACC_THREADS * self.VEC))  # 每线程处理的向量块数

        bidx, _, _ = cute.arch.block_idx()  # block 编号（决定轮转起始组号）
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 统一化 warp 编号
        tidx, _, _ = cute.arch.thread_idx()  # 线程编号

        # ----- shared memory layout
        smem = utils.SmemAllocator()  # 动态共享内存分配器

        # Mbarrier block first (16-byte alignment is plenty).
        acc_load_mbar = smem.allocate_array(Int64, num_elems=2 * stages_acc)  # 累加装载流水线的 mbarrier：每级 2 个
        acc_smem = smem.allocate_tensor(  # 累加级缓冲：[B, H, stages_acc]，order=(1,0,2) 表示 H 最快、B 次之
            BFloat16,
            cute.make_ordered_layout((B, H, stages_acc), order=(1, 0, 2)),  # 元素 (b, h, s) 位于 b*H + h + s*B*H：每行连续
            byte_alignment=128,  # 128B 对齐满足 cp.async.bulk 要求
        )

        # ACC load pipe (TMA G2S → register accumulation):
        #   Producer = warp 0 (1 signalling thread, lane 0).
        #   Consumer = warps 1-4 (128 threads, 4 lane-0 signallers →
        #   consumer_group = 4).  Exact tx = min(B, rows_left) * H_BYTES per
        #   stage (the producer prescans its row count).
        acc_load_pipe = pipeline.PipelineTmaAsync.create(  # 装载流水线：warp0 生产、4 个 ACC warp 消费
            barrier_storage=acc_load_mbar,  # mbarrier 存储区
            num_stages=stages_acc,  # 级数
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),  # 生产者按单线程计数
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),  # 消费者按 4 计数（每 warp lane0 到达）
            tx_count=0,  # 事务计数在运行期按 min(B, rows_left) 动态声明，此处占位
        )

        n_groups = dup_counts_tensor[0]  # 有效重复组数（设备端读取，无主机同步）

        # ============================================
        # Warp 0 — G2S producer: B rows per stage over the flat row stream
        # (each group contributes its primary row first, then its duplicates)
        # ============================================
        # warp0：装载生产者。把本 CTA 轮转到的组展开为扁平行流（先主行后重复行），
        # 每凑满 B 行为一级发起 cp.async.bulk 拷贝并做一次 expect_tx 握手
        if warp_idx == 0:
            acc_load_state = pipeline.make_pipeline_state(  # 初始化生产者流水线状态
                pipeline.PipelineUserType.Producer, stages_acc
            )

            lane = cute.arch.lane_idx()  # 本 warp 内 lane 号

            # Prescan: this CTA's total row count, Σ (dup_count + 1) over
            # its round-robin groups. Lane l sums groups
            # bidx + (32*j + l) * num_sms; the butterfly reduction leaves
            # the warp-uniform total in every lane. Headers are 12 B each
            # and re-read by the main loop right after — cache-hot, so the
            # prescan costs ~n_groups / (num_sms * 32) iterations.
            rows_left = Int32(0)  # 预扫描的本 CTA 剩余总行数
            gj = Int32(bidx + lane * num_sms)  # 本 lane 负责的组号序列起点（跨步 num_sms）
            while gj < n_groups:  # 累加各组 (重复行数 + 主行 1 行)
                rows_left += dup_groups_tensor[gj * 3 + 2] + Int32(1)  # 组头第 3 个字段为 dup_n
                gj += Int32(num_sms * 32)  # 32 个 lane 并行扫组，步长 num_sms*32
            for off in (16, 8, 4, 2, 1):  # 蝶形归约：32 lane 的部分和汇聚为 warp 统一总数
                rows_left += cute.arch.shuffle_sync_bfly(rows_left, Int32(off))

            gi = Int32(bidx)  # 主循环当前组号（轮转起点 = block 编号）
            lane = cute.arch.lane_idx()  # lane 号（重新取一次）
            cnt = Int32(0)  # 当前级内已装行数（0..B-1）

            pref_dup_groups_base = gi * 3  # 当前组在组头表中的偏移（每组 3 个 int32）
            pref_dup_group_val = Int32(0)  # 预取的组头值（lane<3 各持一个字段）
            # gi may already be past the compact prefix (empty CTA): the
            # header row would be garbage and its dup_start would feed the
            # dependent dup_loffs load below — guard the load so row_pref
            # derives from an all-zero header instead.
            if gi < n_groups:  # 越界（空 CTA）时保持全零头，避免垃圾 dup_start 污染依赖加载
                if lane < Int32(3):  # lane0/1/2 分别读组头的三个字段
                    pref_dup_group_val = dup_groups_tensor[pref_dup_groups_base + lane]
            primary_loff = cute.arch.shuffle_sync(pref_dup_group_val, Int32(0))  # 从 lane0 广播 primary_loff
            dup_start_loff = cute.arch.shuffle_sync(pref_dup_group_val, Int32(1))  # 从 lane1 广播 dup_start
            dup_count = cute.arch.shuffle_sync(pref_dup_group_val, Int32(2))  # 从 lane2 广播 dup_n
            row_pref = primary_loff  # 每 lane 预取一行号：lane0 持主行
            if lane > Int32(0) and lane <= dup_count:  # lane j (1..dup_count) 持第 j 个重复行
                row_pref = dup_loffs_tensor[dup_start_loff + lane - 1]  # 从紧凑槽位表读重复行号

            while gi < n_groups:  # 轮转遍历本 CTA 的所有组
                gi_next = gi + Int32(num_sms)  # 下一组号（用于循环内预取）
                next_dup_group_base = gi_next * 3  # 下一组的组头偏移
                next_dup_group_val = Int32(0)  # 预取的下一组头值
                # Last group of this CTA: gi_next is past the compact prefix,
                # so the header row is garbage and the loop-tail dup_loffs
                # load would index with a garbage dup_start — guard the load
                # (an all-zero header makes the tail recompute read nothing).
                if gi_next < n_groups:  # 越界时保持全零头，使尾部重算读到空
                    if lane < Int32(3):  # lane0/1/2 并行读下一组头
                        next_dup_group_val = dup_groups_tensor[next_dup_group_base + lane]

                for k in cutlass.range(dup_count + 1, unroll=1):  # 遍历本组共 dup_n+1 行（先主行后重复行）
                    # Lane j holds row j, so the shuffle covers k <= 31; the
                    # launcher asserts plan.K <= 32 (dup_count <= 31).
                    row_loff = cute.arch.shuffle_sync(row_pref, k)  # 从持有第 k 行的 lane 广播行号（k≤31）

                    if cnt == Int32(0):  # 每级的第一行：做一次级级握手
                        acc_load_pipe.sync_object_empty.wait(  # 等待该级空槽就绪（消费者已释放）
                            acc_load_state.index, acc_load_state.phase
                        )
                        if lane == 0:  # 仅 lane0 声明事务计数并到达 mbarrier
                            # rows_left still includes the current row, so
                            # the final partial stage announces exactly its
                            # real row count — the full barrier always
                            # completes without padding copies.
                            tx_rows = Int32(B)  # 满级按 B 行声明
                            if rows_left < tx_rows:  # 末尾不满一级时按真实剩余行数声明
                                tx_rows = rows_left
                            mbar = acc_load_pipe.producer_get_barrier(  # 取该级的 full mbarrier
                                acc_load_state
                            )
                            cute.arch.mbarrier_arrive_and_expect_tx(  # 到达并声明期望字节数：TMA 完成事务自动抵扣
                                mbar, tx_rows * H_BYTES
                            )

                    if lane == 0:  # cp.async.bulk 为单线程指令，仅 lane0 发起
                        g_row_int = (  # 源行（分片内第 row_loff 行）全局内存字节地址
                            hidden.iterator + Int64(row_loff) * Int64(H)
                        ).toint()
                        s_row_int = (  # 目的行（共享内存当前级内第 cnt 行）字节地址
                            acc_smem.iterator
                            + acc_load_state.index * B * H + cnt * H
                        ).toint()
                        mbar_int = acc_load_pipe.producer_get_barrier(  # 该级 full mbarrier 地址
                            acc_load_state
                        ).toint()
                        cp_async_bulk_g2s(  # 发起一行 bf16 的异步批量拷贝
                            s_row_int.ir_value(),
                            g_row_int.ir_value(),
                            Int32(H_BYTES).ir_value(),
                            mbar_int.ir_value(),
                        )

                    rows_left -= Int32(1)  # 剩余行数递减
                    cnt += Int32(1)  # 级内行计数递增
                    if cnt == Int32(B):  # 凑满一级：重置计数并推进流水线状态
                        cnt = Int32(0)
                        acc_load_state.advance()

                primary_loff = cute.arch.shuffle_sync(next_dup_group_val, Int32(0))  # 从预取值恢复下一组 primary_loff
                dup_start_loff = cute.arch.shuffle_sync(next_dup_group_val, Int32(1))  # 恢复下一组 dup_start
                dup_count = cute.arch.shuffle_sync(next_dup_group_val, Int32(2))  # 恢复下一组 dup_n
                row_pref = primary_loff  # 重新按 lane 预取下一组的行号
                if lane > Int32(0) and lane <= dup_count:  # lane j 持第 j 个重复行号
                    row_pref = dup_loffs_tensor[dup_start_loff + lane - 1]
                gi += Int32(num_sms)  # 轮转推进到下一组

        # ============================================
        # Warps 1...4 — fp32 ACC consumers (accumulate in registers)
        # ============================================
        # warps1..4：累加消费者。逐行从共享内存读入 fp32 寄存器累加，
        # 每组处理完后把 fp32 结果转 bf16 写回该组主行
        elif warp_idx >= 1 and warp_idx <= 4:
            acc_use_state = pipeline.make_pipeline_state(  # 初始化消费者流水线状态
                pipeline.PipelineUserType.Consumer, stages_acc
            )

            acc_tid = tidx - 32  # ACC 组内线程号（0..127）

            # VEC==1 cannot use Tensor.store of a single bf16 element: CuTe
            # requires narrow-precision vector stores to be 32-bit aligned.
            # Keep that shape on the original scalar path; VEC>=2 uses the
            # vectorized path below.
            if cutlass.const_expr(VEC == 1):  # 向量宽度 1：寄存器按 CHUNKS 个标量 fp32 布局
                acc_reg = cute.make_rmem_tensor(
                    cute.make_layout((CHUNKS,)), Float32
                )
            else:  # 向量宽度 ≥2：寄存器按 [VEC, CHUNKS] 布局，便于整向量读写
                acc_reg = cute.make_rmem_tensor(
                    cute.make_layout((VEC, CHUNKS)), Float32
                )

            gi = Int32(bidx)  # 当前组号（与生产者相同的轮转序列）
            cnt = Int32(0)  # 级内行计数（与生产者镜像）
            lane = cute.arch.lane_idx()  # lane 号

            pref_dup_group_base = gi * 3  # 当前组头偏移
            pref_dup_group_val = Int32(0)  # 预取的组头值
            if gi < n_groups:  # 越界保护：空 CTA 用全零头
                if lane < Int32(3):  # lane0/1/2 并行读组头三字段
                    pref_dup_group_val = dup_groups_tensor[pref_dup_group_base + lane]
            primary_loff = cute.arch.shuffle_sync(pref_dup_group_val, Int32(0))  # 广播 primary_loff
            dup_count = cute.arch.shuffle_sync(pref_dup_group_val, Int32(2))  # 广播 dup_n

            while gi < n_groups:  # 轮转遍历本 CTA 的组
                next_gi = gi + Int32(num_sms)  # 下一组号
                next_dup_group_base = next_gi * 3  # 下一组头偏移
                next_dup_group_val = Int32(0)  # 预取的下一组头值
                if next_gi < n_groups:  # 越界保护
                    if lane < Int32(3):  # lane0/1/2 并行读下一组头
                        next_dup_group_val = dup_groups_tensor[next_dup_group_base + lane]

                # Prologue: zero accumulator regs.
                # Thread-local — no cross-warp barrier needed.
                acc_reg.fill(0.0)  # 每组开始前清零累加寄存器（线程本地，无需跨 warp 屏障）

                for k in cutlass.range(dup_count + 1, unroll=1):  # 逐行累加本组 dup_n+1 行
                    if cnt == Int32(0):  # 每级第一行：等待该级 TMA 装载完成
                        acc_load_pipe.consumer_wait(acc_use_state)

                    row_ptr = (  # 当前行在共享内存中的起始指针（级基址 + 级内行偏移）
                        acc_smem.iterator
                        + acc_use_state.index * B * H + cnt * H
                    )
                    if cutlass.const_expr(VEC == 1):  # 标量路径：逐元素 LDS + FADD
                        for c in cutlass.range_constexpr(CHUNKS):  # 遍历本线程负责的块
                            idx = c * acc_threads + acc_tid  # 该块覆盖的隐藏维下标（跨线程交织）
                            acc_reg[c] = acc_reg[c] + Float32(  # 读 bf16 转 fp32 累加
                                acc_smem[cnt, idx, acc_use_state.index]
                            )
                    else:
                        # fp32 accumulate: acc_reg[:, c] += f32(stage row
                        # chunk)
                        for c in cutlass.range_constexpr(CHUNKS):  # 逐向量块累加
                            s_vec = cute.make_tensor(  # 把共享内存当前块包装为长度 VEC 的 bf16 向量视图
                                cute.make_ptr(
                                    BFloat16,
                                    (
                                        row_ptr
                                        + c * acc_threads * VEC + acc_tid * VEC  # 块基址 + 线程内向量偏移
                                    ).toint(),
                                    cute.AddressSpace.smem,
                                    assumed_align=VEC_BYTES,  # 按 VEC 字节对齐假设
                                ),
                                cute.make_layout((VEC,)),
                            )
                            acc_slice = acc_reg[(None, c)]  # 取累加寄存器的第 c 个向量切片
                            acc_slice.store(  # 向量化 FADD 后写回寄存器
                                acc_slice.load() + s_vec.load().to(Float32)
                            )

                    cnt += Int32(1)  # 级内行计数递增
                    if cnt == Int32(B):  # 处理完一级（与生产者镜像的级边界）
                        cnt = Int32(0)
                        # ACC warps must finish reading the stage before
                        # consumer_release lets the producer overwrite it.
                        # consumer_release signals from each warp's lane 0
                        # only (PipelineTmaAsync signalling thread;
                        # consumer_group=4), so a warp-level sync makes that
                        # arrive cover all 32 lanes' reads — cross-warp
                        # completion is already gated by the 4-count empty
                        # mbarrier.
                        cute.arch.fence_view_async_shared()  # 使本线程的共享内存读可见性有序
                        cute.arch.sync_warp()  # warp 内同步：确保本 warp 32 线程读完该级
                        acc_load_pipe.consumer_release(acc_use_state)  # 释放该级（每 warp lane0 代表到达，共 4 计数）
                        acc_use_state.advance()  # 推进消费者状态

                # Epilogue: fp32 → bf16 back to the primary row of the NVL
                # shard. Use scalar stores for VEC==1; vectorized stores for
                # VEC>=2.
                out_row_ptr = hidden.iterator + Int64(primary_loff) * Int64(H)  # 该组主行的全局内存起始地址
                if cutlass.const_expr(VEC == 1):  # 标量路径：逐元素转 bf16 后普通 store
                    for c in cutlass.range_constexpr(CHUNKS):  # 遍历本线程负责的块
                        idx = c * acc_threads + acc_tid  # 隐藏维下标
                        hidden[primary_loff, idx] = BFloat16(acc_reg[c])  # 写回主行（标量 bf16 存储）
                else:
                    for c in cutlass.range_constexpr(CHUNKS):  # 向量化路径：逐向量块写回
                        g_vec = cute.make_tensor(  # 把主行对应块包装为长度 VEC 的 bf16 全局内存视图
                            cute.make_ptr(
                                BFloat16,
                                (
                                    out_row_ptr
                                    + c * acc_threads * VEC + acc_tid * VEC  # 块基址 + 线程内向量偏移
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=VEC_BYTES,  # 按 VEC 字节对齐假设
                            ),
                            cute.make_layout((VEC,)),
                        )
                        g_vec.store(acc_reg[(None, c)].load().to(BFloat16))  # fp32 转 bf16 后向量化 STG

                primary_loff = cute.arch.shuffle_sync(next_dup_group_val, Int32(0))  # 恢复下一组 primary_loff
                dup_count = cute.arch.shuffle_sync(next_dup_group_val, Int32(2))  # 恢复下一组 dup_n
                gi += Int32(num_sms)  # 轮转推进到下一组

        if cutlass.const_expr(self.pdl_trigger):  # PDL 模式：提前触发后继（combine）kernel 的启动
            pdl_trigger_dependents(tidx)


# ============================================================================
# Host launcher with per-shape compile cache
# ============================================================================
# 宿主侧发射器：校验 plan 与 ctx 的一致性，按形状缓存编译产物并发射

# 查询设备每 block 可选入的最大动态共享内存，按设备号缓存
@functools.lru_cache(maxsize=None)
def _max_smem_per_block_optin(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


# 校验 plan 与 ctx 的 dedup 元数据一致性（形状/ dtype/连续性/设备）
def _check_prologue_plan(ctx: dict, plan: MoonEPCommPlan, dev: torch.device) -> None:
    assert isinstance(plan, MoonEPCommPlan), "plan must be a MoonEPCommPlan"  # plan 类型检查
    NvS = int(ctx['NvS'])  # 取 ctx 中的分片行数

    assert plan.NvS == NvS, f"plan.NvS must match ctx NvS={NvS}, got {plan.NvS}"  # plan 与 ctx 的 NvS 必须一致

    def _check_tensor(t: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:  # 通用张量校验
        assert t.dtype == torch.int32 and t.is_contiguous(), \
            f"{name} must be contiguous int32"  # 必须是连续 int32
        assert tuple(t.shape) == shape, \
            f"{name} must be shape {shape}, got {tuple(t.shape)}"  # 形状必须匹配
        assert t.device == dev, \
            f"{name} must be on {dev}, got {t.device}"  # 必须与分片同设备

    _check_tensor(plan.dup_groups, "dup_groups", (NvS, 3))  # 校验组头表
    _check_tensor(plan.dup_loffs, "dup_loffs", (NvS,))  # 校验重复行槽位表
    _check_tensor(plan.dup_counts, "dup_counts", (2,))  # 校验计数区


# 按 (H, R, NvS, num_sms, pdl_trigger) 缓存 cute.compile 的编译结果
@functools.lru_cache(maxsize=None)
def _get_compiled(
    H: int,  # 隐藏维
    R: int,  # rank 数（缓存键）
    NvS: int,  # 分片行数
    num_sms: int,  # 网格 block 数
    device_index: int,  # 目标 GPU 编号
    pdl_trigger: bool,  # 是否启用 PDL 触发
):
    smem_budget = _max_smem_per_block_optin(device_index) - 1024  # 预留 1KB 余量（与 C++ 实现一致）
    kernel = CombinePrologueKernel(  # 用真实形状参数构造 kernel 对象
        H=H, R=R, NvS=NvS, num_sms=num_sms, smem_budget=smem_budget,
        pdl_trigger=pdl_trigger,
    )

    bf16_ptr = make_ptr(BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)  # 代表性 bf16 空指针
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)  # 代表性 int32 空指针
    stream_arg = cuda.CUstream(0)  # 代表性默认流

    return cute.compile(  # 以代表实参完成 DSL 编译
        kernel,
        bf16_ptr,  # hidden_buf_local
        i32_ptr,   # dup_groups
        i32_ptr,   # dup_loffs
        i32_ptr,   # dup_counts
        stream_arg,
    )


# launch_combine_prologue 的 Python 入口：校验入参并绑定真实地址发射 kernel
def launch_combine_prologue(
    ctx: dict,  # 会话上下文：hidden_buf_local/维度/SM 数等
    plan: MoonEPCommPlan,  # 通信规划：携带 dedup 元数据张量
    *,
    pdl_trigger: bool = False,  # 是否在收尾时用 PDL 触发后继 kernel
):
    """Accumulate duplicate rows into their primary row on the NVL shard.

    Must run after the combine input has been staged into
    ``ctx['hidden_buf_local']`` and before ``launch_combine`` on the same
    stream. Consumes the plan-owned ``dup_groups`` / ``dup_loffs`` /
    ``dup_counts`` from a fresh dispatch builder (reuse paths pass the saved
    tensors unchanged).
    """
    H = int(ctx['H'])  # 取隐藏维
    R = int(ctx['R'])  # 取 rank 数
    NvS = int(ctx['NvS'])  # 取分片行数
    num_sms = int(ctx['num_sms_dedup'])  # 取 dedup 路径使用的 SM 数
    dev = ctx['hidden_buf_local'].device  # 分片所在设备
    device_index = dev.index  # GPU 编号

    assert H % CombinePrologueKernel.ACC_THREADS == 0, \
        f"H must be a multiple of ACC_THREADS={CombinePrologueKernel.ACC_THREADS} " \
        "(also covers the 16-B bulk-copy alignment requirement)"  # H 须为 128 的倍数
    assert ctx['hidden_buf_local'].dtype == torch.bfloat16 and \
        ctx['hidden_buf_local'].is_contiguous()  # 分片必须是连续 bf16
    assert ctx['hidden_buf_local'].is_cuda, "hidden_buf_local must be a CUDA tensor"  # 分片必须在 GPU 上
    assert tuple(ctx['hidden_buf_local'].shape) == (NvS, H), \
        f"hidden_buf_local must be shape ({NvS}, {H}), " \
        f"got {tuple(ctx['hidden_buf_local'].shape)}"  # 分片形状必须为 (NvS, H)
    assert device_index is not None, "combine prologue requires a CUDA device"  # 必须有明确的 GPU 编号
    assert plan.K <= 32, \
        "combine prologue's lane-owned row-loff prefetch covers dup_count <= 31 " \
        f"(shuffle width); got K={plan.K}"  # lane 预取依赖 shuffle 宽度，K 不能超过 32
    _check_prologue_plan(ctx, plan, dev)  # 校验 plan 元数据一致性

    prologue_compiled = _get_compiled(H, R, NvS, num_sms, device_index, bool(pdl_trigger))  # 取（或触发）编译

    hbf_local_ptr = make_ptr(BFloat16, ctx['hidden_buf_local'].data_ptr(),  # 本地分片指针
                             cute.AddressSpace.gmem, assumed_align=16)
    dup_groups_ptr = make_ptr(Int32, plan.dup_groups.data_ptr(),  # 组头表指针
                              cute.AddressSpace.gmem, assumed_align=16)
    dup_loffs_ptr = make_ptr(Int32, plan.dup_loffs.data_ptr(),  # 重复行槽位表指针
                             cute.AddressSpace.gmem, assumed_align=16)
    dup_counts_ptr = make_ptr(Int32, plan.dup_counts.data_ptr(),  # 计数区指针（8B 对齐即可）
                              cute.AddressSpace.gmem, assumed_align=8)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)  # 绑定当前 PyTorch CUDA 流

    prologue_compiled(  # 调用编译好的 kernel：绑定真实地址并发射
        hbf_local_ptr,  # 本地分片
        dup_groups_ptr,  # 组头表
        dup_loffs_ptr,  # 重复行槽位表
        dup_counts_ptr,  # 计数区
        stream,  # CUDA 流
    )
