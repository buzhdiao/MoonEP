
# ============================================================
# Bit widths for the packed int32 dedup encodings (single source of
# truth for planning / dispatch kernels and tests).
# 打包进 int32 的去重（dedup）编码所用的位宽；是规划内核、dispatch 内核
# 与测试共同引用的唯一真源（single source of truth）。
# ============================================================
# 2^7 = 128, which is enough for current mainstream EP settings
# RANK_BITS：rank 编号字段的位宽。2^7 = 128 个 rank，足以覆盖当前主流的 EP 规模。
RANK_BITS = 7
# Topk up to 32 uses a single b32 kmask; KIDX_BITS is wider so the packed
# kidx field keeps room for future non-bitmask encodings.
# KIDX_BITS：topk 专家索引（kidx）字段的位宽。topk≤32 时用一个 b32 位掩码即可表达，
# 而这里刻意取更宽的 7 位，为将来非位掩码（non-bitmask）编码方式预留打包空间。
KIDX_BITS = 7

# Number of dedup-builder warps per dispatch CTA. The builder's chunk scans
# are latency-bound with a single warp (one dependent-load chain per SM);
# splitting each CTA chunk over several warps divides that wall time so the
# structure generation hides under the dispatch NVLink transfer. The exposed
# tail flattens out beyond a few warps, so 4 is the sweet spot.
# 每个 dispatch CTA 中负责构建去重结构（dedup builder）的 warp 数。单 warp 时
# builder 的分块扫描受访存延迟限制（每个 SM 只有一条相互依赖的加载链）；把 CTA
# 的分块拆给多个 warp 并行，可等比缩短这段墙钟时间，让去重结构生成隐藏在
# dispatch 的 NVLink 传输之下。warp 数再多时暴露出的尾延迟趋于平坦，故 4 为最佳值。
DEDUP_BUILDER_WARPS = 4
