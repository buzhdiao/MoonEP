# 中文说明：本文件是 DeepSeek-V3（论文《DeepSeek-V3 Technical Report》, arXiv:2412.19437）
# 官方推理实现 model.py 的副本，放在 papers/ 下作为论文基线模型的参考实现。
# 与论文结构的对应关系（详见各节注释）：
#   - MLA（Multi-Head Latent Attention，论文 §2.1.1）           -> MLA 类
#   - MoE（DeepSeekMoE + 共享专家 + 无辅助损失负载均衡，§2.1.2） -> Gate / Expert / MoE 类
#   - 多 token 预测（MTP）与训练部分不在本文件内（推理专用）
#   - FP8 混合精度（细粒度块状量化，§3.3）                       -> linear()/Linear 类
#   - YaRN 长度外推（§2.1.1 RoPE 部分）                          -> precompute_freqs_cis()
# 注意：本副本的 ModelArgs 默认值是缩小版演示配置（dim=2048/27 层/64 专家），
#       与官方发布的 V3 权重（dim=7168/61 层/256 专家）不同；结构逻辑与论文一致。
# 仅依赖标准库 math 与类型标注
import math
# dataclass：声明模型超参数集合 ModelArgs
from dataclasses import dataclass
# 类型标注工具
from typing import Tuple, Optional, Literal

# PyTorch 主库
import torch
# nn 模块：定义各网络层
from torch import nn
# 函数式接口：embedding/linear/silu/softmax 等算子
import torch.nn.functional as F
# torch 分布式：张量并行 / 专家并行的集合通信
import torch.distributed as dist

# 从同目录 kernel.py 导入 FP8 相关算子：激活量化、权重反量化、FP8 GEMM
from kernel import act_quant, weight_dequant, fp8_gemm


# 以下全局量由 Transformer.__init__ 依据进程组初始化（默认单进程）
world_size = 1
rank = 0
# FP8 量化块大小（论文 §3.3：1x128 激活 / 128x128 权重的细粒度分块）
block_size = 128
# GEMM 实现选择："bf16"（权重反量化成 bf16）或 "fp8"（在线量化 + FP8 GEMM）
gemm_impl: Literal["bf16", "fp8"] = "bf16"
# 注意力实现选择："naive"（缓存完整 K/V）或 "absorb"（吸收矩阵后的低秩缓存）
attn_impl: Literal["naive", "absorb"] = "absorb"

# 模型超参数集合；字段默认值为缩小演示配置
@dataclass
class ModelArgs:
    # 最大批大小（决定 KV cache 预分配的第一维）
    max_batch_size: int = 8
    # 最大序列长度（4x4096，配合 YaRN 外推）
    max_seq_len: int = 4096 * 4
    # 权重精度："bf16" 或 "fp8"
    dtype: Literal["bf16", "fp8"] = "bf16"
    # 词表大小
    vocab_size: int = 102400
    # 隐藏维度（论文记号 d）
    dim: int = 2048
    # 稠密层 FFN 中间维
    inter_dim: int = 10944
    # MoE 专家 FFN 中间维（moe_inter_dim）
    moe_inter_dim: int = 1408
    # Transformer 层数（论文记号 N_layers）
    n_layers: int = 27
    # 前 n_dense_layers 层用稠密 FFN，其余层用 MoE（论文：前 3 层稠密）
    n_dense_layers: int = 1
    # 注意力头数
    n_heads: int = 16
    # moe
    # —— MoE 相关超参 ——
    # 路由专家总数（论文记号 N_r）
    n_routed_experts: int = 64
    # 共享专家数（论文记号 N_s，V3 为 1，此处演示用 2）
    n_shared_experts: int = 2
    # 每 token 激活的专家数（论文记号 K_r）
    n_activated_experts: int = 6
    # 分组路由的组数（论文 n_g；V3 为 8 组选 4，此处演示为 1 组=关闭分组）
    n_expert_groups: int = 1
    # 每次路由最多允许的组数（论文记号 K_g）
    n_limited_groups: int = 1
    # 路由打分函数："softmax" 或 "sigmoid"（V3 用 sigmoid）
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    # 路由权重缩放系数（论文记号 s）
    route_scale: float = 1.
    # mla
    # —— MLA 相关超参 ——
    # Q 的低秩压缩维度；0 表示不压缩 Q（V3 为 1536）
    q_lora_rank: int = 0
    # KV 联合低秩压缩维度（论文记号 d_c，V3 为 512）
    kv_lora_rank: int = 512
    # Q/K 的 nope（非旋转位置编码）部分头维（论文记号 d_h）
    qk_nope_head_dim: int = 128
    # Q/K 的 RoPE（解耦旋转位置编码）部分头维（论文记号 d_R^h）
    qk_rope_head_dim: int = 64
    # V 头维（论文记号 d_v）
    v_head_dim: int = 128
    # yarn
    # —— YaRN（长度外推）超参 ——
    # 预训练序列长度（论文记号 L）
    original_seq_len: int = 4096
    # RoPE 基频 theta
    rope_theta: float = 10000.0
    # YaRN 缩放因子（论文记号 s=40）
    rope_factor: float = 40
    # 高频/低频修正边界参数 beta_fast / beta_slow
    beta_fast: int = 32
    beta_slow: int = 1
    # 注意力温度缩放系数（论文记号 \gamma，YaRN mscale）
    mscale: float = 1.


# 词并行 Embedding：把词表按 world_size 切分到各 rank（对应论文实现的张量并行切分）
class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        # 调用基类构造
        super().__init__()
        # 保存词表大小与嵌入维度
        self.vocab_size = vocab_size
        self.dim = dim
        # 词表必须能被并行度整除
        assert vocab_size % world_size == 0
        # 本 rank 负责的词表片段大小
        self.part_vocab_size = (vocab_size // world_size)
        # 本 rank 负责的词表区间 [start, end)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        # 本 rank 的部分嵌入矩阵
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            # 标记不属于本 rank 词表区间的 token
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            # 把 token id 平移到本地词表坐标
            x = x - self.vocab_start_idx
            # 越界 token 置 0（防止越界访问，稍后其输出会被清零）
            x[mask] = 0
        # 查本地嵌入表
        y = F.embedding(x, self.weight)
        if world_size > 1:
            # 非本区间 token 的嵌入清零
            y[mask] = 0
            # 各 rank 结果求和，得到完整词表的嵌入
            dist.all_reduce(y)
        return y


# 统一的线性层计算入口：按权重精度与 gemm_impl 分派到不同实现
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight.element_size() > 1:
        # 权重元素大于 1 字节（bf16 等）：直接走标准 F.linear
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        # 权重是 FP8 存储但选择 bf16 计算：先按块反量化权重再算
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        # FP8 路径：激活按 block_size 在线量化，再用 FP8 GEMM（对应论文 §3.3 细粒度量化）
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            # FP8 GEMM 不支持 bias 融合，手动加上
            y += bias
        return y


# 基础线性层：支持 FP8 权重存储（带分块 scale）与可选 bias
class Linear(nn.Module):
    # 默认权重精度（Transformer.__init__ 会按 args.dtype 改写）
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        # 调用基类构造
        super().__init__()
        # 保存输入/输出维度
        self.in_features = in_features
        self.out_features = out_features
        # 权重按 [out, in] 存储；dtype 未指定时用类默认
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            # FP8（1 字节）权重：额外创建分块 scale 矩阵，形状为 [ceil(out/128), ceil(in/128)]
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            # 非 FP8 权重：注册空的 scale 占位，保持接口统一
            self.register_parameter("scale", None)
        if bias:
            # 注意：part_out_features 仅由 ColumnParallelLinear 子类定义，
            # 基础 Linear 默认 bias=False，开启时依赖子类先行设置该属性
            self.bias = nn.Parameter(torch.empty(self.part_out_features))
        else:
            # 无 bias：注册 None 占位
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 统一走 linear() 分派（bf16 / 反量化 / FP8）
        return linear(x, self.weight, self.bias)


# 列并行线性层：按输出维切分到各 TP rank（对应 Megatron 风格的列并行）
class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        # 输出维必须能被并行度整除
        assert out_features % world_size == 0
        # 本 rank 的输出维切片大小
        self.part_out_features = out_features // world_size
        # 以切片后的输出维构造基础 Linear
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 各 rank 独立计算自己的输出切片（调用方负责后续拼接/使用方式）
        y = linear(x, self.weight, self.bias)
        return y


# 行并行线性层：按输入维切分，输出需 all_reduce 求和
class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        # 输入维必须能被并行度整除
        assert in_features % world_size == 0
        # 本 rank 的输入维切片大小
        self.part_in_features = in_features // world_size
        # 以切片后的输入维构造基础 Linear
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 各 rank 计算部分和（bias 延后到 all_reduce 之后只加一次）
        y = linear(x, self.weight)
        if world_size > 1:
            # 跨 rank 求和得到完整输出
            dist.all_reduce(y)
        if self.bias is not None:
            # 求和后统一加 bias（避免每个 rank 各加一次）
            y += self.bias
        return y


# RMSNorm 归一化层（论文中的 RMSNorm）
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        # 调用基类构造
        super().__init__()
        # 数值稳定项
        self.eps = eps
        # 可学习的逐维缩放参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # 升到 fp32 计算均值，保证数值精度
        x = x.float()
        # RMS 归一化：x / sqrt(mean(x^2) + eps)
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 转回参数精度并乘缩放系数
        return y.type_as(self.weight) * self.weight


# 预计算 RoPE 的复数旋转因子 freqs_cis；序列长度超过预训练长度时应用 YaRN 外推
def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    # 旋转作用在 Q/K 的 RoPE 部分头维上
    dim = args.qk_rope_head_dim
    # 需要支持的最大序列长度
    seqlen = args.max_seq_len
    # YaRN 高频修正边界
    beta_fast = args.beta_fast
    # YaRN 低频修正边界
    beta_slow = args.beta_slow
    # RoPE 基频
    base = args.rope_theta
    # YaRN 插值因子
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        # 论文附录：求"旋转 num_rotations 圈"对应的维度索引（低频/高频分界）
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        # 把上下界旋转圈数换算成维度区间 [low, high]，并夹紧到 [0, dim-1]
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        # 生成 0->1 的线性斜坡（YaRN 的混合权重），min==max 时加小量避免除零
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # 标准 RoPE 基频序列：theta_i = base^(-2i/d)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        # YaRN：仅对 [low, high] 区间内的频率做位置插值（/factor），
        # 低频端保持外推、高频端保持插值，smooth 为逐频混合系数
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # 位置索引 0..seqlen-1
    t = torch.arange(seqlen)
    # 外积得到每个 (位置, 频率) 的相位角
    freqs = torch.outer(t, freqs)
    # 转成单位复数 e^{i*theta}，供后续复数乘法旋转
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


# 对输入的最后维做 RoPE 旋转：相邻两元素视为一个复数，乘 e^{i*theta}
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # 记住原始 dtype，旋转在 fp32 下进行后再转回
    dtype = x.dtype
    # [.., d] -> [.., d/2, 2] -> 复数
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    # 把旋转因子广播到 (1, seq, 1, d/2) 以匹配 (b, s, h, d/2)
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # 复数乘法实现旋转，再转回实数并展平最后一维
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


# MLA：多头潜在注意力（论文 §2.1.1）
# 核心思想：把 Q 投影到低秩潜在空间（可选），K/V 联合压缩成低秩潜在向量 c_KV，
# 显著压缩 KV cache；nope 部分做低秩吸收，RoPE 部分解耦（decoupled）单独携带。
class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        # 调用基类构造
        super().__init__()
        # 隐藏维 d
        self.dim = args.dim
        # 总头数
        self.n_heads = args.n_heads
        # 本 TP rank 负责的头数
        self.n_local_heads = args.n_heads // world_size
        # Q 低秩压缩维（0 = 不压缩）
        self.q_lora_rank = args.q_lora_rank
        # KV 联合压缩维 d_c
        self.kv_lora_rank = args.kv_lora_rank
        # nope / rope 头维与总 QK 头维
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        # V 头维
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            # 不压缩 Q：直接一个大投影到所有头的 QK 维
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # 压缩 Q（论文式 1-3）：先降维 wq_a -> RMSNorm -> 升维 wq_b
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        # KV 联合压缩（论文式 4）：一次投影同时产出 c_KV（d_c）与共享的 k_R（d_R^h）
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        # 对 c_KV 做 RMSNorm（论文式 5 前的归一化）
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # 从 c_KV 恢复各头的 k_nope 与 v（论文式 5），列并行
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        # 输出投影（论文式 6），行并行（内部带 all_reduce）
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        # softmax 缩放 = 1/sqrt(d_h + d_R^h)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            # YaRN 注意力温度修正：mscale^2 乘到 softmax 缩放上（论文附录）
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            # naive 实现缓存完整 K（nope+rope）与 V，形状 [B, S, H_local, d]
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            # absorb 实现只缓存压缩向量：c_KV（d_c）与解耦的 k_R（d_R^h），
            # 显存占用从 O(H*(d_h+d_R)) 降到 O(d_c)+O(d_R)，是 MLA 的核心收益
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        # 取批大小与序列长度
        bsz, seqlen, _ = x.size()
        # 本段 token 写入 KV cache 的结束位置
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            # 不压缩 Q：直接投影
            q = self.wq(x)
        else:
            # 压缩 Q：低秩投影 + 归一化 + 升维
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        # 恢复头维：(b, s, H_local, d_h)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        # 拆分 nope 与 rope 两部分
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # 只对 q 的 rope 部分施加旋转位置编码
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        # KV 联合压缩投影，得到 [b, s, d_c + d_R^h]
        kv = self.wkv_a(x)
        # 拆出压缩向量 c_KV 与所有头共享的 k_R
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # k_R 也施加旋转（unsqueeze 出头维以复用 apply_rotary_emb）
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            # —— naive 路径：先恢复完整 K/V，再标准注意力 ——
            # 拼回完整 Q（nope + rope）
            q = torch.cat([q_nope, q_pe], dim=-1)
            # 从 c_KV 恢复各头 k_nope 与 v
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            # 拆分 k_nope 与 v
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # k 的 rope 部分对所有头共享，广播后与 k_nope 拼接
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            # 把本段 K/V 写入缓存
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            # 标准点积注意力打分（含缓存中的历史 token）
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            # —— absorb 路径：把 W_UK 吸收进 Q，注意力在压缩空间进行 ——
            # 取出（必要时反量化）wkv_b 权重并整理成 [H_local, d_h+d_v, d_c]
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            # q_nope 吸收 W_UK：从 per-head 空间变换到压缩空间（论文式 10 的等价形式）
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            # 只缓存 c_KV 与 k_R（MLA 的 KV cache 压缩）
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            # 打分 = q_nope·c_KV（吸收后）+ q_pe·k_R（解耦 rope 部分）
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            # 叠加因果/填充掩码（-inf 屏蔽）
            scores += mask.unsqueeze(1)
        # softmax 归一化（fp32 计算后转回输入精度）
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            # 加权求和 V 得到各头输出
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            # absorb 路径：先在压缩空间加权求和 c_KV，再用 W_UV 恢复到 V 空间（论文式 11）
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        # 拼接头维并做输出投影（内部含 all_reduce）
        x = self.wo(x.flatten(2))
        return x


# 稠密前馈网络：SwiGLU 结构（论文 FFN），w1/w3 升维、w2 降维
class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        # 调用基类构造
        super().__init__()
        # gate 投影（列并行）
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        # 降维投影（行并行）
        self.w2 = RowParallelLinear(inter_dim, dim)
        # up 投影（列并行）
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU：w2( SiLU(w1 x) * w3 x )
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# MoE 路由门控（论文 §2.1.2）：打分、可选分组限制、取 top-k 专家并归一化权重
class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        # 调用基类构造
        super().__init__()
        # 保存隐藏维
        self.dim = args.dim
        # 每 token 激活专家数 K_r
        self.topk = args.n_activated_experts
        # 专家分组数 n_g
        self.n_groups = args.n_expert_groups
        # 最多选择的组数 K_g
        self.topk_groups = args.n_limited_groups
        # 打分函数类型
        self.score_func = args.score_func
        # 路由权重缩放系数
        self.route_scale = args.route_scale
        # 路由权重矩阵 [N_r, d]
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        # 无辅助损失负载均衡的偏置 b_i（论文 §2.1.2）：仅完整 V3 配置（d=7168）启用
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 对每个 token 与各专家做点积打分
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            # softmax 打分（演示配置）
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            # sigmoid 打分（V3 论文采用）
            scores = scores.sigmoid()
        # 保存原始打分：最终路由权重由原始分（不带均衡偏置）取出
        original_scores = scores
        if self.bias is not None:
            # 仅用于选择专家的均衡偏置：只影响 top-k 选择，不进入最终权重
            scores = scores + self.bias
        if self.n_groups > 1:
            # 分组路由（论文 §2.1.2）：把专家分成 n_g 组
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                # 每组取组内最高分作为组代表分
                group_scores = scores.amax(dim=-1)
            else:
                # 有偏置时取组内前 2 名之和（论文实现细节）
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            # 选出得分最高的 K_g 个组
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            # 构造组掩码：未选中的组内专家得分置 0
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        # 在（可能被组限制过的）得分上取 top-k 专家
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        # 用原始得分取出这 k 个专家的路由权重
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            # sigmoid 打分时在选中的 k 个专家内归一化（论文实现）
            weights /= weights.sum(dim=-1, keepdim=True)
        # 乘路由缩放系数
        weights *= self.route_scale
        # 返回路由权重与选中的专家索引
        return weights.type_as(x), indices


# 单个路由专家：与稠密 MLP 相同的 SwiGLU 结构（但用普通 Linear，不做 TP 切分）
class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        # 调用基类构造
        super().__init__()
        # gate 投影
        self.w1 = Linear(dim, inter_dim)
        # 降维投影
        self.w2 = Linear(inter_dim, dim)
        # up 投影
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU：w2( SiLU(w1 x) * w3 x )
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# MoE 层（论文 §2.1.2）：路由专家按 EP 切分到各 rank + 共享专家 + 求和聚合
class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        # 调用基类构造
        super().__init__()
        # 保存隐藏维
        self.dim = args.dim
        # 专家总数必须能被 EP 并行度整除
        assert args.n_routed_experts % world_size == 0
        self.n_routed_experts = args.n_routed_experts
        # 本 rank 持有的专家数
        self.n_local_experts = args.n_routed_experts // world_size
        # 每 token 激活专家数
        self.n_activated_experts = args.n_activated_experts
        # 本 rank 的全局专家区间 [start, end)
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        # 路由门控
        self.gate = Gate(args)
        # 路由专家列表：区间外的位置放 None（不创建权重，节省显存）
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        # 共享专家：所有 token 都会经过（论文：N_s 个共享专家，容量 N_s*moe_inter_dim）
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 记录输入形状，便于最后还原
        shape = x.size()
        # 展平成 [tokens, d]
        x = x.view(-1, self.dim)
        # 路由：得到每个 token 的专家权重与索引
        weights, indices = self.gate(x)
        # 输出累加容器
        y = torch.zeros_like(x)
        # 统计每个专家被路由到的 token 数（跨全部 token）
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            # 本 rank 只处理自己持有的专家 i
            if counts[i] == 0:
                # 没有 token 路由到该专家则跳过
                continue
            # 取出专家模块
            expert = self.experts[i]
            # 找到路由到专家 i 的 (token 位置, top-k 槽位)
            idx, top = torch.where(indices == i)
            # 计算 expert 输出并乘路由权重累加（对应论文式 12 的按专家分组计算）
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        # 共享专家对所有 token 计算（论文式 13 的共享专家部分）
        z = self.shared_experts(x)
        if world_size > 1:
            # EP 聚合：各 rank 只算了本地专家的部分和，all_reduce 求完整输出
            dist.all_reduce(y)
        # 还原输入形状
        return (y + z).view(shape)


# 单个 Transformer 块：MLA + FFN（前 n_dense_layers 层稠密，其余 MoE），Pre-Norm 残差
class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        # 调用基类构造
        super().__init__()
        # 多头潜在注意力
        self.attn = MLA(args)
        # 前若干层用稠密 FFN，其余用 MoE（论文：前 3 层稠密）
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        # 注意力前的 RMSNorm
        self.attn_norm = RMSNorm(args.dim)
        # FFN 前的 RMSNorm
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # 注意力子层：Pre-Norm + 残差
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        # FFN 子层：Pre-Norm + 残差
        x = x + self.ffn(self.ffn_norm(x))
        return x


# 完整 Transformer：词嵌入 + N 层 Block + 终归一化 + 输出头
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        # 声明使用模块级全局变量
        global world_size, rank
        # 从进程组读取并行度与 rank（未初始化分布式时退化为单进程）
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        # 按 args.dtype 设定全局线性层默认权重精度（bf16 或 fp8）
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        # 调用基类构造
        super().__init__()
        # 保存最大序列长度
        self.max_seq_len = args.max_seq_len
        # 词并行嵌入
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        # 逐层构建 Transformer 块
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        # 输出前的最终 RMSNorm
        self.norm = RMSNorm(args.dim)
        # 输出头（LM head）：列并行，按词表切分
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        # 预计算并缓存 RoPE/YaRN 旋转因子（非持久 buffer，不存入 checkpoint）
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        # 本段输入的序列长度
        seqlen = tokens.size(1)
        # 词嵌入
        h = self.embed(tokens)
        # 截取本段位置对应的旋转因子
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        # 掩码默认为空
        mask = None
        if seqlen > 1:
            # 多 token 输入（预填充阶段）需要因果掩码：上三角置 -inf
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        # 逐层前向
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        # 终归一化后只取最后一个位置的隐藏态（自回归解码只需最后一步）
        h = self.norm(h)[:, -1]
        # 输出头投影得到本 rank 词表分片的 logits
        logits = self.head(h)
        if world_size > 1:
            # all_gather 各 rank 的词表分片并拼接成完整词表 logits
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


if __name__ == "__main__":
    # 冒烟测试：默认 bf16 精度、CUDA 设备、固定随机种子
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    # 使用默认（缩小版）超参数
    args = ModelArgs()
    # 随机造 2 条长 128 的 token 序列
    x = torch.randint(0, args.vocab_size, (2, 128))
    # 构建模型并前向，打印输出 logits 形状（应为 [2, vocab_size]）
    model = Transformer(args)
    print(model(x).size())
