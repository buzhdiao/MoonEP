> 中文版说明文档，对应原文：[README.md](README.md)（结构与章节保持一致，代码、命令、链接不做翻译）

# MoonEP

MoonEP 是一个专家并行(Expert Parallelism)通信库,通过**动态冗余专家**让各 rank 之间的 token 负载保持完美均衡。

**记号**:`S` = 每个 rank 的输入 token 数,`K` = 每个 token 的路由 top-k。

1. **完美均衡**:无论路由多么倾斜,每个 rank 都恰好接收 `S × K` 个 token。少量冗余专家根据当前的路由输出在线规划,并在专家计算之前完成预取;它们的梯度在反向传播中归约回各自的归属 rank。
2. **在线规划**:一个近似最优的 GPU 规划 kernel,开销可忽略不计
3. **零拷贝与静态形状**:融合的 permute/unpermute——token 直接被发送到远端 rank 上按专家分组后的最终位置,并把缓冲区视图直接交给后续计算。只需要固定大小的 `S × K` 缓冲区,而静态已知的形状消除了每层 MoE 的 host 侧同步。

## 性能

两个 benchmark 均在 H20 上运行,EP=8,并扫描路由不均衡度:

$$\text{maxvio} = \max_e \left( \frac{T_e}{\bar{T}} \right) - 1$$

其中 $T_e$ 是路由到专家 $e$ 的 token 数,$\bar{T}$ 是完美均衡下每个专家的期望 token 数(maxvio = 0 表示完美均衡)。

**与 DeepEP v2 的通信对比**([benchmarks/bench_vs_deepep.py](benchmarks/bench_vs_deepep.py)):

<img src="figure/comm_vs_deepep.png" alt="MoonEP 与 DeepEP v2 通信对比" width="800">

- **零拷贝让裸通信更快**:token 直接写入远端 rank 上按专家分组后的最终位置——无需 permute 输入、也无需 permute 输出——并且通信缓冲区的视图直接交给计算,消除了占据 epilogue 主要开销的"通信缓冲区 → 用户缓冲区"拷贝。在每个不均衡级别上,MoonEP 的通信时间都稳定低于 DeepEP v2。
- **完美均衡使其对不均衡免疫**:随着 maxvio 增大,MoonEP 的通信时间几乎保持平坦,而 DeepEP v2——其延迟由最热的 rank 决定——持续恶化。
- **对比已计入 MoonEP 额外的 kernel**:MoonEP 增加了 DeepEP 不需要的规划和权重预取 kernel,这些开销已经叠加在上面的柱状图中。即使把整条关键路径都算上,MoonEP 的 dispatch 总时间也与 DeepEP v2 仅 dispatch 一项持平,并在不均衡加大时反超;而 combine 在每个级别上都显著更快。

**端到端训练**:

<img src="figure/e2e_vs_deepep.png" alt="MoonEP 与 DeepEP 端到端训练对比" width="800">

- **DeepEP 随不均衡而退化**:最热的 rank 接收更多 token,因此迭代时间随 maxvio 增大而持续攀升;与此同时,不断变化的激活形状会碎片化 GPU 显存,直到高不均衡下训练 OOM。
- **MoonEP 不受影响**:每个 rank 在每一层都恰好计算 `S × K` 个 token,因此迭代时间在每个不均衡级别上都保持平坦;完全静态的显存形状意味着没有碎片化,训练永不 OOM。

## 支持的设备

- NVIDIA GPU
- Zhenwu PPU(审核中,即将推出)

## 用法

### 集成

**记号**:`S` = 每个 rank 的输入 token 数,`K` = 每个 token 的路由 top-k,`E` = EP 组内路由专家总数,`R` = EP rank 数(EP 通信域大小),`epn = E/R` = 每个 rank 的本地专家数以及预取/归约槽位数,`NvS` = 每个 rank 被接收的 token 槽位数(`S × K` 个真实 token 加上按 VM group 的填充),`H` = 隐藏维度,`H'` = 专家 FFN 中间维度。

MoonEP 的通信 API 对每个投影接收一个本地专家张量 `[epn, H, H']` 加上一个覆盖所有 rank 的预取缓冲区视图 `[R, epn, H, H']`。集成 MoonEP 的框架把本 rank 的本地专家和它的本地预取切片拼成一个连续的 `[2*epn, H, H']` 计算视图,供 VM group GEMM 使用;规划器产出的 `cu_seqlens[2*epn]` 选中其中的活跃行。

#### 权重缓冲区

<img src="figure/weight_buffer.png" alt="MoonEP 权重缓冲区布局" width="1000">

对每个专家投影(gate/up/down),框架构建两个相关视图:

- **通信视图 `[R, epn, H, H']`**:把所有 rank 的预取池直接映射进来,供 `buffer.prefetch_weight` 使用。
- **计算视图 `[2*epn, H, H']`**:行 `[0, epn)` 别名到本 rank 的本地参数权重;行 `[epn, 2*epn)` 别名到本 rank 的预取切片。VM group GEMM 和 `cu_seqlens` 使用这个紧凑的行序,无需映射其他 rank 的参数权重。

每个 rank 的预取池是进程级全局的、被所有层共享,因此额外的物理开销是每个投影总共 `epn` 份专家权重,而不是每层一份。

每个 rank 拥有 `epn` 个预取槽位。规划器向每个目标 rank 搬运的专家至多来自一个远端归属组,因此这些槽位足以覆盖所有远端专家段。

#### 梯度缓冲区(仅训练需要)

<img src="figure/grad_buffer.png" alt="MoonEP 梯度缓冲区布局与梯度归约" width="1000">

训练在 fp32 下镜像这套紧凑权重布局:

- **本地梯度 `[epn, H, H']`**:本 rank 的参数梯度。
- **计算梯度视图 `[2*epn, H, H']`**:本地梯度后跟本 rank 的归约缓冲区切片。尾部存放的是临时的预取槽位梯度,与框架自身的参数梯度归约保持分离。
- **归约缓冲区**:每个 rank 把全部 `R` 个归约缓冲区映射为一个 `[R, epn, H, H']` 视图。`reduce_grad` 让每个 rank 从各个 rank 的归约缓冲区中读取存放着自家专家梯度的槽位(经 NVLink 的远端读取),累加进自己的本地参数梯度,然后把自己已消费的槽位清零,供下一个 microbatch 使用。

### API 演练

```python
from moonep import Buffer

buffer = Buffer(S=4096, H=7168, K=8, E=256, num_ep_ranks=8,
                num_sms=32, token_padding=128)
```

- `num_sms=None` 时默认为 32。当前实现在内部推导 `epn = E // num_ep_ranks`。
- `dispatch` / `combine` / `prefetch_weight` / `reduce_grad` 都接受 `async_finish=True`,在通信流上执行并返回一个 CUDA event。
- `combine` 默认 `inter_rank_sync=True`,即在 staging(向 NVL 分片搬运数据)之前执行一次显式的 rank 同步。传入 `inter_rank_sync=False` 可跳过这次 staging 前同步;combine kernel 自身仍会在入口处执行跨 rank barrier。

#### dispatch 前向

```python
hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
    hidden_sh,          # [S, H] bf16
    route_weights_sk,   # [S, K] fp32
    topk_experts_sk,    # [S, K] int32
    tokens_per_expert,  # [E] int32,本 rank 的计数
)
# hidden_nvsh:       [NvS, H] bf16 —— dispatch 后的 token,按物理 VM group 顺序排列
# route_weights_nvs: [NvS] fp32
# cu_seqlens:        [2*epn] int32 —— 每个 VM group 行的填充后 token 结束偏移
# plan:              MoonEPCommPlan —— 保存它,供 prefetch/combine 以及两侧反向使用

buffer.prefetch_weight(
    plan=plan,
    local_gate_weight=local_gate_weight,        # [epn, H, H'] bf16
    local_up_weight=local_up_weight,            # [epn, H, H'] bf16
    local_down_weight=local_down_weight,        # [epn, H, H'] bf16
    gate_prefetch_buffer=gate_prefetch_buffer,  # [R, epn, H, H'] bf16
    up_prefetch_buffer=up_prefetch_buffer,      # [R, epn, H, H'] bf16
    down_prefetch_buffer=down_prefetch_buffer,  # [R, epn, H, H'] bf16
)
# 框架的 GEMM 视图把 local_*_weight 与本 rank 的
# *_prefetch_buffer 切片拼接为 [2*epn, H, H']。
```

#### dispatch 反向

dispatch 的反向:把每个 token 的 K 份已 dispatch 梯度副本按 token 序求和——本质是一次 combine——并把冗余专家的权重梯度归约回归属 rank。

```python
grad_hidden_sh, _, _ = buffer.combine(
    plan=plan,
    hidden_nvsh=grad_hidden_nvsh,    # [NvS, H] bf16
)
# grad_hidden_sh: [S, H] bf16

buffer.reduce_grad(
    plan=plan,
    local_gate_grad=local_gate_grad,          # [epn, H, H'] fp32
    local_up_grad=local_up_grad,              # [epn, H, H'] fp32
    local_down_grad=local_down_grad,          # [epn, H, H'] fp32
    gate_reduce_buffer=gate_reduce_buffer,  # [R, epn, H, H'] fp32
    up_reduce_buffer=up_reduce_buffer,      # [R, epn, H, H'] fp32
    down_reduce_buffer=down_reduce_buffer,  # [R, epn, H, H'] fp32
)
# 框架的 GEMM 梯度视图把 local_*_grad 与本 rank 的
# *_reduce_buffer 切片拼接为 [2*epn, H, H']。
```

#### combine 前向

```python
output_sh, gathered_route_weights_sk, _ = buffer.combine(
    plan=plan,
    hidden_nvsh=expert_output_nvsh,       # [NvS, H] bf16
    route_weights_nvs=route_weights_nvs,  # [NvS] fp32,可选
)
# output_sh:                 [S, H] bf16 —— combine 后的 token 序输出
# gathered_route_weights_sk: [S, K] fp32 或 None —— 聚回 token 序的路由权重
```

#### combine 反向

combine 的反向:用已保存的 plan 把输出梯度重新 scatter 回 VM group 顺序——相当于重新 dispatch,但跳过规划,也无需预取。

```python
grad_expert_output_nvsh, _, _, _ = buffer.dispatch(
    grad_output_sh,    # [S, H] bf16
    plan=plan,
)
# grad_expert_output_nvsh: [NvS, H] bf16
```

#### zero_copy

默认情况下 `dispatch` 返回全新张量,`combine` 则先把输入拷贝进 NVL 分片。当两侧都传 `zero_copy=True` 时,`dispatch` 返回通信缓冲区的视图,专家 FFN 原地读写它们——边界处完全没有任何拷贝:

```python
hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
    hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
    zero_copy=True,
)
# hidden_nvsh / route_weights_nvs 是 NVL 缓冲区的视图;
# 专家 FFN 必须在 hidden_nvsh 上原地写出其输出
output_sh, gathered_route_weights_sk, _ = buffer.combine(
    plan=plan,
    hidden_nvsh=hidden_nvsh,
    route_weights_nvs=route_weights_nvs,
    zero_copy=True,  # 断言这些输入正是 dispatch 返回的那些视图
)
```

- 这些视图别名了会被下一次 `dispatch` / `combine` 覆盖的缓冲区状态——不要跨通信调用持有它们(autograd 不得将它们保存到 backward;那种场景需要 `zero_copy=False`)。

```python
# 在销毁进程组之前,显式释放 Buffer 持有的 VMM/NVLink 资源
buffer.destroy()
```

## 构建与测试

```bash
pip install -e .

# 运行测试(需要多 GPU + NVLink)
torchrun --nproc_per_node=8 -m pytest tests/test_planning.py
torchrun --nproc_per_node=8 -m pytest tests/test_dispatch.py
torchrun --nproc_per_node=8 -m pytest tests/test_combine.py
torchrun --nproc_per_node=8 -m pytest tests/test_e2e.py
torchrun --nproc_per_node=8 -m pytest tests/test_grad_reduce.py
torchrun --nproc_per_node=8 -m pytest tests/test_prefetch.py
```

## 致谢

本库受以下工作启发:

- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [Echo](https://arxiv.org/abs/2603.07685)
- [UltraEP](https://github.com/Dots-Infra/UltraEP)
- AcclEP(阿里巴巴的 EP 通信库)

## 引用

```bibtex
@misc{moonep2026,
      title = {MoonEP: A Perfectly Balanced Expert Parallelism Library via Dynamic Redundant Experts},
      author = {Yutian Chen, Cong Li, Yucheng Wang, Ming Wei},
      year = {2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/MoonshotAI/MoonEP}},
}
```
