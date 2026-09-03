# RoPE 与 Transformer Decoder Block 性能实验

## 1. 实验目标

本实验实现旋转位置编码（Rotary Position Embedding，RoPE），并将 RoPE、Causal Multi-Head Self-Attention、两次 RMSNorm、SwiGLU FFN 和两条残差连接组合为完整的 Pre-Norm Transformer Decoder Block。

实验目标包括：

1. 验证 RoPE 的 Shape、范数保持和位置相关性。
2. 验证完整 Decoder Block 的输出形状、参数量、残差路径与因果性。
3. 分析序列长度和数据类型对完整 Decoder Block 延迟与吞吐量的影响。
4. 比较 FP32、FP16、BF16 的理论参数显存和执行期间额外峰值显存。
5. 为后续构建多层 Decoder-only Transformer 和实现 KV Cache 建立正确性及性能基线。

## 2. Decoder Block 结构

本实验采用 Pre-Norm 结构。Attention 子层计算为：

```text
h = x + Attention(RMSNorm(x))
```

FFN 子层计算为：

```text
y = h + FFN(RMSNorm(h))
```

完整数据流为：

```text
x
│
├── residual ─────────────────────┐
│                                 │
└── RMSNorm → RoPE Attention ─── (+)
                                  │
                                  h
                                  │
├── residual ─────────────────────┐
│                                 │
└── RMSNorm → SwiGLU FFN ─────── (+)
                                  │
                                  y
```

输入和输出形状均为：

```text
(B, S, H)
```

## 3. RoPE 原理

RoPE 对 Attention 中 Q、K 的相邻维度执行二维旋转。对于一对元素 `(x₀, x₁)`：

```text
x₀' = x₀ cosθ - x₁ sinθ
x₁' = x₀ sinθ + x₁ cosθ
```

旋转角由 token 位置和维度频率共同决定：

```text
θ = position × frequency
```

RoPE 只作用于 Q 和 K，不作用于 V。二维旋转保持向量范数，同时使 Q、K 的点积包含相对位置信息。

当 `position = 0` 时：

```text
cos(0) = 1
sin(0) = 0
```

因此位置 0 的 Q、K 不发生旋转。

## 4. 实验环境与配置

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 操作系统：WSL2 Ubuntu
- 深度学习框架：PyTorch
- Batch Size：1
- Hidden Size：1024
- Intermediate Size：4096
- Number of Heads：16
- Head Dimension：64
- Bias：关闭
- RoPE Base：10000
- RMSNorm epsilon：1e-6
- Sequence Length：1、16、128、512
- 数据类型：FP32、FP16、BF16
- Warm-up 次数：10
- 正式迭代次数：50
- 独立运行次数：3
- GPU 计时方式：CUDA Event

PyTorch、CUDA 和驱动版本可根据实际运行环境补充：

```text
PyTorch version:
PyTorch CUDA version:
NVIDIA driver version:
```

## 5. 参数量与理论参数显存

Decoder Block 包含：

```text
Attention 参数：4 × H²
SwiGLU FFN 参数：3 × H × I
两个 RMSNorm 参数：2 × H
```

因此参数总量为：

```text
参数量 = 4H² + 3HI + 2H
       = 4 × 1024² + 3 × 1024 × 4096 + 2 × 1024
       = 16,779,264
```

RoPE 的逆频率通过 `register_buffer(..., persistent=False)` 保存，不属于可训练参数，也不计入上述参数量。

不同数据类型的理论参数显存为：

| 数据类型 | 单参数字节数 | 参数显存（MiB） |
|:---:|---:|---:|
| FP32 | 4 | 64.0078 |
| FP16 | 2 | 32.0039 |
| BF16 | 2 | 32.0039 |

FP16 和 BF16 将理论参数显存降低约 50%。

## 6. 正确性验证

单元测试验证了以下性质：

1. RoPE 不改变 Q、K 的 Shape。
2. RoPE 旋转前后的向量范数保持一致。
3. 位置 0 的 Q、K 不发生旋转。
4. 相同向量处于不同位置时得到不同的旋转结果。
5. 显式传入 `position_ids` 时能够得到正确 Shape。
6. Decoder Block 输入输出 Shape 一致。
7. 参数量符合 `4H² + 3HI + 2H`。
8. Attention 和 FFN 权重均置零时，输出等于输入，验证残差路径正确。
9. 修改未来 token 不影响过去位置输出，验证完整 Decoder Block 仍满足因果性。
10. 非法 RoPE 维度和 Base 能够触发预期异常。

## 7. 实验方法

对于每种 Sequence Length 和数据类型：

1. 创建相同配置的 Transformer Decoder Block。
2. 创建输入隐藏状态和对应的 `position_ids`。
3. 执行 10 次预热。
4. 使用 CUDA Event 测量 50 次前向计算的平均延迟。
5. 根据 `sequence_length / latency` 计算模块 token 吞吐量。
6. 根据参数数量和数据类型计算理论参数显存。
7. 使用 PyTorch CUDA Memory API 记录计时区间内相对基线的额外峰值显存。
8. 独立运行三轮，报告均值与样本标准差。

当前 `tokens/s` 是单个 Decoder Block 处理隐藏状态的吞吐量，不是完整 LLM 每秒生成的 token 数。

## 8. 实验结果

延迟和吞吐量以三轮实验的“均值 ± 样本标准差”表示。参数量、理论参数显存和额外峰值显存在同一配置的三轮实验中保持一致。

| Sequence Length | 数据类型 | 延迟（ms） | 吞吐量（tokens/s） | 相对 FP32 吞吐 | 参数显存（MiB） | 额外峰值显存（MiB） | 输出形状 |
|---:|:---:|---:|---:|---:|---:|---:|:---:|
| 1 | FP32 | 0.8345 ± 0.0959 | 1,209.0 ± 138.5 | 1.00× | 64.0078 | 0.0625 | `(1, 1, 1024)` |
| 1 | FP16 | 0.9890 ± 0.3107 | 1,099.5 ± 421.8 | 0.91× | 32.0039 | 0.03125 | `(1, 1, 1024)` |
| 1 | BF16 | 0.9033 ± 0.2319 | 1,167.2 ± 351.7 | 0.97× | 32.0039 | 0.03125 | `(1, 1, 1024)` |
| 16 | FP32 | 0.8593 ± 0.0260 | 18,631.0 ± 557.9 | 1.00× | 64.0078 | 1.0 | `(1, 16, 1024)` |
| 16 | FP16 | 0.7118 ± 0.2680 | 24,369.5 ± 7,539.1 | 1.31× | 32.0039 | 0.5 | `(1, 16, 1024)` |
| 16 | BF16 | 0.5680 ± 0.0124 | 28,178.9 ± 609.7 | 1.51× | 32.0039 | 0.5 | `(1, 16, 1024)` |
| 128 | FP32 | 1.2425 ± 0.0380 | 103,084.0 ± 3,204.2 | 1.00× | 64.0078 | 8.0 | `(1, 128, 1024)` |
| 128 | FP16 | 0.9789 ± 0.1784 | 134,045.9 ± 27,033.3 | 1.30× | 32.0039 | 4.0 | `(1, 128, 1024)` |
| 128 | BF16 | 0.8314 ± 0.2163 | 160,273.5 ± 36,506.4 | 1.55× | 32.0039 | 4.0 | `(1, 128, 1024)` |
| 512 | FP32 | 3.4424 ± 0.0746 | 148,782.1 ± 3,262.2 | 1.00× | 64.0078 | 44.25 | `(1, 512, 1024)` |
| 512 | FP16 | 1.3416 ± 0.0386 | 381,854.7 ± 10,968.5 | 2.57× | 32.0039 | 44.25 | `(1, 512, 1024)` |
| 512 | BF16 | 1.3595 ± 0.0058 | 376,620.3 ± 1,600.7 | 2.53× | 32.0039 | 44.25 | `(1, 512, 1024)` |

## 9. 结果分析

### 9.1 完整 Decoder Block 保持正确的 Shape 与因果性

所有配置的输出形状均为 `(1, S, 1024)`，与输入形状一致。RoPE 范数保持、位置 0、显式位置编号、残差路径和 Decoder 因果性测试均用于验证完整数据流，而不仅是确认程序能够运行。

完整 Block 的参数量为 16,779,264，与理论公式一致。这意味着 Attention、FFN 和两个 RMSNorm 已被正确注册为子模块和模型参数。

### 9.2 低精度将参数显存降低一半

FP32 参数显存约为 64.01 MiB，FP16/BF16 约为 32.00 MiB。参数显存不会随序列长度改变，因为相同 Decoder Block 在不同输入长度下使用相同权重。

这一结果只代表参数存储，不包含 Attention Score、Softmax、FFN 中间激活、RoPE 三角函数结果和其他临时张量。

### 9.3 长序列下低精度吞吐优势明显

Sequence Length 为 512 时：

```text
FP32：148,782.1 tokens/s
FP16：381,854.7 tokens/s，约为 FP32 的 2.57 倍
BF16：376,620.3 tokens/s，约为 FP32 的 2.53 倍
```

此时 Attention 和 FFN 的矩阵计算规模较大，低精度计算和较小参数存储的优势能够得到更充分利用。FP16 与 BF16 的结果接近，其中 BF16 三轮延迟的标准差仅为 0.0058 ms，但不能据此推广到其他硬件或 Shape。

### 9.4 短序列下低精度不一定更快

Sequence Length 为 1 时，FP16/BF16 平均吞吐仅为 FP32 的 0.91 倍和 0.97 倍，即低精度并未带来稳定收益。

完整 Decoder Block 包含多个小规模操作：

- 两次 RMSNorm；
- Q、K、V 和输出投影；
- RoPE 角度、正弦和余弦计算；
- Causal Mask 构造；
- Attention Score 与 Softmax；
- 三个 FFN 投影；
- 两次残差加法；
- FP32 中间计算及数据类型转换。

在单 token Shape 下，矩阵计算规模不足以主导总延迟，Kernel Launch、类型转换和逐元素计算等固定开销占比较高。因此“低精度参数更小”并不等于“所有输入规模下延迟都更低”。

### 9.5 部分 FP16/BF16 测量存在较大波动

Sequence Length 为 1 的 FP16 延迟为：

```text
0.9890 ± 0.3107 ms
```

Sequence Length 为 128 的 BF16 延迟为：

```text
0.8314 ± 0.2163 ms
```

原始数据中，同一配置的个别轮次明显变慢。可能原因包括笔记本 GPU 动态频率、温度、功耗模式、后台负载、测试顺序及短 Kernel 的调度波动。

因此，短序列和高标准差配置只能用于观察趋势，不适合对小幅性能差异做强结论。后续可增加运行次数并报告中位数、P50 和 P95。

### 9.6 额外峰值显存随序列长度增加

Sequence Length 从 1 增加至 128 时，额外峰值显存表现出明显的精度差异：

```text
S = 1：FP32 0.0625 MiB，FP16/BF16 0.03125 MiB
S = 16：FP32 1 MiB，FP16/BF16 0.5 MiB
S = 128：FP32 8 MiB，FP16/BF16 4 MiB
```

这些配置中低精度额外峰值约为 FP32 的一半，说明中间激活存储受到元素字节数影响。

当 Sequence Length 为 512 时，三种精度的额外峰值均为 44.25 MiB。一个重要原因是当前 Attention 会显式构造 `S × S` Score，并将 Score 转换为 FP32 执行 Softmax；RoPE、Mask、Softmax、Attention 权重和 FFN 中间结果的生命周期与 CUDA 内存分配粒度也会共同影响峰值。

因此，该结果不能解释为 FP16/BF16 完全没有节省激活显存，而应解释为：

> 在当前 Eager 实现和长度 512 的执行峰值处，FP32 Softmax 及其他同时存活的临时张量主导了额外峰值，使三种精度得到相同的测量值。

要验证具体是哪一个 Tensor 主导峰值，需要使用 PyTorch Profiler Memory Timeline 或逐步记录内存，而不能仅依靠一个峰值数字推断。

### 9.7 额外峰值显存不等于完整推理显存

Benchmark 在模型和输入已经分配、预热完成后记录基线：

```text
peak_extra_memory = peak_memory - baseline_memory
```

因此 `peak_extra_memory` 主要反映计时区间中新增加的临时内存，不包含完整参数显存，也不等于 `nvidia-smi` 中看到的进程显存。

另外，PyTorch CUDA 缓存分配器中的 reserved memory 与 allocated memory 含义不同。本实验使用的是 `memory_allocated`，报告结论不能直接推广到显卡物理显存总占用。

### 9.8 当前结果仍不是完整 LLM 推理性能

本实验只运行一个 Decoder Block，没有 Token Embedding、多层堆叠、Final RMSNorm、LM Head 和 Sampling，也没有 KV Cache。

尤其是 Sequence Length 为 1 时，当前 Attention 只能看到当前 token，并不能访问历史 K/V，因此不代表真实 Decode。真实增量解码需要：

```text
Query length = 1
KV length = 历史长度 + 1
```

后续需要构建多层 Decoder-only 模型，并用 KV Cache 比较 Prefill 和逐 token Decode。

## 10. 主要结论

1. RoPE 和完整 Pre-Norm Transformer Decoder Block 已完成，并通过 Shape、范数、参数量、残差和因果性测试。
2. 当前 Decoder Block 参数量为 16,779,264。
3. FP16/BF16 将理论参数显存从约 64 MiB 降低到约 32 MiB。
4. 低精度收益随序列长度增大而更加明显。
5. Sequence Length 为 512 时，FP16/BF16 吞吐约为 FP32 的 2.57 倍和 2.53 倍。
6. Sequence Length 为 1 时，固定开销和实验波动较大，低精度没有稳定加速。
7. 执行期间额外峰值显存随序列长度明显增加。
8. 长度 512 时三种精度得到相同额外峰值，说明峰值由 FP32 Softmax 和多个临时张量共同主导，不能只根据参数精度判断。
9. 当前 `tokens/s` 是单个 Decoder Block 的隐藏状态处理吞吐，不是文本生成吞吐。
10. 当前实现尚无 KV Cache，不能代表真实增量 Decode 性能。

## 11. 实验局限

1. 只在一张 RTX 4060 Laptop GPU 上测试。
2. 只测试一个 Decoder Block，没有堆叠完整模型。
3. Attention 为 PyTorch Eager 实现，不是融合 Kernel。
4. 每次前向都会计算 RoPE 的三角函数并构造 Causal Mask。
5. Softmax 显式转换到 FP32。
6. 尚未实现 GQA/MQA 和 KV Cache。
7. Batch Size 固定为 1。
8. 只进行了三轮独立实验。
9. 未控制和记录 GPU 频率、温度及功耗。
10. 额外峰值显存不能定位具体 Tensor 或 Kernel。

## 12. 后续改进

1. 加入 Token Embedding、多层 Decoder、Final RMSNorm 和 LM Head。
2. 实现 Greedy 自回归生成。
3. 实现 KV Cache，验证完整序列与增量解码 logits 一致性。
4. 缓存 RoPE 的 Cos/Sin，避免每次前向重复计算。
5. 缓存或避免显式构造 Causal Mask。
6. 使用 PyTorch Profiler 分析 Decoder Block 的 Kernel 和内存时间线。
7. 对比手写 Attention 与 PyTorch SDPA 的性能和显存。
8. 后续实现融合 RMSNorm CUDA Kernel。
