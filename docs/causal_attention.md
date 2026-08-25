# Causal Multi-Head Self-Attention 性能实验

## 1. 实验目标

本实验使用 PyTorch Eager 模式实现带因果掩码的多头自注意力（Causal Multi-Head Self-Attention），并分析序列长度和数据类型对推理延迟、吞吐量、模型参数显存及 Attention Score 中间张量显存的影响。

实验目标包括：

1. 理解 Q、K、V 投影、多头拆分、缩放点积注意力和多头合并的完整数据流。
2. 使用因果掩码保证当前位置不能访问未来 token。
3. 将手写实现与 PyTorch `scaled_dot_product_attention` 参考实现进行正确性对比。
4. 验证 Attention Score 元素数量和理论显存随序列长度平方增长。
5. 比较 FP32、FP16 和 BF16 在不同序列长度下的性能表现。
6. 为后续加入 RoPE、KV Cache 和完整 Transformer Decoder Block 建立基线。

## 2. 计算原理

输入隐藏状态的形状为：

```text
X: (B, S, H)
```

首先执行 Q、K、V 线性投影：

```text
Q = XWq
K = XWk
V = XWv
```

将隐藏维度拆分为多个 Attention Head：

```text
(B, S, H)
-> (B, S, num_heads, head_dim)
-> (B, num_heads, S, head_dim)
```

其中：

```text
head_dim = hidden_size / num_heads
```

缩放点积注意力计算为：

```text
Attention(Q, K, V)
    = Softmax(QK^T / sqrt(head_dim) + CausalMask)V
```

Attention Score 的形状为：

```text
(B, num_heads, S, S)
```

因此，Score 元素数量随序列长度呈平方增长：

```text
score_elements = B × num_heads × S²
```

## 3. 因果掩码

对于长度为 4 的序列，因果掩码允许的访问关系为：

```text
        Key position
        0  1  2  3
Query 0 ✓  ×  ×  ×
      1 ✓  ✓  ×  ×
      2 ✓  ✓  ✓  ×
      3 ✓  ✓  ✓  ✓
```

测试中修改位置 4～7 的输入后，位置 0～3 的输出保持不变，说明未来 token 不会影响过去位置的输出。

## 4. 实验环境与配置

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 操作系统：WSL2 Ubuntu
- 深度学习框架：PyTorch
- Batch Size：1
- Hidden Size：1024
- Number of Heads：16
- Head Dimension：64
- Bias：关闭
- Sequence Length：1、16、128、512
- 数据类型：FP32、FP16、BF16
- Warm-up 次数：10
- 正式迭代次数：50
- 独立运行次数：3
- GPU 计时方式：CUDA Event
- Attention 实现：手写 PyTorch Eager 实现

PyTorch、CUDA 和驱动版本可根据实际运行环境补充：

```text
PyTorch version:
PyTorch CUDA version:
NVIDIA driver version:
```

## 5. 参数量与理论参数显存

当前实现使用四个 `H × H` 权重矩阵：

```text
query_proj  = H × H
key_proj    = H × H
value_proj  = H × H
output_proj = H × H
```

参数量为：

```text
参数量 = 4 × H²
       = 4 × 1024²
       = 4,194,304
```

不同数据类型下的理论参数显存为：

| 数据类型 | 单参数字节数 | 参数显存（MiB） |
|:---:|---:|---:|
| FP32 | 4 | 16 |
| FP16 | 2 | 8 |
| BF16 | 2 | 8 |

参数量与序列长度无关，但 Attention Score 等中间张量会随序列长度增长。

## 6. 正确性验证

单元测试覆盖以下性质：

1. Attention 输入输出形状均为 `(B, S, H)`。
2. `head_dim = hidden_size / num_heads`。
3. 参数量符合 `4 × H²`。
4. 手写 Attention 输出与 PyTorch SDPA 参考实现一致。
5. 修改未来 token 不会改变过去位置的输出。
6. 修改后的未来位置输出能够发生变化，证明因果性测试有效。
7. 非法 Hidden Size、Head 数量及输入形状能够触发预期异常。

## 7. 实验方法

对于每种 Sequence Length 和数据类型：

1. 创建相同配置的 Causal Self-Attention 模块。
2. 执行 10 次预热。
3. 使用 CUDA Event 测量 50 次前向计算的平均延迟。
4. 根据 `sequence_length / latency` 计算模块 token 吞吐量。
5. 根据数据类型计算参数和 Attention Score 的理论显存。
6. 独立运行三轮，并报告均值与样本标准差。

当前 `tokens/s` 只表示单个 Attention 模块处理隐藏状态的吞吐量，并不是完整 LLM 的生成速度。

## 8. 实验结果

下表中的延迟和吞吐量为三轮实验的“均值 ± 样本标准差”。Score 显存、Softmax FP32 存储量及参数显存在三轮实验中为固定理论值。

| Sequence Length | 数据类型 | 延迟（ms） | 吞吐量（tokens/s） | Score 元素数 | Score 理论显存（MiB） | Softmax FP32 存储量（MiB） | 参数显存（MiB） |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | FP32 | 0.2777 ± 0.0637 | 3,751.9 ± 988.9 | 16 | 0.000061 | 0.000061 | 16 |
| 1 | FP16 | 0.2381 ± 0.0242 | 4,226.6 ± 405.9 | 16 | 0.000031 | 0.000061 | 8 |
| 1 | BF16 | 0.3816 ± 0.1884 | 3,311.1 ± 2,134.7 | 16 | 0.000031 | 0.000061 | 8 |
| 16 | FP32 | 0.2593 ± 0.0885 | 65,991.5 ± 18,846.8 | 4,096 | 0.015625 | 0.015625 | 16 |
| 16 | FP16 | 0.2739 ± 0.0773 | 61,233.0 ± 14,934.5 | 4,096 | 0.007812 | 0.015625 | 8 |
| 16 | BF16 | 0.2287 ± 0.0128 | 70,090.3 ± 3,873.2 | 4,096 | 0.007812 | 0.015625 | 8 |
| 128 | FP32 | 0.3693 ± 0.0093 | 346,747.2 ± 8,808.1 | 262,144 | 1.0 | 1.0 | 16 |
| 128 | FP16 | 0.2650 ± 0.0156 | 484,069.0 ± 28,257.8 | 262,144 | 0.5 | 1.0 | 8 |
| 128 | BF16 | 0.3176 ± 0.1273 | 441,473.1 ± 144,070.1 | 262,144 | 0.5 | 1.0 | 8 |
| 512 | FP32 | 1.6918 ± 0.1107 | 303,465.7 ± 19,145.4 | 4,194,304 | 16.0 | 16.0 | 16 |
| 512 | FP16 | 0.7724 ± 0.0187 | 663,093.9 ± 16,266.5 | 4,194,304 | 8.0 | 16.0 | 8 |
| 512 | BF16 | 0.8448 ± 0.1183 | 613,462.7 ± 79,478.9 | 4,194,304 | 8.0 | 16.0 | 8 |

### 8.1 低精度相对于 FP32 的平均吞吐变化

| Sequence Length | FP16 / FP32 | BF16 / FP32 |
|---:|---:|---:|
| 1 | 1.13× | 0.88× |
| 16 | 0.93× | 1.06× |
| 128 | 1.40× | 1.27× |
| 512 | 2.19× | 2.02× |

## 9. 结果分析

### 9.1 Attention Score 显存呈平方增长

在 Batch Size 为 1、Head 数为 16 时：

```text
S = 16：   4,096 个 Score 元素
S = 128：  262,144 个 Score 元素
S = 512：  4,194,304 个 Score 元素
```

Sequence Length 从 16 增长到 128，即增长 8 倍，Score 元素数增长 64 倍；从 128 增长到 512，即增长 4 倍，Score 元素数增长 16 倍。这与 `O(S²)` 的理论一致。

FP32 Score 理论显存从 Sequence Length 16 时的 0.015625 MiB 增长到 Sequence Length 512 时的 16 MiB。对于更长上下文或更多 Batch，普通 Attention 显式构造 Score Tensor 会带来显著的显存压力。

### 9.2 参数显存与中间激活显存具有不同增长规律

Attention 参数量固定为 4,194,304，与 Sequence Length 无关。FP32、FP16/BF16 参数显存分别为 16 MiB 和 8 MiB。

相比之下，Score Tensor 由输入长度决定，随 `S²` 增长。因此分析大模型推理显存时，必须区分：

- 模型参数显存；
- Q/K/V 等线性增长的激活显存；
- Attention Score 等平方增长的中间显存；
- 后续自回归推理中的 KV Cache 显存。

### 9.3 低精度在长序列下收益更明显

Sequence Length 为 512 时：

```text
FP32：303,465.7 tokens/s
FP16：663,093.9 tokens/s，约为 FP32 的 2.19 倍
BF16：613,462.7 tokens/s，约为 FP32 的 2.02 倍
```

此时矩阵乘法和 Attention Score 计算规模较大，FP16/BF16 的参数及 Score Tensor 存储量减半，低精度计算路径也能得到更充分利用。

Sequence Length 为 1 或 16 时，低精度没有表现出稳定优势。小规模计算中，Linear、Mask 构造、Softmax、类型转换和多个 Kernel Launch 的固定开销占比较高，矩阵计算本身不足以主导总延迟。

### 9.4 Softmax FP32 转换会削弱部分低精度显存和性能收益

当前实现使用：

```python
attention_weights = torch.softmax(
    attention_scores.float(),
    dim=-1,
).to(query.dtype)
```

FP16/BF16 Score 在执行 Softmax 前会转换为 FP32。例如 Sequence Length 为 512 时：

```text
低精度 Score 理论存储量：8 MiB
对应 FP32 Softmax Tensor：16 MiB
```

此外，Softmax 结果还会转换回输入数据类型。当前表格中的 Softmax FP32 存储量只是该 FP32 Tensor 的理论大小，不等于整个模块的实际峰值显存，也不能简单与 Score 显存相加；当输入已经是 FP32 时，`.float()` 也不一定产生新的物理副本。

这说明仅将输入和权重转换为低精度，并不能消除 Eager Attention 的所有中间 Tensor。FlashAttention 等融合实现的核心价值之一，就是避免将完整 `S × S` Score 和 Softmax 中间结果反复写入显存。

### 9.5 延迟没有严格按照 S² 增长

虽然 Score 计算复杂度为 `O(S²)`，实测总延迟并未在所有长度区间严格按平方增长。原因包括：

1. Q、K、V 和输出投影的计算复杂度约为 `O(SH²)`。
2. 短序列下 GPU 利用率较低，固定开销占比较高。
3. 序列变长后，更大的矩阵计算可以提高并行度。
4. 当前实现还包含动态创建 Mask、FP32 Softmax 和数据类型转换。

因此，理论复杂度描述的是渐近趋势，不能直接用来预测消费级 GPU 上每个具体 Shape 的实际延迟。

### 9.6 短序列和部分 BF16 结果波动明显

Sequence Length 为 1 时，BF16 延迟为：

```text
0.3816 ± 0.1884 ms
```

Sequence Length 为 128 时，BF16 吞吐量为：

```text
441,473.1 ± 144,070.1 tokens/s
```

原始数据中，BF16 在第三轮的 Sequence Length 128 和 512 测试出现明显变慢。可能影响因素包括笔记本 GPU 动态频率、温度、功耗限制、后台负载、固定测试顺序以及底层 Kernel 选择。

因此：

- 不应根据单轮结果断言 BF16 或 FP16 始终更快；
- 短序列下几个百分点的差异没有充分统计意义；
- 长度 512 的 FP16 结果波动较小，能更可靠地说明长序列低精度收益。

### 9.7 Sequence Length 1 不是完整的 Decode

当前实现没有 KV Cache。当输入 Sequence Length 为 1 时，Q、K、V 都只包含当前 token，无法访问历史 token。

真实 Decode 阶段应当使用：

```text
Query：当前 1 个 token
Key/Value：当前 token + 历史 KV Cache
```

因此，本实验的 `S=1` 只能表示极小规模 Attention Shape，不能作为真实 LLM Decode 性能。后续需要实现 KV Cache，并分别测量 Prefill 与逐 token Decode。

## 10. 主要结论

1. 手写 Causal Multi-Head Self-Attention 通过了 Shape、参数量、参考实现和因果性验证。
2. 当前配置下 Attention 参数量为 4,194,304。
3. FP16/BF16 将理论参数显存从 16 MiB 降低到 8 MiB。
4. Attention Score 元素数量与显存随 Sequence Length 平方增长。
5. 长序列下低精度收益明显，Sequence Length 512 时 FP16/BF16 吞吐分别约为 FP32 的 2.19 倍和 2.02 倍。
6. 短序列下固定开销和运行波动占比较高，低精度没有稳定优势。
7. FP32 Softmax 转换会产生额外数据转换和潜在中间存储开销。
8. 参数显存固定不变，但中间激活显存随输入长度增长。
9. 当前 `tokens/s` 是单模块吞吐量，不是完整 LLM 生成速度。
10. 当前实现尚无 KV Cache，不能代表真实 Decode 性能。

## 11. 实验局限

1. 只在一张 RTX 4060 Laptop GPU 上测试。
2. 使用手写 PyTorch Eager Attention，不是融合 Attention Kernel。
3. 每次前向都会重新构造 Causal Mask。
4. 尚未加入 RoPE、GQA/MQA 和 KV Cache。
5. Batch Size 固定为 1。
6. 只进行了三轮独立实验。
7. 测试顺序固定，可能存在顺序偏差。
8. 未记录 GPU 频率、温度与功耗。
9. 显存数据来自公式计算，不是实际峰值显存。
10. 未使用 PyTorch Profiler 或 Nsight 分析底层 Kernel。

## 12. 后续改进

1. 加入 RoPE 位置编码。
2. 将 Attention、两次 RMSNorm、FFN 和残差组合为完整 Decoder Block。
3. 实现 KV Cache，区分 Prefill 和 Decode。
4. 对比手写 Attention 与 PyTorch SDPA 的 GPU 性能。
5. 使用 PyTorch Profiler 分析 Mask、Softmax、Matmul 和类型转换开销。
6. 测量 `torch.cuda.max_memory_allocated()` 得到实际峰值显存。
7. 对比 MHA、GQA 和 MQA 的 KV Cache 显存。
8. 后续研究 FlashAttention 如何避免显式存储完整 Score Tensor。
