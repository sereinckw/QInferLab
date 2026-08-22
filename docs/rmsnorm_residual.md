# RMSNorm 与 Pre-Norm FFN 子层性能实验

## 1. 实验目标

本实验在已实现的 SwiGLU 前馈网络基础上加入 RMSNorm 与残差连接，构建一个 Pre-Norm FFN 子层：

```text
output = input + FFN(RMSNorm(input))
```

实验目标包括：

1. 验证 RMSNorm、SwiGLU FFN 和残差连接的实现正确性。
2. 分析 RMSNorm、独立 FFN 和完整 Block 的 GPU 延迟。
3. 比较 FP32、FP16 和 BF16 下的吞吐量与参数显存。
4. 观察序列长度对模块开销占比和 GPU 利用率的影响。
5. 为后续加入 Self-Attention、组成完整 Transformer Decoder Block 做准备。

## 2. 计算结构

### 2.1 RMSNorm

对于隐藏向量 $x$，RMSNorm 的计算公式为：

```text
RMS(x) = sqrt(mean(x²) + eps)
RMSNorm(x) = x / RMS(x) × weight
```

RMSNorm 在输入最后一个维度上进行归一化，不减去均值，且不改变 Tensor 的形状：

```text
(B, S, H) -> (B, S, H)
```

当前实现将输入临时转换为 FP32 完成平方、归约和缩放，然后转换回原始数据类型，以提高低精度输入下的数值稳定性。

### 2.2 Pre-Norm FFN 子层

完整子层的计算过程为：

```text
residual = input
normalized = RMSNorm(input)
ffn_output = SwiGLU(normalized)
output = residual + ffn_output
```

SwiGLU FFN 的公式为：

```text
FFN(x) = Down(SiLU(Gate(x)) * Up(x))
```

## 3. 实验环境与配置

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 操作系统：WSL2 Ubuntu
- 深度学习框架：PyTorch
- Batch Size：1
- Hidden Size：1024
- Intermediate Size：4096
- Bias：关闭
- RMSNorm epsilon：1e-6
- Sequence Length：1、16、128、512
- 数据类型：FP32、FP16、BF16
- Warm-up 次数：10
- 正式迭代次数：100
- 独立运行次数：3
- GPU 计时方式：CUDA Event

PyTorch、CUDA 和驱动版本可根据实际运行环境补充：

```text
PyTorch version:
PyTorch CUDA version:
NVIDIA driver version:
```

## 4. 参数量与理论参数显存

SwiGLU FFN 包含三个权重矩阵，RMSNorm 包含一个长度为 `H` 的可学习权重，因此完整 Block 的参数量为：

```text
参数量 = 3 × H × I + H
       = 3 × 1024 × 4096 + 1024
       = 12,583,936
```

不同数据类型下的理论参数显存为：

| 数据类型 | 单参数字节数 | 参数显存（MiB） |
|:---:|---:|---:|
| FP32 | 4 | 48.0039 |
| FP16 | 2 | 24.0020 |
| BF16 | 2 | 24.0020 |

FP16 和 BF16 的理论参数显存均为 FP32 的一半。这里不包含输入、输出、中间激活、CUDA Context 和 PyTorch 缓存分配器占用，因此不能视为程序的实际峰值显存。

## 5. 正确性验证

单元测试对以下性质进行了验证：

1. RMSNorm 保持输入输出形状不变。
2. RMSNorm 输出与显式公式计算结果一致。
3. 使用单位缩放权重时，归一化结果最后一维的平方均值接近 1。
4. 完整 Block 保持输入输出形状不变。
5. Block 参数量符合 `3 × H × I + H`。
6. 将 FFN 参数全部置零后，Block 输出等于输入，证明残差路径实现正确。
7. 非法的 Hidden Size 和 epsilon 能够触发预期异常。

## 6. 实验方法

对于每种 Sequence Length 和数据类型，分别测量：

1. 独立 RMSNorm 延迟。
2. 独立 SwiGLU FFN 延迟。
3. 完整 Pre-Norm FFN Block 延迟。
4. 完整 Block 的 token 吞吐量。

每项数据均来自三轮独立运行，表中以“均值 ± 样本标准差”表示。

`Block overhead` 按下式计算：

```text
(Block latency - FFN latency) / FFN latency × 100%
```

`RMSNorm share` 按下式计算：

```text
RMSNorm latency / Block latency × 100%
```

需要注意，各模块是在不同计时区间中独立测量的。上述两个比例只用于粗略观察，不能视为严格的 Kernel 耗时分解。

## 7. 实验结果

| Sequence Length | 数据类型 | RMSNorm 延迟（ms） | FFN 延迟（ms） | Block 延迟（ms） | Block 吞吐量（tokens/s） | Block Overhead | RMSNorm Share |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | FP32 | 0.0494 ± 0.0022 | 0.2169 ± 0.0020 | 0.2372 ± 0.0039 | 4,215.8 ± 70.0 | 9.36% ± 0.84% | 20.82% ± 1.28% |
| 1 | FP16 | 0.0929 ± 0.0466 | 0.0741 ± 0.0060 | 0.1515 ± 0.0047 | 6,603.6 ± 208.9 | 105.08% ± 11.47% | 61.96% ± 33.19% |
| 1 | BF16 | 0.0709 ± 0.0060 | 0.1190 ± 0.0160 | 0.2246 ± 0.0644 | 4,742.6 ± 1,534.1 | 90.00% ± 58.85% | 32.99% ± 7.59% |
| 16 | FP32 | 0.0792 ± 0.0362 | 0.3411 ± 0.0106 | 0.3636 ± 0.0126 | 44,046.2 ± 1,553.7 | 6.58% ± 0.61% | 21.73% ± 9.61% |
| 16 | FP16 | 0.1217 ± 0.0901 | 0.0781 ± 0.0238 | 0.1807 ± 0.0783 | 99,395.4 ± 38,440.5 | 125.76% ± 28.42% | 62.05% ± 18.95% |
| 16 | BF16 | 0.1033 ± 0.0143 | 0.0701 ± 0.0073 | 0.1654 ± 0.0479 | 101,612.3 ± 25,250.9 | 136.20% ± 61.87% | 65.28% ± 17.63% |
| 128 | FP32 | 0.0510 ± 0.0025 | 0.4876 ± 0.0512 | 0.5475 ± 0.0460 | 234,869.0 ± 19,460.1 | 12.66% ± 8.76% | 9.36% ± 0.91% |
| 128 | FP16 | 0.0860 ± 0.0273 | 0.1657 ± 0.0081 | 0.2116 ± 0.0279 | 612,164.9 ± 81,353.1 | 27.59% ± 14.32% | 42.16% ± 18.93% |
| 128 | BF16 | 0.0985 ± 0.0190 | 0.1407 ± 0.0009 | 0.2349 ± 0.0863 | 588,966.4 ± 179,554.0 | 66.97% ± 61.70% | 43.56% ± 6.57% |
| 512 | FP32 | 0.0551 ± 0.0084 | 1.9008 ± 0.0398 | 2.0826 ± 0.0706 | 246,036.5 ± 8,376.9 | 9.62% ± 5.10% | 2.65% ± 0.46% |
| 512 | FP16 | 0.0839 ± 0.0094 | 0.5075 ± 0.0121 | 0.5813 ± 0.0157 | 881,251.4 ± 23,790.0 | 14.55% ± 2.62% | 14.48% ± 2.02% |
| 512 | BF16 | 0.1016 ± 0.0409 | 0.5472 ± 0.0130 | 0.5984 ± 0.0077 | 855,687.3 ± 10,913.0 | 9.38% ± 1.25% | 17.00% ± 6.94% |

### 7.1 低精度相对于 FP32 的完整 Block 吞吐提升

| Sequence Length | FP16 / FP32 | BF16 / FP32 |
|---:|---:|---:|
| 1 | 1.57× | 1.12× |
| 16 | 2.26× | 2.31× |
| 128 | 2.61× | 2.51× |
| 512 | 3.58× | 3.48× |

## 8. 结果分析

### 8.1 FP16 和 BF16 将参数显存降低一半

数据类型不会改变模型参数数量，但 FP16/BF16 每个参数只占 2 字节，因此完整子层的理论参数显存从约 48 MiB 降至约 24 MiB。

这说明降低权重精度可以直接减少模型存储需求。不过，本实验只是将整个模块转换到低精度，并不属于 INT8 或 INT4 权重量化。

### 8.2 长序列下低精度计算优势更加明显

Sequence Length 为 1 时，FP16 和 BF16 相对于 FP32 的完整 Block 吞吐提升分别只有 1.57 倍和 1.12 倍；Sequence Length 增加到 512 后，提升扩大到 3.58 倍和 3.48 倍。

短序列对应的计算规模较小，Kernel Launch、类型转换、归约和逐元素计算等固定开销占比较高。随着序列长度增大，FFN 中矩阵乘法的规模上升，低精度矩阵计算的吞吐优势得到更充分的发挥。

### 8.3 RMSNorm 参数很少，但运行开销不能只由参数量判断

RMSNorm 只有 1024 个可学习参数，远少于 FFN 的 12,582,912 个参数，但它仍需读取输入的全部 `(B, S, H)` 元素，并执行平方、归约、倒数平方根、缩放和写回。

因此，“参数量少”不等于“运行时间可以忽略”。对于 Sequence Length 为 1 的小规模计算，RMSNorm 和相关调度开销相对于 FFN 尤其明显。

### 8.4 当前低精度 RMSNorm 没有表现出延迟优势

实验中，FP16/BF16 RMSNorm 延迟多数不低于 FP32。例如 Sequence Length 为 512 时：

```text
FP32 RMSNorm：0.0551 ms
FP16 RMSNorm：0.0839 ms
BF16 RMSNorm：0.1016 ms
```

一个重要原因是当前实现显式执行了：

```python
hidden_states_fp32 = hidden_states.float()
...
normalized_states = normalized_states.to(input_dtype)
```

因此低精度输入需要先转换为 FP32，完成归约和缩放后再转换回原始类型。Eager PyTorch 还可能为平方、均值、倒数平方根、乘法和类型转换分别启动 Kernel。这些转换与中间 Tensor 开销会抵消低精度数据存储带来的收益。

这也为后续 CUDA RMSNorm 和算子融合实验提供了明确动机：减少中间 Tensor、数据类型转换和 Kernel Launch。

### 8.5 长序列下 FFN 是主要耗时来源

Sequence Length 为 512 时，独立 FFN 延迟远高于 RMSNorm 延迟：

```text
FP32：RMSNorm 0.0551 ms，FFN 1.9008 ms
FP16：RMSNorm 0.0839 ms，FFN 0.5075 ms
BF16：RMSNorm 0.1016 ms，FFN 0.5472 ms
```

随着序列长度增大，三个 Linear 层的矩阵乘法计算量快速增加，因此 FFN 逐渐成为完整子层的主要耗时部分。

### 8.6 短序列结果波动明显

Sequence Length 为 1 和 16 时，部分 FP16/BF16 数据的标准差较大。例如 Sequence Length 为 16 的 FP16 RMSNorm 延迟为：

```text
0.1217 ± 0.0901 ms
```

这类微秒级 Kernel 容易受到 GPU 动态频率、功耗、温度、后台负载、缓存状态和固定测试顺序的影响。三轮测试数量也不足以稳定估计尾部波动。

因此，短序列部分只能说明总体趋势，不适合对几个百分点的差异作强结论。

### 8.7 Overhead 和 RMSNorm Share 不是严格时间分解

部分 `Block overhead` 超过 100%，部分 `RMSNorm share` 甚至接近或超过 100%。这不表示 RMSNorm 在数学上占用了超过整个 Block 的执行时间。

出现该现象是因为：

1. RMSNorm、FFN 和完整 Block 在不同计时区间中分别运行。
2. 三段测试期间 GPU 频率和系统状态可能不同。
3. 完整 Block 还包含残差加法及额外 Kernel Launch。
4. Eager PyTorch 中各操作间存在中间 Tensor 和内存分配行为。
5. 对短 Kernel 做差会放大测量噪声。

因此，不能直接假设：

```text
Block latency = RMSNorm latency + FFN latency
```

若要获得可靠的算子时间分解，应使用 PyTorch Profiler 或 Nsight Systems 记录同一次 Block 前向过程中的 Kernel 时间线。

### 8.8 FP16 与 BF16 没有稳定的绝对胜负

Sequence Length 为 512 时，FP16 与 BF16 的完整 Block 吞吐量分别为约 881,251 和 855,687 tokens/s，FP16 略高；在 Sequence Length 为 16 时，BF16 略高于 FP16。

结合短序列下较大的标准差，目前只能认为两种低精度格式整体处于相近性能水平，不能断言其中一种在所有形状下始终更快。

## 9. 主要结论

1. RMSNorm、SwiGLU FFN 和残差连接已经组成一个正确运行的 Pre-Norm FFN 子层。
2. 完整子层参数量为 12,583,936，其中绝大多数参数位于三个 Linear 权重矩阵中。
3. FP16/BF16 将理论参数显存从约 48 MiB 降低到约 24 MiB。
4. 低精度在所有序列长度下都提高了完整 Block 的平均吞吐量。
5. 低精度收益随序列长度增大而更加明显，在 Sequence Length 为 512 时达到约 3.5 倍。
6. 长序列下 FFN 矩阵乘法是主要耗时来源。
7. RMSNorm 虽然参数量极少，但仍有不可忽略的数据读取、归约和 Kernel Launch 开销。
8. 当前 RMSNorm 中的 FP32 转换提高了数值稳定性，但也增加了低精度路径的执行开销。
9. 短序列数据波动较大，不能仅凭单次结果判断 FP16 与 BF16 的优劣。
10. 分开测量得到的 Overhead 和 Share 只能粗略参考，不能代替 Profiler 的同一次执行时间线。

## 10. 实验局限

1. 只在一张 RTX 4060 Laptop GPU 上进行测试。
2. 只测试单个 FFN 子层，不是完整 Transformer 或完整 LLM。
3. Batch Size 固定为 1。
4. 只进行了三轮独立实验。
5. 未记录 GPU 温度、频率和功耗。
6. 各模块采用独立计时，无法严格分解完整 Block 延迟。
7. 测试顺序固定，可能存在热机和频率变化带来的顺序偏差。
8. 只统计理论参数显存，没有测量实际峰值显存。
9. 未使用 PyTorch Profiler 或 Nsight 分析底层 Kernel。
10. 未比较不同 epsilon 和不同 RMSNorm 实现的数值误差。

## 11. 后续改进

1. 使用 PyTorch Profiler 分析一次完整 Block 前向中的 Kernel 时间线。
2. 将独立实验次数增加到 10 次，并报告中位数、P50 和 P95。
3. 随机化不同数据类型和序列长度的测试顺序。
4. 测量 `torch.cuda.max_memory_allocated()` 得到实际峰值显存。
5. 对比自定义 RMSNorm 与 `torch.nn.RMSNorm` 的正确性和性能。
6. 尝试 `torch.compile`，观察能否减少 Eager 模式下的 Kernel Launch 和中间 Tensor。
7. 后续使用 C++/CUDA 实现融合 RMSNorm Kernel。
8. 加入 Multi-Head Self-Attention，组成完整 Transformer Decoder Block。

