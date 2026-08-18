# SwiGLU 前馈网络性能实验

## 1. 实验目标

本实验实现一个基于 PyTorch 的 SwiGLU 前馈网络，并在 NVIDIA GeForce RTX 4060 Laptop GPU 上测试不同序列长度和数据类型对推理性能的影响。

实验重点包括：

1. 理解 SwiGLU 前馈网络的计算结构。
2. 分析 FP32、FP16 和 BF16 的参数显存差异。
3. 比较不同数据类型下的推理延迟和吞吐量。
4. 观察短序列与长序列对 GPU 利用率的影响。
5. 为后续 Transformer Decoder Block 和大模型推理优化实验建立基础。

## 2. 网络结构

实验中的前馈网络使用如下 SwiGLU 结构：

```text
FFN(x) = Down(SiLU(Gate(x)) ⊙ Up(x))
```

其中：

- `Gate`：将隐藏维度从 `H` 投影到中间维度 `I`。
- `Up`：将隐藏维度从 `H` 投影到中间维度 `I`。
- `SiLU`：非线性激活函数。
- `⊙`：逐元素乘法。
- `Down`：将中间维度从 `I` 投影回隐藏维度 `H`。

数据形状变化如下：

```text
输入：
(B, S, H)

Gate 和 Up 投影后：
(B, S, I)

逐元素相乘后：
(B, S, I)

Down 投影后：
(B, S, H)
```

其中：

- `B`：Batch Size。
- `S`：Sequence Length。
- `H`：Hidden Size。
- `I`：Intermediate Size。

## 3. 实验环境

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 操作系统：WSL2 Ubuntu
- 深度学习框架：PyTorch
- Batch Size：1
- Hidden Size：1024
- Intermediate Size：4096
- Bias：关闭
- 激活函数：SiLU
- 测试数据类型：FP32、FP16、BF16
- Sequence Length：1、16、128、512
- Warm-up 次数：10
- 正式迭代次数：50
- 独立实验次数：3

PyTorch 版本和 CUDA 版本可根据实际环境补充：

```text
PyTorch version:
PyTorch CUDA version:
NVIDIA driver version:
```

## 4. 参数量与理论参数显存

SwiGLU 前馈网络包含三个权重矩阵：

```text