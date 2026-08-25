# QInferLab

QInferLab 是一个面向学习与实验的大语言模型量化、推理和 GPU 性能优化项目。

项目从可验证的矩阵乘法基线开始，逐步实现 Decoder-only Transformer 的核心组件，并通过单元测试、CUDA Event、多轮 Benchmark 和实验报告，分析不同精度、序列长度及实现方式对延迟、吞吐量、数值误差和显存的影响。

当前项目处于 Transformer 核心组件构建阶段，所有性能结论均限定在本仓库的实验硬件、Tensor Shape 和实现方式下。

## 项目目标

- 从零实现 Decoder-only Transformer 的核心推理模块。
- 理解 Prefill、Decode 和 KV Cache 的数据流与性能特征。
- 实现 INT8 weight-only 量化并分析精度、存储和性能权衡。
- 使用 C++/CUDA Extension 开发并优化自定义算子。
- 在真实小型 LLM 上建立 TTFT、TPOT、吞吐量和显存评测。
- 保持代码、测试、实验数据与技术结论可复现、可解释。

## 当前已实现

- CPU/GPU 矩阵乘法 Benchmark。
- 基于 CUDA Event 的 GPU 正确计时。
- FP32、FP16、BF16 矩阵乘法延迟、吞吐量、存储和误差对比。
- SwiGLU Feed-Forward Network。
- RMSNorm。
- Pre-Norm + FFN + Residual 子层。
- 带因果掩码的 Multi-Head Self-Attention。
- Attention 与 PyTorch SDPA 参考实现的正确性对比。
- 因果掩码测试：未来 token 不影响过去位置输出。
- Attention Score 的 `O(S²)` 元素数量和理论显存分析。
- 模块级正确性测试、性能测试和中文实验报告。

## 当前实验结论

- FP16/BF16 的参数和矩阵理论存储量通常为 FP32 的一半。
- 短序列下 Kernel Launch、归约、Mask 和类型转换等固定开销占比较高，低精度不一定更快。
- 长序列下低精度矩阵计算优势更明显。
- 在本项目的 Causal Attention 实验中，Sequence Length 为 512 时，FP16/BF16 吞吐分别约为 FP32 的 2.19 倍和 2.02 倍。
- Attention Score 元素数量随序列长度平方增长，但模型参数量不随序列长度改变。
- 当前 RMSNorm 和 Attention 均在部分运算中使用 FP32 中间结果，以换取更好的数值稳定性，同时也引入类型转换和中间 Tensor 开销。

这些吞吐量是单个实验模块处理隐藏状态的吞吐量，不是完整 LLM 的文本生成速度。

## 项目结构

```text
QInferLab/
├── qinferlab/
│   └── layers.py                 # Attention、RMSNorm、SwiGLU 等核心层
├── benchmarks/
│   ├── matmul.py                 # CPU/GPU 矩阵乘法基线
│   ├── precision_matmul.py       # FP32/FP16/BF16 矩阵乘法实验
│   ├── feed_forward.py           # SwiGLU FFN 性能实验
│   ├── rmsnorm_block.py          # RMSNorm 与 Pre-Norm FFN Block 实验
│   └── causal_attention.py       # Causal Self-Attention 性能实验
├── tests/
│   ├── test_feed_forward.py      # FFN 正确性测试
│   ├── test_rms_norm.py          # RMSNorm 与残差连接测试
│   └── test_attention.py         # Attention 参考实现与因果性测试
├── docs/
│   ├── matmul_benchmarks.md      # 矩阵乘法实验报告
│   ├── precision_matmul.md       # 低精度矩阵乘法报告
│   ├── feed_forward.md           # SwiGLU FFN 实验报告
│   ├── rmsnorm_residual.md       # RMSNorm 与残差连接报告
│   └── causal_attention.md       # Causal Attention 实验报告
└── README.md
```

## 环境与运行

主要实验环境：

- WSL2 Ubuntu
- NVIDIA GeForce RTX 4060 Laptop GPU
- Conda 环境：`pytorch`
- PyTorch CUDA

进入项目并激活环境：

```bash
cd ~/projects/QInferLab
conda activate pytorch
```

运行全部单元测试：

```bash
python -m pytest -v
```

运行各项性能实验：

```bash
python -m benchmarks.precision_matmul
python -m benchmarks.feed_forward
python -m benchmarks.rmsnorm_block
python -m benchmarks.causal_attention
```

实验 CSV 默认保存在 `artifacts/`。生成的 CSV、模型权重和其他大型文件不提交到 GitHub。

## 实验文档

- [矩阵乘法基线](docs/matmul_benchmarks.md)
- [FP32、FP16 与 BF16 矩阵乘法](docs/precision_matmul.md)
- [SwiGLU FFN](docs/feed_forward.md)
- [RMSNorm 与 Pre-Norm FFN 子层](docs/rmsnorm_residual.md)
- [Causal Multi-Head Self-Attention](docs/causal_attention.md)

## 项目进度

- [x] 配置 WSL2 与 PyTorch CUDA 环境
- [x] 实现 CPU/GPU 矩阵乘法基线
- [x] 使用 CUDA Event 正确测量 GPU 延迟
- [x] 比较 FP32、FP16、BF16 的性能与误差
- [x] 实现并测试 SwiGLU FFN
- [x] 实现并测试 RMSNorm 与残差连接
- [x] 实现 Causal Multi-Head Self-Attention
- [x] 验证 Attention 因果性及与 PyTorch 参考实现的一致性
- [x] 分析 Attention Score 的平方显存增长
- [ ] 实现 RoPE 位置编码
- [ ] 组合完整 Transformer Decoder Block
- [ ] 实现最小 Decoder-only Transformer 和自回归生成
- [ ] 实现 KV Cache，并比较 Prefill/Decode 性能
- [ ] 实现 INT8 weight-only 量化
- [ ] 开发并优化自定义 CUDA Kernel
- [ ] 在真实小型 LLM 上完成部署与端到端 Benchmark

## 当前局限

- 当前 Attention 为手写 PyTorch Eager 实现，不是融合 Kernel。
- 尚未加入 RoPE、GQA/MQA 和 KV Cache。
- 当前 `sequence_length=1` 不能代表带历史 KV Cache 的真实 Decode。
- 当前显存分析部分来自理论公式，尚未系统测量实际峰值显存。
- 当前实验主要使用 Batch Size 1 和单张消费级 GPU。
- 笔记本 GPU 受动态频率、温度和功耗影响，短 Kernel 测量存在波动。

## 后续路线

```text
RoPE
  -> 完整 Decoder Block
  -> 最小 Decoder-only Transformer
  -> 自回归生成
  -> KV Cache 与 Prefill/Decode Benchmark
  -> INT8 Weight-only 量化
  -> C++/CUDA 自定义算子
  -> 真实小型 LLM 与推理服务 Benchmark
```
