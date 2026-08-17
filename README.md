# QInferLab

A learning-oriented LLM quantization and inference optimization project.

# 项目结构
- `benchmarks/matmul.py`: basic CPU/GPU matrix multiplication benchmark
- `benchmarks/precision_matmul.py`: precision, latency, memory, and error comparison

## Current Progress

- [✅] Set up the WSL2 and PyTorch CUDA environment
- [✅] Implement a CPU/GPU matrix multiplication benchmark
- [✅] Measure GPU latency with CUDA Events
- [✅] Compare FP32, FP16, and BF16 matrix multiplication
- [ ] Implement Transformer inference
- [ ] Add KV Cache
- [ ] Implement weight-only quantization
- [ ] Develop custom CUDA kernels