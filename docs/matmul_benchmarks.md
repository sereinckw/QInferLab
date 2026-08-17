# Matrix Multiplication Benchmark

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Operating system: WSL2 Ubuntu
- PyTorch version:2.5.1+cu121
- PyTorch CUDA version:
- Data type: FP32

## Method

The benchmark compares square matrix multiplication on the CPU and GPU.

GPU latency is measured with CUDA Events. Each experiment includes warm-up iterations before the measured iterations.

## Results

| Matrix size | CPU latency (ms) | GPU latency (ms) | Speedup | GPU TFLOPS |
|---:|---:|---:|---:|---:|
| 512 | 1.006 | 0.052 | 19.39 | 5.177 |
| 1024 | 7.631 | 0.341 | 22.35 | 6.290 |
| 2048 | 56.086 | 2.483 | 22.59 | 6.920 |
| 4096 | 430.488 | 19.624 | 21.94 | 7.003 |

## Observations

1. 重复运行结果存在一定波动
2. 矩阵增大后 GPU 吞吐上升
3. 小矩阵无法充分利用 GPU

## Questions

1. Why does GPU execution require synchronization for accurate timing?
2. Why is warm-up necessary?
3. Why does the observed GPU throughput change with matrix size?