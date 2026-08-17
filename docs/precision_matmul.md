# Precision Matrix Multiplication Benchmark

## Goal

Compare FP32, FP16, and BF16 matrix multiplication on an RTX 4060
Laptop GPU.

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Operating system: WSL2 Ubuntu
- PyTorch version:
- PyTorch CUDA version:
- Power mode:

## Method

For each matrix size, the benchmark:

1. Creates identical FP32 input matrices.
2. Uses FP32 output as the numerical reference.
3. Converts the inputs to FP16 or BF16.
4. Performs warm-up iterations.
5. Measures GPU latency with CUDA Events.
6. Computes throughput and numerical error.

## Results

## Results

The reported latency and throughput values are the mean ± sample standard
deviation across three benchmark runs.

| Matrix size | Data type | Latency (ms) | Throughput (TFLOPS) | Tensor memory (MiB) | Max absolute error | Relative L2 error |
|---:|:---:|---:|---:|---:|---:|---:|
| 512 | FP32 | 0.0892 ± 0.0495 | 3.565 ± 1.499 | 3.0 | 0.000000 | 0 |
| 512 | FP16 | 0.0249 ± 0.0012 | 10.799 ± 0.518 | 1.5 | 0.045601 | 3.602e-4 |
| 512 | BF16 | 0.0245 ± 0.0009 | 10.963 ± 0.386 | 1.5 | 0.361198 | 2.871e-3 |
| 1024 | FP32 | 0.3429 ± 0.0026 | 6.262 ± 0.047 | 12.0 | 0.000000 | 0 |
| 1024 | FP16 | 0.1185 ± 0.0010 | 18.124 ± 0.155 | 6.0 | 0.090759 | 3.884e-4 |
| 1024 | BF16 | 0.1183 ± 0.0013 | 18.156 ± 0.195 | 6.0 | 0.813538 | 3.110e-3 |
| 2048 | FP32 | 2.4484 ± 0.0592 | 7.020 ± 0.172 | 48.0 | 0.000000 | 0 |
| 2048 | FP16 | 0.6996 ± 0.0727 | 24.724 ± 2.427 | 24.0 | 0.102814 | 3.592e-4 |
| 2048 | BF16 | 0.6270 ± 0.0461 | 27.500 ± 2.017 | 24.0 | 0.804306 | 2.873e-3 |
| 4096 | FP32 | 17.6263 ± 0.2844 | 7.799 ± 0.126 | 192.0 | 0.000000 | 0 |
| 4096 | FP16 | 4.6040 ± 0.0577 | 29.855 ± 0.377 | 96.0 | 0.176147 | 3.599e-4 |
| 4096 | BF16 | 4.4818 ± 0.0727 | 30.671 ± 0.493 | 96.0 | 1.274231 | 2.878e-3 |

### Initial observations

1. FP16 and BF16 require exactly half the theoretical tensor storage of FP32.
2. For the 4096 × 4096 case, FP16 and BF16 achieve approximately four times the throughput of FP32.
3. Throughput generally increases with matrix size, suggesting that larger workloads utilize the GPU more effectively.
4. FP16 has a lower relative numerical error than BF16 in these experiments.
5. BF16 is slightly faster than FP16 for the two largest matrix sizes, although the difference is small relative to run-to-run variation.
6. The third FP32 measurement at matrix size 512 is an outlier, producing a much higher latency than the first two runs. Small workloads are more sensitive to scheduling, power-state, and measurement noise.
7. FP32 reports zero error because the FP32 computation itself is used as the reference output; this does not mean FP32 computation has no numerical error relative to exact arithmetic.


## Current explanation

Low-precision data types reduce the number of bytes required to store
matrix elements. They may also use specialized GPU hardware for higher
throughput. However, converting values to lower precision introduces
numerical error.

## Open questions

1. Why do small matrices underutilize the GPU?
2. What is a Tensor Core?
3. What is the difference between FP16 and BF16?
4. Why does lower precision not always produce lower latency?