import argparse
import time
import torch

def benchmark_cpu(
    matrix_size:int,
    warmup_iterations:int,
    benchmark_iterations:int
)->float:
    """Measure the average CPU matrix multiplication latency  in milliseconds. """
    matrix_a = torch.randn(matrix_size,matrix_size,dtype=torch.float32)
    matrix_b = torch.randn(matrix_size,matrix_size,dtype=torch.float32)

    for _ in range(warmup_iterations):
        _=matrix_a@matrix_b

    start_time=time.perf_counter()

    for _ in range(benchmark_iterations):
        _=matrix_a@matrix_b
    
    end_time=time.perf_counter()

    total_time=end_time-start_time
    average_time_ms=total_time*1000/benchmark_iterations

    return average_time_ms

def benchmark_gpu(
    matrix_size:int,
    warmup_iterations:int,
    benchmark_iterations:int,
)->float:
    """Measure the average GPU matrix multiplication latency  in milliseconds. """
    matrix_a=torch.randn(
        matrix_size,
        matrix_size,
        device="cuda",
        dtype=torch.float32,
    )

    matrix_b=torch.randn(
        matrix_size,
        matrix_size,
        device="cuda",
        dtype=torch.float32,
    )
    
    for _ in range(warmup_iterations):
        _=matrix_a@matrix_b

    torch.cuda.synchronize()
    start_event=torch.cuda.Event(enable_timing=True)
    end_event=torch.cuda.Event(enable_timing=True)

    start_event.record()

    for _ in range(benchmark_iterations):
        _=matrix_a@matrix_b
    
    end_event.record()
    end_event.synchronize()
    total_time_ms=start_event.elapsed_time(end_event)
    average_time_ms=total_time_ms/benchmark_iterations
    return average_time_ms

def calculate_tflops(matrix_size:int,latency_ms:float)->float:
    """Estimate throughput for square matrix multiplication"""
    number_of_operations=2*matrix_size**3
    latency_seconds=latency_ms/1000
    return number_of_operations/latency_seconds/1e12

def parse_arguments()->argparse.Namespace:
    parser=argparse.ArgumentParser(
        description="Benchmark square matrix multiplication on CPU and GPU."

    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Width an height of each square matrix.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up iterations.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of measured iterations.",
    )
    return parser.parse_args()

def main()->None:
    args=parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current environment.")
    
    print(f"Pytorch version:{torch.__version__}")
    print(f"GPU:{torch.cuda.get_device_name(0)}")
    print(f"Matrix shape: ({args.size},{args.size})")
    print(f"Data type:{torch.float32}")
    print()

    cpu_latency_ms=benchmark_cpu(
        matrix_size=args.size,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
    )
    gpu_latency_ms=benchmark_gpu(
        matrix_size=args.size,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
    )
    gpu_tflops=calculate_tflops(
        matrix_size=args.size,
        latency_ms=gpu_latency_ms,
    )

    print(f"CPU average latency:{cpu_latency_ms:.3f}ms")
    print(f"GPU average latency:{gpu_latency_ms:.3f}ms")
    print(f"GPU estimated throughput:{gpu_tflops:.3f} TFLOPS")
    print(f"Observed speedup:{cpu_latency_ms/gpu_latency_ms:.2f}x")

if __name__=="__main__":
    main()