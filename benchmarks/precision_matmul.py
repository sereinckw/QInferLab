import argparse
import csv
from pathlib import Path
import torch

DTYPES={
    "float32":torch.float32,
    "float16":torch.float16,
    "bfloat16":torch.bfloat16,
}

def benchmark_matmul(
    matrix_size:int,
    dtype_name:str,
    warmup_iterations:int,
    benchmark_iterations:int,
)->dict:
    """Benchmark one matrix size and one type on the GPU."""
    dtype=DTYPES[dtype_name]
    torch.manual_seed(0)
    matrix_a_fp32=torch.randn(
        matrix_size,
        matrix_size,
        device="cuda",
        dtype=torch.float32,
    )
    matrix_b_fp32=torch.randn(
        matrix_size,
        matrix_size,
        device="cuda",
        dtype=torch.float32,
    )
    with torch.inference_mode():
        reference_output=matrix_a_fp32@matrix_b_fp32
        matrix_a=matrix_a_fp32.to(dtype)
        matrix_b=matrix_b_fp32.to(dtype)

        for _ in range(warmup_iterations):
            _=matrix_a@matrix_b
        
        torch.cuda.synchronize()
        start_event=torch.cuda.Event(enable_timing=True)
        end_event=torch.cuda.Event(enable_timing=True)
        start_event.record()
        output=None

        for _ in range(benchmark_iterations):
            output=matrix_a@matrix_b
        
        end_event.record()
        torch.cuda.synchronize()
        
        total_time_ms=start_event.elapsed_time(end_event)
        average_latency=total_time_ms/benchmark_iterations

        output_fp32=output.float()

        absolute_difference=(output_fp32-reference_output).abs()
        max_absolute_error=absolute_difference.max().item()

        reference_norm=torch.linalg.vector_norm(reference_output)
        error_norm=torch.linalg.vector_norm(output_fp32-reference_output)
        relative_l2_error=(error_norm/reference_norm).item()

    number_of_operations=2*matrix_size**3
    latency_seconds=average_latency/1000
    estimated_tflops=number_of_operations/latency_seconds/1e12

    btype_per_element=torch.empty([],dtype=dtype).element_size()

    tensor_memory_mib=(
        3*matrix_size*matrix_size*btype_per_element
    )/(1024**2)

    return{
        "matrix_size":matrix_size,
        "dtype":dtype_name,
        "latency_ms":average_latency,
        "estimated_tflops":estimated_tflops,
        "tensor_memory_mib":tensor_memory_mib,
        "max_absolute_error":max_absolute_error,
        "relative_l2_error":relative_l2_error,
    }

def save_result(results:list[dict],output_path:Path)->None:
    """Save benchmark results to a CSV file."""
    output_path.parent.mkdir(parents=True,exist_ok=True)
    field_names=list(results[0].keys())
    with output_path.open("w",newline="",encoding='utf-8') as csv_file:
        writer=csv.DictWriter(
            csv_file,
            fieldnames=field_names,
        )
        writer.writeheader()
        writer.writerows(results)

def print_results(results:list[dict])->None:
    """Print benchmark results as a readable table."""
    header=(
        f"{'Size':>6}"
        f"{'Dtype':>10}"
        f"{'Latency(ms)':>14}"
        f"{'TFLOPS':>10}"
        f"{'Memory(MiB)':>13}"
        f"{'Max error':>12}"
        f"{'Relative L2':>14}"
    )
    print(header)
    print("-"*len(header))

    for result in results:
        print(
            f"{result['matrix_size']:>6}"
            f"{result['dtype']:>10}"
            f"{result['latency_ms']:>14.4f}"
            f"{result['estimated_tflops']:>10.3f}"
            f"{result['tensor_memory_mib']:>13.2f}"
            f"{result['max_absolute_error']:>12.6f}"
            f"{result['relative_l2_error']:>14.6e}"
        )

def parse_arguments()->argparse.Namespace:
    parser=argparse.ArgumentParser(
        description="Compare FP32, FP16, and BF16 matrix multiplication"
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[512,1024,2048,4096],
        help="Square matrix sizes to benchmark."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations."
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of measured iterations."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/precision_matmul.csv"),
        help="Path of the output csv file."
    )

    return parser.parse_args()

def main()->None:
    args=parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.set_float32_matmul_precision("highest")

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    results=[]

    for matrix_size in args.sizes:
        for dtype_name in DTYPES:
            result=benchmark_matmul(
                matrix_size=matrix_size,
                dtype_name=dtype_name,
                warmup_iterations=args.warmup,
                benchmark_iterations=args.iterations,
            )
            results.append(result)

    print_results(results)
    save_result(results,args.output)

    print()
    print(f"Results saved to:{args.output}")

if __name__=="__main__":
    main()
