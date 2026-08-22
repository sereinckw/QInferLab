import argparse
import csv
from pathlib import Path
import torch
from torch import nn
from qinferlab.layers import PreNormFeedForwardBlock

DTYPES={
    "float32":torch.float32,
    "float16":torch.float16,
    "bfloat16":torch.bfloat16,
}

def meaasure_latency(
    module:nn.Module,
    hidden_states:torch.Tensor,
    warmup_iterations:int,
    benchmark_iterations:int,
)->float:
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _=module(hidden_states)

        torch.cuda.synchronize()
        start_time=torch.cuda.Event(enable_timing=True)
        end_time=torch.cuda.Event(enable_timing=True)

        start_time.record()
        for _ in range(benchmark_iterations):
            _=module(hidden_states)
        end_time.record()

        end_time.synchronize()
        total_time_ms=start_time.elapsed_time(end_time)
        average_latency_ms=total_time_ms/benchmark_iterations
        return average_latency_ms

def run_benchmark(
    hidden_size:int,
    intermediate_size:int,
    sequence_length:int,
    dtype_name:str,
    warmup_iterations:int,
    benchmark_iterations:int,
)->dict:
    dtype=DTYPES[dtype_name]
    device=torch.device("cuda")

    torch.manual_seed(0)

    block=PreNormFeedForwardBlock(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ).to(device=device,dtype=dtype)
    block.eval()

    hidden_states=torch.randn(1,sequence_length,hidden_size,device=device,dtype=dtype)

    rmsnorm_latency_ms=meaasure_latency(
        module=block.input_layer_norm,
        hidden_states=hidden_states,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
    )

    feed_forward_latency_ms=meaasure_latency(
        module=block.feed_forward,
        hidden_states=hidden_states,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
    )

    block_latency_ms=meaasure_latency(
        module=block,
        hidden_states=hidden_states,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
    )

    block_tokens_per_second=sequence_length/(block_latency_ms/1000)
    block_overhead_percent=(block_latency_ms-feed_forward_latency_ms)/feed_forward_latency_ms * 100

    rmsnorm_share_percent=rmsnorm_latency_ms/block_latency_ms * 100

    parameter_count=block.parameter_count()
    bytes_per_parameter=torch.empty([],dtype=dtype).element_size()

    parameter_memory_mib=parameter_count*bytes_per_parameter/(1024*1024)

    return{
        "sequence_length": sequence_length,
        "dtype": dtype_name,
        "rmsnorm_latency_ms": rmsnorm_latency_ms,
        "feed_forward_latency_ms": feed_forward_latency_ms,
        "block_latency_ms": block_latency_ms,
        "block_tokens_per_second": block_tokens_per_second,
        "block_overhead_percent": block_overhead_percent,
        "rmsnorm_share_percent": rmsnorm_share_percent,
        "parameter_count": parameter_count,
        "parameter_memory_mib": parameter_memory_mib,
    }

def save_results(
    results:list[dict],
    output_path:Path,
)->None:
    output_path.parent.mkdir(parents=True,exist_ok=True)

    with output_path.open("w",newline="",encoding="utf-8") as csv_file:
        writer=csv.DictWriter(csv_file,fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

def print_results(results: list[dict]) -> None:
    header = (
        f"{'Seq':>5} "
        f"{'Dtype':>9} "
        f"{'Norm(ms)':>10} "
        f"{'FFN(ms)':>10} "
        f"{'Block(ms)':>11} "
        f"{'Tokens/s':>13} "
        f"{'Overhead':>10}"
    )

    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['sequence_length']:>5} "
            f"{result['dtype']:>9} "
            f"{result['rmsnorm_latency_ms']:>10.4f} "
            f"{result['feed_forward_latency_ms']:>10.4f} "
            f"{result['block_latency_ms']:>11.4f} "
            f"{result['block_tokens_per_second']:>13.1f} "
            f"{result['block_overhead_percent']:>9.2f}%"
        )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RMSNorm and a pre-norm FFN block."
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[1, 16, 128, 512],
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/rmsnorm_block.csv"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Intermediate size: {args.intermediate_size}")
    print()

    results = []

    for sequence_length in args.sequence_lengths:
        for dtype_name in DTYPES:
            result = run_benchmark(
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                sequence_length=sequence_length,
                dtype_name=dtype_name,
                warmup_iterations=args.warmup,
                benchmark_iterations=args.iterations,
            )
            results.append(result)

    print_results(results)
    save_results(results, args.output)

    print()
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()