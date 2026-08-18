import argparse
import csv
from pathlib import Path

import torch

from qinferlab.layers import SwiGLUFeedForward


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def benchmark_feed_forward(
    hidden_size: int,
    intermediate_size: int,
    sequence_length: int,
    dtype_name: str,
    warmup_iterations: int,
    benchmark_iterations: int,
) -> dict:
    dtype = DTYPES[dtype_name]
    device = torch.device("cuda")

    torch.manual_seed(0)

    model = SwiGLUFeedForward(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ).to(
        device=device,
        dtype=dtype,
    )

    model.eval()

    hidden_states = torch.randn(
        1,
        sequence_length,
        hidden_size,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = model(hidden_states)

        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        output = None

        for _ in range(benchmark_iterations):
            output = model(hidden_states)

        end_event.record()
        end_event.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    average_latency_ms = total_time_ms / benchmark_iterations

    tokens_per_second = (
        sequence_length / (average_latency_ms / 1000)
    )

    parameter_count = model.parameter_count()
    bytes_per_parameter = torch.empty([], dtype=dtype).element_size()

    parameter_memory_mib = (
        parameter_count * bytes_per_parameter
    ) / (1024**2)

    return {
        "sequence_length": sequence_length,
        "dtype": dtype_name,
        "latency_ms": average_latency_ms,
        "tokens_per_second": tokens_per_second,
        "parameter_count": parameter_count,
        "parameter_memory_mib": parameter_memory_mib,
        "output_shape": str(tuple(output.shape)),
    }


def save_results(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)


def print_results(results: list[dict]) -> None:
    header = (
        f"{'Seq len':>8} "
        f"{'Dtype':>10} "
        f"{'Latency(ms)':>14} "
        f"{'Tokens/s':>14} "
        f"{'Param MiB':>12}"
    )

    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['sequence_length']:>8} "
            f"{result['dtype']:>10} "
            f"{result['latency_ms']:>14.4f} "
            f"{result['tokens_per_second']:>14.2f} "
            f"{result['parameter_memory_mib']:>12.2f}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a SwiGLU feed-forward network."
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
        default=50,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/feed_forward.csv"),
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
            result = benchmark_feed_forward(
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