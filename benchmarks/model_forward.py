import argparse
import csv
from pathlib import Path

import torch

from qinferlab.model import DecoderOnlyTransformer
from qinferlab.model import ModelConfig


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def benchmark_model(
    config: ModelConfig,
    sequence_length: int,
    dtype_name: str,
    warmup_iterations: int,
    benchmark_iterations: int,
) -> dict:
    dtype = DTYPES[dtype_name]
    device = torch.device("cuda")

    torch.manual_seed(0)

    model = DecoderOnlyTransformer(config).to(
        device=device,
        dtype=dtype,
    )
    model.eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, sequence_length),
        device=device,
    )

    position_ids = torch.arange(
        sequence_length,
        device=device,
    ).unsqueeze(0)

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            warmup_output = model(
                input_ids=input_ids,
                position_ids=position_ids,
            )

        torch.cuda.synchronize()
        del warmup_output

        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        logits = None

        for _ in range(benchmark_iterations):
            logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
            )

        end_event.record()
        end_event.synchronize()

        peak_memory = torch.cuda.max_memory_allocated()

    latency_ms = (
        start_event.elapsed_time(end_event)
        / benchmark_iterations
    )

    input_tokens_per_second = (
        sequence_length / (latency_ms / 1000)
    )

    parameter_count = model.parameter_count()

    parameter_memory_mib = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    ) / (1024**2)

    peak_extra_memory_mib = (
        peak_memory - baseline_memory
    ) / (1024**2)

    return {
        "sequence_length": sequence_length,
        "dtype": dtype_name,
        "latency_ms": latency_ms,
        "input_tokens_per_second": input_tokens_per_second,
        "parameter_count": parameter_count,
        "parameter_memory_mib": parameter_memory_mib,
        "peak_extra_memory_mib": peak_extra_memory_mib,
        "logits_shape": str(tuple(logits.shape)),
    }


def save_results(
    results: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)


def print_results(results: list[dict]) -> None:
    header = (
        f"{'Seq':>5} "
        f"{'Dtype':>9} "
        f"{'Latency(ms)':>13} "
        f"{'Input tok/s':>14} "
        f"{'Param MiB':>11} "
        f"{'Peak extra':>12}"
    )

    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['sequence_length']:>5} "
            f"{result['dtype']:>9} "
            f"{result['latency_ms']:>13.4f} "
            f"{result['input_tokens_per_second']:>14.1f} "
            f"{result['parameter_memory_mib']:>11.2f} "
            f"{result['peak_extra_memory_mib']:>12.2f}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a minimal decoder-only Transformer."
        )
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
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
        default=5,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/model_forward.csv"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_sequence_length=max(args.sequence_lengths),
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Layers: {config.num_layers}")
    print(f"Heads: {config.num_heads}")
    print()

    results = []

    for sequence_length in args.sequence_lengths:
        for dtype_name in DTYPES:
            result = benchmark_model(
                config=config,
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