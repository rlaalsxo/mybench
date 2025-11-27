# mybench/cli.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List
from benchmarks import Benchmark, BENCHMARK_CHOICES, benchmarks_from_choices
from evaluator_utils import evaluator_from_benchmark
from results import BenchmarkResults
from samples import find_samples_in_dirs

def run_benchmarks(
    benchmarks: List[Benchmark],
    sample_dirs: List[Path],
    output_dir: Path,
    stride: int = 1,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> None:
    """
    우리 프레임워크용 run_benchmarks:
    - 샘플 로딩
    - 각 벤치마크 evaluator 실행
    - 결과 저장/시각화
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 샘플 모으기
    samples = find_samples_in_dirs(sample_dirs)
    if not samples:
        print("[ERROR] No samples found in given sample_dirs.")
        return

    print(f"[INFO] Found {len(samples)} samples.")

    aggregate_results: dict[str, dict[str, float]] = {}

    for idx, benchmark in enumerate(benchmarks):
        print(f"[INFO] Running benchmark {idx+1} of {len(benchmarks)}: {benchmark.value}")

        results_dir = output_dir / benchmark.value
        if results_dir.exists() and not overwrite:
            print(f"[INFO] {benchmark.value} already exists, skipping (use --overwrite to redo).")
            continue

        # results_dir.mkdir(parents=True, exist_ok=True)

        evaluator = evaluator_from_benchmark(
            benchmark,
            stride=stride,
            max_frames=max_frames,
        )
        results: BenchmarkResults = evaluator(samples)

        aggregate_results[benchmark.value] = results.get_aggregate_metrics()

        results.save_results(output_dir)
        results.plot(output_dir)

        # 옵션: 피클로 저장하고 싶으면 여기서 results 를 피클링해도 됨

    with open(output_dir / "benchmark_metrics.json", "w") as f:
        json.dump(aggregate_results, f, indent=2, sort_keys=True)

    print(f"[INFO] Aggregate metrics written to {output_dir / 'benchmark_metrics.json'}")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample-only benchmark framework (BioEmu-style)."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    parser_eval = subparsers.add_parser(
        "eval",
        help="Run benchmarks on samples.",
        description="Run benchmarks on samples.",
    )
    parser_eval.add_argument(
        "output_dir",
        type=Path,
        help="결과를 저장할 디렉터리.",
    )
    parser_eval.add_argument(
        "--benchmarks",
        "-b",
        nargs="+",
        type=str,
        choices=BENCHMARK_CHOICES,
        required=True,
        help="List of benchmarks to evaluate. If set to `all`, all available benchmarks will be run.",
    )
    parser_eval.add_argument(
        "--sample_dirs",
        "-s",
        nargs="+",
        type=Path,
        required=True,
        help="xtc/pdb 샘플이 들어 있는 디렉터리(여러 개 가능).",
    )
    parser_eval.add_argument(
        "--stride",
        type=int,
        default=1,
        help="프레임 subsampling stride.",
    )
    parser_eval.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="최대 프레임 수 (None 이면 전체 사용).",
    )
    parser_eval.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 결과가 있어도 덮어쓰기.",
    )

    return parser.parse_args()

def cli() -> None:
    args = parse_arguments()

    if args.cmd == "eval":
        benches = benchmarks_from_choices(args.benchmarks)
        run_benchmarks(
            benchmarks=benches,
            sample_dirs=args.sample_dirs,
            output_dir=args.output_dir,
            stride=args.stride,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
    else:
        raise NotImplementedError(f"Unrecognized command {args.cmd}")

if __name__ == "__main__":
    cli()