# mybench/evaluator_utils.py
from __future__ import annotations
from collections.abc import Callable
from functools import partial
from typing import Iterable
from benchmarks import Benchmark
from results import BenchmarkResults
from samples import SampleSpec
from eval.basic_stats import evaluate_basic_stats

Evaluator = Callable[[Iterable[SampleSpec]], BenchmarkResults]

def evaluator_from_benchmark(benchmark: Benchmark, **kwargs) -> Evaluator:
    if benchmark == Benchmark.BASIC_STATS:
        return partial(evaluate_basic_stats, **kwargs)
    else:
        raise ValueError(f"Unrecognized benchmark {benchmark}")
