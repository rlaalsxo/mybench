# mybench/evaluator_utils.py
from __future__ import annotations
from collections.abc import Callable
from functools import partial
from typing import Iterable

from benchmarks import Benchmark
from results import BenchmarkResults
from samples import SampleSpec
from eval.basic_stats import evaluate_basic_stats
from eval.fnc_self import evaluate_fnc_self
from eval.folding_free_energies_self import evaluate_folding_free_energies_self

Evaluator = Callable[[Iterable[SampleSpec]], BenchmarkResults]


def evaluator_from_benchmark(benchmark: Benchmark, **kwargs) -> Evaluator:
    if benchmark == Benchmark.BASIC_STATS:
        return partial(evaluate_basic_stats, **kwargs)
    elif benchmark == Benchmark.FNC_SELF:
        return partial(evaluate_fnc_self, **kwargs)
    elif benchmark == Benchmark.FOLDING_FREE_ENERGIES:
        return partial(evaluate_folding_free_energies_self, **kwargs)
    else:
        raise ValueError(f"Unrecognized benchmark {benchmark}")
