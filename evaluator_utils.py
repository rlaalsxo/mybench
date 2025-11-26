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
from eval.tica_landscape import evaluate_tica_landscape
from eval.md_emulation_self import evaluate_md_emulation_self
from eval.dssp_self import evaluate_dssp_self

Evaluator = Callable[[Iterable[SampleSpec]], BenchmarkResults]


def evaluator_from_benchmark(benchmark: Benchmark, **kwargs) -> Evaluator:
    if benchmark == Benchmark.BASIC_STATS:
        return partial(evaluate_basic_stats, **kwargs)
    elif benchmark == Benchmark.FNC_SELF:
        return partial(evaluate_fnc_self, **kwargs)
    elif benchmark == Benchmark.FOLDING_FREE_ENERGIES:
        return partial(evaluate_folding_free_energies_self, **kwargs)
    elif benchmark == Benchmark.TICA_LANDSCAPE:
        return partial(evaluate_tica_landscape, **kwargs)
    elif benchmark == Benchmark.MD_EMULATION_SELF:
        return partial(evaluate_md_emulation_self, **kwargs)
    elif benchmark == Benchmark.DSSP_SELF:
        return partial(evaluate_dssp_self, **kwargs)
    else:
        raise ValueError(f"Unrecognized benchmark {benchmark}")
