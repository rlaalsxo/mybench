# mybench/evaluator_utils.py
from __future__ import annotations
from collections.abc import Callable
from functools import partial
from typing import Iterable
from benchmarks import Benchmark
from results import BenchmarkResults
from samples import SampleSpec
from eval.basic_stats import evaluate_basic_stats
from eval.fnc_self import evaluate_fnc_self  # ← 새로 추가

# samples: Iterable[SampleSpec] 를 받아서 BenchmarkResults 를 반환하는 callable
Evaluator = Callable[[Iterable[SampleSpec]], BenchmarkResults]

def evaluator_from_benchmark(benchmark: Benchmark, **kwargs) -> Evaluator:
    """
    Benchmark 타입에 따라 적절한 evaluator 함수를 돌려주는 유틸.

    kwargs 는 stride, max_frames 같은 공통 옵션을 partial 로 미리 묶어서 넘기기 위한 용도입니다.
    """
    if benchmark == Benchmark.BASIC_STATS:
        # evaluate_basic_stats(samples, **kwargs) 형태로 호출되도록 partial 생성
        return partial(evaluate_basic_stats, **kwargs)

    elif benchmark == Benchmark.FNC_SELF:
        # evaluate_fnc_self(samples, **kwargs) 형태로 호출되도록 partial 생성
        return partial(evaluate_fnc_self, **kwargs)

    else:
        raise ValueError(f"Unrecognized benchmark {benchmark}")
