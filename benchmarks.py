# mybench/benchmarks.py
from __future__ import annotations
from enum import Enum
from typing import Literal, List

class Benchmark(str, Enum):
    BASIC_STATS = "basic_stats"
    FNC_SELF = "fnc_self"


BENCHMARK_CHOICES: list[str] = [b.value for b in Benchmark] + ["all"]


def benchmarks_from_choices(
    choices: list[Literal["basic_stats", "fnc_self", "all"]],
) -> List[Benchmark]:
    """
    CLI 등에서 받은 문자열 리스트를 Benchmark 리스트로 변환.
    'all' 이 포함되어 있으면 모든 Benchmark 를 반환.
    """
    if "all" in choices:
        return [b for b in Benchmark]
    else:
        return [Benchmark(c) for c in set(choices)]
