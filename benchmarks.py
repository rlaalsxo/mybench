# mybench/benchmarks.py
from enum import Enum

class Benchmark(Enum):
    BASIC_STATS = "basic_stats"
    # 나중에 free energy map, CV 분석 등을 추가하고 싶으면:
    # FREE_ENERGY_2D = "free_energy_2d"
