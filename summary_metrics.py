# mybench/summary_metrics.py

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import operator

def coverage(
    results: Dict[str, np.ndarray],
    nsuccess: int = 1,
    xmin: float = 0.0,
    xmax: float = 1.0,
    num: int = 100,
    larger_is_better: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    여러 시스템에 대해 1D metric 분포가 주어졌을 때,
    threshold 를 바꿔가며 "coverage" 곡선을 계산합니다.

    정의(원본 SINGLECONF 로직을 1D에 맞게 단순화):
    - 각 시스템 s에 대해 metric 값 배열 x_s (shape: [n_frames])가 있다고 할 때,
    - 각 threshold r 에 대해
        성공 프레임 수 C_s(r) = | { i : x_s[i] 가 (r 보다 좋음) } |
      이 nsuccess 이상이면, 그 시스템은 r 에서 "covered" 로 간주.
    - coverage(r) = covered 시스템 비율.

    Args:
        results:
            { 시스템 이름 -> metric 값 1D array } 딕셔너리.
            예: { "sample1": fnc_array, "sample2": fnc_array, ... }
        nsuccess:
            한 시스템이 covered 로 인정되기 위해 필요한 "성공 프레임" 최소 개수.
        xmin, xmax:
            threshold 스캔 구간.
        num:
            threshold 분할 개수 (linspace 개수).
        larger_is_better:
            True 면 "값이 클수록 좋음" (>= r), False 면 "작을수록 좋음" (<= r).

    Returns:
        thresholds: shape (num,) 의 threshold 배열
        coverage:  shape (num,) 의 coverage 값
    """
    thresholds = np.linspace(xmin, xmax, num=num)
    coverage_vals = np.zeros_like(thresholds, dtype=float)

    # 비교 연산자 선택
    better_op = operator.ge if larger_is_better else operator.le

    n_systems = len(results)
    if n_systems == 0:
        return thresholds, coverage_vals  # 모두 0

    for j, r in enumerate(thresholds):
        covered_count = 0
        for x in results.values():
            if x.size == 0:
                continue
            successes = better_op(x, r).astype(int).sum()
            if successes >= nsuccess:
                covered_count += 1
        coverage_vals[j] = covered_count / float(n_systems)

    return thresholds, coverage_vals


def coverage_bootstrap(
    results: Dict[str, np.ndarray],
    nsuccess: int = 1,
    nbootstrap: int = 20,
    nsample: int | None = None,
    xmin: float = 0.0,
    xmax: float = 1.0,
    num: int = 100,
    larger_is_better: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    coverage() 를 bootstrap resampling 으로 반복 계산하여
    threshold 마다 coverage 분포(평균 ± 표준편차)를 얻습니다.

    Args:
        results:
            { 시스템 이름 -> metric 값 1D array }.
        nsuccess:
            한 시스템이 covered 로 인정되기 위한 최소 성공 프레임 수.
        nbootstrap:
            bootstrap 반복 횟수.
        nsample:
            각 bootstrap 에서 뽑을 frame 수.
            None 이면 각 시스템에서 가능한 frame 수 최대값을 사용.
        xmin, xmax, num, larger_is_better:
            coverage() 에 전달할 파라미터.

    Returns:
        thresholds: shape (num,)
        coverages: shape (nbootstrap, num)
                   각 bootstrap 반복에서의 coverage 곡선.
    """
    if len(results) == 0:
        thresholds = np.linspace(xmin, xmax, num=num)
        return thresholds, np.zeros((nbootstrap, num), dtype=float)

    if nsample is None:
        nsample = max(x.size for x in results.values() if x.size > 0)

    all_coverages = []
    thresholds: np.ndarray | None = None

    for _ in range(nbootstrap):
        resampled: Dict[str, np.ndarray] = {}
        for name, x in results.items():
            if x.size == 0:
                resampled[name] = x
                continue
            idx = np.random.randint(0, x.size, size=nsample)
            resampled[name] = x[idx]

        th, cov = coverage(
            resampled,
            nsuccess=nsuccess,
            xmin=xmin,
            xmax=xmax,
            num=num,
            larger_is_better=larger_is_better,
        )
        if thresholds is None:
            thresholds = th
        all_coverages.append(cov)

    assert thresholds is not None
    coverages_arr = np.stack(all_coverages, axis=0)  # [nbootstrap, num]

    return thresholds, coverages_arr


def k_recall(
    results: Dict[str, np.ndarray],
    k: int = 1,
    larger_is_better: bool = True,
) -> Dict[str, float]:
    """
    각 시스템에 대해 'best k' metric 값의 평균을 반환합니다.

    정의:
        - larger_is_better=True:
            x 를 내림차순 정렬한 뒤 상위 k 개의 평균
        - larger_is_better=False:
            x 를 오름차순 정렬한 뒤 상위 k 개의 평균

    Args:
        results:
            { 시스템 이름 -> metric 값 1D array }
        k:
            "best" 로 볼 frame 개수.
            (실제 frame 수보다 크면, 가능한 frame 전체를 사용.)
        larger_is_better:
            True면 값이 클수록 좋고, False면 작을수록 좋다고 가정.

    Returns:
        { 시스템 이름 -> k-recall 값 } 딕셔너리.
    """
    recalls: Dict[str, float] = {}

    for name, x in results.items():
        if x.size == 0:
            continue

        x_sorted = np.sort(x)
        if larger_is_better:
            x_sorted = x_sorted[::-1]

        k_eff = min(k, x_sorted.size)
        recalls[name] = float(np.mean(x_sorted[:k_eff]))

    return recalls


def k_recall_bootstrap(
    results: Dict[str, np.ndarray],
    k: int = 1,
    nbootstrap: int = 20,
    nsample: int | None = None,
    larger_is_better: bool = True,
) -> Dict[str, Tuple[float, float]]:
    """
    k_recall 을 bootstrap 으로 반복 계산하여
    각 시스템별 (mean, std)를 반환합니다.

    Args:
        results:
            { 시스템 이름 -> metric 값 1D array }
        k:
            k-recall 의 k.
        nbootstrap:
            bootstrap 반복 횟수.
        nsample:
            각 bootstrap 에서 뽑을 frame 수.
            None 이면 각 시스템에서 가능한 frame 수 최대값을 사용.
        larger_is_better:
            True면 값이 클수록 좋고, False면 작을수록 좋다고 가정.

    Returns:
        { 시스템 이름 -> (k-recall 평균, 표준편차) }
    """
    if len(results) == 0:
        return {}

    if nsample is None:
        nsample = max(x.size for x in results.values() if x.size > 0)

    all_recalls: Dict[str, list[float]] = {name: [] for name in results.keys()}

    for _ in range(nbootstrap):
        resampled: Dict[str, np.ndarray] = {}
        for name, x in results.items():
            if x.size == 0:
                resampled[name] = x
                continue
            idx = np.random.randint(0, x.size, size=nsample)
            resampled[name] = x[idx]

        recalls_b = k_recall(
            resampled,
            k=k,
            larger_is_better=larger_is_better,
        )
        for name, val in recalls_b.items():
            all_recalls[name].append(val)

    out: Dict[str, Tuple[float, float]] = {}
    for name, vals in all_recalls.items():
        if len(vals) == 0:
            continue
        arr = np.asarray(vals, dtype=float)
        out[name] = (float(arr.mean()), float(arr.std(ddof=0)))

    return out