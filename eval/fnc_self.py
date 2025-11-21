# mybench/eval/fnc_self.py

from __future__ import annotations
from typing import Iterable, Optional, List

import mdtraj as md
import numpy as np

from bioemu_benchmarks.eval.multiconf.metrics import fraction_native_contacts
from results import SingleSampleFNC, FNCResults
from samples import SampleSpec  # basic_stats 에서 사용한 것과 동일한 타입이라고 가정합니다.
from summary_metrics import (
    coverage_bootstrap,
    k_recall_bootstrap,
)

def evaluate_fnc_self(
    samples: Iterable[SampleSpec],
    stride: int = 1,
    max_frames: Optional[int] = None,
    exclude_n_neighbours: int = 3,
    coverage_nbootstrap: int = 20,
    coverage_nsample: Optional[int] = None,
    coverage_nsuccess: int = 1,
    coverage_num_thresholds: int = 100,
    krecall_k: int = 1,
    krecall_nbootstrap: int = 20,
    krecall_nsample: Optional[int] = None,
) -> FNCResults:
    """
    각 샘플에 대해 'self-reference FNC' 를 계산하는 evaluator.

    정의:
      - 기준 구조(reference)는 같은 trajectory의 첫 프레임(traj[0])
      - 전체 chain(별도 residue subset 없음)에 대해
        bioemu 원본 fraction_native_contacts 로 FNC 계산
      - fold/unfold 라벨은 사용하지 않고,
        단순히 "기준 frame contact 패턴이 얼마나 유지되는지"만 본다.

    추가로, 모든 샘플의 FNC 분포를 모아서
      - coverage_bootstrap
      - k_recall_bootstrap
    을 계산하고, 결과를 FNCResults 인스턴스에 속성으로 저장합니다.

    Args:
        samples:
            (pdb_path, xtc_path, name, ...) 정보를 가진 SampleSpec iterable.
        stride:
            프레임 스텝 (예: 10 이면 0, 10, 20, ... 만 사용)
        max_frames:
            최대 프레임 수 제한 (None 이면 제한 없음)
        exclude_n_neighbours:
            fraction_native_contacts 에서 사용하는
            이웃 residue (|i-j| <= k) 제거 기준. (원본과 동일하게 3 등)
        coverage_nbootstrap:
            coverage 부트스트랩 반복 횟수.
        coverage_nsample:
            coverage 부트스트랩에서 각 샘플당 뽑을 frame 수.
            None 이면 각 샘플의 frame 수 최댓값을 사용.
        coverage_nsuccess:
            한 샘플이 threshold r 에서 "covered" 로 인정되기 위한
            최소 성공 frame 개수.
        coverage_num_thresholds:
            coverage 곡선을 계산할 threshold 개수 (linspace 분할 수).
        krecall_k:
            k-recall 에서 사용할 k 값 (상위 k frame).
        krecall_nbootstrap:
            k-recall 부트스트랩 반복 횟수.
        krecall_nsample:
            k-recall 부트스트랩에서 각 샘플당 뽑을 frame 수.
            None 이면 각 샘플의 frame 수 최댓값을 사용.

    Returns:
        FNCResults:
            - samples: 각 샘플별 FNC 시계열
            - (동적 속성으로)
              coverage_thresholds: np.ndarray (threshold 축)
              coverage_bootstrap: np.ndarray (nbootstrap, nthreshold)
              krecall_bootstrap: Dict[str, (mean, std)]
    """
    metrics_list: List[SingleSampleFNC] = []

    # 1) 각 샘플에서 self-reference FNC 계산
    for spec in samples:
        print(f"[INFO] FNC_SELF: loading {spec.name}")
        traj = md.load(spec.xtc_path.as_posix(), top=spec.pdb_path.as_posix())

        # stride 적용
        if stride > 1:
            traj = traj[::stride]

        # max_frames 제한
        if max_frames is not None and traj.n_frames > max_frames:
            traj = traj[:max_frames]

        n_frames = traj.n_frames
        if n_frames == 0:
            # 프레임이 전혀 없는 경우는 스킵
            continue

        frame_idx = np.arange(n_frames, dtype=int)

        # 기준 구조: 첫 프레임만 가진 1-frame trajectory
        ref_traj = traj[0]

        # 전체 chain 사용 → matching_resids=None, reference_resid_pairs=None
        # 정의/수식은 bioemu metrics.py 의 fraction_native_contacts 그대로.
        fnc = fraction_native_contacts(
            traj_i=ref_traj,
            traj_j=traj,
            matching_resids=None,
            reference_resid_pairs=None,
            threshold=8.0,  # 원본과 동일 (Å)
            exclude_n_neighbours=exclude_n_neighbours,
        )
        # fnc shape: (n_frames,)  --- 각 frame에서의 FNC 값

        metrics_list.append(
            SingleSampleFNC(
                name=spec.name,
                frame_idx=frame_idx,
                fnc=fnc,
            )
        )

    # 2) FNCResults 객체 생성
    results = FNCResults(samples=metrics_list)

    # 3) summary_metrics 를 이용해 coverage / k-recall 계산
    #    FNC 값이므로 [0, 1] 범위, 값이 클수록 "좋다" 라고 가정합니다.
    if metrics_list:
        fnc_dict = {s.name: s.fnc for s in metrics_list}

        # coverage (bootstrap)
        thresholds, coverages = coverage_bootstrap(
            fnc_dict,
            nsuccess=coverage_nsuccess,
            nbootstrap=coverage_nbootstrap,
            nsample=coverage_nsample,
            xmin=0.0,
            xmax=1.0,
            num=coverage_num_thresholds,
            larger_is_better=True,
        )

        # k-recall (bootstrap)
        krec = k_recall_bootstrap(
            fnc_dict,
            k=krecall_k,
            nbootstrap=krecall_nbootstrap,
            nsample=krecall_nsample,
            larger_is_better=True,
        )

        # FNCResults 인스턴스에 동적으로 속성으로 부착
        # (원하시면 FNCResults dataclass 에 정식 필드로 추가하셔도 됩니다.)
        results.coverage_thresholds = thresholds           # np.ndarray (nthreshold,)
        results.coverage_bootstrap = coverages             # np.ndarray (nbootstrap, nthreshold)
        results.krecall_bootstrap = krec                   # Dict[str, (mean, std)]
    else:
        # 샘플이 하나도 없으면 빈 형태로 속성만 추가
        results.coverage_thresholds = np.array([], dtype=float)
        results.coverage_bootstrap = np.zeros((0, 0), dtype=float)
        results.krecall_bootstrap = {}

    return results