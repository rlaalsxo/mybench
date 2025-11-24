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
      - 동시에 같은 trajectory에서 첫 프레임 기준 RMSD (nm) 도 계산한다.

    추가로, 모든 샘플의 FNC 분포를 모아서
      - coverage_bootstrap
      - k_recall_bootstrap
    을 계산하고, 결과를 FNCResults 인스턴스에 속성으로 저장합니다.
    """
    metrics_list: List[SingleSampleFNC] = []

    # 1) 각 샘플에서 self-reference FNC + RMSD 계산
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

        # (추가) RMSD (첫 프레임 기준, nm)
        rmsd_nm = md.rmsd(traj, traj, 0)  # shape: (n_frames,)

        # 기준 구조: 첫 프레임만 가진 1-frame trajectory
        ref_traj = traj[0]

        # 전체 chain 사용 → matching_resids=None, reference_resid_pairs=None
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
                rmsd_nm=rmsd_nm,  # 여기가 핵심 추가
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

        results.coverage_thresholds = thresholds           # np.ndarray (nthreshold,)
        results.coverage_bootstrap = coverages             # np.ndarray (nbootstrap, nthreshold)
        results.krecall_bootstrap = krec                   # Dict[str, (mean, std)]
    else:
        results.coverage_thresholds = np.array([], dtype=float)
        results.coverage_bootstrap = np.zeros((0, 0), dtype=float)
        results.krecall_bootstrap = {}

    return results