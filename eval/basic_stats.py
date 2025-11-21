# mybench/eval/basic_stats.py
from __future__ import annotations
from typing import Iterable, Optional, List
import mdtraj as md
import numpy as np
from results import BasicStatsResults, SingleSampleMetrics
from samples import SampleSpec

def evaluate_basic_stats(
    samples: Iterable[SampleSpec],
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> BasicStatsResults:
    """
    BASIC_STATS 벤치마크:
    - 각 샘플에서 RMSD(첫 프레임 기준) / Rg 계산
    - 결과는 BasicStatsResults 로 반환
    """
    metrics_list: List[SingleSampleMetrics] = []

    for spec in samples:
        print(f"[INFO] BASIC_STATS: loading {spec.name}")
        traj = md.load(spec.xtc_path.as_posix(), top=spec.pdb_path.as_posix())

        if stride > 1:
            traj = traj[::stride]

        if max_frames is not None and len(traj) > max_frames:
            traj = traj[:max_frames]

        # 프레임 수만 알면 됨 (시간은 사용 안 함)
        # n_frames = traj.n_frames  # 필요하면 유지

        rmsd_nm = md.rmsd(traj, traj, 0)  # 첫 프레임 기준
        rg_nm = md.compute_rg(traj)

        metrics_list.append(
            SingleSampleMetrics(
                name=spec.name,
                rmsd_nm=rmsd_nm,
                rg_nm=rg_nm,
            )
        )

    return BasicStatsResults(samples=metrics_list)