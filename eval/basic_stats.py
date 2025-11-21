# mybench/eval/basic_stats.py
from __future__ import annotations
from pathlib import Path
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

        n_frames = traj.n_frames

        # 시간 축: ps → ns (time 필드가 없으면 frame index 사용)
        if traj.time is not None and len(traj.time) == n_frames:
            time_ps = traj.time
            time_ns = time_ps / 1000.0
        else:
            time_ns = np.arange(n_frames, dtype=float)

        rmsd_nm = md.rmsd(traj, traj, 0)  # 첫 프레임 기준
        rg_nm = md.compute_rg(traj)

        metrics_list.append(
            SingleSampleMetrics(
                name=spec.name,
                time_ns=time_ns,
                rmsd_nm=rmsd_nm,
                rg_nm=rg_nm,
            )
        )

    return BasicStatsResults(samples=metrics_list)