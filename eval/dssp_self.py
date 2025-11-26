# mybench/eval/dssp_self.py
from __future__ import annotations
from typing import Iterable, Optional, List
import mdtraj as md
import numpy as np
from results import SingleSampleDSSP, DSSPResults
from samples import SampleSpec  # basic_stats / fnc_self 등에서 사용하던 타입과 동일 가정

def _compute_sample_dssp(traj: md.Trajectory) -> np.ndarray:
    """
    Trajectory 전체에 대해 mdtraj DSSP 코드를 계산한다.

    반환:
        dssp_codes: shape (n_frames, n_residues), dtype='<U1'
                    각 원소는 'H','E','C' 등의 1글자 secondary structure 코드.
    """
    dssp_raw = md.compute_dssp(traj)  # 보통 dtype='|S1' (bytes)
    dssp_codes = dssp_raw.astype("<U1")  # Unicode 문자열로 변환
    return dssp_codes


def _compute_dssp_fractions(
    dssp_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DSSP 코드 행렬에서 helix/sheet/coil 비율을 frame 별로 계산한다.

    정의 (일반적인 DSSP 관습 사용):
      - helix: H, G, I
      - sheet: E, B
      - coil:  위 두 그룹에 속하지 않는 나머지 (T, S, C, ' ' 등)
    """
    if dssp_codes.ndim != 2:
        raise ValueError(
            f"dssp_codes.ndim={dssp_codes.ndim} 이지만 2D 배열이어야 합니다."
        )

    n_frames, n_residues = dssp_codes.shape
    if n_residues == 0:
        # residue 가 없으면 전부 0
        zeros = np.zeros(n_frames, dtype=float)
        return zeros, zeros, zeros

    helix_set = np.array(["H", "G", "I"], dtype="<U1")
    sheet_set = np.array(["E", "B"], dtype="<U1")

    helix_mask = np.isin(dssp_codes, helix_set)
    sheet_mask = np.isin(dssp_codes, sheet_set)

    helix_frac = helix_mask.sum(axis=1) / float(n_residues)
    sheet_frac = sheet_mask.sum(axis=1) / float(n_residues)
    coil_frac = 1.0 - helix_frac - sheet_frac

    # 수치 오차 방지를 위해 [0,1] 범위로 살짝 클램프
    helix_frac = np.clip(helix_frac, 0.0, 1.0)
    sheet_frac = np.clip(sheet_frac, 0.0, 1.0)
    coil_frac = np.clip(coil_frac, 0.0, 1.0)

    return helix_frac, sheet_frac, coil_frac

def evaluate_dssp_self(
    samples: Iterable[SampleSpec],
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> DSSPResults:
    """
    REFERENCE 없이 각 샘플 trajectory 자체에 대해 DSSP 를 계산하는 evaluator.

    각 SampleSpec 에 대해:
      - trajectory 로드 (+ stride, max_frames 적용)
      - md.compute_dssp(traj) 로 frame × residue DSSP 코드 계산
      - frame 별 helix/sheet/coil 비율 계산
      - SingleSampleDSSP 로 묶어서 반환

    Args:
        samples:
            (pdb_path, xtc_path, name, ...) 를 담은 SampleSpec iterable
        stride:
            frame subsampling 간격 (예: 10 이면 0,10,20,...)
        max_frames:
            최대 frame 수 제한 (None 이면 제한 없음)

    Returns:
        DSSPResults: 각 샘플별 DSSP 코드와 helix/sheet/coil fraction 을 포함
    """
    samples_dssp: List[SingleSampleDSSP] = []

    for spec in samples:
        print(f"[INFO] DSSP_SELF: loading {spec.name}")
        traj = md.load(spec.xtc_path.as_posix(), top=spec.pdb_path.as_posix())

        # stride 적용
        if stride > 1:
            traj = traj[::stride]

        # max_frames 제한
        if max_frames is not None and traj.n_frames > max_frames:
            traj = traj[:max_frames]

        if traj.n_frames == 0:
            print(f"[WARN] {spec.name}: n_frames=0, DSSP 계산을 건너뜁니다.")
            continue

        # 1) DSSP 코드 계산 (reference 전혀 사용하지 않음)
        dssp_codes = _compute_sample_dssp(traj)  # (T, N)

        # 2) frame 별 helix/sheet/coil 비율 계산
        helix_frac, sheet_frac, coil_frac = _compute_dssp_fractions(dssp_codes)

        frame_idx = np.arange(traj.n_frames, dtype=int)

        samples_dssp.append(
            SingleSampleDSSP(
                name=spec.name,
                frame_idx=frame_idx,
                dssp_codes=dssp_codes,
                helix_frac=helix_frac,
                sheet_frac=sheet_frac,
                coil_frac=coil_frac,
            )
        )

    return DSSPResults(samples=samples_dssp)