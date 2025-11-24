# mybench/eval/tica_landscape.py
from __future__ import annotations
from typing import Iterable, Optional, List
from pathlib import Path
import mdtraj as md
import numpy as np
from results import SingleSampleTICA, TICAResults
from samples import SampleSpec  # BASIC_STATS, FNC_SELF 에서 쓰던 것과 동일한 타입 가정

def _compute_tica_2d(features: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    매우 단순한 선형 TICA 구현.
    - features: (n_frames, n_features)
    - lag: time-lag (frame 단위)

    반환:
        (n_frames - lag, 2) 형태의 TICA 좌표 (2개의 가장 느린 성분)
    """
    n_frames, n_feat = features.shape
    if n_frames <= lag + 1:
        raise ValueError(f"TICA 계산에 충분한 frame 이 없습니다: n_frames={n_frames}, lag={lag}")

    # 평균 0 으로 맞추기
    X = features - features.mean(axis=0, keepdims=True)

    X0 = X[:-lag]         # t
    Xtau = X[lag:]        # t + lag

    # 공분산 / time-lag 공분산
    T = X0.shape[0]
    C0 = (X0.T @ X0) / (T - 1)
    Ctau = (X0.T @ Xtau) / (T - 1)

    # 간단한 regularization
    eps = 1e-6
    C0_reg = C0 + eps * np.eye(n_feat, dtype=C0.dtype)

    # C0^{-1} Ctau 의 고유값/고유벡터 (일반적인 TICA 형식)
    A = np.linalg.solve(C0_reg, Ctau)
    # 실수 행렬이지만 수치오차로 복소수 성분이 조금 생길 수 있으니 실수부만 사용
    eigvals, eigvecs = np.linalg.eig(A)

    # 고유값 절댓값 기준 내림차순 정렬 (느린 모드가 |lambda| 큰 쪽)
    idx = np.argsort(np.abs(np.real(eigvals)))[::-1]
    eigvecs = np.real(eigvecs[:, idx])

    # 첫 두 개 성분만 사용
    W = eigvecs[:, :2]              # (n_features, 2)
    Y = X0 @ W                      # (n_frames - lag, 2)

    return Y


def _traj_to_feature_matrix(traj: md.Trajectory, max_ca_atoms: Optional[int] = 120) -> np.ndarray:
    """
    TICA 에 사용할 feature 벡터 생성:
    - 모든 frame 을 첫 frame 에 RMSD 기반으로 superpose
    - backbone Cα 좌표만 사용 (nm 단위)
    - 필요하면 Cα atom 수를 max_ca_atoms 로 downsample

    반환:
        features: shape (n_frames, n_features)
    """
    # 모든 frame 을 첫 frame 에 align (전후 이동 제거)
    traj_aligned = traj.superpose(traj, 0)

    # CA atom 선택
    ca_indices = traj_aligned.topology.select("name CA")
    if ca_indices.size == 0:
        raise ValueError("Cα atom 을 찾을 수 없습니다. PDB/Topology 를 확인하세요.")

    # 너무 큰 시스템에서는 Cα 개수를 제한 (간단한 subsampling)
    if max_ca_atoms is not None and ca_indices.size > max_ca_atoms:
        # 균등 간격으로 몇 개만 선택
        step = ca_indices.size / max_ca_atoms
        chosen = [ca_indices[int(i * step)] for i in range(max_ca_atoms)]
        ca_indices = np.array(chosen, dtype=int)

    traj_ca = traj_aligned.atom_slice(ca_indices)

    # (n_frames, n_atoms, 3) -> (n_frames, 3 * n_atoms)
    n_frames = traj_ca.n_frames
    feats = traj_ca.xyz.reshape(n_frames, -1)  # 단위는 여전히 nm

    return feats


def evaluate_tica_landscape(
    samples: Iterable[SampleSpec],
    stride: int = 1,
    max_frames: Optional[int] = None,
    tica_lag: int = 1,
    max_ca_atoms: Optional[int] = 120,
) -> TICAResults:
    """
    각 샘플에 대해:
      - trajectory 로부터 Cα 좌표 feature 추출
      - 간단한 선형 TICA 로 2개의 성분 계산
      - TICA 공간에서 free-energy surface 를 그릴 수 있도록 좌표를 저장

    Args:
        samples:
            (pdb_path, xtc_path, name, ...) 를 담은 SampleSpec iterable
        stride:
            frame subsampling 간격 (예: 10 이면 0,10,20,...)
        max_frames:
            최대 frame 수 제한 (None 이면 제한 없음)
        tica_lag:
            TICA time-lag (frame 단위). stride 를 포함해서 생각해야 합니다.
        max_ca_atoms:
            feature 로 사용할 Cα atom 최대 개수 (너무 큰 시스템에서 메모리 폭주 방지)

    Returns:
        TICAResults: 각 샘플별 TICA 좌표를 포함한 결과 객체
    """
    samples_tica: List[SingleSampleTICA] = []

    for spec in samples:
        print(f"[INFO] TICA_LANDSCAPE: loading {spec.name}")
        traj = md.load(spec.xtc_path.as_posix(), top=spec.pdb_path.as_posix())

        # stride 적용
        if stride > 1:
            traj = traj[::stride]

        # max_frames 제한
        if max_frames is not None and traj.n_frames > max_frames:
            traj = traj[:max_frames]

        if traj.n_frames <= tica_lag + 1:
            print(
                f"[WARN] {spec.name}: n_frames={traj.n_frames} 로 TICA 계산 불가능 (lag={tica_lag}). 건너뜀."
            )
            continue

        # feature matrix 생성
        feats = _traj_to_feature_matrix(traj, max_ca_atoms=max_ca_atoms)  # (T, D)

        # 2D TICA
        tica_xy = _compute_tica_2d(feats, lag=tica_lag)  # (T - lag, 2)

        samples_tica.append(
            SingleSampleTICA(
                name=spec.name,
                tica_xy=tica_xy,
            )
        )

    return TICAResults(samples=samples_tica)
