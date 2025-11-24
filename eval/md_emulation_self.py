# mybench/eval/md_emulation_self.py
from __future__ import annotations
from typing import Iterable, Optional, List

import mdtraj as md
import numpy as np

from results import SingleSampleMDEmulationSelf, MDEmulationSelfResults
from samples import SampleSpec  # basic_stats / fnc_self 등에서 사용하던 타입과 동일 가정


def _get_ca_coordinates(traj: md.Trajectory, n_trim: int = 2) -> np.ndarray:
    """
    Trajectory 로부터 Cα 좌표를 추출하고, 양 끝에서 n_trim residue 를 잘라낸다.

    반환:
        coords: shape (n_frames, n_ca_trimmed, 3), 단위는 nm
    """
    top = traj.topology
    n_residues = top.n_residues

    if 2 * n_trim >= n_residues:
        raise ValueError(
            f"n_trim={n_trim} 이(가) 너무 큽니다 (n_residues={n_residues}). "
            "2*n_trim < n_residues 여야 합니다."
        )

    # mdtraj 의 resid 는 0-based residue index
    atom_indices = top.select(
        f"name CA and resid {n_trim} to {n_residues - 1 - n_trim}"
    )
    if atom_indices.size == 0:
        raise ValueError(
            "trim 이후 Cα atom 을 찾을 수 없습니다. topology 또는 n_trim 을 확인하세요."
        )

    # traj.xyz: (n_frames, n_atoms, 3)
    coords = traj.xyz[:, atom_indices]  # (n_frames, n_ca_trimmed, 3)
    return coords


def _compute_distance_matrices(
    coords: np.ndarray,
    exclude_neighbors: int = 2,
) -> np.ndarray:
    """
    Cα 좌표로부터 pairwise 거리 행렬을 계산하고,
    서열상 |i-j| <= exclude_neighbors 인 이웃은 0 으로 마스킹한다.

    Args:
        coords: shape (n_frames, n_ca, 3)
        exclude_neighbors: 이웃으로 간주할 residue 거리 (포함)

    Returns:
        dist: shape (n_frames, n_ca, n_ca)
    """
    # pairwise 거리: ||x_i - x_j|| (nm)
    # coords[:, None, :, :] - coords[:, :, None, :] → (n_frames, n_ca, n_ca, 3)
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (n_frames, n_ca, n_ca)

    # 서열 이웃 마스크
    n_ca = coords.shape[1]
    entry_idx = np.arange(n_ca)
    neighbor_mask = np.abs(entry_idx[:, None] - entry_idx[None, :]) <= exclude_neighbors
    dist[:, neighbor_mask] = 0.0

    return dist


def _compute_contact_features(
    traj: md.Trajectory,
    n_trim: int = 2,
    exclude_neighbors: int = 2,
    effective_distance: float = 0.8,
) -> np.ndarray:
    """
    MD emulation 원본 로직과 같은 contact-map 기반 feature 를 계산한다.

    절차:
      1) Cα 좌표 추출 (양 끝 n_trim residue 잘라냄)
      2) Cα 거리 행렬 계산, 서열 이웃(|i-j| <= exclude_neighbors) 은 0 으로 설정
      3) contact-like feature: f = exp(- d / effective_distance), 상한 1.0
      4) 상삼각 요소만 펼쳐서 2D feature matrix 로 변환

    Returns:
        features: shape (n_frames, n_features)
    """
    coords = _get_ca_coordinates(traj, n_trim=n_trim)  # (T, n_ca, 3)
    dist = _compute_distance_matrices(
        coords,
        exclude_neighbors=exclude_neighbors,
    )  # (T, n_ca, n_ca)

    # contact-like feature
    features = dist / float(effective_distance)
    features = np.minimum(np.exp(-features), 1.0)

    # 상삼각 (대각 포함)만 사용
    dim_features = features.shape[-1]
    idx_i, idx_j = np.triu_indices(dim_features)
    features = features[:, idx_i, idx_j]  # (T, n_upper)

    return features


def _compute_pca_projection_2d(features: np.ndarray) -> np.ndarray:
    """
    간단한 PCA 기반 2D 선형 투영을 수행한다.

    - features: (n_frames, n_features)
    - 평균을 제거한 뒤 SVD 를 사용해 첫 두 개 주성분 방향으로 투영.

    Returns:
        proj_xy: shape (n_frames, 2)
    """
    n_frames, n_feat = features.shape
    if n_frames < 3:
        raise ValueError(
            f"PCA 투영에 필요한 frame 수가 부족합니다: n_frames={n_frames} (최소 3 권장)"
        )

    X = features - features.mean(axis=0, keepdims=True)  # 중심화

    # SVD: X = U S V^T, 여기서 V^T[:2] 가 첫 두 principal directions
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # 최소 차원보다 principal components 수를 요구하면 안 되므로 방어
    n_components = min(2, Vt.shape[0])
    if n_components < 2:
        raise ValueError(
            f"PCA 성분이 2개 미만입니다. (Vt.shape[0]={Vt.shape[0]})"
        )

    W = Vt[:2].T  # (n_features, 2)
    proj_xy = X @ W  # (n_frames, 2)

    return proj_xy


def evaluate_md_emulation_self(
    samples: Iterable[SampleSpec],
    stride: int = 1,
    max_frames: Optional[int] = None,
    n_trim: int = 2,
    exclude_neighbors: int = 2,
    effective_distance: float = 0.8,
    temperature_K: float = 300.0,
) -> MDEmulationSelfResults:
    """
    MD emulation 원본 아이디어를 "레퍼런스 없이 self 방식"으로 적용하는 evaluator.

    각 SampleSpec 에 대해:
      - trajectory 로드 (+ stride, max_frames 적용)
      - contact-map 기반 feature 계산 (_compute_contact_features)
      - PCA 로 2D 투영 (_compute_pca_projection_2d)
      - 결과 2D 좌표를 SingleSampleMDEmulationSelf 로 묶어서 반환

    여기서는 MD_EMULATION_ASSET_DIR, reference_projections, precomputed projection_params
    등을 전혀 사용하지 않고, 각 샘플 자체에서 projection 을 학습합니다.
    """
    samples_proj: List[SingleSampleMDEmulationSelf] = []

    for spec in samples:
        print(f"[INFO] MD_EMULATION_SELF: loading {spec.name}")
        traj = md.load(spec.xtc_path.as_posix(), top=spec.pdb_path.as_posix())

        # stride 적용
        if stride > 1:
            traj = traj[::stride]

        # max_frames 제한
        if max_frames is not None and traj.n_frames > max_frames:
            traj = traj[:max_frames]

        if traj.n_frames < 3:
            print(
                f"[WARN] {spec.name}: n_frames={traj.n_frames} < 3, "
                "md_emulation_self 투영 건너뜁니다."
            )
            continue

        try:
            feats = _compute_contact_features(
                traj,
                n_trim=n_trim,
                exclude_neighbors=exclude_neighbors,
                effective_distance=effective_distance,
            )  # (T, D)
        except ValueError as e:
            print(f"[WARN] {spec.name}: feature 계산 실패: {e}. 건너뜁니다.")
            continue

        try:
            proj_xy = _compute_pca_projection_2d(feats)  # (T, 2)
        except ValueError as e:
            print(f"[WARN] {spec.name}: PCA 투영 실패: {e}. 건너뜁니다.")
            continue

        samples_proj.append(
            SingleSampleMDEmulationSelf(
                name=spec.name,
                proj_xy=proj_xy,
            )
        )

    return MDEmulationSelfResults(
        samples=samples_proj,
        temperature_K=temperature_K,
        n_trim=n_trim,
        exclude_neighbors=exclude_neighbors,
        effective_distance=effective_distance,
    )