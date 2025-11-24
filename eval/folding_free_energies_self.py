# mybench/eval/folding_free_energies_self.py
from __future__ import annotations
from typing import Iterable, Optional, List

import mdtraj as md
import numpy as np

# BioEmu FOLDING_FREE_ENERGIES에서 쓰는 FNC 정의 재사용
from bioemu_benchmarks.eval.folding_free_energies.fraction_native_contacts import (
    get_fnc_from_samples_trajectory,
    FNCSettings,
)

from results import SingleSampleFoldingFE, FoldingFreeEnergyResults
from samples import SampleSpec  # 기존 basic_stats, fnc_self 에서 쓰시던 타입 그대로 사용

# BioEmu free_energies.py 에 정의된 볼츠만 상수와 동일 값
K_BOLTZMANN = 0.001987203599772605  # kcal / (mol·K)


def _foldedness_from_fnc(
    fnc: np.ndarray,
    p_fold_thr: float,
    steepness: float,
) -> np.ndarray:
    """
    BioEmu free_energies._foldedness_from_fnc 와 같은 정의:
    FNC → foldedness p_fold (0~1) 로 가는 sigmoid.
    """
    return 1.0 / (1.0 + np.exp(-2.0 * steepness * (fnc - p_fold_thr)))


def _compute_dG_from_fnc(
    fnc: np.ndarray,
    temperature: float,
    p_fold_thr: float,
    steepness: float,
) -> float:
    """
    BioEmu free_energies._compute_dG 와 같은 수식:
    - FNC → foldedness p_fold
    - p_fold 평균 → dG = - k_B T ln(p/(1-p))
    """
    p_fold = _foldedness_from_fnc(fnc, p_fold_thr=p_fold_thr, steepness=steepness).mean()
    p_fold = float(np.clip(p_fold, 1e-10, 1.0 - 1e-10))

    ratio = p_fold / (1.0 - p_fold)
    ratio = float(np.clip(ratio, 1e-10, 1e10))

    dG = -np.log(ratio) * K_BOLTZMANN * temperature
    return float(dG)


def evaluate_folding_free_energies_self(
    samples: Iterable[SampleSpec],
    stride: int = 1,
    max_frames: Optional[int] = None,
    temperature_K: float = 295.0,
    p_fold_thr: float = 0.5,
    steepness: float = 10.0,
) -> FoldingFreeEnergyResults:
    """
    FOLDING_FREE_ENERGIES 의 아이디어를
    '우리 샘플만' 가지고 self-reference 로 돌리는 버전.

    - 각 샘플에 대해:
      * trajectory 첫 프레임을 reference 구조로 사용
      * BioEmu folding_free_energies.fraction_native_contacts.get_fnc_from_samples_trajectory
        로 FNC(t) 계산
      * FNC(t) → foldedness(t) (sigmoid)
      * foldedness 평균 → dG (kcal/mol)

    Args:
        samples:
            SampleSpec iterable (pdb_path, xtc_path, name, ... 포함)
        stride:
            프레임 스텝 (예: 10 → 0,10,20,... 만 사용)
        max_frames:
            최대 프레임 수 제한 (None 이면 제한 없음)
        temperature_K:
            dG 계산에 사용할 온도 (K 단위), 기본 295 K
        p_fold_thr:
            foldedness 0.5 가 되는 FNC 기준값
        steepness:
            sigmoid 기울기

    Returns:
        FoldingFreeEnergyResults:
            각 샘플별 FNC(t), foldedness(t), dG, p_fold_mean 등을 포함
    """
    metrics_list: List[SingleSampleFoldingFE] = []

    for spec in samples:
        print(f"[INFO] FOLDING_FE_SELF: loading {spec.name}")
        traj = md.load(spec.xtc_path.as_posix(), top=spec.pdb_path.as_posix())

        # stride 적용
        if stride > 1:
            traj = traj[::stride]

        # max_frames 제한
        if max_frames is not None and traj.n_frames > max_frames:
            traj = traj[:max_frames]

        n_frames = traj.n_frames
        if n_frames == 0:
            continue

        frame_idx = np.arange(n_frames, dtype=int)

        # 기준 구조: 첫 프레임 (1-frame Trajectory)
        reference = traj[0]

        # BioEmu의 FNC 정의를 그대로 사용
        fnc = get_fnc_from_samples_trajectory(
            samples=traj,
            reference_conformation=reference,
            sequence_separation=FNCSettings.sequence_separation,
            contact_cutoff=FNCSettings.contact_cutoff,
            contact_beta=FNCSettings.contact_beta,
            contact_lambda=FNCSettings.contact_lambda,
            contact_delta=FNCSettings.contact_delta,
        )
        # fnc: shape (n_frames,), 0~1

        foldedness = _foldedness_from_fnc(fnc, p_fold_thr=p_fold_thr, steepness=steepness)
        p_fold_mean = float(foldedness.mean())
        dg = _compute_dG_from_fnc(
            fnc,
            temperature=temperature_K,
            p_fold_thr=p_fold_thr,
            steepness=steepness,
        )

        metrics_list.append(
            SingleSampleFoldingFE(
                name=spec.name,
                frame_idx=frame_idx,
                fnc=fnc,
                foldedness=foldedness,
                dg_kcal_per_mol=dg,
                p_fold_mean=p_fold_mean,
            )
        )

    return FoldingFreeEnergyResults(
        samples=metrics_list,
        temperature_K=temperature_K,
        p_fold_thr=p_fold_thr,
        steepness=steepness,
    )
