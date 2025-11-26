# mybench/eval/folding_free_energies_self.py
from __future__ import annotations
from typing import Iterable, Optional, List

import mdtraj as md
import numpy as np

# BioEmu FOLDING_FREE_ENGERGIES 에서 쓰는 FNC 정의 재사용
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
    FNC 값으로부터 foldedness f_FNC(x)를 계산한다.

    논문 Eq. (6)에 해당하는 정의:
        f_FNC(x) = H(Q(x) - Q_threshold)

    여기서 H는 Heaviside step 함수이다.
    - fnc >= p_fold_thr 이면 1.0 (folded)
    - fnc <  p_fold_thr 이면 0.0 (unfolded)

    참고:
        이전 버전에서는 sigmoid(smooth) 함수를 사용했으나,
        현재 구현은 논문 정의에 맞게 step 함수로 변경하였다.
        steepness 인자는 인터페이스 호환성만 유지하며 실제 계산에는 사용하지 않는다.
    """
    return (fnc >= p_fold_thr).astype(float)


def _compute_dG_from_fnc(
    fnc: np.ndarray,
    temperature: float,
    p_fold_thr: float,
    steepness: float,
) -> float:
    """
    FNC 궤적으로부터 folding free energy ΔG (kcal/mol) 를 계산한다.

    절차:
        1) FNC(t)로부터 foldedness(t) = f_FNC(t) 를 계산
           (위 _foldedness_from_fnc, Eq. 6)
        2) p_fold = <foldedness> (시간 평균)
        3) ΔG = - k_B T ln( p_fold / (1 - p_fold) )  (Eq. 5 에서 정리)

    수치 안정성을 위해 p_fold 는 [1e-10, 1-1e-10] 범위로 잘라준다.
    """
    p_fold = _foldedness_from_fnc(
        fnc,
        p_fold_thr=p_fold_thr,
        steepness=steepness,
    ).mean()
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

    각 샘플에 대해:
        - trajectory 첫 프레임을 reference(native) 구조로 사용
        - BioEmu folding_free_energies.fraction_native_contacts.get_fnc_from_samples_trajectory
          로 FNC(t) 계산
        - FNC(t) → foldedness(t) (Heaviside step 함수, Eq. 6)
        - foldedness 평균 p_fold → ΔG = - k_B T ln(p/(1-p)) (Eq. 5)

    Args:
        samples:
            SampleSpec iterable (pdb_path, xtc_path, name, ... 포함)
        stride:
            프레임 스텝 (예: 10 → 0,10,20,... 만 사용)
        max_frames:
            최대 프레임 수 제한 (None 이면 제한 없음)
        temperature_K:
            ΔG 계산에 사용할 온도 (K 단위), 기본 295 K
        p_fold_thr:
            folded / unfolded 를 나누는 FNC threshold
        steepness:
            과거 sigmoid 버전과의 호환성을 위해 남겨둔 인자.
            현재 구현에서는 사용하지 않는다.

    Returns:
        FoldingFreeEnergyResults:
            각 샘플별 FNC(t), foldedness(t), ΔG, p_fold_mean 등을 포함
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

        foldedness = _foldedness_from_fnc(
            fnc,
            p_fold_thr=p_fold_thr,
            steepness=steepness,
        )
        p_fold_mean = float(foldedness.mean())
        dg = _compute_dG_from_fnc(
            fnc,
            temperature=temperature_K,
            p_fold_thr=p_fold_thr,
            steepness=steepness,
        )
        print(
            f"[DEBUG] FOLDING_FE_SELF: {spec.name} "
            f"p_fold_mean={p_fold_mean:.4f}, dG={dg:.4f} kcal/mol"
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