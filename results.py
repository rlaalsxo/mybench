# mybench/results.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from bioemu_benchmarks.eval.multiconf.plot import plot_smoothed_1d_free_energy
from bioemu_benchmarks.eval.md_emulation.state_metric import (
    DistributionMetricSettings,
    DistributionMetrics2D,
)
from matplotlib.patches import Circle
import mdtraj as md

# BioEmu free_energies.py 와 동일한 볼츠만 상수 (kcal / (mol·K))
K_BOLTZMANN = 0.001987203599772605  # kcal / (mol·K)

class BenchmarkResults(ABC):
    @abstractmethod
    def get_aggregate_metrics(self) -> Dict[str, float]:
        ...

    @abstractmethod
    def save_results(self, output_dir: Path) -> None:
        ...

    @abstractmethod
    def plot(self, output_dir: Path) -> None:
        ...

# ----------------------------------------------------------------------
# BASIC STATS (RMSD / Rg)
# ----------------------------------------------------------------------

@dataclass
class SingleSampleMetrics:
    name: str
    rmsd_nm: np.ndarray
    rg_nm: np.ndarray

@dataclass
class BasicStatsResults(BenchmarkResults):
    samples: List[SingleSampleMetrics]

    def get_aggregate_metrics(self) -> Dict[str, float]:
        if not self.samples:
            return {}
        rmsd_means = [float(s.rmsd_nm.mean()) for s in self.samples]
        rg_means = [float(s.rg_nm.mean()) for s in self.samples]
        return {
            "num_samples": float(len(self.samples)),
            "rmsd_mean_nm_over_samples": float(np.mean(rmsd_means)),
            "rmsd_std_nm_over_samples": float(np.std(rmsd_means)),
            "rg_mean_nm_over_samples": float(np.mean(rg_means)),
            "rg_std_nm_over_samples": float(np.std(rg_means)),
        }

    def save_results(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # summaries = []
        # for s in self.samples:
        #     sample_dir = output_dir / s.name
        #     sample_dir.mkdir(parents=True, exist_ok=True)

        #     n_frames = len(s.rmsd_nm)
        #     df = pd.DataFrame(
        #         {
        #             "frame": np.arange(n_frames, dtype=int),
        #             "rmsd_nm": s.rmsd_nm,
        #             "rg_nm": s.rg_nm,
        #         }
        #     )
        #     df.to_csv(sample_dir / "metrics.csv", index=False)

        #     summary = {
        #         "sample": s.name,
        #         "n_frames": int(n_frames),
        #         "rmsd_mean_nm": float(np.mean(s.rmsd_nm)),
        #         "rmsd_std_nm": float(np.std(s.rmsd_nm)),
        #         "rmsd_min_nm": float(np.min(s.rmsd_nm)),
        #         "rmsd_max_nm": float(np.max(s.rmsd_nm)),
        #         "rg_mean_nm": float(np.mean(s.rg_nm)),
        #         "rg_std_nm": float(np.std(s.rg_nm)),
        #         "rg_min_nm": float(np.min(s.rg_nm)),
        #         "rg_max_nm": float(np.max(s.rg_nm)),
        #     }
        #     with open(sample_dir / "summary.json", "w") as f:
        #         json.dump(summary, f, indent=2, sort_keys=True)

        #     summaries.append(summary)

        # if summaries:
        #     summary_df = pd.DataFrame(summaries)
        #     summary_df.to_csv(output_dir / "all_samples_summary.csv", index=False)

    def plot(self, output_dir: Path) -> None:
        """
        각 샘플마다:
        - RMSD vs Frame  → rmsd_vs_frame.png
        - Rg vs Frame    → rg_vs_frame.png
        - RMSD histogram → rmsd_hist.png
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # for s in self.samples:
        #     sample_dir = output_dir / s.name
        #     sample_dir.mkdir(parents=True, exist_ok=True)

        #     n_frames = len(s.rmsd_nm)
        #     frames = np.arange(n_frames, dtype=int)

        #     # 1) RMSD vs Frame
        #     fig, ax = plt.subplots(figsize=(8, 4))
        #     ax.plot(frames, s.rmsd_nm)
        #     ax.set_xlabel("Frame")
        #     ax.set_ylabel("RMSD (nm)")
        #     ax.set_title(f"{s.name} - RMSD vs Frame")
        #     fig.savefig(sample_dir / "rmsd_vs_frame.png", dpi=200)
        #     plt.close(fig)

        #     # 2) Rg vs Frame
        #     fig, ax = plt.subplots(figsize=(8, 4))
        #     ax.plot(frames, s.rg_nm)
        #     ax.set_xlabel("Frame")
        #     ax.set_ylabel("Radius of gyration (nm)")
        #     ax.set_title(f"{s.name} - Rg vs Frame")
        #     fig.savefig(sample_dir / "rg_vs_frame.png", dpi=200)
        #     plt.close(fig)

        #     # 3) RMSD histogram
        #     fig, ax = plt.subplots(figsize=(8, 4))
        #     ax.hist(s.rmsd_nm, bins=40)
        #     ax.set_xlabel("RMSD (nm)")
        #     ax.set_ylabel("Count")
        #     ax.set_title(f"{s.name} - RMSD distribution")
        #     fig.savefig(sample_dir / "rmsd_hist.png", dpi=200)
        #     plt.close(fig)

# ----------------------------------------------------------------------
# FNC SELF (fraction of native contacts 기반 분석)
# ----------------------------------------------------------------------

@dataclass
class SingleSampleFNC:
    """
    한 샘플(trajectory)에 대한 FNC 정보.

    name      : 샘플 이름
    frame_idx : 프레임 인덱스 (0, 1, 2, ...)
    fnc       : 각 프레임에서의 fraction of native contacts 값
    rmsd_nm   : 각 프레임에서의 RMSD (nm, 첫 프레임 기준)
    """
    name: str
    frame_idx: np.ndarray
    fnc: np.ndarray
    rmsd_nm: np.ndarray

@dataclass
class FNCResults(BenchmarkResults):
    """
    여러 샘플에 대해 self-reference FNC를 계산한 결과 모음.

    samples:
        각 샘플의 FNC 시계열
    coverage_thresholds:
        coverage 곡선에 사용한 threshold 배열 (shape: [n_thresholds])
    coverage_bootstrap:
        bootstrap 반복마다의 coverage 곡선 (shape: [n_bootstrap, n_thresholds])
    krecall_bootstrap:
        샘플별 k-recall (mean, std) 딕셔너리
    """
    samples: List[SingleSampleFNC]
    coverage_thresholds: np.ndarray | None = None
    coverage_bootstrap: np.ndarray | None = None
    krecall_bootstrap: Dict[str, Tuple[float, float]] | None = None

    def get_aggregate_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if not self.samples:
            return metrics

        # 전체 FNC 분포 요약
        all_fnc = np.concatenate([s.fnc for s in self.samples])
        metrics.update(
            {
                "fnc_mean": float(all_fnc.mean()),
                "fnc_std": float(all_fnc.std()),
                "fnc_min": float(all_fnc.min()),
                "fnc_max": float(all_fnc.max()),
            }
        )

        # coverage AUC (bootstrap 평균 기준)
        if (
            self.coverage_thresholds is not None
            and self.coverage_bootstrap is not None
            and self.coverage_bootstrap.size > 0
        ):
            cov_mean = self.coverage_bootstrap.mean(axis=0)
            coverage_auc = float(np.trapz(cov_mean, self.coverage_thresholds))
            metrics["fnc_coverage_auc"] = coverage_auc

        # k-recall 평균 / 표준편차의 평균
        if self.krecall_bootstrap:
            k_means = [m for (m, _) in self.krecall_bootstrap.values()]
            k_stds = [s for (_, s) in self.krecall_bootstrap.values()]
            metrics["fnc_krecall_mean_mean"] = float(np.mean(k_means))
            metrics["fnc_krecall_std_mean"] = float(np.mean(k_stds))

        return metrics

    def save_results(self, output_dir: Path) -> None:
        """
        샘플별 FNC 시계열과 요약 통계,
        coverage / k-recall 요약을 디스크에 저장.
        """
        # output_dir.mkdir(parents=True, exist_ok=True)

        # summaries = []
        # for s in self.samples:
        #     sample_dir = output_dir / s.name
        #     sample_dir.mkdir(parents=True, exist_ok=True)

        #     n_frames = len(s.fnc)
        #     df = pd.DataFrame(
        #         {
        #             "frame": s.frame_idx,
        #             "fnc": s.fnc,
        #         }
        #     )
        #     df.to_csv(sample_dir / "fnc_timeseries.csv", index=False)

        #     summary = {
        #         "sample": s.name,
        #         "n_frames": int(n_frames),
        #         "fnc_mean": float(np.mean(s.fnc)),
        #         "fnc_std": float(np.std(s.fnc)),
        #         "fnc_min": float(np.min(s.fnc)),
        #         "fnc_max": float(np.max(s.fnc)),
        #     }
        #     with open(sample_dir / "fnc_summary.json", "w") as f:
        #         json.dump(summary, f, indent=2, sort_keys=True)

        #     summaries.append(summary)

        # if summaries:
        #     summary_df = pd.DataFrame(summaries)
        #     summary_df.to_csv(output_dir / "all_samples_fnc_summary.csv", index=False)

        # # coverage / k-recall 요약 저장
        # if (
        #     self.coverage_thresholds is not None
        #     and self.coverage_bootstrap is not None
        #     and self.coverage_bootstrap.size > 0
        # ):
        #     cov_mean = self.coverage_bootstrap.mean(axis=0)
        #     cov_std = self.coverage_bootstrap.std(axis=0)
        #     df_cov = pd.DataFrame(
        #         {
        #             "threshold": self.coverage_thresholds,
        #             "coverage_mean": cov_mean,
        #             "coverage_std": cov_std,
        #         }
        #     )
        #     df_cov.to_csv(output_dir / "fnc_coverage_bootstrap.csv", index=False)

        # if self.krecall_bootstrap:
        #     df_k = pd.DataFrame(
        #         [
        #             {
        #                 "sample": name,
        #                 "krecall_mean": float(m),
        #                 "krecall_std": float(s),
        #             }
        #             for name, (m, s) in self.krecall_bootstrap.items()
        #         ]
        #     )
        #     df_k.to_csv(output_dir / "fnc_krecall_bootstrap.csv", index=False)

    def plot(self, output_dir: Path) -> None:
        """
        각 샘플마다:
        - FNC vs Frame
        - FNC 기반 1D free energy (–log p(FNC), 0~1 구간, 임의 단위)

        추가로:
        - bin 기반 free energy 그리드 및 확률을 CSV로 저장
        - 각 프레임별 free energy 값을 CSV로 저장
        """
        # output_dir.mkdir(parents=True, exist_ok=True)

        # from bioemu_benchmarks.eval.multiconf.plot import plot_smoothed_1d_free_energy

        # print(
        #     "[DEBUG] using plot_smoothed_1d_free_energy from "
        #     "bioemu_benchmarks.eval.multiconf.plot:",
        #     plot_smoothed_1d_free_energy,
        # )

        # for s in self.samples:
        #     sample_dir = output_dir / s.name
        #     sample_dir.mkdir(parents=True, exist_ok=True)

        #     # 1) FNC vs Frame
        #     fig, ax = plt.subplots(figsize=(8, 4))
        #     ax.plot(s.frame_idx, s.fnc)
        #     ax.set_xlabel("Frame")
        #     ax.set_ylabel("fraction of native contacts")
        #     ax.set_title(f"{s.name} - FNC vs Frame")
        #     fig.savefig(sample_dir / "fnc_vs_frame.png", dpi=200)
        #     plt.close(fig)

        #     # 2) 우리가 직접 계산하는 1D free energy (–log p, arbitrary units)
        #     #    - histogram: [0,1] 구간, 50 bins, density=True
        #     hist, edges = np.histogram(
        #         s.fnc,
        #         bins=50,
        #         range=(0.0, 1.0),
        #         density=True,
        #     )
        #     centers = 0.5 * (edges[:-1] + edges[1:])   # bin center
        #     P = hist                                  # density=True → 확률밀도 (적분 1)
        #     F = -np.log(P + 1e-12)                    # –log p, 임의 단위

        #     finite = np.isfinite(F)
        #     if np.any(finite):
        #         F = F - np.min(F[finite])             # 최소값을 0으로 shift

        #         F_fin = F[finite]
        #         print(
        #             f"[DEBUG] FNC_FE_1D_GRID {s.name}: "
        #             f"min(F_grid)={F_fin.min():.3f}, "
        #             f"max(F_grid)={F_fin.max():.3f}, "
        #             f"mean(F_grid)={F_fin.mean():.3f}"
        #         )

        #         # bin 기반 free energy 그리드 CSV
        #         df_grid = pd.DataFrame(
        #             {
        #                 "fnc_center": centers,
        #                 "prob_density": P,
        #                 "free_energy_arb": F,
        #             }
        #         )
        #         df_grid.to_csv(
        #             sample_dir / "fnc_free_energy_1d_grid.csv",
        #             index=False,
        #         )

        #         # 각 프레임별 free energy (해당 bin의 F 값 부여)
        #         idx = np.searchsorted(edges, s.fnc, side="right") - 1
        #         valid = (idx >= 0) & (idx < F.shape[0])

        #         fe_per_frame = np.full(s.fnc.shape, np.nan, dtype=float)
        #         fe_per_frame[valid] = F[idx[valid]]

        #         fe_valid = fe_per_frame[np.isfinite(fe_per_frame)]
        #         if fe_valid.size > 0:
        #             print(
        #                 f"[DEBUG] FNC_FE_1D_FRAMES {s.name}: "
        #                 f"min(F_frame)={fe_valid.min():.3f}, "
        #                 f"max(F_frame)={fe_valid.max():.3f}, "
        #                 f"mean(F_frame)={fe_valid.mean():.3f}, "
        #                 f"n_valid={fe_valid.size}/{fe_per_frame.size}"
        #             )
        #         else:
        #             print(
        #                 f"[DEBUG] FNC_FE_1D_FRAMES {s.name}: "
        #                 f"no valid frame free energies (all NaN)."
        #             )

        #         df_frames = pd.DataFrame(
        #             {
        #                 "frame": s.frame_idx,
        #                 "fnc": s.fnc,
        #                 "free_energy_arb": fe_per_frame,
        #             }
        #         )
        #         df_frames.to_csv(
        #             sample_dir / "fnc_free_energy_1d_per_frame.csv",
        #             index=False,
        #         )

        #     # 3) BioEmu의 plot_smoothed_1d_free_energy로 그림 (기존 방식 유지)
        #     fig2, ax2 = plt.subplots(figsize=(8, 4))
        #     plot_smoothed_1d_free_energy(
        #         s.fnc,
        #         range=(0.0, 1.0),
        #         ax=ax2,
        #     )
        #     ax2.set_xlabel("fraction of native contacts")
        #     ax2.set_ylabel("free energy (arb. units)")
        #     ax2.set_title(f"{s.name} - FNC free energy")
        #     fig2.savefig(sample_dir / "fnc_free_energy.png", dpi=200)
        #     plt.close(fig2)

def _find_1d_basins(
    F: np.ndarray,
    local_min_bins: List[int],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    1D free energy F에서 local minimum 인덱스를 기준으로 basin 인덱스들을 계산.
    F는 이미 스무딩된 free energy(F_s)를 쓰는 것을 가정한다.
    """
    F = np.asarray(F, float)
    n = len(F)

    basins: List[np.ndarray] = []
    basin_bin_id = np.full(n, -1, dtype=int)

    if len(local_min_bins) == 0:
        return basins, basin_bin_id

    basin_id = 0

    for m in local_min_bins:
        if not (0 <= m < n):
            continue

        # minimum에서 왼쪽으로 갈 때 F가 단조 증가(또는 일정)하는 구간을 포함
        left = m
        while left - 1 >= 0:
            if not np.isfinite(F[left - 1]):
                break
            # 다시 내려가기 시작하면 중단
            if F[left - 1] < F[left]:
                break
            left -= 1

        # minimum에서 오른쪽으로 갈 때 F가 단조 증가(또는 일정)하는 구간을 포함
        right = m
        while right + 1 < n:
            if not np.isfinite(F[right + 1]):
                break
            if F[right + 1] < F[right]:
                break
            right += 1

        idxs = np.arange(left, right + 1)
        basins.append(idxs)
        basin_bin_id[idxs] = basin_id
        basin_id += 1

    return basins, basin_bin_id


def _smooth_free_energy(F: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    F = np.asarray(F, float)
    if F.size < 3:
        return F.copy()

    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()

    F_pad = np.pad(F, radius, mode="edge")
    F_smooth = np.convolve(F_pad, kernel, mode="same")
    F_smooth = F_smooth[radius:-radius]
    return F_smooth


def _find_local_and_global_minima_1d(
    F: np.ndarray,
    P: np.ndarray,
    min_prominence: float = 1.0,
    smoothing_sigma: float = 2.0,
    min_width: int = 3,
) -> Tuple[List[int], int, np.ndarray]:
    """
    1D free energy F에서 smoothing 후 local minima와 global minimum bin을 찾는다.
    반환값:
        local_min_bins: local minimum 인덱스 리스트 (스무딩된 F_s 기준)
        global_min_bin: global minimum 인덱스 (F_s 기준)
        F_s: 스무딩된 free energy 배열
    """
    F = np.asarray(F, float)
    n = len(F)
    if n < 3:
        return [], -1, F.copy()

    # smoothing
    F_s = _smooth_free_energy(F, sigma=smoothing_sigma)

    # global minimum (스무딩된 F_s 기준)
    global_min_bin = int(np.argmin(F_s))

    local_min: List[int] = []

    for i in range(1, n - 1):
        if not (np.isfinite(F_s[i]) and np.isfinite(F_s[i - 1]) and np.isfinite(F_s[i + 1])):
            continue

        # strict local minimum
        if not (F_s[i] < F_s[i - 1] and F_s[i] < F_s[i + 1]):
            continue

        # prominence
        left_saddle = np.max(F_s[: i + 1])
        right_saddle = np.max(F_s[i:])
        prominence = min(left_saddle - F_s[i], right_saddle - F_s[i])
        if prominence < min_prominence:
            continue

        # width: valley 폭 (F_s 기준)
        left = i
        while left - 1 >= 0 and np.isfinite(F_s[left - 1]):
            # 왼쪽으로 갈수록 F_s가 단조 증가(또는 일정)하는 구간만 포함
            if F_s[left - 1] < F_s[left]:
                break
            left -= 1

        right = i
        while right + 1 < n and np.isfinite(F_s[right + 1]):
            if F_s[right + 1] < F_s[right]:
                break
            right += 1

        width = right - left + 1
        if width < min_width:
            continue

        local_min.append(i)

    if global_min_bin not in local_min:
        local_min.append(global_min_bin)

    return sorted(local_min), global_min_bin, F_s

def _pick_rep_frame_idx(fnc: np.ndarray, target_fnc: float) -> Optional[int]:
    fnc = np.asarray(fnc, float)
    mask = np.isfinite(fnc)
    if not np.any(mask):
        return None
    idx_all = np.where(mask)[0]
    idx_local = np.argmin(np.abs(fnc[mask] - target_fnc))
    return int(idx_all[idx_local])

# ----------------------------------------------------------------------
# FOLDING FREE ENERGIES (self-reference 버전)
# ----------------------------------------------------------------------


@dataclass
class SingleSampleFoldingFE:
    """
    한 샘플(trajectory)에 대한 folding free energy 관련 정보.
    """
    name: str
    frame_idx: np.ndarray
    fnc: np.ndarray
    foldedness: np.ndarray
    dg_kcal_per_mol: float
    p_fold_mean: float
    pdb_path: Path          # 추가
    xtc_path: Path 

@dataclass
class FoldingFreeEnergyResults(BenchmarkResults):
    """
    FOLDING_FREE_ENERGIES 아이디어를 self-reference 로 적용한 결과 모음.
    """
    samples: List[SingleSampleFoldingFE]
    temperature_K: float = 295.0
    p_fold_thr: float = 0.5
    steepness: float = 10.0

    def get_aggregate_metrics(self) -> Dict[str, float]:
        if not self.samples:
            return {}

        dgs = np.array([s.dg_kcal_per_mol for s in self.samples], dtype=float)
        p_folds = np.array([s.p_fold_mean for s in self.samples], dtype=float)
        fnc_means = np.array([float(s.fnc.mean()) for s in self.samples], dtype=float)

        return {
            "num_samples": float(len(self.samples)),
            "dg_mean_kcal_per_mol": float(dgs.mean()),
            "dg_std_kcal_per_mol": float(dgs.std()),
            "p_fold_mean_over_samples": float(p_folds.mean()),
            "p_fold_std_over_samples": float(p_folds.std()),
            "fnc_mean_over_samples": float(fnc_means.mean()),
            "fnc_std_over_samples": float(fnc_means.std()),
        }

    def save_results(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            n_frames = len(s.fnc)
            df = pd.DataFrame(
                {
                    "frame": s.frame_idx,
                    "fnc": s.fnc,
                    "foldedness": s.foldedness,
                }
            )
            # df.to_csv(sample_dir / "folding_fe_timeseries.csv", index=False)

            summary = {
                "sample": s.name,
                "n_frames": int(n_frames),
                "fnc_mean": float(np.mean(s.fnc)),
                "fnc_std": float(np.std(s.fnc)),
                "fnc_min": float(np.min(s.fnc)),
                "fnc_max": float(np.max(s.fnc)),
                "p_fold_mean": float(s.p_fold_mean),
                "p_fold_std": float(s.foldedness.std()),
                "dg_kcal_per_mol": float(s.dg_kcal_per_mol),
                "temperature_K": float(self.temperature_K),
                "p_fold_thr": float(self.p_fold_thr),
                "steepness": float(self.steepness),
            }
            # with open(sample_dir / "folding_fe_summary.json", "w") as f:
            #     json.dump(summary, f, indent=2, sort_keys=True)

            summaries.append(summary)

        # if summaries:
        #     summary_df = pd.DataFrame(summaries)
        #     summary_df.to_csv(output_dir / "all_samples_folding_fe_summary.csv", index=False)

    def plot(self, output_dir: Path) -> None:
        """
        각 샘플마다:
        - FNC 기반 1D free energy (–log p(FNC), 0~1 구간, 임의 단위)
        - basin 탐지 및 프레임별 basin_id 저장
        - free-energy 곡선 위의 global / local minimum 을
        점선 + 'g_m' / 'l_m' 으로 표시
        - 각 minimum에 대응하는 representative 구조를 PDB로 저장
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # FNC 값(target_fnc)에 가장 가까운 프레임 인덱스를 고르는 보조 함수
        def _pick_rep_frame_idx(fnc: np.ndarray, target_fnc: float):
            fnc = np.asarray(fnc, float)
            mask = np.isfinite(fnc)
            if not np.any(mask):
                return None
            idx_all = np.where(mask)[0]
            idx_local = np.argmin(np.abs(fnc[mask] - target_fnc))
            return int(idx_all[idx_local])

        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            # 1) FNC 분포 → 1D free energy grid
            hist, edges = np.histogram(
                s.fnc,
                bins=50,
                range=(0.0, 1.0),
                density=True,
            )
            centers = 0.5 * (edges[:-1] + edges[1:])
            P = hist
            F = -np.log(P + 1e-12)

            finite = np.isfinite(F)
            if not np.any(finite):
                print(
                    f"[WARN] FOLDING_FNC_FE_1D_GRID {s.name}: "
                    f"유효한 free-energy grid 가 없어 plot 생략."
                )
                continue

            # min(F)=0 으로 shift
            F = F - np.min(F[finite])

            F_fin = F[finite]
            print(
                f"[DEBUG] FOLDING_FNC_FE_1D_GRID {s.name}: "
                f"min(F_grid)={F_fin.min():.3f}, "
                f"max(F_grid)={F_fin.max():.3f}, "
                f"mean(F_grid)={F_fin.mean():.3f}"
            )

            # 1-1) local / global minima 및 스무딩된 free energy F_s
            local_min_bins, global_min_bin, F_s = _find_local_and_global_minima_1d(
                F,
                P,
                min_prominence=1.5,
                smoothing_sigma=2.0,
                min_width=3,
            )

            # 1-2) basin 탐지 (스무딩된 free energy F_s 기준)
            basins, basin_bin_id = _find_1d_basins(
                F_s,
                local_min_bins=local_min_bins,
            )

            # 1-3) grid 정보 CSV (원본 F와 basin_id 저장)
            df_grid = pd.DataFrame(
                {
                    "fnc_center": centers,
                    "prob_density": P,
                    "free_energy_arb": F,
                    "basin_id": basin_bin_id.astype(int),
                }
            )
            df_grid.to_csv(
                sample_dir / "folding_fnc_free_energy_1d_grid.csv",
                index=False,
            )

            # 2) 프레임별 free energy 및 basin_id
            idx = np.searchsorted(edges, s.fnc, side="right") - 1
            valid = (idx >= 0) & (idx < F.shape[0])

            fe_per_frame = np.full(s.fnc.shape, np.nan, dtype=float)
            fe_per_frame[valid] = F[idx[valid]]

            basin_id_per_frame = np.full(s.fnc.shape, -1, dtype=int)
            valid_basin = valid & (basin_bin_id[idx] >= 0)
            basin_id_per_frame[valid_basin] = basin_bin_id[idx[valid_basin]]

            fe_valid = fe_per_frame[np.isfinite(fe_per_frame)]
            if fe_valid.size > 0:
                print(
                    f"[DEBUG] FOLDING_FNC_FE_1D_FRAMES {s.name}: "
                    f"min(F_frame)={fe_valid.min():.3f}, "
                    f"max(F_frame)={fe_valid.max():.3f}, "
                    f"mean(F_frame)={fe_valid.mean():.3f}, "
                    f"n_valid={fe_valid.size}/{fe_per_frame.size}"
                )
            else:
                print(
                    f"[DEBUG] FOLDING_FNC_FE_1D_FRAMES {s.name}: "
                    f"no valid frame free energies (all NaN)."
                )

            df_frames = pd.DataFrame(
                {
                    "frame": s.frame_idx,
                    "fnc": s.fnc,
                    "foldedness": s.foldedness,
                    "free_energy_arb": fe_per_frame,
                    "basin_id": basin_id_per_frame,
                }
            )
            df_frames.to_csv(
                sample_dir / "folding_fnc_free_energy_1d_per_frame.csv",
                index=False,
            )

            # 3) g_m / l_m 에 해당하는 representative frame 구조를 PDB로 저장
            try:
                # g_m 에 해당하는 목표 FNC 값
                gm_fnc = centers[global_min_bin]
                # 각 l_m 에 해당하는 목표 FNC 값들 (g_m 제외)
                lm_fncs = [centers[b] for b in local_min_bins if b != global_min_bin]

                gm_frame = _pick_rep_frame_idx(s.fnc, gm_fnc)
                lm_frames = [_pick_rep_frame_idx(s.fnc, q) for q in lm_fncs]

                # trajectory 로드 (필요할 때만)
                if gm_frame is not None or any(fr is not None for fr in lm_frames):
                    traj = md.load(
                        s.xtc_path.as_posix(),
                        top=s.pdb_path.as_posix(),
                    )

                    if gm_frame is not None and 0 <= gm_frame < traj.n_frames:
                        gm_traj = traj[gm_frame]
                        gm_traj.save_pdb(
                            (sample_dir / f"structure_gm.pdb").as_posix()
                        )

                    for k, fr in enumerate(lm_frames):
                        if fr is None or not (0 <= fr < traj.n_frames):
                            continue
                        lm_traj = traj[fr]
                        lm_traj.save_pdb(
                            (sample_dir / f"structure_lm{k}.pdb").as_posix()
                        )
            except Exception as e:
                print(f"[WARN] FOLDING_FNC_FE_1D_PDB {s.name}: 구조 저장 중 오류 발생: {e}")

            # 4) 1D free energy 플롯 + g_m / l_m 점선 표기
            fig3, ax3 = plt.subplots(figsize=(8, 4))

            baseline = float(np.min(F_s)) - 0.5

            # 밝은 회색 영역 + 검은 free-energy 곡선 (스무딩된 F_s 기준)
            ax3.fill_between(
                centers,
                F_s,
                baseline,
                alpha=0.4,
                color="0.7",
            )
            ax3.plot(
                centers,
                F_s,
                linewidth=2.5,
                color="k",
            )

            ymin, ymax = ax3.get_ylim()
            height = ymax - ymin
            label_y = ymin + 0.05 * height  # 텍스트 y 위치

            # global minimum (F_s 기준)
            if global_min_bin >= 0:
                x_g = centers[global_min_bin]
                y_g = F_s[global_min_bin]
                ax3.plot(
                    [x_g, x_g],
                    [label_y, y_g],
                    linestyle=":",
                    linewidth=0.8,
                    color="k",
                )
                ax3.text(
                    x_g,
                    label_y,
                    "g_m",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="k",
                )

            # local minima (global 제외, F_s 기준)
            for b in local_min_bins:
                if b == global_min_bin:
                    continue
                x_l = centers[b]
                y_l = F_s[b]
                ax3.plot(
                    [x_l, x_l],
                    [label_y, y_l],
                    linestyle=":",
                    linewidth=0.8,
                    color="k",
                )
                ax3.text(
                    x_l,
                    label_y,
                    "l_m",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="k",
                )

            ax3.set_xlabel("fraction of native contacts")
            ax3.set_ylabel("free energy (arb. units)")
            ax3.set_title("results - FNC free energy")
            fig3.savefig(sample_dir / "fnc_free_energy.png", dpi=200)
            plt.close(fig3)

# ----------------------------------------------------------------------
# TICA 기반 2D free energy landscape
# ----------------------------------------------------------------------

@dataclass
class SingleSampleTICA:
    """
    한 샘플(trajectory)에 대한 TICA 좌표.

    name    : 샘플 이름
    tica_xy : shape (n_frames_eff, 2), 두 개의 TICA 성분 좌표
              (TICA lag 때문에 원래 frame 수보다 조금 줄어들 수 있음)
    """
    name: str
    tica_xy: np.ndarray


@dataclass
class TICAResults(BenchmarkResults):
    """
    여러 샘플에 대해 TICA 2D 좌표를 계산한 결과.

    samples:
        각 샘플의 2D TICA 좌표
    temperature_K:
        free-energy surface 계산에 사용하는 온도 (K).
        F(x, y) = -k_B T ln p(x, y) + const (kcal/mol) 형태로 사용됩니다.
    """
    samples: List[SingleSampleTICA]
    temperature_K: float = 300.0

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        간단한 요약 통계만 제공합니다.
        (필요하면 여기서 TICA 분산 등 다른 metric 을 더 넣을 수 있습니다.)
        """
        if not self.samples:
            return {}

        n_samples = len(self.samples)
        n_points = [s.tica_xy.shape[0] for s in self.samples]

        return {
            "num_samples": float(n_samples),
            "mean_points_per_sample": float(np.mean(n_points)),
            "std_points_per_sample": float(np.std(n_points)),
        }

    def save_results(self, output_dir: Path) -> None:
        """
        각 샘플별 TICA 좌표를 CSV 로 저장.
        """
        # output_dir.mkdir(parents=True, exist_ok=True)

        # summaries = []
        # for s in self.samples:
        #     sample_dir = output_dir / s.name
        #     sample_dir.mkdir(parents=True, exist_ok=True)

        #     n_points = s.tica_xy.shape[0]
        #     df = pd.DataFrame(
        #         {
        #             "tica1": s.tica_xy[:, 0],
        #             "tica2": s.tica_xy[:, 1],
        #         }
        #     )
        #     df.to_csv(sample_dir / "tica_coords.csv", index=False)

        #     summary = {
        #         "sample": s.name,
        #         "n_points": int(n_points),
        #         "tica1_mean": float(np.mean(s.tica_xy[:, 0])),
        #         "tica2_mean": float(np.mean(s.tica_xy[:, 1])),
        #     }
        #     with open(sample_dir / "tica_summary.json", "w") as f:
        #         json.dump(summary, f, indent=2, sort_keys=True)

        #     summaries.append(summary)

        # if summaries:
        #     summary_df = pd.DataFrame(summaries)
        #     summary_df.to_csv(output_dir / "all_samples_tica_summary.csv", index=False)

    def plot(self, output_dir: Path) -> None:
        """
        각 샘플마다:
        - TICA1 vs TICA2 2D free-energy surface

        BioEmu MD emulation metric 에서 쓰는 것과 동일한 방식으로
        분포를 resample + Gaussian noise 로 스무딩한 뒤,
        F(x, y) = -k_B T ln p(x, y) 를 계산합니다.

        색 범위는 DistributionMetricSettings.energy_cutoff (기본 4 kcal/mol)
        까지만 보여주고, 샘플이 거의 없는 영역은 마스킹합니다.
        """
        # output_dir.mkdir(parents=True, exist_ok=True)

        # # BioEmu md_emulation 의 density 유틸 재사용
        # from bioemu_benchmarks.eval.md_emulation.state_metric import (
        #     DistributionMetricSettings,
        #     resample_with_noise,
        #     histogram_bin_edges,
        #     compute_density_2D,
        # )

        # settings = DistributionMetricSettings()

        # for s in self.samples:
        #     sample_dir = output_dir / s.name
        #     sample_dir.mkdir(parents=True, exist_ok=True)

        #     xy = s.tica_xy
        #     if xy.shape[0] < 10:
        #         print(f"[WARN] {s.name}: TICA points < 10, free-energy plot 생략.")
        #         continue

        #     # 1) resample + Gaussian noise (스무딩)
        #     xy_noised = resample_with_noise(
        #         xy,
        #         num_samples=settings.n_resample,
        #         sigma=settings.sigma_resample,
        #         rng=None,
        #     )

        #     # 2) bin edge 계산 (padding 포함)
        #     edges_x = histogram_bin_edges(
        #         xy_noised[:, 0],
        #         num_bins=settings.num_bins,
        #         padding=settings.padding,
        #     )
        #     edges_y = histogram_bin_edges(
        #         xy_noised[:, 1],
        #         num_bins=settings.num_bins,
        #         padding=settings.padding,
        #     )

        #     # 3) density 계산 (이미 정규화된 p(x,y))
        #     P = compute_density_2D(xy_noised, edges_x, edges_y)  # shape (nx, ny)

        #     # 4) free-energy surface: F = -k_B T ln p + const
        #     kBT = K_BOLTZMANN * self.temperature_K
        #     F = -kBT * np.log(P + 1e-12)

        #     finite = np.isfinite(F) & (P > 0.0)
        #     if not np.any(finite):
        #         print(f"[WARN] {s.name}: 유효한 density grid 가 없어 plot 생략.")
        #         continue

        #     # 최소값을 0으로 shift
        #     F = F - np.nanmin(F[finite])

        #     # 디버깅용 로그 (shift 전/후는 큰 차이 없음)
        #     F_fin = F[finite]
        #     print(
        #         f"[DEBUG] TICA_FE {s.name}: "
        #         f"min(F)={F_fin.min():.3f} kcal/mol, "
        #         f"max(F)={F_fin.max():.3f} kcal/mol, "
        #         f"mean(F)={F_fin.mean():.3f} kcal/mol"
        #     )

        #     # 5) 에너지 cutoff 및 마스킹
        #     F_clipped = np.minimum(F, settings.energy_cutoff)
        #     mask = P <= 0.0           # 샘플이 거의 없는 grid 는 가림
        #     F_masked = np.ma.array(F_clipped, mask=mask)

        #     # grid / density / FE 저장 (수치 확인용)
        #     np.save(sample_dir / "tica_free_energy_F_raw.npy", F)
        #     np.save(sample_dir / "tica_free_energy_F_clipped.npy", F_clipped)
        #     np.save(sample_dir / "tica_free_energy_P.npy", P)
        #     np.save(sample_dir / "tica_free_energy_xedges.npy", edges_x)
        #     np.save(sample_dir / "tica_free_energy_yedges.npy", edges_y)

        #     # 6) bin center 좌표
        #     xc = 0.5 * (edges_x[:-1] + edges_x[1:])
        #     yc = 0.5 * (edges_y[:-1] + edges_y[1:])
        #     Xc, Yc = np.meshgrid(xc, yc, indexing="ij")

        #     # 7) contourf 로 free-energy surface 시각화
        #     fig, ax = plt.subplots(figsize=(4, 4))
        #     cf = ax.contourf(
        #         Xc,
        #         Yc,
        #         F_masked,
        #         levels=20,
        #         cmap="hot",
        #     )
        #     cbar = fig.colorbar(cf, ax=ax)
        #     cbar.set_label("free energy (kcal/mol)")

        #     ax.set_xlabel("TICA 1")
        #     ax.set_ylabel("TICA 2")
        #     ax.set_title(f"{s.name} - TICA 2D free energy")

        #     fig.savefig(sample_dir / "tica_free_energy.png", dpi=300, bbox_inches="tight")
        #     plt.close(fig)

# ----------------------------------------------------------------------
# MD EMULATION SELF (contact-map 기반 2D projection free-energy)
# ----------------------------------------------------------------------

@dataclass
class SingleSampleMDEmulationSelf:
    """
    한 샘플(trajectory)에 대한 MD emulation-style 2D 투영 좌표.

    name    : 샘플 이름
    proj_xy : shape (n_frames, 2), contact-map feature 에 PCA 를 적용한 2D 좌표
    """
    name: str
    proj_xy: np.ndarray


@dataclass
class MDEmulationSelfResults(BenchmarkResults):
    """
    MD emulation 원본 아이디어를 '레퍼런스 없이 self' 로 적용한 결과 모음.

    samples:
        각 샘플의 2D projection 좌표 (contact-map 기반 PCA)
    temperature_K:
        free-energy surface 계산에 사용하는 온도 (K).
        F(x, y) = -k_B T ln p(x, y) + const (kcal/mol) 로 사용합니다.
    n_trim, exclude_neighbors, effective_distance:
        feature 계산에 사용한 설정값 (메타데이터 용도).
    """

    samples: List[SingleSampleMDEmulationSelf]
    temperature_K: float = 300.0
    n_trim: int = 2
    exclude_neighbors: int = 2
    effective_distance: float = 0.8

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        간단한 요약 통계:
        - 샘플 수
        - 샘플당 평균/표준편차 frame 수
        """
        if not self.samples:
            return {}

        n_samples = len(self.samples)
        n_points = np.array([s.proj_xy.shape[0] for s in self.samples], dtype=float)

        return {
            "num_samples": float(n_samples),
            "mean_points_per_sample": float(n_points.mean()),
            "std_points_per_sample": float(n_points.std()),
        }

    def save_results(self, output_dir: Path) -> None:
        """
        각 샘플별 2D projection 좌표를 CSV 로 저장하고,
        간단한 요약 통계를 CSV 로 저장합니다.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            n_points = s.proj_xy.shape[0]
            df = pd.DataFrame(
                {
                    "proj1": s.proj_xy[:, 0],
                    "proj2": s.proj_xy[:, 1],
                }
            )
            df.to_csv(sample_dir / "md_emulation_proj2d.csv", index=False)

            summary = {
                "sample": s.name,
                "n_points": int(n_points),
                "proj1_mean": float(np.mean(s.proj_xy[:, 0])),
                "proj2_mean": float(np.mean(s.proj_xy[:, 1])),
            }
            # with open(sample_dir / "md_emulation_summary.json", "w") as f:
            #     json.dump(summary, f, indent=2, sort_keys=True)

            summaries.append(summary)

        # if summaries:
        #     summary_df = pd.DataFrame(summaries)
        #     summary_df.to_csv(
        #         output_dir / "all_samples_md_emulation_summary.csv",
        #         index=False,
        #     )

    # results.py 내부, MDEmulationSelfResults 클래스의 plot 메서드만 교체

    def plot(self, output_dir: Path) -> None:
        """
        각 샘플마다:
        - proj1 vs proj2 2D free-energy surface
        - FNC basin(C1, C2, ...) 에 해당하는 영역을 원(Circle)으로 표시
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        import copy

        settings = DistributionMetricSettings()
        max_energy = settings.energy_cutoff
        levels = 20

        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            xy = s.proj_xy  # shape (n_frames, 2)
            if xy.shape[0] < 10:
                print(
                    f"[WARN] {s.name}: projection points < 10, "
                    "md_emulation free-energy plot 생략."
                )
                continue

            # 1) DistributionMetrics2D 계산
            metric = DistributionMetrics2D(
                reference_projections=xy,
                n_resample=settings.n_resample,
                sigma_resample=settings.sigma_resample,
                num_bins=settings.num_bins,
                energy_cutoff=settings.energy_cutoff,
                temperature_K=self.temperature_K,
                padding=settings.padding,
                random_seed=42,
            )

            P = metric.density_ref
            low_mask = metric.low_energy_mask
            edges_x = metric.edges_x
            edges_y = metric.edges_y

            # 2) free-energy grid
            kBT = K_BOLTZMANN * self.temperature_K
            F = -kBT * np.log(P + 1e-12)

            finite = np.isfinite(F) & (P > 0.0)
            if not np.any(finite):
                print(f"[WARN] {s.name}: 유효한 density grid 가 없어 plot 생략.")
                continue

            F = F - np.nanmin(F[finite])

            F_fin = F[finite]
            print(
                f"[DEBUG] MD_EMU_SELF_FE {s.name}: "
                f"min(F)={F_fin.min():.3f} kcal/mol, "
                f"max(F)={F_fin.max():.3f} kcal/mol, "
                f"mean(F)={F_fin.mean():.3f} kcal/mol"
            )

            F_for_plot = np.minimum(F, max_energy + 1.0)
            mask = (P <= 0.0) | (~low_mask)
            F_masked = np.ma.array(F_for_plot, mask=mask)

            # 3) 각 frame 의 free energy (기존 로직 유지)
            ix = np.digitize(xy[:, 0], edges_x) - 1
            iy = np.digitize(xy[:, 1], edges_y) - 1

            nx, ny = P.shape
            ix = np.clip(ix, 0, nx - 1)
            iy = np.clip(iy, 0, ny - 1)

            frame_F = np.full(xy.shape[0], np.nan, dtype=float)
            valid_bins = (P[ix, iy] > 0.0) & (low_mask[ix, iy])
            frame_F[valid_bins] = F[ix[valid_bins], iy[valid_bins]]

            df_proj = pd.DataFrame(
                {
                    "proj1": xy[:, 0],
                    "proj2": xy[:, 1],
                    "free_energy": frame_F,
                }
            )
            df_proj.to_csv(sample_dir / "md_emulation_proj2d.csv", index=False)

            # grid 데이터 저장 (기존 그대로)
            np.save(sample_dir / "md_emulation_free_energy_F_raw.npy", F)
            np.save(sample_dir / "md_emulation_free_energy_F_plot.npy", F_for_plot)
            np.save(sample_dir / "md_emulation_free_energy_P.npy", P)
            np.save(sample_dir / "md_emulation_free_energy_low_mask.npy", low_mask)
            np.save(sample_dir / "md_emulation_free_energy_xedges.npy", edges_x)
            np.save(sample_dir / "md_emulation_free_energy_yedges.npy", edges_y)

            # 4) contour 플롯
            xc = 0.5 * (edges_x[:-1] + edges_x[1:])
            yc = 0.5 * (edges_y[:-1] + edges_y[1:])
            Xc, Yc = np.meshgrid(xc, yc, indexing="ij")

            cmap = copy.copy(plt.cm.turbo)
            cmap.set_over(color="w")

            fig, ax = plt.subplots(figsize=(4, 4))
            cf = ax.contourf(
                Xc,
                Yc,
                F_masked,
                levels=levels,
                cmap=cmap,
                vmin=0.0,
                vmax=max_energy,
            )
            cf.set_clim(0.0, max_energy)

            cbar = fig.colorbar(cf, ax=ax, extend="max")
            cbar.set_label("free energy (kcal/mol)")

            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_title(f"MD emulation 2D free energy (self)")

            fig.savefig(
                sample_dir / "md_emulation_free_energy.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

# ----------------------------------------------------------------------
# DSSP SELF (reference 없이 sample 자체 DSSP 통계)
# ----------------------------------------------------------------------

@dataclass
class SingleSampleDSSP:
    """
    한 샘플(trajectory)에 대한 DSSP 정보.

    name        : 샘플 이름
    frame_idx   : 프레임 인덱스 (0, 1, 2, ...)
    dssp_codes  : shape (n_frames, n_residues), 각 원소는 1글자 DSSP 코드('H','E','C',...)
    helix_frac  : 각 프레임에서 헬릭스(H/G/I) 비율 (0~1)
    sheet_frac  : 각 프레임에서 시트(E/B) 비율 (0~1)
    coil_frac   : 각 프레임에서 나머지(코일/턴 등) 비율 (0~1)
    """
    name: str
    frame_idx: np.ndarray
    dssp_codes: np.ndarray
    helix_frac: np.ndarray
    sheet_frac: np.ndarray
    coil_frac: np.ndarray

@dataclass
class DSSPResults(BenchmarkResults):
    """
    여러 샘플에 대해 self DSSP 를 계산한 결과 모음.
    """
    samples: List[SingleSampleDSSP]

    def get_aggregate_metrics(self) -> Dict[str, float]:
        if not self.samples:
            return {}

        helix_means = np.array([s.helix_frac.mean() for s in self.samples], dtype=float)
        sheet_means = np.array([s.sheet_frac.mean() for s in self.samples], dtype=float)
        coil_means  = np.array([s.coil_frac.mean()  for s in self.samples], dtype=float)

        return {
            "num_samples": float(len(self.samples)),
            "helix_frac_mean_over_samples": float(helix_means.mean()),
            "helix_frac_std_over_samples": float(helix_means.std()),
            "sheet_frac_mean_over_samples": float(sheet_means.mean()),
            "sheet_frac_std_over_samples": float(sheet_means.std()),
            "coil_frac_mean_over_samples": float(coil_means.mean()),
            "coil_frac_std_over_samples": float(coil_means.std()),
        }

    def save_results(self, output_dir: Path) -> None:
        """
        샘플별 DSSP 코드와 헬릭스/시트/코일 비율을 저장.
        - dssp_codes.npy : (n_frames, n_residues) 문자열 배열
        - dssp_fractions.csv : frame, helix_frac, sheet_frac, coil_frac
        - dssp_summary.json : 간단한 요약 통계
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            # 1) 전체 DSSP 코드 저장 (numpy 배열)
            np.save(sample_dir / "dssp_codes.npy", s.dssp_codes)

            # 2) per-frame 비율 CSV
            df_frac = pd.DataFrame(
                {
                    "frame": s.frame_idx,
                    "helix_frac": s.helix_frac,
                    "sheet_frac": s.sheet_frac,
                    "coil_frac": s.coil_frac,
                }
            )
            df_frac.to_csv(sample_dir / "dssp_fractions.csv", index=False)

            # 3) 요약 통계 JSON
            summary = {
                "sample": s.name,
                "n_frames": int(len(s.frame_idx)),
                "helix_frac_mean": float(s.helix_frac.mean()),
                "helix_frac_std": float(s.helix_frac.std()),
                "sheet_frac_mean": float(s.sheet_frac.mean()),
                "sheet_frac_std": float(s.sheet_frac.std()),
                "coil_frac_mean": float(s.coil_frac.mean()),
                "coil_frac_std": float(s.coil_frac.std()),
            }
            # with open(sample_dir / "dssp_summary.json", "w") as f:
            #     json.dump(summary, f, indent=2, sort_keys=True)

            summaries.append(summary)

        # # 전체 샘플 요약 CSV
        # if summaries:
        #     summary_df = pd.DataFrame(summaries)
        #     summary_df.to_csv(output_dir / "all_samples_dssp_summary.csv", index=False)

    def plot(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        helix_codes = np.array(["H", "G", "I"], dtype="<U1")
        sheet_codes = np.array(["E", "B"], dtype="<U1")

        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            dssp_codes = s.dssp_codes
            if dssp_codes.ndim != 2:
                continue

            n_frames, n_residues = dssp_codes.shape
            if n_residues == 0:
                continue

            helix_mask = np.isin(dssp_codes, helix_codes)
            sheet_mask = np.isin(dssp_codes, sheet_codes)

            helix_prob_per_res = helix_mask.mean(axis=0)
            sheet_prob_per_res = sheet_mask.mean(axis=0)

            resid_idx = np.arange(n_residues)

            # 축 자체는 3x3 인치로 유지
            fig, ax = plt.subplots(figsize=(3, 3))

            ax.plot(resid_idx, helix_prob_per_res, linewidth=1, label="helix")
            ax.plot(resid_idx, sheet_prob_per_res, linewidth=1, label="sheet")

            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("residue index")
            ax.set_ylabel("fraction")
            ax.set_title(s.name)

            # 범례를 축 밖 오른쪽에 배치
            legend = ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )

            # 범례와 제목까지 포함해서 잘리지 않게 저장
            fig.savefig(
                sample_dir / "dssp_per_residue.png",
                dpi=200,
                bbox_inches="tight",
                bbox_extra_artists=(legend,),
            )
            plt.close(fig)