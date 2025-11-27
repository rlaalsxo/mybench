# mybench/results.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
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
    finite: np.ndarray,
    delta_F_cut: float = 1.0,
    max_depth_from_global: float = 3.0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    1D free energy F (min(F)=0 으로 shift 된 상태)에 대해
    로컬 최소 주변의 연속된 구간을 basin 으로 정의.

    반환
    ----
    basins       : 각 basin 이 포함하는 bin index 배열 리스트
    basin_bin_id : len(F) 배열, 각 bin 의 basin 번호 (없으면 -1)
                   basin_id 0 → C1, 1 → C2, ... 로 라벨링에 사용
    """
    F = np.asarray(F)
    n = F.size
    basins: List[np.ndarray] = []
    basin_bin_id = np.full(n, -1, dtype=int)

    if n < 3:
        return basins, basin_bin_id
    if not np.any(finite):
        return basins, basin_bin_id

    F_finite = F[finite]
    Fmin = np.nanmin(F_finite)

    # 로컬 최소 후보
    cand: List[int] = []
    for i in range(1, n - 1):
        if not (finite[i - 1] and finite[i] and finite[i + 1]):
            continue
        if ((F[i] < F[i - 1] and F[i] <= F[i + 1]) or
            (F[i] <= F[i - 1] and F[i] < F[i + 1])):
            # 너무 얕은 minima 는 제외 (글로벌 최소보다 너무 높으면)
            if F[i] <= Fmin + max_depth_from_global:
                cand.append(i)

    if not cand:
        return basins, basin_bin_id

    cand = np.array(cand, dtype=int)
    # 더 깊은(min F) minima 부터 basin 생성
    order = np.argsort(F[cand])

    next_basin_id = 0
    for j in order:
        i_min = cand[j]
        if basin_bin_id[i_min] >= 0:
            # 이미 다른 basin 에 포함
            continue

        thr = F[i_min] + delta_F_cut

        # 좌측 확장
        left = i_min
        while (left - 1 >= 0 and
               finite[left - 1] and
               F[left - 1] <= thr and
               basin_bin_id[left - 1] < 0):
            left -= 1

        # 우측 확장
        right = i_min
        while (right + 1 < n and
               finite[right + 1] and
               F[right + 1] <= thr and
               basin_bin_id[right + 1] < 0):
            right += 1

        idxs = np.arange(left, right + 1)
        basins.append(idxs)
        basin_bin_id[idxs] = next_basin_id
        next_basin_id += 1

    return basins, basin_bin_id

# ----------------------------------------------------------------------
# FOLDING FREE ENERGIES (self-reference 버전)
# ----------------------------------------------------------------------

@dataclass
class SingleSampleFoldingFE:
    """
    한 샘플(trajectory)에 대한 folding free energy 관련 정보.

    name            : 샘플 이름
    frame_idx       : 프레임 인덱스 (0,1,2,...)
    fnc             : 각 프레임의 fraction of native contacts
    foldedness      : 각 프레임의 foldedness (Heaviside 기반, 0 또는 1)
    dg_kcal_per_mol : 전체 trajectory 기준 추정 ΔG (kcal/mol)
    p_fold_mean     : foldedness 평균 (0~1)
    """
    name: str
    frame_idx: np.ndarray
    fnc: np.ndarray
    foldedness: np.ndarray
    dg_kcal_per_mol: float
    p_fold_mean: float


@dataclass
class FoldingFreeEnergyResults(BenchmarkResults):
    """
    FOLDING_FREE_ENERGIES 아이디어를 self-reference 로 적용한 결과 모음.

    samples:
        각 샘플별 FNC(t), foldedness(t), dG, p_fold_mean
    temperature_K:
        ΔG 계산에 사용한 온도 (K)
    p_fold_thr, steepness:
        FNC → foldedness 에 쓰인 threshold 및 (과거 sigmoid 버전과의)
        호환성을 위한 steepness 인자.
        현재 구현에서는 foldedness 가 Heaviside step 이므로,
        steepness 는 사용되지 않습니다.
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
        """
        샘플별 FNC / foldedness 시계열과 dG 요약을 저장.
        """
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
            with open(sample_dir / "folding_fe_summary.json", "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

            summaries.append(summary)

        # if summaries:
        #     summary_df = pd.DataFrame(summaries)
        #     summary_df.to_csv(output_dir / "all_samples_folding_fe_summary.csv", index=False)

    def plot(self, output_dir: Path) -> None:
        """
        각 샘플마다:
        - FNC 기반 1D free energy (–log p(FNC), 0~1 구간, 임의 단위)
        - basin(움푹 파인 구간) 탐지
        - 각 프레임별 basin_id 저장
        - 플롯 상에 basin 을 C1, C2 ... 로 동그라미 표시
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # basin 정의 하이퍼파라미터
        delta_F_cut = 1.0           # 로컬 최소에서 ΔF <= 이내를 같은 basin 으로
        max_depth_from_global = 3.0 # 글로벌 최소보다 이 값 이상 높으면 무시

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

            # 1-1) basin 탐지 (bin 기준)
            basins, basin_bin_id = _find_1d_basins(
                F,
                finite=finite,
                delta_F_cut=delta_F_cut,
                max_depth_from_global=max_depth_from_global,
            )

            # grid 정보 CSV (선택사항, 있으면 디버깅에 유용)
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

            # 3) 1D free energy 플롯 + C1/C2 동그라미
            fig3, ax3 = plt.subplots(figsize=(8, 4))

            # 기존 BioEmu 스타일 free energy 곡선
            plot_smoothed_1d_free_energy(
                s.fnc,
                range=(0.0, 1.0),
                ax=ax3,
            )

            # y축 범위 기준으로 동그라미 위치(세로)는 아래쪽 일정 높이로 통일
            ymin, ymax = ax3.get_ylim()
            y_circle = ymin + 0.15 * (ymax - ymin)

            colors = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

            for basin_idx, bins in enumerate(basins):
                if bins.size == 0:
                    continue

                # basin 내에서 free energy 가 최소인 bin → "움푹 파인 중심"
                local_min_bin = bins[np.argmin(F[bins])]
                x_center = centers[local_min_bin]

                color = colors[basin_idx % len(colors)]
                label = f"C{basin_idx + 1}"

                # 동그라미 마커
                ax3.scatter(
                    x_center,
                    y_circle,
                    s=80,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.5,
                    zorder=5,
                )
                # 라벨 텍스트
                ax3.text(
                    x_center,
                    y_circle,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    zorder=6,
                )

            ax3.set_xlabel("fraction of native contacts")
            ax3.set_ylabel("free energy (arb. units)")
            ax3.set_title(f"FNC free energy (C1, C2 ... basins)")
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
            with open(sample_dir / "md_emulation_summary.json", "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

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

            # 5) FNC basin(C1, C2, ...) → PCA 평면에 동그라미
            basin_csv = sample_dir / "folding_fnc_free_energy_1d_per_frame.csv"
            if basin_csv.exists():
                df_fnc = pd.read_csv(basin_csv)

                if (
                    df_fnc.shape[0] == xy.shape[0]
                    and "basin_id" in df_fnc.columns
                ):
                    basin_id = df_fnc["basin_id"].to_numpy(dtype=int)
                    colors = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

                    # 존재하는 basin 만 사용
                    unique_basins = np.unique(basin_id[basin_id >= 0])

                    for b in unique_basins:
                        mask_b = basin_id == b
                        # 유효 좌표가 너무 적으면 스킵
                        if np.count_nonzero(mask_b) < 3:
                            continue

                        pts = xy[mask_b]          # (nb, 2)
                        x_center = pts[:, 0].mean()
                        y_center = pts[:, 1].mean()

                        # 중심에서의 거리 분포 → 반지름 설정 (90 percentile)
                        dx = pts[:, 0] - x_center
                        dy = pts[:, 1] - y_center
                        r = np.sqrt(dx * dx + dy * dy)
                        if r.size == 0:
                            continue
                        radius = np.percentile(r, 90.0)
                        if radius <= 0.0:
                            continue

                        color = colors[int(b) % len(colors)]
                        label = f"C{int(b) + 1}"

                        # 동그라미
                        circle = Circle(
                            (x_center, y_center),
                            radius,
                            fill=False,
                            edgecolor=color,
                            linewidth=1.5,
                            alpha=0.9,
                        )
                        ax.add_patch(circle)

                        # 라벨 텍스트 (원 중심에)
                        ax.text(
                            x_center,
                            y_center,
                            label,
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=color,
                            zorder=6,
                        )

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
            with open(sample_dir / "dssp_summary.json", "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

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