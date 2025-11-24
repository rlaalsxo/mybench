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

        summaries = []
        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            n_frames = len(s.rmsd_nm)
            df = pd.DataFrame(
                {
                    "frame": np.arange(n_frames, dtype=int),
                    "rmsd_nm": s.rmsd_nm,
                    "rg_nm": s.rg_nm,
                }
            )
            df.to_csv(sample_dir / "metrics.csv", index=False)

            summary = {
                "sample": s.name,
                "n_frames": int(n_frames),
                "rmsd_mean_nm": float(np.mean(s.rmsd_nm)),
                "rmsd_std_nm": float(np.std(s.rmsd_nm)),
                "rmsd_min_nm": float(np.min(s.rmsd_nm)),
                "rmsd_max_nm": float(np.max(s.rmsd_nm)),
                "rg_mean_nm": float(np.mean(s.rg_nm)),
                "rg_std_nm": float(np.std(s.rg_nm)),
                "rg_min_nm": float(np.min(s.rg_nm)),
                "rg_max_nm": float(np.max(s.rg_nm)),
            }
            with open(sample_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

            summaries.append(summary)

        if summaries:
            summary_df = pd.DataFrame(summaries)
            summary_df.to_csv(output_dir / "all_samples_summary.csv", index=False)

    def plot(self, output_dir: Path) -> None:
        """
        각 샘플마다:
        - RMSD vs Frame  → rmsd_vs_frame.png
        - Rg vs Frame    → rg_vs_frame.png
        - RMSD histogram → rmsd_hist.png
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            n_frames = len(s.rmsd_nm)
            frames = np.arange(n_frames, dtype=int)

            # 1) RMSD vs Frame
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(frames, s.rmsd_nm)
            ax.set_xlabel("Frame")
            ax.set_ylabel("RMSD (nm)")
            ax.set_title(f"{s.name} - RMSD vs Frame")
            fig.savefig(sample_dir / "rmsd_vs_frame.png", dpi=200)
            plt.close(fig)

            # 2) Rg vs Frame
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(frames, s.rg_nm)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Radius of gyration (nm)")
            ax.set_title(f"{s.name} - Rg vs Frame")
            fig.savefig(sample_dir / "rg_vs_frame.png", dpi=200)
            plt.close(fig)

            # 3) RMSD histogram
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(s.rmsd_nm, bins=40)
            ax.set_xlabel("RMSD (nm)")
            ax.set_ylabel("Count")
            ax.set_title(f"{s.name} - RMSD distribution")
            fig.savefig(sample_dir / "rmsd_hist.png", dpi=200)
            plt.close(fig)


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
                }
            )
            df.to_csv(sample_dir / "fnc_timeseries.csv", index=False)

            summary = {
                "sample": s.name,
                "n_frames": int(n_frames),
                "fnc_mean": float(np.mean(s.fnc)),
                "fnc_std": float(np.std(s.fnc)),
                "fnc_min": float(np.min(s.fnc)),
                "fnc_max": float(np.max(s.fnc)),
            }
            with open(sample_dir / "fnc_summary.json", "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

            summaries.append(summary)

        if summaries:
            summary_df = pd.DataFrame(summaries)
            summary_df.to_csv(output_dir / "all_samples_fnc_summary.csv", index=False)

        # coverage / k-recall 요약 저장
        if (
            self.coverage_thresholds is not None
            and self.coverage_bootstrap is not None
            and self.coverage_bootstrap.size > 0
        ):
            cov_mean = self.coverage_bootstrap.mean(axis=0)
            cov_std = self.coverage_bootstrap.std(axis=0)
            df_cov = pd.DataFrame(
                {
                    "threshold": self.coverage_thresholds,
                    "coverage_mean": cov_mean,
                    "coverage_std": cov_std,
                }
            )
            df_cov.to_csv(output_dir / "fnc_coverage_bootstrap.csv", index=False)

        if self.krecall_bootstrap:
            df_k = pd.DataFrame(
                [
                    {
                        "sample": name,
                        "krecall_mean": float(m),
                        "krecall_std": float(s),
                    }
                    for name, (m, s) in self.krecall_bootstrap.items()
                ]
            )
            df_k.to_csv(output_dir / "fnc_krecall_bootstrap.csv", index=False)

    def plot(self, output_dir: Path) -> None:
        """
        각 샘플마다:
        - FNC vs Frame
        - FNC 기반 1D free energy (–log p(FNC), 0~1 구간, 임의 단위)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        from bioemu_benchmarks.eval.multiconf.plot import plot_smoothed_1d_free_energy

        print(
            "[DEBUG] using plot_smoothed_1d_free_energy from "
            "bioemu_benchmarks.eval.multiconf.plot:",
            plot_smoothed_1d_free_energy,
        )

        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            # 1) FNC vs Frame
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(s.frame_idx, s.fnc)
            ax.set_xlabel("Frame")
            ax.set_ylabel("fraction of native contacts")
            ax.set_title(f"{s.name} - FNC vs Frame")
            fig.savefig(sample_dir / "fnc_vs_frame.png", dpi=200)
            plt.close(fig)

            # 2) FNC free energy (1D, –log p, arb. units)
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            plot_smoothed_1d_free_energy(
                s.fnc,
                range=(0.0, 1.0),
                ax=ax2,
            )
            ax2.set_xlabel("fraction of native contacts")
            ax2.set_ylabel("free energy (arb. units)")
            ax2.set_title(f"{s.name} - FNC free energy")
            fig2.savefig(sample_dir / "fnc_free_energy.png", dpi=200)
            plt.close(fig2)


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
            df.to_csv(sample_dir / "folding_fe_timeseries.csv", index=False)

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

        if summaries:
            summary_df = pd.DataFrame(summaries)
            summary_df.to_csv(output_dir / "all_samples_folding_fe_summary.csv", index=False)

    def plot(self, output_dir: Path) -> None:
        """
        각 샘플마다:
        - FNC vs Frame
        - foldedness vs Frame
        - FNC 기반 1D free energy (–log p(FNC), 0~1 구간, 임의 단위)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        from bioemu_benchmarks.eval.multiconf.plot import plot_smoothed_1d_free_energy

        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            # 1) FNC vs Frame
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(s.frame_idx, s.fnc)
            ax.set_xlabel("Frame")
            ax.set_ylabel("fraction of native contacts")
            ax.set_title(f"{s.name} - FNC vs Frame")
            fig.savefig(sample_dir / "fnc_vs_frame.png", dpi=200)
            plt.close(fig)

            # 2) foldedness vs Frame
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(s.frame_idx, s.foldedness)
            ax2.set_xlabel("Frame")
            ax2.set_ylabel("foldedness (p_fold)")
            ax2.set_title(f"{s.name} - foldedness vs Frame")
            fig2.savefig(sample_dir / "foldedness_vs_frame.png", dpi=200)
            plt.close(fig2)

            # 3) FNC free energy (1D, –log p, arb. units)
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            plot_smoothed_1d_free_energy(
                s.fnc,
                range=(0.0, 1.0),
                ax=ax3,
            )
            ax3.set_xlabel("fraction of native contacts")
            ax3.set_ylabel("free energy (arb. units)")
            ax3.set_title(f"{s.name} - FNC free energy")
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
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            n_points = s.tica_xy.shape[0]
            df = pd.DataFrame(
                {
                    "tica1": s.tica_xy[:, 0],
                    "tica2": s.tica_xy[:, 1],
                }
            )
            df.to_csv(sample_dir / "tica_coords.csv", index=False)

            summary = {
                "sample": s.name,
                "n_points": int(n_points),
                "tica1_mean": float(np.mean(s.tica_xy[:, 0])),
                "tica2_mean": float(np.mean(s.tica_xy[:, 1])),
            }
            with open(sample_dir / "tica_summary.json", "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

            summaries.append(summary)

        if summaries:
            summary_df = pd.DataFrame(summaries)
            summary_df.to_csv(output_dir / "all_samples_tica_summary.csv", index=False)

    def plot(self, output_dir: Path) -> None:
        """
        각 샘플마다:
        - TICA1 vs TICA2 2D free-energy surface (히스토그램 기반 -k_B T ln p)
        를 그림으로 저장.

        논문 Fig. 3(a)에 나오는 free energy surfaces (in kcal/mol)에
        대응하는 형태로, 색상 단위는 kcal/mol 입니다.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            xy = s.tica_xy
            if xy.shape[0] < 10:
                # 너무 적으면 의미 있는 분포가 안 나옴
                print(f"[WARN] {s.name}: TICA points < 10, free-energy plot 생략.")
                continue

            x = xy[:, 0]
            y = xy[:, 1]

            # 2D 히스토그램 → 확률
            nbins = 50
            H, xedges, yedges = np.histogram2d(x, y, bins=nbins)
            H = H.astype(float)

            total_counts = H.sum()
            if total_counts <= 0:
                continue

            P = H / total_counts  # 정규화된 확률분포 p(x, y)

            # free-energy surface: F = -k_B T ln p + const (kcal/mol)
            F = -K_BOLTZMANN * self.temperature_K * np.log(P + 1e-12)

            # 최소값을 0으로 맞춤 (상수항 제거)
            finite_mask = np.isfinite(F)
            if np.any(finite_mask):
                F = F - np.nanmin(F[finite_mask])

            # bin center 좌표
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            Xc, Yc = np.meshgrid(xc, yc, indexing="ij")

            fig, ax = plt.subplots(figsize=(4, 4))
            # contourf 로 색채우기
            cf = ax.contourf(Xc, Yc, F, levels=20)
            cbar = fig.colorbar(cf, ax=ax)
            cbar.set_label("free energy (kcal/mol)")

            ax.set_xlabel("TICA 1")
            ax.set_ylabel("TICA 2")
            ax.set_title(f"{s.name} - TICA 2D free energy")

            fig.savefig(sample_dir / "tica_free_energy.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

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

        if summaries:
            summary_df = pd.DataFrame(summaries)
            summary_df.to_csv(
                output_dir / "all_samples_md_emulation_summary.csv",
                index=False,
            )

    def plot(self, output_dir: Path) -> None:
        """
        각 샘플마다:
        - proj1 vs proj2 2D free-energy surface (히스토그램 기반 F = -k_B T ln p)
        를 그림으로 저장.

        TICAResults.plot 과 동일한 방식이지만,
        투영 좌표가 contact-map 기반 PCA 라는 점만 다릅니다.
        색상 단위는 kcal/mol 입니다.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            xy = s.proj_xy
            if xy.shape[0] < 10:
                print(
                    f"[WARN] {s.name}: projection points < 10, "
                    "md_emulation free-energy plot 생략."
                )
                continue

            x = xy[:, 0]
            y = xy[:, 1]

            # 2D 히스토그램 → 확률
            nbins = 50
            H, xedges, yedges = np.histogram2d(x, y, bins=nbins)
            H = H.astype(float)

            total_counts = H.sum()
            if total_counts <= 0:
                continue

            P = H / total_counts  # p(x, y)

            # free-energy surface: F = -k_B T ln p + const (kcal/mol)
            F = -K_BOLTZMANN * self.temperature_K * np.log(P + 1e-12)

            # 최소값을 0으로 shift
            finite_mask = np.isfinite(F)
            if np.any(finite_mask):
                F = F - np.nanmin(F[finite_mask])

            # bin center 좌표
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            Xc, Yc = np.meshgrid(xc, yc, indexing="ij")

            fig, ax = plt.subplots(figsize=(4, 4))
            cf = ax.contourf(Xc, Yc, F, levels=20)
            cbar = fig.colorbar(cf, ax=ax)
            cbar.set_label("free energy (kcal/mol)")

            ax.set_xlabel("projection 1")
            ax.set_ylabel("projection 2")
            ax.set_title(f"{s.name} - MD emulation 2D free energy (self)")

            fig.savefig(
                sample_dir / "md_emulation_free_energy.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
