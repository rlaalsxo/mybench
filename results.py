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
    """
    name: str
    frame_idx: np.ndarray
    fnc: np.ndarray
    rmsd_nm:  np.ndarray

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
        - FNC 기반 1D free energy (–log p(FNC), 0~1 구간)
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

            # 2) FNC free energy (1D)
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

    name         : 샘플 이름
    frame_idx    : 프레임 인덱스 (0,1,2,...)
    fnc          : 각 프레임의 fraction of native contacts
    foldedness   : 각 프레임의 foldedness (sigmoid(FNC))
    dg_kcal_per_mol : 전체 trajectory 기준 추정 ΔG (kcal/mol)
    p_fold_mean  : foldedness 평균 (0~1)
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
        FNC → foldedness sigmoid 파라미터
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
        - FNC 기반 1D free energy (–log p(FNC), 0~1 구간)
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

            # 3) FNC free energy (1D) – BioEmu와 동일한 방식의 –log p(FNC) 스타일
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
