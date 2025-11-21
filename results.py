# results.py
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
        세 개의 이미지를 별도로 저장.
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

    name      : 샘플 이름 (디렉토리 이름 등)
    frame_idx : 프레임 인덱스 (0, 1, 2, ...)
    fnc       : 각 프레임에서의 fraction of native contacts 값
    """
    name: str
    frame_idx: np.ndarray
    fnc: np.ndarray


@dataclass
class FNCResults(BenchmarkResults):
    """
    여러 샘플에 대해 self-reference FNC를 계산한 결과 모음.

    samples:
        각 샘플의 FNC 시계열
    coverage_thresholds:
        coverage 곡선에 사용한 threshold 배열 (shape: [n_thresholds])
    coverage_bootstrap:
        bootstrap 반복마다의 coverage 곡선
        (shape: [n_bootstrap, n_thresholds])
    krecall_bootstrap:
        샘플별 k-recall (mean, std) 딕셔너리
    """
    samples: List[SingleSampleFNC]
    coverage_thresholds: np.ndarray | None = None
    coverage_bootstrap: np.ndarray | None = None
    krecall_bootstrap: Dict[str, Tuple[float, float]] | None = None

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        전체 FNC 분포 및 coverage / k-recall 요약값을 반환.
        """
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
            cov_mean = self.coverage_bootstrap.mean(axis=0)  # [n_thresholds]
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
        를 그림으로 저장.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        from bioemu_benchmarks.eval.multiconf.plot import plot_smoothed_1d_free_energy

        # import 이 실제로 성공했는지 확인용
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