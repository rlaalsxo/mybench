# mybench/results.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BenchmarkResults(ABC):
    """
    bioemu-benchmarks 의 BenchmarkResults 와 비슷한 역할.

    - get_aggregate_metrics(): dict[str, float]
    - save_results(output_dir: Path)
    - plot(output_dir: Path)
    """

    @abstractmethod
    def get_aggregate_metrics(self) -> Dict[str, float]:
        ...

    @abstractmethod
    def save_results(self, output_dir: Path) -> None:
        ...

    @abstractmethod
    def plot(self, output_dir: Path) -> None:
        ...

@dataclass
class SingleSampleMetrics:
    name: str
    time_ns: np.ndarray
    rmsd_nm: np.ndarray
    rg_nm: np.ndarray

@dataclass
class BasicStatsResults(BenchmarkResults):
    """
    BASIC_STATS 벤치마크용 결과.

    - 각 샘플의 per-frame time / RMSD / Rg
    """

    samples: List[SingleSampleMetrics]

    # ---- 요약 메트릭 ----
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """전체 샘플에 대한 간단한 요약 통계를 반환."""
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

    # ---- 파일 저장 ----
    def save_results(self, output_dir: Path) -> None:
        """
        - 각 샘플별 metrics.csv + summary.json
        - 전체 all_samples_summary.csv
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries = []

        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(
                {
                    "frame": np.arange(len(s.time_ns), dtype=int),
                    "time_ns": s.time_ns,
                    "rmsd_nm": s.rmsd_nm,
                    "rg_nm": s.rg_nm,
                }
            )
            df.to_csv(sample_dir / "metrics.csv", index=False)

            summary = {
                "sample": s.name,
                "n_frames": int(len(s.time_ns)),
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

        # 전체 요약
        if summaries:
            summary_df = pd.DataFrame(summaries)
            summary_df.to_csv(output_dir / "all_samples_summary.csv", index=False)

    # ---- 시각화 ----
    def plot(self, output_dir: Path) -> None:
        """
        각 샘플마다:
        - RMSD vs time
        - Rg vs time
        - RMSD 히스토그램
        한 장의 PNG로 저장.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for s in self.samples:
            sample_dir = output_dir / s.name
            sample_dir.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)

            # RMSD vs time
            ax = axes[0]
            ax.plot(s.time_ns, s.rmsd_nm)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("RMSD (nm)")
            ax.set_title(f"{s.name} - RMSD vs Time")

            # Rg vs time
            ax = axes[1]
            ax.plot(s.time_ns, s.rg_nm)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Radius of gyration (nm)")
            ax.set_title(f"{s.name} - Rg vs Time")

            # RMSD histogram
            ax = axes[2]
            ax.hist(s.rmsd_nm, bins=40)
            ax.set_xlabel("RMSD (nm)")
            ax.set_ylabel("Count")
            ax.set_title(f"{s.name} - RMSD distribution")

            fig.savefig(sample_dir / "basic_plots.png", dpi=200)
            plt.close(fig)