#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def plot_cindex(table: Path, out_prefix: Path) -> None:
    df = pd.read_csv(table)
    colors = {"RSF": "#4C78A8", "XGBoost": "#D62728"}
    x = np.arange(len(df))
    y = df["test_cindex"].to_numpy()
    yerr = np.vstack([y - df["ci_low"].to_numpy(), df["ci_high"].to_numpy() - y])

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.errorbar(x, y, yerr=yerr, fmt="o", color="#333333", ecolor="#555555", capsize=4, zorder=2)
    for xi, (_, row) in zip(x, df.iterrows()):
        ax.scatter(xi, row["test_cindex"], s=72, color=colors[row["model"]], edgecolor="white", zorder=3)
        ax.text(xi, row["test_cindex"] + 0.014, f"{row['test_cindex']:.3f}", ha="center", fontsize=8, weight="bold")
    ax.set_xticks(x, df["model"])
    ax.set_ylabel("Test-set C-index")
    ax.set_ylim(0.45, 1.06)
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_title("Held-out Test Set C-index", fontsize=10.5, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_auc(table: Path, out_prefix: Path) -> None:
    df = pd.read_csv(table)
    timepoints = ["1-year", "3-year", "5-year"]
    colors = {"RSF": "#4C78A8", "XGBoost": "#D62728"}
    markers = {"RSF": "o", "XGBoost": "s"}
    x = np.arange(len(timepoints))

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for model in ["RSF", "XGBoost"]:
        sub = df[df["model"] == model].set_index("timepoint").loc[timepoints]
        y = sub["auc"].to_numpy()
        lo = sub["ci95_low"].to_numpy()
        hi = sub["ci95_high"].to_numpy()
        ax.fill_between(x, lo, hi, color=colors[model], alpha=0.13, linewidth=0)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            fmt=markers[model] + "-",
            color=colors[model],
            ecolor=colors[model],
            capsize=3.5,
            markersize=5.5,
            linewidth=1.8,
            label=model,
        )
    ax.set_xticks(x, timepoints)
    ax.set_ylabel("Time-dependent AUC")
    ax.set_ylim(0.35, 1.05)
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.legend(loc="lower right")
    ax.set_title("Held-out Test Set Time-dependent AUC", fontsize=10.5, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot model performance confidence-interval figures.")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_cindex(data_dir / "figure2_cindex_bootstrap_ci_summary.csv", out_dir / "figure2_cindex_bootstrap_ci_replotted")
    plot_auc(data_dir / "figure2_time_dependent_auc_bootstrap_ci_summary.csv", out_dir / "figure2_time_dependent_auc_bootstrap_ci_replotted")


if __name__ == "__main__":
    main()
