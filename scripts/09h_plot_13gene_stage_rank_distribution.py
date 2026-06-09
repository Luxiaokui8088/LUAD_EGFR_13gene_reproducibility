"""
Plot where the locked 13 genes fall within the full-feature stage-association
ranking landscape.

Inputs
- data/full_feature_stage_association.csv

Outputs
- results/analysis/13gene_stage_rank_distribution/<run_id>/figures/13gene_stage_rank_distribution.png
- results/analysis/13gene_stage_rank_distribution/<run_id>/tables/13gene_stage_rank_distribution_summary.csv
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LOCKED_13_GENES_EXP = [
    "CDKAL1_EXP",
    "CIR1_EXP",
    "COQ8A_EXP",
    "USP15_EXP",
    "VPS50_EXP",
    "PRODH_EXP",
    "KRT17_EXP",
    "MAP2K3_EXP",
    "P4HA2_EXP",
    "MINDY1_EXP",
    "TMEM164_EXP",
    "TENT5C_EXP",
    "FCMR_EXP",
]


def resolve_paths() -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = project_root / "results" / "analysis" / "13gene_stage_rank_distribution" / run_id
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return {
        "script_path": script_path,
        "input_csv": project_root / "data" / "full_feature_stage_association.csv",
        "run_dir": run_dir,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
    }


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))
    paths = resolve_paths()
    df = pd.read_csv(paths["input_csv"])

    total_features = len(df)
    locked = df[df["feature"].isin(LOCKED_13_GENES_EXP)].copy()
    locked["rank_pct"] = locked["rank_abs_spearman_stage"] / total_features * 100.0
    locked["direction"] = np.where(locked["spearman_rho_stage"] >= 0, "Positive stage rho", "Negative stage rho")
    locked["fdr_sig"] = locked["spearman_fdr_stage"] < 0.05
    locked = locked.sort_values("rank_pct", ascending=False)

    summary_cols = [
        "feature",
        "rank_abs_spearman_stage",
        "rank_pct",
        "spearman_rho_stage",
        "spearman_p_stage",
        "spearman_fdr_stage",
        "kruskal_p_stage",
        "kruskal_fdr_stage",
        "mean_stage_I",
        "mean_stage_II",
        "mean_stage_III",
        "mean_stage_IV",
        "direction",
        "fdr_sig",
    ]
    locked[summary_cols].to_csv(
        paths["tables_dir"] / "13gene_stage_rank_distribution_summary.csv", index=False
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10.6, 6.8))

    y = np.arange(len(locked))
    colors = locked["direction"].map(
        {"Positive stage rho": "#d73027", "Negative stage rho": "#4575b4"}
    )

    ax.hlines(y=y, xmin=0, xmax=locked["rank_pct"], color="#bdbdbd", linewidth=1.2, zorder=1)
    ax.scatter(
        locked["rank_pct"],
        y,
        s=np.where(locked["fdr_sig"], 90, 55),
        c=colors,
        edgecolor=np.where(locked["fdr_sig"], "#222222", "white"),
        linewidth=0.9,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(locked["feature"], fontsize=9.5)
    ax.set_xlabel("Percentile position in stage-association ranking\n(left = stronger stage association)")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 10, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "10", "25", "50", "75", "100"])
    ax.invert_yaxis()
    ax.set_title("Stage-Association Ranking of the 13 Highlighted Genes", fontsize=13, fontweight="bold")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=False, bottom=False)

    med_pct = locked["rank_pct"].median()
    sig_count = int(locked["fdr_sig"].sum())
    ax.text(
        0.98,
        0.04,
        f"Median percentile: {med_pct:.1f}%\nStage-associated genes (FDR < 0.05): {sig_count}/13",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d73027", markeredgecolor="white", markersize=8, label="Positive stage association"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4575b4", markeredgecolor="white", markersize=8, label="Negative stage association"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#999999", markeredgecolor="#222222", markersize=9, label="FDR < 0.05"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=8.5,
        title="Marker key",
        title_fontsize=9,
        ncol=3,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.30)
    out_fig = paths["figures_dir"] / "13gene_stage_rank_distribution.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "script": str(paths["script_path"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(paths["input_csv"]),
        "n_total_features": int(total_features),
        "n_locked13_found": int(len(locked)),
        "outputs": {
            "figure": str(out_fig),
            "summary_csv": str(paths["tables_dir"] / "13gene_stage_rank_distribution_summary.csv"),
        },
    }
    with open(paths["run_dir"] / "13gene_stage_rank_distribution_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("13-gene stage-rank distribution plot completed.")
    print(f"Run directory: {paths['run_dir']}")


if __name__ == "__main__":
    main()
