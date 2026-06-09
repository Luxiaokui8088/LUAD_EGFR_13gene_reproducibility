"""
Generate revision-ready Kaplan-Meier plots with number-at-risk tables.

This script recreates the key manuscript KM panels in a style close to the
current figures, while adding a clean at-risk table underneath each panel.

Outputs
- results/analysis/km_with_risk_tables/<run_id>/figures/figure2_rsf_km_with_risk_table.{png,tif}
- results/analysis/km_with_risk_tables/<run_id>/figures/figure2_xgb_km_with_risk_table.{png,tif}
- results/analysis/km_with_risk_tables/<run_id>/figures/figure3e_13gene_km_with_risk_table.{png,tif}
- results/analysis/km_with_risk_tables/<run_id>/figures/gse13213_fullcohort_km_with_risk_table.{png,tif}
- results/analysis/km_with_risk_tables/<run_id>/figures/gse13213_egfr_subset_km_with_risk_table.{png,tif}
- results/analysis/km_with_risk_tables/<run_id>/tables/km_plot_manifest.csv
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test


COL_HIGH = "#d62728"
COL_LOW = "#1f77b4"

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
    workspace_root = project_root.parent

    out_root = project_root / "results" / "analysis" / "km_with_risk_tables"
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_root / run_id
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "workspace_root": workspace_root,
        "run_dir": run_dir,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
        "risk_scores": project_root / "results" / "analysis" / "all_samples_risk_scores.csv",
        "expr13": project_root / "results" / "model" / "restored_original_13gene_expression_matrix_293_samples.csv",
        "feature_survival": project_root / "results" / "model" / "restored_original_feature_matrix_with_survival.csv",
        "gse_survival": project_root / "results" / "analysis" / "GSE13213_validation" / "GSE13213_13gene_survival_samples.csv",
    }


def save_figure(fig: plt.Figure, out_prefix: Path) -> None:
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_prefix.with_suffix(".tif"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_p(p_val: float) -> str:
    if pd.isna(p_val):
        return "Log-rank p = NA"
    if p_val < 1e-4:
        return "Log-rank p < 0.0001"
    return f"Log-rank p = {p_val:.4f}"


def to_bool_event(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(int).astype(bool)


def build_testset_df(paths: dict[str, Path], score_col: str) -> pd.DataFrame:
    df = pd.read_csv(paths["risk_scores"])
    df = df[df["split"].astype(str).str.lower() == "test"].copy()
    df = df.rename(
        columns={
            "SampleID": "sample_id",
            "OS_time": "time",
            "OS_status": "event",
            score_col: "score",
        }
    )
    df["event"] = to_bool_event(df["event"])
    return df[["sample_id", "time", "event", "score"]].dropna()


def build_13gene_internal_df(paths: dict[str, Path]) -> pd.DataFrame:
    expr = pd.read_csv(paths["expr13"])
    surv = pd.read_csv(paths["feature_survival"], usecols=["sample_id", "status", "time"])
    df = expr.merge(surv, on="sample_id", how="inner")
    df["score"] = df[LOCKED_13_GENES_EXP].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    df = df.rename(columns={"status": "event"})
    df["event"] = to_bool_event(df["event"])
    return df[["sample_id", "time", "event", "score"]].dropna()


def build_gse_df(paths: dict[str, Path], egfr_only: bool = False) -> pd.DataFrame:
    df = pd.read_csv(paths["gse_survival"]).rename(columns={"Unnamed: 0": "sample_id"})
    if egfr_only:
        df = df[df["EGFR_status"].astype(str) == "Mut"].copy()
    df = df.rename(columns={"time_days": "time", "score_raw": "score"})
    df["event"] = to_bool_event(df["event"])
    return df[["sample_id", "time", "event", "score"]].dropna()


def plot_km_with_risk_table(
    df: pd.DataFrame,
    title: str,
    out_prefix: Path,
    x_label: str,
    time_ticks: list[float] | None = None,
) -> dict[str, float | int | str]:
    df = df[df["time"] > 0].copy()
    cutoff = float(df["score"].median())
    df["risk_group"] = np.where(df["score"] >= cutoff, "High risk", "Low risk")

    high = df[df["risk_group"] == "High risk"].copy()
    low = df[df["risk_group"] == "Low risk"].copy()

    lr = logrank_test(
        high["time"],
        low["time"],
        event_observed_A=high["event"].astype(int),
        event_observed_B=low["event"].astype(int),
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    kmf_high.fit(high["time"], event_observed=high["event"].astype(int), label="High risk")
    kmf_low.fit(low["time"], event_observed=low["event"].astype(int), label="Low risk")

    kmf_high.plot_survival_function(
        ax=ax,
        ci_show=True,
        color=COL_HIGH,
        linewidth=2.4,
        label=f"High risk (n={len(high)}, events={int(high['event'].sum())})",
    )
    kmf_low.plot_survival_function(
        ax=ax,
        ci_show=True,
        color=COL_LOW,
        linewidth=2.4,
        label=f"Low risk (n={len(low)}, events={int(low['event'].sum())})",
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel("Overall survival probability", fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=11)
    ax.legend(
        loc="upper right",
        fontsize=10.5,
        title="Risk group",
        title_fontsize=10.5,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
    )
    ax.text(
        0.03,
        0.08,
        format_p(float(lr.p_value)),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#777777", alpha=0.95),
        zorder=5,
    )

    if time_ticks is not None:
        ax.set_xticks(time_ticks)

    add_at_risk_counts(kmf_high, kmf_low, ax=ax, fontsize=10, rows_to_show=["At risk"])
    plt.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.28)
    save_figure(fig, out_prefix)

    return {
        "figure": out_prefix.name,
        "n_total": int(len(df)),
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "events_high": int(high["event"].sum()),
        "events_low": int(low["event"].sum()),
        "median_cutoff": cutoff,
        "logrank_p": float(lr.p_value),
    }


def main() -> None:
    paths = resolve_paths()

    manifest_rows = []

    manifest_rows.append(
        plot_km_with_risk_table(
            build_testset_df(paths, "RSF_RiskScore"),
            title="RSF Kaplan-Meier Curve",
            out_prefix=paths["figures_dir"] / "figure2_rsf_km_with_risk_table",
            x_label="Time (days)",
        )
    )

    manifest_rows.append(
        plot_km_with_risk_table(
            build_testset_df(paths, "XGBoost_RiskScore"),
            title="XGBoost Kaplan-Meier Curve",
            out_prefix=paths["figures_dir"] / "figure2_xgb_km_with_risk_table",
            x_label="Time (days)",
        )
    )

    manifest_rows.append(
        plot_km_with_risk_table(
            build_13gene_internal_df(paths),
            title="13-gene Module Score Kaplan-Meier Curve",
            out_prefix=paths["figures_dir"] / "figure3e_13gene_km_with_risk_table",
            x_label="Time (days)",
        )
    )

    gse_ticks = [0, 365, 730, 1095, 1460, 1825]
    manifest_rows.append(
        plot_km_with_risk_table(
            build_gse_df(paths, egfr_only=False),
            title="13-gene Risk Signature - GSE13213 Full Cohort",
            out_prefix=paths["figures_dir"] / "gse13213_fullcohort_km_with_risk_table",
            x_label="Time (days)",
            time_ticks=gse_ticks,
        )
    )

    manifest_rows.append(
        plot_km_with_risk_table(
            build_gse_df(paths, egfr_only=True),
            title="13-gene Risk Signature - GSE13213 EGFR-mutant Subset",
            out_prefix=paths["figures_dir"] / "gse13213_egfr_subset_km_with_risk_table",
            x_label="Time (days)",
            time_ticks=gse_ticks,
        )
    )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(paths["tables_dir"] / "km_plot_manifest.csv", index=False)

    metadata = {
        "run_dir": str(paths["run_dir"]),
        "figures_dir": str(paths["figures_dir"]),
        "tables_dir": str(paths["tables_dir"]),
        "n_figures": int(len(manifest)),
    }
    with open(paths["run_dir"] / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved KM figures with risk tables to: {paths['figures_dir']}")
    print(f"Manifest table: {paths['tables_dir'] / 'km_plot_manifest.csv'}")


if __name__ == "__main__":
    main()
