"""
Quantify how strongly the locked 13 genes associate with pathological stage
relative to the full restored feature space.

Inputs
- results/model/restored_original_expression_matrix_293_samples.csv
- results/model/restored_original_13gene_expression_matrix_293_samples.csv
- intermediate_data/LUAD_EGFR_mut_3_datasets_integrated_FINAL.csv

Outputs
- results/analysis/13gene_stage_association/<run_id>/tables/full_feature_stage_association.csv
- results/analysis/13gene_stage_association/<run_id>/tables/locked13_stage_association.csv
- results/analysis/13gene_stage_association/<run_id>/tables/locked13_stage_association_summary.csv
- results/analysis/13gene_stage_association/<run_id>/tables/stage_label_counts.csv
- results/analysis/13gene_stage_association/<run_id>/13gene_stage_association_manifest.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr


LOCKED_13_GENES_EXP = [
    "CDKAL1_EXP", "CIR1_EXP", "COQ8A_EXP", "USP15_EXP", "VPS50_EXP",
    "PRODH_EXP", "KRT17_EXP", "MAP2K3_EXP", "P4HA2_EXP", "MINDY1_EXP",
    "TMEM164_EXP", "TENT5C_EXP", "FCMR_EXP",
]


def resolve_paths() -> Dict[str, Path]:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    workspace_root = project_root.parent

    out_root = project_root / "results" / "analysis" / "13gene_stage_association"
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_root / run_id
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    return {
        "script_path": script_path,
        "project_root": project_root,
        "workspace_root": workspace_root,
        "expr_full": project_root / "results" / "model" / "restored_original_expression_matrix_293_samples.csv",
        "expr_13": project_root / "results" / "model" / "restored_original_13gene_expression_matrix_293_samples.csv",
        "stage_source": project_root / "intermediate_data" / "LUAD_EGFR_mut_3_datasets_integrated_FINAL.csv",
        "run_dir": run_dir,
        "tables_dir": tables_dir,
    }


def standardize_stage(raw: object) -> Optional[str]:
    if pd.isna(raw):
        return None
    text = str(raw).strip().upper()
    if not text or text in {"UNKNOWN", "[DISCREPANCY]", "NAN"}:
        return None
    text = text.replace("STAGE ", "")
    if text.startswith("I") and not text.startswith("IV"):
        if text in {"I", "IA", "IB"}:
            return "I"
        if text in {"II", "IIA", "IIB"}:
            return "II"
        if text in {"III", "IIIA", "IIIB", "IIIC"}:
            return "III"
    if text in {"IV", "IVA", "IVB"}:
        return "IV"
    return None


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    vals = pvals.astype(float).values
    n = len(vals)
    order = np.argsort(vals)
    ranked = vals[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        curr = min(prev, ranked[i] * n / rank)
        adj[i] = curr
        prev = curr
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0.0, 1.0)
    return pd.Series(out, index=pvals.index)


def stage_association_table(expr: pd.DataFrame, stage_ord: pd.Series, stage_group: pd.Series) -> pd.DataFrame:
    rows = []
    valid_index = stage_ord.index
    expr = expr.loc[valid_index]
    for gene in expr.columns:
        values = pd.to_numeric(expr[gene], errors="coerce")
        keep = values.notna() & stage_ord.notna()
        if keep.sum() < 3 or values[keep].nunique() <= 1:
            rho, rho_p = np.nan, np.nan
        else:
            rho, rho_p = spearmanr(values[keep], stage_ord[keep])

        groups = []
        for stage_label in ["I", "II", "III", "IV"]:
            idx = stage_group[stage_group == stage_label].index
            arr = values.loc[idx].dropna().values
            if len(arr) > 0:
                groups.append(arr)
        group_unique_total = sum(len(np.unique(g)) for g in groups)
        if len(groups) >= 2 and group_unique_total > len(groups):
            kw_stat, kw_p = kruskal(*groups)
        else:
            kw_stat, kw_p = np.nan, np.nan

        rows.append({
            "feature": gene,
            "spearman_rho_stage": float(rho) if pd.notna(rho) else np.nan,
            "spearman_p_stage": float(rho_p) if pd.notna(rho_p) else np.nan,
            "abs_spearman_rho_stage": float(abs(rho)) if pd.notna(rho) else np.nan,
            "kruskal_h_stage": float(kw_stat) if pd.notna(kw_stat) else np.nan,
            "kruskal_p_stage": float(kw_p) if pd.notna(kw_p) else np.nan,
            "mean_stage_I": float(values.loc[stage_group == "I"].mean()),
            "mean_stage_II": float(values.loc[stage_group == "II"].mean()),
            "mean_stage_III": float(values.loc[stage_group == "III"].mean()),
            "mean_stage_IV": float(values.loc[stage_group == "IV"].mean()),
        })
    df = pd.DataFrame(rows)
    df["spearman_fdr_stage"] = benjamini_hochberg(df["spearman_p_stage"].fillna(1.0))
    df["kruskal_fdr_stage"] = benjamini_hochberg(df["kruskal_p_stage"].fillna(1.0))
    df = df.sort_values(["abs_spearman_rho_stage", "kruskal_h_stage"], ascending=[False, False]).reset_index(drop=True)
    df["rank_abs_spearman_stage"] = np.arange(1, len(df) + 1)
    return df


def main() -> None:
    paths = resolve_paths()
    expr_full = pd.read_csv(paths["expr_full"], index_col=0, low_memory=False)
    expr_13 = pd.read_csv(paths["expr_13"], index_col=0, low_memory=False)
    stage_df = pd.read_csv(paths["stage_source"], usecols=["Unnamed: 0", "stage"], low_memory=False)
    stage_df = stage_df.rename(columns={"Unnamed: 0": "sample_id"}).drop_duplicates("sample_id").set_index("sample_id")
    common_ids = expr_full.index.intersection(stage_df.index)
    stage_df = stage_df.loc[common_ids].copy()
    stage_df["stage_collapsed"] = stage_df["stage"].map(standardize_stage)
    stage_df = stage_df[stage_df["stage_collapsed"].notna()].copy()
    stage_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
    stage_df["stage_ordinal"] = stage_df["stage_collapsed"].map(stage_map).astype(float)

    expr_full = expr_full.loc[stage_df.index]
    expr_13 = expr_13.loc[stage_df.index]

    full_assoc = stage_association_table(expr_full, stage_df["stage_ordinal"], stage_df["stage_collapsed"])
    locked13_assoc = full_assoc[full_assoc["feature"].isin(LOCKED_13_GENES_EXP)].copy()
    locked13_assoc = locked13_assoc.sort_values("rank_abs_spearman_stage").reset_index(drop=True)
    locked13_assoc["top_1pct"] = locked13_assoc["rank_abs_spearman_stage"] <= max(1, int(np.ceil(len(full_assoc) * 0.01)))
    locked13_assoc["top_5pct"] = locked13_assoc["rank_abs_spearman_stage"] <= max(1, int(np.ceil(len(full_assoc) * 0.05)))
    locked13_assoc["top_10pct"] = locked13_assoc["rank_abs_spearman_stage"] <= max(1, int(np.ceil(len(full_assoc) * 0.10)))

    summary = pd.DataFrame([{
        "n_samples_with_valid_stage": int(len(stage_df)),
        "n_samples_in_restored_expression": int(expr_full.shape[0]),
        "n_features_tested": int(len(full_assoc)),
        "n_locked13_present": int(len(locked13_assoc)),
        "locked13_median_rank_abs_spearman": float(locked13_assoc["rank_abs_spearman_stage"].median()),
        "locked13_mean_abs_spearman": float(locked13_assoc["abs_spearman_rho_stage"].mean()),
        "locked13_max_abs_spearman": float(locked13_assoc["abs_spearman_rho_stage"].max()),
        "locked13_top_1pct_count": int(locked13_assoc["top_1pct"].sum()),
        "locked13_top_5pct_count": int(locked13_assoc["top_5pct"].sum()),
        "locked13_top_10pct_count": int(locked13_assoc["top_10pct"].sum()),
    }])

    stage_counts = (
        stage_df["stage_collapsed"]
        .value_counts()
        .rename_axis("stage_collapsed")
        .reset_index(name="n_samples")
        .sort_values("stage_collapsed")
    )

    full_out = paths["tables_dir"] / "full_feature_stage_association.csv"
    locked_out = paths["tables_dir"] / "locked13_stage_association.csv"
    summary_out = paths["tables_dir"] / "locked13_stage_association_summary.csv"
    counts_out = paths["tables_dir"] / "stage_label_counts.csv"

    full_assoc.to_csv(full_out, index=False)
    locked13_assoc.to_csv(locked_out, index=False)
    summary.to_csv(summary_out, index=False)
    stage_counts.to_csv(counts_out, index=False)

    manifest = {
        "script": str(paths["script_path"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "expr_full": str(paths["expr_full"]),
            "expr_13": str(paths["expr_13"]),
            "stage_source": str(paths["stage_source"]),
        },
        "outputs": {
            "full_feature_stage_association": str(full_out),
            "locked13_stage_association": str(locked_out),
            "locked13_stage_association_summary": str(summary_out),
            "stage_label_counts": str(counts_out),
        },
    }
    with open(paths["run_dir"] / "13gene_stage_association_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("13-gene stage association analysis completed.")
    print(f"Run directory: {paths['run_dir']}")
    print(f"Locked13 summary: {summary_out}")


if __name__ == "__main__":
    main()
