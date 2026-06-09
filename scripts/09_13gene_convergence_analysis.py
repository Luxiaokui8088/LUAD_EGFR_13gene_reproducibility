"""
Script 09: Quantitative convergence analysis for the locked 13-gene signature.

Purpose
- Quantify why the final 13-gene panel is a compact and stable module,
  without using full-model performance metrics.

Inputs (workspace-level)
- intermediate_data/LUAD_EGFR_mut_3_datasets_integrated_FINAL.csv
- results/model/gene_level_xgb_features_811.csv
- data/GSE13213/GSE13213_13gene_mapping.csv

Outputs (project-level)
- results/analysis/13gene_convergence/<run_id>/
  - tables/
  - figures/
  - logs/
  - 13gene_convergence_manifest.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 20260608
BOOTSTRAP_N = 500
RANDOM_BASELINE_N = 500
KSCAN_RANDOM_N = 300
KSCAN_MIN = 8
KSCAN_MAX = 20
MIN_LOCKED_GENES_PRESENT = 3

FALLBACK_13_GENES = [
    "CDKAL1", "CIR1", "COQ8A", "FCMR", "KRT17", "MAP2K3", "MINDY1",
    "P4HA2", "PRODH", "TENT5C", "TMEM164", "USP15", "VPS50",
]


def normalize_gene_name(value: str) -> str:
    return str(value).strip().upper().replace("_EXP", "")


def resolve_paths() -> Dict[str, Path]:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    workspace_root = project_root.parent

    in_integrated = project_root / "results" / "model" / "restored_original_feature_matrix_293_samples.csv"
    in_ranked = project_root / "results" / "model" / "gene_level_xgb_features_811.csv"
    in_mapping = project_root / "data" / "GSE13213" / "GSE13213_13gene_mapping.csv"

    out_root = project_root / "results" / "analysis" / "13gene_convergence"
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_root / run_id
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    logs_dir = run_dir / "logs"

    for d in [tables_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "script_path": script_path,
        "project_root": project_root,
        "workspace_root": workspace_root,
        "in_integrated": in_integrated,
        "in_ranked": in_ranked,
        "in_mapping": in_mapping,
        "out_root": out_root,
        "run_dir": run_dir,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "logs_dir": logs_dir,
    }


def load_13_genes(mapping_path: Path) -> Tuple[List[str], str]:
    if mapping_path.exists():
        df_map = pd.read_csv(mapping_path)
        symbol_col = None
        for col in ["symbol_canonical", "gene", "symbol", "Gene"]:
            if col in df_map.columns:
                symbol_col = col
                break
        if symbol_col is not None:
            genes = sorted({normalize_gene_name(x) for x in df_map[symbol_col].dropna().tolist() if str(x).strip()})
            if len(genes) >= 10:
                return genes, "mapping_file"
    return FALLBACK_13_GENES.copy(), "fallback_list"


def load_ranked_genes(ranked_path: Path) -> List[str]:
    df_rank = pd.read_csv(ranked_path)
    if "Feature" in df_rank.columns:
        raw = df_rank["Feature"].astype(str).tolist()
    else:
        raw = df_rank.iloc[:, 0].astype(str).tolist()
    genes = []
    for item in raw:
        g = item.replace("_EXP", "").strip().upper()
        if g and g not in genes:
            genes.append(g)
    return genes


def extract_expression_matrix(integrated_path: Path) -> pd.DataFrame:
    df = pd.read_csv(integrated_path, low_memory=False)
    expr_cols = [c for c in df.columns if c.endswith("_EXP")]
    if not expr_cols:
        raise ValueError("No *_EXP expression columns found in integrated matrix.")
    expr = df[expr_cols].copy()
    expr.columns = [c.replace("_EXP", "").upper() for c in expr.columns]
    expr = expr.loc[:, ~expr.columns.duplicated()].copy()
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=0, how="all")
    expr = expr.dropna(axis=1, how="all")
    return expr


def pca_first_variance(df_gene: pd.DataFrame) -> float:
    x = df_gene.values
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    if x.shape[1] <= 1:
        return np.nan
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=1, random_state=RANDOM_SEED)
    pca.fit(x_scaled)
    return float(pca.explained_variance_ratio_[0])


def cohesion_metrics(df_gene: pd.DataFrame) -> Dict[str, float]:
    corr = df_gene.corr(method="spearman")
    corr_values = corr.values
    tri = np.triu_indices_from(corr_values, k=1)
    upper = corr_values[tri]
    upper = upper[~np.isnan(upper)]
    abs_upper = np.abs(upper)
    return {
        "n_genes": int(df_gene.shape[1]),
        "mean_abs_spearman": float(np.mean(abs_upper)) if abs_upper.size else np.nan,
        "median_abs_spearman": float(np.median(abs_upper)) if abs_upper.size else np.nan,
        "min_abs_spearman": float(np.min(abs_upper)) if abs_upper.size else np.nan,
        "max_abs_spearman": float(np.max(abs_upper)) if abs_upper.size else np.nan,
        "pca_var_explained_pc1": pca_first_variance(df_gene),
    }


def module_score_table(df_gene: pd.DataFrame) -> pd.DataFrame:
    z = df_gene.apply(lambda col: zscore(col, nan_policy="omit"), axis=0)
    z = z.fillna(0.0)
    score = z.mean(axis=1)
    return pd.DataFrame({
        "sample_index": df_gene.index,
        "module_score_zmean": score.values,
    })


def bootstrap_vs_random(
    expr: pd.DataFrame,
    locked_genes: List[str],
    gene_universe: List[str],
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    rows = []
    n = expr.shape[0]
    locked_arr = np.array(locked_genes)
    universe_arr = np.array(gene_universe)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_df = expr.iloc[idx]

        locked_df = boot_df[locked_arr]
        locked_m = cohesion_metrics(locked_df)

        rand_genes = rng.choice(universe_arr, size=len(locked_genes), replace=False)
        rand_df = boot_df[rand_genes]
        rand_m = cohesion_metrics(rand_df)

        rows.append({
            "iter": i,
            "group": "locked_13",
            "mean_abs_spearman": locked_m["mean_abs_spearman"],
            "pca_var_explained_pc1": locked_m["pca_var_explained_pc1"],
        })
        rows.append({
            "iter": i,
            "group": "random_13",
            "mean_abs_spearman": rand_m["mean_abs_spearman"],
            "pca_var_explained_pc1": rand_m["pca_var_explained_pc1"],
        })
    return pd.DataFrame(rows)


def random_baseline_distribution(
    expr: pd.DataFrame,
    gene_universe: List[str],
    set_size: int,
    rng: np.random.Generator,
    n_iter: int,
) -> pd.DataFrame:
    arr = np.array(gene_universe)
    rows = []
    for i in range(n_iter):
        g = rng.choice(arr, size=set_size, replace=False)
        m = cohesion_metrics(expr[list(g)])
        rows.append({
            "iter": i,
            "mean_abs_spearman": m["mean_abs_spearman"],
            "pca_var_explained_pc1": m["pca_var_explained_pc1"],
        })
    return pd.DataFrame(rows)


def k_scan(
    expr: pd.DataFrame,
    ranked_genes: List[str],
    gene_universe: List[str],
    rng: np.random.Generator,
    k_min: int,
    k_max: int,
    n_rand: int,
) -> pd.DataFrame:
    rows = []
    universe_arr = np.array(gene_universe)

    filtered_ranked = [g for g in ranked_genes if g in gene_universe]
    max_k = min(k_max, len(filtered_ranked), len(gene_universe))
    min_k = max(k_min, 2)

    for k in range(min_k, max_k + 1):
        topk = filtered_ranked[:k]
        obs = cohesion_metrics(expr[topk])

        rand_vals_corr = []
        rand_vals_pca = []
        for _ in range(n_rand):
            g = rng.choice(universe_arr, size=k, replace=False)
            m = cohesion_metrics(expr[list(g)])
            rand_vals_corr.append(m["mean_abs_spearman"])
            rand_vals_pca.append(m["pca_var_explained_pc1"])

        rows.append({
            "k": k,
            "obs_mean_abs_spearman": obs["mean_abs_spearman"],
            "obs_pca_var_explained_pc1": obs["pca_var_explained_pc1"],
            "rand_mean_abs_spearman_mean": float(np.nanmean(rand_vals_corr)),
            "rand_mean_abs_spearman_p95": float(np.nanpercentile(rand_vals_corr, 95)),
            "rand_pca_var_explained_pc1_mean": float(np.nanmean(rand_vals_pca)),
            "rand_pca_var_explained_pc1_p95": float(np.nanpercentile(rand_vals_pca, 95)),
        })

    return pd.DataFrame(rows)


def plot_corr_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("13-gene Spearman Correlation Heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_vs_random(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    locked = df[df["group"] == "locked_13"]["mean_abs_spearman"].values
    rand = df[df["group"] == "random_13"]["mean_abs_spearman"].values
    axes[0].hist(rand, bins=30, alpha=0.6, label="random_13", color="#1f77b4")
    axes[0].hist(locked, bins=30, alpha=0.6, label="locked_13", color="#d62728")
    axes[0].set_title("Bootstrap Cohesion")
    axes[0].set_xlabel("Mean |Spearman|")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    locked_p = df[df["group"] == "locked_13"]["pca_var_explained_pc1"].values
    rand_p = df[df["group"] == "random_13"]["pca_var_explained_pc1"].values
    axes[1].hist(rand_p, bins=30, alpha=0.6, label="random_13", color="#1f77b4")
    axes[1].hist(locked_p, bins=30, alpha=0.6, label="locked_13", color="#d62728")
    axes[1].set_title("Bootstrap PC1 Variance")
    axes[1].set_xlabel("PC1 explained variance")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kscan(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(df["k"], df["obs_mean_abs_spearman"], marker="o", color="#d62728", label="observed top-k")
    axes[0].plot(df["k"], df["rand_mean_abs_spearman_mean"], marker="o", color="#1f77b4", label="random mean")
    axes[0].plot(df["k"], df["rand_mean_abs_spearman_p95"], linestyle="--", color="#1f77b4", label="random p95")
    axes[0].axvline(13, color="gray", linestyle=":")
    axes[0].set_title("k-scan: Cohesion")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Mean |Spearman|")
    axes[0].legend(fontsize=8)

    axes[1].plot(df["k"], df["obs_pca_var_explained_pc1"], marker="o", color="#d62728", label="observed top-k")
    axes[1].plot(df["k"], df["rand_pca_var_explained_pc1_mean"], marker="o", color="#1f77b4", label="random mean")
    axes[1].plot(df["k"], df["rand_pca_var_explained_pc1_p95"], linestyle="--", color="#1f77b4", label="random p95")
    axes[1].axvline(13, color="gray", linestyle=":")
    axes[1].set_title("k-scan: PC1 Variance")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("PC1 explained variance")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    paths = resolve_paths()
    rng = np.random.default_rng(RANDOM_SEED)

    if not paths["in_integrated"].exists():
        raise FileNotFoundError(f"Integrated matrix not found: {paths['in_integrated']}")
    if not paths["in_ranked"].exists():
        raise FileNotFoundError(f"Ranked feature file not found: {paths['in_ranked']}")

    expr = extract_expression_matrix(paths["in_integrated"])
    gene_universe = expr.columns.tolist()

    genes_13_raw, genes_source = load_13_genes(paths["in_mapping"])
    genes_13_present = [g for g in genes_13_raw if g in gene_universe]
    genes_13_missing = sorted(list(set(genes_13_raw) - set(genes_13_present)))

    if len(genes_13_present) < MIN_LOCKED_GENES_PRESENT:
        raise RuntimeError(
            f"Too few locked genes are present in the expression matrix "
            f"({len(genes_13_present)}/{len(genes_13_raw)} found; minimum {MIN_LOCKED_GENES_PRESENT})."
        )

    ranked_genes = load_ranked_genes(paths["in_ranked"])

    # Locked 13-gene metrics
    expr_13 = expr[genes_13_present].copy()
    m13 = cohesion_metrics(expr_13)
    corr13 = expr_13.corr(method="spearman")

    cohesion_13_path = paths["tables_dir"] / "13gene_convergence_cohesion_locked13.csv"
    pd.DataFrame([m13]).to_csv(cohesion_13_path, index=False)

    module_score_path = paths["tables_dir"] / "13gene_convergence_module_score_samples.csv"
    module_score_table(expr_13).to_csv(module_score_path, index=False)

    corr_path = paths["tables_dir"] / "13gene_convergence_locked13_spearman_matrix.csv"
    corr13.to_csv(corr_path)

    # Random baseline for locked size
    random_baseline_df = random_baseline_distribution(
        expr=expr,
        gene_universe=gene_universe,
        set_size=len(genes_13_present),
        rng=rng,
        n_iter=RANDOM_BASELINE_N,
    )
    rand_path = paths["tables_dir"] / "13gene_convergence_random_baseline_lockedsize.csv"
    random_baseline_df.to_csv(rand_path, index=False)

    # Bootstrap stability
    boot_df = bootstrap_vs_random(
        expr=expr,
        locked_genes=genes_13_present,
        gene_universe=gene_universe,
        rng=rng,
        n_boot=BOOTSTRAP_N,
    )
    boot_path = paths["tables_dir"] / "13gene_convergence_bootstrap_vs_random.csv"
    boot_df.to_csv(boot_path, index=False)

    # k-scan around 13
    kscan_df = k_scan(
        expr=expr,
        ranked_genes=ranked_genes,
        gene_universe=gene_universe,
        rng=rng,
        k_min=KSCAN_MIN,
        k_max=KSCAN_MAX,
        n_rand=KSCAN_RANDOM_N,
    )
    kscan_path = paths["tables_dir"] / "13gene_convergence_kscan.csv"
    kscan_df.to_csv(kscan_path, index=False)

    # Figures
    fig_corr = paths["figures_dir"] / "13gene_convergence_corr_heatmap_locked13.png"
    plot_corr_heatmap(corr13, fig_corr)

    fig_boot = paths["figures_dir"] / "13gene_convergence_bootstrap_vs_random.png"
    plot_bootstrap_vs_random(boot_df, fig_boot)

    fig_kscan = paths["figures_dir"] / "13gene_convergence_kscan.png"
    plot_kscan(kscan_df, fig_kscan)

    # Summary table
    summary = {
        "n_samples": int(expr.shape[0]),
        "n_genes_universe": int(expr.shape[1]),
        "locked_genes_source": genes_source,
        "n_locked_genes_requested": int(len(genes_13_raw)),
        "n_locked_genes_present": int(len(genes_13_present)),
        "n_locked_genes_missing": int(len(genes_13_missing)),
        "locked_gene_minimum_required": int(MIN_LOCKED_GENES_PRESENT),
        "locked13_mean_abs_spearman": m13["mean_abs_spearman"],
        "locked13_pca_var_explained_pc1": m13["pca_var_explained_pc1"],
        "bootstrap_n": BOOTSTRAP_N,
        "random_baseline_n": RANDOM_BASELINE_N,
        "kscan_random_n": KSCAN_RANDOM_N,
    }
    summary_path = paths["tables_dir"] / "13gene_convergence_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    # Log
    log_path = paths["logs_dir"] / "13gene_convergence_run.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("13-gene convergence analysis completed.\n")
        f.write(f"Run directory: {paths['run_dir']}\n")
        f.write(f"Samples: {expr.shape[0]} | Gene universe: {expr.shape[1]}\n")
        f.write(
            f"Locked genes present: {len(genes_13_present)}/{len(genes_13_raw)} "
            f"(minimum required: {MIN_LOCKED_GENES_PRESENT})\n"
        )
        if genes_13_missing:
            f.write("Missing locked genes: " + ", ".join(genes_13_missing) + "\n")

    outputs = [
        str(cohesion_13_path),
        str(module_score_path),
        str(corr_path),
        str(rand_path),
        str(boot_path),
        str(kscan_path),
        str(summary_path),
        str(fig_corr),
        str(fig_boot),
        str(fig_kscan),
        str(log_path),
    ]

    manifest = {
        "script": str(paths["script_path"]),
        "run_dir": str(paths["run_dir"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "integrated_matrix": str(paths["in_integrated"]),
            "ranked_features": str(paths["in_ranked"]),
            "mapping_file": str(paths["in_mapping"]),
        },
        "parameters": {
            "random_seed": RANDOM_SEED,
            "bootstrap_n": BOOTSTRAP_N,
            "random_baseline_n": RANDOM_BASELINE_N,
            "kscan_random_n": KSCAN_RANDOM_N,
            "kscan_min": KSCAN_MIN,
            "kscan_max": KSCAN_MAX,
        },
        "locked_genes": {
            "source": genes_source,
            "requested": genes_13_raw,
            "present": genes_13_present,
            "missing": genes_13_missing,
        },
        "outputs": outputs,
    }

    manifest_path = paths["run_dir"] / "13gene_convergence_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("13-gene convergence analysis completed.")
    print(f"Run directory: {paths['run_dir']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
