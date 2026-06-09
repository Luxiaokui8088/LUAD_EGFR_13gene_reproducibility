"""
Script 08: External validation of 13-gene risk signature in GSE13213
Loads the GSE13213 microarray cohort, maps 13 risk genes to Agilent probes
(GPL6480), computes an arithmetic-mean risk score, and evaluates:
  1. Overall: log-rank, univariate Cox, and 1/3/5-year time-point AUC
  2. EGFR-mutant subset (n=45): same metrics
  3. Mutation association: Mann-Whitney U + Fisher exact for EGFR / K-ras / p53

Expected raw data layout (place under data/GSE13213/):
  - GSE13213_series_matrix.txt.gz
  - GSE13213_AD117_patient_info.txt
  - GPL6480.annot.gz   (download from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL6480)

Output (in results/analysis/GSE13213_validation/):
  GSE13213_clinical_cleaned.csv
  GSE13213_expression_matrix.csv.gz
  GSE13213_sample_map.csv
  GSE13213_13gene_survival_summary.csv
  GSE13213_13gene_timepoint_auc.csv
  GSE13213_13gene_survival_samples.csv
  GSE13213_egfr_subset_survival_summary.csv
  GSE13213_egfr_subset_timepoint_auc.csv
  GSE13213_13gene_mutation_association.csv
  GSE13213_13gene_mapping.csv
  -- Visualisation (PNG 300 dpi + TIFF 600 dpi) --
  GSE13213_fullcohort_KM.{png,tif}
  GSE13213_fullcohort_ROC_1y3y5y.{png,tif}
  GSE13213_egfr_subset_KM.{png,tif}
  GSE13213_egfr_subset_ROC_1y3y5y.{png,tif}
"""

import csv
import gzip
import itertools
import ssl
import urllib.request

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata, mannwhitneyu, fisher_exact, spearmanr, kruskal
from statsmodels.duration.survfunc import survdiff
from statsmodels.duration.hazard_regression import PHReg
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

print("--- [08] External validation of 13-gene signature in GSE13213 ---")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Resolve GSE13213 data directory robustly across different repository layouts
_DATA_DIR_CANDIDATES = [
    PROJECT_ROOT / "data" / "GSE13213",       # expected local layout
    PROJECT_ROOT.parent / "GSE13213",          # workspace-level layout
]
DATA_DIR = next((p for p in _DATA_DIR_CANDIDATES if p.exists()), _DATA_DIR_CANDIDATES[0])

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "GSE13213_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SERIES_MATRIX_FILE = DATA_DIR / "GSE13213_series_matrix.txt.gz"
PATIENT_INFO_FILE = DATA_DIR / "GSE13213_AD117_patient_info.txt"
ANNOT_PATH = DATA_DIR / "GPL6480.annot.gz"
ANNOT_URL = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6480/annot/GPL6480.annot.gz"

print(f"Data directory resolved to: {DATA_DIR}")

# 13-gene risk signature (CDKAL1, CIR1, COQ8A, USP15, VPS50, PRODH,
#                         KRT17, MAP2K3, P4HA2, MINDY1, TMEM164, TENT5C, FCMR)
GENE_SET = [
    "CDKAL1", "CIR1", "COQ8A", "USP15", "VPS50",
    "PRODH", "KRT17", "MAP2K3", "P4HA2", "MINDY1",
    "TMEM164", "TENT5C", "FCMR",
]
# Legacy gene aliases in GPL6480
ALIAS_MAP = {
    "COQ8A":  ["ADCK3"],
    "TENT5C": ["FAM46C"],
    "MINDY1": ["FAM63A"],
}

RNG = np.random.default_rng(20260420)
N_BOOTSTRAP = 1000


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------
def auc_binary(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s, method="average")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bootstrap_auc_ci(score_arr, event_arr, time_arr, t_day, B=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(1)
    pos = (event_arr == 1) & (time_arr <= t_day)
    neg = time_arr > t_day
    keep = pos | neg
    y = np.where(pos[keep], 1, 0)
    s = score_arr[keep]
    n = len(y)
    auc_obs = auc_binary(y, s)
    auc_adj_obs = max(auc_obs, 1.0 - auc_obs) if pd.notna(auc_obs) else np.nan
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n < 20:
        return auc_obs, np.nan, np.nan, auc_adj_obs, np.nan, np.nan, n, n_pos, n_neg
    auc_bs, auc_adj_bs = [], []
    idx = np.arange(n)
    for _ in range(B):
        bidx = rng.choice(idx, size=n, replace=True)
        auc_b = auc_binary(y[bidx], s[bidx])
        auc_bs.append(auc_b)
        auc_adj_bs.append(max(auc_b, 1.0 - auc_b) if pd.notna(auc_b) else np.nan)
    lo, hi = np.nanpercentile(auc_bs, [2.5, 97.5])
    lo_adj, hi_adj = np.nanpercentile(auc_adj_bs, [2.5, 97.5])
    return auc_obs, lo, hi, auc_adj_obs, lo_adj, hi_adj, n, n_pos, n_neg


def survival_stats(time_arr, event_arr, score_arr, B=1000, rng=None):
    """Return log-rank chi2/p, Cox HR (per 1 SD), and 1/3/5-year AUC."""
    if rng is None:
        rng = np.random.default_rng(1)
    # Log-rank: median split
    order = np.argsort(score_arr, kind="mergesort")
    group = np.zeros(len(score_arr), dtype=int)
    group[order[len(order) // 2:]] = 1
    if len(np.unique(group)) < 2:
        chi2_lr, p_lr = np.nan, np.nan
    else:
        chi2_lr, p_lr = survdiff(time_arr, event_arr, group)
    # Cox (per 1 SD)
    score_z = (score_arr - score_arr.mean()) / (score_arr.std(ddof=0) + 1e-9)
    cox_fit = PHReg(
        endog=time_arr, exog=score_z, status=event_arr, ties="breslow"
    ).fit(disp=False)
    coef = float(cox_fit.params[0])
    se = float(cox_fit.bse[0])
    hr = float(np.exp(coef))
    hr_lo = float(np.exp(coef - 1.96 * se))
    hr_hi = float(np.exp(coef + 1.96 * se))
    p_cox = float(cox_fit.pvalues[0])
    # AUC at 1/3/5 years
    auc_rows = []
    for t_day, t_label in [(365, "1y"), (3 * 365, "3y"), (5 * 365, "5y")]:
        auc, lo, hi, auc_adj, lo_adj, hi_adj, n_used, n_pos, n_neg = bootstrap_auc_ci(
            score_arr, event_arr, time_arr, t_day, B=B, rng=rng
        )
        auc_rows.append({
            "timepoint": t_label, "t_day": t_day,
            "n_used": n_used, "n_pos_event_by_t": n_pos, "n_neg_survive_beyond_t": n_neg,
            "auc": auc, "auc_ci95_low": lo, "auc_ci95_high": hi,
            "auc_oriented": auc_adj, "auc_oriented_ci95_low": lo_adj, "auc_oriented_ci95_high": hi_adj,
        })
    summary = {
        "n_total": len(time_arr), "n_events": int(event_arr.sum()),
        "logrank_chi2": float(chi2_lr) if pd.notna(chi2_lr) else np.nan,
        "logrank_p": float(p_lr) if pd.notna(p_lr) else np.nan,
        "cox_hr_per_1sd": hr, "cox_hr_ci95_low": hr_lo, "cox_hr_ci95_high": hr_hi,
        "cox_p": p_cox,
    }
    return summary, pd.DataFrame(auc_rows)


# -----------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------
# Colour palette (colour-blind friendly)
_COL_HIGH = "#d62728"   # high-risk group
_COL_LOW  = "#1f77b4"   # low-risk group
_ROC_COLS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 1y / 3y / 5y ROC


def _save_fig(fig: plt.Figure, out_prefix: Path) -> None:
    """Save figure as PNG (300 dpi) and TIFF (600 dpi)."""
    fig.savefig(str(out_prefix) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".tif", dpi=600, bbox_inches="tight", format="tiff")
    plt.close(fig)


def plot_km_curve(df: pd.DataFrame, title: str, out_prefix: Path) -> None:
    """
    Draw a Kaplan-Meier curve (median-split on risk score) with 95% CI,
    log-rank p-value annotation, and an at-risk table.

    Parameters
    ----------
    df : DataFrame with columns  time_days / event / score_raw
    title : Figure title string
    out_prefix : Output path without extension
    """
    df = df.dropna(subset=["time_days", "event", "score_raw"]).copy()
    df = df[df["time_days"] > 0].copy()

    median_score = df["score_raw"].median()
    df["risk_group"] = np.where(df["score_raw"] >= median_score, "High risk", "Low risk")

    high = df[df["risk_group"] == "High risk"]
    low  = df[df["risk_group"] == "Low risk"]

    lr = logrank_test(
        high["time_days"], low["time_days"],
        event_observed_A=high["event"].astype(int),
        event_observed_B=low["event"].astype(int),
    )
    p_val = lr.p_value

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()

    kmf_high.fit(high["time_days"], event_observed=high["event"].astype(int), label="High risk")
    kmf_low.fit(low["time_days"],   event_observed=low["event"].astype(int),  label="Low risk")

    kmf_high.plot_survival_function(
        ax=ax, ci_show=True, color=_COL_HIGH, linewidth=2.0,
        label=f"High risk (n={len(high)}, events={int(high['event'].sum())})",
    )
    kmf_low.plot_survival_function(
        ax=ax, ci_show=True, color=_COL_LOW, linewidth=2.0,
        label=f"Low risk  (n={len(low)}, events={int(low['event'].sum())})",
    )

    # p-value annotation (inside main panel, lower-left)
    p_text = f"Log-rank p = {p_val:.4f}" if p_val >= 0.0001 else "Log-rank p < 0.0001"
    ax.text(
        0.03, 0.08, p_text,
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#666666", alpha=0.9),
        zorder=5,
    )

    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Overall survival probability", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(
        fontsize=9,
        loc="upper right",
        title="Risk group",
        title_fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )
    ax.tick_params(labelsize=9)

    # At-risk table below the plot
    time_points = [0, 365, 730, 1095, 1460, 1825]
    try:
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(kmf_high, kmf_low, ax=ax, fontsize=8, rows_to_show=["At risk"])
    except Exception:
        # Fallback: manual at-risk annotation
        y_base = -0.22
        ax.text(-0.08, y_base + 0.06, "At risk:", transform=ax.transAxes,
                fontsize=8, ha="left", color="black")
        for i, (kmf, col) in enumerate([(kmf_high, _COL_HIGH), (kmf_low, _COL_LOW)]):
            counts = [int((kmf.durations >= t).sum()) for t in time_points]
            y_pos = y_base - i * 0.07
            x_positions = [t / df["time_days"].max() for t in time_points]
            for xp, cnt in zip(x_positions, counts):
                ax.text(xp, y_pos, str(cnt), transform=ax.transAxes,
                        fontsize=7, ha="center", color=col)

    fig.tight_layout()
    _save_fig(fig, out_prefix)
    print(f"  KM curve saved: {out_prefix}.png / .tif")


def _roc_at_timepoint(time_arr, event_arr, score_arr, t_day):
    """Compute FPR/TPR pairs for a time-point ROC curve (incident/dynamic)."""
    pos  = (event_arr == 1) & (time_arr <= t_day)
    neg  = time_arr > t_day
    keep = pos | neg
    y = np.where(pos[keep], 1, 0)
    s = score_arr[keep]

    order    = np.argsort(-s)
    y_sorted = y[order]
    s_sorted = s[order]
    thresholds = np.concatenate([[s_sorted[0] + 1], s_sorted, [s_sorted[-1] - 1]])

    tpr_list, fpr_list = [0.0], [0.0]
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    for thresh in thresholds[1:]:
        pred_pos = s >= thresh
        tp = int((pred_pos & (y == 1)).sum())
        fp = int((pred_pos & (y == 0)).sum())
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    return np.array(fpr_list), np.array(tpr_list)


def plot_roc_curves(
    df: pd.DataFrame,
    title: str,
    out_prefix: Path,
    rng: np.random.Generator,
    B: int = 1000,
) -> None:
    """
    Draw 1-year, 3-year, and 5-year time-dependent ROC curves on one axes,
    with AUC ± 95% CI bootstrap annotations.

    Parameters
    ----------
    df : DataFrame with columns  time_days / event / score_raw
    title : Figure title string
    out_prefix : Output path without extension
    rng : numpy random Generator (for reproducible bootstrap)
    B : number of bootstrap iterations
    """
    df = df.dropna(subset=["time_days", "event", "score_raw"]).copy()
    df = df[df["time_days"] > 0].copy()

    time_arr  = df["time_days"].values
    event_arr = df["event"].astype(int).values
    score_arr = df["score_raw"].values

    timepoints = [(365, "1-year"), (3 * 365, "3-year"), (5 * 365, "5-year")]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    for (t_day, t_label), color in zip(timepoints, _ROC_COLS):
        fpr, tpr = _roc_at_timepoint(time_arr, event_arr, score_arr, t_day)
        # AUC + CI from bootstrap helper
        auc, lo, hi, auc_adj, lo_adj, hi_adj, n_used, n_pos, n_neg = bootstrap_auc_ci(
            score_arr, event_arr, time_arr, t_day, B=B, rng=rng
        )
        if pd.isna(auc) or n_pos == 0:
            continue
        auc_disp = max(auc, 1.0 - auc)
        if pd.notna(lo_adj) and pd.notna(hi_adj):
            label = f"{t_label} AUC = {auc_disp:.3f} (95% CI {lo_adj:.3f}–{hi_adj:.3f})"
        else:
            label = f"{t_label} AUC = {auc_disp:.3f}"
        # Orient curve so it lies above diagonal
        if auc < 0.5:
            fpr, tpr = 1.0 - fpr, 1.0 - tpr
        ax.plot(fpr, tpr, color=color, linewidth=2.0, label=label)

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0, label="Reference")
    ax.set_xlabel("1 − Specificity (FPR)", fontsize=11)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.tick_params(labelsize=9)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    _save_fig(fig, out_prefix)
    print(f"  ROC curves saved: {out_prefix}.png / .tif")


# -----------------------------------------------------------------------
# Step 1: Load expression matrix and clinical data
# -----------------------------------------------------------------------
print("\nStep 1: Loading expression matrix and clinical data ...")

sample_titles = None
sample_geo_ids = None
with gzip.open(SERIES_MATRIX_FILE, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("!Sample_title"):
            sample_titles = next(csv.reader([line.rstrip("\n")], delimiter="\t", quotechar='"'))[1:]
        if line.startswith("!Sample_geo_accession"):
            sample_geo_ids = next(csv.reader([line.rstrip("\n")], delimiter="\t", quotechar='"'))[1:]
        if sample_titles is not None and sample_geo_ids is not None:
            break

if sample_titles is None or sample_geo_ids is None:
    raise ValueError("Failed to read sample titles or GEO accessions from series matrix.")

sample_map = pd.DataFrame({"geo_accession": sample_geo_ids, "sample_title": sample_titles})
sample_map["annotation"] = sample_map["sample_title"].str.extract(r"Patient\s+(\d+)")[0]
sample_map["annotation"] = sample_map["annotation"].apply(
    lambda x: f"AD{int(x):03d}" if pd.notna(x) else np.nan
)

expr_rows = []
in_table = False
with gzip.open(SERIES_MATRIX_FILE, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.rstrip("\n")
        if s.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        if s.startswith("!series_matrix_table_end"):
            break
        if in_table:
            expr_rows.append(next(csv.reader([s], delimiter="\t", quotechar='"')))

expr_df = pd.DataFrame(expr_rows[1:], columns=expr_rows[0])
expr_df.columns = [str(c).strip().strip('"') for c in expr_df.columns]
probe_col = "ID_REF" if "ID_REF" in expr_df.columns else expr_df.columns[0]
expr_df.set_index(probe_col, inplace=True)
expr_df.index = [i.replace('"', "").strip() for i in expr_df.index]
expr_df.columns = [c.replace('"', "").strip() for c in expr_df.columns]
expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
expr_df.dropna(how="all", inplace=True)

geo_to_anno = sample_map.dropna(subset=["annotation"]).set_index("geo_accession")["annotation"].to_dict()
expr_df = expr_df.rename(columns=geo_to_anno)
expr_df = expr_df.loc[:, ~expr_df.columns.duplicated()]

clinical_df = pd.read_csv(PATIENT_INFO_FILE, sep="\t")
clinical_df.columns = [c.strip() for c in clinical_df.columns]
clinical_df = clinical_df.rename(columns={"Stage (Pathological )": "Stage (Pathological)"})
clinical_df = clinical_df.apply(
    lambda col: col.astype(str).str.strip() if col.dtype == "object" else col
)
clinical_df["Annotation"] = clinical_df["Annotation"].astype(str).str.strip()
clinical_df = clinical_df.set_index("Annotation", drop=False)

common_ids = sorted(set(expr_df.columns).intersection(set(clinical_df.index)))
expr_df = expr_df[common_ids]
clinical_df = clinical_df.loc[common_ids].copy()
print(f"  Aligned samples: {len(common_ids)}")

# Save cleaned files
clinical_df.to_csv(OUTPUT_DIR / "GSE13213_clinical_cleaned.csv", index=False)
expr_df.to_csv(OUTPUT_DIR / "GSE13213_expression_matrix.csv.gz", index=True, compression="gzip")
sample_map.set_index("annotation").loc[
    [x for x in common_ids if x in sample_map["annotation"].values]
].reset_index().to_csv(OUTPUT_DIR / "GSE13213_sample_map.csv", index=False)
print("  Clinical, expression, and sample-map files saved.")

# -----------------------------------------------------------------------
# Step 2: Download GPL6480 annotation if not present
# -----------------------------------------------------------------------
if not ANNOT_PATH.exists():
    print(f"\nStep 2: Downloading GPL6480 annotation from NCBI ...")
    ssl_ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(ANNOT_URL, context=ssl_ctx) as resp, open(ANNOT_PATH, "wb") as out_f:
        out_f.write(resp.read())
    print("  Download complete.")
else:
    print("\nStep 2: GPL6480 annotation already present.")

# -----------------------------------------------------------------------
# Step 3: Map 13 genes to GPL6480 probes and compute risk score
# -----------------------------------------------------------------------
print("\nStep 3: Mapping 13 genes to GPL6480 probes ...")

ann_lines = []
capture = False
with gzip.open(ANNOT_PATH, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.rstrip("\n")
        if s.startswith("!platform_table_begin"):
            capture = True
            continue
        if s.startswith("!platform_table_end"):
            break
        if capture:
            ann_lines.append(next(csv.reader([s], delimiter="\t", quotechar='"')))

ann_df = pd.DataFrame(ann_lines[1:], columns=ann_lines[0])
ann_df = ann_df[["ID", "Gene symbol"]].rename(columns={"ID": "probe", "Gene symbol": "symbol"})
ann_df["symbol"] = ann_df["symbol"].fillna("").astype(str)
ann_df["symbol_tokens"] = ann_df["symbol"].str.split("///").apply(
    lambda xs: [x.strip() for x in xs if str(x).strip() != ""]
)

probe_gene_pairs = []
for _, row in ann_df.iterrows():
    toks = row["symbol_tokens"]
    if not toks:
        continue
    for gene in GENE_SET:
        if gene in toks:
            probe_gene_pairs.append((row["probe"], gene))
            continue
        for alias in ALIAS_MAP.get(gene, []):
            if alias in toks:
                probe_gene_pairs.append((row["probe"], gene))
                break

probe_map = pd.DataFrame(probe_gene_pairs, columns=["probe", "symbol_canonical"]).drop_duplicates()
mapped_genes = sorted(probe_map["symbol_canonical"].unique().tolist())
missing_genes = sorted(set(GENE_SET) - set(mapped_genes))
print(f"  Mapped genes: {mapped_genes}")
print(f"  Missing genes: {missing_genes}")

probe_map.to_csv(OUTPUT_DIR / "GSE13213_13gene_mapping.csv", index=False)

# Compute per-gene median expression, then arithmetic mean across genes
expr_probe = expr_df.copy()
expr_probe["probe"] = expr_probe.index
expr_probe = expr_probe.reset_index(drop=True)
merged_expr = expr_probe.merge(probe_map, on="probe", how="inner")
gene_expr = merged_expr.groupby("symbol_canonical")[expr_df.columns].median()

used_genes = [g for g in GENE_SET if g in gene_expr.index]
risk_score = gene_expr.loc[used_genes].mean(axis=0)  # arithmetic mean
print(f"  Risk score computed using {len(used_genes)} genes.")

# -----------------------------------------------------------------------
# Step 4: Full-cohort survival analysis
# -----------------------------------------------------------------------
print("\nStep 4: Full-cohort survival analysis (n=117) ...")

clin_use = clinical_df.copy().set_index("Annotation", drop=False)
surv_df = pd.DataFrame({"score_raw": risk_score})
surv_df["time_days"] = pd.to_numeric(
    clin_use.loc[surv_df.index, "Survival (days)"], errors="coerce"
)
surv_df["event"] = clin_use.loc[surv_df.index, "Status"].str.lower().map({"dead": 1, "alive": 0})
surv_df = surv_df.dropna(subset=["score_raw", "time_days", "event"]).copy()
surv_df = surv_df[surv_df["time_days"] > 0].copy()

surv_df["EGFR_status"] = clin_use.loc[surv_df.index, "EGFR status"]
surv_df["KRAS_status"] = clin_use.loc[surv_df.index, "K-ras Status"]
surv_df["P53_status"]  = clin_use.loc[surv_df.index, "p53 Status"]

summary_all, auc_all = survival_stats(
    surv_df["time_days"].values,
    surv_df["event"].astype(int).values,
    surv_df["score_raw"].values,
    B=N_BOOTSTRAP, rng=RNG,
)
print(f"  Full cohort — log-rank p={summary_all['logrank_p']:.4f}, "
      f"Cox HR={summary_all['cox_hr_per_1sd']:.3f}, p={summary_all['cox_p']:.4f}")

surv_df.to_csv(OUTPUT_DIR / "GSE13213_13gene_survival_samples.csv", index=True)
pd.DataFrame([summary_all]).to_csv(OUTPUT_DIR / "GSE13213_13gene_survival_summary.csv", index=False)
auc_all.to_csv(OUTPUT_DIR / "GSE13213_13gene_timepoint_auc.csv", index=False)

# -----------------------------------------------------------------------
# Step 4b: Full-cohort visualisation (KM curve + time-dependent ROC)
# -----------------------------------------------------------------------
print("\nStep 4b: Generating full-cohort KM and ROC plots ...")
plot_km_curve(
    surv_df,
    title=f"13-gene Risk Signature — Full Cohort (n={len(surv_df)})",
    out_prefix=OUTPUT_DIR / "GSE13213_fullcohort_KM",
)
plot_roc_curves(
    surv_df,
    title=f"Time-dependent ROC — Full Cohort (n={len(surv_df)})",
    out_prefix=OUTPUT_DIR / "GSE13213_fullcohort_ROC_1y3y5y",
    rng=RNG,
    B=N_BOOTSTRAP,
)

# -----------------------------------------------------------------------
# Step 5: EGFR-mutant subset survival analysis (n=45)
# -----------------------------------------------------------------------
print("\nStep 5: EGFR-mutant subset survival analysis ...")

egfr_mut_ids = surv_df.index[surv_df["EGFR_status"] == "Mut"].tolist()
surv_egfr = surv_df.loc[egfr_mut_ids].copy()
print(f"  EGFR-mutant samples available for analysis: {len(surv_egfr)}")

if len(surv_egfr) >= 10:
    summary_egfr, auc_egfr = survival_stats(
        surv_egfr["time_days"].values,
        surv_egfr["event"].astype(int).values,
        surv_egfr["score_raw"].values,
        B=N_BOOTSTRAP, rng=RNG,
    )
    print(f"  EGFR-mut — log-rank p={summary_egfr['logrank_p']:.4f}, "
          f"Cox HR={summary_egfr['cox_hr_per_1sd']:.3f}, p={summary_egfr['cox_p']:.4f}")
    pd.DataFrame([summary_egfr]).to_csv(
        OUTPUT_DIR / "GSE13213_egfr_subset_survival_summary.csv", index=False
    )
    auc_egfr.to_csv(OUTPUT_DIR / "GSE13213_egfr_subset_timepoint_auc.csv", index=False)

    # -------------------------------------------------------------------
    # Step 5b: EGFR-mutant subset visualisation (KM curve + ROC)
    # -------------------------------------------------------------------
    print("\nStep 5b: Generating EGFR-mutant subset KM and ROC plots ...")
    plot_km_curve(
        surv_egfr,
        title=f"13-gene Risk Signature — EGFR-mutant Subset (n={len(surv_egfr)})",
        out_prefix=OUTPUT_DIR / "GSE13213_egfr_subset_KM",
    )
    plot_roc_curves(
        surv_egfr,
        title=f"Time-dependent ROC — EGFR-mutant Subset (n={len(surv_egfr)})",
        out_prefix=OUTPUT_DIR / "GSE13213_egfr_subset_ROC_1y3y5y",
        rng=RNG,
        B=N_BOOTSTRAP,
    )
else:
    print("  Insufficient EGFR-mutant samples for survival analysis.")

# -----------------------------------------------------------------------
# Step 6: Mutation association analysis
# -----------------------------------------------------------------------
print("\nStep 6: Mutation association analysis ...")

mut_analysis = surv_df.dropna(subset=["EGFR_status", "KRAS_status", "P53_status"]).copy()
mut_analysis["EGFR_mut"] = mut_analysis["EGFR_status"].map({"Mut": 1, "Wt": 0})
mut_analysis["KRAS_mut"] = mut_analysis["KRAS_status"].map({"Mut": 1, "Wt": 0})
mut_analysis["P53_mut"]  = mut_analysis["P53_status"].map({"Mut": 1, "Wt": 0})
mut_analysis = mut_analysis.dropna(subset=["EGFR_mut", "KRAS_mut", "P53_mut"]).copy()

mutation_rows = []
for short_name, mut_col in [("EGFR", "EGFR_mut"), ("K-ras", "KRAS_mut"), ("p53", "P53_mut")]:
    d = mut_analysis.dropna(subset=[mut_col, "score_raw"]).copy()
    mut_scores = d.loc[d[mut_col] == 1, "score_raw"]
    wt_scores  = d.loc[d[mut_col] == 0, "score_raw"]

    if len(mut_scores) > 0 and len(wt_scores) > 0:
        mw_stat, mw_p = mannwhitneyu(mut_scores, wt_scores, alternative="two-sided")
    else:
        mw_stat, mw_p = np.nan, np.nan

    d["score_high"] = (d["score_raw"] >= d["score_raw"].median()).astype(int)
    ct = pd.crosstab(d["score_high"], d[mut_col])
    if ct.shape == (2, 2):
        odds_ratio, fisher_p = fisher_exact(ct.values)
    else:
        odds_ratio, fisher_p = np.nan, np.nan

    mutation_rows.append({
        "mutation_label": short_name,
        "n_total": int(len(d)),
        "n_mut": int((d[mut_col] == 1).sum()),
        "n_wt": int((d[mut_col] == 0).sum()),
        "score_median_mut": float(mut_scores.median()) if len(mut_scores) > 0 else np.nan,
        "score_median_wt": float(wt_scores.median()) if len(wt_scores) > 0 else np.nan,
        "mannwhitney_u": float(mw_stat) if pd.notna(mw_stat) else np.nan,
        "mannwhitney_p": float(mw_p) if pd.notna(mw_p) else np.nan,
        "high_score_mut_fisher_or": float(odds_ratio) if pd.notna(odds_ratio) else np.nan,
        "high_score_mut_fisher_p": float(fisher_p) if pd.notna(fisher_p) else np.nan,
    })

mutation_df = pd.DataFrame(mutation_rows)
mutation_df.to_csv(OUTPUT_DIR / "GSE13213_13gene_mutation_association.csv", index=False)
print(mutation_df.to_string(index=False))

print("\n--- [08] External validation complete ---")
print(f"All outputs saved to: {OUTPUT_DIR}")
