"""
Script 04: Integrate three cohorts, apply ComBat batch correction,
           and filter to EGFR-mutant samples.

Reads the three per-cohort CSVs produced by scripts 01-03, finds shared
gene columns, concatenates, applies ComBat correction, then saves:
  - Full integrated cohort (all mutation statuses)
  - EGFR-mutant subset (the primary training input)

Supports two ComBat backends (auto-detected):
  pip install combat           -> uses combat.pycombat
  pip install pycombat         -> uses pycombat.Combat

Output:
  intermediate_data/LUAD_ALL_SAMPLES_integrated_corrected.csv
  intermediate_data/LUAD_EGFR_mut_3_datasets_integrated_FINAL.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# --- ComBat backend detection ---
COMBAT_BACKEND = None
_pycombat_fn = None
_CombatClass = None

try:
    from combat.pycombat import pycombat as _pycombat_fn
    COMBAT_BACKEND = "combat"
except Exception:
    try:
        from pycombat import Combat as _CombatClass
        COMBAT_BACKEND = "pycombat"
    except Exception:
        pass

if COMBAT_BACKEND is None:
    raise ImportError(
        "No ComBat backend found. Install one of: 'pip install combat' or 'pip install pycombat'"
    )

print(f"--- [04] Integrating three cohorts (ComBat backend: {COMBAT_BACKEND}) ---")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_data"
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

TCGA_PATH = INTERMEDIATE_DIR / "TCGA_ALL_SAMPLES_for_integration.csv"
ONCOSG_PATH = INTERMEDIATE_DIR / "OncoSG_ALL_SAMPLES_for_integration.csv"
GSE31210_PATH = INTERMEDIATE_DIR / "GSE31210_ALL_SAMPLES_for_integration.csv"

ALL_OUTPUT_FILE = INTERMEDIATE_DIR / "LUAD_ALL_SAMPLES_integrated_corrected.csv"
EGFR_FINAL_FILE = INTERMEDIATE_DIR / "LUAD_EGFR_mut_3_datasets_integrated_FINAL.csv"

NON_GENE_COLS = [
    "age", "gender", "stage", "smoking_status", "smoking_history",
    "smoking_pack_years", "OS_time", "OS_status", "dataset",
    "EGFR_mutation_status", "OS", "OS.time",
]


def load_and_standardize(name, path):
    """Load a per-cohort CSV and normalize gene column names to UPPERCASE_EXP."""
    print(f"\nLoading and standardizing: {name}")
    if not path.exists():
        print(f"  File not found: {path}")
        return None

    try:
        df = pd.read_csv(path, index_col=0, low_memory=False)

        non_gene = list(NON_GENE_COLS)
        non_gene.extend([c for c in df.columns if str(c).endswith("_mut")])

        gene_cols = [c for c in df.columns if c not in non_gene]

        rename_map = {}
        for col in gene_cols:
            new_col = str(col).upper().replace("_EXP", "")
            new_col = "".join(filter(str.isalnum, new_col))
            new_col = f"{new_col}_EXP"
            rename_map[col] = new_col

        df = df.rename(columns=rename_map)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        print(f"  Standardized. Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"  ERROR processing {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


paths = {"TCGA": TCGA_PATH, "OncoSG": ONCOSG_PATH, "GSE31210": GSE31210_PATH}
all_dfs = {name: load_and_standardize(name, p) for name, p in paths.items()}
all_dfs = {k: v for k, v in all_dfs.items() if v is not None}

if len(all_dfs) < 2:
    raise RuntimeError("At least two cohorts must be loaded successfully.")

# Find shared gene columns
print("\nIdentifying shared gene features ...")
common_genes = sorted(
    list(
        set.intersection(*[set(df.filter(like="_EXP").columns) for df in all_dfs.values()])
    )
)
print(f"  Shared gene count: {len(common_genes)}")
if not common_genes:
    raise RuntimeError("No shared gene columns found across cohorts.")

# Merge
print("\nMerging cohorts ...")
merged_df = pd.concat(all_dfs.values(), sort=False)
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep="first")]
print(f"  Total samples after merge: {merged_df.shape[0]}")

# Impute missing values
clin_numeric_cols = ["age", "smoking_pack_years", "OS_time"]
existing_clin_numeric = [c for c in clin_numeric_cols if c in merged_df.columns]
if existing_clin_numeric:
    medians = merged_df[existing_clin_numeric].median()
    merged_df[existing_clin_numeric] = merged_df[existing_clin_numeric].fillna(medians)

if "stage" in merged_df.columns:
    merged_df["stage"] = merged_df["stage"].fillna("Unknown")
if "gender" in merged_df.columns:
    merged_df["gender"] = merged_df["gender"].fillna("Unknown")

mut_cols = [c for c in merged_df.columns if "_mut" in c]
if mut_cols:
    merged_df[mut_cols] = merged_df[mut_cols].fillna(0).astype(int)

exp_cols = [c for c in common_genes if c in merged_df.columns]
if exp_cols:
    merged_df[exp_cols] = merged_df[exp_cols].fillna(0)

if "EGFR_mutation_status" in merged_df.columns:
    merged_df["EGFR_mutation_status"] = (
        pd.to_numeric(merged_df["EGFR_mutation_status"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

print("  Data cleaning complete.")
print("  EGFR=1 count per cohort:")
print(merged_df.groupby("dataset")["EGFR_mutation_status"].sum())

# ComBat batch correction
print("\nRunning ComBat batch correction ...")
expression_df = merged_df[common_genes].copy()
batch_series = merged_df["dataset"].astype(str)

variance_by_batch = expression_df.groupby(batch_series).var()
zero_var_genes = variance_by_batch.columns[
    (variance_by_batch == 0).any() | variance_by_batch.isna().any()
]
if not zero_var_genes.empty:
    expression_df = expression_df.drop(columns=zero_var_genes)
    print(f"  Removed zero-variance / NaN-variance genes: {len(zero_var_genes)}")

n_batches = len(set(batch_series.tolist()))
if expression_df.shape[1] == 0 or n_batches <= 1:
    raise RuntimeError("ComBat conditions not met (insufficient genes or batches).")

print(f"  Running ComBat on {n_batches} batches, {expression_df.shape[1]} genes ...")
if COMBAT_BACKEND == "combat":
    corrected = _pycombat_fn(expression_df.T, batch_series.tolist())
    corrected_df = corrected.T
else:
    corrected_arr = _CombatClass().fit_transform(
        expression_df.values, batch_series.values
    )
    corrected_df = pd.DataFrame(
        corrected_arr, index=expression_df.index, columns=expression_df.columns
    )

print("  ComBat correction complete.")

# Save full integrated file
print("\nSaving full integrated cohort ...")
final_all = merged_df.drop(columns=common_genes).join(corrected_df, how="inner")
print("  EGFR=1 count per cohort (post-ComBat):")
print(final_all.groupby("dataset")["EGFR_mutation_status"].sum())
final_all.to_csv(ALL_OUTPUT_FILE)
print(f"  Saved: {ALL_OUTPUT_FILE}  (shape: {final_all.shape})")

# Filter to EGFR-mutant samples
print("\nFiltering to EGFR-mutant samples ...")
if "EGFR_mutation_status" not in final_all.columns:
    raise RuntimeError("EGFR_mutation_status column missing; cannot filter.")

final_egfr = final_all[final_all["EGFR_mutation_status"] == 1].copy()
final_egfr.to_csv(EGFR_FINAL_FILE)
print(f"  Saved: {EGFR_FINAL_FILE}  (shape: {final_egfr.shape})")
print("  Dataset distribution:")
print(final_egfr["dataset"].value_counts(dropna=False))
