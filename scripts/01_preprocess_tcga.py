"""
Script 01: Preprocess TCGA-LUAD data
Loads clinical, survival, mutation, and expression data from TCGA-LUAD,
annotates EGFR mutation status, and saves a unified per-patient CSV.

Expected raw data layout (place under data/TCGA_LUAD/):
  - TCGA.LUAD.sampleMap_LUAD_clinicalMatrix
  - survival_LUAD_survival.txt
  - mc3_LUAD_mc3.txt.gz
  - TCGA.LUAD.sampleMap_HiSeqV2.gz

Output:
  intermediate_data/TCGA_ALL_SAMPLES_for_integration.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

print("--- [01] Preprocessing TCGA-LUAD data ---")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "TCGA_LUAD"
OUTPUT_DIR = PROJECT_ROOT / "intermediate_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL_PATH = DATA_DIR / "TCGA.LUAD.sampleMap_LUAD_clinicalMatrix"
SURVIVAL_PATH = DATA_DIR / "survival_LUAD_survival.txt"
MUT_PATH = DATA_DIR / "mc3_LUAD_mc3.txt.gz"
EXPR_PATH = DATA_DIR / "TCGA.LUAD.sampleMap_HiSeqV2.gz"
OUTPUT_PATH = OUTPUT_DIR / "TCGA_ALL_SAMPLES_for_integration.csv"


def pick_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


try:
    df_clin_raw = pd.read_csv(CLINICAL_PATH, sep="\t", low_memory=False)
    df_surv_raw = pd.read_csv(SURVIVAL_PATH, sep="\t", low_memory=False)
    df_mut_raw = pd.read_csv(MUT_PATH, sep="\t", low_memory=False)
    df_exp_raw = pd.read_csv(EXPR_PATH, sep="\t", low_memory=False)

    patient_col = pick_first_existing(
        df_clin_raw.columns,
        ["_PATIENT", "bcr_patient_barcode", "patient_id", "sampleID"],
    )
    age_col = pick_first_existing(
        df_clin_raw.columns,
        ["age_at_initial_pathologic_diagnosis", "age_at_index.demographic", "age"],
    )
    gender_col = pick_first_existing(
        df_clin_raw.columns, ["gender", "gender.demographic"]
    )
    stage_col = pick_first_existing(
        df_clin_raw.columns,
        ["pathologic_stage", "ajcc_pathologic_stage.diagnoses", "stage"],
    )
    smoking_col = pick_first_existing(
        df_clin_raw.columns,
        ["number_pack_years_smoked", "pack_years_smoked.exposures", "smoking_pack_years"],
    )

    if patient_col is None:
        raise ValueError("TCGA clinical table: patient ID column not found.")

    df_clin = pd.DataFrame(
        {
            "patient_id": df_clin_raw[patient_col].astype(str).str.slice(0, 12),
            "age": pd.to_numeric(df_clin_raw[age_col], errors="coerce")
            if age_col
            else np.nan,
            "gender": df_clin_raw[gender_col] if gender_col else "Unknown",
            "stage": df_clin_raw[stage_col] if stage_col else "Unknown",
            "smoking_pack_years": pd.to_numeric(
                df_clin_raw[smoking_col], errors="coerce"
            )
            if smoking_col
            else np.nan,
        }
    )
    df_clin = df_clin.dropna(subset=["patient_id"]).drop_duplicates(subset=["patient_id"])

    surv_patient_col = pick_first_existing(
        df_surv_raw.columns, ["_PATIENT", "patient_id", "sample"]
    )
    if surv_patient_col is None:
        raise ValueError("TCGA survival table: patient ID column not found.")

    df_surv = pd.DataFrame(
        {
            "patient_id": df_surv_raw[surv_patient_col].astype(str).str.slice(0, 12),
            "OS_status": pd.to_numeric(df_surv_raw["OS"], errors="coerce"),
            "OS_time": pd.to_numeric(df_surv_raw["OS.time"], errors="coerce"),
        }
    )
    df_surv = df_surv.dropna(subset=["patient_id"]).drop_duplicates(subset=["patient_id"])

    df_meta = pd.merge(df_clin, df_surv, on="patient_id", how="inner")
    df_meta["dataset"] = "TCGA"

    required_mut_cols = {"sample", "gene", "effect"}
    if not required_mut_cols.issubset(df_mut_raw.columns):
        raise ValueError(
            f"TCGA mutation file missing required columns: {required_mut_cols}"
        )

    # EGFR non-intronic mutations
    egfr_mut_patients = (
        df_mut_raw.loc[
            (df_mut_raw["gene"].astype(str).str.upper() == "EGFR")
            & (df_mut_raw["effect"].astype(str) != "Intron"),
            "sample",
        ]
        .astype(str)
        .str.slice(0, 12)
        .dropna()
        .unique()
    )
    print(f"TCGA EGFR non-intron sample count: {len(egfr_mut_patients)}")
    df_meta["EGFR_mutation_status"] = df_meta["patient_id"].isin(egfr_mut_patients).astype(int)

    gene_col_exp = df_exp_raw.columns[0]
    df_exp = df_exp_raw.rename(columns={gene_col_exp: "gene_symbol"}).set_index(
        "gene_symbol"
    )
    df_exp.columns = pd.Index(df_exp.columns).astype(str).str.slice(0, 12)
    df_exp = df_exp.T.groupby(level=0).mean(numeric_only=True).T

    exp_t = df_exp.T
    exp_t.index.name = "patient_id"
    exp_t = exp_t[~exp_t.index.duplicated(keep="first")]

    df_final = df_meta.set_index("patient_id").join(exp_t, how="inner")
    df_final.to_csv(OUTPUT_PATH)

    print(f"TCGA preprocessing complete. Output: {OUTPUT_PATH}")
    print(f"  Shape: {df_final.shape}")
    print("  EGFR mutation distribution:")
    print(df_final["EGFR_mutation_status"].value_counts(dropna=False))

except Exception as e:
    print(f"ERROR: TCGA preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
