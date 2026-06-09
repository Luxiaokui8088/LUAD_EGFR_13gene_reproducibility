"""
Script 02: Preprocess OncoSG-LUAD data
Loads clinical, mutation, and expression data from the OncoSG 2020 LUAD cohort,
annotates EGFR mutation status, and saves a unified per-patient CSV.

Expected raw data layout:
  data/oncoSG/luad_oncosg_2020/
    - data_mutations.txt
    - data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt
  data/oncoSG/luad_oncosg_2020_clinical_data.tsv

Output:
  intermediate_data/OncoSG_ALL_SAMPLES_for_integration.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

print("--- [02] Preprocessing OncoSG-LUAD data ---")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "oncoSG" / "luad_oncosg_2020"
CLINICAL_PATH = PROJECT_ROOT / "data" / "oncoSG" / "luad_oncosg_2020_clinical_data.tsv"
MUT_PATH = RAW_DIR / "data_mutations.txt"
EXPR_PATH = RAW_DIR / "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt"
OUTPUT_DIR = PROJECT_ROOT / "intermediate_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "OncoSG_ALL_SAMPLES_for_integration.csv"


def parse_os_status(x):
    s = str(x).strip().upper()
    if s.startswith("1") or "DECEASE" in s or "DEAD" in s:
        return 1
    return 0


try:
    df_clin = pd.read_csv(CLINICAL_PATH, sep="\t", comment="#", low_memory=False)
    df_mut = pd.read_csv(MUT_PATH, sep="\t", comment="#", low_memory=False)
    df_exp = pd.read_csv(EXPR_PATH, sep="\t", comment="#", low_memory=False)

    required_cols = {
        "patient_id": "Patient ID",
        "age": "Age",
        "gender": "Sex",
        "stage": "Stage",
        "smoking_status": "Smoking status",
        "smoking_pack_years": "Person Cigarette Smoking History Pack Year Value",
        "OS_time_month": "Overall survival months",
        "OS_status_raw": "Overall survival status",
    }

    missing = [v for v in required_cols.values() if v not in df_clin.columns]
    if missing:
        raise ValueError(f"OncoSG clinical table missing columns: {missing}")

    clin = pd.DataFrame(
        {
            "patient_id": df_clin[required_cols["patient_id"]].astype(str).str.strip(),
            "age": pd.to_numeric(df_clin[required_cols["age"]], errors="coerce"),
            "gender": df_clin[required_cols["gender"]],
            "stage": df_clin[required_cols["stage"]],
            "smoking_status": df_clin[required_cols["smoking_status"]],
            "smoking_pack_years": pd.to_numeric(
                df_clin[required_cols["smoking_pack_years"]], errors="coerce"
            ),
            # Convert months to days
            "OS_time": pd.to_numeric(
                df_clin[required_cols["OS_time_month"]], errors="coerce"
            )
            * 30.44,
            "OS_status": df_clin[required_cols["OS_status_raw"]]
            .apply(parse_os_status)
            .astype(int),
        }
    ).dropna(subset=["patient_id"]).drop_duplicates(subset=["patient_id"])

    clin["dataset"] = "OncoSG"

    egfr_mut_ids = (
        df_mut.loc[
            df_mut["Hugo_Symbol"].astype(str).str.upper() == "EGFR",
            "Tumor_Sample_Barcode",
        ]
        .astype(str)
        .str.slice(0, 4)
        .unique()
    )
    clin["EGFR_mutation_status"] = clin["patient_id"].isin(egfr_mut_ids).astype(int)

    df_exp = df_exp.set_index("Hugo_Symbol")
    if "Entrez_Gene_Id" in df_exp.columns:
        df_exp = df_exp.drop(columns=["Entrez_Gene_Id"])

    exp_t = df_exp.T
    exp_t.index = exp_t.index.astype(str).str.strip()
    exp_t = exp_t[~exp_t.index.duplicated(keep="first")]

    df_final = clin.set_index("patient_id").join(exp_t, how="inner")
    df_final.to_csv(OUTPUT_PATH)

    print(f"OncoSG preprocessing complete. Output: {OUTPUT_PATH}")
    print(f"  Shape: {df_final.shape}")
    print("  EGFR mutation distribution:")
    print(df_final["EGFR_mutation_status"].value_counts(dropna=False))

except Exception as e:
    print(f"ERROR: OncoSG preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
