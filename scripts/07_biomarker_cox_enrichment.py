"""
Script 07: Biomarker-level univariate Cox regression and functional enrichment
For each model (RSF and XGBoost), runs:
  1. Batch univariate Cox regression for all prioritised genes
  2. Pathway enrichment analysis (GO BP / KEGG / Reactome) via Enrichr
  3. Transcription factor enrichment (TRRUST) via Enrichr

Input:
  results/model/FINAL_RSF_BIOMARKERS_PERMUTATION.csv
  results/model/gene_level_xgb_features_811.csv
  intermediate_data/LUAD_EGFR_mut_3_datasets_integrated_FINAL.csv

Output (in results/analysis/<Model>_analysis/):
  prognosis_summary.csv
  enrichment_pathways.csv / enrichment_pathways.png
  enrichment_tfs.csv / enrichment_tfs.png
"""

import pandas as pd
import numpy as np
from pathlib import Path

import gseapy as gp
from lifelines import CoxPHFitter
from tqdm.auto import tqdm
from gseapy.plot import dotplot

print("--- [07] Biomarker Cox regression and functional enrichment ---")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_RESULTS_DIR = PROJECT_ROOT / "results" / "model"
EGFR_SAMPLES_FILE = (
    PROJECT_ROOT / "intermediate_data" / "LUAD_EGFR_mut_3_datasets_integrated_FINAL.csv"
)
OUTPUT_BASE_DIR = PROJECT_ROOT / "results" / "analysis"
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILES = {
    "RSF":     MODEL_RESULTS_DIR / "FINAL_RSF_BIOMARKERS_PERMUTATION.csv",
    "XGBoost": MODEL_RESULTS_DIR / "gene_level_xgb_features_811.csv",
}

PATHWAY_DBS = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"]
TF_DBS = ["TRRUST_Transcription_Factors_2019"]

# Load EGFR-mutant cohort (shared across models)
print("\nLoading EGFR-mutant cohort ...")
try:
    df_egfr = pd.read_csv(EGFR_SAMPLES_FILE, index_col=0)
except Exception as e:
    print(f"ERROR loading main data file: {e}")
    exit(1)

# -----------------------------------------------------------------------
# Main analysis loop (one pass per model)
# -----------------------------------------------------------------------
for model_name, feature_path in FEATURE_FILES.items():
    print("\n" + "#" * 60)
    print(f"Analysing model: {model_name}")

    # Step A: Load gene list
    print(f"\n  [A] Loading biomarker list for {model_name} ...")
    try:
        df_features = pd.read_csv(feature_path)
        feature_col = "Feature" if "Feature" in df_features.columns else df_features.columns[0]
        gene_list_with_suffix = df_features[feature_col].tolist()
        gene_list = [g.replace("_EXP", "") for g in gene_list_with_suffix if "_EXP" in g]
        if not gene_list:
            print(f"  WARNING: No genes found in feature file for {model_name}. Skipping.")
            continue
        print(f"  Loaded {len(gene_list)} genes.")
    except Exception as e:
        print(f"  ERROR loading {feature_path}: {e}")
        continue

    model_output_dir = OUTPUT_BASE_DIR / f"{model_name}_analysis"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Step B: Batch univariate Cox regression
    print(f"\n  [B] Running univariate Cox regression ({len(gene_list)} genes) ...")
    try:
        df_lifelines = df_egfr[
            ["OS_time", "OS_status"] + [f"{g}_EXP" for g in gene_list if f"{g}_EXP" in df_egfr.columns]
        ].copy()
        df_lifelines.rename(columns={"OS_time": "time", "OS_status": "status"}, inplace=True)

        results_list = []
        cph = CoxPHFitter()
        for gene in tqdm(gene_list, desc=f"Cox ({model_name})"):
            col = f"{gene}_EXP"
            if col not in df_lifelines.columns:
                continue
            cph.fit(
                df_lifelines[[col, "time", "status"]],
                duration_col="time",
                event_col="status",
            )
            results_list.append(
                {
                    "Gene": gene,
                    "Hazard_Ratio": cph.hazard_ratios_[0],
                    "P_Value": cph.summary.p.values[0],
                }
            )

        results_df = pd.DataFrame(results_list).sort_values("P_Value")
        results_df.to_csv(model_output_dir / "prognosis_summary.csv", index=False)
        print(f"  Cox regression complete. Top 10 results:")
        print(results_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"  ERROR in Cox regression: {e}")

    # Step C: Pathway enrichment
    print(f"\n  [C] Running pathway enrichment ...")
    try:
        enr_pathway = gp.enrichr(
            gene_list=gene_list,
            gene_sets=PATHWAY_DBS,
            organism="Human",
            outdir=None,
        )
        if not enr_pathway.results.empty:
            enr_pathway.results.to_csv(model_output_dir / "enrichment_pathways.csv")
            dotplot(
                enr_pathway.res2d,
                title=f"Enriched Pathways ({model_name})",
                ofname=str(model_output_dir / "enrichment_pathways.png"),
                top_term=20,
            )
            print("  Pathway enrichment complete. Results and figure saved.")
        else:
            print("  No significant pathway enrichment results.")
    except Exception as e:
        print(f"  ERROR in pathway enrichment: {e}")

    # Step D: Transcription factor enrichment
    print(f"\n  [D] Running TF enrichment ...")
    try:
        enr_tf = gp.enrichr(
            gene_list=gene_list,
            gene_sets=TF_DBS,
            organism="Human",
            outdir=None,
        )
        if not enr_tf.results.empty:
            enr_tf.results.to_csv(model_output_dir / "enrichment_tfs.csv")
            dotplot(
                enr_tf.res2d,
                title=f"Enriched TFs ({model_name})",
                ofname=str(model_output_dir / "enrichment_tfs.png"),
                top_term=20,
            )
            print("  TF enrichment complete. Results and figure saved.")
        else:
            print("  No significant TF enrichment results.")
    except Exception as e:
        print(f"  ERROR in TF enrichment: {e}")

print("\n--- [07] Biomarker analysis complete ---")
