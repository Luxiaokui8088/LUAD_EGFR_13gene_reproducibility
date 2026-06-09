"""
Script 06: Decode RSF feature importance via permutation importance
Rebuilds the exact 8:1:1 test set used during training and computes
permutation-based feature importance for the trained RSF model.
Also replays the XGBoost C-index for cross-verification.

Input:
  results/model/gene_level_rsf_model_811.joblib
  results/model/gene_level_xgb_model_811.joblib  (optional, for replay)
  intermediate_data/LUAD_EGFR_mut_3_datasets_integrated_FINAL.csv
  h.all.v2025.1.Hs.symbols.gmt

Output:
  results/model/FINAL_RSF_BIOMARKERS_PERMUTATION.csv
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sksurv.metrics import concordance_index_censored

print("--- [06] RSF permutation importance ---")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "results" / "model"
EGFR_SAMPLES_FILE = (
    PROJECT_ROOT / "intermediate_data" / "LUAD_EGFR_mut_3_datasets_integrated_FINAL.csv"
)
GMT_PATH = PROJECT_ROOT / "h.all.v2025.1.Hs.symbols.gmt"
OUTPUT_FEATURES_PATH = MODEL_DIR / "FINAL_RSF_BIOMARKERS_PERMUTATION.csv"
RSF_MODEL_PATH = MODEL_DIR / "gene_level_rsf_model_811.joblib"
XGB_MODEL_PATH = MODEL_DIR / "gene_level_xgb_model_811.joblib"

TOP_PATHWAYS = [
    "HALLMARK_HEME_METABOLISM",
    "HALLMARK_KRAS_SIGNALING_DN",
    "HALLMARK_ESTROGEN_RESPONSE_EARLY",
    "HALLMARK_P53_PATHWAY",
]

# -----------------------------------------------------------------------
# Rebuild the exact test set (must match script 05 exactly)
# -----------------------------------------------------------------------
print("Rebuilding test set with the same 8:1:1 split as training ...")

df_egfr = pd.read_csv(EGFR_SAMPLES_FILE, index_col=0, low_memory=False)

genes_in_top_pathways: set = set()
with open(GMT_PATH, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in TOP_PATHWAYS:
            genes_in_top_pathways.update({f"{g.upper()}_EXP" for g in parts[2:]})

genes_in_top_pathways_sorted = sorted(genes_in_top_pathways)

clin_cols = ["age", "gender", "stage", "dataset", "smoking_pack_years"]
existing_clin_cols = [c for c in clin_cols if c in df_egfr.columns]

df_clin_encoded = pd.get_dummies(
    df_egfr[existing_clin_cols],
    columns=["gender", "stage", "dataset"],
    drop_first=True,
    dtype=float,
)
existing_gene_cols = [g for g in genes_in_top_pathways_sorted if g in df_egfr.columns]
X = pd.concat([df_clin_encoded, df_egfr[existing_gene_cols]], axis=1)

y_df = df_egfr[["OS_status", "OS_time"]]
y = np.array(
    [(bool(s), float(t)) for s, t in y_df.itertuples(index=False)],
    dtype=[("status", bool), ("time", np.float64)],
)

# Identical split parameters as script 05
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y["status"]
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp["status"]
)

scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=np.number).columns
scaler.fit(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(f"Test set rebuilt. Shape: {X_test.shape}")

# -----------------------------------------------------------------------
# RSF: validate and compute permutation importance
# -----------------------------------------------------------------------
if not RSF_MODEL_PATH.exists():
    raise FileNotFoundError(f"RSF model not found: {RSF_MODEL_PATH}")

rsf_model = joblib.load(RSF_MODEL_PATH)
rsf_c_index = rsf_model.score(X_test, y_test)
print(f"RSF replay C-index: {rsf_c_index:.5f}")


def c_index_scorer(estimator, X_eval, y_eval):
    risk_scores = estimator.predict(X_eval)
    return concordance_index_censored(
        y_eval["status"], y_eval["time"], risk_scores
    )[0]


print("Computing permutation importance (n_repeats=10) ...")
result = permutation_importance(
    rsf_model,
    X_test,
    y_test,
    scoring=c_index_scorer,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

feature_importance_df = pd.DataFrame(
    {"Feature": X_test.columns, "Importance_Drop": result.importances_mean}
)
feature_importance_df = (
    feature_importance_df[feature_importance_df["Importance_Drop"] > 0]
    .sort_values("Importance_Drop", ascending=False)
)
feature_importance_df.to_csv(OUTPUT_FEATURES_PATH, index=False)
print(f"RSF permutation importance saved: {OUTPUT_FEATURES_PATH}")
print(f"  Number of informative features: {len(feature_importance_df)}")
print("\nTop 20 features:")
print(feature_importance_df.head(20).to_string(index=False))

# -----------------------------------------------------------------------
# XGBoost: replay C-index
# -----------------------------------------------------------------------
if XGB_MODEL_PATH.exists():
    print("\nReplaying XGBoost C-index ...")
    xgb_model = joblib.load(XGB_MODEL_PATH)
    X_test_xgb = X_test.copy()
    X_test_xgb.columns = (
        pd.Index(X_test_xgb.columns.astype(str))
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False)
        .str.replace(">", "_", regex=False)
    )
    xgb_risk = xgb_model.predict(X_test_xgb)
    xgb_c_index = concordance_index_censored(
        y_test["status"], y_test["time"], xgb_risk
    )[0]
    print(f"XGBoost replay C-index: {xgb_c_index:.5f}")

print("\n--- [06] Permutation importance complete ---")
