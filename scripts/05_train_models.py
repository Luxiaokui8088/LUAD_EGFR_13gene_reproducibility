"""
Script 05: Train XGBoost and RSF survival models (8:1:1 split)
Trains XGBoost (Cox objective) and Random Survival Forest on 293 EGFR-mutant
LUAD samples using Bayesian hyperparameter optimisation with a predefined
validation split.

Input:
  intermediate_data/LUAD_EGFR_mut_3_datasets_integrated_FINAL.csv
  h.all.v2025.1.Hs.symbols.gmt   (MSigDB Hallmark gene set file)

Output (in results/model/):
  gene_level_xgb_model_811.joblib
  gene_level_rsf_model_811.joblib
  gene_level_xgb_features_811.csv   (non-zero XGBoost feature importances)
  gene_level_model_summary_811.csv  (C-index comparison table)
  X_train.csv / X_val.csv / X_test.csv / y_train.csv / y_val.csv / y_test.csv
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer

print("--- [05] Training XGBoost and RSF models (8:1:1 split) ---")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EGFR_SAMPLES_FILE = (
    PROJECT_ROOT / "intermediate_data" / "LUAD_EGFR_mut_3_datasets_integrated_FINAL.csv"
)
GMT_PATH = PROJECT_ROOT / "h.all.v2025.1.Hs.symbols.gmt"
OUTPUT_DIR = PROJECT_ROOT / "results" / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Four Hallmark pathways selected as feature space
TOP_PATHWAYS = [
    "HALLMARK_HEME_METABOLISM",
    "HALLMARK_KRAS_SIGNALING_DN",
    "HALLMARK_ESTROGEN_RESPONSE_EARLY",
    "HALLMARK_P53_PATHWAY",
]

# -----------------------------------------------------------------------
# Build feature matrix X and survival labels y
# -----------------------------------------------------------------------
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

y_df = df_egfr[["OS_status", "OS_time"]].copy()
y_sksurv = np.array(
    [(bool(s), float(t)) for s, t in y_df.itertuples(index=False)],
    dtype=[("status", bool), ("time", np.float64)],
)
y_xgb = np.where(y_df["OS_status"].astype(bool), y_df["OS_time"], -y_df["OS_time"])

print(f"Feature matrix shape: {X.shape}")

# -----------------------------------------------------------------------
# 8:1:1 train / validation / test split  (random_state=42, stratified)
# -----------------------------------------------------------------------
X_train, X_temp, y_sksurv_train, y_sksurv_temp, y_xgb_train, y_xgb_temp = (
    train_test_split(X, y_sksurv, y_xgb, test_size=0.2, random_state=42, stratify=y_sksurv["status"])
)
X_val, X_test, y_sksurv_val, y_sksurv_test, y_xgb_val, y_xgb_test = (
    train_test_split(
        X_temp, y_sksurv_temp, y_xgb_temp,
        test_size=0.5, random_state=42, stratify=y_sksurv_temp["status"],
    )
)
print(
    f"Split complete (8:1:1) — train: {len(X_train)}, "
    f"val: {len(X_val)}, test: {len(X_test)}"
)

# Save split indices for reproducibility
y_train_df = pd.DataFrame({"OS_status": y_sksurv_train["status"].astype(int), "OS_time": y_sksurv_train["time"]}, index=X_train.index)
y_val_df   = pd.DataFrame({"OS_status": y_sksurv_val["status"].astype(int),   "OS_time": y_sksurv_val["time"]},   index=X_val.index)
y_test_df  = pd.DataFrame({"OS_status": y_sksurv_test["status"].astype(int),  "OS_time": y_sksurv_test["time"]},  index=X_test.index)

X_train.to_csv(OUTPUT_DIR / "X_train.csv")
X_val.to_csv(OUTPUT_DIR / "X_val.csv")
X_test.to_csv(OUTPUT_DIR / "X_test.csv")
y_train_df.to_csv(OUTPUT_DIR / "y_train.csv")
y_val_df.to_csv(OUTPUT_DIR / "y_val.csv")
y_test_df.to_csv(OUTPUT_DIR / "y_test.csv")
print("Train/val/test split CSVs saved.")

# Standardize numeric features
scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=np.number).columns
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
print("Feature standardization complete.")

# Predefined split for BayesSearchCV (train=0, validation=-1)
split_index = np.zeros(len(X_train) + len(X_val))
split_index[len(X_train):] = -1
pds = PredefinedSplit(test_fold=split_index)

X_train_val = pd.concat([X_train, X_val])
y_sksurv_train_val = np.concatenate([y_sksurv_train, y_sksurv_val])

final_results: dict = {}

# -----------------------------------------------------------------------
# Random Survival Forest
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("Training Random Survival Forest ...")
try:
    rsf = RandomSurvivalForest(n_estimators=500, n_jobs=-1, random_state=42)
    search_spaces_rsf = {
        "max_features": Real(0.1, 1.0),
        "min_samples_leaf": Integer(10, 50),
    }

    bayes_search_rsf = BayesSearchCV(
        rsf, search_spaces_rsf, n_iter=32, cv=pds, n_jobs=-1, random_state=42
    ).fit(X_train_val, y_sksurv_train_val)

    print(f"  Bayesian optimisation complete.")
    print(f"  Best validation C-index: {bayes_search_rsf.best_score_:.5f}")
    print(f"  Best hyperparameters:    {bayes_search_rsf.best_params_}")

    final_rsf = bayes_search_rsf.best_estimator_
    final_rsf.fit(X_train_val, y_sksurv_train_val)

    test_score_rsf = final_rsf.score(X_test, y_sksurv_test)
    final_results["RSF"] = test_score_rsf
    print(f"  RSF test set C-index: {test_score_rsf:.5f}")

    joblib.dump(final_rsf, OUTPUT_DIR / "gene_level_rsf_model_811.joblib")
    print("  RSF model saved.")

except Exception as e:
    print(f"  ERROR: RSF training failed: {e}")
    final_results["RSF"] = np.nan

# -----------------------------------------------------------------------
# XGBoost (Cox proportional hazard objective)
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("Training XGBoost (Cox objective) ...")
try:
    xgb_model = xgb.XGBRegressor(
        objective="survival:cox",
        random_state=42,
        n_jobs=-1,
        eval_metric="cox-nloglik",
    )
    search_spaces_xgb = {
        "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
        "max_depth": Integer(3, 8),
        "n_estimators": Integer(100, 400),
    }

    # XGBoost does not accept [ ] < > in feature names
    safe_cols = (
        pd.Index(X_train.columns.astype(str))
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False)
        .str.replace(">", "_", regex=False)
    )
    col_map = dict(zip(safe_cols, X_train.columns))

    X_train_xgb = X_train.copy(); X_train_xgb.columns = safe_cols
    X_val_xgb   = X_val.copy();   X_val_xgb.columns   = safe_cols
    X_test_xgb  = X_test.copy();  X_test_xgb.columns  = safe_cols
    X_train_val_xgb = pd.concat([X_train_xgb, X_val_xgb])
    y_xgb_train_val = np.concatenate([y_xgb_train, y_xgb_val])

    bayes_search_xgb = BayesSearchCV(
        xgb_model, search_spaces_xgb, n_iter=32, cv=pds, n_jobs=-1, random_state=42
    ).fit(X_train_val_xgb, y_xgb_train_val)

    print(f"  Bayesian optimisation complete.")
    print(f"  Best validation C-index: {bayes_search_xgb.best_score_:.5f}")
    print(f"  Best hyperparameters:    {bayes_search_xgb.best_params_}")

    final_xgb = bayes_search_xgb.best_estimator_
    final_xgb.fit(X_train_val_xgb, y_xgb_train_val)

    risk_scores_xgb = final_xgb.predict(X_test_xgb)
    test_score_xgb = concordance_index_censored(
        y_sksurv_test["status"], y_sksurv_test["time"], risk_scores_xgb
    )[0]
    final_results["XGBoost"] = test_score_xgb
    print(f"  XGBoost test set C-index: {test_score_xgb:.5f}")

    joblib.dump(final_xgb, OUTPUT_DIR / "gene_level_xgb_model_811.joblib")
    print("  XGBoost model saved.")

    # Save non-zero feature importances (original column names)
    importances = pd.Series(final_xgb.feature_importances_, index=safe_cols)
    importances.index = [col_map.get(c, c) for c in importances.index]
    feature_df = (
        importances[importances > 0]
        .sort_values(ascending=False)
        .to_frame("Importance")
    )
    feature_df.to_csv(OUTPUT_DIR / "gene_level_xgb_features_811.csv")
    print(f"  XGBoost feature importances saved ({len(feature_df)} non-zero features).")

except Exception as e:
    print(f"  ERROR: XGBoost training failed: {e}")
    final_results["XGBoost"] = np.nan

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
summary_df = pd.DataFrame.from_dict(
    final_results, orient="index", columns=["Test Set C-index"]
).sort_values("Test Set C-index", ascending=False)
print(summary_df)
summary_df.to_csv(OUTPUT_DIR / "gene_level_model_summary_811.csv")

print("\n--- [05] Model training complete ---")
