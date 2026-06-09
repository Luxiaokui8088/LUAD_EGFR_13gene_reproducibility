"""
Restore the original 293-sample feature matrix used for the 8:1:1 XGBoost split.

Inputs
- results/model/X_train.csv, X_val.csv, X_test.csv
- results/model/y_train.csv, y_val.csv, y_test.csv

Outputs
- results/model/restored_original_feature_matrix_293_samples.csv
- results/model/restored_original_expression_matrix_293_samples.csv
- results/model/restored_original_13gene_expression_matrix_293_samples.csv
- results/model/restored_original_feature_matrix_with_survival.csv
- results/model/restored_original_feature_matrix_manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LOCKED_13_GENES_EXP = [
    "CDKAL1_EXP", "CIR1_EXP", "COQ8A_EXP", "USP15_EXP", "VPS50_EXP",
    "PRODH_EXP", "KRT17_EXP", "MAP2K3_EXP", "P4HA2_EXP", "MINDY1_EXP",
    "TMEM164_EXP", "TENT5C_EXP", "FCMR_EXP",
]


def load_split_table(base_dir: Path, stem: str) -> pd.DataFrame:
    return pd.read_csv(base_dir / f"{stem}.csv", index_col=0, low_memory=False)


def main() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    workspace_root = project_root.parent

    split_dir = project_root / "results" / "model"
    out_dir = project_root / "results" / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tables = [
        load_split_table(split_dir, "X_train"),
        load_split_table(split_dir, "X_val"),
        load_split_table(split_dir, "X_test"),
    ]
    y_tables = [
        load_split_table(split_dir, "y_train"),
        load_split_table(split_dir, "y_val"),
        load_split_table(split_dir, "y_test"),
    ]

    x_all = pd.concat(x_tables, axis=0)
    y_all = pd.concat(y_tables, axis=0)

    if x_all.index.has_duplicates:
        dup = x_all.index[x_all.index.duplicated()].tolist()[:10]
        raise RuntimeError(f"Duplicated sample IDs found in X tables: {dup}")
    if y_all.index.has_duplicates:
        dup = y_all.index[y_all.index.duplicated()].tolist()[:10]
        raise RuntimeError(f"Duplicated sample IDs found in y tables: {dup}")
    if set(x_all.index) != set(y_all.index):
        missing_in_y = sorted(set(x_all.index) - set(y_all.index))[:10]
        missing_in_x = sorted(set(y_all.index) - set(x_all.index))[:10]
        raise RuntimeError(
            "X/y sample IDs do not match. "
            f"Missing in y: {missing_in_y}; missing in X: {missing_in_x}"
        )

    y_all = y_all.loc[x_all.index]

    expr_cols = [c for c in x_all.columns if str(c).endswith("_EXP")]
    clin_cols = [c for c in x_all.columns if c not in expr_cols]
    locked_13_present = [c for c in LOCKED_13_GENES_EXP if c in x_all.columns]

    x_out = out_dir / "restored_original_feature_matrix_293_samples.csv"
    expr_out = out_dir / "restored_original_expression_matrix_293_samples.csv"
    locked_13_out = out_dir / "restored_original_13gene_expression_matrix_293_samples.csv"
    merged_out = out_dir / "restored_original_feature_matrix_with_survival.csv"
    manifest_out = out_dir / "restored_original_feature_matrix_manifest.json"

    x_all.to_csv(x_out)
    x_all[expr_cols].to_csv(expr_out)
    x_all[locked_13_present].to_csv(locked_13_out)
    pd.concat([x_all, y_all], axis=1).to_csv(merged_out)

    manifest = {
        "script": str(script_path),
        "inputs": {
            "split_dir": str(split_dir),
            "x_files": ["X_train.csv", "X_val.csv", "X_test.csv"],
            "y_files": ["y_train.csv", "y_val.csv", "y_test.csv"],
        },
        "outputs": {
            "feature_matrix": str(x_out),
            "expression_only_matrix": str(expr_out),
            "locked_13_expression_matrix": str(locked_13_out),
            "feature_plus_survival": str(merged_out),
        },
        "shape": {
            "samples": int(x_all.shape[0]),
            "features_total": int(x_all.shape[1]),
            "expression_features": int(len(expr_cols)),
            "clinical_or_encoded_features": int(len(clin_cols)),
        },
        "clinical_or_encoded_feature_names": clin_cols,
        "locked_13_genes_requested": LOCKED_13_GENES_EXP,
        "locked_13_genes_present": locked_13_present,
    }
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Restoration complete.")
    print(f"Feature matrix: {x_out}")
    print(f"Expression-only matrix: {expr_out}")
    print(f"Feature+survival matrix: {merged_out}")
    print(
        f"Shape: {x_all.shape[0]} samples x {x_all.shape[1]} features "
        f"({len(expr_cols)} expression + {len(clin_cols)} clinical/encoded)"
    )


if __name__ == "__main__":
    main()
