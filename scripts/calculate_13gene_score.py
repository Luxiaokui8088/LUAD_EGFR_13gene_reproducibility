#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the locked 13-gene score from a sample-by-feature expression matrix."
    )
    parser.add_argument("--expression", required=True, help="CSV expression matrix; rows=samples, columns=features.")
    parser.add_argument(
        "--signature",
        default=str(Path(__file__).resolve().parents[1] / "data" / "locked_13gene_signature.csv"),
        help="Locked signature CSV with a feature_name column.",
    )
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument(
        "--sample-column",
        default=None,
        help="Optional sample ID column. If omitted, the first CSV column is used as the index.",
    )
    args = parser.parse_args()

    sig = pd.read_csv(args.signature)
    features = sig["feature_name"].dropna().astype(str).tolist()

    if args.sample_column:
        expr = pd.read_csv(args.expression).set_index(args.sample_column)
    else:
        expr = pd.read_csv(args.expression, index_col=0)

    present = [f for f in features if f in expr.columns]
    missing = [f for f in features if f not in expr.columns]
    if not present:
        raise ValueError("None of the locked 13-gene features were found in the expression matrix.")

    score = expr[present].mean(axis=1)
    out = pd.DataFrame(
        {
            "sample_id": expr.index.astype(str),
            "locked13_score": score.to_numpy(),
            "n_signature_genes_used": len(present),
            "missing_signature_genes": ";".join(missing) if missing else "",
        }
    )
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
