"""
Script 03: Preprocess GSE31210 data
Parses the GEO series matrix file and local GPL570 annotation to produce
a per-sample expression + clinical CSV with EGFR mutation status.

Expected raw data layout (place under data/GSE31210/):
  - GSE31210_series_matrix.txt.gz
  - GPL570.annot.gz

Output:
  intermediate_data/GSE31210_ALL_SAMPLES_for_integration.csv
"""

import gzip
from io import StringIO
import numpy as np
import pandas as pd
from pathlib import Path

print("--- [03] Preprocessing GSE31210 data ---")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "GSE31210"
SERIES_MATRIX_PATH = DATA_DIR / "GSE31210_series_matrix.txt.gz"
GPL_ANNOT_PATH = DATA_DIR / "GPL570.annot.gz"
OUTPUT_DIR = PROJECT_ROOT / "intermediate_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "GSE31210_ALL_SAMPLES_for_integration.csv"


def read_gzip_table_between_markers(path, begin_marker, end_marker):
    with gzip.open(path, "rt", errors="ignore") as f:
        lines = f.readlines()

    begin_idx = end_idx = None
    for i, line in enumerate(lines):
        if line.startswith(begin_marker):
            begin_idx = i + 1
        if line.startswith(end_marker):
            end_idx = i
            break

    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        raise ValueError(
            f"Cannot locate table markers ({begin_marker} / {end_marker}) in {path}"
        )

    table_text = "".join(lines[begin_idx:end_idx])
    return pd.read_csv(StringIO(table_text), sep="\t", low_memory=False)


try:
    # Step 1: Load probe-level expression matrix
    print("Step 1: Loading probe-level expression matrix ...")
    expr_probe = read_gzip_table_between_markers(
        SERIES_MATRIX_PATH,
        "!series_matrix_table_begin",
        "!series_matrix_table_end",
    )
    probe_col = expr_probe.columns[0]
    expr_probe = expr_probe.rename(columns={probe_col: "probe_id"}).set_index("probe_id")
    expr_probe_t = expr_probe.T
    standard_sample_ids = expr_probe_t.index.tolist()
    print(f"  Expression matrix loaded. Number of samples: {len(standard_sample_ids)}")

    # Step 2: Parse clinical information from series matrix header
    print("Step 2: Parsing clinical information ...")
    sample_data = {}
    with gzip.open(SERIES_MATRIX_PATH, "rt", errors="ignore") as f:
        gsm_line = ""
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                gsm_line = (
                    line.strip().replace('!Sample_geo_accession = "', "").replace('"', "")
                )
                break
        gsm_ids = gsm_line.split("\t")

        for gsm_id in standard_sample_ids:
            sample_data[gsm_id] = {}

        f.seek(0)
        for line in f:
            if line.startswith("!Sample_characteristics_ch1"):
                cleaned = (
                    line.strip()
                    .replace('!Sample_characteristics_ch1 = "', "")
                    .replace('"', "")
                )
                chars = cleaned.split("\t")
                for i, char_string in enumerate(chars):
                    if i < len(gsm_ids) and gsm_ids[i] in sample_data:
                        parts = char_string.split(": ", 1)
                        if len(parts) == 2:
                            sample_data[gsm_ids[i]][parts[0].strip().lower()] = parts[1].strip()

    clinical_raw = pd.DataFrame.from_dict(sample_data, orient="index")

    clinical = pd.DataFrame(index=clinical_raw.index)
    clinical["age"] = pd.to_numeric(clinical_raw.get("age (years)"), errors="coerce")
    clinical["gender"] = clinical_raw.get("gender")
    clinical["stage"] = clinical_raw.get("pathological stage")
    clinical["smoking_status"] = clinical_raw.get("smoking status")
    clinical["OS_time"] = pd.to_numeric(
        clinical_raw.get("days before death/censor"), errors="coerce"
    )
    clinical["OS_status"] = (
        clinical_raw.get("death").astype(str).str.lower() == "dead"
    ).astype(int)
    # Strict annotation: only "EGFR mutation +" is counted as mutant
    gstatus = clinical_raw.get("gene alteration status").astype(str).str.strip()
    clinical["EGFR_mutation_status"] = (gstatus == "EGFR mutation +").astype(int)
    clinical["dataset"] = "GSE31210"
    print("  Clinical information parsed.")

    # Step 3: Map probes to gene symbols using local GPL570 annotation
    print("Step 3: Mapping probes to gene symbols via GPL570 annotation ...")
    annot = read_gzip_table_between_markers(
        GPL_ANNOT_PATH,
        "!platform_table_begin",
        "!platform_table_end",
    )

    id_col = "ID" if "ID" in annot.columns else annot.columns[0]
    symbol_col_candidates = [
        c for c in annot.columns if "gene symbol" in c.lower() or "symbol" in c.lower()
    ]
    if not symbol_col_candidates:
        raise ValueError("GPL570 annotation table: gene symbol column not found.")
    symbol_col = symbol_col_candidates[0]

    annot_sub = annot[[id_col, symbol_col]].copy()
    annot_sub = annot_sub.dropna(subset=[id_col, symbol_col])
    annot_sub[symbol_col] = (
        annot_sub[symbol_col].astype(str).str.split("///").str[0].str.strip()
    )
    annot_sub = annot_sub[
        annot_sub[symbol_col].ne("") & annot_sub[symbol_col].ne("---")
    ]
    annot_sub = annot_sub.drop_duplicates(subset=[id_col], keep="first")

    probe_to_gene = pd.Series(
        annot_sub[symbol_col].values, index=annot_sub[id_col]
    ).to_dict()

    expr_probe_t = expr_probe_t.rename(columns=probe_to_gene)
    mapped_cols = [c for c in expr_probe_t.columns if c in set(probe_to_gene.values())]
    expr_gene = expr_probe_t[mapped_cols]
    expr_gene = expr_gene.T.groupby(level=0).mean().T
    print(f"  Probe mapping complete. Number of genes: {expr_gene.shape[1]}")

    # Step 4: Merge and save
    print("Step 4: Merging clinical and expression data ...")
    df_final = clinical.join(expr_gene, how="inner")
    df_final.to_csv(OUTPUT_PATH)

    print(f"GSE31210 preprocessing complete. Output: {OUTPUT_PATH}")
    print(f"  Shape: {df_final.shape}")
    print("  EGFR mutation distribution:")
    print(df_final["EGFR_mutation_status"].value_counts(dropna=False))

except Exception as e:
    print(f"ERROR: GSE31210 preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
