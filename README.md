# LUAD EGFR-mutant 13-gene Prognostic Signature Reproducibility Package

This repository/package contains public-facing reproducibility materials for the revised manuscript on an EGFR-mutant lung adenocarcinoma prognostic signature.

The package is intentionally limited to low-risk, manuscript-facing materials:

- locked 13-gene signature definition;
- locked XGBoost feature-importance table used in the manuscript;
- source data tables for model-performance and validation analyses;
- source data tables for 13-gene stage-association analyses;
- scripts for cohort preprocessing, data cleaning, feature construction, model training/optimization, external validation, 13-gene score calculation.

It does not include private patient-level raw data, personal paths, local working logs, or historical model-search attempts.

## Folder Structure

```text
reproducibility_package/
  README.md
  requirements.txt
  data/
  scripts/
  docs/
```

See `scripts/README_pipeline.md` for the full script-level workflow.

## Key Data Files

- `data/locked_13gene_signature.csv`  
  Locked 13-gene signature definition.

- `data/supplementary_table_s2_locked_xgboost_features.csv`  
  Locked manuscript-facing XGBoost feature-importance table.

- `data/figure2_cindex_bootstrap_ci_summary.csv`  
  Bootstrap 95% confidence intervals for held-out test-set C-index.

- `data/figure2_time_dependent_auc_bootstrap_ci_summary.csv`  
  Bootstrap 95% confidence intervals for held-out test-set 1-, 3-, and 5-year time-dependent AUC.

- `data/gse13213_egfr_subset_survival_summary.csv` and `data/gse13213_egfr_subset_timepoint_auc.csv`  
  External validation results in the EGFR-mutant subset of GSE13213.

- `data/locked13_stage_association.csv` and `data/full_feature_stage_association.csv`  
  Stage-association source data used to quantify the 13-gene signature within the full feature background.

## 13-gene Score Definition

The locked 13-gene score is calculated as the arithmetic mean of the normalized expression values of the locked signature features:

```text
CDKAL1_EXP, CIR1_EXP, COQ8A_EXP, USP15_EXP, VPS50_EXP, PRODH_EXP,
KRT17_EXP, MAP2K3_EXP, P4HA2_EXP, MINDY1_EXP, TMEM164_EXP,
TENT5C_EXP, FCMR_EXP
```

For validation datasets, the same locked gene list should be used without re-optimization. If a platform lacks one or more genes, calculate the score from the mapped genes and report the number of genes used.

## Example Usage

Run the main preprocessing and model-training pipeline:

```bash
python scripts/01_preprocess_tcga.py
python scripts/02_preprocess_oncosg.py
python scripts/03_preprocess_gse31210.py
python scripts/04_integrate_and_filter.py
python scripts/05_train_models.py
```

Calculate a 13-gene score from a sample-by-feature expression matrix:

```bash
python scripts/calculate_13gene_score.py \
  --expression path/to/expression_matrix.csv \
  --out path/to/locked13_scores.csv
```

## Raw Data

Raw public datasets should be accessed from their original repositories or accession records. This package provides processed source data tables for the revised analyses and does not redistribute raw controlled or private patient-level data.

See `docs/data_availability_statement.md` for a suggested manuscript data-availability statement.
