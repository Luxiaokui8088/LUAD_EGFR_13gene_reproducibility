# Reviewer Response Mapping

This document maps the public reproducibility materials to the reviewer concerns addressed during revision.

## Reviewer 1: Statistical Reporting and Presentation

### Concern

Report the XGBoost model C-index with 95% confidence interval, add number-at-risk tables to Kaplan-Meier curves, and clarify the 13-gene signature.

### Materials

- `data/figure2_cindex_bootstrap_ci_summary.csv`
- `figures/figure2_cindex_bootstrap_ci.pdf`
- `figures/figure2_rsf_km_with_at_risk_table.png`
- `figures/figure2_xgboost_km_with_at_risk_table.png`
- `data/locked_13gene_signature.csv`
- `scripts/calculate_13gene_score.py`

### Response Summary

Bootstrap 95% confidence intervals were added for held-out test-set C-index. Kaplan-Meier panels were updated with number-at-risk tables. The locked 13-gene signature and score calculation are provided as reproducible source files.

## Reviewer 3 Major Concern 1: Small Held-out Test Set

### Concern

The held-out test set is small; model performance estimates should include uncertainty, including C-index and time-dependent AUC bootstrap confidence intervals.

### Materials

- `data/figure2_cindex_bootstrap_ci_summary.csv`
- `data/figure2_time_dependent_auc_bootstrap_ci_summary.csv`
- `figures/figure2_cindex_bootstrap_ci.pdf`
- `figures/figure2_time_dependent_auc_bootstrap_ci.pdf`
- `scripts/plot_model_performance_ci.py`

### Response Summary

The revised analyses quantify uncertainty using bootstrap 95% confidence intervals for both Harrell's C-index and 1-, 3-, and 5-year time-dependent AUC in the held-out test set. The confidence intervals are presented to reflect the limited event count in the test set.

## Reviewer 3 Major Concern 2: External Validation in EGFR-mutant Subset

### Concern

Because the training cohort was restricted to EGFR-mutant LUAD, external validation should be repeated within the EGFR-mutant subset of GSE13213 rather than only in the mixed full cohort.

### Materials

- `data/gse13213_egfr_subset_survival_summary.csv`
- `data/gse13213_egfr_subset_timepoint_auc.csv`
- `figures/gse13213_egfr_subset_km.png`
- `figures/gse13213_egfr_subset_roc_1y3y5y.png`
- `scripts/08_external_validation_gse13213.py`

### Response Summary

The GSE13213 validation was repeated in the EGFR-mutant subset. The deposited tables and figures provide the subset Kaplan-Meier, Cox, and time-dependent ROC results.

## Reviewer 3 Major Concern 3: Quantitative Support for the 13-gene Signature

### Concern

The transition from a larger ranked feature list to the 13-gene signature should be supported quantitatively rather than relying only on visual heatmap inspection.

### Materials

- `data/supplementary_table_s2_locked_xgboost_features.csv`
- `data/locked_13gene_signature.csv`
- `data/full_feature_stage_association.csv`
- `data/locked13_stage_association.csv`
- `data/locked13_stage_association_summary.csv`
- `data/locked13_convergence_summary.csv`
- `figures/locked13_stage_rank_distribution.png`
- `scripts/09_13gene_convergence_analysis.py`
- `scripts/09b_13gene_stage_association.py`
- `scripts/09h_plot_13gene_stage_rank_distribution.py`

### Response Summary

The locked 13-gene signature is documented together with full-feature stage-association analysis, locked 13-gene association summaries, and module-convergence summaries. These materials support the interpretation of the 13-gene set as a locked, clinically interpretable module rather than an arbitrary visual subset.

## Reviewer 3 Minor Concern 2: Multiple-testing Correction

### Concern

Multiple-testing correction should be described for gene-level and pairwise analyses.

### Materials

- `data/full_feature_stage_association.csv`
- `data/locked13_stage_association.csv`
- `scripts/09b_13gene_stage_association.py`

### Response Summary

The stage-association outputs include FDR-adjusted statistics using Benjamini-Hochberg correction where applicable.

## Reviewer 3 Minor Concern 4: Reproducibility Resources

### Concern

The authors should provide code, hyperparameters, signature details, and source data for reproducibility.

### Materials

- `README.md`
- `requirements.txt`
- `scripts/README_pipeline.md`
- `docs/model_hyperparameters.md`
- `docs/data_availability_statement.md`
- `data/locked_13gene_signature.csv`
- `data/supplementary_table_s2_locked_xgboost_features.csv`
- all scripts under `scripts/`

### Response Summary

The public package contains the cleaned pipeline scripts for data preprocessing, integration, model training and optimization, external validation, 13-gene score calculation, and revision-specific statistical analyses. It also includes the locked signature, model hyperparameters, processed figure source data, and a suggested data-availability statement.

## Materials Intentionally Not Included

The package does not include private raw patient-level data, local working logs, historical exploratory retraining attempts, or model binaries. Public raw datasets should be accessed from their original repositories or accession records.

