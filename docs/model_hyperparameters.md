# Model Hyperparameters and Evaluation Notes

This file records the manuscript-facing model settings used for revision documentation.

## XGBoost

- Objective: `survival:cox`
- Learning rate: `0.02155609982756265`
- Maximum tree depth: `3`
- Number of estimators: `100`
- Evaluation metric: C-index and time-dependent AUC in the held-out test set

## Random Survival Forest

- Maximum features: `0.13561195982829205`
- Minimum samples per leaf: `10`
- Number of estimators: `500`
- Evaluation metric: C-index and time-dependent AUC in the held-out test set

## Data Split

- Development design: 8:1:1 train/validation/test split
- Held-out test set size: `n = 30`

## Revision Additions

- Bootstrap 95% confidence intervals for held-out test-set C-index
- Bootstrap 95% confidence intervals for 1-, 3-, and 5-year time-dependent AUC
- Kaplan-Meier number-at-risk tables
- EGFR-mutant subset validation in GSE13213

