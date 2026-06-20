# Reviewer sensitivity tables

## Files generated

- Initialization table: `hyb_adam_um_outputs/reviewer_sensitivity_tables/reviewer_initialization_sensitivity_table.tex`
- Hyperparameter table: `hyb_adam_um_outputs/reviewer_sensitivity_tables/reviewer_hyperparameter_sensitivity_table.tex`
- Initialization check CSV: `hyb_adam_um_outputs/reviewer_sensitivity_tables/reviewer_initialization_sensitivity_table_check.csv`
- Hyperparameter check CSV: `hyb_adam_um_outputs/reviewer_sensitivity_tables/reviewer_hyperparameter_sensitivity_table_check.csv`

## Initialization sensitivity: data-driven summary

- 30% missing: best RMSE_miss = Random observed range (0.0139653); RMSE spread across initializations = 0.00614699.
  Lowest final Delta: Random observed range (39.5866).
- 50% missing: best RMSE_miss = Baseline: row/column mean (0.013441); RMSE spread across initializations = 0.00899021.
  Lowest final Delta: Baseline: row/column mean (25.7351).
- 65% missing: best RMSE_miss = Baseline: row/column mean (0.0155939); RMSE spread across initializations = 0.00699318.
  Lowest final Delta: Random observed range (16.8305).
- 85% missing: best RMSE_miss = Random observed range (0.0398943); RMSE spread across initializations = 0.00173177.
  Lowest final Delta: Observed mean (4.09329).

## Hyperparameter sensitivity: data-driven summary

- Best mean RMSE_miss: FD step $10^{-4}$ (0.0133879); worst: No clipping (0.0307918).
- Lowest mean final Delta: FD step $10^{-4}$ (24.865).
- Fastest mean runtime: FD step $5\cdot 10^{-6}$ (36.1211 s).
- Maximum absolute RMSE_miss change relative to baseline: 129.09%.

## Important note about tree metrics

The initialization table could not include the following tree metrics because they were not present in the sensitivity CSV files: RF_norm, NJ_vs_ML_patristic_RMSE. The Hyb-Adam-only sensitivity runner computes matrix/objective metrics, not RF/patristic tree metrics. To include tree sensitivity metrics, save completed sensitivity matrices and run the tree-comparison pipeline on those matrices, then merge using the documented keys.

## Warnings / merge notes

- Optional tree-metric file not found: hyb_adam_um_outputs/initialization_sensitivity/hyb_adam_um_initialization_tree_metrics.csv
- Optional tree-metric file not found: hyb_adam_um_outputs/hyperparameter_sensitivity/hyb_adam_um_hyperparam_tree_metrics.csv
