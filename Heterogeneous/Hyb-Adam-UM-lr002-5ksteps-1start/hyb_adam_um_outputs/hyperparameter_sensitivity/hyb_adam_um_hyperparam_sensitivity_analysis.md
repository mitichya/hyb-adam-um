# Hyperparameter sensitivity analysis
The sensitivity experiment reuses frozen masks so that differences between rows are attributable to optimizer hyperparameters rather than to different missing-entry patterns. Metrics RMSE_miss, MAE_miss, Pearson_miss, and Spearman_miss are computed only on artificially hidden entries.
## Automatic summary
- Best mean RMSE_miss: **old_global_mean_init** (0.0253223).
- Best mean MAE_miss: **schedule_slow_decay** (0.0135622).
- Lowest mean final Delta: **lr_init_low** (21.9144).
- Fastest mean runtime: **fd_step_large** (35.0153 s).

## Relative changes versus baseline
- **old_global_mean_init**: RMSE_miss -0.24%, Delta_final -0.23%, runtime -0.13% relative to baseline.
- **two_restarts**: RMSE_miss +0.06%, Delta_final -0.07%, runtime +98.99% relative to baseline.
- **lr_init_low**: RMSE_miss -0.02%, Delta_final -1.71%, runtime +0.07% relative to baseline.
- **lr_init_high**: RMSE_miss -0.06%, Delta_final +3.17%, runtime +0.10% relative to baseline.
- **schedule_fast_decay**: RMSE_miss +0.16%, Delta_final -0.02%, runtime -0.88% relative to baseline.
- **schedule_slow_decay**: RMSE_miss -0.01%, Delta_final +2.80%, runtime -0.57% relative to baseline.
- **clip_low**: RMSE_miss -0.15%, Delta_final -0.12%, runtime -0.78% relative to baseline.
- **clip_high**: RMSE_miss -0.01%, Delta_final -0.01%, runtime -0.73% relative to baseline.
- **clip_none**: RMSE_miss +29.47%, Delta_final +89.09%, runtime -0.97% relative to baseline.
- **fd_step_small**: RMSE_miss -0.09%, Delta_final -0.37%, runtime -0.81% relative to baseline.
- **fd_step_large**: RMSE_miss +2.52%, Delta_final +1.04%, runtime -1.67% relative to baseline.

## Interpretation for the manuscript/rebuttal
If the RMSE_miss and Delta_final columns remain close across the tested learning-rate schedules, clipping thresholds, and finite-difference steps, then the method can be described as numerically stable within this local hyperparameter range. If one group changes substantially, report that group as the main sensitivity source and use the baseline as a fixed protocol for the main comparison. The observed-entry sanity columns should remain near machine precision; otherwise the completion routine is modifying entries that were not masked.
