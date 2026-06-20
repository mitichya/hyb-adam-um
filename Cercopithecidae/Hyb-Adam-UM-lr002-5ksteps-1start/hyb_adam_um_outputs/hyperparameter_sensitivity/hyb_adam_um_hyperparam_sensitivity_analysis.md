# Hyperparameter sensitivity analysis
The sensitivity experiment reuses frozen masks so that differences between rows are attributable to optimizer hyperparameters rather than to different missing-entry patterns. Metrics RMSE_miss, MAE_miss, Pearson_miss, and Spearman_miss are computed only on artificially hidden entries.
## Automatic summary
- Best mean RMSE_miss: **fd_step_large** (0.0133879).
- Best mean MAE_miss: **fd_step_large** (0.00509239).
- Lowest mean final Delta: **fd_step_large** (24.865).
- Fastest mean runtime: **fd_step_large** (35.1637 s).

## Relative changes versus baseline
- **old_global_mean_init**: RMSE_miss +58.51%, Delta_final +19.47%, runtime -1.34% relative to baseline.
- **two_restarts**: RMSE_miss +0.06%, Delta_final -0.51%, runtime +94.15% relative to baseline.
- **lr_init_low**: RMSE_miss +34.32%, Delta_final +7.12%, runtime -3.19% relative to baseline.
- **lr_init_high**: RMSE_miss +30.97%, Delta_final +41.60%, runtime -2.73% relative to baseline.
- **schedule_fast_decay**: RMSE_miss +0.10%, Delta_final -0.37%, runtime -3.94% relative to baseline.
- **schedule_slow_decay**: RMSE_miss -0.05%, Delta_final +10.26%, runtime -3.34% relative to baseline.
- **clip_low**: RMSE_miss -0.35%, Delta_final +1.01%, runtime -3.87% relative to baseline.
- **clip_high**: RMSE_miss +23.82%, Delta_final +7.46%, runtime -3.37% relative to baseline.
- **clip_none**: RMSE_miss +129.09%, Delta_final +796.64%, runtime -3.65% relative to baseline.
- **fd_step_small**: RMSE_miss -0.16%, Delta_final +0.47%, runtime -3.15% relative to baseline.
- **fd_step_large**: RMSE_miss -0.40%, Delta_final -3.38%, runtime -4.25% relative to baseline.

## Interpretation for the manuscript/rebuttal
If the RMSE_miss and Delta_final columns remain close across the tested learning-rate schedules, clipping thresholds, and finite-difference steps, then the method can be described as numerically stable within this local hyperparameter range. If one group changes substantially, report that group as the main sensitivity source and use the baseline as a fixed protocol for the main comparison. The observed-entry sanity columns should remain near machine precision; otherwise the completion routine is modifying entries that were not masked.
