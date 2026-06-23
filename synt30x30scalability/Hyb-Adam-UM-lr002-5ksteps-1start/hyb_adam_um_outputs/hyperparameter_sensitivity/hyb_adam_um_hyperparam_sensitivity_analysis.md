# Hyperparameter sensitivity analysis
The sensitivity experiment reuses frozen masks so that differences between rows are attributable to optimizer hyperparameters rather than to different missing-entry patterns. Metrics RMSE_miss, MAE_miss, Pearson_miss, and Spearman_miss are computed only on artificially hidden entries.
## Automatic summary
- Best mean RMSE_miss: **old_global_mean_init** (0.0279633).
- Best mean MAE_miss: **schedule_slow_decay** (0.0162363).
- Lowest mean final Delta: **lr_init_low** (865.751).
- Fastest mean runtime: **schedule_fast_decay** (781 s).

## Relative changes versus baseline
- **old_global_mean_init**: RMSE_miss -0.79%, Delta_final -0.21%, runtime -0.26% relative to baseline.
- **two_restarts**: RMSE_miss +0.14%, Delta_final -0.27%, runtime +99.76% relative to baseline.
- **lr_init_low**: RMSE_miss +0.67%, Delta_final -0.88%, runtime -1.14% relative to baseline.
- **lr_init_high**: RMSE_miss +0.54%, Delta_final +0.05%, runtime +0.02% relative to baseline.
- **schedule_fast_decay**: RMSE_miss -0.00%, Delta_final -0.10%, runtime -1.17% relative to baseline.
- **schedule_slow_decay**: RMSE_miss -0.32%, Delta_final +1.60%, runtime +1.04% relative to baseline.
- **clip_low**: RMSE_miss +0.16%, Delta_final -0.21%, runtime +2.22% relative to baseline.
- **clip_high**: RMSE_miss +0.25%, Delta_final -0.30%, runtime +2.83% relative to baseline.
- **clip_none**: RMSE_miss +15.07%, Delta_final +48.18%, runtime +11.34% relative to baseline.
- **fd_step_small**: RMSE_miss -0.06%, Delta_final +0.05%, runtime +2.37% relative to baseline.
- **fd_step_large**: RMSE_miss +0.23%, Delta_final -0.49%, runtime +2.57% relative to baseline.

## Interpretation for the manuscript/rebuttal
If the RMSE_miss and Delta_final columns remain close across the tested learning-rate schedules, clipping thresholds, and finite-difference steps, then the method can be described as numerically stable within this local hyperparameter range. If one group changes substantially, report that group as the main sensitivity source and use the baseline as a fixed protocol for the main comparison. The observed-entry sanity columns should remain near machine precision; otherwise the completion routine is modifying entries that were not masked.
