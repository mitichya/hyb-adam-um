# -*- coding: utf-8 -*-
"""
Hyb-Adam-UM distance-matrix completion experiment
=================================================

This is a full rewrite of the previous Hyb-Adam-UM-only script with the requested
reviewer-oriented diagnostics and tables.

Implemented changes
-------------------
1. Replicates increased to 30; optimizer epochs set to 6000.
2. Source/reference matrix changed to:
       Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv
3. Runtime is measured for every run and summarized as mean ± std by missingness.
4. RMSE_miss, MAE_miss, Pearson_miss, Spearman_miss are computed ONLY on entries
   hidden by the artificial mask.
5. For every missingness level the summary reports:
       n_missing / n_observed / optimized_variables
   and average convergence epochs.
6. Failure counts are reported:
       n_success, n_failed
7. Sanity check on observed entries is reported:
       max_abs_error_observed, mean_abs_error_observed
8. Hyb-Adam-UM diagnostics are stored as numeric columns:
       Delta_init, Delta_final, Delta_reduction_percent,
       best_epoch, epochs_used, convergence_epoch, final_learning_rate
9. Epoch-level/checkpoint optimizer logs are written to files instead of
   flooding Jupyter notebook output.
10. Every detailed row contains:
       mask_seed, missingness_actual
11. A separate hyperparameter-sensitivity experiment is included for:
       learning-rate schedule, clipping threshold, finite-difference step size.
    It produces a detailed table, summary table, and a Markdown analysis file.

Important computational note
----------------------------
The optimizer uses central finite differences over all missing lower-triangle
variables. This is expensive for large matrices and high missingness. For large
n, this full-objective finite-difference implementation can require substantial runtime;
SENSITIVITY_REPS can be reduced for quick pilot runs while keeping the main
protocol unchanged.
"""

from __future__ import annotations

import os
import json
import time
import shutil
import warnings
from dataclasses import dataclass, replace, asdict
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# -------------------------- CONFIG ---------------------------
# ============================================================

# Reference matrix. The first path is for a normal working directory; the second
# path is convenient when running in a notebook/container where /mnt/data is used.
ORIG_CANDIDATES: Tuple[str, ...] = (
    "Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv",
    "/mnt/data/Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv",
)

# Missingness experiment.
MISSING_FRACS: Tuple[float, ...] = (0.30, 0.50, 0.65, 0.85)
REPS: int = 30
BASE_SEED: int = 55

# Output folders.
OUTPUT_ROOT: str = "hyb_adam_um_outputs"
MASKED_DIR: str = os.path.join(OUTPUT_ROOT, "masked_matrices")
COMPLETED_DIR: str = os.path.join(OUTPUT_ROOT, "completed_matrices")
TABLES_DIR: str = os.path.join(OUTPUT_ROOT, "tables")
SENSITIVITY_DIR: str = os.path.join(OUTPUT_ROOT, "hyperparameter_sensitivity")
LOGS_DIR: str = os.path.join(OUTPUT_ROOT, "logs")
EPOCH_LOGS_DIR: str = os.path.join(LOGS_DIR, "epoch_logs")

# Missing value marker used in artificially incomplete matrices.
MISSING_VAL: float = -1.0

# Robust-delta constants.
OMEGA: float = 2.0
EPS_NUM: float = 1.0e-12


@dataclass(frozen=True)
class OptimizerConfig:
    """Manual Adam + finite-difference configuration."""

    epochs: int = 10000
    print_every: int = 500

    lr_init: float = 0.04
    lr_milestones: Tuple[int, ...] = (700, 2000, 4000, 6000, 8000)
    lr_factor: float = 0.5
    lr_min: float = 1.0e-4
    sched_patience_blocks: int = 7

    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = 5.0
    h_central_diff: float = 5.0e-5

    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1.0e-8

    seed_hyb: int = 42

    # Conservative improvements that keep the same main algorithm:
    # robust-Delta objective + finite-difference Adam over missing entries.
    # "global_mean" reproduces the old initialization exactly.
    # "two_way_mean" is usually more stable because it uses observed row/column scale.
    init_strategy: str = "two_way_mean"
    n_restarts: int = 2
    restart_jitter_scale: float = 0.03
    max_distance_factor: Optional[float] = 1.50
    normalize_objective_by_triplets: bool = True

    # Convergence detection is evaluated only at logging/checkpoint epochs.
    # A run is considered converged after this many consecutive checkpoints
    # without relative improvement larger than convergence_rel_tol.
    convergence_rel_tol: float = 1.0e-7
    convergence_patience_checks: int = 5
    early_stop: bool = False


BASE_OPT_CONFIG = OptimizerConfig()

# Hyperparameter sensitivity block.
RUN_HYPERPARAM_SENSITIVITY: bool = True
SENSITIVITY_MISSING_FRAC: float = 0.50
SENSITIVITY_REPS: int = 5
# Sensitivity runs reuse frozen masks from the main experiment. By default only
# the first 5 replicates are used to keep the reviewer-response table feasible.

# Notebook-safe logging. Epoch/checkpoint diagnostics are written to files.
# Keep this False for Jupyter notebooks to avoid thousands of printed lines.
PRINT_EPOCH_PROGRESS_TO_CONSOLE: bool = False


# ============================================================
# ------------------------ I/O HELPERS ------------------------
# ============================================================


def ensure_dirs() -> None:
    for d in (
        OUTPUT_ROOT,
        MASKED_DIR,
        COMPLETED_DIR,
        TABLES_DIR,
        SENSITIVITY_DIR,
        LOGS_DIR,
        EPOCH_LOGS_DIR,
    ):
        os.makedirs(d, exist_ok=True)



def display_or_print(df: pd.DataFrame, title: Optional[str] = None) -> None:
    if title:
        print(f"\n=== {title} ===")
    try:
        from IPython.display import display

        display(df)
    except Exception:
        print(df.to_string(index=False))



def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def append_text_log(path: Optional[str], message: str, also_print: bool = False) -> None:
    """Append one message to a text log file, optionally mirroring it to console."""
    if also_print:
        print(message)
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def save_epoch_log(records: List[Dict[str, object]], path: Optional[str]) -> None:
    """Save checkpoint-level optimizer history to CSV."""
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def safe_tag(text: str) -> str:
    """Convert an arbitrary label into a filesystem-safe short tag."""
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"



def load_distance_matrix_from_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load a square distance matrix from a CSV file.

    Supported common formats:
    - row labels in the first column and column labels in the header;
    - no row labels, only numeric square matrix;
    - delimiter inferred by pandas, including comma/semicolon/tab.
    """

    # Case 1: labels in first column, headers in first row.
    try:
        df = pd.read_csv(path, sep=None, engine="python", index_col=0)
        num = df.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if num.shape[0] == num.shape[1] and num.shape[0] > 1:
            labels = [str(x) for x in num.index.to_list()]
            M = num.to_numpy(dtype=float)
            return M, labels
    except Exception:
        pass

    # Case 2: pure numeric square matrix or label column accidentally included.
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        num = df.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if num.shape[0] == num.shape[1] and num.shape[0] > 1:
            labels = [f"T{i + 1}" for i in range(num.shape[0])]
            M = num.to_numpy(dtype=float)
            return M, labels
    except Exception:
        pass

    # Case 3: final fallback to numpy text loading.
    try:
        M = np.loadtxt(path, delimiter=",")
    except Exception:
        M = np.loadtxt(path)
    labels = [f"T{i + 1}" for i in range(M.shape[0])]
    return np.asarray(M, dtype=float), labels



def load_matrix_with_candidates(candidates: Sequence[str]) -> Tuple[np.ndarray, List[str], str]:
    for p in candidates:
        if os.path.exists(p):
            M, labels = load_distance_matrix_from_csv(p)
            # Preserve the previous script's scale guard for legacy NW-style files.
            if np.nanmax(M) > 500:
                M = M / 1000.0
            return M, labels, p
    raise FileNotFoundError(f"Reference matrix not found. Tried: {list(candidates)}")


# ============================================================
# ----------------------- MATRIX HELPERS ----------------------
# ============================================================


def symmetrize_full(D: np.ndarray) -> np.ndarray:
    M = 0.5 * (np.asarray(D, dtype=float) + np.asarray(D, dtype=float).T)
    np.fill_diagonal(M, 0.0)
    return M



def symmetrize_with_missing(D: np.ndarray) -> np.ndarray:
    """Symmetrize while preserving missing pairs as MISSING_VAL."""
    M = np.asarray(D, dtype=float).copy()
    n = M.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = M[i, j], M[j, i]
            if a >= 0 and b >= 0:
                v = 0.5 * (a + b)
            elif a >= 0:
                v = a
            elif b >= 0:
                v = b
            else:
                v = MISSING_VAL
            M[i, j] = M[j, i] = v
    np.fill_diagonal(M, 0.0)
    return M



def _finite_fill(v: np.ndarray, fallback: float = 1.0) -> float:
    arr = np.asarray(v, dtype=float)
    finite = np.isfinite(arr)
    if finite.any():
        return float(np.nanmedian(arr[finite]))
    return float(fallback)



def sanitize_reference_distance_matrix(D: np.ndarray, name: str = "D_ref") -> np.ndarray:
    """
    Strictly validate and lightly normalize a complete reference distance matrix.

    Important: this function intentionally does NOT median-fill NaN/Inf/negative
    values. The reference matrix is the ground truth of the benchmark, so bad
    entries must be reported explicitly rather than silently repaired.
    """
    M = np.asarray(D, dtype=float).copy()
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {M.shape}.")

    n = M.shape[0]
    off = ~np.eye(n, dtype=bool)

    if np.any(~np.isfinite(M)):
        raise ValueError(f"{name} contains NaN/Inf values.")

    if np.any(M[off] < 0):
        raise ValueError(f"{name} contains negative off-diagonal distances.")

    # Symmetrize small numerical/asymmetric CSV noise, then enforce distance basics.
    M = 0.5 * (M + M.T)
    M = np.maximum(M, 0.0)
    np.fill_diagonal(M, 0.0)

    if not np.isfinite(M).all():
        raise ValueError(f"{name} has non-finite entries after sanitization.")
    if np.any(M[off] < 0):
        raise ValueError(f"{name} has negative off-diagonal distances after sanitization.")
    if not np.allclose(M, M.T, atol=1.0e-12):
        raise ValueError(f"{name} is not symmetric after sanitization.")
    if not np.allclose(np.diag(M), 0.0, atol=1.0e-12):
        raise ValueError(f"{name} diagonal is not zero after sanitization.")

    return M

def finalize_completed_matrix(
    D_hat: np.ndarray,
    D_incomplete: np.ndarray,
    observed_pairs: np.ndarray,
    preserve_observed: bool = True,
) -> np.ndarray:
    """
    Finalize completed matrix while preserving observed entries exactly.

    Unlike the previous sanitize_distance_matrix() call, this function does not
    quantile-clip entries, because clipping can silently change observed values
    and invalidate the sanity check.
    """
    M = np.asarray(D_hat, dtype=float).copy()
    M = 0.5 * (M + M.T)
    M = np.maximum(M, 0.0)
    np.fill_diagonal(M, 0.0)

    if preserve_observed and observed_pairs.size > 0:
        i, j = observed_pairs[:, 0], observed_pairs[:, 1]
        vals = D_incomplete[i, j]
        M[i, j] = vals
        M[j, i] = vals
        np.fill_diagonal(M, 0.0)

    if not np.isfinite(M).all():
        raise FloatingPointError("Completed matrix contains non-finite values.")
    return M



def lower_pairs(n: int) -> np.ndarray:
    i, j = np.tril_indices(n, k=-1)
    return np.column_stack([i, j]).astype(np.int32)



def values_on_pairs(M: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.array([], dtype=float)
    return np.asarray(M[pairs[:, 0], pairs[:, 1]], dtype=float)


# ============================================================
# ----------------------- ROBUST DELTA ------------------------
# ============================================================


def robust_delta_per_triplet_numpy(M: np.ndarray, triplets: np.ndarray) -> np.ndarray:
    """Return robust delta(i,j,k) for each triplet."""
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    a = M[i, j]
    b = M[i, k]
    c = M[j, k]

    S = np.stack([a, b, c], axis=1)
    S_sorted = -np.sort(-S, axis=1)
    a, b, c = S_sorted[:, 0], S_sorted[:, 1], S_sorted[:, 2]

    viol = a >= (b + c)
    denom_v = np.maximum(b + c, EPS_NUM)
    delta1 = np.maximum(a / denom_v, OMEGA)

    denomA = np.maximum(2.0 * b * c, EPS_NUM)
    denomB = np.maximum(2.0 * a * c, EPS_NUM)
    denomG = np.maximum(2.0 * a * b, EPS_NUM)

    cosA = np.clip((b * b + c * c - a * a) / denomA, -1.0, 1.0)
    cosB = np.clip((a * a + c * c - b * b) / denomB, -1.0, 1.0)
    cosG = np.clip((a * a + b * b - c * c) / denomG, -1.0, 1.0)

    A = np.arccos(cosA)
    B = np.arccos(cosB)
    G = np.arccos(cosG)

    Ang = np.stack([A, B, G], axis=1)
    Ang_sorted = -np.sort(-Ang, axis=1)
    A, B, G = Ang_sorted[:, 0], Ang_sorted[:, 1], Ang_sorted[:, 2]
    delta2 = (A - B) / np.maximum(G, EPS_NUM)

    return np.where(viol, delta1, delta2)



def robust_delta_sum_numpy(M: np.ndarray, triplets: np.ndarray) -> float:
    return float(robust_delta_per_triplet_numpy(M, triplets).sum())



def compute_normalized_delta(M: np.ndarray, triplets: np.ndarray) -> float:
    """Log-compressed mean delta in [0,1], using max_reasonable_delta=100."""
    if triplets.size == 0:
        return float("nan")
    delta_vals = robust_delta_per_triplet_numpy(M, triplets)
    max_reasonable_delta = 100.0
    delta_norm_vals = np.log1p(delta_vals) / np.log1p(max_reasonable_delta)
    delta_norm_vals = np.clip(delta_norm_vals, 0.0, 1.0)
    return float(np.mean(delta_norm_vals))


# ============================================================
# -------------------------- METRICS --------------------------
# ============================================================


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(d * d)))



def mae(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))



def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]
    if len(a) < 2 or np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])



def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]
    if len(a) < 2:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    return pearson_corr(ra, rb)



def completion_metrics_on_missing(
    D_completed: np.ndarray,
    D_reference: np.ndarray,
    missing_pairs: np.ndarray,
) -> Dict[str, float]:
    pred = values_on_pairs(D_completed, missing_pairs)
    true = values_on_pairs(D_reference, missing_pairs)
    return {
        "RMSE_miss": rmse(pred, true),
        "MAE_miss": mae(pred, true),
        "Pearson_miss": pearson_corr(pred, true),
        "Spearman_miss": spearman_corr(pred, true),
    }



def observed_sanity_metrics(
    D_completed: np.ndarray,
    D_reference: np.ndarray,
    observed_pairs: np.ndarray,
) -> Dict[str, float]:
    pred = values_on_pairs(D_completed, observed_pairs)
    true = values_on_pairs(D_reference, observed_pairs)
    if len(pred) == 0:
        return {
            "max_abs_error_observed": float("nan"),
            "mean_abs_error_observed": float("nan"),
        }
    err = np.abs(pred - true)
    return {
        "max_abs_error_observed": float(np.max(err)),
        "mean_abs_error_observed": float(np.mean(err)),
    }



def mean_std_nan(x: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean_val, std_val



def fmt_pm(mean_val: float, std_val: float, decimals: int = 6) -> str:
    if not np.isfinite(mean_val):
        return "nan"
    if not np.isfinite(std_val):
        return f"{mean_val:.{decimals}f} ± nan"
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


# ============================================================
# ---------------------- MASK GENERATION ----------------------
# ============================================================


def simulate_missing(
    D_full: np.ndarray,
    frac_missing: float,
    mask_seed: int,
) -> Dict[str, object]:
    """Mask lower-triangle entries and mirror them symmetrically."""
    rng = np.random.RandomState(mask_seed)
    n = D_full.shape[0]
    pairs = lower_pairs(n)
    m_total = len(pairs)
    n_missing = int(round(frac_missing * m_total))

    keep = np.ones(m_total, dtype=bool)
    if n_missing > 0:
        drop_idx = rng.choice(m_total, size=n_missing, replace=False)
        keep[drop_idx] = False

    observed_pairs = pairs[keep]
    missing_pairs = pairs[~keep]

    D_inc = D_full.copy().astype(float)
    if len(missing_pairs) > 0:
        i, j = missing_pairs[:, 0], missing_pairs[:, 1]
        D_inc[i, j] = MISSING_VAL
        D_inc[j, i] = MISSING_VAL
    np.fill_diagonal(D_inc, 0.0)
    D_inc = symmetrize_with_missing(D_inc)

    missingness_actual = len(missing_pairs) / m_total if m_total else float("nan")

    return {
        "D_inc": D_inc,
        "observed_pairs": observed_pairs,
        "missing_pairs": missing_pairs,
        "n_pairs_total": int(m_total),
        "n_missing": int(len(missing_pairs)),
        "n_observed": int(len(observed_pairs)),
        "missingness_actual": float(missingness_actual),
        "mask_seed": int(mask_seed),
    }



def build_mask_registry(D0: np.ndarray) -> List[Dict[str, object]]:
    registry: List[Dict[str, object]] = []
    for frac in MISSING_FRACS:
        for rep in range(1, REPS + 1):
            mask_seed = BASE_SEED + (rep - 1)
            rec = simulate_missing(D0, frac, mask_seed)
            rec.update(
                {
                    "frac_requested": float(frac),
                    "pct_missing": int(round(100 * frac)),
                    "replicate": int(rep),
                }
            )
            registry.append(rec)
    return registry



def save_masked_matrices(mask_registry: List[Dict[str, object]], labels: List[str]) -> None:
    os.makedirs(MASKED_DIR, exist_ok=True)
    for rec in mask_registry:
        pct = int(rec["pct_missing"])
        rep = int(rec["replicate"])
        fn = f"missing_p{pct}_rep{rep:02d}_seed{rec['mask_seed']}.csv"
        path = os.path.join(MASKED_DIR, fn)
        pd.DataFrame(rec["D_inc"], index=labels, columns=labels).to_csv(path)
        rec["masked_file"] = fn


# ============================================================
# ---------------------- HYB-ADAM-UM CORE ---------------------
# ============================================================


def _observed_offdiag_values(M_in: np.ndarray) -> np.ndarray:
    """Return observed lower-triangle off-diagonal distances."""
    M = np.asarray(M_in, dtype=float)
    lower_mask = np.tril(np.ones(M.shape, dtype=bool), k=-1)
    vals = M[lower_mask & (M >= 0.0)]
    vals = vals[np.isfinite(vals)]
    return vals.astype(np.float64)


def _observed_row_means(M_in: np.ndarray, fallback: float) -> np.ndarray:
    """Observed row means, excluding diagonal and missing entries."""
    M = np.asarray(M_in, dtype=float)
    n = M.shape[0]
    out = np.full(n, float(fallback), dtype=np.float64)
    for i in range(n):
        mask = (M[i, :] >= 0.0) & np.isfinite(M[i, :])
        mask[i] = False
        if np.any(mask):
            out[i] = float(np.mean(M[i, mask]))
    return out


def make_initial_missing_values(
    M_in: np.ndarray,
    missing_pairs: np.ndarray,
    strategy: str = "two_way_mean",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build an initial vector for missing lower-triangle entries.

    This only changes the starting point, not the Hyb-Adam-UM objective.
    strategy="global_mean" reproduces the previous script's initialization.
    strategy="two_way_mean" blends per-taxon observed means with the global mean.
    """
    n_missing = int(len(missing_pairs))
    obs_vals = _observed_offdiag_values(M_in)

    if len(obs_vals) > 0:
        global_mean = float(np.mean(obs_vals))
        global_median = float(np.median(obs_vals))
        obs_max = float(np.max(obs_vals))
    else:
        global_mean = 1.0
        global_median = 1.0
        obs_max = 1.0

    if n_missing == 0:
        return np.array([], dtype=np.float64), {
            "initial_global_mean": global_mean,
            "initial_global_median": global_median,
            "observed_max_distance": obs_max,
        }

    strategy_norm = str(strategy).strip().lower()
    if strategy_norm in {"global", "mean", "global_mean", "old"}:
        x0 = np.full(n_missing, global_mean, dtype=np.float64)
    elif strategy_norm in {"two_way", "two_way_mean", "row_mean", "rowcol", "row_col_mean"}:
        row_means = _observed_row_means(M_in, fallback=global_mean)
        i, j = missing_pairs[:, 0], missing_pairs[:, 1]
        pair_mean = 0.5 * (row_means[i] + row_means[j])
        # Shrink toward the global mean. This avoids overreacting when a taxon has
        # very few observed distances at high missingness.
        x0 = 0.75 * pair_mean + 0.25 * global_mean
    else:
        raise ValueError(
            f"Unknown init_strategy={strategy!r}. Use 'global_mean' or 'two_way_mean'."
        )

    x0 = np.asarray(x0, dtype=np.float64)
    x0[~np.isfinite(x0)] = global_mean
    x0 = np.maximum(x0, 0.0)
    return x0, {
        "initial_global_mean": global_mean,
        "initial_global_median": global_median,
        "observed_max_distance": obs_max,
    }


def setup_problem_for_hyb_adam(
    M_in: np.ndarray,
    triplets: np.ndarray,
    init_strategy: str = "two_way_mean",
):
    """Prepare lower-triangle parameterization over missing entries."""
    n = M_in.shape[0]
    lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    given_mask_lower = lower_mask & (M_in >= 0.0)
    missing_mask_lower = lower_mask & (M_in < 0.0)

    given_pairs = np.array(np.where(given_mask_lower)).T.astype(np.int32)
    missing_pairs = np.array(np.where(missing_mask_lower)).T.astype(np.int32)

    given_vals = M_in[given_mask_lower].astype(np.float64)
    x0, init_stats = make_initial_missing_values(M_in, missing_pairs, strategy=init_strategy)

    def assemble_full(xvec: np.ndarray) -> np.ndarray:
        M = np.zeros((n, n), dtype=np.float64)
        if len(given_pairs) > 0:
            gi, gj = given_pairs[:, 0], given_pairs[:, 1]
            M[gi, gj] = given_vals
        if len(missing_pairs) > 0:
            mi, mj = missing_pairs[:, 0], missing_pairs[:, 1]
            M[mi, mj] = xvec
        M = M + M.T
        np.fill_diagonal(M, 0.0)
        np.maximum(M, 0.0, out=M)
        return M

    return x0, given_pairs, missing_pairs, given_vals, triplets, assemble_full, init_stats


def central_diff_grad(x: np.ndarray, f: Callable[[np.ndarray], float], h: float) -> np.ndarray:
    g = np.zeros_like(x)
    for k in range(x.size):
        x[k] += h
        f1 = f(x)
        x[k] -= 2.0 * h
        f2 = f(x)
        x[k] += h
        g[k] = (f1 - f2) / (2.0 * h)
    return g


def _make_restart_initial_x(
    x0: np.ndarray,
    restart_idx: int,
    rng: np.random.RandomState,
    jitter_scale: float,
    scale_value: float,
    upper_bound: Optional[float],
) -> np.ndarray:
    """Create deterministic/noisy restart around the same biologically scaled initial point."""
    x = np.asarray(x0, dtype=np.float64).copy()
    if restart_idx > 0 and x.size > 0 and jitter_scale > 0.0:
        noise_sd = float(jitter_scale) * max(float(scale_value), EPS_NUM)
        x = x + rng.normal(loc=0.0, scale=noise_sd, size=x.shape)
    x = np.maximum(x, 0.0)
    if upper_bound is not None and np.isfinite(upper_bound) and upper_bound > 0:
        x = np.minimum(x, float(upper_bound))
    return x


def _optimize_one_restart(
    x_start: np.ndarray,
    assemble_full: Callable[[np.ndarray], np.ndarray],
    all_triplets: np.ndarray,
    opt: OptimizerConfig,
    restart_idx: int,
    upper_bound: Optional[float],
    log_path: Optional[str],
    verbose: bool,
) -> Dict[str, object]:
    """Run the same Adam finite-difference optimizer from one initial point."""
    ntrip = max(int(len(all_triplets)), 1)

    def objective_delta_raw(xvec: np.ndarray) -> float:
        return robust_delta_sum_numpy(assemble_full(xvec), all_triplets)

    def objective_for_grad(xvec: np.ndarray) -> float:
        val = objective_delta_raw(xvec)
        if opt.normalize_objective_by_triplets:
            val = val / ntrip
        return float(val)

    x = np.asarray(x_start, dtype=np.float64).copy()
    Delta_init = float(objective_delta_raw(x))
    obj_init = float(objective_for_grad(x))

    append_text_log(
        log_path,
        f"Restart {restart_idx}: initial Delta={Delta_init:.6f}; objective_for_grad={obj_init:.8g}",
        also_print=verbose,
    )

    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    lr = float(opt.lr_init)

    best_x = x.copy()
    best_obj = float(obj_init)
    best_delta = float(Delta_init)
    best_epoch = 0

    last_block_best = float(obj_init)
    no_improve_blocks = 0

    convergence_epoch: Optional[int] = None
    convergence_no_improve_checks = 0
    convergence_best_at_check = float(obj_init)

    epochs_used = 0
    epoch_records: List[Dict[str, object]] = []

    for epoch in range(1, opt.epochs + 1):
        epochs_used = epoch
        t += 1

        g = central_diff_grad(x, objective_for_grad, h=opt.h_central_diff)

        if opt.weight_decay > 0.0:
            g = g + opt.weight_decay * x

        if opt.clip_grad_norm is not None:
            g_norm = float(np.linalg.norm(g))
            if g_norm > opt.clip_grad_norm and g_norm > 0:
                g = g * (opt.clip_grad_norm / g_norm)

        m = opt.beta1 * m + (1.0 - opt.beta1) * g
        v = opt.beta2 * v + (1.0 - opt.beta2) * (g * g)
        m_hat = m / (1.0 - (opt.beta1 ** t))
        v_hat = v / (1.0 - (opt.beta2 ** t))
        x -= lr * (m_hat / (np.sqrt(v_hat) + opt.adam_eps))
        x = np.maximum(x, 0.0)
        if upper_bound is not None and np.isfinite(upper_bound) and upper_bound > 0:
            x = np.minimum(x, float(upper_bound))

        if epoch in opt.lr_milestones and lr > opt.lr_min + 1.0e-12:
            new_lr = max(opt.lr_min, lr * opt.lr_factor)
            if new_lr < lr - 1.0e-12:
                append_text_log(
                    log_path,
                    f"Restart {restart_idx}, epoch {epoch:05d}: lr milestone {lr:.6g} -> {new_lr:.6g}",
                    also_print=verbose,
                )
            lr = new_lr

        checkpoint = (epoch == 1) or (epoch % opt.print_every == 0) or (epoch == opt.epochs)
        if checkpoint:
            current_obj = float(objective_for_grad(x))
            current_delta = float(objective_delta_raw(x))

            if current_obj < best_obj - 1.0e-12:
                best_obj = current_obj
                best_delta = current_delta
                best_x = x.copy()
                best_epoch = int(epoch)

            append_text_log(
                log_path,
                f"Restart {restart_idx}, epoch {epoch:5d} | Delta={current_delta:.6f} | "
                f"best Delta={best_delta:.6f} | obj={current_obj:.8g} | lr={lr:.6g}",
                also_print=verbose,
            )
            epoch_records.append(
                {
                    "restart": int(restart_idx),
                    "epoch": int(epoch),
                    "full_delta": float(current_delta),
                    "best_delta": float(best_delta),
                    "objective_for_grad": float(current_obj),
                    "best_objective_for_grad": float(best_obj),
                    "best_epoch": int(best_epoch),
                    "learning_rate": float(lr),
                    "optimized_variables": int(x.size),
                    "event": "checkpoint",
                }
            )

            # Plateau schedule.
            if current_obj < last_block_best - 1.0e-12:
                no_improve_blocks = 0
                last_block_best = float(current_obj)
            else:
                no_improve_blocks += 1
                if no_improve_blocks >= opt.sched_patience_blocks and lr > opt.lr_min + 1.0e-12:
                    new_lr = max(opt.lr_min, lr * opt.lr_factor)
                    if new_lr < lr - 1.0e-12:
                        append_text_log(
                            log_path,
                            f"Restart {restart_idx}, epoch {epoch:05d}: lr plateau {lr:.6g} -> {new_lr:.6g}",
                            also_print=verbose,
                        )
                        if epoch_records:
                            epoch_records[-1]["event"] = "checkpoint_lr_plateau"
                    lr = new_lr
                    no_improve_blocks = 0

            # Convergence diagnostics.
            denom = max(abs(convergence_best_at_check), EPS_NUM)
            rel_improvement = (convergence_best_at_check - best_obj) / denom
            if rel_improvement > opt.convergence_rel_tol:
                convergence_best_at_check = float(best_obj)
                convergence_no_improve_checks = 0
            else:
                convergence_no_improve_checks += 1
                if (
                    convergence_epoch is None
                    and convergence_no_improve_checks >= opt.convergence_patience_checks
                ):
                    convergence_epoch = int(epoch)
                    append_text_log(
                        log_path,
                        f"Restart {restart_idx}: convergence detected at epoch {convergence_epoch}",
                        also_print=verbose,
                    )
                    if epoch_records:
                        epoch_records[-1]["event"] = "checkpoint_convergence_detected"
                    if opt.early_stop:
                        break

    if convergence_epoch is None:
        convergence_epoch = int(best_epoch if best_epoch > 0 else epochs_used)

    return {
        "restart": int(restart_idx),
        "best_x": best_x,
        "Delta_init": float(Delta_init),
        "Delta_final": float(best_delta),
        "objective_init": float(obj_init),
        "objective_final": float(best_obj),
        "best_epoch": int(best_epoch),
        "epochs_used": int(epochs_used),
        "convergence_epoch": int(convergence_epoch),
        "final_learning_rate": float(lr),
        "epoch_records": epoch_records,
    }


def hyb_adam_um_impute(
    D_in: np.ndarray,
    trip_all: np.ndarray,
    opt: OptimizerConfig,
    verbose: bool = False,
    log_path: Optional[str] = None,
    epoch_log_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Optimize missing lower-triangle entries with the original robust-Delta objective.

    Improvements are intentionally conservative:
    - better observed-scale initialization;
    - optional noisy restarts around that initialization;
    - gradient objective normalization by number of triplets;
    - optional upper clipping to the observed distance scale.

    The objective being minimized remains robust_delta_sum_numpy(assemble_full(x), triplets).
    """
    rng = np.random.RandomState(opt.seed_hyb)
    all_epoch_records: List[Dict[str, object]] = []

    # Start a fresh text log for this optimization run.
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Hyb-Adam-UM optimizer log\n")
            f.write(json.dumps(asdict(opt), ensure_ascii=False) + "\n")

    x0, _, missing_pairs, _, all_triplets, assemble_full, init_stats = setup_problem_for_hyb_adam(
        D_in,
        trip_all,
        init_strategy=opt.init_strategy,
    )

    obs_max = float(init_stats.get("observed_max_distance", np.nan))
    init_scale = float(init_stats.get("initial_global_median", np.nan))
    if not np.isfinite(init_scale) or init_scale <= 0:
        init_scale = float(init_stats.get("initial_global_mean", 1.0))
    if not np.isfinite(init_scale) or init_scale <= 0:
        init_scale = 1.0

    upper_bound: Optional[float]
    if opt.max_distance_factor is not None and np.isfinite(obs_max) and obs_max > 0:
        upper_bound = float(opt.max_distance_factor) * obs_max
    else:
        upper_bound = None

    if x0.size == 0:
        M0 = assemble_full(x0)
        delta0 = robust_delta_sum_numpy(M0, all_triplets)
        all_epoch_records.append(
            {
                "restart": 0,
                "epoch": 0,
                "full_delta": float(delta0),
                "best_delta": float(delta0),
                "best_epoch": 0,
                "learning_rate": float(opt.lr_init),
                "event": "skipped_no_missing_pairs",
            }
        )
        save_epoch_log(all_epoch_records, epoch_log_path)
        info = {
            "success": True,
            "skipped": True,
            "reason": "no missing pairs",
            "optimized_variables": 0,
            "Delta_init": float(delta0),
            "Delta_final": float(delta0),
            "Delta_reduction_percent": 0.0,
            "best_epoch": 0,
            "epochs_used": 0,
            "convergence_epoch": 0,
            "final_learning_rate": opt.lr_init,
            "init_strategy": opt.init_strategy,
            "n_restarts": 0,
            "best_restart": 0,
            "upper_bound": float(upper_bound) if upper_bound is not None else float("nan"),
            "run_log_file": log_path or "",
            "epoch_log_file": epoch_log_path or "",
        }
        return M0, info

    n_restarts = max(1, int(opt.n_restarts))
    restart_results: List[Dict[str, object]] = []

    append_text_log(
        log_path,
        f"Initial strategy={opt.init_strategy}; restarts={n_restarts}; "
        f"upper_bound={upper_bound}; optimized_variables={x0.size}; "
        f"normalize_objective_by_triplets={opt.normalize_objective_by_triplets}",
        also_print=verbose,
    )

    for r in range(n_restarts):
        x_start = _make_restart_initial_x(
            x0=x0,
            restart_idx=r,
            rng=rng,
            jitter_scale=opt.restart_jitter_scale,
            scale_value=init_scale,
            upper_bound=upper_bound,
        )
        res = _optimize_one_restart(
            x_start=x_start,
            assemble_full=assemble_full,
            all_triplets=all_triplets,
            opt=opt,
            restart_idx=r,
            upper_bound=upper_bound,
            log_path=log_path,
            verbose=verbose,
        )
        restart_results.append(res)
        all_epoch_records.extend(res["epoch_records"])

    save_epoch_log(all_epoch_records, epoch_log_path)

    finite_results = [r for r in restart_results if np.isfinite(r["Delta_final"])]
    if not finite_results:
        raise FloatingPointError("All Hyb-Adam-UM restarts produced non-finite objective values.")

    best_res = min(finite_results, key=lambda z: float(z["Delta_final"]))
    best_x = np.asarray(best_res["best_x"], dtype=np.float64)
    M_best = assemble_full(best_x)
    M_best = 0.5 * (M_best + M_best.T)
    M_best = np.maximum(M_best, 0.0)
    np.fill_diagonal(M_best, 0.0)

    Delta_init_best_restart = float(best_res["Delta_init"])
    Delta_final = float(best_res["Delta_final"])
    if np.isfinite(Delta_init_best_restart) and abs(Delta_init_best_restart) > EPS_NUM:
        Delta_reduction_percent = 100.0 * (Delta_init_best_restart - Delta_final) / Delta_init_best_restart
    else:
        Delta_reduction_percent = float("nan")

    append_text_log(
        log_path,
        f"Selected restart {best_res['restart']} with final Delta={Delta_final:.6f}; "
        f"reduction={Delta_reduction_percent:.6f}%; best_epoch={best_res['best_epoch']}; "
        f"epochs_used={best_res['epochs_used']}; final_lr={best_res['final_learning_rate']:.6g}",
        also_print=verbose,
    )

    info = {
        "success": bool(np.isfinite(Delta_final)),
        "skipped": False,
        "optimized_variables": int(x0.size),
        "Delta_init": Delta_init_best_restart,
        "Delta_final": Delta_final,
        "Delta_reduction_percent": float(Delta_reduction_percent),
        "best_epoch": int(best_res["best_epoch"]),
        "epochs_used": int(best_res["epochs_used"]),
        "convergence_epoch": int(best_res["convergence_epoch"]),
        "final_learning_rate": float(best_res["final_learning_rate"]),
        "init_strategy": opt.init_strategy,
        "n_restarts": int(n_restarts),
        "best_restart": int(best_res["restart"]),
        "best_restart_initial_Delta": Delta_init_best_restart,
        "best_restart_final_Delta": Delta_final,
        "upper_bound": float(upper_bound) if upper_bound is not None else float("nan"),
        "initial_global_mean": float(init_stats.get("initial_global_mean", np.nan)),
        "initial_global_median": float(init_stats.get("initial_global_median", np.nan)),
        "observed_max_distance": obs_max,
        "run_log_file": log_path or "",
        "epoch_log_file": epoch_log_path or "",
    }
    return M_best, info


# ============================================================
# ----------------------- SINGLE RUNNER -----------------------
# ============================================================


def is_valid_completed_distance_matrix(D_completed: np.ndarray, D_reference: np.ndarray) -> bool:
    """Implementation-level validity check for a completed distance matrix."""
    if D_completed.shape != D_reference.shape:
        return False
    if not np.isfinite(D_completed).all():
        return False
    if not np.allclose(D_completed, D_completed.T, atol=1.0e-12):
        return False
    if not np.allclose(np.diag(D_completed), 0.0, atol=1.0e-12):
        return False
    if float(np.min(D_completed)) < -1.0e-12:
        return False
    return True


def run_single_mask(
    rec: Dict[str, object],
    D_reference: np.ndarray,
    labels: List[str],
    trip_all: np.ndarray,
    ntri: int,
    Delta_original: float,
    Delta_normalized_original: float,
    opt: OptimizerConfig,
    method_label: str = "Hyb-Adam-UM",
    save_completed: bool = True,
    completed_prefix: str = "HybAdamUM_completed",
    verbose: bool = False,
    write_optimizer_logs: bool = True,
) -> Dict[str, object]:
    pct = int(rec["pct_missing"])
    rep = int(rec["replicate"])
    D_inc = np.asarray(rec["D_inc"], dtype=float)
    missing_pairs = np.asarray(rec["missing_pairs"], dtype=np.int32)
    observed_pairs = np.asarray(rec["observed_pairs"], dtype=np.int32)

    row: Dict[str, object] = {
        "method": method_label,
        "pct_missing": pct,
        "frac_missing_requested": float(rec["frac_requested"]),
        "missingness_actual": float(rec["missingness_actual"]),
        "replicate": rep,
        "mask_seed": int(rec["mask_seed"]),
        "n_pairs_total": int(rec["n_pairs_total"]),
        "n_missing": int(rec["n_missing"]),
        "n_observed": int(rec["n_observed"]),
        "optimized_variables": int(rec["n_missing"]),
        # success means numerical/runtime validity, not necessarily objective improvement.
        "success": False,
        "success_numerical": False,
        "success_valid_matrix": False,
        "objective_improved": False,
        "error_message": "",
        "masked_file": rec.get("masked_file", ""),
        "completed_file": "",
        "run_log_file": "",
        "epoch_log_file": "",
    }

    run_tag = safe_tag(f"{completed_prefix}_p{pct}_rep{rep:02d}_seed{row['mask_seed']}")
    run_log_path = os.path.join(LOGS_DIR, f"{run_tag}.log") if write_optimizer_logs else None
    epoch_log_path = os.path.join(EPOCH_LOGS_DIR, f"{run_tag}_epochs.csv") if write_optimizer_logs else None

    t0 = time.perf_counter()
    try:
        D_completed_raw, info = hyb_adam_um_impute(
            D_inc,
            trip_all=trip_all,
            opt=opt,
            verbose=bool(verbose and PRINT_EPOCH_PROGRESS_TO_CONSOLE),
            log_path=run_log_path,
            epoch_log_path=epoch_log_path,
        )
        D_completed = finalize_completed_matrix(
            D_completed_raw,
            D_incomplete=D_inc,
            observed_pairs=observed_pairs,
            preserve_observed=True,
        )
        runtime_seconds = time.perf_counter() - t0

        miss_metrics = completion_metrics_on_missing(D_completed, D_reference, missing_pairs)
        sanity = observed_sanity_metrics(D_completed, D_reference, observed_pairs)

        # Hard implementation sanity check. Observed entries must be preserved.
        if np.isfinite(sanity["max_abs_error_observed"]) and sanity["max_abs_error_observed"] > 1.0e-10:
            raise RuntimeError(
                "Observed entries were modified: "
                f"max_abs_error_observed={sanity['max_abs_error_observed']:.6g}"
            )

        Delta_total = robust_delta_sum_numpy(D_completed, trip_all)
        Delta_norm = compute_normalized_delta(D_completed, trip_all)
        Delta_per_triangle = Delta_total / ntri if ntri > 0 else float("nan")
        Delta_relative = Delta_total / Delta_original if abs(Delta_original) > EPS_NUM else float("nan")

        Delta_init = float(info.get("Delta_init", np.nan))
        Delta_final = float(info.get("Delta_final", np.nan))

        success_numerical = bool(
            np.isfinite(Delta_init)
            and np.isfinite(Delta_final)
            and np.isfinite(runtime_seconds)
            and np.isfinite(Delta_total)
            and np.isfinite(D_completed).all()
            and bool(info.get("success", True))
        )
        success_valid_matrix = bool(is_valid_completed_distance_matrix(D_completed, D_reference))
        objective_improved = bool(
            np.isfinite(Delta_init)
            and np.isfinite(Delta_final)
            and Delta_final <= Delta_init * (1.0 - 1.0e-8)
        )
        success = bool(success_numerical and success_valid_matrix)

        completed_file = ""
        if save_completed:
            completed_file = f"{completed_prefix}_p{pct}_rep{rep:02d}_seed{row['mask_seed']}.csv"
            completed_path = os.path.join(COMPLETED_DIR, completed_file)
            pd.DataFrame(D_completed, index=labels, columns=labels).to_csv(completed_path)

        row.update(
            {
                "success": success,
                "success_numerical": success_numerical,
                "success_valid_matrix": success_valid_matrix,
                "objective_improved": objective_improved,
                "runtime_seconds": float(runtime_seconds),
                "completed_file": completed_file,
                "run_log_file": info.get("run_log_file", run_log_path or ""),
                "epoch_log_file": info.get("epoch_log_file", epoch_log_path or ""),
                "Delta_total_completed": float(Delta_total),
                "Delta_normalized_completed": float(Delta_norm),
                "Delta_per_triangle_completed": float(Delta_per_triangle),
                "Delta_relative_to_original": float(Delta_relative),
                "Delta_original": float(Delta_original),
                "Delta_normalized_original": float(Delta_normalized_original),
                **miss_metrics,
                **sanity,
                "Delta_init": Delta_init,
                "Delta_final": Delta_final,
                "Delta_reduction_percent": float(info.get("Delta_reduction_percent", np.nan)),
                "best_epoch": int(info.get("best_epoch", -1)),
                "epochs_used": int(info.get("epochs_used", -1)),
                "convergence_epoch": int(info.get("convergence_epoch", -1)),
                "final_learning_rate": float(info.get("final_learning_rate", np.nan)),
                "init_strategy": info.get("init_strategy", opt.init_strategy),
                "n_restarts": int(info.get("n_restarts", opt.n_restarts)),
                "best_restart": int(info.get("best_restart", -1)),
                "best_restart_initial_Delta": float(info.get("best_restart_initial_Delta", np.nan)),
                "best_restart_final_Delta": float(info.get("best_restart_final_Delta", np.nan)),
                "upper_bound": float(info.get("upper_bound", np.nan)),
                "initial_global_mean": float(info.get("initial_global_mean", np.nan)),
                "initial_global_median": float(info.get("initial_global_median", np.nan)),
                "observed_max_distance": float(info.get("observed_max_distance", np.nan)),
                "optimizer_config_json": json.dumps(asdict(opt), ensure_ascii=False),
            }
        )
    except Exception as e:
        runtime_seconds = time.perf_counter() - t0
        row.update(
            {
                "success": False,
                "success_numerical": False,
                "success_valid_matrix": False,
                "objective_improved": False,
                "runtime_seconds": float(runtime_seconds),
                "error_message": repr(e),
                "run_log_file": run_log_path or "",
                "epoch_log_file": epoch_log_path or "",
                "Delta_total_completed": float("nan"),
                "Delta_normalized_completed": float("nan"),
                "Delta_per_triangle_completed": float("nan"),
                "Delta_relative_to_original": float("nan"),
                "Delta_original": float(Delta_original),
                "Delta_normalized_original": float(Delta_normalized_original),
                "RMSE_miss": float("nan"),
                "MAE_miss": float("nan"),
                "Pearson_miss": float("nan"),
                "Spearman_miss": float("nan"),
                "max_abs_error_observed": float("nan"),
                "mean_abs_error_observed": float("nan"),
                "Delta_init": float("nan"),
                "Delta_final": float("nan"),
                "Delta_reduction_percent": float("nan"),
                "best_epoch": -1,
                "epochs_used": -1,
                "convergence_epoch": -1,
                "final_learning_rate": float("nan"),
                "init_strategy": opt.init_strategy,
                "n_restarts": opt.n_restarts,
                "best_restart": -1,
                "best_restart_initial_Delta": float("nan"),
                "best_restart_final_Delta": float("nan"),
                "upper_bound": float("nan"),
                "initial_global_mean": float("nan"),
                "initial_global_median": float("nan"),
                "observed_max_distance": float("nan"),
                "optimizer_config_json": json.dumps(asdict(opt), ensure_ascii=False),
            }
        )
        append_text_log(run_log_path, f"FAILED p{pct} rep{rep}: {repr(e)}", also_print=True)

    return row


# ============================================================
# -------------------------- SUMMARIES ------------------------
# ============================================================


SUMMARY_NUMERIC_COLUMNS: Tuple[str, ...] = (
    "missingness_actual",
    "n_missing",
    "n_observed",
    "optimized_variables",
    "RMSE_miss",
    "MAE_miss",
    "Pearson_miss",
    "Spearman_miss",
    "runtime_seconds",
    "Delta_init",
    "Delta_final",
    "Delta_reduction_percent",
    "best_epoch",
    "epochs_used",
    "convergence_epoch",
    "final_learning_rate",
    "n_restarts",
    "best_restart",
    "best_restart_initial_Delta",
    "best_restart_final_Delta",
    "upper_bound",
    "initial_global_mean",
    "initial_global_median",
    "observed_max_distance",
    "Delta_total_completed",
    "Delta_normalized_completed",
    "Delta_per_triangle_completed",
    "Delta_relative_to_original",
    "max_abs_error_observed",
    "mean_abs_error_observed",
)



def summarize_by_missingness(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows: List[Dict[str, object]] = []
    formatted_rows: List[Dict[str, object]] = []

    for pct, group0 in results_df.groupby("pct_missing", sort=True):
        success_mask = group0["success"].astype(bool)
        group = group0[success_mask].copy()
        n_success = int(success_mask.sum())
        n_failed = int(len(group0) - n_success)
        n_objective_improved = int(group0.get("objective_improved", pd.Series(False, index=group0.index)).astype(bool).sum())
        n_not_improved = int(len(group0) - n_objective_improved)

        num_row: Dict[str, object] = {
            "pct_missing": int(pct),
            "n_runs": int(len(group0)),
            "n_success": n_success,
            "n_failed": n_failed,
            "n_objective_improved": n_objective_improved,
            "n_not_improved": n_not_improved,
        }
        fmt_row: Dict[str, object] = {
            "% Missing": f"{int(pct)}%",
            "n_runs": int(len(group0)),
            "n_success": n_success,
            "n_failed": n_failed,
            "n_objective_improved": n_objective_improved,
            "n_not_improved": n_not_improved,
        }

        for col in SUMMARY_NUMERIC_COLUMNS:
            if col in group.columns:
                mean_val, std_val = mean_std_nan(group[col].to_numpy(dtype=float))
            else:
                mean_val, std_val = float("nan"), float("nan")
            num_row[f"{col}_mean"] = mean_val
            num_row[f"{col}_std"] = std_val

        fmt_specs = {
            "missingness_actual": 4,
            "n_missing": 1,
            "n_observed": 1,
            "optimized_variables": 1,
            "RMSE_miss": 6,
            "MAE_miss": 6,
            "Pearson_miss": 6,
            "Spearman_miss": 6,
            "runtime_seconds": 3,
            "Delta_init": 4,
            "Delta_final": 4,
            "Delta_reduction_percent": 3,
            "best_epoch": 1,
            "epochs_used": 1,
            "convergence_epoch": 1,
            "final_learning_rate": 6,
            "n_restarts": 1,
            "best_restart": 1,
            "best_restart_initial_Delta": 4,
            "best_restart_final_Delta": 4,
            "upper_bound": 6,
            "initial_global_mean": 6,
            "initial_global_median": 6,
            "observed_max_distance": 6,
            "Delta_total_completed": 4,
            "Delta_normalized_completed": 6,
            "Delta_per_triangle_completed": 6,
            "Delta_relative_to_original": 6,
            "max_abs_error_observed": 12,
            "mean_abs_error_observed": 12,
        }
        for col, dec in fmt_specs.items():
            fmt_row[col] = fmt_pm(num_row[f"{col}_mean"], num_row[f"{col}_std"], dec)

        numeric_rows.append(num_row)
        formatted_rows.append(fmt_row)

    numeric_df = pd.DataFrame(numeric_rows)
    formatted_df = pd.DataFrame(formatted_rows)
    return numeric_df, formatted_df


def make_delta_training_style_table(
    results_df: pd.DataFrame,
    Delta_original: float,
    Delta_normalized_original: float,
    ntri: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = [
        {
            "% Missing": "Original",
            "Hyb-Adam-UM Delta_total": f"{Delta_original:.4f}",
            "Delta_normalized": f"{Delta_normalized_original:.6f}",
            "Delta_per_triangle": f"{Delta_original / ntri:.6f}" if ntri > 0 else "nan",
            "Delta_relative_to_original": "1.000000",
            "n_success": "-",
            "n_failed": "-",
        }
    ]

    for pct, group0 in results_df.groupby("pct_missing", sort=True):
        group = group0[group0["success"] == True]
        n_success = int(group0["success"].sum())
        n_failed = int(len(group0) - n_success)

        dt_m, dt_s = mean_std_nan(group["Delta_total_completed"].to_numpy(dtype=float))
        dn_m, dn_s = mean_std_nan(group["Delta_normalized_completed"].to_numpy(dtype=float))
        dpt_m, dpt_s = mean_std_nan(group["Delta_per_triangle_completed"].to_numpy(dtype=float))
        dr_m, dr_s = mean_std_nan(group["Delta_relative_to_original"].to_numpy(dtype=float))

        rows.append(
            {
                "% Missing": f"{int(pct)}%",
                "Hyb-Adam-UM Delta_total": fmt_pm(dt_m, dt_s, 4),
                "Delta_normalized": fmt_pm(dn_m, dn_s, 6),
                "Delta_per_triangle": fmt_pm(dpt_m, dpt_s, 6),
                "Delta_relative_to_original": fmt_pm(dr_m, dr_s, 6),
                "n_success": n_success,
                "n_failed": n_failed,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# --------------- HYPERPARAMETER SENSITIVITY ------------------
# ============================================================


def make_sensitivity_grid(base: OptimizerConfig) -> List[Tuple[str, str, OptimizerConfig]]:
    """
    Return named configurations for sensitivity analysis.

    The grid varies the three reviewer-mentioned groups:
    - learning-rate schedule;
    - clipping threshold;
    - finite-difference step size.
    """
    return [
        (
            "baseline",
            "two-way initialization, 2 restarts, default lr schedule, clip=5.0, h=5e-5",
            base,
        ),
        (
            "old_global_mean_init",
            "old initialization only: global mean, 1 restart",
            replace(base, init_strategy="global_mean", n_restarts=1),
        ),
        (
            "single_restart",
            "two-way initialization, one restart only",
            replace(base, n_restarts=1),
        ),
        (
            "lr_init_low",
            "lower initial learning rate: lr_init=0.02",
            replace(base, lr_init=0.02),
        ),
        (
            "lr_init_high",
            "higher initial learning rate: lr_init=0.08",
            replace(base, lr_init=0.08),
        ),
        (
            "schedule_fast_decay",
            "earlier LR milestones: 300, 1000, 2500",
            replace(base, lr_milestones=(300, 1000, 2500)),
        ),
        (
            "schedule_slow_decay",
            "later LR milestones: 1500, 3000, 5000",
            replace(base, lr_milestones=(1500, 3000, 5000)),
        ),
        (
            "clip_low",
            "stronger gradient clipping: clip=1.0",
            replace(base, clip_grad_norm=1.0),
        ),
        (
            "clip_high",
            "weaker gradient clipping: clip=10.0",
            replace(base, clip_grad_norm=10.0),
        ),
        (
            "clip_none",
            "no gradient clipping",
            replace(base, clip_grad_norm=None),
        ),
        (
            "fd_step_small",
            "smaller central-difference step: h=1e-5",
            replace(base, h_central_diff=1.0e-5),
        ),
        (
            "fd_step_large",
            "larger central-difference step: h=1e-4",
            replace(base, h_central_diff=1.0e-4),
        ),
    ]



def summarize_sensitivity(sens_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for variant, group0 in sens_df.groupby("sensitivity_variant", sort=False):
        success_mask = group0["success"].astype(bool)
        group = group0[success_mask]
        n_success = int(success_mask.sum())
        n_failed = int(len(group0) - n_success)
        n_objective_improved = int(group0.get("objective_improved", pd.Series(False, index=group0.index)).astype(bool).sum())

        row: Dict[str, object] = {
            "sensitivity_variant": variant,
            "variant_description": group0["variant_description"].iloc[0],
            "n_runs": int(len(group0)),
            "n_success": n_success,
            "n_failed": n_failed,
            "n_objective_improved": n_objective_improved,
            "n_not_improved": int(len(group0) - n_objective_improved),
            "lr_init": group0["lr_init"].iloc[0],
            "lr_milestones": group0["lr_milestones"].iloc[0],
            "clip_grad_norm": group0["clip_grad_norm"].iloc[0],
            "h_central_diff": group0["h_central_diff"].iloc[0],
            "init_strategy": group0["init_strategy"].iloc[0] if "init_strategy" in group0.columns else "",
            "n_restarts": group0["n_restarts"].iloc[0] if "n_restarts" in group0.columns else np.nan,
        }

        for col in (
            "RMSE_miss",
            "MAE_miss",
            "Pearson_miss",
            "Spearman_miss",
            "runtime_seconds",
            "Delta_init",
            "Delta_final",
            "Delta_reduction_percent",
            "best_epoch",
            "epochs_used",
            "convergence_epoch",
            "final_learning_rate",
            "max_abs_error_observed",
            "mean_abs_error_observed",
        ):
            m, ss = mean_std_nan(group[col].to_numpy(dtype=float)) if col in group.columns else (float("nan"), float("nan"))
            row[f"{col}_mean"] = m
            row[f"{col}_std"] = ss
            row[col] = fmt_pm(m, ss, 6 if col not in {"runtime_seconds", "Delta_init", "Delta_final"} else 4)

        rows.append(row)
    return pd.DataFrame(rows)


def write_sensitivity_analysis(summary_df: pd.DataFrame, path: str) -> None:
    """Write a small data-driven Markdown analysis from the sensitivity summary."""
    lines: List[str] = []
    lines.append("# Hyperparameter sensitivity analysis\n")
    lines.append(
        "The sensitivity experiment reuses frozen masks so that differences between "
        "rows are attributable to optimizer hyperparameters rather than to different "
        "missing-entry patterns. Metrics RMSE_miss, MAE_miss, Pearson_miss, and "
        "Spearman_miss are computed only on artificially hidden entries.\n"
    )

    if summary_df.empty:
        lines.append("No successful sensitivity runs were available.\n")
    else:
        success_df = summary_df[summary_df["n_success"] > 0].copy()
        if success_df.empty:
            lines.append("All sensitivity variants failed; inspect the detailed CSV for errors.\n")
        else:
            best_rmse = success_df.loc[success_df["RMSE_miss_mean"].idxmin()]
            best_mae = success_df.loc[success_df["MAE_miss_mean"].idxmin()]
            best_delta = success_df.loc[success_df["Delta_final_mean"].idxmin()]
            fastest = success_df.loc[success_df["runtime_seconds_mean"].idxmin()]

            lines.append("## Automatic summary\n")
            lines.append(
                f"- Best mean RMSE_miss: **{best_rmse['sensitivity_variant']}** "
                f"({best_rmse['RMSE_miss_mean']:.6g}).\n"
            )
            lines.append(
                f"- Best mean MAE_miss: **{best_mae['sensitivity_variant']}** "
                f"({best_mae['MAE_miss_mean']:.6g}).\n"
            )
            lines.append(
                f"- Lowest mean final Delta: **{best_delta['sensitivity_variant']}** "
                f"({best_delta['Delta_final_mean']:.6g}).\n"
            )
            lines.append(
                f"- Fastest mean runtime: **{fastest['sensitivity_variant']}** "
                f"({fastest['runtime_seconds_mean']:.6g} s).\n"
            )

            baseline_rows = success_df[success_df["sensitivity_variant"] == "baseline"]
            if len(baseline_rows) == 1:
                base = baseline_rows.iloc[0]
                lines.append("\n## Relative changes versus baseline\n")
                for _, row in success_df.iterrows():
                    if row["sensitivity_variant"] == "baseline":
                        continue
                    def rel_change(col: str) -> float:
                        base_val = float(base[f"{col}_mean"])
                        val = float(row[f"{col}_mean"])
                        if not np.isfinite(base_val) or abs(base_val) <= EPS_NUM:
                            return float("nan")
                        return 100.0 * (val - base_val) / base_val

                    rmse_change = rel_change("RMSE_miss")
                    delta_change = rel_change("Delta_final")
                    time_change = rel_change("runtime_seconds")
                    lines.append(
                        f"- **{row['sensitivity_variant']}**: "
                        f"RMSE_miss {rmse_change:+.2f}%, "
                        f"Delta_final {delta_change:+.2f}%, "
                        f"runtime {time_change:+.2f}% relative to baseline.\n"
                    )

            lines.append("\n## Interpretation for the manuscript/rebuttal\n")
            lines.append(
                "If the RMSE_miss and Delta_final columns remain close across the tested "
                "learning-rate schedules, clipping thresholds, and finite-difference steps, "
                "then the method can be described as numerically stable within this local "
                "hyperparameter range. If one group changes substantially, report that group "
                "as the main sensitivity source and use the baseline as a fixed protocol for "
                "the main comparison. The observed-entry sanity columns should remain near "
                "machine precision; otherwise the completion routine is modifying entries "
                "that were not masked.\n"
            )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved: {path}")



def run_hyperparameter_sensitivity(
    mask_registry: List[Dict[str, object]],
    D_reference: np.ndarray,
    labels: List[str],
    trip_all: np.ndarray,
    ntri: int,
    Delta_original: float,
    Delta_normalized_original: float,
    base_opt: OptimizerConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== Running hyperparameter sensitivity analysis ===")
    variants = make_sensitivity_grid(base_opt)

    selected_reps = set(range(1, SENSITIVITY_REPS + 1))
    selected_masks = [
        rec
        for rec in mask_registry
        if abs(float(rec["frac_requested"]) - SENSITIVITY_MISSING_FRAC) < 1.0e-12
        and int(rec["replicate"]) in selected_reps
    ]

    if not selected_masks:
        print("No masks selected for sensitivity analysis.")
        return pd.DataFrame(), pd.DataFrame()

    sensitivity_rows: List[Dict[str, object]] = []
    for variant_name, description, opt in variants:
        print(f"\n--- Sensitivity variant: {variant_name} ({description}) ---")
        for rec in selected_masks:
            pct = int(rec["pct_missing"])
            rep = int(rec["replicate"])
            print(f"  p{pct} rep{rep:02d}, seed={rec['mask_seed']}")
            row = run_single_mask(
                rec=rec,
                D_reference=D_reference,
                labels=labels,
                trip_all=trip_all,
                ntri=ntri,
                Delta_original=Delta_original,
                Delta_normalized_original=Delta_normalized_original,
                opt=opt,
                method_label="Hyb-Adam-UM",
                save_completed=False,
                completed_prefix=f"Sensitivity_{variant_name}",
                verbose=False,
            )
            row.update(
                {
                    "sensitivity_variant": variant_name,
                    "variant_description": description,
                    "lr_init": opt.lr_init,
                    "lr_milestones": str(tuple(opt.lr_milestones)),
                    "clip_grad_norm": opt.clip_grad_norm,
                    "h_central_diff": opt.h_central_diff,
                    "init_strategy": opt.init_strategy,
                    "n_restarts": opt.n_restarts,
                }
            )
            sensitivity_rows.append(row)

    sens_df = pd.DataFrame(sensitivity_rows)
    sens_summary_df = summarize_sensitivity(sens_df)

    detailed_path = os.path.join(SENSITIVITY_DIR, "hyb_adam_um_hyperparam_sensitivity_detailed.csv")
    summary_path = os.path.join(SENSITIVITY_DIR, "hyb_adam_um_hyperparam_sensitivity_summary.csv")
    analysis_path = os.path.join(SENSITIVITY_DIR, "hyb_adam_um_hyperparam_sensitivity_analysis.md")

    save_csv(sens_df, detailed_path)
    save_csv(sens_summary_df, summary_path)
    write_sensitivity_analysis(sens_summary_df, analysis_path)

    display_or_print(sens_summary_df, "Hyperparameter sensitivity summary")
    return sens_df, sens_summary_df


# ============================================================
# ---------------------------- MAIN ---------------------------
# ============================================================


def main() -> None:
    ensure_dirs()

    D_orig, labels, used_orig_path = load_matrix_with_candidates(ORIG_CANDIDATES)
    D0 = sanitize_reference_distance_matrix(D_orig, "D_reference")

    n = D0.shape[0]
    if len(labels) != n:
        labels = [f"T{i + 1}" for i in range(n)]

    trip_all = np.array(list(combinations(range(n), 3)), dtype=np.int32)
    ntri = n * (n - 1) * (n - 2) // 6

    print("=== Reference matrix ===")
    print(f"File: {used_orig_path}")
    print(f"n taxa/items: {n}")
    print(f"n lower-triangle pairs: {n * (n - 1) // 2}")
    print(f"n triplets: {ntri}")

    Delta_original = robust_delta_sum_numpy(D0, trip_all)
    Delta_normalized_original = compute_normalized_delta(D0, trip_all)
    print(f"Original Delta_total = {Delta_original:.6f}")
    print(f"Original Delta_normalized = {Delta_normalized_original:.6f}")
    print(f"Original Delta_per_triangle = {Delta_original / ntri:.6f}" if ntri > 0 else "Original Delta_per_triangle = nan")

    # Save clean reference matrix for reproducibility.
    pd.DataFrame(D0, index=labels, columns=labels).to_csv(os.path.join(OUTPUT_ROOT, "reference_matrix_sanitized.csv"))

    # Build and freeze all masks.
    print("\n=== Building frozen masks ===")
    mask_registry = build_mask_registry(D0)
    save_masked_matrices(mask_registry, labels)
    mask_meta_df = pd.DataFrame(
        [
            {
                "pct_missing": rec["pct_missing"],
                "frac_missing_requested": rec["frac_requested"],
                "replicate": rec["replicate"],
                "mask_seed": rec["mask_seed"],
                "missingness_actual": rec["missingness_actual"],
                "n_pairs_total": rec["n_pairs_total"],
                "n_missing": rec["n_missing"],
                "n_observed": rec["n_observed"],
                "optimized_variables": rec["n_missing"],
                "masked_file": rec.get("masked_file", ""),
            }
            for rec in mask_registry
        ]
    )
    save_csv(mask_meta_df, os.path.join(TABLES_DIR, "mask_registry.csv"))

    print("\n=== Running Hyb-Adam-UM on all masks ===")
    results: List[Dict[str, object]] = []
    for rec in mask_registry:
        pct = int(rec["pct_missing"])
        rep = int(rec["replicate"])
        print(f"\nProcessing {pct}% missing, replicate {rep:02d}, seed={rec['mask_seed']}...")
        row = run_single_mask(
            rec=rec,
            D_reference=D0,
            labels=labels,
            trip_all=trip_all,
            ntri=ntri,
            Delta_original=Delta_original,
            Delta_normalized_original=Delta_normalized_original,
            opt=BASE_OPT_CONFIG,
            method_label="Hyb-Adam-UM",
            save_completed=True,
            completed_prefix="HybAdamUM_completed",
            verbose=True,
        )
        results.append(row)

    results_df = pd.DataFrame(results)
    detailed_path = os.path.join(TABLES_DIR, "hyb_adam_um_all_masks_detailed.csv")
    save_csv(results_df, detailed_path)
    display_or_print(results_df, "Hyb-Adam-UM detailed results")

    summary_numeric_df, summary_formatted_df = summarize_by_missingness(results_df)
    save_csv(summary_numeric_df, os.path.join(TABLES_DIR, "hyb_adam_um_summary_numeric_by_missingness.csv"))
    save_csv(summary_formatted_df, os.path.join(TABLES_DIR, "hyb_adam_um_summary_formatted_by_missingness.csv"))
    display_or_print(summary_formatted_df, "Hyb-Adam-UM summary by missingness")

    delta_table_df = make_delta_training_style_table(
        results_df=results_df,
        Delta_original=Delta_original,
        Delta_normalized_original=Delta_normalized_original,
        ntri=ntri,
    )
    save_csv(delta_table_df, os.path.join(TABLES_DIR, "hyb_adam_um_delta_sum_training_style.csv"))
    display_or_print(delta_table_df, "Delta sum table")

    # Separate hyperparameter-sensitivity table and analysis.
    if RUN_HYPERPARAM_SENSITIVITY:
        run_hyperparameter_sensitivity(
            mask_registry=mask_registry,
            D_reference=D0,
            labels=labels,
            trip_all=trip_all,
            ntri=ntri,
            Delta_original=Delta_original,
            Delta_normalized_original=Delta_normalized_original,
            base_opt=BASE_OPT_CONFIG,
        )

    # Archive everything for convenience.
    archive_base = OUTPUT_ROOT.rstrip(os.sep)
    try:
        zip_path = shutil.make_archive(archive_base, "zip", OUTPUT_ROOT)
        print(f"\nSaved ZIP archive: {zip_path}")
    except Exception as e:
        print(f"Could not create ZIP archive: {repr(e)}")

    print("\n=== ALL PROCESSING COMPLETE ===")
    print(f"Reference matrix used: {used_orig_path}")
    print(f"Detailed results: {detailed_path}")
    print(f"Formatted summary: {os.path.join(TABLES_DIR, 'hyb_adam_um_summary_formatted_by_missingness.csv')}")
    print(f"Numeric summary: {os.path.join(TABLES_DIR, 'hyb_adam_um_summary_numeric_by_missingness.csv')}")
    print(f"Delta table: {os.path.join(TABLES_DIR, 'hyb_adam_um_delta_sum_training_style.csv')}")
    if RUN_HYPERPARAM_SENSITIVITY:
        print(f"Sensitivity folder: {SENSITIVITY_DIR}")


if __name__ == "__main__":
    main()
