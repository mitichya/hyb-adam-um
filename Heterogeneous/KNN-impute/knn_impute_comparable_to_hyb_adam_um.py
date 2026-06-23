# -*- coding: utf-8 -*-
"""
KNN-impute baseline for distance-matrix completion
==================================================

This script follows the same reviewer-oriented comparison protocol used for
Hyb-Adam-UM / MW* / LRMC / NJ* experiments.

Protocol implemented
--------------------
1. Uses the reference matrix:
       Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv
2. Missingness levels:
       30%, 50%, 65%, 85%
3. Replicates:
       30 per missingness level
4. Metrics are computed ONLY on artificially hidden lower-triangle entries:
       RMSE_miss, MAE_miss, Pearson_miss, Spearman_miss
5. Runtime is measured for every run and summarized as mean ± std.
6. Detailed rows include:
       mask_seed, missingness_actual,
       n_missing, n_observed, optimized_variables, imputed_variables,
       n_success, n_failed,
       max_abs_error_observed, mean_abs_error_observed
7. Observed entries are preserved exactly in the completed output matrix.

Method note
-----------
KNN-impute is used here as a generic famous imputation baseline. Each row of the
distance matrix is treated as a distance-profile vector. Missing values are
encoded as NaN and imputed using sklearn.impute.KNNImputer with nan_euclidean
distance. The output is symmetrized and observed entries are re-imposed exactly.

KNN-impute is not an optimizer in the same sense as Hyb-Adam-UM, so:
       optimized_variables = 0
       imputed_variables   = n_missing
       epochs_used         = 0
       convergence_epoch   = 0
"""

from __future__ import annotations

import os
import time
import json
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# -------------------------- CONFIG ---------------------------
# ============================================================

ORIG_CANDIDATES: Tuple[str, ...] = (
    "Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv",
    "/mnt/data/Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv",
)

MISSING_FRACS: Tuple[float, ...] = (0.30, 0.50, 0.65, 0.85)
REPS: int = 30
BASE_SEED: int = 55

MISSING_VAL: float = -1.0

OUTPUT_ROOT: str = "knn_impute_outputs"
MASKED_DIR: str = os.path.join(OUTPUT_ROOT, "masked_matrices")
COMPLETED_DIR: str = os.path.join(OUTPUT_ROOT, "completed_matrices")
TABLES_DIR: str = os.path.join(OUTPUT_ROOT, "tables")

METHOD_LABEL: str = "KNN-impute"

# KNN-impute hyperparameters.
# n_neighbors=5 is the classical/default choice in many KNN-imputation settings.
KNN_N_NEIGHBORS: int = 5
KNN_WEIGHTS: str = "distance"       # "uniform" or "distance"
KNN_METRIC: str = "nan_euclidean"   # sklearn KNNImputer standard metric


# ============================================================
# -------------------------- I/O ------------------------------
# ============================================================

def ensure_dirs() -> None:
    for d in (OUTPUT_ROOT, MASKED_DIR, COMPLETED_DIR, TABLES_DIR):
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


def load_distance_matrix_from_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load a square distance matrix from a CSV file.

    Supported formats:
    - row labels in first column and column labels in header;
    - pure numeric square matrix;
    - delimiter inferred by pandas, including comma/semicolon/tab.
    """

    # Case 1: row labels + header.
    try:
        df = pd.read_csv(path, sep=None, engine="python", index_col=0)
        num = df.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if num.shape[0] == num.shape[1] and num.shape[0] > 1:
            labels = [str(x) for x in num.index.to_list()]
            return num.to_numpy(dtype=float), labels
    except Exception:
        pass

    # Case 2: pure numeric table.
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        num = df.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if num.shape[0] == num.shape[1] and num.shape[0] > 1:
            labels = [f"T{i + 1}" for i in range(num.shape[0])]
            return num.to_numpy(dtype=float), labels
    except Exception:
        pass

    # Fallback.
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
            # Legacy guard for old mtDNA matrices stored in 10^-3 units.
            if np.nanmax(M) > 500:
                M = M / 1000.0
            return M, labels, p
    raise FileNotFoundError(f"Reference matrix not found. Tried: {list(candidates)}")


# ============================================================
# --------------------- MATRIX HELPERS ------------------------
# ============================================================

def symmetrize_full(D: np.ndarray) -> np.ndarray:
    M = 0.5 * (np.asarray(D, dtype=float) + np.asarray(D, dtype=float).T)
    np.fill_diagonal(M, 0.0)
    return M


def symmetrize_with_missing(D: np.ndarray) -> np.ndarray:
    """Symmetrize while preserving missing pairs encoded as MISSING_VAL."""
    M = np.asarray(D, dtype=float).copy()
    n = M.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = M[i, j], M[j, i]
            a_ok = a >= 0.0
            b_ok = b >= 0.0
            if a_ok and b_ok:
                v = 0.5 * (a + b)
            elif a_ok:
                v = a
            elif b_ok:
                v = b
            else:
                v = MISSING_VAL
            M[i, j] = M[j, i] = v
    np.fill_diagonal(M, 0.0)
    return M


def sanitize_reference_distance_matrix(D: np.ndarray, name: str = "D_ref") -> np.ndarray:
    """
    Strict validation and minimal normalization for a complete reference matrix.

    This function intentionally does not silently fill NaN/Inf/negative entries,
    because the reference matrix is the benchmark ground truth.
    """
    M = np.asarray(D, dtype=float).copy()
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be square, got shape {M.shape}.")

    n = M.shape[0]
    off = ~np.eye(n, dtype=bool)

    if np.any(~np.isfinite(M)):
        raise ValueError(f"{name} contains NaN/Inf values.")
    if np.any(M[off] < 0.0):
        raise ValueError(f"{name} contains negative off-diagonal distances.")

    M = 0.5 * (M + M.T)
    M = np.maximum(M, 0.0)
    np.fill_diagonal(M, 0.0)

    if not np.allclose(M, M.T, atol=1.0e-12):
        raise ValueError(f"{name} is not symmetric after sanitization.")
    if not np.allclose(np.diag(M), 0.0, atol=1.0e-12):
        raise ValueError(f"{name} diagonal is not zero after sanitization.")
    if not np.isfinite(M).all():
        raise ValueError(f"{name} contains non-finite values after sanitization.")

    return M


def finalize_completed_matrix(
    D_hat: np.ndarray,
    D_incomplete: np.ndarray,
    observed_pairs: np.ndarray,
    preserve_observed: bool = True,
) -> np.ndarray:
    """
    Symmetrize and finalize completed matrix while preserving observed entries exactly.
    """
    M = np.asarray(D_hat, dtype=float).copy()

    if not np.isfinite(M).all():
        raise FloatingPointError("Completed matrix contains non-finite values before finalization.")

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
        raise FloatingPointError("Completed matrix contains non-finite values after finalization.")
    return M


def lower_pairs(n: int) -> np.ndarray:
    i, j = np.tril_indices(n, k=-1)
    return np.column_stack([i, j]).astype(np.int32)


def values_on_pairs(M: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.array([], dtype=float)
    return np.asarray(M[pairs[:, 0], pairs[:, 1]], dtype=float)


def is_valid_completed_distance_matrix(D_completed: np.ndarray, D_reference: np.ndarray) -> bool:
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
    if len(a) < 2 or np.std(a) <= 0.0 or np.std(b) <= 0.0:
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
# ----------------------- KNN-IMPUTE CORE ---------------------
# ============================================================

def fallback_fill_remaining_nan(X: np.ndarray, obs_nan: np.ndarray) -> np.ndarray:
    """
    Rare fallback if KNNImputer leaves NaN values.
    Fill remaining NaNs by column median, then global median, then zero.
    """
    Y = np.asarray(X, dtype=float).copy()
    if np.isfinite(Y).all():
        return Y

    col_medians = np.nanmedian(np.where(obs_nan, Y, np.nan), axis=0)
    global_med = np.nanmedian(Y[np.isfinite(Y)])
    if not np.isfinite(global_med):
        global_med = 0.0

    rows, cols = np.where(~np.isfinite(Y))
    for i, j in zip(rows, cols):
        v = col_medians[j] if np.isfinite(col_medians[j]) else global_med
        Y[i, j] = v
    return Y


def knn_impute_completion(
    D_in: np.ndarray,
    n_neighbors: int = KNN_N_NEIGHBORS,
    weights: str = KNN_WEIGHTS,
    metric: str = KNN_METRIC,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Complete a distance matrix using sklearn KNNImputer.

    Missing entries encoded as MISSING_VAL are converted to NaN. Rows are treated
    as distance-profile vectors. The output is returned before observed-entry
    reimposition; finalization is handled outside.
    """
    try:
        from sklearn.impute import KNNImputer
    except Exception as e:
        raise ImportError(
            "KNN-impute baseline requires scikit-learn. Install it with: pip install scikit-learn"
        ) from e

    n = D_in.shape[0]
    X = np.asarray(D_in, dtype=float).copy()
    X[X < 0.0] = np.nan
    np.fill_diagonal(X, 0.0)

    # Do not allow more neighbors than available rows.
    k_eff = int(max(1, min(n_neighbors, max(1, n - 1))))

    imputer = KNNImputer(
        n_neighbors=k_eff,
        weights=weights,
        metric=metric,
        copy=True,
    )

    X_imp = imputer.fit_transform(X)

    # Fallback guard for extremely sparse cases.
    X_imp = fallback_fill_remaining_nan(X_imp, obs_nan=np.isfinite(X))

    X_imp = 0.5 * (X_imp + X_imp.T)
    X_imp = np.maximum(X_imp, 0.0)
    np.fill_diagonal(X_imp, 0.0)

    info = {
        "success": True,
        "n_neighbors_requested": int(n_neighbors),
        "n_neighbors_effective": int(k_eff),
        "weights": str(weights),
        "metric": str(metric),
        "epochs_used": 0,
        "convergence_epoch": 0,
    }
    return X_imp, info


# ============================================================
# ----------------------- SINGLE RUNNER -----------------------
# ============================================================

def run_single_mask(
    rec: Dict[str, object],
    D_reference: np.ndarray,
    labels: List[str],
    method_label: str = METHOD_LABEL,
    save_completed: bool = True,
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
        # KNN-impute is not a continuous optimizer.
        "optimized_variables": 0,
        "imputed_variables": int(rec["n_missing"]),
        "success": False,
        "success_numerical": False,
        "success_valid_matrix": False,
        "error_message": "",
        "masked_file": rec.get("masked_file", ""),
        "completed_file": "",
        "n_neighbors_requested": int(KNN_N_NEIGHBORS),
        "n_neighbors_effective": int(min(KNN_N_NEIGHBORS, max(1, D_reference.shape[0] - 1))),
        "knn_weights": str(KNN_WEIGHTS),
        "knn_metric": str(KNN_METRIC),
        "epochs_used": 0,
        "convergence_epoch": 0,
        "optimizer_config_json": json.dumps(
            {
                "n_neighbors": KNN_N_NEIGHBORS,
                "weights": KNN_WEIGHTS,
                "metric": KNN_METRIC,
            },
            ensure_ascii=False,
        ),
    }

    t0 = time.perf_counter()
    try:
        D_completed_raw, info = knn_impute_completion(
            D_inc,
            n_neighbors=KNN_N_NEIGHBORS,
            weights=KNN_WEIGHTS,
            metric=KNN_METRIC,
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

        # Hard observed-entry sanity check.
        if np.isfinite(sanity["max_abs_error_observed"]) and sanity["max_abs_error_observed"] > 1.0e-10:
            raise RuntimeError(
                "Observed entries were modified: "
                f"max_abs_error_observed={sanity['max_abs_error_observed']:.6g}"
            )

        success_numerical = bool(
            np.isfinite(runtime_seconds)
            and np.isfinite(D_completed).all()
            and bool(info.get("success", True))
        )
        success_valid_matrix = bool(is_valid_completed_distance_matrix(D_completed, D_reference))
        success = bool(success_numerical and success_valid_matrix)

        completed_file = ""
        if save_completed:
            completed_file = f"KNNimpute_completed_p{pct}_rep{rep:02d}_seed{row['mask_seed']}.csv"
            completed_path = os.path.join(COMPLETED_DIR, completed_file)
            pd.DataFrame(D_completed, index=labels, columns=labels).to_csv(completed_path)

        row.update(
            {
                "success": success,
                "success_numerical": success_numerical,
                "success_valid_matrix": success_valid_matrix,
                "runtime_seconds": float(runtime_seconds),
                "completed_file": completed_file,
                **miss_metrics,
                **sanity,
                "n_neighbors_requested": int(info.get("n_neighbors_requested", KNN_N_NEIGHBORS)),
                "n_neighbors_effective": int(info.get("n_neighbors_effective", KNN_N_NEIGHBORS)),
                "knn_weights": str(info.get("weights", KNN_WEIGHTS)),
                "knn_metric": str(info.get("metric", KNN_METRIC)),
                "epochs_used": int(info.get("epochs_used", 0)),
                "convergence_epoch": int(info.get("convergence_epoch", 0)),
            }
        )
    except Exception as e:
        runtime_seconds = time.perf_counter() - t0
        row.update(
            {
                "success": False,
                "success_numerical": False,
                "success_valid_matrix": False,
                "runtime_seconds": float(runtime_seconds),
                "error_message": repr(e),
                "RMSE_miss": float("nan"),
                "MAE_miss": float("nan"),
                "Pearson_miss": float("nan"),
                "Spearman_miss": float("nan"),
                "max_abs_error_observed": float("nan"),
                "mean_abs_error_observed": float("nan"),
            }
        )
        print(f"FAILED p{pct} rep{rep}: {repr(e)}")

    return row


# ============================================================
# -------------------------- SUMMARIES ------------------------
# ============================================================

SUMMARY_NUMERIC_COLUMNS: Tuple[str, ...] = (
    "missingness_actual",
    "n_missing",
    "n_observed",
    "optimized_variables",
    "imputed_variables",
    "RMSE_miss",
    "MAE_miss",
    "Pearson_miss",
    "Spearman_miss",
    "runtime_seconds",
    "max_abs_error_observed",
    "mean_abs_error_observed",
    "n_neighbors_effective",
    "epochs_used",
    "convergence_epoch",
)


def summarize_by_missingness(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows: List[Dict[str, object]] = []
    formatted_rows: List[Dict[str, object]] = []

    for pct in sorted(results_df["pct_missing"].unique()):
        sub_all = results_df[results_df["pct_missing"] == pct].copy()
        sub_ok = sub_all[sub_all["success"] == True].copy()

        n_total = int(len(sub_all))
        n_success = int(len(sub_ok))
        n_failed = int(n_total - n_success)

        num_row: Dict[str, object] = {
            "method": METHOD_LABEL,
            "pct_missing": int(pct),
            "n_total": n_total,
            "n_success": n_success,
            "n_failed": n_failed,
        }
        fmt_row: Dict[str, object] = {
            "method": METHOD_LABEL,
            "% Missing": f"{int(pct)}%",
            "n_total": n_total,
            "n_success": n_success,
            "n_failed": n_failed,
        }

        for col in SUMMARY_NUMERIC_COLUMNS:
            if col not in sub_ok.columns:
                continue
            mean_val, std_val = mean_std_nan(sub_ok[col].to_numpy(dtype=float))
            num_row[f"{col}_mean"] = mean_val
            num_row[f"{col}_std"] = std_val

        fmt_row["missingness_actual"] = fmt_pm(
            num_row.get("missingness_actual_mean", np.nan),
            num_row.get("missingness_actual_std", np.nan),
            decimals=6,
        )
        fmt_row["n_missing"] = fmt_pm(
            num_row.get("n_missing_mean", np.nan),
            num_row.get("n_missing_std", np.nan),
            decimals=2,
        )
        fmt_row["n_observed"] = fmt_pm(
            num_row.get("n_observed_mean", np.nan),
            num_row.get("n_observed_std", np.nan),
            decimals=2,
        )
        fmt_row["optimized_variables"] = fmt_pm(
            num_row.get("optimized_variables_mean", np.nan),
            num_row.get("optimized_variables_std", np.nan),
            decimals=2,
        )
        fmt_row["imputed_variables"] = fmt_pm(
            num_row.get("imputed_variables_mean", np.nan),
            num_row.get("imputed_variables_std", np.nan),
            decimals=2,
        )
        fmt_row["RMSE_miss"] = fmt_pm(
            num_row.get("RMSE_miss_mean", np.nan),
            num_row.get("RMSE_miss_std", np.nan),
            decimals=6,
        )
        fmt_row["MAE_miss"] = fmt_pm(
            num_row.get("MAE_miss_mean", np.nan),
            num_row.get("MAE_miss_std", np.nan),
            decimals=6,
        )
        fmt_row["Pearson_miss"] = fmt_pm(
            num_row.get("Pearson_miss_mean", np.nan),
            num_row.get("Pearson_miss_std", np.nan),
            decimals=6,
        )
        fmt_row["Spearman_miss"] = fmt_pm(
            num_row.get("Spearman_miss_mean", np.nan),
            num_row.get("Spearman_miss_std", np.nan),
            decimals=6,
        )
        fmt_row["runtime_seconds"] = fmt_pm(
            num_row.get("runtime_seconds_mean", np.nan),
            num_row.get("runtime_seconds_std", np.nan),
            decimals=4,
        )
        fmt_row["max_abs_error_observed"] = fmt_pm(
            num_row.get("max_abs_error_observed_mean", np.nan),
            num_row.get("max_abs_error_observed_std", np.nan),
            decimals=6,
        )
        fmt_row["mean_abs_error_observed"] = fmt_pm(
            num_row.get("mean_abs_error_observed_mean", np.nan),
            num_row.get("mean_abs_error_observed_std", np.nan),
            decimals=6,
        )
        fmt_row["n_neighbors_effective"] = fmt_pm(
            num_row.get("n_neighbors_effective_mean", np.nan),
            num_row.get("n_neighbors_effective_std", np.nan),
            decimals=2,
        )
        fmt_row["epochs_used"] = fmt_pm(
            num_row.get("epochs_used_mean", np.nan),
            num_row.get("epochs_used_std", np.nan),
            decimals=2,
        )
        fmt_row["convergence_epoch"] = fmt_pm(
            num_row.get("convergence_epoch_mean", np.nan),
            num_row.get("convergence_epoch_std", np.nan),
            decimals=2,
        )

        numeric_rows.append(num_row)
        formatted_rows.append(fmt_row)

    return pd.DataFrame(numeric_rows), pd.DataFrame(formatted_rows)


# ============================================================
# ---------------------------- MAIN ---------------------------
# ============================================================

def main() -> None:
    ensure_dirs()

    D_raw, labels, used_path = load_matrix_with_candidates(ORIG_CANDIDATES)
    D0 = sanitize_reference_distance_matrix(symmetrize_full(D_raw), "D_ref")
    n = D0.shape[0]

    # If labels are missing/misaligned, repair them.
    if len(labels) != n:
        labels = [f"T{i + 1}" for i in range(n)]

    print("=== KNN-IMPUTE BASELINE ===")
    print(f"Reference matrix: {used_path}")
    print(f"n = {n}")
    print(f"missingness levels = {MISSING_FRACS}")
    print(f"replicates per level = {REPS}")
    print(f"KNN n_neighbors = {KNN_N_NEIGHBORS}, weights = {KNN_WEIGHTS}, metric = {KNN_METRIC}")
    print()

    mask_registry = build_mask_registry(D0)
    save_masked_matrices(mask_registry, labels)

    results: List[Dict[str, object]] = []
    for rec in mask_registry:
        pct = int(rec["pct_missing"])
        rep = int(rec["replicate"])
        print(f"Processing {pct}% missing, replicate {rep:02d}, seed {rec['mask_seed']}...")
        row = run_single_mask(
            rec=rec,
            D_reference=D0,
            labels=labels,
            method_label=METHOD_LABEL,
            save_completed=True,
        )
        results.append(row)

    results_df = pd.DataFrame(results)

    # Add per-level n_success/n_failed to every detailed row as requested.
    results_df["n_success"] = 0
    results_df["n_failed"] = 0
    for pct in sorted(results_df["pct_missing"].unique()):
        idx = results_df["pct_missing"] == pct
        n_success = int(results_df.loc[idx, "success"].sum())
        n_failed = int(idx.sum() - n_success)
        results_df.loc[idx, "n_success"] = n_success
        results_df.loc[idx, "n_failed"] = n_failed

    detailed_path = os.path.join(TABLES_DIR, "knn_impute_all_masks_detailed.csv")
    save_csv(results_df, detailed_path)

    summary_numeric, summary_formatted = summarize_by_missingness(results_df)

    numeric_path = os.path.join(TABLES_DIR, "knn_impute_summary_numeric_by_missingness.csv")
    formatted_path = os.path.join(TABLES_DIR, "knn_impute_summary_formatted_by_missingness.csv")

    save_csv(summary_numeric, numeric_path)
    save_csv(summary_formatted, formatted_path)

    display_or_print(results_df, "KNN-impute detailed results")
    display_or_print(summary_formatted, "KNN-impute summary, mean ± std over successful replicates")

    print("\n=== DONE ===")
    print(f"Masked matrices:    {MASKED_DIR}/")
    print(f"Completed matrices: {COMPLETED_DIR}/")
    print(f"Detailed results:   {detailed_path}")
    print(f"Summary numeric:    {numeric_path}")
    print(f"Summary formatted:  {formatted_path}")


if __name__ == "__main__":
    main()
