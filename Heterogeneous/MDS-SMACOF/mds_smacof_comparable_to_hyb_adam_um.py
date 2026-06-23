# -*- coding: utf-8 -*-
"""
MDS-SMACOF baseline for distance-matrix completion
==================================================

This script follows the same reviewer-oriented comparison protocol used for
Hyb-Adam-UM / MW* / LRMC / NJ* / KNN-impute experiments.

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
This is a weighted/incomplete MDS-SMACOF completion baseline.

Given observed distances D_ij on Omega_obs, it solves approximately

    min_X sum_{(i,j) in Omega_obs} ( ||x_i - x_j|| - D_ij )^2,

using the SMACOF majorization update with weights W_ij = 1 for observed
off-diagonal pairs and W_ij = 0 for missing pairs. Missing distances are then
filled by Euclidean distances ||x_i - x_j||.

Observed entries are re-imposed exactly after completion, so sanity-check errors
on observed entries should be zero up to numerical precision.

Important:
MDS-SMACOF is a distance-geometry baseline, not a phylogenetic/tree-metric
method. It tests whether a Euclidean embedding stress model is sufficient for
the completion task.
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

OUTPUT_ROOT: str = "mds_smacof_outputs"
MASKED_DIR: str = os.path.join(OUTPUT_ROOT, "masked_matrices")
COMPLETED_DIR: str = os.path.join(OUTPUT_ROOT, "completed_matrices")
TABLES_DIR: str = os.path.join(OUTPUT_ROOT, "tables")

METHOD_LABEL: str = "MDS-SMACOF"

# MDS-SMACOF hyperparameters.
# For distance completion, a moderate dimension is usually more useful than 2D
# visualization. FULL_DIM=True uses dim=n-1. Otherwise MDS_DIM is used.
FULL_DIM: bool = True
MDS_DIM: int = 3

MAX_ITER: int = 1000
TOL: float = 1.0e-7
EPS_DIST: float = 1.0e-12

INIT_METHOD: str = "classical_meanfill"  # "classical_meanfill" or "random"
RANDOM_INIT_SCALE: float = 1.0e-2


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


def pairwise_euclidean_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances without requiring scipy/sklearn."""
    G = X @ X.T
    sq = np.diag(G)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2)
    np.fill_diagonal(D, 0.0)
    return D


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
# ----------------------- MDS-SMACOF CORE ---------------------
# ============================================================

def mean_fill_missing_distances(D_in: np.ndarray) -> np.ndarray:
    """Create a complete initialization matrix by mean-filling missing entries."""
    D = np.asarray(D_in, dtype=float).copy()
    n = D.shape[0]
    off = ~np.eye(n, dtype=bool)

    known = (D >= 0.0) & off
    if np.any(known):
        fill = float(np.mean(D[known]))
    else:
        fill = 1.0

    D[D < 0.0] = fill
    D = 0.5 * (D + D.T)
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def classical_mds_initialization(D_complete: np.ndarray, dim: int) -> np.ndarray:
    """
    Classical MDS initialization from a complete dissimilarity matrix.
    """
    D = np.asarray(D_complete, dtype=float)
    n = D.shape[0]
    dim = int(max(1, min(dim, n - 1)))

    D2 = D * D
    J = np.eye(n) - np.ones((n, n), dtype=float) / n
    B = -0.5 * J @ D2 @ J
    B = 0.5 * (B + B.T)

    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    pos = np.maximum(evals[:dim], 0.0)
    X = evecs[:, :dim] * np.sqrt(pos)[None, :]

    # If all positive eigenvalues are zero, use a small deterministic fallback.
    if not np.isfinite(X).all() or np.linalg.norm(X) < EPS_DIST:
        rng = np.random.RandomState(0)
        X = RANDOM_INIT_SCALE * rng.randn(n, dim)
        X -= X.mean(axis=0, keepdims=True)

    return X


def smacof_stress(X: np.ndarray, Delta: np.ndarray, W: np.ndarray) -> float:
    """Raw weighted stress over upper-triangle pairs."""
    D_x = pairwise_euclidean_distances(X)
    iu, ju = np.triu_indices(X.shape[0], k=1)
    w = W[iu, ju]
    diff = D_x[iu, ju] - Delta[iu, ju]
    return float(np.sum(w * diff * diff))


def weighted_smacof_completion(
    D_in: np.ndarray,
    observed_pairs: np.ndarray,
    dim: int,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
    init_method: str = INIT_METHOD,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Weighted/incomplete SMACOF.

    Parameters
    ----------
    D_in:
        Incomplete distance matrix with missing entries encoded as MISSING_VAL.
    observed_pairs:
        Lower-triangle observed pairs.
    dim:
        Embedding dimension.

    Returns
    -------
    D_completed_raw:
        Euclidean distance matrix from the final embedding.
    info:
        Diagnostics.
    """
    Dm = symmetrize_with_missing(D_in)
    n = Dm.shape[0]
    dim_eff = int(max(1, min(dim, n - 1)))

    Delta = np.where(Dm >= 0.0, Dm, 0.0)
    Delta = 0.5 * (Delta + Delta.T)
    np.fill_diagonal(Delta, 0.0)

    W = np.zeros((n, n), dtype=float)
    if observed_pairs.size > 0:
        i, j = observed_pairs[:, 0], observed_pairs[:, 1]
        W[i, j] = 1.0
        W[j, i] = 1.0
    np.fill_diagonal(W, 0.0)

    # V = diag(W 1) - W is the graph Laplacian of the observed-distance graph.
    v_diag = np.sum(W, axis=1)
    V = np.diag(v_diag) - W

    # Moore-Penrose inverse handles the centering nullspace and possible
    # disconnected observed graph. Disconnected cases are allowed but may give
    # weak completions; failures are caught by validity checks.
    V_pinv = np.linalg.pinv(V, rcond=1.0e-12)

    if init_method == "classical_meanfill":
        D_init = mean_fill_missing_distances(Dm)
        X = classical_mds_initialization(D_init, dim_eff)
    elif init_method == "random":
        rng = np.random.RandomState(seed)
        X = RANDOM_INIT_SCALE * rng.randn(n, dim_eff)
        X -= X.mean(axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown INIT_METHOD: {init_method}")

    # Center initialization.
    X = X - X.mean(axis=0, keepdims=True)

    stress_prev = smacof_stress(X, Delta, W)
    stress_init = float(stress_prev)
    stress_best = float(stress_prev)
    X_best = X.copy()
    best_iter = 0
    convergence_iter = max_iter
    last_rel_change = float("nan")

    for it in range(1, max_iter + 1):
        D_x = pairwise_euclidean_distances(X)

        # B(X): offdiag b_ij = - w_ij * delta_ij / d_ij(X)
        ratio = np.zeros((n, n), dtype=float)
        mask = (W > 0.0) & (D_x > EPS_DIST)
        ratio[mask] = W[mask] * Delta[mask] / D_x[mask]

        B = -ratio
        np.fill_diagonal(B, -np.sum(B, axis=1))

        X_new = V_pinv @ (B @ X)
        X_new = X_new - X_new.mean(axis=0, keepdims=True)

        stress_new = smacof_stress(X_new, Delta, W)

        if stress_new < stress_best:
            stress_best = float(stress_new)
            X_best = X_new.copy()
            best_iter = int(it)

        denom = max(abs(stress_prev), EPS_DIST)
        last_rel_change = abs(stress_prev - stress_new) / denom

        X = X_new
        stress_prev = stress_new

        if last_rel_change < tol:
            convergence_iter = int(it)
            break

    D_completed_raw = pairwise_euclidean_distances(X_best)

    stress_reduction_percent = (
        100.0 * (stress_init - stress_best) / max(abs(stress_init), EPS_DIST)
        if np.isfinite(stress_init) else float("nan")
    )

    info = {
        "success": bool(np.isfinite(D_completed_raw).all()),
        "embedding_dim": int(dim_eff),
        "optimized_variables": int(n * dim_eff),
        "stress_init": float(stress_init),
        "stress_final": float(stress_best),
        "stress_reduction_percent": float(stress_reduction_percent),
        "best_iter": int(best_iter),
        "epochs_used": int(it if "it" in locals() else 0),
        "convergence_epoch": int(convergence_iter),
        "last_rel_change": float(last_rel_change),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "init_method": str(init_method),
    }
    return D_completed_raw, info


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

    n = D_reference.shape[0]
    dim = n - 1 if FULL_DIM else min(MDS_DIM, n - 1)

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
        "optimized_variables": int(n * dim),
        "imputed_variables": int(rec["n_missing"]),
        "success": False,
        "success_numerical": False,
        "success_valid_matrix": False,
        "error_message": "",
        "masked_file": rec.get("masked_file", ""),
        "completed_file": "",
        "embedding_dim": int(dim),
        "full_dim": bool(FULL_DIM),
        "max_iter": int(MAX_ITER),
        "tol": float(TOL),
        "init_method": str(INIT_METHOD),
        "optimizer_config_json": json.dumps(
            {
                "FULL_DIM": FULL_DIM,
                "MDS_DIM": MDS_DIM,
                "MAX_ITER": MAX_ITER,
                "TOL": TOL,
                "INIT_METHOD": INIT_METHOD,
                "RANDOM_INIT_SCALE": RANDOM_INIT_SCALE,
            },
            ensure_ascii=False,
        ),
    }

    t0 = time.perf_counter()
    try:
        D_completed_raw, info = weighted_smacof_completion(
            D_in=D_inc,
            observed_pairs=observed_pairs,
            dim=dim,
            max_iter=MAX_ITER,
            tol=TOL,
            init_method=INIT_METHOD,
            seed=10000 * pct + rep,
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
            and np.isfinite(info.get("stress_final", np.nan))
        )
        success_valid_matrix = bool(is_valid_completed_distance_matrix(D_completed, D_reference))
        success = bool(success_numerical and success_valid_matrix)

        completed_file = ""
        if save_completed:
            completed_file = f"MDS_SMACOF_completed_p{pct}_rep{rep:02d}_seed{row['mask_seed']}.csv"
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
                "embedding_dim": int(info.get("embedding_dim", dim)),
                "optimized_variables": int(info.get("optimized_variables", n * dim)),
                "stress_init": float(info.get("stress_init", np.nan)),
                "stress_final": float(info.get("stress_final", np.nan)),
                "stress_reduction_percent": float(info.get("stress_reduction_percent", np.nan)),
                "best_epoch": int(info.get("best_iter", -1)),
                "epochs_used": int(info.get("epochs_used", -1)),
                "convergence_epoch": int(info.get("convergence_epoch", -1)),
                "last_rel_change": float(info.get("last_rel_change", np.nan)),
                "max_iter": int(info.get("max_iter", MAX_ITER)),
                "tol": float(info.get("tol", TOL)),
                "init_method": str(info.get("init_method", INIT_METHOD)),
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
                "stress_init": float("nan"),
                "stress_final": float("nan"),
                "stress_reduction_percent": float("nan"),
                "best_epoch": -1,
                "epochs_used": -1,
                "convergence_epoch": -1,
                "last_rel_change": float("nan"),
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
    "embedding_dim",
    "stress_init",
    "stress_final",
    "stress_reduction_percent",
    "best_epoch",
    "epochs_used",
    "convergence_epoch",
    "last_rel_change",
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
        fmt_row["embedding_dim"] = fmt_pm(
            num_row.get("embedding_dim_mean", np.nan),
            num_row.get("embedding_dim_std", np.nan),
            decimals=2,
        )
        fmt_row["stress_init"] = fmt_pm(
            num_row.get("stress_init_mean", np.nan),
            num_row.get("stress_init_std", np.nan),
            decimals=6,
        )
        fmt_row["stress_final"] = fmt_pm(
            num_row.get("stress_final_mean", np.nan),
            num_row.get("stress_final_std", np.nan),
            decimals=6,
        )
        fmt_row["stress_reduction_percent"] = fmt_pm(
            num_row.get("stress_reduction_percent_mean", np.nan),
            num_row.get("stress_reduction_percent_std", np.nan),
            decimals=4,
        )
        fmt_row["best_epoch"] = fmt_pm(
            num_row.get("best_epoch_mean", np.nan),
            num_row.get("best_epoch_std", np.nan),
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
        fmt_row["last_rel_change"] = fmt_pm(
            num_row.get("last_rel_change_mean", np.nan),
            num_row.get("last_rel_change_std", np.nan),
            decimals=8,
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

    dim = n - 1 if FULL_DIM else min(MDS_DIM, n - 1)

    print("=== MDS-SMACOF BASELINE ===")
    print(f"Reference matrix: {used_path}")
    print(f"n = {n}")
    print(f"embedding_dim = {dim}")
    print(f"missingness levels = {MISSING_FRACS}")
    print(f"replicates per level = {REPS}")
    print(f"max_iter = {MAX_ITER}, tol = {TOL}, init_method = {INIT_METHOD}")
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

    detailed_path = os.path.join(TABLES_DIR, "mds_smacof_all_masks_detailed.csv")
    save_csv(results_df, detailed_path)

    summary_numeric, summary_formatted = summarize_by_missingness(results_df)

    numeric_path = os.path.join(TABLES_DIR, "mds_smacof_summary_numeric_by_missingness.csv")
    formatted_path = os.path.join(TABLES_DIR, "mds_smacof_summary_formatted_by_missingness.csv")

    save_csv(summary_numeric, numeric_path)
    save_csv(summary_formatted, formatted_path)

    display_or_print(results_df, "MDS-SMACOF detailed results")
    display_or_print(summary_formatted, "MDS-SMACOF summary, mean ± std over successful replicates")

    print("\n=== DONE ===")
    print(f"Masked matrices:    {MASKED_DIR}/")
    print(f"Completed matrices: {COMPLETED_DIR}/")
    print(f"Detailed results:   {detailed_path}")
    print(f"Summary numeric:    {numeric_path}")
    print(f"Summary formatted:  {formatted_path}")


if __name__ == "__main__":
    main()
