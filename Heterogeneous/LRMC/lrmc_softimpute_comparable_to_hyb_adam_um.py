# -*- coding: utf-8 -*-
"""
LRMC / Soft-Impute distance-matrix completion experiment
========================================================

Reviewer-oriented rewrite of the previous standalone LRMC script so that its
outputs are directly comparable with the Hyb-Adam-UM and MW-proj/MW* scripts.

Implemented changes
-------------------
1. Replicates increased to 30.
2. Source/reference matrix changed to:
       Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv
3. Runtime is measured for every run and summarized as mean ± std by missingness.
4. RMSE_miss, MAE_miss, Pearson_miss, Spearman_miss are computed ONLY on entries
   hidden by the artificial mask.
5. For every missingness level the summary reports:
       n_missing / n_observed / optimized_variables
   and average iterations/epochs to convergence.
6. Failure counts are reported:
       n_success, n_failed
7. Sanity check on observed entries is reported:
       max_abs_error_observed, mean_abs_error_observed
8. Every detailed row contains:
       mask_seed, missingness_actual
9. Completed matrices preserve all observed entries exactly.
10. The Soft-Impute SVD step uses NumPy's built-in LAPACK SVD by default
    (deterministic and less error-prone than a custom block-power SVD for n=15).

Notes on optimized_variables for LRMC
-------------------------------------
Soft-Impute does not explicitly optimize one independent scalar variable per
missing distance in the same way as Hyb-Adam-UM. For a fair benchmark diagnostic,
this script reports optimized_variables = n_missing, i.e. the number of hidden
lower-triangle distance entries whose values are iteratively imputed and then
evaluated.
"""

from __future__ import annotations

import os
import json
import time
import warnings
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# -------------------------- CONFIG ---------------------------
# ============================================================

MISSING_VAL: float = -1.0

# Reference matrix. The first path is for a normal working directory; the second
# path is convenient when running in a notebook/container where /mnt/data is used.
ORIG_CANDIDATES: Tuple[str, ...] = (
    "Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv",
    "/mnt/data/Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv",
)

# Same missingness protocol as the Hyb-Adam-UM reviewer script.
MISSING_FRACS: Tuple[float, ...] = (0.30, 0.50, 0.65, 0.85)
REPS: int = 30
BASE_SEED: int = 55

# Output folders.
OUTPUT_ROOT: str = "lrmc_softimpute_outputs"
MASKED_DIR: str = os.path.join(OUTPUT_ROOT, "masked_matrices")
COMPLETED_DIR: str = os.path.join(OUTPUT_ROOT, "completed_matrices")
TABLES_DIR: str = os.path.join(OUTPUT_ROOT, "tables")
LOGS_DIR: str = os.path.join(OUTPUT_ROOT, "logs")

# Robust-delta constants, included only for optional cross-method diagnostics.
OMEGA: float = 2.0
EPS_NUM: float = 1.0e-12


@dataclass(frozen=True)
class SoftImputeConfig:
    """Soft-Impute / LRMC configuration."""

    tau: float = 0.02
    max_iters: int = 6000
    tol: float = 1.0e-7
    rank: Optional[int] = None        # None -> min(10, n)
    svd_backend: str = "numpy_full"   # deterministic built-in full SVD
    enforce_nonneg_final: bool = True
    seed_lrmc: int = 0                # kept for reproducible optional initializations


BASE_LRMC_CONFIG = SoftImputeConfig()


# ============================================================
# ------------------------ I/O HELPERS ------------------------
# ============================================================

def ensure_dirs() -> None:
    for d in (OUTPUT_ROOT, MASKED_DIR, COMPLETED_DIR, TABLES_DIR, LOGS_DIR):
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


def safe_tag(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def load_distance_matrix_from_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load a square distance matrix from a CSV/TXT file.

    Supported common formats:
    - row labels in first column and column labels in header;
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
            return num.to_numpy(dtype=float), labels
    except Exception:
        pass

    # Case 2: pure numeric square matrix or label column accidentally included.
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        num = df.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if num.shape[0] == num.shape[1] and num.shape[0] > 1:
            labels = [f"T{i + 1}" for i in range(num.shape[0])]
            return num.to_numpy(dtype=float), labels
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
            # Preserve legacy scale guard for old NW-style files.
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


def sanitize_reference_distance_matrix(D: np.ndarray, name: str = "D_ref") -> np.ndarray:
    """
    Strictly validate and lightly normalize a complete reference distance matrix.

    The reference matrix is the ground truth of the benchmark, so NaN/Inf/negative
    off-diagonal values are reported explicitly rather than silently repaired.
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


def lower_pairs(n: int) -> np.ndarray:
    i, j = np.tril_indices(n, k=-1)
    return np.column_stack([i, j]).astype(np.int32)


def values_on_pairs(M: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.array([], dtype=float)
    return np.asarray(M[pairs[:, 0], pairs[:, 1]], dtype=float)


def finalize_completed_matrix(
    D_hat: np.ndarray,
    D_incomplete: np.ndarray,
    observed_pairs: np.ndarray,
    preserve_observed: bool = True,
    enforce_nonneg: bool = True,
) -> np.ndarray:
    """
    Finalize completed matrix while preserving observed entries exactly.
    """
    M = np.asarray(D_hat, dtype=float).copy()
    M = 0.5 * (M + M.T)
    if enforce_nonneg:
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
# ----------------------- ROBUST DELTA ------------------------
# ============================================================

def robust_delta_per_triplet_numpy(M: np.ndarray, triplets: np.ndarray) -> np.ndarray:
    """Return robust delta(i,j,k) for each triplet."""
    if triplets.size == 0:
        return np.array([], dtype=float)

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
    obs_mask_full = D_inc >= 0
    np.fill_diagonal(obs_mask_full, True)

    return {
        "D_inc": D_inc,
        "obs_mask_full": obs_mask_full,
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
# ---------------------- LRMC / SOFT IMPUTE -------------------
# ============================================================

def singular_value_threshold_builtin(
    Y: np.ndarray,
    tau: float,
    rank: int,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Built-in deterministic singular-value shrinkage:
        S_tau(Y) = U diag(max(s - tau, 0)) V^T
    """
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    rank = min(rank, len(s))
    U = U[:, :rank]
    s = s[:rank]
    Vt = Vt[:rank, :]

    s_thr = np.maximum(s - tau, 0.0)
    active = s_thr > 0.0
    if not np.any(active):
        X_new = np.zeros_like(Y, dtype=float)
        final_rank = 0
    else:
        X_new = (U[:, active] * s_thr[active]) @ Vt[active, :]
        final_rank = int(np.sum(active))

    return X_new, final_rank, s


def soft_impute_lrmc(
    D_in: np.ndarray,
    obs_mask: np.ndarray,
    cfg: SoftImputeConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Soft-Impute LRMC:
        Y = P_O(D_obs) + P_M(X)
        X <- S_tau(Y)

    The returned matrix is complete but final exact observed-entry preservation is
    also enforced outside this function by finalize_completed_matrix().
    """
    n = D_in.shape[0]
    rank = cfg.rank if cfg.rank is not None else min(10, n)
    rank = int(max(1, min(rank, n)))

    M_obs = np.where(obs_mask, D_in, 0.0)
    X = M_obs.copy()

    rel_change = float("nan")
    convergence_epoch: Optional[int] = None
    iters_used = 0
    final_rank = 0
    last_singular_values: np.ndarray = np.array([], dtype=float)

    for t in range(1, cfg.max_iters + 1):
        iters_used = t

        # Standard Soft-Impute projection.
        Y = np.where(obs_mask, M_obs, X)
        Y = 0.5 * (Y + Y.T)
        np.fill_diagonal(Y, 0.0)

        if cfg.svd_backend != "numpy_full":
            raise ValueError(f"Unsupported svd_backend={cfg.svd_backend!r}. Use 'numpy_full'.")

        X_new, final_rank, last_singular_values = singular_value_threshold_builtin(
            Y=Y,
            tau=cfg.tau,
            rank=rank,
        )

        X_new = 0.5 * (X_new + X_new.T)
        np.fill_diagonal(X_new, 0.0)

        denom = np.linalg.norm(X, "fro") + EPS_NUM
        rel_change = float(np.linalg.norm(X_new - X, "fro") / denom)
        X = X_new

        if rel_change < cfg.tol:
            convergence_epoch = int(t)
            break

    if convergence_epoch is None:
        # Conservative proxy when the tolerance is not reached.
        convergence_epoch = int(iters_used)

    # Finalize inside the algorithm: preserve observed entries.
    D_out = np.where(obs_mask, D_in, X)
    D_out = 0.5 * (D_out + D_out.T)
    np.fill_diagonal(D_out, 0.0)

    if cfg.enforce_nonneg_final:
        miss_mask = ~obs_mask
        D_out[miss_mask] = np.maximum(D_out[miss_mask], 0.0)
        D_out = 0.5 * (D_out + D_out.T)
        np.fill_diagonal(D_out, 0.0)

    info = {
        "success": bool(np.isfinite(D_out).all()),
        "iters_used": int(iters_used),
        "epochs_used": int(iters_used),
        "convergence_epoch": int(convergence_epoch),
        "final_rank": int(final_rank),
        "last_rel_change": float(rel_change),
        "last_singular_values_json": json.dumps(
            [float(x) for x in np.asarray(last_singular_values, dtype=float)],
            ensure_ascii=False,
        ),
        "lrmc_config_json": json.dumps(asdict(cfg), ensure_ascii=False),
    }
    return D_out, info


# ============================================================
# ----------------------- SINGLE RUNNER -----------------------
# ============================================================

def run_single_mask(
    rec: Dict[str, object],
    D_reference: np.ndarray,
    labels: List[str],
    trip_all: np.ndarray,
    ntri: int,
    Delta_original: float,
    Delta_normalized_original: float,
    cfg: SoftImputeConfig,
    method_label: str = "LRMC-SoftImpute",
    save_completed: bool = True,
    completed_prefix: str = "LRMC_SoftImpute_completed",
) -> Dict[str, object]:
    pct = int(rec["pct_missing"])
    rep = int(rec["replicate"])
    D_inc = np.asarray(rec["D_inc"], dtype=float)
    obs_mask = np.asarray(rec["obs_mask_full"], dtype=bool)
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
        "success": False,
        "success_numerical": False,
        "success_valid_matrix": False,
        "error_message": "",
        "masked_file": rec.get("masked_file", ""),
        "completed_file": "",
    }

    # Make the optimizer seed deterministic per mask, even though numpy_full SVD is deterministic.
    cfg_run = SoftImputeConfig(**{**asdict(cfg), "seed_lrmc": int(10000 * pct + rep)})

    t0 = time.perf_counter()
    try:
        D_completed_raw, info = soft_impute_lrmc(D_in=D_inc, obs_mask=obs_mask, cfg=cfg_run)
        D_completed = finalize_completed_matrix(
            D_completed_raw,
            D_incomplete=D_inc,
            observed_pairs=observed_pairs,
            preserve_observed=True,
            enforce_nonneg=cfg.enforce_nonneg_final,
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

        success_numerical = bool(
            np.isfinite(runtime_seconds)
            and np.isfinite(D_completed).all()
            and np.isfinite(Delta_total)
            and bool(info.get("success", True))
        )
        success_valid_matrix = bool(is_valid_completed_distance_matrix(D_completed, D_reference))
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
                "runtime_seconds": float(runtime_seconds),
                "completed_file": completed_file,
                "Delta_total_completed": float(Delta_total),
                "Delta_normalized_completed": float(Delta_norm),
                "Delta_per_triangle_completed": float(Delta_per_triangle),
                "Delta_relative_to_original": float(Delta_relative),
                "Delta_original": float(Delta_original),
                "Delta_normalized_original": float(Delta_normalized_original),
                **miss_metrics,
                **sanity,
                "iters_used": int(info.get("iters_used", -1)),
                "epochs_used": int(info.get("epochs_used", -1)),
                "convergence_epoch": int(info.get("convergence_epoch", -1)),
                "final_rank": int(info.get("final_rank", -1)),
                "last_rel_change": float(info.get("last_rel_change", np.nan)),
                "last_singular_values_json": info.get("last_singular_values_json", "[]"),
                "lrmc_config_json": info.get("lrmc_config_json", json.dumps(asdict(cfg_run), ensure_ascii=False)),
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
                "iters_used": -1,
                "epochs_used": -1,
                "convergence_epoch": -1,
                "final_rank": -1,
                "last_rel_change": float("nan"),
                "last_singular_values_json": "[]",
                "lrmc_config_json": json.dumps(asdict(cfg_run), ensure_ascii=False),
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
    "RMSE_miss",
    "MAE_miss",
    "Pearson_miss",
    "Spearman_miss",
    "runtime_seconds",
    "iters_used",
    "epochs_used",
    "convergence_epoch",
    "final_rank",
    "last_rel_change",
    "Delta_total_completed",
    "Delta_normalized_completed",
    "Delta_per_triangle_completed",
    "Delta_relative_to_original",
    "max_abs_error_observed",
    "mean_abs_error_observed",
)


def make_summary_tables(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows: List[Dict[str, object]] = []
    formatted_rows: List[Dict[str, object]] = []

    for pct in sorted(results_df["pct_missing"].unique()):
        sub_all = results_df[results_df["pct_missing"] == pct].copy()
        sub_success = sub_all[sub_all["success"] == True].copy()

        n_total = int(len(sub_all))
        n_success = int(sub_all["success"].sum())
        n_failed = int(n_total - n_success)

        numeric_row: Dict[str, object] = {
            "pct_missing": int(pct),
            "n_replicates": n_total,
            "n_success": n_success,
            "n_failed": n_failed,
        }
        formatted_row: Dict[str, object] = {
            "% Missing": f"{int(pct)}%",
            "n_replicates": n_total,
            "n_success": n_success,
            "n_failed": n_failed,
        }

        for col in SUMMARY_NUMERIC_COLUMNS:
            if col not in sub_all.columns:
                continue

            # Use successful runs for performance/error summaries; use all runs for fixed mask counts.
            source = sub_all if col in {
                "missingness_actual",
                "n_missing",
                "n_observed",
                "optimized_variables",
            } else sub_success

            mean_val, std_val = mean_std_nan(source[col].to_numpy(dtype=float))
            numeric_row[f"{col}_mean"] = mean_val
            numeric_row[f"{col}_std"] = std_val

            decimals = 2 if col in {"runtime_seconds", "convergence_epoch", "epochs_used", "iters_used"} else 6
            formatted_row[col] = fmt_pm(mean_val, std_val, decimals=decimals)

        numeric_rows.append(numeric_row)
        formatted_rows.append(formatted_row)

    return pd.DataFrame(numeric_rows), pd.DataFrame(formatted_rows)


# ============================================================
# ---------------------------- MAIN ---------------------------
# ============================================================

def main() -> None:
    ensure_dirs()

    D_orig, labels, used_orig_path = load_matrix_with_candidates(ORIG_CANDIDATES)
    D0 = sanitize_reference_distance_matrix(symmetrize_full(D_orig), "D_ref")
    n = D0.shape[0]

    if len(labels) != n:
        labels = [f"T{i + 1}" for i in range(n)]

    trip_all = np.array(list(combinations(range(n), 3)), dtype=np.int32)
    ntri = int(len(trip_all))
    Delta_original = robust_delta_sum_numpy(D0, trip_all)
    Delta_normalized_original = compute_normalized_delta(D0, trip_all)

    print("=== ORIGINAL MATRIX INFO ===")
    print(f"Used file: {used_orig_path}")
    print(f"n = {n}")
    print(f"number of lower-triangle pairs = {len(lower_pairs(n))}")
    print(f"number of triplets = {ntri}")
    print(f"Delta_original = {Delta_original:.6f}")
    print(f"Delta_normalized_original = {Delta_normalized_original:.6f}")
    print()

    # Build and save masks.
    mask_registry = build_mask_registry(D0)
    save_masked_matrices(mask_registry, labels)

    print("=== Running LRMC / Soft-Impute on all masks ===")
    print(f"Missingness levels: {MISSING_FRACS}")
    print(f"Replicates per level: {REPS}")
    print(f"Soft-Impute config: {BASE_LRMC_CONFIG}")
    print()

    rows: List[Dict[str, object]] = []
    for rec in mask_registry:
        pct = int(rec["pct_missing"])
        rep = int(rec["replicate"])
        print(f"  Processing {pct}% missing, replicate {rep:02d}, seed={rec['mask_seed']}...")
        row = run_single_mask(
            rec=rec,
            D_reference=D0,
            labels=labels,
            trip_all=trip_all,
            ntri=ntri,
            Delta_original=Delta_original,
            Delta_normalized_original=Delta_normalized_original,
            cfg=BASE_LRMC_CONFIG,
            method_label="LRMC-SoftImpute",
            save_completed=True,
            completed_prefix="LRMC_SoftImpute_completed",
        )
        rows.append(row)

    results_df = pd.DataFrame(rows)
    detail_path = os.path.join(TABLES_DIR, "lrmc_softimpute_all_masks_detailed.csv")
    save_csv(results_df, detail_path)

    summary_numeric, summary_formatted = make_summary_tables(results_df)
    numeric_path = os.path.join(TABLES_DIR, "lrmc_softimpute_summary_numeric_by_missingness.csv")
    formatted_path = os.path.join(TABLES_DIR, "lrmc_softimpute_summary_formatted_by_missingness.csv")
    save_csv(summary_numeric, numeric_path)
    save_csv(summary_formatted, formatted_path)

    # Small manifest for reproducibility.
    manifest = {
        "method": "LRMC-SoftImpute",
        "used_reference_matrix": used_orig_path,
        "n": int(n),
        "missing_fracs": list(MISSING_FRACS),
        "reps": int(REPS),
        "base_seed": int(BASE_SEED),
        "missing_value": float(MISSING_VAL),
        "softimpute_config": asdict(BASE_LRMC_CONFIG),
        "outputs": {
            "masked_dir": MASKED_DIR,
            "completed_dir": COMPLETED_DIR,
            "detail_csv": detail_path,
            "summary_numeric_csv": numeric_path,
            "summary_formatted_csv": formatted_path,
        },
    }
    manifest_path = os.path.join(OUTPUT_ROOT, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved: {manifest_path}")

    display_or_print(results_df, "LRMC detailed results")
    display_or_print(summary_formatted, "LRMC summary by missingness")

    print("\n=== ALL PROCESSING COMPLETE ===")
    print(f"Masked matrices:    {MASKED_DIR}/")
    print(f"Completed matrices: {COMPLETED_DIR}/")
    print(f"Detailed results:   {detail_path}")
    print(f"Summary numeric:    {numeric_path}")
    print(f"Summary formatted:  {formatted_path}")


if __name__ == "__main__":
    main()
