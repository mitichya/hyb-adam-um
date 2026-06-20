# -*- coding: utf-8 -*-
"""
MW-proj / MW* distance-matrix completion experiment
===================================================

This version is deliberately aligned with the Hyb-Adam-UM experiment script, so
the two algorithms can be compared by joining/stacking their detailed and
summary CSV files.

Main reviewer-oriented protocol
-------------------------------
1. Replicates increased to 30.
2. Source/reference matrix changed to:
       Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv
3. Runtime is measured for every run and summarized as mean ± std by missingness.
4. RMSE_miss, MAE_miss, Pearson_miss, Spearman_miss are computed ONLY on entries
   hidden by the artificial mask.
5. For every missingness level the summary reports:
       n_missing / n_observed / optimized_variables
   where optimized_variables for MW-proj means the number of final tree branch
   lengths fitted by weighted NNLS.
6. Failure counts are reported:
       n_success, n_failed
7. Sanity check on observed entries is reported:
       max_abs_error_observed, mean_abs_error_observed
8. Every detailed row contains:
       mask_seed, missingness_actual
9. The output file names and column names follow the Hyb-Adam-UM script style.

Implementation note on built-in solvers
---------------------------------------
There is no standard, widely available Python library function for the complete
Makarenkov-Lapointe MW* / MW-proj reconstruction workflow. To reduce the amount
of custom numerical code, this script uses scipy.optimize.nnls for the weighted
least-squares branch-length subproblem whenever SciPy is available. The MW Step-A
minimax filling and Step-B stepwise topology search are implemented explicitly.

Missing entries are encoded as -1.
"""

from __future__ import annotations

import os
import json
import time
import shutil
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from scipy.optimize import nnls as scipy_nnls
    NNLS_BACKEND = "scipy.optimize.nnls"
except Exception:
    scipy_nnls = None
    NNLS_BACKEND = "numpy_lstsq_clipped_fallback"


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

OUTPUT_ROOT: str = "mw_proj_outputs"
MASKED_DIR: str = os.path.join(OUTPUT_ROOT, "masked_matrices")
COMPLETED_DIR: str = os.path.join(OUTPUT_ROOT, "completed_matrices")
TABLES_DIR: str = os.path.join(OUTPUT_ROOT, "tables")
LOGS_DIR: str = os.path.join(OUTPUT_ROOT, "logs")

MISSING_VAL: float = -1.0
PRESERVE_OBSERVED_ENTRIES: bool = True
METHOD_LABEL: str = "MW-proj"

# Robust-delta constants, kept identical to the Hyb-Adam-UM script.
OMEGA: float = 2.0
EPS_NUM: float = 1.0e-12


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


def append_text_log(path: Optional[str], message: str, also_print: bool = False) -> None:
    if also_print:
        print(message)
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


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
    Load a square distance matrix from a CSV file.

    Supported formats:
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

    # Final fallback.
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
            # Legacy guard, copied from the Hyb-Adam-UM script.
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
    np.fill_diagonal(M, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = M[i, j], M[j, i]
            a_ok = (a != MISSING_VAL)
            b_ok = (b != MISSING_VAL)
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


def finalize_completed_matrix(
    D_hat: np.ndarray,
    D_incomplete: np.ndarray,
    observed_pairs: np.ndarray,
    preserve_observed: bool = True,
) -> np.ndarray:
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


def observed_mask_from_pairs(n: int, observed_pairs: np.ndarray) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    if observed_pairs.size > 0:
        i, j = observed_pairs[:, 0], observed_pairs[:, 1]
        mask[i, j] = True
        mask[j, i] = True
    np.fill_diagonal(mask, True)
    return mask


def mean_fill_initial_matrix(D_inc: np.ndarray, observed_pairs: np.ndarray) -> np.ndarray:
    M = np.asarray(D_inc, dtype=float).copy()
    vals = values_on_pairs(M, observed_pairs)
    fill_value = float(np.mean(vals)) if len(vals) > 0 else 1.0
    M[M == MISSING_VAL] = fill_value
    M = symmetrize_full(M)
    M = np.maximum(M, 0.0)
    np.fill_diagonal(M, 0.0)
    return M


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
# ------------------ MW STEP A: MINIMAX FILL ------------------
# ============================================================

def choose_stepA_method_thresholds(n: int, frac_missing: float) -> str:
    """
    Thresholds stated in the Makarenkov-Lapointe MW* paper text:
      8x8:   Additive if <20% missing, otherwise Ultrametric
      16x16: Additive if <30% missing, otherwise Ultrametric
      24x24: Additive if <40% missing, otherwise Ultrametric
    For larger matrices, the 24x24 threshold is used as a pragmatic extension.
    """
    if n <= 8:
        thr = 0.20
    elif n <= 16:
        thr = 0.30
    elif n <= 24:
        thr = 0.40
    else:
        thr = 0.40
    return "additive" if frac_missing < thr else "ultrametric"


def ultrametric_minimax_table1(D_partial: np.ndarray, max_loops: int = 10_000) -> Tuple[np.ndarray, np.ndarray, int]:
    D = symmetrize_with_missing(D_partial)
    n = D.shape[0]
    missing0 = (D == MISSING_VAL) & (~np.eye(n, dtype=bool))
    est_mask = np.zeros_like(D, dtype=bool)

    loops = 0
    while loops < max_loops:
        loops += 1
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] != MISSING_VAL:
                    continue
                best = None
                for k in range(n):
                    if k == i or k == j:
                        continue
                    dik, djk = D[i, k], D[j, k]
                    if dik == MISSING_VAL or djk == MISSING_VAL:
                        continue
                    cand = max(dik, djk)
                    if best is None or cand < best:
                        best = cand
                if best is not None:
                    D[i, j] = D[j, i] = float(best)
                    est_mask[i, j] = est_mask[j, i] = True
                    changed = True
        if not changed:
            break

    filled = missing0 & (D != MISSING_VAL)
    est_mask = est_mask & filled
    return D, est_mask, loops


def additive_minimax_table2(D_partial: np.ndarray, max_loops: int = 10_000) -> Tuple[np.ndarray, np.ndarray, int]:
    D = symmetrize_with_missing(D_partial)
    n = D.shape[0]
    missing0 = (D == MISSING_VAL) & (~np.eye(n, dtype=bool))
    est_mask = np.zeros_like(D, dtype=bool)

    loops = 0
    while loops < max_loops:
        loops += 1
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] != MISSING_VAL:
                    continue
                best = None
                for k in range(n):
                    if k == i or k == j:
                        continue
                    dik, djk = D[i, k], D[j, k]
                    if dik == MISSING_VAL or djk == MISSING_VAL:
                        continue
                    for l in range(n):
                        if l == i or l == j or l == k:
                            continue
                        dil, djl, dkl = D[i, l], D[j, l], D[k, l]
                        if dil == MISSING_VAL or djl == MISSING_VAL or dkl == MISSING_VAL:
                            continue
                        cand = max(dik + djl, dil + djk) - dkl
                        if best is None or cand < best:
                            best = cand
                if best is not None:
                    D[i, j] = D[j, i] = float(best)
                    est_mask[i, j] = est_mask[j, i] = True
                    changed = True
        if not changed:
            break

    filled = missing0 & (D != MISSING_VAL)
    est_mask = est_mask & filled
    return D, est_mask, loops


def mw_star_stepA_build_D_and_W(
    D_inc: np.ndarray,
    obs_mask_initial: np.ndarray,
    frac_missing: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    D0p = symmetrize_with_missing(D_inc)
    n = D0p.shape[0]

    method = choose_stepA_method_thresholds(n, frac_missing)
    if method == "ultrametric":
        D_A, est_mask, loops = ultrametric_minimax_table1(D0p)
    else:
        D_A, est_mask, loops = additive_minimax_table2(D0p)

    known0 = obs_mask_initial.copy()
    np.fill_diagonal(known0, True)
    known0 = known0 & (~np.eye(n, dtype=bool))

    W = np.zeros((n, n), dtype=float)
    W[known0] = 1.0
    W[est_mask] = 0.5
    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)

    D_A = symmetrize_with_missing(D_A)
    np.fill_diagonal(D_A, 0.0)

    unfilled_mask = (D_A == MISSING_VAL) & (~np.eye(n, dtype=bool))
    n_stepA_estimated_pairs = int(np.sum(np.tril(est_mask, k=-1)))
    n_stepA_unfilled_pairs = int(np.sum(np.tril(unfilled_mask, k=-1)))

    info = {
        "stepA_method": method,
        "stepA_loops": int(loops),
        "stepA_estimated_pairs": n_stepA_estimated_pairs,
        "stepA_unfilled_pairs": n_stepA_unfilled_pairs,
    }
    return D_A, W, info


# ============================================================
# ------------------- TREE REPRESENTATION --------------------
# ============================================================

def build_edge_index(edges_uv: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int, int]], Dict[Tuple[int, int], int]]:
    edge_map: Dict[Tuple[int, int], int] = {}
    edges2: List[Tuple[int, int, int]] = []
    eid = 0
    for (u, v) in edges_uv:
        key = (u, v) if u < v else (v, u)
        if key in edge_map:
            continue
        edge_map[key] = eid
        edges2.append((u, v, eid))
        eid += 1
    return edges2, edge_map


def build_adjacency(edges: List[Tuple[int, int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    adj: Dict[int, List[Tuple[int, int]]] = {}
    for u, v, eid in edges:
        adj.setdefault(u, []).append((v, eid))
        adj.setdefault(v, []).append((u, eid))
    return adj


def path_edge_ids(adj: Dict[int, List[Tuple[int, int]]], src: int, dst: int) -> List[int]:
    q = [src]
    parent = {src: (-1, -1)}
    head = 0
    while head < len(q):
        x = q[head]
        head += 1
        if x == dst:
            break
        for y, eid in adj.get(x, []):
            if y in parent:
                continue
            parent[y] = (x, eid)
            q.append(y)
    if dst not in parent:
        raise RuntimeError("Tree path not found.")
    eids = []
    cur = dst
    while cur != src:
        prev, eid = parent[cur]
        eids.append(eid)
        cur = prev
    eids.reverse()
    return eids


def patristic_matrix_from_tree(n_leaves: int, edges: List[Tuple[int, int, int]], lengths: np.ndarray) -> np.ndarray:
    adj = build_adjacency(edges)
    elen = {eid: float(lengths[eid]) for _, _, eid in edges}

    D = np.zeros((n_leaves, n_leaves), dtype=float)
    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            eids = path_edge_ids(adj, i, j)
            D[i, j] = D[j, i] = sum(elen[eid] for eid in eids)
    np.fill_diagonal(D, 0.0)
    return D


# ============================================================
# ---------------- WEIGHTED NNLS BRANCH FITTING ---------------
# ============================================================

def solve_nonnegative_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve min ||A x - b||_2 subject to x >= 0.

    Uses scipy.optimize.nnls when available. The fallback is intentionally simple
    and should only be used if SciPy is unavailable.
    """
    if A.size == 0:
        return np.zeros(A.shape[1], dtype=float)

    if scipy_nnls is not None:
        x, _ = scipy_nnls(A, b)
        return np.asarray(x, dtype=float)

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def fit_branch_lengths_min_Qw(
    edges_uv: List[Tuple[int, int]],
    D_A: np.ndarray,
    W: np.ndarray,
    leaf_set: List[int],
) -> Tuple[np.ndarray, float, int, int]:
    """
    min_{x>=0} sum_{i<j in leaf_set} W_ij * (d_T(i,j;x) - D_A(i,j))^2
    using weighted NNLS.
    """
    edges, _ = build_edge_index(edges_uv)
    if not edges:
        return np.zeros(0, dtype=float), 0.0, 0, 0

    adj = build_adjacency(edges)
    m_edges = max(eid for _, _, eid in edges) + 1

    L = sorted(leaf_set)
    pairs = []
    for a in range(len(L)):
        i = L[a]
        for b in range(a + 1, len(L)):
            j = L[b]
            wij = float(W[i, j])
            dij = float(D_A[i, j])
            if wij <= 0.0 or dij == MISSING_VAL:
                continue
            pairs.append((i, j, wij, dij))

    if not pairs:
        return np.zeros(m_edges, dtype=float), 0.0, m_edges, 0

    A = np.zeros((len(pairs), m_edges), dtype=float)
    y = np.zeros(len(pairs), dtype=float)
    wvec = np.zeros(len(pairs), dtype=float)

    for p, (i, j, wij, dij) in enumerate(pairs):
        eids = path_edge_ids(adj, i, j)
        A[p, eids] = 1.0
        y[p] = dij
        wvec[p] = wij

    sw = np.sqrt(np.maximum(wvec, 0.0))
    Aw = A * sw[:, None]
    yw = y * sw

    x = solve_nonnegative_least_squares(Aw, yw)
    resid = Aw @ x - yw
    Qw = float(resid @ resid)
    return x, Qw, m_edges, len(pairs)


# ============================================================
# ---------------- MW STEP B: TOPOLOGY SEARCH -----------------
# ============================================================

def global_score_eq9(W: np.ndarray) -> np.ndarray:
    return np.sum(W, axis=0)


def best_graft_for_candidate(
    cand: int,
    edges_uv: List[Tuple[int, int]],
    next_internal_id: int,
    D_A: np.ndarray,
    W: np.ndarray,
    leaf_set_current: List[int],
) -> Tuple[List[Tuple[int, int]], np.ndarray, float, int, int]:
    leaf_set_new = list(leaf_set_current) + [cand]

    if len(edges_uv) == 0:
        u = next_internal_id
        variant = [(leaf_set_current[0], u), (u, cand)]
        x, Qw, nvars, nfit_pairs = fit_branch_lengths_min_Qw(variant, D_A, W, leaf_set_new)
        return variant, x, Qw, nvars, nfit_pairs

    best_edges = None
    best_x = None
    best_Qw = None
    best_nvars = 0
    best_nfit_pairs = 0

    for (a, b) in edges_uv:
        u = next_internal_id
        variant: List[Tuple[int, int]] = []
        for (x0, y0) in edges_uv:
            if (x0 == a and y0 == b) or (x0 == b and y0 == a):
                continue
            variant.append((x0, y0))
        variant.extend([(a, u), (u, b), (u, cand)])

        xlen, Qw, nvars, nfit_pairs = fit_branch_lengths_min_Qw(variant, D_A, W, leaf_set_new)
        if best_Qw is None or Qw < best_Qw - 1e-15:
            best_Qw = Qw
            best_edges = variant
            best_x = xlen
            best_nvars = nvars
            best_nfit_pairs = nfit_pairs

    assert best_edges is not None and best_x is not None and best_Qw is not None
    return best_edges, best_x, float(best_Qw), best_nvars, best_nfit_pairs


def select_next_taxon(
    leaves_in_tree: List[int],
    remaining: List[int],
    W: np.ndarray,
    D_A: np.ndarray,
    edges_uv: List[Tuple[int, int]],
    next_internal_id: int,
    score_global: np.ndarray,
) -> int:
    L = list(leaves_in_tree)

    prim = [(float(np.sum(W[L, cand])), cand) for cand in remaining]
    max_s = max(s for s, _ in prim)
    cands1 = [cand for s, cand in prim if abs(s - max_s) < 1e-12]
    if len(cands1) == 1:
        return cands1[0]

    bestQ: Dict[int, float] = {}
    for cand in cands1:
        _, _, q, _, _ = best_graft_for_candidate(
            cand=cand,
            edges_uv=edges_uv,
            next_internal_id=next_internal_id,
            D_A=D_A,
            W=W,
            leaf_set_current=L,
        )
        bestQ[cand] = q

    minQ = min(bestQ.values())
    cands2 = [c for c in cands1 if abs(bestQ[c] - minQ) < 1e-12]
    if len(cands2) == 1:
        return cands2[0]

    gs = [(float(score_global[c]), c) for c in cands2]
    maxg = max(v for v, _ in gs)
    cands3 = [c for v, c in gs if abs(v - maxg) < 1e-12]
    return min(cands3)


def mw_star_stepB_build_tree(
    D_A: np.ndarray,
    W: np.ndarray,
    exhaustive_start_pairs: bool = True,
) -> Tuple[List[Tuple[int, int, int]], np.ndarray, Dict[str, object]]:
    n = D_A.shape[0]
    score_g = global_score_eq9(W)

    start_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if D_A[i, j] != MISSING_VAL]
    if len(start_pairs) == 0:
        raise RuntimeError("MW-proj: no known distances after Step A; cannot start Step B.")

    if not exhaustive_start_pairs:
        start_pairs = [start_pairs[0]]

    best_final = None  # (Qw, edges, lengths, nvars, nfit_pairs, start_pair)
    n_weighted_lsq_fits = 0
    n_graft_evaluations = 0

    for (i0, j0) in start_pairs:
        edges_uv = [(i0, j0)]
        leaves_in_tree = [i0, j0]
        remaining = [k for k in range(n) if k not in (i0, j0)]
        next_internal = n

        while remaining:
            cand = select_next_taxon(
                leaves_in_tree=leaves_in_tree,
                remaining=remaining,
                W=W,
                D_A=D_A,
                edges_uv=edges_uv,
                next_internal_id=next_internal,
                score_global=score_g,
            )
            # Count all possible split positions for this chosen graft.
            n_graft_evaluations += max(1, len(edges_uv))

            edges_uv, _, _, _, _ = best_graft_for_candidate(
                cand=cand,
                edges_uv=edges_uv,
                next_internal_id=next_internal,
                D_A=D_A,
                W=W,
                leaf_set_current=leaves_in_tree,
            )
            n_weighted_lsq_fits += max(1, len(edges_uv))
            next_internal += 1
            leaves_in_tree.append(cand)
            remaining.remove(cand)

        lengths, Qw, nvars, nfit_pairs = fit_branch_lengths_min_Qw(edges_uv, D_A, W, leaf_set=list(range(n)))
        n_weighted_lsq_fits += 1
        edges, _ = build_edge_index(edges_uv)

        if best_final is None or Qw < best_final[0] - 1e-15:
            best_final = (Qw, edges, lengths, nvars, nfit_pairs, (i0, j0))

    assert best_final is not None
    Qw_best, edges_best, lengths_best, nvars_best, nfit_pairs_best, start_pair_best = best_final

    info = {
        "MW_Qw_final": float(Qw_best),
        "optimized_variables": int(nvars_best),
        "weighted_fit_pairs": int(nfit_pairs_best),
        "final_tree_edges": int(nvars_best),
        "n_start_pairs": int(len(start_pairs)),
        "best_start_pair": str(tuple(int(x) for x in start_pair_best)),
        "n_weighted_lsq_fits": int(n_weighted_lsq_fits),
        "n_graft_evaluations": int(n_graft_evaluations),
        "nnls_backend": NNLS_BACKEND,
    }
    return edges_best, lengths_best, info


# ============================================================
# ---------------------- MW COMPLETION CORE -------------------
# ============================================================

def mw_proj_completion(
    D_inc: np.ndarray,
    observed_pairs: np.ndarray,
    frac_missing: float,
    preserve_observed: bool = PRESERVE_OBSERVED_ENTRIES,
) -> Tuple[np.ndarray, Dict[str, object]]:
    n = D_inc.shape[0]
    obs_mask_initial = observed_mask_from_pairs(n, observed_pairs)

    D_A, W, info_A = mw_star_stepA_build_D_and_W(D_inc, obs_mask_initial, frac_missing)
    edges, lengths, info_B = mw_star_stepB_build_tree(D_A, W, exhaustive_start_pairs=True)
    D_pat = patristic_matrix_from_tree(D_A.shape[0], edges, lengths)

    D_hat = finalize_completed_matrix(
        D_hat=D_pat,
        D_incomplete=D_inc,
        observed_pairs=observed_pairs,
        preserve_observed=preserve_observed,
    )

    info: Dict[str, object] = {}
    info.update(info_A)
    info.update(info_B)
    return D_hat, info


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


def run_single_mask(
    rec: Dict[str, object],
    D_reference: np.ndarray,
    labels: List[str],
    trip_all: np.ndarray,
    ntri: int,
    Delta_original: float,
    Delta_normalized_original: float,
    method_label: str = METHOD_LABEL,
    save_completed: bool = True,
    completed_prefix: str = "MWproj_completed",
    verbose: bool = False,
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
        "optimized_variables": float("nan"),
        "success": False,
        "success_numerical": False,
        "success_valid_matrix": False,
        "objective_improved": False,
        "error_message": "",
        "masked_file": rec.get("masked_file", ""),
        "completed_file": "",
        "run_log_file": "",
    }

    run_tag = safe_tag(f"{completed_prefix}_p{pct}_rep{rep:02d}_seed{row['mask_seed']}")
    run_log_path = os.path.join(LOGS_DIR, f"{run_tag}.log")

    t0 = time.perf_counter()
    try:
        with open(run_log_path, "w", encoding="utf-8") as f:
            f.write("MW-proj run log\n")
            f.write(f"method={method_label}\n")
            f.write(f"nnls_backend={NNLS_BACKEND}\n")
            f.write(json.dumps({k: row[k] for k in row if k not in {'error_message'}}, ensure_ascii=False) + "\n")

        D_initial = mean_fill_initial_matrix(D_inc, observed_pairs)
        Delta_init = robust_delta_sum_numpy(D_initial, trip_all)

        D_completed, info = mw_proj_completion(
            D_inc=D_inc,
            observed_pairs=observed_pairs,
            frac_missing=float(rec["frac_requested"]),
            preserve_observed=PRESERVE_OBSERVED_ENTRIES,
        )
        runtime_seconds = time.perf_counter() - t0

        miss_metrics = completion_metrics_on_missing(D_completed, D_reference, missing_pairs)
        sanity = observed_sanity_metrics(D_completed, D_reference, observed_pairs)

        if np.isfinite(sanity["max_abs_error_observed"]) and sanity["max_abs_error_observed"] > 1.0e-10:
            raise RuntimeError(
                "Observed entries were modified: "
                f"max_abs_error_observed={sanity['max_abs_error_observed']:.6g}"
            )

        Delta_total = robust_delta_sum_numpy(D_completed, trip_all)
        Delta_norm = compute_normalized_delta(D_completed, trip_all)
        Delta_per_triangle = Delta_total / ntri if ntri > 0 else float("nan")
        Delta_relative = Delta_total / Delta_original if abs(Delta_original) > EPS_NUM else float("nan")
        Delta_final = Delta_total
        Delta_reduction_percent = (
            100.0 * (Delta_init - Delta_final) / Delta_init
            if np.isfinite(Delta_init) and abs(Delta_init) > EPS_NUM
            else float("nan")
        )

        success_numerical = bool(
            np.isfinite(runtime_seconds)
            and np.isfinite(Delta_total)
            and np.isfinite(D_completed).all()
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
                "run_log_file": run_log_path,
                "Delta_total_completed": float(Delta_total),
                "Delta_normalized_completed": float(Delta_norm),
                "Delta_per_triangle_completed": float(Delta_per_triangle),
                "Delta_relative_to_original": float(Delta_relative),
                "Delta_original": float(Delta_original),
                "Delta_normalized_original": float(Delta_normalized_original),
                **miss_metrics,
                **sanity,
                # Kept with Hyb-style names for table compatibility.
                # For MW-proj, Delta_init is a mean-fill diagnostic baseline;
                # Delta_final is the robust Delta of the final completed matrix.
                "Delta_init": float(Delta_init),
                "Delta_final": float(Delta_final),
                "Delta_reduction_percent": float(Delta_reduction_percent),
                # MW-proj is not epoch-trained; use 0 to keep numeric summaries valid.
                "best_epoch": 0,
                "epochs_used": 0,
                "convergence_epoch": 0,
                "final_learning_rate": float("nan"),
                **info,
            }
        )

        append_text_log(
            run_log_path,
            (
                f"SUCCESS runtime={runtime_seconds:.6f}s "
                f"RMSE_miss={row['RMSE_miss']:.8g} "
                f"MAE_miss={row['MAE_miss']:.8g} "
                f"Delta_final={row['Delta_final']:.8g} "
                f"optimized_variables={row['optimized_variables']}"
            ),
            also_print=verbose,
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
                "run_log_file": run_log_path,
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
                "best_epoch": 0,
                "epochs_used": 0,
                "convergence_epoch": 0,
                "final_learning_rate": float("nan"),
                "stepA_method": "",
                "stepA_loops": float("nan"),
                "stepA_estimated_pairs": float("nan"),
                "stepA_unfilled_pairs": float("nan"),
                "MW_Qw_final": float("nan"),
                "weighted_fit_pairs": float("nan"),
                "final_tree_edges": float("nan"),
                "n_start_pairs": float("nan"),
                "best_start_pair": "",
                "n_weighted_lsq_fits": float("nan"),
                "n_graft_evaluations": float("nan"),
                "nnls_backend": NNLS_BACKEND,
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
    "Delta_total_completed",
    "Delta_normalized_completed",
    "Delta_per_triangle_completed",
    "Delta_relative_to_original",
    "max_abs_error_observed",
    "mean_abs_error_observed",
    # MW-specific diagnostics retained in numeric summary.
    "MW_Qw_final",
    "weighted_fit_pairs",
    "final_tree_edges",
    "n_start_pairs",
    "n_weighted_lsq_fits",
    "n_graft_evaluations",
    "stepA_loops",
    "stepA_estimated_pairs",
    "stepA_unfilled_pairs",
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
        stepA_mode = group0["stepA_method"].mode().iloc[0] if "stepA_method" in group0.columns and len(group0["stepA_method"].dropna()) > 0 else ""

        num_row: Dict[str, object] = {
            "pct_missing": int(pct),
            "n_runs": int(len(group0)),
            "n_success": n_success,
            "n_failed": n_failed,
            "n_objective_improved": n_objective_improved,
            "n_not_improved": n_not_improved,
            "stepA_method_mode": stepA_mode,
            "nnls_backend": NNLS_BACKEND,
        }
        fmt_row: Dict[str, object] = {
            "% Missing": f"{int(pct)}%",
            "n_runs": int(len(group0)),
            "n_success": n_success,
            "n_failed": n_failed,
            "n_objective_improved": n_objective_improved,
            "n_not_improved": n_not_improved,
            "stepA_method_mode": stepA_mode,
            "nnls_backend": NNLS_BACKEND,
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
            "Delta_total_completed": 4,
            "Delta_normalized_completed": 6,
            "Delta_per_triangle_completed": 6,
            "Delta_relative_to_original": 6,
            "max_abs_error_observed": 12,
            "mean_abs_error_observed": 12,
            "MW_Qw_final": 6,
            "weighted_fit_pairs": 1,
            "final_tree_edges": 1,
            "n_start_pairs": 1,
            "n_weighted_lsq_fits": 1,
            "n_graft_evaluations": 1,
            "stepA_loops": 1,
            "stepA_estimated_pairs": 1,
            "stepA_unfilled_pairs": 1,
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
    delta_col = f"{METHOD_LABEL} Delta_total"
    rows: List[Dict[str, object]] = [
        {
            "% Missing": "Original",
            delta_col: f"{Delta_original:.4f}",
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
                delta_col: fmt_pm(dt_m, dt_s, 4),
                "Delta_normalized": fmt_pm(dn_m, dn_s, 6),
                "Delta_per_triangle": fmt_pm(dpt_m, dpt_s, 6),
                "Delta_relative_to_original": fmt_pm(dr_m, dr_s, 6),
                "n_success": n_success,
                "n_failed": n_failed,
            }
        )
    return pd.DataFrame(rows)


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
    print(f"method: {METHOD_LABEL}")
    print(f"NNLS backend: {NNLS_BACKEND}")
    print(f"Preserve observed entries: {PRESERVE_OBSERVED_ENTRIES}")

    Delta_original = robust_delta_sum_numpy(D0, trip_all)
    Delta_normalized_original = compute_normalized_delta(D0, trip_all)
    print("\n=== Delta for reference matrix ===")
    print(f"Delta_original_total        = {Delta_original:.6f}")
    print(f"Delta_original_normalized   = {Delta_normalized_original:.6f}")
    print(f"Delta_original_per_triangle = {Delta_original / ntri:.6f}" if ntri > 0 else "Delta_original_per_triangle = nan")

    mask_registry = build_mask_registry(D0)
    save_masked_matrices(mask_registry, labels)

    print(f"\n=== Running {METHOD_LABEL} on all masks ===")
    print(f"Missingness levels: {MISSING_FRACS}")
    print(f"Replicates per level: {REPS}")
    print(f"Total runs: {len(mask_registry)}")

    results: List[Dict[str, object]] = []
    for rec in mask_registry:
        pct = int(rec["pct_missing"])
        rep = int(rec["replicate"])
        print(f"  Processing p{pct} rep{rep:02d}, seed={rec['mask_seed']} ...")
        row = run_single_mask(
            rec=rec,
            D_reference=D0,
            labels=labels,
            trip_all=trip_all,
            ntri=ntri,
            Delta_original=Delta_original,
            Delta_normalized_original=Delta_normalized_original,
            method_label=METHOD_LABEL,
            save_completed=True,
            completed_prefix="MWproj_completed",
            verbose=False,
        )
        results.append(row)

    results_df = pd.DataFrame(results)
    detailed_path = os.path.join(TABLES_DIR, "mw_proj_all_masks_detailed.csv")
    save_csv(results_df, detailed_path)
    display_or_print(results_df, f"{METHOD_LABEL} detailed results")

    summary_numeric_df, summary_formatted_df = summarize_by_missingness(results_df)
    save_csv(summary_numeric_df, os.path.join(TABLES_DIR, "mw_proj_summary_numeric_by_missingness.csv"))
    save_csv(summary_formatted_df, os.path.join(TABLES_DIR, "mw_proj_summary_formatted_by_missingness.csv"))
    display_or_print(summary_formatted_df, f"{METHOD_LABEL} summary by missingness")

    delta_table_df = make_delta_training_style_table(
        results_df=results_df,
        Delta_original=Delta_original,
        Delta_normalized_original=Delta_normalized_original,
        ntri=ntri,
    )
    save_csv(delta_table_df, os.path.join(TABLES_DIR, "mw_proj_delta_sum_training_style.csv"))
    display_or_print(delta_table_df, "Delta sum table")

    archive_base = OUTPUT_ROOT.rstrip(os.sep)
    try:
        zip_path = shutil.make_archive(archive_base, "zip", OUTPUT_ROOT)
        print(f"\nSaved ZIP archive: {zip_path}")
    except Exception as e:
        print(f"Could not create ZIP archive: {repr(e)}")

    print("\n=== ALL PROCESSING COMPLETE ===")
    print(f"Reference matrix used: {used_orig_path}")
    print(f"Detailed results: {detailed_path}")
    print(f"Formatted summary: {os.path.join(TABLES_DIR, 'mw_proj_summary_formatted_by_missingness.csv')}")
    print(f"Numeric summary: {os.path.join(TABLES_DIR, 'mw_proj_summary_numeric_by_missingness.csv')}")
    print(f"Delta table: {os.path.join(TABLES_DIR, 'mw_proj_delta_sum_training_style.csv')}")


if __name__ == "__main__":
    main()
