# -*- coding: utf-8 -*-
"""
STRICT NJ* distance-matrix completion experiment
================================================

Reviewer-oriented rewrite of the previous STRICT NJ* script so that its outputs
are directly comparable with the Hyb-Adam-UM, MW-proj/MW*, and LRMC scripts.

Implemented changes
-------------------
1. Replicates increased to 30.
2. Source/reference matrix changed to:
       Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv
3. Missingness levels are aligned with the main protocol:
       30%, 50%, 65%, 85%
   The previous 90% level is removed.
4. Runtime is measured for every run and summarized as mean ± std by missingness.
5. RMSE_miss, MAE_miss, Pearson_miss, Spearman_miss are computed ONLY on entries
   hidden by the artificial mask.
6. For every missingness level the summary reports:
       n_missing / n_observed / optimized_variables
   plus NJ*-specific diagnostics such as tree_edges_final.
7. Failure counts are reported:
       n_success, n_failed
8. Sanity check on observed entries is reported:
       max_abs_error_observed, mean_abs_error_observed
9. Every detailed row contains:
       mask_seed, missingness_actual
10. Completed matrices preserve all observed entries exactly.

Notes on optimized_variables for NJ*
------------------------------------
STRICT NJ* is a constructive tree algorithm, not a continuous optimizer. Therefore
this script reports optimized_variables = 0. For transparency, it additionally
reports imputed_variables = n_missing and tree_edges_final, which is usually
2*n - 3 for a fully resolved unrooted tree.

Missing entries are encoded as -1.0.
"""

from __future__ import annotations

import os
import json
import time
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# -------------------------- CONFIG ---------------------------
# ============================================================

MISSING_VAL: float = -1.0
S_TOP: int = 15
PRESERVE_OBSERVED: bool = True

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
OUTPUT_ROOT: str = "njstar_strict_outputs"
MASKED_DIR: str = os.path.join(OUTPUT_ROOT, "masked_matrices")
COMPLETED_DIR: str = os.path.join(OUTPUT_ROOT, "completed_matrices")
TABLES_DIR: str = os.path.join(OUTPUT_ROOT, "tables")
LOGS_DIR: str = os.path.join(OUTPUT_ROOT, "logs")

# Robust-delta constants, included for cross-method diagnostics.
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


def safe_tag(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def append_text_log(path: Optional[str], message: str, also_print: bool = False) -> None:
    if also_print:
        print(message)
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def load_distance_matrix_from_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load a square distance matrix from a CSV/TXT file.

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


def symmetrize_with_missing(D: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    """Symmetrize while preserving missing pairs as MISSING_VAL."""
    M = np.asarray(D, dtype=float).copy()
    n = M.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = M[i, j], M[j, i]
            a_ok = a >= 0.0
            b_ok = b >= 0.0
            if a_ok and b_ok:
                v = 0.5 * (a + b) if abs(a - b) > tol else a
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
    Strictly validate and lightly normalize a complete reference distance matrix.

    Important: this function intentionally does NOT median-fill NaN/Inf/negative
    values. The reference matrix is the benchmark ground truth, so bad entries
    must be reported explicitly rather than silently repaired.
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


def finalize_completed_matrix(
    D_hat: np.ndarray,
    D_incomplete: np.ndarray,
    observed_pairs: np.ndarray,
    preserve_observed: bool = True,
) -> np.ndarray:
    """Finalize completed matrix while preserving observed entries exactly."""
    M = np.asarray(D_hat, dtype=float).copy()
    if not np.isfinite(M).all():
        raise ValueError("Completed matrix has non-finite entries.")
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

def simulate_missing(D_full: np.ndarray, frac_missing: float, mask_seed: int) -> Dict[str, object]:
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
# ----------------------- NJ* TREE CORE -----------------------
# ============================================================

def _add_edge(adj: Dict[str, List[Tuple[str, float]]], a: str, b: str, w: float) -> None:
    w = float(w)
    if not np.isfinite(w) or w < 0.0:
        w = 0.0
    adj.setdefault(a, []).append((b, w))
    adj.setdefault(b, []).append((a, w))


def count_undirected_edges(adj: Dict[str, List[Tuple[str, float]]]) -> int:
    seen = set()
    for u, nbrs in adj.items():
        for v, _ in nbrs:
            key = tuple(sorted((str(u), str(v))))
            seen.add(key)
    return len(seen)


def patristic_matrix_tree(adj: Dict[str, List[Tuple[str, float]]], leaf_names: List[str]) -> np.ndarray:
    leaves = leaf_names[:]
    idx = {nm: i for i, nm in enumerate(leaves)}
    n = len(leaves)
    Dp = np.zeros((n, n), dtype=float)

    for s in leaves:
        dist = {s: 0.0}
        stack = [(s, None)]
        while stack:
            u, parent = stack.pop()
            du = dist[u]
            for v, w in adj.get(u, []):
                if v == parent:
                    continue
                dist[v] = du + float(w)
                stack.append((v, u))

        si = idx[s]
        for t in leaves:
            Dp[si, idx[t]] = float(dist.get(t, np.inf))

    if not np.isfinite(Dp).all():
        raise RuntimeError("Patristic computation produced non-finite distances; tree may be disconnected.")

    Dp = 0.5 * (Dp + Dp.T)
    np.fill_diagonal(Dp, 0.0)
    Dp = np.maximum(Dp, 0.0)
    return Dp


def _to_nan(D_in: np.ndarray) -> np.ndarray:
    D = D_in.astype(float).copy()
    D[D < 0.0] = np.nan
    np.fill_diagonal(D, 0.0)
    return D


def _count_components_known_graph(Dnan: np.ndarray) -> int:
    """Connected components of the graph whose edges are finite off-diagonal distances."""
    n = Dnan.shape[0]
    fin = np.isfinite(Dnan)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if fin[i, j]:
                adj[i].append(j)
                adj[j].append(i)
    seen = [False] * n
    comps = 0
    for s in range(n):
        if seen[s]:
            continue
        comps += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comps


def _support_set_S(D: np.ndarray, x: int, y: int) -> np.ndarray:
    """S_xy = { i : d_xi and d_yi are known }. Assumes D is NaN for missing."""
    ok = np.isfinite(D[x, :]) & np.isfinite(D[y, :])
    return np.where(ok)[0]


def _criterion6_top_pairs(D: np.ndarray, s_top: int) -> List[Tuple[int, int, float]]:
    """
    Criterion (6): score6(x,y) = (sum_{i in S_xy}(d_xi + d_yi))/(|S_xy|-2) - d_xy.
    Return top-s pairs (x<y) by descending score6.
    """
    r = D.shape[0]
    F = np.isfinite(D)
    A = np.where(F, D, 0.0)

    Ff = F.astype(np.float64)
    Fi = F.astype(np.int32)

    C = Fi @ Fi.T
    m = C - 2
    sum_x = A @ Ff.T
    Rxy = sum_x + sum_x.T

    score = np.full((r, r), -np.inf, dtype=float)
    valid = np.isfinite(D) & (m >= 1)
    np.fill_diagonal(valid, False)
    score[valid] = (Rxy[valid] / m[valid]) - D[valid]

    iu, ju = np.triu_indices(r, k=1)
    vals = score[iu, ju]
    mask = np.isfinite(vals)
    if not np.any(mask):
        return []

    iu = iu[mask]
    ju = ju[mask]
    vals = vals[mask]

    if vals.size > s_top:
        keep = np.argpartition(vals, -s_top)[-s_top:]
        iu, ju, vals = iu[keep], ju[keep], vals[keep]

    order = np.argsort(vals)[::-1]
    return [(int(iu[k]), int(ju[k]), float(vals[k])) for k in order]


def _nstar_stats_7_to_11(D: np.ndarray, x: int, y: int) -> Tuple[float, int, int, float]:
    """
    Compute tie-break stats (Eqs.7-11), used lexicographically:
      1) Nbar  = Ntilde / |C_xy|
      2) C_cnt = |C_xy|
      3) Mstar = |Miss(x) Δ Miss(y)|
      4) Nprime = sum positive margins
    """
    r = D.shape[0]
    dxy = D[x, y]
    if not np.isfinite(dxy):
        return -np.inf, 0, 0, -np.inf

    miss_x = set(np.where(~np.isfinite(D[:, x]))[0].tolist())
    miss_y = set(np.where(~np.isfinite(D[:, y]))[0].tolist())
    miss_x.discard(x)
    miss_y.discard(y)
    Mstar = int(len(miss_x.symmetric_difference(miss_y)))

    Ntilde = 0.0
    C_cnt = 0
    Nprime = 0.0

    for i in range(r):
        if i == x or i == y:
            continue
        for j in range(i + 1, r):
            if j == x or j == y:
                continue

            dij = D[i, j]
            if not np.isfinite(dij):
                continue

            t1_avail = np.isfinite(D[x, i]) and np.isfinite(D[y, j])
            t2_avail = np.isfinite(D[x, j]) and np.isfinite(D[y, i])
            if not (t1_avail or t2_avail):
                continue

            C_cnt += 1

            if t1_avail:
                t1 = D[x, i] + D[y, j] - dxy - dij
                if t1 >= 0.0:
                    Ntilde += 1.0
                if t1 > 0.0:
                    Nprime += float(t1)

            if t2_avail:
                t2 = D[x, j] + D[y, i] - dxy - dij
                if t2 >= 0.0:
                    Ntilde += 1.0
                if t2 > 0.0:
                    Nprime += float(t2)

    Nbar = float(Ntilde / C_cnt) if C_cnt > 0 else 0.0
    if C_cnt == 0:
        Nprime = 0.0

    return float(Nbar), int(C_cnt), int(Mstar), float(Nprime)


def _estimate_branch_lengths_eq12(D: np.ndarray, x: int, y: int, S: np.ndarray) -> Tuple[float, float]:
    """
    Eq. (12) for NJ* with equal weights:
      l_x = d_xy/2 + w * sum_{i in S\\{x,y}} (d_xi - d_yi), w = 1/(2(|S|-2))
      l_y = d_xy - l_x
    """
    dxy = float(D[x, y])
    m = int(len(S) - 2)
    if m <= 0:
        raise RuntimeError("Eq(12) undefined: |S_xy|-2 <= 0.")

    others = [i for i in S.tolist() if i != x and i != y]
    diff_sum = float(np.sum(D[x, others] - D[y, others])) if len(others) > 0 else 0.0
    w = 1.0 / (2.0 * m)

    lx = 0.5 * dxy + w * diff_sum
    ly = dxy - lx

    # Practical safeguard: nonnegative branch lengths.
    if lx < 0.0:
        lx = 0.0
        ly = dxy
    if ly < 0.0:
        ly = 0.0
        lx = dxy

    return float(lx), float(ly)


def _reduce_matrix_eq13(D: np.ndarray, x: int, y: int, lx: float, ly: float) -> np.ndarray:
    """
    Eq. (13), lambda = 1/2:
      if both d_xi and d_yi present: d_ui = 0.5*((d_xi - lx) + (d_yi - ly))
      if only one present:          d_ui = d_xi - lx OR d_yi - ly
      if none present:              missing (NaN)
    """
    r = D.shape[0]
    keep = [k for k in range(r) if k not in (x, y)]
    Dk = D[np.ix_(keep, keep)].copy()

    du = np.full((len(keep),), np.nan, dtype=float)
    for a_idx, i in enumerate(keep):
        dxi = D[x, i]
        dyi = D[y, i]
        if np.isfinite(dxi) and np.isfinite(dyi):
            val = 0.5 * ((dxi - lx) + (dyi - ly))
            du[a_idx] = val
        elif np.isfinite(dxi):
            du[a_idx] = dxi - lx
        elif np.isfinite(dyi):
            du[a_idx] = dyi - ly
        else:
            du[a_idx] = np.nan

    du = np.where(np.isfinite(du), np.maximum(0.0, du), np.nan)

    r2 = len(keep) + 1
    Dnew = np.full((r2, r2), np.nan, dtype=float)
    Dnew[:len(keep), :len(keep)] = Dk
    Dnew[:len(keep), -1] = du
    Dnew[-1, :len(keep)] = du
    Dnew[-1, -1] = 0.0
    return Dnew


def nj_star_tree_strict(
    D_incomplete: np.ndarray,
    leaf_names: List[str],
    s_top: int = S_TOP,
) -> Tuple[Optional[Dict[str, List[Tuple[str, float]]]], str, str, Dict[str, object]]:
    """
    STRICT NJ* tree inference on an incomplete matrix.
    Returns (adjacency_or_None, status, reason, diagnostics) with status in {"OK","FAIL"}.
    """
    D = _to_nan(symmetrize_with_missing(D_incomplete))
    nodes = leaf_names[:]

    diagnostics: Dict[str, object] = {
        "initial_known_graph_components": int(_count_components_known_graph(D)),
        "joins_performed": 0,
        "n_pair_selection_failures": 0,
        "s_top": int(s_top),
        "last_active_clusters": int(len(nodes)),
        "tree_edges_final": -1,
    }

    if diagnostics["initial_known_graph_components"] > 1:
        return None, "FAIL", f"Known-distance graph is disconnected ({diagnostics['initial_known_graph_components']} components).", diagnostics

    adj: Dict[str, List[Tuple[str, float]]] = {nm: [] for nm in nodes}
    next_internal_id = 1

    while len(nodes) > 2:
        r = len(nodes)
        diagnostics["last_active_clusters"] = int(r)

        top_pairs = _criterion6_top_pairs(D, s_top)
        if not top_pairs:
            diagnostics["n_pair_selection_failures"] = int(diagnostics["n_pair_selection_failures"]) + 1
            return None, "FAIL", f"No eligible pair at r={r}: need d_xy known and |S_xy|>=3.", diagnostics

        best = None
        best_key = None
        for x, y, sc6 in top_pairs:
            S = _support_set_S(D, x, y)
            if len(S) < 3:
                continue

            Nbar, C_cnt, Mstar, Nprime = _nstar_stats_7_to_11(D, x, y)
            key = (Nbar, C_cnt, Mstar, Nprime, sc6)

            if best_key is None or key > best_key:
                best_key = key
                best = (x, y, S)

        if best is None:
            diagnostics["n_pair_selection_failures"] = int(diagnostics["n_pair_selection_failures"]) + 1
            return None, "FAIL", f"Top-s pairs existed but none had |S_xy|>=3 at r={r}.", diagnostics

        x, y, S = best
        nx, ny = nodes[x], nodes[y]

        try:
            lx, ly = _estimate_branch_lengths_eq12(D, x, y, S)
        except RuntimeError as e:
            return None, "FAIL", f"Eq(12) undefined at r={r} for ({nx},{ny}): {str(e)}", diagnostics

        u = f"U{next_internal_id}"
        next_internal_id += 1
        adj.setdefault(u, [])
        _add_edge(adj, u, nx, lx)
        _add_edge(adj, u, ny, ly)
        diagnostics["joins_performed"] = int(diagnostics["joins_performed"]) + 1

        D = _reduce_matrix_eq13(D, x, y, lx, ly)
        keep_idx = [k for k in range(r) if k not in (x, y)]
        nodes = [nodes[k] for k in keep_idx] + [u]

    a, b = nodes[0], nodes[1]
    d = D[0, 1]
    if not np.isfinite(d):
        diagnostics["last_active_clusters"] = 2
        return None, "FAIL", "Reached r=2 but final distance is missing; cannot connect last two clusters.", diagnostics
    _add_edge(adj, a, b, float(d))

    diagnostics["last_active_clusters"] = 2
    diagnostics["tree_edges_final"] = int(count_undirected_edges(adj))
    return adj, "OK", "OK", diagnostics


def nj_star_complete_strict(
    D_inc: np.ndarray,
    labels: List[str],
) -> Tuple[Optional[np.ndarray], str, str, Dict[str, object]]:
    adj, status, reason, diagnostics = nj_star_tree_strict(D_inc, labels, s_top=S_TOP)
    if status != "OK" or adj is None:
        return None, status, reason, diagnostics

    D_pat = patristic_matrix_tree(adj, labels)
    return D_pat, "OK", "OK", diagnostics


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
    method_label: str = "NJstar-STRICT",
    save_completed: bool = True,
    verbose: bool = True,
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
        "imputed_variables": int(rec["n_missing"]),
        "optimized_variables": 0,
        "success": False,
        "success_numerical": False,
        "success_valid_matrix": False,
        "status": "FAIL",
        "reason": "",
        "error_message": "",
        "masked_file": rec.get("masked_file", ""),
        "completed_file": "",
        "run_log_file": "",
        "s_top": int(S_TOP),
    }

    run_tag = safe_tag(f"NJstar_STRICT_p{pct}_rep{rep:02d}_seed{row['mask_seed']}")
    run_log_path = os.path.join(LOGS_DIR, f"{run_tag}.log")
    row["run_log_file"] = run_log_path

    t0 = time.perf_counter()
    try:
        D_completed_raw, status, reason, info = nj_star_complete_strict(D_inc, labels)
        runtime_seconds = time.perf_counter() - t0
        row.update(info)
        row["status"] = status
        row["reason"] = reason

        append_text_log(
            run_log_path,
            f"NJ* strict p={pct} rep={rep} seed={row['mask_seed']} status={status} reason={reason}",
            also_print=False,
        )

        if status != "OK" or D_completed_raw is None:
            raise RuntimeError(reason)

        D_completed = finalize_completed_matrix(
            D_completed_raw,
            D_incomplete=D_inc,
            observed_pairs=observed_pairs,
            preserve_observed=PRESERVE_OBSERVED,
        )

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
            and np.isfinite(Delta_total)
            and np.isfinite(D_completed).all()
            and status == "OK"
        )
        success_valid_matrix = bool(is_valid_completed_distance_matrix(D_completed, D_reference))
        success = bool(success_numerical and success_valid_matrix)

        completed_file = ""
        if save_completed:
            completed_file = f"NJstar_STRICT_completed_p{pct}_rep{rep:02d}_seed{row['mask_seed']}.csv"
            completed_path = os.path.join(COMPLETED_DIR, completed_file)
            pd.DataFrame(D_completed, index=labels, columns=labels).to_csv(completed_path)

        row.update(
            {
                "success": success,
                "success_numerical": success_numerical,
                "success_valid_matrix": success_valid_matrix,
                "status": "OK" if success else status,
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
                # These aliases keep NJ* summary compatible with optimizer-based tables.
                "epochs_used": int(info.get("joins_performed", -1)),
                "convergence_epoch": int(info.get("joins_performed", -1)),
                "njstar_config_json": json.dumps({"S_TOP": S_TOP, "PRESERVE_OBSERVED": PRESERVE_OBSERVED}, ensure_ascii=False),
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
                "status": "FAIL",
                "error_message": repr(e),
                "reason": row.get("reason", "") or repr(e),
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
                "joins_performed": int(row.get("joins_performed", -1)) if str(row.get("joins_performed", "")).lstrip("-").isdigit() else -1,
                "tree_edges_final": int(row.get("tree_edges_final", -1)) if str(row.get("tree_edges_final", "")).lstrip("-").isdigit() else -1,
                "epochs_used": -1,
                "convergence_epoch": -1,
                "njstar_config_json": json.dumps({"S_TOP": S_TOP, "PRESERVE_OBSERVED": PRESERVE_OBSERVED}, ensure_ascii=False),
            }
        )
        append_text_log(run_log_path, f"FAILED p{pct} rep{rep}: {repr(e)}", also_print=verbose)

    return row


# ============================================================
# -------------------------- SUMMARIES ------------------------
# ============================================================

SUMMARY_NUMERIC_COLUMNS: Tuple[str, ...] = (
    "missingness_actual",
    "n_missing",
    "n_observed",
    "imputed_variables",
    "optimized_variables",
    "RMSE_miss",
    "MAE_miss",
    "Pearson_miss",
    "Spearman_miss",
    "runtime_seconds",
    "joins_performed",
    "epochs_used",
    "convergence_epoch",
    "tree_edges_final",
    "initial_known_graph_components",
    "n_pair_selection_failures",
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

            # Use all runs for fixed mask counts and failure diagnostics; successful runs for performance/error summaries.
            source = sub_all if col in {
                "missingness_actual",
                "n_missing",
                "n_observed",
                "imputed_variables",
                "optimized_variables",
                "initial_known_graph_components",
                "n_pair_selection_failures",
            } else sub_success

            mean_val, std_val = mean_std_nan(source[col].to_numpy(dtype=float))
            numeric_row[f"{col}_mean"] = mean_val
            numeric_row[f"{col}_std"] = std_val

            decimals = 2 if col in {"runtime_seconds", "joins_performed", "epochs_used", "convergence_epoch", "tree_edges_final"} else 6
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

    # Keep labels synchronized if a numeric-only loader was used.
    if len(labels) != n:
        labels = [f"T{i + 1}" for i in range(n)]

    trip_all = np.array(list(combinations(range(n), 3)), dtype=np.int32)
    ntri = int(len(trip_all))
    Delta_original = robust_delta_sum_numpy(D0, trip_all)
    Delta_normalized_original = compute_normalized_delta(D0, trip_all)

    print("=== REFERENCE MATRIX INFO ===")
    print(f"Used file: {used_orig_path}")
    print(f"n = {n}")
    print(f"n_pairs_lower_triangle = {n * (n - 1) // 2}")
    print(f"n_triplets = {ntri}")
    print(f"Delta_original = {Delta_original:.6f}")
    print(f"Delta_normalized_original = {Delta_normalized_original:.6f}")
    print()

    mask_registry = build_mask_registry(D0)
    save_masked_matrices(mask_registry, labels)

    print(f"=== Running STRICT NJ* on {len(mask_registry)} masks ===")
    results: List[Dict[str, object]] = []
    for rec in mask_registry:
        pct = int(rec["pct_missing"])
        rep = int(rec["replicate"])
        print(f"  Processing {pct}% missing, replicate {rep:02d}, seed {rec['mask_seed']}...")
        row = run_single_mask(
            rec=rec,
            D_reference=D0,
            labels=labels,
            trip_all=trip_all,
            ntri=ntri,
            Delta_original=Delta_original,
            Delta_normalized_original=Delta_normalized_original,
            method_label="NJstar-STRICT",
            save_completed=True,
            verbose=True,
        )
        results.append(row)

    results_df = pd.DataFrame(results)
    detailed_path = os.path.join(TABLES_DIR, "njstar_strict_all_masks_detailed.csv")
    save_csv(results_df, detailed_path)

    summary_numeric, summary_formatted = make_summary_tables(results_df)
    numeric_path = os.path.join(TABLES_DIR, "njstar_strict_summary_numeric_by_missingness.csv")
    formatted_path = os.path.join(TABLES_DIR, "njstar_strict_summary_formatted_by_missingness.csv")
    save_csv(summary_numeric, numeric_path)
    save_csv(summary_formatted, formatted_path)

    metadata = {
        "method": "NJstar-STRICT",
        "reference_matrix_file": used_orig_path,
        "n_taxa": int(n),
        "n_pairs_lower_triangle": int(n * (n - 1) // 2),
        "n_triplets": int(ntri),
        "missing_fracs": list(MISSING_FRACS),
        "reps": int(REPS),
        "base_seed": int(BASE_SEED),
        "s_top": int(S_TOP),
        "preserve_observed": bool(PRESERVE_OBSERVED),
        "output_root": OUTPUT_ROOT,
        "detailed_csv": detailed_path,
        "summary_numeric_csv": numeric_path,
        "summary_formatted_csv": formatted_path,
        "Delta_original": float(Delta_original),
        "Delta_normalized_original": float(Delta_normalized_original),
        "optimized_variables_definition": "0 because STRICT NJ* is constructive, not a continuous optimizer; see imputed_variables and tree_edges_final.",
    }
    metadata_path = os.path.join(OUTPUT_ROOT, "run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved: {metadata_path}")

    display_or_print(results_df, "NJ* STRICT detailed results")
    display_or_print(summary_formatted, "NJ* STRICT summary by missingness")

    print("\n=== ALL PROCESSING COMPLETE ===")
    print(f"Masked matrices:    {MASKED_DIR}/")
    print(f"Completed matrices: {COMPLETED_DIR}/")
    print(f"Detailed results:   {detailed_path}")
    print(f"Summary numeric:    {numeric_path}")
    print(f"Summary formatted:  {formatted_path}")


if __name__ == "__main__":
    main()
