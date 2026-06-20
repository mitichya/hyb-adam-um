# -*- coding: utf-8 -*-
"""
Run Hyb-Adam-UM on two synthetic 30x30 reference matrices.

Input expected from the generator script:
    synthetic30_tree_distances/reference_matrices/
        Dref_synthetic30_ultrametric_Tlabels.csv
        Dref_synthetic30_additive_nonultrametric_Tlabels.csv

Output:
    hyb_adam_um_synthetic30_outputs/
        tables/
            hyb_adam_um_synthetic30_all_detailed.csv
            hyb_adam_um_synthetic30_summary_numeric.csv
            hyb_adam_um_synthetic30_summary_formatted.csv
            hyb_adam_um_synthetic30_paper_table.csv
            mask_registry_synthetic30.csv
        completed_matrices/
        masked_matrices/
        logs/

This script is intentionally Hyb-Adam-UM only. It is designed for the
reviewer-response synthetic scalability / robustness experiment.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CONFIG
# ============================================================

SYNTHETIC_ROOT = Path("synthetic30_tree_distances")

REFERENCE_FILES = {
    "synthetic_ultrametric": SYNTHETIC_ROOT / "reference_matrices" / "Dref_synthetic30_ultrametric_Tlabels.csv",
    "synthetic_additive_nonultrametric": SYNTHETIC_ROOT / "reference_matrices" / "Dref_synthetic30_additive_nonultrametric_Tlabels.csv",
}

OUTPUT_ROOT = Path("hyb_adam_um_synthetic30_outputs")
MASKED_DIR = OUTPUT_ROOT / "masked_matrices"
COMPLETED_DIR = OUTPUT_ROOT / "completed_matrices"
TABLES_DIR = OUTPUT_ROOT / "tables"
LOGS_DIR = OUTPUT_ROOT / "logs"
EPOCH_LOGS_DIR = LOGS_DIR / "epoch_logs"

MISSING_FRACS: Tuple[float, ...] = (0.30, 0.50, 0.65, 0.85)
REPS: int = 30
BASE_SEED: int = 20260606

# Missing marker inside incomplete matrices.
MISSING_VAL = -1.0

# Robust Delta constants.
OMEGA = 2.0
EPS_NUM = 1.0e-12

# Set this to True only if you want epoch messages in terminal.
PRINT_EPOCH_PROGRESS_TO_CONSOLE = False


@dataclass(frozen=True)
class OptimizerConfig:
    epochs: int = 5000
    print_every: int = 500

    lr_init: float = 0.02
    lr_milestones: Tuple[int, ...] = (700, 2000, 4000)
    lr_factor: float = 0.5
    lr_min: float = 1.0e-4
    sched_patience_blocks: int = 7

    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = 5.0
    h_central_diff: float = 1.0e-5

    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1.0e-8

    seed_hyb: int = 42

    init_strategy: str = "two_way_mean"
    max_distance_factor: Optional[float] = 1.50
    normalize_objective_by_triplets: bool = True

    convergence_rel_tol: float = 1.0e-7
    convergence_patience_checks: int = 5
    early_stop: bool = False


BASE_OPT_CONFIG = OptimizerConfig()


# ============================================================
# I/O HELPERS
# ============================================================

def ensure_dirs() -> None:
    for d in (OUTPUT_ROOT, MASKED_DIR, COMPLETED_DIR, TABLES_DIR, LOGS_DIR, EPOCH_LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def safe_tag(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def append_text_log(path: Optional[Path], message: str, also_print: bool = False) -> None:
    if also_print:
        print(message)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def save_epoch_log(records: List[Dict[str, object]], path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def load_distance_matrix_from_csv(path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load a square distance matrix from CSV.

    Supported:
    - first column = row labels, header = column labels;
    - pure numeric square matrix.
    """
    path = Path(path)

    # Usual labeled matrix.
    try:
        df = pd.read_csv(path, index_col=0)
        num = df.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if num.shape[0] == num.shape[1] and num.shape[0] > 1:
            return num.to_numpy(dtype=float), [str(x) for x in num.index.to_list()]
    except Exception:
        pass

    # Pure numeric matrix.
    df = pd.read_csv(path, header=None)
    num = df.apply(pd.to_numeric, errors="coerce")
    num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if num.shape[0] != num.shape[1] or num.shape[0] <= 1:
        raise ValueError(f"Could not read a square numeric matrix from {path}")
    labels = [f"T{i + 1}" for i in range(num.shape[0])]
    return num.to_numpy(dtype=float), labels


def sanitize_reference_distance_matrix(D: np.ndarray, name: str) -> np.ndarray:
    M = np.asarray(D, dtype=float).copy()
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be square, got shape {M.shape}")
    if not np.isfinite(M).all():
        raise ValueError(f"{name} contains NaN/Inf")
    n = M.shape[0]
    off = ~np.eye(n, dtype=bool)
    if np.any(M[off] < 0):
        raise ValueError(f"{name} contains negative off-diagonal distances")

    M = 0.5 * (M + M.T)
    M = np.maximum(M, 0.0)
    np.fill_diagonal(M, 0.0)
    return M


# ============================================================
# PAIR / MASK HELPERS
# ============================================================

def lower_pairs(n: int) -> np.ndarray:
    i, j = np.tril_indices(n, k=-1)
    return np.column_stack([i, j]).astype(np.int32)


def values_on_pairs(M: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.array([], dtype=float)
    return np.asarray(M[pairs[:, 0], pairs[:, 1]], dtype=float)


def symmetrize_with_missing(D: np.ndarray) -> np.ndarray:
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


def simulate_missing(D_full: np.ndarray, frac_missing: float, mask_seed: int) -> Dict[str, object]:
    """
    Mask lower-triangle entries and mirror them symmetrically.

    The same mask_seed can be used for both matrix types, so the ultrametric
    and additive tests are directly comparable.
    """
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

    return {
        "D_inc": D_inc,
        "observed_pairs": observed_pairs,
        "missing_pairs": missing_pairs,
        "n_pairs_total": int(m_total),
        "n_missing": int(len(missing_pairs)),
        "n_observed": int(len(observed_pairs)),
        "missingness_actual": float(len(missing_pairs) / m_total if m_total else np.nan),
        "mask_seed": int(mask_seed),
    }


def build_mask_registry_for_matrix(D0: np.ndarray, matrix_type: str) -> List[Dict[str, object]]:
    registry: List[Dict[str, object]] = []
    for frac in MISSING_FRACS:
        pct = int(round(100 * frac))
        for rep in range(1, REPS + 1):
            # Same seed across matrix types for fair comparison.
            mask_seed = BASE_SEED + 10000 + pct * 100 + rep
            rec = simulate_missing(D0, frac, mask_seed)
            rec.update(
                {
                    "matrix_type": matrix_type,
                    "frac_requested": float(frac),
                    "pct_missing": pct,
                    "replicate": rep,
                }
            )
            registry.append(rec)
    return registry


def save_masked_matrices(mask_registry: List[Dict[str, object]], labels: List[str]) -> None:
    for rec in mask_registry:
        matrix_type = str(rec["matrix_type"])
        pct = int(rec["pct_missing"])
        rep = int(rec["replicate"])
        seed = int(rec["mask_seed"])

        out_dir = MASKED_DIR / matrix_type
        out_dir.mkdir(parents=True, exist_ok=True)

        fn = f"Dobs_{matrix_type}_p{pct}_rep{rep:02d}_seed{seed}.csv"
        pd.DataFrame(rec["D_inc"], index=labels, columns=labels).to_csv(out_dir / fn)
        rec["masked_file"] = str(out_dir / fn)


# ============================================================
# ROBUST DELTA
# ============================================================

def robust_delta_per_triplet_numpy(M: np.ndarray, triplets: np.ndarray) -> np.ndarray:
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
# METRICS
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
        return {"max_abs_error_observed": float("nan"), "mean_abs_error_observed": float("nan")}
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


def fmt_pm(mean_val: float, std_val: float, decimals: int = 4) -> str:
    if not np.isfinite(mean_val):
        return "N/A"
    if not np.isfinite(std_val):
        return f"{mean_val:.{decimals}f} ± N/A"
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


# ============================================================
# HYB-ADAM-UM CORE
# ============================================================

def _observed_offdiag_values(M_in: np.ndarray) -> np.ndarray:
    M = np.asarray(M_in, dtype=float)
    lower_mask = np.tril(np.ones(M.shape, dtype=bool), k=-1)
    vals = M[lower_mask & (M >= 0.0)]
    vals = vals[np.isfinite(vals)]
    return vals.astype(np.float64)


def _observed_row_means(M_in: np.ndarray, fallback: float) -> np.ndarray:
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
    strategy: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    obs_vals = _observed_offdiag_values(M_in)
    if len(obs_vals) > 0:
        global_mean = float(np.mean(obs_vals))
        global_median = float(np.median(obs_vals))
        obs_max = float(np.max(obs_vals))
    else:
        global_mean = global_median = obs_max = 1.0

    if len(missing_pairs) == 0:
        return np.array([], dtype=np.float64), {
            "initial_global_mean": global_mean,
            "initial_global_median": global_median,
            "observed_max_distance": obs_max,
        }

    strategy_norm = str(strategy).strip().lower()

    if strategy_norm in {"global", "mean", "global_mean", "old"}:
        x0 = np.full(len(missing_pairs), global_mean, dtype=np.float64)
    elif strategy_norm in {"two_way", "two_way_mean", "row_mean", "rowcol", "row_col_mean"}:
        row_means = _observed_row_means(M_in, fallback=global_mean)
        i, j = missing_pairs[:, 0], missing_pairs[:, 1]
        pair_mean = 0.5 * (row_means[i] + row_means[j])
        x0 = 0.75 * pair_mean + 0.25 * global_mean
    else:
        raise ValueError(f"Unknown init_strategy={strategy!r}")

    x0 = np.asarray(x0, dtype=np.float64)
    x0[~np.isfinite(x0)] = global_mean
    x0 = np.maximum(x0, 0.0)

    return x0, {
        "initial_global_mean": global_mean,
        "initial_global_median": global_median,
        "observed_max_distance": obs_max,
    }


def setup_problem_for_hyb_adam(M_in: np.ndarray, triplets: np.ndarray, init_strategy: str):
    n = M_in.shape[0]
    lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)

    given_mask_lower = lower_mask & (M_in >= 0.0)
    missing_mask_lower = lower_mask & (M_in < 0.0)

    given_pairs = np.array(np.where(given_mask_lower)).T.astype(np.int32)
    missing_pairs = np.array(np.where(missing_mask_lower)).T.astype(np.int32)

    given_vals = M_in[given_mask_lower].astype(np.float64)
    x0, init_stats = make_initial_missing_values(M_in, missing_pairs, init_strategy)

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

    return x0, missing_pairs, triplets, assemble_full, init_stats


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


def hyb_adam_um_impute(
    D_in: np.ndarray,
    trip_all: np.ndarray,
    opt: OptimizerConfig,
    log_path: Optional[Path] = None,
    epoch_log_path: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Hyb-Adam-UM optimizer log\n")
            f.write(json.dumps(asdict(opt), ensure_ascii=False) + "\n")

    x0, missing_pairs, all_triplets, assemble_full, init_stats = setup_problem_for_hyb_adam(
        D_in,
        trip_all,
        opt.init_strategy,
    )

    ntrip = max(int(len(all_triplets)), 1)
    obs_max = float(init_stats.get("observed_max_distance", np.nan))
    upper_bound: Optional[float]
    if opt.max_distance_factor is not None and np.isfinite(obs_max) and obs_max > 0:
        upper_bound = float(opt.max_distance_factor) * obs_max
    else:
        upper_bound = None

    def objective_delta_raw(xvec: np.ndarray) -> float:
        return robust_delta_sum_numpy(assemble_full(xvec), all_triplets)

    def objective_for_grad(xvec: np.ndarray) -> float:
        val = objective_delta_raw(xvec)
        if opt.normalize_objective_by_triplets:
            val /= ntrip
        return float(val)

    if x0.size == 0:
        M0 = assemble_full(x0)
        delta0 = objective_delta_raw(x0)
        return M0, {
            "success": True,
            "optimized_variables": 0,
            "Delta_init": float(delta0),
            "Delta_final": float(delta0),
            "Delta_reduction_percent": 0.0,
            "best_epoch": 0,
            "epochs_used": 0,
            "convergence_epoch": 0,
            "final_learning_rate": opt.lr_init,
            "upper_bound": float(upper_bound) if upper_bound is not None else float("nan"),
            **init_stats,
        }

    x = x0.copy()
    if upper_bound is not None:
        x = np.minimum(x, upper_bound)

    Delta_init = float(objective_delta_raw(x))
    obj_init = float(objective_for_grad(x))

    m = np.zeros_like(x)
    v = np.zeros_like(x)
    lr = float(opt.lr_init)

    best_x = x.copy()
    best_obj = obj_init
    best_delta = Delta_init
    best_epoch = 0

    last_block_best = obj_init
    no_improve_blocks = 0

    convergence_epoch: Optional[int] = None
    convergence_no_improve_checks = 0
    convergence_best_at_check = obj_init

    epoch_records: List[Dict[str, object]] = []

    append_text_log(
        log_path,
        f"Initial Delta={Delta_init:.6f}; optimized_variables={x.size}; upper_bound={upper_bound}",
        also_print=verbose,
    )

    for epoch in range(1, opt.epochs + 1):
        g = central_diff_grad(x, objective_for_grad, h=opt.h_central_diff)

        if opt.weight_decay > 0:
            g = g + opt.weight_decay * x

        if opt.clip_grad_norm is not None:
            g_norm = float(np.linalg.norm(g))
            if g_norm > opt.clip_grad_norm and g_norm > 0:
                g = g * (opt.clip_grad_norm / g_norm)

        m = opt.beta1 * m + (1.0 - opt.beta1) * g
        v = opt.beta2 * v + (1.0 - opt.beta2) * (g * g)

        m_hat = m / (1.0 - opt.beta1 ** epoch)
        v_hat = v / (1.0 - opt.beta2 ** epoch)

        x -= lr * (m_hat / (np.sqrt(v_hat) + opt.adam_eps))
        x = np.maximum(x, 0.0)
        if upper_bound is not None:
            x = np.minimum(x, upper_bound)

        if epoch in opt.lr_milestones and lr > opt.lr_min + 1.0e-12:
            lr = max(opt.lr_min, lr * opt.lr_factor)

        checkpoint = (epoch == 1) or (epoch % opt.print_every == 0) or (epoch == opt.epochs)
        if checkpoint:
            current_obj = float(objective_for_grad(x))
            current_delta = float(objective_delta_raw(x))

            if current_obj < best_obj - 1.0e-12:
                best_obj = current_obj
                best_delta = current_delta
                best_x = x.copy()
                best_epoch = epoch

            epoch_records.append(
                {
                    "epoch": epoch,
                    "Delta": current_delta,
                    "best_Delta": best_delta,
                    "objective_for_grad": current_obj,
                    "best_epoch": best_epoch,
                    "learning_rate": lr,
                    "optimized_variables": int(x.size),
                }
            )

            append_text_log(
                log_path,
                f"epoch {epoch:05d}: Delta={current_delta:.6f}; best={best_delta:.6f}; lr={lr:.6g}",
                also_print=verbose and PRINT_EPOCH_PROGRESS_TO_CONSOLE,
            )

            # Plateau LR schedule.
            if current_obj < last_block_best - 1.0e-12:
                no_improve_blocks = 0
                last_block_best = current_obj
            else:
                no_improve_blocks += 1
                if no_improve_blocks >= opt.sched_patience_blocks and lr > opt.lr_min + 1.0e-12:
                    lr = max(opt.lr_min, lr * opt.lr_factor)
                    no_improve_blocks = 0

            # Convergence diagnostics.
            denom = max(abs(convergence_best_at_check), EPS_NUM)
            rel_improvement = (convergence_best_at_check - best_obj) / denom
            if rel_improvement > opt.convergence_rel_tol:
                convergence_best_at_check = best_obj
                convergence_no_improve_checks = 0
            else:
                convergence_no_improve_checks += 1
                if convergence_epoch is None and convergence_no_improve_checks >= opt.convergence_patience_checks:
                    convergence_epoch = epoch
                    if opt.early_stop:
                        break

    if convergence_epoch is None:
        convergence_epoch = best_epoch if best_epoch > 0 else opt.epochs

    save_epoch_log(epoch_records, epoch_log_path)

    M_best = assemble_full(best_x)
    M_best = 0.5 * (M_best + M_best.T)
    M_best = np.maximum(M_best, 0.0)
    np.fill_diagonal(M_best, 0.0)

    if abs(Delta_init) > EPS_NUM:
        Delta_reduction_percent = 100.0 * (Delta_init - best_delta) / Delta_init
    else:
        Delta_reduction_percent = float("nan")

    info = {
        "success": bool(np.isfinite(best_delta)),
        "optimized_variables": int(x0.size),
        "Delta_init": float(Delta_init),
        "Delta_final": float(best_delta),
        "Delta_reduction_percent": float(Delta_reduction_percent),
        "best_epoch": int(best_epoch),
        "epochs_used": int(opt.epochs),
        "convergence_epoch": int(convergence_epoch),
        "final_learning_rate": float(lr),
        "upper_bound": float(upper_bound) if upper_bound is not None else float("nan"),
        "init_strategy": opt.init_strategy,
        **init_stats,
    }

    return M_best, info


# ============================================================
# RUNNER
# ============================================================

def finalize_completed_matrix(
    D_hat: np.ndarray,
    D_incomplete: np.ndarray,
    observed_pairs: np.ndarray,
) -> np.ndarray:
    M = np.asarray(D_hat, dtype=float).copy()
    M = 0.5 * (M + M.T)
    M = np.maximum(M, 0.0)
    np.fill_diagonal(M, 0.0)

    # Re-impose observed entries exactly.
    if observed_pairs.size > 0:
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


def run_single_mask(
    rec: Dict[str, object],
    D_reference: np.ndarray,
    labels: List[str],
    trip_all: np.ndarray,
    ntri: int,
    Delta_original: float,
    Delta_normalized_original: float,
    opt: OptimizerConfig,
) -> Dict[str, object]:
    matrix_type = str(rec["matrix_type"])
    pct = int(rec["pct_missing"])
    rep = int(rec["replicate"])
    seed = int(rec["mask_seed"])

    D_inc = np.asarray(rec["D_inc"], dtype=float)
    missing_pairs = np.asarray(rec["missing_pairs"], dtype=np.int32)
    observed_pairs = np.asarray(rec["observed_pairs"], dtype=np.int32)

    row: Dict[str, object] = {
        "matrix_type": matrix_type,
        "method": "Hyb-Adam-UM",
        "pct_missing": pct,
        "frac_missing_requested": float(rec["frac_requested"]),
        "missingness_actual": float(rec["missingness_actual"]),
        "replicate": rep,
        "mask_seed": seed,
        "n_pairs_total": int(rec["n_pairs_total"]),
        "n_missing": int(rec["n_missing"]),
        "n_observed": int(rec["n_observed"]),
        "optimized_variables": int(rec["n_missing"]),
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

    run_tag = safe_tag(f"{matrix_type}_p{pct}_rep{rep:02d}_seed{seed}")
    log_path = LOGS_DIR / f"{run_tag}.log"
    epoch_log_path = EPOCH_LOGS_DIR / f"{run_tag}_epochs.csv"

    t0 = time.perf_counter()
    try:
        D_completed_raw, info = hyb_adam_um_impute(
            D_inc,
            trip_all=trip_all,
            opt=opt,
            log_path=log_path,
            epoch_log_path=epoch_log_path,
            verbose=False,
        )

        D_completed = finalize_completed_matrix(D_completed_raw, D_inc, observed_pairs)
        runtime_seconds = time.perf_counter() - t0

        sanity = observed_sanity_metrics(D_completed, D_reference, observed_pairs)
        if np.isfinite(sanity["max_abs_error_observed"]) and sanity["max_abs_error_observed"] > 1.0e-10:
            raise RuntimeError(
                f"Observed entries were modified: max_abs_error_observed={sanity['max_abs_error_observed']:.6g}"
            )

        miss_metrics = completion_metrics_on_missing(D_completed, D_reference, missing_pairs)

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
            and bool(info.get("success", True))
        )
        success_valid_matrix = bool(is_valid_completed_distance_matrix(D_completed, D_reference))
        objective_improved = bool(
            np.isfinite(Delta_init)
            and np.isfinite(Delta_final)
            and Delta_final <= Delta_init * (1.0 - 1.0e-8)
        )
        success = bool(success_numerical and success_valid_matrix)

        completed_dir = COMPLETED_DIR / matrix_type
        completed_dir.mkdir(parents=True, exist_ok=True)
        completed_file = completed_dir / f"HybAdamUM_completed_{matrix_type}_p{pct}_rep{rep:02d}_seed{seed}.csv"
        pd.DataFrame(D_completed, index=labels, columns=labels).to_csv(completed_file)

        row.update(
            {
                "success": success,
                "success_numerical": success_numerical,
                "success_valid_matrix": success_valid_matrix,
                "objective_improved": objective_improved,
                "runtime_seconds": float(runtime_seconds),
                "completed_file": str(completed_file),
                "run_log_file": str(log_path),
                "epoch_log_file": str(epoch_log_path),
                "Delta_original": float(Delta_original),
                "Delta_normalized_original": float(Delta_normalized_original),
                "Delta_total_completed": float(Delta_total),
                "Delta_normalized_completed": float(Delta_norm),
                "Delta_per_triangle_completed": float(Delta_per_triangle),
                "Delta_relative_to_original": float(Delta_relative),
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
                "run_log_file": str(log_path),
                "epoch_log_file": str(epoch_log_path),
                "Delta_original": float(Delta_original),
                "Delta_normalized_original": float(Delta_normalized_original),
                "Delta_total_completed": float("nan"),
                "Delta_normalized_completed": float("nan"),
                "Delta_per_triangle_completed": float("nan"),
                "Delta_relative_to_original": float("nan"),
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
                "upper_bound": float("nan"),
                "initial_global_mean": float("nan"),
                "initial_global_median": float("nan"),
                "observed_max_distance": float("nan"),
                "optimizer_config_json": json.dumps(asdict(opt), ensure_ascii=False),
            }
        )
        append_text_log(log_path, f"FAILED: {repr(e)}", also_print=True)

    return row


# ============================================================
# SUMMARIES
# ============================================================

SUMMARY_NUMERIC_COLUMNS = (
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
)


def summarize_by_matrix_and_missingness(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows: List[Dict[str, object]] = []
    formatted_rows: List[Dict[str, object]] = []

    group_cols = ["matrix_type", "pct_missing"]

    for (matrix_type, pct), group0 in results_df.groupby(group_cols, sort=True):
        success_mask = group0["success"].astype(bool)
        group = group0[success_mask].copy()

        n_success = int(success_mask.sum())
        n_failed = int(len(group0) - n_success)
        n_objective_improved = int(group0["objective_improved"].astype(bool).sum())
        n_not_improved = int(len(group0) - n_objective_improved)

        num_row: Dict[str, object] = {
            "matrix_type": matrix_type,
            "pct_missing": int(pct),
            "n_runs": int(len(group0)),
            "n_success": n_success,
            "n_failed": n_failed,
            "n_objective_improved": n_objective_improved,
            "n_not_improved": n_not_improved,
        }

        fmt_row: Dict[str, object] = {
            "Matrix type": matrix_type,
            "% Missing": f"{int(pct)}%",
            "n_runs": int(len(group0)),
            "n_success": n_success,
            "n_failed": n_failed,
            "Success": f"{n_success}/{len(group0)}",
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

        decimals = {
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
        }

        for col, dec in decimals.items():
            fmt_row[col] = fmt_pm(num_row[f"{col}_mean"], num_row[f"{col}_std"], dec)

        numeric_rows.append(num_row)
        formatted_rows.append(fmt_row)

    return pd.DataFrame(numeric_rows), pd.DataFrame(formatted_rows)


def make_paper_table(summary_numeric_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact reviewer/paper table.

    RMSE and MAE are multiplied by 100, so they are reported as ×10^{-2},
    consistent with your main paper tables.
    """
    rows = []
    for _, r in summary_numeric_df.iterrows():
        rows.append(
            {
                "Matrix type": r["matrix_type"],
                "Missingness": f"{int(r['pct_missing'])}%",
                "RMSE_miss (x10^-2)": fmt_pm(100.0 * r["RMSE_miss_mean"], 100.0 * r["RMSE_miss_std"], 2),
                "MAE_miss (x10^-2)": fmt_pm(100.0 * r["MAE_miss_mean"], 100.0 * r["MAE_miss_std"], 2),
                "Pearson_miss": fmt_pm(r["Pearson_miss_mean"], r["Pearson_miss_std"], 3),
                "Spearman_miss": fmt_pm(r["Spearman_miss_mean"], r["Spearman_miss_std"], 3),
                "Time (s)": fmt_pm(r["runtime_seconds_mean"], r["runtime_seconds_std"], 2),
                "Success": f"{int(r['n_success'])}/{int(r['n_runs'])}",
                "Delta reduction (%)": fmt_pm(r["Delta_reduction_percent_mean"], r["Delta_reduction_percent_std"], 2),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dirs()

    all_results: List[Dict[str, object]] = []
    all_mask_meta: List[Dict[str, object]] = []

    for matrix_type, ref_path in REFERENCE_FILES.items():
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Missing reference matrix for {matrix_type}: {ref_path}\n"
                "Run the synthetic matrix generator first."
            )

        D_ref_raw, labels = load_distance_matrix_from_csv(ref_path)
        D0 = sanitize_reference_distance_matrix(D_ref_raw, matrix_type)

        n = D0.shape[0]
        if len(labels) != n:
            labels = [f"T{i + 1}" for i in range(n)]

        trip_all = np.array(list(combinations(range(n), 3)), dtype=np.int32)
        ntri = n * (n - 1) * (n - 2) // 6

        Delta_original = robust_delta_sum_numpy(D0, trip_all)
        Delta_normalized_original = compute_normalized_delta(D0, trip_all)

        print("\n" + "=" * 72)
        print(f"Matrix type: {matrix_type}")
        print(f"Reference: {ref_path}")
        print(f"n = {n}")
        print(f"pairs = {n * (n - 1) // 2}")
        print(f"triplets = {ntri}")
        print(f"Delta_original = {Delta_original:.6f}")
        print(f"Delta_normalized_original = {Delta_normalized_original:.6f}")

        # Save sanitized reference.
        ref_out_dir = OUTPUT_ROOT / "reference_matrices"
        ref_out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(D0, index=labels, columns=labels).to_csv(
            ref_out_dir / f"Dref_{matrix_type}_sanitized.csv"
        )

        mask_registry = build_mask_registry_for_matrix(D0, matrix_type)
        save_masked_matrices(mask_registry, labels)

        for rec in mask_registry:
            all_mask_meta.append(
                {
                    "matrix_type": rec["matrix_type"],
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
            )

        print(f"Running Hyb-Adam-UM on {len(mask_registry)} masks for {matrix_type}...")

        for rec in mask_registry:
            pct = int(rec["pct_missing"])
            rep = int(rec["replicate"])
            print(f"  {matrix_type}: {pct}% missing, rep {rep:02d}, seed={rec['mask_seed']}")

            row = run_single_mask(
                rec=rec,
                D_reference=D0,
                labels=labels,
                trip_all=trip_all,
                ntri=ntri,
                Delta_original=Delta_original,
                Delta_normalized_original=Delta_normalized_original,
                opt=BASE_OPT_CONFIG,
            )
            all_results.append(row)

    mask_meta_df = pd.DataFrame(all_mask_meta)
    save_csv(mask_meta_df, TABLES_DIR / "mask_registry_synthetic30.csv")

    results_df = pd.DataFrame(all_results)
    detailed_path = TABLES_DIR / "hyb_adam_um_synthetic30_all_detailed.csv"
    save_csv(results_df, detailed_path)

    summary_numeric_df, summary_formatted_df = summarize_by_matrix_and_missingness(results_df)
    save_csv(summary_numeric_df, TABLES_DIR / "hyb_adam_um_synthetic30_summary_numeric.csv")
    save_csv(summary_formatted_df, TABLES_DIR / "hyb_adam_um_synthetic30_summary_formatted.csv")

    paper_table_df = make_paper_table(summary_numeric_df)
    save_csv(paper_table_df, TABLES_DIR / "hyb_adam_um_synthetic30_paper_table.csv")

    print("\n=== Paper-ready synthetic 30x30 table ===")
    print(paper_table_df.to_string(index=False))

    # Archive all outputs.
    try:
        zip_path = shutil.make_archive(str(OUTPUT_ROOT), "zip", OUTPUT_ROOT)
        print(f"\nSaved ZIP archive: {zip_path}")
    except Exception as e:
        print(f"\nCould not create ZIP archive: {repr(e)}")

    print("\n=== COMPLETE ===")
    print(f"Detailed results: {detailed_path}")
    print(f"Numeric summary: {TABLES_DIR / 'hyb_adam_um_synthetic30_summary_numeric.csv'}")
    print(f"Formatted summary: {TABLES_DIR / 'hyb_adam_um_synthetic30_summary_formatted.csv'}")
    print(f"Paper table: {TABLES_DIR / 'hyb_adam_um_synthetic30_paper_table.csv'}")


if __name__ == "__main__":
    main()
