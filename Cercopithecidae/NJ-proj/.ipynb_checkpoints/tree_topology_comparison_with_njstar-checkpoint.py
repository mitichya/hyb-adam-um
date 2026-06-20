#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tree-topology comparison for completed distance matrices
=======================================================

Purpose
-------
This script evaluates whether completed distance matrices recover the reference
phylogenetic topology. This version includes discovery support for STRICT NJ*.

For every completed matrix found for the enabled methods, it:

1. reads the IQ-TREE maximum-likelihood reference tree;
2. reconstructs a Neighbor-Joining (NJ) tree from the completed matrix;
3. compares the NJ topology against the ML reference topology using unrooted
   Robinson-Foulds-style split metrics;
4. also reports patristic-distance agreement between the reconstructed NJ tree
   and the ML reference tree.

This is complementary to the ML-topology branch-length comparison script:
that script fixes the ML topology and fits branch lengths; this script lets the
completed matrix induce its own NJ topology and asks whether the topology is
closer to the ML reference tree.

Expected input files
--------------------
Reference ML tree produced by your IQ-TREE script:
    mtDNA15_ML_tree/mtDNA15_IQTREE_ML.treefile

Reference matrix / labels:
    hyb_adam_um_outputs/reference_matrix_sanitized.csv
or:
    Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv

Completed matrices from enabled methods, for example:
    hyb_adam_um_outputs/completed_matrices/HybAdamUM_completed_p30_rep01_seed55.csv
    mw_proj_outputs/completed_matrices/MWproj_completed_p30_rep01_seed55.csv
    knn_impute_outputs/completed_matrices/KNNimpute_completed_p30_rep01_seed55.csv
    lrmc_softimpute_outputs/completed_matrices/LRMC_SoftImpute_completed_p30_rep01_seed55.csv
    njstar_strict_outputs/completed_matrices/NJstar_STRICT_completed_p30_rep01_seed55.csv
    mds_smacof_outputs/completed_matrices/MDSSMACOF_completed_p30_rep01_seed55.csv

STRICT NJ* metadata is read from:
    njstar_strict_outputs/tables/njstar_strict_all_masks_detailed.csv

Dependencies
------------
    pip install numpy pandas biopython

Outputs
-------
    tree_topology_comparison/
        method_file_inventory.csv
        reference_ml_splits.csv
        reference_ml_patristic_matrix.csv
        reference_nj_from_Dref.treefile
        reference_nj_from_Dref_metrics.csv
        reconstructed_trees/*.treefile
        tree_topology_detailed_all_methods.csv
        tree_topology_summary_numeric.csv
        tree_topology_summary_meanstd.csv
        tree_topology_comparison.zip
"""

from __future__ import annotations

import os
import re
import math
import shutil
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set, FrozenSet

import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# -------------------------- CONFIG ---------------------------
# ============================================================

ML_TREE_CANDIDATES: Tuple[str, ...] = (
    "mtDNA15_ML_tree/mtDNA15_IQTREE_ML.treefile",
    "./mtDNA15_ML_tree/mtDNA15_IQTREE_ML.treefile",
    "/mnt/data/mtDNA15_ML_tree/mtDNA15_IQTREE_ML.treefile",
)

REFERENCE_MATRIX_CANDIDATES: Tuple[str, ...] = (
    "hyb_adam_um_outputs/reference_matrix_sanitized.csv",
    "./hyb_adam_um_outputs/reference_matrix_sanitized.csv",
    "/mnt/data/hyb_adam_um_outputs/reference_matrix_sanitized.csv",
    "Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv",
    "./Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv",
    "/mnt/data/Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv",
)

LABEL_MAPPING_CANDIDATES: Tuple[str, ...] = (
    "mtDNA15_ML_tree/label_mapping.tsv",
    "./mtDNA15_ML_tree/label_mapping.tsv",
    "/mnt/data/mtDNA15_ML_tree/label_mapping.tsv",
)

# Fallback order used only when a matrix is purely numeric and therefore loaded
# as T1, T2, ..., T15. This is the insertion order from build_ml_tree_iqtree.py.
KNOWN_MTDNA15_LABELS_IN_MATRIX_ORDER: Tuple[str, ...] = (
    "Allenopithecus_nigroviridis_KJ434962",
    "Cercocebus_atys_KT159932",
    "Cercocebus_chrysogaster_KC757390",
    "Cercocebus_torquatus_KJ434959",
    "Cercopithecus_aethiops_AY863426",
    "Cercopithecus_albogularis_KC757391",
    "Cercopithecus_diana_KJ434958",
    "Cercopithecus_lhoesti_KJ434957",
    "Cercopithecus_mitis_KJ434956",
    "Cercopithecus_neglectus_MW160353",
    "Chlorocebus_aethiops_C1_KU682691",
    "Chlorocebus_aethiops_MN816163",
    "Chlorocebus_cynosuros_C3_KU682693",
    "Chlorocebus_cynosuros_KM262190",
    "Chlorocebus_djamdjamensis_C5_KU682695",
)

# Optional Hyb mask registry. Useful as a fallback for common mask counts.
MASK_REGISTRY_CANDIDATES: Tuple[str, ...] = (
    "hyb_adam_um_outputs/tables/mask_registry.csv",
    "./hyb_adam_um_outputs/tables/mask_registry.csv",
    "/mnt/data/hyb_adam_um_outputs/tables/mask_registry.csv",
)

# Optional method-specific detailed result tables. The LRMC script you provided
# writes lrmc_softimpute_all_masks_detailed.csv with runtime, RMSE_miss,
# n_missing/n_observed, and convergence diagnostics.
METHOD_METADATA_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "Hyb-Adam-UM": (
        "hyb_adam_um_outputs/tables/hyb_adam_um_all_masks_detailed.csv",
        "./hyb_adam_um_outputs/tables/hyb_adam_um_all_masks_detailed.csv",
        "/mnt/data/hyb_adam_um_outputs/tables/hyb_adam_um_all_masks_detailed.csv",
    ),
    "MW-proj": (
        "mw_proj_outputs/tables/mw_proj_all_masks_detailed.csv",
        "./mw_proj_outputs/tables/mw_proj_all_masks_detailed.csv",
        "/mnt/data/mw_proj_outputs/tables/mw_proj_all_masks_detailed.csv",
    ),
    "LRMC-SoftImpute": (
        "lrmc_softimpute_outputs/tables/lrmc_softimpute_all_masks_detailed.csv",
        "./lrmc_softimpute_outputs/tables/lrmc_softimpute_all_masks_detailed.csv",
        "/mnt/data/lrmc_softimpute_outputs/tables/lrmc_softimpute_all_masks_detailed.csv",
    ),
    "KNN-Impute": (
        "knn_impute_outputs/tables/knn_impute_all_masks_detailed.csv",
        "./knn_impute_outputs/tables/knn_impute_all_masks_detailed.csv",
        "/mnt/data/knn_impute_outputs/tables/knn_impute_all_masks_detailed.csv",
    ),
    "NJstar-STRICT": (
        "njstar_strict_outputs/tables/njstar_strict_all_masks_detailed.csv",
        "./njstar_strict_outputs/tables/njstar_strict_all_masks_detailed.csv",
        "/mnt/data/njstar_strict_outputs/tables/njstar_strict_all_masks_detailed.csv",
    ),
    "MDS-SMACOF": (
        "mds_smacof_outputs/tables/mds_smacof_all_masks_detailed.csv",
        "./mds_smacof_outputs/tables/mds_smacof_all_masks_detailed.csv",
        "/mnt/data/mds_smacof_outputs/tables/mds_smacof_all_masks_detailed.csv",
    ),
}

# Use None to evaluate every method for which completed matrices are found.
# Example for exactly three methods:
#     METHODS_TO_COMPARE = ("Hyb-Adam-UM", "MW-proj", "NJstar-STRICT")
# Example for STRICT NJ* only:
#     METHODS_TO_COMPARE = ("NJstar-STRICT",)
METHODS_TO_COMPARE: Optional[Tuple[str, ...]] = None

OUT_DIR = "tree_topology_comparison"
TREES_DIR = os.path.join(OUT_DIR, "reconstructed_trees")


@dataclass(frozen=True)
class MethodSpec:
    method: str
    search_dirs: Tuple[str, ...]
    regexes: Tuple[str, ...]


METHOD_SPECS: Tuple[MethodSpec, ...] = (
    MethodSpec(
        method="Hyb-Adam-UM",
        search_dirs=(
            "hyb_adam_um_outputs/completed_matrices",
            "hyb_adam_um_completed_matrices",
            "/mnt/data/hyb_adam_um_outputs/completed_matrices",
            "/mnt/data/hyb_adam_um_completed_matrices",
        ),
        regexes=(
            r"^HybAdamUM_completed_p(?P<pct>\d+)_rep(?P<rep>\d+)(?:_seed(?P<seed>\d+))?\.csv$",
        ),
    ),
    MethodSpec(
        method="MW-proj",
        search_dirs=(
            "mw_proj_outputs/completed_matrices",
            "mw_proj_completed_matrices",
            "/mnt/data/mw_proj_outputs/completed_matrices",
            "/mnt/data/mw_proj_completed_matrices",
        ),
        regexes=(
            r"^(?:MWproj|MWProj|MW_Proj|MW-Proj|MW-proj)_completed_p(?P<pct>\d+)_rep(?P<rep>\d+)(?:_seed(?P<seed>\d+))?\.csv$",
            r"^mw.*?proj.*?p(?P<pct>\d+).*?rep(?P<rep>\d+)(?:.*?seed(?P<seed>\d+))?\.csv$",
        ),
    ),
    MethodSpec(
        method="LRMC-SoftImpute",
        search_dirs=(
            "lrmc_softimpute_outputs/completed_matrices",
            "lrmc_softimpute_completed_matrices",
            "LRMC_outputs/completed_matrices",
            "/mnt/data/lrmc_softimpute_outputs/completed_matrices",
            "/mnt/data/lrmc_softimpute_completed_matrices",
        ),
        regexes=(
            # Main filename written by your LRMC script:
            #     LRMC_SoftImpute_completed_p30_rep01_seed55.csv
            r"^(?:LRMC_SoftImpute|LRMC-SoftImpute|LRMCSoftImpute|SoftImpute)_completed_p(?P<pct>\d+)_rep(?P<rep>\d+)(?:_seed(?P<seed>\d+))?\.csv$",
            r"^lrmc.*?p(?P<pct>\d+).*?rep(?P<rep>\d+)(?:.*?seed(?P<seed>\d+))?\.csv$",
            r"^soft.*?impute.*?p(?P<pct>\d+).*?rep(?P<rep>\d+)(?:.*?seed(?P<seed>\d+))?\.csv$",
        ),
    ),
    MethodSpec(
        method="KNN-Impute",
        search_dirs=(
            "knn_impute_outputs/completed_matrices",
            "knn_impute_completed_matrices",
            "KNN_outputs/completed_matrices",
            "/mnt/data/knn_impute_outputs/completed_matrices",
            "/mnt/data/knn_impute_completed_matrices",
        ),
        regexes=(
            r"^(?:KNN|KNNImpute|KNNImputer|KNN_Impute)_completed_p(?P<pct>\d+)_rep(?P<rep>\d+)(?:_seed(?P<seed>\d+))?\.csv$",
            r"^knn.*?p(?P<pct>\d+).*?rep(?P<rep>\d+)(?:.*?seed(?P<seed>\d+))?\.csv$",
        ),
    ),
    MethodSpec(
        method="NJstar-STRICT",
        search_dirs=(
            "njstar_strict_outputs/completed_matrices",
            "njstar_completed_matrices",
            "NJstar_outputs/completed_matrices",
            "/mnt/data/njstar_strict_outputs/completed_matrices",
            "/mnt/data/njstar_completed_matrices",
        ),
        regexes=(
            # Main filename written by your STRICT NJ* script:
            #     NJstar_STRICT_completed_p30_rep01_seed55.csv
            r"^(?:NJstar_STRICT|NJStar_STRICT|NJstar-STRICT|NJStar-STRICT|NJSTRICT|NJ_Star_STRICT)_completed_p(?P<pct>\d+)_rep(?P<rep>\d+)(?:_seed(?P<seed>\d+))?\.csv$",
            r"^nj.*?strict.*?p(?P<pct>\d+).*?rep(?P<rep>\d+)(?:.*?seed(?P<seed>\d+))?\.csv$",
            r"^njstar.*?p(?P<pct>\d+).*?rep(?P<rep>\d+)(?:.*?seed(?P<seed>\d+))?\.csv$",
        ),
    ),
    MethodSpec(
        method="MDS-SMACOF",
        search_dirs=(
            "mds_smacof_outputs/completed_matrices",
            "mds_smacof_completed_matrices",
            "MDS_SMACOF_outputs/completed_matrices",
            "/mnt/data/mds_smacof_outputs/completed_matrices",
            "/mnt/data/mds_smacof_completed_matrices",
        ),
        regexes=(
            r"^(?:MDSSMACOF|MDS_SMACOF|MDS-SMACOF|SMACOF)_completed_p(?P<pct>\d+)_rep(?P<rep>\d+)(?:_seed(?P<seed>\d+))?\.csv$",
            r"^mds.*?smacof.*?p(?P<pct>\d+).*?rep(?P<rep>\d+)(?:.*?seed(?P<seed>\d+))?\.csv$",
        ),
    ),
)

TOPOLOGY_SUMMARY_METRICS: Tuple[str, ...] = (
    "RF",
    "nRF",
    "shared_splits",
    "ref_splits",
    "est_splits",
    "split_precision",
    "split_recall",
    "split_F1",
    "nj_vs_ML_pat_MAE",
    "nj_vs_ML_pat_RMSE",
    "nj_vs_ML_pat_Pearson",
    "nj_vs_ML_pat_Spearman",
    "input_vs_ML_pat_MAE",
    "input_vs_ML_pat_RMSE",
    "input_vs_ML_pat_Pearson",
    "input_vs_ML_pat_Spearman",
)

METADATA_COLUMNS_TO_MERGE: Tuple[str, ...] = (
    "missingness_actual",
    "n_pairs_total",
    "n_missing",
    "n_observed",
    "optimized_variables",
    "imputed_variables",
    "n_neighbors_requested",
    "n_neighbors_effective",
    "knn_weights",
    "knn_metric",
    "s_top",
    "status",
    "reason",
    "joins_performed",
    "tree_edges_final",
    "initial_known_graph_components",
    "n_pair_selection_failures",
    "RMSE_miss",
    "MAE_miss",
    "Pearson_miss",
    "Spearman_miss",
    "runtime_seconds",
    "success",
    "success_numerical",
    "success_valid_matrix",
    "iters_used",
    "epochs_used",
    "convergence_epoch",
    "final_rank",
    "Delta_normalized_completed",
    "Delta_relative_to_original",
    "max_abs_error_observed",
    "mean_abs_error_observed",
)


# ============================================================
# ----------------------- BASIC HELPERS -----------------------
# ============================================================


def first_existing(candidates: Sequence[str], what: str) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {what}. Tried: {list(candidates)}")


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TREES_DIR, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def display_or_print(df: pd.DataFrame, title: Optional[str] = None) -> None:
    if title:
        print(f"\n=== {title} ===")
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string(index=False))


def safe_tag(text: str) -> str:
    out: List[str] = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "item"


def upper_pairs(n: int) -> np.ndarray:
    i, j = np.triu_indices(n, k=1)
    return np.column_stack([i, j]).astype(np.int32)


def values_on_pairs(M: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return np.asarray(M[pairs[:, 0], pairs[:, 1]], dtype=float)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]
    if a.size < 2 or np.std(a) <= 0.0 or np.std(b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]
    if a.size < 2:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    return pearson_corr(ra, rb)


def vector_metrics(pred: np.ndarray, true: np.ndarray, prefix: str) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(true)
    pred = pred[ok]
    true = true[ok]
    if pred.size == 0:
        return {
            f"{prefix}_MAE": float("nan"),
            f"{prefix}_RMSE": float("nan"),
            f"{prefix}_Pearson": float("nan"),
            f"{prefix}_Spearman": float("nan"),
        }
    diff = pred - true
    return {
        f"{prefix}_MAE": float(np.mean(np.abs(diff))),
        f"{prefix}_RMSE": float(np.sqrt(np.mean(diff * diff))),
        f"{prefix}_Pearson": pearson_corr(pred, true),
        f"{prefix}_Spearman": spearman_corr(pred, true),
    }


def mean_std_nan(x: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean_val, std_val


def fmt_pm(mean_val: float, std_val: float, decimals: int = 6) -> str:
    if not np.isfinite(mean_val):
        return "nan"
    if not np.isfinite(std_val):
        return f"{mean_val:.{decimals}f} ± nan"
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


# ============================================================
# --------------------- MATRIX LOADING ------------------------
# ============================================================


def load_distance_matrix_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load a square distance matrix from CSV with row/column labels when present."""
    # Preferred case: pandas matrix written with index and columns.
    try:
        df = pd.read_csv(path, sep=None, engine="python", index_col=0)
        num = df.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if num.shape[0] == num.shape[1] and num.shape[0] > 1:
            row_labels = [str(x) for x in num.index.to_list()]
            col_labels = [str(x) for x in num.columns.to_list()]
            if set(row_labels) == set(col_labels):
                num = num.loc[col_labels, col_labels]
                labels = col_labels
            else:
                labels = row_labels
            return num.to_numpy(dtype=float), labels
    except Exception:
        pass

    # Fallback: numeric CSV with no labels.
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        num = df.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if num.shape[0] == num.shape[1] and num.shape[0] > 1:
            labels = [f"T{i + 1}" for i in range(num.shape[0])]
            return num.to_numpy(dtype=float), labels
    except Exception:
        pass

    # Final fallback for whitespace/comma numeric text.
    try:
        M = np.loadtxt(path, delimiter=",")
    except Exception:
        M = np.loadtxt(path)
    M = np.asarray(M, dtype=float)
    labels = [f"T{i + 1}" for i in range(M.shape[0])]
    return M, labels


def sanitize_complete_distance_matrix(D: np.ndarray, name: str) -> np.ndarray:
    M = np.asarray(D, dtype=float).copy()
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be square, got {M.shape}")
    if not np.isfinite(M).all():
        raise ValueError(f"{name} contains NaN/Inf")

    n = M.shape[0]
    off = ~np.eye(n, dtype=bool)
    if np.any(M[off] < -1e-12):
        raise ValueError(f"{name} contains negative off-diagonal distances; is it still incomplete?")

    M = 0.5 * (M + M.T)
    M = np.maximum(M, 0.0)
    np.fill_diagonal(M, 0.0)
    return M


# ============================================================
# ---------------------- LABEL MATCHING -----------------------
# ============================================================


ACCESSION_RE = re.compile(r"\b[A-Z]{2}\d{6}\b")


def is_generic_t_labels(labels: Sequence[str]) -> bool:
    labels = [str(x) for x in labels]
    return labels == [f"T{i + 1}" for i in range(len(labels))]


def read_label_mapping_tsv(n_expected: int, tree_labels: Sequence[str]) -> Optional[List[str]]:
    tree_set = set(map(str, tree_labels))
    for path in LABEL_MAPPING_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, sep="\t")
            if "new_label" not in df.columns:
                continue
            labels = [str(x) for x in df["new_label"].tolist()]
            if len(labels) == n_expected and set(labels) == tree_set:
                return labels
        except Exception:
            continue
    return None


def resolve_generic_matrix_labels(
    matrix_labels: Sequence[str],
    tree_labels: Sequence[str],
    *,
    verbose: bool = True,
) -> Tuple[List[str], bool]:
    matrix_labels = [str(x) for x in matrix_labels]
    tree_labels = [str(x) for x in tree_labels]

    if not is_generic_t_labels(matrix_labels):
        return matrix_labels, False

    recovered = read_label_mapping_tsv(len(matrix_labels), tree_labels)
    if recovered is not None:
        if verbose:
            print("Generic labels T1..Tn detected; recovered taxon order from label_mapping.tsv.")
        return recovered, True

    if len(matrix_labels) == len(KNOWN_MTDNA15_LABELS_IN_MATRIX_ORDER) and set(tree_labels) == set(KNOWN_MTDNA15_LABELS_IN_MATRIX_ORDER):
        if verbose:
            print("Generic labels T1..T15 detected; using mtDNA15 accession order from build_ml_tree_iqtree.py.")
        return list(KNOWN_MTDNA15_LABELS_IN_MATRIX_ORDER), True

    return matrix_labels, False


def label_keys(label: str) -> List[str]:
    s = str(label).strip()
    keys: List[str] = []

    m = ACCESSION_RE.search(s.upper())
    if m:
        keys.append("acc:" + m.group(0))

    keys.append("exact:" + s)
    norm = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    keys.append("norm:" + norm)

    if m:
        species = s[: m.start()].rstrip("_ -.")
        species_norm = re.sub(r"[^A-Za-z0-9]+", "_", species).strip("_").lower()
        if species_norm:
            keys.append("species:" + species_norm)
    return keys


def build_tree_to_matrix_label_map(tree_labels: Sequence[str], matrix_labels: Sequence[str]) -> Dict[str, str]:
    tree_labels = [str(x) for x in tree_labels]
    matrix_labels = [str(x) for x in matrix_labels]

    if set(tree_labels) == set(matrix_labels):
        return {x: x for x in tree_labels}

    key_to_matrix: Dict[str, List[str]] = {}
    for lab in matrix_labels:
        for k in label_keys(lab):
            key_to_matrix.setdefault(k, []).append(lab)

    mapping: Dict[str, str] = {}
    missing: List[str] = []
    ambiguous: List[Tuple[str, str, List[str]]] = []

    for tlab in tree_labels:
        found: Optional[str] = None
        for k in label_keys(tlab):
            candidates = sorted(set(key_to_matrix.get(k, [])))
            if len(candidates) == 1:
                found = candidates[0]
                break
            if len(candidates) > 1:
                ambiguous.append((tlab, k, candidates))
        if found is None:
            missing.append(tlab)
        else:
            mapping[tlab] = found

    if missing or len(set(mapping.values())) != len(mapping):
        msg = ["Could not match tree labels to matrix labels."]
        if missing:
            msg.append("Missing tree labels: " + ", ".join(missing[:10]))
        if ambiguous:
            msg.append("Ambiguous examples: " + repr(ambiguous[:3]))
        msg.append("Tree labels example: " + repr(tree_labels[:5]))
        msg.append("Matrix labels example: " + repr(matrix_labels[:5]))
        raise ValueError("\n".join(msg))

    return mapping


def reorder_matrix_to_tree_labels(
    D: np.ndarray,
    matrix_labels: Sequence[str],
    tree_labels: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    resolved_labels, _ = resolve_generic_matrix_labels(matrix_labels, tree_labels)
    mapping = build_tree_to_matrix_label_map(tree_labels, resolved_labels)
    idx = {str(lab): i for i, lab in enumerate(resolved_labels)}
    order = [idx[mapping[tlab]] for tlab in tree_labels]
    return np.asarray(D, dtype=float)[np.ix_(order, order)], [mapping[tlab] for tlab in tree_labels]


# ============================================================
# ---------------------- TREE UTILITIES -----------------------
# ============================================================


def read_tree_labels(tree_path: str) -> List[str]:
    tree = Phylo.read(tree_path, "newick")
    labels = [str(t.name) for t in tree.get_terminals()]
    if len(labels) != len(set(labels)):
        raise ValueError("Reference tree contains duplicate terminal labels.")
    return labels


def distance_matrix_to_biopython(D: np.ndarray, labels: Sequence[str]) -> DistanceMatrix:
    D = sanitize_complete_distance_matrix(D, "NJ input distance matrix")
    names = [str(x) for x in labels]
    n = len(names)
    if D.shape != (n, n):
        raise ValueError(f"Matrix shape {D.shape} does not match {n} labels.")

    lower_tri: List[List[float]] = []
    for i in range(n):
        row = [float(D[i, j]) for j in range(i + 1)]
        row[-1] = 0.0
        lower_tri.append(row)
    return DistanceMatrix(names=names, matrix=lower_tri)


def build_nj_tree_from_matrix(D: np.ndarray, labels: Sequence[str]):
    dm = distance_matrix_to_biopython(D, labels)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)
    # Biopython may create unnamed internal nodes; that is fine for topology.
    return tree


def terminal_set_for_clade(clade) -> Set[str]:
    return {str(t.name) for t in clade.get_terminals()}


def unrooted_internal_splits(tree, leaf_labels: Sequence[str]) -> Set[FrozenSet[str]]:
    """
    Return nontrivial unrooted internal splits represented by the smaller side.

    Terminal/singleton splits are excluded. Complementary splits are normalized
    to the smaller side, so arbitrary Newick rooting does not affect the result.
    """
    all_leaves = set(map(str, leaf_labels))
    n = len(all_leaves)
    splits: Set[FrozenSet[str]] = set()

    for clade in tree.find_clades(order="preorder"):
        if clade is tree.root:
            continue
        desc = terminal_set_for_clade(clade)
        if not desc or desc == all_leaves:
            continue
        comp = all_leaves - desc
        side = desc if len(desc) <= len(comp) else comp
        # Exclude terminal branches and full/complementary terminal branches.
        if 2 <= len(side) <= n - 2:
            splits.add(frozenset(side))
    return splits


def split_id(split: FrozenSet[str]) -> str:
    return "|".join(sorted(split))


def split_metrics(ref_splits: Set[FrozenSet[str]], est_splits: Set[FrozenSet[str]]) -> Dict[str, float]:
    shared = ref_splits & est_splits
    fp = est_splits - ref_splits
    fn = ref_splits - est_splits
    rf = len(fp) + len(fn)
    denom = len(ref_splits) + len(est_splits)
    precision = len(shared) / len(est_splits) if len(est_splits) else float("nan")
    recall = len(shared) / len(ref_splits) if len(ref_splits) else float("nan")
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {
        "RF": float(rf),
        "nRF": float(rf / denom) if denom else float("nan"),
        "shared_splits": float(len(shared)),
        "ref_splits": float(len(ref_splits)),
        "est_splits": float(len(est_splits)),
        "split_precision": float(precision),
        "split_recall": float(recall),
        "split_F1": float(f1),
        "false_positive_splits": ";".join(split_id(s) for s in sorted(fp, key=lambda x: (len(x), sorted(x)))),
        "false_negative_splits": ";".join(split_id(s) for s in sorted(fn, key=lambda x: (len(x), sorted(x)))),
    }


def patristic_matrix_from_tree(tree, labels: Sequence[str]) -> np.ndarray:
    labels = [str(x) for x in labels]
    terminals_by_name = {str(t.name): t for t in tree.get_terminals()}
    missing = [lab for lab in labels if lab not in terminals_by_name]
    if missing:
        raise ValueError("Tree is missing labels: " + ", ".join(missing[:10]))

    n = len(labels)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        ti = terminals_by_name[labels[i]]
        for j in range(i + 1, n):
            tj = terminals_by_name[labels[j]]
            # Bio.Phylo.TreeMixin.distance sums branch lengths and treats None as zero.
            d = tree.distance(ti, tj)
            M[i, j] = M[j, i] = float(d if np.isfinite(d) else 0.0)
    np.fill_diagonal(M, 0.0)
    return M


def write_tree(tree, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Phylo.write(tree, path, "newick")


# ============================================================
# -------------------- FILE DISCOVERY -------------------------
# ============================================================


@dataclass
class MatrixFileRecord:
    method: str
    pct_missing: int
    replicate: int
    seed: Optional[int]
    path: str
    filename: str


def discover_method_files(spec: MethodSpec) -> List[MatrixFileRecord]:
    out: List[MatrixFileRecord] = []
    seen_paths: set[str] = set()
    compiled = [re.compile(r, flags=re.IGNORECASE) for r in spec.regexes]

    for d in spec.search_dirs:
        if not os.path.isdir(d):
            continue
        for path in sorted(Path(d).glob("**/*.csv")):
            resolved = str(path.resolve())
            if resolved in seen_paths:
                continue
            for rgx in compiled:
                m = rgx.match(path.name)
                if not m:
                    continue
                seed_s = m.groupdict().get("seed")
                out.append(
                    MatrixFileRecord(
                        method=spec.method,
                        pct_missing=int(m.group("pct")),
                        replicate=int(m.group("rep")),
                        seed=int(seed_s) if seed_s is not None else None,
                        path=str(path),
                        filename=path.name,
                    )
                )
                seen_paths.add(resolved)
                break

    return sorted(out, key=lambda r: (r.pct_missing, r.replicate, r.seed if r.seed is not None else -1, r.filename))


def discover_all_files() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    enabled = set(METHODS_TO_COMPARE) if METHODS_TO_COMPARE is not None else None

    for spec in METHOD_SPECS:
        if enabled is not None and spec.method not in enabled:
            continue
        records = discover_method_files(spec)
        for r in records:
            rows.append(
                {
                    "method": r.method,
                    "pct_missing": r.pct_missing,
                    "replicate": r.replicate,
                    "mask_seed": r.seed,
                    "file": r.filename,
                    "path": r.path,
                }
            )
        if len(records) == 0:
            print(f"WARNING: no completed matrices found for method: {spec.method}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            "No completed matrix files were found. Edit METHOD_SPECS search_dirs/regexes "
            "or set METHODS_TO_COMPARE to existing methods."
        )
    return df.sort_values(["method", "pct_missing", "replicate", "file"]).reset_index(drop=True)


def load_mask_registry() -> Optional[pd.DataFrame]:
    try:
        path = first_existing(MASK_REGISTRY_CANDIDATES, "mask registry")
    except FileNotFoundError:
        print("WARNING: mask_registry.csv not found. Method-specific metadata will still be used when available.")
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={"seed": "mask_seed"})
    keep = [
        c for c in [
            "pct_missing", "replicate", "mask_seed", "missingness_actual", "n_pairs_total",
            "n_missing", "n_observed", "optimized_variables", "masked_file",
        ] if c in df.columns
    ]
    df = df[keep].copy()
    print(f"Loaded mask registry: {path}")
    return df


def load_method_metadata() -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for method, candidates in METHOD_METADATA_CANDIDATES.items():
        if METHODS_TO_COMPARE is not None and method not in set(METHODS_TO_COMPARE):
            continue
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"WARNING: failed to read metadata for {method}: {e}")
            continue
        if df.empty:
            continue
        df = df.rename(columns={"seed": "mask_seed"}).copy()
        df["method"] = method
        keep = ["method", "pct_missing", "replicate", "mask_seed"]
        keep += [c for c in METADATA_COLUMNS_TO_MERGE if c in df.columns]
        keep += [c for c in ["masked_file", "completed_file", "error_message"] if c in df.columns]
        df = df[[c for c in keep if c in df.columns]].copy()
        rows.append(df)
        print(f"Loaded {method} metadata: {path}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def merge_aux_metadata(inventory: pd.DataFrame, mask_df: Optional[pd.DataFrame], method_meta: pd.DataFrame) -> pd.DataFrame:
    inv = inventory.copy()

    # First merge common mask metadata from Hyb registry, if available.
    if mask_df is not None and not mask_df.empty:
        if "mask_seed" in inv.columns and "mask_seed" in mask_df.columns:
            inv = inv.merge(
                mask_df,
                on=["pct_missing", "replicate", "mask_seed"],
                how="left",
                suffixes=("", "_mask"),
            )
        else:
            inv = inv.merge(
                mask_df.drop(columns=["mask_seed"], errors="ignore"),
                on=["pct_missing", "replicate"],
                how="left",
                suffixes=("", "_mask"),
            )

    # Then merge method-specific metadata. These columns override/fill registry columns.
    if method_meta is not None and not method_meta.empty:
        if "mask_seed" in method_meta.columns:
            merged = inv.merge(
                method_meta,
                on=["method", "pct_missing", "replicate", "mask_seed"],
                how="left",
                suffixes=("", "_method"),
            )
        else:
            merged = inv.merge(
                method_meta,
                on=["method", "pct_missing", "replicate"],
                how="left",
                suffixes=("", "_method"),
            )
        # Prefer method-specific columns when present; otherwise keep existing values.
        for col in list(METADATA_COLUMNS_TO_MERGE) + ["masked_file", "completed_file", "error_message"]:
            mc = f"{col}_method"
            if mc in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[mc].combine_first(merged[col])
                else:
                    merged[col] = merged[mc]
                merged = merged.drop(columns=[mc])
        inv = merged

    return inv


# ============================================================
# --------------------- TOPOLOGY ANALYSIS ---------------------
# ============================================================


def evaluate_one_matrix(
    row: pd.Series,
    design: Dict[str, object],
) -> Dict[str, object]:
    tree_path = str(design["tree_path"])
    tree_labels = list(design["tree_labels"])
    ref_splits = design["ref_splits"]
    ml_pat_vec = np.asarray(design["ml_pat_vec"], dtype=float)
    pairs = np.asarray(design["pairs"], dtype=np.int32)

    method = str(row["method"])
    pct = int(row["pct_missing"])
    rep = int(row["replicate"])
    seed = row.get("mask_seed", None)
    seed_int = int(seed) if pd.notna(seed) else -1
    path = str(row["path"])

    out: Dict[str, object] = {
        "method": method,
        "pct_missing": pct,
        "replicate": rep,
        "mask_seed": seed_int if seed_int >= 0 else np.nan,
        "file": row.get("file", os.path.basename(path)),
        "path": path,
        "success_tree": False,
        "error_message_tree": "",
        "nj_tree_file": "",
    }

    # Carry useful completion metadata into the topology detailed table.
    for col in list(METADATA_COLUMNS_TO_MERGE) + ["missingness_actual", "n_missing", "n_observed", "optimized_variables"]:
        if col in row.index and pd.notna(row[col]):
            out[col] = row[col]

    try:
        D_raw, matrix_labels = load_distance_matrix_csv(path)
        D_reordered, matched_labels = reorder_matrix_to_tree_labels(D_raw, matrix_labels, tree_labels)
        D = sanitize_complete_distance_matrix(D_reordered, path)

        nj_tree = build_nj_tree_from_matrix(D, tree_labels)
        nj_splits = unrooted_internal_splits(nj_tree, tree_labels)
        metrics = split_metrics(ref_splits, nj_splits)

        nj_pat_M = patristic_matrix_from_tree(nj_tree, tree_labels)
        nj_pat_vec = values_on_pairs(nj_pat_M, pairs)
        input_vec = values_on_pairs(D, pairs)

        metrics.update(vector_metrics(nj_pat_vec, ml_pat_vec, "nj_vs_ML_pat"))
        metrics.update(vector_metrics(input_vec, ml_pat_vec, "input_vs_ML_pat"))

        tree_file = f"{safe_tag(method)}_NJ_p{pct}_rep{rep:02d}"
        if seed_int >= 0:
            tree_file += f"_seed{seed_int}"
        tree_file += ".treefile"
        tree_out_path = os.path.join(TREES_DIR, tree_file)
        write_tree(nj_tree, tree_out_path)

        out.update(metrics)
        out.update(
            {
                "success_tree": True,
                "nj_tree_file": tree_file,
                "n_taxa": len(tree_labels),
                "matched_matrix_labels_json": pd.Series(matched_labels).to_json(force_ascii=False),
            }
        )
    except Exception as e:
        out.update(
            {
                "success_tree": False,
                "error_message_tree": repr(e),
                "RF": float("nan"),
                "nRF": float("nan"),
                "shared_splits": float("nan"),
                "ref_splits": float("nan"),
                "est_splits": float("nan"),
                "split_precision": float("nan"),
                "split_recall": float("nan"),
                "split_F1": float("nan"),
                "nj_vs_ML_pat_MAE": float("nan"),
                "nj_vs_ML_pat_RMSE": float("nan"),
                "nj_vs_ML_pat_Pearson": float("nan"),
                "nj_vs_ML_pat_Spearman": float("nan"),
                "input_vs_ML_pat_MAE": float("nan"),
                "input_vs_ML_pat_RMSE": float("nan"),
                "input_vs_ML_pat_Pearson": float("nan"),
                "input_vs_ML_pat_Spearman": float("nan"),
            }
        )
        print(f"FAILED tree comparison: {method} p{pct} rep{rep:02d}: {repr(e)}")

    return out


def make_summary_tables(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows: List[Dict[str, object]] = []
    formatted_rows: List[Dict[str, object]] = []

    for (method, pct), sub_all in results_df.groupby(["method", "pct_missing"], sort=True):
        sub_all = sub_all.copy()
        sub_success = sub_all[sub_all["success_tree"] == True].copy()
        n_total = int(len(sub_all))
        n_success = int(sub_all["success_tree"].sum())
        n_failed = int(n_total - n_success)

        numeric_row: Dict[str, object] = {
            "method": method,
            "pct_missing": int(pct),
            "n_replicates": n_total,
            "n_success_tree": n_success,
            "n_failed_tree": n_failed,
        }
        formatted_row: Dict[str, object] = {
            "method": method,
            "% Missing": f"{int(pct)}%",
            "n_replicates": n_total,
            "n_success_tree": n_success,
            "n_failed_tree": n_failed,
        }

        cols = list(TOPOLOGY_SUMMARY_METRICS)
        for extra in [
            "missingness_actual", "n_missing", "n_observed", "optimized_variables",
            "RMSE_miss", "MAE_miss", "Pearson_miss", "Spearman_miss", "runtime_seconds",
            "iters_used", "epochs_used", "convergence_epoch", "final_rank",
        ]:
            if extra in results_df.columns and extra not in cols:
                cols.append(extra)

        for col in cols:
            if col not in sub_all.columns:
                continue
            source = sub_all if col in {"missingness_actual", "n_missing", "n_observed", "optimized_variables"} else sub_success
            vals = pd.to_numeric(source[col], errors="coerce").to_numpy(dtype=float)
            mean_val, std_val = mean_std_nan(vals)
            numeric_row[f"{col}_mean"] = mean_val
            numeric_row[f"{col}_std"] = std_val

            decimals = 2 if col in {"runtime_seconds", "iters_used", "epochs_used", "convergence_epoch"} else 6
            formatted_row[col] = fmt_pm(mean_val, std_val, decimals=decimals)

        numeric_rows.append(numeric_row)
        formatted_rows.append(formatted_row)

    return pd.DataFrame(numeric_rows), pd.DataFrame(formatted_rows)


def build_design(tree_path: str) -> Dict[str, object]:
    ref_tree = Phylo.read(tree_path, "newick")
    tree_labels = [str(t.name) for t in ref_tree.get_terminals()]
    if len(tree_labels) != len(set(tree_labels)):
        raise ValueError("Reference ML tree contains duplicate terminal labels.")

    pairs = upper_pairs(len(tree_labels))
    ml_pat_M = patristic_matrix_from_tree(ref_tree, tree_labels)
    ml_pat_vec = values_on_pairs(ml_pat_M, pairs)
    ref_splits = unrooted_internal_splits(ref_tree, tree_labels)

    return {
        "tree_path": tree_path,
        "ref_tree": ref_tree,
        "tree_labels": tree_labels,
        "pairs": pairs,
        "ml_pat_M": ml_pat_M,
        "ml_pat_vec": ml_pat_vec,
        "ref_splits": ref_splits,
    }


def write_reference_outputs(design: Dict[str, object], D_ref: np.ndarray, ref_matrix_labels: Sequence[str]) -> None:
    tree_labels = list(design["tree_labels"])
    ref_splits = design["ref_splits"]
    ml_pat_M = np.asarray(design["ml_pat_M"], dtype=float)
    ml_pat_vec = np.asarray(design["ml_pat_vec"], dtype=float)
    pairs = np.asarray(design["pairs"], dtype=np.int32)

    split_rows = []
    for s in sorted(ref_splits, key=lambda x: (len(x), sorted(x))):
        split_rows.append({"split_size": len(s), "split": split_id(s)})
    save_csv(pd.DataFrame(split_rows), os.path.join(OUT_DIR, "reference_ml_splits.csv"))

    pd.DataFrame(ml_pat_M, index=tree_labels, columns=tree_labels).to_csv(
        os.path.join(OUT_DIR, "reference_ml_patristic_matrix.csv")
    )
    print(f"Saved: {os.path.join(OUT_DIR, 'reference_ml_patristic_matrix.csv')}")

    # Also compute NJ(D_ref) vs ML. This is a useful sanity baseline for the
    # reference distance matrix itself.
    D_ref_tree_order, _ = reorder_matrix_to_tree_labels(D_ref, ref_matrix_labels, tree_labels)
    D_ref_tree_order = sanitize_complete_distance_matrix(D_ref_tree_order, "reference matrix")
    ref_nj = build_nj_tree_from_matrix(D_ref_tree_order, tree_labels)
    ref_nj_path = os.path.join(OUT_DIR, "reference_nj_from_Dref.treefile")
    write_tree(ref_nj, ref_nj_path)
    print(f"Saved: {ref_nj_path}")

    ref_nj_splits = unrooted_internal_splits(ref_nj, tree_labels)
    metrics = split_metrics(ref_splits, ref_nj_splits)
    ref_nj_pat_M = patristic_matrix_from_tree(ref_nj, tree_labels)
    metrics.update(vector_metrics(values_on_pairs(ref_nj_pat_M, pairs), ml_pat_vec, "nj_vs_ML_pat"))
    metrics.update(vector_metrics(values_on_pairs(D_ref_tree_order, pairs), ml_pat_vec, "input_vs_ML_pat"))
    metrics_df = pd.DataFrame([{k: v for k, v in metrics.items() if not isinstance(v, str)}])
    metrics_df["false_positive_splits"] = metrics.get("false_positive_splits", "")
    metrics_df["false_negative_splits"] = metrics.get("false_negative_splits", "")
    save_csv(metrics_df, os.path.join(OUT_DIR, "reference_nj_from_Dref_metrics.csv"))


# ============================================================
# ---------------------------- MAIN ---------------------------
# ============================================================


def main() -> None:
    ensure_dirs()

    tree_path = first_existing(ML_TREE_CANDIDATES, "IQ-TREE ML reference tree")
    design = build_design(tree_path)
    tree_labels = list(design["tree_labels"])

    print("=== ML REFERENCE TREE INFO ===")
    print(f"Tree file: {tree_path}")
    print(f"n_taxa = {len(tree_labels)}")
    print(f"reference internal splits = {len(design['ref_splits'])}")
    print()

    ref_matrix_path = first_existing(REFERENCE_MATRIX_CANDIDATES, "reference matrix")
    D_ref, ref_labels = load_distance_matrix_csv(ref_matrix_path)
    if np.nanmax(D_ref) > 500:
        D_ref = D_ref / 1000.0
    D_ref = sanitize_complete_distance_matrix(D_ref, "reference matrix")
    print("=== REFERENCE MATRIX INFO ===")
    print(f"Matrix file: {ref_matrix_path}")
    print(f"shape = {D_ref.shape}")
    print()

    write_reference_outputs(design, D_ref, ref_labels)

    inventory = discover_all_files()
    mask_df = load_mask_registry()
    method_meta = load_method_metadata()
    inventory = merge_aux_metadata(inventory, mask_df, method_meta)
    save_csv(inventory, os.path.join(OUT_DIR, "method_file_inventory.csv"))

    print("\n=== Running NJ topology comparison for completed matrices ===")
    rows: List[Dict[str, object]] = []
    for _, rec in inventory.iterrows():
        print(f"  {rec['method']} | {int(rec['pct_missing'])}% | rep {int(rec['replicate']):02d} | {rec['file']}")
        rows.append(evaluate_one_matrix(rec, design))

    results_df = pd.DataFrame(rows)
    detail_path = os.path.join(OUT_DIR, "tree_topology_detailed_all_methods.csv")
    save_csv(results_df, detail_path)

    summary_numeric, summary_meanstd = make_summary_tables(results_df)
    summary_num_path = os.path.join(OUT_DIR, "tree_topology_summary_numeric.csv")
    summary_fmt_path = os.path.join(OUT_DIR, "tree_topology_summary_meanstd.csv")
    save_csv(summary_numeric, summary_num_path)
    save_csv(summary_meanstd, summary_fmt_path)

    zip_base = os.path.join(OUT_DIR, "tree_topology_comparison")
    zip_path = shutil.make_archive(zip_base, "zip", OUT_DIR)
    print(f"Saved: {zip_path}")

    display_or_print(summary_meanstd, "Tree topology summary")

    print("\n=== ALL PROCESSING COMPLETE ===")
    print(f"Output directory: {OUT_DIR}/")
    print(f"Detailed results: {detail_path}")
    print(f"Summary numeric:  {summary_num_path}")
    print(f"Summary mean±std: {summary_fmt_path}")
    print(f"NJ tree files:    {TREES_DIR}/")
    print(f"ZIP archive:      {zip_path}")


if __name__ == "__main__":
    main()
