#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-topology branch-length comparison for completed distance matrices
===================================================================

Purpose
-------
This script replaces the old NJ/RF benchmark with a benchmark against the
IQ-TREE maximum-likelihood (ML) reference tree.

It does NOT reconstruct NJ trees from the completed matrices. Instead, for each
completed distance matrix it:

1. reads the IQ-TREE ML reference tree with branch lengths;
2. fixes the ML topology;
3. fits nonnegative branch lengths on that fixed ML topology to the completed
   distance matrix by least squares / NNLS;
4. compares the fitted branch lengths against the branch lengths of the ML
   reference tree;
5. also compares the resulting fitted patristic distances against the ML
   reference patristic distances.

This directly tests whether a matrix-completion method improves a biologically
meaningful quantity: branch-length accuracy on the reference ML phylogeny.

Expected input files
--------------------
Reference tree produced by your IQ-TREE script:
    mtDNA15_ML_tree/mtDNA15_IQTREE_ML.treefile

Reference matrix / labels from the new Hyb-Adam-UM protocol:
    hyb_adam_um_outputs/reference_matrix_sanitized.csv
or:
    Dref_MAFFT_pairwise_deletion_pdistance_Tlabels.csv

Frozen mask metadata from the new Hyb-Adam-UM protocol:
    hyb_adam_um_outputs/tables/mask_registry.csv

Completed matrices from enabled methods, for example:
    hyb_adam_um_outputs/completed_matrices/HybAdamUM_completed_p30_rep01_seed55.csv
    mw_proj_outputs/completed_matrices/MWproj_completed_p30_rep01_seed55.csv
    knn_impute_outputs/completed_matrices/KNNImpute_completed_p30_rep01_seed55.csv
    mds_smacof_outputs/completed_matrices/MDSSMACOF_completed_p30_rep01_seed55.csv

MW-proj metadata is read from:
    mw_proj_outputs/tables/mw_proj_all_masks_detailed.csv

If your output folders/prefixes differ, edit METHOD_SPECS below.

Dependencies
------------
    pip install numpy pandas scipy biopython

Outputs
-------
    ml_branch_length_comparison/
        method_file_inventory.csv
        reference_ml_branch_lengths.csv
        reference_ml_patristic_matrix.csv
        ml_branch_length_detailed_all_methods.csv
        ml_branch_lengths_long_all_methods.csv
        ml_branch_length_summary_numeric.csv
        ml_branch_length_summary_meanstd.csv
        ml_branch_length_comparison.zip
"""

from __future__ import annotations

import os
import re
import math
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import Phylo

try:
    from scipy.optimize import nnls as scipy_nnls
except Exception:  # pragma: no cover - fallback for minimal environments
    scipy_nnls = None

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
# It lets the script compare unlabeled CSV matrices with the IQ-TREE labels.
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

MASK_REGISTRY_CANDIDATES: Tuple[str, ...] = (
    "hyb_adam_um_outputs/tables/mask_registry.csv",
    "./hyb_adam_um_outputs/tables/mask_registry.csv",
    "/mnt/data/hyb_adam_um_outputs/tables/mask_registry.csv",
)

# Optional method-specific detailed result tables. These are useful when a method
# has its own runtime, optimized-variable count, RMSE_miss, etc. MW-proj writes
# exactly this detailed table in the script you provided.
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
    "KNN-Impute": (
        "knn_impute_outputs/tables/knn_impute_all_masks_detailed.csv",
        "./knn_impute_outputs/tables/knn_impute_all_masks_detailed.csv",
        "/mnt/data/knn_impute_outputs/tables/knn_impute_all_masks_detailed.csv",
    ),
    "MDS-SMACOF": (
        "mds_smacof_outputs/tables/mds_smacof_all_masks_detailed.csv",
        "./mds_smacof_outputs/tables/mds_smacof_all_masks_detailed.csv",
        "/mnt/data/mds_smacof_outputs/tables/mds_smacof_all_masks_detailed.csv",
    ),
}

# Use None to evaluate every method for which completed matrices are found.
# Example for exactly three methods:
#     METHODS_TO_COMPARE = ("Hyb-Adam-UM", "MW-proj", "KNN-Impute")
METHODS_TO_COMPARE: Optional[Tuple[str, ...]] = None

OUT_DIR = "ml_branch_length_comparison"

# Edit these if your baseline scripts write to different folders or filenames.
# The regex must contain named groups pct and rep; seed is optional.
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
            "hyb_adam_um_completed_matrices",       # legacy folder from the old plotting script
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
            # Main filename written by your MW script:
            #     MWproj_completed_p30_rep01_seed55.csv
            r"^(?:MWproj|MWProj|MW_Proj|MW-Proj|MW-proj)_completed_p(?P<pct>\d+)_rep(?P<rep>\d+)(?:_seed(?P<seed>\d+))?\.csv$",
            r"^mw.*?proj.*?p(?P<pct>\d+).*?rep(?P<rep>\d+)(?:.*?seed(?P<seed>\d+))?\.csv$",
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

SUMMARY_METRICS: Tuple[str, ...] = (
    "edge_BL_MAE",
    "edge_BL_RMSE",
    "edge_BL_rel_RMSE",
    "edge_BL_Pearson",
    "edge_BL_Spearman",
    "terminal_BL_MAE",
    "terminal_BL_RMSE",
    "internal_BL_MAE",
    "internal_BL_RMSE",
    "fit_pat_MAE",
    "fit_pat_RMSE",
    "fit_pat_Pearson",
    "fit_pat_Spearman",
    "input_vs_ML_pat_MAE",
    "input_vs_ML_pat_RMSE",
    "input_vs_ML_pat_Pearson",
    "input_vs_ML_pat_Spearman",
    "tree_length_abs_error",
    "tree_length_rel_error",
)


# ============================================================
# ----------------------- BASIC HELPERS -----------------------
# ============================================================


def first_existing(candidates: Sequence[str], what: str) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {what}. Tried: {list(candidates)}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def as_float_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def upper_pairs(n: int) -> np.ndarray:
    i, j = np.triu_indices(n, k=1)
    return np.column_stack([i, j]).astype(np.int32)


def values_on_pairs(M: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return np.asarray(M[pairs[:, 0], pairs[:, 1]], dtype=float)


def safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0


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
            f"{prefix}_Bias": float("nan"),
            f"{prefix}_MaxAE": float("nan"),
            f"{prefix}_Pearson": float("nan"),
            f"{prefix}_Spearman": float("nan"),
            f"{prefix}_rel_RMSE": float("nan"),
            f"{prefix}_rel_MAE": float("nan"),
        }
    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom_rmse = float(np.sqrt(np.mean(true * true))) if true.size else float("nan")
    denom_mae = float(np.mean(np.abs(true))) if true.size else float("nan")
    return {
        f"{prefix}_MAE": mae,
        f"{prefix}_RMSE": rmse,
        f"{prefix}_Bias": float(np.mean(diff)),
        f"{prefix}_MaxAE": float(np.max(np.abs(diff))),
        f"{prefix}_Pearson": pearson_corr(pred, true),
        f"{prefix}_Spearman": spearman_corr(pred, true),
        f"{prefix}_rel_RMSE": rmse / denom_rmse if denom_rmse and np.isfinite(denom_rmse) else float("nan"),
        f"{prefix}_rel_MAE": mae / denom_mae if denom_mae and np.isfinite(denom_mae) else float("nan"),
    }


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
            # If both axes are labelled with the same set, reorder rows to columns.
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

    # Completed matrices should no longer contain the artificial missing marker.
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
    """True for labels T1, T2, ..., Tn created by the CSV fallback loader."""
    labels = [str(x) for x in labels]
    return labels == [f"T{i + 1}" for i in range(len(labels))]


def read_label_mapping_tsv(n_expected: int, tree_labels: Sequence[str]) -> Optional[List[str]]:
    """
    Try to recover taxon order from mtDNA15_ML_tree/label_mapping.tsv.

    The IQ-TREE-building script writes columns:
        new_label, accession, original_header
    """
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
    """
    Replace generic T1..Tn labels by real taxon labels when possible.

    Returns
    -------
    resolved_labels, used_recovery
    """
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
    """Return possible matching keys for taxon labels."""
    s = str(label).strip()
    keys: List[str] = []

    # Accession such as KJ434962, KT159932, etc.
    m = ACCESSION_RE.search(s.upper())
    if m:
        keys.append("acc:" + m.group(0))

    # Exact and normalized versions.
    keys.append("exact:" + s)
    norm = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    keys.append("norm:" + norm)

    # Sometimes labels are Species_accession; also try species part before accession.
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
    """
    Reorder a matrix into the leaf order of the ML tree.

    If the matrix was loaded from an unlabeled numeric CSV, the loader assigns
    generic labels T1..Tn. In that case we first recover the real mtDNA15 labels
    either from label_mapping.tsv or from the accession order used in the ML-tree
    construction script, then use those recovered labels only for indexing logic.
    """
    resolved_labels, recovered = resolve_generic_matrix_labels(matrix_labels, tree_labels)
    mapping = build_tree_to_matrix_label_map(tree_labels, resolved_labels)
    idx = {str(lab): i for i, lab in enumerate(resolved_labels)}
    order = [idx[mapping[tlab]] for tlab in tree_labels]
    return np.asarray(D, dtype=float)[np.ix_(order, order)], [mapping[tlab] for tlab in tree_labels]


# ============================================================
# ------------------ ML TREE LINEAR ALGEBRA ------------------
# ============================================================


@dataclass
class MLTopologyDesign:
    tree_path: str
    tree_labels: List[str]
    pairs: np.ndarray
    A: np.ndarray
    b_ref: np.ndarray
    edge_ids: List[str]
    edge_types: List[str]
    edge_descendant_counts: List[int]
    ml_patristic_vec: np.ndarray
    ml_patristic_matrix: np.ndarray


def _edge_id_from_indices(desc: set[int], n: int, tree_labels: Sequence[str]) -> Tuple[str, str, int]:
    """
    Stable edge ID from the split induced by an edge.

    For unrooted trees, the two sides of a split are equivalent. We therefore
    store the smaller side as the ID. This also makes the output stable if the
    Newick root is arbitrary.
    """
    desc = set(desc)
    comp = set(range(n)) - desc
    side = desc if len(desc) <= len(comp) else comp
    names = sorted(tree_labels[i] for i in side)
    edge_type = "terminal" if len(side) == 1 else "internal"
    prefix = "terminal:" if edge_type == "terminal" else "internal:"
    return prefix + "|".join(names), edge_type, len(side)


def build_ml_topology_design(tree_path: str) -> MLTopologyDesign:
    tree = Phylo.read(tree_path, "newick")
    terminals = tree.get_terminals()
    tree_labels = [str(t.name) for t in terminals]
    if len(tree_labels) != len(set(tree_labels)):
        raise ValueError("The ML tree contains duplicate terminal labels.")

    n = len(tree_labels)
    label_to_idx = {lab: i for i, lab in enumerate(tree_labels)}
    pairs = upper_pairs(n)

    edge_desc_sets: List[set[int]] = []
    edge_ids: List[str] = []
    edge_types: List[str] = []
    edge_descendant_counts: List[int] = []
    b_ref: List[float] = []

    def desc_indices(clade) -> set[int]:
        return {label_to_idx[str(t.name)] for t in clade.get_terminals()}

    def add_edge_from_desc(desc: set[int], length: float) -> None:
        eid, etype, side_count = _edge_id_from_indices(desc, n, tree_labels)
        edge_desc_sets.append(set(desc))
        edge_ids.append(eid)
        edge_types.append(etype)
        edge_descendant_counts.append(side_count)
        b_ref.append(float(length) if np.isfinite(length) else 0.0)

    def recurse_below(parent) -> None:
        for child in parent.clades:
            add_edge_from_desc(desc_indices(child), float(child.branch_length or 0.0))
            recurse_below(child)

    root = tree.root

    # IQ-TREE writes an unrooted tree, but Newick parsers can represent an arbitrary
    # degree-2 root. Pairwise distances cannot identify the two root-adjacent
    # branches separately; only their sum is meaningful. Suppress such a root and
    # store the combined internal edge.
    if len(root.clades) == 2:
        c1, c2 = root.clades
        combined_len = float(c1.branch_length or 0.0) + float(c2.branch_length or 0.0)
        add_edge_from_desc(desc_indices(c1), combined_len)
        # Add only the edges below the two root children, not the two root edges again.
        recurse_below(c1)
        recurse_below(c2)
    else:
        recurse_below(root)

    m_pairs = pairs.shape[0]
    m_edges = len(edge_desc_sets)
    A = np.zeros((m_pairs, m_edges), dtype=float)

    for e, desc in enumerate(edge_desc_sets):
        for pidx, (i, j) in enumerate(pairs):
            A[pidx, e] = 1.0 if ((int(i) in desc) ^ (int(j) in desc)) else 0.0

    b_ref_arr = np.asarray(b_ref, dtype=float)
    ml_pat_vec = A @ b_ref_arr
    ml_pat_M = np.zeros((n, n), dtype=float)
    for val, (i, j) in zip(ml_pat_vec, pairs):
        ml_pat_M[i, j] = ml_pat_M[j, i] = float(val)

    return MLTopologyDesign(
        tree_path=tree_path,
        tree_labels=tree_labels,
        pairs=pairs,
        A=A,
        b_ref=b_ref_arr,
        edge_ids=edge_ids,
        edge_types=edge_types,
        edge_descendant_counts=edge_descendant_counts,
        ml_patristic_vec=ml_pat_vec,
        ml_patristic_matrix=ml_pat_M,
    )

def fit_nonnegative_branch_lengths(A: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, str]:
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if not ok.all():
        A_fit = A[ok, :]
        y_fit = y[ok]
    else:
        A_fit = A
        y_fit = y

    if A_fit.shape[0] == 0:
        raise ValueError("No finite pairwise distances available for branch-length fitting.")

    if scipy_nnls is not None:
        x, rnorm = scipy_nnls(A_fit, y_fit)
        return np.asarray(x, dtype=float), float(rnorm), "scipy.optimize.nnls"

    # Fallback: ordinary least squares followed by truncation. Prefer scipy NNLS when possible.
    x, *_ = np.linalg.lstsq(A_fit, y_fit, rcond=None)
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    rnorm = float(np.linalg.norm(A_fit @ x - y_fit))
    return x, rnorm, "numpy_lstsq_clipped_fallback"


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
            if str(path.resolve()) in seen_paths:
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
                seen_paths.add(str(path.resolve()))
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
            "No completed matrix files were found. Edit METHOD_SPECS search_dirs/regexes."
        )
    return df.sort_values(["method", "pct_missing", "replicate", "file"]).reset_index(drop=True)


MASK_LEVEL_METADATA_COLUMNS: Tuple[str, ...] = (
    "pct_missing",
    "replicate",
    "mask_seed",
    "missingness_actual",
    "n_pairs_total",
    "n_missing",
    "n_observed",
    "masked_file",
)

METHOD_LEVEL_METADATA_COLUMNS: Tuple[str, ...] = (
    "method",
    "pct_missing",
    "replicate",
    "mask_seed",
    "missingness_actual",
    "n_pairs_total",
    "n_missing",
    "n_observed",
    "optimized_variables",
    "runtime_seconds",
    "RMSE_miss",
    "MAE_miss",
    "Pearson_miss",
    "Spearman_miss",
    "Delta_init",
    "Delta_final",
    "Delta_reduction_percent",
    "Delta_total_completed",
    "Delta_normalized_completed",
    "Delta_per_triangle_completed",
    "Delta_relative_to_original",
    "max_abs_error_observed",
    "mean_abs_error_observed",
    "MW_Qw_final",
    "weighted_fit_pairs",
    "final_tree_edges",
    "n_start_pairs",
    "n_weighted_lsq_fits",
    "n_graft_evaluations",
    "stepA_method",
    "stepA_loops",
    "stepA_estimated_pairs",
    "stepA_unfilled_pairs",
    "masked_file",
)


def _first_existing_optional(candidates: Sequence[str]) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _normalize_metadata_df(df: pd.DataFrame, method: Optional[str], columns: Sequence[str]) -> pd.DataFrame:
    df = df.rename(columns={"seed": "mask_seed"}).copy()
    if method is not None:
        df["method"] = method
    keep = [c for c in columns if c in df.columns]
    required = ["pct_missing", "replicate"]
    for c in required:
        if c not in keep:
            raise ValueError(f"Metadata table for {method or 'mask registry'} lacks required column: {c}")
    out = df[keep].copy()
    for c in ["pct_missing", "replicate", "mask_seed"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "method" in out.columns:
        out["method"] = out["method"].astype(str)
    return out


def load_mask_registry() -> Optional[pd.DataFrame]:
    """
    Load mask/method metadata for the branch-length comparison.

    The old comparison script only loaded Hyb's mask_registry.csv. That is not
    enough for MW-proj because your MW script writes method diagnostics to
    mw_proj_outputs/tables/mw_proj_all_masks_detailed.csv. This function loads
    both generic mask metadata and any method-specific detailed tables that are
    present, then merge_mask_metadata() applies them in the right order.
    """
    frames: List[pd.DataFrame] = []

    generic_path = _first_existing_optional(MASK_REGISTRY_CANDIDATES)
    if generic_path is not None:
        df = pd.read_csv(generic_path)
        df = _normalize_metadata_df(df, method=None, columns=MASK_LEVEL_METADATA_COLUMNS)
        df["__metadata_scope"] = "generic_mask"
        df["__metadata_source"] = generic_path
        frames.append(df)
        print(f"Loaded generic mask registry: {generic_path}")
    else:
        print("WARNING: generic mask_registry.csv not found; trying method-specific detailed tables.")

    enabled = set(METHODS_TO_COMPARE) if METHODS_TO_COMPARE is not None else None
    for method, candidates in METHOD_METADATA_CANDIDATES.items():
        if enabled is not None and method not in enabled:
            continue
        path = _first_existing_optional(candidates)
        if path is None:
            continue
        df = pd.read_csv(path)
        df = _normalize_metadata_df(df, method=method, columns=METHOD_LEVEL_METADATA_COLUMNS)
        df["__metadata_scope"] = "method_specific"
        df["__metadata_source"] = path
        frames.append(df)
        print(f"Loaded {method} detailed metadata: {path}")

    if not frames:
        print("WARNING: no mask/method metadata found. The benchmark will still run, but metadata columns will be unavailable.")
        return None

    return pd.concat(frames, ignore_index=True, sort=False)


def _fill_from_metadata(left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if right is None or right.empty:
        return left
    if any(k not in left.columns for k in keys) or any(k not in right.columns for k in keys):
        return left
    if "mask_seed" in keys and not left["mask_seed"].notna().any():
        return left

    r = right.dropna(subset=[k for k in keys if k in right.columns]).copy()
    if r.empty:
        return left
    r = r.drop_duplicates(list(keys), keep="first")

    merged = left.merge(r, on=list(keys), how="left", suffixes=("", "__meta"))

    for c in list(r.columns):
        if c in keys:
            continue
        meta_c = f"{c}__meta"
        if meta_c in merged.columns:
            if c in merged.columns:
                merged[c] = merged[c].combine_first(merged[meta_c])
                merged = merged.drop(columns=[meta_c])
            else:
                merged = merged.rename(columns={meta_c: c})

    return merged


def merge_mask_metadata(inventory: pd.DataFrame, mask_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if mask_df is None or mask_df.empty:
        return inventory

    out = inventory.copy()

    method_meta = mask_df[mask_df.get("__metadata_scope", "") == "method_specific"].copy()
    generic_meta = mask_df[mask_df.get("__metadata_scope", "") == "generic_mask"].copy()

    # Method-specific metadata first. This prevents generic metadata from
    # overwriting MW-specific optimized_variables/runtime/RMSE diagnostics.
    if not method_meta.empty:
        out = _fill_from_metadata(out, method_meta, ["method", "pct_missing", "replicate", "mask_seed"])
        out = _fill_from_metadata(out, method_meta.drop(columns=["mask_seed"], errors="ignore"), ["method", "pct_missing", "replicate"])

    # Generic mask metadata supplies n_missing/n_observed for methods using the
    # same frozen masks but not writing their own detailed table.
    if not generic_meta.empty:
        out = _fill_from_metadata(out, generic_meta, ["pct_missing", "replicate", "mask_seed"])
        out = _fill_from_metadata(out, generic_meta.drop(columns=["mask_seed"], errors="ignore"), ["pct_missing", "replicate"])

    return out


# ============================================================
# -------------------- SINGLE EVALUATION ----------------------
# ============================================================


def evaluate_one_matrix(
    row: pd.Series,
    design: MLTopologyDesign,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    path = str(row["path"])
    D_raw, matrix_labels = load_distance_matrix_csv(path)
    D_reordered, matched_labels = reorder_matrix_to_tree_labels(D_raw, matrix_labels, design.tree_labels)
    D = sanitize_complete_distance_matrix(D_reordered, name=path)

    y_input = values_on_pairs(D, design.pairs)
    b_hat, rnorm, solver = fit_nonnegative_branch_lengths(design.A, y_input)
    y_fit = design.A @ b_hat

    terminal_mask = np.array([t == "terminal" for t in design.edge_types], dtype=bool)
    internal_mask = ~terminal_mask

    result: Dict[str, object] = {
        "method": row["method"],
        "pct_missing": int(row["pct_missing"]),
        "replicate": int(row["replicate"]),
        "mask_seed": int(row["mask_seed"]) if pd.notna(row.get("mask_seed", np.nan)) else np.nan,
        "file": row["file"],
        "path": path,
        "success": True,
        "error_message": "",
        "n_taxa": len(design.tree_labels),
        "n_pairs": int(design.pairs.shape[0]),
        "n_edges": int(len(design.b_ref)),
        "n_terminal_edges": int(terminal_mask.sum()),
        "n_internal_edges": int(internal_mask.sum()),
        "nnls_rnorm": float(rnorm),
        "branch_fit_solver": solver,
        "tree_length_ref": float(np.sum(design.b_ref)),
        "tree_length_hat": float(np.sum(b_hat)),
        "tree_length_abs_error": float(abs(np.sum(b_hat) - np.sum(design.b_ref))),
        "tree_length_rel_error": float(abs(np.sum(b_hat) - np.sum(design.b_ref)) / np.sum(design.b_ref)) if np.sum(design.b_ref) > 0 else float("nan"),
    }

    # Add mask/completion metadata if available in the inventory row.
    # For MW-proj this includes runtime, hidden-entry RMSE/MAE/correlations,
    # MW weighted-fit diagnostics, and optimized_variables from the MW detailed table.
    metadata_to_copy = [
        "missingness_actual", "n_pairs_total", "n_missing", "n_observed",
        "optimized_variables", "masked_file", "runtime_seconds",
        "RMSE_miss", "MAE_miss", "Pearson_miss", "Spearman_miss",
        "Delta_init", "Delta_final", "Delta_reduction_percent",
        "Delta_total_completed", "Delta_normalized_completed",
        "Delta_per_triangle_completed", "Delta_relative_to_original",
        "max_abs_error_observed", "mean_abs_error_observed",
        "MW_Qw_final", "weighted_fit_pairs", "final_tree_edges",
        "n_start_pairs", "n_weighted_lsq_fits", "n_graft_evaluations",
        "stepA_method", "stepA_loops", "stepA_estimated_pairs",
        "stepA_unfilled_pairs", "__metadata_scope", "__metadata_source",
    ]
    for c in metadata_to_copy:
        if c in row.index:
            result[c] = row[c]

    result.update(vector_metrics(b_hat, design.b_ref, "edge_BL"))
    if terminal_mask.any():
        result.update(vector_metrics(b_hat[terminal_mask], design.b_ref[terminal_mask], "terminal_BL"))
    else:
        result.update({k: float("nan") for k in vector_metrics([], [], "terminal_BL")})
    if internal_mask.any():
        result.update(vector_metrics(b_hat[internal_mask], design.b_ref[internal_mask], "internal_BL"))
    else:
        result.update({k: float("nan") for k in vector_metrics([], [], "internal_BL")})

    result.update(vector_metrics(y_fit, design.ml_patristic_vec, "fit_pat"))
    result.update(vector_metrics(y_input, design.ml_patristic_vec, "input_vs_ML_pat"))

    bl_long = pd.DataFrame(
        {
            "method": row["method"],
            "pct_missing": int(row["pct_missing"]),
            "replicate": int(row["replicate"]),
            "mask_seed": result["mask_seed"],
            "file": row["file"],
            "edge_id": design.edge_ids,
            "edge_type": design.edge_types,
            "edge_descendant_count": design.edge_descendant_counts,
            "branch_length_ref_ML": design.b_ref,
            "branch_length_hat_from_completed_matrix": b_hat,
            "branch_length_error": b_hat - design.b_ref,
            "branch_length_abs_error": np.abs(b_hat - design.b_ref),
        }
    )
    return result, bl_long


def evaluate_all(inventory: pd.DataFrame, design: MLTopologyDesign) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detailed_rows: List[Dict[str, object]] = []
    bl_long_frames: List[pd.DataFrame] = []

    for _, row in inventory.iterrows():
        print(f"Evaluating {row['method']} p{int(row['pct_missing'])} rep{int(row['replicate']):02d}: {row['file']}")
        try:
            res, bl_long = evaluate_one_matrix(row, design)
            detailed_rows.append(res)
            bl_long_frames.append(bl_long)
        except Exception as e:
            err = {
                "method": row["method"],
                "pct_missing": int(row["pct_missing"]),
                "replicate": int(row["replicate"]),
                "mask_seed": row.get("mask_seed", np.nan),
                "file": row["file"],
                "path": row["path"],
                "success": False,
                "error_message": repr(e),
            }
            detailed_rows.append(err)
            print(f"  FAILED: {repr(e)}")

    detailed = pd.DataFrame(detailed_rows)
    bl_long = pd.concat(bl_long_frames, ignore_index=True) if bl_long_frames else pd.DataFrame()
    return detailed, bl_long


# ============================================================
# ------------------------- SUMMARY ---------------------------
# ============================================================


def summarize_numeric(detailed: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (method, pct), g0 in detailed.groupby(["method", "pct_missing"], sort=True):
        ok = g0[g0["success"] == True].copy()
        row: Dict[str, object] = {
            "method": method,
            "pct_missing": int(pct),
            "n_files": int(len(g0)),
            "n_success": int(len(ok)),
            "n_failed": int(len(g0) - len(ok)),
        }
        for metric in SUMMARY_METRICS:
            if metric in ok.columns:
                vals = pd.to_numeric(ok[metric], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
                row[f"{metric}_std"] = safe_std(vals) if vals.size else float("nan")
            else:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pct_missing", "method"]).reset_index(drop=True)


def summarize_formatted(summary_num: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, r in summary_num.iterrows():
        row: Dict[str, object] = {
            "method": r["method"],
            "% Missing": f"{int(r['pct_missing'])}%",
            "n_files": int(r["n_files"]),
            "n_success": int(r["n_success"]),
            "n_failed": int(r["n_failed"]),
        }
        for metric in SUMMARY_METRICS:
            row[metric] = fmt_pm(float(r[f"{metric}_mean"]), float(r[f"{metric}_std"]), 6)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# --------------------------- MAIN ----------------------------
# ============================================================


def main() -> None:
    ensure_dir(OUT_DIR)

    tree_path = first_existing(ML_TREE_CANDIDATES, "IQ-TREE ML treefile")
    print(f"Using ML reference tree: {tree_path}")
    design = build_ml_topology_design(tree_path)
    print(f"ML tree: n_taxa={len(design.tree_labels)}, n_edges={len(design.b_ref)}, n_pairs={design.pairs.shape[0]}")

    # Save ML reference branch lengths and patristic distances.
    ref_bl_df = pd.DataFrame(
        {
            "edge_id": design.edge_ids,
            "edge_type": design.edge_types,
            "edge_descendant_count": design.edge_descendant_counts,
            "branch_length_ref_ML": design.b_ref,
        }
    )
    ref_bl_df.to_csv(os.path.join(OUT_DIR, "reference_ml_branch_lengths.csv"), index=False)
    pd.DataFrame(design.ml_patristic_matrix, index=design.tree_labels, columns=design.tree_labels).to_csv(
        os.path.join(OUT_DIR, "reference_ml_patristic_matrix.csv")
    )

    # Reference matrix is loaded only to verify label compatibility early.
    ref_matrix_path = first_existing(REFERENCE_MATRIX_CANDIDATES, "reference matrix")
    Dref, ref_labels = load_distance_matrix_csv(ref_matrix_path)
    _Dref_reordered, matched = reorder_matrix_to_tree_labels(Dref, ref_labels, design.tree_labels)
    print(f"Reference matrix labels matched to ML tree: {ref_matrix_path}")

    inventory = discover_all_files()
    mask_df = load_mask_registry()
    inventory = merge_mask_metadata(inventory, mask_df)
    inventory.to_csv(os.path.join(OUT_DIR, "method_file_inventory.csv"), index=False)

    print("\nFiles discovered:")
    print(inventory.groupby(["method", "pct_missing"]).size().to_string())

    detailed, bl_long = evaluate_all(inventory, design)

    detailed_path = os.path.join(OUT_DIR, "ml_branch_length_detailed_all_methods.csv")
    bl_long_path = os.path.join(OUT_DIR, "ml_branch_lengths_long_all_methods.csv")
    detailed.to_csv(detailed_path, index=False)
    bl_long.to_csv(bl_long_path, index=False)

    summary_num = summarize_numeric(detailed)
    summary_fmt = summarize_formatted(summary_num)
    summary_num_path = os.path.join(OUT_DIR, "ml_branch_length_summary_numeric.csv")
    summary_fmt_path = os.path.join(OUT_DIR, "ml_branch_length_summary_meanstd.csv")
    summary_num.to_csv(summary_num_path, index=False)
    summary_fmt.to_csv(summary_fmt_path, index=False)

    print("\n=== Branch-length summary mean ± std ===")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 260):
        print(summary_fmt.to_string(index=False))

    try:
        zip_path = shutil.make_archive(OUT_DIR, "zip", OUT_DIR)
        print(f"\nSaved ZIP archive: {zip_path}")
    except Exception as e:
        print(f"Could not create ZIP archive: {repr(e)}")

    print("\nSaved outputs:")
    print(f"  {detailed_path}")
    print(f"  {bl_long_path}")
    print(f"  {summary_num_path}")
    print(f"  {summary_fmt_path}")
    print(f"  {os.path.join(OUT_DIR, 'method_file_inventory.csv')}")


if __name__ == "__main__":
    main()
