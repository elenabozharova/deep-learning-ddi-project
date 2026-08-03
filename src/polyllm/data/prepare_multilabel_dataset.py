"""
Milestone 1 — Pair-level multi-label dataset preparation for PolyLLM.

Transforms ChChSe-Decagon_polypharmacy.csv.gz into four outputs:
  data/processed/polyllm_pairs.parquet       canonical drug-pair table
  data/processed/polyllm_label_mapping.csv   retained side-effect index
  data/processed/polyllm_labels.npy          multi-hot label matrix (uint8)
  outputs/polyllm/data_audit.json            processing audit trail

All transformations are deterministic.  No random operations are performed.

Run from the project root:
    python src/polyllm/data/prepare_multilabel_dataset.py
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401 — imported for version reporting

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: list[str] = [
    "# STITCH 1",
    "STITCH 2",
    "Polypharmacy Side Effect",
    "Side Effect Name",
]

COL_DRUG1_RAW = "# STITCH 1"
COL_DRUG2_RAW = "STITCH 2"
COL_SE_ID = "Polypharmacy Side Effect"
COL_SE_NAME = "Side Effect Name"

DEFAULT_INPUT = Path("data/raw/ChChSe-Decagon_polypharmacy.csv.gz")
DEFAULT_PAIRS_OUT = Path("data/processed/polyllm_pairs.parquet")
DEFAULT_LABEL_MAPPING_OUT = Path("data/processed/polyllm_label_mapping.csv")
DEFAULT_LABELS_OUT = Path("data/processed/polyllm_labels.npy")
DEFAULT_AUDIT_OUT = Path("outputs/polyllm/data_audit.json")
DEFAULT_MIN_PAIR_FREQ = 500

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _write_bytes_atomic(data: bytes, path: Path) -> None:
    """Write bytes to path using a sibling temp file, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(str(tmp), str(path))
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------

def load_raw(path: Path) -> pd.DataFrame:
    """Load the compressed CSV with every column read as a string."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, compression="gzip", dtype=str)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Step 2: Column validation
# ---------------------------------------------------------------------------

def validate_columns(
    df: pd.DataFrame,
    required: list[str] | None = None,
) -> None:
    """Raise ValueError if any required column is absent."""
    if required is None:
        required = REQUIRED_COLUMNS
    observed = list(df.columns)
    missing = [c for c in required if c not in observed]
    if missing:
        raise ValueError(
            "Required columns missing from input.\n"
            f"  Required : {required}\n"
            f"  Observed : {observed}\n"
            f"  Missing  : {missing}"
        )


# ---------------------------------------------------------------------------
# Step 3: Whitespace normalisation
# ---------------------------------------------------------------------------

def normalize_whitespace(
    df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Strip leading/trailing whitespace from the specified string columns.
    Returns (modified_df, {col: number_of_values_changed}).
    """
    changed: dict[str, int] = {}
    df = df.copy()
    for col in cols:
        stripped = df[col].str.strip()
        n = int((stripped != df[col]).sum())
        changed[col] = n
        if n:
            logger.info("Whitespace stripped in %r: %d values changed", col, n)
        df[col] = stripped
    return df, changed


# ---------------------------------------------------------------------------
# Step 4: Missing-value check
# ---------------------------------------------------------------------------

def check_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """
    Count missing or blank values per required column after normalisation.

    Raises ValueError if STITCH identifier or side-effect ID columns have
    any missing/blank entries.  A missing side-effect name is logged as a
    warning but is not blocking.

    Returns counts for the audit.
    """
    counts: dict[str, int] = {}
    blocking: list[str] = []

    for col in REQUIRED_COLUMNS:
        # Count NaN and empty-string-after-strip as missing
        total = int((df[col].fillna("").str.strip() == "").sum())
        counts[col] = total
        if col != COL_SE_NAME and total > 0:
            blocking.append(f"  {col!r}: {total} missing or blank")
        elif col == COL_SE_NAME and total > 0:
            logger.warning(
                "%r has %d missing/blank values — these rows will be excluded from "
                "name-conflict validation but the side-effect ID is still usable.",
                col,
                total,
            )

    if blocking:
        raise ValueError(
            "Missing values found in identifier columns — cannot continue:\n"
            + "\n".join(blocking)
        )
    return counts


# ---------------------------------------------------------------------------
# Step 5: Side-effect identity validation
# ---------------------------------------------------------------------------

def validate_side_effect_names(
    df: pd.DataFrame,
) -> tuple[int, int, list[dict[str, Any]]]:
    """
    Verify that each side-effect identifier maps to at most one distinct name.

    Raises ValueError if any identifier has conflicting (multiple non-empty)
    names — the data cannot be safely processed without resolving the conflict.

    Returns:
        id_conflict_count   — number of IDs with >1 distinct name (must be 0)
        name_conflict_count — number of names mapping to >1 ID (informational)
        examples            — up to 3 conflict examples for the audit
    """
    valid = df[[COL_SE_ID, COL_SE_NAME]].copy()
    valid = valid[valid[COL_SE_NAME].fillna("").str.strip() != ""]

    id_to_names = valid.groupby(COL_SE_ID)[COL_SE_NAME].agg(lambda x: frozenset(x.unique()))
    id_conflicts = id_to_names[id_to_names.map(len) > 1]
    id_conflict_count = int(len(id_conflicts))

    examples: list[dict[str, Any]] = []
    if id_conflict_count > 0:
        for se_id, names in list(id_conflicts.items())[:3]:
            examples.append({"side_effect_id": se_id, "conflicting_names": sorted(names)})
        raise ValueError(
            f"Side-effect identifier-to-name conflicts: {id_conflict_count} identifier(s) "
            f"map to multiple distinct non-empty names.\n"
            f"Representative examples: {examples}\n"
            "Resolve conflicts before continuing."
        )

    name_to_ids = valid.groupby(COL_SE_NAME)[COL_SE_ID].agg(lambda x: frozenset(x.unique()))
    name_conflict_count = int((name_to_ids.map(len) > 1).sum())
    if name_conflict_count:
        logger.info(
            "%d side-effect name(s) map to multiple identifiers (informational only).",
            name_conflict_count,
        )

    return id_conflict_count, name_conflict_count, examples


# ---------------------------------------------------------------------------
# Step 6: Exact duplicate removal
# ---------------------------------------------------------------------------

def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove rows that are identical across all four required source columns.
    Returns (deduplicated_df, number_of_duplicates_removed).
    """
    original = len(df)
    df_out = df.drop_duplicates(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    removed = original - len(df_out)
    logger.info("Exact duplicate rows removed: %d (remaining: %d)", removed, len(df_out))
    return df_out, removed


# ---------------------------------------------------------------------------
# Step 7: Pair canonicalisation
# ---------------------------------------------------------------------------

def canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add drug_1 and drug_2 columns so that drug_1 <= drug_2 lexicographically.

    String comparison only — STITCH identifiers are never parsed as integers.
    """
    df = df.copy()
    a = df[COL_DRUG1_RAW]
    b = df[COL_DRUG2_RAW]
    df["drug_1"] = a.where(a <= b, b)
    df["drug_2"] = b.where(a <= b, a)
    return df


# ---------------------------------------------------------------------------
# Step 8: Self-pair detection and exclusion
# ---------------------------------------------------------------------------

def split_self_pairs(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Separate self-pairs (drug_1 == drug_2) from the working dataset.

    Returns (non_self_df, self_df, stats_dict).
    Self-pairs are excluded from modelling because the task concerns interactions
    between two *distinct* drugs.
    """
    mask = df["drug_1"] == df["drug_2"]
    self_df = df[mask]
    non_self_df = df[~mask].reset_index(drop=True)

    stats: dict[str, int] = {
        "self_pair_row_count": int(mask.sum()),
        "unique_self_pair_count": int(
            self_df[["drug_1", "drug_2"]].drop_duplicates().shape[0]
        ),
        "unique_se_in_self_pairs": int(self_df[COL_SE_ID].nunique()),
    }
    if stats["self_pair_row_count"]:
        logger.info(
            "Self-pairs excluded: %d rows, %d unique pairs, %d unique side effects.",
            stats["self_pair_row_count"],
            stats["unique_self_pair_count"],
            stats["unique_se_in_self_pairs"],
        )
    return non_self_df, self_df, stats


# ---------------------------------------------------------------------------
# Step 9: Canonical duplicate triple removal
# ---------------------------------------------------------------------------

def remove_canonical_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove triples that are identical after canonicalisation.

    A triple is the combination (drug_1, drug_2, Polypharmacy Side Effect).
    This collapses both repeated source rows and reversed A–B / B–A entries
    that represent the same drug-pair–side-effect association.
    """
    original = len(df)
    df_out = df.drop_duplicates(
        subset=["drug_1", "drug_2", COL_SE_ID]
    ).reset_index(drop=True)
    removed = original - len(df_out)
    logger.info("Canonical duplicate triples removed: %d (remaining: %d)", removed, len(df_out))
    return df_out, removed


# ---------------------------------------------------------------------------
# Step 10: Side-effect frequency counting
# ---------------------------------------------------------------------------

def count_se_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count unique canonical drug pairs per side-effect identifier.

    Precondition: df contains unique (drug_1, drug_2, SE_ID) triples, so
    groupby size() equals the unique canonical pair count per side effect.

    Returns DataFrame with columns [side_effect_id, side_effect_name, unique_pair_count].
    """
    counts = (
        df.groupby(COL_SE_ID, sort=True)
        .size()
        .rename("unique_pair_count")
    )
    # One name per ID is guaranteed by validate_side_effect_names
    names = (
        df.groupby(COL_SE_ID, sort=True)[COL_SE_NAME]
        .first()
        .rename("side_effect_name")
    )
    freq = pd.concat([counts, names], axis=1).reset_index()
    freq = freq.rename(columns={COL_SE_ID: "side_effect_id"})
    freq["unique_pair_count"] = freq["unique_pair_count"].astype(int)
    return freq


# ---------------------------------------------------------------------------
# Step 11: Side-effect filtering
# ---------------------------------------------------------------------------

def filter_side_effects(
    freq_df: pd.DataFrame,
    min_pairs: int = DEFAULT_MIN_PAIR_FREQ,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retain side effects with unique_pair_count >= min_pairs (inclusive).
    Returns (retained_df, excluded_df).
    """
    mask = freq_df["unique_pair_count"] >= min_pairs
    return freq_df[mask].copy(), freq_df[~mask].copy()


# ---------------------------------------------------------------------------
# Step 12: Build pair table
# ---------------------------------------------------------------------------

def build_pair_table(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique canonical pairs, sort lexicographically, assign pair_id.

    Returns DataFrame with columns [pair_id, drug_1, drug_2].
    pair_id is zero-based and sequential.
    """
    pairs = (
        df_filtered[["drug_1", "drug_2"]]
        .drop_duplicates()
        .sort_values(["drug_1", "drug_2"])
        .reset_index(drop=True)
    )
    pairs.insert(0, "pair_id", range(len(pairs)))
    pairs["pair_id"] = pairs["pair_id"].astype(int)
    return pairs


# ---------------------------------------------------------------------------
# Step 13: Build label mapping
# ---------------------------------------------------------------------------

def build_label_mapping(retained_freq: pd.DataFrame) -> pd.DataFrame:
    """
    Sort retained side effects by side_effect_id, assign label_index.

    Returns DataFrame with columns [label_index, side_effect_id,
    side_effect_name, unique_pair_count].
    Ordering is lexicographic by side_effect_id, independent of frequency.
    """
    mapping = (
        retained_freq[["side_effect_id", "side_effect_name", "unique_pair_count"]]
        .sort_values("side_effect_id")
        .reset_index(drop=True)
        .copy()
    )
    mapping.insert(0, "label_index", range(len(mapping)))
    mapping["label_index"] = mapping["label_index"].astype(int)
    return mapping


# ---------------------------------------------------------------------------
# Step 14: Build multi-hot label matrix
# ---------------------------------------------------------------------------

def build_label_matrix(
    df_filtered: pd.DataFrame,
    pair_table: pd.DataFrame,
    label_mapping: pd.DataFrame,
) -> np.ndarray:
    """
    Build a dense binary matrix of shape (n_pairs, n_labels) with dtype uint8.

    Row i corresponds to pair_id == i.
    Column j corresponds to label_index == j.
    Entry is 1 if that pair is associated with that side effect, else 0.
    """
    n_pairs = len(pair_table)
    n_labels = len(label_mapping)

    # Build index maps
    pair_id_map: dict[tuple[str, str], int] = dict(
        zip(
            zip(pair_table["drug_1"], pair_table["drug_2"]),
            pair_table["pair_id"],
        )
    )
    label_id_map: dict[str, int] = dict(
        zip(label_mapping["side_effect_id"], label_mapping["label_index"])
    )

    # Unique canonical triples (defensive dedup)
    triples = df_filtered[["drug_1", "drug_2", COL_SE_ID]].drop_duplicates()

    # Map to integer indices via merge
    triples = triples.merge(
        pair_table[["drug_1", "drug_2", "pair_id"]],
        on=["drug_1", "drug_2"],
        how="left",
    )
    triples = triples.copy()
    triples["label_index"] = triples[COL_SE_ID].map(label_id_map)
    triples = triples.dropna(subset=["pair_id", "label_index"])

    matrix = np.zeros((n_pairs, n_labels), dtype=np.uint8)
    rows = triples["pair_id"].astype(int).values
    cols = triples["label_index"].astype(int).values
    matrix[rows, cols] = 1
    return matrix


# ---------------------------------------------------------------------------
# In-memory invariant check (pre-write)
# ---------------------------------------------------------------------------

def validate_invariants(
    pair_table: pd.DataFrame,
    label_mapping: pd.DataFrame,
    matrix: np.ndarray,
    min_pair_freq: int,
) -> None:
    """Validate in-memory outputs before writing.  Raises ValueError on failure."""
    n_pairs = len(pair_table)
    n_labels = len(label_mapping)
    issues: list[str] = []

    if not (pair_table["drug_1"] < pair_table["drug_2"]).all():
        issues.append("Some pairs violate drug_1 < drug_2.")
    if list(pair_table["pair_id"]) != list(range(n_pairs)):
        issues.append("pair_id is not sequential 0-indexed.")
    if pair_table[["drug_1", "drug_2"]].duplicated().any():
        issues.append("Duplicate canonical pairs in pair table.")
    if (pair_table["drug_1"] == pair_table["drug_2"]).any():
        issues.append("Self-pair(s) found in pair table.")
    if list(label_mapping["label_index"]) != list(range(n_labels)):
        issues.append("label_index is not sequential 0-indexed.")
    if matrix.dtype != np.uint8:
        issues.append(f"Matrix dtype is {matrix.dtype!r}, expected uint8.")
    if not ((matrix == 0) | (matrix == 1)).all():
        issues.append("Matrix contains values other than 0 and 1.")
    if matrix.shape != (n_pairs, n_labels):
        issues.append(
            f"Matrix shape {matrix.shape} != expected ({n_pairs}, {n_labels})."
        )
    if not (matrix.sum(axis=1) >= 1).all():
        zero_count = int((matrix.sum(axis=1) < 1).sum())
        issues.append(f"{zero_count} pair(s) have no positive labels.")
    below = int((matrix.sum(axis=0) < min_pair_freq).sum())
    if below:
        issues.append(
            f"{below} label(s) have fewer than {min_pair_freq} positive pairs."
        )

    if issues:
        raise ValueError(
            "Pre-write validation failed:\n"
            + "\n".join(f"  - {i}" for i in issues)
        )
    logger.info("Pre-write validation passed.")


# ---------------------------------------------------------------------------
# Post-run validation (reloads from disk)
# ---------------------------------------------------------------------------

def validate_outputs_from_disk(
    pairs_path: Path,
    label_mapping_path: Path,
    labels_path: Path,
    min_pair_freq: int,
) -> None:
    """
    Reload all three outputs from disk and re-verify invariants independently.
    Raises AssertionError with a descriptive message on any failure.
    """
    pairs = pd.read_parquet(pairs_path)
    mapping = pd.read_csv(
        label_mapping_path,
        dtype={"side_effect_id": str, "side_effect_name": str},
    )
    matrix = np.load(labels_path)

    n_pairs = len(pairs)
    n_labels = len(mapping)

    def _check(condition: bool, msg: str) -> None:
        if not condition:
            raise AssertionError(f"Post-run validation FAILED — {msg}")

    _check(
        (pairs["drug_1"] < pairs["drug_2"]).all(),
        "drug_1 < drug_2 violated for at least one row.",
    )
    _check(
        list(pairs["pair_id"]) == list(range(n_pairs)),
        f"pair_id is not exactly 0..{n_pairs - 1}.",
    )
    _check(
        not pairs[["drug_1", "drug_2"]].duplicated().any(),
        "Duplicate canonical pairs in saved pair table.",
    )
    _check(
        not (pairs["drug_1"] == pairs["drug_2"]).any(),
        "Self-pairs found in saved pair table.",
    )
    _check(
        list(mapping["label_index"]) == list(range(n_labels)),
        f"label_index is not exactly 0..{n_labels - 1}.",
    )
    _check(matrix.dtype == np.uint8, f"Matrix dtype is {matrix.dtype!r}, not uint8.")
    _check(
        bool(((matrix == 0) | (matrix == 1)).all()),
        "Matrix contains values other than 0 and 1.",
    )
    _check(
        matrix.shape == (n_pairs, n_labels),
        f"Matrix shape {matrix.shape} != ({n_pairs}, {n_labels}).",
    )
    _check(
        bool((matrix.sum(axis=1) >= 1).all()),
        "At least one pair has no positive labels.",
    )
    _check(
        bool((matrix.sum(axis=0) >= min_pair_freq).all()),
        f"At least one label has fewer than {min_pair_freq} positive pairs.",
    )
    _check(pairs.isna().sum().sum() == 0, "NaN values in saved pair table.")
    _check(mapping.isna().sum().sum() == 0, "NaN values in saved label mapping.")

    # uint8 cannot hold NaN or inf; assertion is vacuously satisfied but we confirm dtype
    _check(matrix.dtype == np.uint8, "Matrix dtype must be uint8 (precludes NaN/inf).")

    logger.info("All post-run disk validations passed.")


# ---------------------------------------------------------------------------
# Audit construction
# ---------------------------------------------------------------------------

def _software_versions() -> dict[str, str]:
    import pyarrow as _pa
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": _pa.__version__,
    }


def build_audit(
    input_path: Path,
    df_raw: pd.DataFrame,
    df_after_exact_dedup: pd.DataFrame,
    self_pair_stats: dict[str, int],
    canonical_dup_count: int,
    df_unique: pd.DataFrame,
    freq_df: pd.DataFrame,
    retained_freq: pd.DataFrame,
    excluded_freq: pd.DataFrame,
    pair_table: pd.DataFrame,
    label_mapping: pd.DataFrame,
    matrix: np.ndarray,
    missing_counts: dict[str, int],
    whitespace_counts: dict[str, int],
    se_id_conflicts: int,
    se_name_conflicts: int,
    min_pair_freq: int,
) -> dict[str, Any]:
    labels_per_pair = matrix.sum(axis=1)
    unique_pairs_before_filter = int(
        df_unique[["drug_1", "drug_2"]].drop_duplicates().shape[0]
    )
    unique_drugs_before_filter = int(
        pd.concat([df_unique["drug_1"], df_unique["drug_2"]]).nunique()
    )
    final_drugs = int(
        pd.concat([pair_table["drug_1"], pair_table["drug_2"]]).nunique()
    )

    return {
        "input_path": str(input_path),
        "input_file_size_bytes": int(input_path.stat().st_size),
        "required_columns": REQUIRED_COLUMNS,
        "observed_columns": list(df_raw.columns),
        "raw_row_count": int(len(df_raw)),
        "exact_duplicate_row_count": int(len(df_raw) - len(df_after_exact_dedup)),
        "rows_after_exact_duplicate_removal": int(len(df_after_exact_dedup)),
        "unique_drug_count_before_filtering": unique_drugs_before_filter,
        "self_pair_row_count": self_pair_stats["self_pair_row_count"],
        "unique_self_pair_count": self_pair_stats["unique_self_pair_count"],
        "rows_after_self_pair_exclusion": int(
            len(df_after_exact_dedup) - self_pair_stats["self_pair_row_count"]
        ),
        "canonical_duplicate_triple_count": canonical_dup_count,
        "unique_canonical_pair_count_before_label_filtering": unique_pairs_before_filter,
        "original_side_effect_count": int(len(freq_df)),
        "minimum_pair_frequency_threshold": min_pair_freq,
        "retained_side_effect_count": int(len(retained_freq)),
        "excluded_side_effect_count": int(len(excluded_freq)),
        "side_effect_pair_frequency_stats": {
            "min": int(freq_df["unique_pair_count"].min()),
            "max": int(freq_df["unique_pair_count"].max()),
            "median": float(freq_df["unique_pair_count"].median()),
        },
        "retained_triple_count": int(
            (df_unique[COL_SE_ID].isin(set(retained_freq["side_effect_id"]))).sum()
        ),
        "final_pair_count": int(len(pair_table)),
        "final_unique_drug_count": final_drugs,
        "mean_labels_per_pair": float(round(float(labels_per_pair.mean()), 4)),
        "median_labels_per_pair": float(np.median(labels_per_pair)),
        "minimum_labels_per_pair": int(labels_per_pair.min()),
        "maximum_labels_per_pair": int(labels_per_pair.max()),
        "label_matrix_shape": list(matrix.shape),
        "label_matrix_dtype": str(matrix.dtype),
        "missing_value_counts": missing_counts,
        "whitespace_normalization_counts": whitespace_counts,
        "side_effect_identifier_name_conflict_count": se_id_conflicts,
        "side_effect_name_identifier_conflict_count": se_name_conflicts,
        "software_versions": _software_versions(),
        "random_seed": None,
        "random_seed_note": (
            "No random operation is used in deterministic dataset preparation. "
            "Seed 42 applies to later split creation (Milestone 2+)."
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_preprocessing(
    input_path: Path = DEFAULT_INPUT,
    pairs_output: Path = DEFAULT_PAIRS_OUT,
    label_mapping_output: Path = DEFAULT_LABEL_MAPPING_OUT,
    labels_output: Path = DEFAULT_LABELS_OUT,
    audit_output: Path = DEFAULT_AUDIT_OUT,
    min_pair_frequency: int = DEFAULT_MIN_PAIR_FREQ,
) -> dict[str, Any]:
    """
    Execute the full Milestone 1 preprocessing pipeline.

    Writes outputs atomically — a failed run leaves no partially-valid files.
    Calls validate_outputs_from_disk after writing to independently verify
    the saved files.

    Returns the audit dict.
    """
    logger.info("=== Milestone 1 preprocessing started ===")
    logger.info("Input : %s", input_path)
    logger.info("Min pair frequency : %d", min_pair_frequency)

    # Ensure output parent directories exist before any processing
    for p in [pairs_output, label_mapping_output, labels_output, audit_output]:
        p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df_raw = load_raw(input_path)

    # 2. Validate columns
    validate_columns(df_raw)

    # 3. Normalise whitespace (on a copy; record what changed)
    df, whitespace_counts = normalize_whitespace(df_raw, REQUIRED_COLUMNS)

    # 4. Check missing values (after normalisation so blanks are detected)
    missing_counts = check_missing_values(df)

    # 5. Validate side-effect identity
    se_id_conflicts, se_name_conflicts, _ = validate_side_effect_names(df)

    # 6. Exact duplicate removal
    df_dedup, _ = remove_exact_duplicates(df)

    # 7. Pair canonicalisation
    df_canon = canonicalize_pairs(df_dedup)

    # 8. Self-pair exclusion
    df_no_self, _self_df, self_stats = split_self_pairs(df_canon)

    # 9. Canonical duplicate triple removal
    df_unique, canonical_dup_count = remove_canonical_duplicates(df_no_self)

    # 10. Side-effect frequency
    freq_df = count_se_frequency(df_unique)

    # 11. Filter side effects
    retained_freq, excluded_freq = filter_side_effects(freq_df, min_pair_frequency)
    retained_se_ids = set(retained_freq["side_effect_id"])
    logger.info(
        "Side effects: %d total, %d retained (>= %d pairs), %d excluded.",
        len(freq_df),
        len(retained_freq),
        min_pair_frequency,
        len(excluded_freq),
    )

    # 12. Filter triples to retained side effects
    df_filtered = df_unique[df_unique[COL_SE_ID].isin(retained_se_ids)].copy()
    logger.info("Retained triples: %d", len(df_filtered))

    # 13. Pair table
    pair_table = build_pair_table(df_filtered)
    logger.info("Final pairs: %d", len(pair_table))

    # 14. Label mapping
    label_mapping = build_label_mapping(retained_freq)
    logger.info("Final labels: %d", len(label_mapping))

    # 15. Label matrix
    matrix = build_label_matrix(df_filtered, pair_table, label_mapping)
    logger.info("Label matrix shape: %s", matrix.shape)

    # 16. Pre-write invariant check
    validate_invariants(pair_table, label_mapping, matrix, min_pair_frequency)

    # 17. Build audit
    audit = build_audit(
        input_path=input_path,
        df_raw=df_raw,
        df_after_exact_dedup=df_dedup,
        self_pair_stats=self_stats,
        canonical_dup_count=canonical_dup_count,
        df_unique=df_unique,
        freq_df=freq_df,
        retained_freq=retained_freq,
        excluded_freq=excluded_freq,
        pair_table=pair_table,
        label_mapping=label_mapping,
        matrix=matrix,
        missing_counts=missing_counts,
        whitespace_counts=whitespace_counts,
        se_id_conflicts=se_id_conflicts,
        se_name_conflicts=se_name_conflicts,
        min_pair_freq=min_pair_frequency,
    )

    # 18. Write outputs atomically
    buf = io.BytesIO()
    pair_table.to_parquet(buf, index=False)
    _write_bytes_atomic(buf.getvalue(), pairs_output)
    logger.info("Wrote pairs: %s", pairs_output)

    _write_bytes_atomic(
        label_mapping.to_csv(index=False).encode(),
        label_mapping_output,
    )
    logger.info("Wrote label mapping: %s", label_mapping_output)

    buf = io.BytesIO()
    np.save(buf, matrix, allow_pickle=False)
    _write_bytes_atomic(buf.getvalue(), labels_output)
    logger.info("Wrote label matrix: %s", labels_output)

    _write_bytes_atomic(
        json.dumps(audit, indent=2, ensure_ascii=False).encode(),
        audit_output,
    )
    logger.info("Wrote audit: %s", audit_output)

    # 19. Post-run disk validation
    validate_outputs_from_disk(pairs_output, label_mapping_output, labels_output, min_pair_frequency)

    logger.info("=== Milestone 1 preprocessing complete ===")
    return audit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Milestone 1 — prepare pair-level multi-label dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--pairs-output", type=Path, default=DEFAULT_PAIRS_OUT)
    p.add_argument("--label-mapping-output", type=Path, default=DEFAULT_LABEL_MAPPING_OUT)
    p.add_argument("--labels-output", type=Path, default=DEFAULT_LABELS_OUT)
    p.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUT)
    p.add_argument(
        "--min-pair-frequency",
        type=int,
        default=DEFAULT_MIN_PAIR_FREQ,
        help="Minimum unique canonical pair count for a side effect to be retained.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)
    audit = run_preprocessing(
        input_path=args.input,
        pairs_output=args.pairs_output,
        label_mapping_output=args.label_mapping_output,
        labels_output=args.labels_output,
        audit_output=args.audit_output,
        min_pair_frequency=args.min_pair_frequency,
    )
    print(f"\nDone.  Final pairs: {audit['final_pair_count']:,}  "
          f"Labels: {audit['retained_side_effect_count']:,}  "
          f"Matrix: {audit['label_matrix_shape']}")


if __name__ == "__main__":
    main()
