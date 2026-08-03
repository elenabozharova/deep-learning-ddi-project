"""
Milestone 3 — Create permanent pair-random train/validation/test splits.

Protocol
--------
1. Compute integer split sizes using the rounding rule:
       n_test       = round(0.10 * n_pairs)
       n_validation = round(0.10 * n_pairs)
       n_train      = n_pairs - n_validation - n_test
2. Shuffle the complete pair-ID array with:
       rng = np.random.default_rng(42)
3. Assign: first n_train IDs → train, next n_validation → validation,
   remaining → test.
4. Sort IDs numerically within each split (stable, readable output).
5. Write one CSV per split containing only `pair_id`.
6. Write a full audit JSON.

Scope boundary: does NOT generate embeddings, features, or models.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PAIRS_INPUT   = Path("data/processed/polyllm_pairs.parquet")
DEFAULT_LABELS_INPUT  = Path("data/processed/polyllm_labels.npy")
DEFAULT_LABEL_MAP_INPUT = Path("data/processed/polyllm_label_mapping.csv")
DEFAULT_TRAIN_OUTPUT  = Path("data/splits/train_pair_ids.csv")
DEFAULT_VAL_OUTPUT    = Path("data/splits/validation_pair_ids.csv")
DEFAULT_TEST_OUTPUT   = Path("data/splits/test_pair_ids.csv")
DEFAULT_AUDIT_OUTPUT  = Path("outputs/polyllm/split_audit.json")

RANDOM_SEED    = 42
TRAIN_FRACTION = 0.80
VAL_FRACTION   = 0.10
TEST_FRACTION  = 0.10


# ---------------------------------------------------------------------------
# Step 1: Validate inputs
# ---------------------------------------------------------------------------

def validate_pair_table(pairs_df: pd.DataFrame) -> None:
    """Raise ValueError if the pair table does not meet Milestone 3 preconditions."""
    required = {"pair_id", "drug_1", "drug_2"}
    missing = required - set(pairs_df.columns)
    if missing:
        raise ValueError(f"Pair table missing columns: {sorted(missing)}")

    n = len(pairs_df)
    ids = pairs_df["pair_id"].to_numpy()
    expected = np.arange(n)
    if not np.array_equal(np.sort(ids), expected):
        raise ValueError(
            f"pair_id must be exactly 0..{n - 1}. "
            f"Got range {ids.min()}..{ids.max()} with {len(np.unique(ids))} unique values."
        )


def validate_label_matrix(pairs_df: pd.DataFrame, labels: np.ndarray) -> None:
    """Raise ValueError if the label matrix row count does not match the pair table."""
    if labels.ndim != 2:
        raise ValueError(f"Label matrix must be 2-D; got {labels.ndim}-D.")
    if labels.shape[0] != len(pairs_df):
        raise ValueError(
            f"Label matrix has {labels.shape[0]} rows but pair table has {len(pairs_df)} rows."
        )


# ---------------------------------------------------------------------------
# Step 2: Compute split sizes
# ---------------------------------------------------------------------------

def compute_split_sizes(
    n_pairs: int,
    val_frac: float = VAL_FRACTION,
    test_frac: float = TEST_FRACTION,
) -> tuple[int, int, int]:
    """
    Return (n_train, n_validation, n_test) using the approved rounding rule.

        n_test       = round(test_frac  * n_pairs)
        n_validation = round(val_frac   * n_pairs)
        n_train      = n_pairs - n_validation - n_test
    """
    n_test = round(test_frac * n_pairs)
    n_val  = round(val_frac  * n_pairs)
    n_train = n_pairs - n_val - n_test
    return n_train, n_val, n_test


# ---------------------------------------------------------------------------
# Step 3: Create splits
# ---------------------------------------------------------------------------

def create_splits(
    pair_ids: np.ndarray,
    seed: int = RANDOM_SEED,
    val_frac: float = VAL_FRACTION,
    test_frac: float = TEST_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle pair_ids with the given seed and partition into train/val/test.

    Returns three sorted numpy arrays: (train_ids, val_ids, test_ids).
    Sorting after assignment is purely cosmetic and does not change membership.
    """
    n = len(pair_ids)
    n_train, n_val, n_test = compute_split_sizes(n, val_frac, test_frac)

    rng = np.random.default_rng(seed)
    shuffled = pair_ids.copy().astype(np.int64)
    rng.shuffle(shuffled)

    train_ids = np.sort(shuffled[:n_train])
    val_ids   = np.sort(shuffled[n_train : n_train + n_val])
    test_ids  = np.sort(shuffled[n_train + n_val :])

    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Step 4: Validate split properties
# ---------------------------------------------------------------------------

def validate_splits(
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    all_pair_ids: np.ndarray,
) -> dict[str, bool]:
    """
    Run all required split validation checks.

    Returns a dict mapping check name → bool (True = passed).
    Raises ValueError on any failure so callers fail fast.
    """
    results: dict[str, bool] = {}

    def _check(name: str, condition: bool, msg: str) -> None:
        results[name] = condition
        if not condition:
            raise ValueError(f"Split validation failed [{name}]: {msg}")

    # No duplicates within splits
    _check("no_duplicates_train", len(train_ids) == len(np.unique(train_ids)),
           "Duplicate pair IDs in train split.")
    _check("no_duplicates_validation", len(val_ids) == len(np.unique(val_ids)),
           "Duplicate pair IDs in validation split.")
    _check("no_duplicates_test", len(test_ids) == len(np.unique(test_ids)),
           "Duplicate pair IDs in test split.")

    # Pairwise disjoint
    tv = np.intersect1d(train_ids, val_ids)
    _check("train_validation_disjoint", len(tv) == 0,
           f"train ∩ validation = {tv[:5]} (and possibly more).")
    tt = np.intersect1d(train_ids, test_ids)
    _check("train_test_disjoint", len(tt) == 0,
           f"train ∩ test = {tt[:5]} (and possibly more).")
    vt = np.intersect1d(val_ids, test_ids)
    _check("validation_test_disjoint", len(vt) == 0,
           f"validation ∩ test = {vt[:5]} (and possibly more).")

    # Complete coverage
    union = np.sort(np.concatenate([train_ids, val_ids, test_ids]))
    expected = np.sort(all_pair_ids)
    _check("complete_coverage", np.array_equal(union, expected),
           "Union of splits does not equal all pair IDs.")

    # All IDs exist in pair table
    pair_id_set = set(int(x) for x in all_pair_ids)
    all_split_ids = np.concatenate([train_ids, val_ids, test_ids])
    alien = [int(x) for x in all_split_ids if int(x) not in pair_id_set]
    _check("all_ids_exist", len(alien) == 0,
           f"Split IDs not in pair table: {alien[:5]}.")

    # Every pair ID occurs exactly once
    occurrence_counts = {}
    for pid in all_split_ids:
        occurrence_counts[int(pid)] = occurrence_counts.get(int(pid), 0) + 1
    multi = [k for k, v in occurrence_counts.items() if v != 1]
    _check("each_id_once", len(multi) == 0,
           f"Pair IDs appearing ≠1 time: {multi[:5]}.")

    return results


# ---------------------------------------------------------------------------
# Step 5: Label-distribution audit
# ---------------------------------------------------------------------------

def _label_stats_for_split(
    split_ids: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Compute label distribution statistics for one split."""
    sub = labels[split_ids]
    n_pairs = len(sub)
    n_labels = sub.shape[1]

    positives_per_pair = sub.sum(axis=1)
    positives_per_label = sub.sum(axis=0)

    labels_with_positives = int((positives_per_label > 0).sum())
    labels_with_zero = int((positives_per_label == 0).sum())
    zero_label_indices = [int(i) for i in np.where(positives_per_label == 0)[0]]

    prevalence = (positives_per_label / n_pairs).tolist()

    return {
        "pair_count": n_pairs,
        "total_positive_entries": int(sub.sum()),
        "mean_labels_per_pair": float(round(positives_per_pair.mean(), 4)),
        "median_labels_per_pair": float(np.median(positives_per_pair)),
        "min_labels_per_pair": int(positives_per_pair.min()),
        "max_labels_per_pair": int(positives_per_pair.max()),
        "labels_with_at_least_one_positive": labels_with_positives,
        "labels_with_zero_positives": labels_with_zero,
        "zero_positive_label_indices": zero_label_indices,
        "min_positives_per_label": int(positives_per_label.min()),
        "median_positives_per_label": float(np.median(positives_per_label)),
        "max_positives_per_label": int(positives_per_label.max()),
        "prevalence_per_label": prevalence,
    }


def compute_label_distribution_summary(
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    labels: np.ndarray,
    top_k: int = 10,
) -> dict[str, Any]:
    """Build the full label-distribution audit section."""
    all_ids = np.arange(labels.shape[0])

    full_stats  = _label_stats_for_split(all_ids, labels)
    train_stats = _label_stats_for_split(train_ids, labels)
    val_stats   = _label_stats_for_split(val_ids, labels)
    test_stats  = _label_stats_for_split(test_ids, labels)

    full_prev  = np.array(full_stats["prevalence_per_label"])
    train_prev = np.array(train_stats["prevalence_per_label"])
    val_prev   = np.array(val_stats["prevalence_per_label"])
    test_prev  = np.array(test_stats["prevalence_per_label"])

    def _diff_summary(split_prev: np.ndarray, split_name: str) -> dict[str, Any]:
        abs_diff = np.abs(split_prev - full_prev)
        top_idx = np.argsort(abs_diff)[::-1][:top_k]
        return {
            "max_abs_prevalence_diff": float(round(abs_diff.max(), 6)),
            "mean_abs_prevalence_diff": float(round(abs_diff.mean(), 6)),
            "top_labels_by_prevalence_diff": [
                {
                    "label_index": int(i),
                    "full_prevalence": float(round(full_prev[i], 6)),
                    f"{split_name}_prevalence": float(round(split_prev[i], 6)),
                    "abs_diff": float(round(abs_diff[i], 6)),
                }
                for i in top_idx
            ],
        }

    return {
        "full_dataset": full_stats,
        "train": train_stats,
        "validation": val_stats,
        "test": test_stats,
        "train_vs_full": _diff_summary(train_prev, "train"),
        "validation_vs_full": _diff_summary(val_prev, "validation"),
        "test_vs_full": _diff_summary(test_prev, "test"),
    }


# ---------------------------------------------------------------------------
# Step 6: Drug-overlap audit
# ---------------------------------------------------------------------------

def compute_drug_overlap(
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    pairs_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute drug overlap across splits."""

    def _drugs(ids: np.ndarray) -> set[str]:
        sub = pairs_df[pairs_df["pair_id"].isin(set(ids.tolist()))]
        return set(sub["drug_1"].tolist()) | set(sub["drug_2"].tolist())

    train_drugs = _drugs(train_ids)
    val_drugs   = _drugs(val_ids)
    test_drugs  = _drugs(test_ids)
    all_drugs   = train_drugs | val_drugs | test_drugs

    in_all   = train_drugs & val_drugs & test_drugs
    only_train = train_drugs - val_drugs - test_drugs
    only_val   = val_drugs   - train_drugs - test_drugs
    only_test  = test_drugs  - train_drugs - val_drugs

    # Test-pair eligibility
    test_pairs = pairs_df[pairs_df["pair_id"].isin(set(test_ids.tolist()))]
    n_test_pairs = len(test_pairs)

    both_in_train = (
        test_pairs["drug_1"].isin(train_drugs) & test_pairs["drug_2"].isin(train_drugs)
    )
    one_in_train = (
        test_pairs["drug_1"].isin(train_drugs) ^ test_pairs["drug_2"].isin(train_drugs)
    )
    neither_in_train = ~(
        test_pairs["drug_1"].isin(train_drugs) | test_pairs["drug_2"].isin(train_drugs)
    )

    def pct(n: int) -> float:
        return float(round(100 * n / n_test_pairs, 2)) if n_test_pairs else 0.0

    test_in_train_pct = pct(int(test_drugs & train_drugs == test_drugs))

    return {
        "unique_drugs_train": len(train_drugs),
        "unique_drugs_validation": len(val_drugs),
        "unique_drugs_test": len(test_drugs),
        "unique_drugs_total": len(all_drugs),
        "drugs_in_all_three_splits": len(in_all),
        "drugs_unique_to_train": len(only_train),
        "drugs_unique_to_validation": len(only_val),
        "drugs_unique_to_test": len(only_test),
        "test_drugs_also_in_train_count": len(test_drugs & train_drugs),
        "test_drugs_also_in_train_pct": float(round(
            100 * len(test_drugs & train_drugs) / len(test_drugs), 2
        )) if test_drugs else 0.0,
        "test_pairs_both_drugs_in_train": int(both_in_train.sum()),
        "test_pairs_both_drugs_in_train_pct": pct(int(both_in_train.sum())),
        "test_pairs_one_drug_in_train": int(one_in_train.sum()),
        "test_pairs_one_drug_in_train_pct": pct(int(one_in_train.sum())),
        "test_pairs_neither_drug_in_train": int(neither_in_train.sum()),
        "test_pairs_neither_drug_in_train_pct": pct(int(neither_in_train.sum())),
        "interpretation": (
            "The split evaluates unseen drug combinations involving mostly known drugs, "
            "not generalization to completely unseen drugs."
        ),
    }


# ---------------------------------------------------------------------------
# Step 7: Build audit
# ---------------------------------------------------------------------------

def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def build_split_audit(
    n_pairs: int,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    validation_checks: dict[str, bool],
    label_distribution: dict[str, Any],
    drug_overlap: dict[str, Any],
    input_paths: dict[str, str],
) -> dict[str, Any]:
    n_train = len(train_ids)
    n_val   = len(val_ids)
    n_test  = len(test_ids)

    return {
        "random_seed": RANDOM_SEED,
        "split_method": "pair-random",
        "rounding_rule": (
            "n_test = round(0.10 * n_pairs); "
            "n_validation = round(0.10 * n_pairs); "
            "n_train = n_pairs - n_validation - n_test"
        ),
        "total_pairs": n_pairs,
        "train_count": n_train,
        "validation_count": n_val,
        "test_count": n_test,
        "split_percentages": {
            "train": float(round(100 * n_train / n_pairs, 4)),
            "validation": float(round(100 * n_val / n_pairs, 4)),
            "test": float(round(100 * n_test / n_pairs, 4)),
        },
        "overlap_validation": validation_checks,
        "coverage_validation": {
            "union_equals_all_pairs": validation_checks.get("complete_coverage", False),
        },
        "label_distribution_summary": label_distribution,
        "drug_overlap_summary": drug_overlap,
        "software_versions": _software_versions(),
        "input_file_paths": input_paths,
    }


# ---------------------------------------------------------------------------
# Step 8: Disk-level write helpers
# ---------------------------------------------------------------------------

def _write_bytes_atomic(data: bytes, path: Path) -> None:
    """Atomic file write via temp-then-replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(str(tmp), str(path))
    except OSError:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def write_split_csv(ids: np.ndarray, path: Path) -> None:
    """Write a split CSV with a single column `pair_id`."""
    lines = ["pair_id"] + [str(x) for x in ids]
    _write_bytes_atomic(("\n".join(lines) + "\n").encode("utf-8"), path)


# ---------------------------------------------------------------------------
# Step 9: Disk-level validation (reload and re-check)
# ---------------------------------------------------------------------------

def validate_splits_from_disk(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    all_pair_ids: np.ndarray,
) -> None:
    """Reload split CSVs and re-run all validation checks."""
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if list(df.columns) != ["pair_id"]:
            raise ValueError(
                f"{name} CSV has unexpected columns: {list(df.columns)}"
            )

    validate_splits(
        train_df["pair_id"].to_numpy(),
        val_df["pair_id"].to_numpy(),
        test_df["pair_id"].to_numpy(),
        all_pair_ids,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_split_pipeline(
    pairs_input:    Path = DEFAULT_PAIRS_INPUT,
    labels_input:   Path = DEFAULT_LABELS_INPUT,
    label_map_input:Path = DEFAULT_LABEL_MAP_INPUT,
    train_output:   Path = DEFAULT_TRAIN_OUTPUT,
    val_output:     Path = DEFAULT_VAL_OUTPUT,
    test_output:    Path = DEFAULT_TEST_OUTPUT,
    audit_output:   Path = DEFAULT_AUDIT_OUTPUT,
    seed:           int  = RANDOM_SEED,
) -> dict[str, Any]:
    logger.info("=== Milestone 3 split pipeline started ===")

    # 1. Load inputs
    pairs_df = pd.read_parquet(pairs_input)
    labels   = np.load(labels_input)
    logger.info("Loaded %d pairs, label matrix %s.", len(pairs_df), labels.shape)

    # 2. Validate inputs
    validate_pair_table(pairs_df)
    validate_label_matrix(pairs_df, labels)

    all_pair_ids = pairs_df["pair_id"].to_numpy()
    n_pairs = len(all_pair_ids)

    # 3. Compute sizes
    n_train, n_val, n_test = compute_split_sizes(n_pairs)
    logger.info("Split sizes: train=%d  validation=%d  test=%d", n_train, n_val, n_test)

    # 4. Create splits
    train_ids, val_ids, test_ids = create_splits(all_pair_ids, seed=seed)

    # 5. Validate in memory
    checks = validate_splits(train_ids, val_ids, test_ids, all_pair_ids)
    logger.info("In-memory validation: %d checks, all passed.", len(checks))

    # 6. Write CSVs
    write_split_csv(train_ids, train_output)
    write_split_csv(val_ids,   val_output)
    write_split_csv(test_ids,  test_output)
    logger.info("Wrote: %s  %s  %s", train_output, val_output, test_output)

    # 7. Validate from disk
    validate_splits_from_disk(train_output, val_output, test_output, all_pair_ids)
    logger.info("Disk-reload validation passed.")

    # 8. Label-distribution audit
    label_dist = compute_label_distribution_summary(train_ids, val_ids, test_ids, labels)

    zero_val  = label_dist["validation"]["zero_positive_label_indices"]
    zero_test = label_dist["test"]["zero_positive_label_indices"]
    if zero_val:
        logger.warning(
            "%d label(s) have zero positives in the validation split: %s",
            len(zero_val), zero_val[:10],
        )
    if zero_test:
        logger.warning(
            "%d label(s) have zero positives in the test split: %s",
            len(zero_test), zero_test[:10],
        )

    # 9. Drug-overlap audit
    drug_overlap = compute_drug_overlap(train_ids, val_ids, test_ids, pairs_df)
    logger.info(
        "Drug overlap: %d/%d test drugs appear in train (%.1f%%).",
        drug_overlap["test_drugs_also_in_train_count"],
        drug_overlap["unique_drugs_test"],
        drug_overlap["test_drugs_also_in_train_pct"],
    )

    # 10. Build and write audit
    audit = build_split_audit(
        n_pairs, train_ids, val_ids, test_ids,
        checks, label_dist, drug_overlap,
        input_paths={
            "pairs":     str(pairs_input),
            "labels":    str(labels_input),
            "label_map": str(label_map_input),
        },
    )
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomic(
        json.dumps(audit, indent=2, ensure_ascii=False).encode("utf-8"),
        audit_output,
    )
    logger.info("Wrote audit: %s", audit_output)
    logger.info("=== Milestone 3 complete ===")

    return audit


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    run_split_pipeline()


if __name__ == "__main__":
    main()
