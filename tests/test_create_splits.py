"""
Tests for polyllm.data.create_splits — Milestone 3.

All tests use synthetic pair tables and label matrices.
No live data files are read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polyllm.data.create_splits import (
    compute_split_sizes,
    compute_drug_overlap,
    compute_label_distribution_summary,
    create_splits,
    run_split_pipeline,
    validate_label_matrix,
    validate_pair_table,
    validate_splits,
    validate_splits_from_disk,
    write_split_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pairs(n: int, drugs_per_pair: int = 10) -> pd.DataFrame:
    """Synthetic pair table with sequential pair_ids and artificial drug IDs."""
    np.random.seed(0)
    n_drugs = max(drugs_per_pair, 2)
    drug_ids = [f"CID{str(i).zfill(9)}" for i in range(n_drugs)]
    rows = []
    for i in range(n):
        d1 = drug_ids[i % n_drugs]
        d2 = drug_ids[(i + 1) % n_drugs]
        if d1 > d2:
            d1, d2 = d2, d1
        rows.append({"pair_id": i, "drug_1": d1, "drug_2": d2})
    return pd.DataFrame(rows)


def _make_labels(n_pairs: int, n_labels: int, seed: int = 7) -> np.ndarray:
    """Synthetic uint8 label matrix with non-trivial sparsity."""
    rng = np.random.default_rng(seed)
    # Each pair gets ~5 labels on average
    mat = (rng.random((n_pairs, n_labels)) < (5 / n_labels)).astype(np.uint8)
    return mat


# ---------------------------------------------------------------------------
# 1 & 2. Deterministic split + correct 80/10/10 size calculation
# ---------------------------------------------------------------------------

class TestSplitSizes:
    def test_exact_sizes_for_100_pairs(self):
        n_train, n_val, n_test = compute_split_sizes(100)
        assert n_test == 10
        assert n_val == 10
        assert n_train == 80
        assert n_train + n_val + n_test == 100

    def test_exact_sizes_for_63472_pairs(self):
        n_train, n_val, n_test = compute_split_sizes(63_472)
        assert n_test == 6_347        # round(0.10 * 63472) = round(6347.2) = 6347
        assert n_val  == 6_347
        assert n_train == 63_472 - 6_347 - 6_347
        assert n_train + n_val + n_test == 63_472

    def test_total_always_equals_n(self):
        for n in [10, 99, 100, 101, 500, 1000, 63_472]:
            n_tr, n_v, n_te = compute_split_sizes(n)
            assert n_tr + n_v + n_te == n, f"Failed for n={n}"

    def test_sizes_are_non_negative(self):
        for n in [3, 10, 100, 1000]:
            n_tr, n_v, n_te = compute_split_sizes(n)
            assert n_tr >= 0 and n_v >= 0 and n_te >= 0


class TestDeterministicSplit:
    def test_seed42_produces_fixed_output(self):
        ids = np.arange(100)
        a_tr, a_v, a_te = create_splits(ids, seed=42)
        b_tr, b_v, b_te = create_splits(ids, seed=42)
        np.testing.assert_array_equal(a_tr, b_tr)
        np.testing.assert_array_equal(a_v, b_v)
        np.testing.assert_array_equal(a_te, b_te)

    def test_output_ids_are_sorted(self):
        ids = np.arange(100)
        tr, v, te = create_splits(ids, seed=42)
        assert np.all(tr[:-1] < tr[1:]), "train not sorted"
        assert np.all(v[:-1]  < v[1:]),  "validation not sorted"
        assert np.all(te[:-1] < te[1:]), "test not sorted"

    def test_membership_differs_from_sequential_assignment(self):
        """After shuffle, first n_train IDs are not simply 0..n_train-1."""
        ids = np.arange(1000)
        tr, _, _ = create_splits(ids, seed=42)
        sequential_train = np.arange(800)
        assert not np.array_equal(np.sort(tr), sequential_train)


# ---------------------------------------------------------------------------
# 3 & 4. Complete coverage and no overlap
# ---------------------------------------------------------------------------

class TestCoverageAndOverlap:
    def test_union_equals_all_ids(self):
        ids = np.arange(100)
        tr, v, te = create_splits(ids, seed=42)
        union = np.sort(np.concatenate([tr, v, te]))
        np.testing.assert_array_equal(union, ids)

    def test_train_validation_disjoint(self):
        ids = np.arange(100)
        tr, v, _ = create_splits(ids, seed=42)
        assert len(np.intersect1d(tr, v)) == 0

    def test_train_test_disjoint(self):
        ids = np.arange(100)
        tr, _, te = create_splits(ids, seed=42)
        assert len(np.intersect1d(tr, te)) == 0

    def test_validation_test_disjoint(self):
        ids = np.arange(100)
        _, v, te = create_splits(ids, seed=42)
        assert len(np.intersect1d(v, te)) == 0

    def test_validate_splits_passes_on_valid_input(self):
        ids = np.arange(200)
        tr, v, te = create_splits(ids, seed=42)
        results = validate_splits(tr, v, te, ids)
        assert all(results.values())


# ---------------------------------------------------------------------------
# 5. No duplicate IDs within a split
# ---------------------------------------------------------------------------

class TestNoDuplicates:
    def test_no_duplicates_in_any_split(self):
        ids = np.arange(300)
        tr, v, te = create_splits(ids, seed=42)
        assert len(tr) == len(np.unique(tr))
        assert len(v)  == len(np.unique(v))
        assert len(te) == len(np.unique(te))


# ---------------------------------------------------------------------------
# 6. Invalid/non-sequential pair IDs fail clearly
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_non_sequential_pair_ids_raise(self):
        pairs = pd.DataFrame({"pair_id": [0, 1, 3, 4], "drug_1": ["A"] * 4, "drug_2": ["B"] * 4})
        with pytest.raises(ValueError, match="pair_id"):
            validate_pair_table(pairs)

    def test_duplicate_pair_ids_raise(self):
        pairs = pd.DataFrame({"pair_id": [0, 1, 1, 2], "drug_1": ["A"] * 4, "drug_2": ["B"] * 4})
        with pytest.raises(ValueError, match="pair_id"):
            validate_pair_table(pairs)

    def test_missing_required_column_raises(self):
        pairs = pd.DataFrame({"pair_id": [0, 1], "drug_1": ["A", "B"]})
        with pytest.raises(ValueError, match="missing columns"):
            validate_pair_table(pairs)

    def test_wrong_start_index_raises(self):
        pairs = pd.DataFrame({"pair_id": [1, 2, 3], "drug_1": ["A"] * 3, "drug_2": ["B"] * 3})
        with pytest.raises(ValueError, match="pair_id"):
            validate_pair_table(pairs)


# ---------------------------------------------------------------------------
# 7. Label row-count mismatch fails clearly
# ---------------------------------------------------------------------------

class TestLabelMatrixValidation:
    def test_row_count_mismatch_raises(self):
        pairs = pd.DataFrame({"pair_id": [0, 1, 2], "drug_1": ["A"] * 3, "drug_2": ["B"] * 3})
        labels = np.zeros((5, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="row"):
            validate_label_matrix(pairs, labels)

    def test_1d_array_raises(self):
        pairs = pd.DataFrame({"pair_id": [0, 1], "drug_1": ["A"] * 2, "drug_2": ["B"] * 2})
        labels = np.zeros(2, dtype=np.uint8)
        with pytest.raises(ValueError, match="2-D"):
            validate_label_matrix(pairs, labels)

    def test_valid_matrix_does_not_raise(self):
        pairs = _make_pairs(50)
        labels = _make_labels(50, 20)
        validate_label_matrix(pairs, labels)


# ---------------------------------------------------------------------------
# 8. Disk-loaded validation
# ---------------------------------------------------------------------------

class TestDiskValidation:
    def test_roundtrip_csv_validates(self, tmp_path):
        n = 100
        pairs = _make_pairs(n)
        ids = pairs["pair_id"].to_numpy()
        tr, v, te = create_splits(ids, seed=42)

        train_p = tmp_path / "train.csv"
        val_p   = tmp_path / "val.csv"
        test_p  = tmp_path / "test.csv"

        write_split_csv(tr, train_p)
        write_split_csv(v,  val_p)
        write_split_csv(te, test_p)

        validate_splits_from_disk(train_p, val_p, test_p, ids)

    def test_csv_has_only_pair_id_column(self, tmp_path):
        n = 50
        pairs = _make_pairs(n)
        ids = pairs["pair_id"].to_numpy()
        tr, _, _ = create_splits(ids, seed=42)

        p = tmp_path / "train.csv"
        write_split_csv(tr, p)
        df = pd.read_csv(p)

        assert list(df.columns) == ["pair_id"]

    def test_csv_values_are_sorted(self, tmp_path):
        ids = np.arange(100)
        tr, _, _ = create_splits(ids, seed=42)
        p = tmp_path / "train.csv"
        write_split_csv(tr, p)
        loaded = pd.read_csv(p)["pair_id"].to_numpy()
        np.testing.assert_array_equal(loaded, np.sort(loaded))

    def test_wrong_columns_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"pair_id": [0, 1], "extra": [1, 2]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="unexpected columns"):
            validate_splits_from_disk(p, p, p, np.arange(2))


# ---------------------------------------------------------------------------
# 9 & 10. Same seed → identical; different seed → different membership
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_identical_membership(self):
        ids = np.arange(500)
        tr1, v1, te1 = create_splits(ids, seed=42)
        tr2, v2, te2 = create_splits(ids, seed=42)
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(te1, te2)

    def test_different_seed_changes_membership(self):
        ids = np.arange(500)
        tr1, _, _ = create_splits(ids, seed=42)
        tr2, _, _ = create_splits(ids, seed=99)
        assert not np.array_equal(tr1, tr2)


# ---------------------------------------------------------------------------
# 11. Split files contain only pair_id
# ---------------------------------------------------------------------------

class TestSplitFileSchema:
    def test_all_csv_files_have_only_pair_id(self, tmp_path):
        pairs = _make_pairs(100)
        labels = _make_labels(100, 10)
        pairs_path = tmp_path / "pairs.parquet"
        labels_path = tmp_path / "labels.npy"
        label_map_path = tmp_path / "label_map.csv"
        pairs.to_parquet(pairs_path, index=False)
        np.save(labels_path, labels)
        pd.DataFrame({
            "label_index": range(10),
            "side_effect_id": [f"C{i:07d}" for i in range(10)],
            "side_effect_name": [f"se{i}" for i in range(10)],
            "unique_pair_count": [50] * 10,
        }).to_csv(label_map_path, index=False)

        train_p = tmp_path / "train.csv"
        val_p   = tmp_path / "val.csv"
        test_p  = tmp_path / "test.csv"
        audit_p = tmp_path / "audit.json"

        run_split_pipeline(
            pairs_input=pairs_path,
            labels_input=labels_path,
            label_map_input=label_map_path,
            train_output=train_p,
            val_output=val_p,
            test_output=test_p,
            audit_output=audit_p,
        )

        for path in [train_p, val_p, test_p]:
            df = pd.read_csv(path)
            assert list(df.columns) == ["pair_id"], f"{path.name} has extra columns"


# ---------------------------------------------------------------------------
# 12. Label and drug overlap statistics on a small fixture
# ---------------------------------------------------------------------------

class TestLabelAndDrugStats:
    """Fixed small fixture for deterministic stat verification."""

    N_PAIRS  = 100
    N_LABELS = 20
    N_DRUGS  = 10
    SEED     = 42

    def _run(self, tmp_path):
        pairs  = _make_pairs(self.N_PAIRS, drugs_per_pair=self.N_DRUGS)
        labels = _make_labels(self.N_PAIRS, self.N_LABELS)
        ids    = pairs["pair_id"].to_numpy()
        tr, v, te = create_splits(ids, seed=self.SEED)
        return pairs, labels, tr, v, te

    def test_label_stats_pair_counts_sum_to_total(self, tmp_path):
        pairs, labels, tr, v, te = self._run(tmp_path)
        summary = compute_label_distribution_summary(tr, v, te, labels)
        assert summary["train"]["pair_count"] + summary["validation"]["pair_count"] + \
               summary["test"]["pair_count"] == self.N_PAIRS

    def test_label_stats_labels_with_positives_le_n_labels(self, tmp_path):
        pairs, labels, tr, v, te = self._run(tmp_path)
        summary = compute_label_distribution_summary(tr, v, te, labels)
        for split in ["train", "validation", "test"]:
            assert summary[split]["labels_with_at_least_one_positive"] <= self.N_LABELS

    def test_prevalence_per_label_has_correct_length(self, tmp_path):
        pairs, labels, tr, v, te = self._run(tmp_path)
        summary = compute_label_distribution_summary(tr, v, te, labels)
        for split in ["full_dataset", "train", "validation", "test"]:
            assert len(summary[split]["prevalence_per_label"]) == self.N_LABELS

    def test_drug_overlap_unique_drugs_within_total(self, tmp_path):
        pairs, labels, tr, v, te = self._run(tmp_path)
        overlap = compute_drug_overlap(tr, v, te, pairs)
        assert overlap["unique_drugs_train"] <= overlap["unique_drugs_total"]
        assert overlap["unique_drugs_validation"] <= overlap["unique_drugs_total"]
        assert overlap["unique_drugs_test"] <= overlap["unique_drugs_total"]

    def test_drug_overlap_test_pair_categories_sum_to_n_test(self, tmp_path):
        pairs, labels, tr, v, te = self._run(tmp_path)
        overlap = compute_drug_overlap(tr, v, te, pairs)
        assert (
            overlap["test_pairs_both_drugs_in_train"] +
            overlap["test_pairs_one_drug_in_train"] +
            overlap["test_pairs_neither_drug_in_train"]
        ) == len(te)

    def test_drug_in_all_three_le_min_split_count(self, tmp_path):
        pairs, labels, tr, v, te = self._run(tmp_path)
        overlap = compute_drug_overlap(tr, v, te, pairs)
        min_split = min(
            overlap["unique_drugs_train"],
            overlap["unique_drugs_validation"],
            overlap["unique_drugs_test"],
        )
        assert overlap["drugs_in_all_three_splits"] <= min_split

    def test_audit_json_has_required_keys(self, tmp_path):
        pairs  = _make_pairs(self.N_PAIRS, drugs_per_pair=self.N_DRUGS)
        labels = _make_labels(self.N_PAIRS, self.N_LABELS)
        pairs_path = tmp_path / "pairs.parquet"
        labels_path = tmp_path / "labels.npy"
        label_map_path = tmp_path / "label_map.csv"
        pairs.to_parquet(pairs_path, index=False)
        np.save(labels_path, labels)
        pd.DataFrame({
            "label_index": range(self.N_LABELS),
            "side_effect_id": [f"C{i:07d}" for i in range(self.N_LABELS)],
            "side_effect_name": [f"se{i}" for i in range(self.N_LABELS)],
            "unique_pair_count": [50] * self.N_LABELS,
        }).to_csv(label_map_path, index=False)

        audit_p = tmp_path / "audit.json"
        run_split_pipeline(
            pairs_input=pairs_path,
            labels_input=labels_path,
            label_map_input=label_map_path,
            train_output=tmp_path / "train.csv",
            val_output=tmp_path / "val.csv",
            test_output=tmp_path / "test.csv",
            audit_output=audit_p,
        )

        audit = json.loads(audit_p.read_text())
        required_keys = [
            "random_seed", "split_method", "rounding_rule",
            "total_pairs", "train_count", "validation_count", "test_count",
            "split_percentages", "overlap_validation", "coverage_validation",
            "label_distribution_summary", "drug_overlap_summary",
            "software_versions", "input_file_paths",
        ]
        for k in required_keys:
            assert k in audit, f"Missing audit key: {k!r}"

    def test_audit_train_plus_val_plus_test_equals_total(self, tmp_path):
        pairs  = _make_pairs(self.N_PAIRS, drugs_per_pair=self.N_DRUGS)
        labels = _make_labels(self.N_PAIRS, self.N_LABELS)
        pairs_path = tmp_path / "pairs.parquet"
        labels_path = tmp_path / "labels.npy"
        label_map_path = tmp_path / "label_map.csv"
        pairs.to_parquet(pairs_path, index=False)
        np.save(labels_path, labels)
        pd.DataFrame({
            "label_index": range(self.N_LABELS),
            "side_effect_id": [f"C{i:07d}" for i in range(self.N_LABELS)],
            "side_effect_name": [f"se{i}" for i in range(self.N_LABELS)],
            "unique_pair_count": [50] * self.N_LABELS,
        }).to_csv(label_map_path, index=False)

        audit_p = tmp_path / "audit.json"
        audit = run_split_pipeline(
            pairs_input=pairs_path,
            labels_input=labels_path,
            label_map_input=label_map_path,
            train_output=tmp_path / "train.csv",
            val_output=tmp_path / "val.csv",
            test_output=tmp_path / "test.csv",
            audit_output=audit_p,
        )

        assert audit["train_count"] + audit["validation_count"] + audit["test_count"] \
               == audit["total_pairs"]

    def test_validate_splits_detects_overlap(self):
        ids = np.arange(10)
        tr  = np.array([0, 1, 2, 3])
        v   = np.array([3, 4])     # 3 overlaps with train
        te  = np.array([5, 6, 7, 8, 9])
        with pytest.raises(ValueError, match="train ∩ validation"):
            validate_splits(tr, v, te, ids)

    def test_validate_splits_detects_missing_ids(self):
        ids = np.arange(10)
        tr  = np.array([0, 1, 2, 3, 4, 5, 6, 7])  # missing 8 and 9 from union
        v   = np.array([4, 5])  # overlap with train too — simplest failure first
        te  = np.array([6, 7])
        with pytest.raises(ValueError):
            validate_splits(tr, v, te, ids)
