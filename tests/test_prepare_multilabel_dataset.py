"""
Tests for polyllm.data.prepare_multilabel_dataset — Milestone 1.

All tests use synthetic in-memory data.
The 34 MB raw dataset is never loaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polyllm.data.prepare_multilabel_dataset import (
    COL_DRUG1_RAW,
    COL_DRUG2_RAW,
    COL_SE_ID,
    COL_SE_NAME,
    REQUIRED_COLUMNS,
    build_label_mapping,
    build_label_matrix,
    build_pair_table,
    canonicalize_pairs,
    check_missing_values,
    count_se_frequency,
    filter_side_effects,
    load_raw,
    normalize_whitespace,
    remove_canonical_duplicates,
    remove_exact_duplicates,
    run_preprocessing,
    split_self_pairs,
    validate_columns,
    validate_side_effect_names,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(rows: list[tuple]) -> pd.DataFrame:
    """Build a DataFrame with the four required columns from a list of tuples."""
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def _write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, compression="gzip")


# ---------------------------------------------------------------------------
# 1. Required-column validation
# ---------------------------------------------------------------------------

class TestValidateColumns:
    def test_passes_with_all_required_columns(self):
        df = _df([("CIDA", "CIDB", "SE1", "headache")])
        validate_columns(df)  # must not raise

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"STITCH 2": ["CIDB"], COL_SE_ID: ["SE1"], COL_SE_NAME: ["h"]})
        with pytest.raises(ValueError, match="# STITCH 1"):
            validate_columns(df)

    # 2. Literal '#' in column 1
    def test_requires_literal_hash_in_column1(self):
        """A DataFrame with 'STITCH 1' (no #) must fail validation."""
        df = pd.DataFrame({
            "STITCH 1": ["CIDA"],       # missing the '#'
            "STITCH 2": ["CIDB"],
            COL_SE_ID: ["SE1"],
            COL_SE_NAME: ["headache"],
        })
        with pytest.raises(ValueError, match="# STITCH 1"):
            validate_columns(df)

    def test_error_message_shows_required_observed_missing(self):
        df = pd.DataFrame({"X": [1], "Y": [2]})
        with pytest.raises(ValueError) as exc:
            validate_columns(df)
        msg = str(exc.value)
        assert "Required" in msg
        assert "Observed" in msg
        assert "Missing" in msg


# ---------------------------------------------------------------------------
# 3. STITCH identifiers remain strings
# ---------------------------------------------------------------------------

class TestLoadRawPreservesStrings:
    def test_stitch_identifiers_are_strings(self, tmp_path):
        """pandas must not silently convert CID identifiers to integers."""
        df = _df([("CID000003121", "CID000003640", "C0013604", "edema")])
        csv_path = tmp_path / "test.csv.gz"
        _write_csv_gz(df, csv_path)

        loaded = load_raw(csv_path)
        # pandas 3+ uses StringDtype; older pandas uses object — both are string dtypes
        assert pd.api.types.is_string_dtype(loaded[COL_DRUG1_RAW])
        assert pd.api.types.is_string_dtype(loaded[COL_DRUG2_RAW])
        # Leading zeros and CID prefix must be preserved (not converted to int)
        assert loaded[COL_DRUG1_RAW].iloc[0] == "CID000003121"
        assert loaded[COL_DRUG2_RAW].iloc[0] == "CID000003640"
        # Values must not be integer
        assert not isinstance(loaded[COL_DRUG1_RAW].iloc[0], int)

    def test_se_identifier_is_string(self, tmp_path):
        df = _df([("CIDA", "CIDB", "C0013604", "edema")])
        csv_path = tmp_path / "test.csv.gz"
        _write_csv_gz(df, csv_path)
        loaded = load_raw(csv_path)
        assert pd.api.types.is_string_dtype(loaded[COL_SE_ID])
        assert loaded[COL_SE_ID].iloc[0] == "C0013604"
        assert not isinstance(loaded[COL_SE_ID].iloc[0], int)


# ---------------------------------------------------------------------------
# 4 & 5. Pair canonicalisation
# ---------------------------------------------------------------------------

class TestCanonicalizePairs:
    def test_ab_and_ba_canonicalize_to_same_pair(self):
        """Reversed rows (A,B) and (B,A) must produce identical canonical pairs."""
        df = _df([
            ("CIDB", "CIDA", "SE1", "headache"),
            ("CIDA", "CIDB", "SE1", "headache"),
        ])
        result = canonicalize_pairs(df)
        assert result["drug_1"].iloc[0] == result["drug_1"].iloc[1]
        assert result["drug_2"].iloc[0] == result["drug_2"].iloc[1]

    def test_lexicographic_ordering_drug1_le_drug2(self):
        """drug_1 must always be lexicographically <= drug_2."""
        df = _df([
            ("CIDC", "CIDA", "SE1", "h"),
            ("CIDA", "CIDC", "SE2", "h"),
            ("CIDB", "CIDB", "SE3", "h"),  # self-pair: drug_1 == drug_2
        ])
        result = canonicalize_pairs(df)
        assert (result["drug_1"] <= result["drug_2"]).all()

    def test_canonical_pair_uses_string_not_integer_comparison(self):
        """Identifiers must be compared as strings (lexicographic), not as numbers."""
        # Lexicographically "CID000000009" > "CID000000010" is False
        # because "9" < "1" is False... wait, "9" > "1" so "CID000000009" > "CID000000010"
        # Numerically 9 < 10.  If using string comparison: '9' > '1' so CID9 > CID10.
        a = "CID000000002"
        b = "CID000000010"
        # Lexicographic: "CID000000002" < "CID000000010" (because '2' < '1'... no)
        # '0' < '1'... let's think: compare char by char:
        # "CID000000002" vs "CID000000010"
        # All equal until position 9: '0' vs '1' → '0' < '1' → a < b
        assert a < b  # confirm expected lexicographic ordering
        df = _df([(b, a, "SE1", "h")])
        result = canonicalize_pairs(df)
        assert result["drug_1"].iloc[0] == a
        assert result["drug_2"].iloc[0] == b

    def test_does_not_alter_identifier_strings(self):
        """Canonicalisation must not modify the identifier strings themselves."""
        df = _df([("CID000003640", "CID000003121", "SE1", "h")])
        result = canonicalize_pairs(df)
        assert result["drug_1"].iloc[0] == "CID000003121"
        assert result["drug_2"].iloc[0] == "CID000003640"


# ---------------------------------------------------------------------------
# 6. Exact duplicate row removal
# ---------------------------------------------------------------------------

class TestRemoveExactDuplicates:
    def test_removes_identical_rows(self):
        df = _df([
            ("CIDA", "CIDB", "SE1", "headache"),
            ("CIDA", "CIDB", "SE1", "headache"),  # exact dup
            ("CIDA", "CIDB", "SE2", "nausea"),
        ])
        result, count = remove_exact_duplicates(df)
        assert count == 1
        assert len(result) == 2

    def test_rows_differing_in_any_column_are_not_duplicates(self):
        df = _df([
            ("CIDA", "CIDB", "SE1", "headache"),
            ("CIDA", "CIDB", "SE1", "Headache"),  # different name capitalisation
        ])
        result, count = remove_exact_duplicates(df)
        assert count == 0
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 7. Canonical duplicate triple removal after pair reversal
# ---------------------------------------------------------------------------

class TestRemoveCanonicalDuplicates:
    def test_reversed_pair_collapses_after_canonicalization(self):
        """(A,B,SE) and (B,A,SE) must produce exactly one canonical triple."""
        df = _df([
            ("CIDB", "CIDA", "SE1", "headache"),
            ("CIDA", "CIDB", "SE1", "headache"),
        ])
        canon = canonicalize_pairs(df)
        dedup, count = remove_canonical_duplicates(canon)
        assert count == 1
        assert len(dedup) == 1

    def test_different_side_effects_are_not_duplicates(self):
        df = _df([
            ("CIDA", "CIDB", "SE1", "headache"),
            ("CIDA", "CIDB", "SE2", "nausea"),
        ])
        canon = canonicalize_pairs(df)
        dedup, count = remove_canonical_duplicates(canon)
        assert count == 0
        assert len(dedup) == 2


# ---------------------------------------------------------------------------
# 8. Self-pair detection and exclusion
# ---------------------------------------------------------------------------

class TestSelfPairs:
    def test_detects_self_pairs(self):
        df = _df([
            ("CIDA", "CIDA", "SE1", "headache"),  # self-pair
            ("CIDA", "CIDB", "SE1", "headache"),
        ])
        df = canonicalize_pairs(df)
        non_self, self_df, stats = split_self_pairs(df)
        assert stats["self_pair_row_count"] == 1
        assert stats["unique_self_pair_count"] == 1
        assert len(non_self) == 1

    def test_self_pairs_are_excluded_from_result(self):
        df = _df([
            ("CIDA", "CIDA", "SE1", "headache"),
        ])
        df = canonicalize_pairs(df)
        non_self, _, stats = split_self_pairs(df)
        assert len(non_self) == 0
        assert stats["self_pair_row_count"] == 1

    def test_no_self_pairs_returns_unchanged(self):
        df = _df([
            ("CIDA", "CIDB", "SE1", "h"),
            ("CIDB", "CIDC", "SE1", "h"),
        ])
        df = canonicalize_pairs(df)
        non_self, _, stats = split_self_pairs(df)
        assert stats["self_pair_row_count"] == 0
        assert len(non_self) == 2


# ---------------------------------------------------------------------------
# 9. SE frequency counts unique pairs, not raw rows
# ---------------------------------------------------------------------------

class TestSEFrequency:
    def test_counts_unique_canonical_pairs_not_rows(self):
        """
        Three rows with the same (A,B,SE1) must contribute 1 to the SE1 count,
        not 3, after canonical dedup.
        """
        df = _df([
            ("CIDA", "CIDB", "SE1", "headache"),
            ("CIDA", "CIDB", "SE1", "headache"),  # exact dup
            ("CIDB", "CIDA", "SE1", "headache"),  # reversed dup
        ])
        _, _ = remove_exact_duplicates(df)
        df_clean, _ = remove_exact_duplicates(df)
        df_canon = canonicalize_pairs(df_clean)
        df_no_self, _, _ = split_self_pairs(df_canon)
        df_unique, _ = remove_canonical_duplicates(df_no_self)

        freq = count_se_frequency(df_unique)
        se1_count = int(freq.loc[freq["side_effect_id"] == "SE1", "unique_pair_count"].iloc[0])
        assert se1_count == 1, f"Expected 1 unique pair for SE1, got {se1_count}"

    def test_two_different_pairs_same_se(self):
        df = _df([
            ("CIDA", "CIDB", "SE1", "h"),
            ("CIDC", "CIDD", "SE1", "h"),
        ])
        df_canon = canonicalize_pairs(df)
        df_no_self, _, _ = split_self_pairs(df_canon)
        df_unique, _ = remove_canonical_duplicates(df_no_self)
        freq = count_se_frequency(df_unique)
        assert int(freq.loc[freq["side_effect_id"] == "SE1", "unique_pair_count"].iloc[0]) == 2


# ---------------------------------------------------------------------------
# 10 & 11. Frequency threshold is inclusive; below-threshold labels excluded
# ---------------------------------------------------------------------------

class TestFilterSideEffects:
    def _make_freq_df(self, counts: dict[str, int]) -> pd.DataFrame:
        rows = [(se, f"name_{se}", n) for se, n in counts.items()]
        return pd.DataFrame(rows, columns=["side_effect_id", "side_effect_name", "unique_pair_count"])

    def test_threshold_is_inclusive(self):
        """A SE with exactly min_pairs pairs must be retained."""
        freq = self._make_freq_df({"SE1": 5, "SE2": 4, "SE3": 6})
        retained, excluded = filter_side_effects(freq, min_pairs=5)
        assert set(retained["side_effect_id"]) == {"SE1", "SE3"}
        assert set(excluded["side_effect_id"]) == {"SE2"}

    def test_excludes_below_threshold(self):
        freq = self._make_freq_df({"SE1": 10, "SE2": 3})
        retained, excluded = filter_side_effects(freq, min_pairs=5)
        assert "SE2" not in set(retained["side_effect_id"])
        assert "SE2" in set(excluded["side_effect_id"])

    def test_threshold_zero_retains_all(self):
        freq = self._make_freq_df({"SE1": 0, "SE2": 1})
        retained, _ = filter_side_effects(freq, min_pairs=0)
        assert len(retained) == 2


# ---------------------------------------------------------------------------
# 12. Pairs with no retained labels are excluded
# ---------------------------------------------------------------------------

class TestPairsWithNoRetainedLabels:
    def test_pairs_dropped_when_all_labels_filtered(self, tmp_path):
        """
        Pair (C,D) only has SE2 which falls below the threshold.
        It must not appear in the final pair table.
        """
        # Pair (A,B): SE1=3 pairs (just itself 3 times — but we need 3 unique pairs)
        # Use threshold=2 so SE1 is retained (it appears in 2 unique pairs)
        # and SE2 is excluded (only in 1 pair)
        rows = [
            ("CIDA", "CIDB", "SE1", "h"),
            ("CIDC", "CIDD", "SE1", "h"),  # SE1 in 2 unique pairs → retained at threshold=2
            ("CIDE", "CIDF", "SE2", "n"),  # SE2 in 1 unique pair → excluded at threshold=2
        ]
        df = _df(rows)
        df_canon = canonicalize_pairs(df)
        df_no_self, _, _ = split_self_pairs(df_canon)
        df_unique, _ = remove_canonical_duplicates(df_no_self)
        freq = count_se_frequency(df_unique)
        retained, _ = filter_side_effects(freq, min_pairs=2)
        retained_ids = set(retained["side_effect_id"])
        df_filtered = df_unique[df_unique[COL_SE_ID].isin(retained_ids)]
        pairs = build_pair_table(df_filtered)

        pair_set = set(zip(pairs["drug_1"], pairs["drug_2"]))
        assert ("CIDA", "CIDB") in pair_set
        assert ("CIDC", "CIDD") in pair_set
        assert ("CIDE", "CIDF") not in pair_set


# ---------------------------------------------------------------------------
# 13 & 14. Deterministic ordering and sequential index assignment
# ---------------------------------------------------------------------------

class TestDeterministicOrdering:
    def test_pair_table_sorted_by_drug1_then_drug2(self):
        rows = [
            ("CIDB", "CIDC", "SE1", "h"),
            ("CIDA", "CIDC", "SE1", "h"),
            ("CIDA", "CIDB", "SE1", "h"),
        ]
        df = _df(rows)
        df_canon = canonicalize_pairs(df)
        df_no_self, _, _ = split_self_pairs(df_canon)
        df_unique, _ = remove_canonical_duplicates(df_no_self)
        freq = count_se_frequency(df_unique)
        retained, _ = filter_side_effects(freq, min_pairs=1)
        df_filtered = df_unique[df_unique[COL_SE_ID].isin(set(retained["side_effect_id"]))]
        pairs = build_pair_table(df_filtered)

        assert list(pairs["drug_1"]) == sorted(pairs["drug_1"].tolist())
        # Within same drug_1, drug_2 must be sorted
        for d1 in pairs["drug_1"].unique():
            sub = pairs[pairs["drug_1"] == d1]["drug_2"].tolist()
            assert sub == sorted(sub)

    def test_pair_id_is_sequential_zero_based(self):
        rows = [("CIDB", "CIDC", "SE1", "h"), ("CIDA", "CIDB", "SE1", "h")]
        df = _df(rows)
        df_canon = canonicalize_pairs(df)
        df_no_self, _, _ = split_self_pairs(df_canon)
        df_unique, _ = remove_canonical_duplicates(df_no_self)
        freq = count_se_frequency(df_unique)
        retained, _ = filter_side_effects(freq, min_pairs=1)
        df_filtered = df_unique[df_unique[COL_SE_ID].isin(set(retained["side_effect_id"]))]
        pairs = build_pair_table(df_filtered)
        assert list(pairs["pair_id"]) == list(range(len(pairs)))

    def test_label_mapping_sorted_by_side_effect_id(self):
        rows = [
            ("CIDA", "CIDB", "SE003", "gamma"),
            ("CIDC", "CIDD", "SE001", "alpha"),
            ("CIDE", "CIDF", "SE002", "beta"),
        ]
        df = _df(rows)
        df_canon = canonicalize_pairs(df)
        df_no_self, _, _ = split_self_pairs(df_canon)
        df_unique, _ = remove_canonical_duplicates(df_no_self)
        freq = count_se_frequency(df_unique)
        retained, _ = filter_side_effects(freq, min_pairs=1)
        mapping = build_label_mapping(retained)
        assert list(mapping["side_effect_id"]) == sorted(mapping["side_effect_id"].tolist())

    def test_label_index_is_sequential_zero_based(self):
        rows = [("CIDA", "CIDB", "SE2", "b"), ("CIDC", "CIDD", "SE1", "a")]
        df = _df(rows)
        df_canon = canonicalize_pairs(df)
        df_no_self, _, _ = split_self_pairs(df_canon)
        df_unique, _ = remove_canonical_duplicates(df_no_self)
        freq = count_se_frequency(df_unique)
        retained, _ = filter_side_effects(freq, min_pairs=1)
        mapping = build_label_mapping(retained)
        assert list(mapping["label_index"]) == list(range(len(mapping)))


# ---------------------------------------------------------------------------
# 15–19. Label matrix correctness, dtype, shape consistency
# ---------------------------------------------------------------------------

class TestLabelMatrix:
    def _run_pipeline(self, rows, min_pairs=1):
        df = _df(rows)
        df_canon = canonicalize_pairs(df)
        df_no_self, _, _ = split_self_pairs(df_canon)
        df_unique, _ = remove_canonical_duplicates(df_no_self)
        freq = count_se_frequency(df_unique)
        retained, _ = filter_side_effects(freq, min_pairs=min_pairs)
        df_filtered = df_unique[df_unique[COL_SE_ID].isin(set(retained["side_effect_id"]))]
        pairs = build_pair_table(df_filtered)
        mapping = build_label_mapping(retained)
        matrix = build_label_matrix(df_filtered, pairs, mapping)
        return pairs, mapping, matrix

    def test_correct_matrix_contents(self):
        """
        Pairs (A,B) and (A,C) have SE1.  Pairs (A,B) and (B,C) have SE2.
        Expected matrix (sorted pairs: AB, AC, BC):
            SE1  SE2
        AB:  1    1
        AC:  1    0
        BC:  0    1
        """
        rows = [
            ("CIDA", "CIDB", "SE1", "h"),
            ("CIDA", "CIDB", "SE2", "n"),
            ("CIDA", "CIDC", "SE1", "h"),
            ("CIDB", "CIDC", "SE2", "n"),
        ]
        pairs, mapping, matrix = self._run_pipeline(rows)

        pair_ids = dict(zip(zip(pairs["drug_1"], pairs["drug_2"]), pairs["pair_id"]))
        label_ids = dict(zip(mapping["side_effect_id"], mapping["label_index"]))

        assert matrix[pair_ids[("CIDA", "CIDB")], label_ids["SE1"]] == 1
        assert matrix[pair_ids[("CIDA", "CIDB")], label_ids["SE2"]] == 1
        assert matrix[pair_ids[("CIDA", "CIDC")], label_ids["SE1"]] == 1
        assert matrix[pair_ids[("CIDA", "CIDC")], label_ids["SE2"]] == 0
        assert matrix[pair_ids[("CIDB", "CIDC")], label_ids["SE1"]] == 0
        assert matrix[pair_ids[("CIDB", "CIDC")], label_ids["SE2"]] == 1

    # 16. Values are only 0 and 1
    def test_matrix_values_are_zero_or_one(self):
        rows = [("CIDA", "CIDB", "SE1", "h"), ("CIDC", "CIDD", "SE1", "h")]
        _, _, matrix = self._run_pipeline(rows)
        assert set(np.unique(matrix)).issubset({0, 1})

    # 17. Dtype is uint8
    def test_matrix_dtype_is_uint8(self):
        rows = [("CIDA", "CIDB", "SE1", "h")]
        _, _, matrix = self._run_pipeline(rows)
        assert matrix.dtype == np.uint8

    # 18. Pair-table rows == matrix rows
    def test_pair_table_rows_equal_matrix_rows(self):
        rows = [
            ("CIDA", "CIDB", "SE1", "h"),
            ("CIDC", "CIDD", "SE1", "h"),
            ("CIDA", "CIDB", "SE2", "n"),
        ]
        pairs, _, matrix = self._run_pipeline(rows)
        assert len(pairs) == matrix.shape[0]

    # 19. Label-mapping rows == matrix columns
    def test_label_mapping_rows_equal_matrix_columns(self):
        rows = [
            ("CIDA", "CIDB", "SE1", "h"),
            ("CIDA", "CIDB", "SE2", "n"),
            ("CIDA", "CIDB", "SE3", "d"),
        ]
        _, mapping, matrix = self._run_pipeline(rows)
        assert len(mapping) == matrix.shape[1]


# ---------------------------------------------------------------------------
# 20. Missing required values fail clearly
# ---------------------------------------------------------------------------

class TestMissingValues:
    def test_nan_in_drug1_column_raises(self):
        df = _df([("CIDA", "CIDB", "SE1", "h")])
        df.loc[0, COL_DRUG1_RAW] = None
        df, _ = normalize_whitespace(df, REQUIRED_COLUMNS)
        with pytest.raises(ValueError, match="# STITCH 1"):
            check_missing_values(df)

    def test_blank_string_in_drug2_column_raises(self):
        df = _df([("CIDA", "   ", "SE1", "h")])
        df, _ = normalize_whitespace(df, REQUIRED_COLUMNS)
        with pytest.raises(ValueError, match="STITCH 2"):
            check_missing_values(df)

    def test_nan_in_se_id_raises(self):
        df = _df([("CIDA", "CIDB", None, "h")])
        df, _ = normalize_whitespace(df, REQUIRED_COLUMNS)
        with pytest.raises(ValueError, match="Polypharmacy Side Effect"):
            check_missing_values(df)

    def test_missing_se_name_does_not_raise(self):
        """Missing side-effect name is a warning, not a hard error."""
        df = _df([("CIDA", "CIDB", "SE1", None)])
        df, _ = normalize_whitespace(df, REQUIRED_COLUMNS)
        counts = check_missing_values(df)  # must not raise
        assert counts[COL_SE_NAME] == 1


# ---------------------------------------------------------------------------
# 21. Conflicting names for one SE identifier fail clearly
# ---------------------------------------------------------------------------

class TestSideEffectNameConflicts:
    def test_conflicting_names_for_same_id_raises(self):
        """Two non-empty names for the same SE identifier must raise ValueError."""
        df = _df([
            ("CIDA", "CIDB", "SE1", "headache"),
            ("CIDC", "CIDD", "SE1", "migraine"),  # different name, same ID
        ])
        with pytest.raises(ValueError, match="conflicts"):
            validate_side_effect_names(df)

    def test_single_name_per_id_passes(self):
        df = _df([
            ("CIDA", "CIDB", "SE1", "headache"),
            ("CIDC", "CIDD", "SE1", "headache"),  # same name, fine
        ])
        conflicts, _, _ = validate_side_effect_names(df)
        assert conflicts == 0


# ---------------------------------------------------------------------------
# 22. Re-running on identical data produces identical outputs
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_two_runs_produce_identical_outputs(self, tmp_path):
        rows = [
            ("CIDA", "CIDB", "SE1", "headache"),
            ("CIDA", "CIDB", "SE2", "nausea"),
            ("CIDC", "CIDD", "SE1", "headache"),
            ("CIDB", "CIDA", "SE1", "headache"),  # reversed dup
        ]
        df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
        csv_path = tmp_path / "input.csv.gz"
        df.to_csv(csv_path, index=False, compression="gzip")

        def _run(run_id: str):
            out = tmp_path / run_id
            run_preprocessing(
                input_path=csv_path,
                pairs_output=out / "pairs.parquet",
                label_mapping_output=out / "mapping.csv",
                labels_output=out / "labels.npy",
                audit_output=out / "audit.json",
                min_pair_frequency=1,
            )
            return out

        out1 = _run("run1")
        out2 = _run("run2")

        pairs1 = pd.read_parquet(out1 / "pairs.parquet")
        pairs2 = pd.read_parquet(out2 / "pairs.parquet")
        pd.testing.assert_frame_equal(pairs1, pairs2)

        m1 = pd.read_csv(out1 / "mapping.csv")
        m2 = pd.read_csv(out2 / "mapping.csv")
        pd.testing.assert_frame_equal(m1, m2)

        mat1 = np.load(out1 / "labels.npy")
        mat2 = np.load(out2 / "labels.npy")
        np.testing.assert_array_equal(mat1, mat2)


# ---------------------------------------------------------------------------
# 23. Output directories are created when absent
# ---------------------------------------------------------------------------

class TestOutputDirectoryCreation:
    def test_missing_output_dirs_are_created(self, tmp_path):
        rows = [
            ("CIDA", "CIDB", "SE1", "headache"),
            ("CIDC", "CIDD", "SE1", "headache"),
        ]
        df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
        csv_path = tmp_path / "input.csv.gz"
        df.to_csv(csv_path, index=False, compression="gzip")

        pairs_out = tmp_path / "deep" / "nested" / "pairs.parquet"
        label_out = tmp_path / "other" / "mapping.csv"
        labels_out = tmp_path / "yet" / "another" / "labels.npy"
        audit_out = tmp_path / "audit_dir" / "data_audit.json"

        run_preprocessing(
            input_path=csv_path,
            pairs_output=pairs_out,
            label_mapping_output=label_out,
            labels_output=labels_out,
            audit_output=audit_out,
            min_pair_frequency=1,
        )

        assert pairs_out.exists()
        assert label_out.exists()
        assert labels_out.exists()
        assert audit_out.exists()


# ---------------------------------------------------------------------------
# 24. Original input file is not modified
# ---------------------------------------------------------------------------

class TestInputFileUnchanged:
    def test_input_file_bytes_unchanged_after_run(self, tmp_path):
        rows = [
            ("CIDA", "CIDB", "SE1", "headache"),
            ("CIDC", "CIDD", "SE1", "headache"),
        ]
        df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
        csv_path = tmp_path / "input.csv.gz"
        df.to_csv(csv_path, index=False, compression="gzip")

        before = hashlib.md5(csv_path.read_bytes()).hexdigest()

        run_preprocessing(
            input_path=csv_path,
            pairs_output=tmp_path / "pairs.parquet",
            label_mapping_output=tmp_path / "mapping.csv",
            labels_output=tmp_path / "labels.npy",
            audit_output=tmp_path / "audit.json",
            min_pair_frequency=1,
        )

        after = hashlib.md5(csv_path.read_bytes()).hexdigest()
        assert before == after, "Input file was modified during preprocessing."


# ---------------------------------------------------------------------------
# Full pipeline integration smoke test
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    """End-to-end test with a small synthetic dataset."""

    def _make_dataset(self, tmp_path: Path, min_pairs: int = 2) -> dict:
        """
        Build synthetic data with:
          - SE1: in 3 unique canonical pairs → retained at threshold=2
          - SE2: in 1 unique pair → excluded at threshold=2
          - 1 reversed pair that collapses
          - 1 self-pair that is excluded
        """
        rows = [
            ("CIDA", "CIDB", "SE1", "headache"),
            ("CIDA", "CIDC", "SE1", "headache"),
            ("CIDB", "CIDC", "SE1", "headache"),
            ("CIDB", "CIDA", "SE1", "headache"),  # reversed dup of row 0
            ("CIDA", "CIDA", "SE1", "headache"),  # self-pair
            ("CIDE", "CIDF", "SE2", "nausea"),    # SE2: only 1 pair
        ]
        df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
        csv_path = tmp_path / "input.csv.gz"
        df.to_csv(csv_path, index=False, compression="gzip")

        audit = run_preprocessing(
            input_path=csv_path,
            pairs_output=tmp_path / "pairs.parquet",
            label_mapping_output=tmp_path / "mapping.csv",
            labels_output=tmp_path / "labels.npy",
            audit_output=tmp_path / "audit.json",
            min_pair_frequency=min_pairs,
        )
        return audit

    def test_audit_counts_are_correct(self, tmp_path):
        audit = self._make_dataset(tmp_path, min_pairs=2)
        assert audit["self_pair_row_count"] == 1
        assert audit["unique_self_pair_count"] == 1
        # Reversed pair is a canonical dup
        assert audit["canonical_duplicate_triple_count"] >= 1
        # SE1 retained (3 pairs), SE2 excluded (1 pair)
        assert audit["retained_side_effect_count"] == 1
        assert audit["excluded_side_effect_count"] == 1

    def test_final_pairs_all_have_drug1_lt_drug2(self, tmp_path):
        self._make_dataset(tmp_path, min_pairs=2)
        pairs = pd.read_parquet(tmp_path / "pairs.parquet")
        assert (pairs["drug_1"] < pairs["drug_2"]).all()

    def test_matrix_row_col_align_with_ids(self, tmp_path):
        self._make_dataset(tmp_path, min_pairs=2)
        pairs = pd.read_parquet(tmp_path / "pairs.parquet")
        mapping = pd.read_csv(tmp_path / "mapping.csv", dtype=str)
        matrix = np.load(tmp_path / "labels.npy")
        # Row 0 of matrix must correspond to pair_id 0
        assert matrix.shape[0] == len(pairs)
        assert matrix.shape[1] == len(mapping)
        # Every pair has at least one label
        assert (matrix.sum(axis=1) >= 1).all()

    def test_audit_json_is_valid_and_has_required_keys(self, tmp_path):
        self._make_dataset(tmp_path, min_pairs=2)
        audit_text = (tmp_path / "audit.json").read_text(encoding="utf-8")
        audit = json.loads(audit_text)
        required_keys = [
            "input_path", "raw_row_count", "exact_duplicate_row_count",
            "self_pair_row_count", "canonical_duplicate_triple_count",
            "retained_side_effect_count", "final_pair_count",
            "label_matrix_shape", "label_matrix_dtype",
            "software_versions", "random_seed",
        ]
        for k in required_keys:
            assert k in audit, f"Missing audit key: {k!r}"
        assert audit["random_seed"] is None

    def test_software_versions_in_audit(self, tmp_path):
        self._make_dataset(tmp_path, min_pairs=2)
        audit = json.loads((tmp_path / "audit.json").read_text())
        sv = audit["software_versions"]
        assert "python" in sv
        assert "pandas" in sv
        assert "numpy" in sv
        assert "pyarrow" in sv
        assert sv["python"].startswith("3.11")
