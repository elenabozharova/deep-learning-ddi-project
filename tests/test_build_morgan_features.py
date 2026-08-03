"""
Milestone 6A tests — build_morgan_features.py

All tests use small synthetic SMILES and pair tables. No real data files loaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

import polyllm.features.build_morgan_features as bmf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_SMILES = ["C", "CC", "CCC", "c1ccccc1", "CCO", "CC(O)=O"]
N_DRUGS = len(VALID_SMILES)
N_PAIRS = 8
FP_SIZE = bmf.MORGAN_FP_SIZE  # 2048


def _mapping_df(smiles: list[str] | None = None) -> pd.DataFrame:
    if smiles is None:
        smiles = VALID_SMILES
    return pd.DataFrame({
        "stitch_id": [f"CID{str(i).zfill(9)}" for i in range(len(smiles))],
        "parsed_pubchem_cid": list(range(len(smiles))),
        "standardized_smiles": smiles,
        "mapping_status": ["mapped"] * len(smiles),
    })


def _pairs_df(n_drugs: int = N_DRUGS, n_pairs: int = N_PAIRS) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    pairs: list[dict] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < n_pairs:
        a, b = sorted(rng.choice(n_drugs, size=2, replace=False).tolist())
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append({
                "pair_id": len(pairs),
                "drug_1": f"CID{str(a).zfill(9)}",
                "drug_2": f"CID{str(b).zfill(9)}",
            })
    return pd.DataFrame(pairs)


def _gen():
    return bmf.make_generator()


def _drug_matrix_and_sorted(mapping_df: pd.DataFrame | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    if mapping_df is None:
        mapping_df = _mapping_df()
    return bmf.generate_drug_fingerprints(mapping_df, _gen())


# ---------------------------------------------------------------------------
# Scenario 1: SMILES validation
# ---------------------------------------------------------------------------

class TestSmilesValidation:
    def test_invalid_smiles_raises(self):
        bad = _mapping_df(["C", "INVALID_SMILES_XYZ", "CC"])
        with pytest.raises(ValueError, match="RDKit could not parse"):
            bmf.generate_drug_fingerprints(bad, _gen())

    def test_null_smiles_raises_in_input_validation(self):
        mapping = _mapping_df()
        mapping.loc[0, "standardized_smiles"] = None
        pairs = _pairs_df()
        with pytest.raises(ValueError, match="null/empty"):
            bmf.validate_mapping_df(mapping, pairs)

    def test_all_valid_smiles_pass(self):
        matrix, _ = _drug_matrix_and_sorted()
        assert matrix.shape[0] == N_DRUGS


# ---------------------------------------------------------------------------
# Scenario 2: Duplicate STITCH ID detection
# ---------------------------------------------------------------------------

class TestDuplicateStitchId:
    def test_duplicate_stitch_id_raises(self):
        mapping = _mapping_df()
        mapping.loc[2, "stitch_id"] = mapping.loc[0, "stitch_id"]
        pairs = _pairs_df()
        with pytest.raises(ValueError, match="Duplicate stitch_ids"):
            bmf.validate_mapping_df(mapping, pairs)


# ---------------------------------------------------------------------------
# Scenario 3: Missing pair-table drug detection
# ---------------------------------------------------------------------------

class TestMissingPairDrug:
    def test_pair_drug_absent_from_mapping_raises(self):
        mapping = _mapping_df()
        pairs   = _pairs_df()
        pairs.loc[0, "drug_1"] = "CID999999999"
        with pytest.raises(ValueError, match="absent from mapping"):
            bmf.validate_mapping_df(mapping, pairs)

    def test_missing_column_in_pairs_raises(self):
        mapping = _mapping_df()
        pairs   = _pairs_df().drop(columns=["drug_1"])
        with pytest.raises(ValueError, match="missing columns"):
            bmf.validate_mapping_df(mapping, pairs)


# ---------------------------------------------------------------------------
# Scenario 4: Deterministic drug ordering
# ---------------------------------------------------------------------------

class TestDeterministicDrugOrdering:
    def test_drugs_sorted_by_stitch_id(self):
        mapping = _mapping_df().sample(frac=1, random_state=5).reset_index(drop=True)
        _, sorted_df = _drug_matrix_and_sorted(mapping)
        assert list(sorted_df["stitch_id"]) == sorted(sorted_df["stitch_id"].tolist())

    def test_shuffle_gives_same_matrix(self):
        mapping = _mapping_df()
        m1, _ = bmf.generate_drug_fingerprints(mapping, _gen())
        m2, _ = bmf.generate_drug_fingerprints(
            mapping.sample(frac=1, random_state=42).reset_index(drop=True), _gen()
        )
        assert np.array_equal(m1, m2)


# ---------------------------------------------------------------------------
# Scenario 5: Deterministic pair ordering
# ---------------------------------------------------------------------------

class TestDeterministicPairOrdering:
    def test_reversed_pairs_gives_same_matrix(self):
        mapping = _mapping_df()
        pairs   = _pairs_df()
        matrix, sorted_df = _drug_matrix_and_sorted(mapping)
        lookup  = bmf.build_stitch_lookup(bmf.build_drug_index(sorted_df, matrix))

        pm_fwd = bmf.build_pair_fingerprints(pairs, matrix, lookup)
        pm_rev = bmf.build_pair_fingerprints(
            pairs.iloc[::-1].reset_index(drop=True), matrix, lookup
        )
        assert np.array_equal(pm_fwd, pm_rev)


# ---------------------------------------------------------------------------
# Scenario 6: Fingerprint size is 2048
# ---------------------------------------------------------------------------

class TestFingerprintSize:
    def test_drug_fp_size_2048(self):
        matrix, _ = _drug_matrix_and_sorted()
        assert matrix.shape[1] == 2048

    def test_pair_fp_size_2048(self):
        mapping = _mapping_df()
        matrix, sorted_df = _drug_matrix_and_sorted(mapping)
        lookup  = bmf.build_stitch_lookup(bmf.build_drug_index(sorted_df, matrix))
        pair_m  = bmf.build_pair_fingerprints(_pairs_df(), matrix, lookup)
        assert pair_m.shape[1] == 2048


# ---------------------------------------------------------------------------
# Scenario 7: Drug values are binary {0, 1}
# ---------------------------------------------------------------------------

class TestDrugValuesBinary:
    def test_all_values_zero_or_one(self):
        matrix, _ = _drug_matrix_and_sorted()
        assert np.all((matrix == 0) | (matrix == 1))

    def test_dtype_uint8(self):
        matrix, _ = _drug_matrix_and_sorted()
        assert matrix.dtype == np.uint8


# ---------------------------------------------------------------------------
# Scenario 8: Pair values are only {0, 1, 2}
# ---------------------------------------------------------------------------

class TestPairValues:
    def test_pair_values_in_0_1_2(self):
        mapping = _mapping_df()
        matrix, sorted_df = _drug_matrix_and_sorted(mapping)
        lookup  = bmf.build_stitch_lookup(bmf.build_drug_index(sorted_df, matrix))
        pair_m  = bmf.build_pair_fingerprints(_pairs_df(), matrix, lookup)
        assert np.isin(pair_m, [0, 1, 2]).all()

    def test_pair_dtype_uint8(self):
        mapping = _mapping_df()
        matrix, sorted_df = _drug_matrix_and_sorted(mapping)
        lookup  = bmf.build_stitch_lookup(bmf.build_drug_index(sorted_df, matrix))
        pair_m  = bmf.build_pair_fingerprints(_pairs_df(), matrix, lookup)
        assert pair_m.dtype == np.uint8

    def test_validate_pair_matrix_invalid_value_raises(self):
        bad = np.full((N_PAIRS, FP_SIZE), 3, dtype=np.uint8)
        with pytest.raises(ValueError, match="invalid values"):
            bmf.validate_pair_matrix(bad, n_pairs=N_PAIRS)


# ---------------------------------------------------------------------------
# Scenario 9: Symmetry — A+B == B+A
# ---------------------------------------------------------------------------

class TestSymmetry:
    def test_swap_drugs_gives_same_fingerprint(self):
        mapping = _mapping_df()
        matrix, sorted_df = _drug_matrix_and_sorted(mapping)
        lookup  = bmf.build_stitch_lookup(bmf.build_drug_index(sorted_df, matrix))
        pairs   = _pairs_df()

        pairs_swapped = pairs.copy()
        pairs_swapped["drug_1"] = pairs["drug_2"]
        pairs_swapped["drug_2"] = pairs["drug_1"]

        pm_fwd = bmf.build_pair_fingerprints(pairs,         matrix, lookup)
        pm_rev = bmf.build_pair_fingerprints(pairs_swapped, matrix, lookup)
        assert np.array_equal(pm_fwd, pm_rev)

    def test_verify_symmetry_passes(self):
        mapping = _mapping_df()
        matrix, sorted_df = _drug_matrix_and_sorted(mapping)
        lookup  = bmf.build_stitch_lookup(bmf.build_drug_index(sorted_df, matrix))
        bmf.verify_symmetry(_pairs_df(), matrix, lookup, n_checks=3)


# ---------------------------------------------------------------------------
# Scenario 10: Pair result equals direct saved-drug fingerprint addition
# ---------------------------------------------------------------------------

class TestPairEqualsDirectAddition:
    def test_spot_check_all_pairs(self):
        mapping = _mapping_df()
        matrix, sorted_df = _drug_matrix_and_sorted(mapping)
        lookup  = bmf.build_stitch_lookup(bmf.build_drug_index(sorted_df, matrix))
        pairs   = _pairs_df()
        pm      = bmf.build_pair_fingerprints(pairs, matrix, lookup)

        for _, row in pairs.iterrows():
            pid = int(row["pair_id"])
            i1  = lookup[row["drug_1"]]
            i2  = lookup[row["drug_2"]]
            expected = (
                matrix[i1].astype(np.uint16) + matrix[i2].astype(np.uint16)
            ).astype(np.uint8)
            assert np.array_equal(pm[pid], expected), f"Mismatch at pair_id {pid}"


# ---------------------------------------------------------------------------
# Scenario 11: Row alignment
# ---------------------------------------------------------------------------

class TestRowAlignment:
    def test_drug_row_i_is_fingerprint_index_i(self):
        mapping = _mapping_df()
        matrix, sorted_df = _drug_matrix_and_sorted(mapping)
        drug_idx = bmf.build_drug_index(sorted_df, matrix)
        assert list(drug_idx["fingerprint_index"]) == list(range(N_DRUGS))

    def test_pair_row_i_is_pair_id_i(self):
        mapping = _mapping_df()
        matrix, sorted_df = _drug_matrix_and_sorted(mapping)
        lookup = bmf.build_stitch_lookup(bmf.build_drug_index(sorted_df, matrix))
        pairs  = _pairs_df()
        pm     = bmf.build_pair_fingerprints(pairs, matrix, lookup)
        assert pm.shape[0] == N_PAIRS
        # Row 0 must correspond to pair_id 0
        row0 = pairs[pairs["pair_id"] == 0].iloc[0]
        i1   = lookup[row0["drug_1"]]
        i2   = lookup[row0["drug_2"]]
        expected = (matrix[i1].astype(np.uint16) + matrix[i2].astype(np.uint16)).astype(np.uint8)
        assert np.array_equal(pm[0], expected)


# ---------------------------------------------------------------------------
# Scenario 12: dtype is uint8 (both matrices)
# ---------------------------------------------------------------------------

class TestDtype:
    def test_drug_dtype(self):
        m, _ = _drug_matrix_and_sorted()
        assert m.dtype == np.uint8

    def test_pair_dtype(self):
        mapping = _mapping_df()
        matrix, sorted_df = _drug_matrix_and_sorted(mapping)
        lookup  = bmf.build_stitch_lookup(bmf.build_drug_index(sorted_df, matrix))
        pm      = bmf.build_pair_fingerprints(_pairs_df(), matrix, lookup)
        assert pm.dtype == np.uint8

    def test_validate_drug_wrong_dtype_raises(self):
        bad = np.ones((N_DRUGS, FP_SIZE), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            bmf.validate_drug_matrix(bad, n_drugs=N_DRUGS)

    def test_validate_pair_wrong_dtype_raises(self):
        bad = np.ones((N_PAIRS, FP_SIZE), dtype=np.int32)
        with pytest.raises(ValueError, match="uint8"):
            bmf.validate_pair_matrix(bad, n_pairs=N_PAIRS)


# ---------------------------------------------------------------------------
# Scenario 13: Output-directory creation
# ---------------------------------------------------------------------------

class TestOutputDirCreation:
    def test_nested_output_dir_created(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "out.npy"
        drug_matrix = np.ones((N_DRUGS, FP_SIZE), dtype=np.uint8)
        bmf._save_npy_atomic(drug_matrix, deep)
        assert deep.exists()

    def test_write_atomic_creates_dir(self, tmp_path):
        p = tmp_path / "sub" / "file.csv"
        bmf._write_atomic(b"col\n1\n", p)
        assert p.exists()


# ---------------------------------------------------------------------------
# Scenario 14: Disk-loaded validation
# ---------------------------------------------------------------------------

class TestDiskValidation:
    def _setup(self, tmp_path, n_drugs=N_DRUGS, n_pairs=N_PAIRS):
        mapping = _mapping_df()
        pairs   = _pairs_df()
        matrix, sorted_df = _drug_matrix_and_sorted(mapping)
        lookup  = bmf.build_stitch_lookup(bmf.build_drug_index(sorted_df, matrix))
        pm      = bmf.build_pair_fingerprints(pairs, matrix, lookup)
        didx    = bmf.build_drug_index(sorted_df, matrix)
        pidx    = bmf.build_pair_index(pairs, lookup)

        dp  = tmp_path / "drug_fp.npy"
        dip = tmp_path / "drug_idx.csv"
        pp  = tmp_path / "pair_fp.npy"
        pip = tmp_path / "pair_idx.csv"

        np.save(dp, matrix); didx.to_csv(dip, index=False)
        np.save(pp, pm);     pidx.to_csv(pip, index=False)
        return dp, dip, pp, pip, matrix, pm, pairs

    def test_valid_artifacts_pass(self, tmp_path):
        dp, dip, pp, pip, _, _, pairs = self._setup(tmp_path)
        bmf.validate_from_disk(dp, dip, pp, pip, pairs, n_drugs=N_DRUGS)

    def test_bad_drug_shape_raises(self, tmp_path):
        dp, dip, pp, pip, matrix, _, pairs = self._setup(tmp_path)
        np.save(dp, matrix[:3])  # wrong row count
        with pytest.raises(ValueError):
            bmf.validate_from_disk(dp, dip, pp, pip, pairs, n_drugs=N_DRUGS)

    def test_non_binary_drug_values_raise(self, tmp_path):
        dp, dip, pp, pip, matrix, _, pairs = self._setup(tmp_path)
        bad = matrix.copy()
        bad[0, 0] = 3
        np.save(dp, bad)
        with pytest.raises(ValueError, match="outside"):
            bmf.validate_from_disk(dp, dip, pp, pip, pairs, n_drugs=N_DRUGS)


# ---------------------------------------------------------------------------
# Scenario 15: Overwrite protection
# ---------------------------------------------------------------------------

class TestOverwriteProtection:
    def test_skips_when_all_artifacts_exist(self, tmp_path):
        mapping = _mapping_df()
        pairs   = _pairs_df()
        matrix, sorted_df = _drug_matrix_and_sorted(mapping)
        lookup  = bmf.build_stitch_lookup(bmf.build_drug_index(sorted_df, matrix))
        pm      = bmf.build_pair_fingerprints(pairs, matrix, lookup)
        didx    = bmf.build_drug_index(sorted_df, matrix)
        pidx    = bmf.build_pair_index(pairs, lookup)

        dp  = tmp_path / "drug_fp.npy";  dip = tmp_path / "drug_idx.csv"
        pp  = tmp_path / "pair_fp.npy";  pip = tmp_path / "pair_idx.csv"
        mp  = tmp_path / "mapping.csv";  parq = tmp_path / "pairs.parquet"

        np.save(dp, matrix); didx.to_csv(dip, index=False)
        np.save(pp, pm);     pidx.to_csv(pip, index=False)
        mapping.to_csv(mp, index=False)
        pairs.to_parquet(parq, index=False)

        calls: list[int] = []
        original = bmf.validate_mapping_df
        def _spy(*a, **kw):
            calls.append(1)
            return original(*a, **kw)
        bmf.validate_mapping_df = _spy

        try:
            result = bmf.run_morgan_pipeline(
                mapping_input=mp,
                pairs_input=parq,
                drug_fp_output=dp,
                drug_idx_output=dip,
                pair_fp_output=pp,
                pair_idx_output=pip,
                audit_output=tmp_path / "audit.json",
                overwrite=False,
            )
        finally:
            bmf.validate_mapping_df = original

        assert len(calls) == 0
        assert "drug_matrix_shape" in result


# ---------------------------------------------------------------------------
# Scenario 16: Repeated runs produce identical artifact hashes
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_identical_hashes_on_second_run(self, tmp_path):
        mapping = _mapping_df()
        pairs   = _pairs_df()
        mp  = tmp_path / "mapping.csv"
        parq = tmp_path / "pairs.parquet"
        mapping.to_csv(mp, index=False)
        pairs.to_parquet(parq, index=False)

        kwargs = dict(
            mapping_input=mp, pairs_input=parq,
            drug_fp_output=tmp_path / "drug.npy",
            drug_idx_output=tmp_path / "drug.csv",
            pair_fp_output=tmp_path / "pair.npy",
            pair_idx_output=tmp_path / "pair.csv",
            audit_output=tmp_path / "audit.json",
        )

        bmf.run_morgan_pipeline(**kwargs, overwrite=True)
        sha1_drug = bmf.sha256_file(tmp_path / "drug.npy")
        sha1_pair = bmf.sha256_file(tmp_path / "pair.npy")

        bmf.run_morgan_pipeline(**kwargs, overwrite=True)
        sha2_drug = bmf.sha256_file(tmp_path / "drug.npy")
        sha2_pair = bmf.sha256_file(tmp_path / "pair.npy")

        assert sha1_drug == sha2_drug
        assert sha1_pair == sha2_pair
