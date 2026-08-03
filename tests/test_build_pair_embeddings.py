"""
Milestone 5 tests — build_pair_embeddings.py

All tests use small synthetic data; no real files are loaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import polyllm.features.build_pair_embeddings as bpe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DRUGS  = 6
N_PAIRS  = 8
HIDDEN   = 4


def _drug_matrix(n: int = N_DRUGS, h: int = HIDDEN) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, h)).astype(np.float32)


def _drug_index(n: int = N_DRUGS) -> pd.DataFrame:
    return pd.DataFrame({
        "embedding_index": list(range(n)),
        "stitch_id": [f"CID{str(i).zfill(9)}" for i in range(n)],
    })


def _pairs_df(n_drugs: int = N_DRUGS, n_pairs: int = N_PAIRS) -> pd.DataFrame:
    """Generate n_pairs canonical pairs from n_drugs drugs."""
    rng = np.random.default_rng(1)
    pairs = []
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


def _stitch_lookup(drug_index_df: pd.DataFrame) -> dict[str, int]:
    return bpe.build_stitch_lookup(drug_index_df)


# ---------------------------------------------------------------------------
# Scenario 1: Input validation — drug matrix
# ---------------------------------------------------------------------------

class TestValidateDrugInputs:
    def test_missing_column_raises(self):
        matrix = _drug_matrix()
        idx = _drug_index().drop(columns=["stitch_id"])
        with pytest.raises(ValueError, match="missing columns"):
            bpe.validate_drug_inputs(matrix, idx)

    def test_1d_matrix_raises(self):
        idx = _drug_index()
        with pytest.raises(ValueError, match="2-D"):
            bpe.validate_drug_inputs(np.ones(N_DRUGS, dtype=np.float32), idx)

    def test_wrong_dtype_raises(self):
        matrix = _drug_matrix().astype(np.float64)
        idx = _drug_index()
        with pytest.raises(ValueError, match="float32"):
            bpe.validate_drug_inputs(matrix, idx)

    def test_row_mismatch_raises(self):
        matrix = _drug_matrix(n=4)          # 4 rows but index has 6
        idx = _drug_index(n=6)
        with pytest.raises(ValueError, match="rows"):
            bpe.validate_drug_inputs(matrix, idx)

    def test_duplicate_stitch_ids_raise(self):
        matrix = _drug_matrix()
        idx = _drug_index()
        idx.loc[2, "stitch_id"] = idx.loc[0, "stitch_id"]
        with pytest.raises(ValueError, match="duplicate stitch_ids"):
            bpe.validate_drug_inputs(matrix, idx)

    def test_non_sequential_embedding_index_raises(self):
        matrix = _drug_matrix()
        idx = _drug_index()
        idx.loc[0, "embedding_index"] = 99
        with pytest.raises(ValueError, match="embedding_index"):
            bpe.validate_drug_inputs(matrix, idx)

    def test_valid_inputs_pass(self):
        matrix = _drug_matrix()
        idx = _drug_index()
        bpe.validate_drug_inputs(matrix, idx)  # must not raise


# ---------------------------------------------------------------------------
# Scenario 2: Input validation — pairs table
# ---------------------------------------------------------------------------

class TestValidatePairsDf:
    def _known(self, n: int = N_DRUGS) -> set[str]:
        return {f"CID{str(i).zfill(9)}" for i in range(n)}

    def test_missing_column_raises(self):
        pairs = _pairs_df().drop(columns=["drug_1"])
        with pytest.raises(ValueError, match="missing columns"):
            bpe.validate_pairs_df(pairs, self._known())

    def test_duplicate_pair_ids_raise(self):
        pairs = _pairs_df()
        pairs.loc[1, "pair_id"] = 0
        with pytest.raises(ValueError, match="duplicate pair_ids"):
            bpe.validate_pairs_df(pairs, self._known())

    def test_non_sequential_pair_ids_raise(self):
        pairs = _pairs_df()
        pairs.loc[0, "pair_id"] = 999
        with pytest.raises(ValueError, match="sequential"):
            bpe.validate_pairs_df(pairs, self._known())

    def test_unknown_drug_raises(self):
        pairs = _pairs_df()
        pairs.loc[0, "drug_1"] = "CID999999999"
        with pytest.raises(ValueError, match="no drug embedding"):
            bpe.validate_pairs_df(pairs, self._known())

    def test_valid_pairs_pass(self):
        pairs = _pairs_df()
        bpe.validate_pairs_df(pairs, self._known())  # must not raise


# ---------------------------------------------------------------------------
# Scenario 3: Sum operation is correct
# ---------------------------------------------------------------------------

class TestSumOperation:
    def test_pair_is_sum_of_drug_embeddings(self):
        matrix = _drug_matrix()
        idx    = _drug_index()
        pairs  = _pairs_df()
        lookup = _stitch_lookup(idx)

        result = bpe.build_pair_matrix(pairs, matrix, lookup)

        for i, row in pairs.iterrows():
            i1 = lookup[row["drug_1"]]
            i2 = lookup[row["drug_2"]]
            expected = matrix[i1] + matrix[i2]
            assert np.allclose(result[i], expected, atol=1e-6), \
                f"Row {i} mismatch: got {result[i]} expected {expected}"

    def test_output_shape(self):
        matrix = _drug_matrix()
        pairs  = _pairs_df()
        lookup = _stitch_lookup(_drug_index())
        result = bpe.build_pair_matrix(pairs, matrix, lookup)
        assert result.shape == (N_PAIRS, HIDDEN)

    def test_output_dtype_float32(self):
        result = bpe.build_pair_matrix(
            _pairs_df(), _drug_matrix(), _stitch_lookup(_drug_index())
        )
        assert result.dtype == np.float32

    def test_pairs_sorted_by_pair_id(self):
        """Reversing pair order in the input DF must not change the output."""
        matrix = _drug_matrix()
        pairs  = _pairs_df()
        lookup = _stitch_lookup(_drug_index())

        result_fwd = bpe.build_pair_matrix(pairs, matrix, lookup)
        result_rev = bpe.build_pair_matrix(pairs.iloc[::-1].reset_index(drop=True), matrix, lookup)

        assert np.allclose(result_fwd, result_rev, atol=1e-6)


# ---------------------------------------------------------------------------
# Scenario 4: Symmetry
# ---------------------------------------------------------------------------

class TestSymmetry:
    def test_swapping_drug_order_gives_same_embedding(self):
        matrix = _drug_matrix()
        idx    = _drug_index()
        lookup = _stitch_lookup(idx)

        pairs = _pairs_df()
        # Build a swapped version
        pairs_swapped = pairs.copy()
        pairs_swapped["drug_1"] = pairs["drug_2"]
        pairs_swapped["drug_2"] = pairs["drug_1"]

        result_fwd = bpe.build_pair_matrix(pairs, matrix, lookup)
        result_rev = bpe.build_pair_matrix(pairs_swapped, matrix, lookup)

        assert np.allclose(result_fwd, result_rev, atol=1e-6)

    def test_verify_symmetry_passes(self):
        matrix = _drug_matrix()
        idx    = _drug_index()
        pairs  = _pairs_df()
        lookup = _stitch_lookup(idx)
        bpe.verify_symmetry(pairs, matrix, lookup, n_checks=3)  # must not raise


# ---------------------------------------------------------------------------
# Scenario 5: In-memory validation
# ---------------------------------------------------------------------------

class TestInMemoryValidation:
    def test_wrong_shape_raises(self):
        matrix = np.ones((5, HIDDEN), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            bpe.validate_pair_matrix_in_memory(matrix, n_pairs=10, hidden_dim=HIDDEN)

    def test_wrong_dtype_raises(self):
        matrix = np.ones((N_PAIRS, HIDDEN), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            bpe.validate_pair_matrix_in_memory(matrix, n_pairs=N_PAIRS, hidden_dim=HIDDEN)

    def test_nan_raises(self):
        matrix = np.ones((N_PAIRS, HIDDEN), dtype=np.float32)
        matrix[0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            bpe.validate_pair_matrix_in_memory(matrix, n_pairs=N_PAIRS, hidden_dim=HIDDEN)

    def test_all_zero_row_raises(self):
        matrix = np.ones((N_PAIRS, HIDDEN), dtype=np.float32)
        matrix[3, :] = 0.0
        with pytest.raises(ValueError, match="all-zero"):
            bpe.validate_pair_matrix_in_memory(matrix, n_pairs=N_PAIRS, hidden_dim=HIDDEN)

    def test_valid_matrix_passes(self):
        matrix = np.ones((N_PAIRS, HIDDEN), dtype=np.float32)
        bpe.validate_pair_matrix_in_memory(matrix, n_pairs=N_PAIRS, hidden_dim=HIDDEN)


# ---------------------------------------------------------------------------
# Scenario 6: Disk validation
# ---------------------------------------------------------------------------

class TestDiskValidation:
    def _write_artifacts(self, tmp_path, matrix, index_df):
        ep = tmp_path / "pair_emb.npy"
        ip = tmp_path / "pair_idx.csv"
        np.save(ep, matrix)
        index_df.to_csv(ip, index=False)
        return ep, ip

    def test_valid_artifacts_pass(self, tmp_path):
        pairs  = _pairs_df()
        matrix = np.ones((N_PAIRS, HIDDEN), dtype=np.float32)
        index  = bpe.build_pair_index(pairs)
        ep, ip = self._write_artifacts(tmp_path, matrix, index)
        bpe.validate_outputs_from_disk(ep, ip, pairs, HIDDEN)  # no raise

    def test_wrong_shape_raises(self, tmp_path):
        pairs  = _pairs_df()
        matrix = np.ones((3, HIDDEN), dtype=np.float32)  # 3 ≠ N_PAIRS
        index  = bpe.build_pair_index(pairs)
        ep, ip = self._write_artifacts(tmp_path, matrix, index)
        with pytest.raises(ValueError, match="shape"):
            bpe.validate_outputs_from_disk(ep, ip, pairs, HIDDEN)

    def test_non_finite_raises(self, tmp_path):
        pairs  = _pairs_df()
        matrix = np.ones((N_PAIRS, HIDDEN), dtype=np.float32)
        matrix[1, 2] = float("inf")
        index  = bpe.build_pair_index(pairs)
        ep, ip = self._write_artifacts(tmp_path, matrix, index)
        with pytest.raises(ValueError, match="non-finite"):
            bpe.validate_outputs_from_disk(ep, ip, pairs, HIDDEN)

    def test_wrong_index_rows_raises(self, tmp_path):
        pairs  = _pairs_df()
        matrix = np.ones((N_PAIRS, HIDDEN), dtype=np.float32)
        # truncate index to 3 rows
        index  = bpe.build_pair_index(pairs).head(3)
        ep, ip = self._write_artifacts(tmp_path, matrix, index)
        with pytest.raises(ValueError, match="pair index has"):
            bpe.validate_outputs_from_disk(ep, ip, pairs, HIDDEN)


# ---------------------------------------------------------------------------
# Scenario 7: Pair index schema
# ---------------------------------------------------------------------------

class TestPairIndex:
    def test_schema(self):
        pairs = _pairs_df()
        index = bpe.build_pair_index(pairs)
        assert "pair_id"       in index.columns
        assert "drug_1"        in index.columns
        assert "drug_2"        in index.columns
        assert "embedding_row" in index.columns

    def test_sorted_by_pair_id(self):
        pairs = _pairs_df().sample(frac=1, random_state=7).reset_index(drop=True)
        index = bpe.build_pair_index(pairs)
        assert list(index["pair_id"]) == list(range(N_PAIRS))

    def test_embedding_row_equals_pair_id(self):
        pairs = _pairs_df()
        index = bpe.build_pair_index(pairs)
        assert (index["embedding_row"] == index["pair_id"]).all()

    def test_row_count(self):
        pairs = _pairs_df()
        index = bpe.build_pair_index(pairs)
        assert len(index) == N_PAIRS


# ---------------------------------------------------------------------------
# Scenario 8: Audit fields
# ---------------------------------------------------------------------------

class TestAuditFields:
    def test_required_keys_present(self, tmp_path):
        matrix = np.ones((N_PAIRS, HIDDEN), dtype=np.float32)
        pairs  = _pairs_df()
        ep = tmp_path / "e.npy"
        ip = tmp_path / "i.csv"
        np.save(ep, matrix)
        bpe.build_pair_index(pairs).to_csv(ip, index=False)

        audit = bpe.build_pair_embedding_audit(
            matrix, pairs, HIDDEN,
            drug_embed_path=Path("data/features/chemberta_drug_embeddings.npy"),
            pair_embed_path=ep,
            pair_index_path=ip,
        )
        for key in ["operation", "symmetric", "pair_matrix_shape", "pair_matrix_dtype",
                    "total_pairs", "hidden_dim", "finite_value_check", "all_zero_row_count",
                    "l2_norm_stats", "software_versions", "artifact_sha256",
                    "input_paths", "output_paths"]:
            assert key in audit, f"Missing audit key: {key}"

    def test_symmetric_flag_true(self, tmp_path):
        matrix = np.ones((N_PAIRS, HIDDEN), dtype=np.float32)
        pairs  = _pairs_df()
        ep = tmp_path / "e.npy"; ip = tmp_path / "i.csv"
        np.save(ep, matrix); bpe.build_pair_index(pairs).to_csv(ip, index=False)
        audit = bpe.build_pair_embedding_audit(matrix, pairs, HIDDEN, Path("d"), ep, ip)
        assert audit["symmetric"] is True

    def test_operation_mentions_sum(self, tmp_path):
        matrix = np.ones((N_PAIRS, HIDDEN), dtype=np.float32)
        pairs  = _pairs_df()
        ep = tmp_path / "e.npy"; ip = tmp_path / "i.csv"
        np.save(ep, matrix); bpe.build_pair_index(pairs).to_csv(ip, index=False)
        audit = bpe.build_pair_embedding_audit(matrix, pairs, HIDDEN, Path("d"), ep, ip)
        assert "sum" in audit["operation"].lower()


# ---------------------------------------------------------------------------
# Scenario 9: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_output_on_repeated_calls(self):
        matrix = _drug_matrix()
        idx    = _drug_index()
        pairs  = _pairs_df()
        lookup = _stitch_lookup(idx)

        r1 = bpe.build_pair_matrix(pairs, matrix, lookup)
        r2 = bpe.build_pair_matrix(pairs, matrix, lookup)
        assert np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Scenario 10: Overwrite guard
# ---------------------------------------------------------------------------

class TestOverwriteGuard:
    def test_skips_when_artifacts_exist(self, tmp_path):
        pairs  = _pairs_df()
        matrix = np.ones((N_PAIRS, HIDDEN), dtype=np.float32)
        index  = bpe.build_pair_index(pairs)

        ep = tmp_path / "pair_emb.npy"
        ip = tmp_path / "pair_idx.csv"
        np.save(ep, matrix)
        index.to_csv(ip, index=False)

        # Write real parquet and npy/csv inputs
        drug_matrix = _drug_matrix()
        drug_idx    = _drug_index()
        dp  = tmp_path / "drug_emb.npy"
        dip = tmp_path / "drug_idx.csv"
        pp  = tmp_path / "pairs.parquet"
        np.save(dp, drug_matrix)
        drug_idx.to_csv(dip, index=False)
        pairs.to_parquet(pp, index=False)

        calls: list[int] = []
        original = bpe.validate_drug_inputs
        def _spy(*a, **kw):
            calls.append(1)
            return original(*a, **kw)
        bpe.validate_drug_inputs = _spy

        try:
            result = bpe.run_pair_embedding_pipeline(
                drug_embed_input=dp,
                drug_index_input=dip,
                pairs_input=pp,
                pair_embed_output=ep,
                pair_index_output=ip,
                audit_output=tmp_path / "audit.json",
                overwrite=False,
            )
        finally:
            bpe.validate_drug_inputs = original

        assert len(calls) == 0
        assert "pair_matrix_shape" in result


# ---------------------------------------------------------------------------
# Scenario 11: SHA-256
# ---------------------------------------------------------------------------

class TestSha256:
    def test_deterministic(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello world")
        assert bpe.sha256_file(p) == bpe.sha256_file(p)

    def test_known_value(self, tmp_path):
        content = b"chemberta pair embeddings"
        p = tmp_path / "f.bin"
        p.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert bpe.sha256_file(p) == expected


# ---------------------------------------------------------------------------
# Scenario 12: Full pipeline integration (synthetic data)
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_end_to_end(self, tmp_path):
        drug_matrix = _drug_matrix()
        drug_idx    = _drug_index()
        pairs       = _pairs_df()

        dp  = tmp_path / "drug_emb.npy"
        dip = tmp_path / "drug_idx.csv"
        pp  = tmp_path / "pairs.parquet"
        ep  = tmp_path / "pair_emb.npy"
        ip  = tmp_path / "pair_idx.csv"
        ap  = tmp_path / "audit.json"

        np.save(dp, drug_matrix)
        drug_idx.to_csv(dip, index=False)
        pairs.to_parquet(pp, index=False)

        audit = bpe.run_pair_embedding_pipeline(
            drug_embed_input=dp,
            drug_index_input=dip,
            pairs_input=pp,
            pair_embed_output=ep,
            pair_index_output=ip,
            audit_output=ap,
            overwrite=True,
        )

        # Verify matrix written
        result = np.load(ep)
        assert result.shape == (N_PAIRS, HIDDEN)
        assert result.dtype == np.float32

        # Verify index written
        idx_df = pd.read_csv(ip)
        assert len(idx_df) == N_PAIRS
        assert list(idx_df["pair_id"]) == list(range(N_PAIRS))

        # Verify audit
        assert audit["pair_matrix_shape"] == [N_PAIRS, HIDDEN]
        assert audit["symmetric"] is True

        # Spot-check one row
        row = pairs.iloc[0]
        lookup = bpe.build_stitch_lookup(drug_idx)
        expected = drug_matrix[lookup[row["drug_1"]]] + drug_matrix[lookup[row["drug_2"]]]
        assert np.allclose(result[0], expected, atol=1e-6)
