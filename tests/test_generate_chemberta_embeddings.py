"""
Milestone 4 tests — generate_chemberta_embeddings.py
All 17 scenarios.  No HuggingFace downloads; tokenizer and model are mocked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

import polyllm.features.generate_chemberta_embeddings as gem


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

HIDDEN_DIM = 8  # small for speed


def _make_enc(n_seqs: int, seq_len: int = 6) -> dict[str, torch.Tensor]:
    """Fake batch encoding with ones attention masks."""
    return {
        "input_ids":      torch.ones(n_seqs, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(n_seqs, seq_len, dtype=torch.long),
    }


def _fake_tokenizer(n_seqs: int = 1, seq_len: int = 6) -> MagicMock:
    """Return a callable mock that produces deterministic batch encodings."""
    tok = MagicMock()

    def _call(smiles, **kwargs):
        batch = smiles if isinstance(smiles, list) else [smiles]
        enc = {
            "input_ids":      torch.ones(len(batch), seq_len, dtype=torch.long),
            "attention_mask": torch.ones(len(batch), seq_len, dtype=torch.long),
        }
        if kwargs.get("return_tensors") is None:
            # compute_token_lengths path: return plain lists
            return {k: v[0].tolist() for k, v in enc.items()}
        return enc

    tok.side_effect = _call
    return tok


def _fake_model(hidden_dim: int = HIDDEN_DIM) -> MagicMock:
    """Return a mock model that produces deterministic last_hidden_state output."""
    model = MagicMock()
    model.parameters.return_value = iter([])  # no parameters → smoke-test frozen check passes

    def _forward(**enc):
        n, seq_len = enc["input_ids"].shape
        hs = torch.full((n, seq_len, hidden_dim), 0.5)
        return SimpleNamespace(last_hidden_state=hs)

    model.side_effect = _forward
    return model


def _mapping_df(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "stitch_id": f"CID{str(i).zfill(9)}",
                "parsed_pubchem_cid": i,
                "preferred_name": f"drug_{i}",
                "pubchem_smiles": f"CC{i}",
                "standardized_smiles": f"CC{i}",
                "inchikey": f"ABCDEFGHIJKLMNO-X-{i}",
                "mapping_source": "pubchem_batch",
                "mapping_status": "mapped",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _pairs_df(stitch_ids: list[str]) -> pd.DataFrame:
    """Build a minimal pairs table using the first two drugs."""
    n = min(2, len(stitch_ids))
    pairs = [{"pair_id": 0, "drug_1": stitch_ids[0], "drug_2": stitch_ids[n - 1]}]
    return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Scenario 1: Input validation — missing columns
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_mapping_column_raises(self):
        df = _mapping_df(3).drop(columns=["standardized_smiles"])
        pairs = _pairs_df(["CID000000000"])
        with pytest.raises(ValueError, match="missing required columns"):
            gem.validate_mapping_df(df, pairs)

    def test_non_mapped_status_raises(self):
        df = _mapping_df(3)
        df.loc[0, "mapping_status"] = "pubchem_not_found"
        pairs = _pairs_df(df["stitch_id"].tolist())
        with pytest.raises(ValueError, match="non-mapped status"):
            gem.validate_mapping_df(df, pairs)

    def test_duplicate_stitch_ids_raise(self):
        df = _mapping_df(3)
        df.loc[2, "stitch_id"] = df.loc[0, "stitch_id"]
        pairs = _pairs_df(df["stitch_id"].tolist())
        with pytest.raises(ValueError, match="Duplicate stitch_ids"):
            gem.validate_mapping_df(df, pairs)

    def test_null_smiles_raises(self):
        df = _mapping_df(3)
        df.loc[0, "standardized_smiles"] = None
        pairs = _pairs_df(df["stitch_id"].tolist())
        with pytest.raises(ValueError, match="null/empty"):
            gem.validate_mapping_df(df, pairs)

    def test_pair_drug_missing_from_mapping_raises(self):
        df = _mapping_df(2)
        pairs = pd.DataFrame([{"pair_id": 0, "drug_1": "CID999999999", "drug_2": "CID000000000"}])
        with pytest.raises(ValueError, match="absent from mapping"):
            gem.validate_mapping_df(df, pairs)

    def test_valid_mapping_passes(self):
        df = _mapping_df(3)
        pairs = _pairs_df(df["stitch_id"].tolist())
        gem.validate_mapping_df(df, pairs)  # must not raise


# ---------------------------------------------------------------------------
# Scenario 2: Tokenization audit
# ---------------------------------------------------------------------------

class TestTokenizationAudit:
    def test_lengths_returned(self):
        smiles = ["C", "CC", "CCC"]
        ids = ["CID000000000", "CID000000001", "CID000000002"]
        tok = _fake_tokenizer(seq_len=6)
        result = gem.tokenization_audit(smiles, ids, tok, max_length=64)
        assert result["min_token_length"] == 6
        assert result["max_token_length"] == 6
        assert len(result["per_drug_token_lengths"]) == 3

    def test_truncation_flagged(self):
        smiles = ["C", "CC"]
        ids = ["CID000000000", "CID000000001"]
        tok = _fake_tokenizer(seq_len=8)  # 8 > max_length=5
        result = gem.tokenization_audit(smiles, ids, tok, max_length=5)
        assert result["count_exceeding_max"] == 2
        assert "CID000000000" in result["truncated_stitch_ids"]

    def test_no_truncation_no_ids_listed(self):
        smiles = ["C"]
        ids = ["CID000000000"]
        tok = _fake_tokenizer(seq_len=4)
        result = gem.tokenization_audit(smiles, ids, tok, max_length=64)
        assert result["count_exceeding_max"] == 0
        assert result["truncated_stitch_ids"] == []


# ---------------------------------------------------------------------------
# Scenario 3: Mean pooling
# ---------------------------------------------------------------------------

class TestMeanPool:
    def test_all_tokens_unmasked(self):
        hs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        am = torch.ones(1, 2, dtype=torch.long)
        out = gem.mean_pool(hs, am)
        expected = torch.tensor([[2.0, 3.0]])
        assert torch.allclose(out, expected)

    def test_padding_excluded(self):
        hs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])  # padded last
        am = torch.tensor([[1, 1, 0]])
        out = gem.mean_pool(hs, am)
        expected = torch.tensor([[2.0, 3.0]])
        assert torch.allclose(out, expected)

    def test_batch_of_two(self):
        hs = torch.ones(2, 3, 4)  # all 1.0
        am = torch.ones(2, 3, dtype=torch.long)
        out = gem.mean_pool(hs, am)
        assert out.shape == (2, 4)
        assert torch.allclose(out, torch.ones(2, 4))

    def test_fully_masked_clamp(self):
        hs = torch.ones(1, 3, 2)
        am = torch.zeros(1, 3, dtype=torch.long)
        out = gem.mean_pool(hs, am)
        # denominator clamped to 1 → result is sum of zeros = 0
        assert torch.allclose(out, torch.zeros(1, 2))


# ---------------------------------------------------------------------------
# Scenario 4: Smoke test
# ---------------------------------------------------------------------------

class TestSmokeTest:
    def test_smoke_passes_with_valid_mock(self):
        tok = _fake_tokenizer(seq_len=6)
        model = _fake_model(HIDDEN_DIM)
        device = torch.device("cpu")
        dim = gem.smoke_test(["C", "CC", "CCC"], tok, model, device)
        assert dim == HIDDEN_DIM

    def test_smoke_fails_on_nan(self):
        tok = _fake_tokenizer(seq_len=6)
        model = MagicMock()
        model.parameters.return_value = iter([])

        def _nan_forward(**enc):
            n, seq_len = enc["input_ids"].shape
            hs = torch.full((n, seq_len, HIDDEN_DIM), float("nan"))
            return SimpleNamespace(last_hidden_state=hs)

        model.side_effect = _nan_forward

        with pytest.raises(RuntimeError, match="NaN or Inf"):
            gem.smoke_test(["C", "CC", "CCC"], tok, model, torch.device("cpu"))

    def test_smoke_fails_on_frozen_violation(self):
        tok = _fake_tokenizer(seq_len=6)
        model = MagicMock()

        # Return a real parameter with requires_grad=True
        real_param = torch.nn.Parameter(torch.ones(1))
        model.parameters.return_value = iter([real_param])

        def _fwd(**enc):
            n, seq_len = enc["input_ids"].shape
            return SimpleNamespace(last_hidden_state=torch.ones(n, seq_len, HIDDEN_DIM))

        model.side_effect = _fwd

        with pytest.raises(RuntimeError, match="requires_grad"):
            gem.smoke_test(["C", "CC", "CCC"], tok, model, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Scenario 5: embed_drugs produces correct shape and dtype
# ---------------------------------------------------------------------------

class TestEmbedDrugs:
    def test_output_shape(self):
        smiles = [f"C{i}" for i in range(10)]
        tok = _fake_tokenizer(seq_len=6)
        model = _fake_model(HIDDEN_DIM)
        result = gem.embed_drugs(smiles, tok, model, batch_size=4, device=torch.device("cpu"))
        assert result.shape == (10, HIDDEN_DIM)

    def test_output_dtype_float32(self):
        smiles = ["C", "CC", "CCC"]
        tok = _fake_tokenizer(seq_len=6)
        model = _fake_model(HIDDEN_DIM)
        result = gem.embed_drugs(smiles, tok, model, batch_size=8, device=torch.device("cpu"))
        assert result.dtype == np.float32

    def test_batch_boundary(self):
        smiles = [f"C{i}" for i in range(6)]
        tok = _fake_tokenizer(seq_len=6)
        model = _fake_model(HIDDEN_DIM)
        # batch_size=6 → exactly one batch
        result = gem.embed_drugs(smiles, tok, model, batch_size=6, device=torch.device("cpu"))
        assert result.shape == (6, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Scenario 6: Drug index schema
# ---------------------------------------------------------------------------

class TestDrugIndex:
    def test_schema_and_sha256(self):
        mapping = _mapping_df(3)
        smiles = mapping["standardized_smiles"].tolist()
        tok_lengths = [6, 7, 8]
        index = gem.build_drug_index(mapping, tok_lengths, smiles, max_length=64)

        assert list(index.columns) == gem.REQUIRED_INDEX_COLUMNS
        assert len(index) == 3
        assert list(index["embedding_index"]) == [0, 1, 2]

        expected_sha = hashlib.sha256(smiles[0].encode("utf-8")).hexdigest()
        assert index.iloc[0]["smiles_sha256"] == expected_sha

    def test_truncation_flag(self):
        mapping = _mapping_df(2)
        smiles = mapping["standardized_smiles"].tolist()
        tok_lengths = [10, 4]  # 10 > max 8
        index = gem.build_drug_index(mapping, tok_lengths, smiles, max_length=8)
        assert index.iloc[0]["was_truncated"] == True   # noqa: E712 (numpy bool compat)
        assert index.iloc[1]["was_truncated"] == False  # noqa: E712

    def test_smiles_source_column_recorded(self):
        mapping = _mapping_df(2)
        smiles = mapping["standardized_smiles"].tolist()
        index = gem.build_drug_index(mapping, [5, 5], smiles, max_length=64)
        assert all(index["smiles_source_column"] == "standardized_smiles")


# ---------------------------------------------------------------------------
# Scenario 7: Post-run disk validation
# ---------------------------------------------------------------------------

class TestDiskValidation:
    def test_valid_artifacts_pass(self, tmp_path):
        n = 5
        matrix = np.ones((n, HIDDEN_DIM), dtype=np.float32)
        mapping = _mapping_df(n)
        smiles = mapping["standardized_smiles"].tolist()
        tok_lengths = [5] * n
        index_df = gem.build_drug_index(mapping, tok_lengths, smiles)
        pairs = _pairs_df(mapping["stitch_id"].tolist())

        embed_path = tmp_path / "emb.npy"
        index_path = tmp_path / "idx.csv"
        np.save(embed_path, matrix)
        index_df.to_csv(index_path, index=False)

        gem.validate_outputs_from_disk(embed_path, index_path, pairs, HIDDEN_DIM, expected_drug_count=n)

    def test_wrong_row_count_raises(self, tmp_path):
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float32)  # 3 ≠ 5
        mapping = _mapping_df(5)
        smiles = mapping["standardized_smiles"].tolist()
        index_df = gem.build_drug_index(mapping, [5] * 5, smiles)
        pairs = _pairs_df(mapping["stitch_id"].tolist())

        embed_path = tmp_path / "emb.npy"
        index_path = tmp_path / "idx.csv"
        np.save(embed_path, matrix)
        index_df.to_csv(index_path, index=False)

        # Matrix has 3 rows but index has 5 rows — shape mismatch detected
        with pytest.raises(ValueError, match="Matrix has"):
            gem.validate_outputs_from_disk(embed_path, index_path, pairs, HIDDEN_DIM, expected_drug_count=5)

    def test_non_finite_raises(self, tmp_path):
        n = 5
        matrix = np.ones((n, HIDDEN_DIM), dtype=np.float32)
        matrix[2, 3] = float("nan")
        mapping = _mapping_df(n)
        smiles = mapping["standardized_smiles"].tolist()
        index_df = gem.build_drug_index(mapping, [5] * n, smiles)
        pairs = _pairs_df(mapping["stitch_id"].tolist())

        embed_path = tmp_path / "emb.npy"
        index_path = tmp_path / "idx.csv"
        np.save(embed_path, matrix)
        index_df.to_csv(index_path, index=False)

        with pytest.raises(ValueError, match="non-finite"):
            gem.validate_outputs_from_disk(embed_path, index_path, pairs, HIDDEN_DIM, expected_drug_count=n)

    def test_missing_pair_drug_raises(self, tmp_path):
        n = 5
        matrix = np.ones((n, HIDDEN_DIM), dtype=np.float32)
        mapping = _mapping_df(n)
        smiles = mapping["standardized_smiles"].tolist()
        index_df = gem.build_drug_index(mapping, [5] * n, smiles)
        # pairs reference a drug not in index
        pairs = pd.DataFrame([{"pair_id": 0, "drug_1": "CID999999999", "drug_2": "CID000000000"}])

        embed_path = tmp_path / "emb.npy"
        index_path = tmp_path / "idx.csv"
        np.save(embed_path, matrix)
        index_df.to_csv(index_path, index=False)

        with pytest.raises(ValueError, match="pair-table drugs missing"):
            gem.validate_outputs_from_disk(embed_path, index_path, pairs, HIDDEN_DIM, expected_drug_count=n)


# ---------------------------------------------------------------------------
# Scenario 8: Audit JSON schema
# ---------------------------------------------------------------------------

class TestAuditSchema:
    def test_required_fields_present(self, tmp_path):
        mapping = _mapping_df(5)
        smiles = mapping["standardized_smiles"].tolist()
        tok_lengths = [5] * 5
        tok_audit = {
            "min_token_length": 5,
            "median_token_length": 5.0,
            "mean_token_length": 5.0,
            "max_token_length": 5,
            "count_exceeding_max": 0,
            "percent_exceeding_max": 0.0,
            "truncated_stitch_ids": [],
            "per_drug_token_lengths": tok_lengths,
        }
        matrix = np.ones((5, HIDDEN_DIM), dtype=np.float32)
        index_df = gem.build_drug_index(mapping, tok_lengths, smiles)
        pairs = _pairs_df(mapping["stitch_id"].tolist())

        embed_path = tmp_path / "emb.npy"
        index_path = tmp_path / "idx.csv"
        audit_path = tmp_path / "audit.json"
        np.save(embed_path, matrix)
        index_df.to_csv(index_path, index=False)

        audit = gem.build_embedding_audit(
            mapping, index_df, matrix, tok_audit,
            resolved_revision="abc123",
            device_str="cpu",
            cuda_available=False,
            batch_size=32,
            embed_path=embed_path,
            index_path=index_path,
            audit_path=audit_path,
            tokenizer_class="RobertaTokenizerFast",
            model_class="RobertaModel",
        )
        required_keys = [
            "model_name", "resolved_model_revision", "tokenizer_class", "model_class",
            "source_smiles_column", "maximum_token_length", "pooling_method",
            "special_tokens_included_in_pooling", "batch_size", "device",
            "cuda_available", "total_drugs", "embedding_dimension", "embedding_shape",
            "embedding_dtype", "finite_value_check", "all_zero_row_count",
            "token_length_summary", "truncated_drug_count", "truncated_drug_ids",
            "software_versions", "input_paths", "output_paths", "artifact_sha256",
        ]
        for k in required_keys:
            assert k in audit, f"Missing audit key: {k}"

    def test_special_tokens_flag_true(self, tmp_path):
        mapping = _mapping_df(3)
        smiles = mapping["standardized_smiles"].tolist()
        tok_audit = {"min_token_length": 5, "median_token_length": 5.0, "mean_token_length": 5.0,
                     "max_token_length": 5, "count_exceeding_max": 0, "percent_exceeding_max": 0.0,
                     "truncated_stitch_ids": [], "per_drug_token_lengths": [5, 5, 5]}
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float32)
        index_df = gem.build_drug_index(mapping, [5, 5, 5], smiles)
        embed_path = tmp_path / "e.npy"; index_path = tmp_path / "i.csv"
        np.save(embed_path, matrix); index_df.to_csv(index_path, index=False)
        audit = gem.build_embedding_audit(
            mapping, index_df, matrix, tok_audit, None, "cpu", False, 32,
            embed_path, index_path, tmp_path / "a.json", "Tok", "Mdl",
        )
        assert audit["special_tokens_included_in_pooling"] is True


# ---------------------------------------------------------------------------
# Scenario 9: Overwrite guard — skip if artifacts exist
# ---------------------------------------------------------------------------

class TestOverwriteGuard:
    def test_skips_when_artifacts_exist(self, tmp_path):
        n = 5
        mapping = _mapping_df(n)
        smiles = mapping["standardized_smiles"].tolist()
        tok_lengths = [5] * n
        matrix = np.ones((n, HIDDEN_DIM), dtype=np.float32)
        index_df = gem.build_drug_index(mapping, tok_lengths, smiles)

        embed_path = tmp_path / "emb.npy"
        index_path = tmp_path / "idx.csv"
        pairs_path = tmp_path / "pairs.parquet"

        # Write a real parquet file so read_parquet in the guard path works
        pairs = _pairs_df(mapping["stitch_id"].tolist())
        pairs.to_parquet(pairs_path, index=False)

        np.save(embed_path, matrix)
        index_df.to_csv(index_path, index=False)

        # Count how many times validate_mapping_df is called via a side-effect check
        original_fn = gem.validate_mapping_df
        calls: list[int] = []

        def _counting_validate(*a, **kw):
            calls.append(1)
            return original_fn(*a, **kw)

        gem.validate_mapping_df = _counting_validate
        try:
            result = gem.run_embedding_pipeline(
                mapping_input=Path("unused_mapping.csv"),
                pairs_input=pairs_path,
                embed_output=embed_path,
                index_output=index_path,
                audit_output=tmp_path / "audit.json",
                overwrite=False,
            )
        finally:
            gem.validate_mapping_df = original_fn

        # validate_mapping_df must NOT have been called (skipped path taken)
        assert len(calls) == 0
        # Result is the audit dict from _audit_existing
        assert "embedding_shape" in result


# ---------------------------------------------------------------------------
# Scenario 10: Deterministic embedding order (sorted stitch_ids)
# ---------------------------------------------------------------------------

class TestDeterministicOrder:
    def test_sort_by_stitch_id(self):
        mapping = _mapping_df(5)
        # Reverse the order
        mapping = mapping.sort_values("stitch_id", ascending=False).reset_index(drop=True)
        tok = _fake_tokenizer(seq_len=6)
        model = _fake_model(HIDDEN_DIM)

        smiles = mapping.sort_values("stitch_id")["standardized_smiles"].tolist()
        result = gem.embed_drugs(smiles, tok, model, batch_size=5, device=torch.device("cpu"))
        assert result.shape == (5, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Scenario 11: compute_token_lengths — special tokens, no truncation
# ---------------------------------------------------------------------------

class TestComputeTokenLengths:
    def test_no_truncation_flag_in_call(self):
        smiles = ["C", "CC"]
        call_kwargs: list[dict] = []

        class CaptureTok:
            def __call__(self, smi, **kwargs):
                call_kwargs.append(kwargs)
                return {"input_ids": [1, 2, 3, 4, 5]}

        tok = CaptureTok()
        gem.compute_token_lengths(smiles, tok)

        for kw in call_kwargs:
            assert kw.get("truncation") is False
            assert kw.get("add_special_tokens") is True

    def test_length_matches_ids(self):
        class FixedTok:
            def __call__(self, smi, **kwargs):
                return {"input_ids": list(range(10))}

        tok = FixedTok()
        lengths = gem.compute_token_lengths(["C", "CC"], tok)
        assert lengths == [10, 10]


# ---------------------------------------------------------------------------
# Scenario 12: SHA-256 determinism
# ---------------------------------------------------------------------------

class TestSha256:
    def test_deterministic(self, tmp_path):
        path = tmp_path / "test.bin"
        path.write_bytes(b"hello")
        sha1 = gem.sha256_file(path)
        sha2 = gem.sha256_file(path)
        assert sha1 == sha2

    def test_known_value(self, tmp_path):
        path = tmp_path / "test.bin"
        content = b"test content"
        path.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert gem.sha256_file(path) == expected


# ---------------------------------------------------------------------------
# Scenario 13: All-zero row detection
# ---------------------------------------------------------------------------

class TestAllZeroRows:
    def test_all_zero_row_raises_in_memory(self):
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float32)
        matrix[1, :] = 0.0
        with pytest.raises(ValueError, match="all-zero rows"):
            gem._validate_matrix_in_memory(matrix, HIDDEN_DIM)

    def test_valid_matrix_passes(self):
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float32)
        gem._validate_matrix_in_memory(matrix, HIDDEN_DIM)  # no raise


# ---------------------------------------------------------------------------
# Scenario 14: NaN / Inf detection
# ---------------------------------------------------------------------------

class TestFiniteCheck:
    def test_nan_raises_in_memory(self):
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float32)
        matrix[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            gem._validate_matrix_in_memory(matrix, HIDDEN_DIM)

    def test_inf_raises_in_memory(self):
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float32)
        matrix[2, 2] = float("inf")
        with pytest.raises(ValueError, match="NaN or Inf"):
            gem._validate_matrix_in_memory(matrix, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Scenario 15: dtype enforcement
# ---------------------------------------------------------------------------

class TestDtypeEnforcement:
    def test_float64_raises(self):
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            gem._validate_matrix_in_memory(matrix, HIDDEN_DIM)

    def test_float32_passes(self):
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float32)
        gem._validate_matrix_in_memory(matrix, HIDDEN_DIM)  # no raise


# ---------------------------------------------------------------------------
# Scenario 16: SMILES source column constant
# ---------------------------------------------------------------------------

class TestSmilesSourceColumn:
    def test_constant_value(self):
        assert gem.SMILES_COLUMN == "standardized_smiles"

    def test_index_records_column(self):
        mapping = _mapping_df(2)
        index = gem.build_drug_index(
            mapping, [5, 5], mapping["standardized_smiles"].tolist()
        )
        assert (index["smiles_source_column"] == "standardized_smiles").all()

    def test_unexpected_rdkit_column_raises(self):
        """If rdkit_canonical_isomeric_smiles ever appears, validate_mapping_df must raise."""
        df = _mapping_df(3)
        df["rdkit_canonical_isomeric_smiles"] = df["standardized_smiles"]
        pairs = _pairs_df(df["stitch_id"].tolist())
        with pytest.raises(ValueError, match="rdkit_canonical_isomeric_smiles"):
            gem.validate_mapping_df(df, pairs)


# ---------------------------------------------------------------------------
# Scenario 17: Requires_grad_(False) enforced in smoke test
# ---------------------------------------------------------------------------

class TestModelFrozen:
    def test_frozen_model_passes_smoke(self):
        tok = _fake_tokenizer(seq_len=6)
        # The fake model's parameters() returns an empty iterator → passes freeze check
        model = _fake_model(HIDDEN_DIM)
        dim = gem.smoke_test(["C", "CC", "CCC"], tok, model, torch.device("cpu"))
        assert dim == HIDDEN_DIM

    def test_unfrozen_param_fails_smoke(self):
        tok = _fake_tokenizer(seq_len=6)
        model = MagicMock()
        real_param = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        model.parameters.return_value = iter([real_param])

        def _fwd(**enc):
            n, sl = enc["input_ids"].shape
            return SimpleNamespace(last_hidden_state=torch.ones(n, sl, HIDDEN_DIM))

        model.side_effect = _fwd

        with pytest.raises(RuntimeError, match="requires_grad"):
            gem.smoke_test(["C", "CC", "CCC"], tok, model, torch.device("cpu"))
