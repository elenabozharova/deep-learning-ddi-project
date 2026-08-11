"""
Milestone 10b tests — generate_side_effect_embeddings.py
No HuggingFace downloads; tokenizer and model are mocked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

import polyllm.features.generate_side_effect_embeddings as gse


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

HIDDEN_DIM = 8  # small for speed


def _fake_tokenizer(seq_len: int = 6) -> MagicMock:
    tok = MagicMock()

    def _call(names, **kwargs):
        batch = names if isinstance(names, list) else [names]
        enc = {
            "input_ids":      torch.ones(len(batch), seq_len, dtype=torch.long),
            "attention_mask": torch.ones(len(batch), seq_len, dtype=torch.long),
        }
        if kwargs.get("return_tensors") is None:
            return {k: v[0].tolist() for k, v in enc.items()}
        return enc

    tok.side_effect = _call
    return tok


def _fake_model(hidden_dim: int = HIDDEN_DIM) -> MagicMock:
    model = MagicMock()
    model.parameters.return_value = iter([])

    def _forward(**enc):
        n, seq_len = enc["input_ids"].shape
        hs = torch.full((n, seq_len, hidden_dim), 0.5)
        return SimpleNamespace(last_hidden_state=hs)

    model.side_effect = _forward
    return model


def _mapping_df(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "label_index": i,
            "side_effect_id": f"C{str(i).zfill(7)}",
            "side_effect_name": f"condition_{i}",
            "unique_pair_count": 100 + i,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scenario 1: Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_column_raises(self):
        df = _mapping_df(3).drop(columns=["side_effect_name"])
        with pytest.raises(ValueError, match="missing required columns"):
            gse.validate_mapping_df(df)

    def test_non_sequential_label_index_raises(self):
        df = _mapping_df(3)
        df.loc[1, "label_index"] = 5
        with pytest.raises(ValueError, match="0..n-1"):
            gse.validate_mapping_df(df)

    def test_duplicate_side_effect_ids_raise(self):
        df = _mapping_df(3)
        df.loc[2, "side_effect_id"] = df.loc[0, "side_effect_id"]
        with pytest.raises(ValueError, match="Duplicate side_effect_ids"):
            gse.validate_mapping_df(df)

    def test_null_name_raises(self):
        df = _mapping_df(3)
        df.loc[0, "side_effect_name"] = None
        with pytest.raises(ValueError, match="null/empty"):
            gse.validate_mapping_df(df)

    def test_valid_mapping_passes(self):
        df = _mapping_df(3)
        gse.validate_mapping_df(df)  # must not raise


# ---------------------------------------------------------------------------
# Scenario 2: Tokenization audit
# ---------------------------------------------------------------------------

class TestTokenizationAudit:
    def test_lengths_returned(self):
        names = ["abdominal pain", "edema", "nightmare"]
        ids = ["C0000737", "C0013604", "C0028084"]
        tok = _fake_tokenizer(seq_len=6)
        result = gse.tokenization_audit(names, ids, tok, max_length=64)
        assert result["min_token_length"] == 6
        assert result["max_token_length"] == 6
        assert len(result["per_side_effect_token_lengths"]) == 3

    def test_truncation_flagged(self):
        names = ["a", "b"]
        ids = ["C0000001", "C0000002"]
        tok = _fake_tokenizer(seq_len=8)  # 8 > max_length=5
        result = gse.tokenization_audit(names, ids, tok, max_length=5)
        assert result["count_exceeding_max"] == 2
        assert "C0000001" in result["truncated_side_effect_ids"]

    def test_no_truncation_no_ids_listed(self):
        names = ["a"]
        ids = ["C0000001"]
        tok = _fake_tokenizer(seq_len=4)
        result = gse.tokenization_audit(names, ids, tok, max_length=64)
        assert result["count_exceeding_max"] == 0
        assert result["truncated_side_effect_ids"] == []


# ---------------------------------------------------------------------------
# Scenario 3: Mean pooling (same formula as Milestone 4 — verifies the
# module didn't accidentally diverge, e.g. by pooling CLS-only)
# ---------------------------------------------------------------------------

class TestMeanPool:
    def test_all_tokens_unmasked(self):
        hs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        am = torch.ones(1, 2, dtype=torch.long)
        out = gse.mean_pool(hs, am)
        assert torch.allclose(out, torch.tensor([[2.0, 3.0]]))

    def test_padding_excluded(self):
        hs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        am = torch.tensor([[1, 1, 0]])
        out = gse.mean_pool(hs, am)
        assert torch.allclose(out, torch.tensor([[2.0, 3.0]]))

    def test_fully_masked_clamp(self):
        hs = torch.ones(1, 3, 2)
        am = torch.zeros(1, 3, dtype=torch.long)
        out = gse.mean_pool(hs, am)
        assert torch.allclose(out, torch.zeros(1, 2))


# ---------------------------------------------------------------------------
# Scenario 4: Smoke test
# ---------------------------------------------------------------------------

class TestSmokeTest:
    def test_smoke_passes_with_valid_mock(self):
        tok = _fake_tokenizer(seq_len=6)
        model = _fake_model(HIDDEN_DIM)
        dim = gse.smoke_test(["a", "b", "c"], tok, model, torch.device("cpu"))
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
            gse.smoke_test(["a", "b", "c"], tok, model, torch.device("cpu"))

    def test_smoke_fails_on_frozen_violation(self):
        tok = _fake_tokenizer(seq_len=6)
        model = MagicMock()
        real_param = torch.nn.Parameter(torch.ones(1))
        model.parameters.return_value = iter([real_param])

        def _fwd(**enc):
            n, seq_len = enc["input_ids"].shape
            return SimpleNamespace(last_hidden_state=torch.ones(n, seq_len, HIDDEN_DIM))

        model.side_effect = _fwd
        with pytest.raises(RuntimeError, match="requires_grad"):
            gse.smoke_test(["a", "b", "c"], tok, model, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Scenario 5: embed_side_effects shape/dtype/batching
# ---------------------------------------------------------------------------

class TestEmbedSideEffects:
    def test_output_shape(self):
        names = [f"cond_{i}" for i in range(10)]
        tok = _fake_tokenizer(seq_len=6)
        model = _fake_model(HIDDEN_DIM)
        result = gse.embed_side_effects(names, tok, model, batch_size=4, device=torch.device("cpu"))
        assert result.shape == (10, HIDDEN_DIM)

    def test_output_dtype_float32(self):
        names = ["a", "b", "c"]
        tok = _fake_tokenizer(seq_len=6)
        model = _fake_model(HIDDEN_DIM)
        result = gse.embed_side_effects(names, tok, model, batch_size=8, device=torch.device("cpu"))
        assert result.dtype == np.float32

    def test_batch_boundary(self):
        names = [f"cond_{i}" for i in range(6)]
        tok = _fake_tokenizer(seq_len=6)
        model = _fake_model(HIDDEN_DIM)
        result = gse.embed_side_effects(names, tok, model, batch_size=6, device=torch.device("cpu"))
        assert result.shape == (6, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Scenario 6: Side-effect index schema
# ---------------------------------------------------------------------------

class TestSideEffectIndex:
    def test_schema_and_sha256(self):
        mapping = _mapping_df(3)
        names = mapping["side_effect_name"].tolist()
        tok_lengths = [6, 7, 8]
        index = gse.build_side_effect_index(mapping, tok_lengths, names, max_length=64)

        assert list(index.columns) == gse.REQUIRED_INDEX_COLUMNS
        assert len(index) == 3
        assert list(index["embedding_index"]) == [0, 1, 2]
        assert list(index["label_index"]) == [0, 1, 2]

        expected_sha = hashlib.sha256(names[0].encode("utf-8")).hexdigest()
        assert index.iloc[0]["side_effect_name_sha256"] == expected_sha

    def test_truncation_flag(self):
        mapping = _mapping_df(2)
        names = mapping["side_effect_name"].tolist()
        index = gse.build_side_effect_index(mapping, [10, 4], names, max_length=8)
        assert index.iloc[0]["was_truncated"] == True   # noqa: E712
        assert index.iloc[1]["was_truncated"] == False  # noqa: E712


# ---------------------------------------------------------------------------
# Scenario 7: Post-run disk validation
# ---------------------------------------------------------------------------

class TestDiskValidation:
    def test_valid_artifacts_pass(self, tmp_path):
        n = 5
        matrix = np.ones((n, HIDDEN_DIM), dtype=np.float32)
        mapping = _mapping_df(n)
        names = mapping["side_effect_name"].tolist()
        index_df = gse.build_side_effect_index(mapping, [5] * n, names)

        embed_path = tmp_path / "emb.npy"
        index_path = tmp_path / "idx.csv"
        np.save(embed_path, matrix)
        index_df.to_csv(index_path, index=False)

        gse.validate_outputs_from_disk(embed_path, index_path, HIDDEN_DIM, expected_side_effect_count=n)

    def test_wrong_row_count_raises(self, tmp_path):
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float32)
        mapping = _mapping_df(5)
        names = mapping["side_effect_name"].tolist()
        index_df = gse.build_side_effect_index(mapping, [5] * 5, names)

        embed_path = tmp_path / "emb.npy"
        index_path = tmp_path / "idx.csv"
        np.save(embed_path, matrix)
        index_df.to_csv(index_path, index=False)

        with pytest.raises(ValueError, match="Matrix has"):
            gse.validate_outputs_from_disk(embed_path, index_path, HIDDEN_DIM, expected_side_effect_count=5)

    def test_non_finite_raises(self, tmp_path):
        n = 5
        matrix = np.ones((n, HIDDEN_DIM), dtype=np.float32)
        matrix[2, 3] = float("nan")
        mapping = _mapping_df(n)
        names = mapping["side_effect_name"].tolist()
        index_df = gse.build_side_effect_index(mapping, [5] * n, names)

        embed_path = tmp_path / "emb.npy"
        index_path = tmp_path / "idx.csv"
        np.save(embed_path, matrix)
        index_df.to_csv(index_path, index=False)

        with pytest.raises(ValueError, match="non-finite"):
            gse.validate_outputs_from_disk(embed_path, index_path, HIDDEN_DIM, expected_side_effect_count=n)

    def test_duplicate_side_effect_id_in_index_raises(self, tmp_path):
        n = 5
        matrix = np.ones((n, HIDDEN_DIM), dtype=np.float32)
        mapping = _mapping_df(n)
        names = mapping["side_effect_name"].tolist()
        index_df = gse.build_side_effect_index(mapping, [5] * n, names)
        index_df.loc[1, "side_effect_id"] = index_df.loc[0, "side_effect_id"]

        embed_path = tmp_path / "emb.npy"
        index_path = tmp_path / "idx.csv"
        np.save(embed_path, matrix)
        index_df.to_csv(index_path, index=False)

        with pytest.raises(ValueError, match="not all unique"):
            gse.validate_outputs_from_disk(embed_path, index_path, HIDDEN_DIM, expected_side_effect_count=n)


# ---------------------------------------------------------------------------
# Scenario 8: Audit JSON schema
# ---------------------------------------------------------------------------

class TestAuditSchema:
    def test_required_fields_present(self, tmp_path):
        mapping = _mapping_df(5)
        names = mapping["side_effect_name"].tolist()
        tok_lengths = [5] * 5
        tok_audit = {
            "min_token_length": 5, "median_token_length": 5.0, "mean_token_length": 5.0,
            "max_token_length": 5, "count_exceeding_max": 0, "percent_exceeding_max": 0.0,
            "truncated_side_effect_ids": [], "per_side_effect_token_lengths": tok_lengths,
        }
        matrix = np.ones((5, HIDDEN_DIM), dtype=np.float32)
        index_df = gse.build_side_effect_index(mapping, tok_lengths, names)

        embed_path = tmp_path / "emb.npy"
        index_path = tmp_path / "idx.csv"
        np.save(embed_path, matrix)
        index_df.to_csv(index_path, index=False)

        audit = gse.build_embedding_audit(
            index_df, matrix, tok_audit,
            resolved_revision="abc123", device_str="cpu", cuda_available=False,
            batch_size=32, embed_path=embed_path, index_path=index_path,
            tokenizer_class="BertTokenizer", model_class="BertModel",
        )
        required_keys = [
            "model_name", "resolved_model_revision", "tokenizer_class", "model_class",
            "source_name_column", "maximum_token_length", "pooling_method",
            "special_tokens_included_in_pooling", "provenance", "batch_size", "device",
            "cuda_available", "total_side_effects", "embedding_dimension", "embedding_shape",
            "embedding_dtype", "finite_value_check", "all_zero_row_count",
            "token_length_summary", "truncated_side_effect_count", "truncated_side_effect_ids",
            "software_versions", "input_paths", "output_paths", "artifact_sha256",
        ]
        for k in required_keys:
            assert k in audit, f"Missing audit key: {k}"
        assert audit["model_name"] == "bert-base-uncased"

    def test_special_tokens_flag_true(self, tmp_path):
        mapping = _mapping_df(3)
        names = mapping["side_effect_name"].tolist()
        tok_audit = {"min_token_length": 5, "median_token_length": 5.0, "mean_token_length": 5.0,
                     "max_token_length": 5, "count_exceeding_max": 0, "percent_exceeding_max": 0.0,
                     "truncated_side_effect_ids": [], "per_side_effect_token_lengths": [5, 5, 5]}
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float32)
        index_df = gse.build_side_effect_index(mapping, [5, 5, 5], names)
        embed_path = tmp_path / "e.npy"; index_path = tmp_path / "i.csv"
        np.save(embed_path, matrix); index_df.to_csv(index_path, index=False)
        audit = gse.build_embedding_audit(
            index_df, matrix, tok_audit, None, "cpu", False, 32,
            embed_path, index_path, "Tok", "Mdl",
        )
        assert audit["special_tokens_included_in_pooling"] is True


# ---------------------------------------------------------------------------
# Scenario 9: Overwrite guard — skip if artifacts exist
# ---------------------------------------------------------------------------

class TestOverwriteGuard:
    def test_skips_when_artifacts_exist(self, tmp_path):
        n = 5
        mapping = _mapping_df(n)
        names = mapping["side_effect_name"].tolist()
        tok_lengths = [5] * n
        matrix = np.ones((n, HIDDEN_DIM), dtype=np.float32)
        index_df = gse.build_side_effect_index(mapping, tok_lengths, names)

        embed_path = tmp_path / "emb.npy"
        index_path = tmp_path / "idx.csv"
        np.save(embed_path, matrix)
        index_df.to_csv(index_path, index=False)

        original_fn = gse.validate_mapping_df
        calls: list[int] = []

        def _counting_validate(*a, **kw):
            calls.append(1)
            return original_fn(*a, **kw)

        gse.validate_mapping_df = _counting_validate
        try:
            result = gse.run_embedding_pipeline(
                mapping_input=Path("unused_mapping.csv"),
                embed_output=embed_path,
                index_output=index_path,
                audit_output=tmp_path / "audit.json",
                overwrite=False,
            )
        finally:
            gse.validate_mapping_df = original_fn

        assert len(calls) == 0
        assert "embedding_shape" in result


# ---------------------------------------------------------------------------
# Scenario 10: SHA-256 determinism
# ---------------------------------------------------------------------------

class TestSha256:
    def test_deterministic(self, tmp_path):
        path = tmp_path / "test.bin"
        path.write_bytes(b"hello")
        assert gse.sha256_file(path) == gse.sha256_file(path)

    def test_known_value(self, tmp_path):
        path = tmp_path / "test.bin"
        content = b"test content"
        path.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert gse.sha256_file(path) == expected


# ---------------------------------------------------------------------------
# Scenario 11: All-zero row / NaN / Inf / dtype in-memory checks
# ---------------------------------------------------------------------------

class TestInMemoryValidation:
    def test_all_zero_row_raises(self):
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float32)
        matrix[1, :] = 0.0
        with pytest.raises(ValueError, match="all-zero rows"):
            gse._validate_matrix_in_memory(matrix, HIDDEN_DIM)

    def test_nan_raises(self):
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float32)
        matrix[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            gse._validate_matrix_in_memory(matrix, HIDDEN_DIM)

    def test_float64_raises(self):
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            gse._validate_matrix_in_memory(matrix, HIDDEN_DIM)

    def test_valid_matrix_passes(self):
        matrix = np.ones((3, HIDDEN_DIM), dtype=np.float32)
        gse._validate_matrix_in_memory(matrix, HIDDEN_DIM)  # no raise


# ---------------------------------------------------------------------------
# Scenario 12: Model/config constants match the authors' confirmed recipe
# ---------------------------------------------------------------------------

class TestConstantsMatchAuthorsRecipe:
    def test_model_name(self):
        assert gse.MODEL_NAME == "bert-base-uncased"

    def test_max_token_length(self):
        assert gse.MAX_TOKEN_LENGTH == 64

    def test_name_column(self):
        assert gse.NAME_COLUMN == "side_effect_name"
