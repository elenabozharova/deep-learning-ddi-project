"""
Tests for polyllm.features.generate_biomedical_side_effect_embeddings
(Experiment 1 SE1/SE2 pipeline). No HuggingFace downloads; tokenizer and
model are mocked, matching the Milestone 10b test convention.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

import polyllm.features.generate_biomedical_side_effect_embeddings as gbe

HIDDEN_DIM = 8


def _fake_tokenizer(seq_len: int = 6) -> MagicMock:
    tok = MagicMock()

    def _call(names, **kwargs):
        batch = names if isinstance(names, list) else [names]
        enc = {
            "input_ids": torch.ones(len(batch), seq_len, dtype=torch.long),
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
    model.requires_grad_ = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    return model


def _mapping_df(n: int = 5) -> pd.DataFrame:
    rows = [
        {
            "label_index": i,
            "side_effect_id": f"C{str(i).zfill(7)}",
            "side_effect_name": f"condition_{i}",
            "unique_pair_count": 100 + i,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_expected_keys(self):
        assert set(gbe.MODEL_REGISTRY) == {"pubmedbert", "sapbert"}

    def test_canonical_checkpoints(self):
        assert gbe.MODEL_REGISTRY["pubmedbert"]["model_name"] == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        assert gbe.MODEL_REGISTRY["sapbert"]["model_name"] == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

    def test_pooling_identical_and_stated_as_mean_for_both(self):
        for key in gbe.MODEL_REGISTRY:
            assert "mean" in gbe.MODEL_REGISTRY[key]["pooling"].lower()

    def test_unknown_model_key_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown model_key"):
            gbe.run_embedding_pipeline("not-a-model")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_basic_stats(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        diag = gbe.compute_embedding_diagnostics(matrix)
        assert diag["zero_vector_count"] == 0
        assert diag["duplicate_row_groups"] == 1  # rows 0 and 2 are identical
        assert diag["embedding_norm_mean"] == pytest.approx(1.0)

    def test_zero_vector_detected(self):
        matrix = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        diag = gbe.compute_embedding_diagnostics(matrix)
        assert diag["zero_vector_count"] == 1


# ---------------------------------------------------------------------------
# Full pipeline (mocked HF)
# ---------------------------------------------------------------------------

class TestPipeline:
    def _run(self, tmp_path, model_key, mapping_df):
        mapping_path = tmp_path / "mapping.csv"
        mapping_df.to_csv(mapping_path, index=False)

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()), \
             patch("transformers.AutoModel.from_pretrained", return_value=_fake_model()), \
             patch("huggingface_hub.model_info", side_effect=Exception("offline in tests")):
            return gbe.run_embedding_pipeline(
                model_key,
                mapping_input=mapping_path,
                features_dir=tmp_path / "features",
                audit_dir=tmp_path / "audit",
            )

    def test_pubmedbert_pipeline_runs(self, tmp_path):
        mapping_df = _mapping_df(5)
        audit = self._run(tmp_path, "pubmedbert", mapping_df)
        assert audit["embedding_dimension"] == HIDDEN_DIM
        assert audit["total_side_effects"] == 5
        matrix = np.load(tmp_path / "features" / "pubmedbert_side_effect_embeddings.npy")
        assert matrix.shape == (5, HIDDEN_DIM)
        assert np.isfinite(matrix).all()

    def test_sapbert_pipeline_runs(self, tmp_path):
        mapping_df = _mapping_df(5)
        audit = self._run(tmp_path, "sapbert", mapping_df)
        assert audit["model_name"] == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        assert "mean" in audit["pooling_method"].lower()

    def test_label_index_ordering_preserved(self, tmp_path):
        mapping_df = _mapping_df(6)
        self._run(tmp_path, "pubmedbert", mapping_df)
        index_df = pd.read_csv(tmp_path / "features" / "pubmedbert_side_effect_index.csv")
        assert (index_df["label_index"].to_numpy() == mapping_df["label_index"].to_numpy()).all()
        assert (index_df["embedding_index"].to_numpy() == np.arange(6)).all()

    def test_all_names_processed(self, tmp_path):
        mapping_df = _mapping_df(7)
        audit = self._run(tmp_path, "pubmedbert", mapping_df)
        assert audit["total_side_effects"] == 7

    def test_diagnostics_included_in_audit(self, tmp_path):
        mapping_df = _mapping_df(5)
        audit = self._run(tmp_path, "pubmedbert", mapping_df)
        assert "diagnostics" in audit
        assert "embedding_norm_mean" in audit["diagnostics"]

    def test_rerun_reuses_existing_artifacts(self, tmp_path):
        mapping_df = _mapping_df(5)
        self._run(tmp_path, "pubmedbert", mapping_df)
        mapping_path = tmp_path / "mapping.csv"
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
            result = gbe.run_embedding_pipeline(
                "pubmedbert", mapping_input=mapping_path,
                features_dir=tmp_path / "features", audit_dir=tmp_path / "audit",
            )
            mock_tok.assert_not_called()
        assert result["note"] == "existing artifacts reused"

    def test_no_source_model_change_across_both(self, tmp_path):
        """SE1 and SE2 must differ only by model_name/pooling metadata, not by pipeline structure."""
        mapping_df = _mapping_df(5)
        audit1 = self._run(tmp_path, "pubmedbert", mapping_df)
        audit2 = self._run(tmp_path, "sapbert", mapping_df)
        assert audit1["source_name_column"] == audit2["source_name_column"]
        assert audit1["maximum_token_length"] == audit2["maximum_token_length"]
        assert audit1["batch_size"] == audit2["batch_size"]
