"""
Tests for Milestone 9b: recompute_ap_at_k.py.

Covers the core recompute_for_model() function against synthetic fixture
artifacts (no real model checkpoints, no dependency on the real 63,472-pair
dataset) and confirms the recompute never mutates its inputs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyllm.recompute_ap_at_k import recompute_for_model  # noqa: E402


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_fixture(tmp_path: Path, n_total_pairs: int = 20, n_labels: int = 4, n_test: int = 8):
    """Build a synthetic model_dir (test_predictions.npz + test_metrics.json)
    and a separate labels.npy, mirroring the real M6B/M7 artifact layout."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    rng = np.random.default_rng(0)
    pair_ids = np.sort(rng.choice(n_total_pairs, size=n_test, replace=False)).astype(np.int64)
    probabilities = rng.random((n_test, n_labels)).astype(np.float32)

    np.savez(
        model_dir / "test_predictions.npz",
        pair_ids=pair_ids,
        probabilities=probabilities,
        selected_threshold=np.float32(0.3),
    )
    (model_dir / "test_metrics.json").write_text(
        json.dumps({"sample_mean_ap_at_50": 0.1234, "macro_auroc": 0.5})
    )

    labels_path = tmp_path / "labels.npy"
    full_labels = (rng.random((n_total_pairs, n_labels)) > 0.5).astype(np.uint8)
    np.save(labels_path, full_labels)

    return model_dir, labels_path, pair_ids, probabilities, full_labels


class TestRecomputeForModel:
    def test_matches_direct_metrics_call(self, tmp_path):
        model_dir, labels_path, pair_ids, probs, full_labels = _make_fixture(tmp_path)
        result = recompute_for_model(model_dir, labels_path, k=5)

        from polyllm.metrics import average_precision_at_k_multi_label
        y_true = full_labels[pair_ids].astype(np.int64)
        expected = average_precision_at_k_multi_label(y_true, probs, k=5)

        assert result["mean_ap_at_k"] == pytest.approx(expected["mean_ap_at_k"])
        assert result["n_pairs_evaluated"] == len(pair_ids)
        assert result["n_labels_evaluated"] == probs.shape[1]

    def test_carries_existing_metric_for_comparison(self, tmp_path):
        model_dir, labels_path, *_ = _make_fixture(tmp_path)
        result = recompute_for_model(model_dir, labels_path, k=5)
        comp = result["comparison_to_existing_repro_metric"]
        assert comp["existing_metric_value"] == pytest.approx(0.1234)
        assert comp["existing_metric_name"] == "sample_mean_ap_at_50"

    def test_does_not_modify_existing_artifacts(self, tmp_path):
        model_dir, labels_path, *_ = _make_fixture(tmp_path)
        pred_path = model_dir / "test_predictions.npz"
        metrics_path = model_dir / "test_metrics.json"

        pred_hash_before = _sha256_bytes(pred_path)
        metrics_hash_before = _sha256_bytes(metrics_path)
        labels_hash_before = _sha256_bytes(labels_path)

        recompute_for_model(model_dir, labels_path, k=5)

        assert _sha256_bytes(pred_path) == pred_hash_before
        assert _sha256_bytes(metrics_path) == metrics_hash_before
        assert _sha256_bytes(labels_path) == labels_hash_before
        # recompute_for_model itself must not have written a sibling file --
        # that is main()'s responsibility, not the pure computation function.
        assert not (model_dir / "ap_at_k_recompute.json").exists()

    def test_shape_mismatch_raises(self, tmp_path):
        model_dir, labels_path, pair_ids, probs, full_labels = _make_fixture(
            tmp_path, n_labels=4
        )
        # Corrupt: labels.npy has a different number of label columns
        bad_labels_path = tmp_path / "bad_labels.npy"
        rng = np.random.default_rng(1)
        np.save(bad_labels_path, (rng.random((20, 3)) > 0.5).astype(np.uint8))
        with pytest.raises(ValueError, match="Shape mismatch"):
            recompute_for_model(model_dir, bad_labels_path, k=5)

    def test_records_source_artifact_hashes(self, tmp_path):
        model_dir, labels_path, *_ = _make_fixture(tmp_path)
        result = recompute_for_model(model_dir, labels_path, k=5)
        hashes = result["source_artifact_hashes"]
        assert hashes["test_predictions_npz"] == _sha256_bytes(model_dir / "test_predictions.npz")
        assert hashes["test_metrics_json"] == _sha256_bytes(model_dir / "test_metrics.json")
        assert hashes["labels_npy"] == _sha256_bytes(labels_path)

    def test_deterministic_across_repeated_calls(self, tmp_path):
        model_dir, labels_path, *_ = _make_fixture(tmp_path)
        r1 = recompute_for_model(model_dir, labels_path, k=5)
        r2 = recompute_for_model(model_dir, labels_path, k=5)
        assert r1["mean_ap_at_k"] == r2["mean_ap_at_k"]
