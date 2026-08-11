"""
Milestone 10f tests — evaluate_gnn.py
Uses small synthetic hetero graphs; no real data files or checkpoint loaded.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import polyllm.evaluate_gnn as eg
import polyllm.graph.build_graph as bg
import polyllm.models.gnn as gm
import polyllm.train_gnn as tg


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _synthetic_full_test(n_pdrugs=100, n_seffect=15, pdrugs_dim=8, density=0.3, seed=0):
    rng = np.random.default_rng(seed)
    pdrugs_x = rng.normal(size=(n_pdrugs, pdrugs_dim)).astype(np.float32)
    seffect_x = rng.normal(size=(n_seffect, 4)).astype(np.float32)
    labels = (rng.random((n_pdrugs, n_seffect)) < density).astype(np.uint8)
    labels[0, 0] = 1
    full = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
    train, val, test = bg.split_graph(full, seed=42)
    return full, test


def _tiny_config() -> dict:
    cfg = dict(tg.DEFAULT_CONFIG)
    cfg["hidden_channels"] = 8
    return cfg


# ---------------------------------------------------------------------------
# Scenario 1: compute_test_metrics — matches sklearn/our metrics module directly
# ---------------------------------------------------------------------------

class TestComputeTestMetrics:
    def test_matches_sklearn_auc_and_auprc(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=500).astype(np.float64)
        y_pred = rng.random(size=500)

        result = eg.compute_test_metrics(y_true, y_pred)

        assert result["auc"] == pytest.approx(roc_auc_score(y_true, y_pred))
        assert result["auprc"] == pytest.approx(average_precision_score(y_true, y_pred))

    def test_returns_all_three_keys(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2])
        result = eg.compute_test_metrics(y_true, y_pred)
        assert set(result.keys()) == {"auc", "auprc", "ap_at_50"}

    def test_perfect_predictions_give_auc_one(self):
        y_true = np.array([1.0, 1.0, 0.0, 0.0])
        y_pred = np.array([1.0, 0.9, 0.2, 0.1])
        result = eg.compute_test_metrics(y_true, y_pred)
        assert result["auc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scenario 2: compare_to_paper — std-dev arithmetic
# ---------------------------------------------------------------------------

class TestCompareToPaper:
    def test_std_dev_math(self):
        metrics = {"auc": 0.9228 + 2 * 0.0039, "auprc": 0.8944, "ap_at_50": 0.9599}
        comparison = eg.compare_to_paper(metrics)
        assert comparison["auc"]["std_devs_from_paper"] == pytest.approx(2.0)
        assert comparison["auprc"]["std_devs_from_paper"] == pytest.approx(0.0)

    def test_exact_match_gives_zero_std_devs(self):
        metrics = {k: v["mean"] for k, v in eg.PAPER_TARGET.items()}
        comparison = eg.compare_to_paper(metrics)
        for key in metrics:
            assert comparison[key]["std_devs_from_paper"] == pytest.approx(0.0)

    def test_paper_target_matches_table5_deepchem_row(self):
        assert eg.PAPER_TARGET["auc"]["mean"] == 0.9228
        assert eg.PAPER_TARGET["auprc"]["mean"] == 0.8944
        assert eg.PAPER_TARGET["ap_at_50"]["mean"] == 0.9599


# ---------------------------------------------------------------------------
# Scenario 3: collect_test_predictions — end-to-end on a synthetic graph,
# using an in-memory single-batch "loader" (same pattern as test_train_gnn.py)
# ---------------------------------------------------------------------------

class TestCollectTestPredictions:
    def test_output_shapes_and_label_values(self):
        full, test = _synthetic_full_test()
        config = _tiny_config()
        device = torch.device("cpu")
        model = tg.build_model(full, config, device)
        global_pos_edges = gm.build_global_positive_edge_set(full)

        y_true, y_pred = eg.collect_test_predictions(model, [test], global_pos_edges, device)

        assert y_true.shape == y_pred.shape
        assert set(np.unique(y_true).tolist()).issubset({0.0, 1.0})
        assert np.isfinite(y_pred).all()

    def test_positive_count_matches_test_split_supervision_edges(self):
        full, test = _synthetic_full_test()
        config = _tiny_config()
        device = torch.device("cpu")
        model = tg.build_model(full, config, device)
        global_pos_edges = gm.build_global_positive_edge_set(full)

        y_true, y_pred = eg.collect_test_predictions(model, [test], global_pos_edges, device)

        expected_positive = test[gm.EDGE_TYPE].edge_label_index.size(1)
        assert int(y_true.sum()) == expected_positive

    def test_does_not_update_model_parameters(self):
        full, test = _synthetic_full_test()
        config = _tiny_config()
        device = torch.device("cpu")
        model = tg.build_model(full, config, device)
        global_pos_edges = gm.build_global_positive_edge_set(full)

        param_before = next(model.parameters()).data.clone()
        eg.collect_test_predictions(model, [test], global_pos_edges, device)
        assert torch.equal(param_before, next(model.parameters()).data)
