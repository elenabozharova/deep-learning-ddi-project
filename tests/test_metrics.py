"""Tests for multi-label metrics (Milestone 6B)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polyllm.metrics import (
    apply_threshold,
    macro_auprc,
    macro_auroc,
    macro_f1,
    micro_auprc,
    micro_f1,
    per_label_auprc,
    per_label_auroc,
    per_label_f1,
    sample_mean_ap_at_50,
    select_threshold,
)


def _ones_zeros(n_samples, n_labels, seed=0):
    """Simple reproducible binary array."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_samples, n_labels)).astype(np.float32)


def _scores(n_samples, n_labels, seed=1):
    rng = np.random.default_rng(seed)
    return rng.random(size=(n_samples, n_labels)).astype(np.float32)


# ---------------------------------------------------------------------------
# Scenario 17 — Global threshold-grid selection
# ---------------------------------------------------------------------------

class TestThresholdSelection:
    def test_selected_in_grid(self):
        y_true  = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        y_score = np.array([[0.8, 0.2], [0.3, 0.9], [0.7, 0.6]], dtype=np.float32)
        result  = select_threshold(y_true, y_score)
        assert 0.05 <= result["selected_threshold"] <= 0.95

    def test_grid_has_91_entries(self):
        y_true  = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_score = np.array([[0.8, 0.2], [0.3, 0.9]], dtype=np.float32)
        result  = select_threshold(y_true, y_score)
        assert len(result["threshold_grid"]) == 91

    def test_validation_micro_f1_matches_selected(self):
        y_true  = _ones_zeros(50, 10)
        y_score = _scores(50, 10)
        result  = select_threshold(y_true, y_score)
        thr = result["selected_threshold"]
        y_pred = apply_threshold(y_score, thr)
        from sklearn.metrics import f1_score
        expected_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        assert abs(result["validation_micro_f1"] - expected_f1) < 1e-6


# ---------------------------------------------------------------------------
# Scenario 18 — Deterministic tie-breaking (smallest threshold wins)
# ---------------------------------------------------------------------------

class TestThresholdTieBreaking:
    def test_tie_break_selects_smallest(self):
        # All-zeros true labels → all thresholds give micro-F1 = 0 → tie
        y_true  = np.zeros((10, 5), dtype=np.float32)
        y_score = _scores(10, 5)
        result  = select_threshold(y_true, y_score)
        assert result["selected_threshold"] == pytest.approx(0.05)

    def test_tie_break_rule_documented(self):
        y_true  = np.zeros((5, 3), dtype=np.float32)
        y_score = _scores(5, 3)
        result  = select_threshold(y_true, y_score)
        assert "tie_break_rule" in result
        assert "smallest" in result["tie_break_rule"].lower()


# ---------------------------------------------------------------------------
# Scenario 19 — Micro and macro F1
# ---------------------------------------------------------------------------

class TestF1Metrics:
    def test_micro_f1_perfect(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        y_pred = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        assert micro_f1(y_true, y_pred) == pytest.approx(1.0)

    def test_micro_f1_zero(self):
        y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_pred = np.zeros((2, 2), dtype=np.int32)
        assert micro_f1(y_true, y_pred) == pytest.approx(0.0)

    def test_macro_f1_returns_dict(self):
        y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_pred = np.array([[1, 0], [0, 1]], dtype=np.int32)
        res = macro_f1(y_true, y_pred)
        assert "macro_f1" in res
        assert "included_labels" in res
        assert "skipped_labels" in res

    def test_macro_f1_perfect(self):
        y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_pred = np.array([[1, 0], [0, 1]], dtype=np.int32)
        res = macro_f1(y_true, y_pred)
        assert res["macro_f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scenario 20 — Micro and macro AUPRC
# ---------------------------------------------------------------------------

class TestAUPRCMetrics:
    def test_micro_auprc_range(self):
        y_true  = _ones_zeros(100, 10)
        y_score = _scores(100, 10)
        val = micro_auprc(y_true, y_score)
        assert 0.0 <= val <= 1.0

    def test_macro_auprc_returns_dict(self):
        y_true  = _ones_zeros(50, 5)
        y_score = _scores(50, 5)
        res = macro_auprc(y_true, y_score)
        assert "macro_auprc" in res
        assert "included_labels" in res
        assert "skipped_labels" in res

    def test_macro_auprc_perfect_predictor(self):
        # If score == true, AUPRC should be 1.0 per label
        y_true  = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        y_score = y_true.copy()
        res = macro_auprc(y_true, y_score)
        assert res["macro_auprc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scenario 21 — Macro AUROC
# ---------------------------------------------------------------------------

class TestMacroAUROC:
    def test_macro_auroc_range(self):
        y_true  = _ones_zeros(100, 10)
        y_score = _scores(100, 10)
        res = macro_auroc(y_true, y_score)
        assert 0.0 <= res["macro_auroc"] <= 1.0

    def test_macro_auroc_returns_dict(self):
        y_true  = _ones_zeros(50, 5)
        y_score = _scores(50, 5)
        res = macro_auroc(y_true, y_score)
        assert "macro_auroc" in res
        assert "included_labels" in res
        assert "skipped_labels" in res

    def test_macro_auroc_perfect(self):
        y_true  = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_score = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        res = macro_auroc(y_true, y_score)
        assert res["macro_auroc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scenario 22 — Undefined-label metric handling
# ---------------------------------------------------------------------------

class TestUndefinedLabelHandling:
    def test_all_negative_label_skipped_from_auroc(self):
        # Label 0: all negative → AUROC undefined, skip
        # Label 1: mixed → defined
        y_true  = np.array([[0, 1], [0, 0], [0, 1]], dtype=np.float32)
        y_score = np.array([[0.3, 0.8], [0.2, 0.1], [0.7, 0.9]], dtype=np.float32)
        res = macro_auroc(y_true, y_score)
        assert res["skipped_labels"] == 1
        assert res["included_labels"] == 1

    def test_all_negative_label_skipped_from_auprc(self):
        y_true  = np.array([[0, 1], [0, 0], [0, 1]], dtype=np.float32)
        y_score = np.array([[0.3, 0.8], [0.2, 0.1], [0.7, 0.9]], dtype=np.float32)
        res = macro_auprc(y_true, y_score)
        assert res["skipped_labels"] == 1

    def test_per_label_auroc_nan_for_all_negative(self):
        y_true  = np.array([[0, 1], [0, 0]], dtype=np.float32)
        y_score = np.array([[0.3, 0.8], [0.2, 0.1]], dtype=np.float32)
        per = per_label_auroc(y_true, y_score)
        assert np.isnan(per[0])
        assert not np.isnan(per[1])

    def test_f1_nan_only_when_no_activity(self):
        # Label 0: no true positives AND never predicted → undefined
        # Label 1: true positives exist → defined (even if F1 = 0)
        y_true = np.array([[0, 1], [0, 0]], dtype=np.float32)
        y_pred = np.array([[0, 0], [0, 0]], dtype=np.int32)
        per = per_label_f1(y_true, y_pred)
        assert np.isnan(per[0])   # no activity at all
        assert not np.isnan(per[1])  # has true positive; F1=0 (predicted=0)

    def test_macro_does_not_substitute_zero_for_nan(self):
        # Only label 0 has positives; label 1 is all-negative
        y_true  = np.array([[1, 0], [0, 0]], dtype=np.float32)
        y_score = np.array([[0.9, 0.5], [0.2, 0.3]], dtype=np.float32)
        per = per_label_auprc(y_true, y_score)
        assert not np.isnan(per[0])
        assert np.isnan(per[1])  # not substituted with 0


# ---------------------------------------------------------------------------
# Scenario 23 — AP@50 manual examples
# ---------------------------------------------------------------------------

class TestAP50:
    def test_manual_case_two_true_labels(self):
        # 1 pair, 5 labels
        # true=[1,0,1,0,0], scores=[0.9,0.8,0.7,0.6,0.5]
        # Ranked by score desc: [0,1,2,3,4]
        # top_true = [1,0,1,0,0]
        # prec@k = [1, 0.5, 2/3, 2/4, 2/5]
        # prec@k * top_true = [1, 0, 2/3, 0, 0]  → sum = 5/3
        # denom = min(2,50)=2
        # AP = (5/3)/2 = 5/6 ≈ 0.8333
        y_true  = np.array([[1, 0, 1, 0, 0]], dtype=np.float32)
        y_score = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
        val = sample_mean_ap_at_50(y_true, y_score)
        assert val == pytest.approx(5 / 6, abs=1e-5)

    def test_zero_true_labels_contributes_zero(self):
        y_true  = np.array([[0, 0, 0]], dtype=np.float32)
        y_score = np.array([[0.9, 0.5, 0.1]], dtype=np.float32)
        val = sample_mean_ap_at_50(y_true, y_score)
        assert val == pytest.approx(0.0)

    def test_all_true_labels_in_top_positions(self):
        # 1 pair, 3 labels all true; scores put them all at top
        # true=[1,1,1], scores=[0.9,0.8,0.7]
        # Ranked: [0,1,2], top_true=[1,1,1]
        # prec@k=[1,1,1], sum=3, denom=min(3,50)=3 → AP=1.0
        y_true  = np.array([[1, 1, 1]], dtype=np.float32)
        y_score = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
        val = sample_mean_ap_at_50(y_true, y_score)
        assert val == pytest.approx(1.0)

    def test_cap_at_50_labels(self):
        # 1 pair, 100 labels; 60 true; all at top
        # top 50 should all be true → AP = 1.0 (50 true in top-50, denom=50)
        n = 100
        y_true  = np.zeros((1, n), dtype=np.float32)
        y_true[0, :60] = 1.0  # first 60 are true
        y_score = np.linspace(1.0, 0.01, n).reshape(1, -1).astype(np.float32)
        val = sample_mean_ap_at_50(y_true, y_score)
        assert val == pytest.approx(1.0)

    def test_average_over_multiple_pairs(self):
        # Pair 1: AP = 5/6; Pair 2: AP = 0.0 → mean = 5/12
        y_true  = np.array([[1, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
        y_score = np.array([[0.9, 0.8, 0.7, 0.6, 0.5],
                            [0.9, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
        val = sample_mean_ap_at_50(y_true, y_score)
        assert val == pytest.approx(5 / 12, abs=1e-5)


# ---------------------------------------------------------------------------
# Scenario 24 — Per-label metric alignment
# ---------------------------------------------------------------------------

class TestPerLabelAlignment:
    def test_per_label_auroc_length(self):
        y_true  = _ones_zeros(50, 7)
        y_score = _scores(50, 7)
        per = per_label_auroc(y_true, y_score)
        assert per.shape == (7,)

    def test_per_label_auprc_length(self):
        y_true  = _ones_zeros(50, 7)
        y_score = _scores(50, 7)
        per = per_label_auprc(y_true, y_score)
        assert per.shape == (7,)

    def test_per_label_f1_length(self):
        y_true = _ones_zeros(50, 7)
        y_pred = (y_true > 0.5).astype(np.int32)
        per = per_label_f1(y_true, y_pred)
        assert per.shape == (7,)

    def test_apply_threshold(self):
        y_score = np.array([0.3, 0.6, 0.5, 0.1], dtype=np.float32)
        pred = apply_threshold(y_score, 0.5)
        expected = np.array([0, 1, 1, 0], dtype=np.int32)
        np.testing.assert_array_equal(pred, expected)
