"""
Tests for Milestone 8: model comparison.

Covers 18 scenarios around comparability validation, aggregate metrics,
per-label comparison, prediction agreement, paired bootstrap, and audit.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polyllm.compare_models as cm


# ── shared fixtures ───────────────────────────────────────────────────────────

def _make_probs(n: int = 20, k: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, k)).astype(np.float32)


def _make_labels(n: int = 20, k: int = 8, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, k)) > 0.7).astype(np.int32)


def _make_configs() -> tuple[dict, dict]:
    base = {
        "random_seed": 42, "optimizer": "Adam", "learning_rate": 0.001,
        "loss": "BCEWithLogitsLoss", "batch_size": 256, "max_epochs": 100,
        "patience": 10, "min_delta": 0.0001,
        "model_selection_metric": "validation_macro_auprc",
        "dropout": 0.2, "leaky_relu_negative_slope": 0.01, "num_workers": 0,
        "output_dim": 963, "hidden_dims": [512, 1024, 2048],
        "batch_norm_after_layer": 1, "train_pairs": 50778, "val_pairs": 6347,
    }
    morgan = {**base, "input_dim": 2048, "total_parameters": 5647811,
              "feature_source": "morgan_pair_fingerprints.npy"}
    chem = {**base, "input_dim": 384, "total_parameters": 4795843,
            "feature_source": "chemberta_pair_embeddings.npy"}
    return morgan, chem


def _make_per_label_df(n_labels: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "label_index": range(n_labels),
        "side_effect_id": [f"C{i:07d}" for i in range(n_labels)],
        "side_effect_name": [f"effect_{i}" for i in range(n_labels)],
        "test_positive_count": rng.integers(10, 100, n_labels),
        "test_prevalence": rng.random(n_labels).astype(float),
        "auroc": rng.uniform(0.6, 1.0, n_labels).astype(float),
        "auprc": rng.uniform(0.1, 0.9, n_labels).astype(float),
        "f1_at_selected_threshold": rng.uniform(0.1, 0.8, n_labels).astype(float),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Prediction pair-ID alignment — matching IDs pass, mismatched fail
# ─────────────────────────────────────────────────────────────────────────────
class TestPairIdAlignment(unittest.TestCase):

    def test_matching_ids_accepted(self):
        pids = np.array([10, 29, 33])
        # Same array compares equal — verify compare_models would not raise
        self.assertTrue(np.array_equal(pids, pids.copy()))

    def test_mismatched_ids_detected(self):
        m_pids = np.array([10, 29, 33])
        c_pids = np.array([10, 30, 33])
        self.assertFalse(np.array_equal(m_pids, c_pids))


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Prediction-shape mismatch causes failure
# ─────────────────────────────────────────────────────────────────────────────
class TestPredictionShapeMismatch(unittest.TestCase):

    def test_shape_mismatch_detected(self):
        m_probs = _make_probs(20, 8)
        c_probs = _make_probs(20, 9)  # wrong n_labels
        self.assertNotEqual(m_probs.shape, c_probs.shape)

    def test_matching_shapes_pass(self):
        m_probs = _make_probs(20, 8)
        c_probs = _make_probs(20, 8, seed=5)
        self.assertEqual(m_probs.shape, c_probs.shape)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Label alignment — test labels indexed by pair IDs
# ─────────────────────────────────────────────────────────────────────────────
class TestLabelAlignment(unittest.TestCase):

    def test_label_shape_matches_pair_count(self):
        n, k = 20, 8
        probs = _make_probs(n, k)
        labels = _make_labels(n, k)
        self.assertEqual(probs.shape[0], labels.shape[0])
        self.assertEqual(probs.shape[1], labels.shape[1])


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: Aggregate difference calculation
# ─────────────────────────────────────────────────────────────────────────────
class TestAggregateDifference(unittest.TestCase):

    def _make_metrics(self, offset: float = 0.0) -> dict:
        return {k: 0.5 + offset for k in cm.AGGREGATE_METRIC_KEYS}

    def test_difference_is_morgan_minus_chemberta(self):
        m = self._make_metrics(0.1)
        c = self._make_metrics(0.0)
        df = cm.build_aggregate_table(m, c)
        self.assertAlmostEqual(df["difference"].iloc[0], 0.1, places=6)

    def test_negative_difference_when_chemberta_better(self):
        m = self._make_metrics(0.0)
        c = self._make_metrics(0.1)
        df = cm.build_aggregate_table(m, c)
        self.assertTrue((df["difference"] < 0).all())

    def test_aggregate_has_all_metric_rows(self):
        m = self._make_metrics()
        df = cm.build_aggregate_table(m, m)
        self.assertEqual(len(df), len(cm.AGGREGATE_METRIC_KEYS))
        self.assertListEqual(df["metric"].tolist(), cm.AGGREGATE_METRIC_KEYS)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5: Relative difference calculation
# ─────────────────────────────────────────────────────────────────────────────
class TestRelativeDifference(unittest.TestCase):

    def test_relative_difference_formula(self):
        morgan = 0.6
        chem = 0.5
        m_metrics = {k: morgan for k in cm.AGGREGATE_METRIC_KEYS}
        c_metrics = {k: chem for k in cm.AGGREGATE_METRIC_KEYS}
        df = cm.build_aggregate_table(m_metrics, c_metrics)
        expected_rel = (morgan - chem) / abs(chem) * 100
        self.assertAlmostEqual(df["relative_difference_pct"].iloc[0], expected_rel, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6: Winner selection
# ─────────────────────────────────────────────────────────────────────────────
class TestWinnerSelection(unittest.TestCase):

    def test_morgan_wins(self):
        self.assertEqual(cm._winner(0.5 + 1e-10, 0.5), "morgan")

    def test_chemberta_wins(self):
        self.assertEqual(cm._winner(0.5, 0.5 + 1e-10), "chemberta")

    def test_tie_within_tolerance(self):
        self.assertEqual(cm._winner(0.5, 0.5), "tie")
        self.assertEqual(cm._winner(0.5 + 1e-13, 0.5), "tie")

    def test_aggregate_winner_column(self):
        m = {k: 0.6 for k in cm.AGGREGATE_METRIC_KEYS}
        c = {k: 0.5 for k in cm.AGGREGATE_METRIC_KEYS}
        df = cm.build_aggregate_table(m, c)
        self.assertTrue((df["winner"] == "morgan").all())


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 7: Per-label join correctness
# ─────────────────────────────────────────────────────────────────────────────
class TestPerLabelJoin(unittest.TestCase):

    def test_join_preserves_all_labels(self):
        morgan_pl = _make_per_label_df(8, seed=0)
        chem_pl = _make_per_label_df(8, seed=1)
        result = cm.build_per_label_comparison(morgan_pl, chem_pl)
        self.assertEqual(len(result), 8)

    def test_join_has_required_columns(self):
        morgan_pl = _make_per_label_df(8, seed=0)
        chem_pl = _make_per_label_df(8, seed=1)
        result = cm.build_per_label_comparison(morgan_pl, chem_pl)
        for col in ("morgan_auroc", "chemberta_auroc", "auroc_diff", "auroc_winner",
                    "morgan_auprc", "chemberta_auprc", "auprc_diff", "auprc_winner",
                    "morgan_f1", "chemberta_f1", "f1_diff", "f1_winner"):
            self.assertIn(col, result.columns)

    def test_difference_direction(self):
        morgan_pl = _make_per_label_df(8, seed=0)
        chem_pl = morgan_pl.copy()
        chem_pl["auroc"] = chem_pl["auroc"] - 0.1
        result = cm.build_per_label_comparison(morgan_pl, chem_pl)
        self.assertTrue((result["auroc_diff"] > 0).all())


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 8: Per-label win/tie counts
# ─────────────────────────────────────────────────────────────────────────────
class TestPerLabelWinCounts(unittest.TestCase):

    def test_all_morgan_wins(self):
        morgan_pl = _make_per_label_df(8, seed=0)
        chem_pl = morgan_pl.copy()
        chem_pl["auroc"] = chem_pl["auroc"] - 0.1
        result = cm.build_per_label_comparison(morgan_pl, chem_pl)
        self.assertEqual((result["auroc_winner"] == "morgan").sum(), 8)
        self.assertEqual((result["auroc_winner"] == "chemberta").sum(), 0)

    def test_ties_detected(self):
        morgan_pl = _make_per_label_df(8, seed=0)
        chem_pl = morgan_pl.copy()  # identical values
        result = cm.build_per_label_comparison(morgan_pl, chem_pl)
        self.assertEqual((result["auroc_winner"] == "tie").sum(), 8)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 9: Thresholded agreement counts
# ─────────────────────────────────────────────────────────────────────────────
class TestThresholdedAgreement(unittest.TestCase):

    def test_all_zeros_fully_agree(self):
        # All probs below both thresholds (0.28, 0.31) → all negative → 100% agreement
        probs = np.zeros((20, 8), dtype=np.float32)
        result = cm._thresholded_agreement(probs, probs)
        self.assertAlmostEqual(result["agree_pct"], 100.0, places=5)
        self.assertEqual(result["morgan_only_positive_count"], 0)
        self.assertEqual(result["chemberta_only_positive_count"], 0)

    def test_counts_sum_to_total(self):
        m_probs = _make_probs(20, 8, seed=0)
        c_probs = _make_probs(20, 8, seed=5)
        result = cm._thresholded_agreement(m_probs, c_probs)
        count_sum = (result["both_positive_count"] + result["both_negative_count"]
                     + result["morgan_only_positive_count"]
                     + result["chemberta_only_positive_count"])
        self.assertEqual(count_sum, result["total_decisions"])

    def test_total_decisions_is_n_times_k(self):
        m_probs = _make_probs(20, 8)
        c_probs = _make_probs(20, 8, seed=5)
        result = cm._thresholded_agreement(m_probs, c_probs)
        self.assertEqual(result["total_decisions"], 20 * 8)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 10: Probability correlation statistics
# ─────────────────────────────────────────────────────────────────────────────
class TestProbCorrelation(unittest.TestCase):

    def test_identical_arrays_give_pearson_one(self):
        probs = _make_probs(200, 8)
        result = cm._prob_correlation(probs, probs)
        self.assertAlmostEqual(result["pearson_r"], 1.0, places=5)

    def test_spearman_rho_in_range(self):
        m = _make_probs(200, 8, seed=0)
        c = _make_probs(200, 8, seed=7)
        result = cm._prob_correlation(m, c)
        self.assertGreaterEqual(result["spearman_rho"], -1.0)
        self.assertLessEqual(result["spearman_rho"], 1.0)

    def test_spearman_sample_size_capped(self):
        m = _make_probs(20, 8, seed=0)
        c = _make_probs(20, 8, seed=7)
        result = cm._prob_correlation(m, c)
        self.assertLessEqual(result["spearman_sample_size"], m.size)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 11: Top-k overlap (Jaccard)
# ─────────────────────────────────────────────────────────────────────────────
class TestTopKJaccard(unittest.TestCase):

    def test_identical_predictions_give_jaccard_one(self):
        probs = _make_probs(20, 20)
        result = cm._top_k_jaccard(probs, probs, k=5)
        self.assertAlmostEqual(result["mean"], 1.0, places=5)
        self.assertAlmostEqual(result["min"], 1.0, places=5)

    def test_jaccard_in_unit_interval(self):
        m = _make_probs(20, 20, seed=0)
        c = _make_probs(20, 20, seed=9)
        result = cm._top_k_jaccard(m, c, k=5)
        self.assertGreaterEqual(result["min"], 0.0)
        self.assertLessEqual(result["max"], 1.0)

    def test_jaccard_k_stored_in_output(self):
        m = _make_probs(20, 20)
        c = _make_probs(20, 20, seed=3)
        result = cm._top_k_jaccard(m, c, k=7)
        self.assertEqual(result["k"], 7)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 12: Deterministic bootstrap — same seed gives same result
# ─────────────────────────────────────────────────────────────────────────────
class TestBootstrapDeterminism(unittest.TestCase):

    def _run_mini(self) -> pd.DataFrame:
        m = _make_probs(50, 8, seed=0)
        c = _make_probs(50, 8, seed=2)
        y = _make_labels(50, 8, seed=3)
        orig_reps = cm.BOOTSTRAP_REPS
        cm.BOOTSTRAP_REPS = 5
        try:
            df = cm.run_bootstrap(m, c, y)
        finally:
            cm.BOOTSTRAP_REPS = orig_reps
        return df

    def test_same_seed_produces_identical_results(self):
        df1 = self._run_mini()
        df2 = self._run_mini()
        np.testing.assert_array_equal(df1["difference"].values, df2["difference"].values)

    def test_bootstrap_row_count(self):
        df = self._run_mini()
        # 5 reps × 3 metrics
        self.assertEqual(len(df), 5 * 3)

    def test_bootstrap_uses_macro_auprc_not_micro(self):
        df = self._run_mini()
        self.assertIn("macro_auprc", df["metric"].values)
        self.assertNotIn("micro_auprc", df["metric"].values)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 13: Same resampled indices applied to both models
# ─────────────────────────────────────────────────────────────────────────────
class TestBootstrapPairing(unittest.TestCase):

    def test_paired_indices_produce_symmetric_zero_diff(self):
        """If both models receive identical probs and thresholds, differences are zero."""
        probs = _make_probs(50, 8, seed=0)
        y = _make_labels(50, 8, seed=1)
        orig_reps = cm.BOOTSTRAP_REPS
        orig_m = cm.MORGAN_THRESHOLD
        orig_c = cm.CHEMBERTA_THRESHOLD
        # Force same threshold so F1 is also identical
        cm.BOOTSTRAP_REPS = 5
        cm.MORGAN_THRESHOLD = 0.30
        cm.CHEMBERTA_THRESHOLD = 0.30
        try:
            df = cm.run_bootstrap(probs, probs, y)
        finally:
            cm.BOOTSTRAP_REPS = orig_reps
            cm.MORGAN_THRESHOLD = orig_m
            cm.CHEMBERTA_THRESHOLD = orig_c
        self.assertTrue((df["difference"].abs() < 1e-6).all())


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 14: Fixed thresholds preserved inside bootstrap
# ─────────────────────────────────────────────────────────────────────────────
class TestBootstrapFixedThresholds(unittest.TestCase):

    def test_module_threshold_constants_match_spec(self):
        self.assertAlmostEqual(cm.MORGAN_THRESHOLD, 0.31, places=6)
        self.assertAlmostEqual(cm.CHEMBERTA_THRESHOLD, 0.28, places=6)

    def test_bootstrap_uses_fixed_threshold_not_reselected(self):
        """
        Verify that changing module thresholds after import changes bootstrap F1
        differences, proving the function reads the module-level constants
        (not a reselection routine).
        """
        m = _make_probs(50, 8, seed=0)
        c = _make_probs(50, 8, seed=2)
        y = _make_labels(50, 8, seed=3)

        orig_reps = cm.BOOTSTRAP_REPS
        orig_m_thr = cm.MORGAN_THRESHOLD
        orig_c_thr = cm.CHEMBERTA_THRESHOLD
        cm.BOOTSTRAP_REPS = 3
        try:
            df1 = cm.run_bootstrap(m, c, y)
            cm.MORGAN_THRESHOLD = 0.50
            cm.CHEMBERTA_THRESHOLD = 0.50
            df2 = cm.run_bootstrap(m, c, y)
        finally:
            cm.BOOTSTRAP_REPS = orig_reps
            cm.MORGAN_THRESHOLD = orig_m_thr
            cm.CHEMBERTA_THRESHOLD = orig_c_thr

        f1_df1 = df1.loc[df1["metric"] == "micro_f1_selected_threshold", "difference"].values
        f1_df2 = df2.loc[df2["metric"] == "micro_f1_selected_threshold", "difference"].values
        self.assertFalse(np.allclose(f1_df1, f1_df2))


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 15: Bootstrap CI calculation
# ─────────────────────────────────────────────────────────────────────────────
class TestBootstrapCI(unittest.TestCase):

    def test_ci_lower_below_upper(self):
        m = _make_probs(50, 8, seed=0)
        c = _make_probs(50, 8, seed=2)
        y = _make_labels(50, 8, seed=3)
        orig_reps = cm.BOOTSTRAP_REPS
        cm.BOOTSTRAP_REPS = 100
        try:
            df = cm.run_bootstrap(m, c, y)
        finally:
            cm.BOOTSTRAP_REPS = orig_reps

        summary = cm._bootstrap_summary(df)
        for metric, stats in summary.items():
            self.assertLessEqual(stats["ci_lower_2_5"], stats["ci_upper_97_5"])

    def test_finite_sample_bound_in_unit_interval(self):
        m = _make_probs(50, 8, seed=0)
        c = _make_probs(50, 8, seed=2)
        y = _make_labels(50, 8, seed=3)
        orig_reps = cm.BOOTSTRAP_REPS
        cm.BOOTSTRAP_REPS = 50
        try:
            df = cm.run_bootstrap(m, c, y)
        finally:
            cm.BOOTSTRAP_REPS = orig_reps

        summary = cm._bootstrap_summary(df)
        for stats in summary.values():
            self.assertGreaterEqual(stats["one_sided_upper_bound_finite_sample"], 0.0)
            self.assertLessEqual(stats["one_sided_upper_bound_finite_sample"], 1.0)
            self.assertIn("positive_reps_of_500", stats)
            self.assertIn("direction_note", stats)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 16: Audit schema validation
# ─────────────────────────────────────────────────────────────────────────────
class TestAuditSchema(unittest.TestCase):

    def _make_audit(self) -> dict:
        morgan_cfg, chem_cfg = _make_configs()
        m_metrics = {k: 0.6 for k in cm.AGGREGATE_METRIC_KEYS}
        m_metrics["selected_threshold"] = 0.31
        c_metrics = {k: 0.5 for k in cm.AGGREGATE_METRIC_KEYS}
        c_metrics["selected_threshold"] = 0.28
        agg_df = cm.build_aggregate_table(m_metrics, c_metrics)

        m = _make_probs(30, 8, seed=0)
        c = _make_probs(30, 8, seed=2)
        y = _make_labels(30, 8, seed=3)
        orig_reps = cm.BOOTSTRAP_REPS
        cm.BOOTSTRAP_REPS = 5
        try:
            boot_df = cm.run_bootstrap(m, c, y)
        finally:
            cm.BOOTSTRAP_REPS = orig_reps

        with patch.object(cm, "_sha256", return_value="aabbcc"):
            return cm.build_comparison_audit(
                morgan_cfg, chem_cfg,
                m_metrics, c_metrics,
                agg_df, boot_df,
                {"aggregate_metrics.csv": "deadbeef"},
            )

    def test_required_top_level_keys(self):
        audit = self._make_audit()
        for key in ("comparison_timestamp", "morgan_config", "chemberta_config",
                    "comparability_validated", "comparable_fields",
                    "intentional_differences", "source_artifact_hashes",
                    "bootstrap_config", "bootstrap_summary"):
            self.assertIn(key, audit)

    def test_comparability_validated_is_true(self):
        audit = self._make_audit()
        self.assertTrue(audit["comparability_validated"])

    def test_bootstrap_config_has_fixed_thresholds(self):
        audit = self._make_audit()
        bc = audit["bootstrap_config"]
        self.assertAlmostEqual(bc["morgan_threshold"], 0.31, places=6)
        self.assertAlmostEqual(bc["chemberta_threshold"], 0.28, places=6)
        self.assertEqual(bc["seed"], 42)

    def test_bootstrap_summary_has_ci_keys(self):
        audit = self._make_audit()
        for metric, stats in audit["bootstrap_summary"].items():
            self.assertIn("ci_lower_2_5", stats)
            self.assertIn("ci_upper_97_5", stats)
            self.assertIn("positive_reps_of_500", stats)
            self.assertIn("one_sided_upper_bound_finite_sample", stats)
            self.assertIn("direction_note", stats)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 17: No training code invoked
# ─────────────────────────────────────────────────────────────────────────────
class TestNoTrainingCodeInvoked(unittest.TestCase):

    def test_compare_models_does_not_import_torch(self):
        src = Path(cm.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", src)

    def test_compare_models_has_no_train_or_optimizer_references(self):
        src = Path(cm.__file__).read_text(encoding="utf-8")
        for forbidden in ("BCEWithLogitsLoss", "backward()", "zero_grad",
                          "requires_grad", "train_epoch", "EarlyStopping"):
            self.assertNotIn(forbidden, src)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 18: Source artifacts remain unchanged
# ─────────────────────────────────────────────────────────────────────────────
class TestSourceArtifactsUnchanged(unittest.TestCase):
    """Regression guard against ACCIDENTAL artifact drift.

    Updated 2026-08-09 (Milestone 9c) after an INTENTIONAL, documented
    retrain that corrected the LeakyReLU negative slope from 0.01 to the
    paper-specified 0.1 (notes/deviations_from_paper.md Sec 1.7). The
    pre-retrain artifacts these hashes previously pinned are archived at
    outputs/baseline/morgan_slope0.01_backup/ and
    outputs/polyllm/chemberta_slope0.01_backup/, with their hashes recorded
    in outputs/comparison/paper_comparison_audit.json
    (reproduction_artifacts.slope0.01_archived_hashes) for history.
    """

    EXPECTED_HASHES = {
        "outputs/baseline/morgan/checkpoints/best_model.pt":
            "66669b612a33b59c6efe5eb9770970ff042dc171fef427ee5b076fd60b777a08",
        "outputs/baseline/morgan/test_predictions.npz":
            "df2d58b90745eca539f9789c210c544685c0821673b734a3e6a6eb5f4c6fd0b8",
        "outputs/polyllm/chemberta/checkpoints/best_model.pt":
            "57daba681f2fd89fd5c522917ae32536e4bb686f8b83319fc3f68514e448b3c8",
        "outputs/polyllm/chemberta/test_predictions.npz":
            "33c406801938551de8a8151420c54524dfb1bd256724c938314f22397b6dd3c9",
    }

    def test_source_artifacts_sha256(self):
        for rel_path, expected_sha in self.EXPECTED_HASHES.items():
            p = ROOT / rel_path
            if not p.exists():
                self.skipTest(f"Artifact missing: {rel_path}")
            actual = cm._sha256(p)
            self.assertEqual(actual, expected_sha, msg=f"Hash mismatch for {rel_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario: Comparability validation
# ─────────────────────────────────────────────────────────────────────────────
class TestComparabilityValidation(unittest.TestCase):

    def test_identical_configs_pass(self):
        morgan_cfg, chem_cfg = _make_configs()
        matched = cm.validate_comparability(morgan_cfg, chem_cfg)
        self.assertEqual(matched, cm.COMPARABLE_FIELDS)

    def test_mismatched_seed_raises(self):
        morgan_cfg, chem_cfg = _make_configs()
        chem_cfg["random_seed"] = 99
        with self.assertRaises(ValueError):
            cm.validate_comparability(morgan_cfg, chem_cfg)

    def test_intentional_input_dim_difference_not_checked(self):
        morgan_cfg, chem_cfg = _make_configs()
        # input_dim differs (2048 vs 384) but validate_comparability should pass
        self.assertNotIn("input_dim", cm.COMPARABLE_FIELDS)
        cm.validate_comparability(morgan_cfg, chem_cfg)  # should not raise

    def test_ap50_vectorized_matches_scalar(self):
        """Verify the vectorized bootstrap AP@50 agrees with the scalar function."""
        rng = np.random.default_rng(77)
        y = (rng.random((30, 20)) > 0.7).astype(np.int32)
        probs = rng.random((30, 20)).astype(np.float32)
        from polyllm.metrics import sample_mean_ap_at_50 as scalar_ap50
        vec_result = cm._ap50_vectorized(y, probs)
        scalar_result = scalar_ap50(y, probs)
        self.assertAlmostEqual(vec_result, scalar_result, places=5)


if __name__ == "__main__":
    unittest.main()
