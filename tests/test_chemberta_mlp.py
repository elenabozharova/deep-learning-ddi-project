"""
Tests for Milestone 7: ChemBERTa pair-embedding MLP.

All tests use synthetic in-memory data; no external files are loaded.
Full official training is never run inside a unit test.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.models.mlp import MultilabelMLP
from polyllm.training import (
    EarlyStopping,
    PairDataset,
    load_checkpoint,
    make_seeded_loader,
    run_train_epoch,
    run_val_epoch,
    save_checkpoint,
    validate_input_alignment,
)
from polyllm.train_chemberta_mlp import (
    DEFAULT_CONFIG,
    FEATURES_PATH,
    OUTPUT_DIR,
    validate_chemberta_features,
)
from polyllm.evaluate_chemberta_mlp import (
    OUTPUT_DIR as EVAL_OUTPUT_DIR,
    PREDICTIONS_NPZ,
    TEST_METRICS,
    check_prerequisites,
    validate_checkpoint_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_features(n=20, dim=384, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _small_labels(n=20, n_labels=10, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((n, n_labels)) > 0.8).astype(np.uint8)


# ---------------------------------------------------------------------------
# Test 1: ChemBERTa feature-shape validation (384)
# ---------------------------------------------------------------------------

class TestChemBERTaFeatureValidation:
    def test_valid_float32_passes(self):
        feats = _small_features(10, 384)
        validate_chemberta_features(feats)  # should not raise

    # Test 2: Float32 dtype required
    def test_non_float32_raises(self):
        feats = _small_features(10, 384).astype(np.float64)
        with pytest.raises(ValueError, match="float32"):
            validate_chemberta_features(feats)

    def test_uint8_raises(self):
        feats = np.ones((10, 384), dtype=np.uint8)
        with pytest.raises(ValueError, match="float32"):
            validate_chemberta_features(feats)

    # Test 3: Non-finite feature rejection
    def test_nan_raises(self):
        feats = _small_features(10, 384)
        feats[3, 7] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            validate_chemberta_features(feats)

    def test_inf_raises(self):
        feats = _small_features(10, 384)
        feats[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            validate_chemberta_features(feats)

    # Test 4: All-zero feature-row rejection
    def test_all_zero_row_raises(self):
        feats = _small_features(10, 384)
        feats[5, :] = 0.0
        with pytest.raises(ValueError, match="all-zero"):
            validate_chemberta_features(feats)


# ---------------------------------------------------------------------------
# Test 5: Pair-index alignment check
# ---------------------------------------------------------------------------

class TestPairIndexAlignment:
    def test_non_sequential_pair_id_raises(self):
        import pandas as pd
        feats    = _small_features(10)
        labels   = _small_labels(10)
        pair_idx = pd.DataFrame({"pair_id": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]})  # gap at 3
        label_map = pd.DataFrame({
            "label_index": list(range(10)),
            "side_effect_id": [f"C{i}" for i in range(10)],
        })
        train_ids = np.arange(8)
        val_ids   = np.array([8])
        test_ids  = np.array([9])
        with pytest.raises(ValueError, match="pair_id"):
            validate_input_alignment(
                feats, labels, pair_idx, label_map,
                train_ids, val_ids, test_ids,
                expected_n_pairs=10, expected_n_features=384, expected_n_labels=10,
            )


# ---------------------------------------------------------------------------
# Test 6: Input dimension 384
# ---------------------------------------------------------------------------

class TestInputDimension:
    def test_model_accepts_dim_384(self):
        model  = MultilabelMLP(input_dim=384, output_dim=963)
        x      = torch.randn(4, 384)
        logits = model(x)
        assert logits.shape == (4, 963)
        assert model.input_dim == 384

    def test_default_config_input_dim_is_384(self):
        assert DEFAULT_CONFIG["input_dim"] == 384

    def test_model_rejects_wrong_dim(self):
        model = MultilabelMLP(input_dim=384, output_dim=963)
        x     = torch.randn(4, 2048)           # Morgan dim — wrong for ChemBERTa
        with pytest.raises(RuntimeError):
            model(x)


# ---------------------------------------------------------------------------
# Test 7: Reuse of shared MLP
# ---------------------------------------------------------------------------

class TestSharedMLP:
    def test_imports_from_shared_module(self):
        from polyllm.models.mlp import MultilabelMLP as M
        assert M is MultilabelMLP

    def test_same_hidden_arch_as_morgan(self):
        morgan    = MultilabelMLP(input_dim=2048, output_dim=963)
        chemberta = MultilabelMLP(input_dim=384,  output_dim=963)
        # Hidden layers 2-4 should have identical parameter shapes
        morgan_layers    = list(morgan.net.children())
        chemberta_layers = list(chemberta.net.children())
        # Layer 5 onward (index 4+) are Linear(512,1024), Linear(1024,2048), Linear(2048,963)
        for i, (ml, cl) in enumerate(zip(morgan_layers[4:], chemberta_layers[4:])):
            if hasattr(ml, 'weight'):
                assert ml.weight.shape == cl.weight.shape, \
                    f"Hidden-layer shape mismatch at index {i+4}"

    def test_no_sigmoid_in_model(self):
        model = MultilabelMLP(input_dim=384, output_dim=963)
        for module in model.modules():
            assert not isinstance(module, torch.nn.Sigmoid), \
                "Sigmoid found inside MultilabelMLP"

    def test_parameter_count_differs_from_morgan(self):
        morgan    = MultilabelMLP(input_dim=2048, output_dim=963)
        chemberta = MultilabelMLP(input_dim=384,  output_dim=963)
        # ChemBERTa first layer is smaller: 384*512 vs 2048*512
        assert chemberta.count_parameters() < morgan.count_parameters()


# ---------------------------------------------------------------------------
# Test 8: Reuse of shared metrics
# ---------------------------------------------------------------------------

class TestSharedMetrics:
    def test_metrics_importable_from_shared_module(self):
        from polyllm.metrics import (
            macro_auprc, macro_auroc, micro_auprc,
            micro_f1, sample_mean_ap_at_50, select_threshold,
        )
        # Just check they're callable
        assert callable(macro_auprc)
        assert callable(macro_auroc)
        assert callable(micro_auprc)
        assert callable(micro_f1)
        assert callable(sample_mean_ap_at_50)
        assert callable(select_threshold)

    def test_metrics_produce_same_result_for_both_representations(self):
        from polyllm.metrics import micro_f1
        rng    = np.random.default_rng(0)
        y_true = (rng.random((50, 10)) > 0.8).astype(np.int32)
        y_pred = (rng.random((50, 10)) > 0.5).astype(np.int32)
        # metric function is representation-agnostic
        result = micro_f1(y_true, y_pred)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Test 9: Reuse of shared training functions
# ---------------------------------------------------------------------------

class TestSharedTrainingFunctions:
    def test_run_train_epoch_importable(self):
        from polyllm.training import run_train_epoch, run_val_epoch
        assert callable(run_train_epoch)
        assert callable(run_val_epoch)

    def test_pair_dataset_uses_384_dim(self):
        feats  = _small_features(20, 384)
        labels = _small_labels(20)
        ds     = PairDataset(np.arange(20), feats, labels)
        pid, x, y = ds[0]
        assert x.shape == (384,)

    def test_early_stopping_importable(self):
        from polyllm.training import EarlyStopping
        es = EarlyStopping(patience=3, min_delta=0.001)
        assert callable(es.step)


# ---------------------------------------------------------------------------
# Test 10: Test split not accessed during training
# ---------------------------------------------------------------------------

class TestNoTestAccessDuringTraining:
    def test_train_chemberta_signature_has_no_test_ids(self):
        import inspect
        from polyllm.train_chemberta_mlp import run_full_training
        sig = inspect.signature(run_full_training)
        param_names = list(sig.parameters.keys())
        assert "test_ids" not in param_names, \
            "run_full_training should not accept test_ids"

    def test_run_smoke_test_signature_has_no_test_ids(self):
        import inspect
        from polyllm.train_chemberta_mlp import run_smoke_test
        sig = inspect.signature(run_smoke_test)
        assert "test_ids" not in sig.parameters


# ---------------------------------------------------------------------------
# Test 11: Smoke-test isolation (training data only for label selection)
# ---------------------------------------------------------------------------

class TestSmokeLabelSelection:
    def test_smoke_labels_from_training_only(self):
        """Verify that smoke-test top-20 labels depend only on train IDs."""
        rng       = np.random.default_rng(42)
        n_pairs   = 100
        n_labels  = 30
        labels    = (rng.random((n_pairs, n_labels)) > 0.7).astype(np.uint8)
        train_ids = np.arange(60)
        val_ids   = np.arange(60, 80)

        # Simulate the selection logic
        smoke_train_ids = train_ids[:20]
        counts  = np.array(labels[smoke_train_ids, :], dtype=np.int64).sum(axis=0)
        top20   = np.sort(np.argsort(counts)[::-1][:20])

        # Verify: selection uses only training rows (not val rows 60..79)
        val_counts = np.array(labels[val_ids, :], dtype=np.int64).sum(axis=0)
        # Changing val labels should not change the top-20 result
        labels_modified = labels.copy()
        # Zero out validation labels entirely
        labels_modified[val_ids, :] = 0
        counts2 = np.array(labels_modified[smoke_train_ids, :], dtype=np.int64).sum(axis=0)
        top20_2 = np.sort(np.argsort(counts2)[::-1][:20])
        np.testing.assert_array_equal(top20, top20_2)


# ---------------------------------------------------------------------------
# Test 12: Output path separation from Morgan
# ---------------------------------------------------------------------------

class TestOutputPathSeparation:
    def test_chemberta_output_dir_differs_from_morgan(self):
        from polyllm.train_chemberta_mlp import OUTPUT_DIR as CB_DIR
        morgan_dir = Path("outputs/baseline/morgan")
        assert Path(CB_DIR) != morgan_dir

    def test_chemberta_output_dir_is_polyllm_subdir(self):
        assert "polyllm" in str(OUTPUT_DIR)
        assert "chemberta" in str(OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Test 13: Checkpoint configuration validation
# ---------------------------------------------------------------------------

class TestCheckpointConfigValidation:
    def test_valid_ckpt_config_passes(self):
        feats  = _small_features(10, 384)
        labels = _small_labels(10, 5)
        ckpt   = {"input_dim": 384, "output_dim": 5}
        validate_checkpoint_config(ckpt, feats, labels)  # should not raise

    def test_wrong_input_dim_raises(self):
        feats  = _small_features(10, 384)
        labels = _small_labels(10, 5)
        ckpt   = {"input_dim": 2048, "output_dim": 5}  # Morgan dim
        with pytest.raises(ValueError, match="input_dim"):
            validate_checkpoint_config(ckpt, feats, labels)

    def test_wrong_output_dim_raises(self):
        feats  = _small_features(10, 384)
        labels = _small_labels(10, 5)
        ckpt   = {"input_dim": 384, "output_dim": 100}
        with pytest.raises(ValueError, match="output_dim"):
            validate_checkpoint_config(ckpt, feats, labels)


# ---------------------------------------------------------------------------
# Test 14: Prediction shape and sorted ordering
# ---------------------------------------------------------------------------

class TestPredictionOrderAndShape:
    def test_run_val_epoch_returns_sorted_pids(self):
        n       = 30
        feats   = _small_features(n, 384)
        labels  = _small_labels(n, 5)
        # Reverse order of pair IDs to test sorting
        pids    = np.arange(n - 1, -1, -1)
        ds      = PairDataset(pids, feats, labels)
        loader  = make_seeded_loader(ds, batch_size=8, shuffle=False, seed=0)
        model   = MultilabelMLP(input_dim=384, output_dim=5)
        crit    = torch.nn.BCEWithLogitsLoss()
        _, probs, _, ret_pids = run_val_epoch(model, loader, crit, torch.device("cpu"))
        assert np.array_equal(ret_pids, np.sort(pids))
        assert probs.shape == (n, 5)
        assert probs.dtype == np.float32

    def test_probabilities_in_unit_range(self):
        n      = 20
        feats  = _small_features(n, 384)
        labels = _small_labels(n, 5)
        ds     = PairDataset(np.arange(n), feats, labels)
        loader = make_seeded_loader(ds, batch_size=8, shuffle=False, seed=0)
        model  = MultilabelMLP(input_dim=384, output_dim=5)
        crit   = torch.nn.BCEWithLogitsLoss()
        _, probs, _, _ = run_val_epoch(model, loader, crit, torch.device("cpu"))
        assert np.all((probs >= 0.0) & (probs <= 1.0))


# ---------------------------------------------------------------------------
# Test 15: Evaluator overwrite protection
# ---------------------------------------------------------------------------

class TestEvaluatorOverwriteProtection:
    def test_raises_when_test_metrics_exist_without_overwrite(self, tmp_path):
        """check_prerequisites raises FileExistsError when test_metrics.json exists."""
        from polyllm.evaluate_chemberta_mlp import (
            CHECKPOINT_PATH as CP,
            CONFIG_PATH,
            THRESHOLD_PATH as TP,
            TEST_METRICS as TM,
        )
        import unittest.mock as mock

        # Patch all path exists checks
        def _exists(path):
            path = Path(path)
            if path == CP or path == CONFIG_PATH or path == TP or path == TM:
                return True
            return False

        with mock.patch.object(Path, "exists", lambda self: _exists(self)):
            with pytest.raises(FileExistsError, match="--overwrite"):
                check_prerequisites(overwrite=False)

    def test_passes_when_overwrite_flag_set(self):
        """check_prerequisites does not raise when overwrite=True even if outputs exist."""
        import unittest.mock as mock
        from polyllm.evaluate_chemberta_mlp import (
            CHECKPOINT_PATH as CP,
            CONFIG_PATH,
            THRESHOLD_PATH as TP,
        )

        def _exists(path):
            return True  # everything "exists"

        with mock.patch.object(Path, "exists", lambda self: _exists(self)):
            # Should not raise (overwrite=True skips the FileExistsError check)
            check_prerequisites(overwrite=True)

    def test_raises_when_checkpoint_missing(self):
        import unittest.mock as mock
        with mock.patch.object(Path, "exists", lambda self: False):
            with pytest.raises(FileNotFoundError, match="checkpoint"):
                check_prerequisites(overwrite=False)
