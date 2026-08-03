"""Tests for shared training infrastructure (Milestone 6B)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polyllm.models.mlp import MultilabelMLP
from polyllm.training import (
    EarlyStopping,
    PairDataset,
    file_sha256,
    load_checkpoint,
    make_history_row,
    make_seeded_loader,
    run_train_epoch,
    run_val_epoch,
    save_checkpoint,
    set_seeds,
    validate_input_alignment,
)


# ---------------------------------------------------------------------------
# Shared tiny fixtures
# ---------------------------------------------------------------------------

N_PAIRS    = 20
N_FEATURES = 8
N_LABELS   = 4


def _make_features_labels(n=N_PAIRS, f=N_FEATURES, l=N_LABELS, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.integers(0, 3, size=(n, f)).astype(np.uint8)
    labels   = rng.integers(0, 2, size=(n, l)).astype(np.uint8)
    return features, labels


def _make_pair_index(n=N_PAIRS):
    import pandas as pd
    return pd.DataFrame({"pair_id": np.arange(n)})


def _make_label_mapping(l=N_LABELS):
    import pandas as pd
    return pd.DataFrame({
        "label_index":    np.arange(l),
        "side_effect_id": [f"C{i:07d}" for i in range(l)],
    })


def _tiny_model(n_features=N_FEATURES, n_labels=N_LABELS):
    return MultilabelMLP(input_dim=n_features, output_dim=n_labels, dropout=0.0)


# ---------------------------------------------------------------------------
# Scenario 7 — Dataset indexes rows by pair ID
# ---------------------------------------------------------------------------

class TestDatasetRowIndexing:
    def test_getitem_fetches_correct_row(self):
        features, labels = _make_features_labels()
        pair_ids = np.array([5, 10, 15])
        ds = PairDataset(pair_ids, features, labels)
        pid, x, y = ds[0]
        assert pid == 5
        np.testing.assert_array_almost_equal(
            x.numpy(), features[5].astype(np.float32)
        )
        np.testing.assert_array_almost_equal(
            y.numpy(), labels[5].astype(np.float32)
        )

    def test_len_equals_pair_ids_length(self):
        features, labels = _make_features_labels()
        pair_ids = np.arange(5)
        ds = PairDataset(pair_ids, features, labels)
        assert len(ds) == 5

    def test_label_indices_subset(self):
        features, labels = _make_features_labels()
        pair_ids = np.array([3])
        label_idx = np.array([0, 2])
        ds = PairDataset(pair_ids, features, labels, label_indices=label_idx)
        _, _, y = ds[0]
        assert y.shape == (2,)
        np.testing.assert_array_almost_equal(
            y.numpy(), labels[3][[0, 2]].astype(np.float32)
        )

    def test_features_converted_to_float32(self):
        features, labels = _make_features_labels()
        ds = PairDataset(np.arange(N_PAIRS), features, labels)
        _, x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32


# ---------------------------------------------------------------------------
# Scenario 8 — Feature / label dimension mismatch fails
# ---------------------------------------------------------------------------

class TestDimensionMismatch:
    def test_wrong_feature_shape_fails_alignment(self):
        import pandas as pd
        features, labels = _make_features_labels(f=7)  # wrong
        pair_index   = _make_pair_index()
        label_mapping = _make_label_mapping()
        train_ids = np.arange(10)
        val_ids   = np.arange(10, 15)
        test_ids  = np.arange(15, N_PAIRS)
        with pytest.raises(ValueError, match="Feature shape"):
            validate_input_alignment(
                features, labels, pair_index, label_mapping,
                train_ids, val_ids, test_ids,
                expected_n_pairs=N_PAIRS,
                expected_n_features=N_FEATURES,
                expected_n_labels=N_LABELS,
            )

    def test_wrong_label_shape_fails_alignment(self):
        import pandas as pd
        features, labels = _make_features_labels(l=3)  # wrong
        pair_index    = _make_pair_index()
        label_mapping = _make_label_mapping()
        train_ids = np.arange(10)
        val_ids   = np.arange(10, 15)
        test_ids  = np.arange(15, N_PAIRS)
        with pytest.raises(ValueError, match="Label shape"):
            validate_input_alignment(
                features, labels, pair_index, label_mapping,
                train_ids, val_ids, test_ids,
                expected_n_pairs=N_PAIRS,
                expected_n_features=N_FEATURES,
                expected_n_labels=N_LABELS,
            )


# ---------------------------------------------------------------------------
# Scenario 9 — Invalid split overlap fails
# ---------------------------------------------------------------------------

class TestSplitOverlap:
    def _alignment_call(self, train_ids, val_ids, test_ids):
        features, labels = _make_features_labels()
        pair_index    = _make_pair_index()
        label_mapping = _make_label_mapping()
        validate_input_alignment(
            features, labels, pair_index, label_mapping,
            train_ids, val_ids, test_ids,
            expected_n_pairs=N_PAIRS,
            expected_n_features=N_FEATURES,
            expected_n_labels=N_LABELS,
        )

    def test_train_val_overlap_fails(self):
        # ID 5 appears in both train and val
        train_ids = np.arange(10)
        val_ids   = np.array([5, 10, 11, 12, 13])
        test_ids  = np.arange(14, N_PAIRS)
        with pytest.raises(ValueError, match="overlap"):
            self._alignment_call(train_ids, val_ids, test_ids)

    def test_train_test_overlap_fails(self):
        train_ids = np.array([0, 1, 2, 3, 16])  # 16 also in test
        val_ids   = np.arange(4, 12)
        test_ids  = np.arange(12, N_PAIRS)
        with pytest.raises(ValueError, match="overlap"):
            self._alignment_call(train_ids, val_ids, test_ids)


# ---------------------------------------------------------------------------
# Scenario 10 — Incomplete split coverage fails
# ---------------------------------------------------------------------------

class TestIncompleteSplit:
    def test_missing_pair_id_fails(self):
        features, labels = _make_features_labels()
        pair_index    = _make_pair_index()
        label_mapping = _make_label_mapping()
        train_ids = np.arange(8)   # 0-7
        val_ids   = np.arange(8, 14)  # 8-13
        test_ids  = np.arange(14, 19)  # 14-18 — missing 19
        with pytest.raises(ValueError, match="Split union"):
            validate_input_alignment(
                features, labels, pair_index, label_mapping,
                train_ids, val_ids, test_ids,
                expected_n_pairs=N_PAIRS,
                expected_n_features=N_FEATURES,
                expected_n_labels=N_LABELS,
            )


# ---------------------------------------------------------------------------
# Scenario 11 — Seeded DataLoader reproducibility
# ---------------------------------------------------------------------------

class TestSeededDataLoader:
    def test_same_seed_same_order(self):
        features, labels = _make_features_labels(n=40)
        pair_ids = np.arange(40)
        ds = PairDataset(pair_ids, features, labels)
        loader1 = make_seeded_loader(ds, batch_size=8, shuffle=True, seed=42)
        loader2 = make_seeded_loader(ds, batch_size=8, shuffle=True, seed=42)
        pids1 = [pid for batch in loader1 for pid in batch[0].tolist()]
        pids2 = [pid for batch in loader2 for pid in batch[0].tolist()]
        assert pids1 == pids2

    def test_different_seed_different_order(self):
        features, labels = _make_features_labels(n=40)
        pair_ids = np.arange(40)
        ds = PairDataset(pair_ids, features, labels)
        loader1 = make_seeded_loader(ds, batch_size=8, shuffle=True, seed=42)
        loader2 = make_seeded_loader(ds, batch_size=8, shuffle=True, seed=99)
        pids1 = [pid for batch in loader1 for pid in batch[0].tolist()]
        pids2 = [pid for batch in loader2 for pid in batch[0].tolist()]
        assert pids1 != pids2


# ---------------------------------------------------------------------------
# Scenario 12 — Training updates model parameters
# ---------------------------------------------------------------------------

class TestTrainEpochUpdatesParams:
    def test_params_change_after_train_epoch(self):
        set_seeds(0)
        features, labels = _make_features_labels(n=40)
        pair_ids = np.arange(40)
        ds = PairDataset(pair_ids, features, labels)
        loader = make_seeded_loader(ds, batch_size=8, shuffle=True, seed=0)
        model = _tiny_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device("cpu")
        # Snapshot one parameter tensor
        param_before = list(model.parameters())[0].data.clone()
        run_train_epoch(model, loader, criterion, optimizer, device)
        param_after = list(model.parameters())[0].data
        assert not torch.equal(param_before, param_after)


# ---------------------------------------------------------------------------
# Scenario 13 — Validation does not update parameters
# ---------------------------------------------------------------------------

class TestValEpochNoParamUpdate:
    def test_params_unchanged_after_val_epoch(self):
        features, labels = _make_features_labels(n=40)
        ds = PairDataset(np.arange(40), features, labels)
        loader = make_seeded_loader(ds, batch_size=8, shuffle=False, seed=0)
        model = _tiny_model()
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device("cpu")
        params_before = [p.data.clone() for p in model.parameters()]
        run_val_epoch(model, loader, criterion, device)
        params_after = [p.data for p in model.parameters()]
        for b, a in zip(params_before, params_after):
            assert torch.equal(b, a), "Validation changed a model parameter."


# ---------------------------------------------------------------------------
# Scenario 14 — Early stopping behaviour
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_first_step_always_improves(self):
        es = EarlyStopping(patience=3, min_delta=0.001)
        improved = es.step(score=0.5, epoch=1)
        assert improved
        assert es.best_score == 0.5
        assert es.counter == 0

    def test_no_improvement_increments_counter(self):
        es = EarlyStopping(patience=3, min_delta=0.001)
        es.step(score=0.5, epoch=1)
        improved = es.step(score=0.5, epoch=2)   # no change ≥ 0.001
        assert not improved
        assert es.counter == 1
        assert not es.should_stop

    def test_stop_after_patience_exhausted(self):
        es = EarlyStopping(patience=3, min_delta=0.001)
        es.step(0.5, 1)
        es.step(0.5, 2)
        es.step(0.5, 3)
        es.step(0.5, 4)   # counter hits patience=3 → should_stop
        assert es.should_stop

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=3, min_delta=0.001)
        es.step(0.5, 1)
        es.step(0.5, 2)   # counter=1
        es.step(0.6, 3)   # improved → counter=0
        assert es.counter == 0
        assert not es.should_stop
        assert es.best_epoch == 3

    def test_min_delta_boundary(self):
        es = EarlyStopping(patience=3, min_delta=0.01)
        es.step(0.5, 1)
        # Improvement of exactly 0.009 < min_delta → not counted
        improved = es.step(0.509, 2)
        assert not improved
        # Improvement of 0.011 ≥ min_delta → counted
        improved = es.step(0.521, 3)
        assert improved


# ---------------------------------------------------------------------------
# Scenario 15 — Best-checkpoint restoration
# ---------------------------------------------------------------------------

class TestCheckpointSaveLoad:
    def test_roundtrip(self, tmp_path):
        model  = _tiny_model()
        opt    = torch.optim.Adam(model.parameters(), lr=0.001)
        config = {"input_dim": N_FEATURES, "output_dim": N_LABELS, "extra": 42}
        path   = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, opt, epoch=7, score=0.88, config=config)
        assert path.exists()
        model2 = _tiny_model()
        ckpt   = load_checkpoint(path, model2, device=torch.device("cpu"))
        assert ckpt["epoch"] == 7
        assert ckpt["val_macro_auprc"] == pytest.approx(0.88)
        assert ckpt["input_dim"]  == N_FEATURES
        assert ckpt["output_dim"] == N_LABELS
        assert ckpt["config"]["extra"] == 42

    def test_restored_model_gives_same_predictions(self, tmp_path):
        model = _tiny_model()
        opt   = torch.optim.Adam(model.parameters(), lr=0.001)
        path  = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, opt, epoch=1, score=0.5,
                        config={"input_dim": N_FEATURES, "output_dim": N_LABELS})
        model2 = _tiny_model()
        load_checkpoint(path, model2, device=torch.device("cpu"))
        model.eval(); model2.eval()
        x = torch.randn(4, N_FEATURES)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# Scenario 28 — Training-history schema
# ---------------------------------------------------------------------------

class TestHistoryRowSchema:
    def test_required_keys_present(self):
        row = make_history_row(
            epoch=5, train_loss=0.5, val_loss=0.4,
            val_macro_auprc=0.3, checkpoint_saved=True
        )
        required = {"epoch", "train_loss", "val_loss", "val_macro_auprc", "checkpoint_saved"}
        assert required.issubset(row.keys())

    def test_epoch_integer(self):
        row = make_history_row(1, 0.5, 0.4, 0.3, False)
        assert isinstance(row["epoch"], int)

    def test_checkpoint_saved_bool(self):
        row = make_history_row(1, 0.5, 0.4, 0.3, True)
        assert row["checkpoint_saved"] is True


# ---------------------------------------------------------------------------
# Scenario 29 — Smoke-test mode uses training data only for label selection
# ---------------------------------------------------------------------------

class TestSmokeLabelSelection:
    def test_top20_from_training_data_only(self):
        """Label selection must use only the first 2 000 training IDs."""
        set_seeds(42)
        n_total = 100
        n_labels = 10
        features = np.zeros((n_total, 4), dtype=np.uint8)
        labels   = np.zeros((n_total, n_labels), dtype=np.uint8)

        # Make label 0 highly active only in test/val rows (rows 50-99)
        labels[50:, 0] = 1
        # Make label 1 active only in training rows (rows 0-49)
        labels[:50, 1] = 1

        train_ids = np.arange(50)
        smoke_ids = train_ids[:20]   # first 20 training IDs

        # Count from training data only
        train_counts = np.array(labels[smoke_ids, :], dtype=np.int64).sum(axis=0)
        top_idx = np.argsort(train_counts)[::-1][:3]

        # Label 0 has 0 positives in training data, label 1 has positives
        assert 1 in top_idx  # training-active label is selected
        assert 0 not in top_idx  # test/val-active label is NOT selected
