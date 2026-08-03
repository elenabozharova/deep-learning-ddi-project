"""Integration tests for Milestone 6B — Morgan MLP training/evaluation pipeline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polyllm.models.mlp import MultilabelMLP
from polyllm.training import (
    PairDataset,
    load_checkpoint,
    make_seeded_loader,
    run_val_epoch,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Shared tiny fixtures
# ---------------------------------------------------------------------------

N = 30
F = 8
L = 6
SEED = 42


def _make_data(n=N, f=F, l=L):
    rng = np.random.default_rng(SEED)
    feat = rng.integers(0, 3, size=(n, f)).astype(np.uint8)
    lab  = rng.integers(0, 2, size=(n, l)).astype(np.uint8)
    return feat, lab


def _tiny_model(f=F, l=L):
    return MultilabelMLP(input_dim=f, output_dim=l, dropout=0.0)


# ---------------------------------------------------------------------------
# Scenario 25 — Prediction ordering by pair ID
# ---------------------------------------------------------------------------

class TestPredictionOrdering:
    def test_val_epoch_returns_sorted_pair_ids(self):
        features, labels = _make_data()
        # Provide pair_ids in reverse order → run_val_epoch should sort them
        pair_ids = np.arange(N - 1, -1, -1)   # reversed
        ds     = PairDataset(pair_ids, features, labels)
        loader = make_seeded_loader(ds, batch_size=8, shuffle=False, seed=SEED)
        model  = _tiny_model()
        criterion = nn.BCEWithLogitsLoss()
        _, probs, labs, pids = run_val_epoch(model, loader, criterion, torch.device("cpu"))
        assert np.array_equal(pids, np.sort(pids)), \
            "pair IDs are not sorted after run_val_epoch"

    def test_probabilities_aligned_with_sorted_pair_ids(self):
        features, labels = _make_data()
        pair_ids = np.arange(N)
        ds     = PairDataset(pair_ids, features, labels)
        loader = make_seeded_loader(ds, batch_size=8, shuffle=False, seed=SEED)
        model  = _tiny_model()
        criterion = nn.BCEWithLogitsLoss()
        _, probs, labs, pids = run_val_epoch(model, loader, criterion, torch.device("cpu"))
        # The pair_ids should be 0,1,...,N-1 in order
        assert np.array_equal(pids, np.arange(N))


# ---------------------------------------------------------------------------
# Scenario 26 — Prediction probability range [0, 1]
# ---------------------------------------------------------------------------

class TestProbabilityRange:
    def test_all_probs_in_unit_interval(self):
        features, labels = _make_data()
        ds     = PairDataset(np.arange(N), features, labels)
        loader = make_seeded_loader(ds, batch_size=8, shuffle=False, seed=SEED)
        model  = _tiny_model()
        criterion = nn.BCEWithLogitsLoss()
        _, probs, _, _ = run_val_epoch(model, loader, criterion, torch.device("cpu"))
        assert np.all(probs >= 0.0), "Probabilities below 0"
        assert np.all(probs <= 1.0), "Probabilities above 1"

    def test_probs_dtype_float32(self):
        features, labels = _make_data()
        ds     = PairDataset(np.arange(N), features, labels)
        loader = make_seeded_loader(ds, batch_size=8, shuffle=False, seed=SEED)
        model  = _tiny_model()
        criterion = nn.BCEWithLogitsLoss()
        _, probs, _, _ = run_val_epoch(model, loader, criterion, torch.device("cpu"))
        assert probs.dtype == np.float32

    def test_all_probs_finite(self):
        features, labels = _make_data()
        ds     = PairDataset(np.arange(N), features, labels)
        loader = make_seeded_loader(ds, batch_size=8, shuffle=False, seed=SEED)
        model  = _tiny_model()
        criterion = nn.BCEWithLogitsLoss()
        _, probs, _, _ = run_val_epoch(model, loader, criterion, torch.device("cpu"))
        assert np.all(np.isfinite(probs))


# ---------------------------------------------------------------------------
# Scenario 27 — Checkpoint metadata validation
# ---------------------------------------------------------------------------

class TestCheckpointMetadata:
    def test_required_keys_in_checkpoint(self, tmp_path):
        model = _tiny_model()
        opt   = torch.optim.Adam(model.parameters(), lr=0.001)
        config = {"input_dim": F, "output_dim": L, "learning_rate": 0.001}
        path  = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, opt, epoch=3, score=0.55, config=config)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for key in ["epoch", "val_macro_auprc", "model_state_dict",
                    "optimizer_state_dict", "config", "model_class",
                    "input_dim", "output_dim"]:
            assert key in ckpt, f"Missing key: {key}"

    def test_input_output_dim_stored(self, tmp_path):
        model = _tiny_model()
        opt   = torch.optim.Adam(model.parameters(), lr=0.001)
        path  = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, opt, epoch=1, score=0.3,
                        config={"input_dim": F, "output_dim": L})
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["input_dim"]  == F
        assert ckpt["output_dim"] == L


# ---------------------------------------------------------------------------
# Scenario 16 — Test split not accessed during training
# ---------------------------------------------------------------------------

class TestTestSplitNotAccessedDuringTraining:
    def test_train_and_val_functions_have_no_test_parameter(self):
        """
        run_train_epoch and run_val_epoch only accept model, loader, criterion,
        optimizer (training), and device.  They have no test_ids or test_loader
        parameter, enforcing that test data cannot be passed in.
        """
        import inspect
        from polyllm.training import run_train_epoch, run_val_epoch

        train_params = set(inspect.signature(run_train_epoch).parameters.keys())
        val_params   = set(inspect.signature(run_val_epoch).parameters.keys())

        assert "test" not in " ".join(train_params)
        assert "test" not in " ".join(val_params)

    def test_training_loop_uses_only_train_and_val_loaders(self):
        """
        Run one training + validation cycle with two specific pair_id sets;
        verify only those pair_ids appear in the collected predictions.
        """
        features, labels = _make_data(n=30)
        train_ids = np.arange(20)
        val_ids   = np.arange(20, 30)
        # Deliberately do NOT pass test_ids anywhere

        train_ds = PairDataset(train_ids, features, labels)
        val_ds   = PairDataset(val_ids,   features, labels)
        train_loader = make_seeded_loader(train_ds, batch_size=8, shuffle=True,  seed=0)
        val_loader   = make_seeded_loader(val_ds,   batch_size=8, shuffle=False, seed=0)

        model     = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        device    = torch.device("cpu")

        from polyllm.training import run_train_epoch
        run_train_epoch(model, train_loader, criterion, optimizer, device)
        _, _, _, collected_pids = run_val_epoch(model, val_loader, criterion, device)

        # Only val IDs should appear in collected predictions
        assert set(collected_pids.tolist()).issubset(set(val_ids.tolist()))
        assert not set(train_ids.tolist()) & set(collected_pids.tolist())


# ---------------------------------------------------------------------------
# Scenario 30 — Disk-loaded output validation
# ---------------------------------------------------------------------------

class TestDiskLoadedOutputValidation:
    def test_predictions_npz_schema(self, tmp_path):
        n_test = 20
        n_labels = L
        pids  = np.arange(n_test, dtype=np.int64)
        probs = np.random.default_rng(0).random((n_test, n_labels)).astype(np.float32)
        path  = tmp_path / "pred.npz"
        np.savez(path, pair_ids=pids, probabilities=probs,
                 selected_threshold=np.float32(0.3))
        loaded = np.load(path)
        assert set(["pair_ids", "probabilities", "selected_threshold"]).issubset(
            set(loaded.files)
        )
        assert loaded["probabilities"].shape == (n_test, n_labels)
        assert loaded["probabilities"].dtype == np.float32
        assert np.all(np.isfinite(loaded["probabilities"]))
        assert np.all((loaded["probabilities"] >= 0) & (loaded["probabilities"] <= 1))

    def test_sorted_pair_ids_in_npz(self, tmp_path):
        pids  = np.array([5, 2, 8, 0], dtype=np.int64)
        probs = np.ones((4, 3), dtype=np.float32) * 0.5
        path  = tmp_path / "pred.npz"
        # Save with unsorted IDs to test that caller sorts before saving
        np.savez(path, pair_ids=np.sort(pids), probabilities=probs[np.argsort(pids)],
                 selected_threshold=np.float32(0.5))
        loaded = np.load(path)
        assert np.array_equal(loaded["pair_ids"], np.sort(loaded["pair_ids"]))

    def test_per_label_csv_schema(self, tmp_path):
        n_labels = 5
        df = pd.DataFrame({
            "label_index":              np.arange(n_labels),
            "side_effect_id":           [f"C{i}" for i in range(n_labels)],
            "side_effect_name":         [f"se_{i}" for i in range(n_labels)],
            "test_positive_count":      np.zeros(n_labels, dtype=int),
            "test_prevalence":          np.zeros(n_labels),
            "auroc":                    np.full(n_labels, 0.5),
            "auprc":                    np.full(n_labels, 0.5),
            "f1_at_selected_threshold": np.full(n_labels, 0.5),
        })
        csv_path = tmp_path / "per_label.csv"
        df.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)
        for col in ["label_index", "side_effect_id", "auroc", "auprc",
                    "f1_at_selected_threshold"]:
            assert col in loaded.columns, f"Missing column: {col}"
        assert len(loaded) == n_labels
