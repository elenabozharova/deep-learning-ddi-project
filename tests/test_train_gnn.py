"""
Milestone 10e tests — train_gnn.py
Uses small synthetic hetero graphs (via build_graph.py's tested helpers);
no real data files or the full saved graph are loaded.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import polyllm.graph.build_graph as bg
import polyllm.models.gnn as gm
import polyllm.train_gnn as tg


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _synthetic_full_train_val(n_pdrugs=100, n_seffect=15, pdrugs_dim=8, density=0.3, seed=0):
    rng = np.random.default_rng(seed)
    pdrugs_x = rng.normal(size=(n_pdrugs, pdrugs_dim)).astype(np.float32)
    seffect_x = rng.normal(size=(n_seffect, 4)).astype(np.float32)
    labels = (rng.random((n_pdrugs, n_seffect)) < density).astype(np.uint8)
    labels[0, 0] = 1
    full = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
    train, val, test = bg.split_graph(full, seed=42)
    return full, train, val, test, pdrugs_dim


def _tiny_config(**overrides) -> dict:
    cfg = dict(tg.DEFAULT_CONFIG)
    cfg["hidden_channels"] = 8
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Scenario 1: set_seeds — determinism
# ---------------------------------------------------------------------------

class TestSetSeeds:
    def test_reproducible_torch_randn(self):
        tg.set_seeds(123)
        a = torch.randn(5)
        tg.set_seeds(123)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_reproducible_numpy(self):
        tg.set_seeds(7)
        a = np.random.rand(5)
        tg.set_seeds(7)
        b = np.random.rand(5)
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Scenario 2: EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_first_call_is_improvement(self, tmp_path):
        model = torch.nn.Linear(2, 2)
        es = tg.EarlyStopping(patience=2, path=tmp_path / "ckpt.pt")
        improved = es.step(1.0, epoch=1, model=model)
        assert improved is True
        assert es.best_epoch == 1
        assert (tmp_path / "ckpt.pt").exists()

    def test_improvement_resets_counter(self, tmp_path):
        model = torch.nn.Linear(2, 2)
        es = tg.EarlyStopping(patience=2, path=tmp_path / "ckpt.pt")
        es.step(1.0, epoch=1, model=model)
        es.step(2.0, epoch=2, model=model)  # worse -> counter=1
        assert es.counter == 1
        improved = es.step(0.5, epoch=3, model=model)  # better -> resets
        assert improved is True
        assert es.counter == 0
        assert es.best_epoch == 3

    def test_stops_after_patience_exhausted(self, tmp_path):
        model = torch.nn.Linear(2, 2)
        es = tg.EarlyStopping(patience=2, path=tmp_path / "ckpt.pt")
        es.step(1.0, epoch=1, model=model)   # best
        es.step(2.0, epoch=2, model=model)   # worse, counter=1
        assert not es.early_stop
        es.step(2.0, epoch=3, model=model)   # worse, counter=2 -> stop
        assert es.early_stop

    def test_checkpoint_reflects_best_not_last(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        model = torch.nn.Linear(2, 2)
        es = tg.EarlyStopping(patience=5, path=path)

        with torch.no_grad():
            model.weight.fill_(1.0)
        es.step(1.0, epoch=1, model=model)  # best -> saved with weight=1.0

        with torch.no_grad():
            model.weight.fill_(2.0)
        es.step(2.0, epoch=2, model=model)  # worse -> NOT saved

        saved_state = torch.load(path, weights_only=True)
        assert torch.allclose(saved_state["weight"], torch.ones(2, 2))


# ---------------------------------------------------------------------------
# Scenario 3: run_train_epoch / run_eval_epoch on a real (synthetic) graph,
# using an in-memory loader-equivalent (LinkNeighborLoader requires pyg-lib
# and is exercised at full scale manually; here we hand-roll a trivial
# single-batch "loader" to test the loss/backward plumbing in isolation).
# ---------------------------------------------------------------------------

class TestEpochLoops:
    def _model_and_data(self):
        full, train, val, test, pdrugs_dim = _synthetic_full_train_val()
        config = _tiny_config()
        device = torch.device("cpu")
        model = tg.build_model(full, config, device)
        global_pos_edges = gm.build_global_positive_edge_set(full)
        return model, train, val, global_pos_edges, device

    def test_train_epoch_updates_parameters_and_returns_finite_loss(self):
        model, train, val, global_pos_edges, device = self._model_and_data()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        param_before = next(model.parameters()).data.clone()
        loss = tg.run_train_epoch(model, [train], global_pos_edges, optimizer, device)

        assert np.isfinite(loss)
        assert not torch.equal(param_before, next(model.parameters()).data)

    def test_eval_epoch_does_not_update_parameters(self):
        model, train, val, global_pos_edges, device = self._model_and_data()

        param_before = next(model.parameters()).data.clone()
        loss = tg.run_eval_epoch(model, [val], global_pos_edges, device)

        assert np.isfinite(loss)
        assert torch.equal(param_before, next(model.parameters()).data)

    def test_eval_epoch_is_deterministic_in_eval_mode(self):
        """model.eval() disables dropout, so repeated eval passes on the same
        data (and same sampled negatives, via a fixed seed) should agree."""
        model, train, val, global_pos_edges, device = self._model_and_data()
        tg.set_seeds(0)
        loss1 = tg.run_eval_epoch(model, [val], global_pos_edges, device)
        tg.set_seeds(0)
        loss2 = tg.run_eval_epoch(model, [val], global_pos_edges, device)
        assert loss1 == pytest.approx(loss2)


# ---------------------------------------------------------------------------
# Scenario 4: build_link_loader — constructs a real LinkNeighborLoader whose
# first batch is well-formed (requires pyg-lib; skipped if unavailable)
# ---------------------------------------------------------------------------

class TestBuildLinkLoader:
    def test_loader_yields_valid_batch(self):
        pytest.importorskip("pyg_lib")
        full, train, val, test, pdrugs_dim = _synthetic_full_train_val(n_pdrugs=100, n_seffect=15)
        loader = tg.build_link_loader(train, batch_size=16, num_neighbors=[4, 2], shuffle=False)
        batch = next(iter(loader))
        assert gm.EDGE_TYPE in batch.edge_types
        assert batch[gm.EDGE_TYPE].edge_label_index.size(1) <= 16
        assert "node_id" in batch["pdrugs"]
        assert "node_id" in batch["seffect"]


# ---------------------------------------------------------------------------
# Scenario 5: config building from CLI args
# ---------------------------------------------------------------------------

class TestBuildConfig:
    def test_build_config_applies_overrides(self):
        import argparse
        ns = argparse.Namespace(
            smoke_test=False, seed=7, hidden_channels=32, lr=0.005, epochs=3, patience=1, overwrite=True,
        )
        cfg = tg.build_config(ns)
        assert cfg["random_seed"] == 7
        assert cfg["hidden_channels"] == 32
        assert cfg["learning_rate"] == 0.005
        assert cfg["max_epochs"] == 3
        assert cfg["patience"] == 1

    def test_default_config_matches_authors_recipe(self):
        assert tg.DEFAULT_CONFIG["hidden_channels"] == 64
        assert tg.DEFAULT_CONFIG["learning_rate"] == 0.01
        assert tg.DEFAULT_CONFIG["max_epochs"] == 10
        assert tg.DEFAULT_CONFIG["patience"] == 2
        assert tg.DEFAULT_CONFIG["num_neighbors"] == [20, 10]
        assert tg.DEFAULT_CONFIG["train_batch_size"] == 65536
        assert tg.DEFAULT_CONFIG["eval_batch_size"] == 2048
        assert tg.DEFAULT_CONFIG["shuffle_loaders"] is False
