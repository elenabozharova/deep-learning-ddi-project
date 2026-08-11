"""
Milestone 10d tests — models/gnn.py
Uses small synthetic hetero graphs throughout (built via build_graph.py's
already-tested helpers); no real data files loaded.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import polyllm.graph.build_graph as bg
import polyllm.models.gnn as gm


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _synthetic_data(n_pdrugs=200, n_seffect=20, pdrugs_dim=8, seffect_dim=4, density=0.3, seed=0):
    rng = np.random.default_rng(seed)
    pdrugs_x = rng.normal(size=(n_pdrugs, pdrugs_dim)).astype(np.float32)
    seffect_x = rng.normal(size=(n_seffect, seffect_dim)).astype(np.float32)
    labels = (rng.random((n_pdrugs, n_seffect)) < density).astype(np.uint8)
    labels[0, 0] = 1
    data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
    return data, pdrugs_dim, seffect_dim


def _model(data, pdrugs_dim, seffect_dim, hidden_channels=16):
    return gm.GNNLinkPredictor(
        num_pdrugs=data["pdrugs"].num_nodes,
        num_seffect=data["seffect"].num_nodes,
        pdrugs_input_dim=pdrugs_dim,
        hidden_channels=hidden_channels,
        metadata=data.metadata(),
        seffect_input_dim=seffect_dim,
    )


def _dicts(split_data):
    x_dict = {"pdrugs": split_data["pdrugs"].x, "seffect": split_data["seffect"].x}
    edge_index_dict = split_data.edge_index_dict
    node_id_dict = {
        "pdrugs": split_data["pdrugs"].node_id,
        "seffect": split_data["seffect"].node_id,
    }
    return x_dict, edge_index_dict, node_id_dict


# ---------------------------------------------------------------------------
# Scenario 1: GNNEncoder — shape and dead-code omission
# ---------------------------------------------------------------------------

class TestGNNEncoder:
    def test_forward_shape_preserved(self):
        enc = gm.GNNEncoder(hidden_channels=16)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        enc.eval()
        out = enc(x, edge_index)
        assert out.shape == (10, 16)

    def test_no_bn3_bn4(self):
        enc = gm.GNNEncoder(hidden_channels=16)
        assert not hasattr(enc, "bn3")
        assert not hasattr(enc, "bn4")

    def test_no_dropout_rate_param(self):
        import inspect
        sig = inspect.signature(gm.GNNEncoder.__init__)
        assert "dropout_rate" not in sig.parameters

    def test_dropout_p_is_08(self):
        assert gm.GNNEncoder.DROPOUT_P == 0.8


# ---------------------------------------------------------------------------
# Scenario 2: DotProductClassifier — matches authors' formula exactly
# ---------------------------------------------------------------------------

class TestDotProductClassifier:
    def test_dot_product_correctness(self):
        clf = gm.DotProductClassifier()
        x_pdrugs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x_seffect = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        edge_label_index = torch.tensor([[0, 1], [1, 0]])  # (pdrugs=0,seffect=1), (pdrugs=1,seffect=0)
        out = clf(x_pdrugs, x_seffect, edge_label_index)
        expected = torch.tensor([
            (1.0 * 7.0 + 2.0 * 8.0),
            (3.0 * 5.0 + 4.0 * 6.0),
        ])
        assert torch.allclose(out, expected)

    def test_integrate_returns_elementwise(self):
        clf = gm.DotProductClassifier()
        x_pdrugs = torch.tensor([[1.0, 2.0]])
        x_seffect = torch.tensor([[3.0, 4.0]])
        edge_label_index = torch.tensor([[0], [0]])
        out = clf(x_pdrugs, x_seffect, edge_label_index, integrate=True)
        assert torch.allclose(out, torch.tensor([[3.0, 8.0]]))

    def test_output_shape_matches_edge_count(self):
        clf = gm.DotProductClassifier()
        x_pdrugs = torch.randn(5, 4)
        x_seffect = torch.randn(3, 4)
        edge_label_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]])
        out = clf(x_pdrugs, x_seffect, edge_label_index)
        assert out.shape == (4,)


# ---------------------------------------------------------------------------
# Scenario 3: GNNLinkPredictor — end-to-end encode/predict on a real
# (synthetic) hetero graph, including the reverse edge type
# ---------------------------------------------------------------------------

class TestGNNLinkPredictor:
    def test_encode_shapes(self):
        data, pdrugs_dim, seffect_dim = _synthetic_data()
        train, _, _ = bg.split_graph(data, seed=42)
        model = _model(data, pdrugs_dim, seffect_dim)
        model.eval()
        x_dict, edge_index_dict, node_id_dict = _dicts(train)
        with torch.no_grad():
            enc = model.encode(x_dict, edge_index_dict, node_id_dict)
        assert enc["pdrugs"].shape == (200, 16)
        assert enc["seffect"].shape == (20, 16)

    def test_predict_edges_shape(self):
        data, pdrugs_dim, seffect_dim = _synthetic_data()
        train, _, _ = bg.split_graph(data, seed=42)
        model = _model(data, pdrugs_dim, seffect_dim)
        model.eval()
        x_dict, edge_index_dict, node_id_dict = _dicts(train)
        edge_label_index = train[gm.EDGE_TYPE].edge_label_index
        with torch.no_grad():
            enc = model.encode(x_dict, edge_index_dict, node_id_dict)
            pred = model.predict_edges(enc, edge_label_index)
        assert pred.shape == (edge_label_index.size(1),)

    def test_forward_matches_encode_then_predict(self):
        data, pdrugs_dim, seffect_dim = _synthetic_data()
        train, _, _ = bg.split_graph(data, seed=42)
        model = _model(data, pdrugs_dim, seffect_dim)
        model.eval()
        x_dict, edge_index_dict, node_id_dict = _dicts(train)
        edge_label_index = train[gm.EDGE_TYPE].edge_label_index
        with torch.no_grad():
            enc = model.encode(x_dict, edge_index_dict, node_id_dict)
            pred_split = model.predict_edges(enc, edge_label_index)
            pred_forward = model(x_dict, edge_index_dict, node_id_dict, edge_label_index)
        assert torch.allclose(pred_split, pred_forward)

    def test_seffect_input_dim_defaults_to_768(self):
        assert gm.GNNLinkPredictor.SEFFECT_INPUT_DIM == 768

    def test_no_dangerous_seffects_param(self):
        import inspect
        sig = inspect.signature(gm.GNNLinkPredictor.__init__)
        assert "dangerous_seffects_ids" not in sig.parameters


# ---------------------------------------------------------------------------
# Scenario 4: Negative sampling
# ---------------------------------------------------------------------------

class TestNegativeSampling:
    def test_global_positive_edge_set_matches_edge_count(self):
        data, _, _ = _synthetic_data()
        pos_edges = gm.build_global_positive_edge_set(data)
        assert len(pos_edges) == data[gm.EDGE_TYPE].edge_index.size(1)

    def test_sampled_negatives_are_not_true_positives(self):
        data, _, _ = _synthetic_data(density=0.3)
        train, _, _ = bg.split_graph(data, seed=42)
        global_pos = gm.build_global_positive_edge_set(data)
        neg_edge_index = gm.sample_negative_edges(train, global_pos, torch.device("cpu"))
        neg_edges = set(map(tuple, neg_edge_index.T.tolist()))
        assert neg_edges.isdisjoint(global_pos)

    def test_sampled_negatives_nonempty(self):
        data, _, _ = _synthetic_data(density=0.3)
        train, _, _ = bg.split_graph(data, seed=42)
        global_pos = gm.build_global_positive_edge_set(data)
        neg_edge_index = gm.sample_negative_edges(train, global_pos, torch.device("cpu"))
        assert neg_edge_index.size(1) > 0

    def test_assemble_supervision_edges_label_counts(self):
        data, _, _ = _synthetic_data(density=0.3)
        train, _, _ = bg.split_graph(data, seed=42)
        global_pos = gm.build_global_positive_edge_set(data)
        edge_label_index, edge_label = gm.assemble_supervision_edges(train, global_pos, torch.device("cpu"))

        n_pos = train[gm.EDGE_TYPE].edge_label_index.size(1)
        n_total = edge_label_index.size(1)
        assert edge_label.shape == (n_total,)
        assert int(edge_label.sum().item()) == n_pos
        assert edge_label_index.shape == (2, n_total)

    def test_assemble_supervision_edges_positives_come_first(self):
        data, _, _ = _synthetic_data(density=0.3)
        train, _, _ = bg.split_graph(data, seed=42)
        global_pos = gm.build_global_positive_edge_set(data)
        edge_label_index, edge_label = gm.assemble_supervision_edges(train, global_pos, torch.device("cpu"))

        n_pos = train[gm.EDGE_TYPE].edge_label_index.size(1)
        assert torch.all(edge_label[:n_pos] == 1)
        assert torch.all(edge_label[n_pos:] == 0)

    def test_val_split_negatives_also_excluded_from_global_positives(self):
        """val/test carry only positive edge_label (Milestone 10c) — negatives
        must be sampled the same way for eval as for train."""
        data, _, _ = _synthetic_data(density=0.3)
        _, val, _ = bg.split_graph(data, seed=42)
        global_pos = gm.build_global_positive_edge_set(data)
        edge_label_index, edge_label = gm.assemble_supervision_edges(val, global_pos, torch.device("cpu"))
        neg_edges = set(map(tuple, edge_label_index[:, edge_label == 0].T.tolist()))
        assert neg_edges.isdisjoint(global_pos)

    def test_local_batch_indices_filtered_via_global_node_id_not_local_index(self, monkeypatch):
        """
        Regression test for the local/global index-space fix (see gnn.py's
        module docstring). Simulates what a real LinkNeighborLoader mini-batch
        looks like: node_id is a non-identity subset of the full graph's
        global ids (here [5, 7] for pdrugs, [3, 9] for seffect — deliberately
        disjoint from the small {0, 1} local index range so a coincidental
        collision can't fake a pass), and the batch's edge_index is expressed
        in LOCAL 0/1 coordinates. `negative_sampling` itself is monkeypatched
        to a fixed return value so this test is deterministic rather than
        depending on PyG's internal RNG: it returns exactly the two local
        pairs (0,0) and (1,0), of which local (0,0) maps to GLOBAL (5,3) — a
        true positive — and local (1,0) maps to GLOBAL (7,3), a true
        negative. A filter comparing LOCAL tuples directly against the
        GLOBALLY-indexed positive set would keep (0,0) as a "negative" (since
        the literal tuple (0,0) is absent from global_pos_edges even though
        it IS a true positive once mapped) — exactly the bug being fixed.
        """
        global_pos_edges = {(5, 3), (7, 9)}  # true positives, in GLOBAL indices

        from torch_geometric.data import HeteroData

        batch = HeteroData()
        batch["pdrugs"].node_id = torch.tensor([5, 7], dtype=torch.long)   # local0->5, local1->7
        batch["seffect"].node_id = torch.tensor([3, 9], dtype=torch.long)  # local0->3, local1->9
        batch["pdrugs"].x = torch.zeros(2, 4)
        batch["seffect"].x = torch.zeros(2, 4)
        batch[gm.EDGE_TYPE].edge_index = torch.tensor([[1], [1]], dtype=torch.long)  # local (1,1) = global (7,9)
        batch[gm.EDGE_TYPE].edge_label_index = torch.tensor([[1], [1]], dtype=torch.long)
        batch[gm.EDGE_TYPE].edge_label = torch.tensor([1.0])

        fixed_neg = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)  # local (0,0) and (1,0)
        monkeypatch.setattr(gm, "negative_sampling", lambda **kwargs: fixed_neg)

        # Sanity check the trap: a naive local-tuple-vs-global-set comparison
        # would NOT catch local (0,0) as a positive, even though it is one.
        assert (0, 0) not in global_pos_edges

        neg_edge_index = gm.sample_negative_edges(batch, global_pos_edges, torch.device("cpu"))
        surviving_local_pairs = set(map(tuple, neg_edge_index.T.tolist()))

        # local (0,0) -> global (5,3), a true positive -> must be dropped.
        assert (0, 0) not in surviving_local_pairs
        # local (1,0) -> global (7,3), a true negative -> must survive.
        assert (1, 0) in surviving_local_pairs
