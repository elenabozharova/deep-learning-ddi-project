"""
Milestone 10c tests — build_graph.py
Uses small synthetic node/edge counts throughout; no real data files loaded.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import polyllm.graph.build_graph as bg


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _synthetic_arrays(n_pdrugs=200, n_seffect=20, density=0.3, seed=0):
    rng = np.random.default_rng(seed)
    pdrugs_x = rng.normal(size=(n_pdrugs, 8)).astype(np.float32)
    seffect_x = rng.normal(size=(n_seffect, 4)).astype(np.float32)
    labels = (rng.random((n_pdrugs, n_seffect)) < density).astype(np.uint8)
    # Guarantee at least one edge
    labels[0, 0] = 1
    return pdrugs_x, seffect_x, labels


# ---------------------------------------------------------------------------
# Scenario 1: Input validation
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_valid_passes(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays()
        bg.validate_inputs(pdrugs_x, seffect_x, labels)  # no raise

    def test_1d_labels_raises(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays()
        with pytest.raises(ValueError, match="must be 2D"):
            bg.validate_inputs(pdrugs_x, seffect_x, labels.flatten())

    def test_pdrugs_row_mismatch_raises(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_pdrugs=200)
        with pytest.raises(ValueError, match="pdrugs feature count"):
            bg.validate_inputs(pdrugs_x[:100], seffect_x, labels)

    def test_seffect_row_mismatch_raises(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_seffect=20)
        with pytest.raises(ValueError, match="seffect feature count"):
            bg.validate_inputs(pdrugs_x, seffect_x[:10], labels)

    def test_nan_in_pdrugs_raises(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays()
        pdrugs_x[0, 0] = np.nan
        with pytest.raises(ValueError, match="pdrugs feature matrix"):
            bg.validate_inputs(pdrugs_x, seffect_x, labels)

    def test_nan_in_seffect_raises(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays()
        seffect_x[0, 0] = np.nan
        with pytest.raises(ValueError, match="seffect feature matrix"):
            bg.validate_inputs(pdrugs_x, seffect_x, labels)

    def test_zero_edges_raises(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays()
        labels = np.zeros_like(labels)
        with pytest.raises(ValueError, match="zero positive entries"):
            bg.validate_inputs(pdrugs_x, seffect_x, labels)


# ---------------------------------------------------------------------------
# Scenario 2: build_hetero_data structure
# ---------------------------------------------------------------------------

class TestBuildHeteroData:
    def test_node_counts(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_pdrugs=200, n_seffect=20)
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        assert data["pdrugs"].num_nodes == 200
        assert data["seffect"].num_nodes == 20

    def test_node_id_is_long_tensor_arange(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_pdrugs=50, n_seffect=10)
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        assert data["pdrugs"].node_id.dtype == torch.long
        assert torch.equal(data["pdrugs"].node_id, torch.arange(50))
        assert data["seffect"].node_id.dtype == torch.long
        assert torch.equal(data["seffect"].node_id, torch.arange(10))

    def test_feature_dtype_float32(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays()
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        assert data["pdrugs"].x.dtype == torch.float32
        assert data["seffect"].x.dtype == torch.float32

    def test_edge_count_matches_label_sum(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays()
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        assert data[bg.EDGE_TYPE].edge_index.size(1) == int(labels.sum())

    def test_reverse_edge_type_added(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays()
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        assert bg.REV_EDGE_TYPE in data.edge_types

    def test_edge_index_matches_nonzero_labels(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_pdrugs=30, n_seffect=6)
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        expected_pdrugs, expected_seffect = np.nonzero(labels)
        edge_index = data[bg.EDGE_TYPE].edge_index.numpy()
        assert set(zip(edge_index[0], edge_index[1])) == set(zip(expected_pdrugs, expected_seffect))


# ---------------------------------------------------------------------------
# Scenario 3: split_graph + validate_split
# ---------------------------------------------------------------------------

class TestSplitGraph:
    def test_split_produces_three_hetero_data(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_pdrugs=300, n_seffect=30, density=0.4)
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        train, val, test = bg.split_graph(data, seed=42)
        for split in (train, val, test):
            assert bg.EDGE_TYPE in split.edge_types

    def test_val_test_labels_are_all_positive(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_pdrugs=300, n_seffect=30, density=0.4)
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        train, val, test = bg.split_graph(data, seed=42)
        assert torch.all(val[bg.EDGE_TYPE].edge_label == 1)
        assert torch.all(test[bg.EDGE_TYPE].edge_label == 1)

    def test_validate_split_passes_on_real_split(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_pdrugs=300, n_seffect=30, density=0.4)
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        train, val, test = bg.split_graph(data, seed=42)
        bg.validate_split(data, train, val, test)  # no raise

    def test_validate_split_catches_bad_val_fraction(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_pdrugs=300, n_seffect=30, density=0.4)
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        train, val, test = bg.split_graph(data, seed=42)

        # Artificially shrink val's supervision edges far below the expected fraction
        val[bg.EDGE_TYPE].edge_label_index = val[bg.EDGE_TYPE].edge_label_index[:, :1]
        val[bg.EDGE_TYPE].edge_label = val[bg.EDGE_TYPE].edge_label[:1]

        with pytest.raises(ValueError, match="val split fraction"):
            bg.validate_split(data, train, val, test)

    def test_validate_split_catches_negative_label_in_val(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_pdrugs=300, n_seffect=30, density=0.4)
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        train, val, test = bg.split_graph(data, seed=42)

        val[bg.EDGE_TYPE].edge_label[0] = 0

        with pytest.raises(ValueError, match="non-positive edge_label"):
            bg.validate_split(data, train, val, test)

    def test_deterministic_given_seed(self):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_pdrugs=300, n_seffect=30, density=0.4)
        data1 = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        data2 = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        train1, val1, test1 = bg.split_graph(data1, seed=123)
        train2, val2, test2 = bg.split_graph(data2, seed=123)
        assert torch.equal(
            val1[bg.EDGE_TYPE].edge_label_index, val2[bg.EDGE_TYPE].edge_label_index
        )


# ---------------------------------------------------------------------------
# Scenario 4: Audit schema
# ---------------------------------------------------------------------------

class TestGraphAudit:
    def test_required_fields_present(self, tmp_path):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_pdrugs=300, n_seffect=30, density=0.4)
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        train, val, test = bg.split_graph(data, seed=42)

        audit = bg.build_graph_audit(
            data, train, val, test, seed=42,
            graph_output=tmp_path / "g.pt", audit_output=tmp_path / "a.json",
        )
        required_keys = [
            "provenance", "node_types", "edge_type", "reverse_edge_type",
            "total_positive_edges_pdrugs_to_seffect", "split_params", "splits",
            "software_versions", "output_paths",
        ]
        for k in required_keys:
            assert k in audit, f"Missing audit key: {k}"
        assert audit["node_types"]["pdrugs"]["count"] == 300
        assert audit["node_types"]["seffect"]["count"] == 30

    def test_split_params_match_authors_recipe(self, tmp_path):
        pdrugs_x, seffect_x, labels = _synthetic_arrays(n_pdrugs=300, n_seffect=30, density=0.4)
        data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
        train, val, test = bg.split_graph(data, seed=42)
        audit = bg.build_graph_audit(
            data, train, val, test, seed=42,
            graph_output=tmp_path / "g.pt", audit_output=tmp_path / "a.json",
        )
        params = audit["split_params"]
        assert params["num_val"] == 0.1
        assert params["num_test"] == 0.1
        assert params["disjoint_train_ratio"] == 0.3
        assert params["neg_sampling_ratio"] == 0.0
        assert params["add_negative_train_samples"] is False


# ---------------------------------------------------------------------------
# Scenario 5: Constants match the authors' confirmed recipe
# ---------------------------------------------------------------------------

class TestConstantsMatchAuthorsRecipe:
    def test_edge_type(self):
        assert bg.EDGE_TYPE == ("pdrugs", "associated", "seffect")

    def test_rev_edge_type(self):
        assert bg.REV_EDGE_TYPE == ("seffect", "rev_associated", "pdrugs")

    def test_split_fractions(self):
        assert bg.NUM_VAL == 0.1
        assert bg.NUM_TEST == 0.1
        assert bg.DISJOINT_TRAIN_RATIO == 0.3
        assert bg.NEG_SAMPLING_RATIO == 0.0
