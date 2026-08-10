"""Tests for MultilabelMLP architecture (Milestone 6B)."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polyllm.models.mlp import MultilabelMLP


# ---------------------------------------------------------------------------
# Scenario 1 — MLP output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_batch_output_shape(self):
        m = MultilabelMLP(input_dim=16, output_dim=5)
        m.eval()
        x = torch.randn(8, 16)
        assert m(x).shape == (8, 5)

    def test_single_sample_output_shape(self):
        m = MultilabelMLP(input_dim=16, output_dim=5)
        m.eval()
        x = torch.randn(1, 16)
        assert m(x).shape == (1, 5)


# ---------------------------------------------------------------------------
# Scenario 2 — No sigmoid inside the model
# ---------------------------------------------------------------------------

class TestNoSigmoid:
    def test_no_sigmoid_layer(self):
        m = MultilabelMLP(input_dim=16, output_dim=5)
        for mod in m.modules():
            assert not isinstance(mod, nn.Sigmoid), \
                f"Sigmoid found inside model: {mod}"

    def test_logits_can_be_negative(self):
        """Sigmoid-free model can output negative values."""
        torch.manual_seed(0)
        m = MultilabelMLP(input_dim=16, output_dim=100)
        m.eval()
        x = torch.randn(32, 16)
        out = m(x)
        assert out.min().item() < 0.0, \
            "All logits positive — sigmoid may have been applied inside model."


# ---------------------------------------------------------------------------
# Scenario 3 — Input dimension is configurable
# ---------------------------------------------------------------------------

class TestConfigurableInputDim:
    def test_input_dim_2048(self):
        """Morgan fingerprint case."""
        m = MultilabelMLP(input_dim=2048, output_dim=963)
        m.eval()
        x = torch.randn(2, 2048)
        assert m(x).shape == (2, 963)

    def test_input_dim_384(self):
        """ChemBERTa embedding case — same hidden architecture."""
        m = MultilabelMLP(input_dim=384, output_dim=963)
        m.eval()
        x = torch.randn(2, 384)
        assert m(x).shape == (2, 963)


# ---------------------------------------------------------------------------
# Scenario 4 — Output dimension is configurable
# ---------------------------------------------------------------------------

class TestConfigurableOutputDim:
    def test_output_dim_963(self):
        m = MultilabelMLP(input_dim=16, output_dim=963)
        m.eval()
        assert m(torch.randn(1, 16)).shape == (1, 963)

    def test_output_dim_20(self):
        """Smoke-test case."""
        m = MultilabelMLP(input_dim=16, output_dim=20)
        m.eval()
        assert m(torch.randn(1, 16)).shape == (1, 20)


# ---------------------------------------------------------------------------
# Scenario 5 — BatchNorm and Dropout placement
# ---------------------------------------------------------------------------

class TestBatchNormDropoutPlacement:
    def test_exactly_one_batchnorm(self):
        m = MultilabelMLP(input_dim=16, output_dim=5)
        bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm1d)]
        assert len(bns) == 1, f"Expected 1 BatchNorm1d, got {len(bns)}"

    def test_batchnorm_after_first_layer(self):
        m = MultilabelMLP(input_dim=16, output_dim=5)
        bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm1d)]
        assert bns[0].num_features == 512

    def test_exactly_three_dropouts(self):
        m = MultilabelMLP(input_dim=16, output_dim=5)
        drops = [mod for mod in m.modules() if isinstance(mod, nn.Dropout)]
        assert len(drops) == 3, f"Expected 3 Dropout layers, got {len(drops)}"

    def test_layer_order_in_sequential(self):
        m = MultilabelMLP(input_dim=16, output_dim=5)
        layers = list(m.net.children())
        # Expected: Lin LeakyReLU BN Drop Lin LeakyReLU Drop Lin LeakyReLU Drop Lin
        assert isinstance(layers[0],  nn.Linear)
        assert isinstance(layers[1],  nn.LeakyReLU)
        assert isinstance(layers[2],  nn.BatchNorm1d)
        assert isinstance(layers[3],  nn.Dropout)
        assert isinstance(layers[4],  nn.Linear)
        assert isinstance(layers[5],  nn.LeakyReLU)
        assert isinstance(layers[6],  nn.Dropout)
        assert isinstance(layers[7],  nn.Linear)
        assert isinstance(layers[8],  nn.LeakyReLU)
        assert isinstance(layers[9],  nn.Dropout)
        assert isinstance(layers[10], nn.Linear)

    def test_no_batchnorm_after_layers_2_3(self):
        m = MultilabelMLP(input_dim=16, output_dim=5)
        layers = list(m.net.children())
        # Layers 4-10 must not contain BatchNorm
        for i, layer in enumerate(layers[4:], start=4):
            assert not isinstance(layer, nn.BatchNorm1d), \
                f"Unexpected BatchNorm at position {i}: {layer}"


# ---------------------------------------------------------------------------
# Scenario 6 — BCE loss works with raw logits
# ---------------------------------------------------------------------------

class TestBCELoss:
    def test_bce_with_logits_positive(self):
        m = MultilabelMLP(input_dim=16, output_dim=5)
        m.train()
        x = torch.randn(4, 16)
        y = torch.randint(0, 2, (4, 5)).float()
        loss = nn.BCEWithLogitsLoss()(m(x), y)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_bce_backward_no_error(self):
        m = MultilabelMLP(input_dim=16, output_dim=5)
        m.train()
        x = torch.randn(4, 16)
        y = torch.randint(0, 2, (4, 5)).float()
        loss = nn.BCEWithLogitsLoss()(m(x), y)
        loss.backward()  # must not raise

    def test_parameter_count(self):
        m = MultilabelMLP(input_dim=2048, output_dim=963)
        n = m.count_parameters()
        # Rough sanity: must be in the millions
        assert n > 5_000_000, f"Parameter count {n} seems too low"
        assert n < 10_000_000, f"Parameter count {n} seems too high"


# ---------------------------------------------------------------------------
# Scenario 7 — LeakyReLU negative slope matches PolyLLM paper Sec 2.2
# ---------------------------------------------------------------------------

class TestActivationSlopeMatchesPaper:
    def test_default_slope_is_paper_value(self):
        # PolyLLM paper Sec 2.2: "Leaky ReLU ... with a negative slope of
        # 0.1 for each hidden layer." Was 0.01 (PyTorch's own default)
        # until this was corrected on 2026-08-08; see
        # notes/deviations_from_paper.md Sec 1.7.
        assert MultilabelMLP.NEGATIVE_SLOPE == pytest.approx(0.1)

    def test_default_slope_used_when_unspecified(self):
        m = MultilabelMLP(input_dim=16, output_dim=5)
        leaky_relus = [mod for mod in m.modules() if isinstance(mod, nn.LeakyReLU)]
        assert len(leaky_relus) == 3
        for lr in leaky_relus:
            assert lr.negative_slope == pytest.approx(0.1)

    def test_negative_slope_still_overridable(self):
        # Callers (e.g. the smoke-test path) must still be able to pass an
        # explicit value that overrides the paper-matched default.
        m = MultilabelMLP(input_dim=16, output_dim=5, negative_slope=0.2)
        leaky_relus = [mod for mod in m.modules() if isinstance(mod, nn.LeakyReLU)]
        for lr in leaky_relus:
            assert lr.negative_slope == pytest.approx(0.2)
