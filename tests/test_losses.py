"""Tests for BinaryFocalLossWithLogits (Milestone 9d)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polyllm.losses import BinaryFocalLossWithLogits


# ---------------------------------------------------------------------------
# Scenario 1 — Reduces to BCE when gamma=0, alpha disabled
# ---------------------------------------------------------------------------

class TestReducesToBCE:
    def test_matches_bce_with_logits(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 2, (8, 5)).float()

        focal = BinaryFocalLossWithLogits(gamma=0.0, alpha=-1.0, reduction="mean")
        bce = nn.BCEWithLogitsLoss()

        assert focal(logits, targets).item() == pytest.approx(bce(logits, targets).item(), abs=1e-6)

    def test_matches_bce_elementwise_with_reduction_none(self):
        torch.manual_seed(1)
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()

        focal = BinaryFocalLossWithLogits(gamma=0.0, alpha=-1.0, reduction="none")
        bce = nn.BCEWithLogitsLoss(reduction="none")

        assert torch.allclose(focal(logits, targets), bce(logits, targets), atol=1e-6)


# ---------------------------------------------------------------------------
# Scenario 2 — Focusing behaviour (gamma down-weights easy examples)
# ---------------------------------------------------------------------------

class TestFocusingBehaviour:
    def test_confident_correct_prediction_down_weighted(self):
        # target=1, high logit (confident, correct) -> p_t close to 1 ->
        # (1-p_t)^gamma close to 0 -> focal loss << plain BCE
        logits = torch.tensor([[5.0]])
        targets = torch.tensor([[1.0]])

        focal = BinaryFocalLossWithLogits(gamma=2.0, alpha=-1.0, reduction="mean")
        bce = nn.BCEWithLogitsLoss()

        assert focal(logits, targets).item() < bce(logits, targets).item()

    def test_hard_example_less_down_weighted_than_easy(self):
        # target=1: logit=5 (easy/confident) vs logit=0 (hard/uncertain).
        # Focal loss should shrink the easy example's loss much more than
        # the hard example's loss, relative to plain BCE.
        easy_logits = torch.tensor([[5.0]])
        hard_logits = torch.tensor([[0.0]])
        target = torch.tensor([[1.0]])

        focal = BinaryFocalLossWithLogits(gamma=2.0, alpha=-1.0, reduction="mean")
        bce = nn.BCEWithLogitsLoss()

        easy_ratio = focal(easy_logits, target).item() / bce(easy_logits, target).item()
        hard_ratio = focal(hard_logits, target).item() / bce(hard_logits, target).item()

        assert easy_ratio < hard_ratio

    def test_higher_gamma_shrinks_easy_loss_further(self):
        logits = torch.tensor([[5.0]])
        targets = torch.tensor([[1.0]])

        loss_gamma2 = BinaryFocalLossWithLogits(gamma=2.0, alpha=-1.0)(logits, targets)
        loss_gamma5 = BinaryFocalLossWithLogits(gamma=5.0, alpha=-1.0)(logits, targets)

        assert loss_gamma5.item() < loss_gamma2.item()


# ---------------------------------------------------------------------------
# Scenario 3 — Alpha weighting
# ---------------------------------------------------------------------------

class TestAlphaWeighting:
    def test_alpha_upweights_positive_class(self):
        logits = torch.tensor([[0.0]])
        pos_target = torch.tensor([[1.0]])

        loss_alpha_high = BinaryFocalLossWithLogits(gamma=0.0, alpha=0.75)(logits, pos_target)
        loss_alpha_low = BinaryFocalLossWithLogits(gamma=0.0, alpha=0.25)(logits, pos_target)

        assert loss_alpha_high.item() > loss_alpha_low.item()

    def test_alpha_downweights_negative_class(self):
        logits = torch.tensor([[0.0]])
        neg_target = torch.tensor([[0.0]])

        loss_alpha_high = BinaryFocalLossWithLogits(gamma=0.0, alpha=0.75)(logits, neg_target)
        loss_alpha_low = BinaryFocalLossWithLogits(gamma=0.0, alpha=0.25)(logits, neg_target)

        assert loss_alpha_high.item() < loss_alpha_low.item()


# ---------------------------------------------------------------------------
# Scenario 4 — Reduction modes and basic properties
# ---------------------------------------------------------------------------

class TestReductionAndProperties:
    def test_sum_equals_mean_times_n(self):
        torch.manual_seed(2)
        logits = torch.randn(6, 4)
        targets = torch.randint(0, 2, (6, 4)).float()

        mean_loss = BinaryFocalLossWithLogits(reduction="mean")(logits, targets)
        sum_loss = BinaryFocalLossWithLogits(reduction="sum")(logits, targets)

        assert sum_loss.item() == pytest.approx(mean_loss.item() * logits.numel(), rel=1e-4)

    def test_none_reduction_shape(self):
        logits = torch.randn(3, 7)
        targets = torch.randint(0, 2, (3, 7)).float()
        loss = BinaryFocalLossWithLogits(reduction="none")(logits, targets)
        assert loss.shape == (3, 7)

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError):
            BinaryFocalLossWithLogits(reduction="bogus")

    def test_loss_always_nonnegative(self):
        torch.manual_seed(3)
        logits = torch.randn(10, 963) * 5  # wide range, like real logits
        targets = torch.randint(0, 2, (10, 963)).float()
        loss = BinaryFocalLossWithLogits()(logits, targets)
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_backward_no_error(self):
        logits = torch.randn(4, 5, requires_grad=True)
        targets = torch.randint(0, 2, (4, 5)).float()
        loss = BinaryFocalLossWithLogits()(logits, targets)
        loss.backward()  # must not raise
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_integer_target_dtype_accepted(self):
        # Real label tensors in this repo are sometimes int32/uint8; the
        # loss must accept them without dtype errors (as run_val_epoch's
        # BCEWithLogitsLoss usage does today).
        logits = torch.randn(2, 3)
        targets = torch.randint(0, 2, (2, 3)).to(torch.int32)
        loss = BinaryFocalLossWithLogits()(logits, targets)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Scenario 5 — Label smoothing (Milestone 9e, matches Keras
# BinaryFocalCrossentropy(label_smoothing=0.2) with apply_class_balancing=False)
# ---------------------------------------------------------------------------

def _logit(p: float) -> float:
    """Inverse sigmoid, to construct a logit that yields exactly probability p."""
    import math
    return math.log(p / (1 - p))


class TestLabelSmoothing:
    def test_default_label_smoothing_is_zero_and_backward_compatible(self):
        # Milestone 9d instantiated BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25)
        # without a label_smoothing kwarg -- must be numerically unchanged.
        torch.manual_seed(0)
        logits = torch.randn(6, 4)
        targets = torch.randint(0, 2, (6, 4)).float()
        without_kwarg = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25)(logits, targets)
        with_explicit_zero = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25, label_smoothing=0.0)(logits, targets)
        assert without_kwarg.item() == pytest.approx(with_explicit_zero.item())

    def test_confident_positive_matches_hand_derivation(self):
        # target=1, p=0.9, label_smoothing=0.2, alpha disabled, gamma=2.0.
        # Hand-derived (see chat writeup): smoothed target=0.9, p_t=0.82,
        # bce=0.32508297..., focal_factor=(1-0.82)^2=0.0324 -> L=0.01053269
        logits = torch.tensor([[_logit(0.9)]])
        targets = torch.tensor([[1.0]])
        loss = BinaryFocalLossWithLogits(gamma=2.0, alpha=-1.0, label_smoothing=0.2)(logits, targets)
        assert loss.item() == pytest.approx(0.010532689, abs=1e-6)

    def test_confident_negative_matches_hand_derivation(self):
        # target=0, p=0.1 -- symmetric to the positive case above under
        # label_smoothing=0.2 (both map to p_t=0.82), so the loss must match.
        logits = torch.tensor([[_logit(0.1)]])
        targets = torch.tensor([[0.0]])
        loss = BinaryFocalLossWithLogits(gamma=2.0, alpha=-1.0, label_smoothing=0.2)(logits, targets)
        assert loss.item() == pytest.approx(0.010532689, abs=1e-6)

    def test_harder_positive_matches_hand_derivation(self):
        # target=1, p=0.3 (harder/less confident). Hand-derived: smoothed
        # target=0.9, p_t=0.34, bce=1.11924302..., focal=(0.66)^2=0.4356
        # -> L=0.48754226
        logits = torch.tensor([[_logit(0.3)]])
        targets = torch.tensor([[1.0]])
        loss = BinaryFocalLossWithLogits(gamma=2.0, alpha=-1.0, label_smoothing=0.2)(logits, targets)
        assert loss.item() == pytest.approx(0.48754226, abs=1e-5)

    def test_harder_negative_matches_hand_derivation_and_symmetry(self):
        # target=0, p=0.7 -- symmetric to the harder-positive case above.
        logits = torch.tensor([[_logit(0.7)]])
        targets = torch.tensor([[0.0]])
        loss = BinaryFocalLossWithLogits(gamma=2.0, alpha=-1.0, label_smoothing=0.2)(logits, targets)
        assert loss.item() == pytest.approx(0.48754226, abs=1e-5)

    def test_smoothing_changes_targets_before_p_t(self):
        # With smoothing off, target=1 exactly at p=0.9 gives p_t=0.9 (not 0.82).
        # This confirms smoothing is actually applied, not a no-op.
        logits = torch.tensor([[_logit(0.9)]])
        targets = torch.tensor([[1.0]])
        loss_smoothed = BinaryFocalLossWithLogits(gamma=2.0, alpha=-1.0, label_smoothing=0.2)(logits, targets)
        loss_unsmoothed = BinaryFocalLossWithLogits(gamma=2.0, alpha=-1.0, label_smoothing=0.0)(logits, targets)
        assert loss_smoothed.item() != pytest.approx(loss_unsmoothed.item())
        # Unsmoothed case: p_t=0.9, focal=(0.1)^2=0.01, bce=-ln(0.9)=0.10536051565782628
        assert loss_unsmoothed.item() == pytest.approx(0.10536051565782628 * 0.01, abs=1e-8)

    def test_no_alpha_weighting_applied_when_disabled_even_with_smoothing(self):
        # alpha=-1 must mean NO class weighting is applied, regardless of
        # label_smoothing -- this is the exact PolyLLM-author configuration
        # (apply_class_balancing=False). Verify by checking that positive
        # and negative losses at symmetric confidence are equal: if alpha
        # weighting were accidentally still active (e.g. old M9d default
        # alpha=0.25), positive-class loss would be ~1/3 of negative-class
        # loss (alpha_t=0.25 vs 0.75) and this symmetry would break.
        pos_logits = torch.tensor([[_logit(0.9)]])
        pos_targets = torch.tensor([[1.0]])
        neg_logits = torch.tensor([[_logit(0.1)]])
        neg_targets = torch.tensor([[0.0]])
        loss_fn = BinaryFocalLossWithLogits(gamma=2.0, alpha=-1.0, label_smoothing=0.2)
        assert loss_fn(pos_logits, pos_targets).item() == pytest.approx(
            loss_fn(neg_logits, neg_targets).item(), abs=1e-6
        )

    def test_alpha_still_applies_when_explicitly_enabled_alongside_smoothing(self):
        # Sanity check that alpha and label_smoothing are independent knobs:
        # enabling alpha=0.25 alongside smoothing breaks the pos/neg symmetry
        # verified above. Implementation note: this class applies label
        # smoothing to `targets` BEFORE computing alpha_t (matching Keras'
        # order of operations, where label smoothing is applied once at the
        # top and reused everywhere), so alpha_t itself uses the SMOOTHED
        # targets: alpha_t_pos = 0.25*0.9 + 0.75*0.1 = 0.3,
        # alpha_t_neg = 0.25*0.1 + 0.75*0.9 = 0.7 -- a ratio of 3/7, not the
        # "hard-target" 0.25/0.75 = 1/3 one might naively expect. This
        # combined alpha+smoothing interaction is never exercised by the
        # PolyLLM authors' actual run (apply_class_balancing=False), so it
        # is an implementation choice, not a paper-verified behavior.
        pos_logits = torch.tensor([[_logit(0.9)]])
        pos_targets = torch.tensor([[1.0]])
        neg_logits = torch.tensor([[_logit(0.1)]])
        neg_targets = torch.tensor([[0.0]])
        loss_fn = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25, label_smoothing=0.2)
        pos_loss = loss_fn(pos_logits, pos_targets).item()
        neg_loss = loss_fn(neg_logits, neg_targets).item()
        assert pos_loss < neg_loss
        assert pos_loss == pytest.approx(neg_loss * (0.3 / 0.7), rel=1e-4)

    def test_gamma_2_focusing_still_correct_with_smoothing(self):
        # Directly verify the (1 - p_t)^2 factor using the smoothed p_t,
        # independent of the bce term, by comparing gamma=2 against gamma=0
        # (which strips the focal factor entirely) at the same inputs.
        logits = torch.tensor([[_logit(0.3)]])
        targets = torch.tensor([[1.0]])
        loss_gamma0 = BinaryFocalLossWithLogits(gamma=0.0, alpha=-1.0, label_smoothing=0.2)(logits, targets)
        loss_gamma2 = BinaryFocalLossWithLogits(gamma=2.0, alpha=-1.0, label_smoothing=0.2)(logits, targets)
        # p_t = 0.34 (hand-derived above) -> (1 - p_t)^2 = 0.4356
        assert (loss_gamma2 / loss_gamma0).item() == pytest.approx(0.4356, abs=1e-6)

    def test_label_smoothing_out_of_range_raises(self):
        with pytest.raises(ValueError):
            BinaryFocalLossWithLogits(label_smoothing=1.0)
        with pytest.raises(ValueError):
            BinaryFocalLossWithLogits(label_smoothing=-0.1)

    def test_backward_no_error_with_smoothing(self):
        logits = torch.randn(4, 5, requires_grad=True)
        targets = torch.randint(0, 2, (4, 5)).float()
        loss = BinaryFocalLossWithLogits(gamma=2.0, alpha=-1.0, label_smoothing=0.2)(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
