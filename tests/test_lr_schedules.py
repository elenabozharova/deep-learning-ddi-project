"""Tests for keras_exponential_decay_lr (Milestone 9e)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polyllm.lr_schedules import keras_exponential_decay_lr


# ---------------------------------------------------------------------------
# Scenario 1 — Exact values at the user-specified checkpoints
# ---------------------------------------------------------------------------

class TestExactCheckpoints:
    def test_step_0(self):
        lr = keras_exponential_decay_lr(0, 0.005, decay_steps=1000, decay_rate=0.96)
        assert lr == pytest.approx(0.005)

    def test_step_999_still_initial(self):
        lr = keras_exponential_decay_lr(999, 0.005, decay_steps=1000, decay_rate=0.96)
        assert lr == pytest.approx(0.005)

    def test_step_1000_first_decay(self):
        lr = keras_exponential_decay_lr(1000, 0.005, decay_steps=1000, decay_rate=0.96)
        assert lr == pytest.approx(0.005 * 0.96)

    def test_step_1999_still_one_decay(self):
        lr = keras_exponential_decay_lr(1999, 0.005, decay_steps=1000, decay_rate=0.96)
        assert lr == pytest.approx(0.005 * 0.96)

    def test_step_2000_second_decay(self):
        lr = keras_exponential_decay_lr(2000, 0.005, decay_steps=1000, decay_rate=0.96)
        assert lr == pytest.approx(0.005 * (0.96 ** 2))


# ---------------------------------------------------------------------------
# Scenario 2 — Staircase behaviour (piecewise-constant within a block)
# ---------------------------------------------------------------------------

class TestStaircaseBehaviour:
    def test_constant_within_block(self):
        vals = [
            keras_exponential_decay_lr(s, 0.005, decay_steps=1000, decay_rate=0.96)
            for s in (1000, 1250, 1500, 1750, 1999)
        ]
        assert all(v == pytest.approx(vals[0]) for v in vals)

    def test_jumps_at_block_boundary(self):
        just_before = keras_exponential_decay_lr(2999, 0.005, decay_steps=1000, decay_rate=0.96)
        just_after = keras_exponential_decay_lr(3000, 0.005, decay_steps=1000, decay_rate=0.96)
        assert just_before != pytest.approx(just_after)
        assert just_after < just_before

    def test_third_decay_block(self):
        lr = keras_exponential_decay_lr(3500, 0.005, decay_steps=1000, decay_rate=0.96)
        assert lr == pytest.approx(0.005 * (0.96 ** 3))

    def test_non_staircase_is_continuous(self):
        # Without staircase, the exponent is not floored -- values inside a
        # "block" differ continuously rather than being piecewise-constant.
        lr_a = keras_exponential_decay_lr(1000, 0.005, decay_steps=1000, decay_rate=0.96, staircase=False)
        lr_b = keras_exponential_decay_lr(1500, 0.005, decay_steps=1000, decay_rate=0.96, staircase=False)
        assert lr_a != pytest.approx(lr_b)
        assert lr_b == pytest.approx(0.005 * (0.96 ** 1.5))


# ---------------------------------------------------------------------------
# Scenario 3 — Monotonic decay and edge cases
# ---------------------------------------------------------------------------

class TestMonotonicityAndEdgeCases:
    def test_monotonically_non_increasing(self):
        steps = [0, 500, 1000, 5000, 10000, 50000]
        lrs = [keras_exponential_decay_lr(s, 0.005) for s in steps]
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_negative_step_raises(self):
        with pytest.raises(ValueError):
            keras_exponential_decay_lr(-1, 0.005)

    def test_zero_decay_steps_raises(self):
        with pytest.raises(ValueError):
            keras_exponential_decay_lr(1000, 0.005, decay_steps=0)

    def test_matches_paper_epoch_scale_example(self):
        # Sanity-check against the hand-derived epoch-10 value used in the
        # M9d/M9e comparison writeup: ~14 decay clicks by epoch 10 at the
        # authors' true batch size (steps_per_epoch ~1607).
        step = 9 * 1607  # 9 completed epochs' worth of steps
        lr = keras_exponential_decay_lr(step, 0.005, decay_steps=1000, decay_rate=0.96)
        assert lr == pytest.approx(0.005 * (0.96 ** 14), abs=1e-6)
