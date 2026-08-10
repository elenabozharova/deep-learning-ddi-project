"""
Milestone 9e — Keras-compatible per-step exponential LR decay.

The PolyLLM authors' MLPModel.py uses:

    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_rate,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True,
    )

Keras schedules are evaluated per OPTIMIZER STEP (one call per mini-batch
gradient update), using the optimizer's own persistent step counter
(`optimizer.iterations`), not per epoch. With staircase=True the exponent
is floored, so the learning rate is piecewise-constant within blocks of
`decay_steps` steps rather than decaying continuously.

This is fundamentally different from Milestone 9d's implementation, which
used `torch.optim.lr_scheduler.ExponentialLR` stepped once per EPOCH --
see notes/deviations_from_paper.md Sec 1.8/1.9 for why that mismatch, and
the accompanying 8x-smaller batch size (32 vs. our 256 at the time), left
the effective learning rate elevated for much longer than the authors'
schedule intends.
"""
from __future__ import annotations


def keras_exponential_decay_lr(
    step: int,
    initial_learning_rate: float,
    decay_steps: int = 1000,
    decay_rate: float = 0.96,
    staircase: bool = True,
) -> float:
    """Reproduce keras.optimizers.schedules.ExponentialDecay exactly.

        lr(step) = initial_learning_rate * decay_rate ** (step / decay_steps)

    With staircase=True, (step / decay_steps) is floored before
    exponentiating, so:
        steps [0, decay_steps)              -> initial_learning_rate
        steps [decay_steps, 2*decay_steps)  -> initial_learning_rate * decay_rate
        steps [2*decay_steps, 3*decay_steps)-> initial_learning_rate * decay_rate**2
        ...

    `step` is the number of optimizer steps already completed BEFORE the
    upcoming update (i.e. the first update of training uses step=0), matching
    Keras' convention of evaluating the schedule against the optimizer's
    `iterations` counter prior to incrementing it for the current update.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if decay_steps <= 0:
        raise ValueError(f"decay_steps must be > 0, got {decay_steps}")

    exponent = step / decay_steps
    if staircase:
        exponent = exponent // 1  # floor, keeps float type like Keras' tf.floor
    return initial_learning_rate * (decay_rate ** exponent)
