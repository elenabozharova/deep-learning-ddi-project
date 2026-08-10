"""
Milestone 9d/9e — Binary focal loss for multi-label classification.

Milestone 9d shipped this with literature-default gamma=2.0, alpha=0.25
(Lin et al., 2017), since the PolyLLM paper names "Binary Focal Cross
Entropy" without specifying hyperparameters. Milestone 9e then inspected
the PolyLLM authors' actual source (src/mlp/MLPModel.py at
github.com/sadrahkm/PolyLLM) and found their exact call:

    losses.BinaryFocalCrossentropy(label_smoothing=0.2)

with every other Keras default left unchanged, critically including
`apply_class_balancing=False` -- meaning NO alpha weighting is ever
applied in the authors' run, despite alpha=0.25 being the nominal Keras
default value. `label_smoothing` support was added below (default 0.0,
fully backward compatible with Milestone 9d's existing alpha=0.25 usage)
to allow reproducing the authors' exact formula: gamma=2.0,
label_smoothing=0.2, alpha disabled (pass alpha=-1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLossWithLogits(nn.Module):
    """Multi-label binary focal loss, operating on raw logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = p if target == 1 else (1 - p), p = sigmoid(logits),
    and alpha_t = alpha if target == 1 else (1 - alpha).

    If label_smoothing > 0, targets are transformed BEFORE computing p_t,
    the cross-entropy term, and (if enabled) alpha_t -- matching Keras'
    BinaryFocalCrossentropy order of operations exactly:

        targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing

    e.g. label_smoothing=0.2 maps target 1 -> 0.9 and target 0 -> 0.1.

    Reduces to plain BCEWithLogitsLoss when gamma=0, alpha<0 (disabled),
    and label_smoothing=0: (1 - p_t)^0 == 1 for every element, no alpha
    weighting is applied, and targets are untouched.

    Args:
        gamma: focusing parameter; down-weights easy (well-classified)
            examples. gamma=0 recovers unweighted cross-entropy.
        alpha: weight for the positive class in [0, 1]; pass a negative
            value (e.g. -1) to disable alpha weighting entirely -- this is
            required to match the PolyLLM authors' actual configuration
            (apply_class_balancing=False), NOT the alpha=0.25 default used
            in Milestone 9d.
        label_smoothing: soft-target smoothing factor in [0, 1); 0.0
            (default) reproduces Milestone 9d's original hard-target
            behavior. The PolyLLM authors use 0.2.
        reduction: 'mean' (default, matches BCEWithLogitsLoss), 'sum', or
            'none' (per-element loss, no reduction).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction!r}")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * (1 - p_t).pow(self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
