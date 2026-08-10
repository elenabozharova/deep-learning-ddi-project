"""
Reusable multi-label MLP for Milestones 6B and 7.

Architecture (fixed hidden structure, BatchNorm only after layer 1):
    input → Linear(input_dim, 512) → LeakyReLU → BatchNorm1d(512) → Dropout(p)
          → Linear(512, 1024) → LeakyReLU → Dropout(p)
          → Linear(1024, 2048) → LeakyReLU → Dropout(p)
          → Linear(2048, output_dim)

Returns raw logits. No sigmoid inside the model — callers use BCEWithLogitsLoss
during training and torch.sigmoid() during inference.

The same class serves Morgan (input_dim=2048) and ChemBERTa (input_dim=384)
experiments without modifying the hidden architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultilabelMLP(nn.Module):
    """Multi-label MLP with three hidden layers and a fixed expansion schedule."""

    # PolyLLM paper, Section 2.2: "we evaluated several activation functions
    # and selected ... Leaky ReLU ... with a negative slope of 0.1 for each
    # hidden layer." Previously 0.01 (PyTorch's own default) -- corrected
    # 2026-08-08 per the paper's exact quoted text; see
    # notes/deviations_from_paper.md Sec 1.7.
    NEGATIVE_SLOPE: float = 0.1

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        negative_slope: float = NEGATIVE_SLOPE,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            # Layer 1 — only layer with BatchNorm
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            # Layer 2
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),
            # Layer 3
            nn.Linear(1024, 2048),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),
            # Output
            nn.Linear(2048, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
