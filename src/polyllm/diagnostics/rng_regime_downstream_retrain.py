"""
GNN gap diagnostic 2 (part 2) & 6 — RNG-regime downstream-effect comparison.

Diagnostic 2's mechanism check (rng_reseed_comparison.py) already proved,
with no training involved, that the authors' per-forward `set_seed(12)`
reseed makes BOTH dropout masks (via the torch-RNG component of the reseed)
AND `negative_sampling()`'s sampled edges (via the random-module component --
`torch.manual_seed(42)` alone, which is what the authors call a second time
immediately before negative sampling, turned out to have NO effect on it,
since PyG's `negative_sampling` draws from Python's `random.sample()`, not
torch's RNG) IDENTICAL across every repeated call within a run. This script
now tests item 6's request directly: does that RNG behavior materially
change what the model actually learns over a few epochs, as a controlled
one-variable comparison against this repo's normal (non-reseeding) training?

Both regimes share: identical data, identical `DEFAULT_CONFIG` (unchanged
from train_gnn.py -- hidden_channels=64, lr=0.01, Adam, BCE), identical
model-initialization seed, identical number of epochs, identical fixed test
edge sample (reusing `representation_diagnostics.build_fixed_eval_sample` /
`measure_representation` so the numbers are directly comparable to
Diagnostic 4's already-collected "repo" trajectory). The ONLY variable that
differs is what happens to the RNG immediately before each batch's forward
pass and negative sampling.

"repo" regime = train_gnn.py's actual unmodified behavior (seed once at run
start, no reseeding inside the loop) -- reruns a short version of it here
rather than re-reading Diagnostic 4's CSV, so both regimes in this
comparison come from the exact same script/conditions.

"authors" regime = a full reseed (`set_seed(12)`: random/numpy/torch/cuda)
applied immediately before EVERY batch's negative sampling + forward call --
a literal, inline-only reproduction of the authors' `Model.forward`
behavior, collapsed to a single reseed call per batch since Diagnostic 2
already showed the authors' second `torch.manual_seed(42)` call is a no-op
for negative_sampling (which depends on Python's `random` module, already
reset by the `random.seed()` component of `set_seed(12)`). NEVER imported
into or used by src/polyllm/models/gnn.py or train_gnn.py.

Run from the project root:
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/rng_regime_downstream_retrain.py
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/rng_regime_downstream_retrain.py --epochs 3 --sample-size 5000
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.graph.build_graph import EDGE_TYPE  # noqa: E402
from polyllm.models.gnn import assemble_supervision_edges, build_global_positive_edge_set  # noqa: E402
from polyllm.train_gnn import (  # noqa: E402
    DEFAULT_CONFIG,
    GRAPH_PATH,
    build_link_loader,
    build_model,
    load_graph_split,
)
from polyllm.diagnostics.representation_diagnostics import (  # noqa: E402
    SAMPLE_SEED,
    build_fixed_eval_sample,
    measure_representation,
)

OUTPUT_DIR = Path("outputs/polyllm/gnn_diagnostics")
RESULT_CSV = OUTPUT_DIR / "rng_regime_downstream_comparison.csv"

DEFAULT_EPOCHS = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNN gap diagnostic 2/6 — RNG regime downstream-effect comparison.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--sample-size", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["random_seed"])
    return p.parse_args()


def authors_set_seed(seed: int) -> None:
    """Literal translation of the authors' full-reseed set_seed(), used
    ONLY inline in this diagnostic -- never imported into production code."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_train_epoch(model, loader, global_pos_edges, optimizer, device, rng_regime: str) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if rng_regime == "authors":
            # Single reseed collapsing the authors' set_seed(12) [governs
            # both dropout via torch RNG and negative_sampling via the
            # random module] -- their separate torch.manual_seed(42) before
            # negative sampling is omitted as a confirmed no-op (Diagnostic 2).
            authors_set_seed(12)

        edge_label_index, edge_label = assemble_supervision_edges(batch, global_pos_edges, device)

        x_dict = {"pdrugs": batch["pdrugs"].x, "seffect": batch["seffect"].x}
        node_id_dict = {"pdrugs": batch["pdrugs"].node_id, "seffect": batch["seffect"].node_id}
        pred = model(x_dict, batch.edge_index_dict, node_id_dict, edge_label_index)

        loss = F.binary_cross_entropy_with_logits(pred, edge_label)
        loss.backward()
        optimizer.step()

        batch_n = edge_label.size(0)
        total_loss += loss.item() * batch_n
        total_examples += batch_n

    return total_loss / total_examples


def run_regime(regime: str, config: dict, args, full_data, train_data, test_data, global_pos_edges,
                eval_edge_label_index, eval_edge_label, test_x_dict, test_edge_index_dict, test_node_id_dict) -> list[dict]:
    print(f"\n{'=' * 70}\nRNG regime: {regime}\n{'=' * 70}")

    device = torch.device("cpu")
    torch.manual_seed(config["random_seed"])
    model = build_model(full_data, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_loader = build_link_loader(
        train_data, config["train_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )

    rows: list[dict] = []

    def measure(epoch: int) -> None:
        rep = measure_representation(model, test_x_dict, test_edge_index_dict, test_node_id_dict, eval_edge_label_index, eval_edge_label)
        rep["epoch"] = epoch
        rep["rng_regime"] = regime
        rows.append(rep)
        print(
            f"  epoch {epoch}  dot_AUC={rep['auc_dot_product']:.4f}  "
            f"norm_AUC={rep['auc_norm_product']:.4f}  cos_AUC={rep['auc_cosine']:.4f}  "
            f"cos(pos)={rep['mean_cosine_positive']:+.4f}  cos(neg)={rep['mean_cosine_negative']:+.4f}"
        )

    measure(0)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_train_epoch(model, train_loader, global_pos_edges, optimizer, device, regime)
        dt = time.time() - t0
        print(f"Epoch {epoch} ({dt:.1f}s) train_loss={train_loss:.4f}")
        measure(epoch)

    return rows


def main() -> None:
    args = parse_args()
    config = dict(DEFAULT_CONFIG)  # UNCHANGED official hyperparameters
    config["random_seed"] = args.seed
    config["max_epochs"] = args.epochs

    print(f"Loading graph split from {GRAPH_PATH} ...")
    saved = load_graph_split()
    full_data, train_data, test_data = saved["full"], saved["train"], saved["test"]
    global_pos_edges = build_global_positive_edge_set(full_data)

    print("Building fixed-seed evaluation edge sample ...")
    eval_edge_label_index, eval_edge_label = build_fixed_eval_sample(
        test_data, full_data, args.sample_size, SAMPLE_SEED
    )
    test_x_dict = {"pdrugs": test_data["pdrugs"].x, "seffect": test_data["seffect"].x}
    test_edge_index_dict = test_data.edge_index_dict
    test_node_id_dict = {"pdrugs": test_data["pdrugs"].node_id, "seffect": test_data["seffect"].node_id}

    all_rows: list[dict] = []
    for regime in ("repo", "authors"):
        rows = run_regime(
            regime, config, args, full_data, train_data, test_data, global_pos_edges,
            eval_edge_label_index, eval_edge_label, test_x_dict, test_edge_index_dict, test_node_id_dict,
        )
        all_rows.extend(rows)

    fields = [
        "rng_regime", "epoch", "auc_dot_product", "auc_norm_product", "auc_cosine",
        "mean_cosine_positive", "mean_cosine_negative",
        "pdrugs_embedding_norm_mean", "pdrugs_embedding_norm_std",
        "seffect_embedding_norm_mean", "seffect_embedding_norm_std",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved -> {RESULT_CSV}")


if __name__ == "__main__":
    main()
