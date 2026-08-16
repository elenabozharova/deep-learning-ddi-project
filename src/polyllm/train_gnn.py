"""
Milestone 10e — GNN link-prediction training entry point.

Official run:
    .venv-polyllm/Scripts/python.exe src/polyllm/train_gnn.py

Engineering smoke test (2 epochs, first 2 000 train / 512 val supervision edges):
    .venv-polyllm/Scripts/python.exe src/polyllm/train_gnn.py --smoke-test

After training completes, run the separate evaluator (Milestone 10f):
    .venv-polyllm/Scripts/python.exe src/polyllm/evaluate_gnn.py

Loads the graph + split built in Milestone 10c (`data/graph/gnn_link_split.pt`)
and trains the Milestone 10d `GNNLinkPredictor`.

Hyperparameters (from the authors' `params.py::settings['model']['gnn']` and
`graph.py`, traced verbatim 2026-08-11 — see the gnn-authors-source-verified
memory for full quotes):
    hidden_channels=64, lr=0.01, epochs=10, Adam, BCEWithLogitsLoss,
    EarlyStopping(patience=2, monitors mean validation loss),
    LinkNeighborLoader(num_neighbors=[20, 10]),
    train batch_size=65536, val/test batch_size=2048, shuffle=False for all.

Deviations from the literal authors' code (full reasoning in the
gnn-authors-source-verified memory):
- No `accelerate` — plain device-agnostic PyTorch, consistent with every
  other script in this repo.
- No per-forward-pass RNG reseeding (see models/gnn.py docstring) — seeded
  once at run start instead.
- No per-epoch AUC/AUPRC/AP@50 logging during training — matches this repo's
  established train/evaluate split (e.g. train_chemberta_mlp.py vs.
  evaluate_chemberta_mlp.py); final metrics belong to Milestone 10f.
- No `manual_negative`/`dangerous_seffects_ids` hard-negative path (scoped
  out — auxiliary to the paper's Table 5 headline metric).
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import LinkNeighborLoader

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.graph.build_graph import EDGE_TYPE
from polyllm.models.gnn import (
    GNNLinkPredictor,
    assemble_supervision_edges,
    build_global_positive_edge_set,
)

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------

GRAPH_PATH      = Path("data/graph/gnn_link_split.pt")
OUTPUT_DIR      = Path("outputs/polyllm/gnn")
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints/best_model.pt"
HISTORY_PATH    = OUTPUT_DIR / "training_history.csv"
CONFIG_PATH     = OUTPUT_DIR / "training_config.json"
SMOKE_OUTPUT_DIR = OUTPUT_DIR / "smoke_test"

# ---------------------------------------------------------------------------
# Default training configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "random_seed":       42,
    "hidden_channels":   64,
    "optimizer":         "Adam",
    "learning_rate":      0.01,
    "loss":              "BCEWithLogitsLoss",
    "max_epochs":        10,
    "patience":           2,
    "num_neighbors":     [20, 10],
    "train_batch_size":  65536,
    "eval_batch_size":    2048,
    "shuffle_loaders":   False,  # authors' graph.py explicitly overrides link_loader()'s own shuffle=True default
    "provenance": (
        "Hyperparameters from the PolyLLM authors' params.py::settings['model']['gnn'] "
        "and graph.py, traced verbatim 2026-08-11."
    ),
}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_geometric.seed_everything(seed)


# ---------------------------------------------------------------------------
# EarlyStopping — mirrors the authors' EarlyStopping.py semantics
# (score = -val_loss; save checkpoint on improvement; stop after `patience`
# non-improving epochs), made device-agnostic (torch.save/torch.load already
# were) and with no other changes.
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int, path: Path, delta: float = 0.0) -> None:
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def step(self, val_loss: float, epoch: int, model: torch.nn.Module) -> bool:
        """Returns True iff this epoch produced a new best checkpoint."""
        score = -val_loss
        improved = self.best_score is None or score > self.best_score + self.delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.val_loss_min = val_loss
            self.counter = 0
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return improved


# ---------------------------------------------------------------------------
# Loader construction — mirrors the authors' link_loader(), except shuffle
# is an explicit required argument rather than a default that graph.py's own
# call site always overrides anyway.
# ---------------------------------------------------------------------------

def build_link_loader(
    data,
    batch_size: int,
    num_neighbors: list[int],
    shuffle: bool,
) -> LinkNeighborLoader:
    edge_label_index = data[EDGE_TYPE].edge_label_index
    edge_label = data[EDGE_TYPE].edge_label

    return LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        edge_label_index=(EDGE_TYPE, edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ---------------------------------------------------------------------------
# Epoch loops
# ---------------------------------------------------------------------------

def run_train_epoch(
    model: GNNLinkPredictor,
    loader: LinkNeighborLoader,
    global_pos_edges: set[tuple[int, int]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

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


@torch.no_grad()
def run_eval_epoch(
    model: GNNLinkPredictor,
    loader: LinkNeighborLoader,
    global_pos_edges: set[tuple[int, int]],
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)

        edge_label_index, edge_label = assemble_supervision_edges(batch, global_pos_edges, device)

        x_dict = {"pdrugs": batch["pdrugs"].x, "seffect": batch["seffect"].x}
        node_id_dict = {"pdrugs": batch["pdrugs"].node_id, "seffect": batch["seffect"].node_id}
        pred = model(x_dict, batch.edge_index_dict, node_id_dict, edge_label_index)

        loss = F.binary_cross_entropy_with_logits(pred, edge_label)

        batch_n = edge_label.size(0)
        total_loss += loss.item() * batch_n
        total_examples += batch_n

    return total_loss / total_examples


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_graph_split(graph_path: Path = GRAPH_PATH) -> dict[str, Any]:
    return torch.load(graph_path, weights_only=False)


def _derive_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "checkpoint": output_dir / "checkpoints/best_model.pt",
        "history": output_dir / "training_history.csv",
        "config": output_dir / "training_config.json",
    }


def build_model(full_data, config: dict, device: torch.device) -> GNNLinkPredictor:
    model = GNNLinkPredictor(
        num_pdrugs=full_data["pdrugs"].num_nodes,
        num_seffect=full_data["seffect"].num_nodes,
        pdrugs_input_dim=full_data["pdrugs"].x.shape[1],
        hidden_channels=config["hidden_channels"],
        metadata=full_data.metadata(),
        seffect_input_dim=full_data["seffect"].x.shape[1],
    ).to(device)
    return model


def run_full_training(
    config: dict,
    overwrite: bool = False,
    graph_path: Path = GRAPH_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    paths = _derive_output_paths(output_dir)
    checkpoint_path, history_path, config_path = paths["checkpoint"], paths["history"], paths["config"]

    if checkpoint_path.exists() and not overwrite:
        print(f"Checkpoint already exists at {checkpoint_path}.\nPass --overwrite to re-train.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed = config["random_seed"]
    set_seeds(seed)

    print(f"Loading graph split from {graph_path} ...")
    saved = load_graph_split(graph_path)
    full_data, train_data, val_data = saved["full"], saved["train"], saved["val"]
    print(
        f"pdrugs={full_data['pdrugs'].num_nodes}  seffect={full_data['seffect'].num_nodes}  "
        f"train supervision={train_data[EDGE_TYPE].edge_label_index.size(1)}  "
        f"val supervision={val_data[EDGE_TYPE].edge_label_index.size(1)}"
    )

    global_pos_edges = build_global_positive_edge_set(full_data)

    model = build_model(full_data, config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: GNNLinkPredictor  hidden_channels={config['hidden_channels']}  parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_loader = build_link_loader(
        train_data, config["train_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )
    val_loader = build_link_loader(
        val_data, config["eval_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )

    full_config = {
        **config,
        "total_parameters": n_params,
        "device": str(device),
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
    }
    config_path.write_text(json.dumps(full_config, indent=2))
    print(f"Config saved -> {config_path}")

    early_stop = EarlyStopping(patience=config["patience"], path=checkpoint_path)
    history: list[dict] = []

    print(f"\nTraining for up to {config['max_epochs']} epochs (patience={config['patience']}) ...")

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        train_loss = run_train_epoch(model, train_loader, global_pos_edges, optimizer, device)
        val_loss = run_eval_epoch(model, val_loader, global_pos_edges, device)
        improved = early_stop.step(val_loss, epoch, model)
        dt = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "improved": improved, "seconds": round(dt, 1),
        })

        marker = " <- best" if improved else ""
        print(
            f"  Epoch {epoch:2d}/{config['max_epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"({dt:.1f}s){marker}"
        )

        if early_stop.early_stop:
            print(f"\nEarly stopping triggered after epoch {epoch} (no improvement for {config['patience']} epochs).")
            break

    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nHistory saved -> {history_path}")
    print(f"Best epoch: {early_stop.best_epoch}  val_loss={early_stop.val_loss_min:.4f}")
    print(f"Checkpoint saved -> {checkpoint_path}")
    print("\nTraining complete. Run evaluate_gnn.py for test-set metrics.")


# ---------------------------------------------------------------------------
# Smoke test — small edge subset, 2 epochs, correctness checks only
# ---------------------------------------------------------------------------

def run_smoke_test(config: dict) -> None:
    print("\n=== ENGINEERING SMOKE TEST (GNN) ===")
    print(f"Output directory: {SMOKE_OUTPUT_DIR}")
    SMOKE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed = config["random_seed"]
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading graph split from {GRAPH_PATH} ...")
    saved = load_graph_split()
    full_data, train_data, val_data = saved["full"], saved["train"], saved["val"]

    # Shrink supervision edges only (message-passing graph stays full-size,
    # matching how a real mini-batch would still see the full neighborhood).
    n_train_smoke, n_val_smoke = 2000, 512
    train_small = train_data.clone()
    train_small[EDGE_TYPE].edge_label_index = train_data[EDGE_TYPE].edge_label_index[:, :n_train_smoke]
    train_small[EDGE_TYPE].edge_label = train_data[EDGE_TYPE].edge_label[:n_train_smoke]
    val_small = val_data.clone()
    val_small[EDGE_TYPE].edge_label_index = val_data[EDGE_TYPE].edge_label_index[:, :n_val_smoke]
    val_small[EDGE_TYPE].edge_label = val_data[EDGE_TYPE].edge_label[:n_val_smoke]

    global_pos_edges = build_global_positive_edge_set(full_data)

    model = build_model(full_data, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_loader = build_link_loader(train_small, batch_size=512, num_neighbors=config["num_neighbors"], shuffle=False)
    val_loader = build_link_loader(val_small, batch_size=512, num_neighbors=config["num_neighbors"], shuffle=False)

    smoke_ckpt = SMOKE_OUTPUT_DIR / "smoke_checkpoint.pt"
    early_stop = EarlyStopping(patience=2, path=smoke_ckpt)

    last_val_loss = float("nan")
    for epoch in range(1, 3):
        param_snapshot = next(model.parameters()).data.clone()

        train_loss = run_train_epoch(model, train_loader, global_pos_edges, optimizer, device)

        param_after = next(model.parameters()).data
        assert not torch.equal(param_snapshot, param_after), \
            f"Epoch {epoch}: training did not update model parameters."
        assert np.isfinite(train_loss), f"Epoch {epoch}: non-finite train loss."

        val_loss = run_eval_epoch(model, val_loader, global_pos_edges, device)
        assert np.isfinite(val_loss), f"Epoch {epoch}: non-finite val loss."

        early_stop.step(val_loss, epoch, model)
        last_val_loss = val_loss
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    assert smoke_ckpt.exists(), "Checkpoint file was not written."

    # Checkpoint load + reproducibility
    model2 = build_model(full_data, config, device)
    model2.load_state_dict(torch.load(smoke_ckpt, weights_only=True))
    model2.eval()

    val_loss2 = run_eval_epoch(model2, val_loader, global_pos_edges, device)
    print(f"  Reloaded checkpoint val_loss={val_loss2:.4f} (best was {early_stop.val_loss_min:.4f})")

    print("\nSMOKE TEST PASSED")
    print("  OK forward pass (train + eval)")
    print("  OK BCE loss finite")
    print("  OK backward pass and optimizer update")
    print("  OK checkpoint write")
    print("  OK checkpoint load")
    print(f"\nNote: smoke-test metrics are NOT scientific results.")
    print(f"      {n_train_smoke} train / {n_val_smoke} val supervision edges, 2 epochs only.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the GNN link predictor (Milestone 10e).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run engineering smoke test only (2 epochs, small edge subset).")
    p.add_argument("--seed",            type=int,   default=DEFAULT_CONFIG["random_seed"])
    p.add_argument("--hidden-channels", type=int,   default=DEFAULT_CONFIG["hidden_channels"])
    p.add_argument("--lr",              type=float, default=DEFAULT_CONFIG["learning_rate"])
    p.add_argument("--epochs",          type=int,   default=DEFAULT_CONFIG["max_epochs"])
    p.add_argument("--patience",        type=int,   default=DEFAULT_CONFIG["patience"])
    p.add_argument("--overwrite",       action="store_true",
                   help="Overwrite an existing checkpoint.")
    p.add_argument("--graph-path",      type=Path,  default=GRAPH_PATH,
                   help="Graph split .pt to train on (only the seffect.x source should differ across runs).")
    p.add_argument("--output-dir",      type=Path,  default=OUTPUT_DIR,
                   help="Directory for checkpoint/history/config (keep distinct per side-effect encoder).")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg["random_seed"] = args.seed
    cfg["hidden_channels"] = args.hidden_channels
    cfg["learning_rate"] = args.lr
    cfg["max_epochs"] = args.epochs
    cfg["patience"] = args.patience
    return cfg


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.smoke_test:
        run_smoke_test(config)
    else:
        run_full_training(
            config, overwrite=args.overwrite,
            graph_path=args.graph_path, output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
