"""
GNN gap diagnostics 4 & 5 — per-epoch representation diagnostics and the
semantic-projection vs. node-ID-embedding decomposition.

Diagnostic-only training: reuses train_gnn.py's build_model /
build_link_loader / DEFAULT_CONFIG / assemble_supervision_edges /
build_global_positive_edge_set completely UNCHANGED (same hidden_channels=64,
lr=0.01, Adam, BCE, EarlyStopping(patience=2), same
data/graph/gnn_link_split.pt). The only difference from train_gnn.py's
official run is instrumentation: a checkpoint is saved at EVERY epoch
(including epoch 0 = initialization, before any training step) instead of
only the best one, and at each saved epoch a battery of representation
diagnostics is computed on a fixed-seed subsample of the test split.

Does NOT touch outputs/polyllm/gnn/ (the official Milestone 10 result) --
everything here writes to outputs/polyllm/gnn_diagnostics/.

At each epoch, evaluation uses a single FULL-GRAPH (non-mini-batched)
forward pass over test_data (message-passing edges = train+val's full edge
set, per the paper's own stated test protocol, already matched by
RandomLinkSplit -- see notes/deviations_from_paper.md Sec 1.4), since
test_data["pdrugs"/"seffect"].node_id is already the identity permutation
for the full, unsplit-by-batch graph (local == global trivially, unlike a
real LinkNeighborLoader mini-batch). This mirrors the methodology already
used once in the prior root-cause investigation (see
gnn-authors-source-verified memory, "Post-10f deep-dive" section) but now
run systematically across epochs instead of only at the final checkpoint.

Diagnostic 4 measures, for the FULL lifted representation (current,
production pipeline): raw dot-product AUROC, norm-product AUROC,
cosine-similarity AUROC, mean cosine for positive vs. negative edges, and
pdrugs/seffect embedding norm mean/std.

Diagnostic 5 additionally decomposes the lifted input into three variants
using the SAME trained encoder weights at that epoch (no change to
GNNLinkPredictor/GNNEncoder needed -- `model.encoder` is called directly
with a hand-built x_dict):
  (a) full      = pdrugs_lin(x) + pdrugs_emb(node_id)   [same as Diagnostic 4]
  (b) semantic  = pdrugs_lin(x) only (node-ID embedding term zeroed)
  (c) id_only   = pdrugs_emb(node_id) only (semantic-projection term zeroed)
and reports norm-product / cosine / dot-product AUROC for each.

Run from the project root:
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/representation_diagnostics.py
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/representation_diagnostics.py --sample-size 5000  # faster
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.graph.build_graph import EDGE_TYPE  # noqa: E402
from polyllm.models.gnn import build_global_positive_edge_set  # noqa: E402
from polyllm.train_gnn import (  # noqa: E402
    DEFAULT_CONFIG,
    GRAPH_PATH,
    EarlyStopping,
    build_link_loader,
    build_model,
    load_graph_split,
    run_train_epoch,
)

OUTPUT_DIR = Path("outputs/polyllm/gnn_diagnostics")
EPOCH_CKPT_DIR = OUTPUT_DIR / "epoch_checkpoints"
REPRESENTATION_CSV = OUTPUT_DIR / "representation_by_epoch.csv"
DECOMPOSITION_CSV = OUTPUT_DIR / "decomposition_by_epoch.csv"

SAMPLE_SEED = 20260813
DEFAULT_SAMPLE_SIZE = 20_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNN gap diagnostics 4 & 5.")
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                    help="Positive AND negative edges sampled per epoch for scoring (fixed seed).")
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["random_seed"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fixed evaluation-edge sample (identical across all epochs, for comparability)
# ---------------------------------------------------------------------------

def build_fixed_eval_sample(test_data, full_data, sample_size: int, seed: int):
    pos_edge_label_index = test_data[EDGE_TYPE].edge_label_index  # (2, n_pos)
    n_pos_available = pos_edge_label_index.size(1)
    n_pos = min(sample_size, n_pos_available)

    gen = torch.Generator().manual_seed(seed)
    pos_perm = torch.randperm(n_pos_available, generator=gen)[:n_pos]
    pos_sample = pos_edge_label_index[:, pos_perm]

    global_pos_edges = build_global_positive_edge_set(full_data)
    num_pdrugs = test_data["pdrugs"].num_nodes
    num_seffect = test_data["seffect"].num_nodes

    torch.manual_seed(seed)
    neg_candidates = negative_sampling(
        edge_index=test_data[EDGE_TYPE].edge_index,
        num_nodes=(num_pdrugs, num_seffect),
        num_neg_samples=int(sample_size * 1.3),  # oversample; some will collide with true positives
        force_undirected=True,
    )
    keep_mask = torch.tensor(
        [(int(p), int(s)) not in global_pos_edges for p, s in zip(neg_candidates[0], neg_candidates[1])],
        dtype=torch.bool,
    )
    neg_sample = neg_candidates[:, keep_mask][:, :sample_size]

    edge_label_index = torch.cat([pos_sample, neg_sample], dim=-1)
    edge_label = torch.cat([
        torch.ones(pos_sample.size(1)),
        torch.zeros(neg_sample.size(1)),
    ])
    print(
        f"Fixed eval sample built: {pos_sample.size(1)} positive + "
        f"{neg_sample.size(1)} negative edges (seed={seed})."
    )
    return edge_label_index, edge_label


# ---------------------------------------------------------------------------
# Diagnostic 4: representation diagnostics for the FULL lifted representation
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_representation(model, x_dict, edge_index_dict, node_id_dict, edge_label_index, edge_label) -> dict:
    model.eval()
    encoded = model.encode(x_dict, edge_index_dict, node_id_dict)
    return _score_variant(encoded, edge_label_index, edge_label)


def _score_variant(encoded: dict, edge_label_index: torch.Tensor, edge_label: torch.Tensor) -> dict:
    pdrugs_emb = encoded["pdrugs"][edge_label_index[0]]
    seffect_emb = encoded["seffect"][edge_label_index[1]]

    dot = (pdrugs_emb * seffect_emb).sum(dim=-1)
    pdrugs_norm = pdrugs_emb.norm(dim=-1)
    seffect_norm = seffect_emb.norm(dim=-1)
    norm_product = pdrugs_norm * seffect_norm
    cosine = F.cosine_similarity(pdrugs_emb, seffect_emb, dim=-1)

    y = edge_label.numpy()
    pos_mask = edge_label.bool()

    auc_dot = float(roc_auc_score(y, dot.numpy()))
    auc_norm = float(roc_auc_score(y, norm_product.numpy()))
    auc_cosine = float(roc_auc_score(y, cosine.numpy()))

    all_pdrugs_norm = encoded["pdrugs"].norm(dim=-1)
    all_seffect_norm = encoded["seffect"].norm(dim=-1)

    return {
        "auc_dot_product": auc_dot,
        "auc_norm_product": auc_norm,
        "auc_cosine": auc_cosine,
        "mean_cosine_positive": float(cosine[pos_mask].mean().item()),
        "mean_cosine_negative": float(cosine[~pos_mask].mean().item()),
        "pdrugs_embedding_norm_mean": float(all_pdrugs_norm.mean().item()),
        "pdrugs_embedding_norm_std": float(all_pdrugs_norm.std().item()),
        "seffect_embedding_norm_mean": float(all_seffect_norm.mean().item()),
        "seffect_embedding_norm_std": float(all_seffect_norm.std().item()),
    }


# ---------------------------------------------------------------------------
# Diagnostic 5: semantic-only / id-only decomposition
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_decomposition(model, x_dict, edge_index_dict, node_id_dict, edge_label_index, edge_label) -> dict:
    model.eval()

    semantic_pdrugs = model.pdrugs_lin(x_dict["pdrugs"].float())
    id_pdrugs = model.pdrugs_emb(node_id_dict["pdrugs"])
    semantic_seffect = model.seffect_lin(x_dict["seffect"].float())
    id_seffect = model.seffect_emb(node_id_dict["seffect"])

    variants = {
        "full": {"pdrugs": semantic_pdrugs + id_pdrugs, "seffect": semantic_seffect + id_seffect},
        "semantic_only": {"pdrugs": semantic_pdrugs, "seffect": semantic_seffect},
        "id_only": {"pdrugs": id_pdrugs, "seffect": id_seffect},
    }

    results = {}
    for name, lifted in variants.items():
        encoded = model.encoder(lifted, edge_index_dict)
        scored = _score_variant(encoded, edge_label_index, edge_label)
        results[name] = {
            "auc_dot_product": scored["auc_dot_product"],
            "auc_norm_product": scored["auc_norm_product"],
            "auc_cosine": scored["auc_cosine"],
        }
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = dict(DEFAULT_CONFIG)  # UNCHANGED official hyperparameters
    config["random_seed"] = args.seed

    EPOCH_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # diagnostics run on CPU for determinism/simplicity
    print(f"Device: {device}")

    print(f"Loading graph split from {GRAPH_PATH} ...")
    saved = load_graph_split()
    full_data, train_data, val_data, test_data = saved["full"], saved["train"], saved["val"], saved["test"]

    global_pos_edges = build_global_positive_edge_set(full_data)

    print("Building fixed-seed evaluation edge sample from the test split ...")
    eval_edge_label_index, eval_edge_label = build_fixed_eval_sample(
        test_data, full_data, args.sample_size, SAMPLE_SEED
    )

    test_x_dict = {"pdrugs": test_data["pdrugs"].x, "seffect": test_data["seffect"].x}
    test_edge_index_dict = test_data.edge_index_dict
    test_node_id_dict = {"pdrugs": test_data["pdrugs"].node_id, "seffect": test_data["seffect"].node_id}

    torch.manual_seed(config["random_seed"])
    model = build_model(full_data, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_loader = build_link_loader(
        train_data, config["train_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )

    representation_rows: list[dict] = []
    decomposition_rows: list[dict] = []

    def measure_and_record(epoch: int) -> None:
        rep = measure_representation(model, test_x_dict, test_edge_index_dict, test_node_id_dict, eval_edge_label_index, eval_edge_label)
        rep["epoch"] = epoch
        representation_rows.append(rep)
        print(
            f"  epoch {epoch:2d}  dot_AUC={rep['auc_dot_product']:.4f}  "
            f"norm_AUC={rep['auc_norm_product']:.4f}  cos_AUC={rep['auc_cosine']:.4f}  "
            f"cos(pos)={rep['mean_cosine_positive']:+.4f}  cos(neg)={rep['mean_cosine_negative']:+.4f}"
        )

        decomp = measure_decomposition(model, test_x_dict, test_edge_index_dict, test_node_id_dict, eval_edge_label_index, eval_edge_label)
        for variant_name, scores in decomp.items():
            row = {"epoch": epoch, "variant": variant_name, **scores}
            decomposition_rows.append(row)
            print(
                f"    [{variant_name:13s}] dot={scores['auc_dot_product']:.4f}  "
                f"norm={scores['auc_norm_product']:.4f}  cos={scores['auc_cosine']:.4f}"
            )

    # --- epoch 0: initialization, before any training step -----------------
    print("\n=== Epoch 0 (initialization) ===")
    torch.save(model.state_dict(), EPOCH_CKPT_DIR / "epoch_0.pt")
    measure_and_record(0)

    # --- training loop, identical recipe to train_gnn.py::run_full_training,
    #     with per-epoch checkpointing + measurement added -----------------
    early_stop = EarlyStopping(patience=config["patience"], path=EPOCH_CKPT_DIR / "best_by_val_loss.pt")

    val_loader = build_link_loader(
        val_data, config["eval_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )

    from polyllm.train_gnn import run_eval_epoch  # noqa: E402 (local import, same module)

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        train_loss = run_train_epoch(model, train_loader, global_pos_edges, optimizer, device)
        val_loss = run_eval_epoch(model, val_loader, global_pos_edges, device)
        improved = early_stop.step(val_loss, epoch, model)
        dt = time.time() - t0

        torch.save(model.state_dict(), EPOCH_CKPT_DIR / f"epoch_{epoch}.pt")

        print(f"\n=== Epoch {epoch} ({dt:.1f}s, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}{' <- best' if improved else ''}) ===")
        measure_and_record(epoch)

        if early_stop.early_stop:
            print(f"\nEarly stopping triggered after epoch {epoch} (no improvement for {config['patience']} epochs).")
            break

    # --- write outputs -------------------------------------------------------
    rep_fields = [
        "epoch", "auc_dot_product", "auc_norm_product", "auc_cosine",
        "mean_cosine_positive", "mean_cosine_negative",
        "pdrugs_embedding_norm_mean", "pdrugs_embedding_norm_std",
        "seffect_embedding_norm_mean", "seffect_embedding_norm_std",
    ]
    with open(REPRESENTATION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rep_fields)
        writer.writeheader()
        writer.writerows(representation_rows)
    print(f"\nSaved -> {REPRESENTATION_CSV}")

    decomp_fields = ["epoch", "variant", "auc_dot_product", "auc_norm_product", "auc_cosine"]
    with open(DECOMPOSITION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=decomp_fields)
        writer.writeheader()
        writer.writerows(decomposition_rows)
    print(f"Saved -> {DECOMPOSITION_CSV}")


if __name__ == "__main__":
    main()
