"""
Milestone 10f — GNN link-prediction evaluation entry point.

Official run (after train_gnn.py has produced a checkpoint):
    .venv-polyllm/Scripts/python.exe src/polyllm/evaluate_gnn.py

Loads the Milestone 10e checkpoint, runs it over the held-out test split
(Milestone 10c), and computes the three edge-level metrics the authors
report in their Table 5 — traced verbatim from `src/graph/eval.py` and
`src/functions.py` (fetched character-for-character 2026-08-11; see the
gnn-authors-source-verified memory for full quotes):

    AUC   = sklearn.metrics.roc_auc_score(y_true, raw_logits)
    AUPRC = sklearn.metrics.average_precision_score(y_true, raw_logits)
    AP@50 = polyllm.metrics.edge_level_average_precision_at_k(y_true, raw_logits, k=50)

All three operate on RAW LOGITS (no sigmoid) over the FLATTENED concatenation
of every test batch's positive supervision edges + freshly sampled negatives
— not the MLP path's per-label macro framing. Ranking metrics (AUC/AUPRC/AP@k)
are invariant to monotonic transforms, so raw logits vs. sigmoid probabilities
make no difference to the result; the authors' code uses raw logits directly.

Deviation from the literal authors' code: the authors' own `eval.py` also
computes a second "AUC2" via `torchmetrics.BinaryAUROC` purely as a redundant
cross-check of the same ROC-AUC already computed by sklearn (same metric, two
libraries) — not a distinct metric. We report ONE authoritative AUC (sklearn,
already a repo dependency) rather than adding `torchmetrics` as a new
dependency just to duplicate it. Test-time forward pass uses the base
(non-hard-negative, non-`integrate`) path — consistent with Milestone 10d/10e
scoping the authors' optional `manual_negative`/`integrate=True` paths out as
auxiliary to the Table 5 headline metric.

Paper Table 5 target (DeepChem ChemBERTa row — this repo's exact backbone):
    AUC 0.9228 +/- 0.0039, AUPRC 0.8944 +/- 0.0025, AP@50 0.9599 +/- 0.0044
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from sklearn.metrics import average_precision_score, roc_auc_score

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.graph.build_graph import EDGE_TYPE
from polyllm.metrics import edge_level_average_precision_at_k
from polyllm.models.gnn import assemble_supervision_edges, build_global_positive_edge_set
from polyllm.train_gnn import GRAPH_PATH, build_link_loader, build_model, load_graph_split

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------

TRAIN_OUTPUT_DIR = Path("outputs/polyllm/gnn")
CHECKPOINT_PATH  = TRAIN_OUTPUT_DIR / "checkpoints/best_model.pt"
CONFIG_PATH      = TRAIN_OUTPUT_DIR / "training_config.json"

METRICS_OUTPUT_PATH = TRAIN_OUTPUT_DIR / "test_metrics.json"

# Paper Table 5, DeepChem ChemBERTa row (this repo's exact backbone)
PAPER_TARGET = {
    "auc":   {"mean": 0.9228, "std": 0.0039},
    "auprc": {"mean": 0.8944, "std": 0.0025},
    "ap_at_50": {"mean": 0.9599, "std": 0.0044},
}


# ---------------------------------------------------------------------------
# Test-set inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_test_predictions(
    model,
    loader,
    global_pos_edges: set[tuple[int, int]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over every test batch and concatenate raw-logit predictions
    and ground-truth labels into two flat 1D arrays, mirroring the authors'
    `test_preds`/`test_ground_truths` accumulation in `graph.py`.
    """
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        batch = batch.to(device)

        edge_label_index, edge_label = assemble_supervision_edges(batch, global_pos_edges, device)

        x_dict = {"pdrugs": batch["pdrugs"].x, "seffect": batch["seffect"].x}
        node_id_dict = {"pdrugs": batch["pdrugs"].node_id, "seffect": batch["seffect"].node_id}
        pred = model(x_dict, batch.edge_index_dict, node_id_dict, edge_label_index)

        all_preds.append(pred.cpu())
        all_labels.append(edge_label.cpu())

    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = torch.cat(all_labels, dim=0).numpy()
    return y_true, y_pred


def compute_test_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "auprc": float(average_precision_score(y_true, y_pred)),
        "ap_at_50": float(edge_level_average_precision_at_k(y_true, y_pred, k=50)),
    }


def compare_to_paper(metrics: dict) -> dict:
    comparison = {}
    for key, value in metrics.items():
        target = PAPER_TARGET[key]
        std_devs = (value - target["mean"]) / target["std"]
        comparison[key] = {
            "reproduction": value,
            "paper_mean": target["mean"],
            "paper_std": target["std"],
            "std_devs_from_paper": std_devs,
        }
    return comparison


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(checkpoint_path: Path = CHECKPOINT_PATH, config_path: Path = CONFIG_PATH) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run train_gnn.py first."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading graph split from {GRAPH_PATH} ...")
    saved = load_graph_split()
    full_data, test_data = saved["full"], saved["test"]
    print(f"Test supervision edges: {test_data[EDGE_TYPE].edge_label_index.size(1)}")

    global_pos_edges = build_global_positive_edge_set(full_data)

    model = build_model(full_data, config, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"Loaded checkpoint: {checkpoint_path}")

    test_loader = build_link_loader(
        test_data, config["eval_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )

    print("Running inference over the test set ...")
    y_true, y_pred = collect_test_predictions(model, test_loader, global_pos_edges, device)
    print(f"Collected {len(y_true)} test edges ({int(y_true.sum())} positive, {int((y_true == 0).sum())} negative).")

    metrics = compute_test_metrics(y_true, y_pred)
    comparison = compare_to_paper(metrics)

    print("\n=== Test-set results (edge-level link prediction) ===")
    for key in ("auc", "auprc", "ap_at_50"):
        c = comparison[key]
        print(
            f"  {key.upper():8s}  repro={c['reproduction']:.4f}  "
            f"paper={c['paper_mean']:.4f}+/-{c['paper_std']:.4f}  "
            f"({c['std_devs_from_paper']:+.2f} std devs)"
        )

    result = {
        "metrics": metrics,
        "paper_comparison": comparison,
        "n_test_edges": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
        "checkpoint": str(checkpoint_path),
        "training_config": config,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
        "provenance": (
            "AUC/AUPRC/AP@50 formulas traced verbatim from the authors' src/graph/eval.py "
            "and src/functions.py::average_precision_at_k on 2026-08-11. Test-time forward "
            "uses the base path (no manual_negative hard-negative sampling, no integrate=True "
            "elementwise-product path) — both scoped out as auxiliary to the Table 5 headline "
            "metric, consistent with Milestones 10d/10e."
        ),
    }

    METRICS_OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults saved -> {METRICS_OUTPUT_PATH}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the trained GNN on the test split (Milestone 10f).")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--config",     type=Path, default=CONFIG_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(checkpoint_path=args.checkpoint, config_path=args.config)


if __name__ == "__main__":
    main()
