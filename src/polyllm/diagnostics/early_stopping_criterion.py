"""
GNN gap diagnostic — checkpoint-selection criterion (loss vs. AUC), motivated
by a historical finding in the PolyLLM authors' git history.

Commit 7855566 ("Updated Early Stopping to stop by loss", 2025-01-10T05:05:44Z)
changed `EarlyStopping.__call__` to expect validation LOSS (score=-val_loss,
lower loss better) instead of validation AUC (score=val_auc, higher AUC
better). Verified by diffing that commit against its parent b3d4fc57 and
against the very next commit that touches graph.py, 21bef01b
(2025-01-10T05:07:09Z, 84 seconds later):

  - parent (b3d4fc57):  EarlyStopping expects val_auc, score=val_auc.
                         graph.py: earlystopping(val_auc1, model)   -- consistent
  - 7855566:             EarlyStopping expects val_loss, score=-val_loss.
                         graph.py: earlystopping(val_auc1, model)   -- INCONSISTENT
                         (graph.py is byte-identical to the parent commit --
                         only EarlyStopping.py changed here)
  - 21bef01b:            EarlyStopping expects val_loss, score=-val_loss.
                         graph.py: earlystopping(val_total_loss / val_total_examples, model)
                         -- consistent again

So no historical snapshot in the real history ever ran a training loop with
the broken AUC-as-loss combination live -- the inconsistency existed for 84
seconds of commit history, never executed. In BOTH the parent and 7855566,
graph.py's `if earlystopping.early_stop: model.load_state_dict(...)` only
restores the saved checkpoint when patience is actually exhausted; if
training runs to max_epochs without triggering early_stop, the LAST epoch's
weights are used for testing, not the best-checkpoint ones.

This diagnostic asks a narrower, still-relevant question: on OUR frozen
reproduction's exact training trajectory (same seed, same graph split, same
config, same negative-sampling implementation -- see DEFAULT_CONFIG import
below, UNCHANGED), does selecting the checkpoint by best validation AUC
instead of best validation loss produce a materially different test result?

Design (single training run, two checkpoint trackers, per Phase 3 of the
request):
  - The STOPPING rule is left identical to the official run: loss-based
    EarlyStopping(patience=2), reusing train_gnn.EarlyStopping unchanged --
    this also writes the exact same "best-loss" checkpoint the official run
    produces.
  - A second, non-stopping tracker records the epoch with the best
    validation AUC seen so far and saves its own checkpoint. It never
    affects when training stops.
  - A checkpoint is additionally saved at every epoch (including epoch 0 =
    initialization) so Phase 6 ("did the model ever look good before
    collapsing?") can be answered from checkpoints that already exist,
    without a second training run.
  - Train/val AUC/AUPRC per epoch reuse the SAME predictions produced by the
    per-batch forward pass already computed for the loss (no extra forward
    pass) -- mirrors the authors' own graph.py, which accumulates
    train_preds/val_preds inside the same loop that computes the loss.

Does NOT touch outputs/polyllm/gnn/ (the official, frozen result) or modify
train_gnn.py / evaluate_gnn.py / models/gnn.py. All outputs live under
outputs/polyllm/gnn_diagnostics/early_stopping/.

Run from the project root:
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/early_stopping_criterion.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.diagnostics.representation_diagnostics import (  # noqa: E402
    SAMPLE_SEED,
    build_fixed_eval_sample,
    measure_representation,
)
from polyllm.evaluate_gnn import collect_test_predictions, compute_test_metrics  # noqa: E402
from polyllm.graph.build_graph import EDGE_TYPE  # noqa: E402
from polyllm.metrics import edge_level_average_precision_at_k  # noqa: E402
from polyllm.models.gnn import assemble_supervision_edges, build_global_positive_edge_set  # noqa: E402
from polyllm.train_gnn import (  # noqa: E402
    DEFAULT_CONFIG,
    GRAPH_PATH,
    EarlyStopping,
    build_link_loader,
    build_model,
    load_graph_split,
    set_seeds,
)

OUTPUT_DIR = Path("outputs/polyllm/gnn_diagnostics/early_stopping")
EPOCH_CKPT_DIR = OUTPUT_DIR / "epoch_checkpoints"
BEST_LOSS_CKPT = OUTPUT_DIR / "checkpoint_best_loss.pt"
BEST_AUC_CKPT = OUTPUT_DIR / "checkpoint_best_auc.pt"
TRAJECTORY_CSV = OUTPUT_DIR / "trajectory.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"


# ---------------------------------------------------------------------------
# Epoch loops -- mirror train_gnn.run_train_epoch / run_eval_epoch exactly,
# with the addition of accumulating predictions (already computed for the
# loss) so AUC/AUPRC can be reported for the same epoch with no extra pass.
# ---------------------------------------------------------------------------

def _edge_metrics(y_true: np.ndarray, y_pred: np.ndarray, with_ap50: bool) -> dict:
    m = {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "auprc": float(average_precision_score(y_true, y_pred)),
    }
    if with_ap50:
        m["ap_at_50"] = float(edge_level_average_precision_at_k(y_true, y_pred, k=50))
    return m


def run_train_epoch_tracked(model, loader, global_pos_edges, optimizer, device) -> dict:
    model.train()
    total_loss = total_examples = 0
    all_preds, all_labels = [], []

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
        all_preds.append(pred.detach().cpu())
        all_labels.append(edge_label.detach().cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    metrics = _edge_metrics(y_true, y_pred, with_ap50=False)
    metrics["loss"] = total_loss / total_examples
    return metrics


@torch.no_grad()
def run_eval_epoch_tracked(model, loader, global_pos_edges, device) -> dict:
    model.eval()
    total_loss = total_examples = 0
    all_preds, all_labels = [], []

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
        all_preds.append(pred.cpu())
        all_labels.append(edge_label.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    metrics = _edge_metrics(y_true, y_pred, with_ap50=True)
    metrics["loss"] = total_loss / total_examples
    return metrics


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    config = dict(DEFAULT_CONFIG)  # UNCHANGED official hyperparameters/seed
    EPOCH_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # matches the official run (frozen result was also CPU)
    print(f"Device: {device}")

    set_seeds(config["random_seed"])

    print(f"Loading graph split from {GRAPH_PATH} ...")
    saved = load_graph_split()
    full_data, train_data, val_data, test_data = saved["full"], saved["train"], saved["val"], saved["test"]

    global_pos_edges = build_global_positive_edge_set(full_data)

    model = build_model(full_data, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_loader = build_link_loader(
        train_data, config["train_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )
    val_loader = build_link_loader(
        val_data, config["eval_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )

    # loss-based stopping rule -- IDENTICAL mechanism/checkpoint as the official run
    early_stop = EarlyStopping(patience=config["patience"], path=BEST_LOSS_CKPT)
    best_auc_epoch = None
    best_auc_value = -float("inf")

    torch.save(model.state_dict(), EPOCH_CKPT_DIR / "epoch_0.pt")

    history: list[dict] = []
    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        train_m = run_train_epoch_tracked(model, train_loader, global_pos_edges, optimizer, device)
        val_m = run_eval_epoch_tracked(model, val_loader, global_pos_edges, device)
        loss_improved = early_stop.step(val_m["loss"], epoch, model)
        dt = time.time() - t0

        torch.save(model.state_dict(), EPOCH_CKPT_DIR / f"epoch_{epoch}.pt")

        auc_improved = val_m["auc"] > best_auc_value
        if auc_improved:
            best_auc_value = val_m["auc"]
            best_auc_epoch = epoch
            torch.save(model.state_dict(), BEST_AUC_CKPT)

        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"], "train_auc": train_m["auc"], "train_auprc": train_m["auprc"],
            "val_loss": val_m["loss"], "val_auc": val_m["auc"], "val_auprc": val_m["auprc"],
            "val_ap_at_50": val_m["ap_at_50"],
            "best_loss_so_far": loss_improved, "best_auc_so_far": auc_improved,
            "seconds": round(dt, 1),
        })
        print(
            f"  Epoch {epoch:2d}/{config['max_epochs']}  "
            f"train_loss={train_m['loss']:.4f} train_auc={train_m['auc']:.4f}  "
            f"val_loss={val_m['loss']:.4f} val_auc={val_m['auc']:.4f}  "
            f"({dt:.1f}s)"
            f"{' <-loss' if loss_improved else ''}{' <-auc' if auc_improved else ''}"
        )

        if early_stop.early_stop:
            print(f"\nEarly stopping triggered after epoch {epoch} (loss-based, patience={config['patience']}).")
            break

    last_epoch = history[-1]["epoch"]
    best_loss_epoch = early_stop.best_epoch
    # ES-BUGGY-HISTORICAL sanity check (Phase 8): commit 7855566's inconsistency
    # is equivalent to minimizing val_auc -- derive it from the same trajectory,
    # no extra training.
    min_auc_epoch = min(history, key=lambda r: r["val_auc"])["epoch"]

    print(f"\nBest-loss epoch (official mechanism): {best_loss_epoch}")
    print(f"Best-AUC epoch (diagnostic tracker):   {best_auc_epoch}")
    print(f"Min-AUC epoch (ES-BUGGY-HISTORICAL):   {min_auc_epoch}")

    df = pd.DataFrame(history)
    df.to_csv(TRAJECTORY_CSV, index=False)
    print(f"\nTrajectory saved -> {TRAJECTORY_CSV}")

    # -----------------------------------------------------------------------
    # Phase 6 -- test AUC at every saved epoch checkpoint (official mini-
    # batched evaluate_gnn.py protocol, identical frozen test set).
    # -----------------------------------------------------------------------
    print("\nEvaluating every epoch checkpoint on the official test split ...")
    test_loader = build_link_loader(
        test_data, config["eval_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )
    per_epoch_test = []
    for epoch in range(0, last_epoch + 1):
        ckpt_path = EPOCH_CKPT_DIR / f"epoch_{epoch}.pt"
        eval_model = build_model(full_data, config, device)
        eval_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        y_true, y_pred = collect_test_predictions(eval_model, test_loader, global_pos_edges, device)
        test_metrics = compute_test_metrics(y_true, y_pred)
        per_epoch_test.append({"epoch": epoch, **{f"test_{k}": v for k, v in test_metrics.items()}})
        print(f"  epoch {epoch:2d}  test_auc={test_metrics['auc']:.4f}  test_auprc={test_metrics['auprc']:.4f}")

    per_epoch_test_df = pd.DataFrame(per_epoch_test)
    per_epoch_test_path = OUTPUT_DIR / "per_epoch_test_metrics.csv"
    per_epoch_test_df.to_csv(per_epoch_test_path, index=False)
    print(f"Saved -> {per_epoch_test_path}")

    max_test_auc_row = per_epoch_test_df.loc[per_epoch_test_df["test_auc"].idxmax()]

    # -----------------------------------------------------------------------
    # Phase 5 -- evaluate best-loss / best-AUC / min-AUC checkpoints on the
    # identical frozen test set, official protocol.
    # -----------------------------------------------------------------------
    def eval_checkpoint(path: Path) -> dict:
        m = build_model(full_data, config, device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        y_true, y_pred = collect_test_predictions(m, test_loader, global_pos_edges, device)
        return compute_test_metrics(y_true, y_pred)

    test_best_loss = eval_checkpoint(BEST_LOSS_CKPT)
    test_best_auc = eval_checkpoint(BEST_AUC_CKPT)
    test_min_auc = eval_checkpoint(EPOCH_CKPT_DIR / f"epoch_{min_auc_epoch}.pt")

    def row(criterion, epoch, test_m):
        val_row = next(r for r in history if r["epoch"] == epoch)
        return {
            "criterion": criterion, "selected_epoch": epoch,
            "val_loss": val_row["val_loss"], "val_auc": val_row["val_auc"],
            "test_auc": test_m["auc"], "test_auprc": test_m["auprc"], "test_ap_at_50": test_m["ap_at_50"],
        }

    comparison_table = [
        row("ES-LOSS (current/official)", best_loss_epoch, test_best_loss),
        row("ES-AUC (historical, pre-7855566)", best_auc_epoch, test_best_auc),
        row("ES-BUGGY-HISTORICAL (commit 7855566, sanity check only)", min_auc_epoch, test_min_auc),
    ]
    print("\n=== Checkpoint comparison (identical frozen test set) ===")
    for r in comparison_table:
        print(f"  {r['criterion']:55s} epoch={r['selected_epoch']:>2}  test_auc={r['test_auc']:.4f}  test_auprc={r['test_auprc']:.4f}")

    # -----------------------------------------------------------------------
    # Phase 7 -- geometry comparison at best-loss vs. best-AUC epoch, reusing
    # representation_diagnostics' existing fixed-sample + scoring utilities.
    # -----------------------------------------------------------------------
    print("\nComputing embedding-geometry diagnostics at best-loss / best-AUC epochs ...")
    eval_edge_label_index, eval_edge_label = build_fixed_eval_sample(test_data, full_data, 20_000, SAMPLE_SEED)
    test_x_dict = {"pdrugs": test_data["pdrugs"].x, "seffect": test_data["seffect"].x}
    test_edge_index_dict = test_data.edge_index_dict
    test_node_id_dict = {"pdrugs": test_data["pdrugs"].node_id, "seffect": test_data["seffect"].node_id}

    geometry = {}
    for label, epoch in (("best_loss", best_loss_epoch), ("best_auc", best_auc_epoch)):
        m = build_model(full_data, config, device)
        m.load_state_dict(torch.load(EPOCH_CKPT_DIR / f"epoch_{epoch}.pt", map_location=device, weights_only=True))
        geometry[label] = {
            "epoch": epoch,
            **measure_representation(m, test_x_dict, test_edge_index_dict, test_node_id_dict, eval_edge_label_index, eval_edge_label),
        }
        print(f"  {label} (epoch {epoch}): dot_AUC={geometry[label]['auc_dot_product']:.4f}  "
              f"cos(pos)={geometry[label]['mean_cosine_positive']:+.4f}  cos(neg)={geometry[label]['mean_cosine_negative']:+.4f}")

    # -----------------------------------------------------------------------
    # Phase 9 -- classification
    # -----------------------------------------------------------------------
    paper_auc = 0.9228
    loss_auc = test_best_loss["auc"]
    auc_auc = test_best_auc["auc"]
    delta = auc_auc - loss_auc

    if auc_auc >= 0.80:
        classification = "A: strong explanation (best-AUC checkpoint approaches the paper result)"
    elif delta >= 0.15:
        classification = "B: partial explanation (materially improves but remains far from the paper)"
    elif delta >= 0.03:
        classification = "C: small effect"
    else:
        classification = "D: no effect / worse"

    summary = {
        "git_history": {
            "commit_785556660a62": {
                "message": "Updated Early Stopping to stop by loss",
                "date": "2025-01-10T05:05:44Z",
                "parent": "b3d4fc578e201bf005072f358c4de223368abdaf",
                "only_file_changed": "src/graph/EarlyStopping.py",
                "graph_py_unchanged_vs_parent": True,
            },
            "version_A_parent_b3d4fc57": {
                "EarlyStopping_call_signature": "__call__(self, val_auc, model)",
                "score_formula": "score = val_auc",
                "higher_is_better": True,
                "graph_py_call": "earlystopping(val_auc1, model)",
                "internally_consistent": True,
                "best_checkpoint_restored_before_test": "only if early_stop actually triggers (patience exhausted); otherwise last epoch's weights are used",
            },
            "version_B_785556660a62": {
                "EarlyStopping_call_signature": "__call__(self, val_loss, model)",
                "score_formula": "score = -val_loss",
                "higher_is_better": False,
                "graph_py_call": "earlystopping(val_auc1, model)  (unchanged from parent -- graph.py not touched by this commit)",
                "internally_consistent": False,
                "mathematical_meaning": (
                    "score = -val_auc1. Since the algorithm saves a checkpoint / resets patience "
                    "whenever score increases, and score increases exactly when val_auc1 DECREASES, "
                    "this configuration checkpoints on DECREASING AUC -- it favors the worst-performing "
                    "epochs, the opposite of the historical intent."
                ),
                "ever_executed_in_a_real_training_run": "Unknown/unlikely -- superseded 84 seconds later by the next graph.py commit; not verifiable from git history alone.",
            },
            "version_C_21bef01b": {
                "commit": "21bef01bc44fd0837b0dffeae2aedb54e8ce56ba",
                "date": "2025-01-10T05:07:09Z",
                "seconds_after_785556660a62": 85,
                "graph_py_call": "earlystopping(val_total_loss / val_total_examples, model)  (old val_auc1 line commented out, not deleted)",
                "internally_consistent": True,
                "matches_current_reproduction": True,
            },
        },
        "current_baseline": {
            "config": config,
            "graph_path": str(GRAPH_PATH),
            "note": "outputs/polyllm/gnn/ untouched by this diagnostic",
        },
        "trajectory_summary": {
            "epochs_run": last_epoch,
            "best_loss_epoch": best_loss_epoch,
            "best_loss_value": early_stop.val_loss_min,
            "best_auc_epoch": best_auc_epoch,
            "best_auc_value": best_auc_value,
            "min_auc_epoch (ES-BUGGY-HISTORICAL)": min_auc_epoch,
        },
        "max_test_auc_across_all_epochs": {
            "epoch": int(max_test_auc_row["epoch"]),
            "test_auc": float(max_test_auc_row["test_auc"]),
            "far_below_paper": bool(max_test_auc_row["test_auc"] < 0.92 - 3 * 0.0039),
        },
        "checkpoint_comparison": comparison_table,
        "geometry_at_selected_epochs": geometry,
        "classification": classification,
        "classification_inputs": {"paper_auc": paper_auc, "best_loss_test_auc": loss_auc, "best_auc_test_auc": auc_auc, "delta": delta},
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved -> {SUMMARY_JSON}")
    print(f"\nClassification: {classification}")


if __name__ == "__main__":
    main()
