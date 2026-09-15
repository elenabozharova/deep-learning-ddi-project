"""
GNN gap diagnostic -- FINAL combined run.

Tests the three tractable, still-untested hypotheses for the GNN
reproduction gap ALL AT ONCE, on a single training trajectory:

  1. LEGACY SOFTWARE ENVIRONMENT -- PyTorch 2.4.1 + PyTorch Geometric 2.6.1
     (the paper's Sec 3.1 versions). Must be run with .venv-polyllm-legacy.
     Diagnostic 7 tested this alone -> AUC 0.41 -> 0.60.

  2. RESTORED bn3 -- the authors' GNN.py *declares* self.bn3 / self.bn4
     (nn.BatchNorm1d) but never calls them in forward(). The official
     reproduction omitted them as confirmed-dead code. Here bn3 is inserted
     after conv3, before lin1 -- the one place the encoder currently has an
     unnormalised GraphConv output (summed over side-effect degrees of
     500-28,568) feeding straight into a linear layer. This is the single
     most likely cause of the observed training non-convergence
     (mean BCE ~120 at epoch 1, train AUC stuck at 0.59).

  3. CHECKPOINT SELECTION BY VALIDATION AUC -- the authors' git history
     shows early revisions checkpointed on best validation AUC; the released
     code (and the official reproduction) checkpoint on best validation loss.
     The paper says early stopping on "validation performance". The
     2026-08-16 checkpoint-selection diagnostic showed AUC-based selection
     reaching test AUC 0.9202 on ONE trajectory (new env, no bn3).

The run also records the epoch-0 (untrained) test metrics, to check whether
the graph-topology artefact (untrained AUC 0.87-0.94 in 3/5 seeds,
documented 2026-08-17) is still present under this combined configuration.

ISOLATION: no official file is touched. New code only in this script; new
outputs only in outputs/polyllm/gnn_final_diagnostic/. The official frozen
result (outputs/polyllm/gnn/) is read for comparison only.

Run:
    .venv-polyllm-legacy/Scripts/python.exe src/polyllm/diagnostics/gnn_final_combined.py
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import GraphConv, to_hetero

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import polyllm.graph.build_graph as bg  # noqa: E402
import polyllm.models.gnn as gnn_mod  # noqa: E402
from polyllm.models.gnn import (  # noqa: E402
    DotProductClassifier,
    assemble_supervision_edges,
    build_global_positive_edge_set,
)
from polyllm.metrics import edge_level_average_precision_at_k  # noqa: E402
from polyllm.train_gnn import (  # noqa: E402
    DEFAULT_CONFIG,
    build_link_loader,
    run_train_epoch,
    set_seeds,
)

PAPER_TARGET = {
    "auc":   {"mean": 0.9228, "std": 0.0039},
    "auprc": {"mean": 0.8944, "std": 0.0025},
    "ap_at_50": {"mean": 0.9599, "std": 0.0044},
}

BASE_DIR = Path("outputs/polyllm/gnn_final_diagnostic")
OFFICIAL_RESULT_PATH = Path("outputs/polyllm/gnn/test_metrics.json")


# ---------------------------------------------------------------------------
# Hypothesis 2: encoder with the authors' declared-but-unused bn3 restored.
# Everything else is byte-identical to models/gnn.py::GNNEncoder.
# ---------------------------------------------------------------------------

class GNNEncoderBN3(nn.Module):
    DROPOUT_P: float = 0.8

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.conv1 = GraphConv(hidden_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)  # <-- restored
        self.dropout1 = nn.Dropout(self.DROPOUT_P)
        self.dropout2 = nn.Dropout(self.DROPOUT_P)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)  # <-- the one inserted line
        x = self.lin1(x)
        return x


class GNNLinkPredictorBN3(nn.Module):
    """Identical to models/gnn.py::GNNLinkPredictor, but the encoder is
    GNNEncoderBN3. Kept as a local subclass so nothing in models/gnn.py
    is monkeypatched."""

    def __init__(self, num_pdrugs, num_seffect, pdrugs_input_dim,
                 hidden_channels, metadata, seffect_input_dim=768):
        super().__init__()
        self.pdrugs_lin = nn.Linear(pdrugs_input_dim, hidden_channels)
        self.seffect_lin = nn.Linear(seffect_input_dim, hidden_channels)
        self.pdrugs_emb = nn.Embedding(num_pdrugs, hidden_channels)
        self.seffect_emb = nn.Embedding(num_seffect, hidden_channels)
        self.encoder = to_hetero(GNNEncoderBN3(hidden_channels), metadata=metadata)
        self.classifier = DotProductClassifier()

    def encode(self, x_dict, edge_index_dict, node_id_dict):
        lifted = {
            "pdrugs": self.pdrugs_lin(x_dict["pdrugs"].float()) + self.pdrugs_emb(node_id_dict["pdrugs"]),
            "seffect": self.seffect_lin(x_dict["seffect"].float()) + self.seffect_emb(node_id_dict["seffect"]),
        }
        return self.encoder(lifted, edge_index_dict)

    def predict_edges(self, x_dict, edge_label_index, integrate=False):
        return self.classifier(x_dict["pdrugs"], x_dict["seffect"], edge_label_index, integrate)

    def forward(self, x_dict, edge_index_dict, node_id_dict, edge_label_index, integrate=False):
        x_dict = self.encode(x_dict, edge_index_dict, node_id_dict)
        return self.predict_edges(x_dict, edge_label_index, integrate)


# Legacy-toolchain FX-codegen shim (see legacy_env_rerun.py for the full
# explanation) -- pure type-introspection, zero effect on any computation.
GNNEncoderBN3.forward.__annotations__ = typing.get_type_hints(GNNEncoderBN3.forward)


# ---------------------------------------------------------------------------
# Prediction collection (mirrors evaluate_gnn.collect_test_predictions) +
# combined val loss/AUC in a single pass (consistent negatives for both).
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, loader, global_pos_edges, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        eli, el = assemble_supervision_edges(batch, global_pos_edges, device)
        x_dict = {"pdrugs": batch["pdrugs"].x, "seffect": batch["seffect"].x}
        nid = {"pdrugs": batch["pdrugs"].node_id, "seffect": batch["seffect"].node_id}
        pred = model(x_dict, batch.edge_index_dict, nid, eli)
        preds.append(pred.cpu())
        labels.append(el.cpu())
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def metrics_from(y_true, y_pred):
    return {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "auprc": float(average_precision_score(y_true, y_pred)),
        "ap_at_50": float(edge_level_average_precision_at_k(y_true, y_pred, k=50)),
    }


def vs_paper(m):
    return {
        k: {
            "value": m[k],
            "paper_mean": PAPER_TARGET[k]["mean"],
            "std_devs_from_paper": (m[k] - PAPER_TARGET[k]["mean"]) / PAPER_TARGET[k]["std"],
        }
        for k in m
    }


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="model-init / negative-sampling seed; graph split stays fixed at 42")
    args = ap.parse_args()

    out_dir = BASE_DIR if args.seed == 42 else BASE_DIR / f"seed_{args.seed}"
    graph_output = BASE_DIR / "gnn_link_split.pt"          # graph split shared, fixed seed 42
    graph_audit_output = BASE_DIR / "gnn_graph_audit.json"
    ckpt_dir = out_dir / "checkpoints"
    history_path = out_dir / "training_history.csv"
    result_path = out_dir / "results.json"

    print(f"torch {torch.__version__}   torch_geometric {torch_geometric.__version__}   seed={args.seed}")
    assert torch.__version__.startswith("2.4."), "run this with .venv-polyllm-legacy (torch 2.4.1)"

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # --- graph (rebuilt fresh under this env; split seed fixed at 42) -------
    if not graph_output.exists():
        print(f"\n=== Building graph (isolated: {graph_output}) ===")
        bg.run_graph_pipeline(graph_output=graph_output, audit_output=graph_audit_output,
                              seed=bg.RANDOM_SEED, overwrite=True)
    saved = torch.load(graph_output, weights_only=False)
    full_data, train_data, val_data, test_data = (
        saved["full"], saved["train"], saved["val"], saved["test"],
    )
    print(f"  pdrugs={full_data['pdrugs'].num_nodes}  seffect={full_data['seffect'].num_nodes}  "
          f"edges={full_data[('pdrugs','associated','seffect')].edge_index.size(1)}")

    config = dict(DEFAULT_CONFIG)
    config["random_seed"] = args.seed
    config["max_epochs"] = 10  # run the full budget to see the whole AUC trajectory
    print(f"\n=== Config ===\n{json.dumps({k: v for k, v in config.items() if k != 'provenance'}, indent=2)}")
    print("Changes vs official: legacy env + bn3 restored + val-AUC checkpoint tracking")

    CKPT_DIR = ckpt_dir  # local alias so the rest of main() reads naturally
    HISTORY_PATH = history_path
    RESULT_PATH = result_path

    set_seeds(config["random_seed"])
    global_pos_edges = build_global_positive_edge_set(full_data)

    model = GNNLinkPredictorBN3(
        num_pdrugs=full_data["pdrugs"].num_nodes,
        num_seffect=full_data["seffect"].num_nodes,
        pdrugs_input_dim=full_data["pdrugs"].x.shape[1],
        hidden_channels=config["hidden_channels"],
        metadata=full_data.metadata(),
        seffect_input_dim=full_data["seffect"].x.shape[1],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  (official: 4,256,064; +{n_params - 4256064} from bn3)")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_loader = build_link_loader(train_data, config["train_batch_size"], config["num_neighbors"], False)
    val_loader = build_link_loader(val_data, config["eval_batch_size"], config["num_neighbors"], False)

    # --- epoch 0: untrained ------------------------------------------------
    torch.save(model.state_dict(), CKPT_DIR / "epoch_00.pt")

    best_loss, best_loss_epoch = float("inf"), 0
    best_auc, best_auc_epoch = -1.0, 0
    loss_patience_counter, loss_stop_epoch = 0, None
    patience2_epoch = None
    history = []

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        train_loss = run_train_epoch(model, train_loader, global_pos_edges, optimizer, device)

        vy_true, vy_pred = collect_predictions(model, val_loader, global_pos_edges, device)
        val_loss = float(F.binary_cross_entropy_with_logits(
            torch.tensor(vy_pred), torch.tensor(vy_true).float()).item())
        val_auc = float(roc_auc_score(vy_true, vy_pred))
        dt = time.time() - t0

        loss_improved = val_loss < best_loss
        if loss_improved:
            best_loss, best_loss_epoch = val_loss, epoch
            torch.save(model.state_dict(), CKPT_DIR / "best_loss.pt")
            loss_patience_counter = 0
        else:
            loss_patience_counter += 1
            if loss_patience_counter >= config["patience"] and loss_stop_epoch is None:
                loss_stop_epoch = epoch  # where the official patience=2 rule would have stopped
                # freeze the checkpoint that rule would actually have kept (the
                # running best up to here -- NOT any later loss improvement)
                import shutil
                shutil.copy(CKPT_DIR / "best_loss.pt", CKPT_DIR / "patience2_selected.pt")
                patience2_epoch = best_loss_epoch

        auc_improved = val_auc > best_auc
        if auc_improved:
            best_auc, best_auc_epoch = val_auc, epoch
            torch.save(model.state_dict(), CKPT_DIR / "best_auc.pt")

        torch.save(model.state_dict(), CKPT_DIR / f"epoch_{epoch:02d}.pt")
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_auc": val_auc, "seconds": round(dt, 1),
            "loss_improved": loss_improved, "auc_improved": auc_improved,
        })
        print(f"  ep {epoch:2d}/{config['max_epochs']}  train_loss={train_loss:9.4f}  "
              f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  ({dt:.0f}s)"
              f"{'  <-best_loss' if loss_improved else ''}{'  <-best_auc' if auc_improved else ''}")

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    if loss_stop_epoch is None:
        loss_stop_epoch = config["max_epochs"]
    if patience2_epoch is None:  # patience never exhausted -> rule keeps the global best
        patience2_epoch = best_loss_epoch
        import shutil
        shutil.copy(CKPT_DIR / "best_loss.pt", CKPT_DIR / "patience2_selected.pt")
    print(f"\nBest val loss: epoch {best_loss_epoch} ({best_loss:.4f})")
    print(f"Best val AUC : epoch {best_auc_epoch} ({best_auc:.4f})")
    print(f"Official patience=2 (loss) rule would have stopped after epoch {loss_stop_epoch}, "
          f"selecting epoch {best_loss_epoch}")

    # --- test-set evaluation of the three checkpoints ---------------------
    test_loader = build_link_loader(test_data, config["eval_batch_size"], config["num_neighbors"], False)
    results = {}
    for name, ckpt in [
        ("epoch_0_untrained", CKPT_DIR / "epoch_00.pt"),
        ("official_patience2_rule", CKPT_DIR / "patience2_selected.pt"),
        ("global_best_val_loss", CKPT_DIR / "best_loss.pt"),
        ("best_val_auc", CKPT_DIR / "best_auc.pt"),
    ]:
        m = GNNLinkPredictorBN3(
            num_pdrugs=full_data["pdrugs"].num_nodes, num_seffect=full_data["seffect"].num_nodes,
            pdrugs_input_dim=full_data["pdrugs"].x.shape[1], hidden_channels=config["hidden_channels"],
            metadata=full_data.metadata(), seffect_input_dim=full_data["seffect"].x.shape[1],
        ).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        y_true, y_pred = collect_predictions(m, test_loader, global_pos_edges, device)
        met = metrics_from(y_true, y_pred)
        results[name] = {"metrics": met, "vs_paper": vs_paper(met),
                         "n_pos": int(y_true.sum()), "n_neg": int((y_true == 0).sum())}
        print(f"\n--- {name} (checkpoint: {ckpt.name}) ---")
        for k in ("auc", "auprc", "ap_at_50"):
            v = results[name]["vs_paper"][k]
            print(f"    {k.upper():9s} {v['value']:.4f}   paper {v['paper_mean']:.4f}   "
                  f"({v['std_devs_from_paper']:+.1f} sd)")

    official = json.loads(OFFICIAL_RESULT_PATH.read_text())["metrics"] if OFFICIAL_RESULT_PATH.exists() else None

    RESULT_PATH.write_text(json.dumps({
        "provenance": (
            "FINAL combined GNN gap diagnostic: legacy env (torch 2.4.1 / PyG 2.6.1) "
            "+ restored bn3 (authors' declared-but-unused BatchNorm after conv3) "
            "+ validation-AUC checkpoint selection. Single training trajectory. "
            "Isolated; official outputs/polyllm/gnn/ untouched."
        ),
        "config": config,
        "n_parameters": n_params,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
        "best_val_loss_epoch": best_loss_epoch,
        "best_val_auc_epoch": best_auc_epoch,
        "official_patience2_stop_epoch": loss_stop_epoch,
        "official_patience2_selected_epoch": patience2_epoch,
        "results": results,
        "official_gnn_result_for_reference": official,
        "paper_target": PAPER_TARGET,
    }, indent=2))
    print(f"\nSaved -> {RESULT_PATH}")


if __name__ == "__main__":
    main()
