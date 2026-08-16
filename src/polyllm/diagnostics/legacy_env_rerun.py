"""
GNN gap diagnostic 7 — rerun the completely UNCHANGED reproduction under the
paper's stated software versions (PyTorch 2.4.1 + PyTorch Geometric 2.6.1,
vs. this repo's normal 2.12.1 / 2.8.0.post1 -- see notes/deviations_from_paper.md
Sec 1.4, "One newly-discovered, unresolved environment difference").

This script must be run with the LEGACY venv's interpreter
(.venv-polyllm-legacy, which has torch==2.4.1+cpu, torch_geometric==2.6.1,
pyg-lib==0.4.0+pt24cpu installed):

    .venv-polyllm-legacy/Scripts/python.exe src/polyllm/diagnostics/legacy_env_rerun.py

NO architecture, hyperparameter, loss, decoder, or training-loop changes are
made here -- every function called (build_hetero_data/split_graph from
graph/build_graph.py; GNNLinkPredictor/assemble_supervision_edges/
build_global_positive_edge_set from models/gnn.py; DEFAULT_CONFIG/
build_model/build_link_loader/EarlyStopping/run_train_epoch/run_eval_epoch
from train_gnn.py; collect_test_predictions/compute_test_metrics/
compare_to_paper from evaluate_gnn.py) is imported UNCHANGED from the
official modules. Per explicit instruction: no gradient clipping, no
learning-rate experiments, no longer training, no normalized aggregation,
no decoder changes -- this is a pure software-environment swap, nothing else.

The only reason this isn't simply "run train_gnn.py/evaluate_gnn.py
directly" is that those scripts hardcode their output paths to
outputs/polyllm/gnn/, which must not be touched (that's the official
Milestone 10 result). This script calls the same underlying functions with
explicit, isolated output paths instead -- build_graph.py's
run_graph_pipeline() and evaluate_gnn.py's helper functions already accept
path parameters for exactly this purpose; train_gnn.py's EarlyStopping also
takes an explicit checkpoint path.

The graph itself is rebuilt fresh (not loaded from the existing
data/graph/gnn_link_split.pt, which was pickled under the newer
torch_geometric version and might not unpickle cleanly here) from the same,
environment-independent source feature/label arrays
(chemberta_pair_embeddings.npy, bert_side_effect_embeddings.npy,
polyllm_labels.npy) -- these are plain NumPy arrays, not PyG objects, so
there is no cross-version compatibility concern for them.

Outputs (fully isolated from the official run):
    outputs/polyllm/gnn_legacy_env/gnn_link_split.pt
    outputs/polyllm/gnn_legacy_env/gnn_graph_audit.json
    outputs/polyllm/gnn_legacy_env/checkpoints/best_model.pt
    outputs/polyllm/gnn_legacy_env/training_history.csv
    outputs/polyllm/gnn_legacy_env/training_config.json
    outputs/polyllm/gnn_legacy_env/test_metrics.json
"""

from __future__ import annotations

import json
import platform
import sys
import time
import typing
from pathlib import Path

import pandas as pd
import torch
import torch_geometric

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import polyllm.graph.build_graph as bg  # noqa: E402
from polyllm.models.gnn import (  # noqa: E402
    GNNEncoder,
    assemble_supervision_edges,
    build_global_positive_edge_set,
)

# --- Legacy-toolchain-only compatibility shim, NOT a code/architecture change ---
# src/polyllm/models/gnn.py uses `from __future__ import annotations` (PEP 563),
# which stores GNNEncoder.forward's type hints as unevaluated strings. Under
# torch==2.4.1, torch_geometric.nn.to_hetero's FX-based code generation
# resolves those strings into fresh `typing`/`torch.Tensor` objects during
# tracing that are structurally equal but not `is`-identical to the ones it
# registers earlier, and torch/fx/graph.py's `add_global()` asserts strict
# object identity -- raising AssertionError before any model code runs.
# Reproduced from a minimal `to_hetero` + type-annotated forward() + PEP 563
# repro (confirmed the bug disappears if type annotations are absent).
# Fix: resolve the postponed annotations into real objects ONCE, here, before
# to_hetero touches them -- a pure Python type-introspection step with zero
# effect on any tensor computation, weights, or forward-pass behavior. Not
# applied to the actual gnn.py file, only monkeypatched on the imported class
# within this diagnostic script.
GNNEncoder.forward.__annotations__ = typing.get_type_hints(GNNEncoder.forward)
from polyllm.train_gnn import (  # noqa: E402
    DEFAULT_CONFIG,
    EarlyStopping,
    build_link_loader,
    build_model,
    run_eval_epoch,
    run_train_epoch,
    set_seeds,
)
from polyllm.evaluate_gnn import (  # noqa: E402
    PAPER_TARGET,
    collect_test_predictions,
    compare_to_paper,
    compute_test_metrics,
)

OUTPUT_DIR = Path("outputs/polyllm/gnn_legacy_env")
GRAPH_OUTPUT = OUTPUT_DIR / "gnn_link_split.pt"
GRAPH_AUDIT_OUTPUT = OUTPUT_DIR / "gnn_graph_audit.json"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints/best_model.pt"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
CONFIG_PATH = OUTPUT_DIR / "training_config.json"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"

OFFICIAL_RESULT_PATH = Path("outputs/polyllm/gnn/test_metrics.json")


def main() -> None:
    print(f"torch {torch.__version__}  torch_geometric {torch_geometric.__version__}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    # --- Step 1: rebuild the graph fresh under this environment ------------
    print(f"\n=== Building graph (isolated output: {GRAPH_OUTPUT}) ===")
    audit = bg.run_graph_pipeline(
        graph_output=GRAPH_OUTPUT,
        audit_output=GRAPH_AUDIT_OUTPUT,
        seed=bg.RANDOM_SEED,
        overwrite=True,
    )
    print(
        f"pdrugs={audit['node_types']['pdrugs']['count']}  "
        f"seffect={audit['node_types']['seffect']['count']}  "
        f"positive_edges={audit['total_positive_edges_pdrugs_to_seffect']}"
    )

    saved = torch.load(GRAPH_OUTPUT, weights_only=False)
    full_data, train_data, val_data, test_data = saved["full"], saved["train"], saved["val"], saved["test"]

    # --- Step 2: train -- UNCHANGED official recipe -------------------------
    config = dict(DEFAULT_CONFIG)  # completely unmodified
    device = torch.device("cpu")
    print(f"\n=== Training (unchanged DEFAULT_CONFIG, device={device}) ===")
    print(f"Config: {config}")

    set_seeds(config["random_seed"])
    global_pos_edges = build_global_positive_edge_set(full_data)
    model = build_model(full_data, config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_loader = build_link_loader(
        train_data, config["train_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )
    val_loader = build_link_loader(
        val_data, config["eval_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )

    early_stop = EarlyStopping(patience=config["patience"], path=CHECKPOINT_PATH)
    history: list[dict] = []

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        train_loss = run_train_epoch(model, train_loader, global_pos_edges, optimizer, device)
        val_loss = run_eval_epoch(model, val_loader, global_pos_edges, device)
        improved = early_stop.step(val_loss, epoch, model)
        dt = time.time() - t0

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "improved": improved, "seconds": round(dt, 1)})
        marker = " <- best" if improved else ""
        print(f"  Epoch {epoch:2d}/{config['max_epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({dt:.1f}s){marker}")

        if early_stop.early_stop:
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    full_config = {
        **config,
        "total_parameters": n_params,
        "device": str(device),
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
        "provenance": "Diagnostic 7 -- legacy environment rerun, unchanged architecture/hyperparameters.",
    }
    CONFIG_PATH.write_text(json.dumps(full_config, indent=2))
    print(f"Best epoch: {early_stop.best_epoch}  val_loss={early_stop.val_loss_min:.4f}")

    # --- Step 3: evaluate -- UNCHANGED official metric formulas -------------
    print("\n=== Evaluating on the (freshly-built) test split ===")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    test_loader = build_link_loader(
        test_data, config["eval_batch_size"], config["num_neighbors"], config["shuffle_loaders"]
    )
    y_true, y_pred = collect_test_predictions(model, test_loader, global_pos_edges, device)
    print(f"Collected {len(y_true)} test edges ({int(y_true.sum())} positive, {int((y_true == 0).sum())} negative).")

    metrics = compute_test_metrics(y_true, y_pred)
    comparison = compare_to_paper(metrics)

    print("\n=== Legacy-environment test-set results ===")
    for key in ("auc", "auprc", "ap_at_50"):
        c = comparison[key]
        print(f"  {key.upper():8s}  repro={c['reproduction']:.4f}  paper={c['paper_mean']:.4f}+/-{c['paper_std']:.4f}  ({c['std_devs_from_paper']:+.2f} std devs)")

    result = {
        "metrics": metrics,
        "paper_comparison": comparison,
        "n_test_edges": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
        "checkpoint": str(CHECKPOINT_PATH),
        "training_config": full_config,
        "software_versions": full_config["software_versions"],
        "provenance": (
            "Diagnostic 7 (notes/gnn_gap_diagnostics.md). Same architecture, "
            "hyperparameters, data, and evaluation formulas as the official "
            "outputs/polyllm/gnn/ run -- ONLY the PyTorch/PyTorch-Geometric "
            "version differs (2.4.1/2.6.1 here, matching the paper's stated "
            "Sec 3.1 versions, vs 2.12.1/2.8.0.post1 in the official run)."
        ),
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved -> {METRICS_PATH}")

    # --- Step 4: compare directly against the official (non-legacy) result -
    if OFFICIAL_RESULT_PATH.exists():
        official = json.loads(OFFICIAL_RESULT_PATH.read_text())["metrics"]
        print("\n=== Legacy env vs. official (current-versions) run ===")
        for key in ("auc", "auprc", "ap_at_50"):
            print(f"  {key.upper():8s}  legacy={metrics[key]:.4f}  official={official[key]:.4f}  diff={metrics[key]-official[key]:+.4f}")


if __name__ == "__main__":
    main()
