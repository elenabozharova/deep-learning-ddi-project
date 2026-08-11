"""
Milestone 10c — build the bipartite (pdrugs, associated, seffect) HeteroData
graph and apply the authors' RandomLinkSplit protocol.

Protocol (traced verbatim from the PolyLLM authors' actual source,
`src/graph/helpers.py::construct_hetero_data` / `split_data` in
github.com/sadrahkm/PolyLLM)
--------
Nodes  : 'pdrugs' (one per drug pair, features = Milestone 5's ChemBERTa pair
          embeddings, 384-dim) and 'seffect' (one per side effect, features =
          Milestone 10b's BERT embeddings, 768-dim).
Edges  : ('pdrugs', 'associated', 'seffect') — one edge per positive
          (pair, side_effect) label in `polyllm_labels.npy`. `T.ToUndirected()`
          adds the reverse relation ('seffect', 'rev_associated', 'pdrugs').
Split  : `T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
          disjoint_train_ratio=0.3, neg_sampling_ratio=0.0,
          add_negative_train_samples=False,
          edge_types=('pdrugs','associated','seffect'),
          rev_edge_types=('seffect','rev_associated','pdrugs'))`
          — exact authors' parameters. `neg_sampling_ratio=0.0` means NO
          negative edges are baked into ANY split (train/val/test all carry
          only positive `edge_label`s); negatives are instead sampled fresh
          on every forward pass during training/evaluation (Milestone 10d/e).

One documented deviation from the literal authors' code, needed to make the
graph actually usable with `nn.Embedding` + `LinkNeighborLoader`:
- Authors assign `data[node_type].node_id = np.array(items.index)` (a plain
  NumPy array). `nn.Embedding.forward` requires a `LongTensor` index, and
  PyG's neighbor-sampling machinery expects tensor-typed node attributes to
  be sliceable the same way as `x`. We use `torch.arange(num_nodes,
  dtype=torch.long)` instead — the standard PyG idiom (matches PyG's own
  official heterogeneous link-prediction tutorial, which this code closely
  follows) and the only version that is actually runnable end-to-end.

Run from the project root:
    python src/polyllm/graph/build_graph.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42  # repo-wide convention (M1-M10b); authors hardcode seed=12
                   # internally via their own set_seed() — kept at this repo's
                   # standard seed since no headline metric depends on which
                   # exact seed is used, only that one is fixed and reported.

EDGE_TYPE = ("pdrugs", "associated", "seffect")
REV_EDGE_TYPE = ("seffect", "rev_associated", "pdrugs")

NUM_VAL = 0.1
NUM_TEST = 0.1
DISJOINT_TRAIN_RATIO = 0.3
NEG_SAMPLING_RATIO = 0.0  # negatives sampled dynamically per forward pass instead

DEFAULT_PDRUGS_FEATURES  = Path("data/features/chemberta_pair_embeddings.npy")
DEFAULT_SEFFECT_FEATURES = Path("data/features/bert_side_effect_embeddings.npy")
DEFAULT_LABELS_INPUT     = Path("data/processed/polyllm_labels.npy")
DEFAULT_GRAPH_OUTPUT     = Path("data/graph/gnn_link_split.pt")
DEFAULT_AUDIT_OUTPUT     = Path("outputs/polyllm/gnn_graph_audit.json")


# ---------------------------------------------------------------------------
# Step 1: Input validation
# ---------------------------------------------------------------------------

def validate_inputs(
    pdrugs_x: np.ndarray,
    seffect_x: np.ndarray,
    labels: np.ndarray,
) -> None:
    """
    Raise ValueError if the three input arrays are not mutually consistent.

    Checks:
    - labels is a 2D binary/count matrix (num_pdrugs, num_seffect)
    - pdrugs_x row count matches labels.shape[0]
    - seffect_x row count matches labels.shape[1]
    - No NaN/Inf in either feature matrix
    - At least one positive edge exists
    """
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D; got shape {labels.shape}.")

    if pdrugs_x.shape[0] != labels.shape[0]:
        raise ValueError(
            f"pdrugs feature count ({pdrugs_x.shape[0]}) != labels rows ({labels.shape[0]})."
        )

    if seffect_x.shape[0] != labels.shape[1]:
        raise ValueError(
            f"seffect feature count ({seffect_x.shape[0]}) != labels columns ({labels.shape[1]})."
        )

    if not np.isfinite(pdrugs_x).all():
        raise ValueError("pdrugs feature matrix contains NaN or Inf.")

    if not np.isfinite(seffect_x).all():
        raise ValueError("seffect feature matrix contains NaN or Inf.")

    n_edges = int((labels != 0).sum())
    if n_edges == 0:
        raise ValueError("labels matrix has zero positive entries — no edges to build.")


# ---------------------------------------------------------------------------
# Step 2: Build the bipartite HeteroData graph
# ---------------------------------------------------------------------------

def build_hetero_data(
    pdrugs_x: np.ndarray,
    seffect_x: np.ndarray,
    labels: np.ndarray,
) -> HeteroData:
    """
    Construct the undirected bipartite (pdrugs, associated, seffect) graph.

    Mirrors `construct_hetero_data()` in the authors' `src/graph/helpers.py`,
    except `node_id` is `torch.arange(...)` rather than a raw NumPy array —
    see module docstring for why.
    """
    data = HeteroData()

    n_pdrugs = pdrugs_x.shape[0]
    n_seffect = seffect_x.shape[0]

    data["pdrugs"].node_id = torch.arange(n_pdrugs, dtype=torch.long)
    data["seffect"].node_id = torch.arange(n_seffect, dtype=torch.long)

    data["pdrugs"].x = torch.from_numpy(np.asarray(pdrugs_x, dtype=np.float32))
    data["seffect"].x = torch.from_numpy(np.asarray(seffect_x, dtype=np.float32))

    pdrugs_idx, seffect_idx = np.nonzero(labels)
    edge_index = torch.tensor(
        np.stack([pdrugs_idx, seffect_idx], axis=0), dtype=torch.long
    )
    data[EDGE_TYPE].edge_index = edge_index

    data = T.ToUndirected()(data)

    return data


# ---------------------------------------------------------------------------
# Step 3: Apply the authors' RandomLinkSplit protocol
# ---------------------------------------------------------------------------

def split_graph(
    data: HeteroData,
    seed: int = RANDOM_SEED,
) -> tuple[HeteroData, HeteroData, HeteroData]:
    """
    Split the graph into train/val/test via T.RandomLinkSplit, using the
    authors' exact parameters (see module docstring).
    """
    torch_geometric.seed_everything(seed)

    transform = T.RandomLinkSplit(
        num_val=NUM_VAL,
        num_test=NUM_TEST,
        is_undirected=True,
        disjoint_train_ratio=DISJOINT_TRAIN_RATIO,
        neg_sampling_ratio=NEG_SAMPLING_RATIO,
        add_negative_train_samples=False,
        edge_types=EDGE_TYPE,
        rev_edge_types=REV_EDGE_TYPE,
    )
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# Step 4: Post-split validation
# ---------------------------------------------------------------------------

def validate_split(
    full_data: HeteroData,
    train_data: HeteroData,
    val_data: HeteroData,
    test_data: HeteroData,
) -> None:
    """
    Raise ValueError if the split violates its own stated contract.

    Checks:
    - val/test edge_label is all-ones (neg_sampling_ratio=0.0 → positives only)
    - val/test supervision-edge counts are within tolerance of num_val/num_test
      fractions of total positive edges
    - train's message-passing edge_index is a subset of the full graph's edges
    - node counts/features are unchanged across splits (only edges differ)
    """
    total_pos_edges = full_data[EDGE_TYPE].edge_index.size(1)

    for name, split in [("val", val_data), ("test", test_data)]:
        edge_label = split[EDGE_TYPE].edge_label
        if not torch.all(edge_label == 1):
            raise ValueError(
                f"{name} split has non-positive edge_label values; "
                f"expected all-positive since neg_sampling_ratio=0.0."
            )

    val_frac = val_data[EDGE_TYPE].edge_label_index.size(1) / total_pos_edges
    test_frac = test_data[EDGE_TYPE].edge_label_index.size(1) / total_pos_edges
    if not (0.05 < val_frac < 0.15):
        raise ValueError(f"val split fraction {val_frac:.4f} far from expected ~{NUM_VAL}.")
    if not (0.05 < test_frac < 0.15):
        raise ValueError(f"test split fraction {test_frac:.4f} far from expected ~{NUM_TEST}.")

    for split in [train_data, val_data, test_data]:
        if split["pdrugs"].x.shape != full_data["pdrugs"].x.shape:
            raise ValueError("pdrugs node features changed across split.")
        if split["seffect"].x.shape != full_data["seffect"].x.shape:
            raise ValueError("seffect node features changed across split.")

    train_edges = set(map(tuple, train_data[EDGE_TYPE].edge_index.T.tolist()))
    full_edges = set(map(tuple, full_data[EDGE_TYPE].edge_index.T.tolist()))
    if not train_edges.issubset(full_edges):
        raise ValueError("train message-passing edges are not a subset of the full graph's edges.")


# ---------------------------------------------------------------------------
# Step 5: Audit
# ---------------------------------------------------------------------------

def build_graph_audit(
    full_data: HeteroData,
    train_data: HeteroData,
    val_data: HeteroData,
    test_data: HeteroData,
    seed: int,
    graph_output: Path,
    audit_output: Path,
) -> dict[str, Any]:
    def _edge_summary(split: HeteroData) -> dict[str, Any]:
        return {
            "message_passing_edges": int(split[EDGE_TYPE].edge_index.size(1)),
            "supervision_edges": int(split[EDGE_TYPE].edge_label_index.size(1)),
            "supervision_label_sum": float(split[EDGE_TYPE].edge_label.sum().item()),
        }

    return {
        "provenance": (
            "Graph construction and split parameters traced verbatim from "
            "src/graph/helpers.py (construct_hetero_data, split_data) in "
            "github.com/sadrahkm/PolyLLM. node_id uses torch.arange(...) "
            "instead of the authors' raw NumPy array — see module docstring."
        ),
        "node_types": {
            "pdrugs": {
                "count": int(full_data["pdrugs"].num_nodes),
                "feature_dim": int(full_data["pdrugs"].x.shape[1]),
                "feature_source": str(DEFAULT_PDRUGS_FEATURES),
            },
            "seffect": {
                "count": int(full_data["seffect"].num_nodes),
                "feature_dim": int(full_data["seffect"].x.shape[1]),
                "feature_source": str(DEFAULT_SEFFECT_FEATURES),
            },
        },
        "edge_type": list(EDGE_TYPE),
        "reverse_edge_type": list(REV_EDGE_TYPE),
        "total_positive_edges_pdrugs_to_seffect": int(full_data[EDGE_TYPE].edge_index.size(1)),
        "split_params": {
            "num_val": NUM_VAL,
            "num_test": NUM_TEST,
            "is_undirected": True,
            "disjoint_train_ratio": DISJOINT_TRAIN_RATIO,
            "neg_sampling_ratio": NEG_SAMPLING_RATIO,
            "add_negative_train_samples": False,
            "random_seed": seed,
        },
        "splits": {
            "train": _edge_summary(train_data),
            "val": _edge_summary(val_data),
            "test": _edge_summary(test_data),
        },
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
        "output_paths": {
            "graph": str(graph_output),
        },
    }


def _write_atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp), str(path))
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_graph_pipeline(
    pdrugs_features: Path = DEFAULT_PDRUGS_FEATURES,
    seffect_features: Path = DEFAULT_SEFFECT_FEATURES,
    labels_input: Path = DEFAULT_LABELS_INPUT,
    graph_output: Path = DEFAULT_GRAPH_OUTPUT,
    audit_output: Path = DEFAULT_AUDIT_OUTPUT,
    seed: int = RANDOM_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Full Milestone 10c graph-construction-and-split pipeline."""
    logger.info("=== Milestone 10c graph construction pipeline started ===")

    if not overwrite and graph_output.exists() and audit_output.exists():
        logger.info(
            "Artifacts already exist at %s and %s. "
            "Pass --overwrite to regenerate. Loading existing audit.",
            graph_output, audit_output,
        )
        return json.loads(audit_output.read_text(encoding="utf-8"))

    # 1. Load inputs
    pdrugs_x = np.load(pdrugs_features)
    seffect_x = np.load(seffect_features)
    labels = np.load(labels_input)
    logger.info(
        "Loaded pdrugs features %s, seffect features %s, labels %s.",
        pdrugs_x.shape, seffect_x.shape, labels.shape,
    )

    # 2. Validate
    validate_inputs(pdrugs_x, seffect_x, labels)
    logger.info("Input validation passed.")

    # 3. Build graph
    full_data = build_hetero_data(pdrugs_x, seffect_x, labels)
    logger.info(
        "Built HeteroData: pdrugs=%d, seffect=%d, positive edges=%d.",
        full_data["pdrugs"].num_nodes,
        full_data["seffect"].num_nodes,
        full_data[EDGE_TYPE].edge_index.size(1),
    )

    # 4. Split
    train_data, val_data, test_data = split_graph(full_data, seed=seed)
    logger.info(
        "Split — train supervision edges: %d, val: %d, test: %d.",
        train_data[EDGE_TYPE].edge_label_index.size(1),
        val_data[EDGE_TYPE].edge_label_index.size(1),
        test_data[EDGE_TYPE].edge_label_index.size(1),
    )

    # 5. Validate split
    validate_split(full_data, train_data, val_data, test_data)
    logger.info("Split validation passed.")

    # 6. Save atomically
    graph_output.parent.mkdir(parents=True, exist_ok=True)
    tmp_graph = graph_output.parent / (graph_output.name + ".tmp")
    torch.save(
        {"full": full_data, "train": train_data, "val": val_data, "test": test_data},
        tmp_graph,
    )
    os.replace(str(tmp_graph), str(graph_output))
    logger.info("Saved graph split: %s.", graph_output)

    # 7. Audit
    audit = build_graph_audit(
        full_data, train_data, val_data, test_data, seed, graph_output, audit_output
    )
    _write_atomic_json(audit, audit_output)
    logger.info("Wrote audit: %s.", audit_output)
    logger.info("=== Milestone 10c complete. ===")

    return audit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Milestone 10c — build the bipartite GNN graph and link split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdrugs-features",  type=Path, default=DEFAULT_PDRUGS_FEATURES)
    p.add_argument("--seffect-features", type=Path, default=DEFAULT_SEFFECT_FEATURES)
    p.add_argument("--labels-input",     type=Path, default=DEFAULT_LABELS_INPUT)
    p.add_argument("--graph-output",     type=Path, default=DEFAULT_GRAPH_OUTPUT)
    p.add_argument("--audit-output",     type=Path, default=DEFAULT_AUDIT_OUTPUT)
    p.add_argument("--seed",             type=int,  default=RANDOM_SEED)
    p.add_argument("--overwrite",        action="store_true",
                   help="Regenerate even if artifacts already exist.")
    args = p.parse_args(argv)

    run_graph_pipeline(
        pdrugs_features=args.pdrugs_features,
        seffect_features=args.seffect_features,
        labels_input=args.labels_input,
        graph_output=args.graph_output,
        audit_output=args.audit_output,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
