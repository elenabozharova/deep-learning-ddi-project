"""
GNN gap diagnostic 2 & 6, part 1 — RNG/reseeding mechanism check (no training).

The authors' `Model.forward` (see gnn-authors-source-verified memory and
src/polyllm/models/gnn.py's module docstring) calls a full reseed
(`set_seed(12)`: random/numpy/torch/cuda) at the START of every single
forward pass, and `torch.manual_seed(42)` again immediately before negative
sampling within that same call. This repo's actual code
(src/polyllm/train_gnn.py::set_seeds) seeds ONCE at run start and never
reseeds inside forward() or before negative sampling — flagged in the
models/gnn.py docstring as "looks like an unintentional bug [in the
authors' code] rather than a deliberate design choice" because reseeding
before every forward call would make dropout masks identical across calls,
defeating dropout's purpose. This script empirically tests that claim
instead of leaving it as an inference from reading the source.

IMPORTANT: this script does NOT modify src/polyllm/models/gnn.py or
train_gnn.py. The authors'-reseed regime is reproduced only inline, here,
for comparison — never adopted.

Two checks, both against a small synthetic hetero graph (same pattern as
tests/test_gnn_model.py's `_synthetic_data`, no real data files needed):

1. Encoder-dropout determinism: run GNNLinkPredictor.encode() twice on the
   identical input, under two RNG regimes. "repo" regime: seed once, call
   twice with no reseeding in between (dropout should differ between the
   two calls). "authors" regime: full reseed immediately before each call
   (dropout should be IDENTICAL between the two calls, if the hypothesis is
   correct).
2. Negative-sampling determinism: same two-regime comparison, but for
   `torch_geometric.utils.negative_sampling` output instead of dropout,
   testing the authors' second reseed point (torch.manual_seed(42) "before
   negative sampling").

Run from the project root:
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/rng_reseed_comparison.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import negative_sampling

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import polyllm.graph.build_graph as bg  # noqa: E402
import polyllm.models.gnn as gm  # noqa: E402

OUTPUT_PATH = Path("outputs/polyllm/gnn_diagnostics/rng_comparison.json")


def authors_set_seed(seed: int) -> None:
    """Literal translation of the authors' full-reseed set_seed(), used here
    ONLY for this diagnostic comparison -- never imported into production
    code."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthetic_data(n_pdrugs=200, n_seffect=20, pdrugs_dim=8, seffect_dim=4, density=0.3, seed=0):
    rng = np.random.default_rng(seed)
    pdrugs_x = rng.normal(size=(n_pdrugs, pdrugs_dim)).astype(np.float32)
    seffect_x = rng.normal(size=(n_seffect, seffect_dim)).astype(np.float32)
    labels = (rng.random((n_pdrugs, n_seffect)) < density).astype(np.uint8)
    labels[0, 0] = 1
    data = bg.build_hetero_data(pdrugs_x, seffect_x, labels)
    return data, pdrugs_dim, seffect_dim


def build_dicts(data):
    x_dict = {"pdrugs": data["pdrugs"].x, "seffect": data["seffect"].x}
    edge_index_dict = data.edge_index_dict
    node_id_dict = {"pdrugs": data["pdrugs"].node_id, "seffect": data["seffect"].node_id}
    return x_dict, edge_index_dict, node_id_dict


# ---------------------------------------------------------------------------
# Check 1: encoder-dropout determinism across repeated forward calls
# ---------------------------------------------------------------------------

def check_dropout_determinism(model, x_dict, edge_index_dict, node_id_dict) -> dict:
    model.train()  # dropout must be active for this check to mean anything

    # --- "repo" regime: seed once, call twice, no reseeding in between ---
    torch.manual_seed(999)
    out_repo_1 = model.encode(x_dict, edge_index_dict, node_id_dict)["pdrugs"].detach().clone()
    out_repo_2 = model.encode(x_dict, edge_index_dict, node_id_dict)["pdrugs"].detach().clone()
    repo_calls_identical = torch.equal(out_repo_1, out_repo_2)

    # --- "authors" regime: full reseed immediately before EACH call ------
    authors_set_seed(12)
    out_auth_1 = model.encode(x_dict, edge_index_dict, node_id_dict)["pdrugs"].detach().clone()
    authors_set_seed(12)
    out_auth_2 = model.encode(x_dict, edge_index_dict, node_id_dict)["pdrugs"].detach().clone()
    authors_calls_identical = torch.equal(out_auth_1, out_auth_2)

    return {
        "repo_regime_repeated_calls_identical": repo_calls_identical,
        "authors_regime_repeated_calls_identical": authors_calls_identical,
        "repo_regime_max_abs_diff": float((out_repo_1 - out_repo_2).abs().max().item()),
        "authors_regime_max_abs_diff": float((out_auth_1 - out_auth_2).abs().max().item()),
        "interpretation": (
            "If authors_regime_repeated_calls_identical=True and "
            "repo_regime_repeated_calls_identical=False, this confirms the "
            "authors' per-forward reseed makes dropout deterministic across "
            "calls within a run (effectively disabling dropout's "
            "regularization benefit across the run), while this repo's "
            "seed-once approach retains normal per-call dropout stochasticity."
        ),
    }


# ---------------------------------------------------------------------------
# Check 2: negative-sampling determinism under the authors' second reseed
# ---------------------------------------------------------------------------

def check_negative_sampling_determinism(data) -> dict:
    edge_index = data[bg.EDGE_TYPE].edge_index
    num_pdrugs = data["pdrugs"].num_nodes
    num_seffect = data["seffect"].num_nodes
    num_neg = 50

    # "repo" regime: no reseeding between the two sampling calls
    torch.manual_seed(999)
    neg_1 = negative_sampling(
        edge_index=edge_index, num_nodes=(num_pdrugs, num_seffect),
        num_neg_samples=num_neg, force_undirected=True,
    )
    neg_2 = negative_sampling(
        edge_index=edge_index, num_nodes=(num_pdrugs, num_seffect),
        num_neg_samples=num_neg, force_undirected=True,
    )
    repo_identical = torch.equal(neg_1, neg_2)

    # "authors" regime: torch.manual_seed(42) immediately before EACH call
    torch.manual_seed(42)
    neg_auth_1 = negative_sampling(
        edge_index=edge_index, num_nodes=(num_pdrugs, num_seffect),
        num_neg_samples=num_neg, force_undirected=True,
    )
    torch.manual_seed(42)
    neg_auth_2 = negative_sampling(
        edge_index=edge_index, num_nodes=(num_pdrugs, num_seffect),
        num_neg_samples=num_neg, force_undirected=True,
    )
    authors_identical = torch.equal(neg_auth_1, neg_auth_2)

    # --- root-cause check: PyG's negative_sampling draws its candidate ----
    # pool via Python's random.sample(), not torch's RNG (confirmed by
    # reading torch_geometric/utils/_negative_sampling.py::sample() directly
    # -- `torch.tensor(random.sample(range(population), k), ...)`). So
    # torch.manual_seed() alone cannot make it reproducible; random.seed()
    # does. Verified empirically here, not just from source reading.
    random.seed(42)
    neg_rs_1 = negative_sampling(
        edge_index=edge_index, num_nodes=(num_pdrugs, num_seffect),
        num_neg_samples=num_neg, force_undirected=True,
    )
    random.seed(42)
    neg_rs_2 = negative_sampling(
        edge_index=edge_index, num_nodes=(num_pdrugs, num_seffect),
        num_neg_samples=num_neg, force_undirected=True,
    )
    random_seed_alone_identical = torch.equal(neg_rs_1, neg_rs_2)

    return {
        "repo_regime_repeated_sampling_identical": repo_identical,
        "authors_regime_repeated_sampling_identical": authors_identical,
        "random_seed_alone_repeated_sampling_identical": random_seed_alone_identical,
        "interpretation": (
            "authors_regime_repeated_sampling_identical=False even though "
            "the authors' code calls torch.manual_seed(42) immediately "
            "before negative sampling -- this is NOT a null result, it is "
            "explained by a second, independent quirk found by reading "
            "torch_geometric/utils/_negative_sampling.py::sample() directly: "
            "PyG draws its candidate negative-edge pool via Python's builtin "
            "random.sample(), not torch's RNG. torch.manual_seed() has no "
            "effect on it. random_seed_alone_repeated_sampling_identical=True "
            "confirms this: seeding Python's random module (not torch) is "
            "what actually controls negative_sampling()'s reproducibility. "
            "So the authors' torch.manual_seed(42) call before negative "
            "sampling is very likely a no-op for its apparent intended "
            "purpose -- a second, independently-discovered RNG-handling "
            "quirk in the authors' own code, alongside the dropout-reseeding "
            "issue found in Check 1."
        ),
    }


def main() -> None:
    torch.manual_seed(42)
    data, pdrugs_dim, seffect_dim = synthetic_data()
    x_dict, edge_index_dict, node_id_dict = build_dicts(data)

    model = gm.GNNLinkPredictor(
        num_pdrugs=data["pdrugs"].num_nodes,
        num_seffect=data["seffect"].num_nodes,
        pdrugs_input_dim=pdrugs_dim,
        hidden_channels=16,
        metadata=data.metadata(),
        seffect_input_dim=seffect_dim,
    )

    print("=== Check 1: encoder-dropout determinism ===")
    dropout_result = check_dropout_determinism(model, x_dict, edge_index_dict, node_id_dict)
    for k, v in dropout_result.items():
        print(f"  {k}: {v}")

    print("\n=== Check 2: negative-sampling determinism ===")
    negsample_result = check_negative_sampling_determinism(data)
    for k, v in negsample_result.items():
        print(f"  {k}: {v}")

    result = {
        "dropout_determinism": dropout_result,
        "negative_sampling_determinism": negsample_result,
        "note": (
            "Diagnostic only. The 'authors' regime is reproduced inline in "
            "this script for comparison; it is NOT imported into or used by "
            "src/polyllm/models/gnn.py or train_gnn.py."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
