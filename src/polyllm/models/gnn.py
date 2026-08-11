"""
GNN link-prediction model — Milestone 10 (the paper's headline architecture).

Traced verbatim from the PolyLLM authors' src/graph/{GNN,Model,Classifier}.py
in github.com/sadrahkm/PolyLLM (fetched character-for-character on
2026-08-11; see the gnn-authors-source-verified memory for the full quotes).

Kept as separate pieces the way the authors split them across GNN.py /
Model.py / Classifier.py:

GNNEncoder        — homogeneous 3-layer GraphConv encoder, wrapped into a
                     heterogeneous one via `to_hetero` inside GNNLinkPredictor.
DotProductClassifier — scores a candidate edge as the dot product of its
                     endpoint embeddings.
GNNLinkPredictor   — lifts raw pdrugs/seffect features to hidden_channels via
                     a Linear + a learned nn.Embedding per node (the paper's
                     graph has no other node features, so a learned per-node
                     embedding supplements the frozen ones, exactly as the
                     authors do), runs the hetero GNN, and classifies edges.
sample_negative_edges / build_global_positive_edge_set / assemble_supervision_edges
                   — the authors' per-forward-pass negative sampling
                     (val/test carry only positive edges — see Milestone 10c
                     — so negatives must be sampled fresh at both train and
                     eval time), made device-agnostic.

Deviations from the literal authors' code (full reasoning in the
gnn-authors-source-verified memory):
- No `bn3`/`bn4` — confirmed dead code in the authors' own GNN.py (declared,
  never referenced in forward()); omitted rather than reproduced as inert
  clutter.
- No unused `dropout_rate` constructor parameter, for the same reason.
  Dropout probability is a fixed 0.8 — what the authors' code actually runs.
- LeakyReLU keeps PyTorch's default negative_slope=0.01 — the authors' GNN.py
  never overrides it, unlike the MLP path (M9c), and no paper quote
  contradicts 0.01 here.
- No internal RNG reseeding inside forward(). The authors' `Model.forward`
  calls a full reseed (`set_seed(12)`, then `torch.manual_seed(42)` again
  before negative sampling) on every single call — this would make dropout
  masks identical across every forward pass, which looks like an
  unintentional bug rather than a deliberate design choice. Not reproduced;
  seed once at run start instead, like every other script in this repo.
- Negative sampling is device-agnostic instead of the authors' hardcoded
  `.to('cuda')`.
- Negative sampling filters false negatives in GLOBAL node-index space (via
  each node's `node_id`), not local mini-batch indices. The authors' own
  `graph.py` builds `global_pos_edges` once from the full, unsplit graph
  (global indices) but then calls `valid_negative_sampling` on
  `LinkNeighborLoader` mini-batches, whose `edge_index` uses LOCAL indices
  renumbered per batch — comparing the two directly (as the literal code
  does) would fail to filter almost all true collisions. See
  `sample_negative_edges`'s docstring for the fix.
- The optional "hard negative" path keyed on a curated `dangerous_seffects_ids`
  list is out of scope — an auxiliary ablation, not part of the paper's
  Table 5 headline metric.
- Construction takes explicit `num_pdrugs`, `num_seffect`, `pdrugs_input_dim`,
  `hidden_channels`, `metadata` rather than the whole `HeteroData` object, and
  prediction is split into `encode()` + `predict_edges()` rather than one
  monolithic `forward()` that also performs negative sampling — a structural
  cleanup with no effect on the computation for a given set of supervision
  edges.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.utils import negative_sampling

EDGE_TYPE = ("pdrugs", "associated", "seffect")


# ---------------------------------------------------------------------------
# Homogeneous encoder (wrapped into a heterogeneous one via to_hetero)
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """3-layer GraphConv encoder. See module docstring for confirmed dead
    code (bn3/bn4, dropout_rate) intentionally omitted from the authors'
    original GNN.py."""

    DROPOUT_P: float = 0.8

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.conv1 = GraphConv(hidden_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
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
        x = self.lin1(x)
        return x


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class DotProductClassifier(nn.Module):
    """Dot-product edge scorer — identical to the authors' Classifier.py."""

    def forward(
        self,
        x_pdrugs: torch.Tensor,
        x_seffect: torch.Tensor,
        edge_label_index: torch.Tensor,
        integrate: bool = False,
    ) -> torch.Tensor:
        edge_feat_pdrugs = x_pdrugs[edge_label_index[0]]
        edge_feat_seffect = x_seffect[edge_label_index[1]]

        if integrate:
            return edge_feat_pdrugs * edge_feat_seffect
        return (edge_feat_pdrugs * edge_feat_seffect).sum(dim=-1)


# ---------------------------------------------------------------------------
# Heterogeneous link predictor
# ---------------------------------------------------------------------------

class GNNLinkPredictor(nn.Module):
    """
    Lifts pdrugs/seffect node features to `hidden_channels`, runs the
    heterogeneous GNN, and scores candidate edges.

    `metadata` is `HeteroData.metadata()` from Milestone 10c's graph — the
    (node_types, edge_types) tuple `to_hetero` needs to know which relations
    to build separate message-passing weights for.
    """

    SEFFECT_INPUT_DIM: int = 768  # BERT hidden size — default, overridable for testing

    def __init__(
        self,
        num_pdrugs: int,
        num_seffect: int,
        pdrugs_input_dim: int,
        hidden_channels: int,
        metadata: tuple,
        seffect_input_dim: int = SEFFECT_INPUT_DIM,
    ) -> None:
        super().__init__()
        self.pdrugs_lin = nn.Linear(pdrugs_input_dim, hidden_channels)
        self.seffect_lin = nn.Linear(seffect_input_dim, hidden_channels)
        self.pdrugs_emb = nn.Embedding(num_pdrugs, hidden_channels)
        self.seffect_emb = nn.Embedding(num_seffect, hidden_channels)
        self.encoder = to_hetero(GNNEncoder(hidden_channels), metadata=metadata)
        self.classifier = DotProductClassifier()

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        node_id_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        lifted = {
            "pdrugs": self.pdrugs_lin(x_dict["pdrugs"].float()) + self.pdrugs_emb(node_id_dict["pdrugs"]),
            "seffect": self.seffect_lin(x_dict["seffect"].float()) + self.seffect_emb(node_id_dict["seffect"]),
        }
        return self.encoder(lifted, edge_index_dict)

    def predict_edges(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_label_index: torch.Tensor,
        integrate: bool = False,
    ) -> torch.Tensor:
        return self.classifier(x_dict["pdrugs"], x_dict["seffect"], edge_label_index, integrate)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        node_id_dict: dict[str, torch.Tensor],
        edge_label_index: torch.Tensor,
        integrate: bool = False,
    ) -> torch.Tensor:
        x_dict = self.encode(x_dict, edge_index_dict, node_id_dict)
        return self.predict_edges(x_dict, edge_label_index, integrate)


# ---------------------------------------------------------------------------
# Negative sampling (device-agnostic; see module docstring for the
# hardcoded-cuda deviation)
# ---------------------------------------------------------------------------

def build_global_positive_edge_set(full_data) -> set[tuple[int, int]]:
    """
    All (pdrugs_idx, seffect_idx) positive pairs in the FULL, unsplit graph —
    used to filter false negatives out of every sampled batch, no matter
    which split (train/val/test) that batch came from.
    """
    edge_index = full_data[EDGE_TYPE].edge_index
    return set(map(tuple, edge_index.T.tolist()))


def sample_negative_edges(
    data,
    global_pos_edges: set[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """
    Sample random negative edges for this batch's message-passing graph,
    then drop any that collide with a true positive anywhere in the full
    graph (not just this batch) — mirrors `Model.valid_negative_sampling`.

    The realized negative count is <= the requested count (some samples may
    be dropped as false negatives), matching the authors' behavior.

    Index-space fix vs. the literal authors' code: `LinkNeighborLoader`
    mini-batches renumber sampled nodes to a LOCAL 0-indexed space (that's
    what `edge_index` here is expressed in), while `global_pos_edges` is
    built once from the FULL graph in GLOBAL indices (see
    `build_global_positive_edge_set`). The authors' `valid_negative_sampling`
    compares locally-indexed sampled negatives directly against that
    globally-indexed set — for a genuine mini-batch (as opposed to the full,
    unsplit graph, where local==global trivially) this comparison is between
    two different index spaces and would fail to filter almost all true
    collisions. We translate the sampled negatives' local indices to global
    ones via each node type's `node_id` (which PyG's neighbor sampler
    populates with the true global id per local position) before filtering,
    then keep the LOCAL indices for the edges that survive — required
    because embeddings/features are indexed locally within this batch.
    """
    edge_index = data[EDGE_TYPE].edge_index
    num_pos_supervision = data[EDGE_TYPE].edge_label_index.size(1)
    num_pdrugs = data["pdrugs"].num_nodes
    num_seffect = data["seffect"].num_nodes

    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=(num_pdrugs, num_seffect),
        num_neg_samples=num_pos_supervision,
        force_undirected=True,
    )

    pdrugs_node_id = data["pdrugs"].node_id
    seffect_node_id = data["seffect"].node_id
    global_neg_pdrugs = pdrugs_node_id[neg_edge_index[0]].tolist()
    global_neg_seffect = seffect_node_id[neg_edge_index[1]].tolist()

    keep_mask = torch.tensor(
        [
            (gp, gs) not in global_pos_edges
            for gp, gs in zip(global_neg_pdrugs, global_neg_seffect)
        ],
        dtype=torch.bool,
    )
    neg_edge_index = neg_edge_index[:, keep_mask]

    if neg_edge_index.size(1) == 0:
        raise RuntimeError(
            "Negative sampling produced zero valid (non-positive) edges for this batch."
        )

    return neg_edge_index.to(device)


def assemble_supervision_edges(
    data,
    global_pos_edges: set[tuple[int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combine this batch's positive supervision edges with freshly sampled
    negatives into a single (edge_label_index, edge_label) pair, ready for
    `GNNLinkPredictor.predict_edges`.
    """
    pos_edge_label_index = data[EDGE_TYPE].edge_label_index.to(device)
    pos_edge_label = data[EDGE_TYPE].edge_label.to(device)

    neg_edge_index = sample_negative_edges(data, global_pos_edges, device)

    edge_label_index = torch.cat([pos_edge_label_index, neg_edge_index], dim=-1)
    edge_label = torch.cat(
        [pos_edge_label, pos_edge_label.new_zeros(neg_edge_index.size(1))], dim=0
    )

    return edge_label_index, edge_label
