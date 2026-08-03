"""
Shared training infrastructure for multi-label MLP experiments (Milestones 6B and 7).

Provides: deterministic seeding, memory-mapped pair dataset, seeded DataLoader,
one training epoch, validation prediction collection, early stopping,
checkpoint I/O, history-row creation, and artifact alignment validation.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    """Set all relevant seeds for reproducibility.

    Exact cross-hardware reproducibility is not guaranteed (hardware and
    library versions affect floating-point order of operations).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Row-indexed dataset over (optionally memory-mapped) feature and label arrays.

    ``features`` and ``labels`` are indexed by pair_id (row = pair_id).
    ``pair_ids`` is the ordered subset of pair IDs this split exposes.

    Batches are converted to float32 on the fly; source arrays stay in their
    native dtype (e.g. uint8 for Morgan fingerprints) to avoid a full copy.

    ``label_indices`` optionally restricts which label columns are returned,
    used only for the engineering smoke test.
    """

    def __init__(
        self,
        pair_ids: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        label_indices: np.ndarray | None = None,
    ) -> None:
        self.pair_ids = np.asarray(pair_ids, dtype=np.int64)
        self.features = features
        self.labels = labels
        self.label_indices = (
            np.asarray(label_indices, dtype=np.int64) if label_indices is not None else None
        )

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int):
        pid = int(self.pair_ids[idx])
        x = torch.from_numpy(self.features[pid].astype(np.float32))
        if self.label_indices is not None:
            y_arr = self.labels[pid][self.label_indices].astype(np.float32)
        else:
            y_arr = self.labels[pid].astype(np.float32)
        return pid, x, torch.from_numpy(y_arr)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_seeded_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with a deterministic generator."""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=num_workers,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Input alignment validation
# ---------------------------------------------------------------------------

def validate_input_alignment(
    features: np.ndarray,
    labels: np.ndarray,
    pair_index: pd.DataFrame,
    label_mapping: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    *,
    expected_n_pairs: int = 63_472,
    expected_n_features: int = 2_048,
    expected_n_labels: int = 963,
) -> None:
    """Fail loudly before training if any artifact invariant is violated."""
    if features.shape != (expected_n_pairs, expected_n_features):
        raise ValueError(
            f"Feature shape {features.shape} != "
            f"({expected_n_pairs}, {expected_n_features})"
        )
    if labels.shape != (expected_n_pairs, expected_n_labels):
        raise ValueError(
            f"Label shape {labels.shape} != "
            f"({expected_n_pairs}, {expected_n_labels})"
        )
    if len(pair_index) != expected_n_pairs:
        raise ValueError(
            f"pair_index has {len(pair_index)} rows; expected {expected_n_pairs}"
        )

    pair_ids_col = pair_index["pair_id"].to_numpy()
    expected_ids = np.arange(expected_n_pairs, dtype=pair_ids_col.dtype)
    if not np.array_equal(pair_ids_col, expected_ids):
        raise ValueError("pair_index pair_id is not 0..N-1 sequential")

    if len(label_mapping) != expected_n_labels:
        raise ValueError(
            f"label_mapping has {len(label_mapping)} rows; expected {expected_n_labels}"
        )
    lm_sorted = np.sort(label_mapping["label_index"].to_numpy())
    if not np.array_equal(lm_sorted, np.arange(expected_n_labels)):
        raise ValueError("label_mapping label_index is not 0..962 sequential")

    train_set = set(train_ids.tolist())
    val_set   = set(val_ids.tolist())
    test_set  = set(test_ids.tolist())

    for name_a, set_a, name_b, set_b in [
        ("train", train_set, "val",  val_set),
        ("train", train_set, "test", test_set),
        ("val",   val_set,  "test", test_set),
    ]:
        overlap = set_a & set_b
        if overlap:
            raise ValueError(
                f"Split overlap between {name_a} and {name_b}: {len(overlap)} IDs"
            )

    union = train_set | val_set | test_set
    full  = set(range(expected_n_pairs))
    if union != full:
        missing = len(full - union)
        extra   = len(union - full)
        raise ValueError(
            f"Split union != all pair IDs. Missing: {missing}, extra: {extra}"
        )


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor a metric and signal when patience is exhausted."""

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.best_score: float   = -float("inf")
        self.counter: int        = 0
        self.should_stop: bool   = False
        self.best_epoch: int     = 0

    def step(self, score: float, epoch: int) -> bool:
        """Return True when the metric improved by >= min_delta (save checkpoint)."""
        if (score - self.best_score) >= self.min_delta:
            self.best_score = score
            self.counter    = 0
            self.best_epoch = epoch
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One full training epoch. Returns mean batch loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for _pair_ids, x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches  += 1
    return total_loss / n_batches if n_batches else 0.0


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

def run_val_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Collect validation predictions.

    Returns (avg_loss, probabilities, labels, pair_ids), all sorted by pair_id.
    probabilities are float32, range [0, 1].
    No gradients are computed; model parameters are not updated.
    """
    model.eval()
    total_loss   = 0.0
    n_batches    = 0
    all_probs    : list[np.ndarray] = []
    all_labels   : list[np.ndarray] = []
    all_pair_ids : list[int]        = []

    with torch.no_grad():
        for pair_ids_batch, x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            total_loss += float(loss.item())
            n_batches  += 1
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())
            all_pair_ids.extend(pair_ids_batch.tolist())

    probs   = np.concatenate(all_probs,  axis=0)
    labels  = np.concatenate(all_labels, axis=0)
    pids    = np.array(all_pair_ids, dtype=np.int64)

    order  = np.argsort(pids)
    probs  = probs[order].astype(np.float32)
    labels = labels[order]
    pids   = pids[order]

    return (total_loss / n_batches if n_batches else 0.0), probs, labels, pids


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    score: float,
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":                epoch,
            "val_macro_auprc":      score,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config":               config,
            "model_class":          "MultilabelMLP",
            "input_dim":            config["input_dim"],
            "output_dim":           config["output_dim"],
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str | None = None,
) -> dict:
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# History row
# ---------------------------------------------------------------------------

def make_history_row(
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_macro_auprc: float,
    checkpoint_saved: bool,
) -> dict:
    return {
        "epoch":            epoch,
        "train_loss":       round(train_loss,       6),
        "val_loss":         round(val_loss,         6),
        "val_macro_auprc":  round(val_macro_auprc,  6),
        "checkpoint_saved": checkpoint_saved,
    }


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
