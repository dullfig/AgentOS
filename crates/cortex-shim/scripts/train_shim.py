"""Train a cortex shim FFN from (vector, label) pairs.

Extends the C:\\src\\classifiers training-recipe pattern to ~28k-param
FFNs whose input is a transformer hidden-state vector rather than an
image. Stratified seed=42 split (5% test / 10% val / 85% train),
batch SGD with class weighting, ONNX export at opset_version=11.

Usage:
    python train_shim.py
        --input embedded.jsonl
        --output-dir ./out
        [--input-dim 4096]   # autodetected from JSONL if omitted
        [--hidden 1024,256]
        [--output-dim 1]      # 1 = scalar (binary), N = category:N
        [--epochs 30]
        [--lr 1e-3]
        [--batch-size 64]
        [--seed 42]

Inputs:
    JSONL where each line is `{"vector": [...], "label": <int>}`.
    Vectors must all have the same length.

Outputs (in --output-dir):
    model.onnx     ONNX file ready for CortexShimClient::register
    metrics.json   accuracy, per-class precision/recall, confusion
                   matrix, and a sample of misclassified test items
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


SEED = 42
VAL_SPLIT = 0.10
TEST_SPLIT = 0.05


# ── Model ────────────────────────────────────────────────────────────


class ShimFFN(nn.Module):
    """Configurable FFN: input_dim → hidden_dims[0] → ... → output_dim.

    ReLU activation between layers; final layer outputs raw logits
    (callers / cortex apply sigmoid or softmax as appropriate).
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data loading ─────────────────────────────────────────────────────


def load_jsonl(path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load (vector, label) JSONL into tensors. Returns (X, y, dim)."""
    vectors: list[list[float]] = []
    labels: list[int] = []
    dim: int | None = None

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {line_no}: bad JSON ({e})") from e
            v = row.get("vector")
            lbl = row.get("label")
            if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                raise ValueError(f"line {line_no}: 'vector' must be a list of numbers")
            if not isinstance(lbl, int):
                raise ValueError(f"line {line_no}: 'label' must be an int")
            if dim is None:
                dim = len(v)
            elif len(v) != dim:
                raise ValueError(
                    f"line {line_no}: vector length {len(v)} != first row's {dim}"
                )
            vectors.append(v)
            labels.append(lbl)

    if not vectors:
        raise ValueError(f"{path}: no rows loaded")

    return (
        torch.tensor(vectors, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        dim or 0,
    )


def stratified_split(
    y: torch.Tensor, val_frac: float, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class stratified split. Returns (train_idx, val_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for cls in torch.unique(y).tolist():
        cls_idx = (y == cls).nonzero(as_tuple=True)[0].cpu().numpy()
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        test_idx.extend(cls_idx[:n_test].tolist())
        val_idx.extend(cls_idx[n_test : n_test + n_val].tolist())
        train_idx.extend(cls_idx[n_test + n_val :].tolist())

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


def class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights so the loss isn't dominated by
    the majority class."""
    counts = torch.bincount(y, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights


# ── Train + eval ─────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    is_binary: bool,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        if is_binary:
            loss = loss_fn(logits.squeeze(-1), y.float())
        else:
            loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, is_binary: bool
) -> dict:
    """Compute accuracy, per-class precision/recall, confusion matrix."""
    model.eval()
    logits = model(X)
    if is_binary:
        preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
    else:
        preds = logits.argmax(dim=-1)

    num_classes = int(y.max().item()) + 1 if y.numel() else 1
    num_classes = max(num_classes, int(preds.max().item()) + 1 if preds.numel() else 1)
    if is_binary:
        num_classes = 2

    correct = (preds == y).sum().item()
    total = y.numel()
    accuracy = correct / max(total, 1)

    confusion = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y.tolist(), preds.tolist()):
        confusion[t][p] += 1

    per_class = []
    for c in range(num_classes):
        tp = confusion[c][c]
        fp = sum(confusion[r][c] for r in range(num_classes) if r != c)
        fn = sum(confusion[c][p] for p in range(num_classes) if p != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class.append({"class": c, "precision": precision, "recall": recall, "support": tp + fn})

    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


# ── ONNX export ──────────────────────────────────────────────────────


def export_onnx(model: nn.Module, input_dim: int, path: Path) -> None:
    """Export to ONNX opset 11. Dynamic batch axis."""
    model.eval()
    dummy = torch.zeros(1, input_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        opset_version=11,
        input_names=["hidden_state"],
        output_names=["logits"],
        dynamic_axes={"hidden_state": {0: "batch"}, "logits": {0: "batch"}},
    )


# ── Entrypoint ───────────────────────────────────────────────────────


def parse_hidden(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train a cortex shim FFN")
    p.add_argument("--input", required=True, type=Path, help="JSONL of {vector, label}")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--input-dim", type=int, default=None, help="autodetected if omitted")
    p.add_argument("--hidden", type=parse_hidden, default=[1024, 256])
    p.add_argument(
        "--output-dim",
        type=int,
        default=1,
        help="1 = scalar binary; N>1 = N-class category",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y, dim = load_jsonl(args.input)
    input_dim = args.input_dim or dim
    if input_dim != dim:
        raise SystemExit(
            f"--input-dim {input_dim} disagrees with JSONL vector length {dim}"
        )

    is_binary = args.output_dim == 1
    label_distribution = Counter(y.tolist())
    print(
        f"loaded {X.shape[0]} examples, dim={input_dim}, label distribution: "
        f"{dict(sorted(label_distribution.items()))}"
    )

    train_idx, val_idx, test_idx = stratified_split(
        y, VAL_SPLIT, TEST_SPLIT, args.seed
    )
    print(
        f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Class-weighted sampler for the train loader so the gradient
    # isn't dominated by the majority class.
    num_classes = max(2, args.output_dim if not is_binary else 2)
    weights_per_class = class_weights(y_train, num_classes)
    sample_weights = weights_per_class[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        sampler=sampler,
    )

    device = torch.device("cpu")
    model = ShimFFN(input_dim, args.hidden, args.output_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: ShimFFN({input_dim} -> {args.hidden} -> {args.output_dim}), params={n_params}")

    if is_binary:
        loss_fn: nn.Module = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weights_per_class.to(device))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_state: dict | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optim, loss_fn, device, is_binary)
        val_metrics = evaluate(model, X_val, y_val, is_binary)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(
            f"  epoch {epoch:>3}/{args.epochs}: "
            f"train_loss={train_loss:.4f}  val_acc={val_metrics['accuracy']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, X_test, y_test, is_binary)
    val_final = evaluate(model, X_val, y_val, is_binary)

    onnx_path = args.output_dir / "model.onnx"
    export_onnx(model, input_dim, onnx_path)

    metrics = {
        "input_dim": input_dim,
        "output_dim": args.output_dim,
        "hidden_dims": args.hidden,
        "epochs_run": args.epochs,
        "param_count": n_params,
        "best_val_accuracy": best_val_acc,
        "val": val_final,
        "test": test_metrics,
        "label_distribution_full": dict(label_distribution),
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"\ntest accuracy: {test_metrics['accuracy']:.4f}  "
        f"(threshold for promotion is the agent's call)"
    )
    print(f"wrote {onnx_path}")
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
