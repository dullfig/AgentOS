"""Smoke test for train_shim.py.

Generates synthetic two-cluster data, runs the training entrypoint,
and asserts the trained model clears a reasonable accuracy threshold
plus emits a well-formed ONNX file.

Skips gracefully when torch isn't installed so the test file can sit
in the repo without breaking environments that don't yet have the
training stack.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

# Make train_shim.py importable as a sibling module.
SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_shim  # noqa: E402


@pytest.fixture
def synthetic_data(tmp_path: Path) -> Path:
    """Two well-separable clusters in 16-d space, 200 examples per class."""
    rng = np.random.default_rng(0)
    dim = 16
    n_per_class = 200

    cluster_a = rng.normal(loc=+2.0, scale=0.5, size=(n_per_class, dim))
    cluster_b = rng.normal(loc=-2.0, scale=0.5, size=(n_per_class, dim))

    jsonl = tmp_path / "embedded.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for v in cluster_a:
            f.write(json.dumps({"vector": v.tolist(), "label": 0}) + "\n")
        for v in cluster_b:
            f.write(json.dumps({"vector": v.tolist(), "label": 1}) + "\n")
    return jsonl


def test_train_emits_onnx_and_metrics(tmp_path: Path, synthetic_data: Path) -> None:
    out_dir = tmp_path / "out"

    rc = train_shim.main(
        [
            "--input", str(synthetic_data),
            "--output-dir", str(out_dir),
            "--hidden", "32,16",
            "--output-dim", "1",
            "--epochs", "10",
            "--batch-size", "32",
        ]
    )
    assert rc == 0

    onnx_path = out_dir / "model.onnx"
    metrics_path = out_dir / "metrics.json"
    assert onnx_path.exists(), "ONNX file should be written"
    assert metrics_path.exists(), "metrics.json should be written"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["input_dim"] == 16
    assert metrics["output_dim"] == 1

    # Two well-separated clusters → trivially-separable. Loose floor at
    # 0.85 to keep the test stable across torch versions / RNGs.
    assert metrics["test"]["accuracy"] >= 0.85, (
        f"test accuracy too low: {metrics['test']['accuracy']:.3f} "
        f"(metrics: {metrics})"
    )
    assert metrics["best_val_accuracy"] >= 0.85

    # ONNX file should be non-trivially sized (i.e. not zero bytes).
    assert onnx_path.stat().st_size > 1024


def test_load_jsonl_rejects_inconsistent_dim(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"vector": [0.0, 1.0, 2.0], "label": 0}\n'
        '{"vector": [0.0, 1.0], "label": 1}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="vector length"):
        train_shim.load_jsonl(bad)


def test_stratified_split_preserves_class_proportions() -> None:
    y = torch.tensor([0] * 100 + [1] * 50, dtype=torch.long)
    train, val, test = train_shim.stratified_split(y, 0.10, 0.05, seed=42)
    # Each split must contain at least one example of each class.
    for idx in (train, val, test):
        cls = set(y[idx].tolist())
        assert cls == {0, 1}, f"split missing a class: {cls}"
    # No overlap.
    all_idx = set(train.tolist()) | set(val.tolist()) | set(test.tolist())
    assert len(all_idx) == len(train) + len(val) + len(test)


def test_class_weights_inverse_frequency() -> None:
    y = torch.tensor([0, 0, 0, 0, 1, 1])  # 4-to-2 imbalance
    w = train_shim.class_weights(y, num_classes=2)
    # Class 1 has half the count so its weight should be larger.
    assert w[1] > w[0]
