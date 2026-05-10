# cortex-shim training scripts

Python scripts the shim-expert agent invokes to train cortex shims
from `(vector, label)` JSONL data. Lives outside the Rust source tree
because the agent calls them as subprocesses via `CommandExecTool`.

## Files

- **`train_shim.py`** — train an FFN shim, export ONNX + metrics
- **`test_train_shim.py`** — pytest smoke test on synthetic data

## Install

Tested on Python 3.10+.

```
pip install torch numpy pytest
```

PyTorch installs CPU-only by default, which is what we want — shims
are tiny (~28k–~5M parameters depending on hidden_dims) and CPU-trainable
in seconds. Daniel's existing `C:\src\classifiers` setup already has
torch installed and is the canonical reference environment.

## Train a shim

```
python train_shim.py \
    --input  ./training-data/embedded.jsonl \
    --output-dir ./out/should_respond_v3 \
    --input-dim 4096 \
    --hidden 1024,256 \
    --output-dim 1 \
    --epochs 30
```

Outputs:

- `out/should_respond_v3/model.onnx`  — opset 11, ready for `CortexShimClient::register`
- `out/should_respond_v3/metrics.json` — accuracy, per-class precision/recall,
  confusion matrix, train/val/test split sizes

## Input JSONL format

Each line:

```json
{"vector": [..f32..], "label": 0}
{"vector": [..f32..], "label": 1}
```

`vector` length must be consistent across rows; `label` must be a
non-negative integer. For a binary scalar gate use labels 0/1 with
`--output-dim 1`. For an N-way category use labels 0..N-1 with
`--output-dim N`.

## Test

```
python -m pytest test_train_shim.py -v
```

Generates synthetic data with two well-separable clusters, runs the
trainer end-to-end, and asserts >85% test-set accuracy plus a valid
ONNX file.

## How this fits

The shim-expert agent (organism: `shim-expert`) drives the lifecycle:

1. Generate `(text, label)` example pairs from a natural-language
   feedback signal
2. Embed each text via cortex's `/v1/embed` (this gives `vector`)
3. Write the JSONL
4. Invoke `train_shim.py`
5. Read `metrics.json`; if accuracy clears the threshold, register the
   ONNX with cortex via `CortexShimClient::register`
6. Append a rule to the target agent's `shim-rules.json` that gates on
   the new shim

Documented in the integration-claude session memory at
`project_modular_cognition_architecture.md` (Loop 1 spec).
