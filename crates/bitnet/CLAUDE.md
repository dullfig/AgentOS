# agentos-bitnet

Pure-Rust 1.58-bit inference engine. Ternary weights {-1, 0, +1}, zero multiplication.

## Architecture

### Core
- **Tensor** (`tensor.rs`) — 2-bit packed ternary storage (4 weights/byte, 16× vs f32). 8-bit quantized activations with absmax scaling. Float tensors for non-quantized ops.
- **I2S Kernel** (`ops/matmul.rs`) — Ternary matrix-vector product via conditional add/sub/skip. Branch-free inner loop, byte-level unpacking with sub-byte row offset support.
- **LUT Kernel** (`ops/lut.rs`) — TL1 lookup-table kernel: pairs 2 weights → 4-bit index into 9-entry precomputed table. Zero arithmetic in hot loop.
- **Quantize** (`ops/quantize.rs`) — Absmax 8-bit activation quantization. Per-token scaling for batched inference.

### Layers (compose bottom-up, each is a standalone struct with `forward()`)
- **BitLinear** (`layers/bitlinear.rs`) — Ternary linear layer (replaces nn.Linear). Forward: quantize activations → ternary matmul → rescale by γ·scale.
- **RmsNorm** (`layers/rmsnorm.rs`) — Root Mean Square normalization (LLaMA-style, no bias).
- **RoPE** (`layers/rope.rs`) — Rotary Position Embeddings. Precomputed inverse-frequency table, per-head rotation. Supports multi-head and sequence-batch forward.
- **MultiHeadAttention** (`layers/attention.rs`) — Grouped Query Attention (GQA). Ternary Q/K/V/O projections via BitLinear, RoPE on Q and K, causal mask, softmax, output projection. Supports `n_kv_heads < n_heads`.
- **SwiGLU** (`layers/swiglu.rs`) — Gated FFN: `down(SiLU(gate(x)) ⊙ up(x))`. Three ternary projections.
- **TransformerBlock** (`layers/transformer.rs`) — Pre-norm decoder block: attn_norm → attention → residual → ffn_norm → SwiGLU → residual.
- **TransformerModel** (`layers/model.rs`) — Full LLM: embedding lookup → N transformer blocks → final RMSNorm → output projection (linear or tied embedding) → logits. Accepts token IDs, returns logit vectors.

### Loader
- **GGUF Parser** (`gguf.rs`) — GGUF v3 format parser. Loads TQ1_0 (base-3 packed), TQ2_0 (2-bit packed), F32, F16, BF16 tensors. Generic reader enables in-memory testing. `GgufFile::open()` parses header cheaply; tensor data loaded on demand via `load_ternary()` / `load_float()`. Extracts LLaMA `ModelConfig` from metadata.

## Key Invariants

- Ternary encoding: 0b00 = -1, 0b01 = 0, 0b10 = +1 (0b11 unused → zero)
- GGUF TQ2_0 encoding differs (0=neg, 1=zero, 2=pos) — remapped on load via `TQ2_REMAP`
- Matmul accumulates in i32 (safe: 127 × 4096 < i32::MAX)
- LUT kernel must produce identical results to I2S kernel (tested exhaustively)
- Row byte access handles sub-byte offsets when rows don't start on 4-value boundaries
- GGUF dimensions reversed on parse (GGUF stores innermost first, we store outermost first)
- Causal attention mask: token at position t can only attend to positions 0..=t
- Residual connections preserve gradient flow (pre-norm architecture)

## Public API

- `TernaryTensor::pack(values, rows, cols)` — Pack ternary values into 2-bit storage
- `TernaryTensor::from_packed(data, rows, cols)` — From raw bytes (GGUF loading)
- `ActivationTensor::quantize(values, shape)` — Absmax 8-bit quantization
- `ternary_matvec(weights, input)` — I2S kernel, returns i32 accumulators
- `lut_matvec(weights, input)` — LUT kernel, returns i32 accumulators
- `BitLinear::new(weights, weight_scale).forward(input)` — End-to-end ternary linear
- `RmsNorm::new(weight, eps).forward(input)` — Normalization
- `RoPE::new(dim, base).forward(input, pos)` — Rotary position encoding
- `MultiHeadAttention::new(...).forward(input, seq_len, start_pos)` — GQA attention
- `SwiGLU::new(gate, up, down).forward(input)` — Gated FFN
- `TransformerBlock::new(...).forward(input, seq_len, start_pos)` — Single decoder layer
- `TransformerModel::new(...).forward(tokens, start_pos)` — Token IDs → logits
- `TransformerModel::forward_last(tokens, start_pos)` — Logits for last token only
- `GgufFile::open(path)` / `open_reader(reader)` — Parse GGUF header
- `GgufFile::load_ternary(name)` → `(TernaryTensor, f32)` — Load ternary weights
- `GgufFile::load_float(name)` → `FloatTensor` — Load float weights
- `GgufFile::model_config()` → `ModelConfig` — Extract LLaMA hyperparameters

## Testing

150 tests covering: bit packing roundtrips, I2S correctness, LUT-vs-I2S equivalence (exhaustive 9-combo), sub-byte alignment, quantization fidelity, layer forward passes, compression ratio, f16/bf16 conversion (normals/subnormals/inf/NaN), GGUF header/metadata/tensor parsing, TQ1_0 base-3 decode, TQ2_0 remap, RoPE properties (identity at pos 0, magnitude preservation, relative position, frequency hierarchy), softmax stability, GQA head grouping, causal masking, SiLU activation, SwiGLU gating, residual connections, full model forward pass with tied/untied embeddings.

## Roadmap

- [x] GGUF model loader
- [x] RoPE (rotary positional embeddings)
- [x] Multi-head attention with ternary Q/K/V (GQA)
- [x] SwiGLU feed-forward network
- [x] Full transformer forward pass (TransformerModel)
- [ ] KV cache for autoregressive generation
- [ ] Token sampler (top-k, top-p, temperature)
- [ ] SIMD kernels (x86 AVX2/512, ARM NEON) via std::arch
- [ ] Engine trait matching SharedEngine interface
- [ ] Wire GGUF loader → TransformerModel construction (load_model helper)

## Design Principles

- **Modular**: Each layer is a standalone struct. Swap RoPE for ALiBi, SwiGLU for GeGLU — one file change, other tests keep passing.
- **Plain f32 at boundaries**: No custom tensor framework lock-in. Layers talk via `&[f32]`.
- **Composition over inheritance**: TransformerBlock contains attention + FFN, doesn't subclass.
- **Generic reader**: GGUF parser works with `Cursor<Vec<u8>>` in tests, `BufReader<File>` in prod.
- **Zero unsafe**: All SIMD will go through safe abstractions (future).
