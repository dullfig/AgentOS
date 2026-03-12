# agentos-bitnet

Pure-Rust 1.58-bit inference engine. Ternary weights {-1, 0, +1}, zero multiplication.

## Architecture

- **Tensor** (`tensor.rs`) — 2-bit packed ternary storage (4 weights/byte, 16× vs f32). 8-bit quantized activations with absmax scaling. Float tensors for non-quantized ops.
- **I2S Kernel** (`ops/matmul.rs`) — Ternary matrix-vector product via conditional add/sub/skip. Branch-free inner loop, byte-level unpacking with sub-byte row offset support.
- **LUT Kernel** (`ops/lut.rs`) — TL1 lookup-table kernel: pairs 2 weights → 4-bit index into 9-entry precomputed table. Zero arithmetic in hot loop.
- **Quantize** (`ops/quantize.rs`) — Absmax 8-bit activation quantization. Per-token scaling for batched inference.
- **BitLinear** (`layers/bitlinear.rs`) — Ternary linear layer (replaces nn.Linear). Forward: quantize activations → ternary matmul → rescale by γ·scale.
- **RmsNorm** (`layers/rmsnorm.rs`) — Root Mean Square normalization (LLaMA-style, no bias).

## Key Invariants

- Ternary encoding: 0b00 = -1, 0b01 = 0, 0b10 = +1 (0b11 unused → zero)
- Matmul accumulates in i32 (safe: 127 × 4096 < i32::MAX)
- LUT kernel must produce identical results to I2S kernel (tested exhaustively)
- Row byte access handles sub-byte offsets when rows don't start on 4-value boundaries

## Public API

- `TernaryTensor::pack(values, rows, cols)` — Pack ternary values into 2-bit storage
- `TernaryTensor::from_packed(data, rows, cols)` — From raw bytes (GGUF loading)
- `ActivationTensor::quantize(values, shape)` — Absmax 8-bit quantization
- `ternary_matvec(weights, input)` — I2S kernel, returns i32 accumulators
- `ternary_matvec_scaled(weights, input, weight_scale)` — Full pipeline, returns f32
- `lut_matvec(weights, input)` — LUT kernel, returns i32 accumulators
- `BitLinear::new(weights, weight_scale).forward(input)` — End-to-end layer
- `RmsNorm::new(weight, eps).forward(input)` — Normalization

## Testing

50 tests covering: bit packing roundtrips, I2S correctness, LUT-vs-I2S equivalence (exhaustive 9-combo), sub-byte alignment, quantization fidelity, layer forward passes, compression ratio.

## Roadmap

- [ ] GGUF model loader
- [ ] RoPE (rotary positional embeddings)
- [ ] Multi-head attention with ternary Q/K/V
- [ ] SwiGLU feed-forward network
- [ ] Full transformer forward pass
- [ ] KV cache for autoregressive generation
- [ ] Token sampler (top-k, top-p, temperature)
- [ ] SIMD kernels (x86 AVX2/512, ARM NEON) via std::arch
- [ ] Engine trait matching SharedEngine interface
