# agentos-bitnet — Status & Implementation Notes

## What This Is

A pure-Rust 1.58-bit inference engine for ternary LLMs, built as a subproject of AgentOS. Inspired by Microsoft's [BitNet.cpp](https://github.com/microsoft/BitNet) but implemented from scratch in Rust with zero external ML dependencies. The goal: AgentOS owns the full inference stack, so `cargo build` is all you need — no CUDA, no Python, no C toolchain.

## Why It Matters For AgentOS

AgentOS currently splits brain (cloud LLM via Anthropic API) from body (local Rust kernel). The local inference path (`code-llm` crate) exists but is narrowly scoped to constrained decoding for form-filling in the semantic router.

BitNet b1.58 models use ternary weights {-1, 0, +1}, which means:
- **No floating-point multiplication in the matmul hot loop** — just add, subtract, or skip
- **16× memory compression** vs f32 (4 weights per byte)
- **Models run on CPU at usable speed** — Microsoft reports 5-7 tok/s for 100B params on a single CPU

This unlocks:
1. **Fully air-gapped AgentOS** — no cloud dependency for inference
2. **Bob routing locally** — concierge decisions without API roundtrip
3. **Librarian running locally** — context curation at zero API cost
4. **Pi 5 running the full stack** — not just the kernel, the thinking too
5. **Replace code-llm entirely** — one less external dependency, full ownership

## Technical Foundation (BitNet b1.58 Paper)

Reference papers:
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) — BitNet b1.58 architecture
- [bitnet.cpp](https://arxiv.org/abs/2410.16144) — Optimized CPU inference kernels

### How It Works

**Architecture**: Standard LLaMA transformer with all `nn.Linear` layers replaced by `BitLinear`:
- Weights: ternary {-1, 0, +1}, quantized during training via `W̃ = RoundClip(W/γ, -1, 1)` where γ = mean(|W|)
- Activations: 8-bit quantized per-token via absmax scaling
- No bias terms
- Uses RMSNorm (not LayerNorm), SwiGLU activation, rotary positional embeddings (RoPE)

**Forward pass through BitLinear**:
1. Quantize input activations to 8-bit: `scale = max(|x|)/127`, `x_q = round(x/scale)`
2. Ternary matmul (integer accumulation): for each weight, add (+1), subtract (-1), or skip (0)
3. Rescale output: `y = accumulator * activation_scale * weight_scale_γ`

**Two kernel strategies for the matmul**:
- **I2S**: Unpack 2-bit weights from bytes, conditional add/sub per weight. Simple, good baseline.
- **TL1 (Lookup Table)**: Group 2 weights into a 4-bit index. Precompute all 9 possible dot products for each activation pair. Inner loop is pure table lookup — zero arithmetic.

### Weight Encoding

Ternary values pack into 2 bits:
```
0b00 = -1
0b01 =  0
0b10 = +1
0b11 = unused (treated as zero)
```
4 weights per byte, little-endian bit order: `[w0:2][w1:2][w2:2][w3:2]`.

### LUT-9 Indexing

For a pair of weights (w0, w1) and activations (a0, a1), the 9 possible outcomes:
```
idx  w0  w1  result
 0   -1  -1  -a0 - a1
 1   -1   0  -a0
 2   -1  +1  -a0 + a1
 3    0  -1       - a1
 4    0   0         0
 5    0  +1       + a1
 6   +1  -1  +a0 - a1
 7   +1   0  +a0
 8   +1  +1  +a0 + a1
```

The 4-bit packed pair `w0_bits | (w1_bits << 2)` indexes through a 16-entry remap table (`PAIR_TO_LUT_INDEX`) to select the correct LUT entry. The remap handles the sparse mapping (only 9 of 16 bit patterns are valid).

**Important implementation note**: The LUT index formula is `ternary_idx(w0) * 3 + ternary_idx(w1)`, where w0 is in the LOW bits (0-1) and w1 is in the HIGH bits (2-3) of the pair. Getting this ordering wrong causes silent corruption — we caught this in testing via exhaustive I2S cross-validation.

## What's Built (Phase 1 — Foundation)

### Crate: `crates/bitnet/`
- Added to workspace in root `Cargo.toml`
- Dependencies: only `thiserror` and `tracing` (dev: `tempfile`)
- **50 tests, all passing, zero warnings**

### Module Map

```
crates/bitnet/src/
├── lib.rs              # Public API, re-exports
├── tensor.rs           # Core tensor types (480 lines)
│   ├── Ternary         # Enum: Neg/Zero/Pos with 2-bit encoding
│   ├── TernaryTensor   # 2-bit packed weight matrix
│   ├── ActivationTensor # 8-bit quantized with f32 scale
│   └── FloatTensor     # Plain f32 for norms/embeddings
├── ops/
│   ├── mod.rs          # Module declarations
│   ├── matmul.rs       # I2S ternary matvec kernel (250 lines)
│   ├── lut.rs          # TL1 lookup-table matvec kernel (290 lines)
│   └── quantize.rs     # Absmax 8-bit quantization (130 lines)
└── layers/
    ├── mod.rs          # Module declarations
    ├── bitlinear.rs    # BitLinear layer: quantize → matmul → rescale (175 lines)
    └── rmsnorm.rs      # RMSNorm: (x/√mean(x²)+ε) * γ (100 lines)
```

### Test Coverage

| Module | Tests | What's Covered |
|--------|-------|----------------|
| `tensor.rs` | 16 | Bit packing roundtrip, non-aligned cols, set/get, zeros, row_bytes with sub-byte offsets, activation quantize/dequantize, scale correctness, 16× compression ratio |
| `ops/matmul.rs` | 10 | Identity, negation, mixed weights, all-zero/all-pos/all-neg, non-aligned cols, multi-row, scaled output, batch matmul, accumulator range safety |
| `ops/lut.rs` | 8 | LUT-9 value correctness, pair index mapping, exhaustive 9-combo cross-validation vs I2S, aligned/unaligned/odd cols, multi-row, large dimension (512×64) |
| `ops/quantize.rs` | 6 | Absmax scale, zero input, roundtrip fidelity, boundary clamping, per-token independence, in-place dequantize |
| `layers/bitlinear.rs` | 5 | Identity forward, negation forward, scale factor, batch forward, debug format |
| `layers/rmsnorm.rs` | 5 | Unit weight normalize, scale weights, mixed input, zero input with eps, forward_into equivalence |

### Key Design Decisions

1. **Sub-byte row alignment**: When `row * cols` isn't a multiple of 4, row data starts mid-byte. `row_bytes()` returns a `start_offset` that both kernels respect. This costs nothing for aligned dimensions (which real models always have) but prevents subtle bugs.

2. **LUT cross-validation**: Every LUT test verifies against the I2S kernel. This is how we caught the `PAIR_TO_LUT_INDEX` transposition bug. The two kernels serve as mutual oracles.

3. **i32 accumulators**: With 8-bit activations (max 127) and realistic hidden dims (≤ 16384), the worst-case accumulator is `127 × 16384 = 2,080,768`, well within i32 range. Verified by test.

4. **No unsafe**: Matching AgentOS convention. SIMD will be added later via `std::arch` with safe wrappers, gated behind `cfg(target_arch)`.

## What's Next (Phase 2 — Transformer Stack)

Priority order for the next implementation session:

### 2a. GGUF Model Loader (`gguf.rs`)
- Parse GGUF header, metadata, and tensor blocks
- Load ternary weights from I2_S quantized tensors
- Load f32/f16 weights for embeddings and norms
- Extract model hyperparameters (n_layers, n_heads, hidden_dim, vocab_size, etc.)
- This unblocks running real pretrained models

### 2b. Remaining Transformer Layers
- **RoPE** (`layers/rope.rs`): Rotary positional embeddings — precompute sin/cos frequency pairs, apply complex rotation to Q/K
- **Attention** (`layers/attention.rs`): Multi-head self-attention with ternary Q/K/V/O projections, KV cache support
- **SwiGLU FFN** (`layers/ffn.rs`): `SwiGLU(x) = (xW_gate ⊙ SiLU(xW_up)) W_down` — three BitLinear layers per block
- **Embedding** (`layers/embedding.rs`): Token embedding lookup (f16 weights, not ternary)

### 2c. Full Transformer (`transformer.rs`)
- `TransformerBlock`: RmsNorm → Attention → residual → RmsNorm → FFN → residual
- `Transformer`: Embedding → N × TransformerBlock → RmsNorm → linear head → logits
- Forward pass for a single token position (autoregressive)

### 2d. Generation Pipeline
- **KV Cache** (`kv_cache.rs`): Pre-allocated key/value buffers per layer, grow with sequence
- **Sampler** (`sampler.rs`): Temperature, top-k, top-p, repetition penalty
- **Engine** (`engine.rs`): High-level API matching AgentOS `SharedEngine` interface — `load_model()`, `generate()`, `complete_constrained()`

### 2e. SIMD Acceleration
- x86: AVX2 (256-bit), AVX-512 (512-bit) via `std::arch::x86_64`
- ARM: NEON (128-bit) via `std::arch::aarch64`
- Process 16-64 ternary weights per SIMD instruction
- Safe wrappers, feature-gated, scalar fallback always available

## How It Plugs Into AgentOS

Current local inference path:
```
PipelineBuilder::with_local_inference()
  → loads ~/.agentos/models/*.gguf
  → creates SharedEngine (Arc<Mutex<InferenceEngine>>)
  → InferenceEngine from code-llm crate (external dep)
  → used by LocalFormFiller for constrained decoding
  → falls back to CloudFormFiller if local fails
```

Target state:
```
PipelineBuilder::with_local_inference()
  → loads ~/.agentos/models/*.gguf
  → creates SharedEngine (Arc<Mutex<BitNetEngine>>)
  → BitNetEngine from crates/bitnet (workspace crate, no external dep)
  → full general-purpose inference, not just constrained decoding
  → can run Bob, Librarian, specialists locally
  → cloud becomes optional, not required
```

The key interface to implement in `engine.rs`:
```rust
pub struct BitNetEngine { /* loaded model, KV cache, config */ }

impl BitNetEngine {
    pub fn from_gguf(model_path: &Path, config: EngineConfig) -> Result<Self>;
    pub fn generate(&mut self, prompt: &[u32], max_tokens: usize, sampler: &Sampler) -> Vec<u32>;
    pub fn complete_constrained(&mut self, prompt: &str, constraint: &Schema, max_tokens: usize) -> String;
}
```

## Build & Test

```bash
# From repo root (requires sibling repos: rust-pipeline, code-llm, d2)
cargo test -p agentos-bitnet

# If sibling repos aren't present, stub them:
# mkdir -p ../code-llm/src ../rust-pipeline/src ../d2/src
# echo '[package]\nname="code-llm"\nversion="0.1.0"\nedition="2021"' > ../code-llm/Cargo.toml
# (same pattern for rust-pipeline → "rust-pipeline", d2 → "d2-ascii")
# echo '// stub' > ../code-llm/src/lib.rs (etc.)
```

## Files Changed

- `Cargo.toml` (root): Added `crates/bitnet` to workspace members
- `crates/bitnet/Cargo.toml`: New crate manifest
- `crates/bitnet/CLAUDE.md`: Agent-facing quick reference
- `crates/bitnet/STATUS.md`: This file
- `crates/bitnet/src/lib.rs`: Crate root, public API
- `crates/bitnet/src/tensor.rs`: Ternary, ActivationTensor, FloatTensor
- `crates/bitnet/src/ops/mod.rs`: Ops module
- `crates/bitnet/src/ops/matmul.rs`: I2S ternary matmul kernel
- `crates/bitnet/src/ops/lut.rs`: TL1 lookup-table kernel
- `crates/bitnet/src/ops/quantize.rs`: 8-bit activation quantization
- `crates/bitnet/src/layers/mod.rs`: Layers module
- `crates/bitnet/src/layers/bitlinear.rs`: BitLinear ternary layer
- `crates/bitnet/src/layers/rmsnorm.rs`: RMSNorm layer
