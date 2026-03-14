//! Model loader — wires GGUF tensor data into a live TransformerModel.
//!
//! LLaMA/BitNet GGUF tensor naming convention:
//!
//! - `token_embd.weight` — embedding table (float)
//! - `blk.{i}.attn_q.weight` — Q projection (ternary)
//! - `blk.{i}.attn_k.weight` — K projection (ternary)
//! - `blk.{i}.attn_v.weight` — V projection (ternary)
//! - `blk.{i}.attn_output.weight` — O projection (ternary)
//! - `blk.{i}.ffn_gate.weight` — SwiGLU gate (ternary)
//! - `blk.{i}.ffn_up.weight` — SwiGLU up (ternary)
//! - `blk.{i}.ffn_down.weight` — SwiGLU down (ternary)
//! - `blk.{i}.attn_norm.weight` — attention RMSNorm (float)
//! - `blk.{i}.ffn_norm.weight` — FFN RMSNorm (float)
//! - `blk.{i}.attn_sub_norm.weight` — post-attention sub-norm (BitNet b1.58)
//! - `blk.{i}.ffn_sub_norm.weight` — pre-down-projection sub-norm (BitNet b1.58)
//! - `output_norm.weight` — final RMSNorm (float)
//! - `output.weight` — output projection (ternary, or absent if tied)

use tracing::info;

use crate::compute;
use crate::gguf::{GgufFile, GgufError, ModelConfig};
use crate::layers::attention::MultiHeadAttention;
use crate::layers::bitlinear::BitLinear;
use crate::layers::model::{OutputProjection, TransformerModel};
use crate::layers::rmsnorm::RmsNorm;
use crate::layers::rope::RoPELayout;
use crate::layers::swiglu::{GateActivation, SwiGLU};
use crate::layers::transformer::TransformerBlock;
use crate::tokenizer::Tokenizer;

/// A fully loaded model ready for inference.
pub struct LoadedModel {
    /// The transformer model.
    pub model: TransformerModel,
    /// The tokenizer.
    pub tokenizer: Tokenizer,
    /// Model hyperparameters.
    pub config: ModelConfig,
}

/// Load a transformer model and tokenizer from a GGUF file.
///
/// This reads all tensor data into memory — for a 2B model at ternary
/// precision, that's roughly ~400MB of weights.
pub fn load_model(path: &str) -> Result<LoadedModel, GgufError> {
    // Hardware detection and boot banner
    let hw = crate::compute::device::HardwareInfo::detect();
    hw.print_boot_banner();

    let gguf = GgufFile::open(path)?;
    let config = gguf.model_config()?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    eprintln!(
        "  [boot] Model: {} layers, {} embed, {} vocab, {:.0}MB weights",
        config.n_layers,
        config.embedding_dim,
        config.vocab_size,
        // Ternary weights: 2 bits per param, plus float norms/embeddings
        (config.vocab_size * config.embedding_dim) as f64 * 4.0 / (1024.0 * 1024.0)
            + (config.n_layers * config.embedding_dim * config.embedding_dim * 7) as f64 * 0.25
                / (1024.0 * 1024.0),
    );

    info!(
        vocab_size = config.vocab_size,
        embed_dim = config.embedding_dim,
        n_layers = config.n_layers,
        n_heads = config.n_heads,
        n_kv_heads = config.n_kv_heads,
        intermediate = config.intermediate_size,
        rope_theta = config.rope_theta,
        "loading model"
    );

    let embed_dim = config.embedding_dim as usize;
    let n_heads = config.n_heads as usize;
    let n_kv_heads = config.n_kv_heads as usize;
    let head_dim = embed_dim / n_heads;
    let intermediate = config.intermediate_size as usize;

    // Detect architecture string (used for RoPE and activation inference)
    let arch = gguf
        .get_metadata("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    // Determine RoPE layout.
    // BitNet models from HuggingFace use halved (NeoX) RoPE convention.
    // BitNet.cpp's converter does NOT apply the Q/K weight permutation,
    // so we need halved RoPE for bitnet architectures.
    let rope_layout = match config.rope_type {
        2 => {
            info!("using halved (NeoX/HF) RoPE layout");
            RoPELayout::Halved
        }
        _ if arch.contains("bitnet") => {
            info!("detected bitnet architecture, using halved (NeoX/HF) RoPE layout");
            RoPELayout::Halved
        }
        _ => {
            info!("using interleaved (llama.cpp) RoPE layout");
            RoPELayout::Interleaved
        }
    };

    let activation = match config.hidden_act.as_str() {
        "relu2" | "relu_squared" | "squared_relu" => {
            info!("using squared ReLU (relu²) activation (from metadata)");
            GateActivation::ReLU2
        }
        "silu" if arch.contains("bitnet") => {
            // Metadata defaulted to "silu" but architecture is bitnet → use relu²
            info!("detected bitnet architecture, using squared ReLU (relu²) activation");
            GateActivation::ReLU2
        }
        other => {
            info!(act = %other, arch, "using SiLU activation");
            GateActivation::SiLU
        }
    };

    // Auto-detect compute backend (scalar / AVX2 / etc.)
    let backend = compute::detect();
    eprintln!("  [boot] Compute: {} ternary kernel", backend.name());
    info!(backend = backend.name(), "compute backend selected");

    // Embedding table
    let embedding = gguf.load_float("token_embd.weight")?;
    info!("loaded embedding: {:?}", embedding.shape());

    // Transformer blocks
    let mut blocks = Vec::with_capacity(config.n_layers as usize);

    for i in 0..config.n_layers as usize {
        // Attention projections (ternary) — plain BitLinear, no sub-norms inside
        let (q_w, q_s) = gguf.load_ternary(&format!("blk.{i}.attn_q.weight"))?;
        let (k_w, k_s) = gguf.load_ternary(&format!("blk.{i}.attn_k.weight"))?;
        let (v_w, v_s) = gguf.load_ternary(&format!("blk.{i}.attn_v.weight"))?;
        let (o_w, o_s) = gguf.load_ternary(&format!("blk.{i}.attn_output.weight"))?;

        let q_proj = BitLinear::with_backend(q_w, q_s, backend.clone());
        let k_proj = BitLinear::with_backend(k_w, k_s, backend.clone());
        let v_proj = BitLinear::with_backend(v_w, v_s, backend.clone());
        let o_proj = BitLinear::with_backend(o_w, o_s, backend.clone());

        let attention = MultiHeadAttention::with_rope_layout(
            q_proj, k_proj, v_proj, o_proj,
            n_heads, n_kv_heads, head_dim, config.rope_theta, rope_layout,
        );

        // FFN projections (ternary)
        let (gate_w, gate_s) = gguf.load_ternary(&format!("blk.{i}.ffn_gate.weight"))?;
        let (up_w, up_s) = gguf.load_ternary(&format!("blk.{i}.ffn_up.weight"))?;
        let (down_w, down_s) = gguf.load_ternary(&format!("blk.{i}.ffn_down.weight"))?;

        let gate_proj = BitLinear::with_backend(gate_w, gate_s, backend.clone());
        let up_proj = BitLinear::with_backend(up_w, up_s, backend.clone());
        let down_proj = BitLinear::with_backend(down_w, down_s, backend.clone());

        // Optional ffn_sub_norm [intermediate_dim]: applied between activation(gate)⊙up and down
        let ffn = if gguf.tensor_info(&format!("blk.{i}.ffn_sub_norm.weight")).is_some() {
            let w = gguf.load_float(&format!("blk.{i}.ffn_sub_norm.weight"))?;
            let sub_norm = RmsNorm::new(w.data().to_vec(), config.rms_norm_eps);
            SwiGLU::with_sub_norm_and_activation(gate_proj, up_proj, down_proj, sub_norm, activation)
        } else {
            SwiGLU::with_activation(gate_proj, up_proj, down_proj, activation)
        };

        // Norms (float)
        let attn_norm_w = gguf.load_float(&format!("blk.{i}.attn_norm.weight"))?;
        let ffn_norm_w = gguf.load_float(&format!("blk.{i}.ffn_norm.weight"))?;

        let attn_norm = RmsNorm::new(attn_norm_w.data().to_vec(), config.rms_norm_eps);
        let ffn_norm = RmsNorm::new(ffn_norm_w.data().to_vec(), config.rms_norm_eps);

        // Optional attn_sub_norm [embed_dim]: applied to attention output before residual
        let attn_sub_norm = if gguf.tensor_info(&format!("blk.{i}.attn_sub_norm.weight")).is_some() {
            let w = gguf.load_float(&format!("blk.{i}.attn_sub_norm.weight"))?;
            Some(RmsNorm::new(w.data().to_vec(), config.rms_norm_eps))
        } else {
            None
        };

        let block = TransformerBlock::with_sub_norms(
            attn_norm, attention, attn_sub_norm, ffn_norm, ffn,
        );
        blocks.push(block);

        info!(layer = i, "loaded transformer block {}/{}", i + 1, config.n_layers);
    }

    // Final norm
    let final_norm_w = gguf.load_float("output_norm.weight")?;
    let final_norm = RmsNorm::new(final_norm_w.data().to_vec(), config.rms_norm_eps);

    // Output projection: check if "output.weight" exists and its type
    let output_proj = if let Some(out_info) = gguf.tensor_info("output.weight") {
        use crate::gguf::GgmlType;
        match out_info.ggml_type {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                let out_tensor = gguf.load_float("output.weight")?;
                info!(
                    shape = ?out_tensor.shape(),
                    dtype = ?out_info.ggml_type,
                    "loaded output projection (float)"
                );
                OutputProjection::Float(out_tensor)
            }
            _ => {
                // Ternary (I2S, TQ1, TQ2)
                let (out_w, out_s) = gguf.load_ternary("output.weight")?;
                info!(
                    rows = out_w.rows(),
                    cols = out_w.cols(),
                    "loaded output projection (ternary)"
                );
                OutputProjection::Linear(BitLinear::with_backend(out_w, out_s, backend.clone()))
            }
        }
    } else {
        info!("using tied embedding for output projection");
        OutputProjection::TiedEmbedding
    };

    let model = TransformerModel::new(embedding, blocks, final_norm, output_proj);
    info!(
        vocab = model.vocab_size(),
        embed = model.embed_dim(),
        layers = model.n_layers(),
        "model loaded successfully"
    );

    // Sanity check
    assert_eq!(
        model.vocab_size(),
        tokenizer.vocab_size(),
        "model vocab size ({}) != tokenizer vocab size ({})",
        model.vocab_size(),
        tokenizer.vocab_size()
    );

    let _ = intermediate; // used implicitly through GGUF tensor shapes

    Ok(LoadedModel {
        model,
        tokenizer,
        config,
    })
}
