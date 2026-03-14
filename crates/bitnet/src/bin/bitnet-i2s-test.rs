//! Test different I2S interpretations to find the correct one.
//!
//! The I2S format from BitNet.cpp might use:
//! A) Interleaved 128-element blocks, high bits first (current implementation)
//! B) Sequential 2-bit packing, low bits first (simplest interpretation)
//! C) Interleaved 128-element blocks, low bits first
//! D) Sequential 2-bit packing, high bits first
//!
//! This binary tests the model's output quality with each interpretation.

use agentos_bitnet::gguf::GgufFile;
use agentos_bitnet::layers::bitlinear::BitLinear;
use agentos_bitnet::layers::rmsnorm::RmsNorm;
use agentos_bitnet::tensor::{Ternary, TernaryTensor};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: bitnet-i2s-test <model.gguf>");
        std::process::exit(1);
    });

    let gguf = GgufFile::open(&path).expect("failed to open GGUF");
    let config = gguf.model_config().expect("config");

    // Load the embedding table and attn_norm for layer 0
    let embedding = gguf.load_float("token_embd.weight").expect("embedding");
    let attn_norm_w = gguf.load_float("blk.0.attn_norm.weight").expect("attn_norm");
    let attn_norm = RmsNorm::new(attn_norm_w.data().to_vec(), config.rms_norm_eps);

    let embed_dim = config.embedding_dim as usize;

    // Get raw tensor data for blk.0.attn_q.weight
    let q_info = gguf.tensor_info("blk.0.attn_q.weight").expect("no q tensor");
    let q_data = gguf.read_tensor_data_pub(q_info).expect("read q data");
    let q_n_elements = q_info.n_elements;
    let q_rows = q_info.shape[0];
    let q_cols = q_info.shape[1];
    println!("Q tensor: {}x{}, {} elements, {} bytes of data",
        q_rows, q_cols, q_n_elements, q_data.len());

    // Extract scale (last 4 bytes)
    let packed_bytes = (q_n_elements as usize).div_ceil(4);
    let scale = if q_data.len() >= packed_bytes + 4 {
        let sb = &q_data[packed_bytes..packed_bytes + 4];
        f32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]])
    } else {
        1.0
    };
    println!("Scale from I2S: {scale}");

    // Get a test embedding (token "The" = 791 after BOS)
    let tok_id = 791u32; // "The"
    let emb = &embedding.data()[tok_id as usize * embed_dim..(tok_id as usize + 1) * embed_dim];

    // Normalize it (as the real model does)
    let normed = attn_norm.forward(emb);

    println!("\nNormalized embedding first 8: {:?}", &normed[..8]);
    println!("Normalized embedding L2 norm: {:.4}",
        normed.iter().map(|x| x * x).sum::<f32>().sqrt());

    // GGUF TQ2 remap: 0→Neg(-1), 1→Zero(0), 2→Pos(+1), 3→Zero(0)
    let remap = |code: u8| -> Ternary {
        match code {
            0 => Ternary::Neg,
            2 => Ternary::Pos,
            _ => Ternary::Zero,
        }
    };

    // === Interpretation A: Interleaved, high bits first (CURRENT) ===
    let q_a = unpack_interleaved_high(&q_data, q_n_elements as usize, &remap);
    test_interpretation("A: interleaved, high-first", &q_a, q_rows, q_cols, scale, &normed);

    // === Interpretation B: Sequential, low bits first ===
    let q_b = unpack_sequential_low(&q_data, q_n_elements as usize, &remap);
    test_interpretation("B: sequential, low-first", &q_b, q_rows, q_cols, scale, &normed);

    // === Interpretation C: Interleaved, low bits first ===
    let q_c = unpack_interleaved_low(&q_data, q_n_elements as usize, &remap);
    test_interpretation("C: interleaved, low-first", &q_c, q_rows, q_cols, scale, &normed);

    // === Interpretation D: Sequential, high bits first ===
    let q_d = unpack_sequential_high(&q_data, q_n_elements as usize, &remap);
    test_interpretation("D: sequential, high-first", &q_d, q_rows, q_cols, scale, &normed);
}

fn test_interpretation(
    name: &str,
    values: &[Ternary],
    rows: usize,
    cols: usize,
    scale: f32,
    input: &[f32],
) {
    let tensor = TernaryTensor::pack(values, rows, cols);
    let layer = BitLinear::new(tensor, scale);
    let output = layer.forward(input);

    let l2 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
    let rms = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
    let max_abs = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    println!("\n  [{name}]");
    println!("    output L2={l2:.2}, RMS={rms:.4}, max_abs={max_abs:.2}");
    println!("    first 8: {:?}", &output[..8.min(output.len())]);

    // Check if the output head vectors have reasonable structure
    // (a correctly loaded model should produce Q vectors where different heads
    // have distinct patterns, not random noise)
    let head_dim = cols / 20; // n_heads = 20
    let mut head_norms: Vec<f32> = Vec::new();
    for h in 0..20.min(rows / head_dim) {
        let start = h * head_dim;
        let end = start + head_dim;
        let head = &output[start..end];
        let norm = head.iter().map(|x| x * x).sum::<f32>().sqrt();
        head_norms.push(norm);
    }
    let mean_norm = head_norms.iter().sum::<f32>() / head_norms.len() as f32;
    let norm_std = (head_norms.iter().map(|n| (n - mean_norm).powi(2)).sum::<f32>()
        / head_norms.len() as f32).sqrt();
    println!("    head norms: mean={mean_norm:.2}, std={norm_std:.2}, cv={:.3}",
        norm_std / mean_norm);
}

fn unpack_interleaved_high(data: &[u8], n: usize, remap: &dyn Fn(u8) -> Ternary) -> Vec<Ternary> {
    const QK: usize = 128;
    const BPB: usize = 32;
    let mut vals = vec![Ternary::Zero; n];
    let n_blocks = n / QK;
    for bi in 0..n_blocks {
        let bs = bi * BPB;
        let vs = bi * QK;
        for p in 0..BPB {
            let b = data[bs + p];
            vals[vs + p]      = remap((b >> 6) & 0x03);
            vals[vs + p + 32] = remap((b >> 4) & 0x03);
            vals[vs + p + 64] = remap((b >> 2) & 0x03);
            vals[vs + p + 96] = remap(b & 0x03);
        }
    }
    // remainder
    let rem = n % QK;
    if rem > 0 {
        let bs = n_blocks * BPB;
        let vs = n_blocks * QK;
        for i in 0..rem {
            let byte_idx = bs + i / 4;
            let shift = 6 - (i % 4) * 2;
            vals[vs + i] = remap((data[byte_idx] >> shift) & 0x03);
        }
    }
    vals
}

fn unpack_interleaved_low(data: &[u8], n: usize, remap: &dyn Fn(u8) -> Ternary) -> Vec<Ternary> {
    const QK: usize = 128;
    const BPB: usize = 32;
    let mut vals = vec![Ternary::Zero; n];
    let n_blocks = n / QK;
    for bi in 0..n_blocks {
        let bs = bi * BPB;
        let vs = bi * QK;
        for p in 0..BPB {
            let b = data[bs + p];
            vals[vs + p]      = remap(b & 0x03);
            vals[vs + p + 32] = remap((b >> 2) & 0x03);
            vals[vs + p + 64] = remap((b >> 4) & 0x03);
            vals[vs + p + 96] = remap((b >> 6) & 0x03);
        }
    }
    vals
}

fn unpack_sequential_low(data: &[u8], n: usize, remap: &dyn Fn(u8) -> Ternary) -> Vec<Ternary> {
    let mut vals = vec![Ternary::Zero; n];
    for (i, item) in vals.iter_mut().enumerate().take(n) {
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        *item = remap((data[byte_idx] >> shift) & 0x03);
    }
    vals
}

fn unpack_sequential_high(data: &[u8], n: usize, remap: &dyn Fn(u8) -> Ternary) -> Vec<Ternary> {
    let mut vals = vec![Ternary::Zero; n];
    for (i, item) in vals.iter_mut().enumerate().take(n) {
        let byte_idx = i / 4;
        let shift = 6 - (i % 4) * 2;
        *item = remap((data[byte_idx] >> shift) & 0x03);
    }
    vals
}
