//! Benchmark: scalar vs AVX2 ternary matvec on real model weights.
//!
//! Usage: cargo run -p agentos-bitnet --release --bin bitnet-bench -- <model.gguf>

use std::time::Instant;

use agentos_bitnet::compute::{self, ComputeBackend};
use agentos_bitnet::compute::scalar::ScalarBackend;
use agentos_bitnet::gguf::GgufFile;
use agentos_bitnet::ops::quantize::quantize_absmax;

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: bitnet-bench <model.gguf>");
        std::process::exit(1);
    });

    println!("=== BitNet Compute Backend Benchmark ===\n");

    // Detect backend
    let backend = compute::detect();
    println!("Detected backend: {}", backend.name());

    // Load a few representative tensors from the model
    let gguf = GgufFile::open(&model_path).expect("failed to open GGUF");
    let config = gguf.model_config().expect("failed to read config");

    println!("Model: {} layers, embed_dim={}, intermediate={}\n",
        config.n_layers, config.embedding_dim, config.intermediate_size);

    // Load layer 0 Q projection (embed_dim × embed_dim = 2560×2560)
    let (q_weights, _q_scale) = gguf.load_ternary("blk.0.attn_q.weight")
        .expect("failed to load Q weights");
    println!("Q projection: {}×{} ({} packed bytes)",
        q_weights.rows(), q_weights.cols(), q_weights.packed_len());

    // Load layer 0 FFN gate (intermediate × embed_dim = 6912×2560)
    let (gate_weights, _gate_scale) = gguf.load_ternary("blk.0.ffn_gate.weight")
        .expect("failed to load gate weights");
    println!("FFN gate:     {}×{} ({} packed bytes)",
        gate_weights.rows(), gate_weights.cols(), gate_weights.packed_len());

    // Generate a realistic input vector (quantized random-ish data)
    let embed_dim = config.embedding_dim as usize;
    let fake_input: Vec<f32> = (0..embed_dim)
        .map(|i| ((i * 17 + 5) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let (input_q, _scale) = quantize_absmax(&fake_input);

    let scalar = ScalarBackend;
    let warmup = 3;
    let iterations = 20;

    // --- Benchmark Q projection (square matrix) ---
    println!("\n--- Q projection ({}×{}) ---", q_weights.rows(), q_weights.cols());

    // Warmup
    for _ in 0..warmup {
        let _ = scalar.ternary_matvec(&q_weights, &input_q);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = scalar.ternary_matvec(&q_weights, &input_q);
    }
    let scalar_q_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("  Scalar: {:.0} µs/call", scalar_q_us);

    // Warmup
    for _ in 0..warmup {
        let _ = backend.ternary_matvec(&q_weights, &input_q);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = backend.ternary_matvec(&q_weights, &input_q);
    }
    let backend_q_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("  {}: {:.0} µs/call", backend.name(), backend_q_us);
    println!("  Speedup: {:.1}×", scalar_q_us / backend_q_us);

    // Verify correctness
    let y_scalar = scalar.ternary_matvec(&q_weights, &input_q);
    let y_backend = backend.ternary_matvec(&q_weights, &input_q);
    assert_eq!(y_scalar, y_backend, "MISMATCH: scalar vs {} on Q projection", backend.name());
    println!("  Correctness: VERIFIED ✓");

    // --- Benchmark FFN gate (wide matrix) ---
    println!("\n--- FFN gate ({}×{}) ---", gate_weights.rows(), gate_weights.cols());

    for _ in 0..warmup {
        let _ = scalar.ternary_matvec(&gate_weights, &input_q);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = scalar.ternary_matvec(&gate_weights, &input_q);
    }
    let scalar_gate_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("  Scalar: {:.0} µs/call", scalar_gate_us);

    for _ in 0..warmup {
        let _ = backend.ternary_matvec(&gate_weights, &input_q);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = backend.ternary_matvec(&gate_weights, &input_q);
    }
    let backend_gate_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("  {}: {:.0} µs/call", backend.name(), backend_gate_us);
    println!("  Speedup: {:.1}×", scalar_gate_us / backend_gate_us);

    let y_scalar = scalar.ternary_matvec(&gate_weights, &input_q);
    let y_backend = backend.ternary_matvec(&gate_weights, &input_q);
    assert_eq!(y_scalar, y_backend, "MISMATCH: scalar vs {} on FFN gate", backend.name());
    println!("  Correctness: VERIFIED ✓");

    // --- Estimate full-model impact ---
    // Per layer: 4 attn projections (Q,K,V,O) + 3 FFN projections (gate, up, down)
    // K,V are smaller (n_kv_heads < n_heads), but approximate as:
    //   4 × Q-size + 3 × gate-size per layer
    let layers = config.n_layers as f64;
    let scalar_per_layer_ms = (4.0 * scalar_q_us + 3.0 * scalar_gate_us) / 1000.0;
    let backend_per_layer_ms = (4.0 * backend_q_us + 3.0 * backend_gate_us) / 1000.0;
    let scalar_total_ms = scalar_per_layer_ms * layers;
    let backend_total_ms = backend_per_layer_ms * layers;

    println!("\n--- Full model estimate ({} layers) ---", config.n_layers);
    println!("  Scalar: {:.0} ms/token (matmul only)", scalar_total_ms);
    println!("  {}:   {:.0} ms/token (matmul only)", backend.name(), backend_total_ms);
    println!("  Speedup: {:.1}×", scalar_total_ms / backend_total_ms);
    println!("  Est. tok/s (matmul-bound): {:.1} → {:.1}",
        1000.0 / scalar_total_ms, 1000.0 / backend_total_ms);
}
