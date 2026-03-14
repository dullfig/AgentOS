//! Diagnostic tool — sanity-check model loading, tokenization, and tensor unpacking.

use agentos_bitnet::gguf::GgufFile;
use agentos_bitnet::layers::bitlinear::BitLinear;
use agentos_bitnet::layers::rope::RoPE;
use agentos_bitnet::loader::load_model;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bitnet-diag <model.gguf> [--quick]");
        std::process::exit(1);
    }

    let path = &args[1];
    let quick = args.iter().any(|a| a == "--quick");

    // === 0. GGUF Metadata Dump (RoPE and architecture) ===
    println!("=== GGUF Metadata ===");
    let gguf = GgufFile::open(path).expect("failed to open GGUF");
    for (key, val) in gguf.metadata() {
        if key.contains("rope") || key.contains("general.") || key.contains("arch") {
            println!("  {key} = {val:?}");
        }
    }
    // Print model config
    let config = gguf.model_config().expect("failed to read config");
    println!("  [config] rope_theta={}, n_heads={}, n_kv_heads={}, embed_dim={}, head_dim={}",
        config.rope_theta, config.n_heads, config.n_kv_heads, config.embedding_dim,
        config.embedding_dim / config.n_heads);

    // === 1. Tokenizer check ===
    println!("\n=== Tokenizer ===");

    let model_type = gguf
        .get_metadata("tokenizer.ggml.model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    println!("tokenizer model: {model_type}");

    let loaded = load_model(path).expect("failed to load model");
    let tok = &loaded.tokenizer;

    let test_texts = &["hello", "Hello, what is your name?", "The capital of France is"];
    for text in test_texts {
        let tokens = tok.encode(text, true);
        let decoded = tok.decode(&tokens);
        println!("  \"{text}\"");
        println!("    tokens: {:?}", tokens);
        println!("    decoded: \"{decoded}\"");
        for &t in &tokens {
            println!("      {} → {:?}", t, tok.token(t));
        }
    }

    // === 2. Ternary distribution check ===
    println!("\n=== Ternary Weight Distribution ===");
    let tensor_names = ["blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "blk.15.attn_q.weight"];
    for name in &tensor_names {
        let (tw, scale) = gguf.load_ternary(name).expect(name);
        let rows = tw.rows();
        let cols = tw.cols();

        // Count ternary values by unpacking
        let mut neg = 0u64;
        let mut zero = 0u64;
        let mut pos = 0u64;
        for r in 0..rows {
            for c in 0..cols {
                match tw.get(r, c) {
                    agentos_bitnet::tensor::Ternary::Neg => neg += 1,
                    agentos_bitnet::tensor::Ternary::Zero => zero += 1,
                    agentos_bitnet::tensor::Ternary::Pos => pos += 1,
                }
            }
        }
        let total = (neg + zero + pos) as f64;
        println!("  {name} [{rows}×{cols}] scale={scale:.6}");
        println!("    neg={neg} ({:.1}%), zero={zero} ({:.1}%), pos={pos} ({:.1}%)",
            neg as f64 / total * 100.0,
            zero as f64 / total * 100.0,
            pos as f64 / total * 100.0,
        );
    }

    // === 3. Attention Probe (Q·K dot products with RoPE) ===
    println!("\n=== Attention Probe ===");
    attention_probe(&gguf, &loaded);

    if quick {
        println!("\n[--quick mode: skipping full model forward passes]");
        return;
    }

    // === 4. Embedding check ===
    println!("\n=== Embedding Check ===");
    let model = &loaded.model;
    // Feed BOS token through the model to check if embeddings look sane
    let bos = tok.bos_token_id();
    let logits = model.forward(&[bos], 0);
    let vocab_size = model.vocab_size();
    let logits_slice = &logits[..vocab_size];

    let max_val = logits_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_val = logits_slice.iter().copied().fold(f32::INFINITY, f32::min);
    let mean = logits_slice.iter().sum::<f32>() / vocab_size as f32;
    let has_nan = logits_slice.iter().any(|v| v.is_nan());
    let has_inf = logits_slice.iter().any(|v| v.is_infinite());

    println!("  BOS token logits: min={min_val:.4}, max={max_val:.4}, mean={mean:.4}");
    println!("  has_nan={has_nan}, has_inf={has_inf}");

    // Top-5 tokens from BOS
    let mut indexed: Vec<(usize, f32)> = logits_slice.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("  top-5 from BOS:");
    for (id, score) in indexed.iter().take(5) {
        println!("    {} ({:.4}): {:?}", id, score, tok.token(*id as u32));
    }
    println!("  bottom-5 from BOS:");
    for (id, score) in indexed.iter().rev().take(5) {
        println!("    {} ({:.4}): {:?}", id, score, tok.token(*id as u32));
    }

    // === 5. Non-cached vs cached generation comparison ===
    println!("\n=== Prompt Forward (non-cached) ===");
    let prompt = "The capital of France is";
    let prompt_tokens = tok.encode(prompt, true);
    println!("  prompt tokens: {:?}", prompt_tokens);

    // Non-cached: feed all tokens at once
    let logits_all = model.forward(&prompt_tokens, 0);
    let last_start = (prompt_tokens.len() - 1) * vocab_size;
    let last_logits = &logits_all[last_start..last_start + vocab_size];

    let mut indexed: Vec<(usize, f32)> = last_logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("  non-cached top-5 next token:");
    for (id, score) in indexed.iter().take(5) {
        println!("    {} ({:.4}): {:?}", id, score, tok.token(*id as u32));
    }
    // Check where "Paris" ranks
    if let Some(paris_id) = tok.token_id("ĠParis") {
        let paris_score = last_logits[paris_id as usize];
        let rank = indexed.iter().position(|(id, _)| *id == paris_id as usize);
        println!("  'Paris' ({}): score={:.4}, rank={:?}", paris_id, paris_score, rank);
    }

    // Cached: feed tokens through KV cache
    println!("\n=== Prompt Forward (cached) ===");
    let mut cache = model.create_kv_cache(512);
    let cached_logits = model.forward_cached(&prompt_tokens, &mut cache);
    let last_start_c = (prompt_tokens.len() - 1) * vocab_size;
    let last_logits_c = &cached_logits[last_start_c..last_start_c + vocab_size];

    let mut indexed_c: Vec<(usize, f32)> = last_logits_c.iter().copied().enumerate().collect();
    indexed_c.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("  cached top-5 next token:");
    for (id, score) in indexed_c.iter().take(5) {
        println!("    {} ({:.4}): {:?}", id, score, tok.token(*id as u32));
    }

    // Check if non-cached and cached agree
    let agree = indexed[0].0 == indexed_c[0].0;
    println!("  top-1 match: {} (non-cached={}, cached={})", agree, indexed[0].0, indexed_c[0].0);

    // Greedy decode 5 tokens non-cached (one at a time, no KV cache)
    println!("\n=== Greedy Decode (non-cached, 5 tokens) ===");
    let mut gen_tokens = prompt_tokens.clone();
    print!("  \"{prompt}\"");
    for _ in 0..5 {
        let logits = model.forward(&gen_tokens, 0);
        let last = (gen_tokens.len() - 1) * vocab_size;
        let last_l = &logits[last..last + vocab_size];
        let next = last_l.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0 as u32;
        let text = tok.decode(&[next]);
        print!("{text}");
        gen_tokens.push(next);
    }
    println!();
}

/// Probe attention by manually computing Q·K dot products with both RoPE conventions.
/// This tests whether the model's Q/K projections + our RoPE produce position-dependent scores.
fn attention_probe(gguf: &GgufFile, loaded: &agentos_bitnet::loader::LoadedModel) {
    let config = &loaded.config;
    let embed_dim = config.embedding_dim as usize;
    let n_heads = config.n_heads as usize;
    let head_dim = embed_dim / n_heads;

    // Load Q and K projections from layer 0
    let (q_w, q_s) = gguf.load_ternary("blk.0.attn_q.weight").expect("q weights");
    let (k_w, k_s) = gguf.load_ternary("blk.0.attn_k.weight").expect("k weights");
    let q_proj = BitLinear::new(q_w, q_s);
    let k_proj = BitLinear::new(k_w, k_s);

    // Get embedding vectors for two different tokens
    let tok = &loaded.tokenizer;
    let bos_id = tok.bos_token_id();
    let the_tokens = tok.encode("The capital", false);
    let tok0 = if the_tokens.is_empty() { bos_id } else { the_tokens[0] }; // "The"
    let tok1 = if the_tokens.len() > 1 { the_tokens[1] } else { bos_id }; // "capital" or bos

    println!("  tokens used: tok0={} ({:?}), tok1={} ({:?})",
        tok0, tok.token(tok0), tok1, tok.token(tok1));

    // Look up embeddings
    let embed_data = loaded.model.embedding_data();
    let emb0 = &embed_data[tok0 as usize * embed_dim..(tok0 as usize + 1) * embed_dim];
    let emb1 = &embed_data[tok1 as usize * embed_dim..(tok1 as usize + 1) * embed_dim];

    // Project through Q and K
    let q0 = q_proj.forward(emb0);
    let q1 = q_proj.forward(emb1);
    let k0 = k_proj.forward(emb0);
    let k1 = k_proj.forward(emb1);

    println!("  Q0 first 8: {:?}", &q0[..8.min(q0.len())]);
    println!("  K0 first 8: {:?}", &k0[..8.min(k0.len())]);
    println!("  Q0 L2 norm: {:.4}", l2_norm(&q0));
    println!("  K0 L2 norm: {:.4}", l2_norm(&k0));

    // Test head 0 with INTERLEAVED RoPE (our current implementation)
    let rope_interleaved = RoPE::new(head_dim, config.rope_theta);
    let q0_h0 = &q0[..head_dim];
    let q1_h0 = &q1[..head_dim];
    // K head dim may differ from Q head dim if n_kv_heads < n_heads
    let n_kv_heads = config.n_kv_heads as usize;
    let kv_head_dim = k_proj.out_features() / n_kv_heads;
    let k0_h0 = &k0[..kv_head_dim];
    let k1_h0 = &k1[..kv_head_dim];

    println!("  head_dim={head_dim}, kv_head_dim={kv_head_dim}");

    // Interleaved RoPE: pairs (0,1), (2,3), ...
    let rope_kv = RoPE::new(kv_head_dim, config.rope_theta);
    let q0_rope_inter = rope_interleaved.forward(q0_h0, 0);
    let q1_rope_inter = rope_interleaved.forward(q1_h0, 1);
    let k0_rope_inter = rope_kv.forward(k0_h0, 0);
    let k1_rope_inter = rope_kv.forward(k1_h0, 1);

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Dot products with interleaved RoPE
    let dot_00_inter = dot_product(&q0_rope_inter, &k0_rope_inter) * scale;
    let dot_01_inter = dot_product(&q0_rope_inter, &k1_rope_inter) * scale;
    let dot_10_inter = dot_product(&q1_rope_inter, &k0_rope_inter) * scale;
    let dot_11_inter = dot_product(&q1_rope_inter, &k1_rope_inter) * scale;

    println!("  [INTERLEAVED RoPE] head 0 attention scores (scaled by 1/sqrt(head_dim)):");
    println!("    Q(pos=0)·K(pos=0) = {dot_00_inter:.6}");
    println!("    Q(pos=0)·K(pos=1) = {dot_01_inter:.6}");
    println!("    Q(pos=1)·K(pos=0) = {dot_10_inter:.6}");
    println!("    Q(pos=1)·K(pos=1) = {dot_11_inter:.6}");
    println!("    score variance: {:.6}", variance(&[dot_00_inter, dot_01_inter, dot_10_inter, dot_11_inter]));

    // Halved RoPE: pairs (i, i+half)
    let q0_rope_halved = rope_halved(q0_h0, head_dim, config.rope_theta, 0);
    let q1_rope_halved = rope_halved(q1_h0, head_dim, config.rope_theta, 1);
    let k0_rope_halved = rope_halved(k0_h0, kv_head_dim, config.rope_theta, 0);
    let k1_rope_halved = rope_halved(k1_h0, kv_head_dim, config.rope_theta, 1);

    let dot_00_halved = dot_product(&q0_rope_halved, &k0_rope_halved) * scale;
    let dot_01_halved = dot_product(&q0_rope_halved, &k1_rope_halved) * scale;
    let dot_10_halved = dot_product(&q1_rope_halved, &k0_rope_halved) * scale;
    let dot_11_halved = dot_product(&q1_rope_halved, &k1_rope_halved) * scale;

    println!("  [HALVED RoPE] head 0 attention scores (scaled by 1/sqrt(head_dim)):");
    println!("    Q(pos=0)·K(pos=0) = {dot_00_halved:.6}");
    println!("    Q(pos=0)·K(pos=1) = {dot_01_halved:.6}");
    println!("    Q(pos=1)·K(pos=0) = {dot_10_halved:.6}");
    println!("    Q(pos=1)·K(pos=1) = {dot_11_halved:.6}");
    println!("    score variance: {:.6}", variance(&[dot_00_halved, dot_01_halved, dot_10_halved, dot_11_halved]));

    // Without RoPE (baseline)
    let dot_00_none = dot_product(q0_h0, k0_h0) * scale;
    let dot_01_none = dot_product(q0_h0, k1_h0) * scale;
    let dot_10_none = dot_product(q1_h0, k0_h0) * scale;
    let dot_11_none = dot_product(q1_h0, k1_h0) * scale;

    println!("  [NO RoPE] head 0 attention scores (baseline):");
    println!("    Q(tok0)·K(tok0) = {dot_00_none:.6}");
    println!("    Q(tok0)·K(tok1) = {dot_01_none:.6}");
    println!("    Q(tok1)·K(tok0) = {dot_10_none:.6}");
    println!("    Q(tok1)·K(tok1) = {dot_11_none:.6}");
    println!("    score variance: {:.6}", variance(&[dot_00_none, dot_01_none, dot_10_none, dot_11_none]));
}

/// Apply RoPE with halved convention: pairs (i, i+half) instead of (2i, 2i+1).
fn rope_halved(input: &[f32], dim: usize, base: f32, pos: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut output = input.to_vec();
    let p = pos as f32;

    for i in 0..half {
        let freq = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
        let angle = p * freq;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let x0 = input[i];
        let x1 = input[i + half];

        output[i] = x0 * cos_a - x1 * sin_a;
        output[i + half] = x0 * sin_a + x1 * cos_a;
    }
    output
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn variance(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32
}
