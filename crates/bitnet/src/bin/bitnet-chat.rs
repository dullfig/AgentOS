//! Quick-and-dirty chat TUI for testing the ternary inference engine.
//!
//! Usage: bitnet-chat <model.gguf> [--temp 0.7] [--top-k 40] [--top-p 0.9] [--max-tokens 256]

use std::io::{self, BufRead, Write};
use std::time::Instant;

use agentos_bitnet::layers::sampler::SamplerConfig;
use agentos_bitnet::loader::load_model;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: bitnet-chat <model.gguf> [options]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --temp <f32>        Temperature (default: 0.7)");
        eprintln!("  --top-k <usize>     Top-k sampling (default: 40)");
        eprintln!("  --top-p <f32>       Nucleus sampling (default: 0.9)");
        eprintln!("  --max-tokens <n>    Max tokens per response (default: 256)");
        eprintln!("  --max-seq <n>       Max sequence length / KV cache (default: 2048)");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut temperature = 0.7f32;
    let mut top_k = 40usize;
    let mut top_p = 0.9f32;
    let mut max_tokens = 256usize;
    let mut max_seq_len = 2048usize;
    let mut rep_penalty = 1.2f32;
    let mut rep_window = 64usize;

    // Parse optional args
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--temp" => {
                temperature = args[i + 1].parse().expect("invalid --temp value");
                i += 2;
            }
            "--top-k" => {
                top_k = args[i + 1].parse().expect("invalid --top-k value");
                i += 2;
            }
            "--top-p" => {
                top_p = args[i + 1].parse().expect("invalid --top-p value");
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = args[i + 1].parse().expect("invalid --max-tokens value");
                i += 2;
            }
            "--max-seq" => {
                max_seq_len = args[i + 1].parse().expect("invalid --max-seq value");
                i += 2;
            }
            "--rep-penalty" => {
                rep_penalty = args[i + 1].parse().expect("invalid --rep-penalty value");
                i += 2;
            }
            "--rep-window" => {
                rep_window = args[i + 1].parse().expect("invalid --rep-window value");
                i += 2;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    // Load model
    eprintln!("Loading model from: {}", model_path);
    let start = Instant::now();
    let loaded = match load_model(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };
    let load_time = start.elapsed();
    eprintln!(
        "Model loaded in {:.1}s — vocab={}, embed={}, layers={}, ctx={}",
        load_time.as_secs_f64(),
        loaded.config.vocab_size,
        loaded.config.embedding_dim,
        loaded.config.n_layers,
        loaded.config.context_length,
    );
    eprintln!(
        "Sampling: temp={}, top_k={}, top_p={}, max_tokens={}",
        temperature, top_k, top_p, max_tokens,
    );
    eprintln!();
    eprintln!("Type a prompt and press Enter. Ctrl+C to quit.");
    eprintln!("---");

    let model = &loaded.model;
    let tokenizer = &loaded.tokenizer;

    // Detect chat template from model name
    let model_name = loaded.config.model_name.as_deref().unwrap_or("");
    let is_instruct = model_name.to_lowercase().contains("instruct")
        || model_name.to_lowercase().contains("chat");
    if is_instruct {
        eprintln!("Chat mode: instruct model detected, wrapping with <|user|>/<|assistant|> template");
    }

    // Pre-allocate KV cache for the full session
    let mut cache = model.create_kv_cache(max_seq_len);
    let mut total_tokens = 0usize;

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Prompt
        eprint!("\n> ");
        io::stderr().flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Special commands
        if line == "/clear" || line == "/reset" {
            cache.clear();
            total_tokens = 0;
            eprintln!("[KV cache cleared]");
            continue;
        }
        if line == "/quit" || line == "/exit" {
            break;
        }
        if line == "/info" {
            eprintln!(
                "[cache: {}/{} tokens, {:.1}MB]",
                cache.seq_len(),
                max_seq_len,
                cache.memory_bytes() as f64 / (1024.0 * 1024.0),
            );
            continue;
        }

        // Wrap input with chat template if instruct model
        // Format: <|user|>\n{message}\n<|assistant|>\n
        let formatted_input = if is_instruct {
            format!("<|user|>\n{}\n<|assistant|>\n", line)
        } else {
            line.to_string()
        };

        // Tokenize input
        let add_bos = total_tokens == 0; // BOS only on first turn
        let input_tokens = tokenizer.encode(&formatted_input, add_bos);
        let n_input = input_tokens.len();

        // Check if we have room
        if total_tokens + n_input + max_tokens > max_seq_len {
            eprintln!(
                "[Warning: approaching context limit ({}/{}). Use /clear to reset.]",
                total_tokens + n_input,
                max_seq_len,
            );
            if total_tokens + n_input >= max_seq_len {
                eprintln!("[Context full. Use /clear to reset.]");
                continue;
            }
        }

        // Prefill the input
        let prefill_start = Instant::now();
        let prefill_logits = model.forward_cached(&input_tokens, &mut cache);
        let prefill_time = prefill_start.elapsed();
        total_tokens += n_input;

        let sampler_config = SamplerConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: rep_penalty,
            repetition_window: rep_window,
        };
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let mut sampler = agentos_bitnet::layers::sampler::Sampler::new(sampler_config, seed);

        // Sample first token from prefill
        let last_logits_start = (n_input - 1) * model.vocab_size();
        let last_logits = &prefill_logits[last_logits_start..last_logits_start + model.vocab_size()];
        let mut next_token = sampler.sample(last_logits);

        eprintln!(
            "[prefill: {} tokens in {:.1}ms | {:.0} tok/s]",
            n_input,
            prefill_time.as_secs_f64() * 1000.0,
            n_input as f64 / prefill_time.as_secs_f64(),
        );

        // Decode loop — stream tokens to stdout
        let decode_start = Instant::now();
        let mut generated = 0usize;
        let mut recent_text = String::new();

        let eos = tokenizer.eos_token_id();

        // Role markers that signal the model is starting a new turn
        let stop_markers = ["<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>"];

        while generated < max_tokens && next_token != eos && total_tokens < max_seq_len {
            // Decode the token and check for role markers
            let text = tokenizer.decode(&[next_token]);

            // Track recent text to detect multi-token stop markers
            recent_text.push_str(&text);
            if recent_text.len() > 50 {
                recent_text = recent_text[recent_text.len() - 50..].to_string();
            }

            // Stop if we hit a role marker (model is trying to continue the template)
            if stop_markers.iter().any(|m| recent_text.contains(m)) {
                break;
            }

            print!("{}", text);
            stdout.flush().unwrap();

            total_tokens += 1;
            generated += 1;

            // Forward one token
            let logits = model.forward_cached(&[next_token], &mut cache);
            next_token = sampler.sample(&logits);
        }

        let decode_time = decode_start.elapsed();
        println!(); // newline after generation

        if generated > 0 {
            eprintln!(
                "[decode: {} tokens in {:.1}ms | {:.1} tok/s]",
                generated,
                decode_time.as_secs_f64() * 1000.0,
                generated as f64 / decode_time.as_secs_f64(),
            );
        }
    }

    eprintln!("\nBye!");
}
