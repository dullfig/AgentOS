//! Dump GGUF metadata and tensor info from a model file.

use agentos_bitnet::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: gguf-info <model.gguf>");
        std::process::exit(1);
    }

    let gguf = GgufFile::open(&args[1]).expect("failed to open GGUF file");

    println!("=== Metadata ===");
    let mut keys: Vec<&String> = gguf.metadata().keys().collect();
    keys.sort();
    for key in keys {
        let val = &gguf.metadata()[key];
        let display = match val {
            agentos_bitnet::MetadataValue::String(s) => {
                if s.len() > 80 { format!("\"{}...\"", &s[..80]) } else { format!("\"{s}\"") }
            }
            agentos_bitnet::MetadataValue::U32(v) => format!("{v}"),
            agentos_bitnet::MetadataValue::I32(v) => format!("{v}"),
            agentos_bitnet::MetadataValue::F32(v) => format!("{v}"),
            agentos_bitnet::MetadataValue::U64(v) => format!("{v}"),
            agentos_bitnet::MetadataValue::Bool(v) => format!("{v}"),
            agentos_bitnet::MetadataValue::Array(arr) => format!("[array, len={}]", arr.len()),
            other => format!("{other:?}"),
        };
        println!("  {key} = {display}");
    }

    println!("\n=== Tensors ({}) ===", gguf.tensors().len());
    let mut names: Vec<&String> = gguf.tensors().keys().collect();
    names.sort();
    for name in names {
        let info = &gguf.tensors()[name];
        println!("  {name}: {:?} {:?} ({} elements)", info.ggml_type, info.shape, info.n_elements);
    }
}
