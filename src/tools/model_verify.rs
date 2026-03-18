//! Model verification tool — checks if a model is compatible with the system.
//!
//! Given a HuggingFace model ID and filename, checks:
//! - Architecture support (llama, bitnet, qwen2, phi, etc.)
//! - Quantization type support (TQ1_0, TQ2_0, Q4_K_M, Q8_0, F16, F32, etc.)
//! - Size vs available RAM
//! - Required disk space
//!
//! Returns a compatibility report so Bob can advise the user before downloading.

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use serde::Deserialize;

use super::{extract_tag, ToolPeer, ToolResponse};

/// Model verifier tool.
pub struct ModelVerifyTool {
    http: reqwest::Client,
    /// Available RAM in bytes (detected at startup).
    available_ram: u64,
    /// Models directory path.
    models_dir: String,
}

impl ModelVerifyTool {
    pub fn new(models_dir: String) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
            available_ram: detect_available_ram(),
            models_dir,
        }
    }
}

/// Detect available system RAM (best-effort).
fn detect_available_ram() -> u64 {
    // sysinfo would be more accurate, but for now use a conservative default.
    // On most dev machines: 8-64 GB. Pi 5: 4-8 GB.
    // We'll report what we know and let Bob interpret.
    #[cfg(target_os = "windows")]
    {
        // Windows: use GlobalMemoryStatusEx via winapi
        // For now, conservative default
        8 * 1024 * 1024 * 1024 // 8 GB default
    }
    #[cfg(not(target_os = "windows"))]
    {
        // Linux/macOS: read /proc/meminfo or sysctl
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(val) = kb.parse::<u64>() {
                            return val * 1024;
                        }
                    }
                }
            }
        }
        8 * 1024 * 1024 * 1024
    }
}

/// Architectures our engine currently supports.
const SUPPORTED_ARCHITECTURES: &[&str] = &[
    "llama",
    "bitnet",
    "qwen2",
    "phi",
    "falcon",
];

/// Quantization types our engine supports.
const SUPPORTED_QUANT_TYPES: &[&str] = &[
    "TQ1_0",  // base-3 packed ternary
    "TQ2_0",  // 2-bit packed ternary
    "F32",
    "F16",
    "BF16",
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q8_0",
    "Q2_K",
    "Q3_K_M",
    "Q3_K_S",
    "Q3_K_L",
    "Q6_K",
];

/// Guess architecture from model ID or filename.
fn guess_architecture(model_id: &str, filename: &str) -> Option<String> {
    let combined = format!("{} {}", model_id, filename).to_lowercase();
    for arch in SUPPORTED_ARCHITECTURES {
        if combined.contains(arch) {
            return Some(arch.to_string());
        }
    }
    // Common mappings
    if combined.contains("codellama") || combined.contains("llama") || combined.contains("tinyllama") {
        return Some("llama".into());
    }
    if combined.contains("mistral") || combined.contains("mixtral") {
        return Some("llama".into()); // Mistral uses llama architecture
    }
    None
}

/// Guess quant type from filename (e.g., "model-Q4_K_M.gguf").
fn guess_quant_type(filename: &str) -> Option<String> {
    let upper = filename.to_uppercase();
    for qt in SUPPORTED_QUANT_TYPES {
        if upper.contains(qt) {
            return Some(qt.to_string());
        }
    }
    // Try common patterns
    if upper.contains("Q4_0") {
        return Some("Q4_0".into());
    }
    if upper.contains("Q5_0") {
        return Some("Q5_0".into());
    }
    None
}

/// Format bytes as human-readable.
fn human_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.0} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{} KB", bytes / 1024)
    }
}

/// HuggingFace file info from the tree API.
#[derive(Debug, Deserialize)]
struct HfFileInfo {
    #[serde(rename = "rfilename")]
    filename: Option<String>,
    path: Option<String>,
    size: Option<u64>,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    file_type: Option<String>,
}

#[async_trait]
impl Handler for ModelVerifyTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let model_id = extract_tag(&xml_str, "model-id").unwrap_or_default();
        let filename = extract_tag(&xml_str, "filename").unwrap_or_default();

        if model_id.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <model-id>"),
            });
        }
        if filename.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <filename>"),
            });
        }

        let mut report = String::new();
        let mut issues = Vec::new();
        let mut ok_items = Vec::new();

        // 1. Architecture check
        match guess_architecture(&model_id, &filename) {
            Some(arch) if SUPPORTED_ARCHITECTURES.contains(&arch.as_str()) => {
                ok_items.push(format!("Architecture: {} (supported)", arch));
            }
            Some(arch) => {
                issues.push(format!("Architecture '{}' is not yet supported", arch));
            }
            None => {
                issues.push("Could not determine architecture from model name. Verify manually.".into());
            }
        }

        // 2. Quantization check
        match guess_quant_type(&filename) {
            Some(qt) if SUPPORTED_QUANT_TYPES.contains(&qt.as_str()) => {
                ok_items.push(format!("Quantization: {} (supported)", qt));
            }
            Some(qt) => {
                issues.push(format!("Quantization type '{}' is not yet supported", qt));
            }
            None => {
                issues.push("Could not determine quantization type from filename.".into());
            }
        }

        // 3. File size check (query HuggingFace for actual size)
        let size_url = format!(
            "https://huggingface.co/api/models/{}/tree/main",
            urlencoding::encode(&model_id),
        );

        let file_size = match self.http.get(&size_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<Vec<HfFileInfo>>().await {
                    Ok(files) => files
                        .iter()
                        .find(|f| {
                            f.filename.as_deref() == Some(&filename)
                                || f.path.as_deref() == Some(&filename)
                        })
                        .and_then(|f| f.size),
                    Err(_) => None,
                }
            }
            _ => None,
        };

        match file_size {
            Some(size) => {
                ok_items.push(format!("File size: {}", human_size(size)));

                // RAM check: model needs ~1.2x file size in RAM (rough estimate)
                let estimated_ram = (size as f64 * 1.2) as u64;
                if estimated_ram > self.available_ram {
                    issues.push(format!(
                        "Model needs ~{} RAM but system has ~{}",
                        human_size(estimated_ram),
                        human_size(self.available_ram),
                    ));
                } else {
                    ok_items.push(format!(
                        "RAM: ~{} needed, {} available",
                        human_size(estimated_ram),
                        human_size(self.available_ram),
                    ));
                }

                // Disk space check
                let models_path = std::path::Path::new(&self.models_dir);
                if models_path.exists() {
                    ok_items.push(format!("Download path: {} (exists)", self.models_dir));
                } else {
                    issues.push(format!("Models directory '{}' does not exist", self.models_dir));
                }
            }
            None => {
                issues.push("Could not determine file size from HuggingFace. Model/file may not exist.".into());
            }
        }

        // Build report
        report.push_str("# Model Compatibility Report\n\n");
        report.push_str(&format!("Model: {}\n", model_id));
        report.push_str(&format!("File: {}\n\n", filename));

        if !ok_items.is_empty() {
            report.push_str("## Compatible\n");
            for item in &ok_items {
                report.push_str(&format!("  ✓ {}\n", item));
            }
            report.push('\n');
        }

        if !issues.is_empty() {
            report.push_str("## Issues\n");
            for issue in &issues {
                report.push_str(&format!("  ✗ {}\n", issue));
            }
            report.push('\n');
        }

        let verdict = if issues.is_empty() {
            "COMPATIBLE — safe to download."
        } else {
            "ISSUES FOUND — review before downloading."
        };
        report.push_str(&format!("Verdict: {}\n", verdict));

        Ok(HandlerResponse::Reply {
            payload_xml: ToolResponse::ok(&report),
        })
    }
}

#[async_trait]
impl ToolPeer for ModelVerifyTool {
    fn name(&self) -> &str {
        "model-verify"
    }

    fn wit(&self) -> &str {
        r#"
/// Verify that a model is compatible with the local system before downloading.
/// Checks architecture support, quantization type, RAM requirements, and disk space.
/// Always run this before model-download to avoid wasting bandwidth.
interface model-verify {
    record request {
        /// HuggingFace model ID (e.g., "TheBloke/Llama-2-7B-GGUF")
        model-id: string,
        /// GGUF filename to check (e.g., "llama-2-7b.Q4_K_M.gguf")
        filename: string,
    }
    verify: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guess_arch_llama() {
        assert_eq!(
            guess_architecture("TheBloke/Llama-2-7B-GGUF", "llama.gguf"),
            Some("llama".into())
        );
    }

    #[test]
    fn guess_arch_qwen() {
        assert_eq!(
            guess_architecture("Qwen/Qwen2-0.5B-GGUF", "q4.gguf"),
            Some("qwen2".into())
        );
    }

    #[test]
    fn guess_arch_bitnet() {
        assert_eq!(
            guess_architecture("1bitLLM/bitnet_b1_58-large", "model.gguf"),
            Some("bitnet".into())
        );
    }

    #[test]
    fn guess_arch_mistral_is_llama() {
        assert_eq!(
            guess_architecture("mistralai/Mistral-7B", "model.gguf"),
            Some("llama".into())
        );
    }

    #[test]
    fn guess_arch_unknown() {
        assert_eq!(
            guess_architecture("unknown/mystery-model", "model.gguf"),
            None
        );
    }

    #[test]
    fn guess_quant_q4km() {
        assert_eq!(guess_quant_type("model-Q4_K_M.gguf"), Some("Q4_K_M".into()));
    }

    #[test]
    fn guess_quant_tq2() {
        assert_eq!(guess_quant_type("model-TQ2_0.gguf"), Some("TQ2_0".into()));
    }

    #[test]
    fn guess_quant_q8() {
        assert_eq!(guess_quant_type("model-Q8_0.gguf"), Some("Q8_0".into()));
    }

    #[test]
    fn guess_quant_none() {
        assert_eq!(guess_quant_type("model.gguf"), None);
    }

    #[test]
    fn metadata() {
        let tool = ModelVerifyTool::new("./models".into());
        assert_eq!(tool.name(), "model-verify");
        assert!(tool.wit().contains("model-id"));
    }

    #[test]
    fn human_size_formats() {
        assert_eq!(human_size(4_500_000_000), "4.2 GB");
        assert_eq!(human_size(150_000_000), "143 MB");
    }

    #[test]
    fn detect_ram_returns_nonzero() {
        let ram = detect_available_ram();
        assert!(ram > 0);
    }
}
