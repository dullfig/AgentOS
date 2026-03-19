//! Model search tool — queries HuggingFace for GGUF models.
//!
//! Searches the HuggingFace Hub API for models matching a query,
//! filtered to GGUF format. Returns structured results with metadata
//! (name, size, architecture, quant type, downloads) for Bob to
//! present as a comparison table.

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use serde::Deserialize;

use super::{extract_tag, ToolPeer, ToolResponse};

/// HuggingFace model search tool.
pub struct ModelSearchTool {
    http: reqwest::Client,
}

impl ModelSearchTool {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
        }
    }
}

/// HuggingFace API model response (subset of fields).
#[derive(Debug, Deserialize)]
struct HfModel {
    #[serde(rename = "modelId")]
    model_id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

/// A file within a HuggingFace model repo.
#[derive(Debug, Deserialize)]
struct HfSibling {
    #[serde(rename = "rfilename")]
    filename: String,
    #[serde(default)]
    size: Option<u64>,
}

impl HfModel {
    /// Extract GGUF files from siblings.
    fn gguf_files(&self) -> Vec<&HfSibling> {
        self.siblings
            .iter()
            .filter(|s| s.filename.ends_with(".gguf"))
            .collect()
    }
}

/// Format file size in human-readable form.
fn human_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.0} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{} KB", bytes / 1024)
    }
}

#[async_trait]
impl Handler for ModelSearchTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let query = extract_tag(&xml_str, "query").unwrap_or_default();
        if query.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <query>"),
            });
        }

        let limit = extract_tag(&xml_str, "limit")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10);

        // Search HuggingFace API — filter for GGUF tagged models
        let search_query = if query.to_lowercase().contains("gguf") {
            query.clone()
        } else {
            format!("{query} gguf")
        };

        let url = format!(
            "https://huggingface.co/api/models?search={}&sort=downloads&limit={}&full=true",
            urlencoding::encode(&search_query),
            limit.min(20),
        );

        let response = match self.http.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&format!("HTTP request failed: {e}")),
                });
            }
        };

        if !response.status().is_success() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "HuggingFace API returned {}",
                    response.status()
                )),
            });
        }

        let models: Vec<HfModel> = match response.json().await {
            Ok(m) => m,
            Err(e) => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&format!("failed to parse response: {e}")),
                });
            }
        };

        // Format results
        let mut output = String::new();
        let mut result_count = 0;

        for model in &models {
            let gguf_files = model.gguf_files();
            if gguf_files.is_empty() {
                continue;
            }

            output.push_str(&format!(
                "\n## {}\nDownloads: {}\nTags: {}\nGGUF files:\n",
                model.model_id,
                model.downloads,
                model.tags.join(", "),
            ));

            for file in &gguf_files {
                let size_str = file
                    .size
                    .map(|s| human_size(s))
                    .unwrap_or_else(|| "unknown".into());
                output.push_str(&format!("  - {} ({})\n", file.filename, size_str));
            }
            result_count += 1;
        }

        if result_count == 0 {
            output = "No GGUF models found matching your query. Try broader search terms.".into();
        } else {
            output = format!("Found {result_count} model(s) with GGUF files:\n{output}");
        }

        Ok(HandlerResponse::Reply {
            payload_xml: ToolResponse::ok(&output),
        })
    }
}

#[async_trait]
impl ToolPeer for ModelSearchTool {
    fn name(&self) -> &str {
        "model-search"
    }

    fn wit(&self) -> &str {
        r#"
/// Search HuggingFace Hub for GGUF models. Returns model names, file sizes,
/// download counts, and available quantization variants. Use this to help
/// the user find and compare models before downloading.
interface model-search {
    record request {
        /// Search query (e.g., "qwen coder 1.5b", "bitnet", "phi small")
        query: string,
        /// Maximum results to return (default: 10, max: 20)
        limit: option<u32>,
    }
    search: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_size_gb() {
        assert_eq!(human_size(4_500_000_000), "4.2 GB");
    }

    #[test]
    fn human_size_mb() {
        assert_eq!(human_size(150_000_000), "143 MB");
    }

    #[test]
    fn human_size_kb() {
        assert_eq!(human_size(50_000), "48 KB");
    }

    #[test]
    fn gguf_filter() {
        let model = HfModel {
            model_id: "test/model".into(),
            downloads: 100,
            tags: vec![],
            siblings: vec![
                HfSibling { filename: "model.safetensors".into(), size: Some(1000) },
                HfSibling { filename: "model-q4.gguf".into(), size: Some(2000) },
                HfSibling { filename: "readme.md".into(), size: Some(500) },
            ],
        };
        let gguf = model.gguf_files();
        assert_eq!(gguf.len(), 1);
        assert_eq!(gguf[0].filename, "model-q4.gguf");
    }

    #[test]
    fn metadata() {
        let tool = ModelSearchTool::new();
        assert_eq!(tool.name(), "model-search");
        assert!(tool.wit().contains("query"));
    }
}
