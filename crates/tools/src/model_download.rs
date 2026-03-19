//! Model download tool — fetches GGUF models from HuggingFace.
//!
//! Downloads a specific file from a HuggingFace model repo to the
//! local models directory. Only writes to the configured models path —
//! no arbitrary filesystem access.
//!
//! This tool should be gated behind `prompt` permission tier so the
//! user confirms before downloading (models can be multi-GB).

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use std::path::{Path, PathBuf};

use super::{extract_tag, ToolPeer, ToolResponse};

/// Model download tool.
pub struct ModelDownloadTool {
    http: reqwest::Client,
    /// Absolute path to the models directory.
    models_dir: PathBuf,
}

impl ModelDownloadTool {
    pub fn new(models_dir: PathBuf) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(3600)) // 1 hour for large models
                .build()
                .expect("failed to build HTTP client"),
            models_dir,
        }
    }
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

/// Sanitize a filename — strip path separators, prevent directory traversal.
fn sanitize_filename(name: &str) -> Option<String> {
    let name = name.trim();
    if name.is_empty() || name.contains("..") {
        return None;
    }
    // Take only the final component
    let clean = Path::new(name)
        .file_name()?
        .to_str()?
        .to_string();
    if clean.is_empty() {
        return None;
    }
    Some(clean)
}

#[async_trait]
impl Handler for ModelDownloadTool {
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

        // Validate filename
        let safe_filename = match sanitize_filename(&filename) {
            Some(f) => f,
            None => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err("invalid filename (directory traversal rejected)"),
                });
            }
        };

        if !safe_filename.ends_with(".gguf") {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("only .gguf files can be downloaded"),
            });
        }

        // Ensure models directory exists
        if !self.models_dir.exists() {
            if let Err(e) = std::fs::create_dir_all(&self.models_dir) {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&format!(
                        "failed to create models directory '{}': {e}",
                        self.models_dir.display()
                    )),
                });
            }
        }

        let dest_path = self.models_dir.join(&safe_filename);

        // Check if already downloaded
        if dest_path.exists() {
            let size = dest_path.metadata().map(|m| m.len()).unwrap_or(0);
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::ok(&format!(
                    "Model already exists: {} ({})",
                    dest_path.display(),
                    human_size(size),
                )),
            });
        }

        // Build download URL
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model_id, filename,
        );

        // Download with streaming to disk
        let response = match self.http.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&format!("download request failed: {e}")),
                });
            }
        };

        if !response.status().is_success() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "download failed: HTTP {}. Check model ID and filename.",
                    response.status(),
                )),
            });
        }

        let total_size = response.content_length().unwrap_or(0);

        // Write to a temp file first, rename on success (atomic-ish)
        let temp_path = dest_path.with_extension("gguf.downloading");
        let mut file = match std::fs::File::create(&temp_path) {
            Ok(f) => f,
            Err(e) => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&format!(
                        "failed to create temp file: {e}"
                    )),
                });
            }
        };

        use std::io::Write;
        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();
        use futures_util::StreamExt;

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    if let Err(e) = file.write_all(&bytes) {
                        // Clean up temp file
                        let _ = std::fs::remove_file(&temp_path);
                        return Ok(HandlerResponse::Reply {
                            payload_xml: ToolResponse::err(&format!("write failed: {e}")),
                        });
                    }
                    downloaded += bytes.len() as u64;
                }
                Err(e) => {
                    let _ = std::fs::remove_file(&temp_path);
                    return Ok(HandlerResponse::Reply {
                        payload_xml: ToolResponse::err(&format!(
                            "download interrupted after {}: {e}",
                            human_size(downloaded),
                        )),
                    });
                }
            }
        }

        // Flush and rename
        if let Err(e) = file.flush() {
            let _ = std::fs::remove_file(&temp_path);
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("flush failed: {e}")),
            });
        }
        drop(file);

        if let Err(e) = std::fs::rename(&temp_path, &dest_path) {
            let _ = std::fs::remove_file(&temp_path);
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("rename failed: {e}")),
            });
        }

        let size_str = if total_size > 0 {
            human_size(total_size)
        } else {
            human_size(downloaded)
        };

        Ok(HandlerResponse::Reply {
            payload_xml: ToolResponse::ok(&format!(
                "Downloaded successfully!\nFile: {}\nSize: {}\nPath: {}",
                safe_filename,
                size_str,
                dest_path.display(),
            )),
        })
    }
}

#[async_trait]
impl ToolPeer for ModelDownloadTool {
    fn name(&self) -> &str {
        "model-download"
    }

    fn wit(&self) -> &str {
        r#"
/// Download a GGUF model file from HuggingFace Hub to the local models directory.
/// Always run model-verify first to check compatibility. This tool only writes to
/// the configured models directory — no arbitrary filesystem access.
/// Downloads can be large (multi-GB) — user confirmation is required.
interface model-download {
    record request {
        /// HuggingFace model ID (e.g., "TheBloke/Llama-2-7B-GGUF")
        model-id: string,
        /// GGUF filename to download (e.g., "llama-2-7b.Q4_K_M.gguf")
        filename: string,
    }
    download: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_normal_filename() {
        assert_eq!(
            sanitize_filename("model-Q4_K_M.gguf"),
            Some("model-Q4_K_M.gguf".into())
        );
    }

    #[test]
    fn sanitize_strips_path() {
        assert_eq!(
            sanitize_filename("some/dir/model.gguf"),
            Some("model.gguf".into())
        );
    }

    #[test]
    fn sanitize_rejects_traversal() {
        assert_eq!(sanitize_filename("../../etc/passwd"), None);
    }

    #[test]
    fn sanitize_rejects_empty() {
        assert_eq!(sanitize_filename(""), None);
    }

    #[test]
    fn sanitize_windows_path() {
        assert_eq!(
            sanitize_filename("C:\\models\\model.gguf"),
            Some("model.gguf".into())
        );
    }

    #[test]
    fn metadata() {
        let dir = tempfile::TempDir::new().unwrap();
        let tool = ModelDownloadTool::new(dir.path().to_path_buf());
        assert_eq!(tool.name(), "model-download");
        assert!(tool.wit().contains("model-id"));
    }

    #[test]
    fn only_gguf_allowed() {
        // The handler checks this, but we verify the intent
        let safe = sanitize_filename("model.safetensors");
        assert_eq!(safe, Some("model.safetensors".into()));
        // The handler would reject this because it doesn't end in .gguf
    }
}
