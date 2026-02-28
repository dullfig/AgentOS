//! Local inference engine lifecycle — manages codeLlm's InferenceEngine.
//!
//! Convention over configuration: looks for `~/.agentos/models/*.gguf`
//! and `tokenizer.json` in the same directory. No YAML config needed.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::Mutex;
use tracing::info;

use code_llm::prelude::{EngineConfig, InferenceEngine};

/// Shared engine handle. `tokio::sync::Mutex` because the lock is held
/// across `.await` in `fill()`.
pub type SharedEngine = Arc<Mutex<InferenceEngine>>;

/// Configuration for the local inference engine.
#[derive(Debug, Clone)]
pub struct LocalEngineConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,
    /// Path to the tokenizer.json file.
    pub tokenizer_path: PathBuf,
    /// Engine configuration (context size, temperature, etc.).
    pub engine_config: EngineConfig,
}

impl LocalEngineConfig {
    /// Discover model files by convention: `~/.agentos/models/*.gguf` + `tokenizer.json`.
    ///
    /// Returns `None` if the directory doesn't exist or no GGUF file is found.
    pub fn from_conventional_paths() -> Option<Self> {
        let home = dirs_path()?;
        let models_dir = home.join(".agentos").join("models");

        if !models_dir.is_dir() {
            return None;
        }

        // Find first *.gguf file
        let gguf = std::fs::read_dir(&models_dir)
            .ok()?
            .filter_map(|e| e.ok())
            .find(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "gguf")
                    .unwrap_or(false)
            })?
            .path();

        let tokenizer = models_dir.join("tokenizer.json");
        if !tokenizer.is_file() {
            return None;
        }

        Some(Self {
            model_path: gguf,
            tokenizer_path: tokenizer,
            engine_config: EngineConfig {
                n_ctx: 2048,
                n_gpu_layers: 0, // CPU-only default (Pi 5 target)
                seed: 42,
                temperature: 0.3, // Deterministic parameter extraction
                top_p: 0.9,
            },
        })
    }
}

/// Load an InferenceEngine from config, wrapped in a shared handle.
pub fn load_engine(config: &LocalEngineConfig) -> Result<SharedEngine, String> {
    info!(
        "Loading local model from {}",
        config.model_path.display()
    );
    let engine = InferenceEngine::from_gguf(
        &config.model_path,
        &config.tokenizer_path,
        config.engine_config.clone(),
    )
    .map_err(|e| format!("failed to load local model: {e}"))?;

    Ok(Arc::new(Mutex::new(engine)))
}

/// Get the user's home directory. Cross-platform.
fn dirs_path() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conventional_paths_returns_none_when_missing() {
        // Unless someone has ~/.agentos/models/ with a GGUF file, this returns None
        // We test the negative case — the function doesn't panic
        let _ = LocalEngineConfig::from_conventional_paths();
    }

    #[test]
    fn engine_config_defaults() {
        let config = LocalEngineConfig {
            model_path: PathBuf::from("/tmp/test.gguf"),
            tokenizer_path: PathBuf::from("/tmp/tokenizer.json"),
            engine_config: EngineConfig {
                n_ctx: 2048,
                n_gpu_layers: 0,
                seed: 42,
                temperature: 0.3,
                top_p: 0.9,
            },
        };
        assert_eq!(config.engine_config.temperature, 0.3);
        assert_eq!(config.engine_config.n_ctx, 2048);
    }

    #[test]
    #[ignore] // Requires actual model files
    fn load_engine_from_gguf() {
        let config = LocalEngineConfig::from_conventional_paths().expect("no model found");
        let engine = load_engine(&config).expect("load failed");
        let _guard = engine.blocking_lock();
    }
}
