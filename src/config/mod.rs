//! Model configuration — persistent config for providers, models, and API keys.
//!
//! User-level config: `~/.agentos/models.yaml` (providers, keys, models)
//! Project-level config: `.agentos/config.yaml` (default model alias only, safe to commit)
//!
//! Resolution: project config → user config → env var fallback → error.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A configured LLM provider (Anthropic, OpenAI, Ollama, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default)]
    pub models: HashMap<String, String>, // alias → model ID
}

/// Top-level models configuration (user-level file).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelsConfig {
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
}

/// Project-level config (no secrets — safe to commit).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ProjectConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    default_model: Option<String>,
}

/// A fully resolved model reference.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub provider: String,
    pub model_id: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

/// A single entry in the flat model list (for `/models` display).
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub alias: String,
    pub provider: String,
    pub model_id: String,
    pub is_default: bool,
}

/// Path to user-level config directory.
fn user_config_dir() -> Option<PathBuf> {
    dirs_path().map(|p| p.join("models.yaml"))
}

/// Path to `~/.agentos/`.
fn dirs_path() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE")
            .ok()
            .map(|p| PathBuf::from(p).join(".agentos"))
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME")
            .ok()
            .map(|p| PathBuf::from(p).join(".agentos"))
    }
}

/// Default Anthropic models (same as the hardcoded aliases in resolve_model).
fn default_anthropic_models() -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("opus".into(), "claude-opus-4-6".into());
    m.insert("sonnet".into(), "claude-sonnet-4-6".into());
    m.insert("sonnet-4.5".into(), "claude-sonnet-4-5-20250929".into());
    m.insert("haiku".into(), "claude-haiku-4-5-20251001".into());
    m
}

impl ModelsConfig {
    /// Load config from disk, merging user + project files.
    /// Falls back to env var if no config file exists.
    pub fn load() -> Self {
        let mut config = Self::load_user_config();

        // Merge project-level default override
        let project = Self::load_project_config();
        if let Some(default) = project.default_model {
            config.default = Some(default);
        }

        // Env fallback: if no providers configured, check ANTHROPIC_API_KEY
        if config.providers.is_empty() {
            if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
                config.providers.insert(
                    "anthropic".into(),
                    ProviderConfig {
                        api_key: Some(key),
                        base_url: None,
                        models: default_anthropic_models(),
                    },
                );
                if config.default.is_none() {
                    config.default = Some("sonnet".into());
                }
            }
        }

        config
    }

    /// Load just the user-level config file.
    fn load_user_config() -> Self {
        let Some(path) = user_config_dir() else {
            return Self::default();
        };
        match std::fs::read_to_string(&path) {
            Ok(content) => serde_yaml::from_str(&content).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Load just the project-level config file.
    fn load_project_config() -> ProjectConfig {
        match std::fs::read_to_string(".agentos/config.yaml") {
            Ok(content) => serde_yaml::from_str(&content).unwrap_or_default(),
            Err(_) => ProjectConfig::default(),
        }
    }

    /// Save user-level config to `~/.agentos/models.yaml`.
    pub fn save(&self) -> Result<(), String> {
        let Some(dir) = dirs_path() else {
            return Err("Cannot determine home directory".into());
        };
        std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create {}: {e}", dir.display()))?;
        let path = dir.join("models.yaml");
        let yaml = serde_yaml::to_string(self).map_err(|e| format!("YAML serialize error: {e}"))?;
        std::fs::write(&path, yaml).map_err(|e| format!("Failed to write {}: {e}", path.display()))?;
        Ok(())
    }

    /// Save project-level default to `.agentos/config.yaml`.
    pub fn save_project_default(&self) -> Result<(), String> {
        let project = ProjectConfig {
            default_model: self.default.clone(),
        };
        std::fs::create_dir_all(".agentos")
            .map_err(|e| format!("Failed to create .agentos/: {e}"))?;
        let yaml = serde_yaml::to_string(&project)
            .map_err(|e| format!("YAML serialize error: {e}"))?;
        std::fs::write(".agentos/config.yaml", yaml)
            .map_err(|e| format!("Failed to write .agentos/config.yaml: {e}"))?;
        Ok(())
    }

    /// Resolve an alias to a full model reference (provider + model_id + key + url).
    pub fn resolve(&self, alias: &str) -> Option<ResolvedModel> {
        for (provider_name, provider) in &self.providers {
            if let Some(model_id) = provider.models.get(alias) {
                return Some(ResolvedModel {
                    provider: provider_name.clone(),
                    model_id: model_id.clone(),
                    api_key: provider.api_key.clone(),
                    base_url: provider.base_url.clone(),
                });
            }
        }
        None
    }

    /// Resolve alias with fallback to the hardcoded resolve_model().
    /// Returns (model_id, api_key, base_url).
    pub fn resolve_or_fallback(&self, alias: &str) -> Option<ResolvedModel> {
        if let Some(resolved) = self.resolve(alias) {
            return Some(resolved);
        }
        // Fallback: treat alias as full model ID with first available provider's key
        let fallback_id = crate::llm::types::resolve_model(alias);
        // Find any anthropic provider for the key
        if let Some(provider) = self.providers.get("anthropic") {
            return Some(ResolvedModel {
                provider: "anthropic".into(),
                model_id: fallback_id.to_string(),
                api_key: provider.api_key.clone(),
                base_url: provider.base_url.clone(),
            });
        }
        None
    }

    /// Get a flat list of all configured models.
    pub fn all_models(&self) -> Vec<ModelEntry> {
        let mut entries = Vec::new();
        for (provider_name, provider) in &self.providers {
            for (alias, model_id) in &provider.models {
                entries.push(ModelEntry {
                    alias: alias.clone(),
                    provider: provider_name.clone(),
                    model_id: model_id.clone(),
                    is_default: self.default.as_deref() == Some(alias),
                });
            }
        }
        // Sort: default first, then alphabetical
        entries.sort_by(|a, b| {
            b.is_default.cmp(&a.is_default).then(a.alias.cmp(&b.alias))
        });
        entries
    }

    /// Add a model to a provider. Creates the provider if it doesn't exist.
    pub fn add_model(
        &mut self,
        provider: &str,
        alias: &str,
        model_id: &str,
        api_key: Option<String>,
        base_url: Option<String>,
    ) {
        let entry = self
            .providers
            .entry(provider.to_string())
            .or_insert_with(|| ProviderConfig {
                api_key: None,
                base_url: None,
                models: HashMap::new(),
            });
        entry.models.insert(alias.to_string(), model_id.to_string());
        if api_key.is_some() {
            entry.api_key = api_key;
        }
        if base_url.is_some() {
            entry.base_url = base_url;
        }
        // If this is the first model ever, set it as default
        if self.default.is_none() {
            self.default = Some(alias.to_string());
        }
    }

    /// Remove a model by alias. Returns true if found and removed.
    pub fn remove_model(&mut self, alias: &str) -> bool {
        for provider in self.providers.values_mut() {
            if provider.models.remove(alias).is_some() {
                // Clear default if it was this alias
                if self.default.as_deref() == Some(alias) {
                    self.default = None;
                }
                return true;
            }
        }
        false
    }

    /// Set the default model alias. Returns true if the alias exists.
    pub fn set_default(&mut self, alias: &str) -> bool {
        let exists = self
            .providers
            .values()
            .any(|p| p.models.contains_key(alias));
        if exists {
            self.default = Some(alias.to_string());
        }
        exists
    }

    /// Whether any models are configured.
    pub fn has_models(&self) -> bool {
        self.providers.values().any(|p| !p.models.is_empty())
    }

    /// Get all known alias names (for completions).
    pub fn all_aliases(&self) -> Vec<String> {
        self.providers
            .values()
            .flat_map(|p| p.models.keys().cloned())
            .collect()
    }
}

/// Format the model list for display in the chat log.
pub fn format_model_list(config: &ModelsConfig) -> String {
    let models = config.all_models();
    if models.is_empty() {
        return "No models configured. Use /provider <name> to add one.".into();
    }
    let mut lines = vec!["Models:".to_string()];
    for m in &models {
        let marker = if m.is_default { "*" } else { " " };
        let default_tag = if m.is_default { "  (default)" } else { "" };
        lines.push(format!(
            "  {marker} {:<10} {:<12} {}{default_tag}",
            m.alias, m.provider, m.model_id,
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> ModelsConfig {
        let mut config = ModelsConfig::default();
        config.add_model("anthropic", "opus", "claude-opus-4-6", Some("sk-ant-123".into()), None);
        config.add_model("anthropic", "sonnet", "claude-sonnet-4-6", None, None);
        config.add_model("anthropic", "haiku", "claude-haiku-4-5-20251001", None, None);
        config.add_model("openai", "gpt4", "gpt-4o", Some("sk-openai".into()), Some("https://api.openai.com/v1".into()));
        config.set_default("sonnet");
        config
    }

    #[test]
    fn load_from_yaml_string() {
        let yaml = r#"
providers:
  anthropic:
    api_key: sk-ant-test
    models:
      opus: claude-opus-4-6
      sonnet: claude-sonnet-4-6
default: sonnet
"#;
        let config: ModelsConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.default, Some("sonnet".into()));
        assert!(config.providers.contains_key("anthropic"));
        let anthropic = &config.providers["anthropic"];
        assert_eq!(anthropic.api_key, Some("sk-ant-test".into()));
        assert_eq!(anthropic.models.len(), 2);
    }

    #[test]
    fn round_trip_yaml() {
        let config = sample_config();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let back: ModelsConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(back.default, Some("sonnet".into()));
        assert!(back.providers.contains_key("anthropic"));
        assert!(back.providers.contains_key("openai"));
    }

    #[test]
    fn resolve_alias_finds_correct_provider() {
        let config = sample_config();
        let resolved = config.resolve("gpt4").unwrap();
        assert_eq!(resolved.provider, "openai");
        assert_eq!(resolved.model_id, "gpt-4o");
        assert_eq!(resolved.api_key, Some("sk-openai".into()));
        assert_eq!(resolved.base_url, Some("https://api.openai.com/v1".into()));
    }

    #[test]
    fn resolve_unknown_alias_returns_none() {
        let config = sample_config();
        assert!(config.resolve("nonexistent").is_none());
    }

    #[test]
    fn add_model_creates_provider() {
        let mut config = ModelsConfig::default();
        config.add_model("ollama", "local", "llama3", None, Some("http://localhost:11434".into()));
        assert!(config.providers.contains_key("ollama"));
        assert_eq!(config.providers["ollama"].models["local"], "llama3");
        // First model sets default
        assert_eq!(config.default, Some("local".into()));
    }

    #[test]
    fn remove_model() {
        let mut config = sample_config();
        assert!(config.remove_model("gpt4"));
        assert!(config.resolve("gpt4").is_none());
        // Removing non-default shouldn't clear default
        assert_eq!(config.default, Some("sonnet".into()));
    }

    #[test]
    fn remove_default_clears_default() {
        let mut config = sample_config();
        assert!(config.remove_model("sonnet"));
        assert!(config.default.is_none());
    }

    #[test]
    fn set_default() {
        let mut config = sample_config();
        assert!(config.set_default("opus"));
        assert_eq!(config.default, Some("opus".into()));
    }

    #[test]
    fn set_default_unknown_returns_false() {
        let mut config = sample_config();
        assert!(!config.set_default("nonexistent"));
        assert_eq!(config.default, Some("sonnet".into())); // unchanged
    }

    #[test]
    fn env_fallback_when_no_config() {
        // This test checks the in-memory env fallback path
        let mut config = ModelsConfig::default();
        assert!(!config.has_models());

        // Simulate what load() does when env var is set
        config.providers.insert(
            "anthropic".into(),
            ProviderConfig {
                api_key: Some("sk-from-env".into()),
                base_url: None,
                models: default_anthropic_models(),
            },
        );
        config.default = Some("sonnet".into());

        assert!(config.has_models());
        let resolved = config.resolve("opus").unwrap();
        assert_eq!(resolved.model_id, "claude-opus-4-6");
        assert_eq!(resolved.api_key, Some("sk-from-env".into()));
    }

    #[test]
    fn all_models_sorted() {
        let config = sample_config();
        let models = config.all_models();
        assert!(!models.is_empty());
        // Default (sonnet) should be first
        assert_eq!(models[0].alias, "sonnet");
        assert!(models[0].is_default);
    }

    #[test]
    fn format_model_list_output() {
        let config = sample_config();
        let output = format_model_list(&config);
        assert!(output.contains("Models:"));
        assert!(output.contains("sonnet"));
        assert!(output.contains("(default)"));
        assert!(output.contains("anthropic"));
        assert!(output.contains("openai"));
    }

    #[test]
    fn all_aliases() {
        let config = sample_config();
        let aliases = config.all_aliases();
        assert!(aliases.contains(&"opus".to_string()));
        assert!(aliases.contains(&"sonnet".to_string()));
        assert!(aliases.contains(&"gpt4".to_string()));
    }

    #[test]
    fn has_models_empty() {
        let config = ModelsConfig::default();
        assert!(!config.has_models());
    }
}

// ── Agent favorites config ──

/// Persistent list of favorite agents (project-level).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentsConfig {
    #[serde(default)]
    pub favorites: Vec<String>,
}

impl AgentsConfig {
    /// Load from `.agentos/agents.yaml`. Returns default if missing.
    pub fn load() -> Self {
        match std::fs::read_to_string(".agentos/agents.yaml") {
            Ok(content) => serde_yaml::from_str(&content).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Save to `.agentos/agents.yaml`.
    pub fn save(&self) -> Result<(), String> {
        std::fs::create_dir_all(".agentos")
            .map_err(|e| format!("Failed to create .agentos/: {e}"))?;
        let yaml = serde_yaml::to_string(self)
            .map_err(|e| format!("YAML serialize error: {e}"))?;
        std::fs::write(".agentos/agents.yaml", yaml)
            .map_err(|e| format!("Failed to write .agentos/agents.yaml: {e}"))?;
        Ok(())
    }

    /// Add an agent name to favorites (deduplicates).
    pub fn add(&mut self, name: &str) {
        if !self.favorites.iter().any(|f| f == name) {
            self.favorites.push(name.to_string());
        }
    }

    /// Remove an agent name from favorites. Returns true if it was present.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.favorites.len();
        self.favorites.retain(|f| f != name);
        self.favorites.len() < before
    }
}

#[cfg(test)]
mod agents_tests {
    use super::*;

    #[test]
    fn agents_config_add_dedup() {
        let mut config = AgentsConfig::default();
        config.add("coding-agent");
        config.add("coding-agent");
        assert_eq!(config.favorites.len(), 1);
    }

    #[test]
    fn agents_config_remove() {
        let mut config = AgentsConfig::default();
        config.add("coding-agent");
        config.add("email-sender");
        assert!(config.remove("coding-agent"));
        assert_eq!(config.favorites, vec!["email-sender"]);
        assert!(!config.remove("nonexistent"));
    }

    #[test]
    fn agents_config_round_trip() {
        let mut config = AgentsConfig::default();
        config.add("coding-agent");
        config.add("email-sender");
        let yaml = serde_yaml::to_string(&config).unwrap();
        let back: AgentsConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(back.favorites, vec!["coding-agent", "email-sender"]);
    }

    #[test]
    fn agents_config_default_empty() {
        let config = AgentsConfig::default();
        assert!(config.favorites.is_empty());
    }
}
