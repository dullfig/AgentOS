//! Organism — imperative configuration API for AgentOS.
//!
//! The organism is the single source of truth for listeners, profiles,
//! and security configuration. The YAML parser is a client of this API.

pub mod parser;
pub mod profile;

use std::collections::HashMap;

use crate::wasm::capabilities::WasmCapabilities;
use profile::{DispatchTable, SecurityProfile};

/// WASM tool configuration on a listener.
#[derive(Debug, Clone)]
pub struct WasmToolConfig {
    pub path: String,
    pub capabilities: WasmCapabilities,
}

/// Port declaration on a listener (from organism config).
#[derive(Debug, Clone)]
pub struct PortDef {
    pub port: u16,
    pub direction: String,
    pub protocol: String,
    pub hosts: Vec<String>,
}

/// Definition of a listener (from organism config).
#[derive(Debug, Clone)]
pub struct ListenerDef {
    pub name: String,
    pub payload_tag: String,
    pub handler: String,
    pub description: String,
    pub is_agent: bool,
    pub peers: Vec<String>,
    pub model: Option<String>,
    pub ports: Vec<PortDef>,
    /// Whether this LLM listener auto-curates via the librarian before API calls.
    pub librarian: bool,
    /// WASM tool configuration (present when handler == "wasm").
    pub wasm: Option<WasmToolConfig>,
    /// Rich semantic description for embedding-based routing.
    /// Consumed by the embedding model, never enters the thinker's context.
    pub semantic_description: Option<String>,
}

/// Result of a hot-reload diff.
#[derive(Debug)]
pub struct ReloadEvent {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub updated: Vec<String>,
}

/// The organism: single source of truth for configuration.
#[derive(Debug)]
pub struct Organism {
    pub name: String,
    listeners: HashMap<String, ListenerDef>,
    profiles: HashMap<String, SecurityProfile>,
}

impl Organism {
    /// Create a new organism with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            listeners: HashMap::new(),
            profiles: HashMap::new(),
        }
    }

    // ── Listener management ──

    /// Register a listener.
    pub fn register_listener(&mut self, def: ListenerDef) -> Result<(), String> {
        if self.listeners.contains_key(&def.name) {
            return Err(format!("listener '{}' already registered", def.name));
        }
        self.listeners.insert(def.name.clone(), def);
        Ok(())
    }

    /// Unregister a listener by name.
    pub fn unregister_listener(&mut self, name: &str) -> Result<(), String> {
        if self.listeners.remove(name).is_none() {
            return Err(format!("listener '{name}' not found"));
        }
        Ok(())
    }

    /// Get a listener by name.
    pub fn get_listener(&self, name: &str) -> Option<&ListenerDef> {
        self.listeners.get(name)
    }

    /// Get all listener names.
    pub fn listener_names(&self) -> Vec<&str> {
        self.listeners.keys().map(|s| s.as_str()).collect()
    }

    /// Get all listeners.
    pub fn listeners(&self) -> &HashMap<String, ListenerDef> {
        &self.listeners
    }

    // ── Profile management ──

    /// Add a security profile.
    pub fn add_profile(&mut self, profile: SecurityProfile) -> Result<(), String> {
        // Validate: all allowed listeners must exist (unless "all")
        if !profile.allow_all {
            for name in &profile.allowed_listeners {
                if !self.listeners.contains_key(name) {
                    return Err(format!(
                        "profile '{}' references unknown listener '{name}'",
                        profile.name
                    ));
                }
            }
        }
        self.profiles.insert(profile.name.clone(), profile);
        Ok(())
    }

    /// Get a security profile by name.
    pub fn get_profile(&self, name: &str) -> Option<&SecurityProfile> {
        self.profiles.get(name)
    }

    /// Get all profile names.
    pub fn profile_names(&self) -> Vec<&str> {
        self.profiles.keys().map(|s| s.as_str()).collect()
    }

    // ── Hot reload ──

    /// Apply a new configuration, returning what changed.
    pub fn apply_config(&mut self, new: Organism) -> ReloadEvent {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut updated = Vec::new();

        // Find removed listeners
        let old_names: Vec<String> = self.listeners.keys().cloned().collect();
        for name in &old_names {
            if !new.listeners.contains_key(name) {
                self.listeners.remove(name);
                removed.push(name.clone());
            }
        }

        // Find added/updated listeners
        for (name, def) in &new.listeners {
            if self.listeners.contains_key(name) {
                updated.push(name.clone());
            } else {
                added.push(name.clone());
            }
            self.listeners.insert(name.clone(), def.clone());
        }

        // Replace profiles wholesale
        self.profiles = new.profiles;
        self.name = new.name;

        ReloadEvent {
            added,
            removed,
            updated,
        }
    }

    /// Build a dispatch table for a named profile.
    /// The dispatch table contains only the listeners allowed by that profile.
    pub fn dispatch_table(&self, profile_name: &str) -> Result<DispatchTable, String> {
        let profile = self
            .profiles
            .get(profile_name)
            .ok_or_else(|| format!("profile '{profile_name}' not found"))?;

        let mut allowed_listeners = HashMap::new();

        if profile.allow_all {
            allowed_listeners = self.listeners.clone();
        } else {
            for name in &profile.allowed_listeners {
                if let Some(def) = self.listeners.get(name) {
                    allowed_listeners.insert(name.clone(), def.clone());
                }
            }
        }

        Ok(DispatchTable {
            profile_name: profile_name.to_string(),
            listeners: allowed_listeners,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::profile::RetentionPolicy;
    use super::*;
    use std::collections::HashSet;

    fn sample_listener(name: &str) -> ListenerDef {
        ListenerDef {
            name: name.to_string(),
            payload_tag: format!("{name}Request"),
            handler: format!("handlers.{name}.handle"),
            description: format!("{name} handler"),
            is_agent: false,
            peers: vec![],
            model: None,
            ports: vec![],
            librarian: false,
            wasm: None,
            semantic_description: None,
        }
    }

    fn sample_profile(name: &str, listeners: Vec<&str>) -> SecurityProfile {
        SecurityProfile {
            name: name.to_string(),
            linux_user: format!("agentos-{name}"),
            allowed_listeners: listeners.into_iter().map(|s| s.to_string()).collect(),
            allow_all: false,
            journal_retention: RetentionPolicy::Forever,
            network: vec![],
        }
    }

    #[test]
    fn register_and_get_listener() {
        let mut org = Organism::new("test");
        org.register_listener(sample_listener("echo")).unwrap();

        assert!(org.get_listener("echo").is_some());
        assert!(org.get_listener("nonexistent").is_none());
    }

    #[test]
    fn duplicate_listener_rejected() {
        let mut org = Organism::new("test");
        org.register_listener(sample_listener("echo")).unwrap();
        let err = org.register_listener(sample_listener("echo")).unwrap_err();
        assert!(err.contains("already registered"));
    }

    #[test]
    fn unregister_listener() {
        let mut org = Organism::new("test");
        org.register_listener(sample_listener("echo")).unwrap();
        org.unregister_listener("echo").unwrap();
        assert!(org.get_listener("echo").is_none());
    }

    #[test]
    fn profile_with_subset() {
        let mut org = Organism::new("test");
        org.register_listener(sample_listener("file-ops")).unwrap();
        org.register_listener(sample_listener("shell")).unwrap();
        org.register_listener(sample_listener("faq")).unwrap();

        org.add_profile(sample_profile("public", vec!["faq"]))
            .unwrap();

        let table = org.dispatch_table("public").unwrap();
        assert_eq!(table.listeners.len(), 1);
        assert!(table.listeners.contains_key("faq"));
    }

    #[test]
    fn profile_all_listeners() {
        let mut org = Organism::new("test");
        org.register_listener(sample_listener("a")).unwrap();
        org.register_listener(sample_listener("b")).unwrap();

        let profile = SecurityProfile {
            name: "root".into(),
            linux_user: "agentos-root".into(),
            allowed_listeners: HashSet::new(),
            allow_all: true,
            journal_retention: RetentionPolicy::Forever,
            network: vec![],
        };
        org.add_profile(profile).unwrap();

        let table = org.dispatch_table("root").unwrap();
        assert_eq!(table.listeners.len(), 2);
    }

    #[test]
    fn profile_references_missing_listener() {
        let mut org = Organism::new("test");
        let err = org
            .add_profile(sample_profile("bad", vec!["nonexistent"]))
            .unwrap_err();
        assert!(err.contains("unknown listener"));
    }

    #[test]
    fn hot_reload() {
        let mut org = Organism::new("test");
        org.register_listener(sample_listener("a")).unwrap();
        org.register_listener(sample_listener("b")).unwrap();

        let mut new_org = Organism::new("test-v2");
        new_org.register_listener(sample_listener("b")).unwrap();
        new_org.register_listener(sample_listener("c")).unwrap();

        let event = org.apply_config(new_org);
        assert_eq!(event.removed, vec!["a"]);
        assert!(event.added.contains(&"c".to_string()));
        assert!(event.updated.contains(&"b".to_string()));
        assert_eq!(org.name, "test-v2");
    }

    #[test]
    fn missing_profile_dispatch_table_error() {
        let org = Organism::new("test");
        let err = org.dispatch_table("nonexistent").unwrap_err();
        assert!(err.contains("not found"));
    }

    // ── Phase 5: WASM config on ListenerDef ──

    #[test]
    fn register_wasm_listener() {
        let mut org = Organism::new("test");
        org.register_listener(ListenerDef {
            name: "echo".into(),
            payload_tag: "EchoRequest".into(),
            handler: "wasm".into(),
            description: "Echo WASM tool".into(),
            is_agent: false,
            peers: vec![],
            model: None,
            ports: vec![],
            librarian: false,
            wasm: Some(WasmToolConfig {
                path: "tools/echo.wasm".into(),
                capabilities: WasmCapabilities::default(),
            }),
            semantic_description: None,
        })
        .unwrap();

        assert!(org.get_listener("echo").is_some());
        assert!(org.get_listener("echo").unwrap().wasm.is_some());
    }

    #[test]
    fn wasm_config_on_listener_def() {
        let mut org = Organism::new("test");
        org.register_listener(ListenerDef {
            name: "my-tool".into(),
            payload_tag: "MyToolRequest".into(),
            handler: "wasm".into(),
            description: "My WASM tool".into(),
            is_agent: false,
            peers: vec![],
            model: None,
            ports: vec![],
            librarian: false,
            wasm: Some(WasmToolConfig {
                path: "tools/my_tool.wasm".into(),
                capabilities: WasmCapabilities {
                    stdio: true,
                    ..Default::default()
                },
            }),
            semantic_description: None,
        })
        .unwrap();

        let def = org.get_listener("my-tool").unwrap();
        let wasm = def.wasm.as_ref().unwrap();
        assert_eq!(wasm.path, "tools/my_tool.wasm");
        assert!(wasm.capabilities.stdio);
    }

    // ── Semantic Routing: semantic_description field ──

    #[test]
    fn listener_def_semantic_description() {
        let def = sample_listener("echo");
        // Defaults to None
        assert!(def.semantic_description.is_none());

        // Can be set explicitly
        let def_with = ListenerDef {
            semantic_description: Some("This tool echoes messages back.".into()),
            ..sample_listener("echo")
        };
        assert_eq!(
            def_with.semantic_description.as_deref(),
            Some("This tool echoes messages back.")
        );
    }
}
