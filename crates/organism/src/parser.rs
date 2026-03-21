//! YAML parser for organism configuration.
//!
//! Parses `organism.yaml` into an `Organism` struct by calling the
//! imperative API (register_listener, add_profile, etc.).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::Deserialize;

use super::profile::{RetentionPolicy, SecurityProfile};
use super::{
    AgentConfig, BufferConfig, CallableParam, ListenerDef, Organism, PortDef, PythonToolConfig,
    WasmToolConfig,
};
use agentos_events::{EnvGrant, FsGrant, KvGrant, PermissionMap, PermissionTier, WasmCapabilities};

/// Top-level organism YAML configuration.
///
/// Defines an organism: its identity, listeners (handlers), security
/// profiles, and named prompt templates.
#[derive(Debug, Deserialize, JsonSchema)]
struct OrganismYaml {
    /// Organism identity metadata.
    organism: OrganismMeta,
    /// Array of listener definitions — each handles one payload type.
    #[serde(default)]
    listeners: Vec<ListenerYaml>,
    /// Security profiles — map of profile name to access rules.
    #[serde(default)]
    profiles: std::collections::HashMap<String, ProfileYaml>,
    /// Named prompt templates for agent identity. Values can be inline text or `file:path`.
    #[serde(default)]
    prompts: std::collections::HashMap<String, String>,
    /// Onboarding script steps (decision tree run on first launch).
    #[serde(default)]
    onboarding: Vec<OnboardingStepYaml>,
    /// KV store configuration. `true`/`"yes"`/`"memory"` = in-memory,
    /// a path string = on-disk. Omit or `false`/`"no"` = no KV store.
    #[serde(default, rename = "kv-store")]
    kv_store: Option<KvStoreYaml>,
    /// Organism files to import (paths relative to this file's directory).
    /// Listeners and prompts are merged; profiles/onboarding/kv-store are root-only.
    #[serde(default)]
    imports: Vec<String>,
}

/// KV store YAML value — accepts bool or string.
#[derive(Debug, JsonSchema)]
enum KvStoreYaml {
    Enabled(bool),
    Value(String),
}

impl<'de> serde::Deserialize<'de> for KvStoreYaml {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KvVisitor;
        impl<'de> serde::de::Visitor<'de> for KvVisitor {
            type Value = KvStoreYaml;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("bool, \"yes\", \"no\", \"memory\", or a filesystem path")
            }

            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(KvStoreYaml::Enabled(v))
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(KvStoreYaml::Value(v.to_string()))
            }

            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(KvStoreYaml::Value(v))
            }
        }
        deserializer.deserialize_any(KvVisitor)
    }
}

impl KvStoreYaml {
    fn to_config(&self) -> super::KvStoreConfig {
        match self {
            KvStoreYaml::Enabled(true) => super::KvStoreConfig::Memory,
            KvStoreYaml::Enabled(false) => super::KvStoreConfig::None,
            KvStoreYaml::Value(s) => {
                let lower = s.to_lowercase();
                match lower.as_str() {
                    "yes" | "memory" | "true" => super::KvStoreConfig::Memory,
                    "no" | "false" | "" => super::KvStoreConfig::None,
                    _ => super::KvStoreConfig::Disk(s.clone()),
                }
            }
        }
    }
}

/// Organism identity metadata.
#[derive(Debug, Deserialize, JsonSchema)]
struct OrganismMeta {
    /// Unique name for this organism configuration.
    name: String,
}

/// A listener definition — handles one payload type via a handler function.
#[derive(Debug, Deserialize, JsonSchema)]
struct ListenerYaml {
    /// Unique identifier for this listener, kebab-case.
    name: String,
    /// Fully qualified payload class (e.g., `agent.AgentTask`). Last component becomes payload tag.
    payload_class: String,
    /// Handler: dotted path (e.g., `tools.file_read.handle`), `wasm`, or `buffer`.
    handler: String,
    /// Human-readable purpose of this listener.
    description: String,
    /// Agent config. `true` for defaults, or block: `{ prompt, max_tokens, ... }`. Alias: `is_agent`.
    #[serde(default, alias = "is_agent")]
    agent: AgentFieldYaml,
    /// Tools and agents this listener may call. A list of names, or `"auto"` to discover all.
    #[serde(default)]
    tools: ToolsSpec,
    /// LLM model override — `opus`, `sonnet`, or `haiku`. Default: pool default.
    #[serde(default)]
    model: Option<String>,
    /// Network port declarations.
    #[serde(default)]
    ports: Vec<PortYaml>,
    /// `true` to auto-curate context via Haiku librarian. Default: `false`.
    #[serde(default)]
    librarian: bool,
    /// WASM sandboxed tool configuration.
    #[serde(default)]
    wasm: Option<WasmYaml>,
    /// Natural language description for embedding-based semantic routing.
    #[serde(default)]
    semantic_description: Option<String>,
    /// Buffer node: callable tool interface + child pipeline spawn config.
    #[serde(default)]
    buffer: Option<BufferYaml>,
    /// Python tool configuration (handler == "python").
    #[serde(default)]
    python: Option<PythonYaml>,
}

/// Agent field: `true` for defaults, or a configuration block. Untagged for YAML flexibility.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(untagged)]
enum AgentFieldYaml {
    /// Boolean shorthand — `true` enables agent with defaults.
    Bool(bool),
    /// Full agent configuration block.
    Config(AgentConfigYaml),
}

impl Default for AgentFieldYaml {
    fn default() -> Self {
        AgentFieldYaml::Bool(false)
    }
}

/// Agent configuration block.
#[derive(Debug, Deserialize, JsonSchema)]
struct AgentConfigYaml {
    /// Prompt label(s). Use `&` to compose: `"safety & coding_base"`. Labels must exist in `prompts:` section.
    #[serde(default)]
    prompt: Option<String>,
    /// Maximum LLM completion tokens. Default: 4096.
    #[serde(default)]
    max_tokens: Option<u32>,
    /// Maximum semantic routing iterations. Default: 5.
    #[serde(default)]
    max_iterations: Option<usize>,
    /// Maximum tool-call loop iterations. Default: 25.
    #[serde(default)]
    max_agentic_iterations: Option<usize>,
    /// LLM model override — `opus`, `sonnet`, or `haiku`.
    #[serde(default)]
    model: Option<String>,
    /// Per-tool permission tiers: `auto` (no approval), `prompt` (ask user), `deny` (never).
    #[serde(default)]
    permissions: std::collections::HashMap<String, String>,
}

/// WASM sandboxed tool configuration.
#[derive(Debug, Deserialize, JsonSchema)]
struct WasmYaml {
    /// Path to the WASM binary (relative to organism directory).
    path: String,
    /// Sandbox capability grants.
    #[serde(default)]
    capabilities: Option<WasmCapabilitiesYaml>,
}

/// WASM sandbox capabilities — filesystem, environment, and stdio grants.
#[derive(Debug, Deserialize, Default, JsonSchema)]
struct WasmCapabilitiesYaml {
    /// Filesystem mount grants.
    #[serde(default)]
    filesystem: Vec<FsGrantYaml>,
    /// Environment variable grants.
    #[serde(default)]
    env: Vec<EnvGrantYaml>,
    /// Allow stdio access. Default: `false`.
    #[serde(default)]
    stdio: bool,
    /// KV store grants. When present, the tool gets a private namespace
    /// plus any declared read/write access to shared namespaces.
    #[serde(default)]
    kv: Option<KvGrantYaml>,
}

/// KV store access grant.
#[derive(Debug, Deserialize, JsonSchema)]
struct KvGrantYaml {
    /// Shared namespaces this tool can read (e.g., `["stocks", "market"]`).
    #[serde(default)]
    read: Vec<String>,
    /// Shared namespaces this tool can write (e.g., `["market"]`).
    #[serde(default)]
    write: Vec<String>,
}

/// Filesystem mount grant for WASM sandbox.
#[derive(Debug, Deserialize, JsonSchema)]
struct FsGrantYaml {
    /// Host filesystem path to mount.
    host_path: String,
    /// Guest filesystem path (inside WASM).
    guest_path: String,
    /// Mount as read-only. Default: `false`.
    #[serde(default)]
    read_only: bool,
}

/// Environment variable grant for WASM sandbox.
#[derive(Debug, Deserialize, JsonSchema)]
struct EnvGrantYaml {
    /// Environment variable name.
    key: String,
    /// Environment variable value.
    value: String,
}

/// Tool parameter definition for buffer callable interface.
#[derive(Debug, Deserialize, JsonSchema)]
struct CallableParamYaml {
    /// Parameter type: `string`, `integer`, `boolean`, etc.
    #[serde(rename = "type")]
    param_type: String,
    /// Human-readable parameter description.
    #[serde(default)]
    description: Option<String>,
    /// Allowed values (enum constraint).
    #[serde(default, rename = "enum")]
    enum_values: Option<Vec<String>>,
}

/// Buffer node: callable tool interface + child pipeline spawn config.
///
/// A buffer makes a listener callable as a tool. The calling agent sends
/// parameters; the system spawns an isolated child pipeline to execute.
#[derive(Debug, Deserialize, JsonSchema)]
struct BufferYaml {
    /// Tool description shown to the calling agent.
    description: String,
    /// Tool parameters — map of parameter name to type/description.
    #[serde(default)]
    parameters: std::collections::HashMap<String, CallableParamYaml>,
    /// Which parameters are mandatory.
    #[serde(default)]
    required: Vec<String>,
    /// Tools available inside the child pipeline (e.g., `[file-read, command-exec]`).
    #[serde(default)]
    requires: Vec<String>,
    /// Child organism YAML file (relative to `--dir`). Omit to clone current organism (self-referential).
    #[serde(default)]
    organism: Option<String>,
    /// Maximum parallel child instances. Default: 5.
    #[serde(default = "default_max_concurrency")]
    #[schemars(default = "default_max_concurrency")]
    max_concurrency: usize,
    /// Execution timeout in seconds. Default: 300.
    #[serde(default = "default_timeout_secs")]
    #[schemars(default = "default_timeout_secs")]
    timeout_secs: u64,
    /// Forward child events (thinking, tool calls) to parent TUI. Default: false.
    #[serde(default)]
    context_visible: bool,
    /// Child agent takes over TUI focus (interactive mode). Default: false.
    /// When true, the child agent's output appears in chat and user input routes to it.
    #[serde(default)]
    interactive: bool,
}

fn default_max_concurrency() -> usize {
    5
}

fn default_timeout_secs() -> u64 {
    300
}

/// Python tool configuration (handler == "python").
#[derive(Debug, Deserialize, JsonSchema)]
struct PythonYaml {
    /// Path to the .py source file (relative to organism base dir).
    source: String,
}

/// Network port declaration for a listener.
#[derive(Debug, Deserialize, JsonSchema)]
struct PortYaml {
    /// Port number.
    port: u16,
    /// Direction: `inbound` or `outbound`.
    direction: String,
    /// Network protocol: `https`, `http`, `ssh`, etc.
    protocol: String,
    /// Target hosts for outbound connections (e.g., `["api.anthropic.com"]`).
    #[serde(default)]
    hosts: Vec<String>,
}

/// Security profile — access rules for a set of listeners.
#[derive(Debug, Deserialize, JsonSchema)]
struct ProfileYaml {
    /// Linux user for process isolation (e.g., `agentos-root`).
    linux_user: String,
    /// Which listeners this profile may access: `"all"` or a list of names.
    listeners: ListenersSpec,
    /// Message retention policy. Default: `retain_forever`.
    #[serde(default)]
    journal: JournalSpec,
    /// Listeners whose network ports this profile may use.
    #[serde(default)]
    network: Vec<String>,
}

/// Tools spec: `"auto"` for auto-discovery, or a list of listener names.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(untagged)]
enum ToolsSpec {
    /// Auto-discover all available tools and agents at pipeline build time.
    Auto(String),
    /// Explicit list of tool/agent names this listener may call.
    List(Vec<String>),
}

impl Default for ToolsSpec {
    fn default() -> Self {
        ToolsSpec::List(vec![])
    }
}

/// Listeners access spec: `"all"` or a list of listener names.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(untagged)]
enum ListenersSpec {
    /// Grant access to all listeners.
    All(String),
    /// Grant access to specific listeners by name.
    List(Vec<String>),
}

/// Journal retention policy.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(untagged)]
enum JournalSpec {
    /// Simple policy: `"retain_forever"` or `"prune_on_delivery"`.
    Simple(String),
    /// Retain for a number of days.
    WithDays(JournalDaysSpec),
}

impl Default for JournalSpec {
    fn default() -> Self {
        JournalSpec::Simple("retain_forever".into())
    }
}

/// Journal retention by day count.
#[derive(Debug, Deserialize, JsonSchema)]
struct JournalDaysSpec {
    /// Number of days to retain journal messages.
    retain_days: u16,
}

// ── Onboarding script YAML types ──

/// A single step in an onboarding script (serde representation).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(untagged)]
enum OnboardingStepYaml {
    Say { say: String },
    Choice { choice: OnboardingChoiceYaml },
    Open { open: String },
    Wait { wait: String },
}

/// A choice block with a prompt and options.
#[derive(Debug, Deserialize, JsonSchema)]
struct OnboardingChoiceYaml {
    prompt: String,
    options: Vec<OnboardingOptionYaml>,
}

/// A choice option with nested sub-steps.
#[derive(Debug, Deserialize, JsonSchema)]
struct OnboardingOptionYaml {
    label: String,
    #[serde(default)]
    value: String,
    #[serde(default)]
    steps: Vec<OnboardingStepYaml>,
}

/// Convert parsed YAML onboarding steps to domain types.
fn convert_onboarding_steps(yaml_steps: Vec<OnboardingStepYaml>) -> Vec<super::OnboardingStep> {
    yaml_steps.into_iter().map(|s| match s {
        OnboardingStepYaml::Say { say } => super::OnboardingStep::Say(say),
        OnboardingStepYaml::Open { open } => super::OnboardingStep::Open(open),
        OnboardingStepYaml::Wait { wait } => super::OnboardingStep::Wait(wait),
        OnboardingStepYaml::Choice { choice } => super::OnboardingStep::Choice {
            prompt: choice.prompt,
            options: choice.options.into_iter().map(|o| super::OnboardingChoice {
                label: o.label,
                value: o.value,
                steps: convert_onboarding_steps(o.steps),
            }).collect(),
        },
    }).collect()
}

/// Generate the JSON Schema for the organism YAML format.
pub fn generate_schema() -> serde_json::Value {
    let schema = schemars::schema_for!(OrganismYaml);
    serde_json::to_value(schema).expect("schema serialization cannot fail")
}

/// Load an organism from a YAML file, resolving imports recursively.
///
/// Imports are loaded depth-first. Listeners and prompts from imported files
/// are merged into the root organism. Duplicate listeners with matching
/// handler and payload_class are silently deduplicated; conflicts are errors.
/// Circular imports are detected and reported with the full import chain.
/// Diamond imports (A→B→D, A→C→D) are handled correctly — D is loaded once.
pub fn load_organism(path: &Path) -> Result<Organism, String> {
    let mut loading = Vec::new();
    let mut loaded = HashSet::new();
    load_organism_recursive(path, &mut loading, &mut loaded)
}

fn load_organism_recursive(
    path: &Path,
    loading: &mut Vec<PathBuf>,
    loaded: &mut HashSet<PathBuf>,
) -> Result<Organism, String> {
    let canonical = std::fs::canonicalize(path)
        .map_err(|e| format!("failed to resolve {}: {e}", path.display()))?;

    // Diamond: already fully loaded via another import path → return empty
    if loaded.contains(&canonical) {
        return Ok(Organism::new("_diamond_skip"));
    }

    // Circular: currently in the loading chain → error with full cycle
    if let Some(pos) = loading.iter().position(|p| p == &canonical) {
        let cycle: Vec<String> = loading[pos..]
            .iter()
            .chain(std::iter::once(&canonical))
            .map(|p| p.display().to_string())
            .collect();
        return Err(format!("circular import: {}", cycle.join(" → ")));
    }

    loading.push(canonical.clone());

    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let raw: OrganismYaml = serde_yaml::from_str(&contents)
        .map_err(|e| format!("YAML parse error in {}: {e}", path.display()))?;

    let base_dir = path.parent().unwrap_or(Path::new("."));
    let imports = raw.imports.clone();

    // Build this file's own organism
    let mut org = build_organism(raw, Some(base_dir))?;

    // Recursively load and merge imports
    for import_path in &imports {
        let resolved = base_dir.join(import_path);
        let imported = load_organism_recursive(&resolved, loading, loaded)?;
        org.merge_from(imported)
            .map_err(|e| format!("importing '{}': {e}", import_path))?;
    }

    // Validate profiles now that all listeners (local + imported) are registered
    org.validate_profiles()?;

    loading.pop();
    loaded.insert(canonical);
    Ok(org)
}

/// Parse an organism from a YAML string.
///
/// This parses a single YAML document without resolving imports.
/// Use [`load_organism`] to load from a file with import resolution.
pub fn parse_organism(yaml: &str) -> Result<Organism, String> {
    let raw: OrganismYaml =
        serde_yaml::from_str(yaml).map_err(|e| format!("YAML parse error: {e}"))?;
    let org = build_organism(raw, None)?;
    org.validate_profiles()?;
    Ok(org)
}

/// Build an Organism from a parsed YAML struct.
///
/// `base_dir` is used to resolve `file:` prompt references. If `None`,
/// paths are resolved relative to the current working directory.
fn build_organism(raw: OrganismYaml, base_dir: Option<&Path>) -> Result<Organism, String> {
    let mut org = Organism::new(&raw.organism.name);

    // Register prompts (resolve file: prefixes)
    for (name, value) in raw.prompts {
        if let Some(file_path) = value.strip_prefix("file:") {
            let resolved = if let Some(dir) = base_dir {
                dir.join(file_path.trim())
            } else {
                PathBuf::from(file_path.trim())
            };
            let content = std::fs::read_to_string(&resolved)
                .map_err(|e| format!("failed to load prompt file '{}': {e}", resolved.display()))?;
            org.register_prompt(name, content);
        } else {
            org.register_prompt(name, value);
        }
    }

    // Register listeners
    for l in raw.listeners {
        let payload_tag = l
            .payload_class
            .rsplit('.')
            .next()
            .unwrap_or(&l.payload_class)
            .to_string();

        let ports = l
            .ports
            .into_iter()
            .map(|p| PortDef {
                port: p.port,
                direction: p.direction,
                protocol: p.protocol,
                hosts: p.hosts,
            })
            .collect();

        // Resolve agent field: bool or config block
        let (is_agent, agent_config) = match l.agent {
            AgentFieldYaml::Config(cfg) => {
                let mut permissions = PermissionMap::new();
                for (tool, tier_str) in cfg.permissions {
                    let tier = PermissionTier::from_str(&tier_str)
                        .map_err(|e| format!("listener '{}': {e}", l.name))?;
                    permissions.insert(tool, tier);
                }
                let config = AgentConfig {
                    prompt: cfg.prompt,
                    max_tokens: cfg.max_tokens.unwrap_or(4096),
                    max_routing_iterations: cfg.max_iterations.unwrap_or(5),
                    max_agentic_iterations: cfg.max_agentic_iterations.unwrap_or(25),
                    model: cfg.model,
                    permissions,
                };
                (true, Some(config))
            }
            AgentFieldYaml::Bool(b) => {
                if b {
                    (true, Some(AgentConfig::default()))
                } else {
                    (false, None)
                }
            }
        };

        // Resolve buffer: use explicit declaration, or auto-generate default for agents
        let buffer = l.buffer.map(|b| {
            let parameters = b
                .parameters
                .into_iter()
                .map(|(name, p)| CallableParam {
                    name,
                    param_type: p.param_type,
                    description: p.description,
                    enum_values: p.enum_values,
                })
                .collect();
            BufferConfig {
                description: b.description,
                parameters,
                required: b.required,
                requires: b.requires,
                organism: b.organism,
                max_concurrency: b.max_concurrency,
                timeout_secs: b.timeout_secs,
                context_visible: b.context_visible || b.interactive,
                interactive: b.interactive,
            }
        }).or_else(|| {
            if is_agent {
                Some(BufferConfig {
                    description: l.description.clone(),
                    parameters: vec![CallableParam {
                        name: "task".into(),
                        param_type: "string".into(),
                        description: Some("What you want this agent to do".into()),
                        enum_values: None,
                    }],
                    required: vec!["task".into()],
                    requires: vec![],
                    organism: None,
                    max_concurrency: 1,
                    timeout_secs: 300,
                    context_visible: false,
                    interactive: false,
                })
            } else {
                None
            }
        });

        org.register_listener(ListenerDef {
            name: l.name,
            payload_tag,
            handler: l.handler,
            description: l.description,
            is_agent,
            tools: match &l.tools {
                ToolsSpec::Auto(_) => vec![],
                ToolsSpec::List(list) => list.clone(),
            },
            tools_auto: matches!(&l.tools, ToolsSpec::Auto(s) if s == "auto"),
            model: l.model,
            ports,
            librarian: l.librarian,
            semantic_description: l.semantic_description,
            agent_config,
            wasm: l.wasm.map(|w| {
                let caps = match w.capabilities {
                    Some(c) => WasmCapabilities {
                        filesystem: c
                            .filesystem
                            .into_iter()
                            .map(|f| FsGrant {
                                host_path: f.host_path,
                                guest_path: f.guest_path,
                                read_only: f.read_only,
                            })
                            .collect(),
                        env_vars: c
                            .env
                            .into_iter()
                            .map(|e| EnvGrant {
                                key: e.key,
                                value: e.value,
                            })
                            .collect(),
                        stdio: c.stdio,
                        kv: c.kv.map(|k| KvGrant {
                            read: k.read,
                            write: k.write,
                        }),
                    },
                    None => WasmCapabilities::default(),
                };
                WasmToolConfig {
                    path: w.path,
                    capabilities: caps,
                }
            }),
            buffer,
            python: l.python.map(|p| PythonToolConfig {
                source: p.source,
            }),
        })?;
    }

    // Register profiles
    for (name, p) in raw.profiles {
        let (allow_all, allowed_listeners) = match p.listeners {
            ListenersSpec::All(ref s) if s == "all" => (true, HashSet::new()),
            ListenersSpec::All(ref s) => {
                // Single listener name that isn't "all"
                let mut set = HashSet::new();
                set.insert(s.clone());
                (false, set)
            }
            ListenersSpec::List(names) => (false, names.into_iter().collect()),
        };

        let journal_retention = match p.journal {
            JournalSpec::Simple(ref s) if s == "retain_forever" => RetentionPolicy::Forever,
            JournalSpec::Simple(ref s) if s == "prune_on_delivery" => {
                RetentionPolicy::PruneOnDelivery
            }
            JournalSpec::Simple(ref s) => {
                return Err(format!("unknown journal retention: '{s}'"));
            }
            JournalSpec::WithDays(spec) => RetentionPolicy::RetainDays(spec.retain_days),
        };

        org.add_profile_deferred(SecurityProfile {
            name,
            linux_user: p.linux_user,
            allowed_listeners,
            allow_all,
            journal_retention,
            network: p.network,
        });
    }

    // Convert onboarding steps
    org.onboarding = convert_onboarding_steps(raw.onboarding);

    // KV store config
    org.kv_store = raw.kv_store
        .as_ref()
        .map(|k| k.to_config())
        .unwrap_or(super::KvStoreConfig::None);

    Ok(org)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_full_organism() {
        let yaml = r#"
organism:
  name: agentos

listeners:
  - name: coding-agent
    payload_class: handlers.code.CodeRequest
    handler: handlers.code.handle
    description: "Opus coding agent"
    agent: true
    tools: [file-ops, shell]
    model: opus

  - name: file-ops
    payload_class: handlers.files.FileRequest
    handler: handlers.files.handle
    description: "File operations"

  - name: shell
    payload_class: handlers.shell.ShellRequest
    handler: handlers.shell.handle
    description: "Shell execution"

  - name: faq
    payload_class: handlers.faq.FaqRequest
    handler: handlers.faq.handle
    description: "FAQ handler"

profiles:
  root:
    linux_user: agentos-root
    listeners: all
    journal: retain_forever
  admin:
    linux_user: agentos-admin
    listeners: [file-ops, shell, coding-agent]
    journal:
      retain_days: 90
  public:
    linux_user: agentos-public
    listeners: [faq]
    journal: prune_on_delivery
"#;

        let org = parse_organism(yaml).unwrap();
        assert_eq!(org.name, "agentos");
        assert_eq!(org.listener_names().len(), 4);

        // Root profile allows all
        let root_table = org.dispatch_table("root").unwrap();
        assert_eq!(root_table.listeners.len(), 4);

        // Admin profile allows 3
        let admin_table = org.dispatch_table("admin").unwrap();
        assert_eq!(admin_table.listeners.len(), 3);
        assert!(admin_table.has_listener("file-ops"));
        assert!(!admin_table.has_listener("faq"));

        // Public profile allows 1
        let public_table = org.dispatch_table("public").unwrap();
        assert_eq!(public_table.listeners.len(), 1);
        assert!(public_table.has_listener("faq"));
    }

    #[test]
    fn parse_minimal_organism() {
        let yaml = r#"
organism:
  name: minimal
listeners: []
"#;
        let org = parse_organism(yaml).unwrap();
        assert_eq!(org.name, "minimal");
        assert_eq!(org.listener_names().len(), 0);
    }

    #[test]
    fn parse_organism_with_ports_and_network() {
        let yaml = r#"
organism:
  name: agentos-m2

listeners:
  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM inference pool"
    tools: [coding-agent]
    ports:
      - port: 443
        direction: outbound
        protocol: https
        hosts: [api.anthropic.com]

  - name: file-ops
    payload_class: tools.FileOpsRequest
    handler: tools.file_ops.handle
    description: "File operations"

  - name: shell
    payload_class: tools.ShellRequest
    handler: tools.shell.handle
    description: "Shell execution"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [file-ops, shell, llm-pool]
    network: [llm-pool]
    journal:
      retain_days: 90
  public:
    linux_user: agentos-public
    listeners: [file-ops]
    journal: prune_on_delivery
"#;

        let org = parse_organism(yaml).unwrap();
        assert_eq!(org.name, "agentos-m2");

        // LLM pool has port declarations
        let llm = org.get_listener("llm-pool").unwrap();
        assert_eq!(llm.ports.len(), 1);
        assert_eq!(llm.ports[0].port, 443);
        assert_eq!(llm.ports[0].direction, "outbound");
        assert_eq!(llm.ports[0].protocol, "https");
        assert_eq!(llm.ports[0].hosts, vec!["api.anthropic.com"]);

        // File-ops has no ports
        let fops = org.get_listener("file-ops").unwrap();
        assert!(fops.ports.is_empty());

        // Admin profile has network field
        let admin = org.get_profile("admin").unwrap();
        assert_eq!(admin.network, vec!["llm-pool"]);

        // Public profile has empty network
        let public = org.get_profile("public").unwrap();
        assert!(public.network.is_empty());
    }

    #[test]
    fn parse_librarian_flag() {
        let yaml = r#"
organism:
  name: test-librarian

listeners:
  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM pool"
    librarian: true

  - name: echo
    payload_class: handlers.echo.Greeting
    handler: handlers.echo.handle
    description: "Echo"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [llm-pool, echo]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();

        // llm-pool has librarian: true
        let llm = org.get_listener("llm-pool").unwrap();
        assert!(llm.librarian);

        // echo defaults to librarian: false
        let echo = org.get_listener("echo").unwrap();
        assert!(!echo.librarian);
    }

    #[test]
    fn parse_invalid_yaml() {
        let err = parse_organism("{{invalid").unwrap_err();
        assert!(err.contains("YAML parse error"));
    }

    #[test]
    fn profile_references_missing_listener() {
        let yaml = r#"
organism:
  name: bad

profiles:
  broken:
    linux_user: nobody
    listeners: [nonexistent]
    journal: retain_forever
"#;
        let err = parse_organism(yaml).unwrap_err();
        assert!(err.contains("unknown listener"));
    }

    // ── Phase 5: WASM listener parsing ──

    #[test]
    fn parse_wasm_listener() {
        let yaml = r#"
organism:
  name: test-wasm

listeners:
  - name: echo
    payload_class: tools.EchoRequest
    handler: wasm
    description: "Echo tool (WASM)"
    wasm:
      path: tools/echo.wasm

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [echo]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let echo = org.get_listener("echo").unwrap();
        assert_eq!(echo.handler, "wasm");
        let wasm = echo.wasm.as_ref().expect("wasm config should be present");
        assert_eq!(wasm.path, "tools/echo.wasm");
    }

    #[test]
    fn parse_wasm_with_capabilities() {
        let yaml = r#"
organism:
  name: test-wasm-caps

listeners:
  - name: my-tool
    payload_class: tools.MyToolRequest
    handler: wasm
    description: "My custom tool"
    wasm:
      path: tools/my_tool.wasm
      capabilities:
        filesystem:
          - host_path: /data
            guest_path: /data
            read_only: true
        env:
          - key: RUST_LOG
            value: info
        stdio: true

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [my-tool]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let tool = org.get_listener("my-tool").unwrap();
        let wasm = tool.wasm.as_ref().unwrap();
        assert_eq!(wasm.path, "tools/my_tool.wasm");
        assert_eq!(wasm.capabilities.filesystem.len(), 1);
        assert_eq!(wasm.capabilities.filesystem[0].host_path, "/data");
        assert!(wasm.capabilities.filesystem[0].read_only);
        assert_eq!(wasm.capabilities.env_vars.len(), 1);
        assert_eq!(wasm.capabilities.env_vars[0].key, "RUST_LOG");
        assert!(wasm.capabilities.stdio);
    }

    #[test]
    fn parse_listener_without_wasm() {
        let yaml = r#"
organism:
  name: test-no-wasm

listeners:
  - name: file-ops
    payload_class: tools.FileOpsRequest
    handler: tools.file_ops.handle
    description: "File operations"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [file-ops]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let fops = org.get_listener("file-ops").unwrap();
        assert!(fops.wasm.is_none());
    }

    #[test]
    fn parse_wasm_empty_capabilities() {
        let yaml = r#"
organism:
  name: test-wasm-empty

listeners:
  - name: echo
    payload_class: tools.EchoRequest
    handler: wasm
    description: "Echo (no caps)"
    wasm:
      path: tools/echo.wasm

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [echo]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let echo = org.get_listener("echo").unwrap();
        let wasm = echo.wasm.as_ref().unwrap();
        assert!(wasm.capabilities.filesystem.is_empty());
        assert!(wasm.capabilities.env_vars.is_empty());
        assert!(!wasm.capabilities.stdio);
    }

    // ── Python Tools: handler: python parsing ──

    #[test]
    fn parse_python_listener() {
        let yaml = r#"
organism:
  name: test-python

listeners:
  - name: echo-py
    payload_class: tools.EchoRequest
    handler: python
    description: "Echo tool (Python)"
    python:
      source: tools/echo_tool.py

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [echo-py]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let echo = org.get_listener("echo-py").unwrap();
        assert_eq!(echo.handler, "python");
        let py = echo.python.as_ref().expect("python config should be present");
        assert_eq!(py.source, "tools/echo_tool.py");
    }

    #[test]
    fn parse_listener_without_python() {
        let yaml = r#"
organism:
  name: test-no-python

listeners:
  - name: regular
    payload_class: tools.RegularRequest
    handler: tools.regular.handle
    description: "Regular tool"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [regular]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let regular = org.get_listener("regular").unwrap();
        assert!(regular.python.is_none());
    }

    // ── Interactive Buffer: interactive flag parsing ──

    #[test]
    fn parse_interactive_buffer() {
        let yaml = r#"
organism:
  name: test-interactive

listeners:
  - name: expert
    payload_class: buffer.ExpertRequest
    handler: buffer
    description: "Expert agent"
    buffer:
      description: "Expert that takes over the TUI"
      parameters:
        task:
          type: string
      required: [task]
      requires: [file-read]
      organism: expert.yaml
      interactive: true

profiles:
  default:
    linux_user: agentos
    listeners: [expert]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let expert = org.get_listener("expert").unwrap();
        let buf = expert.buffer.as_ref().expect("buffer config present");
        assert!(buf.interactive, "interactive should be true");
        assert!(buf.context_visible, "context_visible implied by interactive");
    }

    #[test]
    fn parse_non_interactive_buffer_default() {
        let yaml = r#"
organism:
  name: test-default

listeners:
  - name: worker
    payload_class: buffer.WorkerRequest
    handler: buffer
    description: "Worker"
    buffer:
      description: "Background worker"
      parameters:
        task:
          type: string
      required: [task]
      requires: [file-read]
      organism: worker.yaml

profiles:
  default:
    linux_user: agentos
    listeners: [worker]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let worker = org.get_listener("worker").unwrap();
        let buf = worker.buffer.as_ref().expect("buffer config present");
        assert!(!buf.interactive, "interactive defaults to false");
        assert!(!buf.context_visible, "context_visible defaults to false");
    }

    // ── Semantic Routing: semantic_description parsing ──

    #[test]
    fn parse_semantic_description_yaml() {
        let yaml = r#"
organism:
  name: test-routing

listeners:
  - name: file-ops
    payload_class: tools.FileOpsRequest
    handler: tools.file_ops.handle
    description: "File operations"
    semantic_description: |
      This tool reads, writes, and manages files on the local filesystem.
      Use it when you need to examine source code or read configuration files.

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [file-ops]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let fops = org.get_listener("file-ops").unwrap();
        let desc = fops.semantic_description.as_ref().unwrap();
        assert!(desc.contains("reads, writes, and manages files"));
    }

    #[test]
    fn parse_missing_semantic_description() {
        let yaml = r#"
organism:
  name: test-no-routing

listeners:
  - name: shell
    payload_class: tools.ShellRequest
    handler: tools.shell.handle
    description: "Shell execution"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [shell]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let shell = org.get_listener("shell").unwrap();
        assert!(shell.semantic_description.is_none());
    }

    // ── YAML-Defined Agents: prompts section, agent config block ──

    #[test]
    fn parse_prompts_section() {
        let yaml = r#"
organism:
  name: test-prompts

prompts:
  greeting: "Hello, agent!"
  safety: |
    You are bounded.
    You do not pursue goals beyond your task.

listeners: []
"#;
        let org = parse_organism(yaml).unwrap();
        assert_eq!(org.get_prompt("greeting"), Some("Hello, agent!"));
        assert!(org.get_prompt("safety").unwrap().contains("You are bounded"));
        assert_eq!(org.prompts().len(), 2);
    }

    #[test]
    fn parse_agent_config_block() {
        let yaml = r#"
organism:
  name: test-agent-config

listeners:
  - name: coding-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Coding agent"
    agent:
      prompt: "safety & coding_base"
      max_tokens: 8192
      max_iterations: 10
      model: haiku
    tools: [file-read]

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [coding-agent]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let agent = org.get_listener("coding-agent").unwrap();
        assert!(agent.is_agent);

        let cfg = agent.agent_config.as_ref().unwrap();
        assert_eq!(cfg.prompt.as_deref(), Some("safety & coding_base"));
        assert_eq!(cfg.max_tokens, 8192);
        assert_eq!(cfg.max_routing_iterations, 10);
        assert_eq!(cfg.max_agentic_iterations, 25);
        assert_eq!(cfg.model.as_deref(), Some("haiku"));
    }

    #[test]
    fn parse_agent_bool_compat() {
        let yaml = r#"
organism:
  name: test-bool

listeners:
  - name: coding-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Coding agent"
    agent: true
    tools: [file-read]

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [coding-agent]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let agent = org.get_listener("coding-agent").unwrap();
        assert!(agent.is_agent);

        // Bool true → default AgentConfig
        let cfg = agent.agent_config.as_ref().unwrap();
        assert_eq!(cfg.prompt, None);
        assert_eq!(cfg.max_tokens, 4096);
        assert_eq!(cfg.max_routing_iterations, 5);
        assert_eq!(cfg.max_agentic_iterations, 25);
        assert_eq!(cfg.model, None);
    }

    #[test]
    fn parse_is_agent_alias() {
        let yaml = r#"
organism:
  name: test-alias

listeners:
  - name: coding-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Coding agent"
    is_agent: true
    tools: [file-read]

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [coding-agent]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let agent = org.get_listener("coding-agent").unwrap();
        assert!(agent.is_agent);
        assert!(agent.agent_config.is_some());
    }

    #[test]
    fn parse_agent_config_defaults() {
        let yaml = r#"
organism:
  name: test-defaults

listeners:
  - name: agent
    payload_class: agent.Task
    handler: agent.handle
    description: "Agent"
    agent:
      prompt: "my_prompt"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [agent]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let agent = org.get_listener("agent").unwrap();
        assert!(agent.is_agent);

        let cfg = agent.agent_config.as_ref().unwrap();
        assert_eq!(cfg.prompt.as_deref(), Some("my_prompt"));
        // Defaults
        assert_eq!(cfg.max_tokens, 4096);
        assert_eq!(cfg.max_routing_iterations, 5);
        assert_eq!(cfg.max_agentic_iterations, 25);
        assert_eq!(cfg.model, None);
    }

    #[test]
    fn parse_max_agentic_iterations() {
        let yaml = r#"
organism:
  name: test-agentic

listeners:
  - name: agent
    payload_class: agent.Task
    handler: agent.handle
    description: "Agent"
    agent:
      prompt: "my_prompt"
      max_iterations: 8
      max_agentic_iterations: 50

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [agent]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let agent = org.get_listener("agent").unwrap();
        let cfg = agent.agent_config.as_ref().unwrap();
        assert_eq!(cfg.max_routing_iterations, 8);
        assert_eq!(cfg.max_agentic_iterations, 50);
    }

    #[test]
    fn parse_agent_false_no_config() {
        let yaml = r#"
organism:
  name: test-false

listeners:
  - name: tool
    payload_class: tools.Request
    handler: tools.handle
    description: "A tool"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [tool]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let tool = org.get_listener("tool").unwrap();
        assert!(!tool.is_agent);
        assert!(tool.agent_config.is_none());
    }

    #[test]
    fn parse_file_prompt() {
        let dir = tempfile::TempDir::new().unwrap();
        let prompt_path = dir.path().join("test_prompt.md");
        std::fs::write(&prompt_path, "You are a test prompt from a file.").unwrap();

        // Use forward slashes for YAML compatibility (avoids hex escape issues)
        let path_str = prompt_path.display().to_string().replace('\\', "/");
        let yaml = format!(
            r#"
organism:
  name: test-file-prompt

prompts:
  from_file: "file:{path_str}"

listeners: []
"#,
        );

        let org = parse_organism(&yaml).unwrap();
        assert_eq!(
            org.get_prompt("from_file"),
            Some("You are a test prompt from a file.")
        );
    }

    // ── Buffer Node: callable + buffer parsing ──

    #[test]
    fn parse_buffer() {
        let yaml = r#"
organism:
  name: test-buffer

listeners:
  - name: email-sender
    payload_class: buffer.EmailSenderRequest
    handler: buffer
    description: "Send marketing email"
    buffer:
      description: "Send a marketing email to a recipient"
      parameters:
        to: { type: string, description: "Recipient email" }
        subject: { type: string, description: "Subject line" }
        body: { type: string, description: "Email body" }
      required: [to, subject, body]
      requires: [command-exec]
      organism: email-agent.yaml
      max_concurrency: 5
      timeout_secs: 120

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [email-sender]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let listener = org.get_listener("email-sender").unwrap();

        // buffer config (tool interface + spawn config unified)
        let buffer = listener.buffer.as_ref().unwrap();
        assert_eq!(buffer.description, "Send a marketing email to a recipient");
        assert_eq!(buffer.parameters.len(), 3);
        assert_eq!(buffer.required, vec!["to", "subject", "body"]);
        assert_eq!(buffer.requires, vec!["command-exec"]);
        assert_eq!(buffer.organism.as_deref(), Some("email-agent.yaml"));
        assert_eq!(buffer.max_concurrency, 5);
        assert_eq!(buffer.timeout_secs, 120);

        // Verify params
        let to_param = buffer.parameters.iter().find(|p| p.name == "to").unwrap();
        assert_eq!(to_param.param_type, "string");
        assert_eq!(to_param.description.as_deref(), Some("Recipient email"));
    }

    #[test]
    fn parse_buffer_defaults() {
        let yaml = r#"
organism:
  name: test-buffer-defaults

listeners:
  - name: worker
    payload_class: buffer.WorkerRequest
    handler: buffer
    description: "Worker"
    buffer:
      description: "Run a worker task"
      parameters:
        task: { type: string, description: "Task to run" }
      required: [task]
      organism: worker.yaml

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [worker]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let listener = org.get_listener("worker").unwrap();

        let buffer = listener.buffer.as_ref().unwrap();
        assert_eq!(buffer.max_concurrency, 5);
        assert_eq!(buffer.timeout_secs, 300);
    }

    #[test]
    fn parse_self_referential_buffer() {
        let yaml = r#"
organism:
  name: test-self-ref

listeners:
  - name: researcher
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Research agent"
    agent:
      prompt: "research_base"
      max_agentic_iterations: 25
    tools: [file-read, researcher]
    buffer:
      description: "Research a sub-topic"
      parameters:
        topic: { type: string, description: "The sub-topic to research" }
      required: [topic]
      max_concurrency: 3

  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"

profiles:
  default:
    linux_user: agentos
    listeners: [researcher, file-read]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let listener = org.get_listener("researcher").unwrap();

        let buffer = listener.buffer.as_ref().unwrap();
        assert_eq!(buffer.organism, None); // self-referential: no child organism
        assert_eq!(buffer.max_concurrency, 3);
        assert_eq!(buffer.description, "Research a sub-topic");
        assert_eq!(buffer.parameters.len(), 1);

        // Also an agent
        assert!(listener.is_agent);
        assert!(listener.agent_config.is_some());

        // Appears in both agent and buffer listener lists
        assert_eq!(org.agent_listeners().len(), 1);
        assert_eq!(org.buffer_listeners().len(), 1);
    }

    #[test]
    fn parse_no_buffer() {
        let yaml = r#"
organism:
  name: test-no-buffer

listeners:
  - name: echo
    payload_class: handlers.echo.Greeting
    handler: handlers.echo.handle
    description: "Echo"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [echo]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let echo = org.get_listener("echo").unwrap();
        assert!(echo.buffer.is_none());
    }

    #[test]
    fn buffer_to_tool_definition() {
        let yaml = r#"
organism:
  name: test-tool-def

listeners:
  - name: email-sender
    payload_class: buffer.EmailSenderRequest
    handler: buffer
    description: "Send email"
    buffer:
      description: "Send a marketing email"
      parameters:
        to: { type: string, description: "Recipient email" }
        count: { type: integer, description: "Number of emails" }
      required: [to]
      organism: email-agent.yaml

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [email-sender]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let listener = org.get_listener("email-sender").unwrap();
        let buffer = listener.buffer.as_ref().unwrap();

        let tool_def = buffer.to_tool_definition("email-sender");
        assert_eq!(tool_def.name, "email-sender");
        assert_eq!(tool_def.description, "Send a marketing email");

        // Check JSON Schema structure
        let schema = tool_def.input_schema.as_object().unwrap();
        assert_eq!(schema["type"], "object");
        let props = schema["properties"].as_object().unwrap();
        assert!(props.contains_key("to"));
        assert!(props.contains_key("count"));
        assert_eq!(props["count"]["type"], "integer");
        let required = schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "to");
    }

    #[test]
    fn buffer_listeners_filter() {
        let yaml = r#"
organism:
  name: test-filter

listeners:
  - name: email-sender
    payload_class: buffer.EmailSenderRequest
    handler: buffer
    description: "Send email"
    buffer:
      description: "Send email"
      parameters:
        to: { type: string }
      required: [to]
      organism: email-agent.yaml

  - name: file-ops
    payload_class: tools.FileOpsRequest
    handler: tools.file_ops.handle
    description: "File ops"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [email-sender, file-ops]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let buffer_listeners = org.buffer_listeners();
        assert_eq!(buffer_listeners.len(), 1);
        assert_eq!(buffer_listeners[0].name, "email-sender");
    }

    // ── Permissions parsing ──

    #[test]
    fn parse_agent_permissions() {
        let yaml = r#"
organism:
  name: test-permissions

listeners:
  - name: coding-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Coding agent"
    agent:
      prompt: "coding_base"
      permissions:
        file-read: auto
        glob: auto
        grep: auto
        file-write: prompt
        file-edit: prompt
        command-exec: deny
    tools: [file-read, file-write, file-edit, glob, grep, command-exec]
  - name: file-read
    payload_class: tools.FileRead
    handler: tools.file_read
    description: "Read files"
  - name: file-write
    payload_class: tools.FileWrite
    handler: tools.file_write
    description: "Write files"
  - name: file-edit
    payload_class: tools.FileEdit
    handler: tools.file_edit
    description: "Edit files"
  - name: glob
    payload_class: tools.Glob
    handler: tools.glob
    description: "Glob search"
  - name: grep
    payload_class: tools.Grep
    handler: tools.grep
    description: "Grep search"
  - name: command-exec
    payload_class: tools.CommandExec
    handler: tools.command_exec
    description: "Execute commands"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [coding-agent, file-read, file-write, file-edit, glob, grep, command-exec]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let agent = org.get_listener("coding-agent").unwrap();
        let cfg = agent.agent_config.as_ref().unwrap();

        use agentos_events::PermissionTier;
        assert_eq!(cfg.permissions.get("file-read"), Some(&PermissionTier::Auto));
        assert_eq!(cfg.permissions.get("glob"), Some(&PermissionTier::Auto));
        assert_eq!(cfg.permissions.get("file-write"), Some(&PermissionTier::Prompt));
        assert_eq!(cfg.permissions.get("command-exec"), Some(&PermissionTier::Deny));
        // Unlisted tool → not in map (resolve_tier will return Prompt)
        assert_eq!(cfg.permissions.get("codebase-index"), None);
    }

    #[test]
    fn parse_agent_no_permissions() {
        let yaml = r#"
organism:
  name: test-no-perms

listeners:
  - name: coding-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Coding agent"
    agent:
      prompt: "coding_base"
    tools: [file-read]
  - name: file-read
    payload_class: tools.FileRead
    handler: tools.file_read
    description: "Read files"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [coding-agent, file-read]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let agent = org.get_listener("coding-agent").unwrap();
        let cfg = agent.agent_config.as_ref().unwrap();
        assert!(cfg.permissions.is_empty());
    }

    #[test]
    fn parse_invalid_permission_tier() {
        let yaml = r#"
organism:
  name: test-bad-tier

listeners:
  - name: coding-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Coding agent"
    agent:
      prompt: "coding_base"
      permissions:
        file-read: always

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [coding-agent]
    journal: retain_forever
"#;
        let err = parse_organism(yaml).unwrap_err();
        assert!(err.contains("unknown permission tier"));
        assert!(err.contains("always"));
    }

    // ── JSON Schema generation ──

    #[test]
    fn generate_schema_produces_valid_json() {
        let schema = generate_schema();
        assert!(schema.is_object());
        let obj = schema.as_object().unwrap();
        assert!(obj.contains_key("$schema") || obj.contains_key("definitions") || obj.contains_key("properties"),
            "Schema should have standard JSON Schema keys");
    }

    /// Workspace root — two levels up from this crate's Cargo.toml.
    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent() // crates/
            .and_then(|p| p.parent()) // workspace root
            .expect("can't find workspace root")
            .to_path_buf()
    }

    #[test]
    fn schema_stays_in_sync() {
        let generated = serde_json::to_string_pretty(&generate_schema()).unwrap();
        let schema_path = workspace_root().join("organisms/organism.schema.json");
        let committed = std::fs::read_to_string(&schema_path)
            .unwrap_or_else(|e| panic!("{}: {e}\nRun: cargo test generate_schema_file -- --ignored", schema_path.display()));
        assert_eq!(
            generated.trim(),
            committed.trim(),
            "Schema drift detected! Regenerate with: cargo test generate_schema_file -- --ignored"
        );
    }

    #[test]
    #[ignore]
    fn generate_schema_file() {
        let schema = generate_schema();
        let pretty = serde_json::to_string_pretty(&schema).unwrap();
        let schema_path = workspace_root().join("organisms/organism.schema.json");
        std::fs::write(&schema_path, format!("{pretty}\n")).unwrap();
        println!("Wrote {}", schema_path.display());
    }

    #[test]
    fn existing_organisms_parse() {
        let root = workspace_root();
        for name in &["default.yaml", "infrastructure.yaml", "coder.yaml", "coder-v2.yaml", "organism-builder.yaml", "agent-expert.yaml", "wiki-expert.yaml", "plan-expert.yaml"] {
            let path = root.join("organisms").join(name);
            let content = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
            parse_organism(&content)
                .unwrap_or_else(|e| panic!("{name} failed to parse: {e}"));
        }
    }

    #[test]
    fn default_organism_loads_with_imports() {
        let root = workspace_root();
        let path = root.join("organisms").join("default.yaml");
        let org = load_organism(&path)
            .unwrap_or_else(|e| panic!("default.yaml failed to load: {e}"));
        assert_eq!(org.name, "bob");
        // Bob exists and has tools: auto
        let bob = org.get_listener("bob").expect("bob listener should exist");
        assert!(bob.tools_auto);
        assert!(bob.is_agent);
        // Infrastructure imported
        assert!(org.get_listener("file-read").is_some());
        assert!(org.get_listener("llm-pool").is_some());
        assert!(org.get_listener("grep").is_some());
        // Specialists declared locally
        assert!(org.get_listener("coder").is_some());
        assert!(org.get_listener("organism-builder").is_some());
        // Bob has auto-generated buffer (since we didn't declare one explicitly)
        assert!(bob.buffer.is_some());
    }

    // ── KV store config parsing ──

    #[test]
    fn kv_store_bool_true() {
        let yaml = "organism:\n  name: test\nkv-store: true\n";
        let org = parse_organism(yaml).unwrap();
        assert_eq!(org.kv_store, super::super::KvStoreConfig::Memory);
    }

    #[test]
    fn kv_store_bool_false() {
        let yaml = "organism:\n  name: test\nkv-store: false\n";
        let org = parse_organism(yaml).unwrap();
        assert_eq!(org.kv_store, super::super::KvStoreConfig::None);
    }

    #[test]
    fn kv_store_yes_string() {
        let yaml = "organism:\n  name: test\nkv-store: yes\n";
        let org = parse_organism(yaml).unwrap();
        assert_eq!(org.kv_store, super::super::KvStoreConfig::Memory);
    }

    #[test]
    fn kv_store_memory_string() {
        let yaml = "organism:\n  name: test\nkv-store: memory\n";
        let org = parse_organism(yaml).unwrap();
        assert_eq!(org.kv_store, super::super::KvStoreConfig::Memory);
    }

    #[test]
    fn kv_store_path_string() {
        let yaml = "organism:\n  name: test\nkv-store: /data/kv\n";
        let org = parse_organism(yaml).unwrap();
        assert_eq!(org.kv_store, super::super::KvStoreConfig::Disk("/data/kv".into()));
    }

    #[test]
    fn kv_store_windows_path() {
        let yaml = "organism:\n  name: test\nkv-store: 'C:\\src\\test'\n";
        let org = parse_organism(yaml).unwrap();
        assert_eq!(org.kv_store, super::super::KvStoreConfig::Disk("C:\\src\\test".into()));
    }

    #[test]
    fn kv_store_omitted_is_none() {
        let yaml = "organism:\n  name: test\n";
        let org = parse_organism(yaml).unwrap();
        assert_eq!(org.kv_store, super::super::KvStoreConfig::None);
    }

    #[test]
    fn kv_store_no_string() {
        let yaml = "organism:\n  name: test\nkv-store: no\n";
        let org = parse_organism(yaml).unwrap();
        assert_eq!(org.kv_store, super::super::KvStoreConfig::None);
    }

    // ── Import resolution tests ──

    /// Helper: write a YAML file and return its path.
    fn write_yaml(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn load_with_imports() {
        let dir = tempfile::TempDir::new().unwrap();

        write_yaml(dir.path(), "tools.yaml", r#"
organism:
  name: tools
listeners:
  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"
  - name: grep
    payload_class: tools.GrepRequest
    handler: tools.grep.handle
    description: "Grep search"
prompts:
  safety: "Be safe."
"#);

        let root = write_yaml(dir.path(), "root.yaml", r#"
organism:
  name: my-agent
imports:
  - tools.yaml
listeners:
  - name: planner
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Planner"
    agent: true
    tools: [file-read, grep]
prompts:
  planner_base: "You are a planner."
profiles:
  admin:
    linux_user: agentos
    listeners: all
    journal: retain_forever
"#);

        let org = load_organism(&root).unwrap();
        assert_eq!(org.name, "my-agent");
        // 3 listeners: planner (local) + file-read + grep (imported)
        assert_eq!(org.listener_names().len(), 3);
        assert!(org.get_listener("planner").is_some());
        assert!(org.get_listener("file-read").is_some());
        assert!(org.get_listener("grep").is_some());
        // 2 prompts: planner_base (local) + safety (imported)
        assert_eq!(org.prompts().len(), 2);
        assert!(org.get_prompt("planner_base").is_some());
        assert!(org.get_prompt("safety").is_some());
    }

    #[test]
    fn load_circular_import_detected() {
        let dir = tempfile::TempDir::new().unwrap();

        write_yaml(dir.path(), "a.yaml", r#"
organism:
  name: a
imports:
  - b.yaml
"#);
        write_yaml(dir.path(), "b.yaml", r#"
organism:
  name: b
imports:
  - a.yaml
"#);

        let err = load_organism(&dir.path().join("a.yaml")).unwrap_err();
        assert!(err.contains("circular import"), "expected circular import error, got: {err}");
    }

    #[test]
    fn load_diamond_import_deduplicates() {
        let dir = tempfile::TempDir::new().unwrap();

        // D: shared tools
        write_yaml(dir.path(), "d.yaml", r#"
organism:
  name: shared-tools
listeners:
  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"
"#);

        // B imports D
        write_yaml(dir.path(), "b.yaml", r#"
organism:
  name: b
imports:
  - d.yaml
listeners:
  - name: coder
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Coder"
    agent: true
"#);

        // C imports D
        write_yaml(dir.path(), "c.yaml", r#"
organism:
  name: c
imports:
  - d.yaml
listeners:
  - name: wiki
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Wiki"
    agent: true
"#);

        // A imports B and C (diamond via D)
        let root = write_yaml(dir.path(), "a.yaml", r#"
organism:
  name: root
imports:
  - b.yaml
  - c.yaml
"#);

        let org = load_organism(&root).unwrap();
        // file-read (from D, once), coder (from B), wiki (from C)
        assert_eq!(org.listener_names().len(), 3);
        assert!(org.get_listener("file-read").is_some());
        assert!(org.get_listener("coder").is_some());
        assert!(org.get_listener("wiki").is_some());
    }

    #[test]
    fn load_import_listener_conflict() {
        let dir = tempfile::TempDir::new().unwrap();

        write_yaml(dir.path(), "tools.yaml", r#"
organism:
  name: tools
listeners:
  - name: file-read
    payload_class: tools.FileReadRequest
    handler: OTHER.handler
    description: "Different handler"
"#);

        let root = write_yaml(dir.path(), "root.yaml", r#"
organism:
  name: root
imports:
  - tools.yaml
listeners:
  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"
"#);

        let err = load_organism(&root).unwrap_err();
        assert!(err.contains("conflict"), "expected conflict error, got: {err}");
    }

    #[test]
    fn load_import_dedup_identical_listeners() {
        let dir = tempfile::TempDir::new().unwrap();

        write_yaml(dir.path(), "tools.yaml", r#"
organism:
  name: tools
listeners:
  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"
"#);

        let root = write_yaml(dir.path(), "root.yaml", r#"
organism:
  name: root
imports:
  - tools.yaml
listeners:
  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"
"#);

        let org = load_organism(&root).unwrap();
        assert_eq!(org.listener_names().len(), 1);
    }

    #[test]
    fn load_no_imports_works() {
        let dir = tempfile::TempDir::new().unwrap();

        let root = write_yaml(dir.path(), "root.yaml", r#"
organism:
  name: simple
listeners:
  - name: echo
    payload_class: tools.EchoRequest
    handler: tools.echo.handle
    description: "Echo"
"#);

        let org = load_organism(&root).unwrap();
        assert_eq!(org.name, "simple");
        assert_eq!(org.listener_names().len(), 1);
    }

    #[test]
    fn load_nested_imports() {
        let dir = tempfile::TempDir::new().unwrap();

        write_yaml(dir.path(), "base-tools.yaml", r#"
organism:
  name: base
listeners:
  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"
"#);

        write_yaml(dir.path(), "extended-tools.yaml", r#"
organism:
  name: extended
imports:
  - base-tools.yaml
listeners:
  - name: grep
    payload_class: tools.GrepRequest
    handler: tools.grep.handle
    description: "Grep"
"#);

        let root = write_yaml(dir.path(), "root.yaml", r#"
organism:
  name: root
imports:
  - extended-tools.yaml
listeners:
  - name: planner
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Planner"
    agent: true
"#);

        let org = load_organism(&root).unwrap();
        // planner (local) + grep (extended-tools) + file-read (base-tools, nested)
        assert_eq!(org.listener_names().len(), 3);
        assert!(org.get_listener("planner").is_some());
        assert!(org.get_listener("grep").is_some());
        assert!(org.get_listener("file-read").is_some());
    }

    #[test]
    fn load_import_prompts_merged() {
        let dir = tempfile::TempDir::new().unwrap();

        write_yaml(dir.path(), "shared.yaml", r#"
organism:
  name: shared
prompts:
  safety: "Be safe."
  bounded: "Stay bounded."
"#);

        let root = write_yaml(dir.path(), "root.yaml", r#"
organism:
  name: root
imports:
  - shared.yaml
prompts:
  my_prompt: "You are an agent."
"#);

        let org = load_organism(&root).unwrap();
        assert_eq!(org.prompts().len(), 3);
        assert_eq!(org.get_prompt("safety"), Some("Be safe."));
        assert_eq!(org.get_prompt("bounded"), Some("Stay bounded."));
        assert_eq!(org.get_prompt("my_prompt"), Some("You are an agent."));
    }

    #[test]
    fn load_import_profiles_not_imported() {
        let dir = tempfile::TempDir::new().unwrap();

        write_yaml(dir.path(), "tools.yaml", r#"
organism:
  name: tools
listeners:
  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"
profiles:
  imported-profile:
    linux_user: someone
    listeners: [file-read]
    journal: retain_forever
"#);

        let root = write_yaml(dir.path(), "root.yaml", r#"
organism:
  name: root
imports:
  - tools.yaml
listeners:
  - name: planner
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Planner"
    agent: true
profiles:
  admin:
    linux_user: agentos
    listeners: [file-read, planner]
    journal: retain_forever
"#);

        let org = load_organism(&root).unwrap();
        // Only the root's profile, not the imported one
        assert!(org.get_profile("admin").is_some());
        assert!(org.get_profile("imported-profile").is_none());
    }

    // ── tools: auto parsing ──

    #[test]
    fn parse_tools_auto() {
        let yaml = r#"
organism:
  name: test-auto
listeners:
  - name: bob
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Bob"
    agent: true
    tools: auto
"#;
        let org = parse_organism(yaml).unwrap();
        let bob = org.get_listener("bob").unwrap();
        assert!(bob.tools_auto, "tools_auto should be true");
        assert!(bob.tools.is_empty(), "tools list should be empty (resolved at build time)");
    }

    #[test]
    fn parse_tools_explicit_list() {
        let yaml = r#"
organism:
  name: test-explicit
listeners:
  - name: coder
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Coder"
    agent: true
    tools: [file-read, grep]
"#;
        let org = parse_organism(yaml).unwrap();
        let coder = org.get_listener("coder").unwrap();
        assert!(!coder.tools_auto);
        assert_eq!(coder.tools, vec!["file-read", "grep"]);
    }

    #[test]
    fn parse_tools_default_empty() {
        let yaml = r#"
organism:
  name: test-default
listeners:
  - name: tool
    payload_class: tools.Request
    handler: tools.handle
    description: "A tool"
"#;
        let org = parse_organism(yaml).unwrap();
        let tool = org.get_listener("tool").unwrap();
        assert!(!tool.tools_auto);
        assert!(tool.tools.is_empty());
    }

    // ── Auto-generated default buffer for agents ──

    #[test]
    fn agent_without_buffer_gets_default() {
        let yaml = r#"
organism:
  name: test-default-buffer
listeners:
  - name: my-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "My agent"
    agent: true
"#;
        let org = parse_organism(yaml).unwrap();
        let agent = org.get_listener("my-agent").unwrap();
        assert!(agent.is_agent);

        // Should have auto-generated buffer
        let buf = agent.buffer.as_ref().expect("agent should have auto-generated buffer");
        assert_eq!(buf.description, "My agent");
        assert_eq!(buf.parameters.len(), 1);
        assert_eq!(buf.parameters[0].name, "task");
        assert_eq!(buf.parameters[0].param_type, "string");
        assert_eq!(buf.required, vec!["task"]);
        assert_eq!(buf.max_concurrency, 1);
        assert_eq!(buf.timeout_secs, 300);
    }

    #[test]
    fn agent_with_explicit_buffer_keeps_it() {
        let yaml = r#"
organism:
  name: test-explicit-buffer
listeners:
  - name: coder
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Coder"
    agent: true
    buffer:
      description: "Write code"
      parameters:
        plan: { type: string, description: "Implementation plan" }
      required: [plan]
      organism: coder.yaml
      max_concurrency: 3
"#;
        let org = parse_organism(yaml).unwrap();
        let coder = org.get_listener("coder").unwrap();
        let buf = coder.buffer.as_ref().unwrap();
        // Explicit values preserved
        assert_eq!(buf.description, "Write code");
        assert_eq!(buf.parameters.len(), 1);
        assert_eq!(buf.parameters[0].name, "plan");
        assert_eq!(buf.organism.as_deref(), Some("coder.yaml"));
        assert_eq!(buf.max_concurrency, 3);
    }

    #[test]
    fn non_agent_without_buffer_stays_none() {
        let yaml = r#"
organism:
  name: test-tool-no-buffer
listeners:
  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"
"#;
        let org = parse_organism(yaml).unwrap();
        let tool = org.get_listener("file-read").unwrap();
        assert!(!tool.is_agent);
        assert!(tool.buffer.is_none(), "non-agent should not get auto buffer");
    }
}
