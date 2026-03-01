//! Language service for organism YAML files.
//!
//! Provides diagnostics (syntax + schema + cross-ref), context-aware
//! completions, and hover information — all as pure functions.

use lsp_types::{
    CompletionItem, CompletionItemKind, Diagnostic, DiagnosticSeverity, Position, Range,
};
use serde_yaml::Value;

use super::{HoverInfo, LanguageService};

/// Language service for organism YAML configuration files.
#[derive(Default)]
pub struct OrganismYamlService;

impl OrganismYamlService {
    pub fn new() -> Self {
        Self
    }
}

impl LanguageService for OrganismYamlService {
    fn diagnostics(&self, content: &str) -> Vec<Diagnostic> {
        let mut diags = Vec::new();

        // Phase 1: YAML syntax check
        let value: Value = match serde_yaml::from_str(content) {
            Ok(v) => v,
            Err(e) => {
                let loc = e.location();
                let (line, col) = loc
                    .map(|l| (l.line() as u32, l.column() as u32))
                    .unwrap_or((0, 0));
                diags.push(Diagnostic {
                    range: Range::new(Position::new(line, col), Position::new(line, col + 1)),
                    severity: Some(DiagnosticSeverity::ERROR),
                    message: format!("YAML syntax error: {e}"),
                    ..Default::default()
                });
                return diags;
            }
        };

        // Phase 2: Schema validation
        let Some(root) = value.as_mapping() else {
            diags.push(make_diag(0, 0, "Root must be a YAML mapping", DiagnosticSeverity::ERROR));
            return diags;
        };

        // Check top-level keys
        let valid_top = ["organism", "listeners", "profiles", "prompts"];
        for (key, _) in root {
            if let Some(name) = key.as_str() {
                if !valid_top.contains(&name) {
                    let line = find_key_line(content, name, 0);
                    diags.push(make_diag(
                        line,
                        0,
                        &format!("Unknown top-level key: '{name}'"),
                        DiagnosticSeverity::WARNING,
                    ));
                }
            }
        }

        // organism: section
        if let Some(org) = root.get("organism") {
            validate_organism_meta(content, org, &mut diags);
        } else {
            diags.push(make_diag(0, 0, "Missing required top-level key: 'organism'", DiagnosticSeverity::ERROR));
        }

        // Collect listener names and prompt labels for cross-reference checks
        let listener_names = collect_listener_names(root);
        let prompt_labels = collect_prompt_labels(root);

        // listeners: section
        if let Some(listeners) = root.get("listeners") {
            validate_listeners(content, listeners, &prompt_labels, &mut diags);
        }

        // profiles: section
        if let Some(profiles) = root.get("profiles") {
            validate_profiles(content, profiles, &listener_names, &mut diags);
        }

        diags
    }

    fn completions(&self, content: &str, pos: Position) -> Vec<CompletionItem> {
        let lines: Vec<&str> = content.lines().collect();
        let line_idx = pos.line as usize;
        if line_idx >= lines.len() {
            return Vec::new();
        }
        let line = lines[line_idx];
        let col = pos.character as usize;

        // Determine indent level (number of leading spaces)
        let indent = line.len() - line.trim_start().len();
        let before_cursor = &line[..col.min(line.len())];
        let trimmed = before_cursor.trim();

        // Parse the document for cross-reference completions
        let value: Option<Value> = serde_yaml::from_str(content).ok();

        // Context: what section are we in?
        let context = determine_context(lines.as_slice(), line_idx);

        match context {
            Context::TopLevel => {
                complete_keys(&["organism", "listeners", "profiles", "prompts"], trimmed)
            }
            Context::Organism => {
                complete_keys(&["name"], trimmed)
            }
            Context::ListenerItem => {
                complete_keys(
                    &[
                        "name", "payload_class", "handler", "description", "agent",
                        "peers", "model", "ports", "librarian", "wasm",
                        "semantic_description", "buffer",
                    ],
                    trimmed,
                )
            }
            Context::AgentBlock => {
                complete_keys(
                    &["prompt", "max_tokens", "max_iterations", "max_agentic_iterations", "model", "permissions"],
                    trimmed,
                )
            }
            Context::BufferBlock => {
                complete_keys(
                    &["description", "parameters", "required", "requires", "organism", "max_concurrency", "timeout_secs"],
                    trimmed,
                )
            }
            Context::Profile => {
                complete_keys(&["linux_user", "listeners", "journal", "network"], trimmed)
            }
            Context::PortItem => {
                complete_keys(&["port", "direction", "protocol", "hosts"], trimmed)
            }
            Context::WasmBlock => {
                complete_keys(&["path", "capabilities"], trimmed)
            }
            // Value completions — prefix is text after the colon
            Context::ValueOf(field) => {
                let value_prefix = trimmed
                    .find(':')
                    .map(|p| trimmed[p + 1..].trim())
                    .unwrap_or("");
                complete_value(&field, value_prefix, indent, &value)
            }
            Context::Unknown => Vec::new(),
        }
    }

    fn hover(&self, content: &str, pos: Position) -> Option<HoverInfo> {
        let lines: Vec<&str> = content.lines().collect();
        let line_idx = pos.line as usize;
        if line_idx >= lines.len() {
            return None;
        }
        let line = lines[line_idx];
        let col = pos.character as usize;

        // Find the word under the cursor
        let word = word_at(line, col)?;

        // Check if it's a key (before ':')
        if let Some(colon_pos) = line.find(':') {
            let key = line[..colon_pos].trim();
            if key == word {
                return hover_for_key(word);
            }
        }

        // Check if it's a value reference (listener name, prompt label, etc.)
        let value: Value = serde_yaml::from_str(content).ok()?;
        let root = value.as_mapping()?;

        // Check if word matches a listener name
        let listener_names = collect_listener_names(root);
        if listener_names.contains(&word.to_string()) {
            if let Some(desc) = find_listener_description(root, word) {
                return Some(HoverInfo {
                    content: format!("**Listener:** {word}\n\n{desc}"),
                    range: None,
                });
            }
        }

        // Check if word matches a prompt label
        let prompt_labels = collect_prompt_labels(root);
        if prompt_labels.contains(&word.to_string()) {
            if let Some(text) = find_prompt_text(root, word) {
                let preview: String = text.lines().take(3).collect::<Vec<_>>().join("\n");
                let suffix = if text.lines().count() > 3 { "\n..." } else { "" };
                return Some(HoverInfo {
                    content: format!("**Prompt:** {word}\n\n```\n{preview}{suffix}\n```"),
                    range: None,
                });
            }
        }

        None
    }
}

// ── Diagnostic helpers ──

fn make_diag(line: u32, col: u32, message: &str, severity: DiagnosticSeverity) -> Diagnostic {
    Diagnostic {
        range: Range::new(
            Position::new(line, col),
            Position::new(line, col + 1),
        ),
        severity: Some(severity),
        message: message.to_string(),
        ..Default::default()
    }
}

/// Find the line number where a key first appears at a given indent level (approx).
fn find_key_line(content: &str, key: &str, _min_indent: usize) -> u32 {
    let needle = format!("{key}:");
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with(&needle) || trimmed.starts_with(&format!("- {needle}")) {
            return i as u32;
        }
    }
    0
}

fn validate_organism_meta(content: &str, value: &Value, diags: &mut Vec<Diagnostic>) {
    let Some(map) = value.as_mapping() else {
        let line = find_key_line(content, "organism", 0);
        diags.push(make_diag(line, 0, "'organism' must be a mapping", DiagnosticSeverity::ERROR));
        return;
    };

    if map.get("name").is_none() {
        let line = find_key_line(content, "organism", 0);
        diags.push(make_diag(
            line, 0,
            "Missing required field: 'organism.name'",
            DiagnosticSeverity::ERROR,
        ));
    }

    let valid = ["name"];
    for (key, _) in map {
        if let Some(name) = key.as_str() {
            if !valid.contains(&name) {
                let line = find_key_line(content, name, 2);
                diags.push(make_diag(
                    line, 0,
                    &format!("Unknown field in organism: '{name}'"),
                    DiagnosticSeverity::WARNING,
                ));
            }
        }
    }
}

fn validate_listeners(
    content: &str,
    value: &Value,
    prompt_labels: &[String],
    diags: &mut Vec<Diagnostic>,
) {
    let Some(list) = value.as_sequence() else {
        let line = find_key_line(content, "listeners", 0);
        diags.push(make_diag(line, 0, "'listeners' must be a sequence", DiagnosticSeverity::ERROR));
        return;
    };

    let required_fields = ["name", "payload_class", "handler", "description"];

    // Collect all listener names for peer cross-ref
    let all_names: Vec<String> = list
        .iter()
        .filter_map(|item| {
            item.as_mapping()
                .and_then(|m| m.get("name"))
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .collect();

    for item in list {
        let Some(map) = item.as_mapping() else {
            diags.push(make_diag(0, 0, "Listener must be a mapping", DiagnosticSeverity::ERROR));
            continue;
        };

        let listener_name = map
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("<unnamed>");

        // Required fields
        for field in &required_fields {
            if map.get(*field).is_none() {
                let line = find_listener_line(content, listener_name);
                diags.push(make_diag(
                    line, 0,
                    &format!("Listener '{listener_name}' missing required field: '{field}'"),
                    DiagnosticSeverity::ERROR,
                ));
            }
        }

        // Unknown fields
        let valid_fields = [
            "name", "payload_class", "handler", "description", "agent", "is_agent",
            "peers", "model", "ports", "librarian", "wasm", "semantic_description",
            "buffer",
        ];
        for (key, _) in map {
            if let Some(name) = key.as_str() {
                if !valid_fields.contains(&name) {
                    let line = find_key_line(content, name, 4);
                    diags.push(make_diag(
                        line, 0,
                        &format!("Unknown listener field: '{name}'"),
                        DiagnosticSeverity::WARNING,
                    ));
                }
            }
        }

        // Type checks
        if let Some(max_tokens) = map.get("max_tokens") {
            if !max_tokens.is_u64() && !max_tokens.is_i64() {
                let line = find_key_line(content, "max_tokens", 4);
                diags.push(make_diag(
                    line, 0,
                    "max_tokens must be a number",
                    DiagnosticSeverity::ERROR,
                ));
            }
        }

        // Peer cross-references
        if let Some(peers) = map.get("peers") {
            if let Some(peer_list) = peers.as_sequence() {
                for peer in peer_list {
                    if let Some(peer_name) = peer.as_str() {
                        if !all_names.contains(&peer_name.to_string()) {
                            let line = find_key_line(content, "peers", 4);
                            diags.push(make_diag(
                                line, 0,
                                &format!(
                                    "Listener '{listener_name}' references unknown peer: '{peer_name}'"
                                ),
                                DiagnosticSeverity::ERROR,
                            ));
                        }
                    }
                }
            }
        }

        // Agent prompt cross-reference
        if let Some(agent) = map.get("agent") {
            if let Some(agent_map) = agent.as_mapping() {
                if let Some(prompt_val) = agent_map.get("prompt") {
                    if let Some(prompt_str) = prompt_val.as_str() {
                        // Prompt composition: "a & b" → check each label
                        for label in prompt_str.split('&').map(str::trim) {
                            if !label.is_empty() && !prompt_labels.contains(&label.to_string()) {
                                let line = find_key_line(content, "prompt", 6);
                                diags.push(make_diag(
                                    line, 0,
                                    &format!("Agent references unknown prompt label: '{label}'"),
                                    DiagnosticSeverity::ERROR,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
}

fn validate_profiles(
    content: &str,
    value: &Value,
    listener_names: &[String],
    diags: &mut Vec<Diagnostic>,
) {
    let Some(map) = value.as_mapping() else {
        let line = find_key_line(content, "profiles", 0);
        diags.push(make_diag(line, 0, "'profiles' must be a mapping", DiagnosticSeverity::ERROR));
        return;
    };

    let valid_fields = ["linux_user", "listeners", "journal", "network"];

    for (key, profile) in map {
        let profile_name = key.as_str().unwrap_or("<unnamed>");
        let Some(profile_map) = profile.as_mapping() else {
            diags.push(make_diag(0, 0, &format!("Profile '{profile_name}' must be a mapping"), DiagnosticSeverity::ERROR));
            continue;
        };

        // Required field: linux_user
        if profile_map.get("linux_user").is_none() {
            let line = find_key_line(content, profile_name, 2);
            diags.push(make_diag(
                line, 0,
                &format!("Profile '{profile_name}' missing required field: 'linux_user'"),
                DiagnosticSeverity::ERROR,
            ));
        }

        // Unknown fields
        for (field_key, _) in profile_map {
            if let Some(name) = field_key.as_str() {
                if !valid_fields.contains(&name) {
                    let line = find_key_line(content, name, 4);
                    diags.push(make_diag(
                        line, 0,
                        &format!("Unknown profile field: '{name}'"),
                        DiagnosticSeverity::WARNING,
                    ));
                }
            }
        }

        // Cross-ref: listeners list
        if let Some(listeners) = profile_map.get("listeners") {
            if let Some(list) = listeners.as_sequence() {
                for item in list {
                    if let Some(name) = item.as_str() {
                        if name != "all" && !listener_names.contains(&name.to_string()) {
                            let line = find_key_line(content, "listeners", 4);
                            diags.push(make_diag(
                                line, 0,
                                &format!(
                                    "Profile '{profile_name}' references unknown listener: '{name}'"
                                ),
                                DiagnosticSeverity::ERROR,
                            ));
                        }
                    }
                }
            }
        }

        // Cross-ref: network list
        if let Some(network) = profile_map.get("network") {
            if let Some(list) = network.as_sequence() {
                for item in list {
                    if let Some(name) = item.as_str() {
                        if !listener_names.contains(&name.to_string()) {
                            let line = find_key_line(content, "network", 4);
                            diags.push(make_diag(
                                line, 0,
                                &format!(
                                    "Profile '{profile_name}' network references unknown listener: '{name}'"
                                ),
                                DiagnosticSeverity::ERROR,
                            ));
                        }
                    }
                }
            }
        }
    }
}

fn find_listener_line(content: &str, name: &str) -> u32 {
    let needle = format!("name: {name}");
    let needle_quoted = format!("name: \"{name}\"");
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.contains(&needle) || trimmed.contains(&needle_quoted) {
            return i as u32;
        }
    }
    0
}

fn collect_listener_names(root: &serde_yaml::Mapping) -> Vec<String> {
    root.get("listeners")
        .and_then(|v| v.as_sequence())
        .map(|list| {
            list.iter()
                .filter_map(|item| {
                    item.as_mapping()
                        .and_then(|m| m.get("name"))
                        .and_then(|v| v.as_str())
                        .map(String::from)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn collect_prompt_labels(root: &serde_yaml::Mapping) -> Vec<String> {
    root.get("prompts")
        .and_then(|v| v.as_mapping())
        .map(|map| {
            map.keys()
                .filter_map(|k| k.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

fn find_listener_description(root: &serde_yaml::Mapping, name: &str) -> Option<String> {
    root.get("listeners")?
        .as_sequence()?
        .iter()
        .find_map(|item| {
            let map = item.as_mapping()?;
            let n = map.get("name")?.as_str()?;
            if n == name {
                map.get("description")?.as_str().map(String::from)
            } else {
                None
            }
        })
}

fn find_prompt_text(root: &serde_yaml::Mapping, label: &str) -> Option<String> {
    root.get("prompts")?
        .as_mapping()?
        .get(label)?
        .as_str()
        .map(String::from)
}

// ── Completion helpers ──

/// Contextual position within the YAML document.
enum Context {
    TopLevel,
    Organism,
    ListenerItem,
    AgentBlock,
    BufferBlock,
    Profile,
    PortItem,
    WasmBlock,
    ValueOf(String),
    Unknown,
}

/// Determine the editing context based on indentation and surrounding keys.
fn determine_context(lines: &[&str], cursor_line: usize) -> Context {
    let line = lines.get(cursor_line).copied().unwrap_or("");
    let indent = line.len() - line.trim_start().len();

    // Check if cursor is on a value position (after "key: ")
    let trimmed = line.trim();
    if let Some(colon_pos) = trimmed.find(':') {
        let key = trimmed[..colon_pos].trim_start_matches("- ").trim();
        let after = trimmed[colon_pos + 1..].trim();
        if after.is_empty() || !after.contains(':') {
            match key {
                "model" | "journal" | "handler" | "direction" | "protocol" | "librarian" => {
                    return Context::ValueOf(key.to_string());
                }
                _ => {}
            }
        }
    }

    // Walk backwards to find parent context
    if indent == 0 {
        return Context::TopLevel;
    }

    for i in (0..cursor_line).rev() {
        let prev = lines[i];
        let prev_indent = prev.len() - prev.trim_start().len();
        let prev_trimmed = prev.trim();

        if prev_indent < indent {
            // This is a parent — what key is it?
            let parent_key = prev_trimmed
                .split(':')
                .next()
                .unwrap_or("")
                .trim_start_matches("- ")
                .trim();

            match parent_key {
                "organism" => return Context::Organism,
                "agent" | "is_agent" => return Context::AgentBlock,
                "buffer" => return Context::BufferBlock,
                "ports" => return Context::PortItem,
                "wasm" => return Context::WasmBlock,
                _ => {}
            }

            // Check if we're inside a listener item (parent is `listeners:` or `- name:`)
            if parent_key == "listeners" {
                return Context::ListenerItem;
            }
            if parent_key == "profiles" || prev_indent == 2 {
                // Check if grandparent is profiles
                for j in (0..i).rev() {
                    let gp = lines[j].trim();
                    if gp.starts_with("profiles:") {
                        return Context::Profile;
                    }
                    if !gp.is_empty() && lines[j].len() - gp.len() < prev_indent {
                        break;
                    }
                }
            }

            // Inside a listener item? Check if any ancestor is a `- name:` pattern under listeners
            if prev_trimmed.starts_with("- name:") || prev_trimmed.starts_with("- ") {
                // Check if grandparent is listeners
                for j in (0..i).rev() {
                    let gp = lines[j].trim();
                    if gp.starts_with("listeners:") {
                        return Context::ListenerItem;
                    }
                    let gp_indent = lines[j].len() - gp.len();
                    if gp_indent == 0 && !gp.is_empty() {
                        break;
                    }
                }
            }

            // Profile block?
            if prev_indent == 2 {
                for j in (0..i).rev() {
                    let gp = lines[j].trim();
                    if gp.starts_with("profiles:") {
                        return Context::Profile;
                    }
                    if lines[j].len() - gp.len() == 0 && !gp.is_empty() {
                        break;
                    }
                }
            }

            return Context::Unknown;
        }
    }

    Context::Unknown
}

fn complete_keys(keys: &[&str], prefix: &str) -> Vec<CompletionItem> {
    let filter = prefix.trim_start_matches("- ");
    keys.iter()
        .filter(|k| filter.is_empty() || k.starts_with(filter))
        .map(|k| CompletionItem {
            label: format!("{k}:"),
            kind: Some(CompletionItemKind::FIELD),
            detail: Some("organism YAML key".to_string()),
            insert_text: Some(format!("{k}: ")),
            ..Default::default()
        })
        .collect()
}

fn complete_value(
    field: &str,
    prefix: &str,
    _indent: usize,
    doc: &Option<Value>,
) -> Vec<CompletionItem> {
    let values: Vec<&str> = match field {
        "model" => vec!["opus", "sonnet", "haiku"],
        "journal" => vec!["retain_forever", "prune_on_delivery"],
        "handler" => vec!["wasm", "buffer"],
        "direction" => vec!["inbound", "outbound"],
        "protocol" => vec!["https", "http", "ssh"],
        "librarian" => vec!["true", "false"],
        _ => return Vec::new(),
    };

    let mut items: Vec<CompletionItem> = values
        .into_iter()
        .filter(|v| prefix.is_empty() || v.starts_with(prefix))
        .map(|v| CompletionItem {
            label: v.to_string(),
            kind: Some(CompletionItemKind::VALUE),
            ..Default::default()
        })
        .collect();

    // For peer/listeners/prompt/network, add document-aware completions
    if let Some(doc) = doc {
        if let Some(root) = doc.as_mapping() {
            match field {
                "peers" | "listeners" | "network" => {
                    let names = collect_listener_names(root);
                    for name in names {
                        if prefix.is_empty() || name.starts_with(prefix) {
                            items.push(CompletionItem {
                                label: name,
                                kind: Some(CompletionItemKind::REFERENCE),
                                ..Default::default()
                            });
                        }
                    }
                }
                "prompt" => {
                    let labels = collect_prompt_labels(root);
                    for label in labels {
                        if prefix.is_empty() || label.starts_with(prefix) {
                            items.push(CompletionItem {
                                label,
                                kind: Some(CompletionItemKind::REFERENCE),
                                ..Default::default()
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }

    items
}

// ── Hover helpers ──

fn hover_for_key(key: &str) -> Option<HoverInfo> {
    let desc = match key {
        "organism" => "Top-level: organism metadata block. Inside `buffer:`: child organism YAML file (relative to `--dir`). Omit to clone current organism.",
        "name" => "Unique name for this organism configuration. *Required.*",
        "listeners" => "Array of listener definitions — each handles one payload type.",
        "profiles" => "Security profiles — map of profile name to access rules.",
        "prompts" => "Named prompt templates for agent identity. Values can be inline or `file:path`.",
        "payload_class" => "Fully qualified payload class (e.g., `agent.AgentTask`). Last component becomes the payload tag.",
        "handler" => "Handler function name (e.g., `tools.file_ops.handle`), or `wasm` for WASM tools.",
        "description" => "Human-readable description of this listener's purpose.",
        "agent" => "Agent configuration block — `true` for defaults, or `{ prompt, max_tokens, max_iterations, model }`.",
        "is_agent" => "Alias for `agent`. Boolean or configuration block.",
        "peers" => "List of listener names this listener may call (dispatch table entries).",
        "model" => "LLM model override — `opus`, `sonnet`, or `haiku`. Default: pool default.",
        "ports" => "Network port declarations — `{ port, direction, protocol, hosts }`.",
        "librarian" => "`true` to auto-curate context via Haiku librarian. Default: `false`.",
        "wasm" => "WASM tool configuration — `{ path, capabilities }`.",
        "semantic_description" => "Natural language description for embedding-based semantic routing.",
        "prompt" => "Prompt label(s). Use `&` to compose: `\"safety & coding_base\"`. Labels must exist in `prompts:` section.",
        "max_tokens" => "Maximum LLM completion tokens. Default: `4096`.",
        "max_iterations" => "Maximum agentic loop iterations. Default: `5`.",
        "linux_user" => "Linux user for process isolation (e.g., `agentos-root`). *Required.*",
        "journal" => "Message retention policy — `retain_forever`, `prune_on_delivery`, or `{ retain_days: N }`.",
        "network" => "List of listener names whose network ports are accessible to this profile.",
        "port" => "Port number (u16).",
        "direction" => "`inbound` or `outbound`.",
        "protocol" => "Network protocol — `https`, `http`, `ssh`, etc.",
        "hosts" => "Target hosts for outbound connections (e.g., `[\"api.anthropic.com\"]`).",
        "path" => "Path to the WASM binary.",
        "capabilities" => "WASM sandbox capabilities — `{ filesystem, env, stdio }`.",
        "buffer" => "Buffer node — callable tool interface + child pipeline spawn config.",
        "max_agentic_iterations" => "Maximum tool-call loop iterations. Default: `25`.",
        "permissions" => "Per-tool permission tiers: `auto`, `prompt` (default), or `deny`.",
        "required" => "List of mandatory parameter names for the buffer tool interface.",
        "requires" => "Tools available inside the child pipeline (e.g., `[file-read, command-exec]`).",
        "max_concurrency" => "Maximum parallel child instances. Default: `5`.",
        "timeout_secs" => "Execution timeout in seconds. Default: `300`.",
        _ => return None,
    };

    Some(HoverInfo {
        content: format!("**{key}**\n\n{desc}"),
        range: None,
    })
}

fn word_at(line: &str, col: usize) -> Option<&str> {
    if col > line.len() {
        return None;
    }
    let bytes = line.as_bytes();

    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'-';

    let mut start = col;
    while start > 0 && is_word(bytes[start - 1]) {
        start -= 1;
    }

    let mut end = col;
    while end < bytes.len() && is_word(bytes[end]) {
        end += 1;
    }

    if start == end {
        return None;
    }

    Some(&line[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn svc() -> OrganismYamlService {
        OrganismYamlService::new()
    }

    // ── Diagnostics ──

    #[test]
    fn diagnostics_valid_yaml_no_errors() {
        let yaml = r#"
organism:
  name: test
listeners:
  - name: echo
    payload_class: tools.EchoRequest
    handler: tools.echo.handle
    description: "Echo handler"
profiles:
  root:
    linux_user: agentos-root
    listeners: all
"#;
        let diags = svc().diagnostics(yaml);
        assert!(diags.is_empty(), "Expected no diagnostics, got: {diags:?}");
    }

    #[test]
    fn diagnostics_syntax_error_has_position() {
        let yaml = "key: [invalid\n";
        let diags = svc().diagnostics(yaml);
        assert!(!diags.is_empty());
        assert!(diags[0].message.contains("syntax"));
        assert_eq!(diags[0].severity, Some(DiagnosticSeverity::ERROR));
    }

    #[test]
    fn diagnostics_missing_organism() {
        let yaml = "listeners: []\n";
        let diags = svc().diagnostics(yaml);
        assert!(diags.iter().any(|d| d.message.contains("organism")));
    }

    #[test]
    fn diagnostics_missing_required_field() {
        let yaml = r#"
organism:
  name: test
listeners:
  - name: echo
    handler: tools.echo.handle
"#;
        let diags = svc().diagnostics(yaml);
        // Missing payload_class and description
        assert!(diags.iter().any(|d| d.message.contains("payload_class")));
        assert!(diags.iter().any(|d| d.message.contains("description")));
    }

    #[test]
    fn diagnostics_unknown_top_level_key() {
        let yaml = r#"
organism:
  name: test
bogus_key: value
"#;
        let diags = svc().diagnostics(yaml);
        assert!(diags.iter().any(|d| d.message.contains("bogus_key")));
    }

    #[test]
    fn diagnostics_dangling_peer_reference() {
        let yaml = r#"
organism:
  name: test
listeners:
  - name: agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Agent"
    peers: [nonexistent]
"#;
        let diags = svc().diagnostics(yaml);
        assert!(diags.iter().any(|d| d.message.contains("nonexistent")));
    }

    #[test]
    fn diagnostics_dangling_prompt_label() {
        let yaml = r#"
organism:
  name: test
prompts:
  safety: "Be safe."
listeners:
  - name: agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Agent"
    agent:
      prompt: "safety & missing_prompt"
"#;
        let diags = svc().diagnostics(yaml);
        assert!(diags.iter().any(|d| d.message.contains("missing_prompt")));
    }

    #[test]
    fn diagnostics_profile_unknown_listener() {
        let yaml = r#"
organism:
  name: test
listeners:
  - name: echo
    payload_class: tools.EchoRequest
    handler: tools.echo.handle
    description: "Echo"
profiles:
  admin:
    linux_user: agentos-admin
    listeners: [echo, ghost]
"#;
        let diags = svc().diagnostics(yaml);
        assert!(diags.iter().any(|d| d.message.contains("ghost")));
    }

    // ── Completions ──

    #[test]
    fn completions_top_level_keys() {
        let yaml = "\n";
        let items = svc().completions(yaml, Position::new(0, 0));
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"organism:"));
        assert!(labels.contains(&"listeners:"));
        assert!(labels.contains(&"profiles:"));
        assert!(labels.contains(&"prompts:"));
    }

    #[test]
    fn completions_listener_fields() {
        let yaml = "listeners:\n  - name: foo\n    \n";
        let items = svc().completions(yaml, Position::new(2, 4));
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"handler:"), "Expected handler: in {labels:?}");
        assert!(labels.contains(&"description:"), "Expected description: in {labels:?}");
    }

    #[test]
    fn completions_model_values() {
        let yaml = "listeners:\n  - name: foo\n    model: \n";
        let items = svc().completions(yaml, Position::new(2, 11));
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"opus"));
        assert!(labels.contains(&"sonnet"));
        assert!(labels.contains(&"haiku"));
    }

    // ── Hover ──

    #[test]
    fn hover_on_field_name() {
        let yaml = "organism:\n  name: test\n";
        let info = svc().hover(yaml, Position::new(0, 2));
        assert!(info.is_some());
        assert!(info.unwrap().content.contains("organism"));
    }

    #[test]
    fn hover_on_listener_reference() {
        let yaml = r#"
organism:
  name: test
listeners:
  - name: echo
    payload_class: tools.EchoRequest
    handler: tools.echo.handle
    description: "Echo tool"
    peers: [echo]
"#;
        let info = svc().hover(yaml, Position::new(8, 13));
        assert!(info.is_some());
        let content = info.unwrap().content;
        assert!(content.contains("echo") || content.contains("Echo"));
    }

    #[test]
    fn hover_on_prompt_label() {
        let yaml = r#"
organism:
  name: test
prompts:
  safety: |
    You are bounded.
    You do not pursue goals beyond your task.
    Always verify before acting.
listeners:
  - name: agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Agent"
    agent:
      prompt: safety
"#;
        let info = svc().hover(yaml, Position::new(14, 14));
        assert!(info.is_some());
        let content = info.unwrap().content;
        assert!(content.contains("bounded"), "Expected prompt preview, got: {content}");
    }

    // ── Helpers ──

    #[test]
    fn word_at_extracts_correctly() {
        assert_eq!(word_at("  name: echo", 8), Some("echo"));
        assert_eq!(word_at("  name: echo", 2), Some("name"));
        assert_eq!(word_at("  : ", 2), None);
    }
}
