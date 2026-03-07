//! System prompt templates for the coding agent.
//!
//! Three prompt modes:
//! - CODING_SYSTEM_PROMPT: Role, capabilities, tool descriptions
//! - PLANNING_PROMPT: Task decomposition instructions (Ralph Method)
//! - EXECUTION_PROMPT: Per-story execution instructions
//!
//! Plus: YAML-defined prompt resolution (labels, composition, interpolation).

use std::collections::HashMap;
use std::path::Path;

/// System prompt for the coding agent.
pub const CODING_SYSTEM_PROMPT: &str = "\
You are a coding agent running inside AgentOS. You have access to tools for file operations, \
shell commands, and codebase indexing. Use these tools to complete the task you've been given.

Rules:
1. Read before you write. Always understand existing code before modifying it.
2. Make the smallest change that solves the problem.
3. Test your changes when possible (run tests, verify output).
4. If a tool call fails, analyze the error and try a different approach.
5. When done, provide a clear summary of what you did.";

/// Planning prompt for task decomposition (Ralph Method).
pub const PLANNING_PROMPT: &str = "\
Decompose this task into atomic stories. Each story must:
1. Fit in one context window (under 8000 tokens of relevant code)
2. Have clear inputs and outputs
3. Be independently testable
4. Build on previous stories in sequence

Respond with a numbered list of stories. Each story should have:
- **Title**: One-line summary
- **Goal**: What this story accomplishes
- **Files**: Which files to read/modify
- **Test**: How to verify this story is done

Do NOT start executing yet. Just produce the plan.";

/// Execution prompt prefix for individual stories.
pub const EXECUTION_PROMPT: &str = "\
Execute this story from the plan. Focus only on this story's goal. \
Use the available tools to read files, make changes, and verify your work.";

/// Build the full system prompt with tool descriptions (legacy path).
pub fn build_system_prompt(tool_descriptions: &[(String, String)]) -> String {
    let mut prompt = CODING_SYSTEM_PROMPT.to_string();

    if !tool_descriptions.is_empty() {
        prompt.push_str("\n\nAvailable tools:\n");
        for (name, description) in tool_descriptions {
            prompt.push_str(&format!("- **{name}**: {description}\n"));
        }
    }

    prompt
}

/// Resolve a prompt specification into a final system prompt string.
///
/// - `None` → legacy `build_system_prompt()` (backward compat)
/// - `Some("label")` → lookup in registry, interpolate `{tool_definitions}`
/// - `Some("a & b")` → split on `&`, lookup each, join with `\n\n`, interpolate
pub fn resolve_prompt(
    spec: Option<&str>,
    registry: &HashMap<String, String>,
    tool_descriptions: &[(String, String)],
) -> Result<String, String> {
    match spec {
        None => Ok(build_system_prompt(tool_descriptions)),
        Some(label_spec) => {
            let composed = compose_labels(label_spec, registry)?;
            Ok(interpolate(&composed, tool_descriptions))
        }
    }
}

/// Compose one or more labels separated by `&` into a single prompt string.
fn compose_labels(spec: &str, registry: &HashMap<String, String>) -> Result<String, String> {
    let labels: Vec<&str> = spec.split('&').map(|s| s.trim()).collect();
    let mut parts = Vec::with_capacity(labels.len());

    for label in labels {
        let content = registry
            .get(label)
            .ok_or_else(|| format!("unknown prompt label: '{label}'"))?;
        parts.push(content.as_str());
    }

    Ok(parts.join("\n\n"))
}

/// Interpolate template variables in a prompt string.
///
/// Currently supports: `{tool_definitions}` → formatted tool list.
fn interpolate(template: &str, tool_descriptions: &[(String, String)]) -> String {
    if !template.contains("{tool_definitions}") {
        return template.to_string();
    }

    let mut tool_block = String::new();
    if !tool_descriptions.is_empty() {
        tool_block.push_str("Available tools:\n");
        for (name, description) in tool_descriptions {
            tool_block.push_str(&format!("- **{name}**: {description}\n"));
        }
    }

    template.replace("{tool_definitions}", &tool_block)
}

/// Load a prompt from a file path.
pub fn load_prompt_file(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path)
        .map_err(|e| format!("failed to load prompt file '{}': {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_is_not_empty() {
        assert!(!CODING_SYSTEM_PROMPT.is_empty());
    }

    #[test]
    fn planning_prompt_is_not_empty() {
        assert!(!PLANNING_PROMPT.is_empty());
    }

    #[test]
    fn build_system_prompt_includes_tools() {
        let tools = vec![
            ("file-ops".into(), "Read and write files".into()),
            ("shell".into(), "Execute commands".into()),
        ];
        let prompt = build_system_prompt(&tools);
        assert!(prompt.contains("file-ops"));
        assert!(prompt.contains("shell"));
        assert!(prompt.contains("Available tools"));
    }

    #[test]
    fn build_system_prompt_no_tools() {
        let prompt = build_system_prompt(&[]);
        assert!(!prompt.contains("Available tools"));
        assert!(prompt.contains("coding agent"));
    }

    // ── YAML-Defined Agents: prompt resolution tests ──

    fn test_registry() -> HashMap<String, String> {
        let mut reg = HashMap::new();
        reg.insert("base".into(), "You are a base agent.\n{tool_definitions}".into());
        reg.insert("safety".into(), "You are bounded. You do not pursue goals beyond your task.".into());
        reg.insert("plain".into(), "Just a plain prompt.".into());
        reg
    }

    fn test_tools() -> Vec<(String, String)> {
        vec![
            ("file-read".into(), "Read files".into()),
            ("bash".into(), "Run commands".into()),
        ]
    }

    #[test]
    fn resolve_single_label() {
        let reg = test_registry();
        let tools = test_tools();
        let result = resolve_prompt(Some("plain"), &reg, &tools).unwrap();
        assert_eq!(result, "Just a plain prompt.");
    }

    #[test]
    fn resolve_composed_labels() {
        let reg = test_registry();
        let tools = test_tools();
        let result = resolve_prompt(Some("safety & plain"), &reg, &tools).unwrap();
        assert!(result.contains("You are bounded"));
        assert!(result.contains("Just a plain prompt"));
        // Joined with double newline
        assert!(result.contains("\n\n"));
    }

    #[test]
    fn resolve_unknown_label_error() {
        let reg = test_registry();
        let tools = test_tools();
        let err = resolve_prompt(Some("nonexistent"), &reg, &tools).unwrap_err();
        assert!(err.contains("unknown prompt label"));
        assert!(err.contains("nonexistent"));
    }

    #[test]
    fn interpolate_tool_definitions() {
        let reg = test_registry();
        let tools = test_tools();
        let result = resolve_prompt(Some("base"), &reg, &tools).unwrap();
        assert!(result.contains("file-read"));
        assert!(result.contains("bash"));
        assert!(result.contains("Available tools"));
        assert!(!result.contains("{tool_definitions}"));
    }

    #[test]
    fn interpolate_empty_tools() {
        let reg = test_registry();
        let result = resolve_prompt(Some("base"), &reg, &[]).unwrap();
        assert!(!result.contains("{tool_definitions}"));
        // With no tools, the tool block is empty
        assert!(!result.contains("Available tools"));
    }

    #[test]
    fn resolve_none_legacy_fallback() {
        let tools = test_tools();
        let result = resolve_prompt(None, &HashMap::new(), &tools).unwrap();
        // Should produce the legacy prompt
        assert!(result.contains("coding agent"));
        assert!(result.contains("file-read"));
    }

    #[test]
    fn load_prompt_file_ok() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.md");
        std::fs::write(&path, "This is a test prompt.").unwrap();

        let content = load_prompt_file(&path).unwrap();
        assert_eq!(content, "This is a test prompt.");
    }

    #[test]
    fn load_prompt_file_missing() {
        let err = load_prompt_file(Path::new("/nonexistent/prompt.md")).unwrap_err();
        assert!(err.contains("failed to load prompt file"));
    }

    #[test]
    fn compose_trims_whitespace() {
        let reg = test_registry();
        // Extra spaces around &
        let result = compose_labels("safety  &  plain", &reg).unwrap();
        assert!(result.contains("You are bounded"));
        assert!(result.contains("Just a plain prompt"));
    }
}
