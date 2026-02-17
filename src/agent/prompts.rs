//! System prompt templates for the coding agent.
//!
//! Three prompt modes:
//! - CODING_SYSTEM_PROMPT: Role, capabilities, tool descriptions
//! - PLANNING_PROMPT: Task decomposition instructions (Ralph Method)
//! - EXECUTION_PROMPT: Per-story execution instructions

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

/// Build the full system prompt with tool descriptions.
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
}
