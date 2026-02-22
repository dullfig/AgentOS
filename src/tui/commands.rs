//! Slash commands for the TUI input bar.
//!
//! Commands start with `/` and are intercepted before reaching the pipeline.
//! Ghost-text autocomplete suggests commands as you type.

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::llm::LlmPool;
use crate::llm::types::resolve_model;

use super::app::{ChatEntry, TuiApp};

/// A slash command definition.
pub struct SlashCommand {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub description: &'static str,
    pub has_arg: bool,
}

/// All known commands.
pub static COMMANDS: &[SlashCommand] = &[
    SlashCommand {
        name: "/exit",
        aliases: &["/quit"],
        description: "Quit BestCode",
        has_arg: false,
    },
    SlashCommand {
        name: "/clear",
        aliases: &[],
        description: "Clear chat log",
        has_arg: false,
    },
    SlashCommand {
        name: "/model",
        aliases: &[],
        description: "Switch default model (opus/sonnet/haiku or full ID)",
        has_arg: true,
    },
    SlashCommand {
        name: "/help",
        aliases: &[],
        description: "List available commands",
        has_arg: false,
    },
];

/// Find the best prefix-matching command for the given input.
/// Input must start with `/`. Returns None if no match.
pub fn suggest(input: &str) -> Option<&'static SlashCommand> {
    if !input.starts_with('/') {
        return None;
    }

    // Extract just the command part (before any space)
    let cmd_part = input.split_whitespace().next().unwrap_or(input);

    // Exact match on name or alias (with possible trailing space/args)
    for cmd in COMMANDS {
        if cmd.name == cmd_part || cmd.aliases.iter().any(|a| *a == cmd_part) {
            return Some(cmd);
        }
    }

    // Prefix match on name
    for cmd in COMMANDS {
        if cmd.name.starts_with(cmd_part) {
            return Some(cmd);
        }
    }

    // Prefix match on aliases
    for cmd in COMMANDS {
        for alias in cmd.aliases {
            if alias.starts_with(cmd_part) {
                return Some(cmd);
            }
        }
    }

    None
}

/// Compute the ghost suffix to overlay after the cursor.
/// Returns the remaining characters of the matched command + trailing space if it takes args.
pub fn ghost_suffix(input: &str) -> Option<String> {
    if !input.starts_with('/') || input.contains(' ') {
        return None;
    }

    let cmd = suggest(input)?;

    // Check if an alias matches better
    let matched_name = if cmd.name.starts_with(input) {
        cmd.name
    } else {
        cmd.aliases
            .iter()
            .find(|a| a.starts_with(input))
            .copied()
            .unwrap_or(cmd.name)
    };

    if matched_name.len() <= input.len() {
        // Already fully typed — show trailing space if it takes args
        if cmd.has_arg {
            return Some(" ".into());
        }
        return None;
    }

    let suffix = &matched_name[input.len()..];
    if cmd.has_arg {
        Some(format!("{suffix} "))
    } else {
        Some(suffix.to_string())
    }
}

/// Result of executing a slash command.
pub struct CommandResult {
    /// Feedback text to display in the chat log (None = silent).
    pub feedback: Option<String>,
    /// Whether the input was handled as a command.
    pub handled: bool,
}

/// Execute a slash command. Mutates app state directly.
pub async fn execute(
    app: &mut TuiApp,
    input: &str,
    pool: Option<&Arc<Mutex<LlmPool>>>,
) -> CommandResult {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd_str = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd_str {
        "/exit" | "/quit" => {
            app.should_quit = true;
            CommandResult {
                feedback: None,
                handled: true,
            }
        }
        "/clear" => {
            app.chat_log.clear();
            app.message_scroll = 0;
            app.message_auto_scroll = true;
            CommandResult {
                feedback: None,
                handled: true,
            }
        }
        "/model" => {
            if arg.is_empty() {
                // Show current model
                let current = if let Some(p) = pool {
                    let p = p.lock().await;
                    p.default_model().to_string()
                } else {
                    "unknown (no pool)".into()
                };
                CommandResult {
                    feedback: Some(format!("Current model: {current}")),
                    handled: true,
                }
            } else if let Some(p) = pool {
                let resolved = resolve_model(arg);
                let mut p = p.lock().await;
                p.set_default_model(arg);
                CommandResult {
                    feedback: Some(format!("Model set to: {resolved}")),
                    handled: true,
                }
            } else {
                CommandResult {
                    feedback: Some("No LLM pool available".into()),
                    handled: true,
                }
            }
        }
        "/help" => {
            let mut lines = Vec::new();
            for cmd in COMMANDS {
                let aliases = if cmd.aliases.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", cmd.aliases.join(", "))
                };
                let arg_hint = if cmd.has_arg { " <arg>" } else { "" };
                lines.push(format!(
                    "  {}{}{} — {}",
                    cmd.name, arg_hint, aliases, cmd.description
                ));
            }
            CommandResult {
                feedback: Some(format!("Commands:\n{}", lines.join("\n"))),
                handled: true,
            }
        }
        _ => CommandResult {
            feedback: Some(format!("Unknown command: {cmd_str}")),
            handled: true,
        },
    }
}

/// Push command feedback into the chat log as a system message.
pub fn push_feedback(app: &mut TuiApp, text: &str) {
    app.chat_log.push(ChatEntry {
        role: "system".into(),
        text: text.into(),
    });
    app.message_auto_scroll = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggest_model() {
        let cmd = suggest("/m").unwrap();
        assert_eq!(cmd.name, "/model");
    }

    #[test]
    fn suggest_exit() {
        let cmd = suggest("/e").unwrap();
        assert_eq!(cmd.name, "/exit");
    }

    #[test]
    fn suggest_quit_alias() {
        let cmd = suggest("/q").unwrap();
        assert_eq!(cmd.name, "/exit"); // /quit is alias of /exit
    }

    #[test]
    fn suggest_no_match() {
        assert!(suggest("/xyz").is_none());
    }

    #[test]
    fn suggest_exact_match() {
        let cmd = suggest("/help").unwrap();
        assert_eq!(cmd.name, "/help");
    }

    #[test]
    fn suggest_not_slash() {
        assert!(suggest("hello").is_none());
    }

    #[test]
    fn ghost_suffix_partial() {
        assert_eq!(ghost_suffix("/mo"), Some("del ".into()));
    }

    #[test]
    fn ghost_suffix_full_with_arg() {
        assert_eq!(ghost_suffix("/model"), Some(" ".into()));
    }

    #[test]
    fn ghost_suffix_full_no_arg() {
        // /help has no args — no ghost suffix when fully typed
        assert_eq!(ghost_suffix("/help"), None);
    }

    #[test]
    fn ghost_suffix_with_space() {
        // Already has a space — no ghost
        assert_eq!(ghost_suffix("/model haiku"), None);
    }

    #[test]
    fn ghost_suffix_slash_only() {
        // Just "/" — matches /clear (first alphabetical prefix match is /exit actually)
        let result = ghost_suffix("/");
        assert!(result.is_some()); // should match something
    }

    #[tokio::test]
    async fn execute_exit() {
        let mut app = TuiApp::new();
        let result = execute(&mut app, "/exit", None).await;
        assert!(result.handled);
        assert!(app.should_quit);
    }

    #[tokio::test]
    async fn execute_quit() {
        let mut app = TuiApp::new();
        let result = execute(&mut app, "/quit", None).await;
        assert!(result.handled);
        assert!(app.should_quit);
    }

    #[tokio::test]
    async fn execute_clear() {
        let mut app = TuiApp::new();
        app.chat_log.push(ChatEntry {
            role: "user".into(),
            text: "hello".into(),
        });
        assert_eq!(app.chat_log.len(), 1);

        let result = execute(&mut app, "/clear", None).await;
        assert!(result.handled);
        assert!(app.chat_log.is_empty());
    }

    #[tokio::test]
    async fn execute_model_switch() {
        let pool = LlmPool::new("test-key".into(), "opus");
        let pool = Arc::new(Mutex::new(pool));

        let mut app = TuiApp::new();
        let result = execute(&mut app, "/model haiku", Some(&pool)).await;
        assert!(result.handled);
        assert!(result.feedback.unwrap().contains("claude-haiku"));

        let p = pool.lock().await;
        assert_eq!(p.default_model(), "claude-haiku-4-5-20251001");
    }

    #[tokio::test]
    async fn execute_model_show_current() {
        let pool = LlmPool::new("test-key".into(), "sonnet");
        let pool = Arc::new(Mutex::new(pool));

        let mut app = TuiApp::new();
        let result = execute(&mut app, "/model", Some(&pool)).await;
        assert!(result.handled);
        assert!(result.feedback.unwrap().contains("claude-sonnet"));
    }

    #[tokio::test]
    async fn execute_help() {
        let mut app = TuiApp::new();
        let result = execute(&mut app, "/help", None).await;
        assert!(result.handled);
        let text = result.feedback.unwrap();
        assert!(text.contains("/exit"));
        assert!(text.contains("/model"));
        assert!(text.contains("/clear"));
        assert!(text.contains("/help"));
    }

    #[tokio::test]
    async fn execute_unknown() {
        let mut app = TuiApp::new();
        let result = execute(&mut app, "/foobar", None).await;
        assert!(result.handled);
        assert!(result.feedback.unwrap().contains("Unknown command"));
    }
}
