//! Slash commands for the TUI input bar.
//!
//! Commands start with `/` and are intercepted before reaching the pipeline.
//! Ghost-text autocomplete suggests commands as you type.

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::llm::LlmPool;

use super::app::{ChatEntry, InputMode, TuiApp};

/// What kind of values an argument accepts.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgKind {
    /// Fixed set of known values (e.g., opus/sonnet/haiku).
    Enum(&'static [&'static str]),
    /// Free-form text — no completions but shows hint.
    Free(&'static str),
}

/// A positional argument in a command.
#[derive(Debug, Clone, PartialEq)]
pub struct ArgSpec {
    pub name: &'static str,
    pub kind: ArgKind,
}

/// A subcommand (e.g., "install" under "/plugins").
#[derive(Debug, Clone, PartialEq)]
pub struct SubcommandSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub args: &'static [ArgSpec],
}

/// A slash command definition.
#[derive(Debug, PartialEq)]
pub struct SlashCommand {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub description: &'static str,
    pub has_arg: bool,
    pub args: &'static [ArgSpec],
    pub subcommands: &'static [SubcommandSpec],
}

/// All known commands.
pub static COMMANDS: &[SlashCommand] = &[
    SlashCommand {
        name: "/exit",
        aliases: &["/quit"],
        description: "Quit AgentOS",
        has_arg: false,
        args: &[],
        subcommands: &[],
    },
    SlashCommand {
        name: "/clear",
        aliases: &[],
        description: "Clear chat log",
        has_arg: false,
        args: &[],
        subcommands: &[],
    },
    SlashCommand {
        name: "/model",
        aliases: &[],
        description: "Switch default model (opus/sonnet/sonnet-4.5/haiku or full ID)",
        has_arg: true,
        args: &[ArgSpec {
            name: "model_id",
            kind: ArgKind::Enum(&["opus", "sonnet", "sonnet-4.5", "haiku"]),
        }],
        subcommands: &[],
    },
    SlashCommand {
        name: "/help",
        aliases: &[],
        description: "List available commands",
        has_arg: false,
        args: &[],
        subcommands: &[],
    },
    SlashCommand {
        name: "/models",
        aliases: &[],
        description: "Manage configured models",
        has_arg: true,
        args: &[],
        subcommands: &[
            SubcommandSpec {
                name: "remove",
                description: "Remove a model",
                args: &[ArgSpec {
                    name: "alias",
                    kind: ArgKind::Free("model alias"),
                }],
            },
            SubcommandSpec {
                name: "default",
                description: "Set default model",
                args: &[ArgSpec {
                    name: "alias",
                    kind: ArgKind::Free("model alias"),
                }],
            },
        ],
    },
    SlashCommand {
        name: "/provider",
        aliases: &[],
        description: "Configure API provider (auto-discovers models)",
        has_arg: true,
        args: &[ArgSpec {
            name: "name",
            kind: ArgKind::Enum(&["anthropic", "openai", "ollama"]),
        }],
        subcommands: &[
            SubcommandSpec {
                name: "remove",
                description: "Remove a provider and all its models",
                args: &[ArgSpec {
                    name: "name",
                    kind: ArgKind::Free("provider name"),
                }],
            },
        ],
    },
    SlashCommand {
        name: "/agents",
        aliases: &[],
        description: "Manage favorite agents",
        has_arg: true,
        args: &[],
        subcommands: &[
            SubcommandSpec {
                name: "add",
                description: "Add agent to favorites",
                args: &[ArgSpec {
                    name: "agent name",
                    kind: ArgKind::Free("agent name"),
                }],
            },
            SubcommandSpec {
                name: "remove",
                description: "Remove agent from favorites",
                args: &[ArgSpec {
                    name: "agent name",
                    kind: ArgKind::Free("agent name"),
                }],
            },
        ],
    },
];

/// Return all commands whose name or alias prefix-matches the input.
/// Used by the command popup to show filtered choices. Sorted alphabetically.
pub fn matching_commands(input: &str) -> Vec<&'static SlashCommand> {
    if !input.starts_with('/') {
        return Vec::new();
    }
    let cmd_part = input.split_whitespace().next().unwrap_or(input);
    let mut results = Vec::new();
    for cmd in COMMANDS {
        if cmd.name.starts_with(cmd_part)
            || cmd.aliases.iter().any(|a| a.starts_with(cmd_part))
        {
            results.push(cmd);
        }
    }
    results.sort_by_key(|c| c.name);
    results
}

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
        if cmd.name == cmd_part || cmd.aliases.contains(&cmd_part) {
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
    let parts: Vec<&str> = input.splitn(3, ' ').collect();
    let cmd_str = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");
    let arg2 = parts.get(2).map(|s| s.trim()).unwrap_or("");

    match cmd_str {
        "/exit" | "/quit" => {
            app.should_quit = true;
            CommandResult {
                feedback: None,
                handled: true,
            }
        }
        "/clear" => {
            // Clear active agent tab if on one
            if let Some(tab) = app.active_agent_tab_mut() {
                tab.chat_log.clear();
                tab.message_scroll = 0;
                tab.message_auto_scroll = true;
            }
            // Bridge: clear global
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
                    "No LLM pool — use /provider to configure".into()
                };
                CommandResult {
                    feedback: Some(format!("Current model: {current}")),
                    handled: true,
                }
            } else if let Some(p) = pool {
                // Rebuild pool for the target alias (handles cross-provider switches)
                let config = app.models_config.lock().await;
                let mut p = p.lock().await;
                match p.rebuild_for_alias(&config, arg) {
                    Ok(()) => CommandResult {
                        feedback: Some(format!("Model set to: {}", p.default_model())),
                        handled: true,
                    },
                    Err(e) => CommandResult {
                        feedback: Some(format!("Failed to switch model: {e}")),
                        handled: true,
                    },
                }
            } else {
                // No pool — try to create one from config
                let config = app.models_config.lock().await;
                match crate::llm::LlmPool::from_config(&config) {
                    Ok(mut new_pool) => {
                        match new_pool.rebuild_for_alias(&config, arg) {
                            Ok(()) => {
                                let model = new_pool.default_model().to_string();
                                drop(config);
                                app.llm_pool = Some(Arc::new(Mutex::new(new_pool)));
                                CommandResult {
                                    feedback: Some(format!("Pool connected. Model set to: {model}")),
                                    handled: true,
                                }
                            }
                            Err(e) => CommandResult {
                                feedback: Some(format!("Failed to switch model: {e}")),
                                handled: true,
                            },
                        }
                    }
                    Err(e) => CommandResult {
                        feedback: Some(format!("No LLM pool and can't create one: {e}. Use /provider first.")),
                        handled: true,
                    },
                }
            }
        }
        "/agents" => {
            if !arg2.is_empty() {
                execute_agents_with_arg(app, arg, arg2)
            } else {
                execute_agents(app, arg)
            }
        }
        "/models" => {
            execute_models(app, arg, arg2, pool).await
        }
        "/provider" => {
            execute_provider(app, arg, arg2).await
        }
        "/help" => {
            let mut lines = Vec::new();
            for cmd in COMMANDS {
                let aliases = if cmd.aliases.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", cmd.aliases.join(", "))
                };
                let arg_hint = if !cmd.args.is_empty() {
                    format!(" {}", cmd.args.iter().map(|a| format!("<{}>", a.name)).collect::<Vec<_>>().join(" "))
                } else if cmd.has_arg {
                    " <arg>".to_string()
                } else {
                    String::new()
                };
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
    app.chat_log.push(ChatEntry::new("system", text));
    app.message_auto_scroll = true;
}

/// Handle `/agents` subcommands.
fn execute_agents(app: &mut TuiApp, subcommand: &str) -> CommandResult {
    match subcommand {
        "" => {
            // List all agent listeners from organism config (via chat log)
            // We don't have the organism here, so list favorites + selected
            let mut lines = vec!["Favorite agents:".to_string()];
            if app.agents_config.favorites.is_empty() {
                lines.push("  (none — use /agents add <name> to add)".into());
            } else {
                for name in &app.agents_config.favorites {
                    let marker = if app.selected_agent.as_deref() == Some(name.as_str()) {
                        "\u{25b6} "
                    } else {
                        "  "
                    };
                    lines.push(format!("{marker}{name}"));
                }
            }
            if let Some(ref sel) = app.selected_agent {
                lines.push(format!("\nActive: {sel}"));
            } else {
                lines.push("\nActive: (default — first agent)".into());
            }
            CommandResult {
                feedback: Some(lines.join("\n")),
                handled: true,
            }
        }
        "add" => {
            CommandResult {
                feedback: Some("Usage: /agents add <name>".into()),
                handled: true,
            }
        }
        "remove" => {
            CommandResult {
                feedback: Some("Usage: /agents remove <name>".into()),
                handled: true,
            }
        }
        _ => {
            // Treat as agent name for add/remove
            // Check if it looks like "add <name>" or "remove <name>"
            CommandResult {
                feedback: Some(format!("Unknown /agents subcommand: {subcommand}. Use add or remove.")),
                handled: true,
            }
        }
    }
}

/// Handle `/agents add <name>` and `/agents remove <name>`.
/// Called from execute() when /agents has 3 parts.
fn execute_agents_with_arg(app: &mut TuiApp, subcommand: &str, name: &str) -> CommandResult {
    match subcommand {
        "add" => {
            if name.is_empty() {
                return CommandResult {
                    feedback: Some("Usage: /agents add <agent-name>".into()),
                    handled: true,
                };
            }
            app.agents_config.add(name);
            let _ = app.agents_config.save();
            app.rebuild_menu();
            CommandResult {
                feedback: Some(format!("Added '{name}' to favorites.")),
                handled: true,
            }
        }
        "remove" => {
            if name.is_empty() {
                return CommandResult {
                    feedback: Some("Usage: /agents remove <agent-name>".into()),
                    handled: true,
                };
            }
            if app.agents_config.remove(name) {
                // If the removed agent was selected, reset to default
                if app.selected_agent.as_deref() == Some(name) {
                    app.selected_agent = None;
                }
                let _ = app.agents_config.save();
                app.rebuild_menu();
                CommandResult {
                    feedback: Some(format!("Removed '{name}' from favorites.")),
                    handled: true,
                }
            } else {
                CommandResult {
                    feedback: Some(format!("'{name}' is not in favorites.")),
                    handled: true,
                }
            }
        }
        _ => CommandResult {
            feedback: Some(format!("Unknown /agents subcommand: {subcommand}. Use add or remove.")),
            handled: true,
        },
    }
}

/// Handle `/models` subcommands.
async fn execute_models(
    app: &mut TuiApp,
    subcommand: &str,
    arg: &str,
    pool: Option<&Arc<Mutex<LlmPool>>>,
) -> CommandResult {
    match subcommand {
        "" => {
            // /models — list available models from API, fall back to config
            let mut output = String::new();

            if let Some(p) = pool {
                let p = p.lock().await;
                match p.list_models().await {
                    Ok(models) => {
                        let current = p.default_model().to_string();
                        output.push_str("Available models (from API):\n");
                        for m in &models {
                            let marker = if m.id == current { " *" } else { "  " };
                            output.push_str(&format!("{marker} {:<40} {}\n", m.id, m.display_name));
                        }
                        if models.is_empty() {
                            output.push_str("  (none returned)\n");
                        }
                        output.push_str(&format!("\nActive: {current}"));
                    }
                    Err(e) => {
                        // API query failed — fall back to config list
                        output.push_str(&format!("API query failed: {e}\n\n"));
                        let config = app.models_config.lock().await;
                        output.push_str(&crate::config::format_model_list(&config));
                    }
                }
            } else {
                // No pool — show config only
                let config = app.models_config.lock().await;
                output.push_str(&crate::config::format_model_list(&config));
            }

            CommandResult {
                feedback: Some(output),
                handled: true,
            }
        }
        "remove" => {
            if arg.is_empty() {
                return CommandResult {
                    feedback: Some("Usage: /models remove <alias>".into()),
                    handled: true,
                };
            }
            let mut config = app.models_config.lock().await;
            if config.remove_model(arg) {
                let _ = config.save();
                drop(config);
                CommandResult {
                    feedback: Some(format!("Removed model: {arg}")),
                    handled: true,
                }
            } else {
                CommandResult {
                    feedback: Some(format!("Unknown model alias: {arg}")),
                    handled: true,
                }
            }
        }
        "default" => {
            if arg.is_empty() {
                return CommandResult {
                    feedback: Some("Usage: /models default <alias>".into()),
                    handled: true,
                };
            }
            let mut config = app.models_config.lock().await;
            if config.set_default(arg) {
                let _ = config.save();
                // Also switch the active pool model
                if let Some(p) = pool {
                    let mut p = p.lock().await;
                    p.set_default_model_from_config(&config, arg);
                }
                drop(config);
                CommandResult {
                    feedback: Some(format!("Default model set to: {arg}")),
                    handled: true,
                }
            } else {
                CommandResult {
                    feedback: Some(format!("Unknown model alias: {arg}")),
                    handled: true,
                }
            }
        }
        _ => CommandResult {
            feedback: Some(format!("Unknown /models subcommand: {subcommand}. Use remove or default.")),
            handled: true,
        },
    }
}

/// Handle `/provider` command.
async fn execute_provider(
    app: &mut TuiApp,
    subcommand_or_name: &str,
    arg: &str,
) -> CommandResult {
    match subcommand_or_name {
        "" => {
            // /provider — list configured providers
            let config = app.models_config.lock().await;
            if config.providers.is_empty() {
                return CommandResult {
                    feedback: Some("No providers configured. Use /provider <name> to add one.\nExample: /provider anthropic".into()),
                    handled: true,
                };
            }
            let mut lines = vec!["Providers:".to_string()];
            for (name, provider) in &config.providers {
                let key_status = if provider.api_key.is_some() { "\u{2713}" } else { "\u{2717}" };
                let model_count = provider.models.len();
                lines.push(format!("  {key_status} {name:<12} ({model_count} models)"));
            }
            CommandResult {
                feedback: Some(lines.join("\n")),
                handled: true,
            }
        }
        "remove" => {
            if arg.is_empty() {
                return CommandResult {
                    feedback: Some("Usage: /provider remove <name>".into()),
                    handled: true,
                };
            }
            let mut config = app.models_config.lock().await;
            if config.providers.remove(arg).is_some() {
                // Clear default if it belonged to the removed provider
                if let Some(ref default) = config.default {
                    if config.resolve(default).is_none() {
                        config.default = None;
                    }
                }
                let _ = config.save();
                CommandResult {
                    feedback: Some(format!("Removed provider: {arg}")),
                    handled: true,
                }
            } else {
                CommandResult {
                    feedback: Some(format!("Unknown provider: {arg}")),
                    handled: true,
                }
            }
        }
        name => {
            // /provider <name> — enter wizard to paste API key
            app.input_mode = InputMode::ProviderWizard {
                provider: name.to_string(),
            };
            app.clear_input();
            CommandResult {
                feedback: None,
                handled: true,
            }
        }
    }
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
        app.chat_log.push(ChatEntry::new("user", "hello"));
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
    async fn execute_model_no_pool_shows_hint() {
        let mut app = TuiApp::new();
        let result = execute(&mut app, "/model", None).await;
        assert!(result.handled);
        assert!(result.feedback.unwrap().contains("/provider"));
    }

    #[tokio::test]
    async fn execute_model_no_pool_creates_from_config() {
        let mut app = TuiApp::new();
        // Set up config with a key
        {
            let mut config = app.models_config.lock().await;
            config.add_model("anthropic", "haiku", "claude-haiku-4-5-20251001", Some("sk-test".into()), None);
        }
        assert!(app.llm_pool.is_none());

        let result = execute(&mut app, "/model haiku", None).await;
        assert!(result.handled);
        assert!(result.feedback.unwrap().contains("Pool connected"));
        // Pool should now exist
        let pool_arc = app.llm_pool.unwrap();
        let p = pool_arc.lock().await;
        assert_eq!(p.default_model(), "claude-haiku-4-5-20251001");
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

    // ── /models tests ──

    #[tokio::test]
    async fn execute_models_list() {
        let mut app = TuiApp::new();
        // Add a model to config
        {
            let mut config = app.models_config.lock().await;
            config.add_model("anthropic", "opus", "claude-opus-4-6", Some("key".into()), None);
            config.set_default("opus");
        }
        let result = execute(&mut app, "/models", None).await;
        assert!(result.handled);
        let text = result.feedback.unwrap();
        assert!(text.contains("opus"));
        assert!(text.contains("anthropic"));
        assert!(text.contains("(default)"));
    }

    #[tokio::test]
    async fn execute_provider_enters_wizard() {
        let mut app = TuiApp::new();
        let result = execute(&mut app, "/provider anthropic", None).await;
        assert!(result.handled);
        assert!(result.feedback.is_none()); // wizard mode, no immediate feedback
        match &app.input_mode {
            InputMode::ProviderWizard { provider } => {
                assert_eq!(provider, "anthropic");
            }
            _ => panic!("expected ProviderWizard"),
        }
    }

    #[tokio::test]
    async fn execute_models_remove() {
        let mut app = TuiApp::new();
        {
            let mut config = app.models_config.lock().await;
            config.add_model("anthropic", "opus", "claude-opus-4-6", Some("key".into()), None);
        }
        let result = execute(&mut app, "/models remove opus", None).await;
        assert!(result.handled);
        assert!(result.feedback.unwrap().contains("Removed"));
        // Verify it's gone
        let config = app.models_config.lock().await;
        assert!(config.resolve("opus").is_none());
    }

    #[tokio::test]
    async fn execute_models_default() {
        let mut app = TuiApp::new();
        {
            let mut config = app.models_config.lock().await;
            config.add_model("anthropic", "opus", "claude-opus-4-6", Some("key".into()), None);
            config.add_model("anthropic", "sonnet", "claude-sonnet-4-6", None, None);
        }
        let result = execute(&mut app, "/models default opus", None).await;
        assert!(result.handled);
        assert!(result.feedback.unwrap().contains("Default model set to: opus"));
        let config = app.models_config.lock().await;
        assert_eq!(config.default, Some("opus".into()));
    }

    #[tokio::test]
    async fn execute_models_default_unknown() {
        let mut app = TuiApp::new();
        let result = execute(&mut app, "/models default nonexistent", None).await;
        assert!(result.handled);
        assert!(result.feedback.unwrap().contains("Unknown model alias"));
    }

    #[tokio::test]
    async fn execute_provider_list() {
        let mut app = TuiApp::new();
        {
            let mut config = app.models_config.lock().await;
            config.add_model("anthropic", "opus", "claude-opus-4-6", Some("sk-test".into()), None);
        }
        let result = execute(&mut app, "/provider", None).await;
        assert!(result.handled);
        let text = result.feedback.unwrap();
        assert!(text.contains("anthropic"));
        assert!(text.contains("\u{2713}")); // checkmark for key present
    }

    #[tokio::test]
    async fn execute_provider_remove() {
        let mut app = TuiApp::new();
        {
            let mut config = app.models_config.lock().await;
            config.add_model("anthropic", "opus", "claude-opus-4-6", Some("sk-test".into()), None);
        }
        let result = execute(&mut app, "/provider remove anthropic", None).await;
        assert!(result.handled);
        assert!(result.feedback.unwrap().contains("Removed provider"));
        let config = app.models_config.lock().await;
        assert!(!config.providers.contains_key("anthropic"));
    }

    #[tokio::test]
    async fn execute_provider_remove_unknown() {
        let mut app = TuiApp::new();
        let result = execute(&mut app, "/provider remove nonexistent", None).await;
        assert!(result.handled);
        assert!(result.feedback.unwrap().contains("Unknown provider"));
    }
}
