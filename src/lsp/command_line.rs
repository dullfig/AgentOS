//! Language service for the command-line input bar.
//!
//! Implements `LanguageService` over slash commands. No tree-sitter —
//! the grammar is trivial (space-delimited tokens). Pure functions on text.

use lsp_types::{
    CompletionItem, CompletionItemKind, Diagnostic, DiagnosticSeverity, Position, Range,
};

use super::{HoverInfo, LanguageService};
use crate::tui::commands::{ArgKind, COMMANDS, SlashCommand};

/// Parsed position within a command-line input.
#[derive(Debug, PartialEq)]
pub enum InputContext<'a> {
    /// Typing a command name (position 0).
    CommandName { prefix: &'a str },
    /// Typing a subcommand (position 1, command has subcommands).
    Subcommand {
        cmd: &'a SlashCommand,
        prefix: &'a str,
    },
    /// Typing an argument value.
    Argument {
        cmd: &'a SlashCommand,
        arg_index: usize,
        prefix: &'a str,
    },
    /// Past all defined arguments — nothing more expected.
    Trailing { cmd: &'a SlashCommand },
    /// Not a command (doesn't start with `/`).
    NotCommand,
}

/// Parse input text into a position context.
pub fn parse_input(input: &str) -> InputContext<'_> {
    if !input.starts_with('/') {
        return InputContext::NotCommand;
    }

    let tokens: Vec<&str> = input.split_whitespace().collect();
    let has_trailing_space = input.ends_with(' ') && !input.ends_with("  ");

    if tokens.is_empty() {
        return InputContext::CommandName { prefix: "/" };
    }

    let cmd_token = tokens[0];

    // Still typing the command name (no space after it yet)
    if tokens.len() == 1 && !has_trailing_space {
        return InputContext::CommandName { prefix: cmd_token };
    }

    // Find matching command (exact match on name or alias)
    let cmd = COMMANDS.iter().find(|c| {
        c.name == cmd_token || c.aliases.contains(&cmd_token)
    });

    let Some(cmd) = cmd else {
        // Unknown command — still treat as command name for diagnostics
        if tokens.len() == 1 {
            return InputContext::CommandName { prefix: cmd_token };
        }
        return InputContext::NotCommand;
    };

    // Determine which positional token we're on
    let arg_tokens = &tokens[1..];
    let current_pos = if has_trailing_space {
        arg_tokens.len() // cursor is at the next (empty) position
    } else {
        arg_tokens.len().saturating_sub(1)
    };

    let current_prefix = if has_trailing_space {
        ""
    } else {
        arg_tokens.last().copied().unwrap_or("")
    };

    // If command has subcommands, position 0 is a subcommand
    if !cmd.subcommands.is_empty() && current_pos == 0 {
        return InputContext::Subcommand {
            cmd,
            prefix: current_prefix,
        };
    }

    // Map to argument spec
    let arg_offset = if !cmd.subcommands.is_empty() { 1 } else { 0 };
    let arg_index = current_pos.saturating_sub(arg_offset);

    if arg_index < cmd.args.len() {
        InputContext::Argument {
            cmd,
            arg_index,
            prefix: current_prefix,
        }
    } else {
        InputContext::Trailing { cmd }
    }
}

/// Language service for the command-line input bar.
#[derive(Default)]
pub struct CommandLineService;

impl CommandLineService {
    pub fn new() -> Self {
        Self
    }
}

impl LanguageService for CommandLineService {
    fn diagnostics(&self, content: &str) -> Vec<Diagnostic> {
        let input = content.lines().next().unwrap_or("");
        if !input.starts_with('/') {
            return Vec::new();
        }

        let mut diags = Vec::new();
        let tokens: Vec<&str> = input.split_whitespace().collect();
        if tokens.is_empty() {
            return diags;
        }

        let cmd_token = tokens[0];
        let cmd = COMMANDS.iter().find(|c| {
            c.name == cmd_token || c.aliases.contains(&cmd_token)
        });

        if cmd.is_none() {
            // Check if it's a partial prefix match (still typing)
            let is_partial = COMMANDS
                .iter()
                .any(|c| c.name.starts_with(cmd_token) || c.aliases.iter().any(|a| a.starts_with(cmd_token)));
            if !is_partial {
                diags.push(Diagnostic {
                    range: Range::new(
                        Position::new(0, 0),
                        Position::new(0, cmd_token.len() as u32),
                    ),
                    severity: Some(DiagnosticSeverity::ERROR),
                    message: format!("Unknown command: {cmd_token}"),
                    ..Default::default()
                });
            }
        }

        if let Some(cmd) = cmd {
            let arg_tokens = &tokens[1..];
            let expected = cmd.args.len() + if cmd.subcommands.is_empty() { 0 } else { 1 };
            if arg_tokens.len() > expected && expected > 0 {
                let extra_start: usize = tokens[..1 + expected].iter().map(|t| t.len() + 1).sum();
                diags.push(Diagnostic {
                    range: Range::new(
                        Position::new(0, extra_start as u32),
                        Position::new(0, input.len() as u32),
                    ),
                    severity: Some(DiagnosticSeverity::WARNING),
                    message: format!(
                        "Too many arguments (expected {})",
                        expected,
                    ),
                    ..Default::default()
                });
            }
        }

        diags
    }

    fn completions(&self, content: &str, _pos: Position) -> Vec<CompletionItem> {
        let input = content.lines().next().unwrap_or("");
        match parse_input(input) {
            InputContext::CommandName { prefix } => {
                let mut items = Vec::new();
                for cmd in COMMANDS {
                    if cmd.name.starts_with(prefix) {
                        items.push(CompletionItem {
                            label: cmd.name.to_string(),
                            kind: Some(CompletionItemKind::FUNCTION),
                            detail: Some(cmd.description.to_string()),
                            insert_text: Some(if cmd.has_arg || !cmd.subcommands.is_empty() {
                                format!("{} ", cmd.name)
                            } else {
                                cmd.name.to_string()
                            }),
                            ..Default::default()
                        });
                    }
                    for alias in cmd.aliases {
                        if alias.starts_with(prefix) {
                            items.push(CompletionItem {
                                label: alias.to_string(),
                                kind: Some(CompletionItemKind::FUNCTION),
                                detail: Some(format!("{} (alias of {})", cmd.description, cmd.name)),
                                insert_text: Some(if cmd.has_arg || !cmd.subcommands.is_empty() {
                                    format!("{} ", alias)
                                } else {
                                    alias.to_string()
                                }),
                                ..Default::default()
                            });
                        }
                    }
                }
                items
            }
            InputContext::Subcommand { cmd, prefix } => {
                cmd.subcommands
                    .iter()
                    .filter(|sc| sc.name.starts_with(prefix))
                    .map(|sc| CompletionItem {
                        label: sc.name.to_string(),
                        kind: Some(CompletionItemKind::ENUM_MEMBER),
                        detail: Some(sc.description.to_string()),
                        insert_text: Some(if sc.args.is_empty() {
                            sc.name.to_string()
                        } else {
                            format!("{} ", sc.name)
                        }),
                        ..Default::default()
                    })
                    .collect()
            }
            InputContext::Argument {
                cmd,
                arg_index,
                prefix,
            } => {
                if let Some(arg) = cmd.args.get(arg_index) {
                    match &arg.kind {
                        ArgKind::Enum(values) => values
                            .iter()
                            .filter(|v| v.starts_with(prefix))
                            .map(|v| CompletionItem {
                                label: v.to_string(),
                                kind: Some(CompletionItemKind::ENUM_MEMBER),
                                detail: Some(format!("{}: {}", arg.name, v)),
                                insert_text: Some(v.to_string()),
                                ..Default::default()
                            })
                            .collect(),
                        ArgKind::Free(_hint) => Vec::new(),
                    }
                } else {
                    Vec::new()
                }
            }
            InputContext::Trailing { .. } | InputContext::NotCommand => Vec::new(),
        }
    }

    fn hover(&self, content: &str, pos: Position) -> Option<HoverInfo> {
        let input = content.lines().next().unwrap_or("");
        if !input.starts_with('/') {
            return None;
        }

        // Find which token the cursor is in
        let col = pos.character as usize;
        let tokens: Vec<(usize, &str)> = input
            .split_whitespace()
            .scan(0usize, |offset, token| {
                let start = input[*offset..].find(token).unwrap_or(0) + *offset;
                *offset = start + token.len();
                Some((start, token))
            })
            .collect();

        let token_index = tokens
            .iter()
            .position(|(start, token)| col >= *start && col <= start + token.len());

        let token_index = token_index?;

        if token_index == 0 {
            // Hovering over command name
            let cmd_token = tokens[0].1;
            let cmd = COMMANDS.iter().find(|c| {
                c.name == cmd_token
                    || c.name.starts_with(cmd_token)
                    || c.aliases.iter().any(|a| *a == cmd_token || a.starts_with(cmd_token))
            })?;

            let mut text = format!("**{}**\n{}", cmd.name, cmd.description);
            if !cmd.aliases.is_empty() {
                text.push_str(&format!(
                    "\nAliases: {}",
                    cmd.aliases.join(", ")
                ));
            }
            if !cmd.args.is_empty() {
                text.push_str("\nArgs:");
                for arg in cmd.args {
                    let kind_str = match &arg.kind {
                        ArgKind::Enum(vals) => format!("[{}]", vals.join("|")),
                        ArgKind::Free(hint) => hint.to_string(),
                    };
                    text.push_str(&format!("\n  {} — {}", arg.name, kind_str));
                }
            }
            Some(HoverInfo {
                content: text,
                range: Some(Range::new(
                    Position::new(0, tokens[0].0 as u32),
                    Position::new(0, (tokens[0].0 + tokens[0].1.len()) as u32),
                )),
            })
        } else {
            // Hovering over an argument
            let cmd_token = tokens[0].1;
            let cmd = COMMANDS.iter().find(|c| {
                c.name == cmd_token || c.aliases.contains(&cmd_token)
            })?;

            let arg_index = token_index - 1;
            let arg = cmd.args.get(arg_index)?;

            let kind_str = match &arg.kind {
                ArgKind::Enum(vals) => format!("one of: {}", vals.join(", ")),
                ArgKind::Free(hint) => hint.to_string(),
            };
            Some(HoverInfo {
                content: format!("**{}**\n{}", arg.name, kind_str),
                range: Some(Range::new(
                    Position::new(0, tokens[token_index].0 as u32),
                    Position::new(
                        0,
                        (tokens[token_index].0 + tokens[token_index].1.len()) as u32,
                    ),
                )),
            })
        }
    }
}

/// Compute position-aware ghost text suffix for the input bar.
/// Extends the original `ghost_suffix` to handle argument positions.
pub fn ghost_suffix(input: &str) -> Option<String> {
    if !input.starts_with('/') {
        return None;
    }

    match parse_input(input) {
        InputContext::CommandName { prefix } => {
            // Same logic as original ghost_suffix for command names
            let cmd = crate::tui::commands::suggest(prefix)?;
            let matched_name = if cmd.name.starts_with(prefix) {
                cmd.name
            } else {
                cmd.aliases
                    .iter()
                    .find(|a| a.starts_with(prefix))
                    .copied()
                    .unwrap_or(cmd.name)
            };

            if matched_name.len() <= prefix.len() {
                if cmd.has_arg || !cmd.args.is_empty() {
                    return Some(" ".into());
                }
                return None;
            }

            let suffix = &matched_name[prefix.len()..];
            if cmd.has_arg || !cmd.args.is_empty() || !cmd.subcommands.is_empty() {
                Some(format!("{suffix} "))
            } else {
                Some(suffix.to_string())
            }
        }
        InputContext::Subcommand { cmd, prefix } => {
            if prefix.is_empty() {
                // Show first subcommand as hint
                cmd.subcommands.first().map(|sc| sc.name.to_string())
            } else {
                // Complete best matching subcommand
                cmd.subcommands
                    .iter()
                    .find(|sc| sc.name.starts_with(prefix))
                    .map(|sc| sc.name[prefix.len()..].to_string())
            }
        }
        InputContext::Argument {
            cmd,
            arg_index,
            prefix,
        } => {
            let arg = cmd.args.get(arg_index)?;
            match &arg.kind {
                ArgKind::Enum(values) => {
                    if prefix.is_empty() {
                        // Show first value as placeholder
                        values.first().map(|v| v.to_string())
                    } else {
                        // Complete best matching value
                        values
                            .iter()
                            .find(|v| v.starts_with(prefix))
                            .map(|v| v[prefix.len()..].to_string())
                    }
                }
                ArgKind::Free(hint) => {
                    if prefix.is_empty() {
                        Some(hint.to_string())
                    } else {
                        None
                    }
                }
            }
        }
        InputContext::Trailing { .. } | InputContext::NotCommand => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_input tests ──

    #[test]
    fn parse_command_name_prefix() {
        match parse_input("/mo") {
            InputContext::CommandName { prefix } => assert_eq!(prefix, "/mo"),
            other => panic!("expected CommandName, got {other:?}"),
        }
    }

    #[test]
    fn parse_command_name_just_slash() {
        match parse_input("/") {
            InputContext::CommandName { prefix } => assert_eq!(prefix, "/"),
            other => panic!("expected CommandName, got {other:?}"),
        }
    }

    #[test]
    fn parse_argument_position() {
        match parse_input("/model ") {
            InputContext::Argument {
                cmd,
                arg_index,
                prefix,
            } => {
                assert_eq!(cmd.name, "/model");
                assert_eq!(arg_index, 0);
                assert_eq!(prefix, "");
            }
            other => panic!("expected Argument, got {other:?}"),
        }
    }

    #[test]
    fn parse_argument_with_prefix() {
        match parse_input("/model hai") {
            InputContext::Argument {
                cmd,
                arg_index,
                prefix,
            } => {
                assert_eq!(cmd.name, "/model");
                assert_eq!(arg_index, 0);
                assert_eq!(prefix, "hai");
            }
            other => panic!("expected Argument, got {other:?}"),
        }
    }

    #[test]
    fn parse_trailing_args() {
        match parse_input("/model haiku extra") {
            InputContext::Trailing { cmd } => {
                assert_eq!(cmd.name, "/model");
            }
            other => panic!("expected Trailing, got {other:?}"),
        }
    }

    #[test]
    fn parse_not_command() {
        assert_eq!(parse_input("hello"), InputContext::NotCommand);
    }

    #[test]
    fn parse_no_arg_command_with_space() {
        // /help + space → trailing (no args defined)
        match parse_input("/help ") {
            InputContext::Trailing { cmd } => {
                assert_eq!(cmd.name, "/help");
            }
            other => panic!("expected Trailing, got {other:?}"),
        }
    }

    // ── Completion tests ──

    #[test]
    fn completions_command_names() {
        let svc = CommandLineService::new();
        let items = svc.completions("/", Position::new(0, 1));
        assert!(items.len() >= 4); // at least /exit, /clear, /model, /help
        assert!(items.iter().any(|i| i.label == "/model"));
        assert!(items.iter().any(|i| i.label == "/exit"));
    }

    #[test]
    fn completions_model_values() {
        let svc = CommandLineService::new();
        let items = svc.completions("/model ", Position::new(0, 7));
        assert_eq!(items.len(), 4);
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"opus"));
        assert!(labels.contains(&"sonnet"));
        assert!(labels.contains(&"sonnet-4.5"));
        assert!(labels.contains(&"haiku"));
    }

    #[test]
    fn completions_filtered_by_prefix() {
        let svc = CommandLineService::new();
        let items = svc.completions("/model h", Position::new(0, 8));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].label, "haiku");
    }

    #[test]
    fn completions_empty_for_no_match() {
        let svc = CommandLineService::new();
        let items = svc.completions("/model xyz", Position::new(0, 10));
        assert!(items.is_empty());
    }

    // ── Ghost text tests ──

    #[test]
    fn ghost_suffix_command_name() {
        assert_eq!(ghost_suffix("/mo"), Some("del ".into()));
    }

    #[test]
    fn ghost_suffix_argument_value() {
        assert_eq!(ghost_suffix("/model h"), Some("aiku".into()));
    }

    #[test]
    fn ghost_suffix_after_space() {
        // After "/model " → show first enum value as placeholder
        assert_eq!(ghost_suffix("/model "), Some("opus".into()));
    }

    #[test]
    fn ghost_suffix_full_command_no_arg() {
        assert_eq!(ghost_suffix("/help"), None);
    }

    #[test]
    fn ghost_suffix_full_command_with_arg() {
        assert_eq!(ghost_suffix("/model"), Some(" ".into()));
    }

    #[test]
    fn ghost_suffix_not_command() {
        assert_eq!(ghost_suffix("hello"), None);
    }

    // ── Diagnostics tests ──

    #[test]
    fn diagnostics_unknown_command() {
        let svc = CommandLineService::new();
        let diags = svc.diagnostics("/foobar");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].severity, Some(DiagnosticSeverity::ERROR));
        assert!(diags[0].message.contains("Unknown command"));
    }

    #[test]
    fn diagnostics_valid_command() {
        let svc = CommandLineService::new();
        let diags = svc.diagnostics("/model haiku");
        assert!(diags.is_empty());
    }

    #[test]
    fn diagnostics_partial_command_no_error() {
        let svc = CommandLineService::new();
        // Still typing — /mo is a valid prefix, should not error
        let diags = svc.diagnostics("/mo");
        assert!(diags.is_empty());
    }

    // ── Hover tests ──

    #[test]
    fn hover_on_command() {
        let svc = CommandLineService::new();
        let hover = svc.hover("/model haiku", Position::new(0, 3));
        assert!(hover.is_some());
        let h = hover.unwrap();
        assert!(h.content.contains("/model"));
        assert!(h.content.contains("Switch default model"));
    }

    #[test]
    fn hover_on_argument() {
        let svc = CommandLineService::new();
        let hover = svc.hover("/model haiku", Position::new(0, 8));
        assert!(hover.is_some());
        let h = hover.unwrap();
        assert!(h.content.contains("model_id"));
        assert!(h.content.contains("opus"));
    }

    #[test]
    fn hover_not_command() {
        let svc = CommandLineService::new();
        let hover = svc.hover("hello world", Position::new(0, 3));
        assert!(hover.is_none());
    }

    // ── /models subcommand tests ──

    #[test]
    fn ghost_suffix_models_command() {
        // /models has subcommands, so it should show trailing space
        assert_eq!(ghost_suffix("/models"), Some(" ".into()));
    }

    #[test]
    fn ghost_suffix_models_subcommand() {
        // /models r → complete to "emove " (remove + space for its arg)
        assert_eq!(ghost_suffix("/models r"), Some("emove".into()));
    }

    #[test]
    fn completions_models_subcommands() {
        let svc = CommandLineService::new();
        let items = svc.completions("/models ", Position::new(0, 8));
        assert_eq!(items.len(), 2);
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"remove"));
        assert!(labels.contains(&"default"));
    }

    #[test]
    fn completions_models_subcommand_filtered() {
        let svc = CommandLineService::new();
        let items = svc.completions("/models d", Position::new(0, 9));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].label, "default");
    }
}
