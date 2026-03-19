//! Safe command tools — virtualized shell commands with constrained interfaces.
//!
//! Each safe command tool wraps a single executable + subcommand with a fixed
//! prefix. The agent can pass additional arguments but cannot change the
//! base command. This eliminates shell injection and enables granular
//! permission tiers:
//!
//!   - `cargo-test` (auto) vs `git-push` (prompt) vs `bash` (always-prompt)
//!
//! The `SafeCommandTool` struct is the framework; each tool is a thin config.
//! WIT interfaces are auto-generated from the tool definition.

use std::time::Duration;

use async_trait::async_trait;
use rust_pipeline::prelude::*;

use super::vdrive_tools::DriveSlot;
use super::{extract_tag, ToolPeer, ToolResponse};

/// Maximum output size before truncation.
const MAX_OUTPUT: usize = 100 * 1024;

/// Configuration for a safe command tool.
pub struct SafeCommandDef {
    /// Tool name (e.g., "cargo-test").
    pub name: &'static str,
    /// Description for the LLM.
    pub description: &'static str,
    /// The executable to run (e.g., "cargo").
    pub executable: &'static str,
    /// Fixed argument prefix (e.g., &["test"] for cargo test).
    pub fixed_args: &'static [&'static str],
    /// Whether the user/agent can pass additional arguments.
    pub user_args: bool,
    /// Timeout in seconds.
    pub timeout_secs: u64,
}

/// A safe command tool instance bound to a VDrive.
pub struct SafeCommandTool {
    def: &'static SafeCommandDef,
    slot: DriveSlot,
}

impl SafeCommandTool {
    pub fn new(def: &'static SafeCommandDef, slot: DriveSlot) -> Self {
        Self { def, slot }
    }
}

// ── Macro to reduce require_drive! duplication ──

macro_rules! require_drive {
    ($slot:expr) => {{
        let guard = $slot.read().await;
        match guard.as_ref() {
            Some(drive) => drive.clone(),
            None => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(
                        "no storage mounted — use /vdrive mount <path> to mount a workspace",
                    ),
                })
            }
        }
    }};
}

#[async_trait]
impl Handler for SafeCommandTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let drive = require_drive!(self.slot);
        let xml_str = String::from_utf8_lossy(&payload.xml);

        // Build the command: executable + fixed_args + optional user args
        let mut args: Vec<String> = self
            .def
            .fixed_args
            .iter()
            .map(|s| s.to_string())
            .collect();

        if self.def.user_args {
            if let Some(extra) = extract_tag(&xml_str, "args") {
                let extra = extra.trim();
                if !extra.is_empty() {
                    // Split on whitespace — no shell interpretation
                    args.extend(extra.split_whitespace().map(|s| s.to_string()));
                }
            }
        }

        let work_dir = drive.root().to_path_buf();

        let mut cmd = tokio::process::Command::new(self.def.executable);
        cmd.args(&args);
        cmd.current_dir(&work_dir);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let result = tokio::time::timeout(
            Duration::from_secs(self.def.timeout_secs),
            cmd.output(),
        )
        .await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let exit_code = output.status.code().unwrap_or(-1);

                let combined = if stderr.is_empty() {
                    format!("exit code: {exit_code}\n{stdout}")
                } else {
                    format!("exit code: {exit_code}\n{stdout}\nstderr:\n{stderr}")
                };

                let truncated = truncate_output(&combined);

                if output.status.success() {
                    Ok(HandlerResponse::Reply {
                        payload_xml: ToolResponse::ok(&truncated),
                    })
                } else {
                    Ok(HandlerResponse::Reply {
                        payload_xml: ToolResponse::err(&truncated),
                    })
                }
            }
            Ok(Err(e)) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "failed to execute {}: {e}",
                    self.def.executable
                )),
            }),
            Err(_) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "{} timed out after {} seconds",
                    self.def.name, self.def.timeout_secs
                )),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for SafeCommandTool {
    fn name(&self) -> &str {
        self.def.name
    }

    fn wit(&self) -> &str {
        self.def.wit()
    }
}

impl SafeCommandDef {
    /// Generate a WIT interface string for this command.
    ///
    /// Stored as a leaked &'static str so it can be returned from wit().
    /// (Called once per tool at registration time.)
    fn wit(&self) -> &'static str {
        let args_field = if self.user_args {
            format!(
                "        /// Additional arguments (e.g., \"--release\", test name)\n\
                 {}        args: option<string>,\n",
                ""
            )
        } else {
            String::new()
        };

        let wit = format!(
            r#"
/// {description}
interface {name} {{
    record request {{
{args_field}    }}
    run: func(req: request) -> result<string, string>;
}}
"#,
            description = self.description,
            name = self.name,
            args_field = args_field,
        );

        // Leak to get 'static lifetime — only called once per tool at startup
        Box::leak(wit.into_boxed_str())
    }
}

fn truncate_output(s: &str) -> String {
    if s.len() > MAX_OUTPUT {
        format!(
            "{}...\n(truncated at {} bytes)",
            &s[..MAX_OUTPUT],
            MAX_OUTPUT
        )
    } else {
        s.to_string()
    }
}

// ═══════════════════════════════════════════════════════════
// Tool definitions — add new virtualized commands here
// ═══════════════════════════════════════════════════════════

pub static CARGO_TEST: SafeCommandDef = SafeCommandDef {
    name: "cargo-test",
    description: "Run cargo test. Pass additional args like test name or --release.",
    executable: "cargo",
    fixed_args: &["test"],
    user_args: true,
    timeout_secs: 300,
};

pub static CARGO_BUILD: SafeCommandDef = SafeCommandDef {
    name: "cargo-build",
    description: "Run cargo build. Pass additional args like --release.",
    executable: "cargo",
    fixed_args: &["build"],
    user_args: true,
    timeout_secs: 300,
};

pub static CARGO_CHECK: SafeCommandDef = SafeCommandDef {
    name: "cargo-check",
    description: "Run cargo check (type-check without building). Fast compilation validation.",
    executable: "cargo",
    fixed_args: &["check"],
    user_args: true,
    timeout_secs: 120,
};

pub static CARGO_CLIPPY: SafeCommandDef = SafeCommandDef {
    name: "cargo-clippy",
    description: "Run cargo clippy linter. Reports code quality warnings.",
    executable: "cargo",
    fixed_args: &["clippy"],
    user_args: true,
    timeout_secs: 300,
};

pub static GIT_STATUS: SafeCommandDef = SafeCommandDef {
    name: "git-status",
    description: "Show git working tree status. No arguments needed.",
    executable: "git",
    fixed_args: &["status"],
    user_args: false,
    timeout_secs: 10,
};

pub static GIT_DIFF: SafeCommandDef = SafeCommandDef {
    name: "git-diff",
    description: "Show git diff. Pass file paths or --staged for staged changes.",
    executable: "git",
    fixed_args: &["diff"],
    user_args: true,
    timeout_secs: 30,
};

pub static GIT_LOG: SafeCommandDef = SafeCommandDef {
    name: "git-log",
    description: "Show git commit log. Pass args like --oneline, -n 10, or a path.",
    executable: "git",
    fixed_args: &["log"],
    user_args: true,
    timeout_secs: 10,
};

pub static GIT_ADD: SafeCommandDef = SafeCommandDef {
    name: "git-add",
    description: "Stage files for commit. Pass file paths to add.",
    executable: "git",
    fixed_args: &["add"],
    user_args: true,
    timeout_secs: 10,
};

pub static GIT_COMMIT: SafeCommandDef = SafeCommandDef {
    name: "git-commit",
    description: "Create a git commit. Pass -m \"message\" for the commit message.",
    executable: "git",
    fixed_args: &["commit"],
    user_args: true,
    timeout_secs: 30,
};

pub static GIT_PUSH: SafeCommandDef = SafeCommandDef {
    name: "git-push",
    description: "Push commits to remote. Irreversible — affects shared repository.",
    executable: "git",
    fixed_args: &["push"],
    user_args: true,
    timeout_secs: 60,
};

/// All available safe command definitions, for bulk registration.
pub static ALL_SAFE_COMMANDS: &[&SafeCommandDef] = &[
    &CARGO_TEST,
    &CARGO_BUILD,
    &CARGO_CHECK,
    &CARGO_CLIPPY,
    &GIT_STATUS,
    &GIT_DIFF,
    &GIT_LOG,
    &GIT_ADD,
    &GIT_COMMIT,
    &GIT_PUSH,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wit_generation_with_args() {
        let wit = CARGO_TEST.wit();
        assert!(wit.contains("interface cargo-test"), "got: {wit}");
        assert!(wit.contains("args: option<string>"), "should have args field");
        assert!(wit.contains("Run cargo test"), "should have description");
    }

    #[test]
    fn wit_generation_without_args() {
        let wit = GIT_STATUS.wit();
        assert!(wit.contains("interface git-status"), "got: {wit}");
        assert!(!wit.contains("args:"), "should not have args field");
    }

    #[test]
    fn all_commands_have_unique_names() {
        let mut names = std::collections::HashSet::new();
        for def in ALL_SAFE_COMMANDS {
            assert!(names.insert(def.name), "duplicate tool name: {}", def.name);
        }
    }

    #[test]
    fn all_commands_generate_valid_wit() {
        for def in ALL_SAFE_COMMANDS {
            let wit = def.wit();
            assert!(wit.contains(&format!("interface {}", def.name)),
                "bad WIT for {}: {}", def.name, wit);
        }
    }

    #[tokio::test]
    async fn no_drive_returns_error() {
        let slot = crate::vdrive_tools::empty_slot();
        let tool = SafeCommandTool::new(&CARGO_CHECK, slot);
        let payload = ValidatedPayload {
            xml: b"<CargoCheckRequest></CargoCheckRequest>".to_vec(),
            tag: "CargoCheckRequest".into(),
        };
        let ctx = HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "cargo-check".into(),
        };
        let result = tool.handle(payload, ctx).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("no storage mounted"), "got: {xml}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[test]
    fn truncate_output_short() {
        let s = "hello";
        assert_eq!(truncate_output(s), "hello");
    }

    #[test]
    fn truncate_output_long() {
        let s = "x".repeat(MAX_OUTPUT + 100);
        let t = truncate_output(&s);
        assert!(t.contains("truncated"));
        assert!(t.len() < s.len());
    }
}
