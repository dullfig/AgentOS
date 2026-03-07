//! CommandExecTool — execute allowed commands with allowlist, timeout, capture.

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use std::time::Duration;
use tokio::process::Command;

use super::{extract_tag, ToolPeer, ToolResponse};

const MAX_OUTPUT: usize = 100 * 1024; // 100KB
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Default command allowlist.
const DEFAULT_ALLOWLIST: &[&str] = &[
    "cargo", "rustc", "npm", "node", "python", "git", "pip", "make", "just", "rustup",
    "wasm-tools", "ls", "dir", "echo", "where", "which", "tree", "rg", "curl", "mkdir",
];

/// Execute allowed shell commands with timeout and output capture.
pub struct CommandExecTool {
    allowlist: Vec<String>,
}

impl CommandExecTool {
    /// Create with default allowlist.
    pub fn new() -> Self {
        Self {
            allowlist: DEFAULT_ALLOWLIST.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Create with custom allowlist.
    pub fn with_allowlist(allowlist: Vec<String>) -> Self {
        Self { allowlist }
    }

    /// Check if a command's first token is in the allowlist.
    fn is_allowed(&self, command: &str) -> bool {
        let first_token = command.split_whitespace().next().unwrap_or("");
        // Extract just the executable name (strip path)
        let exe_name = std::path::Path::new(first_token)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(first_token);

        // Case-insensitive on Windows, case-sensitive otherwise
        self.allowlist.iter().any(|allowed| {
            if cfg!(windows) {
                exe_name.eq_ignore_ascii_case(allowed)
                    || exe_name
                        .strip_suffix(".exe")
                        .map(|s| s.eq_ignore_ascii_case(allowed))
                        .unwrap_or(false)
            } else {
                exe_name == allowed.as_str()
            }
        })
    }

    fn truncate_output(s: &str) -> String {
        if s.len() > MAX_OUTPUT {
            format!("{}...\n(truncated at {} bytes)", &s[..MAX_OUTPUT], MAX_OUTPUT)
        } else {
            s.to_string()
        }
    }
}

impl Default for CommandExecTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Handler for CommandExecTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let command = extract_tag(&xml_str, "command").unwrap_or_default();
        if command.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <command>"),
            });
        }

        // Allowlist check
        if !self.is_allowed(&command) {
            let first = command.split_whitespace().next().unwrap_or("(empty)");
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "command not allowed: {first}. Allowed: {}",
                    self.allowlist.join(", ")
                )),
            });
        }

        let timeout_secs = extract_tag(&xml_str, "timeout")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        let working_dir = extract_tag(&xml_str, "working_dir");

        // Build the command
        let mut cmd = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", &command]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", &command]);
            c
        };

        if let Some(ref dir) = working_dir {
            cmd.current_dir(dir);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        // Execute with timeout
        let result = tokio::time::timeout(Duration::from_secs(timeout_secs), cmd.output()).await;

        match result {
            Ok(Ok(output)) => {
                let stdout = Self::truncate_output(&String::from_utf8_lossy(&output.stdout));
                let stderr = Self::truncate_output(&String::from_utf8_lossy(&output.stderr));
                let exit_code = output.status.code().unwrap_or(-1);

                let response = format!(
                    "exit_code: {exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
                );

                // Report both success and non-zero exit as OK — the caller
                // sees the exit_code in the response text and decides.
                Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::ok(&response),
                })
            }
            Ok(Err(e)) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("execution error: {e}")),
            }),
            Err(_) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "command timed out after {timeout_secs}s: {command}"
                )),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for CommandExecTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn wit(&self) -> &str {
        r#"
/// Execute a shell command. Only allowed commands can be run (cargo, git, npm, etc). Captures stdout, stderr, and exit code.
interface bash {
    record request {
        /// The command to execute
        command: string,
        /// Timeout in seconds (default: 30)
        timeout: option<u32>,
        /// Working directory for the command
        working-dir: option<string>,
    }
    exec: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "command-exec".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "CommandExecRequest".into(),
        }
    }

    fn get_result(resp: HandlerResponse) -> (bool, String) {
        match resp {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                let success = xml.contains("<success>true</success>");
                let content = if success {
                    extract_tag(&xml, "result").unwrap_or_default()
                } else {
                    extract_tag(&xml, "error").unwrap_or_default()
                };
                (success, content)
            }
            _ => panic!("expected Reply"),
        }
    }

    #[test]
    fn allowlist_check_allowed() {
        let tool = CommandExecTool::new();
        assert!(tool.is_allowed("echo hello"));
        assert!(tool.is_allowed("cargo test"));
        assert!(tool.is_allowed("git status"));
        assert!(tool.is_allowed("ls -la"));
    }

    #[test]
    fn allowlist_check_blocked() {
        let tool = CommandExecTool::new();
        assert!(!tool.is_allowed("rm -rf /"));
        assert!(!tool.is_allowed("shutdown /s"));
        assert!(!tool.is_allowed("format C:"));
        assert!(!tool.is_allowed("powershell malicious"));
    }

    #[test]
    fn custom_allowlist() {
        let tool = CommandExecTool::with_allowlist(vec!["myapp".into()]);
        assert!(tool.is_allowed("myapp --flag"));
        assert!(!tool.is_allowed("echo hello"));
    }

    #[tokio::test]
    async fn exec_echo() {
        let tool = CommandExecTool::new();
        let xml = "<CommandExecRequest><command>echo hello world</command></CommandExecRequest>";
        let (ok, content) = get_result(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("exit_code: 0"));
        assert!(content.contains("hello world"));
    }

    #[tokio::test]
    async fn exec_blocked_command() {
        let tool = CommandExecTool::new();
        let xml = "<CommandExecRequest><command>rm -rf /</command></CommandExecRequest>";
        let (ok, content) = get_result(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("not allowed"));
    }

    #[tokio::test]
    async fn exec_with_working_dir() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "found").unwrap();

        let work_dir = dir.path().to_str().unwrap();
        let cmd = if cfg!(windows) { "dir" } else { "ls" };
        let xml = format!(
            "<CommandExecRequest><command>{cmd}</command><working_dir>{work_dir}</working_dir></CommandExecRequest>"
        );
        let tool = CommandExecTool::new();
        let (ok, content) = get_result(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("marker.txt"));
    }

    #[tokio::test]
    async fn exec_missing_command() {
        let tool = CommandExecTool::new();
        let xml = "<CommandExecRequest></CommandExecRequest>";
        let (ok, content) = get_result(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("missing required"));
    }

    #[tokio::test]
    async fn exec_captures_stderr() {
        let tool = CommandExecTool::new();
        // Use a command that writes to stderr
        let cmd = if cfg!(windows) {
            "echo error_output 1>&2"
        } else {
            "echo error_output >&2"
        };
        let xml = format!("<CommandExecRequest><command>{cmd}</command></CommandExecRequest>");
        let (ok, content) = get_result(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("error_output"));
    }

    #[tokio::test]
    async fn exec_nonzero_exit() {
        let tool = CommandExecTool::new();
        let cmd = if cfg!(windows) {
            "cmd /C exit 42"
        } else {
            "echo fail && exit 42"
        };
        // The cmd wrapper means the first token check matters
        // We need a command whose first token is allowed
        let _xml = format!("<CommandExecRequest><command>{cmd}</command></CommandExecRequest>");

        // On Windows, "cmd" is not in the allowlist, but our shell wraps it anyway.
        // Let's use a valid allowed command that exits nonzero
        let xml2 = "<CommandExecRequest><command>git log --nonexistent-flag</command></CommandExecRequest>";
        let (ok, content) = get_result(tool.handle(make_payload(xml2), make_ctx()).await.unwrap());
        assert!(ok); // We still return ok with exit_code info
        assert!(content.contains("exit_code:"));
    }

    #[cfg(windows)]
    #[test]
    fn allowlist_case_insensitive_windows() {
        let tool = CommandExecTool::new();
        assert!(tool.is_allowed("ECHO hello"));
        assert!(tool.is_allowed("Echo hello"));
        assert!(tool.is_allowed("echo.exe hello"));
    }

    #[test]
    fn default_allowlist_comprehensive() {
        let tool = CommandExecTool::new();
        for cmd in DEFAULT_ALLOWLIST {
            assert!(tool.is_allowed(&format!("{cmd} --version")), "should allow: {cmd}");
        }
    }

    #[test]
    fn command_exec_metadata() {
        let tool = CommandExecTool::new();
        assert_eq!(tool.name(), "bash");
        let iface = crate::wit::parser::parse_wit(tool.wit()).unwrap();
        assert_eq!(iface.name, "bash");
        assert_eq!(iface.request_tag(), "BashRequest");
        assert!(iface.request.fields.iter().any(|f| f.name == "command"));
    }
}
