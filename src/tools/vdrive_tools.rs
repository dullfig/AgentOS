//! VDrive-backed tool implementations — sandboxed versions of file-read,
//! file-write, file-edit, glob, and grep.
//!
//! Same WIT interfaces as the system tools. The LLM sees identical tool
//! schemas and response formats. The only difference: all file paths are
//! resolved through the VDrive sandbox.
//!
//! Each tool holds `Option<Arc<VDrive>>`. When `None` (no drive mounted),
//! every operation returns "no storage mounted — use /vdrive mount <path>".
//! This is the agent's only access to the filesystem.

use std::sync::Arc;
use tokio::sync::RwLock;

use async_trait::async_trait;
use rust_pipeline::prelude::*;

use agentos_vdrive::VDrive;

use super::{extract_tag, ToolPeer, ToolResponse};

/// Shared mount point. All VDrive tools reference the same slot.
/// When the user mounts/unmounts a drive, the slot is updated and
/// every tool sees the change immediately.
pub type DriveSlot = Arc<RwLock<Option<Arc<VDrive>>>>;

/// Create an empty (unmounted) drive slot.
pub fn empty_slot() -> DriveSlot {
    Arc::new(RwLock::new(None))
}

const NO_STORAGE: &str = "no storage mounted — use /vdrive mount <path> to mount a workspace";

/// Helper: read the drive from a slot, returning an error response if empty.
macro_rules! require_drive {
    ($slot:expr) => {{
        let guard = $slot.read().await;
        match guard.clone() {
            Some(d) => d,
            None => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(NO_STORAGE),
                });
            }
        }
    }};
}

// ── VDrive File Read ──

pub struct VDriveFileRead {
    slot: DriveSlot,
}

impl VDriveFileRead {
    pub fn new(slot: DriveSlot) -> Self {
        Self { slot }
    }
}

#[async_trait]
impl Handler for VDriveFileRead {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let drive = require_drive!(self.slot);
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let path = extract_tag(&xml_str, "path").unwrap_or_default();
        if path.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <path>"),
            });
        }

        let offset = extract_tag(&xml_str, "offset")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1);
        let limit = extract_tag(&xml_str, "limit")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2000);

        match drive.read_file(&path, offset, limit) {
            Ok(result) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::ok(&result.content),
            }),
            Err(e) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&e.to_string()),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for VDriveFileRead {
    fn name(&self) -> &str {
        "file-read"
    }

    fn wit(&self) -> &str {
        r#"
/// Read file contents with line numbers. Supports offset and limit for large files. Detects binary files.
interface file-read {
    record request {
        /// The file path to read
        path: string,
        /// Starting line number (1-based, default: 1)
        offset: option<u32>,
        /// Maximum lines to read (default: 2000)
        limit: option<u32>,
    }
    read: func(req: request) -> result<string, string>;
}
"#
    }
}

// ── VDrive File Write ──

pub struct VDriveFileWrite {
    slot: DriveSlot,
}

impl VDriveFileWrite {
    pub fn new(slot: DriveSlot) -> Self {
        Self { slot }
    }
}

#[async_trait]
impl Handler for VDriveFileWrite {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let drive = require_drive!(self.slot);
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let path = extract_tag(&xml_str, "path").unwrap_or_default();
        if path.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <path>"),
            });
        }

        let content = extract_tag(&xml_str, "content").unwrap_or_default();

        match drive.write_file(&path, &content) {
            Ok(()) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::ok(&format!(
                    "wrote {} bytes to {path}",
                    content.len()
                )),
            }),
            Err(e) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&e.to_string()),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for VDriveFileWrite {
    fn name(&self) -> &str {
        "file-write"
    }

    fn wit(&self) -> &str {
        r#"
/// Write or create a file. Auto-creates parent directories.
interface file-write {
    record request {
        /// The file path to write
        path: string,
        /// The content to write to the file
        content: string,
    }
    write: func(req: request) -> result<string, string>;
}
"#
    }
}

// ── VDrive File Edit ──

pub struct VDriveFileEdit {
    slot: DriveSlot,
}

impl VDriveFileEdit {
    pub fn new(slot: DriveSlot) -> Self {
        Self { slot }
    }
}

#[async_trait]
impl Handler for VDriveFileEdit {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let drive = require_drive!(self.slot);
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let path = extract_tag(&xml_str, "path").unwrap_or_default();
        if path.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <path>"),
            });
        }

        let old_string = extract_tag(&xml_str, "old_string").unwrap_or_default();
        if old_string.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <old_string>"),
            });
        }

        let new_string = extract_tag(&xml_str, "new_string").unwrap_or_default();

        // Read before edit to generate diff
        let before = match drive.read_file(&path, 1, usize::MAX) {
            Ok(r) => r.content,
            Err(e) => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&e.to_string()),
                });
            }
        };

        match drive.edit_file(&path, &old_string, &new_string, false) {
            Ok(()) => {
                // Generate diff
                let after = drive.read_file(&path, 1, usize::MAX)
                    .map(|r| r.content)
                    .unwrap_or_default();

                let diff = similar::TextDiff::from_lines(&before, &after);
                let mut diff_output = String::new();
                for change in diff.iter_all_changes() {
                    let sign = match change.tag() {
                        similar::ChangeTag::Delete => "-",
                        similar::ChangeTag::Insert => "+",
                        similar::ChangeTag::Equal => " ",
                    };
                    diff_output.push_str(&format!("{sign}{change}"));
                }

                Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::ok(&diff_output),
                })
            }
            Err(e) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&e.to_string()),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for VDriveFileEdit {
    fn name(&self) -> &str {
        "file-edit"
    }

    fn wit(&self) -> &str {
        r#"
/// Surgical text replacement in a file. Replaces old_string with new_string. The old_string must match exactly once. Returns unified diff.
interface file-edit {
    record request {
        /// The file path to edit
        path: string,
        /// The exact text to find and replace (must be unique in the file)
        old-string: string,
        /// The replacement text
        new-string: string,
    }
    edit: func(req: request) -> result<string, string>;
}
"#
    }
}

// ── VDrive Glob ──

pub struct VDriveGlob {
    slot: DriveSlot,
}

impl VDriveGlob {
    pub fn new(slot: DriveSlot) -> Self {
        Self { slot }
    }
}

const GLOB_MAX_RESULTS: usize = 1000;

#[async_trait]
impl Handler for VDriveGlob {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let drive = require_drive!(self.slot);
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let pattern = extract_tag(&xml_str, "pattern").unwrap_or_default();
        if pattern.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <pattern>"),
            });
        }

        match drive.glob(&pattern) {
            Ok(results) => {
                let total = results.len();
                let truncated = total > GLOB_MAX_RESULTS;
                let shown: Vec<&str> = results.iter()
                    .take(GLOB_MAX_RESULTS)
                    .map(|s| s.as_str())
                    .collect();

                let mut output = shown.join("\n");
                if truncated {
                    output.push_str(&format!("\n\n... ({total} total, showing first {GLOB_MAX_RESULTS})"));
                } else {
                    output.push_str(&format!("\n\n{total} files matched"));
                }

                Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::ok(&output),
                })
            }
            Err(e) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&e.to_string()),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for VDriveGlob {
    fn name(&self) -> &str {
        "glob"
    }

    fn wit(&self) -> &str {
        r#"
/// Find files matching a glob pattern (e.g. **/*.rs, src/*.txt).
interface glob {
    record request {
        /// The glob pattern to match (e.g. **/*.rs)
        pattern: string,
    }
    search: func(req: request) -> result<string, string>;
}
"#
    }
}

// ── VDrive Grep ──

pub struct VDriveGrep {
    slot: DriveSlot,
}

impl VDriveGrep {
    pub fn new(slot: DriveSlot) -> Self {
        Self { slot }
    }
}

const GREP_MAX_MATCHES: usize = 500;

#[async_trait]
impl Handler for VDriveGrep {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let drive = require_drive!(self.slot);
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let pattern = extract_tag(&xml_str, "pattern").unwrap_or_default();
        if pattern.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <pattern>"),
            });
        }

        let glob_filter = extract_tag(&xml_str, "glob_filter");
        let case_insensitive = extract_tag(&xml_str, "case_insensitive")
            .map(|s| s == "true")
            .unwrap_or(false);

        let re_pattern = if case_insensitive {
            format!("(?i){pattern}")
        } else {
            pattern
        };

        match drive.grep(&re_pattern, glob_filter.as_deref(), GREP_MAX_MATCHES) {
            Ok(matches) => {
                let total = matches.len();
                let mut output: String = matches.iter()
                    .map(|m| format!("{}:{}:{}", m.path, m.line_num, m.line))
                    .collect::<Vec<_>>()
                    .join("\n");

                if total >= GREP_MAX_MATCHES {
                    output.push_str(&format!("\n\n... (truncated at {GREP_MAX_MATCHES} matches)"));
                } else {
                    output.push_str(&format!("\n\n{total} matches"));
                }

                Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::ok(&output),
                })
            }
            Err(e) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&e.to_string()),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for VDriveGrep {
    fn name(&self) -> &str {
        "grep"
    }

    fn wit(&self) -> &str {
        r#"
/// Regex search across files. Recursively searches directories, skips binary files.
interface grep {
    record request {
        /// Regex pattern to search for
        pattern: string,
        /// Filter files by glob (e.g. *.rs)
        glob-filter: option<string>,
        /// Case insensitive search (default: false)
        case-insensitive: option<bool>,
    }
    search: func(req: request) -> result<string, string>;
}
"#
    }
}

// ── VDrive List Directory ──

pub struct VDriveListDir {
    slot: DriveSlot,
}

impl VDriveListDir {
    pub fn new(slot: DriveSlot) -> Self {
        Self { slot }
    }
}

#[async_trait]
impl Handler for VDriveListDir {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let drive = require_drive!(self.slot);
        let xml_str = String::from_utf8_lossy(&payload.xml);
        let path = extract_tag(&xml_str, "path").unwrap_or_else(|| ".".into());

        match drive.list_dir(&path) {
            Ok(entries) => {
                let mut output = String::new();
                for entry in &entries {
                    let kind = if entry.is_dir { "dir " } else { "file" };
                    output.push_str(&format!("{kind}  {}\n", entry.path));
                }
                output.push_str(&format!("\n{} entries", entries.len()));
                Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::ok(&output),
                })
            }
            Err(e) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&e.to_string()),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for VDriveListDir {
    fn name(&self) -> &str {
        "list-dir"
    }

    fn wit(&self) -> &str {
        r#"
/// List files and directories in a path.
interface list-dir {
    record request {
        /// The directory path to list (default: workspace root)
        path: option<string>,
    }
    list: func(req: request) -> result<string, string>;
}
"#
    }
}

// ── VDrive Command Exec ──

pub struct VDriveCommandExec {
    slot: DriveSlot,
    allowlist: Vec<String>,
}

const DEFAULT_ALLOWLIST: &[&str] = &[
    "cargo", "rustc", "npm", "node", "python", "git", "pip", "make", "just", "rustup",
    "wasm-tools", "ls", "dir", "echo", "where", "which", "tree", "rg", "curl", "mkdir",
];

const MAX_OUTPUT: usize = 100 * 1024;
const DEFAULT_TIMEOUT_SECS: u64 = 30;

impl VDriveCommandExec {
    pub fn new(slot: DriveSlot) -> Self {
        Self {
            slot,
            allowlist: DEFAULT_ALLOWLIST.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn is_allowed(&self, command: &str) -> bool {
        let first_token = command.split_whitespace().next().unwrap_or("");
        let exe_name = std::path::Path::new(first_token)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(first_token);

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

#[async_trait]
impl Handler for VDriveCommandExec {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let drive = require_drive!(self.slot);
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let command = extract_tag(&xml_str, "command").unwrap_or_default();
        if command.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <command>"),
            });
        }

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

        // Always run in the drive root — ignore user-supplied working_dir
        let work_dir = drive.root().to_path_buf();

        let mut cmd = if cfg!(windows) {
            let mut c = tokio::process::Command::new("cmd");
            c.args(["/C", &command]);
            c
        } else {
            let mut c = tokio::process::Command::new("sh");
            c.args(["-c", &command]);
            c
        };

        cmd.current_dir(&work_dir);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            cmd.output(),
        )
        .await;

        match result {
            Ok(Ok(output)) => {
                let stdout = Self::truncate_output(&String::from_utf8_lossy(&output.stdout));
                let stderr = Self::truncate_output(&String::from_utf8_lossy(&output.stderr));
                let exit_code = output.status.code().unwrap_or(-1);

                let response = format!(
                    "exit_code: {exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
                );

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
impl ToolPeer for VDriveCommandExec {
    fn name(&self) -> &str {
        "bash"
    }

    fn wit(&self) -> &str {
        r#"
/// Execute a shell command in the workspace. Only allowed commands can be run (cargo, git, npm, etc). Captures stdout, stderr, and exit code.
interface bash {
    record request {
        /// The command to execute
        command: string,
        /// Timeout in seconds (default: 30)
        timeout: option<u32>,
    }
    exec: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn make_ctx(name: &str) -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: name.into(),
        }
    }

    fn make_payload(xml: &str, tag: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: tag.into(),
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

    fn setup() -> (TempDir, DriveSlot) {
        let dir = TempDir::new().unwrap();
        let vd = Arc::new(VDrive::open(dir.path()).unwrap());
        let slot = Arc::new(RwLock::new(Some(vd)));
        (dir, slot)
    }

    // ── Read ──

    #[tokio::test]
    async fn vdrive_read_file() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("test.txt"), "hello\nworld\n").unwrap();

        let tool = VDriveFileRead::new(vd);
        let xml = "<FileReadRequest><path>test.txt</path></FileReadRequest>";
        let (ok, content) = get_result(tool.handle(make_payload(xml, "FileReadRequest"), make_ctx("file-read")).await.unwrap());
        assert!(ok);
        assert!(content.contains("1| hello"));
        assert!(content.contains("2| world"));
    }

    #[tokio::test]
    async fn vdrive_read_escape_blocked() {
        let (_dir, vd) = setup();
        let tool = VDriveFileRead::new(vd);
        let xml = "<FileReadRequest><path>../../etc/passwd</path></FileReadRequest>";
        let (ok, content) = get_result(tool.handle(make_payload(xml, "FileReadRequest"), make_ctx("file-read")).await.unwrap());
        assert!(!ok);
        assert!(content.contains("not found") || content.contains("escapes"));
    }

    // ── Write ──

    #[tokio::test]
    async fn vdrive_write_file() {
        let (dir, vd) = setup();
        let tool = VDriveFileWrite::new(vd);
        let xml = "<FileWriteRequest><path>output.txt</path><content>hello world</content></FileWriteRequest>";
        let (ok, _) = get_result(tool.handle(make_payload(xml, "FileWriteRequest"), make_ctx("file-write")).await.unwrap());
        assert!(ok);
        assert_eq!(fs::read_to_string(dir.path().join("output.txt")).unwrap(), "hello world");
    }

    #[tokio::test]
    async fn vdrive_write_creates_parents() {
        let (dir, vd) = setup();
        let tool = VDriveFileWrite::new(vd);
        let xml = "<FileWriteRequest><path>a/b/c/deep.rs</path><content>fn main() {}</content></FileWriteRequest>";
        let (ok, _) = get_result(tool.handle(make_payload(xml, "FileWriteRequest"), make_ctx("file-write")).await.unwrap());
        assert!(ok);
        assert_eq!(fs::read_to_string(dir.path().join("a/b/c/deep.rs")).unwrap(), "fn main() {}");
    }

    #[tokio::test]
    async fn vdrive_write_escape_blocked() {
        let (_dir, vd) = setup();
        let tool = VDriveFileWrite::new(vd);
        let xml = "<FileWriteRequest><path>../../escape.txt</path><content>bad</content></FileWriteRequest>";
        let (ok, _) = get_result(tool.handle(make_payload(xml, "FileWriteRequest"), make_ctx("file-write")).await.unwrap());
        assert!(!ok);
    }

    // ── Edit ──

    #[tokio::test]
    async fn vdrive_edit_file() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("code.rs"), "fn foo() {}\nfn bar() {}\n").unwrap();

        let tool = VDriveFileEdit::new(vd);
        let xml = "<FileEditRequest><path>code.rs</path><old_string>fn foo()</old_string><new_string>fn baz()</new_string></FileEditRequest>";
        let (ok, diff) = get_result(tool.handle(make_payload(xml, "FileEditRequest"), make_ctx("file-edit")).await.unwrap());
        assert!(ok);
        assert!(diff.contains("-") || diff.contains("+"));
        assert!(fs::read_to_string(dir.path().join("code.rs")).unwrap().contains("fn baz()"));
    }

    // ── Glob ──

    #[tokio::test]
    async fn vdrive_glob_finds_files() {
        let (dir, vd) = setup();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/main.rs"), "").unwrap();
        fs::write(dir.path().join("src/lib.rs"), "").unwrap();
        fs::write(dir.path().join("README.md"), "").unwrap();

        let tool = VDriveGlob::new(vd);
        let xml = "<GlobRequest><pattern>src/*.rs</pattern></GlobRequest>";
        let (ok, content) = get_result(tool.handle(make_payload(xml, "GlobRequest"), make_ctx("glob")).await.unwrap());
        assert!(ok);
        assert!(content.contains("main.rs"));
        assert!(content.contains("lib.rs"));
        assert!(!content.contains("README"));
        assert!(content.contains("2 files matched"));
    }

    // ── Grep ──

    #[tokio::test]
    async fn vdrive_grep_finds_matches() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("a.txt"), "hello world\ngoodbye\nhello again\n").unwrap();

        let tool = VDriveGrep::new(vd);
        let xml = "<GrepRequest><pattern>hello</pattern></GrepRequest>";
        let (ok, content) = get_result(tool.handle(make_payload(xml, "GrepRequest"), make_ctx("grep")).await.unwrap());
        assert!(ok);
        assert!(content.contains("a.txt:1:hello world"));
        assert!(content.contains("a.txt:3:hello again"));
        assert!(content.contains("2 matches"));
    }

    // ── List Dir ──

    #[tokio::test]
    async fn vdrive_list_dir() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("a.txt"), "").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();

        let tool = VDriveListDir::new(vd);
        let xml = "<ListDirRequest><path>.</path></ListDirRequest>";
        let (ok, content) = get_result(tool.handle(make_payload(xml, "ListDirRequest"), make_ctx("list-dir")).await.unwrap());
        assert!(ok);
        assert!(content.contains("a.txt"));
        assert!(content.contains("sub"));
        assert!(content.contains("2 entries"));
    }

    // ── Metadata ──

    #[test]
    fn vdrive_tools_have_correct_names() {
        let slot = empty_slot();

        assert_eq!(VDriveFileRead::new(slot.clone()).name(), "file-read");
        assert_eq!(VDriveFileWrite::new(slot.clone()).name(), "file-write");
        assert_eq!(VDriveFileEdit::new(slot.clone()).name(), "file-edit");
        assert_eq!(VDriveGlob::new(slot.clone()).name(), "glob");
        assert_eq!(VDriveGrep::new(slot.clone()).name(), "grep");
        assert_eq!(VDriveListDir::new(slot).name(), "list-dir");
    }

    #[test]
    fn vdrive_tools_have_valid_wit() {
        let slot = empty_slot();

        // All WIT interfaces should parse successfully
        for tool in [
            VDriveFileRead::new(slot.clone()).wit(),
            VDriveFileWrite::new(slot.clone()).wit(),
            VDriveFileEdit::new(slot.clone()).wit(),
            VDriveGlob::new(slot.clone()).wit(),
            VDriveGrep::new(slot.clone()).wit(),
            VDriveListDir::new(slot).wit(),
        ] {
            let iface = crate::wit::parser::parse_wit(tool).unwrap();
            assert!(!iface.name.is_empty());
        }
    }

    #[tokio::test]
    async fn vdrive_no_storage_returns_error() {
        let slot = empty_slot();
        let tool = VDriveFileRead::new(slot);
        let xml = "<FileReadRequest><path>test.txt</path></FileReadRequest>";
        let (ok, content) = get_result(
            tool.handle(make_payload(xml, "FileReadRequest"), make_ctx("file-read"))
                .await
                .unwrap(),
        );
        assert!(!ok);
        assert!(content.contains("no storage mounted"));
    }
}
