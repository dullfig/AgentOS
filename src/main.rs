use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing::info;

use agentos::config::{AgentsConfig, ModelsConfig};
use agentos::llm::LlmPool;
use agentos::organism::parser::parse_organism;
use agentos::pipeline::AgentPipelineBuilder;
use agentos::tools::vdrive_tools::{
    self, VDriveFileRead, VDriveFileWrite, VDriveFileEdit,
    VDriveGlob, VDriveGrep, VDriveListDir, VDriveCommandExec,
};
use agentos::tui::runner::run_tui;

/// Default organism configuration embedded in the binary.
const DEFAULT_ORGANISM: &str = r#"
organism:
  name: agentos

prompts:
  coding_base: |
    You are a coding agent running inside AgentOS. You have access to tools for file operations,
    shell commands, and codebase indexing. Use these tools to complete the task you've been given.

    Your output is rendered in a TUI with full markdown support. You can use:
    - Headings, bold, italic, code blocks (with syntax highlighting)
    - Pipe-delimited markdown tables (rendered as box-drawing art)
    - D2 diagrams in fenced code blocks (```d2) for architecture diagrams, flowcharts, and relationships

    Rules:
    1. Read before you write. Always understand existing code before modifying it.
    2. Make the smallest change that solves the problem.
    3. Test your changes when possible (run tests, verify output).
    4. If a tool call fails, analyze the error and try a different approach.
    5. When done, provide a clear summary of what you did.

    {tool_definitions}

  no_paperclipper: |
    You are bounded. You do not pursue goals beyond your task.
    You report uncertainty rather than improvising.

listeners:
  - name: coding-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "AI coding agent"
    agent:
      prompt: "no_paperclipper & coding_base"
      max_tokens: 4096
      max_agentic_iterations: 25
      permissions:
        file-read: auto
        list-dir: auto
        glob: auto
        grep: auto
        codebase-index: auto
        file-write: prompt
        file-edit: prompt
        command-exec: prompt
    librarian: true
    peers: [file-read, file-write, file-edit, glob, grep, list-dir, command-exec, codebase-index]

  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM inference pool"
    librarian: true
    ports:
      - port: 443
        direction: outbound
        protocol: https
        hosts: [api.anthropic.com]

  - name: librarian
    payload_class: librarian.LibrarianRequest
    handler: librarian.handle
    description: "Context curator"
    peers: [llm-pool]

  - name: codebase-index
    payload_class: treesitter.CodeIndexRequest
    handler: treesitter.handle
    description: "Tree-sitter code indexing"

  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"

  - name: file-write
    payload_class: tools.FileWriteRequest
    handler: tools.file_write.handle
    description: "Write files"

  - name: file-edit
    payload_class: tools.FileEditRequest
    handler: tools.file_edit.handle
    description: "Edit files"

  - name: glob
    payload_class: tools.GlobRequest
    handler: tools.glob.handle
    description: "Glob search"

  - name: grep
    payload_class: tools.GrepRequest
    handler: tools.grep.handle
    description: "Grep search"

  - name: list-dir
    payload_class: tools.ListDirRequest
    handler: tools.list_dir.handle
    description: "List directory contents"

  - name: command-exec
    payload_class: tools.CommandExecRequest
    handler: tools.command_exec.handle
    description: "Command execution"

profiles:
  coding:
    linux_user: agentos
    listeners: [coding-agent, file-read, file-write, file-edit, glob, grep, list-dir, command-exec, codebase-index, llm-pool, librarian]
    network: [llm-pool]
    journal: retain_forever
"#;

/// Extension trait to convert Result<T, String> to anyhow::Result<T>.
trait ToAnyhow<T> {
    fn to_anyhow(self) -> Result<T>;
}

impl<T> ToAnyhow<T> for std::result::Result<T, String> {
    fn to_anyhow(self) -> Result<T> {
        self.map_err(|e| anyhow::anyhow!("{e}"))
    }
}

#[derive(Parser)]
#[command(name = "agentos", about = "An operating system for AI coding agents. No compaction, ever.")]
struct Cli {
    /// Working directory (defaults to current)
    #[arg(short, long)]
    dir: Option<String>,

    /// Model to use (default: sonnet → claude-sonnet-4-6)
    #[arg(short, long)]
    model: Option<String>,

    /// Path to organism.yaml (default: embedded)
    #[arg(short, long)]
    organism: Option<String>,

    /// Kernel data directory (default: .agentos/)
    #[arg(long)]
    data: Option<String>,

    /// Enable debug tab (activity trace, diagnostics)
    #[arg(long)]
    debug: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI
    let cli = Cli::parse();
    let debug = cli.debug;
    let work_dir = cli.dir.unwrap_or_else(|| ".".into());
    let model = cli
        .model
        .unwrap_or_else(|| "sonnet".into());
    let data_rel = cli.data.unwrap_or_else(|| ".agentos".into());
    let data_dir = PathBuf::from(&work_dir).join(&data_rel);

    // Set working directory
    std::env::set_current_dir(&work_dir)?;

    // Initialize tracing to file (avoid polluting the TUI)
    let log_dir = PathBuf::from(&data_rel);
    std::fs::create_dir_all(&log_dir)?;
    let log_file = std::fs::File::create(log_dir.join("agentos.log"))?;
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("agentos=info".parse()?),
        )
        .with_writer(log_file)
        .with_ansi(false)
        .init();

    info!("AgentOS starting in {work_dir}");

    // Auto-mount CWD as the agent's workspace if it looks like a project directory
    let drive_slot = vdrive_tools::empty_slot();
    let auto_mount_msg = try_auto_mount(&drive_slot);

    // Parse organism config
    let yaml = if let Some(ref path) = cli.organism {
        std::fs::read_to_string(path)?
    } else {
        DEFAULT_ORGANISM.to_string()
    };
    let org = parse_organism(&yaml).to_anyhow()?;

    // Load models config (user + project + env fallback)
    let models_config = ModelsConfig::load();

    // Load agents favorites (project-level)
    let agents_config = AgentsConfig::load();

    // Create LLM pool: config first, env var fallback. None = no key yet (user configures via TUI).
    let pool = if models_config.has_models() {
        match LlmPool::from_config(&models_config) {
            Ok(p) => {
                info!("Using models from config file");
                Some(p)
            }
            Err(e) => {
                info!("Config exists but pool creation failed: {e}");
                None
            }
        }
    } else {
        match LlmPool::from_env(&model) {
            Ok(p) => {
                info!("Using ANTHROPIC_API_KEY from env");
                Some(p)
            }
            Err(e) => {
                info!("No API key available: {e}");
                None
            }
        }
    };

    info!("Building pipeline with model {model}");

    // Build pipeline — LLM pool is optional (user may configure via TUI)
    let has_pool = pool.is_some();
    let mut builder = AgentPipelineBuilder::new(org, &data_dir).with_debug(debug);
    if let Some(p) = pool {
        builder = builder
            .with_llm_pool(p)
            .to_anyhow()?
            .with_librarian()
            .to_anyhow()?;
    }
    // Try to load local inference engine (optional — graceful if missing)
    builder = builder.with_local_inference().to_anyhow()?;
    let slot = drive_slot.clone();
    let mut pipeline = builder
        .with_code_index()
        .to_anyhow()?
        .register_tool("file-read", VDriveFileRead::new(slot.clone()))
        .to_anyhow()?
        .register_tool("file-write", VDriveFileWrite::new(slot.clone()))
        .to_anyhow()?
        .register_tool("file-edit", VDriveFileEdit::new(slot.clone()))
        .to_anyhow()?
        .register_tool("glob", VDriveGlob::new(slot.clone()))
        .to_anyhow()?
        .register_tool("grep", VDriveGrep::new(slot.clone()))
        .to_anyhow()?
        .register_tool("list-dir", VDriveListDir::new(slot.clone()))
        .to_anyhow()?
        .register_tool("command-exec", VDriveCommandExec::new(slot))
        .to_anyhow()?
        .with_buffer_nodes(&PathBuf::from(&work_dir))
        .to_anyhow()?;
    if has_pool {
        pipeline = pipeline.with_agents().to_anyhow()?;
    }
    let mut pipeline = pipeline.build().to_anyhow()?;

    // Initialize root thread
    pipeline
        .initialize_root("agentos", "coding")
        .await
        .to_anyhow()?;

    info!("Pipeline ready, starting TUI");

    // Start pipeline
    pipeline.run();

    // Run TUI (blocks until quit)
    run_tui(&mut pipeline, debug, &yaml, models_config, agents_config, has_pool, drive_slot, auto_mount_msg).await?;

    // Shutdown
    info!("Shutting down");
    pipeline.shutdown().await;

    Ok(())
}

/// Try to auto-mount the working directory as the agent's workspace.
/// Returns a message to show in the TUI on startup, or None if no mount.
fn try_auto_mount(
    slot: &vdrive_tools::DriveSlot,
) -> Option<String> {
    use std::path::Path;

    let cwd = match std::env::current_dir() {
        Ok(d) => d,
        Err(_) => return None,
    };

    let canonical = match cwd.canonicalize() {
        Ok(c) => c,
        Err(_) => return None,
    };

    let s = canonical.to_string_lossy();

    // Don't auto-mount if CWD is AgentOS's own source directory
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));
    if let Some(ref ed) = exe_dir {
        if canonical.starts_with(ed) {
            return None;
        }
    }

    // Don't auto-mount sensitive paths (roots, home, system dirs)
    let normalized = s.replace('\\', "/");
    let lower = normalized.to_lowercase();

    // Filesystem roots
    if lower == "/" || (lower.len() == 3 && lower.ends_with(":/")) {
        return None;
    }

    // System directories
    if lower.starts_with("c:/windows") || lower.starts_with("c:/program files") {
        return None;
    }

    // Home directory itself (not subdirectories)
    if let Ok(home) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        let home_normalized = home.replace('\\', "/");
        if normalized.trim_end_matches('/') == home_normalized.trim_end_matches('/') {
            return None;
        }
    }

    // Don't auto-mount if --dir was explicitly "." and we're in the cargo project dir
    // (i.e., Cargo.toml exists and package.name == "agentos")
    if Path::new("Cargo.toml").exists() {
        if let Ok(content) = std::fs::read_to_string("Cargo.toml") {
            if content.contains("name = \"agentos\"") {
                return None;
            }
        }
    }

    // Mount it
    match agentos::vdrive::mount(&canonical) {
        Ok(drive) => {
            let name = drive.name().to_string();
            let root = drive.root().display().to_string();
            if let Ok(mut guard) = slot.try_write() {
                *guard = Some(drive);
                Some(format!("Workspace: {name} ({root})"))
            } else {
                None
            }
        }
        Err(_) => None,
    }
}
