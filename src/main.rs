use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing::info;

use agentos::config::{AgentsConfig, ModelsConfig};
use agentos::llm::LlmPool;
use agentos::organism::Organism;
use agentos::organism::parser::parse_organism;
use agentos::pipeline::{AgentPipeline, AgentPipelineBuilder};
use agentos::tools::compile_wasm::CompileWasmTool;
use agentos::tools::list_agents::ListAgentsTool;
use agentos::tools::safe_commands::{SafeCommandTool, ALL_SAFE_COMMANDS};
use agentos::tools::user_channel::UserChannelHandler;
use agentos::tools::validate_organism::ValidateOrganismTool;
use agentos::tools::vdrive_tools::{
    self, DriveSlot, VDriveFileRead, VDriveFileWrite, VDriveFileEdit,
    VDriveGlob, VDriveGrep, VDriveListDir, VDriveCommandExec,
};
use agentos::tui::runner::run_tui;

/// Default organism configuration embedded in the binary.
const DEFAULT_ORGANISM: &str = r#"
organism:
  name: agentos

prompts:
  no_paperclipper: |
    You are bounded. You do not pursue goals beyond your task.
    You report uncertainty rather than improvising.

  bob_base: |
    You are Bob, the AgentOS concierge. You help users accomplish tasks by
    understanding their intent and routing to the right specialist.

    Your specialists:
    - **coder**: Writes code, edits files, runs tests, uses git. For any hands-on coding task.
    - **agent-expert**: Designs, validates, and diagnoses organism YAML configurations.
      For building new agents, fixing broken organisms, or modifying existing ones.
    - **plan-expert**: For complex, multi-file tasks. Surveys the codebase, creates a
      step-by-step plan (plan.md), then delegates each step to coder. Use this when
      the task touches multiple files or modules and needs careful sequencing.
    - **wiki-expert**: Creates and maintains project documentation as a wiki/ folder
      of interlinked markdown files. For documentation requests.

    Your workflow:
    1. Understand what the user wants.
    2. If it's a simple question you can answer directly, do so.
    3. For simple coding tasks (one file, clear change), delegate to coder directly.
    4. For complex coding tasks (multi-file, refactoring, new features spanning modules),
       delegate to plan-expert. It will survey, plan, and call coder for each step.
    5. For organism/agent design tasks, delegate to agent-expert.
    6. For documentation tasks ("document this", "create a wiki", "explain the architecture"),
       delegate to wiki-expert.
    7. Summarize the specialist's results for the user.

    You also have direct tools:
    - **calc**: Evaluates math expressions (Python tool running in WASM sandbox).
      Call it for any arithmetic, scientific calculations, or numeric computation.

    You can also explore the codebase yourself using file-read, glob, grep, and
    codebase-index to answer questions or gather context before delegating.

    Keep responses concise. Don't over-explain. The user knows what they're doing.
    Never list your capabilities unprompted. One or two sentences max for status updates.
    When asked to introduce yourself, respond with exactly: "Bob here. What are we working on?"
    Before calling a specialist, briefly tell the user what you're about to do
    (e.g., "Routing to coder for that." or "Let me get agent-expert on this.").

    {tool_definitions}

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
    6. Prefer specific tools over bash. Use cargo-test, cargo-check, git-status, etc.
       Only use bash as a last resort when no specific tool exists for the command.
    7. If a memory.md exists in the workspace root, read it at the start of each task
       for project context and conventions. When you discover stable patterns,
       architectural decisions, or recurring solutions, update it. Keep it concise.

    {tool_definitions}

listeners:
  # Bob — the concierge. Routes to specialists. Runs on Haiku (fast, cheap).
  - name: bob
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "AgentOS concierge — understands intent, routes to specialists"
    agent:
      prompt: "no_paperclipper & bob_base"
      model: haiku
      max_tokens: 2048
      max_agentic_iterations: 10
      permissions:
        file-read: auto
        glob: auto
        grep: auto
        codebase-index: auto
        list-agents: auto
        coder: auto
        agent-expert: auto
        user: auto
        plan-expert: auto
        wiki-expert: auto
        calc: auto
    librarian: true
    peers: [file-read, glob, grep, list-dir, codebase-index, list-agents, user, coder, plan-expert, agent-expert, wiki-expert, calc]

  # Coder — hands-on coding specialist (buffer → child pipeline)
  - name: coder
    payload_class: buffer.CoderRequest
    handler: buffer
    description: "Coding specialist — writes code, edits files, runs tests, uses git"
    buffer:
      description: "Execute a coding task: write code, edit files, run tests, use git. Describe what needs to be done."
      parameters:
        task:
          type: string
          description: "The coding task to perform — be specific about files, changes, and expected outcome"
      required: [task]
      requires: [file-read, file-write, file-edit, glob, grep, list-dir, bash, cargo-test, cargo-build, cargo-check, cargo-clippy, git-status, git-diff, git-log, git-add, git-commit, git-push]
      organism: organisms/coder-v2.yaml
      max_concurrency: 1
      timeout_secs: 600

  # Plan Expert — surveys codebase, creates plans, delegates steps to coder
  - name: plan-expert
    payload_class: buffer.PlanExpertRequest
    handler: buffer
    description: "Plan Expert — surveys codebase, creates step-by-step plans, delegates to coder"
    buffer:
      description: "Analyze a complex task, survey the codebase, create a structured plan (plan.md), then execute each step via the coder specialist. Use for multi-file changes, refactoring, or new features spanning multiple modules."
      parameters:
        task:
          type: string
          description: "The task to plan and execute — e.g., 'refactor auth module into separate crate', 'add WebSocket support to the server'"
      required: [task]
      requires: [file-read, file-write, file-edit, glob, grep, list-dir, codebase-index, user]
      organism: organisms/plan-expert.yaml
      max_concurrency: 1
      timeout_secs: 1800
      interactive: true

  # Agent Expert — organism design specialist (buffer → child pipeline)
  - name: agent-expert
    payload_class: buffer.AgentExpertRequest
    handler: buffer
    description: "Agent Expert — designs, validates, and diagnoses organism configurations"
    buffer:
      description: "Design, validate, or diagnose an AgentOS organism YAML configuration. Modes: design (create new), diagnose (post-mortem a crash), modify (edit existing)."
      parameters:
        task:
          type: string
          description: "What to do — e.g., 'create a research agent with web search', 'diagnose why organisms/my-agent.yaml crashes on startup'"
      required: [task]
      requires: [file-read, file-write, file-edit, glob, grep, list-dir, validate-organism, user]
      organism: organisms/agent-expert.yaml
      max_concurrency: 1
      timeout_secs: 600
      interactive: true

  # Wiki Expert — documentation specialist (buffer → child pipeline)
  - name: wiki-expert
    payload_class: buffer.WikiExpertRequest
    handler: buffer
    description: "Wiki Expert — creates and maintains project documentation as interlinked markdown"
    buffer:
      description: "Create or update project documentation in a wiki/ folder. Surveys the codebase, writes interlinked markdown pages with index, architecture, module guides, etc."
      parameters:
        task:
          type: string
          description: "What to document — e.g., 'document this project', 'update the architecture page', 'add a page about the tool system'"
      required: [task]
      requires: [file-read, file-write, file-edit, glob, grep, list-dir, codebase-index, user]
      organism: organisms/wiki-expert.yaml
      max_concurrency: 1
      timeout_secs: 900
      interactive: true

  # Infrastructure
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

  # Tools (registered in pipeline, declared here for profile/peer references)
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

  - name: cargo-test
    payload_class: tools.CargoTestRequest
    handler: tools.safe_commands.handle
    description: "Run cargo test"

  - name: cargo-build
    payload_class: tools.CargoBuildRequest
    handler: tools.safe_commands.handle
    description: "Run cargo build"

  - name: cargo-check
    payload_class: tools.CargoCheckRequest
    handler: tools.safe_commands.handle
    description: "Run cargo check"

  - name: cargo-clippy
    payload_class: tools.CargoClippyRequest
    handler: tools.safe_commands.handle
    description: "Run cargo clippy"

  - name: git-status
    payload_class: tools.GitStatusRequest
    handler: tools.safe_commands.handle
    description: "Show git status"

  - name: git-diff
    payload_class: tools.GitDiffRequest
    handler: tools.safe_commands.handle
    description: "Show git diff"

  - name: git-log
    payload_class: tools.GitLogRequest
    handler: tools.safe_commands.handle
    description: "Show git log"

  - name: git-add
    payload_class: tools.GitAddRequest
    handler: tools.safe_commands.handle
    description: "Stage files for commit"

  - name: git-commit
    payload_class: tools.GitCommitRequest
    handler: tools.safe_commands.handle
    description: "Create a git commit"

  - name: git-push
    payload_class: tools.GitPushRequest
    handler: tools.safe_commands.handle
    description: "Push to remote"

  - name: bash
    payload_class: tools.BashRequest
    handler: tools.command_exec.handle
    description: "Shell command (last resort)"

  - name: user
    payload_class: tui.UserRequest
    handler: tui.handle
    description: "Display messages or ask questions to the user"

  - name: list-agents
    payload_class: tools.ListAgentsRequest
    handler: tools.list_agents.handle
    description: "List available specialist agents and their capabilities"

  - name: validate-organism
    payload_class: tools.ValidateOrganismRequest
    handler: tools.validate_organism.handle
    description: "Validate organism YAML configuration"

  - name: compile-wasm
    payload_class: tools.CompileWasmRequest
    handler: tools.compile_wasm.handle
    description: "Compile Python tool to WASM component"

  - name: calc
    payload_class: tools.CalcRequest
    handler: python
    description: "Calculator — evaluates math expressions safely (Python/WASM)"
    python:
      source: tools/samples/calc_tool.py

profiles:
  default:
    linux_user: agentos
    listeners: [bob, coder, plan-expert, agent-expert, wiki-expert, user, file-read, file-write, file-edit, glob, grep, list-dir, cargo-test, cargo-build, cargo-check, cargo-clippy, git-status, git-diff, git-log, git-add, git-commit, git-push, bash, list-agents, validate-organism, compile-wasm, calc, codebase-index, llm-pool, librarian]
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

    // Collect startup errors — TUI always opens, errors display as messages.
    let mut startup_errors: Vec<String> = Vec::new();

    // Auto-mount CWD as the agent's workspace if it looks like a project directory
    let drive_slot = vdrive_tools::empty_slot();
    let auto_mount_msg = try_auto_mount(&drive_slot);

    // Parse organism config
    let yaml = if let Some(ref path) = cli.organism {
        match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                startup_errors.push(format!("Failed to read organism '{}': {e}", path));
                DEFAULT_ORGANISM.to_string()
            }
        }
    } else {
        DEFAULT_ORGANISM.to_string()
    };
    let org = match parse_organism(&yaml) {
        Ok(o) => o,
        Err(e) => {
            startup_errors.push(format!("Organism parse error: {e} — falling back to default"));
            parse_organism(DEFAULT_ORGANISM).expect("embedded organism must parse")
        }
    };

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

    // Build pipeline — LLM pool is optional (user may configure via TUI).
    // If the full build fails, fall back to a bare pipeline so the TUI always opens.
    let has_pool = pool.is_some();
    let slot = drive_slot.clone();
    let build_result = build_pipeline(
        org, &data_dir, debug, pool, has_pool, slot.clone(), &work_dir,
    );

    let (mut pipeline, build_error) = match build_result {
        Ok(p) => (p, None),
        Err(e) => {
            let err_msg = format!("{e}");
            startup_errors.push(format!("Pipeline build failed: {e}"));
            info!("Full pipeline build failed: {e} — falling back to degraded mode");
            // Fall back to bare pipeline so TUI always opens
            let bare_org = parse_organism(DEFAULT_ORGANISM).expect("embedded organism must parse");
            let bare = AgentPipelineBuilder::new(bare_org, &data_dir)
                .build()
                .to_anyhow()?;
            (bare, Some(err_msg))
        }
    };

    // Initialize root thread
    let profile = pipeline.organism().profile_names().into_iter().next()
        .unwrap_or("default");
    if let Err(e) = pipeline.initialize_root("agentos", profile).await {
        startup_errors.push(format!("Root thread init failed: {e}"));
    }

    info!("Pipeline ready, starting TUI");

    // Start pipeline
    pipeline.run();

    // Run TUI (blocks until quit) — always opens, errors shown as messages
    run_tui(&mut pipeline, debug, &yaml, models_config, agents_config, has_pool && build_error.is_none(), drive_slot, auto_mount_msg, startup_errors).await?;

    // Shutdown
    info!("Shutting down");
    pipeline.shutdown().await;

    // Force exit — spawned tasks (buffer child pipelines, in-flight LLM requests)
    // may still be running. The tokio runtime won't exit until all tasks complete,
    // so we force it. All state is WAL-backed, nothing is lost.
    std::process::exit(0);
}

/// Build the full pipeline with all tools, buffers, and agents.
///
/// If anything fails, the entire build fails and main() falls back to a bare
/// pipeline. Builder methods consume `self`, so partial recovery isn't possible.
fn build_pipeline(
    org: Organism,
    data_dir: &PathBuf,
    debug: bool,
    pool: Option<LlmPool>,
    has_pool: bool,
    slot: DriveSlot,
    work_dir: &str,
) -> Result<AgentPipeline, String> {
    let list_agents_tool = ListAgentsTool::from_organism(&org);
    let mut builder = AgentPipelineBuilder::new(org, data_dir).with_debug(debug);

    // LLM pool + dependents (librarian, semantic router)
    if let Some(p) = pool {
        builder = builder
            .with_llm_pool(p)?
            .with_librarian()?;
    }

    // Local inference (optional — graceful if missing)
    builder = builder.with_local_inference()?;

    // Code index
    builder = builder.with_code_index()?;

    // VDrive-sandboxed file tools
    builder = builder
        .register_tool("file-read", VDriveFileRead::new(slot.clone()))?
        .register_tool("file-write", VDriveFileWrite::new(slot.clone()))?
        .register_tool("file-edit", VDriveFileEdit::new(slot.clone()))?
        .register_tool("glob", VDriveGlob::new(slot.clone()))?
        .register_tool("grep", VDriveGrep::new(slot.clone()))?
        .register_tool("list-dir", VDriveListDir::new(slot.clone()))?
        .register_tool("bash", VDriveCommandExec::new(slot.clone()))?;

    // Safe commands (cargo-test, git-status, etc.)
    for def in ALL_SAFE_COMMANDS {
        builder = builder.register_tool(def.name, SafeCommandTool::new(def, slot.clone()))?;
    }

    // validate-organism tool
    builder = builder.register_tool("validate-organism", ValidateOrganismTool::new(slot.clone()))?;

    // list-agents tool (snapshot of organism agents/buffers for Bob)
    builder = builder.register_tool("list-agents", list_agents_tool)?;

    // user channel (display + query bridge to TUI)
    let user_handler = UserChannelHandler::new(builder.event_sender(), builder.query_sender());
    builder = builder.register_tool("user", user_handler)?;

    // compile-wasm tool
    let wit_dir = PathBuf::from(work_dir).join("tools").join("wit");
    builder = builder.register_tool("compile-wasm", CompileWasmTool::new(wit_dir))?;

    // Python tools (handler: "python" listeners in organism)
    let wasm_dir = PathBuf::from(work_dir).join("tools").join("python-runtime");
    builder = builder.with_python_tools(&PathBuf::from(work_dir), &wasm_dir)?;

    // Buffer nodes (child pipelines for coder, agent-expert)
    builder = builder.with_buffer_nodes(&PathBuf::from(work_dir), slot)?;

    // Agent wiring (needs LLM pool)
    if has_pool {
        builder = builder.with_agents()?;
    }

    builder.build()
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
