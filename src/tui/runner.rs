//! TUI runner — main loop that wires everything together.
//!
//! Creates terminal, spawns event loop, runs main TEA loop.

use std::io;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use tokio::sync::Mutex;
use tokio::time::interval;

use rust_pipeline::prelude::build_envelope;

use crate::kernel::Kernel;
use crate::pipeline::AgentPipeline;
use crate::tools::xml_escape;

use super::app::{ContextView, MessageView, ThreadView, TuiApp};
use super::event::TuiMessage;
use super::layout;

/// Refresh TuiApp state from the kernel (brief lock).
pub async fn refresh_from_kernel(app: &mut TuiApp, kernel: &Arc<Mutex<Kernel>>) {
    let k = kernel.lock().await;

    // Refresh threads
    app.threads = k.threads().all_records().map(ThreadView::from).collect();

    // Refresh messages (journal entries)
    app.messages = k.journal().all_entries().map(MessageView::from).collect();

    // Refresh context for selected thread
    if let Some(selected) = app.threads.get(app.selected_thread) {
        if let Ok(inv) = k.contexts().get_inventory(&selected.uuid) {
            app.context = Some(ContextView::from(&inv));
        } else {
            app.context = None;
        }
    } else {
        app.context = None;
    }
    // Lock released here — microseconds
}

/// Inject a task from the input bar into the pipeline.
///
/// Routes to the selected agent if set, otherwise to the first agent listener.
async fn inject_task(
    pipeline: &AgentPipeline,
    kernel: &Arc<Mutex<Kernel>>,
    task: &str,
    selected_agent: Option<&str>,
) {
    let root_uuid = {
        let k = kernel.lock().await;
        k.threads().root_uuid().map(|s| s.to_string())
    };

    // Find the target agent: selected by name, or first available
    let agents = pipeline.organism().agent_listeners();
    let agent_def = if let Some(name) = selected_agent {
        agents.iter().find(|a| a.name == name).or(agents.first())
    } else {
        agents.first()
    };
    let agent_def = match agent_def {
        Some(def) => def,
        None => return, // no agents configured
    };
    let agent_name = agent_def.name.clone();
    let payload_tag = agent_def.payload_tag.clone();

    if let Some(uuid) = root_uuid {
        let escaped = xml_escape(task);
        let xml = format!("<{payload_tag}><task>{escaped}</task></{payload_tag}>");
        if let Ok(envelope) =
            build_envelope("user", &agent_name, &uuid, xml.as_bytes())
        {
            let _ = pipeline
                .inject_checked(envelope, &uuid, "coding", &agent_name)
                .await;
        }
    }
}

/// Run the TUI main loop. Blocks until quit.
pub async fn run_tui(
    pipeline: &mut AgentPipeline,
    debug: bool,
    organism_yaml: &str,
    models_config: crate::config::ModelsConfig,
    agents_config: crate::config::AgentsConfig,
    has_pool: bool,
) -> anyhow::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    io::stdout().execute(crossterm::event::EnableBracketedPaste)?;
    io::stdout().execute(crossterm::event::EnableMouseCapture)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut app = TuiApp::new();
    app.debug_mode = debug;
    app.llm_pool = pipeline.llm_pool();
    app.models_config = std::sync::Arc::new(tokio::sync::Mutex::new(models_config));
    app.agents_config = agents_config;
    app.load_yaml_editor(organism_yaml);
    app.load_organism_graph(pipeline.organism());
    app.rebuild_menu();

    // If no LLM pool at boot, show a helpful welcome message
    if !has_pool {
        super::commands::push_feedback(
            &mut app,
            "No API key configured. Use /provider <name> to set up a provider.\nExample: /provider anthropic",
        );
    }
    let kernel = pipeline.kernel();
    let mut event_rx = pipeline.subscribe();
    let mut approval_rx = pipeline.take_approval_receiver();

    // Dedicated input thread — reads crossterm events and sends through channel.
    // One thread, no polling/spinning. event::read() blocks until input arrives.
    // CRITICAL: Filter on the input thread side. Windows fires Press, Repeat, and
    // Release for every keystroke. If we send all events through the channel, Release
    // events accumulate, burn select iterations, and starve the render branch.
    let (key_tx, mut key_rx) = tokio::sync::mpsc::channel::<Event>(32);
    tokio::task::spawn_blocking(move || {
        while let Ok(ev) = event::read() {
            // Only forward Press events — drop Release/Repeat before they
            // enter the channel. Non-key events (mouse, resize) pass through.
            let dominated = matches!(
                &ev,
                Event::Key(k) if k.kind != KeyEventKind::Press
            );
            if dominated {
                continue;
            }
            if key_tx.blocking_send(ev).is_err() {
                break; // receiver dropped, TUI is shutting down
            }
        }
    });

    let mut tick_interval = interval(Duration::from_millis(250)); // 4Hz
    tick_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut render_interval = interval(Duration::from_millis(33)); // ~30fps
    render_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        // Check quit FIRST so /exit and Ctrl+C respond immediately,
        // even if the pipeline is busy with an agent task.
        if app.should_quit {
            break;
        }

        tokio::select! {
            _ = tick_interval.tick() => {
                refresh_from_kernel(&mut app, &kernel).await;
            }
            _ = render_interval.tick() => {
                terminal.draw(|f| layout::draw(f, &mut app))?;
            }
            Ok(pipeline_event) = event_rx.recv() => {
                app.update(TuiMessage::Pipeline(pipeline_event));
            }
            Some(crossterm_event) = key_rx.recv() => {
                // Input thread already filters to Press-only key events.
                match crossterm_event {
                    Event::Key(key) => {
                        app.update(TuiMessage::Input(key));
                    }
                    Event::Paste(text) => {
                        // Bracketed paste — normalize \r\n → \n, preserve newlines.
                        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
                        if looks_like_block(&normalized) {
                            // Structured content (table, code) — auto-fence.
                            let fenced = format!("```\n{}\n```", normalized.trim());
                            app.input_line.insert_str(&fenced);
                        } else {
                            app.input_line.insert_str(&normalized);
                        }
                    }
                    Event::Mouse(mouse) => {
                        super::mouse::handle_mouse(&mut app, mouse);
                    }
                    _ => {}
                }
            }
            Some(request) = async { match approval_rx.as_mut() { Some(rx) => rx.recv().await, None => std::future::pending().await } } => {
                app.pending_approval = Some(request);
            }
        }

        // Check for pending slash command (set by input handler on Enter with `/`)
        if let Some(cmd) = app.pending_command.take() {
            let pool_ref = app.llm_pool.clone();
            let result = super::commands::execute(&mut app, &cmd, pool_ref.as_ref()).await;
            if let Some(feedback) = result.feedback {
                super::commands::push_feedback(&mut app, &feedback);
            }
        }

        // Check for pending provider completion (set by provider wizard Enter)
        if let Some(pc) = app.pending_provider_completion.take() {
            // Create a temporary client with the given key to discover models
            let base_url = {
                let config = app.models_config.lock().await;
                config.providers.get(&pc.provider).and_then(|p| p.base_url.clone())
            };
            let client = if let Some(ref url) = base_url {
                crate::llm::client::AnthropicClient::with_base_url(pc.api_key.clone(), url.clone())
            } else {
                crate::llm::client::AnthropicClient::new(pc.api_key.clone())
            };

            match client.list_models().await {
                Ok(models) => {
                    let config_arc = app.models_config.clone();
                    let mut config = config_arc.lock().await;

                    // Build the model map from discovered models
                    let mut model_map = std::collections::HashMap::new();
                    for model in &models {
                        let alias = shorten_model_id(&model.id);
                        model_map.insert(alias, model.id.clone());
                    }

                    // Add well-known friendly aliases (opus, sonnet, haiku, etc.)
                    // so /model opus works alongside /model opus-4-6
                    add_friendly_aliases(&mut model_map);

                    // Store provider with key and discovered models
                    if let Some(existing) = config.providers.get_mut(&pc.provider) {
                        existing.api_key = Some(pc.api_key.clone());
                        existing.models = model_map.clone();
                    } else {
                        config.providers.insert(pc.provider.clone(), crate::config::ProviderConfig {
                            api_key: Some(pc.api_key.clone()),
                            base_url: base_url.clone(),
                            models: model_map.clone(),
                        });
                    }

                    // Set default if none
                    if config.default.is_none() {
                        let default = if model_map.contains_key("sonnet") {
                            "sonnet".to_string()
                        } else {
                            model_map.keys().next().cloned().unwrap_or_default()
                        };
                        if !default.is_empty() {
                            config.default = Some(default);
                        }
                    }

                    let _ = config.save();

                    // Format feedback
                    let mut lines = vec![format!("Discovered {} models from {}:", models.len(), pc.provider)];
                    let mut aliases: Vec<_> = model_map.iter().collect();
                    aliases.sort_by_key(|(alias, _)| (*alias).clone());
                    for (alias, model_id) in &aliases {
                        let is_default = config.default.as_deref() == Some(alias.as_str());
                        let marker = if is_default { " *" } else { "  " };
                        lines.push(format!("{marker} {alias:<20} {model_id}"));
                    }

                    // Rebuild or create pool
                    let pool_msg = rebuild_pool(&mut app, &config);
                    drop(config);
                    lines.push(pool_msg);
                    super::commands::push_feedback(&mut app, &lines.join("\n"));
                }
                Err(e) => {
                    super::commands::push_feedback(
                        &mut app,
                        &format!("Failed to discover models from {}: {e}\nCheck your API key and try again with /provider {}", pc.provider, pc.provider),
                    );
                }
            }
        }

        // Check for pending task submission (set by input handler on Enter)
        if let Some(task) = app.pending_task.take() {
            inject_task(pipeline, &kernel, &task, app.selected_agent.as_deref()).await;
        }
    }

    // Restore terminal — always runs, even if agent task is mid-flight.
    // Pipeline tasks are detached (tokio-spawned), so they'll be dropped
    // when the runtime shuts down.
    io::stdout().execute(crossterm::event::DisableMouseCapture)?;
    io::stdout().execute(crossterm::event::DisableBracketedPaste)?;
    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

/// Detect if pasted text looks like a markdown block (table or code).
///
/// Tables: multiple lines where most start/end with `|`.
/// Code: multiple lines with consistent indentation or code-like syntax.
fn looks_like_block(text: &str) -> bool {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 2 {
        return false;
    }

    // Table: majority of lines start and end with `|`
    let pipe_lines = lines
        .iter()
        .filter(|l| {
            let t = l.trim();
            t.starts_with('|') && t.ends_with('|')
        })
        .count();
    if pipe_lines * 2 >= lines.len() {
        return true;
    }

    // Code: majority of non-empty lines start with whitespace (indented)
    let non_empty: Vec<&&str> = lines.iter().filter(|l| !l.trim().is_empty()).collect();
    if non_empty.len() >= 2 {
        let indented = non_empty
            .iter()
            .filter(|l| l.starts_with(' ') || l.starts_with('\t'))
            .count();
        if indented * 2 >= non_empty.len() {
            return true;
        }
    }

    false
}

/// Generate a short alias from a full model ID.
/// E.g., "claude-sonnet-4-6-20250514" → "sonnet-4-6"
///       "claude-opus-4-6" → "opus"
///       "gpt-4o" → "gpt-4o"
fn shorten_model_id(model_id: &str) -> String {
    let s = model_id.strip_prefix("claude-").unwrap_or(model_id);
    // Strip date suffixes like "-20250514" (8 digits after last dash)
    let s = if s.len() > 9 {
        let last_dash = s.rfind('-').unwrap_or(s.len());
        let suffix = &s[last_dash + 1..];
        if suffix.len() == 8 && suffix.chars().all(|c| c.is_ascii_digit()) {
            &s[..last_dash]
        } else {
            s
        }
    } else {
        s
    };
    s.to_string()
}

/// Add well-known friendly aliases that point to the latest model in each family.
/// E.g., "opus" → latest opus model ID, "sonnet" → latest sonnet, etc.
/// Only adds if the friendly name isn't already a shortened alias.
fn add_friendly_aliases(model_map: &mut std::collections::HashMap<String, String>) {
    // Map friendly name → prefix to match against shortened aliases
    let families = [
        ("opus", "opus"),
        ("sonnet", "sonnet"),
        ("haiku", "haiku"),
    ];
    for (friendly, prefix) in &families {
        if model_map.contains_key(*friendly) {
            continue; // already exists as a shortened alias
        }
        // Find the best match: prefer latest version (highest version number)
        let mut best: Option<(&String, &String)> = None;
        for (alias, model_id) in model_map.iter() {
            if alias.starts_with(prefix) {
                match best {
                    None => best = Some((alias, model_id)),
                    Some((_, prev_id)) => {
                        // Prefer longer model ID (more specific version) or lexicographically greater
                        if model_id > prev_id {
                            best = Some((alias, model_id));
                        }
                    }
                }
            }
        }
        if let Some((_, model_id)) = best {
            model_map.insert(friendly.to_string(), model_id.clone());
        }
    }
}

/// Rebuild or create the LlmPool from updated config.
/// If a pool exists, rebuilds it in-place. If not, creates a new one.
/// Returns a suffix message for user feedback (e.g., " — pool connected").
fn rebuild_pool(app: &mut TuiApp, config: &crate::config::ModelsConfig) -> String {
    use crate::llm::LlmPool;

    if let Some(ref pool_arc) = app.llm_pool {
        // Pool exists — rebuild in place
        match pool_arc.try_lock() {
            Ok(mut pool) => match pool.rebuild_from_config(config) {
                Ok(()) => format!("\nPool reloaded (model: {})", pool.default_model()),
                Err(e) => format!("\nWarning: pool rebuild failed: {e}"),
            },
            Err(_) => "\nWarning: pool is busy, changes will apply on next request".into(),
        }
    } else {
        // No pool yet — create one
        match LlmPool::from_config(config) {
            Ok(pool) => {
                let model = pool.default_model().to_string();
                app.llm_pool = Some(std::sync::Arc::new(tokio::sync::Mutex::new(pool)));
                format!("\nPool connected (model: {model})")
            }
            Err(e) => format!("\nWarning: could not create pool: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::events::PipelineEvent;
    use tempfile::TempDir;

    #[tokio::test]
    async fn refresh_populates_threads() {
        let dir = TempDir::new().unwrap();
        let mut kernel = Kernel::open(&dir.path().join("data")).unwrap();
        kernel.initialize_root("org", "admin").unwrap();

        let kernel_arc = Arc::new(Mutex::new(kernel));
        let mut app = TuiApp::new();
        refresh_from_kernel(&mut app, &kernel_arc).await;

        assert!(!app.threads.is_empty());
        assert_eq!(app.threads[0].chain, "system.org");
    }

    #[tokio::test]
    async fn refresh_populates_context() {
        let dir = TempDir::new().unwrap();
        let mut kernel = Kernel::open(&dir.path().join("data")).unwrap();
        let root = kernel.initialize_root("org", "admin").unwrap();
        kernel.contexts_mut().create(&root).unwrap();

        let seg = crate::kernel::context_store::ContextSegment {
            id: "s1".into(),
            tag: "code".into(),
            content: b"fn main()".to_vec(),
            status: crate::kernel::context_store::SegmentStatus::Active,
            relevance: 0.8,
            created_at: 0,
            fold_ref: None,
        };
        kernel.contexts_mut().add_segment(&root, seg).unwrap();

        let kernel_arc = Arc::new(Mutex::new(kernel));
        let mut app = TuiApp::new();
        refresh_from_kernel(&mut app, &kernel_arc).await;

        assert!(app.context.is_some());
        let ctx = app.context.unwrap();
        assert_eq!(ctx.segments.len(), 1);
        assert_eq!(ctx.segments[0].id, "s1");
    }

    #[tokio::test]
    async fn event_log_ring_buffer() {
        let mut app = TuiApp::new();
        // Fill beyond capacity
        for i in 0..300 {
            app.update(TuiMessage::Pipeline(PipelineEvent::MessageInjected {
                thread_id: format!("t-{i}"),
                target: "echo".into(),
                profile: "admin".into(),
            }));
        }
        // Should be capped at 256
        assert_eq!(app.event_log.len(), 256);
    }

    #[test]
    fn runner_quit_on_message() {
        let mut app = TuiApp::new();
        app.update(TuiMessage::Quit);
        assert!(app.should_quit);
    }

    #[tokio::test]
    async fn integration_pipeline_to_tui() {
        use crate::organism::parser::parse_organism;
        use rust_pipeline::prelude::{
            build_envelope, FnHandler, HandlerContext, HandlerResponse, ValidatedPayload,
        };

        let yaml = r#"
organism:
  name: tui-test

listeners:
  - name: echo
    payload_class: handlers.echo.Greeting
    handler: handlers.echo.handle
    description: "Echo handler"
    peers: []

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [echo]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let dir = TempDir::new().unwrap();

        let echo = FnHandler(|p: ValidatedPayload, _ctx: HandlerContext| {
            Box::pin(async move { Ok(HandlerResponse::Reply { payload_xml: p.xml }) })
        });

        let mut pipeline = crate::pipeline::AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .register("echo", echo)
            .unwrap()
            .build()
            .unwrap();

        pipeline.run();

        // Subscribe and inject
        let mut rx = pipeline.subscribe();

        let envelope = build_envelope(
            "test",
            "echo",
            "thread-1",
            b"<Greeting><text>hello tui</text></Greeting>",
        )
        .unwrap();

        pipeline
            .inject_checked(envelope, "thread-1", "admin", "echo")
            .await
            .unwrap();

        // Verify event arrives
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, PipelineEvent::MessageInjected { .. }));

        // Verify kernel has the root data we can refresh from
        let kernel = pipeline.kernel();
        let mut app = TuiApp::new();
        refresh_from_kernel(&mut app, &kernel).await;
        // Messages might be empty (journal not populated by inject_checked alone)
        // but threads should be accessible

        pipeline.shutdown().await;
    }

    #[test]
    fn submit_sets_pending_task() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let mut app = TuiApp::new();
        app.set_input_text("Read README.md");

        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));

        assert_eq!(app.pending_task, Some("Read README.md".into()));
        assert!(app.input_text().is_empty());
        assert_eq!(app.chat_log.len(), 1);
        assert_eq!(app.chat_log[0].role, "user");
    }

    #[test]
    fn typing_goes_to_input_editor() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let mut app = TuiApp::new();

        // Type "hi" — goes directly to input editor
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Char('h'),
            KeyModifiers::NONE,
        )));
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Char('i'),
            KeyModifiers::NONE,
        )));
        assert!(app.input_text().contains("hi"));

        // Backspace
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Backspace,
            KeyModifiers::NONE,
        )));
        assert_eq!(app.input_text(), "h");
    }

    #[test]
    fn esc_clears_input() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let mut app = TuiApp::new();
        app.set_input_text("some text");

        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Esc,
            KeyModifiers::NONE,
        )));

        assert!(app.input_text().is_empty());
        assert!(!app.should_quit);
    }

    #[test]
    fn arrows_scroll_messages() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let mut app = TuiApp::new();
        app.message_scroll = 10;
        app.message_auto_scroll = false;

        // Up scrolls 3 lines at a time
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Up,
            KeyModifiers::NONE,
        )));
        assert_eq!(app.message_scroll, 7);

        // Down scrolls 3 lines at a time
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Down,
            KeyModifiers::NONE,
        )));
        assert_eq!(app.message_scroll, 10);
    }

    #[test]
    fn agent_response_updates_status() {
        use super::super::app::AgentStatus;

        let mut app = TuiApp::new();
        app.agent_status = AgentStatus::Thinking;

        app.update(TuiMessage::Pipeline(PipelineEvent::AgentResponse {
            thread_id: "t1".into(),
            text: "Done! Here is the summary.".into(),
        }));

        assert_eq!(app.agent_status, AgentStatus::Idle);
        assert_eq!(
            app.last_response,
            Some("Done! Here is the summary.".into())
        );
    }

    #[test]
    fn ctrl_keys_switch_tabs() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        use super::super::app::ActiveTab;

        let mut app = TuiApp::new();
        assert_eq!(app.active_tab, ActiveTab::Messages);

        // Ctrl+2 → Threads
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Char('2'),
            KeyModifiers::CONTROL,
        )));
        assert_eq!(app.active_tab, ActiveTab::Threads);

        // Ctrl+3 → Graph
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Char('3'),
            KeyModifiers::CONTROL,
        )));
        assert_eq!(app.active_tab, ActiveTab::Graph);

        // Ctrl+4 → Yaml
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Char('4'),
            KeyModifiers::CONTROL,
        )));
        assert_eq!(app.active_tab, ActiveTab::Yaml);

        // Ctrl+1 → Messages
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Char('1'),
            KeyModifiers::CONTROL,
        )));
        assert_eq!(app.active_tab, ActiveTab::Messages);
    }

    #[test]
    fn home_jumps_to_top() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let mut app = TuiApp::new();
        app.message_scroll = 50;
        app.message_auto_scroll = false;

        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Home,
            KeyModifiers::NONE,
        )));
        assert_eq!(app.message_scroll, 0);
        assert!(!app.message_auto_scroll);
    }

    #[test]
    fn end_jumps_to_bottom() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let mut app = TuiApp::new();
        app.message_scroll = 0;
        app.message_auto_scroll = false;

        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::End,
            KeyModifiers::NONE,
        )));
        assert!(app.message_auto_scroll);
    }

    #[test]
    fn page_up_scrolls_by_viewport() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let mut app = TuiApp::new();
        app.message_scroll = 50;
        app.message_auto_scroll = false;
        app.viewport_height = 20;

        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::PageUp,
            KeyModifiers::NONE,
        )));
        // viewport_height(20) - 2 overlap = 18 lines scrolled
        assert_eq!(app.message_scroll, 32);
    }

    #[test]
    fn arrows_dont_scroll_messages_on_threads_tab() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let mut app = TuiApp::new();
        app.message_scroll = 5;
        app.message_auto_scroll = false;

        // Switch to Threads tab (default focus = ThreadList)
        app.active_tab = super::super::app::ActiveTab::Threads;

        // Up arrow should NOT scroll messages — dispatches to thread list instead
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Up,
            KeyModifiers::NONE,
        )));
        assert_eq!(app.message_scroll, 5); // unchanged
    }

    // ── looks_like_block tests ──

    #[test]
    fn table_detected_as_block() {
        let table = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |";
        assert!(looks_like_block(table));
    }

    #[test]
    fn indented_code_detected_as_block() {
        let code = "fn main() {\n    println!(\"hello\");\n    let x = 42;\n}";
        assert!(looks_like_block(code));
    }

    #[test]
    fn plain_text_not_block() {
        assert!(!looks_like_block("hello world"));
        assert!(!looks_like_block("just a sentence"));
    }

    #[test]
    fn single_line_not_block() {
        assert!(!looks_like_block("| one line |"));
    }

    #[test]
    fn multiline_prose_not_block() {
        let prose = "This is a paragraph.\nIt has multiple sentences.\nBut no structure.";
        assert!(!looks_like_block(prose));
    }
}
