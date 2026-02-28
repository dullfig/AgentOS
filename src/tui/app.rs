//! TuiApp — the TEA model.
//!
//! All state lives here. Update receives TuiMessages, mutates state.
//! View reads state to produce ratatui widgets. No side effects in view.

use std::sync::Arc;
use ratatui::layout::Rect;
use tokio::sync::Mutex;
use tui_menu::{MenuItem, MenuState};

use crate::config::{AgentsConfig, ModelsConfig};
use crate::kernel::context_store::{ContextInventory, SegmentMeta, SegmentStatus};
use crate::kernel::journal::JournalEntry;
use crate::kernel::thread_table::ThreadRecord;
use crate::llm::LlmPool;
use crate::lsp::command_line::CommandLineService;
use crate::lsp::organism::OrganismYamlService;
use crate::lsp::{HoverInfo, LanguageService};
use crate::pipeline::events::PipelineEvent;

use super::event::TuiMessage;

/// Which tab is currently visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTab {
    Messages, // Ctrl+1, default
    Threads,  // Ctrl+2 (threads + context)
    Yaml,     // Ctrl+3 (placeholder)
    Wasm,     // Ctrl+4 (placeholder)
    Debug,    // Ctrl+5 (activity trace — only when debug_mode)
}

/// Which sub-pane has focus within the Threads tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadsFocus {
    ThreadList,
    Conversation,
    ContextTree,
}

/// Actions that can be triggered from the menu bar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MenuAction {
    SwitchTab(ActiveTab),
    NewTask,
    Quit,
    SetModel,
    ShowAbout,
    ShowShortcuts,
    SelectAgent(String),
    ResetAgent,
}

/// Agent processing status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentStatus {
    Idle,
    Thinking,
    ToolCall(String),
    Error(String),
}

/// Current input mode (normal or wizard).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    ProviderWizard {
        provider: String,
    },
}

/// Pending provider wizard completion data (consumed by runner after handle_key).
#[derive(Debug, Clone)]
pub struct ProviderCompletion {
    pub provider: String,
    pub api_key: String,
}

/// Lightweight view of a thread record (no kernel lifetime).
#[derive(Debug, Clone)]
pub struct ThreadView {
    pub uuid: String,
    pub chain: String,
    pub profile: String,
    pub created_at: u64,
}

impl From<&ThreadRecord> for ThreadView {
    fn from(r: &ThreadRecord) -> Self {
        Self {
            uuid: r.uuid.clone(),
            chain: r.chain.clone(),
            profile: r.profile.clone(),
            created_at: r.created_at,
        }
    }
}

/// Lightweight view of a message/journal entry.
#[derive(Debug, Clone)]
pub struct MessageView {
    pub message_id: String,
    pub from: String,
    pub to: String,
    pub status: String,
    pub thread_id: String,
}

impl From<&JournalEntry> for MessageView {
    fn from(e: &JournalEntry) -> Self {
        let status = format!("{:?}", e.status);
        Self {
            message_id: e.message_id.clone(),
            from: e.from.clone(),
            to: e.to.clone(),
            status,
            thread_id: e.thread_id.clone(),
        }
    }
}

/// Lightweight view of a segment.
#[derive(Debug, Clone)]
pub struct SegmentView {
    pub id: String,
    pub tag: String,
    pub size: usize,
    pub status: SegmentStatus,
    pub relevance: f32,
}

impl From<&SegmentMeta> for SegmentView {
    fn from(m: &SegmentMeta) -> Self {
        Self {
            id: m.id.clone(),
            tag: m.tag.clone(),
            size: m.size,
            status: m.status,
            relevance: m.relevance,
        }
    }
}

/// Lightweight view of a thread's context.
#[derive(Debug, Clone)]
pub struct ContextView {
    pub thread_id: String,
    pub segments: Vec<SegmentView>,
    pub active_count: usize,
    pub shelved_count: usize,
    pub folded_count: usize,
    pub total_bytes: usize,
    pub active_bytes: usize,
}

impl From<&ContextInventory> for ContextView {
    fn from(inv: &ContextInventory) -> Self {
        Self {
            thread_id: inv.thread_id.clone(),
            segments: inv.segments.iter().map(SegmentView::from).collect(),
            active_count: inv.active_count,
            shelved_count: inv.shelved_count,
            folded_count: inv.folded_count,
            total_bytes: inv.total_bytes,
            active_bytes: inv.active_bytes,
        }
    }
}

/// Status of an activity trace entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivityStatus {
    InProgress,
    Done,
    Error,
}

/// A single entry in the live activity trace (Threads tab).
#[derive(Debug, Clone)]
pub struct ActivityEntry {
    /// Unix timestamp (seconds).
    pub timestamp: u64,
    /// Short label — tool name or "thinking".
    pub label: String,
    /// Detail — path, command, pattern, etc.
    pub detail: String,
    /// Current status.
    pub status: ActivityStatus,
}

/// A chat message in the conversation log.
#[derive(Debug, Clone)]
pub struct ChatEntry {
    /// "user" or "agent"
    pub role: String,
    /// The message text.
    pub text: String,
}

/// The main TUI application state (TEA model).
pub struct TuiApp {
    /// Which tab is currently visible.
    pub active_tab: ActiveTab,
    /// Whether the app should quit.
    pub should_quit: bool,
    /// Currently selected thread index.
    pub selected_thread: usize,
    /// Scroll offset for the messages pane.
    pub message_scroll: u16,
    /// When true, auto-scroll messages to bottom on next render.
    pub message_auto_scroll: bool,
    /// Recent pipeline events (ring buffer).
    pub event_log: Vec<PipelineEvent>,
    /// Thread list (refreshed from kernel on tick).
    pub threads: Vec<ThreadView>,
    /// Messages/journal entries for the selected thread.
    pub messages: Vec<MessageView>,
    /// Context for the selected thread.
    pub context: Option<ContextView>,
    /// Total input tokens across all API calls.
    pub total_input_tokens: u64,
    /// Total output tokens across all API calls.
    pub total_output_tokens: u64,
    /// Text input widget (ratatui-code-editor, plain text mode).
    pub input_editor: ratatui_code_editor::editor::Editor,
    /// Cached input bar area from last render (needed for editor.input()).
    pub input_area: Rect,
    /// Current agent processing status.
    pub agent_status: AgentStatus,
    /// Task pending injection into the pipeline (set by input, consumed by runner).
    pub pending_task: Option<String>,
    /// Last agent response text (for display).
    pub last_response: Option<String>,
    /// Conversation log (user tasks + agent responses).
    pub chat_log: Vec<ChatEntry>,
    /// Viewport height of the messages pane (set by renderer, used by PageUp/PageDown).
    pub viewport_height: u16,
    /// Live activity trace (ring buffer, Threads tab).
    pub activity_log: Vec<ActivityEntry>,
    /// Scroll offset for the activity trace pane.
    pub activity_scroll: u16,
    /// When true, auto-scroll activity to bottom on next render.
    pub activity_auto_scroll: bool,
    /// Viewport height of the activity trace pane (set by renderer).
    pub activity_viewport_height: u16,
    /// Per-thread conversation entries (populated by ConversationSync events).
    pub thread_conversations: std::collections::HashMap<String, Vec<crate::pipeline::events::ConversationEntry>>,
    /// Scroll offset for the conversation pane (Threads tab).
    pub conversation_scroll: u16,
    /// When true, auto-scroll conversation to bottom on next render.
    pub conversation_auto_scroll: bool,
    /// Viewport height of the conversation pane (set by renderer).
    pub conversation_viewport_height: u16,
    /// LLM pool handle (for `/model` command).
    pub llm_pool: Option<Arc<Mutex<LlmPool>>>,
    /// Command pending async execution (set by input handler on Enter with `/`).
    pub pending_command: Option<String>,
    /// Which sub-pane has focus within the Threads tab.
    pub threads_focus: ThreadsFocus,
    /// Tree widget state for the context tree (Threads tab, context pane).
    pub context_tree_state: tui_tree_widget::TreeState<String>,
    /// Whether debug mode is enabled (--debug flag). Controls Debug tab visibility.
    pub debug_mode: bool,
    /// Menu bar state (tui-menu).
    pub menu_state: MenuState<MenuAction>,
    /// Whether the menu bar has keyboard focus (dropdowns visible).
    pub menu_active: bool,
    /// Command palette selection index (for `/` popup).
    pub command_popup_index: usize,
    /// Language service for command-line input (slash commands).
    pub cmd_service: CommandLineService,
    /// YAML editor widget (ratatui-code-editor). None until organism YAML loaded.
    pub yaml_editor: Option<ratatui_code_editor::editor::Editor>,
    /// YAML validation status. None = valid/untouched, Some(msg) = parse error.
    pub yaml_status: Option<String>,
    /// Cached YAML content area from last render (needed for editor.input()).
    pub yaml_area: Rect,
    /// Language service for organism YAML (diagnostics, completions, hover).
    pub lang_service: OrganismYamlService,
    /// Current diagnostics from the language service.
    pub diagnostics: Vec<lsp_types::Diagnostic>,
    /// Completion items from the language service.
    pub completion_items: Vec<lsp_types::CompletionItem>,
    /// Selected index in the completion popup.
    pub completion_index: usize,
    /// Whether the completion popup is visible.
    pub completion_visible: bool,
    /// Current hover information. None = no hover active.
    pub hover_info: Option<HoverInfo>,
    /// Debounce counter for diagnostics (frames since last edit).
    pub diag_debounce: u8,
    /// Diagnostic summary for status bar ("3 errors, 1 warning").
    pub diag_summary: String,
    /// Current input mode (normal typing vs wizard).
    pub input_mode: InputMode,
    /// Models configuration (shared with commands for mutation).
    pub models_config: Arc<Mutex<ModelsConfig>>,
    /// Pending provider wizard completion (set by wizard Enter, consumed by runner).
    pub pending_provider_completion: Option<ProviderCompletion>,
    /// Currently selected agent name. None = default (first agent).
    pub selected_agent: Option<String>,
    /// Agent favorites config (project-level persistence).
    pub agents_config: AgentsConfig,
}

/// Current time in seconds since Unix epoch.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Maximum number of events in the ring buffer.
const EVENT_LOG_CAPACITY: usize = 256;

/// Maximum number of activity entries in the ring buffer.
const ACTIVITY_LOG_CAPACITY: usize = 512;

/// Build the menu item tree for the menu bar.
pub fn build_menu_items(
    debug_mode: bool,
    favorites: &[String],
    selected: Option<&str>,
) -> Vec<MenuItem<MenuAction>> {
    let mut view_items = vec![
        MenuItem::item("Messages  ^1", MenuAction::SwitchTab(ActiveTab::Messages)),
        MenuItem::item("Threads   ^2", MenuAction::SwitchTab(ActiveTab::Threads)),
        MenuItem::item("YAML      ^3", MenuAction::SwitchTab(ActiveTab::Yaml)),
        MenuItem::item("WASM      ^4", MenuAction::SwitchTab(ActiveTab::Wasm)),
    ];
    if debug_mode {
        view_items.push(MenuItem::item(
            "Debug     ^5",
            MenuAction::SwitchTab(ActiveTab::Debug),
        ));
    }

    // Build Agents menu items from favorites
    let mut agent_items: Vec<MenuItem<MenuAction>> = favorites
        .iter()
        .map(|name| {
            let marker = if selected == Some(name.as_str()) {
                " \u{2713}"
            } else {
                ""
            };
            MenuItem::item(
                format!("{name}{marker}"),
                MenuAction::SelectAgent(name.clone()),
            )
        })
        .collect();
    agent_items.push(MenuItem::item("(default)", MenuAction::ResetAgent));

    vec![
        MenuItem::group(
            "File",
            vec![
                MenuItem::item("New Task", MenuAction::NewTask),
                MenuItem::item("Quit     ^C", MenuAction::Quit),
            ],
        ),
        MenuItem::group("View", view_items),
        MenuItem::group("Agents", agent_items),
        MenuItem::group(
            "Model",
            vec![MenuItem::item("/model ...", MenuAction::SetModel)],
        ),
        MenuItem::group(
            "Help",
            vec![
                MenuItem::item("About", MenuAction::ShowAbout),
                MenuItem::item("Shortcuts", MenuAction::ShowShortcuts),
            ],
        ),
    ]
}

impl TuiApp {
    /// Create a new TuiApp with default state.
    pub fn new() -> Self {
        // Plain text editor for input bar (no syntax highlighting)
        let mut input_editor = ratatui_code_editor::editor::Editor::new("text", "", vec![])
            .expect("plain text editor creation should never fail");
        input_editor.set_show_line_numbers(false);

        Self {
            active_tab: ActiveTab::Messages,
            should_quit: false,
            selected_thread: 0,
            message_scroll: 0,
            message_auto_scroll: true,
            event_log: Vec::new(),
            threads: Vec::new(),
            messages: Vec::new(),
            context: None,
            total_input_tokens: 0,
            total_output_tokens: 0,
            input_editor,
            input_area: Rect::new(0, 0, 80, 3), // sensible default, updated by renderer
            agent_status: AgentStatus::Idle,
            pending_task: None,
            last_response: None,
            chat_log: Vec::new(),
            viewport_height: 20, // sensible default, updated by renderer
            activity_log: Vec::new(),
            activity_scroll: 0,
            activity_auto_scroll: true,
            activity_viewport_height: 20,
            thread_conversations: std::collections::HashMap::new(),
            conversation_scroll: 0,
            conversation_auto_scroll: true,
            conversation_viewport_height: 20,
            llm_pool: None,
            pending_command: None,
            threads_focus: ThreadsFocus::ThreadList,
            context_tree_state: tui_tree_widget::TreeState::default(),
            debug_mode: false,
            menu_state: MenuState::new(build_menu_items(false, &[], None)),
            menu_active: false,
            command_popup_index: 0,
            cmd_service: CommandLineService::new(),
            yaml_editor: None,
            yaml_status: None,
            yaml_area: Rect::default(),
            lang_service: OrganismYamlService::new(),
            diagnostics: Vec::new(),
            completion_items: Vec::new(),
            completion_index: 0,
            completion_visible: false,
            hover_info: None,
            diag_debounce: 0,
            diag_summary: String::new(),
            input_mode: InputMode::Normal,
            models_config: Arc::new(Mutex::new(ModelsConfig::default())),
            pending_provider_completion: None,
            selected_agent: None,
            agents_config: AgentsConfig::default(),
        }
    }

    /// Rebuild the menu item tree (e.g., after toggling debug mode or changing favorites).
    pub fn rebuild_menu(&mut self) {
        self.menu_state = MenuState::new(build_menu_items(
            self.debug_mode,
            &self.agents_config.favorites,
            self.selected_agent.as_deref(),
        ));
    }

    /// High-contrast theme tuned for YAML readability on dark terminals.
    fn yaml_theme() -> Vec<(&'static str, &'static str)> {
        vec![
            // Keys — cyan, stands out as structure
            ("identifier", "#00DDDD"),
            // String values — green
            ("string", "#66DD66"),
            ("string.escape", "#DDDD00"),
            // Numbers — orange
            ("number", "#FFAA44"),
            ("integer", "#FFAA44"),
            ("float", "#FFAA44"),
            // Booleans — magenta
            ("boolean", "#DD77DD"),
            // Null — dim magenta
            ("constant.builtin", "#AA55AA"),
            // Comments — dim gray
            ("comment", "#666666"),
            // Punctuation (: - , > | ?) — white, subtle
            ("punctuation.delimiter", "#AAAAAA"),
            // Brackets ([] {}) — white
            ("punctuation.bracket", "#CCCCCC"),
            // Anchors & aliases — yellow
            ("type", "#DDDD00"),
            // Directives — dim cyan
            ("preproc", "#55AAAA"),
            // Errors — bright red
            ("error", "#FF4444"),
            // Fallbacks for other languages if theme reused
            ("keyword", "#00DDDD"),
            ("function", "#FFAA44"),
            ("variable", "#FFFFFF"),
            ("variable.builtin", "#FFFFFF"),
            ("namespace", "#FFAA44"),
            ("type.builtin", "#FFAA44"),
        ]
    }

    /// Load content into the YAML editor with syntax highlighting.
    pub fn load_yaml_editor(&mut self, content: &str) {
        match ratatui_code_editor::editor::Editor::new("yaml", content, Self::yaml_theme()) {
            Ok(editor) => {
                self.yaml_editor = Some(editor);
                self.yaml_status = None;
                // Run initial diagnostics
                self.run_diagnostics();
            }
            Err(e) => {
                tracing::warn!("Failed to create YAML editor: {e}");
                self.yaml_editor = None;
            }
        }
    }

    /// Run diagnostics on current YAML editor content and update marks + summary.
    pub fn run_diagnostics(&mut self) {
        let Some(ref editor) = self.yaml_editor else { return };
        let content = editor.get_content();
        self.diagnostics = self.lang_service.diagnostics(&content);

        // Convert diagnostics to editor marks (char offset ranges with colors)
        let marks: Vec<(usize, usize, &str)> = self.diagnostics.iter().filter_map(|d| {
            let start_line = d.range.start.line as usize;
            let end_line = d.range.end.line as usize;
            let code = &editor.code_ref();
            let line_count = code.len_lines();
            if start_line >= line_count { return None; }
            let start_char = code.line_to_char(start_line) + d.range.start.character as usize;
            // End: if same position, mark whole line
            let end_char = if start_line == end_line && d.range.start.character == d.range.end.character {
                // Mark to end of line
                code.line_to_char(start_line) + code.line_len(start_line)
            } else {
                let el = end_line.min(line_count.saturating_sub(1));
                code.line_to_char(el) + d.range.end.character as usize
            };
            let color = match d.severity {
                Some(lsp_types::DiagnosticSeverity::ERROR) => "#FF4444",
                Some(lsp_types::DiagnosticSeverity::WARNING) => "#FFAA44",
                _ => "#FFAA44",
            };
            Some((start_char, end_char.min(code.len_chars()), color))
        }).collect();

        // Apply marks to editor
        let editor = self.yaml_editor.as_mut().unwrap();
        if marks.is_empty() {
            editor.remove_marks();
        } else {
            editor.set_marks(marks);
        }

        // Build summary
        let errors = self.diagnostics.iter()
            .filter(|d| d.severity == Some(lsp_types::DiagnosticSeverity::ERROR))
            .count();
        let warnings = self.diagnostics.iter()
            .filter(|d| d.severity == Some(lsp_types::DiagnosticSeverity::WARNING))
            .count();
        self.diag_summary = if errors == 0 && warnings == 0 {
            String::new()
        } else {
            let mut parts = Vec::new();
            if errors > 0 { parts.push(format!("{errors} error{}", if errors == 1 { "" } else { "s" })); }
            if warnings > 0 { parts.push(format!("{warnings} warning{}", if warnings == 1 { "" } else { "s" })); }
            parts.join(", ")
        };
        self.diag_debounce = 0;
    }

    /// Trigger completion at current cursor position.
    pub fn trigger_completions(&mut self) {
        let Some(ref editor) = self.yaml_editor else { return };
        let content = editor.get_content();
        let cursor = editor.get_cursor();
        let code = editor.code_ref();
        let (row, col) = code.point(cursor);
        let pos = lsp_types::Position::new(row as u32, col as u32);
        self.completion_items = self.lang_service.completions(&content, pos);
        self.completion_index = 0;
        self.completion_visible = !self.completion_items.is_empty();
    }

    /// Trigger hover at current cursor position.
    pub fn trigger_hover(&mut self) {
        let Some(ref editor) = self.yaml_editor else { return };
        let content = editor.get_content();
        let cursor = editor.get_cursor();
        let code = editor.code_ref();
        let (row, col) = code.point(cursor);
        let pos = lsp_types::Position::new(row as u32, col as u32);
        self.hover_info = self.lang_service.hover(&content, pos);
    }

    /// Accept the currently selected completion item.
    pub fn accept_completion(&mut self) {
        if !self.completion_visible { return; }
        let Some(item) = self.completion_items.get(self.completion_index) else { return };
        let text = item.insert_text.as_deref().unwrap_or(&item.label);
        let text = text.to_string();
        if let Some(ref mut editor) = self.yaml_editor {
            let cursor = editor.get_cursor();
            let code = editor.code_ref();
            // Find start of current word (for replacement)
            let (row, col) = code.point(cursor);
            let line_start = code.line_to_char(row);
            let line_text: String = code.line(row).to_string();
            // Find word start from cursor col
            let mut word_start = col;
            while word_start > 0 {
                let c = line_text.as_bytes().get(word_start - 1).copied().unwrap_or(b' ');
                if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' {
                    word_start -= 1;
                } else {
                    break;
                }
            }
            // Replace word with completion
            let abs_word_start = line_start + word_start;
            let code = editor.code_mut();
            code.tx();
            code.set_state_before(cursor, None);
            if cursor > abs_word_start {
                code.remove(abs_word_start, cursor);
            }
            code.insert(abs_word_start, &text);
            let new_cursor = abs_word_start + text.chars().count();
            code.set_state_after(new_cursor, None);
            code.commit();
            editor.set_cursor(new_cursor);
        }
        self.completion_visible = false;
        self.completion_items.clear();
        // Trigger diagnostics after completion
        self.diag_debounce = 8;
    }

    /// Handle a TUI message (TEA update).
    pub fn update(&mut self, msg: TuiMessage) {
        match msg {
            TuiMessage::Input(key) => {
                super::input::handle_key(self, key);
            }
            TuiMessage::Pipeline(event) => {
                self.handle_pipeline_event(event);
            }
            TuiMessage::Tick => {
                // Kernel refresh handled externally by runner
                // Debounced diagnostics: decrement counter, run when it hits 0
                if self.diag_debounce > 0 {
                    self.diag_debounce -= 1;
                    if self.diag_debounce == 0 {
                        self.run_diagnostics();
                    }
                }
            }
            TuiMessage::Render => {
                // Render handled externally by runner
            }
            TuiMessage::Quit => {
                self.should_quit = true;
            }
            TuiMessage::SubmitTask(_) => {
                // Handled by runner (task injection into pipeline)
            }
        }
    }

    /// Handle a pipeline event.
    fn handle_pipeline_event(&mut self, event: PipelineEvent) {
        match &event {
            PipelineEvent::TokenUsage {
                input_tokens,
                output_tokens,
                ..
            } => {
                self.total_input_tokens += *input_tokens as u64;
                self.total_output_tokens += *output_tokens as u64;
            }
            PipelineEvent::AgentResponse { text, .. } => {
                if text.starts_with("Error: ") {
                    self.agent_status = AgentStatus::Error(text.clone());
                } else {
                    self.agent_status = AgentStatus::Idle;
                }
                self.last_response = Some(text.clone());
                self.chat_log.push(ChatEntry {
                    role: "agent".into(),
                    text: text.clone(),
                });
                // Auto-scroll to bottom so latest response is visible
                self.message_auto_scroll = true;
                // Complete any pending "thinking" activity
                self.complete_thinking();
            }
            PipelineEvent::AgentThinking { .. } => {
                self.agent_status = AgentStatus::Thinking;
                self.push_activity(ActivityEntry {
                    timestamp: now_secs(),
                    label: "thinking".into(),
                    detail: String::new(),
                    status: ActivityStatus::InProgress,
                });
            }
            PipelineEvent::ToolDispatched {
                tool_name, detail, ..
            } => {
                self.agent_status = AgentStatus::ToolCall(tool_name.clone());
                self.push_activity(ActivityEntry {
                    timestamp: now_secs(),
                    label: tool_name.clone(),
                    detail: detail.clone(),
                    status: ActivityStatus::InProgress,
                });
            }
            PipelineEvent::ToolCompleted {
                tool_name,
                success,
                detail,
                ..
            } => {
                self.complete_activity(tool_name, *success, detail);
            }
            PipelineEvent::ConversationSync {
                thread_id, entries, ..
            } => {
                self.thread_conversations
                    .insert(thread_id.clone(), entries.clone());
                self.conversation_auto_scroll = true;
            }
            _ => {}
        }

        self.event_log.push(event);
        if self.event_log.len() > EVENT_LOG_CAPACITY {
            self.event_log.remove(0);
        }
    }

    /// Move selection up.
    pub fn move_up(&mut self) {
        if self.selected_thread > 0 {
            self.selected_thread -= 1;
        }
    }

    /// Move selection down.
    pub fn move_down(&mut self) {
        let max = if self.threads.is_empty() {
            0
        } else {
            self.threads.len() - 1
        };
        if self.selected_thread < max {
            self.selected_thread += 1;
        }
    }

    /// Scroll messages pane down.
    pub fn scroll_messages_down(&mut self) {
        self.message_auto_scroll = false;
        self.message_scroll = self.message_scroll.saturating_add(1);
    }

    /// Scroll messages pane up.
    pub fn scroll_messages_up(&mut self) {
        self.message_auto_scroll = false;
        self.message_scroll = self.message_scroll.saturating_sub(1);
    }

    /// Push an activity entry, maintaining ring buffer capacity.
    pub fn push_activity(&mut self, entry: ActivityEntry) {
        self.activity_log.push(entry);
        if self.activity_log.len() > ACTIVITY_LOG_CAPACITY {
            self.activity_log.remove(0);
        }
        self.activity_auto_scroll = true;
    }

    /// Find the last matching InProgress activity and mark it Done/Error.
    pub fn complete_activity(&mut self, tool_name: &str, success: bool, detail: &str) {
        for entry in self.activity_log.iter_mut().rev() {
            if entry.label == tool_name && entry.status == ActivityStatus::InProgress {
                entry.status = if success {
                    ActivityStatus::Done
                } else {
                    ActivityStatus::Error
                };
                if !detail.is_empty() {
                    entry.detail = detail.to_string();
                }
                break;
            }
        }
    }

    /// Mark the last "thinking" entry as Done.
    pub fn complete_thinking(&mut self) {
        for entry in self.activity_log.iter_mut().rev() {
            if entry.label == "thinking" && entry.status == ActivityStatus::InProgress {
                entry.status = ActivityStatus::Done;
                break;
            }
        }
    }

    /// Scroll conversation pane down.
    pub fn scroll_conversation_down(&mut self) {
        self.conversation_auto_scroll = false;
        self.conversation_scroll = self.conversation_scroll.saturating_add(1);
    }

    /// Scroll conversation pane up.
    pub fn scroll_conversation_up(&mut self) {
        self.conversation_auto_scroll = false;
        self.conversation_scroll = self.conversation_scroll.saturating_sub(1);
    }

    /// Scroll activity pane down.
    pub fn scroll_activity_down(&mut self) {
        self.activity_auto_scroll = false;
        self.activity_scroll = self.activity_scroll.saturating_add(1);
    }

    /// Scroll activity pane up.
    pub fn scroll_activity_up(&mut self) {
        self.activity_auto_scroll = false;
        self.activity_scroll = self.activity_scroll.saturating_sub(1);
    }

    /// Extract text from the input editor and clear it. Returns None if empty.
    pub fn take_input(&mut self) -> Option<String> {
        let text = self.input_editor.get_content();
        let text = text.trim().to_string();
        if text.is_empty() {
            return None;
        }
        self.input_editor.set_content("");
        Some(text)
    }

    /// Get the current input text (first line).
    pub fn input_text(&self) -> String {
        let content = self.input_editor.get_content();
        content.lines().next().unwrap_or("").to_string()
    }

    /// Clear the input editor.
    pub fn clear_input(&mut self) {
        self.input_editor.set_content("");
    }

    /// Set the input editor content and move cursor to end.
    pub fn set_input_text(&mut self, text: &str) {
        self.input_editor.set_content(text);
        let len = self.input_editor.get_content().chars().count();
        self.input_editor.set_cursor(len);
    }

    /// Whether the wizard is active.
    pub fn in_wizard(&self) -> bool {
        matches!(self.input_mode, InputMode::ProviderWizard { .. })
    }
}

impl Default for TuiApp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::context_store::{ContextInventory, SegmentMeta, SegmentStatus};
    use crate::kernel::thread_table::ThreadRecord;
    use crate::pipeline::events::PipelineEvent;
    use crate::tui::input::handle_key;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn app_default_state() {
        let app = TuiApp::new();
        assert_eq!(app.active_tab, ActiveTab::Messages);
        assert!(!app.should_quit);
        assert_eq!(app.selected_thread, 0);
    }

    #[test]
    fn app_quit_on_ctrl_c() {
        let mut app = TuiApp::new();
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Char('c'),
            KeyModifiers::CONTROL,
        )));
        assert!(app.should_quit);
    }

    #[test]
    fn app_typing_goes_to_input_editor() {
        let mut app = TuiApp::new();
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Char('q'),
            KeyModifiers::NONE,
        )));
        // 'q' types into input editor, doesn't quit
        assert!(!app.should_quit);
        assert!(app.input_text().contains('q'));
    }

    #[test]
    fn app_move_up_down() {
        let mut app = TuiApp::new();
        app.threads = vec![
            ThreadView {
                uuid: "a".into(),
                chain: "system.org".into(),
                profile: "admin".into(),
                created_at: 0,
            },
            ThreadView {
                uuid: "b".into(),
                chain: "system.org.handler".into(),
                profile: "admin".into(),
                created_at: 0,
            },
        ];

        app.move_down();
        assert_eq!(app.selected_thread, 1);

        app.move_up();
        assert_eq!(app.selected_thread, 0);
    }

    #[test]
    fn app_move_clamped() {
        let mut app = TuiApp::new();
        app.move_up();
        assert_eq!(app.selected_thread, 0);

        app.move_down();
        assert_eq!(app.selected_thread, 0);

        app.threads = vec![ThreadView {
            uuid: "a".into(),
            chain: "c".into(),
            profile: "p".into(),
            created_at: 0,
        }];
        app.move_down();
        assert_eq!(app.selected_thread, 0);
    }

    #[test]
    fn tui_message_variants() {
        // Just verify all variants construct without panic
        let _input = TuiMessage::Input(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
        let _pipeline = TuiMessage::Pipeline(PipelineEvent::MessageInjected {
            thread_id: "t".into(),
            target: "e".into(),
            profile: "a".into(),
        });
        let _tick = TuiMessage::Tick;
        let _render = TuiMessage::Render;
        let _quit = TuiMessage::Quit;
        let _submit = TuiMessage::SubmitTask("hello".into());
    }

    #[test]
    fn app_pipeline_event_logged() {
        let mut app = TuiApp::new();
        app.update(TuiMessage::Pipeline(PipelineEvent::MessageInjected {
            thread_id: "t1".into(),
            target: "echo".into(),
            profile: "admin".into(),
        }));
        assert_eq!(app.event_log.len(), 1);
    }

    #[test]
    fn app_token_usage_accumulates() {
        let mut app = TuiApp::new();
        app.update(TuiMessage::Pipeline(PipelineEvent::TokenUsage {
            thread_id: "t1".into(),
            input_tokens: 100,
            output_tokens: 50,
        }));
        app.update(TuiMessage::Pipeline(PipelineEvent::TokenUsage {
            thread_id: "t2".into(),
            input_tokens: 200,
            output_tokens: 100,
        }));
        assert_eq!(app.total_input_tokens, 300);
        assert_eq!(app.total_output_tokens, 150);
    }

    #[test]
    fn thread_view_from_record() {
        let record = ThreadRecord {
            uuid: "uuid-1".into(),
            chain: "system.org".into(),
            profile: "admin".into(),
            created_at: 12345,
        };
        let view = ThreadView::from(&record);
        assert_eq!(view.uuid, "uuid-1");
        assert_eq!(view.chain, "system.org");
        assert_eq!(view.profile, "admin");
        assert_eq!(view.created_at, 12345);
    }

    #[test]
    fn context_view_from_inventory() {
        let inv = ContextInventory {
            thread_id: "t1".into(),
            segments: vec![
                SegmentMeta {
                    id: "s1".into(),
                    tag: "code".into(),
                    size: 100,
                    status: SegmentStatus::Active,
                    relevance: 0.9,
                    created_at: 0,
                },
                SegmentMeta {
                    id: "s2".into(),
                    tag: "msg".into(),
                    size: 50,
                    status: SegmentStatus::Shelved,
                    relevance: 0.3,
                    created_at: 0,
                },
            ],
            active_count: 1,
            shelved_count: 1,
            folded_count: 0,
            total_bytes: 150,
            active_bytes: 100,
        };
        let view = ContextView::from(&inv);
        assert_eq!(view.thread_id, "t1");
        assert_eq!(view.segments.len(), 2);
        assert_eq!(view.active_count, 1);
        assert_eq!(view.shelved_count, 1);
        assert_eq!(view.total_bytes, 150);
        assert_eq!(view.active_bytes, 100);
    }

    #[test]
    fn tab_switching() {
        let mut app = TuiApp::new();
        assert_eq!(app.active_tab, ActiveTab::Messages);

        app.active_tab = ActiveTab::Threads;
        assert_eq!(app.active_tab, ActiveTab::Threads);

        app.active_tab = ActiveTab::Yaml;
        assert_eq!(app.active_tab, ActiveTab::Yaml);

        app.active_tab = ActiveTab::Wasm;
        assert_eq!(app.active_tab, ActiveTab::Wasm);
    }

    #[test]
    fn take_input_returns_text_and_clears() {
        let mut app = TuiApp::new();
        app.set_input_text("hello");
        let text = app.take_input();
        assert_eq!(text, Some("hello".into()));
        // Input should be empty after take
        assert!(app.input_text().is_empty());
    }

    #[test]
    fn take_input_returns_none_when_empty() {
        let mut app = TuiApp::new();
        assert_eq!(app.take_input(), None);
    }

    // ── Activity trace tests ──

    #[test]
    fn activity_populated_on_agent_thinking() {
        let mut app = TuiApp::new();
        app.update(TuiMessage::Pipeline(PipelineEvent::AgentThinking {
            thread_id: "t1".into(),
        }));
        assert_eq!(app.activity_log.len(), 1);
        assert_eq!(app.activity_log[0].label, "thinking");
        assert_eq!(app.activity_log[0].status, ActivityStatus::InProgress);
        assert_eq!(app.agent_status, AgentStatus::Thinking);
    }

    #[test]
    fn activity_populated_on_tool_dispatched() {
        let mut app = TuiApp::new();
        app.update(TuiMessage::Pipeline(PipelineEvent::ToolDispatched {
            thread_id: "t1".into(),
            tool_name: "file-read".into(),
            detail: "src/main.rs".into(),
        }));
        assert_eq!(app.activity_log.len(), 1);
        assert_eq!(app.activity_log[0].label, "file-read");
        assert_eq!(app.activity_log[0].detail, "src/main.rs");
        assert_eq!(app.activity_log[0].status, ActivityStatus::InProgress);
        assert_eq!(app.agent_status, AgentStatus::ToolCall("file-read".into()));
    }

    #[test]
    fn tool_completed_updates_in_place() {
        let mut app = TuiApp::new();
        app.update(TuiMessage::Pipeline(PipelineEvent::ToolDispatched {
            thread_id: "t1".into(),
            tool_name: "file-read".into(),
            detail: "src/main.rs".into(),
        }));
        app.update(TuiMessage::Pipeline(PipelineEvent::ToolCompleted {
            thread_id: "t1".into(),
            tool_name: "file-read".into(),
            success: true,
            detail: String::new(),
        }));
        // Should NOT add a new entry — just update the existing one
        assert_eq!(app.activity_log.len(), 1);
        assert_eq!(app.activity_log[0].status, ActivityStatus::Done);
    }

    #[test]
    fn tool_completed_error_status() {
        let mut app = TuiApp::new();
        app.update(TuiMessage::Pipeline(PipelineEvent::ToolDispatched {
            thread_id: "t1".into(),
            tool_name: "command-exec".into(),
            detail: "rm -rf /".into(),
        }));
        app.update(TuiMessage::Pipeline(PipelineEvent::ToolCompleted {
            thread_id: "t1".into(),
            tool_name: "command-exec".into(),
            success: false,
            detail: "permission denied".into(),
        }));
        assert_eq!(app.activity_log.len(), 1);
        assert_eq!(app.activity_log[0].status, ActivityStatus::Error);
        assert_eq!(app.activity_log[0].detail, "permission denied");
    }

    #[test]
    fn activity_ring_buffer_caps_at_512() {
        let mut app = TuiApp::new();
        for i in 0..600 {
            app.push_activity(ActivityEntry {
                timestamp: i,
                label: format!("tool-{i}"),
                detail: String::new(),
                status: ActivityStatus::Done,
            });
        }
        assert_eq!(app.activity_log.len(), 512);
        // Oldest entries should have been evicted
        assert_eq!(app.activity_log[0].label, "tool-88");
    }

    #[test]
    fn thinking_completed_on_agent_response() {
        let mut app = TuiApp::new();
        app.update(TuiMessage::Pipeline(PipelineEvent::AgentThinking {
            thread_id: "t1".into(),
        }));
        assert_eq!(app.activity_log[0].status, ActivityStatus::InProgress);

        app.update(TuiMessage::Pipeline(PipelineEvent::AgentResponse {
            thread_id: "t1".into(),
            text: "Done!".into(),
        }));
        assert_eq!(app.activity_log[0].status, ActivityStatus::Done);
    }

    #[test]
    fn scroll_activity_up_down() {
        let mut app = TuiApp::new();
        app.activity_scroll = 5;
        app.scroll_activity_up();
        assert_eq!(app.activity_scroll, 4);
        assert!(!app.activity_auto_scroll);

        app.scroll_activity_down();
        assert_eq!(app.activity_scroll, 5);
    }

    #[test]
    fn scroll_activity_clamps_at_zero() {
        let mut app = TuiApp::new();
        app.activity_scroll = 0;
        app.scroll_activity_up();
        assert_eq!(app.activity_scroll, 0);
    }

    #[test]
    fn threads_focus_default() {
        let app = TuiApp::new();
        assert_eq!(app.threads_focus, super::ThreadsFocus::ThreadList);
    }

    #[test]
    fn context_tree_state_default() {
        let app = TuiApp::new();
        assert!(app.context_tree_state.selected().is_empty());
    }

    #[test]
    fn debug_mode_default_false() {
        let app = TuiApp::new();
        assert!(!app.debug_mode);
    }

    #[test]
    fn arrow_keys_on_threads_tab_with_threadlist_focus() {
        let mut app = TuiApp::new();
        app.active_tab = ActiveTab::Threads;
        app.threads_focus = super::ThreadsFocus::ThreadList;
        app.threads = vec![
            ThreadView {
                uuid: "a".into(),
                chain: "system.org".into(),
                profile: "admin".into(),
                created_at: 0,
            },
            ThreadView {
                uuid: "b".into(),
                chain: "system.org.handler".into(),
                profile: "admin".into(),
                created_at: 0,
            },
        ];

        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Down,
            KeyModifiers::NONE,
        )));
        assert_eq!(app.selected_thread, 1);

        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Up,
            KeyModifiers::NONE,
        )));
        assert_eq!(app.selected_thread, 0);
    }

    #[test]
    fn home_end_dispatch_to_active_tab() {
        let mut app = TuiApp::new();
        app.debug_mode = true;
        app.active_tab = ActiveTab::Debug;
        app.activity_scroll = 50;
        app.activity_auto_scroll = true;

        // Home → scroll to top (Debug tab)
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::Home,
            KeyModifiers::NONE,
        )));
        assert_eq!(app.activity_scroll, 0);
        assert!(!app.activity_auto_scroll);

        // End → auto-scroll
        app.update(TuiMessage::Input(KeyEvent::new(
            KeyCode::End,
            KeyModifiers::NONE,
        )));
        assert!(app.activity_auto_scroll);
    }

    // ── YAML editor tests ──

    #[test]
    fn yaml_editor_loads_content() {
        let mut app = TuiApp::new();
        assert!(app.yaml_editor.is_none());

        app.load_yaml_editor("key: value\n");
        assert!(app.yaml_editor.is_some());
        assert!(app.yaml_status.is_none());

        let content = app.yaml_editor.as_ref().unwrap().get_content();
        assert_eq!(content, "key: value\n");
    }

    #[test]
    fn yaml_ctrl_s_validates_good() {
        let mut app = TuiApp::new();
        app.load_yaml_editor("organism:\n  name: test\n");
        app.active_tab = ActiveTab::Yaml;

        // Ctrl+S triggers validation
        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL),
        );

        // Valid YAML → no error status, success message in chat log
        assert!(app.yaml_status.is_none());
        assert!(app.chat_log.iter().any(|e| e.text.contains("validated")));
    }

    #[test]
    fn yaml_ctrl_s_validates_bad() {
        let mut app = TuiApp::new();
        app.load_yaml_editor("key: [invalid\n");
        app.active_tab = ActiveTab::Yaml;

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL),
        );

        // Invalid YAML → error status set
        assert!(app.yaml_status.is_some());
    }

    #[test]
    fn yaml_tab_receives_keys() {
        let mut app = TuiApp::new();
        app.load_yaml_editor("hello");
        app.active_tab = ActiveTab::Yaml;
        app.yaml_area = Rect::new(0, 0, 80, 24);

        // Type a character — should go to editor, not textarea
        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE),
        );

        let content = app.yaml_editor.as_ref().unwrap().get_content();
        assert!(content.contains('x'));
        // Input editor should NOT have received the 'x'
        assert!(app.input_text().is_empty());
    }

    #[test]
    fn yaml_esc_clears_status() {
        let mut app = TuiApp::new();
        app.load_yaml_editor("bad: [yaml\n");
        app.active_tab = ActiveTab::Yaml;
        app.yaml_status = Some("parse error".into());

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
        );

        assert!(app.yaml_status.is_none());
    }

    // ── Conversation sync tests ──

    #[test]
    fn conversation_sync_populates_thread_conversations() {
        use crate::pipeline::events::ConversationEntry;
        let mut app = TuiApp::new();
        let entries = vec![
            ConversationEntry {
                role: "user".into(),
                summary: "Read the file".into(),
                is_tool_use: false,
                tool_name: None,
                is_error: false,
            },
            ConversationEntry {
                role: "assistant".into(),
                summary: "Here's the content.".into(),
                is_tool_use: false,
                tool_name: None,
                is_error: false,
            },
        ];
        app.update(TuiMessage::Pipeline(PipelineEvent::ConversationSync {
            thread_id: "thread-1".into(),
            entries,
        }));
        assert!(app.thread_conversations.contains_key("thread-1"));
        assert_eq!(app.thread_conversations["thread-1"].len(), 2);
        assert!(app.conversation_auto_scroll);
    }

    #[test]
    fn empty_thread_has_no_conversation() {
        let app = TuiApp::new();
        assert!(app.thread_conversations.is_empty());
    }

    #[test]
    fn conversation_scroll_helpers() {
        let mut app = TuiApp::new();
        app.conversation_scroll = 5;
        app.scroll_conversation_up();
        assert_eq!(app.conversation_scroll, 4);
        assert!(!app.conversation_auto_scroll);

        app.scroll_conversation_down();
        assert_eq!(app.conversation_scroll, 5);
    }

    #[test]
    fn conversation_scroll_clamps_at_zero() {
        let mut app = TuiApp::new();
        app.conversation_scroll = 0;
        app.scroll_conversation_up();
        assert_eq!(app.conversation_scroll, 0);
    }

    #[test]
    fn threads_focus_cycles_through_conversation() {
        let mut app = TuiApp::new();
        app.active_tab = ActiveTab::Threads;
        assert_eq!(app.threads_focus, ThreadsFocus::ThreadList);

        // Simulate Tab presses
        app.threads_focus = ThreadsFocus::Conversation;
        assert_eq!(app.threads_focus, ThreadsFocus::Conversation);

        app.threads_focus = ThreadsFocus::ContextTree;
        assert_eq!(app.threads_focus, ThreadsFocus::ContextTree);
    }
}
