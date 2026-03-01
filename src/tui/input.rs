//! Key binding dispatch for the TUI.
//!
//! Ctrl+C quits. Ctrl+1/2/3/4 switches tabs. Enter submits.
//! Esc clears textarea. Up/Down scroll messages. Tab completes
//! slash commands (or cycles sub-pane focus on Threads tab).
//! Everything else is forwarded to the textarea widget.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tui_menu::MenuEvent;

use crate::lsp::LanguageService;

use crate::agent::permissions::ApprovalVerdict;
use super::app::{TabId, AgentStatus, ChatEntry, InputMode, MenuAction, MessagesFocus, ProviderCompletion, ThreadsFocus, TuiApp};

/// Dispatch a selected menu action.
fn dispatch_menu_action(app: &mut TuiApp, action: MenuAction) {
    app.menu_state.reset();
    app.menu_active = false;
    match action {
        MenuAction::SwitchTab(tab) => {
            app.active_tab = tab;
        }
        MenuAction::NewTask => {
            app.clear_input();
        }
        MenuAction::Quit => {
            app.should_quit = true;
        }
        MenuAction::SetModel => {
            // Pre-fill the input with "/model " so user can type the model name
            set_input(app, "/model ");
        }
        MenuAction::ShowAbout => {
            app.chat_log.push(ChatEntry::new(
                "system",
                "AgentOS — an operating system for AI coding agents.\nNo compaction, ever.",
            ));
            app.message_auto_scroll = true;
        }
        MenuAction::ShowShortcuts => {
            let menu_line = if app.debug_mode {
                "  Alt+F/R/I/D/H  Open File/Run/Inspect/Debug/Help menu\n"
            } else {
                "  Alt+F/R/I/H    Open File/Run/Inspect/Help menu\n"
            };
            let activity_line = if app.debug_mode {
                "  Ctrl+Y          YAML      Ctrl+A  Activity\n"
            } else {
                "  Ctrl+Y          YAML\n"
            };
            let text = format!(
                "Keyboard shortcuts:\n\
                 {menu_line}\
                 {activity_line}\
                 \x20 F10             Open/close menu bar\n\
                 \x20 Ctrl+1..9       Switch to tab by position\n\
                 \x20 Ctrl+W          Close active tab\n\
                 \x20 Ctrl+T          Threads   Ctrl+G  Graph\n\
                 \x20 Shift+Enter     Insert newline\n\
                 \x20 Tab             Cycle focus (Threads) / autocomplete (/commands)\n\
                 \x20 Enter           Submit task or confirm\n\
                 \x20 Esc             Clear input\n\
                 \x20 Up/Down         Scroll or navigate\n\
                 \x20 PageUp/Dn       Page scroll\n\
                 \x20 Home/End        Jump to top/bottom\n\
                 \x20 Ctrl+C          Quit"
            );
            app.chat_log.push(ChatEntry::new("system", text));
            app.message_auto_scroll = true;
        }
        MenuAction::OpenAgentTab(name) => {
            app.open_agent_tab(&name);
        }
        MenuAction::CloseTab => {
            let tab = app.active_tab.clone();
            app.close_tab(&tab);
        }
    }
}

/// Push a system message to the chat log.
fn push_feedback(app: &mut TuiApp, text: &str) {
    app.chat_log.push(ChatEntry::new("system", text));
    app.message_auto_scroll = true;
}

/// Toggle a utility tab open/close. If already open and active, close it.
/// If open but not active, switch to it. If not open, open and switch.
fn toggle_utility_tab(app: &mut TuiApp, tab: TabId) {
    if app.active_tab == tab {
        // Already focused — close it
        app.close_tab(&tab);
    } else if app.open_tabs.contains(&tab) {
        // Open but not focused — switch to it
        app.active_tab = tab;
    } else {
        // Not open — open and switch
        app.open_tabs.push(tab.clone());
        app.active_tab = tab;
    }
}

/// Get the current input text (first line).
fn current_input(app: &TuiApp) -> String {
    app.input_text()
}

/// Replace the last (possibly partial) token with the completion text.
/// If the completion starts with `/`, it replaces the whole input (command name).
/// Otherwise it replaces the last whitespace-delimited token.
fn complete_token(input: &str, completion: &str) -> String {
    if completion.starts_with('/') {
        // Command name completion — replaces entire input
        completion.to_string()
    } else {
        // Argument completion — keep prefix up to last token boundary
        let prefix = if input.ends_with(' ') {
            input
        } else {
            input.rsplit_once(' ').map(|(p, _)| p).unwrap_or(input)
        };
        if prefix.ends_with(' ') {
            format!("{prefix}{completion}")
        } else {
            format!("{prefix} {completion}")
        }
    }
}

/// Replace the input editor content with new text and move cursor to end.
fn set_input(app: &mut TuiApp, text: &str) {
    app.set_input_text(text);
}

/// Handle a key event, mutating app state.
pub fn handle_key(app: &mut TuiApp, key: KeyEvent) {
    // Ctrl+C: copy if selection active, quit otherwise
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        if app.text_selection.active {
            // Selection was already copied on mouse-up, just clear it
            app.text_selection.active = false;
            return;
        }
        app.should_quit = true;
        return;
    }

    // Any other keystroke clears text selection
    app.text_selection.active = false;

    // Tool approval mode: [1]/Enter approves, [2]/Esc denies
    if app.pending_approval.is_some() {
        match key.code {
            KeyCode::Char('1') | KeyCode::Enter => {
                if let Some(request) = app.pending_approval.take() {
                    let tool = request.tool_name.clone();
                    let _ = request.response_tx.send(ApprovalVerdict::Approved);
                    push_feedback(app, &format!("Approved: {tool}"));
                }
                return;
            }
            KeyCode::Char('2') | KeyCode::Esc => {
                if let Some(request) = app.pending_approval.take() {
                    let tool = request.tool_name.clone();
                    let _ = request.response_tx.send(ApprovalVerdict::Denied);
                    push_feedback(app, &format!("Denied: {tool}"));
                }
                return;
            }
            _ => return, // Ignore all other keys while approval is pending
        }
    }

    // F10 toggles menu bar
    if key.code == KeyCode::F(10) {
        if app.menu_active {
            app.menu_state.reset();
            app.menu_active = false;
        } else {
            app.menu_state.activate();
            app.menu_active = true;
        }
        return;
    }

    // Alt+letter opens a specific menu group (Windows-style accelerators)
    // Non-debug: File(0) Run(1) Inspect(2) Help(3)
    // Debug:     File(0) Run(1) Inspect(2) Debug(3) Help(4)
    if key.modifiers.contains(KeyModifiers::ALT) {
        let menu_index = match key.code {
            KeyCode::Char('f') => Some(0), // File
            KeyCode::Char('r') => Some(1), // Run
            KeyCode::Char('i') => Some(2), // Inspect
            KeyCode::Char('d') if app.debug_mode => Some(3), // Debug (debug only)
            KeyCode::Char('h') => Some(if app.debug_mode { 4 } else { 3 }), // Help
            _ => None,
        };
        if let Some(index) = menu_index {
            app.menu_state.reset();
            app.menu_state.activate(); // highlights first group (File)
            for _ in 0..index {
                app.menu_state.right(); // navigate to the target group
            }
            app.menu_state.down(); // open the dropdown
            app.menu_active = true;
            return;
        }
    }

    // When menu is active, route keys to menu navigation
    if app.menu_active {
        match key.code {
            KeyCode::Left => app.menu_state.left(),
            KeyCode::Right => app.menu_state.right(),
            KeyCode::Up => app.menu_state.up(),
            KeyCode::Down => app.menu_state.down(),
            KeyCode::Enter => app.menu_state.select(),
            KeyCode::Esc => {
                app.menu_state.reset();
                app.menu_active = false;
            }
            _ => {}
        }
        // Drain and dispatch any selected menu actions
        let events: Vec<_> = app.menu_state.drain_events().collect();
        for event in events {
            let MenuEvent::Selected(action) = event;
            dispatch_menu_action(app, action);
        }
        return;
    }

    // Provider wizard mode: single step — paste API key and Enter
    if let InputMode::ProviderWizard { ref provider } = app.input_mode.clone() {
        match key.code {
            KeyCode::Esc => {
                app.input_mode = InputMode::Normal;
                app.clear_input();
                app.chat_log.push(ChatEntry::new("system", "Provider wizard cancelled."));
                app.message_auto_scroll = true;
                return;
            }
            KeyCode::Enter => {
                let value = app.take_input().unwrap_or_default();
                if value.is_empty() {
                    return; // require non-empty API key
                }
                // Store pending completion for async processing in runner
                app.pending_provider_completion = Some(ProviderCompletion {
                    provider: provider.clone(),
                    api_key: value,
                });
                app.input_mode = InputMode::Normal;
                return;
            }
            _ => {
                // Forward to input line
                app.input_line.handle_key(key);
                return;
            }
        }
    }

    // Ctrl+1..9 switch tabs by position (like browser tabs)
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char(c @ '1'..='9') => {
                let idx = (c as usize) - ('1' as usize);
                if let Some(tab) = app.open_tabs.get(idx) {
                    app.active_tab = tab.clone();
                }
                return;
            }
            // Ctrl+W closes active tab
            KeyCode::Char('w') => {
                let tab = app.active_tab.clone();
                app.close_tab(&tab);
                return;
            }
            // Utility tab shortcuts: Ctrl+T/G/Y/A toggle open/close
            KeyCode::Char('t') => {
                toggle_utility_tab(app, TabId::Threads);
                return;
            }
            KeyCode::Char('g') => {
                toggle_utility_tab(app, TabId::Graph);
                return;
            }
            KeyCode::Char('y') => {
                toggle_utility_tab(app, TabId::Yaml);
                return;
            }
            KeyCode::Char('a') if app.debug_mode => {
                toggle_utility_tab(app, TabId::Activity);
                return;
            }
            _ => {}
        }
    }

    // YAML tab: route keys to code editor when active
    if app.active_tab == TabId::Yaml {
        if let Some(ref mut editor) = app.yaml_editor {
            match key.code {
                // Ctrl+S validates YAML
                KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    let content = editor.get_content();
                    match serde_yaml::from_str::<serde_yaml::Value>(&content) {
                        Ok(_) => {
                            app.yaml_status = None;
                            app.chat_log.push(ChatEntry::new("system", "YAML validated successfully."));
                            app.message_auto_scroll = true;
                        }
                        Err(e) => {
                            app.yaml_status = Some(format!("{e}"));
                        }
                    }
                }
                // Esc on YAML tab: dismiss popups, then clear status, then clear textarea
                KeyCode::Esc => {
                    if app.completion_visible {
                        app.completion_visible = false;
                    } else if app.hover_info.is_some() {
                        app.hover_info = None;
                    } else if app.yaml_status.is_some() {
                        app.yaml_status = None;
                    } else {
                        app.clear_input();
                    }
                }
                // Ctrl+Space triggers completions
                KeyCode::Char(' ') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    app.trigger_completions();
                }
                // Ctrl+H triggers hover
                KeyCode::Char('h') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    app.trigger_hover();
                }
                // Completion popup navigation
                KeyCode::Up if app.completion_visible => {
                    if app.completion_index > 0 {
                        app.completion_index -= 1;
                    }
                }
                KeyCode::Down if app.completion_visible => {
                    if app.completion_index + 1 < app.completion_items.len() {
                        app.completion_index += 1;
                    }
                }
                KeyCode::Tab | KeyCode::Enter if app.completion_visible => {
                    app.accept_completion();
                }
                // Everything else goes to the code editor
                _ => {
                    let area = app.yaml_area;
                    let _ = editor.input(key, &area);
                    // Dismiss hover and completion on any edit
                    app.hover_info = None;
                    if app.completion_visible {
                        app.completion_visible = false;
                    }
                    // Schedule debounced diagnostics (4 ticks ≈ 1s at 4Hz tick rate)
                    app.diag_debounce = 4;
                }
            }
            return;
        }
    }

    // Command popup: when input starts with `/`, use CommandLineService for completions
    {
        let input = current_input(app);
        if input.starts_with('/') {
            let items = app
                .cmd_service
                .completions(&input, lsp_types::Position::new(0, input.len() as u32));
            if !items.is_empty() {
                match key.code {
                    KeyCode::Up => {
                        if app.command_popup_index > 0 {
                            app.command_popup_index -= 1;
                        }
                        return;
                    }
                    KeyCode::Down => {
                        if app.command_popup_index + 1 < items.len() {
                            app.command_popup_index += 1;
                        }
                        return;
                    }
                    KeyCode::Tab => {
                        if let Some(item) = items.get(app.command_popup_index) {
                            let text = item.insert_text.as_deref().unwrap_or(&item.label);
                            let completed = complete_token(&input, text);
                            set_input(app, &completed);
                            app.command_popup_index = 0;
                        }
                        return;
                    }
                    KeyCode::Enter => {
                        if let Some(item) = items.get(app.command_popup_index) {
                            let text = item.insert_text.as_deref().unwrap_or(&item.label);
                            let completed = complete_token(&input, text);
                            // If completed text ends with space, command needs more input
                            if completed.ends_with(' ') {
                                set_input(app, &completed);
                                app.command_popup_index = 0;
                                return;
                            }
                            // Complete value — fill and fall through to submit
                            set_input(app, &completed);
                            app.command_popup_index = 0;
                            // Don't return — let the Enter handler below submit it
                        }
                    }
                    KeyCode::Esc => {
                        app.clear_input();
                        app.command_popup_index = 0;
                        return;
                    }
                    _ => {
                        // Typing continues — reset selection to top, fall through
                        app.command_popup_index = 0;
                    }
                }
            } else {
                app.command_popup_index = 0;
            }
        } else {
            // No popup — reset index
            app.command_popup_index = 0;
        }
    }

    match key.code {
        // Shift+Enter: insert newline (multiline input)
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
            app.input_line.insert_char('\n');
        }
        // Enter: submit task or slash command
        KeyCode::Enter => {
            // On Threads tab with ContextTree focus, toggle the selected node
            if app.active_tab == TabId::Threads
                && app.threads_focus == ThreadsFocus::ContextTree
            {
                app.context_tree_state.toggle_selected();
                return;
            }
            if let Some(text) = app.take_input() {
                if text.starts_with('/') {
                    // Slash command — always allowed, even while agent is busy
                    app.pending_command = Some(text);
                } else if app.agent_status == AgentStatus::Idle {
                    // Push to active agent tab
                    if let Some(tab) = app.active_agent_tab_mut() {
                        tab.chat_log.push(ChatEntry::new("user", text.clone()));
                        tab.agent_status = AgentStatus::Thinking;
                        tab.message_auto_scroll = true;
                    }
                    // Bridge: keep global state
                    app.chat_log.push(ChatEntry::new("user", text.clone()));
                    app.agent_status = AgentStatus::Thinking;
                    app.message_auto_scroll = true;
                    app.pending_task = Some(text);
                } else {
                    // Agent is busy — put text back, don't submit
                    app.set_input_text(&text);
                }
            }
        }
        // Tab: focus cycling on Threads tab, otherwise forward to input
        KeyCode::Tab => {
            if app.active_tab == TabId::Threads {
                app.threads_focus = match app.threads_focus {
                    ThreadsFocus::ThreadList => ThreadsFocus::Conversation,
                    ThreadsFocus::Conversation => ThreadsFocus::ContextTree,
                    ThreadsFocus::ContextTree => ThreadsFocus::ThreadList,
                };
            } else {
                // Normal Tab → insert tab character (or ignore)
                app.input_line.handle_key(key);
            }
        }
        // Clear input
        KeyCode::Esc => {
            app.clear_input();
        }
        // Arrow keys dispatched based on active tab + focus
        KeyCode::Up if app.active_tab.is_agent() => {
            for _ in 0..3 {
                app.scroll_messages_up();
            }
        }
        KeyCode::Down if app.active_tab.is_agent() => {
            for _ in 0..3 {
                app.scroll_messages_down();
            }
        }
        KeyCode::Up if app.active_tab == TabId::Threads => match app.threads_focus {
            ThreadsFocus::ThreadList => {
                app.move_up();
            }
            ThreadsFocus::Conversation => {
                for _ in 0..3 {
                    app.scroll_conversation_up();
                }
            }
            ThreadsFocus::ContextTree => {
                app.context_tree_state.key_up();
            }
        },
        KeyCode::Down if app.active_tab == TabId::Threads => match app.threads_focus {
            ThreadsFocus::ThreadList => {
                app.move_down();
            }
            ThreadsFocus::Conversation => {
                for _ in 0..3 {
                    app.scroll_conversation_down();
                }
            }
            ThreadsFocus::ContextTree => {
                app.context_tree_state.key_down();
            }
        },
        // Activity tab: arrow keys scroll activity trace
        KeyCode::Up if app.active_tab == TabId::Activity => {
            for _ in 0..3 {
                app.scroll_activity_up();
            }
        }
        KeyCode::Down if app.active_tab == TabId::Activity => {
            for _ in 0..3 {
                app.scroll_activity_down();
            }
        }
        // Graph tab: arrow keys scroll
        KeyCode::Up if app.active_tab == TabId::Graph => {
            for _ in 0..3 {
                app.scroll_graph_up();
            }
        }
        KeyCode::Down if app.active_tab == TabId::Graph => {
            for _ in 0..3 {
                app.scroll_graph_down();
            }
        }
        KeyCode::Left if app.active_tab == TabId::Graph => {
            app.scroll_graph_left();
        }
        KeyCode::Right if app.active_tab == TabId::Graph => {
            app.scroll_graph_right();
        }
        // Left/Right on Messages tab: depends on focus
        KeyCode::Left if app.active_tab.is_agent() => {
            match app.messages_focus {
                MessagesFocus::Input => app.input_line.move_left(),
                MessagesFocus::Messages => app.scroll_messages_left(),
            }
        }
        KeyCode::Right if app.active_tab.is_agent() => {
            match app.messages_focus {
                MessagesFocus::Input => app.input_line.move_right(),
                MessagesFocus::Messages => app.scroll_messages_right(),
            }
        }
        // Left/Right on ContextTree focus: collapse/expand
        KeyCode::Left
            if app.active_tab == TabId::Threads
                && app.threads_focus == ThreadsFocus::ContextTree =>
        {
            app.context_tree_state.key_left();
        }
        KeyCode::Right
            if app.active_tab == TabId::Threads
                && app.threads_focus == ThreadsFocus::ContextTree =>
        {
            app.context_tree_state.key_right();
        }
        // Page scroll — full page minus 2 overlap lines, dispatched to active tab
        KeyCode::PageUp => match app.active_tab {
            TabId::Threads if app.threads_focus == ThreadsFocus::Conversation => {
                let page = app.conversation_viewport_height.saturating_sub(2).max(1);
                for _ in 0..page {
                    app.scroll_conversation_up();
                }
            }
            TabId::Activity => {
                let page = app.activity_viewport_height.saturating_sub(2).max(1);
                for _ in 0..page {
                    app.scroll_activity_up();
                }
            }
            TabId::Graph => {
                let page = app.graph_viewport_height.saturating_sub(2).max(1);
                for _ in 0..page {
                    app.scroll_graph_up();
                }
            }
            _ => {
                let page = app.viewport_height.saturating_sub(2).max(1);
                for _ in 0..page {
                    app.scroll_messages_up();
                }
            }
        },
        KeyCode::PageDown => match app.active_tab {
            TabId::Threads if app.threads_focus == ThreadsFocus::Conversation => {
                let page = app.conversation_viewport_height.saturating_sub(2).max(1);
                for _ in 0..page {
                    app.scroll_conversation_down();
                }
            }
            TabId::Activity => {
                let page = app.activity_viewport_height.saturating_sub(2).max(1);
                for _ in 0..page {
                    app.scroll_activity_down();
                }
            }
            TabId::Graph => {
                let page = app.graph_viewport_height.saturating_sub(2).max(1);
                for _ in 0..page {
                    app.scroll_graph_down();
                }
            }
            _ => {
                let page = app.viewport_height.saturating_sub(2).max(1);
                for _ in 0..page {
                    app.scroll_messages_down();
                }
            }
        },
        // Jump to top/bottom, dispatched to active tab + focus
        KeyCode::Home => match app.active_tab {
            TabId::Threads => match app.threads_focus {
                ThreadsFocus::ThreadList => {
                    app.selected_thread = 0;
                }
                ThreadsFocus::Conversation => {
                    app.conversation_scroll = 0;
                    app.conversation_auto_scroll = false;
                }
                ThreadsFocus::ContextTree => {
                    app.context_tree_state.select_first();
                }
            },
            TabId::Activity => {
                app.activity_scroll = 0;
                app.activity_auto_scroll = false;
            }
            TabId::Graph => {
                app.graph_scroll = 0;
                app.graph_h_scroll = 0;
            }
            _ => {
                app.message_scroll = 0;
                app.message_h_scroll = 0;
                app.message_auto_scroll = false;
            }
        },
        KeyCode::End => match app.active_tab {
            TabId::Threads => match app.threads_focus {
                ThreadsFocus::ThreadList => {
                    if !app.threads.is_empty() {
                        app.selected_thread = app.threads.len() - 1;
                    }
                }
                ThreadsFocus::Conversation => {
                    app.conversation_auto_scroll = true;
                }
                ThreadsFocus::ContextTree => {
                    app.context_tree_state.select_last();
                }
            },
            TabId::Activity => {
                app.activity_auto_scroll = true;
            }
            TabId::Graph => {
                // Scroll to bottom
                app.graph_scroll = u16::MAX;
            }
            _ => {
                app.message_auto_scroll = true;
            }
        },
        // Everything else → input line (typing implicitly focuses input)
        _ => {
            if app.active_tab.is_agent() {
                app.messages_focus = MessagesFocus::Input;
            }
            app.input_line.handle_key(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn type_text(app: &mut TuiApp, text: &str) {
        app.set_input_text(text);
    }

    #[test]
    fn tab_completes_slash_command() {
        let mut app = TuiApp::new();
        type_text(&mut app, "/mo");

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
        );

        assert_eq!(app.input_text(), "/model ");
    }

    #[test]
    fn tab_no_slash_forwards_to_editor() {
        let mut app = TuiApp::new();
        type_text(&mut app, "hello");

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
        );

        // Tab forwarded to input editor — input still contains "hello"
        let text = app.input_text();
        assert!(text.contains("hello"));
    }

    #[test]
    fn enter_with_slash_sets_pending_command() {
        let mut app = TuiApp::new();
        type_text(&mut app, "/clear");

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        );

        assert_eq!(app.pending_command, Some("/clear".into()));
        assert!(app.pending_task.is_none());
        // Chat log should NOT have a user entry (commands are not shown as user messages)
        assert!(app.chat_log.is_empty());
    }

    #[test]
    fn enter_without_slash_sets_pending_task() {
        let mut app = TuiApp::new();
        type_text(&mut app, "read the file");

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        );

        assert_eq!(app.pending_task, Some("read the file".into()));
        assert!(app.pending_command.is_none());
        assert_eq!(app.chat_log.len(), 1);
        assert_eq!(app.chat_log[0].role, "user");
    }

    // ── Threads tab focus cycling ──

    #[test]
    fn tab_cycles_threads_focus() {
        let mut app = TuiApp::new();
        app.active_tab = TabId::Threads;
        assert_eq!(app.threads_focus, ThreadsFocus::ThreadList);

        handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.threads_focus, ThreadsFocus::Conversation);

        handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.threads_focus, ThreadsFocus::ContextTree);

        handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.threads_focus, ThreadsFocus::ThreadList);
    }

    #[test]
    fn tab_on_messages_tab_goes_to_editor() {
        let mut app = TuiApp::new();
        app.active_tab = TabId::Agent("planner".into());
        type_text(&mut app, "hello");

        handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));

        // Tab forwarded to input editor (not focus cycling)
        let text = app.input_text();
        assert!(text.contains("hello"));
    }

    #[test]
    fn up_down_on_thread_list() {
        let mut app = TuiApp::new();
        app.active_tab = TabId::Threads;
        app.threads_focus = ThreadsFocus::ThreadList;
        app.threads = vec![
            super::super::app::ThreadView {
                uuid: "a".into(),
                chain: "system.org".into(),
                profile: "admin".into(),
                created_at: 0,
            },
            super::super::app::ThreadView {
                uuid: "b".into(),
                chain: "system.org.handler".into(),
                profile: "admin".into(),
                created_at: 0,
            },
        ];

        handle_key(&mut app, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(app.selected_thread, 1);

        handle_key(&mut app, KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(app.selected_thread, 0);
    }

    #[test]
    fn up_down_on_context_tree() {
        let mut app = TuiApp::new();
        app.active_tab = TabId::Threads;
        app.threads_focus = ThreadsFocus::ContextTree;

        // key_up/key_down on TreeState with no rendered items is a no-op — just verify no panic
        handle_key(&mut app, KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        handle_key(&mut app, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        // No crash = pass
    }

    #[test]
    fn up_down_on_activity_tab() {
        let mut app = TuiApp::new();
        app.active_tab = TabId::Activity;
        app.activity_scroll = 10;

        handle_key(&mut app, KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(app.activity_scroll, 7); // 3 lines per step

        handle_key(&mut app, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(app.activity_scroll, 10);
    }

    #[test]
    fn ctrl_a_toggles_activity_in_debug_mode() {
        let mut app = TuiApp::new();
        app.debug_mode = true;

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL),
        );
        assert_eq!(app.active_tab, TabId::Activity);
    }

    #[test]
    fn ctrl_a_noop_without_debug() {
        let mut app = TuiApp::new();
        // debug_mode is false by default
        let original_tab = app.active_tab.clone();

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL),
        );
        // Tab should not have changed — Ctrl+A is ignored outside debug mode
        assert_eq!(app.active_tab, original_tab);
    }

    #[test]
    fn ctrl_g_toggles_graph() {
        let mut app = TuiApp::new();

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('g'), KeyModifiers::CONTROL),
        );
        assert_eq!(app.active_tab, TabId::Graph);
    }

    #[test]
    fn home_end_on_activity_tab() {
        let mut app = TuiApp::new();
        app.active_tab = TabId::Activity;
        app.activity_scroll = 50;
        app.activity_auto_scroll = false;

        handle_key(&mut app, KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        assert_eq!(app.activity_scroll, 0);
        assert!(!app.activity_auto_scroll);

        handle_key(&mut app, KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
        assert!(app.activity_auto_scroll);
    }

    #[test]
    fn left_right_on_context_tree() {
        let mut app = TuiApp::new();
        app.active_tab = TabId::Threads;
        app.threads_focus = ThreadsFocus::ContextTree;

        // No rendered items yet, just verify no panic
        handle_key(&mut app, KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        handle_key(&mut app, KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
    }

    #[test]
    fn enter_on_context_tree() {
        let mut app = TuiApp::new();
        app.active_tab = TabId::Threads;
        app.threads_focus = ThreadsFocus::ContextTree;

        // Enter should toggle_selected (no-op when nothing selected, but no panic)
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        // Should NOT set pending_task
        assert!(app.pending_task.is_none());
    }

    // ── Menu bar tests ──

    #[test]
    fn menu_f10_toggles_active() {
        let mut app = TuiApp::new();
        assert!(!app.menu_active);

        handle_key(&mut app, KeyEvent::new(KeyCode::F(10), KeyModifiers::NONE));
        assert!(app.menu_active);

        handle_key(&mut app, KeyEvent::new(KeyCode::F(10), KeyModifiers::NONE));
        assert!(!app.menu_active);
    }

    #[test]
    fn menu_esc_closes() {
        let mut app = TuiApp::new();
        // Activate menu
        handle_key(&mut app, KeyEvent::new(KeyCode::F(10), KeyModifiers::NONE));
        assert!(app.menu_active);

        // Esc closes
        handle_key(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!app.menu_active);
    }

    #[test]
    fn menu_action_quit() {
        let mut app = TuiApp::new();
        dispatch_menu_action(&mut app, MenuAction::Quit);
        assert!(app.should_quit);
        assert!(!app.menu_active);
    }

    #[test]
    fn menu_action_switch_tab() {
        let mut app = TuiApp::new();
        app.menu_active = true;
        dispatch_menu_action(&mut app, MenuAction::SwitchTab(TabId::Threads));
        assert_eq!(app.active_tab, TabId::Threads);
        assert!(!app.menu_active);
    }

    #[test]
    fn menu_state_default() {
        let app = TuiApp::new();
        assert!(!app.menu_active);
        // Menu state exists but is not user-activated
        assert_eq!(app.active_tab, TabId::Agent("planner".into()));
    }

    #[test]
    fn enter_on_model_arg_submits_full_command() {
        let mut app = TuiApp::new();
        // Type "/model " — popup shows opus/sonnet/sonnet-4.5/haiku
        type_text(&mut app, "/model ");
        // Cursor down three times to reach "haiku" (index 3)
        handle_key(&mut app, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        handle_key(&mut app, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        handle_key(&mut app, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(app.command_popup_index, 3);
        // Enter should submit "/model haiku" as a command
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.pending_command, Some("/model haiku".into()));
        assert!(app.pending_task.is_none());
    }

    #[test]
    fn tab_on_model_arg_completes_value() {
        let mut app = TuiApp::new();
        type_text(&mut app, "/model h");
        // Tab should complete to "/model haiku"
        handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.input_text(), "/model haiku");
    }

    #[test]
    fn complete_token_preserves_prefix() {
        assert_eq!(complete_token("/model ", "haiku"), "/model haiku");
        assert_eq!(complete_token("/model h", "haiku"), "/model haiku");
        assert_eq!(complete_token("/mo", "/model "), "/model ");
        assert_eq!(complete_token("/", "/exit"), "/exit");
    }

    // ── Conversation focus tests ──

    #[test]
    fn up_down_on_conversation_focus() {
        let mut app = TuiApp::new();
        app.active_tab = TabId::Threads;
        app.threads_focus = ThreadsFocus::Conversation;
        app.conversation_scroll = 10;

        handle_key(&mut app, KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(app.conversation_scroll, 7); // 3 lines per step

        handle_key(&mut app, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(app.conversation_scroll, 10);
    }

    #[test]
    fn home_end_on_conversation_focus() {
        let mut app = TuiApp::new();
        app.active_tab = TabId::Threads;
        app.threads_focus = ThreadsFocus::Conversation;
        app.conversation_scroll = 50;
        app.conversation_auto_scroll = true;

        handle_key(&mut app, KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
        assert_eq!(app.conversation_scroll, 0);
        assert!(!app.conversation_auto_scroll);

        handle_key(&mut app, KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
        assert!(app.conversation_auto_scroll);
    }

    // ── Provider wizard tests ──

    #[test]
    fn provider_wizard_enter_sets_pending() {
        let mut app = TuiApp::new();
        app.input_mode = InputMode::ProviderWizard {
            provider: "anthropic".into(),
        };
        type_text(&mut app, "sk-ant-test-key");

        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(app.input_mode, InputMode::Normal);
        assert!(app.pending_provider_completion.is_some());
        let pc = app.pending_provider_completion.unwrap();
        assert_eq!(pc.provider, "anthropic");
        assert_eq!(pc.api_key, "sk-ant-test-key");
    }

    #[test]
    fn provider_wizard_esc_cancels() {
        let mut app = TuiApp::new();
        app.input_mode = InputMode::ProviderWizard {
            provider: "anthropic".into(),
        };

        handle_key(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));

        assert_eq!(app.input_mode, InputMode::Normal);
        assert!(app.chat_log.iter().any(|e| e.text.contains("cancelled")));
    }

    #[test]
    fn provider_wizard_empty_key_doesnt_submit() {
        let mut app = TuiApp::new();
        app.input_mode = InputMode::ProviderWizard {
            provider: "anthropic".into(),
        };
        // Enter with empty input — should NOT complete
        handle_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert!(app.in_wizard()); // still in wizard
        assert!(app.pending_provider_completion.is_none());
    }

    // ── Debug mode menu tests ──

    #[test]
    fn debug_mode_shows_debug_menu() {
        use super::super::app::build_menu_items;

        let items = build_menu_items(&[], true);
        // Debug mode: File, Run, Inspect, Debug, Help = 5 groups
        assert_eq!(items.len(), 5);
    }

    #[test]
    fn non_debug_omits_debug_menu() {
        use super::super::app::build_menu_items;

        let items = build_menu_items(&[], false);
        // Non-debug: File, Run, Inspect, Help = 4 groups
        assert_eq!(items.len(), 4);
    }
}
