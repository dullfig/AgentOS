//! Key binding dispatch for the TUI.
//!
//! Ctrl+C quits. Ctrl+1/2/3/4 switches tabs. Enter submits.
//! Esc clears textarea. Up/Down scroll messages. Tab completes
//! slash commands (or cycles sub-pane focus on Threads tab).
//! Everything else is forwarded to the textarea widget.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tui_menu::MenuEvent;

use super::app::{ActiveTab, AgentStatus, ChatEntry, MenuAction, ThreadsFocus, TuiApp};
use super::commands;

/// Dispatch a selected menu action.
fn dispatch_menu_action(app: &mut TuiApp, action: MenuAction) {
    app.menu_state.reset();
    app.menu_active = false;
    match action {
        MenuAction::SwitchTab(tab) => {
            if tab == ActiveTab::Debug && !app.debug_mode {
                return; // guard: don't switch to Debug unless enabled
            }
            app.active_tab = tab;
        }
        MenuAction::NewTask => {
            // Focus the input textarea (clear any existing text)
            app.input_textarea.select_all();
            app.input_textarea.cut();
        }
        MenuAction::Quit => {
            app.should_quit = true;
        }
        MenuAction::SetModel => {
            // Pre-fill the input with "/model " so user can type the model name
            set_input(app, "/model ");
        }
        MenuAction::ShowAbout => {
            app.chat_log.push(ChatEntry {
                role: "system".into(),
                text: "BestCode — an AI coding agent built on rust-pipeline.\nNo compaction, ever."
                    .into(),
            });
            app.message_auto_scroll = true;
        }
        MenuAction::ShowShortcuts => {
            app.chat_log.push(ChatEntry {
                role: "system".into(),
                text: concat!(
                    "Keyboard shortcuts:\n",
                    "  F10         Open/close menu bar\n",
                    "  Ctrl+1..5   Switch tabs\n",
                    "  Tab         Cycle focus (Threads) / autocomplete (/commands)\n",
                    "  Enter       Submit task or confirm\n",
                    "  Esc         Clear input\n",
                    "  Up/Down     Scroll or navigate\n",
                    "  PageUp/Dn   Page scroll\n",
                    "  Home/End    Jump to top/bottom\n",
                    "  Ctrl+C      Quit",
                )
                .into(),
            });
            app.message_auto_scroll = true;
        }
    }
}

/// Get the current input text from the textarea (first line).
fn current_input(app: &TuiApp) -> String {
    app.input_textarea.lines().first().cloned().unwrap_or_default()
}

/// Replace the textarea content with new text and move cursor to end.
fn set_input(app: &mut TuiApp, text: &str) {
    app.input_textarea.select_all();
    app.input_textarea.cut();
    for c in text.chars() {
        app.input_textarea.input(crossterm::event::Event::Key(
            KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE),
        ));
    }
}

/// Handle a key event, mutating app state.
pub fn handle_key(app: &mut TuiApp, key: KeyEvent) {
    // Ctrl+C always quits
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        app.should_quit = true;
        return;
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

    // Ctrl+1/2/3/4/5 switch tabs
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('1') => {
                app.active_tab = ActiveTab::Messages;
                return;
            }
            KeyCode::Char('2') => {
                app.active_tab = ActiveTab::Threads;
                return;
            }
            KeyCode::Char('3') => {
                app.active_tab = ActiveTab::Yaml;
                return;
            }
            KeyCode::Char('4') => {
                app.active_tab = ActiveTab::Wasm;
                return;
            }
            KeyCode::Char('5') if app.debug_mode => {
                app.active_tab = ActiveTab::Debug;
                return;
            }
            _ => {}
        }
    }

    match key.code {
        // Submit task or slash command
        KeyCode::Enter => {
            // On Threads tab with ContextTree focus, toggle the selected node
            if app.active_tab == ActiveTab::Threads
                && app.threads_focus == ThreadsFocus::ContextTree
            {
                app.context_tree_state.toggle_selected();
                return;
            }
            if let Some(text) = app.take_input() {
                if text.starts_with('/') {
                    // Slash command — defer to async executor in runner
                    app.pending_command = Some(text);
                } else {
                    app.chat_log.push(ChatEntry {
                        role: "user".into(),
                        text: text.clone(),
                    });
                    app.agent_status = AgentStatus::Thinking;
                    app.message_auto_scroll = true;
                    app.pending_task = Some(text);
                }
            }
        }
        // Tab: on Threads tab cycles sub-pane focus; otherwise autocomplete slash commands
        KeyCode::Tab => {
            if app.active_tab == ActiveTab::Threads {
                app.threads_focus = match app.threads_focus {
                    ThreadsFocus::ThreadList => ThreadsFocus::ContextTree,
                    ThreadsFocus::ContextTree => ThreadsFocus::ThreadList,
                };
            } else {
                let input = current_input(app);
                if input.starts_with('/') && !input.contains(' ') {
                    if let Some(cmd) = commands::suggest(&input) {
                        let mut completion = cmd.name.to_string();
                        if cmd.has_arg {
                            completion.push(' ');
                        }
                        set_input(app, &completion);
                    }
                } else {
                    // Normal Tab → forward to textarea
                    app.input_textarea
                        .input(crossterm::event::Event::Key(key));
                }
            }
        }
        // Clear input
        KeyCode::Esc => {
            app.input_textarea.select_all();
            app.input_textarea.cut();
        }
        // Arrow keys dispatched based on active tab + focus
        KeyCode::Up if app.active_tab == ActiveTab::Messages => {
            for _ in 0..3 {
                app.scroll_messages_up();
            }
        }
        KeyCode::Down if app.active_tab == ActiveTab::Messages => {
            for _ in 0..3 {
                app.scroll_messages_down();
            }
        }
        KeyCode::Up if app.active_tab == ActiveTab::Threads => match app.threads_focus {
            ThreadsFocus::ThreadList => {
                app.move_up();
            }
            ThreadsFocus::ContextTree => {
                app.context_tree_state.key_up();
            }
        },
        KeyCode::Down if app.active_tab == ActiveTab::Threads => match app.threads_focus {
            ThreadsFocus::ThreadList => {
                app.move_down();
            }
            ThreadsFocus::ContextTree => {
                app.context_tree_state.key_down();
            }
        },
        // Debug tab: arrow keys scroll activity trace
        KeyCode::Up if app.active_tab == ActiveTab::Debug => {
            for _ in 0..3 {
                app.scroll_activity_up();
            }
        }
        KeyCode::Down if app.active_tab == ActiveTab::Debug => {
            for _ in 0..3 {
                app.scroll_activity_down();
            }
        }
        // Left/Right on ContextTree focus: collapse/expand
        KeyCode::Left
            if app.active_tab == ActiveTab::Threads
                && app.threads_focus == ThreadsFocus::ContextTree =>
        {
            app.context_tree_state.key_left();
        }
        KeyCode::Right
            if app.active_tab == ActiveTab::Threads
                && app.threads_focus == ThreadsFocus::ContextTree =>
        {
            app.context_tree_state.key_right();
        }
        // Page scroll — full page minus 2 overlap lines, dispatched to active tab
        KeyCode::PageUp => match app.active_tab {
            ActiveTab::Debug => {
                let page = app.activity_viewport_height.saturating_sub(2).max(1);
                for _ in 0..page {
                    app.scroll_activity_up();
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
            ActiveTab::Debug => {
                let page = app.activity_viewport_height.saturating_sub(2).max(1);
                for _ in 0..page {
                    app.scroll_activity_down();
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
            ActiveTab::Threads => match app.threads_focus {
                ThreadsFocus::ContextTree => {
                    app.context_tree_state.select_first();
                }
                ThreadsFocus::ThreadList => {
                    app.selected_thread = 0;
                }
            },
            ActiveTab::Debug => {
                app.activity_scroll = 0;
                app.activity_auto_scroll = false;
            }
            _ => {
                app.message_scroll = 0;
                app.message_auto_scroll = false;
            }
        },
        KeyCode::End => match app.active_tab {
            ActiveTab::Threads => match app.threads_focus {
                ThreadsFocus::ContextTree => {
                    app.context_tree_state.select_last();
                }
                ThreadsFocus::ThreadList => {
                    if !app.threads.is_empty() {
                        app.selected_thread = app.threads.len() - 1;
                    }
                }
            },
            ActiveTab::Debug => {
                app.activity_auto_scroll = true;
            }
            _ => {
                app.message_auto_scroll = true;
            }
        },
        // Everything else → textarea
        _ => {
            app.input_textarea
                .input(crossterm::event::Event::Key(key));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn type_text(app: &mut TuiApp, text: &str) {
        for c in text.chars() {
            app.input_textarea.input(crossterm::event::Event::Key(
                KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE),
            ));
        }
    }

    #[test]
    fn tab_completes_slash_command() {
        let mut app = TuiApp::new();
        type_text(&mut app, "/mo");

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
        );

        assert_eq!(app.input_textarea.lines(), ["/model "]);
    }

    #[test]
    fn tab_no_slash_forwards_to_textarea() {
        let mut app = TuiApp::new();
        type_text(&mut app, "hello");

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
        );

        // Tab should be forwarded to textarea (inserts tab or similar)
        // Main assertion: no crash, input still contains "hello"
        let text = app.input_textarea.lines().join("");
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
        app.active_tab = ActiveTab::Threads;
        assert_eq!(app.threads_focus, ThreadsFocus::ThreadList);

        handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.threads_focus, ThreadsFocus::ContextTree);

        handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.threads_focus, ThreadsFocus::ThreadList);
    }

    #[test]
    fn tab_on_messages_tab_goes_to_textarea() {
        let mut app = TuiApp::new();
        app.active_tab = ActiveTab::Messages;
        type_text(&mut app, "hello");

        handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));

        // Tab forwarded to textarea (not focus cycling)
        let text = app.input_textarea.lines().join("");
        assert!(text.contains("hello"));
    }

    #[test]
    fn up_down_on_thread_list() {
        let mut app = TuiApp::new();
        app.active_tab = ActiveTab::Threads;
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
        app.active_tab = ActiveTab::Threads;
        app.threads_focus = ThreadsFocus::ContextTree;

        // key_up/key_down on TreeState with no rendered items is a no-op — just verify no panic
        handle_key(&mut app, KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        handle_key(&mut app, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        // No crash = pass
    }

    #[test]
    fn up_down_on_debug_tab() {
        let mut app = TuiApp::new();
        app.debug_mode = true;
        app.active_tab = ActiveTab::Debug;
        app.activity_scroll = 10;

        handle_key(&mut app, KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(app.activity_scroll, 7); // 3 lines per step

        handle_key(&mut app, KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(app.activity_scroll, 10);
    }

    #[test]
    fn ctrl_5_switches_to_debug_when_enabled() {
        let mut app = TuiApp::new();
        app.debug_mode = true;

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('5'), KeyModifiers::CONTROL),
        );
        assert_eq!(app.active_tab, ActiveTab::Debug);
    }

    #[test]
    fn ctrl_5_ignored_when_not_debug() {
        let mut app = TuiApp::new();
        assert!(!app.debug_mode);
        app.active_tab = ActiveTab::Messages;

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('5'), KeyModifiers::CONTROL),
        );
        // Should remain on Messages — Ctrl+5 is a no-op without debug_mode
        assert_eq!(app.active_tab, ActiveTab::Messages);
    }

    #[test]
    fn home_end_on_debug_tab() {
        let mut app = TuiApp::new();
        app.debug_mode = true;
        app.active_tab = ActiveTab::Debug;
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
        app.active_tab = ActiveTab::Threads;
        app.threads_focus = ThreadsFocus::ContextTree;

        // No rendered items yet, just verify no panic
        handle_key(&mut app, KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        handle_key(&mut app, KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
    }

    #[test]
    fn enter_on_context_tree() {
        let mut app = TuiApp::new();
        app.active_tab = ActiveTab::Threads;
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
        dispatch_menu_action(&mut app, MenuAction::SwitchTab(ActiveTab::Threads));
        assert_eq!(app.active_tab, ActiveTab::Threads);
        assert!(!app.menu_active);
    }

    #[test]
    fn menu_state_default() {
        let app = TuiApp::new();
        assert!(!app.menu_active);
        // Menu state exists but is not user-activated
        assert_eq!(app.active_tab, ActiveTab::Messages);
    }
}
