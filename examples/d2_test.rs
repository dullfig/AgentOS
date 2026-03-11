//! Standalone D2 diagram test harness.
//!
//! Usage:
//!   cargo run --example d2_test                       # built-in gallery
//!   cargo run --example d2_test -- file1.d2 file2.d2  # custom files appended
//!
//! Controls: Tab / Shift+Tab to cycle, q / Esc to quit.

use std::io;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

/// A named D2 example.
struct Example {
    name: String,
    source: String,
}

fn built_in_examples() -> Vec<Example> {
    vec![
        Example {
            name: "Simple edge".into(),
            source: "a -> b -> c".into(),
        },
        Example {
            name: "Edge labels".into(),
            source: "client -> server: request\nserver -> client: response".into(),
        },
        Example {
            name: "Diamond layout".into(),
            source: "a -> b\na -> c\nb -> d\nc -> d".into(),
        },
        Example {
            name: "Bidirectional".into(),
            source: "frontend <-> backend: REST\nbackend <-> database: SQL".into(),
        },
        Example {
            name: "Cycle".into(),
            source: "a -> b\nb -> c\nc -> a".into(),
        },
        Example {
            name: "Shapes: diamond".into(),
            source: "router: { shape: diamond }\nrequest -> router\nrouter -> handler_a\nrouter -> handler_b".into(),
        },
        Example {
            name: "Shapes: cylinder".into(),
            source: "db: { shape: cylinder }\napp -> db: query\ndb -> app: rows".into(),
        },
        Example {
            name: "Shapes: hexagon".into(),
            source: "queue: { shape: hexagon }\nproducer -> queue\nqueue -> consumer".into(),
        },
        Example {
            name: "Container".into(),
            source: "services: { auth; api; cache }\ndata: { postgres; redis }".into(),
        },
        Example {
            name: "Mixed shapes".into(),
            source: concat!(
                "agent: { shape: diamond }\n",
                "wasm_tool: { shape: cylinder }\n",
                "buffer: { shape: hexagon }\n",
                "agent -> wasm_tool: invoke\n",
                "agent -> buffer: delegate\n",
                "buffer -> wasm_tool: run",
            ).into(),
        },
        Example {
            name: "Chain with labels".into(),
            source: "user -> gateway: HTTP\ngateway -> auth: validate\nauth -> db: lookup\ndb -> auth: result\nauth -> gateway: token\ngateway -> user: 200 OK".into(),
        },
        Example {
            name: "Sequence: login flow".into(),
            source: concat!(
                "d2-config {\n",
                "  type: sequence\n",
                "}\n",
                "User: User\n",
                "Server: Auth Server\n",
                "DB: Database\n",
                "\n",
                "User -> Server: POST /login\n",
                "Server -> DB: SELECT user\n",
                "DB -> Server: user record\n",
                "Server -> Server: validate password\n",
                "Server -> User: 200 OK + token\n",
            ).into(),
        },
        Example {
            name: "Sequence: simple".into(),
            source: concat!(
                "d2-config {\n",
                "  type: sequence\n",
                "}\n",
                "Alice -> Bob: hello\n",
                "Bob -> Alice: hi back\n",
            ).into(),
        },
        Example {
            name: "Sequence: self-message".into(),
            source: concat!(
                "d2-config {\n",
                "  type: sequence\n",
                "}\n",
                "Client -> Server: request\n",
                "Server -> Server: process\n",
                "Server -> Client: response\n",
            ).into(),
        },
        Example {
            name: "Sequence: many actors".into(),
            source: concat!(
                "d2-config {\n",
                "  type: sequence\n",
                "}\n",
                "Browser: Browser\n",
                "CDN: CDN\n",
                "API: API Gateway\n",
                "Auth: Auth Service\n",
                "DB: Database\n",
                "\n",
                "Browser -> CDN: GET /app.js\n",
                "CDN -> Browser: 200 cached\n",
                "Browser -> API: POST /login\n",
                "API -> Auth: validate\n",
                "Auth -> DB: query\n",
                "DB -> Auth: user\n",
                "Auth -> API: token\n",
                "API -> Browser: 200 + JWT\n",
            ).into(),
        },
        Example {
            name: "Wide graph".into(),
            source: "a -> b\na -> c\na -> d\na -> e\nb -> f\nc -> f\nd -> f\ne -> f".into(),
        },
        Example {
            name: "Single node".into(),
            source: "lonely".into(),
        },
        Example {
            name: "Emoji labels".into(),
            source: "rocket: 🚀 Launch\nplanet: 🌍 Earth\nrocket -> planet: travel".into(),
        },
    ]
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Build example list: built-ins + any CLI files
    let mut examples = built_in_examples();
    for path in &args[1..] {
        let source = std::fs::read_to_string(path)?;
        examples.push(Example {
            name: path.clone(),
            source,
        });
    }

    let mut idx: usize = 0;

    // Setup terminal
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    loop {
        let ex = &examples[idx];
        terminal.draw(|frame| {
            let area = frame.area();
            let max_width = (area.width as usize).saturating_sub(2);

            let lines = agentos::tui::diagram::render_d2(&ex.source, max_width);

            let title = format!(
                " {} ({}/{}) ── Tab/Shift+Tab to cycle, q to quit ",
                ex.name,
                idx + 1,
                examples.len(),
            );
            let paragraph = Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title(title));

            frame.render_widget(paragraph, area);
        })?;

        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Tab => {
                    if key.modifiers.contains(KeyModifiers::SHIFT) {
                        idx = if idx == 0 { examples.len() - 1 } else { idx - 1 };
                    } else {
                        idx = (idx + 1) % examples.len();
                    }
                }
                KeyCode::BackTab => {
                    idx = if idx == 0 { examples.len() - 1 } else { idx - 1 };
                }
                KeyCode::Right | KeyCode::Down => {
                    idx = (idx + 1) % examples.len();
                }
                KeyCode::Left | KeyCode::Up => {
                    idx = if idx == 0 { examples.len() - 1 } else { idx - 1 };
                }
                _ => {}
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}
