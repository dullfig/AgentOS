# AgentOS

An operating system for AI coding agents — not a framework, not a library, not a tool.

AgentOS is complete, secure, validated infrastructure where AI agents read, write,
search, and execute with the same guarantees an OS provides to processes. Every
message is untrusted. Every capability is structural. Every state change is durable.

Built on [rust-pipeline](https://github.com/dullfig/rust-pipeline), a zero-trust
async message pipeline where handler responses re-enter as untrusted bytes —
because the most dangerous input is the output you just produced.

**819 tests. ~27,000 lines of Rust. Zero unsafe blocks. No compaction, ever.**

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         Control Room (TUI)       │
                    │  Messages │ Threads │ YAML │ Debug│
                    └────────────────┬────────────────┘
                                     │ event bus
    ┌────────────────────────────────┼────────────────────────────────┐
    │                           Pipeline                              │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
    │  │  Coding   │  │Librarian │  │ Semantic │  │   LLM Pool    │  │
    │  │  Agent    │  │ (Haiku)  │  │  Router  │  │  (Anthropic)  │  │
    │  └────┬─────┘  └──────────┘  └──────────┘  └───────────────┘  │
    │       │ dispatch                                                │
    │  ┌────┴──────────────────────────────────────────────────────┐  │
    │  │                     Tool Peers                            │  │
    │  │  file-read · file-write · file-edit · glob · grep         │  │
    │  │  command-exec · codebase-index · WASM user tools          │  │
    │  └───────────────────────────────────────────────────────────┘  │
    └────────────────────────────────┼────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │            Kernel                │
                    │  WAL · Thread Table · Context    │
                    │  Store · Message Journal         │
                    │  ─────────────────────────       │
                    │  One process. One WAL. Atomic.   │
                    └─────────────────────────────────┘
```

**One thinker (Opus), everything else executes.** Tools don't think. The agent calls tools through the pipeline, tools reply with results, the pipeline routes everything under zero-trust security.

Three pieces of nuclear-proof state compose the kernel:

- **Thread Table** — the call stack. Threads recurse arbitrarily (`root.a.b.c.c.c...`), making the pipeline Turing-complete.
- **Context Store** — virtual memory for attention. Three tiers: expanded (active), folded (summarized), evicted (on disk). The librarian is kswapd, not the OOM killer.
- **Message Journal** — audit trail and tape. Configurable retention: `retain_forever` (coding), `prune_on_delivery` (stateless), `retain_days` (compliance).

## Quick Start

```bash
# Build
cargo build --release

# Run with API key in environment
export ANTHROPIC_API_KEY=sk-ant-...
cargo run

# Or start without a key — configure interactively via TUI
cargo run
# Then: /models add anthropic → wizard prompts for alias, model ID, API key
```

### CLI

```
agentos [OPTIONS]

  -d, --dir <DIR>            Working directory (default: current)
  -m, --model <MODEL>        Model alias (default: sonnet)
  -o, --organism <ORGANISM>  Path to organism.yaml (default: embedded)
      --data <DATA>          Kernel data directory (default: .agentos/)
      --debug                Enable debug tab (activity trace)
```

### TUI Commands

| Command | Description |
|---------|-------------|
| `/model <alias>` | Switch model (opus, sonnet, sonnet-4.5, haiku) |
| `/models` | List models available from API |
| `/models add <provider>` | Interactive wizard to add a provider |
| `/models update <provider>` | Update API key for a provider |
| `/models remove <alias>` | Remove a model |
| `/models default <alias>` | Set the default model |
| `/clear` | Clear chat |
| `/help` | Show all commands |
| `/exit` | Quit |

### Keyboard

| Key | Action |
|-----|--------|
| Enter | Submit task to agent |
| F10 | Toggle menu bar |
| Ctrl+1-5 | Switch tabs (Messages, Threads, YAML, Debug) |
| Alt+letter | Menu accelerators |
| Ctrl+S | Validate YAML (YAML tab) |
| Ctrl+Space | Trigger completions (YAML editor) |
| Ctrl+H | Hover info (YAML editor) |
| Ctrl+C | Quit |

## Security

Security is structural, not behavioral. You cannot prompt-inject your way past
a dispatch table that does not contain the route you are trying to reach.

Three concentric walls:
1. **Dispatch table** — missing route = structural impossibility
2. **Linux user isolation** — each organism runs as its own user
3. **Kernel process isolation** — one WAL, one process, atomic operations

```yaml
# organism.yaml — security profiles
profiles:
  coding:                          # full access
    listeners: [file-read, file-write, file-edit, glob, grep, command-exec]
  researcher:                      # read-only, structurally enforced
    listeners: [file-read, glob, grep]
```

The command-exec tool enforces an allowlist at the token level — the first word
of every command is checked before execution. WASM user tools run in
capability-based sandboxes — they can only access what their WIT interface declares.

## Ships With a Coding Agent

AgentOS ships with a working coding agent — not a toy demo, but a real tool that
exercises every layer of the stack: pipeline, kernel, security profiles, tool
peers, buffer nodes, prompt resolution, and the TUI. Point it at a codebase and
start working.

The default organism (`organisms/default.yaml`) defines three agents:

- **Planner** — read-only access (file-read, glob, grep, codebase-index). Explores
  the codebase, designs an implementation plan, then calls the coder.
- **Coder** — a buffer node that spawns an isolated child pipeline with write tools.
  Executes the plan in its own kernel, its own context store, its own thread table.
  `fork()+exec()` for agents.
- **Coding-agent** — the all-in-one option. Full read/write access in a single agent.

```
User task → Planner (read-only tools)
                ↓ calls coder as a tool
            [buffer] ← isolation boundary
                ↓
            Coder (child pipeline, write tools)
                ↓
            Result flows back through planner to user
```

Agent identity is data, not code. Prompts, model selection, token limits, and
iteration caps are all declared in organism YAML. One agent per file. New agent
types require a YAML file and a prompt — zero Rust.

```yaml
# organisms/default.yaml
prompts:
  coding_base: |
    You are a coding agent running inside AgentOS...
    {tool_definitions}
  no_paperclipper: |
    You are bounded. You do not pursue goals beyond your task.

listeners:
  - name: coding-agent
    handler: agent.handle
    agent:
      prompt: "no_paperclipper & coding_base"  # composition with &
      max_tokens: 4096
    peers: [file-read, file-write, file-edit, glob, grep, command-exec]

  - name: coder
    handler: buffer
    buffer:                                    # buffer IS the tool
      description: "Execute an implementation plan"
      parameters:
        plan: { type: string, description: "The plan to execute" }
      required: [plan]
      organism: organisms/coder.yaml           # child pipeline
      max_concurrency: 1
```

**Semantic routing** discovers tools by embedding similarity — the agent
describes what it needs, the router finds the capability. No hardcoded dispatch
for user-defined tools.

## The Librarian

A Haiku-class model that curates context the way kswapd manages pages —
proactively, by relevance, before pressure forces eviction. Informed by
[research on context rot](https://research.trychroma.com/context-rot):
every irrelevant token degrades the thinker. Focused 300 tokens beats
full 113K history.

The architecture mirrors Conway's
[Self-Memory System](https://www.sciencedirect.com/science/article/pii/S0749596X05000987)
(2005). The context store is the autobiographical knowledge base; the librarian
is the conceptual self; fold summaries are constructed memories, not retrieved ones.
Scratch contexts are working memory — you remember the answer, not the carry digits.

Forgetting is the primary feature of functional memory.

## Model Management

Multi-provider support (Anthropic, OpenAI, Ollama) with persistent config:

```yaml
# ~/.agentos/models.yaml
providers:
  anthropic:
    api_key: sk-ant-...
    models:
      opus: claude-opus-4-6
      sonnet: claude-sonnet-4-6
      haiku: claude-haiku-4-5-20251001
default: sonnet
```

`/models` queries the API to show which models your key supports. `/models add`
walks through an interactive wizard. `/model <alias>` hot-swaps the active model
and rebuilds the HTTP client if the provider changes (e.g., switching from
Anthropic to OpenAI).

Starts without an API key — configure via the TUI, no restart needed.

## The Control Room

A ratatui terminal UI following TEA (The Elm Architecture):

- **Messages tab** — conversation with the agent, markdown rendering, D2 diagram art
- **Threads tab** — three-pane split: thread list, conversation timeline, context tree
- **YAML tab** — tree-sitter syntax-highlighted editor for the organism config, with diagnostics, completions, and hover from an in-process language service
- **Debug tab** — live activity trace with timestamps (enabled with `--debug`)

The YAML editor provides the same intelligence as a real LSP — schema-aware
completions, cross-reference validation, hover documentation — but runs as
pure functions on the editor buffer. No JSON-RPC, no server process.

Command palette (type `/`) shows filtered commands with ghost-text autocomplete.
Menu bar (F10) with dropdown navigation and Alt+letter accelerators.

## Modules

| Module | Purpose |
|--------|---------|
| `kernel/` | WAL, thread table, context store, message journal — durable state |
| `agent/` | Coding agent: agentic loop, tool-use state machine, JSON/XML translation, prompts |
| `pipeline/` | Builder pattern, event bus, organism-to-pipeline wiring |
| `organism/` | YAML config: listeners, profiles, prompts, agent config, buffer config |
| `buffer/` | Buffer nodes: fork()+exec() for callable organisms, ephemeral child pipelines |
| `security/` | Dispatch table enforcement, profile resolution |
| `llm/` | Anthropic API client, LlmPool, model aliasing, list models API |
| `config/` | Multi-provider model config (`~/.agentos/models.yaml`) |
| `tools/` | Six native tool peers: file-read, file-write, file-edit, glob, grep, command-exec |
| `wasm/` | WASM+WIT component runtime, capability-based sandboxing |
| `librarian/` | Haiku-powered context curation, relevance-based paging |
| `routing/` | Semantic router: TF-IDF embeddings, form filler, invisible dispatch |
| `embedding/` | EmbeddingProvider trait, TF-IDF implementation |
| `treesitter/` | Code indexing, symbol extraction via tree-sitter |
| `lsp/` | In-process language intelligence for YAML editor and command line |
| `tui/` | ratatui Control Room: TEA model, multi-tab dashboard, D2 diagrams |
| `ports/` | Port manager, firewall, network protocol validation |

## Building

```bash
cargo build                # debug build
cargo test --lib           # 819 tests, no API key needed, ~7s
cargo test                 # full suite including live API integration tests
cargo clippy               # zero warnings from project code
```

## Target Hardware

Raspberry Pi 5 running the kernel locally, with cloud LLM inference.
The expensive part is the thinking, not the infrastructure.

## License

BUSL-1.1
