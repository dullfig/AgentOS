# AgentOS

An operating system for AI agents — not a framework, not a library, not a tool.

AgentOS is complete, secure, validated infrastructure where AI agents read, write,
search, and execute with the same guarantees an OS provides to processes. Every
message is untrusted. Every capability is structural. Every state change is durable.

Built on [rust-pipeline](https://github.com/dullfig/rust-pipeline), a zero-trust
async message pipeline where handler responses re-enter as untrusted bytes —
because the most dangerous input is the output you just produced.

**761 tests. ~40,000 lines of Rust. Zero unsafe blocks. No compaction, ever.**

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
    │  │   Bob     │  │Librarian │  │ Semantic │  │   LLM Pool    │  │
    │  │ (concierge)│ │ (Haiku)  │  │  Router  │  │  (Anthropic)  │  │
    │  └────┬─────┘  └──────────┘  └──────────┘  └───────────────┘  │
    │       │ delegates                                               │
    │  ┌────┴──────────────────────────────────────────────────────┐  │
    │  │               Specialist Buffers                          │  │
    │  │  coder · plan-expert · wiki-expert · agent-expert         │  │
    │  └────┬─────────────────────────────────────────────────────┘  │
    │       │ dispatch                                                │
    │  ┌────┴──────────────────────────────────────────────────────┐  │
    │  │                     Tool Peers                            │  │
    │  │  file-read · file-write · file-edit · glob · grep         │  │
    │  │  bash · codebase-index · WASM user tools                  │  │
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

**One thinker, specialist executors.** Bob is the concierge — he routes tasks to
the right specialist. Each specialist runs in an isolated child pipeline (buffer)
with only the tools it needs. Tools don't think. The pipeline routes everything
under zero-trust security.

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
| `/vdrive mount <path>` | Mount a workspace (sandbox all file tools to this directory) |
| `/clear` | Clear chat |
| `/help` | Show all commands |
| `/exit` | Quit |

### Keyboard

| Key | Action |
|-----|--------|
| Enter | Submit task to agent |
| F10 | Toggle menu bar |
| Ctrl+1-5 | Switch tabs (Messages, Threads, YAML, Debug, Code) |
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
2. **VDrive sandbox** — all file tools are sandboxed to the mounted workspace. No path escapes.
3. **Kernel process isolation** — one WAL, one process, atomic operations

```yaml
# organism.yaml — security profiles
profiles:
  admin:                             # full access
    listeners: [file-read, file-write, file-edit, glob, grep, bash]
  restricted:                        # read-only, structurally enforced
    listeners: [file-read, glob, grep]
```

The bash tool enforces a command allowlist — only `cargo`, `git`, `npm`, `python`,
and other safe commands pass. WASM user tools run in capability-based sandboxes —
they can only access what their WIT interface declares.

**Injection guard middleware** scans all tool outputs for prompt injection attempts
(regex patterns + semantic classification). Flagged content is quarantined before
reaching the agent.

## Ships With Specialists

AgentOS ships with a working team of AI agents — not toy demos, but real tools
that exercise every layer of the stack. Point it at a codebase and start working.

**Bob** is the concierge. He reads your task and routes it to the right specialist:

| Specialist | What it does |
|------------|-------------|
| **coder** | Writes code, edits files, runs tests, uses git. Spawns in an isolated child pipeline with write tools. `fork()+exec()` for agents. |
| **plan-expert** | Surveys the codebase, creates structured execution plans (plan.md with impact tables and dependency graphs), then delegates each step to the coder. |
| **wiki-expert** | Generates project documentation as a wiki/ folder of interlinked markdown files. |
| **agent-expert** | Designs, validates, and diagnoses organism YAML configurations. |

```
User task → Bob (concierge)
                ↓ routes to specialist
            [buffer] ← isolation boundary
                ↓
            plan-expert (surveys, plans)
                ↓ delegates steps
            [buffer] ← nested isolation
                ↓
            coder (child pipeline, write tools)
                ↓
            Result flows back through chain to user
```

Agent identity is data, not code. Prompts, model selection, token limits, and
iteration caps are all declared in organism YAML. New agent types require a
YAML file and a prompt — zero Rust.

```yaml
# organisms/coder-v2.yaml
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
    peers: [file-read, file-write, file-edit, glob, grep, bash]

  - name: coder
    handler: buffer
    buffer:                                    # buffer IS the tool
      description: "Execute a coding task"
      parameters:
        task: { type: string, description: "The coding task to perform" }
      required: [task]
      organism: organisms/coder-v2.yaml        # child pipeline
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
- **Code tab** — Python tool editor with tree-sitter syntax highlighting

The YAML editor provides the same intelligence as a real LSP — schema-aware
completions, cross-reference validation, hover documentation — but runs as
pure functions on the editor buffer. No JSON-RPC, no server process.

Command palette (type `/`) shows filtered commands with ghost-text autocomplete.
Menu bar (F10) with dropdown navigation and Alt+letter accelerators.

## Modules

| Module | Purpose |
|--------|---------|
| `crates/kernel/` | WAL, thread table, context store, message journal — durable state |
| `crates/events/` | PipelineEvent, ConversationEntry — shared event types (zero deps) |
| `crates/vdrive/` | VDrive sandbox: path-validated workspace isolation, 47 tests |
| `agent/` | Agentic loop, tool-use state machine, middleware (LoopGuard, PermissionGate, DebugGate, InjectionGuard) |
| `pipeline/` | Builder pattern, event bus, organism-to-pipeline wiring |
| `organism/` | YAML config: listeners, profiles, prompts, agent config, buffer config |
| `buffer/` | Buffer nodes: fork()+exec() for callable organisms, ephemeral child pipelines |
| `security/` | Dispatch table enforcement, profile resolution |
| `llm/` | Anthropic API client, LlmPool, model aliasing, list models API |
| `config/` | Multi-provider model config (`~/.agentos/models.yaml`) |
| `tools/` | Native tool peers: file-read, file-write, file-edit, glob, grep, bash, list-agents, user channel |
| `wasm/` | WASM+WIT component runtime, Python→WASM pipeline, capability-based sandboxing |
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
cargo test --lib           # 761 tests, no API key needed
cargo test                 # full suite including live API integration tests
```

## Target Hardware

Raspberry Pi 5 running the kernel locally, with cloud LLM inference.
The expensive part is the thinking, not the infrastructure.

## License

BUSL-1.1
