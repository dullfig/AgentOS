# BestCode

An operating system for AI coding agents — not a framework, not a library, not a tool.

BestCode is the runtime kernel of [AgentOS](https://github.com/dullfig): a complete,
secure, validated infrastructure where AI agents read, write, search, and execute
with the same guarantees an OS provides to processes. Every message is untrusted.
Every capability is structural. Every state change is durable.

Built on [rust-pipeline](https://github.com/dullfig/rust-pipeline), a zero-trust
async message pipeline where handler responses re-enter as untrusted bytes —
because the most dangerous input is the output you just produced.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         Control Room (TUI)       │
                    │   ratatui · TEA · XML renderers  │
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

Three pieces of nuclear-proof state compose the kernel:

- **Thread Table** — the call stack. Threads can recurse arbitrarily (`root.a.b.c.c.c...`), making the pipeline Turing-complete.
- **Context Store** — virtual memory for attention. Three-tier hierarchy: expanded (active working set), folded (compressed summaries), evicted (on disk). The librarian is kswapd, not the OOM killer.
- **Message Journal** — audit trail and tape. Configurable retention: `retain_forever` for coding agents, `prune_on_delivery` for stateless services, `retain_days` for compliance.

## Security

Security is structural, not behavioral. You cannot prompt-inject your way past
a dispatch table that simply does not contain the route you are trying to reach.

```yaml
# organism.yaml — security profiles
profiles:
  coding:                          # full access
    listeners: [file-read, file-write, file-edit, glob, grep, command-exec]
  researcher:                      # read-only, structurally enforced
    listeners: [file-read, glob, grep]
```

Three concentric walls:
1. **Dispatch table** — missing route = structural impossibility
2. **Linux user isolation** — each organism runs as its own user
3. **Kernel process isolation** — one WAL, one process, atomic operations

The command-exec tool enforces an allowlist at the token level — the first word
of every command is checked before execution. No shell injection, no escape.

## The Coding Agent

A stateful agentic loop built on Anthropic's tool-use protocol. One thinker
(Opus), everything else executes. Tools don't think.

The agent maintains per-thread state machines: task arrives, model reasons,
tool calls dispatch through the pipeline as first-class messages, results
return as untrusted bytes, model continues. The full OODA loop with structural
security at every transition.

**Semantic routing** discovers tools by embedding similarity — the agent
describes what it needs, the router finds the capability. No hardcoded
dispatch tables for user-defined tools.

**WASM sandboxing** lets users bring their own tools as WebAssembly components.
Full capability-based security: a WASM tool can only access what its WIT
interface declares.

## The Librarian

A Haiku-class model that curates context the way kswapd manages pages —
proactively, by relevance, before pressure forces eviction. Informed by
[research on context rot](https://research.trychroma.com/context-rot):
every irrelevant token degrades the thinker. Focused 300 tokens beats
full 113K history.

The architecture mirrors Conway's
[Self-Memory System](https://www.sciencedirect.com/science/article/pii/S0749596X05000987)
(2005) — convergent design from first principles. The context store is the
autobiographical knowledge base; the librarian is the conceptual self;
fold summaries are constructed memories, not retrieved ones. Scratch contexts
are working memory — you remember the answer, not the carry digits.

Forgetting is the primary feature of functional memory.

## Modules

| Module | Purpose |
|--------|---------|
| `kernel` | WAL, thread table, context store, message journal |
| `agent` | Coding agent handler, tool-use state machine, JSON/XML translation |
| `librarian` | Haiku-powered context curation, tree-sitter symbol extraction |
| `llm` | Anthropic API client, connection pooling, message types |
| `tools` | Six native tool peers: read, write, edit, glob, grep, exec |
| `wasm` | WASM+WIT component runtime, capability-based sandboxing |
| `routing` | Semantic router, TF-IDF embeddings, form filler |
| `embedding` | Embedding provider trait, TF-IDF implementation |
| `organism` | YAML-driven organism configuration, security profiles |
| `pipeline` | Builder pattern, event bus, pipeline integration |
| `tui` | ratatui Control Room: dashboard, context tree, XML renderers |
| `ports` | Port manager, firewall |
| `security` | Security profile enforcement |

## Numbers

- **~19,000** lines of Rust
- **461** tests (unit + integration)
- **6** native tool peers
- **13** modules
- **0** unsafe blocks

## Building

```bash
# Prerequisites: Rust toolchain, Anthropic API key for live tests
cargo build
cargo test --lib          # 461 tests, no API key needed
cargo test                # includes live integration tests (needs ANTHROPIC_API_KEY)
cargo clippy              # clean
```

## Target Hardware

A Raspberry Pi 5 running the kernel locally, with cloud LLM inference.
The expensive part is the thinking, not the infrastructure.

## License

BUSL-1.1
