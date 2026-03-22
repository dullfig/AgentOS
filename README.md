# AgentOS

**What if agents had an operating system?**

Not a framework you import. Not a library you call. An actual OS — with
process isolation, a filesystem sandbox, durable state, and security that
works even when the agent is trying to break it.

AgentOS exists because of a simple question: *"Won't this take over my
computer?"* The answer was yes. So we built the thing that makes the
answer no.

## The Problem

Every agent framework trusts the agent. The system prompt says "don't do
bad things" and hopes for the best. But prompts aren't security — they're
suggestions. A prompt-injected agent with file access owns your machine.

AgentOS doesn't trust the agent. Security is **structural**, not behavioral.
An agent can't access a file operation that isn't in its dispatch table — not
because a prompt says so, but because the route doesn't exist. You can't
prompt-inject your way past a wall that has no door.

## What It Looks Like

An agent is a YAML file. No code required.

```yaml
organism:
  name: my-coder

imports:
  - infrastructure.yaml          # shared tools + LLM pool

prompts:
  safety: |
    You are bounded. You do not pursue goals beyond your task.
  coder_base: |
    You are a coding agent. Read before you write.
    Make the smallest change that solves the problem.
    {tool_definitions}

listeners:
  - name: coder
    handler: agent.handle
    description: "Coding agent"
    agent:
      prompt: "safety & coder_base"     # compose prompts with &
      model: sonnet                      # per-agent model selection
    tools: [file-read, file-write, file-edit, glob, grep, bash]

profiles:
  default:
    linux_user: agentos
    listeners: all
    journal: retain_forever
```

That's a complete agent. It has a prompt, a model, tools it can use, and
a security profile. Drop it in `organisms/` and it's available to the system.

Want a custom tool? Write Python:

```python
def run(query: str) -> str:
    """Search the codebase for a pattern."""
    import subprocess
    result = subprocess.run(["grep", "-r", query, "."], capture_output=True)
    return result.stdout.decode()
```

The platform compiles it to WASM, sandboxes it, and makes it available
as a tool — with only the capabilities you declare in the YAML. No filesystem
access unless granted. No network unless granted. The sandbox is structural.

## How It Works

```
You → Bob (concierge) → decides what to do
                           ↓
                    Specialist buffers — isolated child pipelines
                    ┌─────────┬──────────┬─────────────┐
                    │ coder   │ planner  │ agent-expert │
                    │ (writes │ (reads,  │ (designs new │
                    │  code)  │  plans)  │  agents)     │
                    └────┬────┴────┬─────┴──────┬──────┘
                         │         │             │
                    Tool peers — sandboxed operations
                    file-read · file-write · glob · grep
                    bash · codebase-index · WASM tools
                         │
                    Kernel — durable state
                    WAL · threads · context · journal
```

**Bob** is the concierge — he reads your task and delegates to the right
specialist. Each specialist runs in an **isolated child pipeline** with
only the tools it needs. It's `fork()+exec()` for agents.

New specialists are just YAML files. Bob discovers them automatically
(`tools: auto`). Install a new agent, Bob can use it — no wiring needed.

## Why an OS?

Because agents need the same things processes do:

| Process needs | Agent needs | AgentOS provides |
|---------------|-------------|------------------|
| Filesystem | Read/write files | **VDrive** — QCOW2 sandbox, no path escapes |
| Process isolation | Don't let agents interfere | **Buffers** — each specialist gets its own pipeline |
| Permissions | Don't let agents do everything | **Dispatch table** — structural, not behavioral |
| Virtual memory | Don't blow the context window | **Context store** — 3-tier VMM (active/folded/evicted) |
| Scheduler | Manage concurrent work | **Thread table** — recursive, arbitrary nesting |
| Audit log | Know what happened | **Journal** — configurable retention per profile |
| Device drivers | Talk to LLMs, tools, APIs | **Listeners** — uniform handler interface |

The **librarian** is kswapd for attention — a Haiku-class model that curates
context proactively, by relevance, before pressure forces eviction. Informed
by [research on context rot](https://research.trychroma.com/context-rot):
every irrelevant token degrades the thinker. Focused 300 tokens beats full
128K history. Forgetting is the primary feature of functional memory.

## Security Model

Three concentric walls, all structural:

1. **Dispatch table** — an agent can only call listeners in its `tools:` list.
   Missing route = impossible, not forbidden. No prompt can add a route.

2. **VDrive sandbox** — all file operations are jailed to the mounted workspace.
   Path traversal is validated at the filesystem layer. `../../etc/passwd` resolves
   to nothing.

3. **WASM capability sandbox** — user tools run in WebAssembly with
   capability-based grants. A tool declares what it needs; the platform
   grants exactly that. No ambient authority.

```yaml
profiles:
  admin:
    listeners: all                    # full access
  read-only:
    listeners: [file-read, glob, grep]  # structurally can't write
```

**Injection guard middleware** scans tool outputs for prompt injection
attempts before they reach the agent. Flagged content is quarantined.

## Organism System

Agents are defined in **organism YAML** files — self-contained configurations
that declare everything an agent needs to run.

**Imports** let organisms share infrastructure:
```yaml
imports:
  - infrastructure.yaml     # tools, LLM pool, librarian
  - shared-prompts.yaml     # common safety prompts
```

**Buffers** make agents callable as tools — other agents delegate to them
through a structured interface:
```yaml
- name: coder
  handler: buffer
  buffer:
    description: "Write a function"
    parameters:
      name: { type: string, description: "Function name" }
      purpose: { type: string, description: "What it should do" }
      language: { type: string, enum: [rust, python, typescript] }
    required: [name, purpose, language]
    organism: organisms/coder.yaml      # runs in isolated child pipeline
```

The calling agent fills in parameters like a tool call. The buffer spawns
a child pipeline, passes the parameters as a prompt, and returns the result.
The caller doesn't know it's talking to another agent — it's just a tool
with a structured interface.

**Auto-discovery** — Bob uses `tools: auto` to discover every tool and
specialist in the organism. Install a new agent, Bob sees it immediately.

## Ships Ready to Work

Point AgentOS at a codebase and start working. It ships with a team:

| Agent | Role |
|-------|------|
| **Bob** | Concierge — reads your task, delegates to the right specialist |
| **coder** | Writes code, edits files, runs tests. Isolated child pipeline with write tools |
| **plan-expert** | Surveys codebase, produces structured execution plans, delegates steps to coder |
| **wiki-expert** | Generates project documentation as interlinked wiki pages |
| **agent-expert** | Designs, validates, and tests new organism configurations |

Agent identity is data, not code. New agents require a YAML file and a
prompt — zero Rust.

## The Control Room

A terminal UI for working with agents:

- **Chat** — conversation with the agent, markdown rendering, D2 diagrams
- **Threads** — thread list, conversation timeline, context tree
- **YAML editor** — syntax highlighting, schema-aware completions, hover docs, diagnostics
- **Python editor** — write custom tools with tree-sitter highlighting and ghost-text completion
- **Debug** — live activity trace (agent thinking, tool calls, routing decisions)

The editors run in-process language services — same intelligence as an LSP
(completions, diagnostics, hover) but as pure functions, no external process.

## Quick Start

```bash
cargo build --release

# Run with API key
export ANTHROPIC_API_KEY=sk-ant-...
cargo run

# Or configure interactively
cargo run
# Then: /models add anthropic
```

## Building

```bash
cargo build                # debug
cargo test --lib           # 900+ tests, no API key needed
cargo test                 # full suite including API integration tests
```

~40,000 lines of Rust. Zero unsafe blocks. Runs on a Raspberry Pi 5.

## Origin Story

In October 2025, a conversation with an AI produced a Python program
that could self-spawn agent swarms. Before running it, the question was
asked: *"Isn't it going to take over my computer?"* The answer: *"Yes."*

That moment — asking the question before hitting enter — became the seed
for everything. Sandboxed VDrive, permission gates, bounded agents, structural
security. The whole architecture traces back to the instinct that an agent
without containment is a program without an operating system.

If you're building agents and the question "won't this take over?" doesn't
have a structural answer, you need an OS.

## License

BUSL-1.1
