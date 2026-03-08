# AgentOS Platform Reference

> For the agent-expert: everything you need to know to design and create agents.

## What is AgentOS?

AgentOS is an operating system for AI agents. It runs on a validated XML message bus
where agents, tools, and infrastructure communicate through typed messages. Everything
is declared in YAML configuration files called "organisms."

## Core Concepts

### Organism
A YAML file that defines a complete agent system: its agents, tools, infrastructure,
prompts, and security profiles. An organism is self-contained — it declares everything
needed to run.

### Listener
The fundamental unit. Every component is a listener — agents, tools, LLM pools,
the librarian. A listener has a name, handles one payload type, and may call other
listeners listed in its `peers`.

### Agent
A listener with `agent: true` (or an agent config block). Agents have:
- A **system prompt** (composed from named templates in `prompts:`)
- **Tools** — other listeners listed in `peers` that the LLM can call
- **Permissions** — per-tool approval tiers: `auto`, `prompt`, or `deny`
- An **agentic loop** — the agent calls tools, reads results, and iterates

### Buffer
A listener that makes another agent callable as a tool. The calling agent sends
parameters; the system spawns an isolated child pipeline to execute. Buffers enable
delegation without direct agent-to-agent communication.

Key properties:
- `description` — what the calling agent sees (this IS the tool description)
- `parameters` — the tool's input schema
- `organism` — path to child YAML (omit to clone current organism for self-referential patterns)
- `requires` — which tools the child pipeline gets
- `max_concurrency` — parallel execution limit
- `timeout_secs` — execution timeout

### Prompt Composition
Prompts are named templates in the top-level `prompts:` section. An agent references
them by label. Use `&` to compose multiple: `"safety & coding_base"`.

The magic variable `{tool_definitions}` is replaced at runtime with the agent's
available tool descriptions. Always include it.

### Security Profile
Profiles control access: which listeners are active, which can use the network,
and message retention policy. Every organism needs at least one profile.

## Architecture Patterns

### Single Agent (simplest)
One agent with direct tool access. Good for focused tasks.

```
User -> agent -> [tools]
```

### Plan-then-Execute
A planner agent explores and plans, then delegates to a coder agent via buffer.
The planner has read-only tools; the coder has write tools.

```
User -> planner -> [read tools, coder-buffer]
                       |
                   coder -> [write tools]
```

### Parallel Research
An agent that can fork itself to investigate sub-topics concurrently.
Self-referential buffer: the agent lists itself as a peer.

```
User -> researcher -> [tools, researcher-buffer]
                           | (up to N parallel)
                       researcher -> [tools, researcher-buffer]
```

### Multi-Specialist
A coordinator routes to multiple specialist agents based on the task.

```
User -> coordinator -> [analyzer-buffer, coder-buffer, reviewer-buffer]
```

## Available Tools

### File Operations
| Tool | Description | When to include |
|------|-------------|-----------------|
| `file-read` | Read file contents | Almost always — agents need to read code |
| `file-write` | Create/overwrite files | When agent creates new files |
| `file-edit` | Edit existing files (search/replace) | When agent modifies existing code |

### Search
| Tool | Description | When to include |
|------|-------------|-----------------|
| `glob` | Find files by pattern | Codebase exploration |
| `grep` | Search file contents | Finding code references |
| `codebase-index` | Tree-sitter structural index | Deep code understanding |

### Execution
| Tool | Description | When to include |
|------|-------------|-----------------|
| `bash` | Run shell commands | Tests, builds, git, system commands |

### Infrastructure
| Component | Purpose | When to include |
|-----------|---------|-----------------|
| `llm-pool` | LLM inference | Always — every agent organism needs this |
| `librarian` | Context curation via Haiku | When agent handles large codebases |

## Design Decisions

### Choosing the Right Pattern

**Use a single agent when:**
- The task is well-scoped (one domain, clear success criteria)
- All needed tools can be given to one agent without risk
- You want simplicity

**Use plan-then-execute when:**
- Exploration and execution are separate concerns
- You want the planner to have read-only access (safety)
- Tasks are complex enough to benefit from explicit planning

**Use parallel research when:**
- The problem decomposes into independent sub-tasks
- You need to explore multiple approaches simultaneously
- Bounded by `max_concurrency` to prevent runaway forking

**Use multi-specialist when:**
- Different sub-tasks need different tool sets or expertise
- You want clear separation of concerns
- A coordinator can route based on task type

### Permission Tiers

- `auto` — tool runs without user approval. Use for read-only, safe operations.
- `prompt` — user must approve each call. Use for writes, deletes, commands.
- `deny` — tool is never allowed. Use to restrict inherited tools.

A good default: read tools on `auto`, write tools on `prompt`.

### Prompt Engineering for Agents

1. **Be specific about role** — "You are a coding agent" not "You are helpful"
2. **State constraints** — what the agent should NOT do matters as much as what it should
3. **Include workflow** — numbered steps guide the agent's approach
4. **Reference tools by name** — the agent needs to know what it has
5. **Include `{tool_definitions}`** — always, so the agent sees its tool schemas
6. **Compose with safety** — prepend `no_paperclipper` or similar boundary prompt

### Naming Conventions

- Listener names: `kebab-case` (e.g., `coding-agent`, `file-read`)
- Prompt labels: `snake_case` (e.g., `coding_base`, `no_paperclipper`)
- Organism names: `kebab-case` (e.g., `coder-child`)
- Profile names: `snake_case` (e.g., `coding`, `building`)

## Constraints

- **No inter-agent communication** — agents cannot call other agents directly. Use buffers.
- **No shared state between buffer children** — each fork is isolated.
- **No dynamic tool registration** — tools are fixed at startup.
- **One agent per child organism** — buffer children have exactly one agent listener.
- **No prompt inheritance** — child organisms must declare their own prompts.
- **Peers must exist** — every name in `peers` must be a declared listener.

## Template: Minimal Agent

```yaml
organism:
  name: my-agent

prompts:
  safety: |
    You are bounded. You do not pursue goals beyond your task.

  agent_base: |
    You are [describe role]. [Describe what the agent does.]

    {tool_definitions}

listeners:
  - name: my-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "[One-line description]"
    agent:
      prompt: "safety & agent_base"
      max_tokens: 4096
      max_agentic_iterations: 25
      permissions:
        file-read: auto
        glob: auto
        grep: auto
    peers: [file-read, glob, grep]

  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM inference pool"
    ports:
      - port: 443
        direction: outbound
        protocol: https
        hosts: [api.anthropic.com]

  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"

  - name: glob
    payload_class: tools.GlobRequest
    handler: tools.glob.handle
    description: "Glob search"

  - name: grep
    payload_class: tools.GrepRequest
    handler: tools.grep.handle
    description: "Grep search"

profiles:
  default:
    linux_user: agentos
    listeners: [my-agent, file-read, glob, grep, llm-pool]
    network: [llm-pool]
    journal: retain_forever
```

## Template: Plan-then-Execute

```yaml
organism:
  name: plan-execute

prompts:
  safety: |
    You are bounded. You do not pursue goals beyond your task.

  planner_prompt: |
    You are a planning agent. Explore the codebase, understand the problem,
    and create a detailed implementation plan. Then delegate to the coder.

    {tool_definitions}

listeners:
  - name: planner
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Planning agent"
    agent:
      prompt: "safety & planner_prompt"
      max_tokens: 4096
      max_agentic_iterations: 25
      permissions:
        file-read: auto
        glob: auto
        grep: auto
        coder: prompt
    peers: [file-read, glob, grep, coder]

  - name: coder
    payload_class: buffer.CoderRequest
    handler: buffer
    description: "Coding agent — executes plans"
    buffer:
      description: "Execute an implementation plan"
      parameters:
        plan:
          type: string
          description: "The implementation plan to execute"
      required: [plan]
      requires: [file-read, file-write, file-edit, glob, grep, bash]
      organism: organisms/coder.yaml
      timeout_secs: 600

  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM inference pool"
    ports:
      - port: 443
        direction: outbound
        protocol: https
        hosts: [api.anthropic.com]

  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"

  - name: glob
    payload_class: tools.GlobRequest
    handler: tools.glob.handle
    description: "Glob search"

  - name: grep
    payload_class: tools.GrepRequest
    handler: tools.grep.handle
    description: "Grep search"

profiles:
  default:
    linux_user: agentos
    listeners: [planner, coder, file-read, glob, grep, llm-pool]
    network: [llm-pool]
    journal: retain_forever
```

## Reference Files

When creating organisms, always consult:
- `organisms/organism.schema.json` — the machine-readable JSON Schema (ground truth)
- `organisms/GUIDE.md` — architectural constraints, patterns, built-in listener templates
- `organisms/default.yaml` — real-world example: planner + coder + organism-builder
