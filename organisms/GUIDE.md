# Organism YAML Guide

> Machine-readable schema: [organism.schema.json](organism.schema.json)

Architectural constraints, patterns, and built-in templates for writing AgentOS organism configuration files. For field types, defaults, and descriptions see the JSON Schema.

## Constraints the platform cannot express

- **No inter-agent communication** — agents cannot call other agents directly. Use a buffer to delegate.
- **No shared state between buffer children** — each fork gets its own kernel, context store, and thread table.
- **No dynamic tool registration** — tools are fixed at pipeline build time.
- **No conditional listeners** — all declared listeners are always available to their profile.
- **No prompt inheritance** — child organisms must redeclare their prompts.
- **One agent per child organism** — buffer takes `agent_listeners().next()`. Multiple agents in a child YAML is undefined.

## Child organism rules

- Child YAML must define exactly **one agent** listener (the entry point)
- Child needs its own `llm-pool` listener for inference
- Child needs its own `prompts:` section (no inheritance from parent)
- The `requires:` list controls which tools are registered in the child pipeline

## Prompt composition

Prompt labels reference entries in the top-level `prompts:` section. Use `&` to concatenate multiple labels. The template variable `{tool_definitions}` is interpolated at runtime with the agent's available tools.

## Self-referential buffers

Omit `organism` from a buffer block to clone the current organism. The agent can fork itself in parallel, bounded by `max_concurrency` and `max_agentic_iterations`:

```yaml
- name: researcher
  handler: agent.handle
  agent:
    prompt: "research_base"
  peers: [file-read, researcher]     # lists itself as peer
  buffer:
    description: "Research a sub-topic in parallel"
    parameters:
      topic: { type: string, description: "Sub-topic to investigate" }
    required: [topic]
    max_concurrency: 3
```

## Common patterns

**Read-only explorer** — agent with only read tools, no write access:
```yaml
peers: [file-read, glob, grep, codebase-index]
permissions:
  file-read: auto
  glob: auto
  grep: auto
  codebase-index: auto
```

**Plan-then-execute** — planner calls coder as buffer:
```yaml
peers: [file-read, glob, grep, codebase-index, coder]
# coder is a buffer node with write tools in its child organism
```

**Parallel researcher** — self-referential buffer for divide-and-conquer:
```yaml
peers: [file-read, researcher]
buffer:
  description: "Research sub-topic"
  parameters:
    topic: { type: string }
  required: [topic]
  max_concurrency: 3
```

## Built-in infrastructure listeners

These are provided by the runtime. Include them when your agent needs LLM inference or context curation:

```yaml
- name: llm-pool
  payload_class: llm.LlmRequest
  handler: llm.handle
  description: "LLM inference pool"
  ports:
    - port: 443
      direction: outbound
      protocol: https
      hosts: [api.anthropic.com]

- name: librarian
  payload_class: librarian.LibrarianRequest
  handler: librarian.handle
  description: "Context curator"
  peers: [llm-pool]

- name: codebase-index
  payload_class: treesitter.CodeIndexRequest
  handler: treesitter.handle
  description: "Tree-sitter code indexing"
```

## Built-in tool listeners

```yaml
- name: file-read
  payload_class: tools.FileReadRequest
  handler: tools.file_read.handle
  description: "Read files"

- name: file-write
  payload_class: tools.FileWriteRequest
  handler: tools.file_write.handle
  description: "Write files"

- name: file-edit
  payload_class: tools.FileEditRequest
  handler: tools.file_edit.handle
  description: "Edit files"

- name: glob
  payload_class: tools.GlobRequest
  handler: tools.glob.handle
  description: "Glob search"

- name: grep
  payload_class: tools.GrepRequest
  handler: tools.grep.handle
  description: "Grep search"

- name: command-exec
  payload_class: tools.CommandExecRequest
  handler: tools.command_exec.handle
  description: "Command execution"
```

## Known tool names for `requires`

`file-read`, `file-write`, `file-edit`, `glob`, `grep`, `command-exec`
