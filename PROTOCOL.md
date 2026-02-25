# AgentOS Pipeline Protocol Specification

Version: 0.1 (draft)
Derived from: Rust implementation (AgentOS) + xml-pipeline Python reference

This specification describes the protocol in language-agnostic terms.
An implementation may use any language, runtime, or serialization format
provided it upholds the stated guarantees.

---

## 1. Message Envelope

The **envelope** is the fundamental unit of communication. Every interaction
in the pipeline — tool calls, agent responses, LLM requests, journal entries —
is an envelope.

### 1.1 Fields

| Field | Type | Description |
|-------|------|-------------|
| `namespace` | URI | Identifies the schema family (e.g., `urn:agentos:tools:v1`) |
| `payload_tag` | string | Discriminator for the payload type (e.g., `FileReadRequest`) |
| `payload` | bytes | Opaque content. Interpreted only by the recipient handler. |
| `sender` | string | Name of the originating handler or external source |
| `thread_id` | string | Thread context for this message (see Section 5) |
| `profile` | string | Security profile governing dispatch (see Section 8) |

### 1.2 Guarantees

- **Payload is opaque.** The pipeline MUST NOT interpret payload content
  for routing decisions. Routing is determined by `payload_tag` and
  the dispatch table — never by payload inspection.
- **Envelope is immutable after creation.** Fields are set once. Handlers
  receive a validated copy; they cannot mutate the envelope in transit.

---

## 2. Pipeline Stages

A message traverses a defined sequence of stages. Implementations MAY
vary the number of stages but MUST preserve the ordering guarantees.

### 2.1 Required Stages

1. **Validation** — Verify envelope structure. Reject malformed envelopes
   before they enter the pipeline. If a schema is registered for the
   `payload_tag`, validate the payload against it.

2. **Security Check** — Resolve the `profile` to a dispatch table.
   Verify that the `payload_tag` has a registered route. If no route
   exists, the message is **structurally rejected** — not filtered,
   not logged, not deferred. It cannot proceed.

3. **Dispatch** — Route the envelope to the handler registered for the
   `payload_tag`. Exactly one handler receives the message.

4. **Response Validation** — Handler output is raw bytes. Before
   re-entering the pipeline, these bytes MUST be validated against the
   handler's declared response schema (Section 3.2). If validation
   fails, the message is rejected — it never reaches the next handler.
   This validation checkpoint is pipeline-enforced; the receiving
   handler is not responsible for validating its input.

5. **Response Re-entry** — Validated handler output re-enters the pipeline
   as a **new envelope** subject to the same security and dispatch stages.
   Handler output is untrusted until it passes the validation checkpoint.

### 2.2 Guarantee: Zero Trust Re-entry

This is the core invariant. A handler's response is not privileged.
It exits as raw bytes, passes through schema validation (stage 4),
re-enters as a validated envelope, gets security-checked, gets
dispatched. A compromised handler cannot escalate privileges because
its output passes through the same gates as external input. The
validation checkpoint between handler output and next-handler dispatch
is the enforcement point — the schema is a firewall rule, not documentation.

### 2.3 Optional Stages

- **Schema Validation** — Type-level validation of payload content against
  a registered schema (XSD, JSON Schema, or equivalent). Runs after
  structural validation, before security check.
- **Canonicalization** — Normalize payload representation before validation
  (e.g., C14N for XML). Prevents equivalent-but-different payloads from
  producing different validation results.
- **Signing** — Cryptographic signature on envelope contents for audit
  integrity. Append-only; does not modify the envelope.
- **Repair** — Attempt to fix malformed payloads before rejection.
  MUST be idempotent. MUST NOT alter semantics.

---

## 3. Handlers

A **handler** is a unit of computation that receives a validated envelope
and produces a response.

### 3.1 Handler Contract

```
handle(payload, context) → Response
```

Where:
- `payload` — validated payload bytes + tag
- `context` — thread_id, sender name, own name (the handler's registered name)

Response is one of:
- **Reply** — send payload back to the sender (reversed routing)
- **Send** — send payload to a named handler (forward routing)
- **Broadcast** — send payload to multiple named handlers
- **Silence** — consume the message, produce no output.
  The pipeline MUST synthesize an acknowledgment envelope back to the
  sender so the sender's thread can continue or terminate cleanly.
  Pure silence without an ACK would stall any sender waiting on a response.
- **Error** — signal failure with a structured error payload

### 3.2 Handler Metadata

Every handler MUST provide:
- **name** — unique identifier within the pipeline
- **description** — human-readable purpose (used by semantic routing)
- **payload_tag** — the tag this handler accepts
- **request_schema** — schema for the payload it accepts (SHOULD be provided)
- **response_schema** — schema for the payload it produces (SHOULD be provided)

Handlers MAY provide:
- **semantic_description** — extended natural-language description for
  embedding-based routing (see Section 10)
- **usage_instructions** — guidance for agents on when/how to invoke
- **peers** — list of handler names this handler may Send to

### 3.3 Guarantee: Handlers Are Isolated

A handler has no direct reference to other handlers, to the dispatch
table, or to the pipeline itself. It receives bytes and returns bytes.
Communication happens only through the pipeline's routing mechanism.

---

## 4. Dispatch Tables

A **dispatch table** maps `payload_tag` → handler. It is the primary
security mechanism.

### 4.1 Properties

- **Static after construction.** The dispatch table for a given profile
  is fixed at pipeline startup. It cannot be modified at runtime by
  handlers or by payload content.
- **Closed world.** If a `payload_tag` is not in the table, the message
  is rejected. There is no wildcard, no fallback, no dynamic registration
  by handlers.
- **Per-profile.** Different security profiles have different dispatch
  tables. A `researcher` profile may route `FileReadRequest` but not
  `FileWriteRequest`.

### 4.2 Alternate Routing Tables

A handler MAY have multiple routing registrations for different payload
tags. This enables a single handler to accept messages from different
sources (e.g., an agent handler accepts both `AgentTask` from users and
`ToolResponse` from tools).

The dispatch table resolves the **first matching route** for the payload
tag within the active profile's allowed handlers.

### 4.3 Guarantee: Structural Impossibility

Security is not a filter that can be bypassed. If the dispatch table
does not contain a route, the message **cannot be delivered** — there is
no code path that leads to the handler. This is the difference between
"access denied" (a policy decision that could be overridden) and
"route not found" (a structural impossibility).

---

## 5. Thread Model

**Threads** are the call stack of the pipeline. They provide context
isolation and enable recursive composition.

### 5.1 Thread Identifiers

Thread IDs are dot-separated hierarchical paths:
```
root
root.task-1
root.task-1.subtask-a
root.task-1.subtask-a.subtask-a    (recursion)
```

### 5.2 Thread Operations

- **Spawn** — Create a child thread. The child inherits the parent's
  security profile (or a more restrictive one — never more permissive).
- **Return** — Complete a thread and return a result to the parent.
  The child thread's context MAY be pruned, folded, or retained
  depending on journal policy.

### 5.3 Thread Properties

| Property | Description |
|----------|-------------|
| `thread_id` | Hierarchical path |
| `profile` | Security profile (dispatch table selection) |
| `state` | Active, Completed, Failed |
| `parent` | Parent thread ID (None for root) |
| `children` | Child thread IDs |

### 5.4 Guarantee: Turing Completeness

Threads can recurse arbitrarily (`root.a.b.c.c.c...`). Combined with
conditional branching (handler logic) and unbounded context (the store),
the pipeline is Turing complete. This is by design — agents need
arbitrary recursion depth for task decomposition.

### 5.5 Guarantee: Profile Monotonicity

A child thread's profile MUST be equal to or more restrictive than its
parent's. Privilege escalation through thread spawning is structurally
impossible.

---

## 6. Context Management

The **context store** is virtual memory for attention. It manages what
the agent can "see" during inference.

### 6.1 Three-Tier Model

| Tier | Description | Analogous To |
|------|-------------|--------------|
| **Expanded** | Full content, actively in the attention window | RAM |
| **Folded** | Summary replaces full content; original on disk | Swap |
| **Evicted** | On disk only; not visible to the agent | Cold storage |

### 6.2 Context Segments

A **segment** is the unit of context management:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier |
| `thread_id` | Which thread owns this segment |
| `content` | Full content (when expanded) or summary (when folded) |
| `content_type` | Classification (message, code, tool_result, etc.) |
| `relevance` | Numeric score (0.0–1.0) relative to current task |
| `status` | Expanded, Folded, or Evicted |
| `byte_size` | Size of full content |
| `token_estimate` | Estimated tokens (for budget management) |

### 6.3 Context Curation (The Librarian)

An automated curator manages context tier transitions:
- **Fold** when relevance drops below threshold
- **Evict** when total context exceeds budget
- **Unfold** when a folded segment becomes relevant again

The curator is proactive — it manages context by relevance **before**
capacity pressure forces eviction. This is kswapd, not the OOM killer.

Key principle: **every irrelevant token in context degrades the thinker.**
Aggressive folding on relevance produces better results than keeping
full history. Focused 300 tokens beats full 128K history.

### 6.4 Fold Summaries

When a segment is folded, its full content is replaced by a constructed
summary. The summary is a **constructed memory**, not a verbatim excerpt.
The original content is retained on disk for potential unfolding.

This mirrors human autobiographical memory: you remember the *answer*
(847), not the *carry digits*. Forgetting procedural detail while
retaining semantic content is the primary feature of functional memory.

### 6.5 Guarantee: No Silent Data Loss

Folding and eviction are reversible. Full content is always retained
on disk. The worst case of a bad curation decision is one extra
round-trip to unfold — never silent data loss.

---

## 7. Message Journal

The **journal** records messages that pass through the pipeline.
It is the audit trail and replay tape. The journal is configurable —
retention policy determines whether it grows indefinitely, prunes on
delivery, or retains for a fixed window. A stateless online concierge
may use `prune_on_delivery`; a coding agent may use `retain_forever`.
The journal is not mandatory-forever — it is mandatory-configurable.

### 7.1 Retention Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `retain_forever` | Never delete | Coding agents (full history) |
| `prune_on_delivery` | Delete after handler confirms receipt | Stateless tools |
| `retain_days(N)` | Delete after N days | Compliance requirements |

### 7.2 Journal Entry Fields

| Field | Description |
|-------|-------------|
| `id` | Monotonically increasing sequence number |
| `timestamp` | Wall-clock time of entry |
| `thread_id` | Thread context |
| `direction` | Inbound (received) or Outbound (sent) |
| `handler` | Handler that produced/consumed this message |
| `payload_tag` | Discriminator |
| `payload_hash` | Integrity hash of payload content |
| `retention` | Active retention policy |

### 7.3 Guarantee: Append-Only

The journal is append-only. Entries are never modified after creation.
Retention policies delete entries; they do not alter them.

---

## 8. Security Model

Security has three concentric layers. Each is independent — compromising
one does not compromise the others.

### 8.1 Layer 1: Dispatch Table (Structural)

The dispatch table is the primary security mechanism (see Section 4).
A missing route is a structural impossibility, not a policy decision.

### 8.2 Layer 2: Process Isolation (OS-Level)

Each organism (pipeline instance) SHOULD run as its own OS user with
minimal privileges. File system access, network access, and IPC are
constrained by OS-level permissions.

### 8.3 Layer 3: Kernel Isolation (Data-Level)

The kernel (WAL, thread table, context store, journal) runs in a single
process with atomic operations. No shared memory with handlers. Handlers
communicate with the kernel only through the pipeline's message protocol.

### 8.4 Profiles

A **profile** defines:
- Which handlers are accessible (the dispatch table)
- Which handlers may access the network (port allowlist)
- The journal retention policy
- The OS user identity

### 8.5 Guarantee: No Runtime Downgrade

A profile cannot be changed mid-conversation. The dispatch table for a
thread is fixed when the thread is created. A prompt injection that
requests elevated access cannot change the structural routing.

---

## 9. Tool Protocol

**Tools** are handlers that perform side effects (file I/O, shell
commands, network requests). They don't think — they execute.

### 9.1 Tool Peer Contract

A tool MUST provide:
- Handler metadata (Section 3.2)
- **Request schema** — defines the payload structure it accepts
- **Response schema** — defines the payload structure it produces

A tool response is always one of:
- **Success** — result payload
- **Error** — structured error with message

### 9.2 Schema-Driven Tool Definitions

Tool schemas SHOULD be automatically derivable from the tool's type
definitions (e.g., a derive macro, decorator, or code generator).
Hand-written schemas are acceptable but error-prone.

The derived schema serves three purposes:
1. **Validation** — reject malformed payloads before the tool sees them
2. **Documentation** — self-describing tools (no separate docs to maintain)
3. **Agent prompting** — schemas are injected into the agent's system
   prompt so the LLM knows how to construct valid tool calls

### 9.3 Guarantee: Tools Are Stateless Per-Call

A tool handler MUST NOT maintain state between invocations that affects
its behavior. Each call is independent. (The tool may have internal
caches or connection pools, but these are invisible to the protocol.)

---

## 10. Semantic Routing

**Semantic routing** enables capability discovery without hardcoded
dispatch tables. An agent describes what it needs in natural language;
the router finds the handler that provides that capability.

### 10.1 Architecture

```
Agent: "I need to read the contents of main.rs"
  → Embedding: [0.82, 0.14, ...]
  → Similarity search against handler descriptions
  → Match: file-read (score: 0.94)
  → Form fill: { path: "main.rs" }
  → Dispatch to file-read handler
```

### 10.2 Components

- **Embedding provider** — converts text to vector representations.
  Implementations may use TF-IDF, ONNX models, or API-based embeddings.
- **Router** — maintains an index of handler descriptions and their
  embeddings. Performs nearest-neighbor search.
- **Form filler** — given a matched handler and the agent's natural
  language request, constructs a valid request payload. May use an
  LLM (model ladder: cheap model first, escalate on failure).

### 10.3 Dispatch Table Masking

The router produces a **ranked list** of candidate handlers, not a single
match. The dispatch table acts as a mask over this list — candidates not
permitted by the active profile are filtered out **before** form-filling.

```
Agent: "delete the temp files"
  → Ranked candidates: [file-erase: 0.94, file-write: 0.87, file-read: 0.82]
  → Dispatch table mask (researcher profile): file-erase NOT allowed
  → Filtered: [file-write: 0.87, file-read: 0.82]
  → Top allowed match: file-write (0.87)
  → Form fill against file-write schema
  → Dispatch
```

This ordering is critical:
1. **Rank** — similarity search produces scored candidates
2. **Mask** — dispatch table removes disallowed handlers
3. **Select** — top remaining candidate proceeds to form-filling
4. **Fill** — LLM constructs valid payload for the selected handler
5. **Dispatch** — standard pipeline dispatch

If the router misreads intent (matches file-erase when the agent meant
file-read), the dispatch table masks file-erase, and file-read bubbles
up as the top allowed candidate — correct behavior despite the misread.

If ALL candidates are masked out, the router returns a structured error:
"no matching capability in your profile." This is a real denial, not a
misread — the agent is genuinely asking for something it cannot do.

Form-filling is expensive (LLM call). Masking before filling ensures no
LLM calls are wasted constructing parameters for a tool that would be
rejected by the security check.

### 10.4 Invisible Dispatch

The agent SHOULD NOT need to know tool names or payload schemas.
It describes intent; the router translates to a concrete tool call.
This enables new tools to be added without modifying agent prompts.

### 10.5 Guarantee: Router Is Advisory

Semantic routing is a convenience layer. The dispatch table (Section 4)
is the authority. The router cannot override structural security — the
dispatch table mask (10.3) ensures that only permitted handlers are
ever considered, regardless of semantic similarity score.

---

## 11. Agent Protocol

An **agent** is a handler that maintains a conversation loop with an
LLM. It is the "thinker" — the only component that uses inference.

### 11.1 Agent Loop

```
1. Receive task (AgentTask payload)
2. Construct prompt (system + context + task)
3. Call LLM for inference
4. Parse response:
   a. Text only → Reply with AgentResponse
   b. Tool calls → Dispatch each to the named tool
5. Receive tool results (ToolResponse payloads)
6. Append results to conversation history
7. Go to step 3 (iterate until done or limit reached)
```

### 11.2 Agent Identity Is Data

An agent's identity — its prompt, model, token limits, iteration caps —
is declared in configuration, not in code. Creating a new agent type
requires a configuration entry and a prompt template. No code changes.

### 11.3 Prompt Composition

Prompts are named blocks that can be composed:
```
"no_paperclipper & coding_base"
```
This concatenates the `no_paperclipper` and `coding_base` prompt blocks
with a newline separator. Prompts support template variables
(e.g., `{tool_definitions}`) interpolated at runtime.

### 11.4 Tool Call Translation

The agent translates between the LLM's native tool-call format and the
pipeline's envelope format. This is a mechanical translation, not
interpretation. The agent does not inspect or modify tool call content.

### 11.5 Guarantee: Bounded Iteration

Agents MUST have a configurable maximum iteration count. An agent that
enters an infinite tool-call loop MUST be terminated after the limit.
The limit is declared in configuration, not hardcoded.

### 11.6 Guarantee: One Thinker

Only agents call LLMs. Tools don't think. The librarian (context
curator) may use a cheaper model for summarization, but it is not
an agent — it does not maintain a conversation or dispatch tool calls.

---

## 12. Organism Configuration

An **organism** is a complete pipeline configuration: handlers, profiles,
prompts, and wiring. It is the single source of truth.

### 12.1 Configuration Scope

| Section | Contents |
|---------|----------|
| `organism.name` | Pipeline instance identity |
| `prompts` | Named prompt blocks (inline or file reference) |
| `listeners` | Handler declarations with metadata |
| `profiles` | Security profiles (dispatch tables, network, journal) |

### 12.2 Listener Declaration

Each listener declares:
- `name` — unique handler identifier
- `payload_class` — the payload tag it handles
- `handler` — implementation reference
- `description` — human-readable purpose
- `peers` — handlers it may communicate with
- `agent` — agent configuration (if this is an agent handler)
- `ports` — network port declarations (for firewall)

### 12.3 Agent Configuration Block

```yaml
agent:
  prompt: "no_paperclipper & coding_base"
  max_tokens: 4096
  max_iterations: 20
  model: opus
```

### 12.4 Guarantee: Configuration Is Static

The organism configuration is loaded at startup and does not change
during runtime. Hot-reloading MAY be supported but MUST NOT alter
the security properties of running threads (see Section 8.5).

---

## 13. WASM Tool Sandboxing

User-defined tools run as WebAssembly components with capability-based
security via the WIT (WebAssembly Interface Type) system.

### 13.1 Capability Model

A WASM tool declares its required capabilities in its WIT interface:
```wit
interface my-tool {
    use wasi:filesystem/types.{descriptor}
    use wasi:http/types.{outgoing-request}

    record request { path: string }
    record response { content: string }

    handle: func(req: request) -> result<response, string>
}
```

The tool can ONLY access what its WIT interface declares. No ambient
authority. No "just this once" escalation.

### 13.2 Host Sovereignty

The host (pipeline) controls what capabilities are actually provided
to the WASM component. A tool may declare it needs filesystem access,
but the host may provide a virtual filesystem scoped to a single
directory. The WIT interface is the maximum; the host decides the actual.

### 13.3 Guarantee: Deterministic Sandboxing

WASM execution is deterministic and memory-isolated. A malicious tool
cannot access host memory, other tools' memory, or the pipeline's
internal state. The sandbox is enforced by the WASM runtime, not by
trust in the tool's code.

---

## 14. Durable State (Kernel)

The **kernel** provides three pieces of nuclear-proof state. Everything
else in the pipeline is ephemeral.

### 14.1 Write-Ahead Log (WAL)

All state mutations are written to a WAL before being applied. The WAL
enables crash recovery: on startup, replay uncommitted entries to
restore consistent state.

### 14.2 Atomicity

State operations (thread creation, context updates, journal appends)
are atomic. A partial write is either completed on recovery or rolled
back. There is no inconsistent intermediate state visible to handlers.

### 14.3 Single-Writer

One process, one WAL. There is no distributed coordination, no
consensus protocol, no split-brain risk. The kernel is intentionally
simple because reliability matters more than throughput.

---

## 15. Callable Organisms (Future)

An organism MAY be exposed as a tool to other organisms. This enables
agent-as-tool composition: one agent calls another agent through the
pipeline's standard tool protocol.

### 15.1 Callable Declaration

```yaml
listeners:
  - name: research-agent
    callable:
      description: "Deep research on a topic"
      parameters:
        - name: query
          type: string
          required: true
      requires: [llm-pool, file-read, grep]
```

### 15.2 Cross-Organism Dispatch

When organism A calls organism B as a tool:
1. A's agent constructs a tool call (standard tool protocol)
2. The pipeline translates to B's `AgentTask` format
3. B executes independently (own threads, own context, own profile)
4. B's result returns as a `ToolResponse` to A

### 15.3 Guarantee: Host Sovereignty

The calling organism controls the dispatch table. Callable organisms
cannot access capabilities beyond what the host's profile allows.
The callee runs in the caller's security context unless explicitly
granted its own profile.

---

## Appendix A: What This Spec Does NOT Prescribe

- **Serialization format.** XML, JSON, MessagePack, Protobuf — the
  pipeline is format-agnostic. Payloads are bytes. However, the chosen
  format MUST have a rigorous schema validation story (Section 2.1,
  stage 4) since validation is a security boundary. XML+XSD is the
  reference implementation. JSON+JSON Schema is viable but note that
  JSON Schema has multiple incompatible draft versions and validator
  implementations that disagree on edge cases — choose a draft, pin
  a validator, and treat it as a hard dependency.
- **Async runtime.** tokio, libuv, OS threads — implementation choice.
- **LLM provider.** Anthropic, OpenAI, local models — the agent
  protocol is provider-agnostic.
- **Programming language.** The C ABI layer, the Rust crate, and WASM
  components are all valid implementations of this protocol.
- **Transport.** In-process, IPC, network — handlers don't know or
  care how messages are delivered.

## Appendix B: Invariants Summary

1. **Zero Trust Re-entry** — handler output is raw bytes, validated against schema before dispatch (2.2)
2. **Structural Security** — missing route = impossibility, not denial (4.3)
3. **Profile Monotonicity** — children never escalate privileges (5.5)
4. **No Runtime Downgrade** — profiles are fixed per-thread (8.5)
5. **No Silent Data Loss** — folding/eviction is reversible (6.5)
6. **Append-Only Journal** — entries are never modified (7.3)
7. **Bounded Iteration** — agents have configurable limits (11.5)
8. **Deterministic Sandboxing** — WASM tools are memory-isolated (13.3)
9. **Single-Writer Atomicity** — one process, one WAL (14.3)
10. **Router Is Advisory** — semantic routing cannot override security (10.4)
