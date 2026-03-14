# AgentOS — Project Notes

## Architecture Overview
AgentOS is a Rust-based agent operating system with a TUI interface. Key crates:
- **agentos** (main binary): TUI + agent orchestration (~40k lines Rust)
- **agentos-kernel**: Durable state engine (WAL, thread table, context store, journal)
- **agentos-events**: Shared event types
- **agentos-vdrive**: QCOW2 virtual drive sandbox

## Memory Management Review (2026-03-11)

An independent analysis was performed by comparing AgentOS's patterns against
memory leak issues found in Claude Code v2.1.74 (which was leaking 4.6GB heap
due to unbounded state accumulation and immutable array copies).

### What's working well
- **Three-tier VMM context store** (Active/Shelved/Folded) with fold_store eviction — exactly what Claude Code is missing
- **Bounded event bus**: `broadcast::channel(256)` with lagged subscribers skipped — prevents queue memory leaks
- **WAL-backed durability** — state survives crashes, not purely in-process
- **Rust ownership** — no accidental full-history clones via immutable patterns

### Items to address

1. ~~**`AgentThread.messages: Vec<Message>` grows unbounded**~~ — **FIXED (2026-03-11)**
   - Added sliding window pruning in `agent/state.rs` (default 80 messages, ~30 turns)
   - First message (original task) pinned, synthetic summary injected at position 1
   - Preserves API alternation constraint (user/assistant)
   - `pruned_count` tracks total dropped messages for diagnostics

2. **`thread.messages.clone()` on every LLM call** (agent/handler.rs:300)
   - Full conversation history cloned per API request
   - Now bounded by the sliding window (max 80 messages), so cost is capped
   - Changing `complete_with_tools` to `&[Message]` would require lifetime annotations
     on `MessagesRequest` — deferred since the clone is now bounded

3. **`journal.sweep()` may not be called automatically** (kernel/journal.rs:163)
   - Retention policy system is well-designed but needs periodic invocation
   - Hook into per-turn checkpoint or a background timer

4. **No streaming for LLM responses** (llm/client.rs)
   - Full JSON buffered before parsing
   - Acceptable for current output sizes, but worth adding for large responses

### Claude Code anti-patterns to avoid
- Unbounded `Map`/`HashMap` caches without eviction (Claude Code has `systemPromptSectionCache`, `planSlugCache`)
- Asymmetric event listener registration (67 addEventListener vs 30 removeEventListener)
- Immutable array copy on every state update (`[...old, new]` pattern)
- Relying on GC for memory management of long-lived objects

### Reference
Full analysis details: `~/.claude/projects/C--Users-Daniel--local-bin/memory/claude-code-analysis.md`
