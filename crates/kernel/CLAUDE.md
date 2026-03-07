# agentos-kernel

Durable state engine for AgentOS. Three stores, one WAL, atomic cross-store ops.

## Architecture

- **WAL** (`wal.rs`) — Append-only, CRC-checked write-ahead log. All mutations flow through here first. Supports atomic batches.
- **ThreadTable** (`thread_table.rs`) — Dot-separated call chains (system.org.handler.subhandler). UUID-indexed. Extends, prunes, inherits security profiles.
- **ContextStore** (`context_store.rs`) — Per-thread named segments (VMM metaphor). Active/Shelved/Folded status. Page-in/page-out via relevance scoring. Fold store for compressed child contexts.
- **Journal** (`journal.rs`) — Message audit trail. Dispatched/Delivered/Failed lifecycle. Retention policies (Forever, PruneOnDelivery, RetainDays).
- **Kernel** (`lib.rs`) — Wraps all three stores. Provides atomic dispatch, prune, and fold operations. WAL replay on crash recovery.

## Key Invariants

- WAL written before state mutation (crash recovery guarantee)
- Atomic batches: all-or-nothing for cross-store ops
- CRC32 integrity on every WAL entry
- Profile inheritance: child threads inherit parent's security profile

## Public API

- `Kernel::open(data_dir)` — Open/create kernel, replay WAL
- `Kernel::initialize_root(organism, profile)` — Boot root thread
- `Kernel::dispatch_message(from, to, thread_id, msg_id)` — Atomic extend+allocate+journal
- `Kernel::prune_thread(thread_id)` — Atomic prune+release+deliver
- `Kernel::fold_thread(thread_id, summary)` — Compress child into parent context
- Accessors: `threads()`, `contexts()`, `journal()`, `wal()`, `data_dir()`

## Testing

66 tests covering: lifecycle, crash recovery, WAL replay, retention sweep, profile propagation.
