# AgentOS Kernel Extraction

## Decision (2026-04-05)

AgentOS should be split into two layers:
- **agentos-kernel** — the library: call chain dispatch, sandboxed memory, WAL, process isolation
- **agentos** — the TUI platform that uses the kernel

## Why

RingHub (and future products) needs the kernel's multi-tenant isolation model, not the full platform. The call chain's alternate dispatch tables — where swapping the root swaps the entire context — naturally provides per-user memory partitioning. Making the root a userID pulls up the right memory bank, KV store, and sandboxed state for that user.

This mechanism already exists for process isolation. Multi-tenancy is a reframe, not a rebuild.

## Integration Path

```
RingHub (Django) → Donna (FastAPI/WS) → agentos-kernel → Concierge organism
```

- Donna bridges HTTP/WebSocket to the kernel
- Each RingHub user = one dispatch root in the call chain
- Concierge organism template is shared, dispatch context is per-user
- neuralkv-core memory naturally partitions per root (each chain sees its own KV store)

## What the kernel crate exposes

- Call chain dispatch with configurable roots
- Sandboxed filesystem per process/user
- WAL for durable state
- Process lifecycle (spawn, isolate, teardown)

## What stays in the platform

- TUI interface
- Interactive shell
- QCOW2 virtual drive management (unless needed by consumers)
