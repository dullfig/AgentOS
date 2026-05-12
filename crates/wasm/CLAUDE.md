# agentos-wasm

WebAssembly component runtime: load components, instantiate them
with capability-gated WASI, run their exports. Includes a
CPython-in-WASM bridge for Python-language consumers.

## Shared-crate status

This crate is **path-importable** by external consumers (memex, future
sibling projects). Per the BYO-WIT pattern in
`project_agentos_topology.md`: AgentOS uses it for its `tool` world;
memex uses it for its `ingestion-driver` world; each consumer
authors its own WIT, compiles its own WASM components, wraps the
runtime in its own peer abstraction. This crate provides the
component-model machinery — it doesn't bake in any contract.

Consumer Cargo.toml:

```toml
[dependencies]
agentos-wit = { path = "../agentos/crates/wit" }
agentos-wasm = { path = "../agentos/crates/wasm" }
agentos-events = { path = "../agentos/crates/events" }  # for shared type defs
```

## Two lifecycle patterns

### Ephemeral (stateless, one-shot)

For tools that take an input and produce an output, no state between
calls. AgentOS's `tool` world uses this pattern. Each invocation
creates a fresh `Store` + `Instance`; total isolation between calls.

Implemented by `WasmToolPeer` (in this crate). Consumers typically
don't reach for `WasmToolPeer` directly — it's tied to AgentOS's
`request_xml → response_xml` shape. Memex doesn't use it; memex
defines its own peer wrapper.

### Stateful (long-lived session)

For workflows that need state across calls — corpus ingestion
iterators, interactive editors, long-running converters. Memex's
`ingestion-driver` world uses this pattern.

Use `WasmComponent::instantiate_session(&runtime, &capabilities)`
to get a `WasmSession`. Hold the session for the workflow's
lifetime; pass `&mut session.store` to your `bindgen!`-generated
typed bindings. The session drops with its consumer.

```rust
let runtime = WasmRuntime::new()?;
let component = runtime.load_component_from_path(driver_wasm_path)?;
let mut session = component.instantiate_session(&runtime, &capabilities)?;

// memex's bindgen!-generated bindings wrap the session
let bindings = my_bindings::IngestionDriver::new(&mut session.store, &session.instance)?;
bindings.call_init(&mut session.store, &corpus_config)?;
while let Some(chunk) = bindings.call_next_chunk(&mut session.store)? {
    handle(chunk);
}
// session drops; Store + Instance + WASI ctx cleaned up
```

When using raw `Func::call` (no bindgen), you must call
`Func::post_return` between calls — wasmtime requires it. Bindgen-
generated wrappers do this for you.

## Python-in-WASM (CPython component)

`PythonRuntime` + `PythonToolPeer` (in `python_runtime.rs`) provide
a CPython interpreter compiled to a WASM component via Bytecode
Alliance's `componentize-py` toolchain. One shared `python-runtime.wasm`
(~42MB), many `.py` tools per consumer.

**Per consumer**: each project compiles its own `python-runtime.wasm`
against its own WIT contract. componentize-py bakes the WIT world
into the resulting `.wasm`. AgentOS's lives at
`tools/python-runtime/python-runtime.wasm`, compiled against
`agentos/wit/python-runtime.wit`. Memex's will live somewhere under
the memex workspace, compiled against memex's WIT. No artifact
bundling at the crate level — the consumer points the runtime at
its own file via `PythonRuntime::load`.

The `PythonToolPeer` here is shaped for AgentOS's tool model
(`get-metadata` + `handle` exports). Memex defines its own Python
peer wrapper matching its `ingestion-driver` shape.

## Capabilities (WASI grants)

`WasmCapabilities` (in `capabilities.rs`) is the structural sandbox.
A WASM component can only use what its capabilities grant — missing
the filesystem capability isn't a policy check, the WASI import
doesn't exist. That's important for memex's ingestion drivers: they
get read-only access to the user's corpus path, nothing else.

Capability shape:
- `filesystem: Vec<FsGrant>` — host path → guest path, read-only/read-write
- `env_vars: Vec<EnvGrant>` — key → value
- `stdio: bool` — inherit stdin/stdout/stderr (off by default)

Set on the session at instantiation time; immutable thereafter.

## What this crate does NOT provide

- **A peer trait** — each consumer wraps the runtime in its own
  handler abstraction. AgentOS's `ToolPeer` is in `agentos-tools`;
  memex defines its own peer.
- **JSON schema generation for LLM tools** — that's `agentos-wit`.
- **Tool registration** — `AgentPipelineBuilder::register_tool` is in
  AgentOS's pipeline crate; memex will have its own equivalent.
- **A bundled python-runtime.wasm** — consumer compiles + ships its
  own.
- **An opinion about WIT world names** — `WasmRuntime::load_component`
  doesn't read metadata or call any export by name. The runtime gives
  you a component; you decide how to call into it. (Note:
  `extract_metadata` IS in this crate today, but it's AgentOS-specific
  — calls the `get-metadata` export expecting AgentOS's tool record
  shape. Memex doesn't use it; memex calls its own exports via its
  own bindings.)

## When to extend

Adding features to this crate:
- More WASI primitives (sockets, threads, etc.): yes, extend here.
- New lifecycle patterns beyond ephemeral/session: yes, extend here.
- AgentOS-tool-specific or memex-specific protocol shapes: NO. Those
  belong in the consumer's wrapper crate.

## Tests

`cargo test -p agentos-wasm` — slow on first run because each
test loads a WASM component (~42MB for python-runtime). After the
first run, wasmtime caches the compiled artifacts and subsequent
runs are faster.
