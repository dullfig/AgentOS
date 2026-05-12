# agentos-wit

WIT (WebAssembly Interface Type) parser + schema deriver. Parses WIT
text and turns it into structured `ToolInterface` values, JSON
schemas for LLM tool definitions, and XML payload schemas for
in-pipeline validation.

## Shared-crate status

This crate is **path-importable** by external consumers (memex, future
sibling projects). Per the BYO-WIT pattern in
`project_agentos_topology.md`: AgentOS uses this for its `tool` world;
memex uses it for its `ingestion-driver` world; future apps use it
for their own. The crate provides parsing + schema derivation — it
doesn't bake in any particular WIT world or contract.

Consumers Cargo.toml:

```toml
[dependencies]
agentos-wit = { path = "../agentos/crates/wit" }
```

## Public API

- **`parser::parse_wit(text: &str) -> Result<ToolInterface, _>`** —
  the entry point. Hand it a WIT interface block, get back a
  parsed structure.
- **`ToolInterface`** — name, description, request record fields,
  plus methods:
  - `to_payload_schema()` — for XML payload validation (rust-pipeline)
  - `to_tool_definition()` — for LLM tool calls (JSON schema)
  - `to_codellm_schema(root_tag)` — for constrained decoding
    (code-llm crate)
- **`ToolRecord`**, **`ToolField`**, **`ToolFieldType`** — the
  parsed field/type tree. Useful when a consumer wants to walk the
  schema directly instead of going through the helpers.

## What this crate doesn't do

- It doesn't load WASM components. That's `agentos-wasm`.
- It doesn't run anything. Pure parse + transform.
- It doesn't enforce a particular WIT world name (the parser handles
  any `interface foo { ... }` block; consumers pick their own
  naming).
- It doesn't know about the `world` keyword's full semantics — it's
  scoped to interface records and exported functions. WIT's resource
  types and complex world composition aren't currently handled
  because nothing in AgentOS or memex needs them yet.

## When to extend the crate

If memex or another consumer needs WIT features the parser doesn't
support (resources, streams, futures), the right move is to extend
the parser here rather than fork. The parser intentionally lives in
one place so all consumers benefit from improvements.

## What stays out

- Tool registration machinery — that's the consumer's responsibility.
  AgentOS has `AgentPipelineBuilder::register_tool` which calls
  `parse_wit` and stuffs the result into its tool table; memex will
  have its own registration shape.
- Bindgen — generating Python/Rust bindings from WIT is `componentize-py`'s
  job (or `wit-bindgen` for Rust). This crate consumes WIT text, it
  doesn't generate code.

## Tests

`cargo test -p agentos-wit` — pure unit tests, no fixtures.
