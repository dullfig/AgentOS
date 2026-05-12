//! WasmSession — long-lived component instance for stateful workflows.
//!
//! `WasmToolPeer` creates a fresh `Store` per `handle()` call: complete
//! isolation between tool invocations, zero state leakage. That's the
//! right shape for AgentOS's stateless tool model (request_xml in,
//! response_xml out, done).
//!
//! Memex's ingestion drivers are a different shape: `init(config)` →
//! `next_chunk()` → `next_chunk()` → … → drop. The iterator's state
//! needs to persist across calls. Same wasmtime component machinery,
//! different lifecycle.
//!
//! `WasmSession` is the abstraction for the long-lived case. The
//! consumer holds the session for the duration of the workflow,
//! passes `&mut session.store` to its `bindgen!`-generated bindings,
//! and drops the session when done. WASI context (filesystem grants,
//! env vars, stdio) is set at session creation and persists for the
//! session's lifetime.
//!
//! This module is the foundation memex builds its `IngestionDriverPeer`
//! on. AgentOS's `WasmToolPeer` also uses it internally (via
//! `instantiate_session` + immediate drop after `handle()`) so both
//! lifecycles share one code path.

use wasmtime::component::Instance;
use wasmtime::Store;

use crate::capabilities::{build_wasi_ctx, WasmCapabilities};
use crate::error::WasmError;
use crate::runtime::{ToolState, WasmComponent, WasmRuntime};

/// A long-lived WASM component instance. State persists across
/// calls to exported functions until the session is dropped.
///
/// The `store` and `instance` fields are `pub` so consumers can pass
/// them to `bindgen!`-generated typed bindings:
///
/// ```ignore
/// let session = component.instantiate_session(&runtime, &caps)?;
/// let bindings = my_bindings::IngestionDriver::new(&mut session.store, &session.instance)?;
/// bindings.call_init(&mut session.store, &config)?;
/// while let Some(chunk) = bindings.call_next_chunk(&mut session.store)? {
///     handle(chunk);
/// }
/// // session drops; Store + Instance cleaned up
/// ```
///
/// For stateless one-shot calls, prefer `WasmToolPeer` (which uses
/// a session internally + drops it immediately).
pub struct WasmSession {
    /// wasmtime Store carrying `ToolState` (WasiCtx + ResourceTable +
    /// tool-specific state). Public so consumers can pass `&mut` to
    /// bindgen calls.
    pub store: Store<ToolState>,
    /// The instantiated component instance. Consumers wrap this in
    /// their typed bindings.
    pub instance: Instance,
}

impl WasmComponent {
    /// Instantiate this component with a long-lived `Store`, returning
    /// a `WasmSession` the caller owns.
    ///
    /// Reuses `WasmRuntime::make_store_and_linker` (which sets up the
    /// linker with WASI imports) so behavior matches what
    /// `execute_wasm_tool` does for one-shot invocations.
    ///
    /// `capabilities` determines the WASI grants — filesystem mounts,
    /// env vars, stdio inheritance. An all-empty `WasmCapabilities`
    /// produces a minimal ToolState (no WASI ctx); any non-empty
    /// grant produces a full ToolState with WasiCtx built from the
    /// grant table. Same shape as `execute_wasm_tool`.
    pub fn instantiate_session(
        &self,
        runtime: &WasmRuntime,
        capabilities: &WasmCapabilities,
    ) -> Result<WasmSession, WasmError> {
        let state = if capabilities.filesystem.is_empty()
            && capabilities.env_vars.is_empty()
            && !capabilities.stdio
        {
            ToolState::minimal()
        } else {
            ToolState::with_ctx(build_wasi_ctx(capabilities)?)
        };

        let (mut store, linker) = runtime.make_store_and_linker(state)?;
        let instance = linker
            .instantiate(&mut store, &self.component)
            .map_err(|e| WasmError::Instantiation(e.to_string()))?;

        Ok(WasmSession { store, instance })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use wasmtime::component::Val;

    fn load_echo() -> (Arc<WasmRuntime>, WasmComponent) {
        let runtime = Arc::new(WasmRuntime::new().unwrap());
        let bytes = std::fs::read(
            crate::workspace_root()
                .join("tests")
                .join("fixtures")
                .join("echo.wasm"),
        )
        .unwrap();
        let component = runtime.load_component(&bytes).unwrap();
        (runtime, component)
    }

    #[test]
    fn session_instantiates_with_minimal_capabilities() {
        let (runtime, component) = load_echo();
        let session = component
            .instantiate_session(&runtime, &WasmCapabilities::default())
            .expect("session creation should succeed with empty capabilities");
        // Session is usable — instance is non-null and store is mutable.
        let _ = &session.instance;
        let _ = &session.store;
    }

    #[test]
    fn session_supports_repeated_export_calls() {
        // The whole point of a session: state lives across calls.
        // The echo tool is stateless so we can't observe accumulated
        // state, but we can prove the instance survives multiple
        // `handle` invocations through one session — which is the
        // lifecycle memex needs.
        //
        // wasmtime requires `Func::post_return` between calls when
        // using the raw `Func::call` API. Memex consumers will use
        // `bindgen!`-generated wrappers that call `post_return`
        // automatically; here we do it manually since we're not
        // using bindings.
        let (runtime, component) = load_echo();
        let mut session = component
            .instantiate_session(&runtime, &WasmCapabilities::default())
            .unwrap();

        let handle_fn = session
            .instance
            .get_func(&mut session.store, "handle")
            .expect("echo exports 'handle'");

        for i in 0..3 {
            let xml = format!("<EchoRequest><message>call {i}</message></EchoRequest>");
            let args = [Val::String(xml.clone().into())];
            let mut results = [Val::Bool(false)];
            handle_fn
                .call(&mut session.store, &args, &mut results)
                .unwrap_or_else(|e| panic!("call {i}: {e}"));
            handle_fn
                .post_return(&mut session.store)
                .unwrap_or_else(|e| panic!("post_return {i}: {e}"));
        }
    }
}
