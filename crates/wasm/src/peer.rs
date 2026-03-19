//! WasmToolPeer — bridges WASM components into the pipeline as Handlers.
//!
//! Each WasmToolPeer wraps a compiled Component. On handle(), it spawns
//! a blocking task with a fresh Store (complete isolation per invocation),
//! instantiates the component, and calls handle(xml).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use wasmtime::component::{Linker, Val};
use wasmtime::Store;

use super::capabilities::WasmCapabilities;
use super::error::WasmError;
use super::runtime::{ToolMetadata, ToolState, WasmComponent, WasmRuntime};
use agentos_events::{ToolPeer, ToolResponse};

/// Timeout for WASM tool execution (5 minutes).
/// Prevents hung tools from blocking the agent thread indefinitely.
const WASM_TOOL_TIMEOUT: Duration = Duration::from_secs(300);

/// A WASM tool component exposed as a pipeline Handler + ToolPeer.
///
/// One Store per invocation — complete isolation, no state leakage.
/// Component is compiled once (expensive), instantiated per-call (cheap).
pub struct WasmToolPeer {
    runtime: Arc<WasmRuntime>,
    component: Arc<WasmComponent>,
    metadata: ToolMetadata,
    capabilities: WasmCapabilities,
}

impl WasmToolPeer {
    /// Create a new WasmToolPeer from a loaded component (no capabilities).
    pub fn new(runtime: Arc<WasmRuntime>, component: Arc<WasmComponent>) -> Self {
        let metadata = component.metadata.clone();
        Self {
            runtime,
            component,
            metadata,
            capabilities: WasmCapabilities::default(),
        }
    }

    /// Create a new WasmToolPeer with explicit capability grants.
    pub fn with_capabilities(
        runtime: Arc<WasmRuntime>,
        component: Arc<WasmComponent>,
        capabilities: WasmCapabilities,
    ) -> Self {
        let metadata = component.metadata.clone();
        Self {
            runtime,
            component,
            metadata,
            capabilities,
        }
    }
}

#[async_trait]
impl Handler for WasmToolPeer {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml = String::from_utf8_lossy(&payload.xml).to_string();
        let runtime = self.runtime.clone();
        let component = self.component.clone();
        let caps = self.capabilities.clone();

        // Bridge async pipeline → sync WASM via spawn_blocking.
        // Fresh Store per invocation = complete isolation.
        // Wrapped in a timeout to prevent hung tools from blocking the agent.
        let task = tokio::task::spawn_blocking(move || {
            execute_wasm_tool(&runtime, &component, &xml, &caps)
        });

        let result = match tokio::time::timeout(WASM_TOOL_TIMEOUT, task).await {
            Ok(join_result) => join_result
                .map_err(|e| PipelineError::Handler(format!("WASM task panicked: {e}")))?
                .map_err(|e: WasmError| PipelineError::Handler(format!("WASM: {e}")))?,
            Err(_) => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(
                        "WASM tool execution timed out after 5 minutes",
                    ),
                });
            }
        };

        let response = if result.0 {
            ToolResponse::ok(&result.1)
        } else {
            ToolResponse::err(&result.1)
        };

        Ok(HandlerResponse::Reply {
            payload_xml: response,
        })
    }
}

#[async_trait]
impl ToolPeer for WasmToolPeer {
    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn wit(&self) -> &str {
        // WASM tools get schemas from component metadata, not WIT text.
        // register_tool() detects the empty string and falls back to
        // the WasmToolRegistry path for ToolDefinitions.
        ""
    }
}

/// Execute a WASM tool call synchronously (called inside spawn_blocking).
///
/// Creates a fresh Store + WASI context, instantiates the component,
/// and calls handle(xml). Returns (success, payload).
/// Capabilities determine the WASI grants for this invocation.
fn execute_wasm_tool(
    runtime: &WasmRuntime,
    component: &WasmComponent,
    xml: &str,
    capabilities: &WasmCapabilities,
) -> Result<(bool, String), WasmError> {
    let state = if capabilities.filesystem.is_empty()
        && capabilities.env_vars.is_empty()
        && !capabilities.stdio
    {
        ToolState::minimal()
    } else {
        ToolState::with_ctx(super::capabilities::build_wasi_ctx(capabilities)?)
    };
    let mut store = Store::new(runtime.engine(), state);

    let mut linker = Linker::new(runtime.engine());
    wasmtime_wasi::p2::add_to_linker_sync(&mut linker)
        .map_err(|e| WasmError::Instantiation(format!("WASI link failed: {e}")))?;

    let instance = linker
        .instantiate(&mut store, &component.component)
        .map_err(|e| WasmError::Instantiation(e.to_string()))?;

    let handle_fn = instance
        .get_func(&mut store, "handle")
        .ok_or_else(|| WasmError::Execution("export 'handle' not found".into()))?;

    let args = [Val::String(xml.into())];
    let mut results = [Val::Bool(false)]; // single record result
    handle_fn
        .call(&mut store, &args, &mut results)
        .map_err(|e| WasmError::Execution(format!("handle call failed: {e}")))?;

    // Extract fields from the tool-result record
    match &results[0] {
        Val::Record(fields) => {
            let success = fields
                .iter()
                .find(|(k, _)| k == "success")
                .and_then(|(_, v)| match v {
                    Val::Bool(b) => Some(*b),
                    _ => None,
                })
                .unwrap_or(false);

            let payload = fields
                .iter()
                .find(|(k, _)| k == "payload")
                .and_then(|(_, v)| match v {
                    Val::String(s) => Some(s.to_string()),
                    _ => None,
                })
                .unwrap_or_default();

            Ok((success, payload))
        }
        other => Err(WasmError::Execution(format!(
            "expected record from handle, got: {:?}",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_echo_peer() -> (Arc<WasmRuntime>, WasmToolPeer) {
        let runtime = Arc::new(WasmRuntime::new().unwrap());
        let bytes = std::fs::read(
            crate::workspace_root()
                .join("tests")
                .join("fixtures")
                .join("echo.wasm"),
        )
        .unwrap();
        let component = Arc::new(runtime.load_component(&bytes).unwrap());
        let peer = WasmToolPeer::new(runtime.clone(), component);
        (runtime, peer)
    }

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "echo".into(),
        }
    }

    #[test]
    fn peer_creation() {
        let (_rt, peer) = load_echo_peer();
        assert_eq!(peer.name(), "echo");
    }

    #[test]
    fn peer_name() {
        let (_rt, peer) = load_echo_peer();
        assert_eq!(peer.name(), "echo");
    }

    #[test]
    fn peer_wit_empty_for_wasm() {
        let (_rt, peer) = load_echo_peer();
        // WASM tools return empty WIT — schemas come from component metadata
        assert!(peer.wit().is_empty());
    }

    #[tokio::test]
    async fn handle_echo_request() {
        let (_rt, peer) = load_echo_peer();
        let payload = ValidatedPayload {
            xml: b"<EchoRequest><message>hello world</message></EchoRequest>".to_vec(),
            tag: "EchoRequest".into(),
        };
        let result = peer.handle(payload, make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("echo: hello world"));
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn handle_returns_reply() {
        let (_rt, peer) = load_echo_peer();
        let payload = ValidatedPayload {
            xml: b"<EchoRequest><message>test</message></EchoRequest>".to_vec(),
            tag: "EchoRequest".into(),
        };
        let result = peer.handle(payload, make_ctx()).await.unwrap();
        assert!(matches!(result, HandlerResponse::Reply { .. }));
    }

    #[tokio::test]
    async fn handle_success_wraps_tool_response() {
        let (_rt, peer) = load_echo_peer();
        let payload = ValidatedPayload {
            xml: b"<EchoRequest><message>hi</message></EchoRequest>".to_vec(),
            tag: "EchoRequest".into(),
        };
        let result = peer.handle(payload, make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("<success>true</success>"));
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn handle_error_wraps_tool_response() {
        // Echo tool returns success for everything, but we test the wrapper
        // by checking that the ToolResponse envelope is correct
        let (_rt, peer) = load_echo_peer();
        let payload = ValidatedPayload {
            xml: b"<EchoRequest><message>test</message></EchoRequest>".to_vec(),
            tag: "EchoRequest".into(),
        };
        let result = peer.handle(payload, make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                // The echo tool returns success, so we have <result>
                assert!(xml.contains("<result>"));
                assert!(xml.contains("</result>"));
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn handle_malformed_xml() {
        let (_rt, peer) = load_echo_peer();
        let payload = ValidatedPayload {
            xml: b"not xml at all garbage".to_vec(),
            tag: "EchoRequest".into(),
        };
        // Should not panic — echo tool handles missing <message> gracefully
        let result = peer.handle(payload, make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                // Should still get a valid ToolResponse back
                assert!(xml.contains("<success>true</success>") || xml.contains("<success>false</success>"));
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn handle_concurrent_calls() {
        let (_rt, peer) = load_echo_peer();
        // Wrap in Arc for shared access
        let peer = Arc::new(peer);

        let peer1 = peer.clone();
        let peer2 = peer.clone();

        let t1 = tokio::spawn(async move {
            let payload = ValidatedPayload {
                xml: b"<EchoRequest><message>first</message></EchoRequest>".to_vec(),
                tag: "EchoRequest".into(),
            };
            peer1.handle(payload, HandlerContext {
                thread_id: "t1".into(),
                from: "agent".into(),
                own_name: "echo".into(),
            }).await
        });

        let t2 = tokio::spawn(async move {
            let payload = ValidatedPayload {
                xml: b"<EchoRequest><message>second</message></EchoRequest>".to_vec(),
                tag: "EchoRequest".into(),
            };
            peer2.handle(payload, HandlerContext {
                thread_id: "t2".into(),
                from: "agent".into(),
                own_name: "echo".into(),
            }).await
        });

        let (r1, r2) = tokio::join!(t1, t2);
        let r1 = r1.unwrap().unwrap();
        let r2 = r2.unwrap().unwrap();

        match r1 {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("echo: first"));
            }
            _ => panic!("expected Reply"),
        }
        match r2 {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("echo: second"));
            }
            _ => panic!("expected Reply"),
        }
    }

    // ── Python WASM component tests ──

    fn load_echo_py_peer() -> (Arc<WasmRuntime>, WasmToolPeer) {
        let runtime = Arc::new(WasmRuntime::new().unwrap());
        let path = crate::workspace_root()
            .join("tests")
            .join("fixtures")
            .join("echo-py.wasm");
        let component = Arc::new(runtime.load_component_from_path(&path).unwrap());
        let peer = WasmToolPeer::new(runtime.clone(), component);
        (runtime, peer)
    }

    #[test]
    fn python_peer_name() {
        let (_rt, peer) = load_echo_py_peer();
        assert_eq!(peer.name(), "echo-py");
    }

    #[tokio::test]
    async fn python_handle_echo() {
        let (_rt, peer) = load_echo_py_peer();
        let payload = ValidatedPayload {
            xml: b"<EchoRequest><message>hello from python</message></EchoRequest>".to_vec(),
            tag: "EchoRequest".into(),
        };
        let result = peer.handle(payload, make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("echo-py: hello from python"), "got: {xml}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn per_invocation_isolation() {
        // Two sequential calls should each get fresh state
        let (_rt, peer) = load_echo_peer();

        for msg in &["alpha", "beta"] {
            let payload = ValidatedPayload {
                xml: format!("<EchoRequest><message>{msg}</message></EchoRequest>").into_bytes(),
                tag: "EchoRequest".into(),
            };
            let result = peer.handle(payload, make_ctx()).await.unwrap();
            match result {
                HandlerResponse::Reply { payload_xml } => {
                    let xml = String::from_utf8(payload_xml).unwrap();
                    assert!(
                        xml.contains(&format!("echo: {msg}")),
                        "expected 'echo: {msg}' in response, got: {xml}"
                    );
                }
                _ => panic!("expected Reply"),
            }
        }
    }
}
