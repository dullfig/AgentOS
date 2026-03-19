//! PythonToolPeer — bridges pure Python tools through the shared Python runtime.
//!
//! The python-runtime.wasm component embeds CPython and exports:
//!   get-metadata(source: string) -> tool-metadata
//!   handle(source: string, request-xml: string) -> tool-result
//!
//! Each PythonToolPeer holds the tool's Python source code. On handle(),
//! it passes the source + request XML to the runtime. Fresh Store per
//! invocation = complete isolation.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use wasmtime::component::{Component, Val};

use super::error::WasmError;
use super::runtime::{ToolMetadata, ToolState, WasmRuntime};
use agentos_events::{ToolPeer, ToolResponse};

/// Timeout for Python tool execution (5 minutes).
const PYTHON_TOOL_TIMEOUT: Duration = Duration::from_secs(300);

/// The shared Python runtime component (loaded once, 42MB).
pub struct PythonRuntime {
    runtime: Arc<WasmRuntime>,
    component: Component,
}

impl PythonRuntime {
    /// Load the python-runtime.wasm component from a file path.
    pub fn load(runtime: Arc<WasmRuntime>, path: &Path) -> Result<Self, WasmError> {
        let component = Component::from_file(runtime.engine(), path)
            .map_err(|e| WasmError::Compilation(format!("python-runtime: {e}")))?;
        Ok(Self { runtime, component })
    }

    /// Extract metadata for a Python tool by calling get-metadata(source).
    pub fn get_metadata(&self, source: &str) -> Result<ToolMetadata, WasmError> {
        let (mut store, linker) = self.runtime.make_store_and_linker(ToolState::minimal())?;

        let instance = linker
            .instantiate(&mut store, &self.component)
            .map_err(|e| WasmError::Instantiation(e.to_string()))?;

        let get_metadata_fn = instance
            .get_func(&mut store, "get-metadata")
            .ok_or_else(|| WasmError::Metadata("export 'get-metadata' not found".into()))?;

        let args = [Val::String(source.into())];
        let mut results = [Val::Bool(false)];
        get_metadata_fn
            .call(&mut store, &args, &mut results)
            .map_err(|e| WasmError::Metadata(format!("get-metadata call failed: {e}")))?;

        parse_metadata_record(&results[0])
    }

    /// Execute a Python tool: handle(source, request_xml).
    fn execute(
        &self,
        source: &str,
        request_xml: &str,
    ) -> Result<(bool, String), WasmError> {
        let (mut store, linker) = self.runtime.make_store_and_linker(ToolState::minimal())?;

        let instance = linker
            .instantiate(&mut store, &self.component)
            .map_err(|e| WasmError::Instantiation(e.to_string()))?;

        let handle_fn = instance
            .get_func(&mut store, "handle")
            .ok_or_else(|| WasmError::Execution("export 'handle' not found".into()))?;

        let args = [Val::String(source.into()), Val::String(request_xml.into())];
        let mut results = [Val::Bool(false)];
        handle_fn
            .call(&mut store, &args, &mut results)
            .map_err(|e| WasmError::Execution(format!("handle call failed: {e}")))?;

        parse_tool_result_record(&results[0])
    }
}

/// A Python tool backed by the shared runtime + a source file.
pub struct PythonToolPeer {
    py_runtime: Arc<PythonRuntime>,
    source: String,
    metadata: ToolMetadata,
}

impl PythonToolPeer {
    /// Create a PythonToolPeer by loading source and extracting metadata.
    pub fn new(
        py_runtime: Arc<PythonRuntime>,
        source: String,
    ) -> Result<Self, WasmError> {
        let metadata = py_runtime.get_metadata(&source)?;
        Ok(Self {
            py_runtime,
            source,
            metadata,
        })
    }

    /// Create from a .py file path.
    pub fn from_file(
        py_runtime: Arc<PythonRuntime>,
        path: &Path,
    ) -> Result<Self, WasmError> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| WasmError::Execution(format!("failed to read {}: {e}", path.display())))?;
        Self::new(py_runtime, source)
    }

    /// Get the cached metadata.
    pub fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }
}

#[async_trait]
impl Handler for PythonToolPeer {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml = String::from_utf8_lossy(&payload.xml).to_string();
        let py_runtime = self.py_runtime.clone();
        let source = self.source.clone();

        let task = tokio::task::spawn_blocking(move || {
            py_runtime.execute(&source, &xml)
        });

        let result = match tokio::time::timeout(PYTHON_TOOL_TIMEOUT, task).await {
            Ok(join_result) => join_result
                .map_err(|e| PipelineError::Handler(format!("Python task panicked: {e}")))?
                .map_err(|e| PipelineError::Handler(format!("Python: {e}")))?,
            Err(_) => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(
                        "Python tool execution timed out after 5 minutes",
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
impl ToolPeer for PythonToolPeer {
    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn wit(&self) -> &str {
        // Python tools use metadata from the runtime, not WIT text
        ""
    }
}

/// Parse a tool-metadata record from WASM Val.
fn parse_metadata_record(val: &Val) -> Result<ToolMetadata, WasmError> {
    let fields = match val {
        Val::Record(fields) => fields,
        other => {
            return Err(WasmError::Metadata(format!(
                "expected record, got: {:?}",
                other
            )))
        }
    };

    fn field_string(fields: &[(String, Val)], name: &str) -> Result<String, WasmError> {
        for (k, v) in fields {
            if k == name {
                return match v {
                    Val::String(s) => Ok(s.to_string()),
                    other => Err(WasmError::Metadata(format!(
                        "field '{name}': expected string, got {:?}",
                        other
                    ))),
                };
            }
        }
        Err(WasmError::Metadata(format!("missing field '{name}'")))
    }

    Ok(ToolMetadata {
        name: field_string(fields, "name")?,
        description: field_string(fields, "description")?,
        semantic_description: field_string(fields, "semantic-description")?,
        request_tag: field_string(fields, "request-tag")?,
        request_schema: field_string(fields, "request-schema")?,
        response_schema: field_string(fields, "response-schema")?,
        input_json_schema: field_string(fields, "input-json-schema")?,
    })
}

/// Parse a tool-result record from WASM Val.
fn parse_tool_result_record(val: &Val) -> Result<(bool, String), WasmError> {
    match val {
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

    fn runtime_wasm_path() -> std::path::PathBuf {
        crate::workspace_root()
            .join("tests")
            .join("fixtures")
            .join("python-runtime.wasm")
    }

    fn sample_tool_source() -> String {
        std::fs::read_to_string(
            crate::workspace_root()
                .join("tools")
                .join("samples")
                .join("echo_tool.py"),
        )
        .expect("echo_tool.py sample not found")
    }

    fn load_python_runtime() -> Arc<PythonRuntime> {
        let wasm_runtime = Arc::new(WasmRuntime::new().unwrap());
        Arc::new(PythonRuntime::load(wasm_runtime, &runtime_wasm_path()).unwrap())
    }

    #[test]
    fn load_runtime_component() {
        let wasm_runtime = Arc::new(WasmRuntime::new().unwrap());
        let result = PythonRuntime::load(wasm_runtime, &runtime_wasm_path());
        assert!(result.is_ok(), "failed to load python-runtime: {:?}", result.err());
    }

    #[test]
    fn extract_metadata_from_source() {
        let py_rt = load_python_runtime();
        let source = sample_tool_source();
        let metadata = py_rt.get_metadata(&source).unwrap();
        assert_eq!(metadata.name, "echo-py");
        assert_eq!(metadata.request_tag, "EchoRequest");
        assert!(!metadata.description.is_empty());
    }

    #[test]
    fn execute_tool_from_source() {
        let py_rt = load_python_runtime();
        let source = sample_tool_source();
        let (success, payload) = py_rt
            .execute(&source, "<EchoRequest><message>hello runtime</message></EchoRequest>")
            .unwrap();
        assert!(success);
        assert!(payload.contains("echo-py: hello runtime"), "got: {payload}");
    }

    #[test]
    fn create_python_tool_peer() {
        let py_rt = load_python_runtime();
        let source = sample_tool_source();
        let peer = PythonToolPeer::new(py_rt, source).unwrap();
        assert_eq!(peer.name(), "echo-py");
        assert_eq!(peer.metadata().request_tag, "EchoRequest");
    }

    #[tokio::test]
    async fn python_peer_handle() {
        let py_rt = load_python_runtime();
        let source = sample_tool_source();
        let peer = PythonToolPeer::new(py_rt, source).unwrap();

        let payload = ValidatedPayload {
            xml: b"<EchoRequest><message>peer test</message></EchoRequest>".to_vec(),
            tag: "EchoRequest".into(),
        };
        let ctx = HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "echo-py".into(),
        };
        let result = peer.handle(payload, ctx).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("echo-py: peer test"), "got: {xml}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[test]
    fn bad_source_returns_error_metadata() {
        let py_rt = load_python_runtime();
        let metadata = py_rt.get_metadata("this is not valid python !!!").unwrap();
        assert!(metadata.description.contains("Error"), "got: {}", metadata.description);
    }

    #[test]
    fn missing_handle_returns_error() {
        let py_rt = load_python_runtime();
        let source = "def get_metadata(): pass  # no handle function";
        let (success, payload) = py_rt
            .execute(source, "<Req></Req>")
            .unwrap();
        assert!(!success);
        assert!(payload.contains("no handle"), "got: {payload}");
    }

    #[tokio::test]
    async fn python_peer_from_file() {
        let py_rt = load_python_runtime();
        let path = crate::workspace_root()
            .join("tools")
            .join("samples")
            .join("echo_tool.py");
        let peer = PythonToolPeer::from_file(py_rt, &path).unwrap();
        assert_eq!(peer.name(), "echo-py");
    }

    // ── @tool decorator tests ──

    fn decorated_tool_source() -> String {
        std::fs::read_to_string(
            crate::workspace_root()
                .join("tools")
                .join("samples")
                .join("echo_decorated.py"),
        )
        .expect("echo_decorated.py sample not found")
    }

    #[test]
    fn decorated_tool_metadata() {
        let py_rt = load_python_runtime();
        let source = decorated_tool_source();
        let metadata = py_rt.get_metadata(&source).unwrap();
        assert_eq!(metadata.name, "echo-py");
        assert_eq!(metadata.request_tag, "EchoPyRequest");
        assert!(!metadata.description.is_empty());
        // JSON schema should have the 'message' and 'times' fields
        let schema: serde_json::Value =
            serde_json::from_str(&metadata.input_json_schema).unwrap();
        assert!(schema["properties"]["message"].is_object());
        assert!(schema["properties"]["times"].is_object());
        assert_eq!(schema["properties"]["times"]["default"], 1);
    }

    #[test]
    fn decorated_tool_handle() {
        let py_rt = load_python_runtime();
        let source = decorated_tool_source();
        let (success, payload) = py_rt
            .execute(&source, "<EchoPyRequest><message>decorated</message></EchoPyRequest>")
            .unwrap();
        assert!(success, "got error: {payload}");
        assert!(payload.contains("echo-py: decorated"), "got: {payload}");
    }

    #[test]
    fn decorated_tool_with_optional_field() {
        let py_rt = load_python_runtime();
        let source = decorated_tool_source();
        let (success, payload) = py_rt
            .execute(
                &source,
                "<EchoPyRequest><message>hi</message><times>3</times></EchoPyRequest>",
            )
            .unwrap();
        assert!(success, "got error: {payload}");
        // "hi" repeated 3 times
        assert_eq!(payload.matches("hi").count(), 3, "got: {payload}");
    }

    #[test]
    fn decorated_tool_missing_required() {
        let py_rt = load_python_runtime();
        let source = decorated_tool_source();
        let (success, payload) = py_rt
            .execute(&source, "<EchoPyRequest></EchoPyRequest>")
            .unwrap();
        assert!(!success);
        assert!(payload.contains("missing required"), "got: {payload}");
    }

    #[tokio::test]
    async fn decorated_tool_as_peer() {
        let py_rt = load_python_runtime();
        let source = decorated_tool_source();
        let peer = PythonToolPeer::new(py_rt, source).unwrap();
        assert_eq!(peer.name(), "echo-py");

        let payload = ValidatedPayload {
            xml: b"<EchoPyRequest><message>peer decorator</message></EchoPyRequest>".to_vec(),
            tag: "EchoPyRequest".into(),
        };
        let ctx = HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "echo-py".into(),
        };
        let result = peer.handle(payload, ctx).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("echo-py: peer decorator"), "got: {xml}");
            }
            _ => panic!("expected Reply"),
        }
    }
}
