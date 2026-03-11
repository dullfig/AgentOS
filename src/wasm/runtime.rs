//! WASM Component Runtime — loads and executes WASM tool components.
//!
//! Uses wasmtime's component model. Each tool is a WASM component that
//! exports `get-metadata()` and `handle()` per the WIT contract.
//! Components are compiled once (expensive), instantiated per-call (cheap).

use std::path::Path;

use wasmtime::component::{Component, Linker, ResourceTable, Val};
use wasmtime::{Engine, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

use super::error::WasmError;

/// Metadata extracted from a WASM tool component.
#[derive(Debug, Clone)]
pub struct ToolMetadata {
    pub name: String,
    pub description: String,
    pub semantic_description: String,
    pub request_tag: String,
    pub request_schema: String,
    pub response_schema: String,
    pub input_json_schema: String,
}

/// A compiled WASM tool component with cached metadata.
pub struct WasmComponent {
    pub component: Component,
    pub metadata: ToolMetadata,
}

impl std::fmt::Debug for WasmComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmComponent")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

/// Store data for WASM tool execution — implements WasiView.
pub(crate) struct ToolState {
    ctx: WasiCtx,
    table: ResourceTable,
}

impl WasiView for ToolState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}

impl ToolState {
    /// Create a minimal tool state for metadata extraction (no capabilities).
    pub fn minimal() -> Self {
        Self {
            ctx: WasiCtxBuilder::new().build(),
            table: ResourceTable::new(),
        }
    }

    /// Create a tool state from a pre-built WasiCtx.
    pub fn with_ctx(ctx: WasiCtx) -> Self {
        Self {
            ctx,
            table: ResourceTable::new(),
        }
    }
}

/// The WASM runtime engine — shared across all tool components.
pub struct WasmRuntime {
    engine: Engine,
}

impl WasmRuntime {
    /// Create a new WASM runtime with default configuration.
    pub fn new() -> Result<Self, WasmError> {
        let mut config = wasmtime::Config::new();
        config.wasm_component_model(true);
        let engine =
            Engine::new(&config).map_err(|e| WasmError::EngineCreation(e.to_string()))?;
        Ok(Self { engine })
    }

    /// Get a reference to the underlying engine.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Load a WASM component from raw bytes.
    ///
    /// Compiles the component and calls `get_metadata()` once to cache metadata.
    pub fn load_component(&self, bytes: &[u8]) -> Result<WasmComponent, WasmError> {
        let component = Component::new(&self.engine, bytes)
            .map_err(|e| WasmError::Compilation(e.to_string()))?;

        let metadata = self.extract_metadata(&component)?;

        Ok(WasmComponent {
            component,
            metadata,
        })
    }

    /// Load a WASM component from a filesystem path.
    pub fn load_component_from_path(&self, path: &Path) -> Result<WasmComponent, WasmError> {
        let component = Component::from_file(&self.engine, path)
            .map_err(|e| WasmError::Compilation(format!("{}: {e}", path.display())))?;

        let metadata = self.extract_metadata(&component)?;

        Ok(WasmComponent {
            component,
            metadata,
        })
    }

    /// Create a fresh Store with a linker for sync execution.
    pub(crate) fn make_store_and_linker(
        &self,
        state: ToolState,
    ) -> Result<(Store<ToolState>, Linker<ToolState>), WasmError> {
        let store = Store::new(&self.engine, state);
        let mut linker = Linker::new(&self.engine);
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)
            .map_err(|e| WasmError::Instantiation(format!("WASI link failed: {e}")))?;
        Ok((store, linker))
    }

    /// Extract metadata by instantiating the component and calling get_metadata().
    fn extract_metadata(&self, component: &Component) -> Result<ToolMetadata, WasmError> {
        let (mut store, linker) = self.make_store_and_linker(ToolState::minimal())?;

        // Instantiate
        let instance = linker
            .instantiate(&mut store, component)
            .map_err(|e| WasmError::Instantiation(e.to_string()))?;

        // Call get-metadata — returns a single record value
        let get_metadata = instance
            .get_func(&mut store, "get-metadata")
            .ok_or_else(|| WasmError::Metadata("export 'get-metadata' not found".into()))?;

        let mut results = vec![Val::Bool(false)]; // 1 record result
        get_metadata
            .call(&mut store, &[], &mut results)
            .map_err(|e| WasmError::Metadata(format!("get-metadata call failed: {e}")))?;

        // Extract fields from the record
        let fields = match &results[0] {
            Val::Record(fields) => fields,
            other => {
                return Err(WasmError::Metadata(format!(
                    "expected record from get-metadata, got: {:?}",
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_wasm_bytes() -> Vec<u8> {
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("fixtures")
                .join("echo.wasm"),
        )
        .expect("echo.wasm fixture not found — run: cargo build --manifest-path tools/echo-tool/Cargo.toml --target wasm32-wasip2 --release")
    }

    fn echo_wasm_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("echo.wasm")
    }

    #[test]
    fn engine_creation() {
        let runtime = WasmRuntime::new();
        assert!(runtime.is_ok());
    }

    #[test]
    fn load_invalid_bytes_fails() {
        let runtime = WasmRuntime::new().unwrap();
        let result = runtime.load_component(b"garbage bytes not wasm");
        assert!(result.is_err());
        match result.unwrap_err() {
            WasmError::Compilation(_) => {} // expected
            other => panic!("expected Compilation error, got: {other}"),
        }
    }

    #[test]
    fn load_empty_bytes_fails() {
        let runtime = WasmRuntime::new().unwrap();
        let result = runtime.load_component(b"");
        assert!(result.is_err());
    }

    #[test]
    fn load_valid_component() {
        let runtime = WasmRuntime::new().unwrap();
        let bytes = echo_wasm_bytes();
        let result = runtime.load_component(&bytes);
        assert!(result.is_ok(), "load failed: {:?}", result.err());
    }

    #[test]
    fn metadata_extraction() {
        let runtime = WasmRuntime::new().unwrap();
        let wc = runtime.load_component(&echo_wasm_bytes()).unwrap();
        let m = &wc.metadata;
        assert!(!m.name.is_empty());
        assert!(!m.description.is_empty());
        assert!(!m.semantic_description.is_empty());
        assert!(!m.request_tag.is_empty());
        assert!(!m.request_schema.is_empty());
        assert!(!m.response_schema.is_empty());
        assert!(!m.input_json_schema.is_empty());
    }

    #[test]
    fn metadata_name_matches() {
        let runtime = WasmRuntime::new().unwrap();
        let wc = runtime.load_component(&echo_wasm_bytes()).unwrap();
        assert_eq!(wc.metadata.name, "echo");
    }

    #[test]
    fn metadata_has_request_tag() {
        let runtime = WasmRuntime::new().unwrap();
        let wc = runtime.load_component(&echo_wasm_bytes()).unwrap();
        assert_eq!(wc.metadata.request_tag, "EchoRequest");
    }

    #[test]
    fn metadata_has_json_schema() {
        let runtime = WasmRuntime::new().unwrap();
        let wc = runtime.load_component(&echo_wasm_bytes()).unwrap();
        // Must parse as valid JSON
        let schema: serde_json::Value =
            serde_json::from_str(&wc.metadata.input_json_schema).unwrap();
        assert_eq!(schema["type"], "object");
    }

    #[test]
    fn metadata_has_semantic_description() {
        let runtime = WasmRuntime::new().unwrap();
        let wc = runtime.load_component(&echo_wasm_bytes()).unwrap();
        assert!(
            wc.metadata.semantic_description.len() > 20,
            "semantic_description should be verbose"
        );
    }

    #[test]
    fn load_from_path() {
        let runtime = WasmRuntime::new().unwrap();
        let result = runtime.load_component_from_path(&echo_wasm_path());
        assert!(result.is_ok(), "load from path failed: {:?}", result.err());
        assert_eq!(result.unwrap().metadata.name, "echo");
    }

    // ── Python WASM component tests ──

    fn echo_py_wasm_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("echo-py.wasm")
    }

    #[test]
    fn load_python_component() {
        let runtime = WasmRuntime::new().unwrap();
        let result = runtime.load_component_from_path(&echo_py_wasm_path());
        assert!(result.is_ok(), "Python component load failed: {:?}", result.err());
    }

    #[test]
    fn python_metadata_extraction() {
        let runtime = WasmRuntime::new().unwrap();
        let wc = runtime.load_component_from_path(&echo_py_wasm_path()).unwrap();
        let m = &wc.metadata;
        assert_eq!(m.name, "echo-py");
        assert_eq!(m.request_tag, "EchoRequest");
        assert!(!m.description.is_empty());
        assert!(!m.semantic_description.is_empty());
        // JSON schema must parse
        let schema: serde_json::Value =
            serde_json::from_str(&m.input_json_schema).unwrap();
        assert_eq!(schema["type"], "object");
    }
}
