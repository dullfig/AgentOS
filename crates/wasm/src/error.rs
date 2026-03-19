//! Error types for the WASM tool runtime.

#[derive(Debug, thiserror::Error)]
pub enum WasmError {
    #[error("engine creation failed: {0}")]
    EngineCreation(String),
    #[error("component compilation failed: {0}")]
    Compilation(String),
    #[error("instantiation failed: {0}")]
    Instantiation(String),
    #[error("metadata extraction failed: {0}")]
    Metadata(String),
    #[error("tool execution failed: {0}")]
    Execution(String),
    #[error("capability error: {0}")]
    Capability(String),
}
