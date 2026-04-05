//! Rhai script runtime for cloud providers.
//!
//! Each provider is a `.rhai` script that implements four functions:
//!   fn search(gpu_type, region) -> Array of GpuOffering maps
//!   fn provision(config) -> CloudInstance map
//!   fn status(instance_id) -> CloudInstance map
//!   fn teardown(instance_id) -> bool
//!
//! The runtime exposes `http_post`, `http_get`, and `http_delete` to scripts
//! so they can call provider APIs. Scripts return Rhai Maps that get converted
//! to our Rust types via serde.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use rhai::{Dynamic, Engine, Scope, AST};
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::provider::CloudProvider;
use crate::types::{CloudInstance, GpuOffering, ProvisionRequest};
use crate::CloudError;

/// A cloud provider backed by a Rhai script.
#[allow(dead_code)]
pub struct ScriptedProvider {
    name: String,
    engine: Engine,
    ast: AST,
    /// API key for the provider (injected into script scope).
    api_key: String,
    /// Shared HTTP client for script calls.
    http: reqwest::Client,
    /// Mutex-protected scope for sequential script execution.
    state: Arc<Mutex<Scope<'static>>>,
}

impl std::fmt::Debug for ScriptedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedProvider")
            .field("name", &self.name)
            .finish()
    }
}

impl ScriptedProvider {
    /// Load a provider from a Rhai script file.
    ///
    /// The script must define: `search`, `provision`, `status`, `teardown`.
    /// `api_key` is made available as `API_KEY` in the script scope.
    pub fn load(name: &str, script_path: &Path, api_key: String) -> Result<Self, CloudError> {
        let mut engine = Engine::new();

        // Provider scripts build nested API payloads — raise Rhai's limits.
        engine.set_max_expr_depths(128, 64);

        // Register HTTP helper functions that scripts can call.
        // These are synchronous wrappers — Rhai doesn't support async natively,
        // so we use blocking reqwest inside a tokio::task::block_in_place.
        register_http_functions(&mut engine);

        let ast = engine.compile_file(script_path.to_path_buf()).map_err(|e| {
            CloudError::ScriptError(format!("Failed to compile {}: {e}", script_path.display()))
        })?;

        // Validate that required functions exist
        let required = ["search", "provision", "status", "teardown"];
        for func in &required {
            if !ast.iter_functions().any(|f| f.name == *func) {
                return Err(CloudError::ScriptError(format!(
                    "Provider script '{}' missing required function: {func}",
                    script_path.display()
                )));
            }
        }

        let mut scope = Scope::new();
        scope.push_constant("API_KEY", api_key.clone());

        debug!("Loaded cloud provider script: {name} from {}", script_path.display());

        Ok(Self {
            name: name.to_string(),
            engine,
            ast,
            api_key,
            http: reqwest::Client::new(),
            state: Arc::new(Mutex::new(scope)),
        })
    }

    /// Load a provider from an inline script string (for testing / api-expert generated scripts).
    pub fn from_source(name: &str, source: &str, api_key: String) -> Result<Self, CloudError> {
        let mut engine = Engine::new();
        engine.set_max_expr_depths(128, 64);
        register_http_functions(&mut engine);

        let ast = engine.compile(source).map_err(|e| {
            CloudError::ScriptError(format!("Failed to compile inline script for {name}: {e}"))
        })?;

        let required = ["search", "provision", "status", "teardown"];
        for func in &required {
            if !ast.iter_functions().any(|f| f.name == *func) {
                return Err(CloudError::ScriptError(format!(
                    "Inline provider script '{name}' missing required function: {func}"
                )));
            }
        }

        let mut scope = Scope::new();
        scope.push_constant("API_KEY", api_key.clone());

        Ok(Self {
            name: name.to_string(),
            engine,
            ast,
            api_key,
            http: reqwest::Client::new(),
            state: Arc::new(Mutex::new(scope)),
        })
    }

    /// Call a Rhai function and convert the result to JSON-compatible Dynamic.
    fn call_fn(&self, scope: &mut Scope, fn_name: &str, args: impl rhai::FuncArgs) -> Result<Dynamic, CloudError> {
        self.engine
            .call_fn(scope, &self.ast, fn_name, args)
            .map_err(|e| CloudError::ScriptError(format!("{fn_name}() failed: {e}")))
    }
}

#[async_trait]
impl CloudProvider for ScriptedProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn search(
        &self,
        gpu_type: Option<&str>,
        region: Option<&str>,
    ) -> Result<Vec<GpuOffering>, CloudError> {
        let gpu = gpu_type.unwrap_or("").to_string();
        let reg = region.unwrap_or("").to_string();

        let mut scope = self.state.lock().await;
        let result = self.call_fn(&mut scope, "search", (gpu, reg))?;

        // Script returns an Array of Maps — convert via serde
        let json = rhai::serde::to_dynamic(&result)
            .and_then(|d| rhai::serde::from_dynamic::<Vec<GpuOffering>>(&d))
            .or_else(|_| {
                // Try direct conversion: result is already a Dynamic array
                rhai::serde::from_dynamic::<Vec<GpuOffering>>(&result)
            })
            .map_err(|e| CloudError::ScriptError(format!("search() returned invalid data: {e}")))?;

        Ok(json)
    }

    async fn provision(&self, request: &ProvisionRequest) -> Result<CloudInstance, CloudError> {
        // Convert request to a Rhai map
        let req_json = serde_json::to_value(request)
            .map_err(|e| CloudError::ScriptError(format!("Failed to serialize request: {e}")))?;
        let req_dynamic: Dynamic = serde_json::from_value::<rhai::Dynamic>(req_json)
            .unwrap_or_else(|_| Dynamic::UNIT);

        let mut scope = self.state.lock().await;
        let result = self.call_fn(&mut scope, "provision", (req_dynamic,))?;

        rhai::serde::from_dynamic::<CloudInstance>(&result)
            .map_err(|e| CloudError::ScriptError(format!("provision() returned invalid data: {e}")))
    }

    async fn status(&self, instance_id: &str) -> Result<CloudInstance, CloudError> {
        let id = instance_id.to_string();
        let mut scope = self.state.lock().await;
        let result = self.call_fn(&mut scope, "status", (id,))?;

        rhai::serde::from_dynamic::<CloudInstance>(&result)
            .map_err(|e| CloudError::ScriptError(format!("status() returned invalid data: {e}")))
    }

    async fn teardown(&self, instance_id: &str) -> Result<(), CloudError> {
        let id = instance_id.to_string();
        let mut scope = self.state.lock().await;
        let _result = self.call_fn(&mut scope, "teardown", (id,))?;
        Ok(())
    }
}

/// Register HTTP helper functions into the Rhai engine.
///
/// These let provider scripts make API calls:
///   let resp = http_post(url, headers_map, body_string);
///   let resp = http_get(url, headers_map);
///   let resp = http_delete(url, headers_map);
///
/// Each returns the response body as a string. Scripts parse JSON via Rhai's
/// built-in `parse_json()`.
fn register_http_functions(engine: &mut Engine) {
    // http_post(url: String, headers: Map, body: String) -> String
    engine.register_fn("http_post", |url: String, headers: rhai::Map, body: String| -> Result<String, Box<rhai::EvalAltResult>> {
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                let client = reqwest::Client::new();
                let mut req = client.post(&url);
                for (k, v) in &headers {
                    if let Some(val) = v.clone().into_string().ok() {
                        req = req.header(k.as_str(), val);
                    }
                }
                req = req.header("content-type", "application/json");
                let resp = req.body(body).send().await
                    .map_err(|e| Box::new(rhai::EvalAltResult::from(e.to_string())))?;
                resp.text().await
                    .map_err(|e| Box::new(rhai::EvalAltResult::from(e.to_string())))
            })
        })
    });

    // http_get(url: String, headers: Map) -> String
    engine.register_fn("http_get", |url: String, headers: rhai::Map| -> Result<String, Box<rhai::EvalAltResult>> {
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                let client = reqwest::Client::new();
                let mut req = client.get(&url);
                for (k, v) in &headers {
                    if let Some(val) = v.clone().into_string().ok() {
                        req = req.header(k.as_str(), val);
                    }
                }
                let resp = req.send().await
                    .map_err(|e| Box::new(rhai::EvalAltResult::from(e.to_string())))?;
                resp.text().await
                    .map_err(|e| Box::new(rhai::EvalAltResult::from(e.to_string())))
            })
        })
    });

    // http_delete(url: String, headers: Map) -> String
    engine.register_fn("http_delete", |url: String, headers: rhai::Map| -> Result<String, Box<rhai::EvalAltResult>> {
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                let client = reqwest::Client::new();
                let mut req = client.delete(&url);
                for (k, v) in &headers {
                    if let Some(val) = v.clone().into_string().ok() {
                        req = req.header(k.as_str(), val);
                    }
                }
                let resp = req.send().await
                    .map_err(|e| Box::new(rhai::EvalAltResult::from(e.to_string())))?;
                resp.text().await
                    .map_err(|e| Box::new(rhai::EvalAltResult::from(e.to_string())))
            })
        })
    });
}

/// Registry of loaded providers.
#[derive(Debug, Default)]
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<ScriptedProvider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load all `.rhai` scripts from a directory.
    /// Each script's filename (minus extension) becomes the provider name.
    /// API keys are looked up from `keys` map by provider name.
    pub fn load_from_dir(
        dir: &Path,
        keys: &HashMap<String, String>,
    ) -> Result<Self, CloudError> {
        let mut registry = Self::new();

        if !dir.exists() {
            debug!("Cloud providers directory does not exist: {}", dir.display());
            return Ok(registry);
        }

        let entries = std::fs::read_dir(dir).map_err(|e| {
            CloudError::ScriptError(format!("Cannot read providers dir {}: {e}", dir.display()))
        })?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "rhai") {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let api_key = keys.get(&name).cloned().unwrap_or_default();

                match ScriptedProvider::load(&name, &path, api_key) {
                    Ok(provider) => {
                        debug!("Registered cloud provider: {name}");
                        registry.providers.insert(name, Arc::new(provider));
                    }
                    Err(e) => {
                        warn!("Failed to load cloud provider {name}: {e}");
                    }
                }
            }
        }

        Ok(registry)
    }

    /// Register a provider from an inline script (e.g. api-expert generated).
    pub fn register_from_source(
        &mut self,
        name: &str,
        source: &str,
        api_key: String,
    ) -> Result<(), CloudError> {
        let provider = ScriptedProvider::from_source(name, source, api_key)?;
        self.providers.insert(name.to_string(), Arc::new(provider));
        Ok(())
    }

    /// Get a provider by name.
    pub fn get(&self, name: &str) -> Option<Arc<ScriptedProvider>> {
        self.providers.get(name).cloned()
    }

    /// List all registered provider names.
    pub fn list(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }
}
