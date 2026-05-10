//! ShimStoreTool — kernel-backed CRUD over the cognitive substrate.
//!
//! Single tool, action-dispatched. Replaces the JSON-file shim-rules
//! tool from `a41a195` now that shim_store is the kernel's fourth
//! pillar (per `project_shim_store_design.md`).
//!
//! Action surface:
//!
//! | action              | purpose                                              |
//! |---------------------|------------------------------------------------------|
//! | `create-store`      | Initialize a new empty shim_store                    |
//! | `delete-store`      | Remove a shim_store entirely                         |
//! | `list-stores`       | Names of all on-disk shim_stores                     |
//! | `read-composition`  | Get a store's composition.json bytes                 |
//! | `update-composition`| Replace a store's composition.json (validated first) |
//! | `add-shim`          | Register a trained shim (manifest + ONNX path) in a store |
//! | `retire-shim`       | Soft-retire a shim (move to `<store>/retired/`)      |
//! | `list-shims`        | List shims in one store                              |
//!
//! The tool follows the dispatch.rs pattern for kernel access: holds a
//! `ShimStoreHandles` populated post-pipeline-build via `connect()`.

use std::path::PathBuf;
use std::sync::Arc;

use agentos_kernel::Kernel;
use agentos_llm::types::ShimAttachment;
use async_trait::async_trait;
use rust_pipeline::prelude::*;
use serde_json::json;
use tokio::sync::Mutex;

use super::{extract_tag, ToolPeer, ToolResponse};

/// Deferred handle to the kernel. Tool is registered pre-build;
/// `connect()` populates the kernel reference once the pipeline has
/// opened its data dir.
#[derive(Clone, Default)]
pub struct ShimStoreHandles {
    pub kernel: Arc<Mutex<Option<Arc<Mutex<Kernel>>>>>,
}

impl ShimStoreHandles {
    pub fn new() -> Self {
        Self::default()
    }

    /// Wire up after `AgentPipelineBuilder::build()`.
    pub async fn connect(&self, kernel: Arc<Mutex<Kernel>>) {
        *self.kernel.lock().await = Some(kernel);
    }
}

/// Tool wrapper around the kernel's shim_store APIs.
#[derive(Clone)]
pub struct ShimStoreTool {
    handles: ShimStoreHandles,
}

impl ShimStoreTool {
    pub fn new(handles: ShimStoreHandles) -> Self {
        Self { handles }
    }

    /// Construct with an already-connected kernel (testing convenience).
    pub fn with_kernel(kernel: Arc<Mutex<Kernel>>) -> Self {
        let handles = ShimStoreHandles {
            kernel: Arc::new(Mutex::new(Some(kernel))),
        };
        Self { handles }
    }

    async fn kernel(&self) -> Result<Arc<Mutex<Kernel>>, String> {
        self.handles
            .kernel
            .lock()
            .await
            .clone()
            .ok_or_else(|| "kernel not connected; pipeline build incomplete".to_string())
    }

    async fn handle_create_store(&self, xml_str: &str) -> Result<String, String> {
        let name = required_tag(xml_str, "name")?;
        // base_compat may be empty; comma-separated string for now (a
        // future revision could accept a JSON array). Empty value =
        // empty Vec, which the kernel accepts.
        let base_compat: Vec<String> = extract_tag(xml_str, "base_compat")
            .map(|s| {
                s.split(',')
                    .map(|x| x.trim().to_string())
                    .filter(|x| !x.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        let kernel = self.kernel().await?;
        let mut k = kernel.lock().await;
        k.create_shim_store(&name, base_compat.clone())
            .map_err(|e| format!("create_shim_store: {e}"))?;
        let path = k.shim_store().path_for(&name);
        Ok(json!({
            "store": name,
            "path": path,
            "base_compat": base_compat,
        })
        .to_string())
    }

    async fn handle_delete_store(&self, xml_str: &str) -> Result<String, String> {
        let name = required_tag(xml_str, "name")?;
        let kernel = self.kernel().await?;
        let mut k = kernel.lock().await;
        if !k.shim_store().exists(&name) {
            return Err(format!("shim_store `{name}` does not exist"));
        }
        k.delete_shim_store(&name)
            .map_err(|e| format!("delete_shim_store: {e}"))?;
        Ok(json!({"deleted": name}).to_string())
    }

    async fn handle_list_stores(&self) -> Result<String, String> {
        let kernel = self.kernel().await?;
        let k = kernel.lock().await;
        let mut stores = k.shim_store().list_stores();
        stores.sort();
        Ok(serde_json::to_string(&stores)
            .map_err(|e| format!("serialize stores: {e}"))?)
    }

    async fn handle_read_composition(&self, xml_str: &str) -> Result<String, String> {
        let name = required_tag(xml_str, "name")?;
        let kernel = self.kernel().await?;
        let k = kernel.lock().await;
        let bytes = k.shim_store().composition_bytes_for(&name).ok_or_else(|| {
            format!("shim_store `{name}` does not exist")
        })?;
        // Return raw bytes verbatim — caller (LLM) will see the
        // composition.json content directly.
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    async fn handle_update_composition(&self, xml_str: &str) -> Result<String, String> {
        let name = required_tag(xml_str, "name")?;
        let composition_raw = required_tag(xml_str, "composition")?;
        // Validate against the live ShimAttachment schema before writing.
        // A typo'd update should not silently brick the agent on next
        // restart; we catch it here while the agent is still running.
        let _validated: ShimAttachment = serde_json::from_str(&composition_raw).map_err(|e| {
            format!("composition does not match ShimAttachment schema: {e}")
        })?;
        let kernel = self.kernel().await?;
        let mut k = kernel.lock().await;
        if !k.shim_store().exists(&name) {
            return Err(format!("shim_store `{name}` does not exist"));
        }
        k.update_composition(&name, composition_raw.into_bytes())
            .map_err(|e| format!("update_composition: {e}"))?;
        Ok(json!({
            "store": name,
            "note": "agents using this store must restart for changes to take effect"
        })
        .to_string())
    }

    async fn handle_add_shim(&self, xml_str: &str) -> Result<String, String> {
        let store = required_tag(xml_str, "store")?;
        let shim_id = required_tag(xml_str, "shim_id")?;
        let manifest = required_tag(xml_str, "manifest")?;
        let onnx_path = required_tag(xml_str, "onnx_path")?;
        let onnx_bytes = std::fs::read(PathBuf::from(&onnx_path))
            .map_err(|e| format!("read onnx file `{onnx_path}`: {e}"))?;

        let kernel = self.kernel().await?;
        let mut k = kernel.lock().await;
        if !k.shim_store().exists(&store) {
            return Err(format!("shim_store `{store}` does not exist"));
        }
        k.add_shim_to_store(&store, &shim_id, manifest.into_bytes(), onnx_bytes)
            .map_err(|e| format!("add_shim_to_store: {e}"))?;
        let path = k
            .shim_store()
            .shims_in(&store)
            .and_then(|s| s.get(&shim_id))
            .map(|r| r.onnx_path.clone());
        Ok(json!({
            "store": store,
            "shim_id": shim_id,
            "onnx_path": path,
        })
        .to_string())
    }

    async fn handle_retire_shim(&self, xml_str: &str) -> Result<String, String> {
        let store = required_tag(xml_str, "store")?;
        let shim_id = required_tag(xml_str, "shim_id")?;
        let kernel = self.kernel().await?;
        let mut k = kernel.lock().await;
        k.retire_shim_from_store(&store, &shim_id)
            .map_err(|e| format!("retire_shim_from_store: {e}"))?;
        Ok(json!({"store": store, "shim_id": shim_id, "retired": true}).to_string())
    }

    async fn handle_list_shims(&self, xml_str: &str) -> Result<String, String> {
        let store = required_tag(xml_str, "store")?;
        let kernel = self.kernel().await?;
        let k = kernel.lock().await;
        let shims = k
            .shim_store()
            .shims_in(&store)
            .ok_or_else(|| format!("shim_store `{store}` does not exist"))?;
        let mut ids: Vec<&str> = shims.keys().map(|s| s.as_str()).collect();
        ids.sort();
        Ok(serde_json::to_string(&ids)
            .map_err(|e| format!("serialize: {e}"))?)
    }
}

fn required_tag(xml_str: &str, name: &str) -> Result<String, String> {
    extract_tag(xml_str, name)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing required <{name}>"))
}

#[async_trait]
impl Handler for ShimStoreTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);
        let action = extract_tag(&xml_str, "action").unwrap_or_default();

        let result = match action.as_str() {
            "create-store" => self.handle_create_store(&xml_str).await,
            "delete-store" => self.handle_delete_store(&xml_str).await,
            "list-stores" => self.handle_list_stores().await,
            "read-composition" => self.handle_read_composition(&xml_str).await,
            "update-composition" => self.handle_update_composition(&xml_str).await,
            "add-shim" => self.handle_add_shim(&xml_str).await,
            "retire-shim" => self.handle_retire_shim(&xml_str).await,
            "list-shims" => self.handle_list_shims(&xml_str).await,
            "" => Err("missing required <action>".into()),
            other => Err(format!(
                "unknown action: {other} (allowed: create-store|delete-store|\
                 list-stores|read-composition|update-composition|add-shim|\
                 retire-shim|list-shims)"
            )),
        };

        let payload_xml = match result {
            Ok(body) => ToolResponse::ok(&body),
            Err(msg) => ToolResponse::err(&msg),
        };
        Ok(HandlerResponse::Reply { payload_xml })
    }
}

#[async_trait]
impl ToolPeer for ShimStoreTool {
    fn name(&self) -> &str {
        "shim-store"
    }

    fn wit(&self) -> &str {
        r#"
/// Manage cortex shim_stores via the kernel's fourth pillar. Each
/// store is a named directory of ONNX shim weights + composition rules
/// + per-shim metadata. Updates require an agent restart to take
/// effect (v1).
interface shim-store {
    record request {
        /// create-store | delete-store | list-stores |
        /// read-composition | update-composition |
        /// add-shim | retire-shim | list-shims
        action: string,
        /// Store name (required for most actions).
        name: option<string>,
        /// Comma-separated base-model names (create-store only).
        base-compat: option<string>,
        /// JSON-serialized ShimAttachment (update-composition).
        composition: option<string>,
        /// Target store (add-shim, retire-shim, list-shims).
        store: option<string>,
        /// Shim id (add-shim, retire-shim).
        shim-id: option<string>,
        /// JSON-serialized ShimManifest sidecar (add-shim).
        manifest: option<string>,
        /// Path to the trained ONNX file on disk (add-shim).
        onnx-path: option<string>,
    }
    invoke: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "shim-expert".into(),
            own_name: "shim-store".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "ShimStore".into(),
        }
    }

    fn parse(resp: HandlerResponse) -> (bool, String) {
        match resp {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                let success = xml.contains("<success>true</success>");
                let body = if success {
                    extract_tag(&xml, "result").unwrap_or_default()
                } else {
                    extract_tag(&xml, "error").unwrap_or_default()
                };
                (success, body)
            }
            _ => panic!("expected Reply"),
        }
    }

    fn fresh_tool() -> (TempDir, ShimStoreTool) {
        let dir = TempDir::new().unwrap();
        let kernel = Kernel::open(&dir.path().join("data")).unwrap();
        let tool = ShimStoreTool::with_kernel(Arc::new(Mutex::new(kernel)));
        (dir, tool)
    }

    #[tokio::test]
    async fn create_store_then_list_includes_it() {
        let (_dir, tool) = fresh_tool();
        let xml = "<ShimStore><action>create-store</action><name>bob</name></ShimStore>";
        let (ok, _) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok);

        let xml = "<ShimStore><action>list-stores</action></ShimStore>";
        let (ok, body) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(body.contains("bob"));
    }

    #[tokio::test]
    async fn create_store_with_base_compat() {
        let (_dir, tool) = fresh_tool();
        let xml = "<ShimStore>\
            <action>create-store</action>\
            <name>bob</name>\
            <base_compat>qwen-2.5-3b, haiku</base_compat>\
        </ShimStore>";
        let (ok, body) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");
        assert!(body.contains("qwen-2.5-3b"));
        assert!(body.contains("haiku"));
    }

    #[tokio::test]
    async fn delete_missing_store_errors() {
        let (_dir, tool) = fresh_tool();
        let xml = "<ShimStore><action>delete-store</action><name>noop</name></ShimStore>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("does not exist"));
    }

    #[tokio::test]
    async fn update_composition_round_trip() {
        let (_dir, tool) = fresh_tool();
        // Create
        let xml = "<ShimStore><action>create-store</action><name>bob</name></ShimStore>";
        parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());

        // Update
        let composition = r#"{"gate_shims":["should_respond"],"steer_shims":[],"inject_shims":[],"shim_rules":[]}"#;
        let xml = format!(
            "<ShimStore><action>update-composition</action><name>bob</name><composition>{}</composition></ShimStore>",
            agentos_events::xml_escape(composition),
        );
        let (ok, body) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");

        // Read
        let xml = "<ShimStore><action>read-composition</action><name>bob</name></ShimStore>";
        let (ok, body) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok);
        let parsed: ShimAttachment = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed.gate_shims, vec!["should_respond"]);
    }

    #[tokio::test]
    async fn update_composition_rejects_invalid_schema() {
        let (_dir, tool) = fresh_tool();
        let xml = "<ShimStore><action>create-store</action><name>bob</name></ShimStore>";
        parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());

        // gate_shims must be an array; this passes a string instead.
        let xml = format!(
            "<ShimStore><action>update-composition</action><name>bob</name><composition>{}</composition></ShimStore>",
            agentos_events::xml_escape(r#"{"gate_shims":"not an array"}"#),
        );
        let (ok, msg) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("ShimAttachment schema"));
    }

    #[tokio::test]
    async fn add_shim_writes_onnx_and_manifest() {
        let (dir, tool) = fresh_tool();
        let xml = "<ShimStore><action>create-store</action><name>bob</name></ShimStore>";
        parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());

        // Place a tiny "ONNX" file on disk for the tool to read.
        let onnx_input = dir.path().join("input.onnx");
        std::fs::write(&onnx_input, b"\x00\x01ONNX-fake-bytes").unwrap();

        let xml = format!(
            "<ShimStore>\
                <action>add-shim</action>\
                <store>bob</store>\
                <shim_id>should_respond</shim_id>\
                <manifest>{{\"id\":\"should_respond\",\"phase\":\"gate\"}}</manifest>\
                <onnx_path>{}</onnx_path>\
            </ShimStore>",
            onnx_input.display(),
        );
        let (ok, body) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");
        assert!(body.contains("should_respond"));

        let xml = "<ShimStore><action>list-shims</action><store>bob</store></ShimStore>";
        let (ok, body) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(body.contains("should_respond"));
    }

    #[tokio::test]
    async fn retire_shim_drops_from_active() {
        let (dir, tool) = fresh_tool();
        let xml = "<ShimStore><action>create-store</action><name>bob</name></ShimStore>";
        parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());

        let onnx_input = dir.path().join("input.onnx");
        std::fs::write(&onnx_input, b"x").unwrap();
        let xml = format!(
            "<ShimStore><action>add-shim</action><store>bob</store>\
             <shim_id>x</shim_id><manifest>{{}}</manifest>\
             <onnx_path>{}</onnx_path></ShimStore>",
            onnx_input.display(),
        );
        parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());

        let xml = "<ShimStore><action>retire-shim</action><store>bob</store><shim_id>x</shim_id></ShimStore>";
        let (ok, _) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok);

        let xml = "<ShimStore><action>list-shims</action><store>bob</store></ShimStore>";
        let (ok, body) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert_eq!(body, "[]");
    }

    #[tokio::test]
    async fn missing_action_errors() {
        let (_dir, tool) = fresh_tool();
        let xml = "<ShimStore></ShimStore>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("<action>"));
    }

    #[tokio::test]
    async fn unknown_action_errors() {
        let (_dir, tool) = fresh_tool();
        let xml = "<ShimStore><action>nope</action></ShimStore>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("unknown action"));
    }

    #[tokio::test]
    async fn unconnected_kernel_errors_cleanly() {
        // Tool registered but kernel never connected (deferred handles).
        let tool = ShimStoreTool::new(ShimStoreHandles::new());
        let xml = "<ShimStore><action>list-stores</action></ShimStore>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("kernel not connected"));
    }
}
