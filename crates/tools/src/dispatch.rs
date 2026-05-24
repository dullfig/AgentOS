//! dispatch tool — spawn a thread for a target agent and inject an initial task.
//!
//! Non-blocking: creates thread, injects envelope, returns immediately.
//! The target agent runs independently on its own thread.
//! If the target is interactive, emits FocusAcquire so the TUI opens a tab.
//!
//! ## Recursion depth limit (security audit H1)
//!
//! Without a depth cap, an attacker who can prompt-inject Bob (or any
//! agent reachable via dispatch) can drive an infinite dispatch chain:
//! Bob → coder → Bob → coder → ... Each link is an LLM call against a
//! paid model — token-budget DoS, kernel WAL growth, process memory.
//!
//! We track depth per thread_id in an in-tool HashMap. Root threads
//! created outside dispatch implicitly have depth 0; each dispatch
//! increments. Above `MAX_DISPATCH_DEPTH`, the call is refused.
//!
//! No platform changes needed — the map lives in the tool. Memory:
//! one entry per active dispatched thread (~40 bytes); bounded by the
//! platform's max concurrent thread count.

use std::collections::HashMap;
use std::sync::Arc;

use agentos_events::PipelineEvent;
use agentos_kernel::Kernel;
use async_trait::async_trait;
use rust_pipeline::prelude::*;
use tokio::sync::{broadcast, Mutex};

use super::{extract_tag, ToolPeer, ToolResponse};
use agentos_organism::Organism;

/// Maximum dispatch chain depth. Bob (depth 0) → coder (1) → tester
/// (2) → reporter (3) → fixer (4). A fifth hop is refused.
///
/// Real workflows fit easily: dispatch chains in practice are 1-2
/// levels (Bob → specialist; specialist → optional sub-buffer). 4
/// gives slack without admitting unbounded recursion.
pub const MAX_DISPATCH_DEPTH: u8 = 4;

/// Cloneable handle for injecting envelopes into the pipeline.
pub type InjectTx = tokio::sync::mpsc::Sender<Vec<u8>>;

/// Shared handles populated after pipeline.build(). The dispatch tool
/// is registered pre-build but needs kernel + inject_tx from the built pipeline.
#[derive(Clone)]
pub struct DispatchHandles {
    pub kernel: Arc<Mutex<Option<Arc<Mutex<Kernel>>>>>,
    pub inject_tx: Arc<Mutex<Option<InjectTx>>>,
}

impl DispatchHandles {
    pub fn new() -> Self {
        Self {
            kernel: Arc::new(Mutex::new(None)),
            inject_tx: Arc::new(Mutex::new(None)),
        }
    }

    /// Wire up after pipeline.build().
    pub async fn connect(&self, kernel: Arc<Mutex<Kernel>>, inject_tx: InjectTx) {
        *self.kernel.lock().await = Some(kernel);
        *self.inject_tx.lock().await = Some(inject_tx);
    }
}

pub struct DispatchTool {
    handles: DispatchHandles,
    event_tx: broadcast::Sender<PipelineEvent>,
    organism: Arc<Organism>,
    /// Depth per thread_id. New thread inherits parent's depth + 1.
    /// Threads not in the map are treated as depth 0 (root threads
    /// created outside dispatch).
    depths: Arc<Mutex<HashMap<String, u8>>>,
}

impl DispatchTool {
    /// Create with all handles available (for testing).
    pub fn new(
        kernel: Arc<Mutex<Kernel>>,
        inject_tx: InjectTx,
        event_tx: broadcast::Sender<PipelineEvent>,
        organism: Arc<Organism>,
    ) -> Self {
        let handles = DispatchHandles {
            kernel: Arc::new(Mutex::new(Some(kernel))),
            inject_tx: Arc::new(Mutex::new(Some(inject_tx))),
        };
        Self {
            handles,
            event_tx,
            organism,
            depths: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create with deferred handles (populated after pipeline.build()).
    pub fn new_deferred(
        handles: DispatchHandles,
        event_tx: broadcast::Sender<PipelineEvent>,
        organism: Arc<Organism>,
    ) -> Self {
        Self {
            handles,
            event_tx,
            organism,
            depths: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl Handler for DispatchTool {
    async fn handle(&self, payload: ValidatedPayload, ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        // Required: target agent name
        let target = match extract_tag(&xml_str, "target") {
            Some(t) => t,
            None => return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("provide <target> agent name"),
            }),
        };

        // Required: initial task/message
        let task = match extract_tag(&xml_str, "task") {
            Some(t) => t,
            None => return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("provide <task> message for the agent"),
            }),
        };

        // Look up target listener in organism
        let listeners = self.organism.listeners();
        let listener_def = match listeners.get(target.as_str()) {
            Some(def) => def,
            None => return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("unknown agent: '{}'", target)),
            }),
        };

        // Must be an agent listener
        if !listener_def.is_agent {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "'{}' is not an agent listener", target
                )),
            });
        }

        // Depth check (security audit H1). Look up caller's depth in
        // the per-tool map; 0 if not present (root thread). New
        // dispatch = caller + 1. Above MAX_DISPATCH_DEPTH, refuse.
        let caller_depth = {
            let map = self.depths.lock().await;
            *map.get(&ctx.thread_id).unwrap_or(&0)
        };
        let new_depth = caller_depth.saturating_add(1);
        if new_depth > MAX_DISPATCH_DEPTH {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "dispatch depth limit exceeded (current chain at {caller_depth}, \
                     max {MAX_DISPATCH_DEPTH}); refusing to spawn deeper"
                )),
            });
        }

        // Get kernel handle
        let kernel_arc = {
            let guard = self.handles.kernel.lock().await;
            match guard.as_ref() {
                Some(k) => k.clone(),
                None => return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err("dispatch not ready — pipeline not yet started"),
                }),
            }
        };

        // Create a new thread for the target agent
        let profile = self.organism
            .profile_names()
            .into_iter()
            .next()
            .unwrap_or("default")
            .to_string();

        let thread_id = {
            let mut kernel = kernel_arc.lock().await;
            kernel.initialize_root(&target, &profile)
                .map_err(|e| PipelineError::Handler(format!("thread creation failed: {e}")))?
        };

        // Record the new thread's depth so any future dispatch FROM
        // this thread sees the chain depth.
        self.depths.lock().await.insert(thread_id.clone(), new_depth);

        // Build the task envelope
        let escaped_task = super::xml_escape(&task);
        let xml = format!(
            "<{tag}><task>{escaped_task}</task></{tag}>",
            tag = listener_def.payload_tag
        );
        let envelope = build_envelope(
            "user",           // from: user initiated this
            &target,          // to: target agent
            &thread_id,       // new thread
            xml.as_bytes(),
        )
        .map_err(|e| PipelineError::Handler(format!("envelope build failed: {e}")))?;

        // Inject into the pipeline (non-blocking send)
        let inject_tx = {
            let guard = self.handles.inject_tx.lock().await;
            match guard.as_ref() {
                Some(tx) => tx.clone(),
                None => return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err("dispatch not ready — pipeline not yet started"),
                }),
            }
        };
        inject_tx
            .send(envelope)
            .await
            .map_err(|e| PipelineError::Handler(format!("inject failed: {e}")))?;

        // If the target has interactive config or is an agent, emit FocusAcquire
        // so the TUI opens a tab and switches to it
        let interactive = listener_def.agent_config
            .as_ref()
            .map(|_| true)  // for now, all dispatched agents are interactive
            .unwrap_or(false);

        if interactive {
            let _ = self.event_tx.send(PipelineEvent::FocusAcquire {
                agent_name: target.clone(),
                parent_agent: ctx.from.clone(),
            });
        }

        Ok(HandlerResponse::Reply {
            payload_xml: ToolResponse::ok(&format!(
                "Dispatched to {} (thread {})",
                target, thread_id
            )),
        })
    }
}

#[async_trait]
impl ToolPeer for DispatchTool {
    fn name(&self) -> &str {
        "dispatch"
    }

    fn wit(&self) -> &str {
        r#"
/// Spawn a thread for a target agent and inject an initial task. Returns immediately — the agent runs independently. If the agent is interactive, the TUI opens a tab for it.
interface dispatch {
    record request {
        /// Name of the agent to dispatch to (must be an agent listener)
        target: string,
        /// Initial task or message for the agent
        task: string,
    }
    dispatch: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "DispatchRequest".into(),
        }
    }

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            from: "bob".into(),
            own_name: "dispatch".into(),
            thread_id: "bob-thread".into(),
        }
    }

    fn make_test_organism() -> Organism {
        use agentos_organism::parser::parse_organism;
        parse_organism(r#"
organism:
  name: test
prompts:
  base: |
    You are a test agent.
listeners:
  - name: coder
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Coder"
    agent:
      prompt: "base"
  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"
  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM pool"
profiles:
  default:
    linux_user: agentos
    listeners: [coder, file-read, llm-pool]
    journal: retain_forever
"#).unwrap()
    }

    #[tokio::test]
    async fn missing_target_returns_error() {
        let tempdir = tempfile::TempDir::new().unwrap();
        let kernel = Kernel::open(tempdir.path()).unwrap();
        let (inject_tx, _rx) = tokio::sync::mpsc::channel(16);
        let (event_tx, _) = broadcast::channel(16);
        let org = Arc::new(make_test_organism());

        let tool = DispatchTool::new(
            Arc::new(Mutex::new(kernel)), inject_tx, event_tx, org,
        );

        let xml = "<DispatchRequest><task>do something</task></DispatchRequest>";
        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("<success>false</success>"), "{s}");
                assert!(s.contains("target"), "{s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn missing_task_returns_error() {
        let tempdir = tempfile::TempDir::new().unwrap();
        let kernel = Kernel::open(tempdir.path()).unwrap();
        let (inject_tx, _rx) = tokio::sync::mpsc::channel(16);
        let (event_tx, _) = broadcast::channel(16);
        let org = Arc::new(make_test_organism());

        let tool = DispatchTool::new(
            Arc::new(Mutex::new(kernel)), inject_tx, event_tx, org,
        );

        let xml = "<DispatchRequest><target>coder</target></DispatchRequest>";
        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("<success>false</success>"), "{s}");
                assert!(s.contains("task"), "{s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn unknown_agent_returns_error() {
        let tempdir = tempfile::TempDir::new().unwrap();
        let kernel = Kernel::open(tempdir.path()).unwrap();
        let (inject_tx, _rx) = tokio::sync::mpsc::channel(16);
        let (event_tx, _) = broadcast::channel(16);
        let org = Arc::new(make_test_organism());

        let tool = DispatchTool::new(
            Arc::new(Mutex::new(kernel)), inject_tx, event_tx, org,
        );

        let xml = "<DispatchRequest><target>nonexistent</target><task>hello</task></DispatchRequest>";
        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("unknown agent"), "{s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn non_agent_listener_returns_error() {
        let tempdir = tempfile::TempDir::new().unwrap();
        let kernel = Kernel::open(tempdir.path()).unwrap();
        let (inject_tx, _rx) = tokio::sync::mpsc::channel(16);
        let (event_tx, _) = broadcast::channel(16);
        let org = Arc::new(make_test_organism());

        let tool = DispatchTool::new(
            Arc::new(Mutex::new(kernel)), inject_tx, event_tx, org,
        );

        let xml = "<DispatchRequest><target>file-read</target><task>hello</task></DispatchRequest>";
        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("not an agent"), "{s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn successful_dispatch() {
        let tempdir = tempfile::TempDir::new().unwrap();
        let kernel = Kernel::open(tempdir.path()).unwrap();
        let (inject_tx, mut inject_rx) = tokio::sync::mpsc::channel(16);
        let (event_tx, mut event_rx) = broadcast::channel(16);
        let org = Arc::new(make_test_organism());

        let tool = DispatchTool::new(
            Arc::new(Mutex::new(kernel)), inject_tx, event_tx, org,
        );

        let xml = "<DispatchRequest><target>coder</target><task>refactor auth module</task></DispatchRequest>";
        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();

        // Should succeed
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("<success>true</success>"), "expected success: {s}");
                assert!(s.contains("Dispatched to coder"), "{s}");
            }
            _ => panic!("expected Reply"),
        }

        // Should have injected an envelope
        let envelope = inject_rx.try_recv().expect("expected injected envelope");
        let envelope_str = String::from_utf8_lossy(&envelope);
        assert!(envelope_str.contains("refactor auth module"), "envelope should contain task");

        // Should have emitted FocusAcquire
        let event = event_rx.try_recv().expect("expected FocusAcquire event");
        match event {
            PipelineEvent::FocusAcquire { agent_name, parent_agent } => {
                assert_eq!(agent_name, "coder");
                assert_eq!(parent_agent, "bob");
            }
            _ => panic!("expected FocusAcquire, got {:?}", event),
        }
    }

    #[tokio::test]
    async fn dispatch_depth_increments_then_caps() {
        // H1 regression: chain dispatches Bob → coder → coder → ...
        // and verify the cap fires at MAX_DISPATCH_DEPTH.
        //
        // We simulate the chain by reusing the same tool instance and
        // manually invoking handle() with a HandlerContext whose
        // thread_id matches the previously-created thread. (In real
        // operation, an agent running on the newly-created thread
        // would be the one issuing the next dispatch — same effect.)
        let tempdir = tempfile::TempDir::new().unwrap();
        let kernel = Kernel::open(tempdir.path()).unwrap();
        let (inject_tx, mut _inject_rx) = tokio::sync::mpsc::channel(64);
        let (event_tx, mut _event_rx) = broadcast::channel(64);
        let org = Arc::new(make_test_organism());

        let tool = DispatchTool::new(
            Arc::new(Mutex::new(kernel)), inject_tx, event_tx, org,
        );

        // Helper: dispatch and extract the new thread_id from the
        // success message ("Dispatched to coder (thread <uuid>)").
        async fn dispatch_once(
            tool: &DispatchTool,
            from_thread: &str,
        ) -> (bool, String) {
            let ctx = HandlerContext {
                from: "bob".into(),
                own_name: "dispatch".into(),
                thread_id: from_thread.into(),
            };
            let xml = "<DispatchRequest><target>coder</target><task>x</task></DispatchRequest>";
            let resp = tool.handle(make_payload(xml), ctx).await.unwrap();
            match resp {
                HandlerResponse::Reply { payload_xml } => {
                    let s = String::from_utf8(payload_xml).unwrap();
                    let ok = s.contains("<success>true</success>");
                    (ok, s)
                }
                _ => panic!("expected Reply"),
            }
        }

        // Depth 1: from a root ("bob-thread", not in the map → depth 0).
        let (ok1, s1) = dispatch_once(&tool, "bob-thread").await;
        assert!(ok1, "{s1}");
        let t1 = extract_thread_id(&s1);

        // Depth 2.
        let (ok2, s2) = dispatch_once(&tool, &t1).await;
        assert!(ok2, "{s2}");
        let t2 = extract_thread_id(&s2);

        // Depth 3.
        let (ok3, s3) = dispatch_once(&tool, &t2).await;
        assert!(ok3, "{s3}");
        let t3 = extract_thread_id(&s3);

        // Depth 4 — the last allowed.
        let (ok4, s4) = dispatch_once(&tool, &t3).await;
        assert!(ok4, "depth 4 should still pass: {s4}");
        let t4 = extract_thread_id(&s4);

        // Depth 5 — refused.
        let (ok5, s5) = dispatch_once(&tool, &t4).await;
        assert!(!ok5, "depth 5 should be refused: {s5}");
        assert!(s5.contains("depth limit"), "got: {s5}");
    }

    fn extract_thread_id(success_response: &str) -> String {
        // "Dispatched to coder (thread <uuid>)"
        let needle = "thread ";
        let start = success_response.find(needle).unwrap() + needle.len();
        let end = success_response[start..]
            .find(')')
            .expect("expected closing paren in dispatch response")
            + start;
        success_response[start..end].to_string()
    }
}
