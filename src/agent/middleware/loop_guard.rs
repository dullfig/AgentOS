//! LoopGuard middleware — prevents runaway agentic loops.
//!
//! Counts ToolResponse dispatches per (thread, agent) pair.
//! Resets on non-ToolResponse messages (new user turns).
//! Short-circuits with an error AgentResponse when the limit is exceeded.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use rust_pipeline::prelude::*;

use crate::agent::translate::xml_escape_text;

/// LoopGuard middleware — limits agentic iterations per thread per agent.
pub struct LoopGuard {
    /// agent_name -> max iterations allowed
    limits: HashMap<String, usize>,
    /// (thread_id, agent_name) -> current iteration count
    counters: Arc<Mutex<HashMap<(String, String), usize>>>,
}

impl LoopGuard {
    /// Create a new LoopGuard with per-agent iteration limits.
    ///
    /// Only agents present in `limits` are tracked. Non-agent targets
    /// (tool handlers, buffers) pass through unaffected.
    pub fn new(limits: HashMap<String, usize>) -> Self {
        Self {
            limits,
            counters: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl Middleware for LoopGuard {
    async fn pre_dispatch(
        &self,
        meta: &DispatchMeta,
        _payload: &ValidatedPayload,
    ) -> Result<PreDispatchVerdict, PipelineError> {
        // Only track agents that have limits configured
        let max = match self.limits.get(&meta.to) {
            Some(&max) => max,
            None => return Ok(PreDispatchVerdict::Continue),
        };

        let key = (meta.thread_id.clone(), meta.to.clone());
        let mut counters = self.counters.lock().await;

        if meta.payload_tag == "ToolResponse" {
            // Agentic cycle — increment counter
            let count = counters.entry(key).or_insert(0);
            *count += 1;

            if *count > max {
                let msg = format!(
                    "Agentic iteration limit reached ({max} iterations). \
                     Stopping to prevent runaway loop."
                );
                let reply_xml = format!(
                    "<AgentResponse><result>{}</result></AgentResponse>",
                    xml_escape_text(&msg)
                );
                return Ok(PreDispatchVerdict::ShortCircuit(HandlerResponse::Reply {
                    payload_xml: reply_xml.into_bytes(),
                }));
            }
        } else {
            // New user turn — reset counter for this thread+agent
            counters.remove(&key);
        }

        Ok(PreDispatchVerdict::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(to: &str, tag: &str, thread: &str) -> DispatchMeta {
        DispatchMeta {
            from: "user".into(),
            to: to.into(),
            thread_id: thread.into(),
            payload_tag: tag.into(),
        }
    }

    fn dummy_payload() -> ValidatedPayload {
        ValidatedPayload {
            xml: b"<ToolResponse><success>true</success><result>ok</result></ToolResponse>"
                .to_vec(),
            tag: "ToolResponse".into(),
        }
    }

    #[tokio::test]
    async fn counter_increments_on_tool_response() {
        let mut limits = HashMap::new();
        limits.insert("agent-1".into(), 3);
        let guard = LoopGuard::new(limits);

        // First ToolResponse — count = 1
        let result = guard
            .pre_dispatch(&meta("agent-1", "ToolResponse", "t1"), &dummy_payload())
            .await
            .unwrap();
        assert!(matches!(result, PreDispatchVerdict::Continue));

        // Verify counter
        let counters = guard.counters.lock().await;
        assert_eq!(
            counters[&("t1".into(), "agent-1".into())],
            1
        );
    }

    #[tokio::test]
    async fn counter_resets_on_non_tool_response() {
        let mut limits = HashMap::new();
        limits.insert("agent-1".into(), 3);
        let guard = LoopGuard::new(limits);

        // Increment twice
        guard
            .pre_dispatch(&meta("agent-1", "ToolResponse", "t1"), &dummy_payload())
            .await
            .unwrap();
        guard
            .pre_dispatch(&meta("agent-1", "ToolResponse", "t1"), &dummy_payload())
            .await
            .unwrap();

        // Non-ToolResponse resets
        let task_payload = ValidatedPayload {
            xml: b"<AgentTask><task>do something</task></AgentTask>".to_vec(),
            tag: "AgentTask".into(),
        };
        guard
            .pre_dispatch(&meta("agent-1", "AgentTask", "t1"), &task_payload)
            .await
            .unwrap();

        // Counter should be gone
        let counters = guard.counters.lock().await;
        assert!(!counters.contains_key(&("t1".into(), "agent-1".into())));
    }

    #[tokio::test]
    async fn short_circuits_at_limit_plus_one() {
        let mut limits = HashMap::new();
        limits.insert("agent-1".into(), 2);
        let guard = LoopGuard::new(limits);

        // Two iterations — should be fine
        let r1 = guard
            .pre_dispatch(&meta("agent-1", "ToolResponse", "t1"), &dummy_payload())
            .await
            .unwrap();
        assert!(matches!(r1, PreDispatchVerdict::Continue));

        let r2 = guard
            .pre_dispatch(&meta("agent-1", "ToolResponse", "t1"), &dummy_payload())
            .await
            .unwrap();
        assert!(matches!(r2, PreDispatchVerdict::Continue));

        // Third iteration — over the limit
        let r3 = guard
            .pre_dispatch(&meta("agent-1", "ToolResponse", "t1"), &dummy_payload())
            .await
            .unwrap();
        match r3 {
            PreDispatchVerdict::ShortCircuit(HandlerResponse::Reply { payload_xml }) => {
                let text = String::from_utf8_lossy(&payload_xml);
                assert!(text.contains("iteration limit"));
            }
            _ => panic!("expected short-circuit at limit+1"),
        }
    }

    #[tokio::test]
    async fn non_agent_targets_ignored() {
        let mut limits = HashMap::new();
        limits.insert("agent-1".into(), 2);
        let guard = LoopGuard::new(limits);

        // Dispatch to a tool (not in limits map) — always continues
        for _ in 0..10 {
            let result = guard
                .pre_dispatch(
                    &meta("file-reader", "ToolResponse", "t1"),
                    &dummy_payload(),
                )
                .await
                .unwrap();
            assert!(matches!(result, PreDispatchVerdict::Continue));
        }
    }
}
