//! DebugGate middleware — pipeline debug backchannel.
//!
//! When `--debug` is active, intercepts every agent→tool dispatch before
//! it reaches the handler, shows it to the TUI, and lets the user
//! approve or deny. Denied calls get a "tool denied" Reply short-circuited
//! back to the calling agent.
//!
//! Agent-to-agent dispatches pass through (only tool calls are gated).
//! Non-agent senders pass through (only agent-initiated calls are gated).

use std::collections::HashSet;

use async_trait::async_trait;
use tokio::sync::{broadcast, mpsc};

use rust_pipeline::prelude::*;

use crate::permissions::{ApprovalVerdict, ToolApprovalRequest};
use agentos_events::PipelineEvent;

/// DebugGate middleware — pre-dispatch gating for debug sessions.
pub struct DebugGate {
    /// Only active when --debug flag is set.
    active: bool,
    /// Which handler names are agents (vs tool handlers).
    agent_names: HashSet<String>,
    /// Shared approval channel (same mpsc as PermissionGate).
    approval_tx: Option<mpsc::Sender<ToolApprovalRequest>>,
    /// Event broadcast for activity log.
    event_tx: Option<broadcast::Sender<PipelineEvent>>,
}

impl DebugGate {
    /// Create a new DebugGate.
    ///
    /// `active` — whether the gate is enabled (from --debug flag).
    /// `agent_names` — set of listener names that are agents.
    /// `approval_tx` — channel to TUI for user approval prompts.
    /// `event_tx` — broadcast channel for pipeline events.
    pub fn new(
        active: bool,
        agent_names: HashSet<String>,
        approval_tx: Option<mpsc::Sender<ToolApprovalRequest>>,
        event_tx: Option<broadcast::Sender<PipelineEvent>>,
    ) -> Self {
        Self {
            active,
            agent_names,
            approval_tx,
            event_tx,
        }
    }

    fn emit(&self, event: PipelineEvent) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(event);
        }
    }
}

#[async_trait]
impl Middleware for DebugGate {
    async fn pre_dispatch(
        &self,
        meta: &DispatchMeta,
        _payload: &ValidatedPayload,
    ) -> Result<PreDispatchVerdict, PipelineError> {
        // 1. Inactive → pass through
        if !self.active {
            return Ok(PreDispatchVerdict::Continue);
        }

        // 2. Non-agent sender → pass through
        if !self.agent_names.contains(&meta.from) {
            return Ok(PreDispatchVerdict::Continue);
        }

        // 3. Agent-to-agent → pass through (don't gate)
        if self.agent_names.contains(&meta.to) {
            return Ok(PreDispatchVerdict::Continue);
        }

        // 4. Emit ToolDispatched event for Activity tab visibility
        self.emit(PipelineEvent::ToolDispatched {
            thread_id: meta.thread_id.clone(),
            agent_name: meta.from.clone(),
            tool_name: meta.to.clone(),
            detail: format!("Debug: {} → {}", meta.payload_tag, meta.to),
        });

        // 5. Send approval request to TUI
        if let Some(ref tx) = self.approval_tx {
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            let request = ToolApprovalRequest {
                tool_name: meta.to.clone(),
                args_summary: format!("Debug: {} → {}", meta.payload_tag, meta.to),
                thread_id: meta.thread_id.clone(),
                response_tx: resp_tx,
            };

            if tx.send(request).await.is_err() {
                // TUI disconnected — headless mode, allow through
                return Ok(PreDispatchVerdict::Continue);
            }

            // 6. Await verdict
            match resp_rx.await {
                Ok(ApprovalVerdict::Approved) => {
                    self.emit(PipelineEvent::ToolApproval {
                        thread_id: meta.thread_id.clone(),
                        agent_name: meta.from.clone(),
                        tool_name: meta.to.clone(),
                        verdict: "approved".into(),
                    });
                    Ok(PreDispatchVerdict::Continue)
                }
                Ok(ApprovalVerdict::Denied) => {
                    self.emit(PipelineEvent::ToolApproval {
                        thread_id: meta.thread_id.clone(),
                        agent_name: meta.from.clone(),
                        tool_name: meta.to.clone(),
                        verdict: "denied_by_debug".into(),
                    });
                    Ok(PreDispatchVerdict::ShortCircuit(HandlerResponse::Reply {
                        payload_xml: b"<ToolResponse><success>false</success>\
                            <result>Tool denied by debug gate</result></ToolResponse>"
                            .to_vec(),
                    }))
                }
                Err(_) => {
                    // Channel disconnected — headless, allow through
                    Ok(PreDispatchVerdict::Continue)
                }
            }
        } else {
            // No approval channel — headless mode, allow through
            Ok(PreDispatchVerdict::Continue)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_names() -> HashSet<String> {
        let mut set = HashSet::new();
        set.insert("coding-agent".into());
        set.insert("review-agent".into());
        set
    }

    fn meta(from: &str, to: &str) -> DispatchMeta {
        DispatchMeta {
            from: from.into(),
            to: to.into(),
            thread_id: "t1".into(),
            payload_tag: "FileReadRequest".into(),
        }
    }

    fn dummy_payload() -> ValidatedPayload {
        ValidatedPayload {
            xml: b"<FileReadRequest><path>test.rs</path></FileReadRequest>".to_vec(),
            tag: "FileReadRequest".into(),
        }
    }

    #[tokio::test]
    async fn inactive_gate_passes_through() {
        let gate = DebugGate::new(false, agent_names(), None, None);
        let result = gate
            .pre_dispatch(&meta("coding-agent", "file-read"), &dummy_payload())
            .await
            .unwrap();
        assert!(matches!(result, PreDispatchVerdict::Continue));
    }

    #[tokio::test]
    async fn non_agent_sender_passes_through() {
        let gate = DebugGate::new(true, agent_names(), None, None);
        // "user" is not in agent_names
        let result = gate
            .pre_dispatch(&meta("user", "file-read"), &dummy_payload())
            .await
            .unwrap();
        assert!(matches!(result, PreDispatchVerdict::Continue));
    }

    #[tokio::test]
    async fn agent_to_agent_passes_through() {
        let gate = DebugGate::new(true, agent_names(), None, None);
        // coding-agent → review-agent: both are agents, should pass
        let result = gate
            .pre_dispatch(&meta("coding-agent", "review-agent"), &dummy_payload())
            .await
            .unwrap();
        assert!(matches!(result, PreDispatchVerdict::Continue));
    }

    #[tokio::test]
    async fn agent_to_tool_approved() {
        let (approval_tx, mut approval_rx) = mpsc::channel(1);
        let gate = DebugGate::new(true, agent_names(), Some(approval_tx), None);

        // Spawn approver
        tokio::spawn(async move {
            if let Some(req) = approval_rx.recv().await {
                let _ = req.response_tx.send(ApprovalVerdict::Approved);
            }
        });

        let result = gate
            .pre_dispatch(&meta("coding-agent", "file-read"), &dummy_payload())
            .await
            .unwrap();
        assert!(matches!(result, PreDispatchVerdict::Continue));
    }

    #[tokio::test]
    async fn agent_to_tool_denied() {
        let (approval_tx, mut approval_rx) = mpsc::channel(1);
        let gate = DebugGate::new(true, agent_names(), Some(approval_tx), None);

        // Spawn denier
        tokio::spawn(async move {
            if let Some(req) = approval_rx.recv().await {
                let _ = req.response_tx.send(ApprovalVerdict::Denied);
            }
        });

        let result = gate
            .pre_dispatch(&meta("coding-agent", "file-read"), &dummy_payload())
            .await
            .unwrap();

        match result {
            PreDispatchVerdict::ShortCircuit(HandlerResponse::Reply { payload_xml }) => {
                let text = String::from_utf8_lossy(&payload_xml);
                assert!(text.contains("Tool denied by debug gate"));
                assert!(text.contains("<success>false</success>"));
            }
            _ => panic!("expected ShortCircuit with denial Reply"),
        }
    }

    #[tokio::test]
    async fn no_approval_channel_passes_through() {
        // Headless mode — no approval_tx
        let gate = DebugGate::new(true, agent_names(), None, None);
        let result = gate
            .pre_dispatch(&meta("coding-agent", "file-read"), &dummy_payload())
            .await
            .unwrap();
        assert!(matches!(result, PreDispatchVerdict::Continue));
    }
}
