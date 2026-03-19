//! PermissionGate middleware — enforces per-agent tool permission policies.
//!
//! Intercepts `HandlerResponse::Send` after handler dispatch and checks
//! the agent's permission tier for the target tool:
//! - **Auto:** pass through
//! - **Deny:** replace with error ToolResponse back to the agent
//! - **Prompt:** send approval request to TUI, await verdict
//!
//! Denied tools get re-injected as error ToolResponses — the agent sees
//! a failed tool call and can adapt naturally.

use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::{broadcast, mpsc};

use rust_pipeline::prelude::*;

use crate::permissions::{
    ApprovalVerdict, PermissionMap, PermissionTier, ToolApprovalRequest, resolve_tier,
};
use agentos_events::PipelineEvent;

/// PermissionGate middleware — post-dispatch permission enforcement.
pub struct PermissionGate {
    /// agent_name -> permission map
    policies: HashMap<String, PermissionMap>,
    /// TUI approval channel
    approval_tx: Option<mpsc::Sender<ToolApprovalRequest>>,
    /// Event broadcast
    event_tx: Option<broadcast::Sender<PipelineEvent>>,
    /// When true, auto-approve all tiers (debug mode — DebugGate handles gating).
    debug_override: bool,
}

impl PermissionGate {
    /// Create a new PermissionGate.
    ///
    /// `policies` maps agent names to their per-tool permission tiers.
    /// Only agents present in the map are checked — non-agent senders
    /// pass through unaffected.
    pub fn new(
        policies: HashMap<String, PermissionMap>,
        approval_tx: Option<mpsc::Sender<ToolApprovalRequest>>,
        event_tx: Option<broadcast::Sender<PipelineEvent>>,
        debug_override: bool,
    ) -> Self {
        Self {
            policies,
            approval_tx,
            event_tx,
            debug_override,
        }
    }

    fn emit(&self, event: PipelineEvent) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(event);
        }
    }

    /// Build an error ToolResponse XML payload for a denied tool call.
    fn denial_payload(tool_name: &str, reason: &str) -> Vec<u8> {
        format!(
            "<ToolResponse><success>false</success>\
             <result>Permission denied for {tool_name}: {reason}</result></ToolResponse>"
        )
        .into_bytes()
    }
}

#[async_trait]
impl Middleware for PermissionGate {
    async fn post_dispatch(
        &self,
        meta: &DispatchMeta,
        _payload: &ValidatedPayload,
        response: HandlerResponse,
    ) -> Result<PostDispatchVerdict, PipelineError> {
        // Debug override: DebugGate handles all gating, skip permission checks
        if self.debug_override {
            if let HandlerResponse::Send { ref to, .. } = response {
                self.emit(PipelineEvent::ToolApproval {
                    thread_id: meta.thread_id.clone(),
                    agent_name: meta.to.clone(),
                    tool_name: to.clone(),
                    verdict: "auto_debug".into(),
                });
            }
            return Ok(PostDispatchVerdict::PassThrough(response));
        }

        // Only intercept Send responses from agents with policies
        let to = match response {
            HandlerResponse::Send { ref to, .. } => to.clone(),
            other => return Ok(PostDispatchVerdict::PassThrough(other)),
        };

        // Look up policies for the sending agent (meta.to is the handler that produced this response)
        let permissions = match self.policies.get(&meta.to) {
            Some(p) => p,
            None => return Ok(PostDispatchVerdict::PassThrough(response)),
        };

        let tier = resolve_tier(permissions, &to);

        match tier {
            PermissionTier::Auto => {
                self.emit(PipelineEvent::ToolApproval {
                    thread_id: meta.thread_id.clone(),
                    agent_name: meta.to.clone(),
                    tool_name: to.clone(),
                    verdict: "auto".into(),
                });
                Ok(PostDispatchVerdict::PassThrough(response))
            }
            PermissionTier::Deny => {
                self.emit(PipelineEvent::ToolApproval {
                    thread_id: meta.thread_id.clone(),
                    agent_name: meta.to.clone(),
                    tool_name: to.clone(),
                    verdict: "denied_by_policy".into(),
                });
                // Replace with error ToolResponse sent back to the agent
                let error_xml = Self::denial_payload(&to, "blocked by policy");
                Ok(PostDispatchVerdict::Replace(HandlerResponse::Send {
                    to: meta.to.clone(), // send back to the originating agent
                    payload_xml: error_xml,
                }))
            }
            PermissionTier::Prompt => {
                if let Some(ref tx) = self.approval_tx {
                    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                    let request = ToolApprovalRequest {
                        tool_name: to.clone(),
                        args_summary: format!("Send to {to}"),
                        thread_id: meta.thread_id.clone(),
                        response_tx: resp_tx,
                    };
                    if tx.send(request).await.is_err() {
                        // TUI disconnected — headless mode, auto-approve
                        return Ok(PostDispatchVerdict::PassThrough(response));
                    }
                    match resp_rx.await {
                        Ok(ApprovalVerdict::Approved) => {
                            self.emit(PipelineEvent::ToolApproval {
                                thread_id: meta.thread_id.clone(),
                                agent_name: meta.to.clone(),
                                tool_name: to.clone(),
                                verdict: "approved".into(),
                            });
                            Ok(PostDispatchVerdict::PassThrough(response))
                        }
                        Ok(ApprovalVerdict::Denied) | Err(_) => {
                            self.emit(PipelineEvent::ToolApproval {
                                thread_id: meta.thread_id.clone(),
                                agent_name: meta.to.clone(),
                                tool_name: to.clone(),
                                verdict: "denied".into(),
                            });
                            let error_xml =
                                Self::denial_payload(&to, "denied by user");
                            Ok(PostDispatchVerdict::Replace(HandlerResponse::Send {
                                to: meta.to.clone(),
                                payload_xml: error_xml,
                            }))
                        }
                    }
                } else {
                    // No TUI → headless mode, auto-approve
                    Ok(PostDispatchVerdict::PassThrough(response))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_for(agent: &str) -> DispatchMeta {
        DispatchMeta {
            from: "user".into(),
            to: agent.into(),
            thread_id: "t1".into(),
            payload_tag: "ToolResponse".into(),
        }
    }

    fn dummy_payload() -> ValidatedPayload {
        ValidatedPayload {
            xml: b"<ToolResponse><success>true</success></ToolResponse>".to_vec(),
            tag: "ToolResponse".into(),
        }
    }

    fn send_response(to: &str) -> HandlerResponse {
        HandlerResponse::Send {
            to: to.into(),
            payload_xml: b"<FileReadRequest><path>test.rs</path></FileReadRequest>".to_vec(),
        }
    }

    #[tokio::test]
    async fn auto_tier_passes_through() {
        let mut perms = PermissionMap::new();
        perms.insert("file-reader".into(), PermissionTier::Auto);
        let mut policies = HashMap::new();
        policies.insert("coding-agent".into(), perms);

        let gate = PermissionGate::new(policies, None, None, false);
        let result = gate
            .post_dispatch(
                &meta_for("coding-agent"),
                &dummy_payload(),
                send_response("file-reader"),
            )
            .await
            .unwrap();

        match result {
            PostDispatchVerdict::PassThrough(HandlerResponse::Send { to, .. }) => {
                assert_eq!(to, "file-reader");
            }
            _ => panic!("expected PassThrough for Auto tier"),
        }
    }

    #[tokio::test]
    async fn deny_tier_replaces_with_error() {
        let mut perms = PermissionMap::new();
        perms.insert("command-exec".into(), PermissionTier::Deny);
        let mut policies = HashMap::new();
        policies.insert("coding-agent".into(), perms);

        let gate = PermissionGate::new(policies, None, None, false);
        let result = gate
            .post_dispatch(
                &meta_for("coding-agent"),
                &dummy_payload(),
                send_response("command-exec"),
            )
            .await
            .unwrap();

        match result {
            PostDispatchVerdict::Replace(HandlerResponse::Send { to, payload_xml }) => {
                assert_eq!(to, "coding-agent", "denial should send back to agent");
                let text = String::from_utf8_lossy(&payload_xml);
                assert!(text.contains("Permission denied"));
                assert!(text.contains("command-exec"));
            }
            _ => panic!("expected Replace for Deny tier"),
        }
    }

    #[tokio::test]
    async fn non_send_passes_through() {
        let mut perms = PermissionMap::new();
        perms.insert("file-reader".into(), PermissionTier::Deny);
        let mut policies = HashMap::new();
        policies.insert("coding-agent".into(), perms);

        let gate = PermissionGate::new(policies, None, None, false);

        // Reply response — should pass through regardless of permissions
        let result = gate
            .post_dispatch(
                &meta_for("coding-agent"),
                &dummy_payload(),
                HandlerResponse::Reply {
                    payload_xml: b"<AgentResponse><result>done</result></AgentResponse>".to_vec(),
                },
            )
            .await
            .unwrap();

        assert!(matches!(
            result,
            PostDispatchVerdict::PassThrough(HandlerResponse::Reply { .. })
        ));

        // None response
        let result = gate
            .post_dispatch(
                &meta_for("coding-agent"),
                &dummy_payload(),
                HandlerResponse::None,
            )
            .await
            .unwrap();

        assert!(matches!(
            result,
            PostDispatchVerdict::PassThrough(HandlerResponse::None)
        ));
    }

    #[tokio::test]
    async fn prompt_tier_with_mocked_approval() {
        let mut perms = PermissionMap::new();
        perms.insert("file-reader".into(), PermissionTier::Prompt);
        let mut policies = HashMap::new();
        policies.insert("coding-agent".into(), perms);

        let (approval_tx, mut approval_rx) = mpsc::channel(1);
        let gate = PermissionGate::new(policies, Some(approval_tx), None, false);

        // Spawn a task to approve the request
        tokio::spawn(async move {
            if let Some(req) = approval_rx.recv().await {
                let _ = req.response_tx.send(ApprovalVerdict::Approved);
            }
        });

        let result = gate
            .post_dispatch(
                &meta_for("coding-agent"),
                &dummy_payload(),
                send_response("file-reader"),
            )
            .await
            .unwrap();

        match result {
            PostDispatchVerdict::PassThrough(HandlerResponse::Send { to, .. }) => {
                assert_eq!(to, "file-reader");
            }
            _ => panic!("expected PassThrough after approval"),
        }
    }

    #[tokio::test]
    async fn prompt_tier_denied_by_user() {
        let mut perms = PermissionMap::new();
        perms.insert("file-reader".into(), PermissionTier::Prompt);
        let mut policies = HashMap::new();
        policies.insert("coding-agent".into(), perms);

        let (approval_tx, mut approval_rx) = mpsc::channel(1);
        let gate = PermissionGate::new(policies, Some(approval_tx), None, false);

        // Spawn a task to deny the request
        tokio::spawn(async move {
            if let Some(req) = approval_rx.recv().await {
                let _ = req.response_tx.send(ApprovalVerdict::Denied);
            }
        });

        let result = gate
            .post_dispatch(
                &meta_for("coding-agent"),
                &dummy_payload(),
                send_response("file-reader"),
            )
            .await
            .unwrap();

        match result {
            PostDispatchVerdict::Replace(HandlerResponse::Send { to, payload_xml }) => {
                assert_eq!(to, "coding-agent", "denial should send back to agent");
                let text = String::from_utf8_lossy(&payload_xml);
                assert!(text.contains("Permission denied"));
            }
            _ => panic!("expected Replace after denial"),
        }
    }

    #[tokio::test]
    async fn unknown_agent_passes_through() {
        let policies = HashMap::new(); // empty — no agents registered
        let gate = PermissionGate::new(policies, None, None, false);

        let result = gate
            .post_dispatch(
                &meta_for("unknown-agent"),
                &dummy_payload(),
                send_response("file-reader"),
            )
            .await
            .unwrap();

        assert!(matches!(
            result,
            PostDispatchVerdict::PassThrough(HandlerResponse::Send { .. })
        ));
    }

    #[tokio::test]
    async fn debug_override_passes_prompt_tier() {
        let mut perms = PermissionMap::new();
        perms.insert("file-reader".into(), PermissionTier::Prompt);
        let mut policies = HashMap::new();
        policies.insert("coding-agent".into(), perms);

        // debug_override=true — should pass through without prompting
        let gate = PermissionGate::new(policies, None, None, true);
        let result = gate
            .post_dispatch(
                &meta_for("coding-agent"),
                &dummy_payload(),
                send_response("file-reader"),
            )
            .await
            .unwrap();

        match result {
            PostDispatchVerdict::PassThrough(HandlerResponse::Send { to, .. }) => {
                assert_eq!(to, "file-reader");
            }
            _ => panic!("expected PassThrough for debug_override on Prompt tier"),
        }
    }

    #[tokio::test]
    async fn debug_override_passes_deny_tier() {
        let mut perms = PermissionMap::new();
        perms.insert("command-exec".into(), PermissionTier::Deny);
        let mut policies = HashMap::new();
        policies.insert("coding-agent".into(), perms);

        // debug_override=true — even Deny tier should pass through
        let gate = PermissionGate::new(policies, None, None, true);
        let result = gate
            .post_dispatch(
                &meta_for("coding-agent"),
                &dummy_payload(),
                send_response("command-exec"),
            )
            .await
            .unwrap();

        match result {
            PostDispatchVerdict::PassThrough(HandlerResponse::Send { to, .. }) => {
                assert_eq!(to, "command-exec");
            }
            _ => panic!("expected PassThrough for debug_override on Deny tier"),
        }
    }

    #[tokio::test]
    async fn debug_override_false_preserves_behavior() {
        let mut perms = PermissionMap::new();
        perms.insert("command-exec".into(), PermissionTier::Deny);
        let mut policies = HashMap::new();
        policies.insert("coding-agent".into(), perms);

        // debug_override=false — Deny tier should still deny
        let gate = PermissionGate::new(policies, None, None, false);
        let result = gate
            .post_dispatch(
                &meta_for("coding-agent"),
                &dummy_payload(),
                send_response("command-exec"),
            )
            .await
            .unwrap();

        match result {
            PostDispatchVerdict::Replace(HandlerResponse::Send { to, payload_xml }) => {
                assert_eq!(to, "coding-agent");
                let text = String::from_utf8_lossy(&payload_xml);
                assert!(text.contains("Permission denied"));
            }
            _ => panic!("expected Replace for Deny tier with debug_override=false"),
        }
    }
}
