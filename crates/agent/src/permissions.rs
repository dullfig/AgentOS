//! Context-aware permission framework.
//!
//! Each agent declares per-tool permission tiers in organism YAML.
//! The handler checks permissions before dispatching tool calls:
//! - `Auto` → execute immediately
//! - `Prompt` → pause for user approval via TUI
//! - `Deny` → reject immediately, agent sees error
//!
//! Permission types (`PermissionTier`, `PermissionMap`, `resolve_tier`)
//! live in `agentos-events` for cross-crate access. Re-exported here
//! for convenience.

// Re-export shared types from events crate
pub use agentos_events::{PermissionMap, PermissionTier, resolve_tier};

/// Request sent to TUI for user approval.
pub struct ToolApprovalRequest {
    /// Tool being invoked.
    pub tool_name: String,
    /// Human-readable summary of the tool arguments.
    pub args_summary: String,
    /// Thread that triggered the request.
    pub thread_id: String,
    /// Oneshot channel to send the verdict back to the handler.
    pub response_tx: tokio::sync::oneshot::Sender<ApprovalVerdict>,
}

/// User's verdict on a tool approval request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalVerdict {
    Approved,
    Denied,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_permission_is_prompt() {
        assert_eq!(PermissionTier::default(), PermissionTier::Prompt);
    }

    #[test]
    fn parse_permission_tiers() {
        assert_eq!(PermissionTier::from_str("auto").unwrap(), PermissionTier::Auto);
        assert_eq!(PermissionTier::from_str("prompt").unwrap(), PermissionTier::Prompt);
        assert_eq!(PermissionTier::from_str("deny").unwrap(), PermissionTier::Deny);
        assert!(PermissionTier::from_str("unknown").is_err());
    }

    #[test]
    fn resolve_tier_defaults_to_prompt() {
        let map = PermissionMap::new();
        assert_eq!(resolve_tier(&map, "anything"), PermissionTier::Prompt);
    }

    #[test]
    fn resolve_tier_finds_auto() {
        let mut map = PermissionMap::new();
        map.insert("file-read".into(), PermissionTier::Auto);
        assert_eq!(resolve_tier(&map, "file-read"), PermissionTier::Auto);
    }
}
