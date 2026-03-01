//! Context-aware permission framework.
//!
//! Each agent declares per-tool permission tiers in organism YAML.
//! The handler checks permissions before dispatching tool calls:
//! - `Auto` → execute immediately
//! - `Prompt` → pause for user approval via TUI
//! - `Deny` → reject immediately, agent sees error

use std::collections::HashMap;

/// Permission tier for a tool within an agent's context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionTier {
    /// Execute immediately, no user interaction.
    Auto,
    /// Pause, show request in TUI, wait for approve/deny.
    Prompt,
    /// Reject immediately, agent sees error and can explain.
    Deny,
}

impl Default for PermissionTier {
    fn default() -> Self {
        PermissionTier::Prompt
    }
}

/// Parse a permission tier from a YAML string value.
impl PermissionTier {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(PermissionTier::Auto),
            "prompt" => Ok(PermissionTier::Prompt),
            "deny" => Ok(PermissionTier::Deny),
            _ => Err(format!("unknown permission tier: '{s}' (expected auto/prompt/deny)")),
        }
    }
}

/// A set of permission policies for an agent's tools.
pub type PermissionMap = HashMap<String, PermissionTier>;

/// Look up the permission tier for a tool. Unlisted tools default to `Prompt`.
pub fn resolve_tier(permissions: &PermissionMap, tool_name: &str) -> PermissionTier {
    permissions
        .get(tool_name)
        .cloned()
        .unwrap_or(PermissionTier::Prompt)
}

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
        assert_eq!(resolve_tier(&map, "file-read"), PermissionTier::Prompt);
    }

    #[test]
    fn resolve_tier_uses_map() {
        let mut map = PermissionMap::new();
        map.insert("file-read".into(), PermissionTier::Auto);
        map.insert("command-exec".into(), PermissionTier::Deny);
        assert_eq!(resolve_tier(&map, "file-read"), PermissionTier::Auto);
        assert_eq!(resolve_tier(&map, "command-exec"), PermissionTier::Deny);
        assert_eq!(resolve_tier(&map, "file-write"), PermissionTier::Prompt);
    }
}
