//! Shared types for the AgentOS platform.
//!
//! Contains event types, LLM message protocol types, and permission types
//! that cross module boundaries. This crate is the foundation — imported by
//! agent, llm, librarian, organism, tui, pipeline, and tools.

use std::collections::HashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

// ── Pipeline Events ──

/// Events emitted by the pipeline for observation.
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    /// A message was successfully injected.
    MessageInjected {
        thread_id: String,
        target: String,
        profile: String,
    },
    /// A message was blocked by security policy.
    SecurityBlocked {
        profile: String,
        target: String,
    },
    /// Token usage from an LLM API call.
    TokenUsage {
        thread_id: String,
        input_tokens: u32,
        output_tokens: u32,
    },
    /// A kernel-level operation occurred.
    KernelOp {
        op: KernelOpType,
        thread_id: String,
    },
    /// Semantic router matched a tool.
    SemanticMatch {
        thread_id: String,
        tool_name: String,
        score: f32,
    },
    /// Form-filler attempt.
    FormFillAttempt {
        thread_id: String,
        tool_name: String,
        model: String,
        success: bool,
    },
    /// Coding agent produced a final response.
    AgentResponse {
        thread_id: String,
        agent_name: String,
        text: String,
    },
    /// Agent is about to call the LLM (thinking).
    AgentThinking {
        thread_id: String,
        agent_name: String,
    },
    /// A tool call has been dispatched.
    ToolDispatched {
        thread_id: String,
        agent_name: String,
        tool_name: String,
        detail: String,
    },
    /// A tool call completed (result received).
    ToolCompleted {
        thread_id: String,
        agent_name: String,
        tool_name: String,
        success: bool,
        detail: String,
    },
    /// Conversation state sync — full conversation for a thread (for TUI display).
    ConversationSync {
        thread_id: String,
        agent_name: String,
        entries: Vec<ConversationEntry>,
    },
    /// Tool permission check result (for activity trace).
    ToolApproval {
        thread_id: String,
        agent_name: String,
        tool_name: String,
        verdict: String, // "approved", "denied", "auto", "denied_by_policy"
    },
    /// Suspected prompt injection detected in tool output.
    InjectionDetected {
        thread_id: String,
        tool_name: String,
        agent_name: String,
    },
    /// User allowed suspected injection through (quarantined).
    InjectionAllowed {
        thread_id: String,
        tool_name: String,
        agent_name: String,
    },
    /// User blocked suspected injection — sanitized output sent to agent.
    InjectionBlocked {
        thread_id: String,
        tool_name: String,
        agent_name: String,
    },
    /// Agent sent a display-only message to the user (no response expected).
    UserDisplay {
        thread_id: String,
        agent_name: String,
        text: String,
    },
    /// Agent is asking the user a question (response expected).
    UserQuery {
        thread_id: String,
        agent_name: String,
        question: String,
    },
    /// Interactive buffer child wants TUI focus (child agent taking over).
    FocusAcquire {
        agent_name: String,
        parent_agent: String,
    },
    /// Interactive buffer child released TUI focus (returning to parent).
    FocusRelease {
        agent_name: String,
        parent_agent: String,
    },
}

/// A conversation entry for TUI display (lightweight, no raw API content).
#[derive(Debug, Clone)]
pub struct ConversationEntry {
    /// "user", "assistant", or "tool_result"
    pub role: String,
    /// Truncated text or tool description.
    pub summary: String,
    /// Was this a tool_use block?
    pub is_tool_use: bool,
    /// Tool name if this was a tool_use or tool_result.
    pub tool_name: Option<String>,
    /// Whether this entry represents an error.
    pub is_error: bool,
}

/// Kernel operation types for event reporting.
#[derive(Debug, Clone)]
pub enum KernelOpType {
    ThreadCreated,
    ThreadPruned,
    ContextAllocated,
    ContextReleased,
    ContextFolded,
}

// ── Permission Types ──

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

impl PermissionTier {
    /// Parse a permission tier from a YAML string value.
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

// ── LLM Message Types ──

/// Tool definition sent in the API request.
/// Describes a tool that Claude can invoke.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// A content block in a response (or assistant message).
/// The API returns these as `{"type": "text", ...}` or `{"type": "tool_use", ...}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

/// Message content that handles the three shapes the API uses:
/// - Simple string (for user messages in requests)
/// - Array of ContentBlocks (for assistant responses with tool_use)
/// - Array of ContentBlocks including ToolResult (for user tool_result messages)
///
/// Serializes as a string when it's just text, or as an array of blocks otherwise.
/// Deserializes from either shape.
#[derive(Debug, Clone)]
pub enum MessageContent {
    /// Plain text content (serializes as a JSON string).
    Text(String),
    /// Array of content blocks (serializes as a JSON array).
    Blocks(Vec<ContentBlock>),
}

impl MessageContent {
    /// Get plain text content, concatenating text blocks if needed.
    pub fn text(&self) -> Option<String> {
        match self {
            MessageContent::Text(s) => Some(s.clone()),
            MessageContent::Blocks(blocks) => {
                let texts: Vec<&str> = blocks
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                if texts.is_empty() {
                    None
                } else {
                    Some(texts.join(""))
                }
            }
        }
    }

    /// Check if this content contains any tool_use blocks.
    pub fn has_tool_use(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Blocks(blocks) => {
                blocks.iter().any(|b| matches!(b, ContentBlock::ToolUse { .. }))
            }
        }
    }

    /// Extract all tool_use blocks.
    pub fn tool_use_blocks(&self) -> Vec<&ContentBlock> {
        match self {
            MessageContent::Text(_) => vec![],
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
                .collect(),
        }
    }
}

impl Serialize for MessageContent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            MessageContent::Text(s) => serializer.serialize_str(s),
            MessageContent::Blocks(blocks) => blocks.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for MessageContent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(s) => Ok(MessageContent::Text(s)),
            serde_json::Value::Array(arr) => {
                let blocks: Vec<ContentBlock> =
                    serde_json::from_value(serde_json::Value::Array(arr))
                        .map_err(serde::de::Error::custom)?;
                Ok(MessageContent::Blocks(blocks))
            }
            other => Err(serde::de::Error::custom(format!(
                "expected string or array for message content, got: {}",
                other
            ))),
        }
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

/// A single message in the conversation.
///
/// `content` is polymorphic: plain text for simple messages,
/// content blocks for tool_use responses and tool_result messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

impl Message {
    /// Create a simple text message.
    pub fn text(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: MessageContent::Text(content.to_string()),
        }
    }

    /// Create a user message with tool results.
    pub fn tool_results(results: Vec<ToolResultBlock>) -> Self {
        let blocks = results
            .into_iter()
            .map(|r| ContentBlock::ToolResult {
                tool_use_id: r.tool_use_id,
                content: Some(r.content),
                is_error: if r.is_error { Some(true) } else { None },
            })
            .collect();
        Self {
            role: "user".to_string(),
            content: MessageContent::Blocks(blocks),
        }
    }

    /// Create an assistant message with content blocks (for conversation replay).
    pub fn assistant_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: MessageContent::Blocks(blocks),
        }
    }
}

/// A tool result to be sent back to the API.
#[derive(Debug, Clone)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    pub content: String,
    pub is_error: bool,
}

// ── WASM Capability Types ──

/// Capability grants for a WASM tool component.
///
/// Default is empty — no access to anything.
#[derive(Debug, Clone, Default)]
pub struct WasmCapabilities {
    pub filesystem: Vec<FsGrant>,
    pub env_vars: Vec<EnvGrant>,
    pub stdio: bool,
    /// KV store grants. None = no KV access. Some = private namespace
    /// plus any additional read/write grants to shared namespaces.
    pub kv: Option<KvGrant>,
}

/// KV store access grant for a WASM tool.
#[derive(Debug, Clone)]
pub struct KvGrant {
    /// Shared namespaces this tool can read (in addition to its own private namespace).
    pub read: Vec<String>,
    /// Shared namespaces this tool can write (in addition to its own private namespace).
    pub write: Vec<String>,
}

/// A filesystem access grant.
#[derive(Debug, Clone)]
pub struct FsGrant {
    pub host_path: String,
    pub guest_path: String,
    pub read_only: bool,
}

/// An environment variable grant.
#[derive(Debug, Clone)]
pub struct EnvGrant {
    pub key: String,
    pub value: String,
}
