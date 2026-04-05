//! Shared types for the cloud provider abstraction.
//!
//! These are the uniform data structures that every provider script must
//! produce, regardless of the underlying API (RunPod GraphQL, Lambda REST, etc.).

use serde::{Deserialize, Serialize};

/// A GPU offering from a cloud provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOffering {
    /// Provider name (e.g. "runpod", "lambda", "vastai").
    pub provider: String,
    /// GPU model (e.g. "A100-80GB", "H100-SXM").
    pub gpu_type: String,
    /// Number of GPUs in this offering.
    pub gpu_count: u32,
    /// VRAM per GPU in GB.
    pub vram_gb: u32,
    /// Price per hour in USD.
    pub price_per_hour: f64,
    /// Region/datacenter (e.g. "US-East", "EU-RO-1").
    pub region: String,
    /// Whether this offering is currently available.
    pub available: bool,
    /// Provider-specific offering ID for provisioning.
    pub offering_id: String,
}

/// Configuration for provisioning a cloud instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvisionRequest {
    /// Which provider to use.
    pub provider: String,
    /// GPU type to request (e.g. "A100-80GB").
    pub gpu_type: String,
    /// Number of GPUs.
    pub gpu_count: u32,
    /// Docker image to deploy (e.g. "dullfig/cortex:latest").
    pub container_image: String,
    /// Ports to expose (e.g. [8080]).
    pub expose_ports: Vec<u16>,
    /// Environment variables for the container.
    pub env_vars: Vec<(String, String)>,
    /// Volume mount path for model weights (provider-dependent).
    pub volume_mount: Option<String>,
    /// Volume size in GB (for network volumes).
    pub volume_size_gb: Option<u32>,
    /// Human-readable name for this instance.
    pub name: String,
}

/// A running cloud instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudInstance {
    /// Provider-assigned instance/pod ID.
    pub instance_id: String,
    /// Provider name.
    pub provider: String,
    /// Current status.
    pub status: InstanceStatus,
    /// Public endpoint URL (once running), e.g. "https://abc123-8080.proxy.runpod.net".
    pub endpoint_url: Option<String>,
    /// SSH connection string (if available).
    pub ssh_command: Option<String>,
    /// GPU type allocated.
    pub gpu_type: String,
    /// Cost per hour in USD.
    pub cost_per_hour: f64,
    /// Total cost accumulated so far in USD.
    pub cost_total: Option<f64>,
    /// Uptime in seconds.
    pub uptime_secs: Option<u64>,
}

/// Instance lifecycle status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstanceStatus {
    /// Being created / image pulling.
    Provisioning,
    /// Running and healthy.
    Running,
    /// Stopped but not destroyed (can resume).
    Stopped,
    /// Being torn down.
    Terminating,
    /// Gone.
    Terminated,
    /// Something went wrong.
    Error,
}

impl std::fmt::Display for InstanceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provisioning => write!(f, "provisioning"),
            Self::Running => write!(f, "running"),
            Self::Stopped => write!(f, "stopped"),
            Self::Terminating => write!(f, "terminating"),
            Self::Terminated => write!(f, "terminated"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// Wire protocol the endpoint speaks — determines which LLM client to use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireProtocol {
    /// OpenAI-compatible /v1/chat/completions
    OpenAi,
    /// Anthropic Messages API /v1/messages
    Anthropic,
}
