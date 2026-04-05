//! CloudProvider trait — the uniform interface every provider must implement.
//!
//! Concrete providers are Rhai scripts that get loaded by the `ScriptedProvider`
//! runtime. This trait exists so Rust code can call providers without caring
//! whether they're native or scripted.

use async_trait::async_trait;

use crate::types::{CloudInstance, GpuOffering, ProvisionRequest};
use crate::CloudError;

/// The uniform cloud provider interface.
///
/// Every provider — RunPod, Lambda, Vast.ai — implements these four operations.
/// The Rhai runtime (`ScriptedProvider`) delegates each call to the corresponding
/// Rhai function in the provider script.
#[async_trait]
pub trait CloudProvider: Send + Sync {
    /// Provider name (e.g. "runpod").
    fn name(&self) -> &str;

    /// Search available GPU offerings.
    ///
    /// `gpu_type`: optional filter (e.g. "A100"), None = all.
    /// `region`: optional region filter, None = all.
    async fn search(
        &self,
        gpu_type: Option<&str>,
        region: Option<&str>,
    ) -> Result<Vec<GpuOffering>, CloudError>;

    /// Provision a new instance.
    ///
    /// Returns the instance immediately (status will be `Provisioning`).
    /// Caller should poll `status()` until it transitions to `Running`.
    async fn provision(&self, request: &ProvisionRequest) -> Result<CloudInstance, CloudError>;

    /// Get current status of an instance.
    async fn status(&self, instance_id: &str) -> Result<CloudInstance, CloudError>;

    /// Tear down an instance. Irreversible.
    async fn teardown(&self, instance_id: &str) -> Result<(), CloudError>;
}
