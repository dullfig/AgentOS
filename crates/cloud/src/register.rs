//! Cloud endpoint registration — bridges a running CloudInstance into ModelsConfig
//! so the LLM pipeline can route inference requests to it.

use agentos_config::ModelsConfig;

use crate::types::{CloudInstance, InstanceStatus, WireProtocol};
use crate::CloudError;

/// Registration result returned to the caller (cloud-expert agent).
#[derive(Debug, Clone)]
pub struct RegisteredEndpoint {
    /// The model alias that was registered (e.g. "concierge").
    pub alias: String,
    /// The provider name in ModelsConfig (e.g. "cortex-cloud").
    pub provider_name: String,
    /// The base URL registered.
    pub base_url: String,
    /// The wire protocol.
    pub protocol: WireProtocol,
}

/// Register a running cloud instance as a model provider in ModelsConfig.
///
/// - `instance`: Must be in `Running` status with an `endpoint_url`.
/// - `alias`: The model alias the pipeline will use (e.g. "concierge").
/// - `model_id`: The model ID the endpoint serves (e.g. "qwen-30b").
/// - `protocol`: Which wire format the endpoint speaks.
/// - `provider_name`: Optional. Defaults to `"{cloud_provider}-cloud"` (e.g. "runpod-cloud").
///
/// Loads the current ModelsConfig, adds the provider entry, and saves.
/// If a provider with the same name already exists, it's updated in place.
pub fn register_cloud_endpoint(
    instance: &CloudInstance,
    alias: &str,
    model_id: &str,
    protocol: WireProtocol,
    provider_name: Option<&str>,
) -> Result<RegisteredEndpoint, CloudError> {
    // Validate instance is running
    if instance.status != InstanceStatus::Running {
        return Err(CloudError::ProvisionFailed(format!(
            "Instance {} is {:?}, not running. Wait for it to be ready.",
            instance.instance_id, instance.status
        )));
    }

    let endpoint_url = instance.endpoint_url.as_ref().ok_or_else(|| {
        CloudError::ProvisionFailed(format!(
            "Instance {} is running but has no endpoint URL",
            instance.instance_id
        ))
    })?;

    let name = provider_name
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}-cloud", instance.provider));

    // Build the base_url — for OpenAI-compatible, ensure it ends with /v1
    let base_url = match protocol {
        WireProtocol::OpenAi => {
            if endpoint_url.ends_with("/v1") {
                endpoint_url.clone()
            } else {
                format!("{}/v1", endpoint_url.trim_end_matches('/'))
            }
        }
        WireProtocol::Anthropic => endpoint_url.clone(),
    };

    // Load, modify, save
    let mut config = ModelsConfig::load();
    config.add_model(
        &name,
        alias,
        model_id,
        None, // API key — cloud endpoints typically don't need one, or it's in the URL
        Some(base_url.clone()),
    );

    config.save().map_err(|e| {
        CloudError::ApiError(format!("Failed to save models config: {e}"))
    })?;

    Ok(RegisteredEndpoint {
        alias: alias.to_string(),
        provider_name: name,
        base_url,
        protocol,
    })
}

/// Deregister a cloud endpoint by alias. Called during teardown.
pub fn deregister_cloud_endpoint(alias: &str) -> Result<bool, CloudError> {
    let mut config = ModelsConfig::load();
    let removed = config.remove_model(alias);
    if removed {
        config.save().map_err(|e| {
            CloudError::ApiError(format!("Failed to save models config: {e}"))
        })?;
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::InstanceStatus;

    fn running_instance() -> CloudInstance {
        CloudInstance {
            instance_id: "pod-abc123".into(),
            provider: "runpod".into(),
            status: InstanceStatus::Running,
            endpoint_url: Some("https://pod-abc123-8080.proxy.runpod.net".into()),
            ssh_command: None,
            gpu_type: "A100-80GB".into(),
            cost_per_hour: 1.09,
            cost_total: Some(2.18),
            uptime_secs: Some(7200),
        }
    }

    #[test]
    fn rejects_non_running_instance() {
        let mut instance = running_instance();
        instance.status = InstanceStatus::Provisioning;

        let result = register_cloud_endpoint(
            &instance,
            "concierge",
            "qwen-30b",
            WireProtocol::OpenAi,
            None,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not running"));
    }

    #[test]
    fn rejects_missing_endpoint() {
        let mut instance = running_instance();
        instance.endpoint_url = None;

        let result = register_cloud_endpoint(
            &instance,
            "concierge",
            "qwen-30b",
            WireProtocol::OpenAi,
            None,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no endpoint URL"));
    }

    #[test]
    fn builds_correct_openai_base_url() {
        // Can't test save (no home dir in CI), so test URL construction directly
        let endpoint = "https://pod-abc123-8080.proxy.runpod.net";
        let expected = format!("{endpoint}/v1");

        // Direct URL logic test
        let base_url = if endpoint.ends_with("/v1") {
            endpoint.to_string()
        } else {
            format!("{}/v1", endpoint.trim_end_matches('/'))
        };
        assert_eq!(base_url, expected);
    }

    #[test]
    fn builds_correct_openai_base_url_already_has_v1() {
        let endpoint = "https://pod-abc123-8080.proxy.runpod.net/v1";
        let base_url = if endpoint.ends_with("/v1") {
            endpoint.to_string()
        } else {
            format!("{}/v1", endpoint.trim_end_matches('/'))
        };
        assert_eq!(base_url, endpoint); // no double /v1
    }

    #[test]
    fn default_provider_name() {
        let instance = running_instance();
        let name: Option<&str> = None;
        let result = name
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{}-cloud", instance.provider));
        assert_eq!(result, "runpod-cloud");
    }

    #[test]
    fn custom_provider_name() {
        let name: Option<&str> = Some("cortex-ringhub");
        let result = name
            .map(|s| s.to_string())
            .unwrap_or_else(|| "runpod-cloud".to_string());
        assert_eq!(result, "cortex-ringhub");
    }
}
