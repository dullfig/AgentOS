#[cfg(test)]
mod tests {
    use crate::provider::CloudProvider;
    use crate::runtime::{ProviderRegistry, ScriptedProvider};

    /// A minimal mock provider script for testing the Rhai runtime.
    /// Uses no HTTP — returns hardcoded data.
    const MOCK_PROVIDER: &str = r#"
fn search(gpu_type, region) {
    let offerings = [];
    offerings.push(#{
        provider: "mock",
        gpu_type: "A100-80GB",
        gpu_count: 1,
        vram_gb: 80,
        price_per_hour: 1.09,
        region: "us-east",
        available: true,
        offering_id: "a100-80gb:secure"
    });
    if gpu_type != "" {
        // Filter: only return if name contains filter
        let filtered = [];
        for o in offerings {
            if o.gpu_type.to_lower().contains(gpu_type.to_lower()) {
                filtered.push(o);
            }
        }
        return filtered;
    }
    offerings
}

fn provision(config) {
    #{
        instance_id: "pod-abc123",
        provider: "mock",
        status: "provisioning",
        endpoint_url: (),
        ssh_command: (),
        gpu_type: config.gpu_type,
        cost_per_hour: 1.09,
        cost_total: (),
        uptime_secs: ()
    }
}

fn status(instance_id) {
    #{
        instance_id: instance_id,
        provider: "mock",
        status: "running",
        endpoint_url: "https://" + instance_id + "-8080.proxy.runpod.net",
        ssh_command: (),
        gpu_type: "A100-80GB",
        cost_per_hour: 1.09,
        cost_total: 2.18,
        uptime_secs: 7200
    }
}

fn teardown(instance_id) {
    true
}
"#;

    #[test]
    fn load_mock_provider() {
        let provider = ScriptedProvider::from_source("mock", MOCK_PROVIDER, "test-key".into());
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().name(), "mock");
    }

    #[test]
    fn missing_function_rejected() {
        let bad_script = r#"
fn search(gpu_type, region) { [] }
fn provision(config) { #{} }
// missing status and teardown
"#;
        let result = ScriptedProvider::from_source("bad", bad_script, "key".into());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing required function"));
    }

    #[tokio::test]
    async fn search_returns_offerings() {

        let provider = ScriptedProvider::from_source("mock", MOCK_PROVIDER, "key".into()).unwrap();
        let offerings = provider.search(None, None).await.unwrap();
        assert_eq!(offerings.len(), 1);
        assert_eq!(offerings[0].gpu_type, "A100-80GB");
        assert_eq!(offerings[0].vram_gb, 80);
        assert!((offerings[0].price_per_hour - 1.09).abs() < 0.01);
    }

    #[tokio::test]
    async fn search_with_filter() {

        let provider = ScriptedProvider::from_source("mock", MOCK_PROVIDER, "key".into()).unwrap();
        let offerings = provider.search(Some("H100"), None).await.unwrap();
        assert!(offerings.is_empty()); // mock only has A100
    }

    #[tokio::test]
    async fn provision_returns_instance() {
        use crate::provider::CloudProvider;
        use crate::types::ProvisionRequest;

        let provider = ScriptedProvider::from_source("mock", MOCK_PROVIDER, "key".into()).unwrap();
        let req = ProvisionRequest {
            provider: "mock".into(),
            gpu_type: "A100-80GB".into(),
            gpu_count: 1,
            container_image: "dullfig/cortex:latest".into(),
            expose_ports: vec![8080],
            env_vars: vec![],
            volume_mount: None,
            volume_size_gb: None,
            name: "test-pod".into(),
        };
        let instance = provider.provision(&req).await.unwrap();
        assert_eq!(instance.instance_id, "pod-abc123");
        assert_eq!(instance.status, crate::types::InstanceStatus::Provisioning);
    }

    #[tokio::test]
    async fn status_returns_running() {

        let provider = ScriptedProvider::from_source("mock", MOCK_PROVIDER, "key".into()).unwrap();
        let instance = provider.status("pod-abc123").await.unwrap();
        assert_eq!(instance.status, crate::types::InstanceStatus::Running);
        assert!(instance.endpoint_url.is_some());
        assert_eq!(instance.uptime_secs, Some(7200));
    }

    #[tokio::test]
    async fn teardown_succeeds() {

        let provider = ScriptedProvider::from_source("mock", MOCK_PROVIDER, "key".into()).unwrap();
        let result = provider.teardown("pod-abc123").await;
        assert!(result.is_ok());
    }

    #[test]
    fn registry_register_from_source() {
        let mut registry = ProviderRegistry::new();
        let result = registry.register_from_source("mock", MOCK_PROVIDER, "key".into());
        assert!(result.is_ok());
        assert!(registry.get("mock").is_some());
        assert_eq!(registry.list().len(), 1);
    }
}
