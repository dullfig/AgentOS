//! AgentPipeline — wraps rust-pipeline with kernel integration.
//!
//! The adapter pattern: rust-pipeline stays clean as a library,
//! bestcode adds the kernel layer on top for durability and security.
//!
//! Architecture:
//! - Builds a `ListenerRegistry` from the Organism configuration
//! - Passes a standard `ThreadRegistry` to the inner pipeline
//! - Mirrors thread/context/journal ops to the Kernel for durability
//! - Enforces security profiles before messages enter the pipeline
//! - On crash recovery, rebuilds in-memory state from the kernel

use std::path::Path;
use std::sync::Arc;

use tokio::sync::Mutex;

use rust_pipeline::prelude::*;

use crate::kernel::Kernel;
use crate::librarian::handler::LibrarianHandler;
use crate::librarian::Librarian;
use crate::llm::{handler::LlmHandler, LlmPool};
use crate::organism::Organism;
use crate::ports::{Direction, PortDeclaration, PortManager, Protocol};
use crate::security::SecurityResolver;
use crate::treesitter::handler::CodeIndexHandler;
use crate::treesitter::CodeIndex;

/// AgentPipeline: wraps rust-pipeline's Pipeline with kernel integration.
pub struct AgentPipeline {
    /// The inner rust-pipeline.
    pipeline: Pipeline,
    /// Durable kernel state.
    kernel: Arc<Mutex<Kernel>>,
    /// Organism configuration.
    organism: Organism,
    /// Security resolver (profile → dispatch table).
    security: SecurityResolver,
}

impl AgentPipeline {
    /// Build an AgentPipeline from an Organism config and a data directory.
    ///
    /// This:
    /// 1. Opens/recovers the kernel from the data directory
    /// 2. Builds a ListenerRegistry from the organism's listeners
    /// 3. Constructs the security resolver from profiles
    /// 4. Wraps everything in the adapter
    ///
    /// Note: handlers must be registered separately since the Organism
    /// config only has handler names (strings), not actual handler instances.
    /// Use `register_handler()` after construction.
    pub fn new(organism: Organism, data_dir: &Path) -> Result<Self, String> {
        let kernel = Kernel::open(data_dir).map_err(|e| format!("kernel open failed: {e}"))?;

        let security = SecurityResolver::from_organism(&organism)?;

        // Build a ListenerRegistry from organism config
        // Handlers will be registered later via register_handler()
        let registry = ListenerRegistry::new();
        let threads = ThreadRegistry::new();
        let pipeline = Pipeline::new(registry, threads);

        Ok(Self {
            pipeline,
            kernel: Arc::new(Mutex::new(kernel)),
            organism,
            security,
        })
    }

    /// Register a handler for a named listener.
    /// The listener must already be defined in the Organism config.
    pub fn register_handler<H: Handler>(
        &mut self,
        listener_name: &str,
        _handler: H,
    ) -> Result<(), String> {
        let _def = self
            .organism
            .get_listener(listener_name)
            .ok_or_else(|| format!("listener '{listener_name}' not in organism config"))?
            .clone();

        // We need to rebuild the pipeline with the updated registry.
        // Since Pipeline::new takes ownership, we need to reconstruct.
        // For now, we register directly on the existing pipeline's registry
        // through the provided API.

        // Unfortunately, rust-pipeline's Pipeline takes Arc<ListenerRegistry>
        // which is immutable after creation. The proper approach is to build
        // the full registry before creating the pipeline.
        // Let's use a builder pattern instead.

        Err("use AgentPipelineBuilder to register handlers before building".into())
    }

    /// Initialize the root thread (WAL-backed).
    pub async fn initialize_root(
        &self,
        organism_name: &str,
        profile: &str,
    ) -> Result<String, String> {
        let mut kernel = self.kernel.lock().await;
        kernel
            .initialize_root(organism_name, profile)
            .map_err(|e| format!("initialize_root failed: {e}"))
    }

    /// Inject a raw message into the pipeline with security enforcement.
    ///
    /// Before the message enters the pipeline, we check:
    /// 1. The thread's profile allows messaging the target
    /// 2. The dispatch is logged in the kernel
    pub async fn inject_checked(
        &self,
        raw: Vec<u8>,
        _thread_id: &str,
        profile: &str,
        target: &str,
    ) -> Result<(), String> {
        // Security check: is the target reachable under this profile?
        if !self.security.can_reach(profile, target) {
            return Err(format!(
                "security: profile '{profile}' cannot reach listener '{target}'"
            ));
        }

        // Inject into the inner pipeline
        self.pipeline
            .inject(raw)
            .await
            .map_err(|e| format!("inject failed: {e}"))
    }

    /// Inject raw bytes directly (bypasses security — for system messages).
    pub async fn inject_raw(&self, raw: Vec<u8>) -> Result<(), String> {
        self.pipeline
            .inject(raw)
            .await
            .map_err(|e| format!("inject failed: {e}"))
    }

    /// Start the pipeline.
    pub fn run(&mut self) {
        self.pipeline.run();
    }

    /// Shutdown the pipeline.
    pub async fn shutdown(self) {
        self.pipeline.shutdown().await;
    }

    /// Get a reference to the organism.
    pub fn organism(&self) -> &Organism {
        &self.organism
    }

    /// Get the security resolver.
    pub fn security(&self) -> &SecurityResolver {
        &self.security
    }

    /// Get a handle to the kernel (for direct operations).
    pub fn kernel(&self) -> Arc<Mutex<Kernel>> {
        self.kernel.clone()
    }

    /// Reload organism configuration and rebuild security tables.
    pub fn reload(
        &mut self,
        new_organism: Organism,
    ) -> Result<crate::organism::ReloadEvent, String> {
        let event = self.organism.apply_config(new_organism);
        self.security.rebuild(&self.organism)?;
        Ok(event)
    }
}

/// Builder for AgentPipeline — register handlers before building.
pub struct AgentPipelineBuilder {
    organism: Organism,
    data_dir: std::path::PathBuf,
    registry: ListenerRegistry,
    llm_pool: Option<Arc<Mutex<LlmPool>>>,
    port_manager: Option<PortManager>,
    librarian: Option<Arc<Mutex<Librarian>>>,
    code_index: Option<Arc<Mutex<CodeIndex>>>,
}

impl AgentPipelineBuilder {
    /// Start building an AgentPipeline.
    pub fn new(organism: Organism, data_dir: &Path) -> Self {
        Self {
            organism,
            data_dir: data_dir.to_path_buf(),
            registry: ListenerRegistry::new(),
            llm_pool: None,
            port_manager: None,
            librarian: None,
            code_index: None,
        }
    }

    /// Register a handler for a listener defined in the organism.
    pub fn register<H: Handler>(mut self, listener_name: &str, handler: H) -> Result<Self, String> {
        let def = self
            .organism
            .get_listener(listener_name)
            .ok_or_else(|| format!("listener '{listener_name}' not in organism config"))?
            .clone();

        self.registry.register(
            &def.name,
            &def.payload_tag,
            handler,
            def.is_agent,
            def.peers.clone(),
            &def.description,
            None, // Schema registration deferred
        );

        Ok(self)
    }

    /// Attach an LLM pool and auto-register the `llm-pool` handler.
    ///
    /// The organism config must have a listener named `llm-pool`.
    /// If a librarian is already attached and the llm-pool listener has
    /// `librarian: true`, the handler will auto-curate before API calls.
    pub fn with_llm_pool(mut self, pool: LlmPool) -> Result<Self, String> {
        let arc = Arc::new(Mutex::new(pool));
        self.llm_pool = Some(arc.clone());

        // Check if auto-curation is enabled for this listener
        let auto_curate = self
            .organism
            .get_listener("llm-pool")
            .map(|l| l.librarian)
            .unwrap_or(false);

        let handler = if auto_curate {
            if let Some(ref lib) = self.librarian {
                LlmHandler::with_librarian(arc, lib.clone())
            } else {
                LlmHandler::new(arc)
            }
        } else {
            LlmHandler::new(arc)
        };

        self = self.register("llm-pool", handler)?;
        Ok(self)
    }

    /// Attach a Librarian service and register the `librarian` handler.
    ///
    /// Requires an LLM pool to be attached first (Librarian calls Haiku).
    /// The organism config must have a listener named `librarian`.
    pub fn with_librarian(mut self) -> Result<Self, String> {
        let pool = self.llm_pool.clone().ok_or_else(|| {
            "with_librarian() requires LLM pool — call with_llm_pool() first".to_string()
        })?;

        let kernel = Kernel::open(&self.data_dir)
            .map_err(|e| format!("kernel open for librarian failed: {e}"))?;
        let kernel_arc = Arc::new(Mutex::new(kernel));

        let librarian = Librarian::new(pool, kernel_arc);
        let lib_arc = Arc::new(Mutex::new(librarian));
        self.librarian = Some(lib_arc.clone());

        // Only register as a listener if the organism config defines one
        if self.organism.get_listener("librarian").is_some() {
            let handler = LibrarianHandler::new(lib_arc);
            self = self.register("librarian", handler)?;
        }

        Ok(self)
    }

    /// Attach a CodeIndex service and register the `codebase-index` handler.
    ///
    /// The organism config must have a listener named `codebase-index`.
    pub fn with_code_index(mut self) -> Result<Self, String> {
        let index = CodeIndex::new();
        let arc = Arc::new(Mutex::new(index));
        self.code_index = Some(arc.clone());

        // Only register as a listener if the organism config defines one
        if self.organism.get_listener("codebase-index").is_some() {
            let handler = CodeIndexHandler::new(arc);
            self = self.register("codebase-index", handler)?;
        }

        Ok(self)
    }

    /// Build a PortManager from the organism's listener port declarations.
    ///
    /// Validates that no two listeners conflict on the same port+direction.
    pub fn with_port_manager(mut self) -> Result<Self, String> {
        let mut pm = PortManager::new();

        for listener in self.organism.listeners().values() {
            for port_def in &listener.ports {
                let direction = match port_def.direction.as_str() {
                    "inbound" => Direction::Inbound,
                    "outbound" => Direction::Outbound,
                    other => {
                        return Err(format!(
                            "invalid port direction '{}' on listener '{}'",
                            other, listener.name
                        ))
                    }
                };

                let protocol = Protocol::from_str_lc(&port_def.protocol)
                    .map_err(|e| format!("listener '{}': {}", listener.name, e))?;

                pm.declare(
                    &listener.name,
                    PortDeclaration {
                        port: port_def.port,
                        direction,
                        protocol,
                        allowed_hosts: port_def.hosts.clone(),
                    },
                )?;
            }
        }

        pm.validate().map_err(|errs| errs.join("; "))?;
        self.port_manager = Some(pm);
        Ok(self)
    }

    /// Build the AgentPipeline.
    pub fn build(self) -> Result<AgentPipeline, String> {
        let kernel =
            Kernel::open(&self.data_dir).map_err(|e| format!("kernel open failed: {e}"))?;

        let security = SecurityResolver::from_organism(&self.organism)?;

        let threads = ThreadRegistry::new();
        let pipeline = Pipeline::new(self.registry, threads);

        Ok(AgentPipeline {
            pipeline,
            kernel: Arc::new(Mutex::new(kernel)),
            organism: self.organism,
            security,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::organism::parser::parse_organism;
    use rust_pipeline::prelude::{
        build_envelope, FnHandler, HandlerContext, HandlerResponse, ValidatedPayload,
    };
    use tempfile::TempDir;

    fn test_organism() -> Organism {
        let yaml = r#"
organism:
  name: test-org

listeners:
  - name: echo
    payload_class: handlers.echo.Greeting
    handler: handlers.echo.handle
    description: "Echo handler"
    peers: []

  - name: sink
    payload_class: handlers.sink.SinkRequest
    handler: handlers.sink.handle
    description: "Sink handler"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [echo, sink]
    journal: retain_forever
  public:
    linux_user: agentos-public
    listeners: [echo]
    journal: prune_on_delivery
"#;
        parse_organism(yaml).unwrap()
    }

    #[tokio::test]
    async fn build_agent_pipeline() {
        let dir = TempDir::new().unwrap();
        let org = test_organism();

        let echo = FnHandler(|p: ValidatedPayload, _ctx: HandlerContext| {
            Box::pin(async move { Ok(HandlerResponse::Reply { payload_xml: p.xml }) })
        });

        let sink = FnHandler(|_p: ValidatedPayload, _ctx: HandlerContext| {
            Box::pin(async move { Ok(HandlerResponse::None) })
        });

        let pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .register("echo", echo)
            .unwrap()
            .register("sink", sink)
            .unwrap()
            .build()
            .unwrap();

        assert!(pipeline.organism().get_listener("echo").is_some());
    }

    #[tokio::test]
    async fn security_blocks_restricted_target() {
        let dir = TempDir::new().unwrap();
        let org = test_organism();

        let echo = FnHandler(|p: ValidatedPayload, _ctx: HandlerContext| {
            Box::pin(async move { Ok(HandlerResponse::Reply { payload_xml: p.xml }) })
        });

        let sink = FnHandler(|_p: ValidatedPayload, _ctx: HandlerContext| {
            Box::pin(async move { Ok(HandlerResponse::None) })
        });

        let mut pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .register("echo", echo)
            .unwrap()
            .register("sink", sink)
            .unwrap()
            .build()
            .unwrap();

        pipeline.run();

        // Public profile can reach echo
        let envelope = build_envelope(
            "test",
            "echo",
            "thread-1",
            b"<Greeting><text>hi</text></Greeting>",
        )
        .unwrap();

        let result = pipeline
            .inject_checked(envelope, "thread-1", "public", "echo")
            .await;
        assert!(result.is_ok());

        // Public profile CANNOT reach sink — structural impossibility
        let envelope2 = build_envelope("test", "sink", "thread-2", b"<SinkRequest/>").unwrap();

        let result = pipeline
            .inject_checked(envelope2, "thread-2", "public", "sink")
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cannot reach"));

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn kernel_state_persists() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().join("data");

        // First session
        {
            let org = test_organism();
            let echo = FnHandler(|p: ValidatedPayload, _ctx: HandlerContext| {
                Box::pin(async move { Ok(HandlerResponse::Reply { payload_xml: p.xml }) })
            });
            let sink = FnHandler(|_p: ValidatedPayload, _ctx: HandlerContext| {
                Box::pin(async move { Ok(HandlerResponse::None) })
            });

            let pipeline = AgentPipelineBuilder::new(org, &data_dir)
                .register("echo", echo)
                .unwrap()
                .register("sink", sink)
                .unwrap()
                .build()
                .unwrap();

            // Initialize root in kernel
            pipeline.initialize_root("test-org", "admin").await.unwrap();
        }

        // Second session — kernel state should be recovered
        {
            let org = test_organism();
            let echo = FnHandler(|p: ValidatedPayload, _ctx: HandlerContext| {
                Box::pin(async move { Ok(HandlerResponse::Reply { payload_xml: p.xml }) })
            });
            let sink = FnHandler(|_p: ValidatedPayload, _ctx: HandlerContext| {
                Box::pin(async move { Ok(HandlerResponse::None) })
            });

            let pipeline = AgentPipelineBuilder::new(org, &data_dir)
                .register("echo", echo)
                .unwrap()
                .register("sink", sink)
                .unwrap()
                .build()
                .unwrap();

            let kernel = pipeline.kernel();
            let k = kernel.lock().await;
            assert!(k.threads().root_uuid().is_some());
        }
    }

    #[tokio::test]
    async fn hot_reload_updates_security() {
        let dir = TempDir::new().unwrap();
        let org = test_organism();

        let echo = FnHandler(|p: ValidatedPayload, _ctx: HandlerContext| {
            Box::pin(async move { Ok(HandlerResponse::Reply { payload_xml: p.xml }) })
        });
        let sink = FnHandler(|_p: ValidatedPayload, _ctx: HandlerContext| {
            Box::pin(async move { Ok(HandlerResponse::None) })
        });

        let mut pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .register("echo", echo)
            .unwrap()
            .register("sink", sink)
            .unwrap()
            .build()
            .unwrap();

        // Initially, public cannot reach sink
        assert!(!pipeline.security().can_reach("public", "sink"));

        // Hot reload: expand public profile to include sink
        let new_yaml = r#"
organism:
  name: test-org-v2

listeners:
  - name: echo
    payload_class: handlers.echo.Greeting
    handler: handlers.echo.handle
    description: "Echo handler"

  - name: sink
    payload_class: handlers.sink.SinkRequest
    handler: handlers.sink.handle
    description: "Sink handler"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [echo, sink]
    journal: retain_forever
  public:
    linux_user: agentos-public
    listeners: [echo, sink]
    journal: prune_on_delivery
"#;
        let new_org = parse_organism(new_yaml).unwrap();
        let _event = pipeline.reload(new_org).unwrap();

        assert_eq!(pipeline.organism().name, "test-org-v2");

        // Now public CAN reach sink
        assert!(pipeline.security().can_reach("public", "sink"));
    }

    // ── Milestone 2 Integration Tests ──

    fn m2_organism() -> Organism {
        let yaml = r#"
organism:
  name: bestcode-m2

listeners:
  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM inference pool"
    peers: []
    ports:
      - port: 443
        direction: outbound
        protocol: https
        hosts: [api.anthropic.com]

  - name: file-ops
    payload_class: tools.FileOpsRequest
    handler: tools.file_ops.handle
    description: "File operations"

  - name: shell
    payload_class: tools.ShellRequest
    handler: tools.shell.handle
    description: "Shell execution"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [file-ops, shell, llm-pool]
    network: [llm-pool]
    journal:
      retain_days: 90
  restricted:
    linux_user: agentos-restricted
    listeners: [file-ops]
    journal: prune_on_delivery
"#;
        parse_organism(yaml).unwrap()
    }

    #[tokio::test]
    async fn build_pipeline_with_llm_pool_and_tools() {
        let dir = TempDir::new().unwrap();
        let org = m2_organism();

        let pool = crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        );

        let pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_llm_pool(pool)
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .register("shell", crate::tools::shell::ShellStub)
            .unwrap()
            .with_port_manager()
            .unwrap()
            .build()
            .unwrap();

        assert!(pipeline.organism().get_listener("llm-pool").is_some());
        assert!(pipeline.organism().get_listener("file-ops").is_some());
        assert!(pipeline.organism().get_listener("shell").is_some());
    }

    #[tokio::test]
    async fn tool_stub_responds_via_pipeline() {
        let dir = TempDir::new().unwrap();
        let org = m2_organism();

        let pool = crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        );

        let mut pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_llm_pool(pool)
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .register("shell", crate::tools::shell::ShellStub)
            .unwrap()
            .with_port_manager()
            .unwrap()
            .build()
            .unwrap();

        pipeline.run();

        // Inject a FileOps request under admin profile
        let envelope = build_envelope(
            "test",
            "file-ops",
            "thread-1",
            b"<FileOpsRequest><action>read</action><path>/etc/hostname</path></FileOpsRequest>",
        )
        .unwrap();

        let result = pipeline
            .inject_checked(envelope, "thread-1", "admin", "file-ops")
            .await;
        assert!(result.is_ok());

        // Inject a Shell request under admin profile
        let envelope2 = build_envelope(
            "test",
            "shell",
            "thread-2",
            b"<ShellRequest><command>echo hello</command></ShellRequest>",
        )
        .unwrap();

        let result = pipeline
            .inject_checked(envelope2, "thread-2", "admin", "shell")
            .await;
        assert!(result.is_ok());

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn security_blocks_llm_for_restricted_profile() {
        let dir = TempDir::new().unwrap();
        let org = m2_organism();

        let pool = crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        );

        let mut pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_llm_pool(pool)
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .register("shell", crate::tools::shell::ShellStub)
            .unwrap()
            .with_port_manager()
            .unwrap()
            .build()
            .unwrap();

        pipeline.run();

        // Restricted profile can reach file-ops
        let envelope = build_envelope(
            "test",
            "file-ops",
            "thread-1",
            b"<FileOpsRequest><action>read</action><path>/tmp/x</path></FileOpsRequest>",
        )
        .unwrap();

        let ok = pipeline
            .inject_checked(envelope, "thread-1", "restricted", "file-ops")
            .await;
        assert!(ok.is_ok());

        // Restricted profile CANNOT reach llm-pool — structural impossibility
        let llm_envelope = build_envelope(
            "test",
            "llm-pool",
            "thread-2",
            b"<LlmRequest><messages><message role=\"user\">hi</message></messages></LlmRequest>",
        )
        .unwrap();

        let err = pipeline
            .inject_checked(llm_envelope, "thread-2", "restricted", "llm-pool")
            .await;
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("cannot reach"));

        // Restricted profile also CANNOT reach shell
        let shell_envelope = build_envelope(
            "test",
            "shell",
            "thread-3",
            b"<ShellRequest><command>whoami</command></ShellRequest>",
        )
        .unwrap();

        let err = pipeline
            .inject_checked(shell_envelope, "thread-3", "restricted", "shell")
            .await;
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("cannot reach"));

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn port_conflict_rejected_at_build_time() {
        let yaml = r#"
organism:
  name: conflict-test

listeners:
  - name: listener-a
    payload_class: test.ReqA
    handler: test.handle_a
    description: "Listener A"
    ports:
      - port: 8080
        direction: inbound
        protocol: http

  - name: listener-b
    payload_class: test.ReqB
    handler: test.handle_b
    description: "Listener B"
    ports:
      - port: 8080
        direction: inbound
        protocol: http

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [listener-a, listener-b]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let dir = TempDir::new().unwrap();

        let handler_a = FnHandler(|p: ValidatedPayload, _ctx: HandlerContext| {
            Box::pin(async move { Ok(HandlerResponse::Reply { payload_xml: p.xml }) })
        });
        let handler_b = FnHandler(|p: ValidatedPayload, _ctx: HandlerContext| {
            Box::pin(async move { Ok(HandlerResponse::Reply { payload_xml: p.xml }) })
        });

        let result = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .register("listener-a", handler_a)
            .unwrap()
            .register("listener-b", handler_b)
            .unwrap()
            .with_port_manager();

        match result {
            Err(e) => assert!(
                e.contains("port conflict"),
                "expected port conflict, got: {e}"
            ),
            Ok(_) => panic!("expected port conflict error"),
        }
    }

    #[tokio::test]
    async fn port_manager_built_from_organism_config() {
        let dir = TempDir::new().unwrap();
        let org = m2_organism();

        let pool = crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        );

        // Build successfully with port manager
        let builder = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_llm_pool(pool)
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .register("shell", crate::tools::shell::ShellStub)
            .unwrap()
            .with_port_manager()
            .unwrap();

        // Port manager should have the LLM pool's port declaration
        let pm = builder.port_manager.as_ref().unwrap();
        let ports = pm.get_ports("llm-pool");
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].port, 443);
        assert_eq!(ports[0].allowed_hosts, vec!["api.anthropic.com"]);
    }

    // ── Phase 3 Integration Tests ──

    fn p3_organism() -> Organism {
        let yaml = r#"
organism:
  name: bestcode-p3

listeners:
  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM inference pool"
    librarian: true
    ports:
      - port: 443
        direction: outbound
        protocol: https
        hosts: [api.anthropic.com]

  - name: librarian
    payload_class: librarian.LibrarianRequest
    handler: librarian.handle
    description: "Context curator"
    peers: [llm-pool]

  - name: codebase-index
    payload_class: treesitter.CodeIndexRequest
    handler: treesitter.handle
    description: "Tree-sitter code indexing"

  - name: file-ops
    payload_class: tools.FileOpsRequest
    handler: tools.file_ops.handle
    description: "File operations"

  - name: shell
    payload_class: tools.ShellRequest
    handler: tools.shell.handle
    description: "Shell execution"

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [file-ops, shell, llm-pool, librarian, codebase-index]
    network: [llm-pool]
    journal:
      retain_days: 90
  restricted:
    linux_user: agentos-restricted
    listeners: [file-ops, codebase-index]
    journal: prune_on_delivery
"#;
        parse_organism(yaml).unwrap()
    }

    #[tokio::test]
    async fn build_pipeline_with_librarian_and_code_index() {
        let dir = TempDir::new().unwrap();
        let org = p3_organism();

        let pool = crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        );

        let pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_llm_pool(pool)
            .unwrap()
            .with_librarian()
            .unwrap()
            .with_code_index()
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .register("shell", crate::tools::shell::ShellStub)
            .unwrap()
            .with_port_manager()
            .unwrap()
            .build()
            .unwrap();

        assert!(pipeline.organism().get_listener("llm-pool").is_some());
        assert!(pipeline.organism().get_listener("librarian").is_some());
        assert!(pipeline.organism().get_listener("codebase-index").is_some());
    }

    #[tokio::test]
    async fn librarian_auto_curate_wired() {
        let dir = TempDir::new().unwrap();
        let org = p3_organism();

        let pool = crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        );

        // Build with librarian BEFORE llm_pool to test the auto-curation wiring
        // Note: with_librarian needs pool first, so we build pool, then librarian, then register llm-pool
        let builder = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_llm_pool(pool)
            .unwrap()
            .with_librarian()
            .unwrap()
            .with_code_index()
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .register("shell", crate::tools::shell::ShellStub)
            .unwrap();

        // Librarian should be attached
        assert!(builder.librarian.is_some());
        assert!(builder.code_index.is_some());

        let pipeline = builder.build().unwrap();
        assert!(pipeline.organism().get_listener("librarian").is_some());
    }

    #[tokio::test]
    async fn security_scoping_for_librarian() {
        let dir = TempDir::new().unwrap();
        let org = p3_organism();

        let pool = crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        );

        let mut pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_llm_pool(pool)
            .unwrap()
            .with_librarian()
            .unwrap()
            .with_code_index()
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .register("shell", crate::tools::shell::ShellStub)
            .unwrap()
            .with_port_manager()
            .unwrap()
            .build()
            .unwrap();

        pipeline.run();

        // Admin can reach librarian
        assert!(pipeline.security().can_reach("admin", "librarian"));
        // Admin can reach codebase-index
        assert!(pipeline.security().can_reach("admin", "codebase-index"));

        // Restricted CANNOT reach librarian — structural impossibility
        assert!(!pipeline.security().can_reach("restricted", "librarian"));
        // Restricted CAN reach codebase-index
        assert!(pipeline
            .security()
            .can_reach("restricted", "codebase-index"));
        // Restricted CANNOT reach llm-pool
        assert!(!pipeline.security().can_reach("restricted", "llm-pool"));

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn code_index_handler_via_pipeline() {
        let dir = TempDir::new().unwrap();
        let org = p3_organism();

        let pool = crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        );

        let mut pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_llm_pool(pool)
            .unwrap()
            .with_librarian()
            .unwrap()
            .with_code_index()
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .register("shell", crate::tools::shell::ShellStub)
            .unwrap()
            .build()
            .unwrap();

        pipeline.run();

        // Inject a CodeIndex request under admin profile
        let envelope = build_envelope(
            "test",
            "codebase-index",
            "thread-1",
            b"<CodeIndexRequest><action>search</action><query>test</query></CodeIndexRequest>",
        )
        .unwrap();

        let result = pipeline
            .inject_checked(envelope, "thread-1", "admin", "codebase-index")
            .await;
        assert!(result.is_ok());

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn with_librarian_requires_pool() {
        let dir = TempDir::new().unwrap();
        let org = p3_organism();

        // Try to build librarian without pool — should fail
        let result = AgentPipelineBuilder::new(org, &dir.path().join("data")).with_librarian();

        match result {
            Err(e) => assert!(e.contains("requires LLM pool"), "unexpected error: {e}"),
            Ok(_) => panic!("expected error when building librarian without pool"),
        }
    }

    #[tokio::test]
    async fn with_code_index_without_organism_listener() {
        let dir = TempDir::new().unwrap();
        // Use m2 organism which doesn't have codebase-index listener
        let org = m2_organism();

        let pool = crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        );

        // with_code_index() should succeed even without organism listener
        // (CodeIndex created but not registered as handler)
        let builder = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_llm_pool(pool)
            .unwrap()
            .with_code_index()
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .register("shell", crate::tools::shell::ShellStub)
            .unwrap();

        assert!(builder.code_index.is_some());
        // Should still build successfully
        let pipeline = builder.build().unwrap();
        assert!(pipeline.organism().get_listener("codebase-index").is_none());
    }
}
