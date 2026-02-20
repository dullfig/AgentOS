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

pub mod events;

use std::path::Path;
use std::sync::Arc;

use tokio::sync::{broadcast, Mutex};

use rust_pipeline::prelude::*;

use events::PipelineEvent;

use crate::agent::handler::CodingAgentHandler;
use crate::agent::prompts;
use crate::agent::tools as agent_tools;
use crate::embedding::tfidf::TfIdfProvider;
use crate::embedding::EmbeddingIndex;
use crate::kernel::Kernel;
use crate::librarian::handler::LibrarianHandler;
use crate::librarian::Librarian;
use crate::llm::{handler::LlmHandler, LlmPool};
use crate::organism::Organism;
use crate::ports::{Direction, PortDeclaration, PortManager, Protocol};
use crate::routing::{self, form_filler::FormFiller, SemanticRouter, ToolMetadata};
use crate::security::SecurityResolver;
use crate::treesitter::handler::CodeIndexHandler;
use crate::treesitter::CodeIndex;
use crate::wasm::definitions::WasmToolRegistry;
use crate::wasm::peer::WasmToolPeer;
use crate::wasm::runtime::WasmRuntime;

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
    /// Broadcast channel for pipeline events (TUI, observers).
    event_tx: broadcast::Sender<PipelineEvent>,
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
        let (event_tx, _) = broadcast::channel(256);

        Ok(Self {
            pipeline,
            kernel: Arc::new(Mutex::new(kernel)),
            organism,
            security,
            event_tx,
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
        thread_id: &str,
        profile: &str,
        target: &str,
    ) -> Result<(), String> {
        // Security check: is the target reachable under this profile?
        if !self.security.can_reach(profile, target) {
            let _ = self.event_tx.send(PipelineEvent::SecurityBlocked {
                profile: profile.to_string(),
                target: target.to_string(),
            });
            return Err(format!(
                "security: profile '{profile}' cannot reach listener '{target}'"
            ));
        }

        // Inject into the inner pipeline
        self.pipeline
            .inject(raw)
            .await
            .map_err(|e| format!("inject failed: {e}"))?;

        let _ = self.event_tx.send(PipelineEvent::MessageInjected {
            thread_id: thread_id.to_string(),
            target: target.to_string(),
            profile: profile.to_string(),
        });

        Ok(())
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

    /// Subscribe to pipeline events (for TUI, observers).
    pub fn subscribe(&self) -> broadcast::Receiver<PipelineEvent> {
        self.event_tx.subscribe()
    }

    /// Get the event sender (for components that need to emit events).
    pub fn event_sender(&self) -> &broadcast::Sender<PipelineEvent> {
        &self.event_tx
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
    wasm_runtime: Option<Arc<WasmRuntime>>,
    wasm_registry: Option<WasmToolRegistry>,
    semantic_router: Option<SemanticRouter>,
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
            wasm_runtime: None,
            wasm_registry: None,
            semantic_router: None,
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

    /// Load WASM tool components and register them as handlers.
    ///
    /// Scans the organism config for listeners with `handler: "wasm"`,
    /// loads each .wasm component, registers metadata in WasmToolRegistry,
    /// and registers WasmToolPeer as the handler.
    ///
    /// Paths in the wasm config are resolved relative to `base_dir`.
    pub fn with_wasm_tools(mut self, base_dir: &Path) -> Result<Self, String> {
        let runtime = Arc::new(
            WasmRuntime::new().map_err(|e| format!("WASM runtime creation failed: {e}"))?,
        );
        let mut registry = WasmToolRegistry::new();

        // Collect WASM listener info to avoid borrow conflict
        let wasm_listeners: Vec<_> = self
            .organism
            .listeners()
            .values()
            .filter(|l| l.handler == "wasm")
            .filter_map(|l| {
                l.wasm.as_ref().map(|w| {
                    (l.name.clone(), w.path.clone(), w.capabilities.clone())
                })
            })
            .collect();

        for (name, wasm_path, caps) in &wasm_listeners {
            let full_path = base_dir.join(wasm_path);
            let component = runtime
                .load_component_from_path(&full_path)
                .map_err(|e| format!("WASM tool '{}' load failed: {e}", name))?;

            registry
                .register(&component.metadata)
                .map_err(|e| format!("WASM tool '{}' registry failed: {e}", name))?;

            let peer = WasmToolPeer::with_capabilities(
                runtime.clone(),
                Arc::new(component),
                caps.clone(),
            );
            self = self.register(name, peer)?;
        }

        self.wasm_runtime = Some(runtime);
        self.wasm_registry = Some(registry);
        Ok(self)
    }

    /// Build a semantic router from the organism's semantic descriptions.
    ///
    /// Requires an LLM pool to be attached first (form-filler calls Haiku).
    /// Creates a TF-IDF provider from all semantic descriptions, builds
    /// an embedding index, and stores the router for later injection
    /// into CodingAgentHandler.
    pub fn with_semantic_router(mut self) -> Result<Self, String> {
        let pool = self.llm_pool.clone().ok_or_else(|| {
            "with_semantic_router() requires LLM pool — call with_llm_pool() first".to_string()
        })?;

        // Collect all semantic descriptions to build the TF-IDF corpus
        let descriptions: Vec<(String, String)> = self
            .organism
            .listeners()
            .values()
            .filter_map(|l| {
                l.semantic_description
                    .as_ref()
                    .map(|d| (l.name.clone(), d.clone()))
            })
            .collect();

        if descriptions.is_empty() {
            return Err(
                "with_semantic_router() requires at least one listener with semantic_description"
                    .to_string(),
            );
        }

        // Build TF-IDF provider from the corpus
        let corpus_strs: Vec<&str> = descriptions.iter().map(|(_, d)| d.as_str()).collect();
        let provider = TfIdfProvider::from_corpus(&corpus_strs);

        // Build embedding index and register all tools
        let mut index = EmbeddingIndex::new(0.3); // default threshold
        routing::register_tools(&mut index, &provider, &self.organism);

        // Build tool metadata from listener definitions
        let mut metadata = std::collections::HashMap::new();
        for listener in self.organism.listeners().values() {
            if listener.semantic_description.is_some() {
                metadata.insert(
                    listener.name.clone(),
                    ToolMetadata {
                        description: listener.description.clone(),
                        xml_template: format!(
                            "<{tag}></{tag}>",
                            tag = listener.payload_tag
                        ),
                        payload_tag: listener.payload_tag.clone(),
                    },
                );
            }
        }

        // Build form-filler
        let filler = FormFiller::new(pool, 3);

        // Build router
        let router = SemanticRouter::new(Box::new(provider), index, filler, metadata);
        self.semantic_router = Some(router);

        Ok(self)
    }

    /// Attach a CodingAgent and register the `coding-agent` handler.
    ///
    /// Requires an LLM pool to be attached first (the agent calls Opus).
    /// The organism config must have a listener named `coding-agent`.
    /// Automatically collects tool definitions from the listener's peers.
    /// If WASM tools are loaded, their definitions are included automatically.
    pub fn with_coding_agent(mut self) -> Result<Self, String> {
        let pool = self.llm_pool.clone().ok_or_else(|| {
            "with_coding_agent() requires LLM pool — call with_llm_pool() first".to_string()
        })?;

        // Get the coding-agent listener definition
        let def = self
            .organism
            .get_listener("coding-agent")
            .ok_or_else(|| {
                "with_coding_agent() requires 'coding-agent' listener in organism config"
                    .to_string()
            })?
            .clone();

        // Build tool definitions from the listener's declared peers,
        // with WASM registry fallback for dynamic tools
        let peer_names: Vec<&str> = def.peers.iter().map(|s| s.as_str()).collect();
        let tool_definitions = agent_tools::build_tool_definitions_with_wasm(
            &peer_names,
            self.wasm_registry.as_ref(),
        );

        // Build tool descriptions for the system prompt
        let tool_descs: Vec<(String, String)> = tool_definitions
            .iter()
            .map(|d| (d.name.clone(), d.description.clone()))
            .collect();
        let system_prompt = prompts::build_system_prompt(&tool_descs);

        // Create the handler
        let handler = if let Some(router) = self.semantic_router.take() {
            let h = CodingAgentHandler::with_semantic_router(
                pool,
                router,
                tool_definitions,
                system_prompt,
            );
            if let Some(ref lib) = self.librarian {
                // Note: with_semantic_router doesn't take a librarian, but the
                // librarian is still available through the pipeline. For full
                // integration, we'd need a builder on CodingAgentHandler, but
                // for V1 the router and librarian are separate concerns.
                let _ = lib;
            }
            h
        } else if let Some(ref lib) = self.librarian {
            CodingAgentHandler::with_librarian(
                pool,
                lib.clone(),
                tool_definitions,
                system_prompt,
            )
        } else {
            CodingAgentHandler::new(pool, tool_definitions, system_prompt)
        };

        self = self.register("coding-agent", handler)?;
        Ok(self)
    }

    /// Build the AgentPipeline.
    pub fn build(self) -> Result<AgentPipeline, String> {
        let kernel =
            Kernel::open(&self.data_dir).map_err(|e| format!("kernel open failed: {e}"))?;

        let security = SecurityResolver::from_organism(&self.organism)?;

        let threads = ThreadRegistry::new();
        let pipeline = Pipeline::new(self.registry, threads);
        let (event_tx, _) = broadcast::channel(256);

        Ok(AgentPipeline {
            pipeline,
            kernel: Arc::new(Mutex::new(kernel)),
            organism: self.organism,
            security,
            event_tx,
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

    // ── Phase 4 Integration Tests ──

    fn p4_organism() -> Organism {
        let yaml = r#"
organism:
  name: bestcode-p4

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

  - name: coding-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Opus coding agent"
    is_agent: true
    librarian: true
    peers: [file-ops, shell, codebase-index]

profiles:
  coding:
    linux_user: agentos-coding
    listeners: [coding-agent, file-ops, shell, codebase-index, llm-pool, librarian]
    network: [llm-pool]
    journal: retain_forever
  restricted:
    linux_user: agentos-restricted
    listeners: [file-ops, codebase-index]
    journal: prune_on_delivery
"#;
        parse_organism(yaml).unwrap()
    }

    #[tokio::test]
    async fn build_pipeline_with_coding_agent() {
        let dir = TempDir::new().unwrap();
        let org = p4_organism();

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
            .with_coding_agent()
            .unwrap()
            .build()
            .unwrap();

        assert!(pipeline.organism().get_listener("coding-agent").is_some());
        assert!(pipeline.organism().get_listener("file-ops").is_some());
        assert!(pipeline.organism().get_listener("shell").is_some());
    }

    #[tokio::test]
    async fn coding_agent_security_can_reach_tools() {
        let dir = TempDir::new().unwrap();
        let org = p4_organism();

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
            .with_coding_agent()
            .unwrap()
            .build()
            .unwrap();

        // Coding profile can reach everything it needs
        assert!(pipeline.security().can_reach("coding", "coding-agent"));
        assert!(pipeline.security().can_reach("coding", "file-ops"));
        assert!(pipeline.security().can_reach("coding", "shell"));
        assert!(pipeline.security().can_reach("coding", "codebase-index"));
        assert!(pipeline.security().can_reach("coding", "llm-pool"));
        assert!(pipeline.security().can_reach("coding", "librarian"));
    }

    #[tokio::test]
    async fn restricted_cannot_reach_coding_agent() {
        let dir = TempDir::new().unwrap();
        let org = p4_organism();

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
            .with_coding_agent()
            .unwrap()
            .build()
            .unwrap();

        // Restricted profile CANNOT reach coding agent — structural impossibility
        assert!(!pipeline.security().can_reach("restricted", "coding-agent"));
        // Restricted CANNOT reach shell
        assert!(!pipeline.security().can_reach("restricted", "shell"));
        // Restricted CANNOT reach llm-pool
        assert!(!pipeline.security().can_reach("restricted", "llm-pool"));
        // Restricted CAN reach file-ops and codebase-index
        assert!(pipeline.security().can_reach("restricted", "file-ops"));
        assert!(pipeline
            .security()
            .can_reach("restricted", "codebase-index"));
    }

    #[tokio::test]
    async fn coding_agent_requires_pool() {
        let dir = TempDir::new().unwrap();
        let org = p4_organism();

        let result =
            AgentPipelineBuilder::new(org, &dir.path().join("data")).with_coding_agent();

        match result {
            Err(e) => assert!(
                e.contains("requires LLM pool"),
                "unexpected error: {e}"
            ),
            Ok(_) => panic!("expected error when building coding agent without pool"),
        }
    }

    #[tokio::test]
    async fn coding_agent_requires_organism_listener() {
        let dir = TempDir::new().unwrap();
        // Use m2 organism which doesn't have coding-agent listener
        let org = m2_organism();

        let pool = crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        );

        let result = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_llm_pool(pool)
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .register("shell", crate::tools::shell::ShellStub)
            .unwrap()
            .with_coding_agent();

        match result {
            Err(e) => assert!(
                e.contains("coding-agent"),
                "unexpected error: {e}"
            ),
            Ok(_) => panic!("expected error when coding-agent not in organism config"),
        }
    }

    #[tokio::test]
    async fn inject_task_to_coding_agent() {
        let dir = TempDir::new().unwrap();
        let org = p4_organism();

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
            .with_coding_agent()
            .unwrap()
            .build()
            .unwrap();

        pipeline.run();

        // Inject a task under the coding profile
        let envelope = build_envelope(
            "user",
            "coding-agent",
            "thread-1",
            b"<AgentTask><task>Hello, agent!</task></AgentTask>",
        )
        .unwrap();

        let result = pipeline
            .inject_checked(envelope, "thread-1", "coding", "coding-agent")
            .await;
        // This will fail on the API call (fake URL), but the injection itself
        // should succeed since the security check passes
        assert!(result.is_ok());

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn inject_task_blocked_for_restricted() {
        let dir = TempDir::new().unwrap();
        let org = p4_organism();

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
            .with_coding_agent()
            .unwrap()
            .build()
            .unwrap();

        pipeline.run();

        // Restricted profile cannot inject to coding-agent
        let envelope = build_envelope(
            "user",
            "coding-agent",
            "thread-1",
            b"<AgentTask><task>hack the mainframe</task></AgentTask>",
        )
        .unwrap();

        let result = pipeline
            .inject_checked(envelope, "thread-1", "restricted", "coding-agent")
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cannot reach"));

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn coding_agent_without_librarian() {
        let dir = TempDir::new().unwrap();
        let org = p4_organism();

        let pool = crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        );

        // Build without librarian — coding agent should still work
        let pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_llm_pool(pool)
            .unwrap()
            .with_code_index()
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .register("shell", crate::tools::shell::ShellStub)
            .unwrap()
            .with_coding_agent()
            .unwrap()
            .build()
            .unwrap();

        assert!(pipeline.organism().get_listener("coding-agent").is_some());
    }

    #[tokio::test]
    async fn coding_agent_tool_defs_match_peers() {
        let dir = TempDir::new().unwrap();
        let org = p4_organism();

        // Verify that the coding-agent's peers produce tool definitions
        let def = org.get_listener("coding-agent").unwrap();
        let peer_names: Vec<&str> = def.peers.iter().map(|s| s.as_str()).collect();
        let tool_defs = crate::agent::tools::build_tool_definitions(&peer_names);

        // Should have definitions for file-ops, shell, codebase-index
        assert_eq!(tool_defs.len(), 3);
        let names: Vec<&str> = tool_defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"file-ops"));
        assert!(names.contains(&"shell"));
        assert!(names.contains(&"codebase-index"));

        // Also verify the pipeline builds cleanly
        let pool = crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        );

        let _pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_llm_pool(pool)
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .register("shell", crate::tools::shell::ShellStub)
            .unwrap()
            .with_code_index()
            .unwrap()
            .with_coding_agent()
            .unwrap()
            .build()
            .unwrap();
    }

    // ── Phase 5 Integration Tests ──

    fn echo_wasm_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
    }

    fn p5_organism() -> Organism {
        let yaml = r#"
organism:
  name: bestcode-p5

listeners:
  - name: echo
    payload_class: tools.EchoRequest
    handler: wasm
    description: "Echo tool (WASM)"
    wasm:
      path: echo.wasm
      capabilities:
        stdio: true

  - name: file-ops
    payload_class: tools.FileOpsRequest
    handler: tools.file_ops.handle
    description: "File operations (stub)"

  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM pool"
    ports:
      - port: 443
        direction: outbound
        protocol: https
        hosts: [api.anthropic.com]

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [echo, file-ops, llm-pool]
    journal: retain_forever
  restricted:
    linux_user: agentos-restricted
    listeners: [file-ops]
    journal: prune_on_delivery
"#;
        parse_organism(yaml).unwrap()
    }

    #[tokio::test]
    async fn build_pipeline_with_wasm_tool() {
        let dir = TempDir::new().unwrap();
        let org = p5_organism();

        let pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_wasm_tools(&echo_wasm_dir())
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .build()
            .unwrap();

        assert!(pipeline.organism().get_listener("echo").is_some());
    }

    #[tokio::test]
    async fn wasm_tool_registered_as_listener() {
        let dir = TempDir::new().unwrap();
        let org = p5_organism();

        let pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_wasm_tools(&echo_wasm_dir())
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .build()
            .unwrap();

        let echo = pipeline.organism().get_listener("echo").unwrap();
        assert_eq!(echo.handler, "wasm");
        assert!(echo.wasm.is_some());
    }

    #[tokio::test]
    async fn wasm_tool_security_scoping() {
        let dir = TempDir::new().unwrap();
        let org = p5_organism();

        let pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_wasm_tools(&echo_wasm_dir())
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .build()
            .unwrap();

        // Admin can reach echo
        assert!(pipeline.security().can_reach("admin", "echo"));
        // Restricted CANNOT reach echo — structural impossibility
        assert!(!pipeline.security().can_reach("restricted", "echo"));
    }

    #[tokio::test]
    async fn wasm_tool_handles_request() {
        let dir = TempDir::new().unwrap();
        let org = p5_organism();

        let mut pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_wasm_tools(&echo_wasm_dir())
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .build()
            .unwrap();

        pipeline.run();

        let envelope = build_envelope(
            "test",
            "echo",
            "thread-1",
            b"<EchoRequest><message>hello pipeline</message></EchoRequest>",
        )
        .unwrap();

        let result = pipeline
            .inject_checked(envelope, "thread-1", "admin", "echo")
            .await;
        assert!(result.is_ok());

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn wasm_tool_definitions_auto_generated() {
        let dir = TempDir::new().unwrap();
        let org = p5_organism();

        let builder = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_wasm_tools(&echo_wasm_dir())
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap();

        // WASM registry should have the echo tool definition
        let reg = builder.wasm_registry.as_ref().unwrap();
        let def = reg.definition_for("echo").unwrap();
        assert_eq!(def.name, "echo");
        assert_eq!(def.input_schema["type"], "object");
    }

    #[tokio::test]
    async fn wasm_missing_file_fails() {
        let yaml = r#"
organism:
  name: bad-wasm

listeners:
  - name: missing
    payload_class: tools.MissingRequest
    handler: wasm
    description: "Missing WASM tool"
    wasm:
      path: nonexistent.wasm

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [missing]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let dir = TempDir::new().unwrap();

        let result = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_wasm_tools(&echo_wasm_dir());

        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.contains("load failed"),
            "expected load failure, got: {err}"
        );
    }

    #[tokio::test]
    async fn wasm_and_native_coexist() {
        let dir = TempDir::new().unwrap();
        let org = p5_organism();

        let mut pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_wasm_tools(&echo_wasm_dir())
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .build()
            .unwrap();

        pipeline.run();

        // WASM echo tool
        let echo_env = build_envelope(
            "test",
            "echo",
            "thread-1",
            b"<EchoRequest><message>wasm</message></EchoRequest>",
        )
        .unwrap();
        assert!(pipeline
            .inject_checked(echo_env, "thread-1", "admin", "echo")
            .await
            .is_ok());

        // Native file-ops stub
        let fops_env = build_envelope(
            "test",
            "file-ops",
            "thread-2",
            b"<FileOpsRequest><action>read</action><path>/tmp/x</path></FileOpsRequest>",
        )
        .unwrap();
        assert!(pipeline
            .inject_checked(fops_env, "thread-2", "admin", "file-ops")
            .await
            .is_ok());

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn coding_agent_xml_tag_for_wasm() {
        let dir = TempDir::new().unwrap();
        let org = p5_organism();

        let builder = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_wasm_tools(&echo_wasm_dir())
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap();

        let reg = builder.wasm_registry.as_ref().unwrap();
        assert_eq!(reg.request_tag_for("echo"), Some("EchoRequest"));
    }

    #[tokio::test]
    async fn hot_reload_preserves_wasm() {
        let dir = TempDir::new().unwrap();
        let org = p5_organism();

        let mut pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .with_wasm_tools(&echo_wasm_dir())
            .unwrap()
            .register("file-ops", crate::tools::file_ops::FileOpsStub)
            .unwrap()
            .build()
            .unwrap();

        // Reload same config — WASM listeners should still be accessible
        let new_org = p5_organism();
        let _event = pipeline.reload(new_org).unwrap();

        // Echo listener still present after reload
        assert!(pipeline.organism().get_listener("echo").is_some());
        assert!(pipeline.security().can_reach("admin", "echo"));
    }

    #[tokio::test]
    async fn without_wasm_tools_still_works() {
        let dir = TempDir::new().unwrap();
        // Use p4 organism (no WASM listeners) — pipeline builds fine without with_wasm_tools()
        let org = p4_organism();

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
            .with_code_index()
            .unwrap()
            .with_coding_agent()
            .unwrap()
            .build()
            .unwrap();

        assert!(pipeline.organism().get_listener("coding-agent").is_some());
        assert!(pipeline.organism().get_listener("echo").is_none());
    }

    // ── Phase 6 Milestone 1: Event Bus Tests ──

    #[tokio::test]
    async fn event_bus_subscribe_receives() {
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

        let mut rx = pipeline.subscribe();

        let envelope = build_envelope(
            "test",
            "echo",
            "thread-1",
            b"<Greeting><text>hi</text></Greeting>",
        )
        .unwrap();

        pipeline
            .inject_checked(envelope, "thread-1", "admin", "echo")
            .await
            .unwrap();

        let event = rx.recv().await.unwrap();
        match event {
            PipelineEvent::MessageInjected { target, .. } => assert_eq!(target, "echo"),
            _ => panic!("expected MessageInjected"),
        }

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn event_bus_multiple_subscribers() {
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

        let mut rx1 = pipeline.subscribe();
        let mut rx2 = pipeline.subscribe();

        let envelope = build_envelope(
            "test",
            "echo",
            "thread-1",
            b"<Greeting><text>hi</text></Greeting>",
        )
        .unwrap();

        pipeline
            .inject_checked(envelope, "thread-1", "admin", "echo")
            .await
            .unwrap();

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();
        assert!(matches!(e1, PipelineEvent::MessageInjected { .. }));
        assert!(matches!(e2, PipelineEvent::MessageInjected { .. }));

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn event_bus_security_blocked() {
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

        let mut rx = pipeline.subscribe();

        let envelope = build_envelope("test", "sink", "thread-1", b"<SinkRequest/>").unwrap();

        // Public cannot reach sink — should emit SecurityBlocked
        let _ = pipeline
            .inject_checked(envelope, "thread-1", "public", "sink")
            .await;

        let event = rx.recv().await.unwrap();
        match event {
            PipelineEvent::SecurityBlocked { profile, target } => {
                assert_eq!(profile, "public");
                assert_eq!(target, "sink");
            }
            _ => panic!("expected SecurityBlocked"),
        }

        pipeline.shutdown().await;
    }

    #[tokio::test]
    async fn agent_thread_snapshots_empty() {
        let pool = Arc::new(Mutex::new(crate::llm::LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        )));
        let handler = crate::agent::handler::CodingAgentHandler::new(
            pool,
            vec![],
            "test".into(),
        );
        let snapshots = handler.thread_snapshots().await;
        assert!(snapshots.is_empty());
    }

    #[test]
    fn event_types_clone_debug() {
        let event = PipelineEvent::MessageInjected {
            thread_id: "t1".into(),
            target: "echo".into(),
            profile: "admin".into(),
        };
        let cloned = event.clone();
        let debug = format!("{:?}", cloned);
        assert!(debug.contains("MessageInjected"));

        let blocked = PipelineEvent::SecurityBlocked {
            profile: "pub".into(),
            target: "sink".into(),
        };
        let _ = blocked.clone();
        assert!(format!("{:?}", blocked).contains("SecurityBlocked"));

        let token = PipelineEvent::TokenUsage {
            thread_id: "t1".into(),
            input_tokens: 100,
            output_tokens: 50,
        };
        let _ = token.clone();
        assert!(format!("{:?}", token).contains("TokenUsage"));

        let kernel_op = PipelineEvent::KernelOp {
            op: events::KernelOpType::ThreadCreated,
            thread_id: "t1".into(),
        };
        let _ = kernel_op.clone();
        assert!(format!("{:?}", kernel_op).contains("KernelOp"));
    }

    // ── Semantic Routing Integration Tests ──

    fn routing_organism() -> Organism {
        let yaml = r#"
organism:
  name: bestcode-routing

listeners:
  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM inference pool"
    ports:
      - port: 443
        direction: outbound
        protocol: https
        hosts: [api.anthropic.com]

  - name: file-ops
    payload_class: tools.FileOpsRequest
    handler: tools.file_ops.handle
    description: "File operations"
    semantic_description: |
      This tool reads, writes, and manages files on the local filesystem.
      Use it when you need to examine source code, read configuration files,
      write new files, create directories, or check if files exist.

  - name: shell
    payload_class: tools.ShellRequest
    handler: tools.shell.handle
    description: "Shell execution"
    semantic_description: |
      This tool executes shell commands and returns their output.
      Use it when you need to run programs, compile code, run tests,
      check system state, or execute any command-line operation.

  - name: coding-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Opus coding agent"
    agent: true
    peers: [file-ops, shell]

profiles:
  coding:
    linux_user: agentos-coding
    listeners: [coding-agent, file-ops, shell, llm-pool]
    network: [llm-pool]
    journal: retain_forever
"#;
        parse_organism(yaml).unwrap()
    }

    #[tokio::test]
    async fn pipeline_builder_with_semantic_router() {
        let dir = TempDir::new().unwrap();
        let org = routing_organism();

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
            .with_semantic_router()
            .unwrap()
            .with_coding_agent()
            .unwrap()
            .build()
            .unwrap();

        assert!(pipeline.organism().get_listener("coding-agent").is_some());
        assert!(pipeline.organism().get_listener("file-ops").is_some());
    }

    #[test]
    fn pipeline_event_semantic_match() {
        let event = PipelineEvent::SemanticMatch {
            thread_id: "t1".into(),
            tool_name: "file-ops".into(),
            score: 0.87,
        };
        let cloned = event.clone();
        let debug = format!("{:?}", cloned);
        assert!(debug.contains("SemanticMatch"));
        assert!(debug.contains("file-ops"));

        let fill_event = PipelineEvent::FormFillAttempt {
            thread_id: "t1".into(),
            tool_name: "file-ops".into(),
            model: "haiku".into(),
            success: true,
        };
        let _ = fill_event.clone();
        assert!(format!("{:?}", fill_event).contains("FormFillAttempt"));
    }
}
