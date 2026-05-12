//! agentos-server bin — boots an AgentPipeline + axum HTTP frontend.
//!
//! Loads an organism from `--organism <path>` (or a stub if omitted),
//! constructs the pipeline, mounts the platform router, starts the
//! HTTP server. Static bearer token from `--auth-token` or
//! `AGENTOS_SERVER_TOKEN` env var.
//!
//! Note on conversation persistence: the registry is in-memory only.
//! A restart drops all materialized instances. See Step 3.5 in the
//! topology doc for the persistence work that lights this up.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::net::TcpListener;
use tracing::info;

use agentos_organism::parser::parse_organism;
use agentos_pipeline::AgentPipelineBuilder;
use agentos_server::{build_router, ServerState};

/// Embedded fallback organism so the bin can boot without an explicit
/// config — useful for smoke tests. Real deployments pass `--organism`.
const STUB_ORGANISM: &str = r#"
organism:
  name: agentos-server-stub

listeners:
  - name: bob
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Bob — chat-bubble agent (no LLM wired in stub)"
    agent:
      prompt: "stub"

profiles:
  default:
    linux_user: agentos
    listeners: [bob]
    journal: retain_forever
"#;

#[derive(Parser)]
#[command(name = "agentos-server", about = "AgentOS HTTP+SSE server (chat-bubble Bob)")]
struct Cli {
    /// Bind address. Defaults to 127.0.0.1:8080.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,

    /// Path to organism YAML. Falls back to a stub if omitted.
    #[arg(long)]
    organism: Option<PathBuf>,

    /// Kernel data directory. Defaults to ./.agentos.
    #[arg(long, default_value = ".agentos")]
    data: PathBuf,

    /// Static bearer token. Falls back to AGENTOS_SERVER_TOKEN env var.
    #[arg(long)]
    auth_token: Option<String>,

    /// Listener name to handle /v1/messages traffic. Defaults to "bob".
    #[arg(long, default_value = "bob")]
    agent: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,agentos=debug")),
        )
        .init();

    let cli = Cli::parse();

    let token = cli
        .auth_token
        .or_else(|| std::env::var("AGENTOS_SERVER_TOKEN").ok())
        .context(
            "no auth token: pass --auth-token or set AGENTOS_SERVER_TOKEN env var",
        )?;

    let yaml = match cli.organism {
        Some(ref p) => std::fs::read_to_string(p)
            .with_context(|| format!("reading organism from {}", p.display()))?,
        None => {
            info!("no --organism provided; using embedded stub");
            STUB_ORGANISM.to_string()
        }
    };
    let org = parse_organism(&yaml).map_err(|e| anyhow::anyhow!("organism parse: {e}"))?;

    let builder = AgentPipelineBuilder::new(org, &cli.data);
    let event_tx = builder.event_sender();
    let mut pipeline = builder
        .build()
        .map_err(|e| anyhow::anyhow!("pipeline build: {e}"))?;

    let profile = pipeline
        .organism()
        .profile_names()
        .into_iter()
        .next()
        .unwrap_or("default")
        .to_string();
    pipeline
        .initialize_root("agentos-server", &profile)
        .await
        .map_err(|e| anyhow::anyhow!("kernel init: {e}"))?;
    pipeline.run();

    let shared_router = Arc::new(pipeline.shared_router(0, Duration::from_secs(60)));
    let _eviction = shared_router.start_eviction_timer();

    let idempotency = agentos_server::idempotency::IdempotencyCache::new();
    // Periodic TTL sweep; the handle is dropped on shutdown which
    // stops the sweeper (the cache itself remains usable).
    let _sweeper = idempotency.clone().spawn_sweeper(
        agentos_server::idempotency::DEFAULT_SWEEP_INTERVAL,
    );

    let state = Arc::new(ServerState {
        router: shared_router,
        events: event_tx,
        organism: Arc::new(pipeline.organism().clone()),
        agent_name: cli.agent,
        auth_token: token,
        idempotency,
    });

    let app = build_router(state);
    let listener = TcpListener::bind(&cli.bind)
        .await
        .with_context(|| format!("binding {}", cli.bind))?;
    info!("agentos-server listening on {}", cli.bind);

    axum::serve(listener, app).await.context("axum serve")?;
    Ok(())
}
