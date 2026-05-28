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

    /// Static bearer token, passed inline. **Visible in `ps`** —
    /// prefer `--auth-token-file` for production deployments.
    #[arg(long)]
    auth_token: Option<String>,

    /// Path to a file containing the bearer token. Whitespace at
    /// either end is trimmed. Preferred over `--auth-token`: the
    /// token doesn't appear in `ps` output, doesn't end up in shell
    /// history, and can have tighter file permissions than the
    /// process can have on its env.
    #[arg(long)]
    auth_token_file: Option<PathBuf>,

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

    // Token sourcing precedence: --auth-token-file > --auth-token >
    // AGENTOS_SERVER_TOKEN env var. The file is the recommended path
    // for production; the inline flag and env var are dev conveniences
    // that leak to local-attacker recon (ps, /proc/<pid>/environ).
    let token = if let Some(ref path) = cli.auth_token_file {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading auth-token-file {}", path.display()))?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            anyhow::bail!("auth-token-file {} is empty", path.display());
        }
        trimmed.to_string()
    } else {
        cli.auth_token
            .or_else(|| std::env::var("AGENTOS_SERVER_TOKEN").ok())
            .context(
                "no auth token: pass --auth-token-file (preferred), --auth-token, \
                 or set AGENTOS_SERVER_TOKEN env var",
            )?
    };

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

    // Periodic gauge updater for the idempotency cache size. Counters
    // and histograms are pull-friendly (each event updates them), but
    // a gauge over a DashMap needs an active sampler — every 10s is
    // plenty for capacity monitoring (the cache changes on human-paced
    // request rates, not microseconds).
    let _gauge_updater = {
        let cache = idempotency.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(10));
            ticker.tick().await;
            loop {
                ticker.tick().await;
                agentos_server::metrics::set_idempotency_cache_entries(cache.len());
            }
        })
    };

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

    // Operator safety: shout when binding to anything other than
    // loopback unless the operator has explicitly opted in. The
    // server has no TLS termination of its own (HTTPS is delegated
    // to a reverse proxy); binding 0.0.0.0 / a routable address
    // without a reverse proxy = bearer token traversing cleartext.
    //
    // Override with AGENTOS_INSECURE_PUBLIC=1 (e.g., behind a
    // confirmed reverse proxy).
    let resolved = listener.local_addr().ok();
    let is_loopback = resolved
        .map(|a| a.ip().is_loopback())
        .unwrap_or(false);
    let insecure_ack = std::env::var("AGENTOS_INSECURE_PUBLIC").ok().as_deref() == Some("1");
    if !is_loopback && !insecure_ack {
        tracing::warn!(
            bind = %cli.bind,
            "agentos-server is binding to a non-loopback address without \
             AGENTOS_INSECURE_PUBLIC=1 set. The bearer token will traverse \
             the network in cleartext unless a reverse proxy terminates TLS \
             in front. If that proxy is configured, set \
             AGENTOS_INSECURE_PUBLIC=1 to silence this warning."
        );
    }
    info!("agentos-server listening on {}", cli.bind);

    axum::serve(listener, app).await.context("axum serve")?;
    Ok(())
}
