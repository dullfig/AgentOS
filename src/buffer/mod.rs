//! Buffer Node — fork()+exec() for callable organisms.
//!
//! A BufferHandler is a pipeline Handler that builds ephemeral child pipelines
//! from organism YAML, injects work, captures results, and returns them as
//! ToolResponses. This is the runtime mechanism for callable organisms.
//!
//! ## Lifecycle
//!
//! 1. Acquire semaphore permit (backpressure at max_concurrency)
//! 2. Build child pipeline (tempdir kernel, shared LlmPool, fresh tool instances)
//! 3. Subscribe to child event bus
//! 4. Inject task message into child pipeline
//! 5. Await PipelineEvent::AgentResponse from child broadcast channel
//! 6. Shutdown child, drop tempdir
//! 7. Return ToolResponse::ok(result) or ToolResponse::err(e) to host

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use tokio::sync::{broadcast, Mutex, Semaphore};

use crate::llm::LlmPool;
use crate::organism::parser::load_organism;
use crate::organism::{BufferConfig, Organism};
use crate::pipeline::events::PipelineEvent;
use crate::pipeline::AgentPipelineBuilder;
use crate::tools::{self, ToolResponse};

/// Buffer handler — manages ephemeral child pipeline lifecycles.
pub struct BufferHandler {
    pool: Arc<Mutex<LlmPool>>,
    child_organism: Organism,
    config: BufferConfig,
    semaphore: Arc<Semaphore>,
    /// Event sender from the host pipeline (for forwarding child events).
    _event_tx: Option<broadcast::Sender<PipelineEvent>>,
}

impl BufferHandler {
    /// Create a new BufferHandler.
    pub fn new(
        pool: Arc<Mutex<LlmPool>>,
        child_organism: Organism,
        config: BufferConfig,
        event_tx: Option<broadcast::Sender<PipelineEvent>>,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrency));
        Self {
            pool,
            child_organism,
            config,
            semaphore,
            _event_tx: event_tx,
        }
    }

    /// Build and run a child pipeline, returning the agent's response text.
    async fn run_child(&self, task_text: &str) -> Result<String, String> {
        // Acquire semaphore permit (backpressure)
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| format!("semaphore acquire failed: {e}"))?;

        // Create temp directory for ephemeral kernel
        let tempdir = tempfile::TempDir::new()
            .map_err(|e| format!("tempdir creation failed: {e}"))?;
        let data_dir = tempdir.path().join("data");

        // Build child pipeline
        let child_org = self.child_organism.clone();
        let mut builder = AgentPipelineBuilder::new(child_org, &data_dir);

        // Inject shared LLM pool
        builder = builder.with_shared_llm_pool(self.pool.clone())?;

        // Register required tools as fresh instances
        builder = register_required_tools(builder, &self.config.requires)?;

        // Wire agents from child organism config
        builder = builder.with_agents()?;

        // Build the child pipeline
        let mut child_pipeline = builder.build()?;

        // Initialize root thread
        let child_profiles: Vec<String> = child_pipeline
            .organism()
            .profile_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let profile = child_profiles
            .first()
            .ok_or_else(|| "child organism has no profiles".to_string())?;
        let root_uuid = child_pipeline
            .initialize_root("buffer-child", profile)
            .await?;

        // Subscribe to child events
        let mut rx = child_pipeline.subscribe();

        // Start child pipeline
        child_pipeline.run();

        // Find the child agent listener
        let agent_def = child_pipeline
            .organism()
            .agent_listeners()
            .into_iter()
            .next()
            .ok_or_else(|| "child organism has no agent listeners".to_string())?
            .clone();

        // Build and inject the task envelope
        let escaped_task = tools::xml_escape(task_text);
        let xml = format!(
            "<{tag}><task>{escaped_task}</task></{tag}>",
            tag = agent_def.payload_tag
        );
        let envelope = build_envelope("user", &agent_def.name, &root_uuid, xml.as_bytes())
            .map_err(|e| format!("envelope build failed: {e}"))?;
        child_pipeline.inject_raw(envelope).await?;

        // Await AgentResponse with timeout
        let timeout = std::time::Duration::from_secs(self.config.timeout_secs);
        let result = tokio::time::timeout(timeout, async {
            loop {
                match rx.recv().await {
                    Ok(PipelineEvent::AgentResponse { text, .. }) => return Ok(text),
                    Ok(_) => continue, // skip other events
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => {
                        return Err("child event bus closed".to_string());
                    }
                }
            }
        })
        .await
        .map_err(|_| {
            format!(
                "buffer timeout after {}s",
                self.config.timeout_secs
            )
        })?;

        // Shutdown child
        child_pipeline.shutdown().await;
        // tempdir drops here

        result
    }
}

#[async_trait]
impl Handler for BufferHandler {
    async fn handle(
        &self,
        payload: ValidatedPayload,
        _ctx: HandlerContext,
    ) -> Result<HandlerResponse, PipelineError> {
        // Extract parameters from the XML payload
        let xml = String::from_utf8_lossy(&payload.xml).to_string();

        // Build a text representation of the parameters for the child agent
        let mut task_parts = Vec::new();
        for param in &self.config.parameters {
            if let Some(value) = tools::extract_tag(&xml, &param.name) {
                task_parts.push(format!("{}: {}", param.name, value));
            }
        }
        let task_text = task_parts.join("\n");

        // Run the child pipeline
        match self.run_child(&task_text).await {
            Ok(result) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::ok(&result),
            }),
            Err(e) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&e),
            }),
        }
    }
}

/// Register fresh tool instances for the required tools.
///
/// Matches tool names to their constructors. Unknown tools are hard errors.
fn register_required_tools(
    mut builder: AgentPipelineBuilder,
    requires: &[String],
) -> Result<AgentPipelineBuilder, String> {
    for name in requires {
        builder = match name.as_str() {
            "file-read" => builder.register_tool(name, crate::tools::file_read::FileReadTool)?,
            "file-write" => builder.register_tool(name, crate::tools::file_write::FileWriteTool)?,
            "file-edit" => builder.register_tool(name, crate::tools::file_edit::FileEditTool)?,
            "glob" => builder.register_tool(name, crate::tools::glob_tool::GlobTool)?,
            "grep" => builder.register_tool(name, crate::tools::grep::GrepTool)?,
            "command-exec" => {
                builder.register_tool(name, crate::tools::command_exec::CommandExecTool::new())?
            }
            _ => return Err(format!("unknown required tool: '{name}'")),
        };
    }
    Ok(builder)
}

/// Resolve the path to a child organism YAML file.
///
/// If the path is relative, resolves it relative to `base_dir`.
pub fn resolve_organism_path(base_dir: &std::path::Path, organism_path: &str) -> PathBuf {
    let path = PathBuf::from(organism_path);
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

/// Load and validate a child organism for buffer use.
pub fn load_child_organism(base_dir: &std::path::Path, organism_path: &str) -> Result<Organism, String> {
    let full_path = resolve_organism_path(base_dir, organism_path);
    load_organism(&full_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::organism::parser::parse_organism;
    use crate::organism::CallableParam;

    #[test]
    fn register_required_tools_known() {
        let yaml = r#"
organism:
  name: child-test

listeners:
  - name: file-read
    payload_class: tools.FileReadRequest
    handler: tools.file_read.handle
    description: "Read files"

  - name: command-exec
    payload_class: tools.CommandExecRequest
    handler: tools.command_exec.handle
    description: "Command execution"

profiles:
  child:
    linux_user: agentos-child
    listeners: all
    journal: prune_on_delivery
"#;
        let org = parse_organism(yaml).unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        let builder = AgentPipelineBuilder::new(org, dir.path());

        let requires = vec!["file-read".to_string(), "command-exec".to_string()];
        let result = register_required_tools(builder, &requires);
        assert!(result.is_ok());
    }

    #[test]
    fn register_required_tools_unknown_fails() {
        let yaml = r#"
organism:
  name: child-test
listeners: []
profiles:
  child:
    linux_user: agentos-child
    listeners: all
    journal: prune_on_delivery
"#;
        let org = parse_organism(yaml).unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        let builder = AgentPipelineBuilder::new(org, dir.path());

        let requires = vec!["nonexistent-tool".to_string()];
        let result = register_required_tools(builder, &requires);
        match result {
            Err(e) => assert!(e.contains("unknown required tool"), "unexpected error: {e}"),
            Ok(_) => panic!("expected error for unknown tool"),
        }
    }

    #[test]
    fn resolve_organism_path_relative() {
        let base = std::path::Path::new("/home/user/project");
        let resolved = resolve_organism_path(base, "email-agent.yaml");
        assert_eq!(resolved, PathBuf::from("/home/user/project/email-agent.yaml"));
    }

    #[test]
    fn resolve_organism_path_absolute() {
        let base = std::path::Path::new("/home/user/project");
        let resolved = resolve_organism_path(base, "/opt/agents/email-agent.yaml");
        assert_eq!(resolved, PathBuf::from("/opt/agents/email-agent.yaml"));
    }

    #[test]
    fn buffer_config_to_tool_definition() {
        let config = BufferConfig {
            description: "Send email".to_string(),
            parameters: vec![
                CallableParam {
                    name: "to".to_string(),
                    param_type: "string".to_string(),
                    description: Some("Recipient".to_string()),
                    enum_values: None,
                },
                CallableParam {
                    name: "count".to_string(),
                    param_type: "integer".to_string(),
                    description: None,
                    enum_values: None,
                },
            ],
            required: vec!["to".to_string()],
            requires: vec![],
            organism: Some("child.yaml".to_string()),
            max_concurrency: 5,
            timeout_secs: 300,
        };

        let def = config.to_tool_definition("email-sender");
        assert_eq!(def.name, "email-sender");
        assert_eq!(def.description, "Send email");

        let schema = def.input_schema.as_object().unwrap();
        assert_eq!(schema["type"], "object");
        let props = schema["properties"].as_object().unwrap();
        assert!(props.contains_key("to"));
        assert_eq!(props["to"]["type"], "string");
        assert_eq!(props["to"]["description"], "Recipient");
        assert_eq!(props["count"]["type"], "integer");
    }
}
