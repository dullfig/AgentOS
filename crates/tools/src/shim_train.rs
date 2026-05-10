//! ShimTrainTool — invoke the Python shim trainer subprocess.
//!
//! Calls `python <script_path> --input <jsonl> --output-dir <dir> ...`
//! and parses the resulting `metrics.json` so the shim-expert agent
//! has structured visibility into training quality before deciding
//! whether to register the shim.
//!
//! Python + torch must be installed on the host. The trainer script
//! lives at `crates/cortex-shim/scripts/train_shim.py` and is configured
//! by the AgentPipelineBuilder at register-tool time so deployments can
//! point it elsewhere if needed (e.g. baked into a container image).

use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use serde_json::json;
use tokio::process::Command;
use tokio::time::timeout;

use super::{extract_tag, ToolPeer, ToolResponse};

/// Subprocess time budget for one training run. Tiny FFNs train in
/// seconds-to-minutes on CPU; 10 minutes is a generous ceiling.
const TRAIN_TIMEOUT: Duration = Duration::from_secs(600);

/// Tool that runs the Python shim trainer.
///
/// Configured at register-tool time with the trainer script's path and
/// the python interpreter to use.
#[derive(Clone, Debug)]
pub struct ShimTrainTool {
    python: String,
    script_path: PathBuf,
}

impl ShimTrainTool {
    /// Construct with explicit paths.
    pub fn new(python: impl Into<String>, script_path: PathBuf) -> Self {
        Self {
            python: python.into(),
            script_path,
        }
    }

    /// Default path: `<workspace_root>/crates/cortex-shim/scripts/train_shim.py`
    /// + the system `python`.
    pub fn with_workspace_default(workspace_root: &std::path::Path) -> Self {
        let script_path = workspace_root
            .join("crates")
            .join("cortex-shim")
            .join("scripts")
            .join("train_shim.py");
        Self::new("python", script_path)
    }

    /// Execute the trainer and return either the parsed `metrics.json`
    /// content (as JSON string) or a structured error.
    async fn execute(&self, args: TrainArgs) -> Result<String, String> {
        if !self.script_path.exists() {
            return Err(format!(
                "trainer script not found at {}",
                self.script_path.display()
            ));
        }

        // The trainer creates output_dir if missing, but we resolve it
        // up front so the success-path metrics_path read is unambiguous.
        std::fs::create_dir_all(&args.output_dir)
            .map_err(|e| format!("create output_dir: {e}"))?;

        let mut cmd = Command::new(&self.python);
        cmd.arg(&self.script_path)
            .arg("--input")
            .arg(&args.input)
            .arg("--output-dir")
            .arg(&args.output_dir);

        if let Some(d) = &args.input_dim {
            cmd.arg("--input-dim").arg(d.to_string());
        }
        if let Some(h) = &args.hidden {
            cmd.arg("--hidden").arg(h);
        }
        if let Some(o) = &args.output_dim {
            cmd.arg("--output-dim").arg(o.to_string());
        }
        if let Some(e) = &args.epochs {
            cmd.arg("--epochs").arg(e.to_string());
        }
        if let Some(lr) = &args.lr {
            cmd.arg("--lr").arg(lr.to_string());
        }
        if let Some(b) = &args.batch_size {
            cmd.arg("--batch-size").arg(b.to_string());
        }
        if let Some(s) = &args.seed {
            cmd.arg("--seed").arg(s.to_string());
        }

        let output = match timeout(TRAIN_TIMEOUT, cmd.output()).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => return Err(format!("spawn python: {e}")),
            Err(_) => return Err(format!("trainer exceeded {TRAIN_TIMEOUT:?}")),
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(format!(
                "trainer exited with {}: stderr={} stdout={}",
                output.status, stderr, stdout
            ));
        }

        let metrics_path = args.output_dir.join("metrics.json");
        let metrics = std::fs::read_to_string(&metrics_path).map_err(|e| {
            format!(
                "trainer succeeded but metrics.json missing at {}: {e}",
                metrics_path.display()
            )
        })?;

        // Validate it parses as JSON before handing back to the agent.
        let parsed: serde_json::Value = serde_json::from_str(&metrics)
            .map_err(|e| format!("metrics.json is not valid JSON: {e}"))?;

        Ok(json!({
            "metrics": parsed,
            "model_path": args.output_dir.join("model.onnx"),
            "metrics_path": metrics_path,
        })
        .to_string())
    }
}

/// Parsed training arguments.
struct TrainArgs {
    input: PathBuf,
    output_dir: PathBuf,
    input_dim: Option<u32>,
    hidden: Option<String>,
    output_dim: Option<u32>,
    epochs: Option<u32>,
    lr: Option<f64>,
    batch_size: Option<u32>,
    seed: Option<u32>,
}

impl TrainArgs {
    fn from_xml(xml_str: &str) -> Result<Self, String> {
        let input = extract_tag(xml_str, "input")
            .ok_or_else(|| "missing required <input>".to_string())?;
        let output_dir = extract_tag(xml_str, "output_dir")
            .ok_or_else(|| "missing required <output_dir>".to_string())?;
        Ok(Self {
            input: PathBuf::from(input),
            output_dir: PathBuf::from(output_dir),
            input_dim: extract_tag(xml_str, "input_dim").and_then(|s| s.parse().ok()),
            hidden: extract_tag(xml_str, "hidden"),
            output_dim: extract_tag(xml_str, "output_dim").and_then(|s| s.parse().ok()),
            epochs: extract_tag(xml_str, "epochs").and_then(|s| s.parse().ok()),
            lr: extract_tag(xml_str, "lr").and_then(|s| s.parse().ok()),
            batch_size: extract_tag(xml_str, "batch_size").and_then(|s| s.parse().ok()),
            seed: extract_tag(xml_str, "seed").and_then(|s| s.parse().ok()),
        })
    }
}

#[async_trait]
impl Handler for ShimTrainTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let args = match TrainArgs::from_xml(&xml_str) {
            Ok(a) => a,
            Err(e) => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&e),
                });
            }
        };

        let result = self.execute(args).await;
        let payload_xml = match result {
            Ok(body) => ToolResponse::ok(&body),
            Err(msg) => ToolResponse::err(&msg),
        };
        Ok(HandlerResponse::Reply { payload_xml })
    }
}

#[async_trait]
impl ToolPeer for ShimTrainTool {
    fn name(&self) -> &str {
        "shim-train"
    }

    fn wit(&self) -> &str {
        r#"
/// Train a shim FFN by invoking the Python trainer subprocess. Reads
/// (vector, label) JSONL, exports an ONNX file plus metrics.json. The
/// shim-expert agent reads metrics.json to decide whether to register
/// the trained shim with cortex.
interface shim-train {
    record request {
        /// path to the (vector, label) JSONL input
        input: string,
        /// directory the trainer writes model.onnx + metrics.json into
        output-dir: string,
        /// optional hyperparameters
        input-dim: option<u32>,
        hidden: option<string>,         // e.g. "1024,256"
        output-dim: option<u32>,
        epochs: option<u32>,
        lr: option<f64>,
        batch-size: option<u32>,
        seed: option<u32>,
    }
    invoke: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "shim-expert".into(),
            own_name: "shim-train".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "ShimTrain".into(),
        }
    }

    fn parse(resp: HandlerResponse) -> (bool, String) {
        match resp {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                let success = xml.contains("<success>true</success>");
                let body = if success {
                    extract_tag(&xml, "result").unwrap_or_default()
                } else {
                    extract_tag(&xml, "error").unwrap_or_default()
                };
                (success, body)
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn missing_required_args_error() {
        let tool = ShimTrainTool::new("python", PathBuf::from("/nope/train_shim.py"));
        let xml = "<ShimTrain></ShimTrain>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("<input>"));
    }

    #[tokio::test]
    async fn missing_script_returns_clear_error() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("data.jsonl");
        std::fs::write(&input, "{}").unwrap();
        let out = dir.path().join("out");

        let tool = ShimTrainTool::new("python", PathBuf::from("/does-not-exist/train_shim.py"));
        let xml = format!(
            "<ShimTrain><input>{}</input><output_dir>{}</output_dir></ShimTrain>",
            input.display(),
            out.display()
        );
        let (ok, msg) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("trainer script not found"));
    }

    /// Stub trainer test: we substitute the Python interpreter with a
    /// shell snippet that just writes a metrics.json into output_dir.
    /// This proves the tool's spawn → wait → parse flow without
    /// requiring a real torch install.
    #[tokio::test]
    #[cfg(unix)]
    async fn end_to_end_with_stub_trainer() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("data.jsonl");
        std::fs::write(&input, "").unwrap();
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).unwrap();

        // Stub script — emits a valid metrics.json.
        let stub = dir.path().join("stub_train.sh");
        std::fs::write(
            &stub,
            "#!/usr/bin/env bash\n\
             set -e\n\
             OUTDIR=\"\"\n\
             while [ \"$#\" -gt 0 ]; do\n\
               case \"$1\" in --output-dir) OUTDIR=\"$2\"; shift 2 ;; *) shift ;; esac\n\
             done\n\
             echo '{\"test\":{\"accuracy\":0.97}}' > \"$OUTDIR/metrics.json\"\n",
        )
        .unwrap();
        std::process::Command::new("chmod")
            .args(["+x", stub.to_str().unwrap()])
            .status()
            .unwrap();

        // Use bash as the interpreter; our "script" is the shell stub.
        let tool = ShimTrainTool::new("bash", stub.clone());
        let xml = format!(
            "<ShimTrain><input>{}</input><output_dir>{}</output_dir></ShimTrain>",
            input.display(),
            out.display()
        );
        let (ok, body) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");
        assert!(body.contains("0.97"), "body: {body}");
    }
}
