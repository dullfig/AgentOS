//! CompileWasmTool — compile Python source to a WASM component.
//!
//! Shells out to `componentize-py` to compile a Python module that
//! implements the AgentOS WIT tool contract into a .wasm component.
//! Returns compiler output (stdout + stderr) so agents can iterate
//! on errors.

use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use tokio::process::Command;

use super::{extract_tag, ToolPeer, ToolResponse};

const COMPILE_TIMEOUT_SECS: u64 = 120;

/// Compile Python source to a WASM component via componentize-py.
pub struct CompileWasmTool {
    /// Path to the WIT directory (contains tool.wit).
    wit_dir: PathBuf,
}

impl CompileWasmTool {
    /// Create with the path to the WIT directory.
    pub fn new(wit_dir: PathBuf) -> Self {
        Self { wit_dir }
    }

    /// Resolve the output path: if not specified, use `<source_dir>/<module>.wasm`.
    fn resolve_output(source_dir: &Path, module: &str, output: Option<&str>) -> PathBuf {
        match output {
            Some(p) => PathBuf::from(p),
            None => source_dir.join(format!("{module}.wasm")),
        }
    }
}

#[async_trait]
impl Handler for CompileWasmTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        // Required: source directory containing app.py + bindings/
        let source_dir = match extract_tag(&xml_str, "source_dir") {
            Some(d) => d,
            None => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(
                        "missing required <source_dir>: path to directory containing the Python module",
                    ),
                });
            }
        };

        // Optional: Python module name (default: "app")
        let module = extract_tag(&xml_str, "module").unwrap_or_else(|| "app".into());

        // Optional: output .wasm path (default: <source_dir>/<module>.wasm)
        let output_tag = extract_tag(&xml_str, "output");
        let source_path = Path::new(&source_dir);
        let output_path =
            Self::resolve_output(source_path, &module, output_tag.as_deref());

        // Validate source directory exists
        if !source_path.is_dir() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "source_dir does not exist or is not a directory: {source_dir}"
                )),
            });
        }

        // Check that the module file exists
        let module_file = source_path.join(format!("{module}.py"));
        if !module_file.exists() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "module file not found: {}",
                    module_file.display()
                )),
            });
        }

        // Check for bindings directory
        let bindings_dir = source_path.join("bindings");
        if !bindings_dir.is_dir() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "bindings directory not found: {}. Run: componentize-py -d <wit_dir> -w tool bindings ./bindings",
                    bindings_dir.display()
                )),
            });
        }

        // Build componentize-py command
        let wit_dir_str = self.wit_dir.to_string_lossy();
        let output_str = output_path.to_string_lossy();

        let mut cmd = Command::new("componentize-py");
        cmd.args([
            "-d",
            &wit_dir_str,
            "-w",
            "tool",
            "componentize",
            "-p",
            &source_dir,
            "-p",
            &bindings_dir.to_string_lossy(),
            "-o",
            &output_str,
            &module,
        ]);
        cmd.current_dir(source_path);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let result = tokio::time::timeout(
            Duration::from_secs(COMPILE_TIMEOUT_SECS),
            cmd.output(),
        )
        .await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let exit_code = output.status.code().unwrap_or(-1);

                if exit_code == 0 {
                    let size = std::fs::metadata(&output_path)
                        .map(|m| m.len())
                        .unwrap_or(0);
                    let response = format!(
                        "Compilation successful.\nOutput: {}\nSize: {} bytes\n{}{}",
                        output_path.display(),
                        size,
                        if !stdout.is_empty() {
                            format!("stdout:\n{stdout}\n")
                        } else {
                            String::new()
                        },
                        if !stderr.is_empty() {
                            format!("stderr:\n{stderr}")
                        } else {
                            String::new()
                        },
                    );
                    Ok(HandlerResponse::Reply {
                        payload_xml: ToolResponse::ok(&response),
                    })
                } else {
                    let response = format!(
                        "Compilation failed (exit code {exit_code}).\nstdout:\n{stdout}\nstderr:\n{stderr}"
                    );
                    Ok(HandlerResponse::Reply {
                        payload_xml: ToolResponse::err(&response),
                    })
                }
            }
            Ok(Err(e)) => {
                let msg = if e.kind() == std::io::ErrorKind::NotFound {
                    "componentize-py not found. Install with: pip install componentize-py"
                        .to_string()
                } else {
                    format!("failed to execute componentize-py: {e}")
                };
                Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&msg),
                })
            }
            Err(_) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "compilation timed out after {COMPILE_TIMEOUT_SECS}s"
                )),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for CompileWasmTool {
    fn name(&self) -> &str {
        "compile-wasm"
    }

    fn wit(&self) -> &str {
        r#"
/// Compile a Python tool to a WASM component. The source directory must contain the Python module file and a bindings/ subdirectory (generated by componentize-py bindings). Returns compiler output including any errors for debugging.
interface compile-wasm {
    record request {
        /// Path to the directory containing the Python source and bindings
        source-dir: string,
        /// Python module name (default: "app")
        module: option<string>,
        /// Output .wasm file path (default: <source_dir>/<module>.wasm)
        output: option<string>,
    }
    exec: func(req: request) -> result<string, string>;
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
            from: "agent".into(),
            own_name: "compile-wasm".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "CompileWasmRequest".into(),
        }
    }

    fn get_result(resp: HandlerResponse) -> (bool, String) {
        match resp {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                let success = xml.contains("<success>true</success>");
                let content = if success {
                    extract_tag(&xml, "result").unwrap_or_default()
                } else {
                    extract_tag(&xml, "error").unwrap_or_default()
                };
                (success, content)
            }
            _ => panic!("expected Reply"),
        }
    }

    #[test]
    fn resolve_output_default() {
        let out = CompileWasmTool::resolve_output(Path::new("/tmp/my-tool"), "app", None);
        assert_eq!(out, PathBuf::from("/tmp/my-tool/app.wasm"));
    }

    #[test]
    fn resolve_output_explicit() {
        let out =
            CompileWasmTool::resolve_output(Path::new("/tmp/my-tool"), "app", Some("/out/tool.wasm"));
        assert_eq!(out, PathBuf::from("/out/tool.wasm"));
    }

    #[tokio::test]
    async fn missing_source_dir() {
        let tool = CompileWasmTool::new(PathBuf::from("/nonexistent/wit"));
        let xml = "<CompileWasmRequest><source_dir>/nonexistent/dir</source_dir></CompileWasmRequest>";
        let (ok, content) = get_result(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("does not exist"));
    }

    #[tokio::test]
    async fn missing_required_source_dir() {
        let tool = CompileWasmTool::new(PathBuf::from("/nonexistent/wit"));
        let xml = "<CompileWasmRequest></CompileWasmRequest>";
        let (ok, content) = get_result(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("missing required"));
    }

    #[tokio::test]
    async fn missing_module_file() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("bindings")).unwrap();
        // No app.py

        let tool = CompileWasmTool::new(PathBuf::from("/nonexistent/wit"));
        let xml = format!(
            "<CompileWasmRequest><source_dir>{}</source_dir></CompileWasmRequest>",
            dir.path().display()
        );
        let (ok, content) = get_result(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("module file not found"));
    }

    #[tokio::test]
    async fn missing_bindings_dir() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("app.py"), "# empty").unwrap();
        // No bindings/

        let tool = CompileWasmTool::new(PathBuf::from("/nonexistent/wit"));
        let xml = format!(
            "<CompileWasmRequest><source_dir>{}</source_dir></CompileWasmRequest>",
            dir.path().display()
        );
        let (ok, content) = get_result(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("bindings directory not found"));
    }

    #[tokio::test]
    async fn compile_echo_py_tool() {
        // Integration test: compile the echo-py example
        let source_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tools")
            .join("echo-py");
        let wit_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wit");

        if !source_dir.join("app.py").exists() || !source_dir.join("bindings").is_dir() {
            eprintln!("Skipping compile test: echo-py source not set up");
            return;
        }

        // Check componentize-py is available
        let check = std::process::Command::new("componentize-py")
            .arg("--version")
            .output();
        if check.is_err() || !check.unwrap().status.success() {
            eprintln!("Skipping compile test: componentize-py not installed");
            return;
        }

        let output_dir = TempDir::new().unwrap();
        let output_path = output_dir.path().join("echo-py-test.wasm");

        let tool = CompileWasmTool::new(wit_dir);
        let xml = format!(
            "<CompileWasmRequest>\
                <source_dir>{}</source_dir>\
                <output>{}</output>\
            </CompileWasmRequest>",
            source_dir.display(),
            output_path.display(),
        );
        let (ok, content) =
            get_result(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok, "compile failed: {content}");
        assert!(content.contains("Compilation successful"));
        assert!(output_path.exists(), "output .wasm not created");
    }

    #[test]
    fn compile_wasm_metadata() {
        let tool = CompileWasmTool::new(PathBuf::from("/tmp/wit"));
        assert_eq!(tool.name(), "compile-wasm");
        let iface = crate::wit::parser::parse_wit(tool.wit()).unwrap();
        assert_eq!(iface.name, "compile-wasm");
        assert_eq!(iface.request_tag(), "CompileWasmRequest");
        assert!(iface.request.fields.iter().any(|f| f.name == "source-dir"));
    }
}
