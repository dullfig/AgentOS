//! package-organism tool — bundle an organism source folder into a .agent package.
//!
//! Reads an organism source folder from the VDrive, validates the YAML,
//! creates a manifest, and writes a self-contained .agent zip file.
//! Source files are never modified.

use std::io::{Cursor, Write};

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use serde::{Deserialize, Serialize};
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

use super::{extract_tag, ToolPeer, ToolResponse};
use super::vdrive_tools::DriveSlot;
use agentos_organism::parser::parse_organism;

/// Manifest embedded in every .agent package.
#[derive(Debug, Serialize, Deserialize)]
struct AgentManifest {
    name: String,
    version: String,
    description: String,
    /// SHA-256 of all files (excluding manifest itself), hex-encoded.
    checksum: String,
}

pub struct PackageOrganismTool {
    slot: DriveSlot,
}

impl PackageOrganismTool {
    pub fn new(slot: DriveSlot) -> Self {
        Self { slot }
    }
}

#[async_trait]
impl Handler for PackageOrganismTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        // Required: source folder path
        let source_dir = match extract_tag(&xml_str, "source") {
            Some(p) => p,
            None => return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(
                    "provide <source> path to the organism source folder",
                ),
            }),
        };

        // Optional: output path (default: {name}.agent next to source folder)
        let output_override = extract_tag(&xml_str, "output");

        // Optional: version (default: 0.1.0)
        let version = extract_tag(&xml_str, "version").unwrap_or_else(|| "0.1.0".to_string());

        // Optional: description override (default: from organism name)
        let desc_override = extract_tag(&xml_str, "description");

        let guard = self.slot.read().await;
        let drive = match guard.as_ref() {
            Some(d) => d,
            None => return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("no workspace mounted"),
            }),
        };

        // Verify source is a directory
        let dir_info = match drive.stat(&source_dir) {
            Ok(info) => info,
            Err(e) => return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("cannot read '{}': {}", source_dir, e)),
            }),
        };
        if !dir_info.is_dir {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("'{}' is not a directory", source_dir)),
            });
        }

        // Find organism.yaml in the source folder
        let org_yaml_path = format!("{}/organism.yaml", source_dir);
        let org_yaml = match drive.read_bytes(&org_yaml_path) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(_) => return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err("organism.yaml is not valid UTF-8"),
                }),
            },
            Err(_) => return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "no organism.yaml found in '{}'", source_dir
                )),
            }),
        };

        // Validate the organism
        let organism = match parse_organism(&org_yaml) {
            Ok(org) => org,
            Err(e) => return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("organism validation failed: {}", e)),
            }),
        };

        // Check breadcrumbs — validate and test must have been run
        let validated_path = format!("{}/.validated", source_dir);
        let tested_path = format!("{}/.tested", source_dir);

        let has_validated = drive.read_bytes(&validated_path).is_ok();
        let has_tested = drive.read_bytes(&tested_path).is_ok();

        if !has_validated && !has_tested {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(
                    "organism has not been validated or tested. Run validate-organism and test-organism first (pass source_dir to leave breadcrumbs).",
                ),
            });
        } else if !has_validated {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(
                    "organism has not been validated. Run validate-organism first (pass source_dir to leave breadcrumb).",
                ),
            });
        } else if !has_tested {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(
                    "organism has not been smoke-tested. Run test-organism first (pass source_dir to leave breadcrumb).",
                ),
            });
        }

        let name = organism.name.clone();
        let description = desc_override.unwrap_or_else(|| format!("{} agent package", name));

        // Collect all files in the source directory recursively
        let glob_pattern = format!("{}/**/*", source_dir);
        let files = match drive.glob(&glob_pattern) {
            Ok(f) => f,
            Err(e) => return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("failed to list files: {}", e)),
            }),
        };

        // Read all files and compute checksum
        let mut file_contents: Vec<(String, Vec<u8>)> = Vec::new();
        let mut hasher = Crc32Hasher::new();

        for file_path in &files {
            // Skip directories and dotfiles (breadcrumbs, build artifacts)
            if let Ok(info) = drive.stat(file_path) {
                if info.is_dir {
                    continue;
                }
            }
            let filename = file_path.rsplit('/').next().unwrap_or(file_path);
            if filename.starts_with('.') {
                continue;
            }

            let bytes = match drive.read_bytes(file_path) {
                Ok(b) => b,
                Err(e) => return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&format!(
                        "failed to read '{}': {}", file_path, e
                    )),
                }),
            };

            // Store with path relative to source_dir
            let rel_path = file_path
                .strip_prefix(&source_dir)
                .unwrap_or(file_path)
                .trim_start_matches('/')
                .trim_start_matches('\\')
                .to_string();

            hasher.update(&bytes);
            file_contents.push((rel_path, bytes));
        }

        let checksum = hasher.finalize_hex();

        // Build manifest
        let manifest = AgentManifest {
            name: name.clone(),
            version,
            description,
            checksum,
        };
        let manifest_json = serde_json::to_string_pretty(&manifest)
            .unwrap_or_else(|_| "{}".to_string());

        // Create zip in memory
        let mut zip_buffer = Cursor::new(Vec::new());
        {
            let mut zip = ZipWriter::new(&mut zip_buffer);
            let options = SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);

            // Write manifest first
            if let Err(e) = zip.start_file("manifest.json", options) {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&format!("zip error: {}", e)),
                });
            }
            let _ = zip.write_all(manifest_json.as_bytes());

            // Write all source files
            for (rel_path, bytes) in &file_contents {
                if let Err(e) = zip.start_file(rel_path, options) {
                    return Ok(HandlerResponse::Reply {
                        payload_xml: ToolResponse::err(&format!(
                            "zip error for '{}': {}", rel_path, e
                        )),
                    });
                }
                let _ = zip.write_all(bytes);
            }

            if let Err(e) = zip.finish() {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&format!("zip finalize error: {}", e)),
                });
            }
        }

        let zip_bytes = zip_buffer.into_inner();
        let zip_size = zip_bytes.len();

        // Write .agent file
        let output_path = output_override.unwrap_or_else(|| {
            // Place next to source folder (sibling)
            let parent = source_dir
                .rsplit_once('/')
                .map(|(p, _)| p.to_string())
                .unwrap_or_default();
            if parent.is_empty() {
                format!("{}.agent", name)
            } else {
                format!("{}/{}.agent", parent, name)
            }
        });

        if let Err(e) = drive.write_bytes(&output_path, &zip_bytes) {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "failed to write '{}': {}", output_path, e
                )),
            });
        }

        let report = format!(
            "Package created: {}\n\
             Files: {}\n\
             Size: {} bytes\n\
             Checksum: {}\n\
             Output: {}",
            name,
            file_contents.len(),
            zip_size,
            manifest.checksum,
            output_path,
        );

        Ok(HandlerResponse::Reply {
            payload_xml: ToolResponse::ok(&report),
        })
    }
}

/// Simple CRC32 hasher for checksumming package contents.
struct Crc32Hasher {
    hasher: crc32fast::Hasher,
}

impl Crc32Hasher {
    fn new() -> Self {
        Self { hasher: crc32fast::Hasher::new() }
    }

    fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    fn finalize_hex(self) -> String {
        format!("{:08x}", self.hasher.finalize())
    }
}

#[async_trait]
impl ToolPeer for PackageOrganismTool {
    fn name(&self) -> &str {
        "package-organism"
    }

    fn wit(&self) -> &str {
        r#"
/// Package an organism source folder into a self-contained .agent file. Validates the organism YAML, bundles all files (prompts, tools, configs) into a zip with manifest and checksum. Source files are never modified.
interface package_organism {
    record request {
        /// Path to organism source folder (must contain organism.yaml)
        source: string,
        /// Output path for .agent file (default: {name}.agent next to source)
        output: option<string>,
        /// Package version (default: 0.1.0)
        version: option<string>,
        /// Description override (default: derived from organism name)
        description: option<string>,
    }
    package: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::vdrive_tools::empty_slot;
    use agentos_vdrive::VDrive;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "PackageOrganismRequest".into(),
        }
    }

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            from: "test".into(),
            own_name: "package-organism".into(),
            thread_id: "test-thread".into(),
        }
    }

    #[tokio::test]
    async fn missing_source_returns_error() {
        let tool = PackageOrganismTool::new(empty_slot());
        let xml = "<PackageOrganismRequest></PackageOrganismRequest>";
        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("<success>false</success>"), "expected error: {s}");
                assert!(s.contains("source"), "{s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn no_drive_mounted_returns_error() {
        let tool = PackageOrganismTool::new(empty_slot());
        let xml = "<PackageOrganismRequest><source>organisms/test</source></PackageOrganismRequest>";
        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("no workspace mounted"), "{s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn packages_valid_organism() {
        let tmpdir = tempfile::TempDir::new().unwrap();
        let drive = VDrive::open(tmpdir.path()).unwrap();

        // Create source folder structure
        drive.write_file("organisms/test-agent/organism.yaml", r#"
organism:
  name: test-agent
prompts:
  safety: |
    You are bounded.
  task: |
    You are a test agent.
listeners:
  - name: my-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Test agent"
    agent:
      prompt: "safety & task"
      max_tokens: 4096
    peers: [llm-pool]
  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM pool"
profiles:
  default:
    linux_user: agentos
    listeners: [my-agent, llm-pool]
    journal: retain_forever
"#).unwrap();

        drive.write_file("organisms/test-agent/prompts/guide.md",
            "# Guide\nThis is a guide."
        ).unwrap();

        // Write breadcrumbs (simulate validate + test having run)
        drive.write_file("organisms/test-agent/.validated",
            "{\"status\":\"ok\",\"timestamp\":1710000000}"
        ).unwrap();
        drive.write_file("organisms/test-agent/.tested",
            "{\"status\":\"ok\",\"timestamp\":1710000000,\"tests\":1,\"input_tokens\":100,\"output_tokens\":50}"
        ).unwrap();

        let slot: DriveSlot = Arc::new(RwLock::new(Some(Arc::new(drive))));
        let tool = PackageOrganismTool::new(slot);

        let xml = "<PackageOrganismRequest>\
            <source>organisms/test-agent</source>\
            <version>1.0.0</version>\
            <description>A test agent package</description>\
            </PackageOrganismRequest>";

        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("<success>true</success>"), "expected success: {s}");
                assert!(s.contains("test-agent"), "{s}");
                assert!(s.contains("Files: 2"), "expected 2 files: {s}");
            }
            _ => panic!("expected Reply"),
        }

        // Verify the .agent file was created
        let agent_bytes = tmpdir.path().join("organisms/test-agent.agent");
        assert!(agent_bytes.exists(), ".agent file should exist");

        // Verify it's a valid zip
        let data = std::fs::read(&agent_bytes).unwrap();
        let reader = Cursor::new(data);
        let mut archive = zip::ZipArchive::new(reader).unwrap();
        let file_names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(file_names.contains(&"manifest.json".to_string()));
        assert!(file_names.contains(&"organism.yaml".to_string()));
        assert!(file_names.iter().any(|f| f.contains("guide.md")));

        // Verify manifest content
        let mut manifest_file = archive.by_name("manifest.json").unwrap();
        let mut manifest_str = String::new();
        std::io::Read::read_to_string(&mut manifest_file, &mut manifest_str).unwrap();
        let manifest: AgentManifest = serde_json::from_str(&manifest_str).unwrap();
        assert_eq!(manifest.name, "test-agent");
        assert_eq!(manifest.version, "1.0.0");
        assert_eq!(manifest.description, "A test agent package");
        assert!(!manifest.checksum.is_empty());
    }

    #[tokio::test]
    async fn missing_organism_yaml_returns_error() {
        let tmpdir = tempfile::TempDir::new().unwrap();
        let drive = VDrive::open(tmpdir.path()).unwrap();

        // Create folder without organism.yaml
        drive.write_file("organisms/bad-agent/readme.md", "hello").unwrap();

        let slot: DriveSlot = Arc::new(RwLock::new(Some(Arc::new(drive))));
        let tool = PackageOrganismTool::new(slot);

        let xml = "<PackageOrganismRequest>\
            <source>organisms/bad-agent</source>\
            </PackageOrganismRequest>";

        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("no organism.yaml"), "{s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn invalid_organism_returns_error() {
        let tmpdir = tempfile::TempDir::new().unwrap();
        let drive = VDrive::open(tmpdir.path()).unwrap();

        drive.write_file("organisms/bad/organism.yaml", "not: valid: [}").unwrap();

        let slot: DriveSlot = Arc::new(RwLock::new(Some(Arc::new(drive))));
        let tool = PackageOrganismTool::new(slot);

        let xml = "<PackageOrganismRequest>\
            <source>organisms/bad</source>\
            </PackageOrganismRequest>";

        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("validation failed"), "{s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn missing_breadcrumbs_returns_error() {
        let tmpdir = tempfile::TempDir::new().unwrap();
        let drive = VDrive::open(tmpdir.path()).unwrap();

        // Valid organism but no breadcrumbs
        drive.write_file("organisms/no-crumbs/organism.yaml", r#"
organism:
  name: no-crumbs
prompts:
  task: |
    You are a test.
listeners:
  - name: my-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Test"
    agent:
      prompt: "task"
    peers: [llm-pool]
  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM"
profiles:
  default:
    linux_user: agentos
    listeners: [my-agent, llm-pool]
    journal: retain_forever
"#).unwrap();

        let slot: DriveSlot = Arc::new(RwLock::new(Some(Arc::new(drive))));
        let tool = PackageOrganismTool::new(slot);

        let xml = "<PackageOrganismRequest>\
            <source>organisms/no-crumbs</source>\
            </PackageOrganismRequest>";

        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("<success>false</success>"), "expected error: {s}");
                assert!(s.contains("validated") || s.contains("tested"), "expected breadcrumb error: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn dotfiles_excluded_from_package() {
        let tmpdir = tempfile::TempDir::new().unwrap();
        let drive = VDrive::open(tmpdir.path()).unwrap();

        drive.write_file("organisms/clean/organism.yaml", r#"
organism:
  name: clean
prompts:
  task: |
    Test.
listeners:
  - name: my-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Test"
    agent:
      prompt: "task"
    peers: [llm-pool]
  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM"
profiles:
  default:
    linux_user: agentos
    listeners: [my-agent, llm-pool]
    journal: retain_forever
"#).unwrap();

        // Write breadcrumbs
        drive.write_file("organisms/clean/.validated", "{\"status\":\"ok\",\"timestamp\":1}").unwrap();
        drive.write_file("organisms/clean/.tested", "{\"status\":\"ok\",\"timestamp\":1}").unwrap();

        let slot: DriveSlot = Arc::new(RwLock::new(Some(Arc::new(drive))));
        let tool = PackageOrganismTool::new(slot);

        let xml = "<PackageOrganismRequest><source>organisms/clean</source></PackageOrganismRequest>";
        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("<success>true</success>"), "expected success: {s}");
            }
            _ => panic!("expected Reply"),
        }

        // Verify dotfiles are NOT in the zip
        let agent_path = tmpdir.path().join("organisms/clean.agent");
        let data = std::fs::read(&agent_path).unwrap();
        let reader = Cursor::new(data);
        let mut archive = zip::ZipArchive::new(reader).unwrap();
        let file_names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(!file_names.iter().any(|f| f.contains(".validated")), "dotfiles should be excluded: {file_names:?}");
        assert!(!file_names.iter().any(|f| f.contains(".tested")), "dotfiles should be excluded: {file_names:?}");
        assert!(file_names.contains(&"manifest.json".to_string()));
        assert!(file_names.contains(&"organism.yaml".to_string()));
    }
}
