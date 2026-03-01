//! Live integration tests for the coding agent.
//!
//! Requires ANTHROPIC_API_KEY in the environment. Skips gracefully if unset.
//! Uses Haiku for cost efficiency — the coding agent is configured to use Haiku
//! instead of Opus for testing purposes.

use std::sync::Arc;

use agentos::agent::handler::CodingAgentHandler;
use agentos::agent::tools::build_tool_definitions;
use agentos::llm::types::{ContentBlock, ToolDefinition};
use agentos::llm::LlmPool;
use rust_pipeline::prelude::*;
use tokio::sync::Mutex;

/// Test: Inject a simple task → agent calls API → gets text response (no tools).
#[tokio::test]
async fn agent_simple_text_response() {
    let pool = match LlmPool::from_env("haiku") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("ANTHROPIC_API_KEY not set — skipping live agent test");
            return;
        }
    };

    let pool = Arc::new(Mutex::new(pool));
    // No tools → the model can't use tools, must respond with text
    let handler = CodingAgentHandler::new("test-agent".into(), pool, vec![], "You are a helpful test agent. Be very brief.".into());

    let payload = ValidatedPayload {
        xml: b"<AgentTask><task>What is 2+2? Reply with just the number.</task></AgentTask>"
            .to_vec(),
        tag: "AgentTask".into(),
    };
    let ctx = HandlerContext {
        thread_id: "live-test-1".into(),
        from: "test".into(),
        own_name: "coding-agent".into(),
    };

    let result = handler.handle(payload, ctx).await.unwrap();
    match result {
        HandlerResponse::Reply { payload_xml } => {
            let xml = String::from_utf8(payload_xml).unwrap();
            println!("Agent response: {xml}");
            assert!(xml.contains("<AgentResponse>"));
            assert!(xml.contains("<result>"));
            // The response should contain "4" somewhere
            assert!(xml.contains('4'), "expected '4' in response: {xml}");
        }
        HandlerResponse::Send { to, .. } => {
            panic!("expected Reply, got Send to {to} (no tools were defined)")
        }
        HandlerResponse::None => panic!("expected Reply, got None"),
    }
}

/// Test: Inject a task with tools → agent calls API → wants to use a tool → returns Send.
#[tokio::test]
async fn agent_tool_call_dispatch() {
    let pool = match LlmPool::from_env("haiku") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("ANTHROPIC_API_KEY not set — skipping live agent test");
            return;
        }
    };

    let pool = Arc::new(Mutex::new(pool));
    let tool_defs = vec![ToolDefinition {
        name: "file-read".into(),
        description: "Read file contents with line numbers.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to read"}
            },
            "required": ["path"]
        }),
    }];

    let handler = CodingAgentHandler::new(
        "test-agent".into(),
        pool,
        tool_defs,
        "You are a coding agent. When asked to read a file, use the file-read tool.".into(),
    );

    let payload = ValidatedPayload {
        xml: b"<AgentTask><task>Read the file src/main.rs</task></AgentTask>".to_vec(),
        tag: "AgentTask".into(),
    };
    let ctx = HandlerContext {
        thread_id: "live-test-2".into(),
        from: "test".into(),
        own_name: "coding-agent".into(),
    };

    let result = handler.handle(payload, ctx).await.unwrap();
    match result {
        HandlerResponse::Send { to, payload_xml } => {
            let xml = String::from_utf8(payload_xml).unwrap();
            println!("Tool call: to={to}, xml={xml}");
            assert_eq!(to, "file-read");
            assert!(xml.contains("<FileReadRequest>"));
            assert!(xml.contains("main.rs"));
        }
        HandlerResponse::Reply { payload_xml } => {
            let xml = String::from_utf8(payload_xml).unwrap();
            // Some models might not call the tool — that's acceptable for a live test
            println!("Got Reply instead of Send (model didn't call tool): {xml}");
        }
        HandlerResponse::None => panic!("expected Send or Reply, got None"),
    }
}

/// Test: Multi-turn tool sequence — task → tool call → fake result → final response.
#[tokio::test]
async fn agent_multi_turn_tool_sequence() {
    let pool = match LlmPool::from_env("haiku") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("ANTHROPIC_API_KEY not set — skipping live agent test");
            return;
        }
    };

    let pool = Arc::new(Mutex::new(pool));
    let tool_defs = build_tool_definitions(&["file-read"]);

    let handler = CodingAgentHandler::new(
        "test-agent".into(),
        pool,
        tool_defs,
        "You are a coding agent. Use file-read to read files when asked. After reading, summarize what you found.".into(),
    );

    // Step 1: Send the task
    let payload1 = ValidatedPayload {
        xml: b"<AgentTask><task>Read the file README.md and tell me what it says.</task></AgentTask>"
            .to_vec(),
        tag: "AgentTask".into(),
    };
    let ctx1 = HandlerContext {
        thread_id: "live-test-3".into(),
        from: "test".into(),
        own_name: "coding-agent".into(),
    };

    let result1 = handler.handle(payload1, ctx1).await.unwrap();

    match result1 {
        HandlerResponse::Send { to, .. } => {
            println!("Step 1 → Send to {to}");
            assert_eq!(to, "file-read");

            // Step 2: Simulate the tool response coming back
            let tool_response = b"<ToolResponse><success>true</success><result># AgentOS\n\nAn operating system for AI coding agents.\n\n## Features\n- Zero trust pipeline\n- WAL-backed kernel\n- Capability-based security</result></ToolResponse>";
            let payload2 = ValidatedPayload {
                xml: tool_response.to_vec(),
                tag: "ToolResponse".into(),
            };
            let ctx2 = HandlerContext {
                thread_id: "live-test-3".into(),
                from: "file-read".into(),
                own_name: "coding-agent".into(),
            };

            let result2 = handler.handle(payload2, ctx2).await.unwrap();

            match result2 {
                HandlerResponse::Reply { payload_xml } => {
                    let xml = String::from_utf8(payload_xml).unwrap();
                    println!("Step 2 → Reply: {xml}");
                    assert!(xml.contains("<AgentResponse>"));
                    assert!(xml.contains("<result>"));
                    // The model should mention something about the README content
                    let lower = xml.to_lowercase();
                    assert!(
                        lower.contains("agentos")
                            || lower.contains("coding")
                            || lower.contains("pipeline")
                            || lower.contains("readme"),
                        "expected response to reference README content: {xml}"
                    );
                }
                HandlerResponse::Send { to, payload_xml: xml_bytes } => {
                    // Model might want to make another tool call — that's fine
                    let xml = String::from_utf8(xml_bytes).unwrap();
                    println!("Step 2 → another Send to {to}: {xml}");
                }
                HandlerResponse::None => panic!("expected Reply or Send after tool result"),
            }
        }
        HandlerResponse::Reply { payload_xml } => {
            let xml = String::from_utf8(payload_xml).unwrap();
            println!("Model didn't use tool (replied directly): {xml}");
        }
        HandlerResponse::None => panic!("expected Send or Reply, got None"),
    }
}

/// Test: Verify that MessagesResponse with both text and tool_use blocks
/// is properly represented (forgiving ingress support at the type level).
#[test]
fn response_with_text_and_tool_use() {
    // This tests that the types support mixed content blocks
    // (the handler's forgiving ingress behavior is tested in unit tests)
    let response = agentos::llm::types::MessagesResponse {
        id: "msg_test".into(),
        model: "test".into(),
        content: vec![
            ContentBlock::Text {
                text: "I'll read that file for you.".into(),
            },
            ContentBlock::ToolUse {
                id: "toolu_forgiving".into(),
                name: "file-read".into(),
                input: serde_json::json!({"path": "test.rs"}),
            },
        ],
        stop_reason: Some("tool_use".into()),
        usage: agentos::llm::types::Usage {
            input_tokens: 10,
            output_tokens: 10,
        },
    };

    // Both text and tool_use should be accessible
    assert_eq!(response.text(), Some("I'll read that file for you."));
    assert!(response.has_tool_use());
    assert_eq!(response.tool_use_blocks().len(), 1);
}

/// Test: End-to-end codebase-index tool usage.
#[tokio::test]
async fn agent_codebase_index_tool() {
    let pool = match LlmPool::from_env("haiku") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("ANTHROPIC_API_KEY not set — skipping");
            return;
        }
    };

    let pool = Arc::new(Mutex::new(pool));
    let tool_defs = build_tool_definitions(&["codebase-index"]);

    let handler = CodingAgentHandler::new(
        "test-agent".into(),
        pool,
        tool_defs,
        "You are a coding agent. Use the codebase-index tool to search for symbols.".into(),
    );

    let payload = ValidatedPayload {
        xml: b"<AgentTask><task>Search for all functions named 'resolve_model' in the codebase using the codebase-index tool.</task></AgentTask>"
            .to_vec(),
        tag: "AgentTask".into(),
    };
    let ctx = HandlerContext {
        thread_id: "live-test-ci".into(),
        from: "test".into(),
        own_name: "coding-agent".into(),
    };

    let result = handler.handle(payload, ctx).await.unwrap();
    match result {
        HandlerResponse::Send { to, payload_xml } => {
            let xml = String::from_utf8(payload_xml).unwrap();
            println!("Codebase index call: to={to}, xml={xml}");
            assert_eq!(to, "codebase-index");
            assert!(xml.contains("<CodeIndexRequest>"));
        }
        HandlerResponse::Reply { payload_xml } => {
            let xml = String::from_utf8(payload_xml).unwrap();
            println!("Model replied directly: {xml}");
        }
        HandlerResponse::None => panic!("expected Send or Reply"),
    }
}
