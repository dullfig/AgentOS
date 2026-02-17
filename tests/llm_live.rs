//! Live round-trip test against the real Anthropic API.
//!
//! Requires ANTHROPIC_API_KEY in the environment. Skips gracefully if unset.
//! Uses Haiku — cheapest model, ~0.001 cent per call.

use bestcode::llm::LlmPool;
use bestcode::llm::types::{ContentBlock, Message, ToolDefinition};

#[tokio::test]
async fn haiku_round_trip() {
    let pool = match LlmPool::from_env("haiku") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("ANTHROPIC_API_KEY not set — skipping live test");
            return;
        }
    };

    let messages = vec![Message::text(
        "user",
        "What is 2+2? Reply with just the number, nothing else.",
    )];

    let resp = pool
        .complete(None, messages, 32, None)
        .await
        .expect("API call failed");

    let text = resp.text().expect("no text in response");
    println!("Model: {}", resp.model);
    println!("Response: {text}");
    println!(
        "Tokens: {} in / {} out",
        resp.usage.input_tokens, resp.usage.output_tokens
    );
    println!("Stop reason: {:?}", resp.stop_reason);

    assert!(text.contains('4'), "expected '4' in response, got: {text}");
    assert!(resp.usage.input_tokens > 0);
    assert!(resp.usage.output_tokens > 0);
}

#[tokio::test]
async fn haiku_tool_use_round_trip() {
    let pool = match LlmPool::from_env("haiku") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("ANTHROPIC_API_KEY not set — skipping live test");
            return;
        }
    };

    let tools = vec![ToolDefinition {
        name: "calculator".into(),
        description: "Evaluate a mathematical expression and return the result.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate"
                }
            },
            "required": ["expression"]
        }),
    }];

    let messages = vec![Message::text(
        "user",
        "Use the calculator tool to compute 17 * 23.",
    )];

    let resp = pool
        .complete_with_tools(None, messages, 256, None, tools)
        .await
        .expect("API call with tools failed");

    println!("Model: {}", resp.model);
    println!("Stop reason: {:?}", resp.stop_reason);
    println!("Content blocks: {}", resp.content.len());
    for (i, block) in resp.content.iter().enumerate() {
        match block {
            ContentBlock::Text { text } => println!("  [{i}] Text: {text}"),
            ContentBlock::ToolUse { id, name, input } => {
                println!("  [{i}] ToolUse: id={id}, name={name}, input={input}")
            }
            ContentBlock::ToolResult { .. } => println!("  [{i}] ToolResult (unexpected)"),
        }
    }

    assert_eq!(
        resp.stop_reason.as_deref(),
        Some("tool_use"),
        "expected stop_reason=tool_use"
    );
    assert!(resp.has_tool_use(), "expected at least one tool_use block");

    let tool_blocks = resp.tool_use_blocks();
    assert!(!tool_blocks.is_empty());
    match tool_blocks[0] {
        ContentBlock::ToolUse { name, input, .. } => {
            assert_eq!(name, "calculator");
            assert!(input.get("expression").is_some());
        }
        _ => panic!("expected ToolUse block"),
    }
}
