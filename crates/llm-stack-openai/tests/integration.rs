//! Integration tests for the `OpenAI` provider.
//!
//! These tests require a valid `OPENAI_API_KEY` environment variable.
//! They are skipped (not failed) when the key is not present.
//!
//! Run with:
//! ```sh
//! OPENAI_API_KEY=sk-... cargo test -p llm-openai --test integration
//! ```

use futures::StreamExt;
use llm_stack::StreamEvent;
use llm_stack::chat::{ChatMessage, ContentBlock, StopReason};
use llm_stack::provider::{ChatParams, JsonSchema, Provider, ToolChoice, ToolDefinition};
use llm_stack_openai::{OpenAiConfig, OpenAiProvider};

/// Helper: create a provider configured for integration tests.
/// Returns `None` (and the test is skipped) if no API key is set.
fn test_provider() -> Option<OpenAiProvider> {
    let api_key = std::env::var("OPENAI_API_KEY").ok()?;
    if api_key.is_empty() {
        return None;
    }
    Some(OpenAiProvider::new(OpenAiConfig {
        api_key,
        // Use gpt-4o-mini for fast, cheap integration tests
        model: "gpt-4o-mini".into(),
        ..Default::default()
    }))
}

macro_rules! skip_without_key {
    () => {
        match test_provider() {
            Some(p) => p,
            None => {
                eprintln!("OPENAI_API_KEY not set, skipping integration test");
                return;
            }
        }
    };
}

#[tokio::test]
async fn test_simple_generate() {
    let provider = skip_without_key!();

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "What is 2+2? Reply with just the number.",
        )],
        max_tokens: Some(32),
        ..Default::default()
    };

    let response = provider.generate(&params).await.unwrap();

    assert_eq!(response.stop_reason, StopReason::EndTurn);
    let text = response.text().expect("should have text");
    assert!(text.contains('4'), "Expected '4' in response: {text}");
    assert!(response.usage.input_tokens > 0);
    assert!(response.usage.output_tokens > 0);
}

#[tokio::test]
async fn test_generate_with_system_prompt() {
    let provider = skip_without_key!();

    let params = ChatParams {
        messages: vec![ChatMessage::user("What are you?")],
        system: Some("You are a helpful pirate. Always respond in pirate speak.".into()),
        max_tokens: Some(128),
        ..Default::default()
    };

    let response = provider.generate(&params).await.unwrap();
    let text = response.text().expect("should have text");
    assert!(!text.is_empty());
}

#[tokio::test]
async fn test_generate_with_temperature() {
    let provider = skip_without_key!();

    let params = ChatParams {
        messages: vec![ChatMessage::user("Say exactly one word: 'hello'")],
        temperature: Some(0.0),
        max_tokens: Some(16),
        ..Default::default()
    };

    let response = provider.generate(&params).await.unwrap();
    assert!(response.text().is_some());
}

#[tokio::test]
async fn test_multi_turn_conversation() {
    let provider = skip_without_key!();

    let params = ChatParams {
        messages: vec![
            ChatMessage::user("My name is TestUser. Remember this."),
            ChatMessage::assistant("Got it! Your name is TestUser. I'll remember that."),
            ChatMessage::user("What is my name?"),
        ],
        max_tokens: Some(64),
        ..Default::default()
    };

    let response = provider.generate(&params).await.unwrap();
    let text = response.text().expect("should have text");
    assert!(
        text.contains("TestUser"),
        "Expected 'TestUser' in response: {text}"
    );
}

#[tokio::test]
async fn test_tool_calling() {
    let provider = skip_without_key!();

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name"
            }
        },
        "required": ["city"]
    });

    let params = ChatParams {
        messages: vec![ChatMessage::user("What is the weather in Tokyo?")],
        tools: Some(vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Get the current weather for a city".into(),
            parameters: JsonSchema::new(schema),
            retry: None,
        }]),
        tool_choice: Some(ToolChoice::Auto),
        max_tokens: Some(256),
        ..Default::default()
    };

    let response = provider.generate(&params).await.unwrap();

    assert_eq!(response.stop_reason, StopReason::ToolUse);
    let tool_calls = response.tool_calls();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "get_weather");
    assert!(tool_calls[0].arguments.get("city").is_some());
}

#[tokio::test]
async fn test_tool_result_roundtrip() {
    let provider = skip_without_key!();

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "city": { "type": "string" }
        },
        "required": ["city"]
    });

    // First call: model should request tool use
    let params1 = ChatParams {
        messages: vec![ChatMessage::user("What's the weather in Paris?")],
        tools: Some(vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Get the current weather for a city".into(),
            parameters: JsonSchema::new(schema.clone()),
            retry: None,
        }]),
        tool_choice: Some(ToolChoice::Required),
        max_tokens: Some(256),
        ..Default::default()
    };

    let resp1 = provider.generate(&params1).await.unwrap();
    assert_eq!(resp1.stop_reason, StopReason::ToolUse);

    let tool_call = &resp1.tool_calls()[0];

    // Second call: provide tool result
    let mut assistant_msg = ChatMessage::assistant("");
    assistant_msg.content = vec![ContentBlock::ToolCall((*tool_call).clone())];

    let params2 = ChatParams {
        messages: vec![
            ChatMessage::user("What's the weather in Paris?"),
            assistant_msg,
            ChatMessage::tool_result(&tool_call.id, "Sunny, 22°C"),
        ],
        tools: Some(vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Get the current weather for a city".into(),
            parameters: JsonSchema::new(schema),
            retry: None,
        }]),
        max_tokens: Some(256),
        ..Default::default()
    };

    let resp2 = provider.generate(&params2).await.unwrap();
    let text = resp2.text().expect("should have text after tool result");
    assert!(
        text.contains("22") || text.contains("Sunny") || text.contains("Paris"),
        "Expected weather info in: {text}"
    );
}

#[tokio::test]
async fn test_max_tokens_limit() {
    let provider = skip_without_key!();

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "Write a very long essay about the history of computing.",
        )],
        max_tokens: Some(10),
        ..Default::default()
    };

    let response = provider.generate(&params).await.unwrap();
    assert_eq!(response.stop_reason, StopReason::MaxTokens);
}

#[tokio::test]
async fn test_invalid_api_key() {
    let provider = OpenAiProvider::new(OpenAiConfig {
        api_key: "sk-invalid-key".into(),
        model: "gpt-4o-mini".into(),
        ..Default::default()
    });

    let params = ChatParams {
        messages: vec![ChatMessage::user("Hello")],
        max_tokens: Some(16),
        ..Default::default()
    };

    let err = provider.generate(&params).await.unwrap_err();
    assert!(
        matches!(err, llm_stack::LlmError::Auth(_)),
        "Expected Auth error, got: {err:?}"
    );
}

#[tokio::test]
async fn test_metadata() {
    let provider = skip_without_key!();
    let meta = provider.metadata();

    assert_eq!(meta.name, "openai");
    assert_eq!(meta.model, "gpt-4o-mini");
    assert!(
        meta.capabilities
            .contains(&llm_stack::provider::Capability::Tools)
    );
    assert!(
        meta.capabilities
            .contains(&llm_stack::provider::Capability::Vision)
    );
}

// ── Streaming tests ──────────────────────────────────────────────────

#[tokio::test]
async fn test_stream_simple_text() {
    let provider = skip_without_key!();

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "What is 2+2? Reply with just the number.",
        )],
        max_tokens: Some(32),
        ..Default::default()
    };

    let mut stream = provider.stream(&params).await.unwrap();

    let mut text = String::new();
    let mut got_done = false;

    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::TextDelta(t) => text.push_str(&t),
            StreamEvent::Done { stop_reason } => {
                assert_eq!(stop_reason, StopReason::EndTurn);
                got_done = true;
            }
            _ => {}
        }
    }

    assert!(got_done, "Should receive Done event");
    assert!(text.contains('4'), "Expected '4' in streamed text: {text}");
}

#[tokio::test]
async fn test_stream_with_usage() {
    let provider = skip_without_key!();

    let params = ChatParams {
        messages: vec![ChatMessage::user("Say hi")],
        max_tokens: Some(16),
        ..Default::default()
    };

    let mut stream = provider.stream(&params).await.unwrap();

    let mut got_usage = false;
    while let Some(event) = stream.next().await {
        if let Ok(StreamEvent::Usage(usage)) = event {
            if usage.input_tokens > 0 || usage.output_tokens > 0 {
                got_usage = true;
            }
        }
    }

    assert!(got_usage, "Should receive at least one Usage event");
}

#[tokio::test]
async fn test_stream_tool_calling() {
    let provider = skip_without_key!();

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "city": { "type": "string" }
        },
        "required": ["city"]
    });

    let params = ChatParams {
        messages: vec![ChatMessage::user("What is the weather in Tokyo?")],
        tools: Some(vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Get the current weather for a city".into(),
            parameters: JsonSchema::new(schema),
            retry: None,
        }]),
        tool_choice: Some(ToolChoice::Required),
        max_tokens: Some(256),
        ..Default::default()
    };

    let mut stream = provider.stream(&params).await.unwrap();

    let mut got_start = false;
    let mut got_delta = false;
    let mut got_complete = false;
    let mut got_done = false;
    let mut tool_name = String::new();

    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::ToolCallStart { name, .. } => {
                tool_name = name;
                got_start = true;
            }
            StreamEvent::ToolCallDelta { .. } => {
                got_delta = true;
            }
            StreamEvent::ToolCallComplete { call, .. } => {
                assert_eq!(call.name, "get_weather");
                assert!(call.arguments.get("city").is_some());
                got_complete = true;
            }
            StreamEvent::Done { stop_reason } => {
                assert_eq!(stop_reason, StopReason::ToolUse);
                got_done = true;
            }
            _ => {}
        }
    }

    assert_eq!(tool_name, "get_weather");
    assert!(got_start, "Should receive ToolCallStart");
    assert!(got_delta, "Should receive ToolCallDelta");
    assert!(got_complete, "Should receive ToolCallComplete");
    assert!(got_done, "Should receive Done with ToolUse reason");
}

#[tokio::test]
async fn test_stream_max_tokens() {
    let provider = skip_without_key!();

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "Write a long essay about the history of computing.",
        )],
        max_tokens: Some(10),
        ..Default::default()
    };

    let mut stream = provider.stream(&params).await.unwrap();

    let mut stop_reason = None;
    while let Some(event) = stream.next().await {
        if let Ok(StreamEvent::Done { stop_reason: sr }) = event {
            stop_reason = Some(sr);
        }
    }

    assert_eq!(stop_reason, Some(StopReason::MaxTokens));
}
