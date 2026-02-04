//! Integration tests for the Ollama provider.
//!
//! These tests require a running Ollama instance at `localhost:11434`
//! with the configured model pulled. They are skipped when Ollama
//! is not reachable.
//!
//! Run with:
//! ```sh
//! # Pull the model first:
//! ollama pull llama3.2
//! # Then run:
//! cargo test -p llm-ollama --test integration
//! ```

use futures::StreamExt;
use llm_stack_core::StreamEvent;
use llm_stack_core::chat::{ChatMessage, ContentBlock, StopReason};
use llm_stack_core::provider::{ChatParams, JsonSchema, Provider, ToolChoice, ToolDefinition};
use llm_stack_ollama::{OllamaConfig, OllamaProvider};

/// The model to use for integration tests.
const TEST_MODEL: &str = "llama3.2";

/// Helper: create a provider and check if Ollama is reachable.
/// Returns `None` (and the test is skipped) if Ollama is not running.
async fn test_provider() -> Option<OllamaProvider> {
    let config = OllamaConfig {
        model: TEST_MODEL.into(),
        ..Default::default()
    };
    let provider = OllamaProvider::new(config);

    // Quick health check — try to reach Ollama
    let client = reqwest::Client::new();
    let health = client
        .get("http://localhost:11434/api/tags")
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await;

    if health.is_err() {
        return None;
    }

    Some(provider)
}

macro_rules! skip_without_ollama {
    () => {
        match test_provider().await {
            Some(p) => p,
            None => {
                eprintln!("Ollama not running, skipping integration test");
                return;
            }
        }
    };
}

#[tokio::test]
async fn test_simple_generate() {
    let provider = skip_without_ollama!();

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
}

#[tokio::test]
async fn test_generate_with_system_prompt() {
    let provider = skip_without_ollama!();

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
    let provider = skip_without_ollama!();

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
    let provider = skip_without_ollama!();

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
    let provider = skip_without_ollama!();

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

    // Ollama may or may not use tools depending on the model
    if response.stop_reason == StopReason::ToolUse {
        let tool_calls = response.tool_calls();
        assert!(!tool_calls.is_empty());
        assert_eq!(tool_calls[0].name, "get_weather");
    }
}

#[tokio::test]
async fn test_tool_result_roundtrip() {
    let provider = skip_without_ollama!();

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "city": { "type": "string" }
        },
        "required": ["city"]
    });

    let tool_def = ToolDefinition {
        name: "get_weather".into(),
        description: "Get the current weather for a city".into(),
        parameters: JsonSchema::new(schema.clone()),
        retry: None,
    };

    // First call
    let params1 = ChatParams {
        messages: vec![ChatMessage::user("What's the weather in Paris?")],
        tools: Some(vec![tool_def.clone()]),
        tool_choice: Some(ToolChoice::Auto),
        max_tokens: Some(256),
        ..Default::default()
    };

    let resp1 = provider.generate(&params1).await.unwrap();

    if resp1.stop_reason != StopReason::ToolUse {
        // Model didn't use tools — skip rest of test
        return;
    }

    let tool_call = &resp1.tool_calls()[0];

    // Second call with tool result
    let mut assistant_msg = ChatMessage::assistant("");
    assistant_msg.content = vec![ContentBlock::ToolCall((*tool_call).clone())];

    let params2 = ChatParams {
        messages: vec![
            ChatMessage::user("What's the weather in Paris?"),
            assistant_msg,
            ChatMessage::tool_result(&tool_call.id, "Sunny, 22°C"),
        ],
        tools: Some(vec![tool_def]),
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
async fn test_metadata() {
    let provider = skip_without_ollama!();
    let meta = provider.metadata();

    assert_eq!(meta.name, "ollama");
    assert_eq!(meta.model, TEST_MODEL);
    assert!(
        meta.capabilities
            .contains(&llm_stack_core::provider::Capability::Tools)
    );
    assert!(
        meta.capabilities
            .contains(&llm_stack_core::provider::Capability::Vision)
    );
}

// ── Streaming tests ──────────────────────────────────────────────────

#[tokio::test]
async fn test_stream_simple_text() {
    let provider = skip_without_ollama!();

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
    let provider = skip_without_ollama!();

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
    let provider = skip_without_ollama!();

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
        tool_choice: Some(ToolChoice::Auto),
        max_tokens: Some(256),
        ..Default::default()
    };

    let mut stream = provider.stream(&params).await.unwrap();

    let mut got_done = false;
    let mut got_tool = false;

    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::ToolCallComplete { .. } => {
                got_tool = true;
            }
            StreamEvent::Done { stop_reason } => {
                got_done = true;
                if got_tool {
                    assert_eq!(stop_reason, StopReason::ToolUse);
                }
            }
            _ => {}
        }
    }

    assert!(got_done, "Should receive Done event");
    // Tool use depends on the model — we don't assert got_tool
}
