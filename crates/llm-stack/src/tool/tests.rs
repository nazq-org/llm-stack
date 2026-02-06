//! Tests for the tool module.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};

use super::*;
use crate::ToolCall;
use crate::chat::{ChatMessage, ChatRole, ContentBlock, StopReason};
use crate::provider::{ChatParams, JsonSchema, ToolDefinition};
use crate::test_helpers::{mock_for, sample_response, sample_tool_response};
use crate::usage::Usage;

fn number_schema() -> JsonSchema {
    JsonSchema::new(json!({
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    }))
}

/// A simple tool handler for testing — adds two numbers.
struct AddTool;

impl ToolHandler<()> for AddTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "add".into(),
            description: "Add two numbers".into(),
            parameters: number_schema(),
            retry: None,
        }
    }

    fn execute<'a>(
        &'a self,
        input: Value,
        _ctx: &'a (),
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let a = input["a"].as_f64().unwrap_or(0.0);
            let b = input["b"].as_f64().unwrap_or(0.0);
            Ok(ToolOutput::new(format!("{}", a + b)))
        })
    }
}

/// A tool that always fails, for testing error paths.
struct FailTool;

impl ToolHandler<()> for FailTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fail".into(),
            description: "Always fails".into(),
            parameters: JsonSchema::new(json!({"type": "object"})),
            retry: None,
        }
    }

    fn execute<'a>(
        &'a self,
        input: Value,
        _ctx: &'a (),
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>> {
        let _ = input;
        Box::pin(async move { Err(ToolError::new("intentional failure")) })
    }
}

// ── ToolHandler tests ───────────────────────────────────────────

#[test]
fn test_tool_handler_is_object_safe() {
    fn assert_object_safe(_: &dyn ToolHandler<()>) {}
    assert_object_safe(&AddTool);
}

#[test]
fn test_tool_error_display() {
    let err = ToolError::new("something broke");
    assert_eq!(format!("{err}"), "something broke");
}

#[test]
fn test_tool_handler_definition() {
    let def = AddTool.definition();
    assert_eq!(def.name, "add");
    assert_eq!(def.description, "Add two numbers");
}

#[tokio::test]
async fn test_tool_handler_execute() {
    let result = AddTool.execute(json!({"a": 2, "b": 3}), &()).await.unwrap();
    assert_eq!(result.content, "5");
}

#[tokio::test]
async fn test_tool_handler_error() {
    let result = FailTool.execute(json!({}), &()).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().message, "intentional failure");
}

#[test]
fn test_fn_tool_handler() {
    let handler = tool_fn(
        ToolDefinition {
            name: "greet".into(),
            description: "Say hello".into(),
            parameters: JsonSchema::new(json!({"type": "object"})),
            retry: None,
        },
        |_input: Value| async { Ok("Hello!".to_string()) },
    );
    assert_eq!(handler.definition().name, "greet");
}

// ── ToolRegistry tests ──────────────────────────────────────────

#[test]
fn test_registry_empty() {
    let registry: ToolRegistry<()> = ToolRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
    assert!(registry.definitions().is_empty());
}

#[test]
fn test_registry_register_and_get() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    assert_eq!(registry.len(), 1);
    assert!(registry.contains("add"));
    assert!(!registry.contains("subtract"));
    assert!(registry.get("add").is_some());
}

#[test]
fn test_registry_definitions() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);
    registry.register(FailTool);

    let defs = registry.definitions();
    assert_eq!(defs.len(), 2);
    let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"add"));
    assert!(names.contains(&"fail"));
}

#[test]
fn test_registry_overwrite_duplicate() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);
    registry.register(AddTool);
    assert_eq!(registry.len(), 1);
}

#[tokio::test]
async fn test_registry_execute_valid() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let call = ToolCall {
        id: "call_1".into(),
        name: "add".into(),
        arguments: json!({"a": 10, "b": 20}),
    };
    let result = registry.execute(&call, &()).await;

    assert!(!result.is_error);
    assert_eq!(result.content, "30");
    assert_eq!(result.tool_call_id, "call_1");
}

#[tokio::test]
async fn test_registry_execute_unknown_tool() {
    let registry: ToolRegistry<()> = ToolRegistry::new();
    let call = ToolCall {
        id: "call_1".into(),
        name: "nonexistent".into(),
        arguments: json!({}),
    };
    let result = registry.execute(&call, &()).await;

    assert!(result.is_error);
    assert!(result.content.contains("Unknown tool"));
}

#[tokio::test]
async fn test_registry_execute_schema_validation_failure() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let call = ToolCall {
        id: "call_1".into(),
        name: "add".into(),
        arguments: json!({"a": "not a number", "b": 5}),
    };
    let result = registry.execute(&call, &()).await;

    assert!(result.is_error);
    assert!(result.content.contains("Invalid arguments"));
}

#[tokio::test]
async fn test_registry_execute_missing_required_field() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let call = ToolCall {
        id: "call_1".into(),
        name: "add".into(),
        arguments: json!({"a": 5}),
    };
    let result = registry.execute(&call, &()).await;

    assert!(result.is_error);
    assert!(result.content.contains("Invalid arguments"));
}

#[tokio::test]
async fn test_registry_execute_handler_error() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(FailTool);

    let call = ToolCall {
        id: "call_1".into(),
        name: "fail".into(),
        arguments: json!({}),
    };
    let result = registry.execute(&call, &()).await;

    assert!(result.is_error);
    assert_eq!(result.content, "intentional failure");
}

#[tokio::test]
async fn test_registry_execute_all_sequential() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let calls = vec![
        ToolCall {
            id: "c1".into(),
            name: "add".into(),
            arguments: json!({"a": 1, "b": 2}),
        },
        ToolCall {
            id: "c2".into(),
            name: "add".into(),
            arguments: json!({"a": 3, "b": 4}),
        },
    ];
    let results = registry.execute_all(&calls, &(), false).await;

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].content, "3");
    assert_eq!(results[1].content, "7");
}

#[tokio::test]
async fn test_registry_execute_all_parallel() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let calls = vec![
        ToolCall {
            id: "c1".into(),
            name: "add".into(),
            arguments: json!({"a": 10, "b": 20}),
        },
        ToolCall {
            id: "c2".into(),
            name: "add".into(),
            arguments: json!({"a": 30, "b": 40}),
        },
    ];
    let results = registry.execute_all(&calls, &(), true).await;

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].content, "30");
    assert_eq!(results[1].content, "70");
}

#[tokio::test]
async fn test_registry_execute_all_with_failure() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);
    registry.register(FailTool);

    let calls = vec![
        ToolCall {
            id: "c1".into(),
            name: "add".into(),
            arguments: json!({"a": 1, "b": 2}),
        },
        ToolCall {
            id: "c2".into(),
            name: "fail".into(),
            arguments: json!({}),
        },
    ];
    let results = registry.execute_all(&calls, &(), true).await;

    assert!(!results[0].is_error);
    assert_eq!(results[0].content, "3");
    assert!(results[1].is_error);
    assert_eq!(results[1].content, "intentional failure");
}

#[test]
fn test_registry_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ToolRegistry>();
}

#[test]
fn test_registry_debug() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);
    let debug = format!("{registry:?}");
    assert!(debug.contains("ToolRegistry"));
    assert!(debug.contains("add"));
}

// ── tool_loop tests ─────────────────────────────────────────────

#[tokio::test]
async fn test_tool_loop_no_tool_calls() {
    let mock = mock_for("test", "test-model");
    mock.queue_response(sample_response("Hello!"));

    let registry: ToolRegistry<()> = ToolRegistry::new();
    let params = ChatParams {
        messages: vec![ChatMessage::user("Hi")],
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, ToolLoopConfig::default(), &())
        .await
        .unwrap();

    assert_eq!(result.iterations, 1);
    assert_eq!(result.response.text(), Some("Hello!"));
}

#[tokio::test]
async fn test_tool_loop_one_iteration() {
    let mock = mock_for("test", "test-model");

    // First call: model requests tool
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "call_1".into(),
        name: "add".into(),
        arguments: json!({"a": 2, "b": 3}),
    }]));
    // Second call: model returns text
    mock.queue_response(sample_response("The answer is 5"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("What is 2 + 3?")],
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, ToolLoopConfig::default(), &())
        .await
        .unwrap();

    assert_eq!(result.iterations, 2);
    assert_eq!(result.response.text(), Some("The answer is 5"));
    assert_eq!(result.total_usage.input_tokens, 200); // 100 * 2 iterations
}

#[tokio::test]
async fn test_tool_loop_multiple_iterations() {
    let mock = mock_for("test", "test-model");

    // Iteration 1: tool call
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }]));
    // Iteration 2: another tool call
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c2".into(),
        name: "add".into(),
        arguments: json!({"a": 3, "b": 4}),
    }]));
    // Iteration 3: final text
    mock.queue_response(sample_response("Done"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Chain calls")],
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, ToolLoopConfig::default(), &())
        .await
        .unwrap();

    assert_eq!(result.iterations, 3);
    assert_eq!(result.response.text(), Some("Done"));
}

#[tokio::test]
async fn test_tool_loop_max_iterations_exceeded() {
    let mock = mock_for("test", "test-model");

    // Queue more tool calls than the limit allows
    for _ in 0..5 {
        mock.queue_response(sample_tool_response(vec![ToolCall {
            id: "c".into(),
            name: "add".into(),
            arguments: json!({"a": 1, "b": 2}),
        }]));
    }

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Loop")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        max_iterations: 3,
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();
    assert!(matches!(
        result.termination_reason,
        TerminationReason::MaxIterations { limit: 3 }
    ));
}

#[tokio::test]
async fn test_tool_loop_approval_deny() {
    let mock = mock_for("test", "test-model");

    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "call_1".into(),
        name: "add".into(),
        arguments: json!({"a": 2, "b": 3}),
    }]));
    mock.queue_response(sample_response("OK denied"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Denied")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        on_tool_call: Some(Arc::new(|_call| ToolApproval::Deny("not allowed".into()))),
        ..Default::default()
    };

    let _result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    // Check the tool result sent back was an error
    let recorded = mock.recorded_calls();
    let last_call = &recorded[1];
    let tool_msgs: Vec<_> = last_call
        .messages
        .iter()
        .filter(|m| m.role == ChatRole::Tool)
        .collect();
    assert!(!tool_msgs.is_empty());
}

#[tokio::test]
async fn test_tool_loop_approval_modify() {
    let mock = mock_for("test", "test-model");

    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "call_1".into(),
        name: "add".into(),
        arguments: json!({"a": 2, "b": 3}),
    }]));
    mock.queue_response(sample_response("Modified"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Modify")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        on_tool_call: Some(Arc::new(|_call| {
            ToolApproval::Modify(json!({"a": 100, "b": 200}))
        })),
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    assert_eq!(result.iterations, 2);
    // The tool was called with modified args, so result should reflect that
    let recorded = mock.recorded_calls();
    let tool_msgs: Vec<_> = recorded[1]
        .messages
        .iter()
        .filter(|m| m.role == ChatRole::Tool)
        .collect();
    // Tool result should contain "300" (100 + 200)
    let tool_content = &tool_msgs[0].content;
    let has_300 = tool_content.iter().any(|b| {
        if let ContentBlock::ToolResult(r) = b {
            r.content == "300"
        } else {
            false
        }
    });
    assert!(has_300, "Expected tool result with '300'");
}

#[tokio::test]
async fn test_tool_loop_parallel_execution() {
    let mock = mock_for("test", "test-model");

    mock.queue_response(sample_tool_response(vec![
        ToolCall {
            id: "c1".into(),
            name: "add".into(),
            arguments: json!({"a": 1, "b": 2}),
        },
        ToolCall {
            id: "c2".into(),
            name: "add".into(),
            arguments: json!({"a": 3, "b": 4}),
        },
    ]));
    mock.queue_response(sample_response("Both done"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Parallel")],
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, ToolLoopConfig::default(), &())
        .await
        .unwrap();

    assert_eq!(result.iterations, 2);
    // Both tool results should be in the second call's messages
    let recorded = mock.recorded_calls();
    let tool_msgs: Vec<_> = recorded[1]
        .messages
        .iter()
        .filter(|m| m.role == ChatRole::Tool)
        .collect();
    assert_eq!(tool_msgs.len(), 2);
}

#[tokio::test]
async fn test_tool_loop_unknown_tool_sends_error_result() {
    let mock = mock_for("test", "test-model");

    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "call_1".into(),
        name: "unknown_tool".into(),
        arguments: json!({}),
    }]));
    mock.queue_response(sample_response("Handled unknown"));

    let registry: ToolRegistry<()> = ToolRegistry::new();
    let params = ChatParams {
        messages: vec![ChatMessage::user("Unknown")],
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, ToolLoopConfig::default(), &())
        .await
        .unwrap();

    assert_eq!(result.iterations, 2);
}

#[tokio::test]
async fn test_tool_loop_usage_accumulation() {
    let mock = mock_for("test", "test-model");

    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }]));
    mock.queue_response(sample_response("Done"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Usage test")],
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, ToolLoopConfig::default(), &())
        .await
        .unwrap();

    // sample_usage returns 100 input, 50 output per call
    assert_eq!(result.total_usage.input_tokens, 200);
    assert_eq!(result.total_usage.output_tokens, 100);
}

// ── tool_loop_stream tests ──────────────────────────────────────

#[tokio::test]
async fn test_tool_loop_stream_no_tools() {
    use crate::stream::StreamEvent;
    use futures::StreamExt;

    let mock = Arc::new(mock_for("test", "test-model"));
    mock.queue_stream(vec![
        StreamEvent::TextDelta("Hello".into()),
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
        },
    ]);

    let registry = Arc::new(ToolRegistry::new());
    let params = ChatParams {
        messages: vec![ChatMessage::user("Hi")],
        ..Default::default()
    };

    let stream = tool_loop_stream(
        mock,
        registry,
        params,
        ToolLoopConfig::default(),
        Arc::new(()),
    );
    let events: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(events.len(), 2);
    assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Hello"));
    assert!(matches!(
        &events[1],
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn
        }
    ));
}

#[tokio::test]
async fn test_tool_loop_stream_one_iteration() {
    use crate::stream::StreamEvent;
    use futures::StreamExt;

    let mock = Arc::new(mock_for("test", "test-model"));

    // First stream: tool call
    mock.queue_stream(vec![
        StreamEvent::ToolCallStart {
            index: 0,
            id: "call_1".into(),
            name: "add".into(),
        },
        StreamEvent::ToolCallDelta {
            index: 0,
            json_chunk: r#"{"a":2,"b":3}"#.into(),
        },
        StreamEvent::ToolCallComplete {
            index: 0,
            call: ToolCall {
                id: "call_1".into(),
                name: "add".into(),
                arguments: json!({"a": 2, "b": 3}),
            },
        },
        StreamEvent::Done {
            stop_reason: StopReason::ToolUse,
        },
    ]);

    // Second stream: text response
    mock.queue_stream(vec![
        StreamEvent::TextDelta("The answer is 5".into()),
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
        },
    ]);

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);
    let registry = Arc::new(registry);

    let params = ChatParams {
        messages: vec![ChatMessage::user("What is 2+3?")],
        ..Default::default()
    };

    let stream = tool_loop_stream(
        mock,
        registry,
        params,
        ToolLoopConfig::default(),
        Arc::new(()),
    );
    let events: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // Should have: ToolCallStart, ToolCallDelta, ToolCallComplete from first stream
    // (Done is swallowed for tool use), then TextDelta + Done from second stream
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCallStart { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "The answer is 5"))
    );
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn
        }
    )));
}

#[tokio::test]
async fn test_tool_loop_stream_max_iterations() {
    use crate::stream::StreamEvent;
    use futures::StreamExt;

    let mock = Arc::new(mock_for("test", "test-model"));

    // Queue more tool-calling streams than the limit
    for _ in 0..5 {
        mock.queue_stream(vec![
            StreamEvent::ToolCallComplete {
                index: 0,
                call: ToolCall {
                    id: "c".into(),
                    name: "add".into(),
                    arguments: json!({"a": 1, "b": 2}),
                },
            },
            StreamEvent::Done {
                stop_reason: StopReason::ToolUse,
            },
        ]);
    }

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);
    let registry = Arc::new(registry);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Loop")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        max_iterations: 2,
        ..Default::default()
    };

    let stream = tool_loop_stream(mock, registry, params, config, Arc::new(()));
    let results: Vec<_> = stream.collect::<Vec<_>>().await;

    // Should have an error
    assert!(results.iter().any(Result::is_err));
}

// ── Usage addition tests ────────────────────────────────────────

#[test]
fn test_usage_add_assign() {
    let mut total = Usage {
        input_tokens: 100,
        output_tokens: 50,
        reasoning_tokens: Some(10),
        cache_read_tokens: None,
        cache_write_tokens: None,
    };
    total += Usage {
        input_tokens: 200,
        output_tokens: 75,
        reasoning_tokens: Some(20),
        cache_read_tokens: Some(5),
        cache_write_tokens: None,
    };

    assert_eq!(total.input_tokens, 300);
    assert_eq!(total.output_tokens, 125);
    assert_eq!(total.reasoning_tokens, Some(30));
    assert_eq!(total.cache_read_tokens, Some(5));
    assert!(total.cache_write_tokens.is_none());
}

#[test]
fn test_usage_add_assign_both_none() {
    let mut a = Usage::default();
    a += Usage::default();
    assert!(a.reasoning_tokens.is_none());
}

// ── ChatMessage::tool_result_full test ──────────────────────────

#[test]
fn test_tool_result_full_message() {
    use crate::chat::ToolResult;

    let result = ToolResult {
        tool_call_id: "c1".into(),
        content: "42".into(),
        is_error: false,
    };
    let msg = ChatMessage::tool_result_full(result);
    assert_eq!(msg.role, ChatRole::Tool);
    assert!(matches!(&msg.content[0], ContentBlock::ToolResult(r) if r.content == "42"));
}

// ── ToolLoopEvent tests ────────────────────────────────────────────

#[test]
fn test_tool_loop_event_debug() {
    let event = ToolLoopEvent::IterationStart {
        iteration: 1,
        message_count: 5,
    };
    let debug = format!("{event:?}");
    assert!(debug.contains("IterationStart"));
    assert!(debug.contains("iteration: 1"));
}

#[test]
fn test_tool_loop_event_clone() {
    let event = ToolLoopEvent::ToolExecutionStart {
        call_id: "c1".into(),
        tool_name: "add".into(),
        arguments: json!({"a": 1}),
    };
    let cloned = event.clone();
    assert!(
        matches!(cloned, ToolLoopEvent::ToolExecutionStart { tool_name, .. } if tool_name == "add")
    );
}

#[tokio::test]
async fn test_tool_loop_emits_iteration_start() {
    use std::sync::Mutex;

    let mock = mock_for("test", "test-model");
    mock.queue_response(sample_response("Hello!"));

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = Arc::clone(&events);

    let registry: ToolRegistry<()> = ToolRegistry::new();
    let params = ChatParams {
        messages: vec![ChatMessage::user("Hi")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        on_event: Some(Arc::new(move |event| {
            events_clone.lock().unwrap().push(event);
        })),
        ..Default::default()
    };

    let _result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    let captured = events.lock().unwrap();
    assert!(
        captured
            .iter()
            .any(|e| matches!(e, ToolLoopEvent::IterationStart { iteration: 1, .. }))
    );
    assert!(captured.iter().any(|e| matches!(
        e,
        ToolLoopEvent::LlmResponseReceived {
            iteration: 1,
            has_tool_calls: false,
            ..
        }
    )));
}

#[tokio::test]
async fn test_tool_loop_emits_tool_execution_events() {
    use std::sync::Mutex;

    let mock = mock_for("test", "test-model");

    // First call: model requests tool
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "call_1".into(),
        name: "add".into(),
        arguments: json!({"a": 2, "b": 3}),
    }]));
    // Second call: model returns text
    mock.queue_response(sample_response("The answer is 5"));

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = Arc::clone(&events);

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("What is 2 + 3?")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        on_event: Some(Arc::new(move |event| {
            events_clone.lock().unwrap().push(event);
        })),
        ..Default::default()
    };

    let _result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    let captured = events.lock().unwrap();

    // Should have: IterationStart(1), LlmResponseReceived(1, has_tools=true),
    // ToolExecutionStart, ToolExecutionEnd, IterationStart(2), LlmResponseReceived(2, has_tools=false)
    assert!(captured.iter().any(
        |e| matches!(e, ToolLoopEvent::ToolExecutionStart { tool_name, .. } if tool_name == "add")
    ));
    assert!(captured.iter().any(|e| matches!(e, ToolLoopEvent::ToolExecutionEnd { tool_name, result, .. } if tool_name == "add" && result.content == "5")));

    // Check iteration count
    let iteration_starts: Vec<_> = captured
        .iter()
        .filter(|e| matches!(e, ToolLoopEvent::IterationStart { .. }))
        .collect();
    assert_eq!(iteration_starts.len(), 2);
}

#[tokio::test]
async fn test_tool_loop_event_duration_is_positive() {
    use std::sync::Mutex;

    let mock = mock_for("test", "test-model");

    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "call_1".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }]));
    mock.queue_response(sample_response("Done"));

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = Arc::clone(&events);

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Add")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        on_event: Some(Arc::new(move |event| {
            events_clone.lock().unwrap().push(event);
        })),
        ..Default::default()
    };

    let _result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    let captured = events.lock().unwrap();
    let end_event = captured
        .iter()
        .find(|e| matches!(e, ToolLoopEvent::ToolExecutionEnd { .. }));
    assert!(end_event.is_some());

    if let Some(ToolLoopEvent::ToolExecutionEnd { duration, .. }) = end_event {
        // Duration should be non-negative (it's a Duration, always >= 0)
        assert!(*duration >= std::time::Duration::ZERO);
    }
}

#[tokio::test]
async fn test_tool_loop_events_with_parallel_execution() {
    use std::sync::Mutex;

    let mock = mock_for("test", "test-model");

    mock.queue_response(sample_tool_response(vec![
        ToolCall {
            id: "c1".into(),
            name: "add".into(),
            arguments: json!({"a": 1, "b": 2}),
        },
        ToolCall {
            id: "c2".into(),
            name: "add".into(),
            arguments: json!({"a": 3, "b": 4}),
        },
    ]));
    mock.queue_response(sample_response("Both done"));

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = Arc::clone(&events);

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Parallel")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        parallel_tool_execution: true,
        on_event: Some(Arc::new(move |event| {
            events_clone.lock().unwrap().push(event);
        })),
        ..Default::default()
    };

    let _result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    let captured = events.lock().unwrap();

    // Should have 2 start events and 2 end events
    let starts: Vec<_> = captured
        .iter()
        .filter(|e| matches!(e, ToolLoopEvent::ToolExecutionStart { .. }))
        .collect();
    let ends: Vec<_> = captured
        .iter()
        .filter(|e| matches!(e, ToolLoopEvent::ToolExecutionEnd { .. }))
        .collect();
    assert_eq!(starts.len(), 2);
    assert_eq!(ends.len(), 2);
}

#[tokio::test]
async fn test_tool_loop_stream_emits_events() {
    use crate::stream::StreamEvent;
    use futures::StreamExt;
    use std::sync::Mutex;

    let mock = Arc::new(mock_for("test", "test-model"));

    // First stream: tool call
    mock.queue_stream(vec![
        StreamEvent::ToolCallStart {
            index: 0,
            id: "call_1".into(),
            name: "add".into(),
        },
        StreamEvent::ToolCallDelta {
            index: 0,
            json_chunk: r#"{"a":2,"b":3}"#.into(),
        },
        StreamEvent::ToolCallComplete {
            index: 0,
            call: ToolCall {
                id: "call_1".into(),
                name: "add".into(),
                arguments: json!({"a": 2, "b": 3}),
            },
        },
        StreamEvent::Done {
            stop_reason: StopReason::ToolUse,
        },
    ]);

    // Second stream: text response
    mock.queue_stream(vec![
        StreamEvent::TextDelta("The answer is 5".into()),
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
        },
    ]);

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = Arc::clone(&events);

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);
    let registry = Arc::new(registry);

    let params = ChatParams {
        messages: vec![ChatMessage::user("What is 2+3?")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        on_event: Some(Arc::new(move |event| {
            events_clone.lock().unwrap().push(event);
        })),
        ..Default::default()
    };

    let stream = tool_loop_stream(mock, registry, params, config, Arc::new(()));
    let _: Vec<_> = stream.collect::<Vec<_>>().await;

    let captured = events.lock().unwrap();

    // Should have iteration starts, LLM response events, and tool execution events
    assert!(
        captured
            .iter()
            .any(|e| matches!(e, ToolLoopEvent::IterationStart { .. }))
    );
    assert!(
        captured
            .iter()
            .any(|e| matches!(e, ToolLoopEvent::LlmResponseReceived { .. }))
    );
    assert!(
        captured
            .iter()
            .any(|e| matches!(e, ToolLoopEvent::ToolExecutionStart { .. }))
    );
    assert!(
        captured
            .iter()
            .any(|e| matches!(e, ToolLoopEvent::ToolExecutionEnd { .. }))
    );
}

#[test]
fn test_tool_loop_config_debug_with_event() {
    let config = ToolLoopConfig {
        on_event: Some(Arc::new(|_| {})),
        ..Default::default()
    };
    let debug = format!("{config:?}");
    assert!(debug.contains("has_on_event: true"));
}

#[test]
fn test_tool_loop_config_debug_without_event() {
    let config = ToolLoopConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("has_on_event: false"));
}

// ── Stop condition tests ─────────────────────────────────────────

#[test]
fn test_stop_decision_equality() {
    assert_eq!(StopDecision::Continue, StopDecision::Continue);
    assert_eq!(StopDecision::Stop, StopDecision::Stop);
    assert_eq!(
        StopDecision::StopWithReason("done".into()),
        StopDecision::StopWithReason("done".into())
    );
    assert_ne!(StopDecision::Continue, StopDecision::Stop);
}

#[test]
fn test_stop_decision_debug() {
    let decision = StopDecision::StopWithReason("token limit".into());
    let debug = format!("{decision:?}");
    assert!(debug.contains("StopWithReason"));
    assert!(debug.contains("token limit"));
}

#[test]
fn test_stop_context_debug() {
    use crate::chat::ChatResponse;

    let response = ChatResponse {
        content: vec![ContentBlock::Text("test".into())],
        usage: Usage::default(),
        stop_reason: StopReason::EndTurn,
        model: "test".into(),
        metadata: std::collections::HashMap::new(),
    };
    let usage = Usage::default();
    let results: Vec<crate::chat::ToolResult> = vec![];
    let ctx = StopContext {
        iteration: 1,
        response: &response,
        total_usage: &usage,
        tool_calls_executed: 5,
        last_tool_results: &results,
    };
    let debug = format!("{ctx:?}");
    assert!(debug.contains("iteration: 1"));
    assert!(debug.contains("tool_calls_executed: 5"));
}

#[test]
fn test_tool_loop_config_debug_with_stop_when() {
    let config = ToolLoopConfig {
        stop_when: Some(Arc::new(|_| StopDecision::Continue)),
        ..Default::default()
    };
    let debug = format!("{config:?}");
    assert!(debug.contains("has_stop_when: true"));
}

#[test]
fn test_tool_loop_config_debug_without_stop_when() {
    let config = ToolLoopConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("has_stop_when: false"));
}

#[tokio::test]
async fn test_tool_loop_stop_on_first_response() {
    let mock = mock_for("test", "test-model");

    // Queue a response that would normally trigger a tool call
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "call_1".into(),
        name: "add".into(),
        arguments: json!({"a": 2, "b": 3}),
    }]));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("What is 2 + 3?")],
        ..Default::default()
    };

    // Stop immediately on first response
    let config = ToolLoopConfig {
        stop_when: Some(Arc::new(|_| StopDecision::Stop)),
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    // Should stop after iteration 1, no tool execution
    assert_eq!(result.iterations, 1);
    // The response should have tool calls (we stopped before executing them)
    assert!(!result.response.tool_calls().is_empty());
}

#[tokio::test]
async fn test_tool_loop_stop_after_tool_call_limit() {
    let mock = mock_for("test", "test-model");

    // First call: model requests tool
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }]));
    // Second call: model requests another tool
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c2".into(),
        name: "add".into(),
        arguments: json!({"a": 3, "b": 4}),
    }]));
    // Third call: would continue if not stopped
    mock.queue_response(sample_response("Final"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Chain calls")],
        ..Default::default()
    };

    // Stop after 1 tool call is executed
    let config = ToolLoopConfig {
        stop_when: Some(Arc::new(|ctx| {
            if ctx.tool_calls_executed >= 1 {
                StopDecision::StopWithReason("Tool call limit".into())
            } else {
                StopDecision::Continue
            }
        })),
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    // Should have run 2 iterations: first executes tool, second checks stop condition
    assert_eq!(result.iterations, 2);
}

#[tokio::test]
async fn test_tool_loop_stop_context_has_last_results() {
    use std::sync::Mutex;

    let mock = mock_for("test", "test-model");

    // First call: tool call
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "call_1".into(),
        name: "add".into(),
        arguments: json!({"a": 2, "b": 3}),
    }]));
    // Second call: another response
    mock.queue_response(sample_response("Done"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let captured = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Test")],
        ..Default::default()
    };

    let config = ToolLoopConfig {
        stop_when: Some(Arc::new(move |ctx| {
            // Capture tool_calls_executed and last_tool_results content
            captured_clone.lock().unwrap().push((
                ctx.iteration,
                ctx.tool_calls_executed,
                ctx.last_tool_results
                    .iter()
                    .map(|r| r.content.clone())
                    .collect::<Vec<_>>(),
            ));
            StopDecision::Continue
        })),
        ..Default::default()
    };

    let _result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    let checks = captured.lock().unwrap();
    // First check: iteration 1, 0 tool calls executed, no results yet
    assert_eq!(checks[0].0, 1);
    assert_eq!(checks[0].1, 0);
    assert!(checks[0].2.is_empty());

    // Second check: iteration 2, 1 tool call executed, has result "5"
    assert_eq!(checks[1].0, 2);
    assert_eq!(checks[1].1, 1);
    assert_eq!(checks[1].2, vec!["5".to_string()]);
}

#[tokio::test]
async fn test_tool_loop_stop_on_specific_tool() {
    let mock = mock_for("test", "test-model");

    // First call: model calls "add"
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }]));
    // Second call: model calls "final_answer" (simulated)
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c2".into(),
        name: "final_answer".into(),
        arguments: json!({"answer": "The result is 3"}),
    }]));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);
    // Register a "final_answer" tool
    registry.register(tool_fn(
        ToolDefinition {
            name: "final_answer".into(),
            description: "Provide final answer".into(),
            parameters: JsonSchema::new(json!({"type": "object"})),
            retry: None,
        },
        |_| async { Ok(ToolOutput::new("acknowledged")) },
    ));

    let params = ChatParams {
        messages: vec![ChatMessage::user("Calculate")],
        ..Default::default()
    };

    // Stop when response contains a call to "final_answer" tool
    let config = ToolLoopConfig {
        stop_when: Some(Arc::new(|ctx| {
            let has_final = ctx
                .response
                .tool_calls()
                .iter()
                .any(|c| c.name == "final_answer");
            if has_final {
                StopDecision::StopWithReason("Final answer provided".into())
            } else {
                StopDecision::Continue
            }
        })),
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    // Should stop on iteration 2 when final_answer is called
    assert_eq!(result.iterations, 2);
    assert!(
        result
            .response
            .tool_calls()
            .iter()
            .any(|c| c.name == "final_answer")
    );
}

#[tokio::test]
async fn test_tool_loop_stream_stop_early() {
    use crate::stream::StreamEvent;
    use futures::StreamExt;

    let mock = Arc::new(mock_for("test", "test-model"));

    // First stream: tool call
    mock.queue_stream(vec![
        StreamEvent::ToolCallComplete {
            index: 0,
            call: ToolCall {
                id: "call_1".into(),
                name: "add".into(),
                arguments: json!({"a": 2, "b": 3}),
            },
        },
        StreamEvent::Done {
            stop_reason: StopReason::ToolUse,
        },
    ]);

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);
    let registry = Arc::new(registry);

    let params = ChatParams {
        messages: vec![ChatMessage::user("What is 2+3?")],
        ..Default::default()
    };

    // Stop immediately
    let config = ToolLoopConfig {
        stop_when: Some(Arc::new(|_| StopDecision::Stop)),
        ..Default::default()
    };

    let stream = tool_loop_stream(mock, registry, params, config, Arc::new(()));
    let events: Vec<_> = stream.collect::<Vec<_>>().await;

    // Should have ToolCallComplete and Done, then stop
    assert!(events.iter().all(Result::is_ok));
    // Should have Done event
    assert!(
        events
            .iter()
            .any(|r| matches!(r, Ok(StreamEvent::Done { .. })))
    );
    // Should NOT have more iterations (no text delta from second iteration)
}

#[tokio::test]
async fn test_tool_loop_stream_stop_after_tool_execution() {
    use crate::stream::StreamEvent;
    use futures::StreamExt;

    let mock = Arc::new(mock_for("test", "test-model"));

    // First stream: tool call
    mock.queue_stream(vec![
        StreamEvent::ToolCallComplete {
            index: 0,
            call: ToolCall {
                id: "call_1".into(),
                name: "add".into(),
                arguments: json!({"a": 2, "b": 3}),
            },
        },
        StreamEvent::Done {
            stop_reason: StopReason::ToolUse,
        },
    ]);

    // Second stream: LLM responds, stop condition triggers on Done
    // Note: In streaming, text deltas are yielded before Done arrives,
    // so the text will appear even when we stop on Done.
    mock.queue_stream(vec![
        StreamEvent::TextDelta("Final response".into()),
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
        },
    ]);

    // Third stream: should never be reached
    mock.queue_stream(vec![
        StreamEvent::TextDelta("This should never appear".into()),
        StreamEvent::Done {
            stop_reason: StopReason::EndTurn,
        },
    ]);

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);
    let registry = Arc::new(registry);

    let params = ChatParams {
        messages: vec![ChatMessage::user("What is 2+3?")],
        ..Default::default()
    };

    // Stop after 1 tool call is executed
    let config = ToolLoopConfig {
        stop_when: Some(Arc::new(|ctx| {
            if ctx.tool_calls_executed >= 1 {
                StopDecision::Stop
            } else {
                StopDecision::Continue
            }
        })),
        ..Default::default()
    };

    let stream = tool_loop_stream(mock, registry, params, config, Arc::new(()));
    let events: Vec<_> = stream.filter_map(|r| async { r.ok() }).collect().await;

    // Should have 2 Done events (first iteration + second iteration that stops)
    let done_count = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::Done { .. }))
        .count();
    assert_eq!(done_count, 2);

    // The second iteration's text DOES appear (streamed before Done triggers stop)
    let has_final_response = events
        .iter()
        .any(|e| matches!(e, StreamEvent::TextDelta(t) if t.contains("Final response")));
    assert!(has_final_response);

    // But the third iteration should NOT be reached
    let has_third_iteration = events
        .iter()
        .any(|e| matches!(e, StreamEvent::TextDelta(t) if t.contains("never appear")));
    assert!(!has_third_iteration);
}

#[tokio::test]
async fn test_tool_loop_stop_continues_when_condition_not_met() {
    let mock = mock_for("test", "test-model");

    // First call: tool call
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }]));
    // Second call: text response
    mock.queue_response(sample_response("The answer is 3"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Add 1 and 2")],
        ..Default::default()
    };

    // Never stop (always Continue)
    let config = ToolLoopConfig {
        stop_when: Some(Arc::new(|_| StopDecision::Continue)),
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    // Should complete normally with 2 iterations
    assert_eq!(result.iterations, 2);
    assert_eq!(result.response.text(), Some("The answer is 3"));
}

// ── Loop detection tests ─────────────────────────────────────────

#[test]
fn test_loop_detection_config_default() {
    let config = LoopDetectionConfig::default();
    assert_eq!(config.threshold, 3);
    assert_eq!(config.action, LoopAction::Warn);
}

#[test]
fn test_loop_action_equality() {
    assert_eq!(LoopAction::Warn, LoopAction::Warn);
    assert_eq!(LoopAction::Stop, LoopAction::Stop);
    assert_eq!(LoopAction::InjectWarning, LoopAction::InjectWarning);
    assert_ne!(LoopAction::Warn, LoopAction::Stop);
}

#[test]
fn test_loop_detection_state_no_loop() {
    use super::loop_detection::LoopDetectionState;

    let mut state = LoopDetectionState::default();
    let calls = vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }];

    // First call - no loop
    assert!(state.update(&calls, 3).is_none());

    // Different call - no loop
    let calls2 = vec![ToolCall {
        id: "c2".into(),
        name: "add".into(),
        arguments: json!({"a": 3, "b": 4}), // different args
    }];
    assert!(state.update(&calls2, 3).is_none());
}

#[test]
fn test_loop_detection_state_detects_loop() {
    use super::loop_detection::LoopDetectionState;

    let mut state = LoopDetectionState::default();
    let calls = vec![ToolCall {
        id: "c1".into(),
        name: "search".into(),
        arguments: json!({"query": "test"}),
    }];

    // First call - no loop
    assert!(state.update(&calls, 3).is_none());
    // Second identical call - no loop yet (count = 2)
    assert!(state.update(&calls, 3).is_none());
    // Third identical call - loop detected!
    let result = state.update(&calls, 3);
    assert!(result.is_some());
    let (name, count) = result.unwrap();
    assert_eq!(name, "search");
    assert_eq!(count, 3);
}

#[test]
fn test_loop_detection_state_reset() {
    use super::loop_detection::LoopDetectionState;

    let mut state = LoopDetectionState::default();
    let calls = vec![ToolCall {
        id: "c1".into(),
        name: "search".into(),
        arguments: json!({"query": "test"}),
    }];

    state.update(&calls, 3);
    state.update(&calls, 3);
    state.reset();

    // After reset, should start fresh
    assert!(state.update(&calls, 3).is_none());
}

#[test]
fn test_tool_loop_config_debug_with_loop_detection() {
    let config = ToolLoopConfig {
        loop_detection: Some(LoopDetectionConfig::default()),
        ..Default::default()
    };
    let debug = format!("{config:?}");
    assert!(debug.contains("loop_detection: Some"));
}

#[tokio::test]
async fn test_tool_loop_detects_loop_warn() {
    use std::sync::Mutex;

    let mock = mock_for("test", "test-model");

    // Queue identical tool calls 3 times
    for _ in 0..3 {
        mock.queue_response(sample_tool_response(vec![ToolCall {
            id: "c".into(),
            name: "search".into(),
            arguments: json!({"query": "foo"}),
        }]));
    }
    // Fourth call: text response
    mock.queue_response(sample_response("Done"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(tool_fn(
        ToolDefinition {
            name: "search".into(),
            description: "Search".into(),
            parameters: JsonSchema::new(json!({"type": "object"})),
            retry: None,
        },
        |_| async { Ok(ToolOutput::new("result")) },
    ));

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = Arc::clone(&events);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Search")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        loop_detection: Some(LoopDetectionConfig {
            threshold: 3,
            action: LoopAction::Warn,
        }),
        on_event: Some(Arc::new(move |event| {
            events_clone.lock().unwrap().push(event);
        })),
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    // Should complete (Warn doesn't stop)
    assert_eq!(result.iterations, 4);

    // Should have emitted LoopDetected event
    let captured = events.lock().unwrap();
    let loop_events: Vec<_> = captured
        .iter()
        .filter(|e| matches!(e, ToolLoopEvent::LoopDetected { .. }))
        .collect();
    assert_eq!(loop_events.len(), 1);
    if let ToolLoopEvent::LoopDetected {
        tool_name,
        consecutive_count,
        action,
    } = &loop_events[0]
    {
        assert_eq!(tool_name, "search");
        assert_eq!(*consecutive_count, 3);
        assert_eq!(*action, LoopAction::Warn);
    }
}

#[tokio::test]
async fn test_tool_loop_detects_loop_stop() {
    let mock = mock_for("test", "test-model");

    // Queue identical tool calls
    for _ in 0..5 {
        mock.queue_response(sample_tool_response(vec![ToolCall {
            id: "c".into(),
            name: "search".into(),
            arguments: json!({"query": "foo"}),
        }]));
    }

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(tool_fn(
        ToolDefinition {
            name: "search".into(),
            description: "Search".into(),
            parameters: JsonSchema::new(json!({"type": "object"})),
            retry: None,
        },
        |_| async { Ok(ToolOutput::new("result")) },
    ));

    let params = ChatParams {
        messages: vec![ChatMessage::user("Search")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        loop_detection: Some(LoopDetectionConfig {
            threshold: 3,
            action: LoopAction::Stop,
        }),
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    // Should return with LoopDetected termination reason
    assert!(matches!(
        result.termination_reason,
        TerminationReason::LoopDetected { ref tool_name, count }
            if tool_name == "search" && count == 3
    ));
}

#[tokio::test]
async fn test_tool_loop_detects_loop_inject_warning() {
    let mock = mock_for("test", "test-model");

    // First 3 calls: identical
    for _ in 0..3 {
        mock.queue_response(sample_tool_response(vec![ToolCall {
            id: "c".into(),
            name: "search".into(),
            arguments: json!({"query": "foo"}),
        }]));
    }
    // After warning, LLM changes behavior
    mock.queue_response(sample_response("I'll try something different"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(tool_fn(
        ToolDefinition {
            name: "search".into(),
            description: "Search".into(),
            parameters: JsonSchema::new(json!({"type": "object"})),
            retry: None,
        },
        |_| async { Ok(ToolOutput::new("result")) },
    ));

    let params = ChatParams {
        messages: vec![ChatMessage::user("Search")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        loop_detection: Some(LoopDetectionConfig {
            threshold: 3,
            action: LoopAction::InjectWarning,
        }),
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    // Should complete
    assert_eq!(result.iterations, 4);

    // Check that warning was injected into messages
    let recorded = mock.recorded_calls();
    // The 4th call should have the warning message
    let last_call = &recorded[3];
    let has_warning = last_call.messages.iter().any(|m| {
        if m.role == ChatRole::System {
            // Check if any content block contains the warning text
            m.content.iter().any(|b| {
                if let ContentBlock::Text(t) = b {
                    t.contains("identical arguments")
                } else {
                    false
                }
            })
        } else {
            false
        }
    });
    assert!(has_warning, "Warning message should be in conversation");
}

#[tokio::test]
async fn test_tool_loop_no_false_positive_different_args() {
    let mock = mock_for("test", "test-model");

    // Queue tool calls with DIFFERENT arguments each time
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "search".into(),
        arguments: json!({"query": "foo"}),
    }]));
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c2".into(),
        name: "search".into(),
        arguments: json!({"query": "bar"}), // different
    }]));
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c3".into(),
        name: "search".into(),
        arguments: json!({"query": "baz"}), // different
    }]));
    mock.queue_response(sample_response("Done"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(tool_fn(
        ToolDefinition {
            name: "search".into(),
            description: "Search".into(),
            parameters: JsonSchema::new(json!({"type": "object"})),
            retry: None,
        },
        |_| async { Ok(ToolOutput::new("result")) },
    ));

    let params = ChatParams {
        messages: vec![ChatMessage::user("Search")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        loop_detection: Some(LoopDetectionConfig {
            threshold: 3,
            action: LoopAction::Stop,
        }),
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &()).await;

    // Should complete without error (no false positive)
    assert!(result.is_ok());
    assert_eq!(result.unwrap().iterations, 4);
}

#[tokio::test]
async fn test_tool_loop_stream_detects_loop_stop() {
    use crate::error::LlmError;
    use crate::stream::StreamEvent;
    use futures::StreamExt;

    let mock = Arc::new(mock_for("test", "test-model"));

    // Queue identical streams
    for _ in 0..5 {
        mock.queue_stream(vec![
            StreamEvent::ToolCallComplete {
                index: 0,
                call: ToolCall {
                    id: "c".into(),
                    name: "search".into(),
                    arguments: json!({"query": "foo"}),
                },
            },
            StreamEvent::Done {
                stop_reason: StopReason::ToolUse,
            },
        ]);
    }

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(tool_fn(
        ToolDefinition {
            name: "search".into(),
            description: "Search".into(),
            parameters: JsonSchema::new(json!({"type": "object"})),
            retry: None,
        },
        |_| async { Ok(ToolOutput::new("result")) },
    ));
    let registry = Arc::new(registry);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Search")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        loop_detection: Some(LoopDetectionConfig {
            threshold: 3,
            action: LoopAction::Stop,
        }),
        ..Default::default()
    };

    let stream = tool_loop_stream(mock, registry, params, config, Arc::new(()));
    let events: Vec<_> = stream.collect::<Vec<_>>().await;

    // Should have an error
    assert!(events.iter().any(Result::is_err));
    let err = events.iter().find(|r| r.is_err()).unwrap();
    assert!(matches!(err, Err(LlmError::ToolExecution { .. })));
}

#[test]
fn test_compute_tool_calls_signature_single() {
    use super::loop_detection::compute_tool_calls_signature;

    let calls = vec![ToolCall {
        id: "c1".into(),
        name: "search".into(),
        arguments: json!({"query": "test"}),
    }];
    let sig = compute_tool_calls_signature(&calls);
    assert_eq!(sig.0, "search");
    assert!(sig.1.contains("query"));
}

#[test]
fn test_compute_tool_calls_signature_multiple() {
    use super::loop_detection::compute_tool_calls_signature;

    let calls = vec![
        ToolCall {
            id: "c1".into(),
            name: "search".into(),
            arguments: json!({"query": "a"}),
        },
        ToolCall {
            id: "c2".into(),
            name: "read".into(),
            arguments: json!({"file": "b"}),
        },
    ];
    let sig = compute_tool_calls_signature(&calls);
    assert_eq!(sig.0, "search+read");
    assert!(sig.1.contains('|'));
}

#[test]
fn test_compute_tool_calls_signature_empty() {
    use super::loop_detection::compute_tool_calls_signature;

    let calls: Vec<ToolCall> = vec![];
    let sig = compute_tool_calls_signature(&calls);
    assert!(sig.0.is_empty());
    assert!(sig.1.is_empty());
}

// ── TerminationReason tests ────────────────────────────────────────

#[test]
fn test_termination_reason_complete() {
    let reason = TerminationReason::Complete;
    assert_eq!(reason, TerminationReason::Complete);
}

#[test]
fn test_termination_reason_stop_condition_without_reason() {
    let reason = TerminationReason::StopCondition { reason: None };
    assert!(matches!(
        reason,
        TerminationReason::StopCondition { reason: None }
    ));
}

#[test]
fn test_termination_reason_stop_condition_with_reason() {
    let reason = TerminationReason::StopCondition {
        reason: Some("budget exceeded".into()),
    };
    assert!(matches!(
        reason,
        TerminationReason::StopCondition { reason: Some(r) } if r == "budget exceeded"
    ));
}

#[test]
fn test_termination_reason_max_iterations() {
    let reason = TerminationReason::MaxIterations { limit: 10 };
    assert!(matches!(
        reason,
        TerminationReason::MaxIterations { limit: 10 }
    ));
}

#[test]
fn test_termination_reason_loop_detected() {
    let reason = TerminationReason::LoopDetected {
        tool_name: "search".into(),
        count: 5,
    };
    assert!(matches!(
        reason,
        TerminationReason::LoopDetected { tool_name, count } if tool_name == "search" && count == 5
    ));
}

#[test]
fn test_termination_reason_clone() {
    let reason = TerminationReason::StopCondition {
        reason: Some("test".into()),
    };
    let cloned = reason.clone();
    assert_eq!(reason, cloned);
}

#[test]
fn test_termination_reason_debug() {
    let reason = TerminationReason::Complete;
    let debug = format!("{reason:?}");
    assert!(debug.contains("Complete"));
}

#[tokio::test]
async fn test_tool_loop_returns_termination_reason_complete() {
    let mock = mock_for("test", "test-model");
    mock.queue_response(sample_response("Done!"));

    let registry: ToolRegistry<()> = ToolRegistry::new();
    let params = ChatParams {
        messages: vec![ChatMessage::user("Hello")],
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, ToolLoopConfig::default(), &())
        .await
        .unwrap();

    assert_eq!(result.termination_reason, TerminationReason::Complete);
}

#[tokio::test]
async fn test_tool_loop_returns_termination_reason_stop_condition() {
    let mock = mock_for("test", "test-model");
    mock.queue_response(sample_response("First response"));

    let registry: ToolRegistry<()> = ToolRegistry::new();
    let params = ChatParams {
        messages: vec![ChatMessage::user("Hello")],
        ..Default::default()
    };

    let config = ToolLoopConfig {
        stop_when: Some(Arc::new(|_ctx| {
            StopDecision::StopWithReason("manual stop".into())
        })),
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    assert!(matches!(
        result.termination_reason,
        TerminationReason::StopCondition { reason: Some(r) } if r == "manual stop"
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeout Tests
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_tool_loop_timeout_returns_immediately() {
    let mock = mock_for("test", "test-model");

    // Queue responses that would normally run for several iterations
    for _ in 0..5 {
        mock.queue_response(sample_tool_response(vec![ToolCall {
            id: "c".into(),
            name: "add".into(),
            arguments: json!({"a": 1, "b": 2}),
        }]));
    }

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Timeout test")],
        ..Default::default()
    };

    // Set a zero timeout - should trigger immediately after first iteration completes
    let config = ToolLoopConfig {
        timeout: Some(Duration::ZERO),
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    assert!(matches!(
        result.termination_reason,
        TerminationReason::Timeout { limit } if limit == Duration::ZERO
    ));
}

#[tokio::test]
async fn test_tool_loop_no_timeout_completes_normally() {
    let mock = mock_for("test", "test-model");

    mock.queue_response(sample_response("Done quickly"));

    let registry: ToolRegistry<()> = ToolRegistry::new();

    let params = ChatParams {
        messages: vec![ChatMessage::user("No timeout")],
        ..Default::default()
    };

    // No timeout configured (None)
    let config = ToolLoopConfig {
        timeout: None,
        ..Default::default()
    };

    let result = tool_loop(&mock, &registry, params, config, &())
        .await
        .unwrap();

    assert!(matches!(
        result.termination_reason,
        TerminationReason::Complete
    ));
}

#[tokio::test]
async fn test_tool_loop_stream_timeout() {
    use futures::StreamExt;

    let mock = Arc::new(mock_for("test", "test-model"));

    // Queue multiple tool responses to keep the loop running
    for _ in 0..10 {
        mock.queue_response(sample_tool_response(vec![ToolCall {
            id: "c".into(),
            name: "add".into(),
            arguments: json!({"a": 1, "b": 2}),
        }]));
    }

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);
    let registry = Arc::new(registry);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Stream timeout")],
        ..Default::default()
    };

    // Zero timeout should trigger immediately
    let config = ToolLoopConfig {
        timeout: Some(Duration::ZERO),
        ..Default::default()
    };

    let ctx = Arc::new(());
    let mut stream = tool_loop_stream(mock, registry, params, config, ctx);

    // Collect events until we get an error (timeout)
    let mut got_timeout_error = false;
    while let Some(result) = stream.next().await {
        // Should be a ToolExecution error mentioning timeout
        if let Err(crate::error::LlmError::ToolExecution { source, .. }) = result {
            if source.to_string().contains("timeout") {
                got_timeout_error = true;
                break;
            }
        }
    }

    assert!(got_timeout_error, "Expected timeout error in stream");
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-Tool Retry Tests
// ─────────────────────────────────────────────────────────────────────────────

/// A tool handler that fails a configurable number of times before succeeding.
struct FlakeyTool {
    fail_count: std::sync::atomic::AtomicU32,
    max_failures: u32,
}

impl FlakeyTool {
    fn new(max_failures: u32) -> Self {
        Self {
            fail_count: std::sync::atomic::AtomicU32::new(0),
            max_failures,
        }
    }
}

impl ToolHandler<()> for FlakeyTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "flakey".into(),
            description: "A tool that fails sometimes".into(),
            parameters: JsonSchema::new(json!({"type": "object"})),
            retry: Some(crate::provider::ToolRetryConfig {
                max_retries: 3,
                initial_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(10),
                backoff_multiplier: 2.0,
                jitter: 0.0, // No jitter for predictable tests
                retry_if: None,
            }),
        }
    }

    fn execute<'a>(
        &'a self,
        _input: Value,
        _ctx: &'a (),
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let count = self
                .fail_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if count < self.max_failures {
                Err(ToolError::new("transient failure"))
            } else {
                Ok(ToolOutput::new("success after retries"))
            }
        })
    }
}

#[tokio::test]
async fn test_tool_retry_succeeds_after_failures() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(FlakeyTool::new(2)); // Fail twice, succeed on third

    let call = ToolCall {
        id: "c1".into(),
        name: "flakey".into(),
        arguments: json!({}),
    };

    let result = registry.execute(&call, &()).await;

    // Should eventually succeed
    assert!(!result.is_error);
    assert_eq!(result.content, "success after retries");
}

#[tokio::test]
async fn test_tool_retry_exhausted() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(FlakeyTool::new(10)); // More failures than max_retries allows

    let call = ToolCall {
        id: "c1".into(),
        name: "flakey".into(),
        arguments: json!({}),
    };

    let result = registry.execute(&call, &()).await;

    // Should fail after exhausting retries
    assert!(result.is_error);
    assert!(result.content.contains("transient failure"));
}

/// A tool that uses `retry_if` predicate to only retry specific errors.
struct SelectiveRetryTool {
    call_count: std::sync::atomic::AtomicU32,
}

impl SelectiveRetryTool {
    fn new() -> Self {
        Self {
            call_count: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

impl ToolHandler<()> for SelectiveRetryTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "selective".into(),
            description: "A tool with selective retry".into(),
            parameters: JsonSchema::new(json!({"type": "object"})),
            retry: Some(crate::provider::ToolRetryConfig {
                max_retries: 3,
                initial_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(10),
                backoff_multiplier: 2.0,
                jitter: 0.0,
                // Only retry errors containing "TRANSIENT"
                retry_if: Some(Arc::new(|msg: &str| msg.contains("TRANSIENT"))),
            }),
        }
    }

    fn execute<'a>(
        &'a self,
        _input: Value,
        _ctx: &'a (),
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let count = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if count == 0 {
                // Permanent error - should not be retried
                Err(ToolError::new("permanent failure"))
            } else {
                Ok(ToolOutput::new("should not reach"))
            }
        })
    }
}

#[tokio::test]
async fn test_tool_retry_predicate_prevents_retry() {
    let tool = SelectiveRetryTool::new();
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(tool);

    let call = ToolCall {
        id: "c1".into(),
        name: "selective".into(),
        arguments: json!({}),
    };

    let result = registry.execute(&call, &()).await;

    // Should fail immediately without retry because error doesn't contain "TRANSIENT"
    assert!(
        result.is_error,
        "Expected is_error=true but got false. Content: {}",
        result.content
    );
    assert!(result.content.contains("permanent failure"));
}

#[test]
fn test_tool_retry_config_default() {
    let config = crate::provider::ToolRetryConfig::default();
    assert_eq!(config.max_retries, 3);
    assert_eq!(config.initial_backoff, Duration::from_millis(100));
    assert_eq!(config.max_backoff, Duration::from_secs(5));
    assert!((config.backoff_multiplier - 2.0).abs() < f64::EPSILON);
    assert!((config.jitter - 0.5).abs() < f64::EPSILON);
    assert!(config.retry_if.is_none());
}

#[test]
fn test_tool_retry_config_partial_eq() {
    let config1 = crate::provider::ToolRetryConfig::default();
    let config2 = crate::provider::ToolRetryConfig::default();
    assert_eq!(config1, config2);

    let config3 = crate::provider::ToolRetryConfig {
        max_retries: 5,
        ..Default::default()
    };
    assert_ne!(config1, config3);
}

#[test]
fn test_tool_retry_config_debug() {
    let config = crate::provider::ToolRetryConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("ToolRetryConfig"));
    assert!(debug.contains("max_retries"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Resumable Tool Loop Tests
// ─────────────────────────────────────────────────────────────────────────────

use super::loop_resumable::{LoopCommand, LoopEvent, ToolLoopHandle};

#[tokio::test]
async fn test_resumable_no_tools_completes() {
    let mock = mock_for("test", "test-model");
    mock.queue_response(sample_response("Hello!"));

    let registry: ToolRegistry<()> = ToolRegistry::new();
    let params = ChatParams {
        messages: vec![ChatMessage::user("Hi")],
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, ToolLoopConfig::default(), &());
    let event = handle.next_event().await;

    assert!(
        matches!(&event, LoopEvent::Completed { termination_reason, .. }
            if *termination_reason == TerminationReason::Complete)
    );
    if let LoopEvent::Completed { response, .. } = &event {
        assert_eq!(response.text(), Some("Hello!"));
    }
    assert!(handle.is_finished());
}

#[tokio::test]
async fn test_resumable_one_tool_iteration() {
    let mock = mock_for("test", "test-model");

    // First: LLM requests tool
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "call_1".into(),
        name: "add".into(),
        arguments: json!({"a": 2, "b": 3}),
    }]));
    // Second: LLM returns text
    mock.queue_response(sample_response("The answer is 5"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("What is 2 + 3?")],
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, ToolLoopConfig::default(), &());

    // First event: tools executed
    let event = handle.next_event().await;
    assert!(matches!(
        &event,
        LoopEvent::ToolsExecuted { iteration: 1, .. }
    ));
    if let LoopEvent::ToolsExecuted {
        results,
        tool_calls,
        ..
    } = &event
    {
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "add");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "5");
    }

    // Resume
    handle.resume(LoopCommand::Continue);

    // Second event: completion
    let event = handle.next_event().await;
    assert!(
        matches!(&event, LoopEvent::Completed { termination_reason, .. }
            if *termination_reason == TerminationReason::Complete)
    );
    if let LoopEvent::Completed {
        response,
        iterations,
        ..
    } = &event
    {
        assert_eq!(*iterations, 2);
        assert_eq!(response.text(), Some("The answer is 5"));
    }
}

#[tokio::test]
async fn test_resumable_inject_messages() {
    let mock = mock_for("test", "test-model");

    // First: tool call
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }]));
    // Second: text response
    mock.queue_response(sample_response("Done with injection"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Add numbers")],
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, ToolLoopConfig::default(), &());

    let event = handle.next_event().await;
    assert!(matches!(&event, LoopEvent::ToolsExecuted { .. }));

    // Inject a message before next iteration
    handle.resume(LoopCommand::InjectMessages(vec![ChatMessage::system(
        "Additional context from worker",
    )]));

    let event = handle.next_event().await;
    assert!(matches!(&event, LoopEvent::Completed { .. }));

    // Verify the injected message was in the LLM call
    let recorded = mock.recorded_calls();
    let last_call = &recorded[1];
    let has_injection = last_call.messages.iter().any(|m| {
        m.content.iter().any(|b| {
            if let ContentBlock::Text(t) = b {
                t.contains("Additional context from worker")
            } else {
                false
            }
        })
    });
    assert!(has_injection, "Injected message should be in conversation");
}

#[tokio::test]
async fn test_resumable_stop_command() {
    let mock = mock_for("test", "test-model");

    // First: tool call
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }]));
    // This should never be reached
    mock.queue_response(sample_response("Should not appear"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Stop early")],
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, ToolLoopConfig::default(), &());

    let event = handle.next_event().await;
    assert!(matches!(&event, LoopEvent::ToolsExecuted { .. }));

    // Stop the loop
    handle.resume(LoopCommand::Stop(Some("task_spawn detected".into())));

    let event = handle.next_event().await;
    assert!(
        matches!(&event, LoopEvent::Completed { termination_reason, .. }
        if matches!(termination_reason,
            TerminationReason::StopCondition { reason: Some(r) } if r == "task_spawn detected"
        ))
    );

    // Only 1 LLM call was made (the second was never reached)
    assert_eq!(mock.recorded_calls().len(), 1);
}

#[tokio::test]
async fn test_resumable_usage_accumulation() {
    let mock = mock_for("test", "test-model");

    // Two tool iterations + final text
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }]));
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c2".into(),
        name: "add".into(),
        arguments: json!({"a": 3, "b": 4}),
    }]));
    mock.queue_response(sample_response("All done"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Chain")],
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, ToolLoopConfig::default(), &());

    // First iteration
    let event = handle.next_event().await;
    if let LoopEvent::ToolsExecuted { total_usage, .. } = &event {
        assert_eq!(total_usage.input_tokens, 100); // 1 LLM call
    }
    handle.resume(LoopCommand::Continue);

    // Second iteration
    let event = handle.next_event().await;
    if let LoopEvent::ToolsExecuted { total_usage, .. } = &event {
        assert_eq!(total_usage.input_tokens, 200); // 2 LLM calls
    }
    handle.resume(LoopCommand::Continue);

    // Completion
    let event = handle.next_event().await;
    if let LoopEvent::Completed { total_usage, .. } = &event {
        // 3 LLM calls * 100 input tokens each
        assert_eq!(total_usage.input_tokens, 300);
        assert_eq!(total_usage.output_tokens, 150);
    }

    let result = handle.into_result();
    assert_eq!(result.total_usage.input_tokens, 300);
}

#[tokio::test]
async fn test_resumable_max_iterations() {
    let mock = mock_for("test", "test-model");

    // Keep returning tool calls
    for _ in 0..5 {
        mock.queue_response(sample_tool_response(vec![ToolCall {
            id: "c".into(),
            name: "add".into(),
            arguments: json!({"a": 1, "b": 2}),
        }]));
    }

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Loop")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        max_iterations: 2,
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, config, &());

    // First iteration: tools executed
    let event = handle.next_event().await;
    assert!(matches!(
        &event,
        LoopEvent::ToolsExecuted { iteration: 1, .. }
    ));
    handle.resume(LoopCommand::Continue);

    // Second iteration: tools executed
    let event = handle.next_event().await;
    assert!(matches!(
        &event,
        LoopEvent::ToolsExecuted { iteration: 2, .. }
    ));
    handle.resume(LoopCommand::Continue);

    // Third: max iterations exceeded
    let event = handle.next_event().await;
    assert!(
        matches!(&event, LoopEvent::Completed { termination_reason, .. }
            if matches!(termination_reason, TerminationReason::MaxIterations { limit: 2 }))
    );
}

#[tokio::test]
async fn test_resumable_depth_exceeded() {
    #[derive(Clone)]
    struct DepthCtx(u32);
    impl super::LoopDepth for DepthCtx {
        fn loop_depth(&self) -> u32 {
            self.0
        }
        fn with_depth(&self, depth: u32) -> Self {
            DepthCtx(depth)
        }
    }

    let mock = mock_for("test", "test-model");
    let params = ChatParams {
        messages: vec![ChatMessage::user("Hi")],
        ..Default::default()
    };

    // Create config with max_depth=1, but context depth is already 1
    let config = ToolLoopConfig {
        max_depth: Some(1),
        ..Default::default()
    };

    let registry_typed: ToolRegistry<DepthCtx> = ToolRegistry::new();
    let ctx = DepthCtx(1); // Already at depth 1, max is 1
    let mut handle = ToolLoopHandle::new(&mock, &registry_typed, params, config, &ctx);

    let event = handle.next_event().await;
    assert!(matches!(
        &event,
        LoopEvent::Error {
            error: crate::LlmError::MaxDepthExceeded {
                current: 1,
                limit: 1
            },
            ..
        }
    ));
    assert!(handle.is_finished());
}

#[tokio::test]
async fn test_resumable_messages_access() {
    let mock = mock_for("test", "test-model");

    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }]));
    mock.queue_response(sample_response("Done"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Go")],
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, ToolLoopConfig::default(), &());

    // Before any iteration: just the user message
    assert_eq!(handle.messages().len(), 1);

    let event = handle.next_event().await;
    assert!(matches!(&event, LoopEvent::ToolsExecuted { .. }));

    // After tool execution: user + assistant (with tool call) + tool result
    assert_eq!(handle.messages().len(), 3);

    handle.resume(LoopCommand::Continue);
    let _ = handle.next_event().await;

    // After completion: messages unchanged (no new tool calls)
    // User + assistant(tool_call) + tool_result + assistant(text) = depends on impl
    assert!(handle.messages().len() >= 3);
}

#[tokio::test]
async fn test_resumable_into_result() {
    let mock = mock_for("test", "test-model");
    mock.queue_response(sample_response("Quick"));

    let registry: ToolRegistry<()> = ToolRegistry::new();
    let params = ChatParams {
        messages: vec![ChatMessage::user("Hi")],
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, ToolLoopConfig::default(), &());
    let _ = handle.next_event().await;

    let result = handle.into_result();
    assert_eq!(result.termination_reason, TerminationReason::Complete);
    assert_eq!(result.iterations, 1);
    assert_eq!(result.total_usage.input_tokens, 100);
}

#[tokio::test]
async fn test_resumable_repeated_next_after_completion() {
    let mock = mock_for("test", "test-model");
    mock.queue_response(sample_response("Done"));

    let registry: ToolRegistry<()> = ToolRegistry::new();
    let params = ChatParams {
        messages: vec![ChatMessage::user("Hi")],
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, ToolLoopConfig::default(), &());

    // First call: completion
    let event = handle.next_event().await;
    assert!(matches!(&event, LoopEvent::Completed { .. }));

    // Second call: same terminal event
    let event = handle.next_event().await;
    assert!(matches!(&event, LoopEvent::Completed { .. }));
}

#[tokio::test]
async fn test_resumable_stop_condition_callback() {
    let mock = mock_for("test", "test-model");

    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 2, "b": 3}),
    }]));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Test stop")],
        ..Default::default()
    };

    // Stop immediately via config callback
    let config = ToolLoopConfig {
        stop_when: Some(Arc::new(|_| StopDecision::Stop)),
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, config, &());
    let event = handle.next_event().await;

    // Should stop without executing tools
    assert!(
        matches!(&event, LoopEvent::Completed { termination_reason, .. }
            if matches!(termination_reason, TerminationReason::StopCondition { reason: None }))
    );
}

#[tokio::test]
async fn test_resumable_convenience_function() {
    use super::loop_resumable::tool_loop_resumable;

    let mock = mock_for("test", "test-model");
    mock.queue_response(sample_response("Via convenience"));

    let registry: ToolRegistry<()> = ToolRegistry::new();
    let params = ChatParams {
        messages: vec![ChatMessage::user("Hi")],
        ..Default::default()
    };

    let mut handle = tool_loop_resumable(&mock, &registry, params, ToolLoopConfig::default(), &());
    let event = handle.next_event().await;

    assert!(matches!(&event, LoopEvent::Completed { .. }));
    if let LoopEvent::Completed { response, .. } = &event {
        assert_eq!(response.text(), Some("Via convenience"));
    }
}

#[tokio::test]
async fn test_resumable_debug_impl() {
    let mock = mock_for("test", "test-model");
    mock.queue_response(sample_response("Debug"));

    let registry: ToolRegistry<()> = ToolRegistry::new();
    let params = ChatParams {
        messages: vec![ChatMessage::user("Hi")],
        ..Default::default()
    };

    let handle = ToolLoopHandle::new(&mock, &registry, params, ToolLoopConfig::default(), &());
    let debug = format!("{handle:?}");
    assert!(debug.contains("ToolLoopHandle"));
    assert!(debug.contains("iterations"));
    assert!(debug.contains("finished"));
}

#[tokio::test]
async fn test_resumable_multi_iteration_with_mixed_commands() {
    let mock = mock_for("test", "test-model");

    // Iteration 1: tool call
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }]));
    // Iteration 2: another tool call (after injection)
    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c2".into(),
        name: "add".into(),
        arguments: json!({"a": 10, "b": 20}),
    }]));
    // Iteration 3: final text
    mock.queue_response(sample_response("All done"));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Mix commands")],
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, ToolLoopConfig::default(), &());

    // Iteration 1: Continue
    let event = handle.next_event().await;
    assert!(matches!(
        &event,
        LoopEvent::ToolsExecuted { iteration: 1, .. }
    ));
    handle.resume(LoopCommand::Continue);

    // Iteration 2: Inject messages
    let event = handle.next_event().await;
    assert!(matches!(
        &event,
        LoopEvent::ToolsExecuted { iteration: 2, .. }
    ));
    handle.resume(LoopCommand::InjectMessages(vec![ChatMessage::user(
        "Worker completed task X",
    )]));

    // Iteration 3: Completion
    let event = handle.next_event().await;
    assert!(matches!(&event, LoopEvent::Completed { iterations: 3, .. }));

    // Verify the injection was in the final LLM call
    let recorded = mock.recorded_calls();
    let last_call = &recorded[2];
    let has_worker_msg = last_call.messages.iter().any(|m| {
        m.content.iter().any(|b| {
            if let ContentBlock::Text(t) = b {
                t.contains("Worker completed task X")
            } else {
                false
            }
        })
    });
    assert!(has_worker_msg);
}

#[tokio::test]
async fn test_resumable_timeout() {
    let mock = mock_for("test", "test-model");

    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c".into(),
        name: "add".into(),
        arguments: json!({"a": 1, "b": 2}),
    }]));

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Timeout")],
        ..Default::default()
    };

    // Zero timeout: should trigger on second iteration
    let config = ToolLoopConfig {
        timeout: Some(Duration::ZERO),
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, config, &());

    // First event might be tools executed or timeout depending on timing.
    // With Duration::ZERO, the timeout check happens at the start of the
    // first iteration, but Instant::now() was set in new() so elapsed > 0.
    let event = handle.next_event().await;
    assert!(
        matches!(&event, LoopEvent::Completed { termination_reason, .. }
            if matches!(termination_reason, TerminationReason::Timeout { .. }))
    );
}

#[tokio::test]
async fn test_resumable_on_event_callback() {
    use std::sync::Mutex;

    let mock = mock_for("test", "test-model");

    mock.queue_response(sample_tool_response(vec![ToolCall {
        id: "c1".into(),
        name: "add".into(),
        arguments: json!({"a": 2, "b": 3}),
    }]));
    mock.queue_response(sample_response("Done"));

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = Arc::clone(&events);

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(AddTool);

    let params = ChatParams {
        messages: vec![ChatMessage::user("Events")],
        ..Default::default()
    };
    let config = ToolLoopConfig {
        on_event: Some(Arc::new(move |event| {
            events_clone.lock().unwrap().push(event);
        })),
        ..Default::default()
    };

    let mut handle = ToolLoopHandle::new(&mock, &registry, params, config, &());

    let _ = handle.next_event().await;
    handle.resume(LoopCommand::Continue);
    let _ = handle.next_event().await;

    let captured = events.lock().unwrap();
    assert!(
        captured
            .iter()
            .any(|e| matches!(e, ToolLoopEvent::IterationStart { .. }))
    );
    assert!(
        captured
            .iter()
            .any(|e| matches!(e, ToolLoopEvent::ToolExecutionStart { .. }))
    );
    assert!(
        captured
            .iter()
            .any(|e| matches!(e, ToolLoopEvent::ToolExecutionEnd { .. }))
    );
    assert!(
        captured
            .iter()
            .any(|e| matches!(e, ToolLoopEvent::LlmResponseReceived { .. }))
    );
}
