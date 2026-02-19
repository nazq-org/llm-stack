//! Provider-agnostic integration tests for tool loop.
//!
//! These tests hit live APIs and are marked `#[ignore]` so they don't
//! run in CI. Run manually with:
//! ```sh
//! ANTHROPIC_API_KEY=sk-ant-... cargo test -p llm-stack --test integration_tool_loop -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! Supported providers (first available wins):
//! - `ANTHROPIC_API_KEY` → Anthropic Claude (Haiku 4.5)
//! - `OPENAI_API_KEY` → `OpenAI` (gpt-4o-mini)
//! - Ollama on `localhost:11434` → Ollama (llama3.2)

use std::sync::Arc;

use futures::StreamExt;
use serde_json::{Value, json};

use llm_stack::DynProvider;
use llm_stack::chat::ChatMessage;
use llm_stack::provider::{ChatParams, JsonSchema, ToolDefinition};
use llm_stack::tool::{
    LoopEvent, TerminationReason, ToolError, ToolLoopConfig, ToolRegistry, tool_fn, tool_loop,
    tool_loop_stream,
};

// ── Provider discovery ───────────────────────────────────────────────

/// Try to construct a provider from available credentials.
/// Returns the first available: Anthropic > `OpenAI` > Ollama.
fn discover_provider() -> Option<(Box<dyn DynProvider>, &'static str)> {
    // Anthropic
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        if !key.is_empty() {
            let p =
                llm_stack_anthropic::AnthropicProvider::new(llm_stack_anthropic::AnthropicConfig {
                    api_key: key,
                    model: "claude-haiku-4-5-20251001".into(),
                    ..Default::default()
                });
            return Some((Box::new(p), "anthropic"));
        }
    }

    // OpenAI
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        if !key.is_empty() {
            let p = llm_stack_openai::OpenAiProvider::new(llm_stack_openai::OpenAiConfig {
                api_key: key,
                model: "gpt-4o-mini".into(),
                ..Default::default()
            });
            return Some((Box::new(p), "openai"));
        }
    }

    // Ollama — sync health check
    if check_ollama() {
        let p = llm_stack_ollama::OllamaProvider::new(llm_stack_ollama::OllamaConfig {
            model: "llama3.2".into(),
            ..Default::default()
        });
        return Some((Box::new(p), "ollama"));
    }

    None
}

fn check_ollama() -> bool {
    // Quick blocking check — integration tests already use tokio runtime
    std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().unwrap(),
        std::time::Duration::from_secs(1),
    )
    .is_ok()
}

macro_rules! skip_without_provider {
    () => {
        match discover_provider() {
            Some((provider, name)) => {
                eprintln!("Using provider: {name}");
                provider
            }
            None => {
                eprintln!("No LLM provider available, skipping integration test");
                return;
            }
        }
    };
}

// ── Tool helpers ─────────────────────────────────────────────────────

fn add_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "add".into(),
        description: "Add two numbers and return the sum".into(),
        parameters: JsonSchema::new(json!({
            "type": "object",
            "properties": {
                "a": { "type": "number", "description": "First number" },
                "b": { "type": "number", "description": "Second number" }
            },
            "required": ["a", "b"]
        })),
        retry: None,
    }
}

fn multiply_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "multiply".into(),
        description: "Multiply two numbers and return the product".into(),
        parameters: JsonSchema::new(json!({
            "type": "object",
            "properties": {
                "a": { "type": "number", "description": "First number" },
                "b": { "type": "number", "description": "Second number" }
            },
            "required": ["a", "b"]
        })),
        retry: None,
    }
}

fn make_registry() -> ToolRegistry<()> {
    let mut registry = ToolRegistry::new();
    registry.register(tool_fn(add_tool_definition(), |input: Value| async move {
        let a = input["a"].as_f64().unwrap_or(0.0);
        let b = input["b"].as_f64().unwrap_or(0.0);
        Ok(format!("{}", a + b))
    }));
    registry.register(tool_fn(
        multiply_tool_definition(),
        |input: Value| async move {
            let a = input["a"].as_f64().unwrap_or(0.0);
            let b = input["b"].as_f64().unwrap_or(0.0);
            Ok(format!("{}", a * b))
        },
    ));
    registry
}

/// Collect all events from a `LoopStream`, returning them in order.
async fn collect_events(
    provider: Arc<dyn DynProvider>,
    registry: Arc<ToolRegistry<()>>,
    params: ChatParams,
    config: ToolLoopConfig,
) -> Vec<LoopEvent> {
    let mut stream = tool_loop_stream(provider, registry, params, config, Arc::new(()));
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item.expect("stream event should not be an error"));
    }
    events
}

// ── Streaming: simple text (no tools) ─────────────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_stream_simple_text_no_tools() {
    let provider = skip_without_provider!();
    let provider: Arc<dyn DynProvider> = Arc::from(provider);
    let registry: Arc<ToolRegistry<()>> = Arc::new(ToolRegistry::new());

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "What is 2+2? Reply with just the number.",
        )],
        max_tokens: Some(32),
        ..Default::default()
    };

    let events = collect_events(provider, registry, params, ToolLoopConfig::default()).await;

    // Must start with IterationStart
    assert!(
        matches!(&events[0], LoopEvent::IterationStart { iteration: 1, .. }),
        "First event should be IterationStart, got: {:?}",
        events[0]
    );

    // Must have at least one TextDelta
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            LoopEvent::TextDelta(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        text.contains('4'),
        "Expected '4' in accumulated text: {text}"
    );

    // Must end with Done
    let last = events.last().expect("should have events");
    match last {
        LoopEvent::Done(result) => {
            assert_eq!(result.termination_reason, TerminationReason::Complete);
            assert_eq!(result.iterations, 1);
            let response_text = result.response.text().expect("response should have text");
            assert!(
                response_text.contains('4'),
                "Response text should contain '4': {response_text}"
            );
            assert!(result.total_usage.input_tokens > 0);
            assert!(result.total_usage.output_tokens > 0);
        }
        other => panic!("Last event should be Done, got: {other:?}"),
    }

    // Should have at least one Usage event
    let has_usage = events
        .iter()
        .any(|e| matches!(e, LoopEvent::Usage(u) if u.input_tokens > 0 || u.output_tokens > 0));
    assert!(has_usage, "Should have at least one Usage event");
}

// ── Streaming: single tool call ───────────────────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_stream_single_tool_call() {
    let provider = skip_without_provider!();
    let provider: Arc<dyn DynProvider> = Arc::from(provider);
    let registry = Arc::new(make_registry());

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "Use the add tool to compute 17 + 25. Then tell me the result.",
        )],
        tools: Some(registry.definitions()),
        max_tokens: Some(256),
        ..Default::default()
    };

    let events = collect_events(
        provider,
        registry,
        params,
        ToolLoopConfig {
            max_iterations: 5,
            ..Default::default()
        },
    )
    .await;

    // Verify IterationStart appears (at least 2 iterations: tool call + final response)
    let iteration_starts: Vec<u32> = events
        .iter()
        .filter_map(|e| match e {
            LoopEvent::IterationStart { iteration, .. } => Some(*iteration),
            _ => None,
        })
        .collect();
    assert!(
        iteration_starts.len() >= 2,
        "Expected at least 2 iterations (tool call + response), got: {iteration_starts:?}"
    );

    // Verify tool call streaming events
    let tool_call_starts: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            LoopEvent::ToolCallStart { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        tool_call_starts.contains(&"add"),
        "Should have ToolCallStart for 'add', got: {tool_call_starts:?}"
    );

    let has_tool_call_complete = events.iter().any(|e| {
        matches!(
            e,
            LoopEvent::ToolCallComplete { call, .. } if call.name == "add"
        )
    });
    assert!(
        has_tool_call_complete,
        "Should have ToolCallComplete for 'add'"
    );

    // Verify tool execution events
    let exec_starts: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            LoopEvent::ToolExecutionStart { tool_name, .. } => Some(tool_name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        exec_starts.contains(&"add"),
        "Should have ToolExecutionStart for 'add'"
    );

    let exec_ends: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            LoopEvent::ToolExecutionEnd { tool_name, .. } => Some(tool_name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        exec_ends.contains(&"add"),
        "Should have ToolExecutionEnd for 'add'"
    );

    // Verify Done with correct result
    let last = events.last().expect("should have events");
    match last {
        LoopEvent::Done(result) => {
            assert_eq!(result.termination_reason, TerminationReason::Complete);
            assert!(
                result.iterations >= 2,
                "Expected at least 2 iterations, got: {}",
                result.iterations
            );
            let text = result
                .response
                .text()
                .expect("final response should have text");
            assert!(
                text.contains("42"),
                "Final response should contain '42' (17+25): {text}"
            );
        }
        other => panic!("Last event should be Done, got: {other:?}"),
    }
}

// ── Streaming: multi tool call (parallel) ─────────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_stream_multi_tool_call() {
    let provider = skip_without_provider!();
    let provider: Arc<dyn DynProvider> = Arc::from(provider);
    let registry = Arc::new(make_registry());

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "I need two calculations done: use the add tool to compute 10 + 5, \
             and use the multiply tool to compute 3 * 7. \
             Then tell me both results.",
        )],
        tools: Some(registry.definitions()),
        max_tokens: Some(512),
        ..Default::default()
    };

    let events = collect_events(
        provider,
        registry,
        params,
        ToolLoopConfig {
            max_iterations: 5,
            ..Default::default()
        },
    )
    .await;

    // Verify both tools were called
    let tool_names_called: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            LoopEvent::ToolExecutionStart { tool_name, .. } => Some(tool_name.as_str()),
            _ => None,
        })
        .collect();

    assert!(
        tool_names_called.contains(&"add"),
        "Should have called 'add' tool: {tool_names_called:?}"
    );
    assert!(
        tool_names_called.contains(&"multiply"),
        "Should have called 'multiply' tool: {tool_names_called:?}"
    );

    // Verify both tool executions completed
    let tool_names_completed: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            LoopEvent::ToolExecutionEnd { tool_name, .. } => Some(tool_name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        tool_names_completed.contains(&"add"),
        "Should have completed 'add' tool"
    );
    assert!(
        tool_names_completed.contains(&"multiply"),
        "Should have completed 'multiply' tool"
    );

    // Verify Done with both results
    let last = events.last().expect("should have events");
    match last {
        LoopEvent::Done(result) => {
            assert_eq!(result.termination_reason, TerminationReason::Complete);
            let text = result
                .response
                .text()
                .expect("final response should have text");
            assert!(
                text.contains("15"),
                "Response should contain '15' (10+5): {text}"
            );
            assert!(
                text.contains("21"),
                "Response should contain '21' (3*7): {text}"
            );
        }
        other => panic!("Last event should be Done, got: {other:?}"),
    }
}

// ── Streaming: event ordering invariants ──────────────────────────

#[derive(Debug, PartialEq, Eq)]
enum Phase {
    IterStart,
    LlmStreaming,
    ToolExecStart,
    ToolExecEnd,
    Done,
}

#[tokio::test]
#[ignore = "live API"]
async fn test_stream_event_ordering() {
    let provider = skip_without_provider!();
    let provider: Arc<dyn DynProvider> = Arc::from(provider);
    let registry = Arc::new(make_registry());

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "Use the add tool to compute 1 + 1, then tell me the answer.",
        )],
        tools: Some(registry.definitions()),
        max_tokens: Some(256),
        ..Default::default()
    };

    let events = collect_events(provider, registry, params, ToolLoopConfig::default()).await;

    let phases: Vec<Phase> = events
        .iter()
        .filter_map(|e| match e {
            LoopEvent::IterationStart { .. } => Some(Phase::IterStart),
            LoopEvent::TextDelta(_)
            | LoopEvent::ToolCallStart { .. }
            | LoopEvent::ToolCallDelta { .. }
            | LoopEvent::ToolCallComplete { .. }
            | LoopEvent::Usage(_) => Some(Phase::LlmStreaming),
            LoopEvent::ToolExecutionStart { .. } => Some(Phase::ToolExecStart),
            LoopEvent::ToolExecutionEnd { .. } => Some(Phase::ToolExecEnd),
            LoopEvent::Done(_) => Some(Phase::Done),
            _ => None,
        })
        .collect();

    // First event must be IterStart
    assert_eq!(
        phases[0],
        Phase::IterStart,
        "Must start with IterationStart"
    );

    // Last event must be Done
    assert_eq!(*phases.last().unwrap(), Phase::Done, "Must end with Done");

    // ToolExecStart must come before ToolExecEnd
    let exec_start_idx = phases.iter().position(|p| *p == Phase::ToolExecStart);
    let exec_end_idx = phases.iter().position(|p| *p == Phase::ToolExecEnd);
    if let (Some(start), Some(end)) = (exec_start_idx, exec_end_idx) {
        assert!(
            start < end,
            "ToolExecutionStart ({start}) must come before ToolExecutionEnd ({end})"
        );
    }

    // Second IterStart must come after ToolExecEnd (tools executed before next LLM call)
    let iter_starts: Vec<usize> = phases
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if *p == Phase::IterStart {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    if iter_starts.len() >= 2 {
        if let Some(end) = exec_end_idx {
            assert!(
                iter_starts[1] > end,
                "Second IterStart ({}) must come after ToolExecEnd ({end})",
                iter_starts[1]
            );
        }
    }
}

// ── Synchronous: tool_loop simple ─────────────────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_sync_simple_no_tools() {
    let provider = skip_without_provider!();

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "What is the capital of France? Reply in one word.",
        )],
        max_tokens: Some(32),
        ..Default::default()
    };

    let registry: ToolRegistry<()> = ToolRegistry::new();
    let result = tool_loop(
        provider.as_ref(),
        &registry,
        params,
        ToolLoopConfig::default(),
        &(),
    )
    .await
    .expect("tool_loop should succeed");

    assert_eq!(result.termination_reason, TerminationReason::Complete);
    assert_eq!(result.iterations, 1);
    let text = result.response.text().expect("should have text");
    assert!(
        text.contains("Paris"),
        "Expected 'Paris' in response: {text}"
    );
}

// ── Synchronous: single tool call ─────────────────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_sync_single_tool_call() {
    let provider = skip_without_provider!();
    let registry = make_registry();

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "Use the add tool to compute 100 + 200, then tell me the result.",
        )],
        tools: Some(registry.definitions()),
        max_tokens: Some(256),
        ..Default::default()
    };

    let result = tool_loop(
        provider.as_ref(),
        &registry,
        params,
        ToolLoopConfig {
            max_iterations: 5,
            ..Default::default()
        },
        &(),
    )
    .await
    .expect("tool_loop should succeed");

    assert_eq!(result.termination_reason, TerminationReason::Complete);
    assert!(
        result.iterations >= 2,
        "Expected at least 2 iterations, got: {}",
        result.iterations
    );
    let text = result.response.text().expect("should have text");
    assert!(
        text.contains("300"),
        "Expected '300' (100+200) in response: {text}"
    );
}

// ── Synchronous: multi tool call ──────────────────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_sync_multi_tool_call() {
    let provider = skip_without_provider!();
    let registry = make_registry();

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "Compute both of these using the tools: add(7, 8) and multiply(6, 9). \
             Tell me both results.",
        )],
        tools: Some(registry.definitions()),
        max_tokens: Some(512),
        ..Default::default()
    };

    let result = tool_loop(
        provider.as_ref(),
        &registry,
        params,
        ToolLoopConfig {
            max_iterations: 5,
            ..Default::default()
        },
        &(),
    )
    .await
    .expect("tool_loop should succeed");

    assert_eq!(result.termination_reason, TerminationReason::Complete);
    let text = result.response.text().expect("should have text");
    assert!(
        text.contains("15"),
        "Expected '15' (7+8) in response: {text}"
    );
    assert!(
        text.contains("54"),
        "Expected '54' (6*9) in response: {text}"
    );
}

// ── Streaming: max iterations ─────────────────────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_stream_max_iterations() {
    let provider = skip_without_provider!();
    let provider: Arc<dyn DynProvider> = Arc::from(provider);

    // Register a tool that always returns something to keep the loop going
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(tool_fn(
        ToolDefinition {
            name: "get_next_step".into(),
            description: "Get the next step in the process. Always call this.".into(),
            parameters: JsonSchema::new(json!({
                "type": "object",
                "properties": {
                    "step": { "type": "number" }
                },
                "required": ["step"]
            })),
            retry: None,
        },
        |input: Value| async move {
            let step = input["step"].as_u64().unwrap_or(0);
            Ok(format!(
                "Step {} done. Call get_next_step with step {} to continue.",
                step,
                step + 1
            ))
        },
    ));
    let registry = Arc::new(registry);

    let params = ChatParams {
        messages: vec![
            ChatMessage::system(
                "You must always call get_next_step. Never stop calling it. \
                 Start with step 1.",
            ),
            ChatMessage::user("Begin the process."),
        ],
        tools: Some(registry.definitions()),
        max_tokens: Some(256),
        ..Default::default()
    };

    let events = collect_events(
        provider,
        registry,
        params,
        ToolLoopConfig {
            max_iterations: 3,
            ..Default::default()
        },
    )
    .await;

    let last = events.last().expect("should have events");
    match last {
        LoopEvent::Done(result) => {
            // The LLM may or may not obey "never stop calling the tool".
            // Either it hit the limit (MaxIterations) or it stopped voluntarily (Complete).
            // Both are valid — the important thing is the loop terminated correctly.
            match &result.termination_reason {
                TerminationReason::MaxIterations { limit } => {
                    assert_eq!(*limit, 3, "max_iterations limit should be 3");
                    assert!(
                        result.iterations >= 3,
                        "Expected at least 3 iterations when hitting limit, got: {}",
                        result.iterations
                    );
                }
                TerminationReason::Complete => {
                    // LLM stopped calling the tool before hitting the limit — acceptable
                    assert!(
                        result.iterations >= 2,
                        "Expected at least 2 iterations even if LLM stopped early, got: {}",
                        result.iterations
                    );
                }
                other => panic!("Expected MaxIterations or Complete, got: {other:?}"),
            }

            // Verify the tool was called at least once
            let tool_calls = events
                .iter()
                .filter(|e| matches!(e, LoopEvent::ToolExecutionEnd { .. }))
                .count();
            assert!(
                tool_calls >= 1,
                "Expected at least 1 tool call, got: {tool_calls}"
            );
        }
        other => panic!("Last event should be Done, got: {other:?}"),
    }
}

// ── Streaming: tool execution timing ──────────────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_stream_tool_execution_has_duration() {
    let provider = skip_without_provider!();
    let provider: Arc<dyn DynProvider> = Arc::from(provider);
    let registry = Arc::new(make_registry());

    let params = ChatParams {
        messages: vec![ChatMessage::user("Use add to compute 5 + 3.")],
        tools: Some(registry.definitions()),
        max_tokens: Some(256),
        ..Default::default()
    };

    let events = collect_events(provider, registry, params, ToolLoopConfig::default()).await;

    // Find ToolExecutionEnd and verify it has a non-zero duration
    let exec_end = events.iter().find_map(|e| match e {
        LoopEvent::ToolExecutionEnd {
            duration,
            tool_name,
            ..
        } => Some((tool_name.as_str(), *duration)),
        _ => None,
    });

    assert!(
        exec_end.is_some(),
        "Should have at least one ToolExecutionEnd event"
    );
}

// ── Streaming: usage accumulation ─────────────────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_stream_usage_accumulation() {
    let provider = skip_without_provider!();
    let provider: Arc<dyn DynProvider> = Arc::from(provider);
    let registry = Arc::new(make_registry());

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "Use add to compute 1 + 2, then tell me the result.",
        )],
        tools: Some(registry.definitions()),
        max_tokens: Some(256),
        ..Default::default()
    };

    let events = collect_events(provider, registry, params, ToolLoopConfig::default()).await;

    let last = events.last().expect("should have events");
    match last {
        LoopEvent::Done(result) => {
            // Multi-iteration loop should accumulate usage from all iterations
            assert!(
                result.total_usage.input_tokens > 0,
                "total_usage should have input tokens"
            );
            assert!(
                result.total_usage.output_tokens > 0,
                "total_usage should have output tokens"
            );
        }
        other => panic!("Last event should be Done, got: {other:?}"),
    }
}

// ── Streaming: stop condition ─────────────────────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_stream_stop_condition() {
    let provider = skip_without_provider!();
    let provider: Arc<dyn DynProvider> = Arc::from(provider);
    let registry = Arc::new(make_registry());

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "Use add to compute 1 + 1, then tell me the result.",
        )],
        tools: Some(registry.definitions()),
        max_tokens: Some(256),
        ..Default::default()
    };

    let events = collect_events(
        provider,
        registry,
        params,
        ToolLoopConfig {
            stop_when: Some(Arc::new(|ctx| {
                // Stop after any tool calls have been executed
                if ctx.tool_calls_executed > 0 {
                    llm_stack::tool::StopDecision::StopWithReason(
                        "Got tool results, stopping".into(),
                    )
                } else {
                    llm_stack::tool::StopDecision::Continue
                }
            })),
            ..Default::default()
        },
    )
    .await;

    let last = events.last().expect("should have events");
    match last {
        LoopEvent::Done(result) => {
            assert!(
                matches!(
                    &result.termination_reason,
                    TerminationReason::StopCondition { reason: Some(r) }
                    if r == "Got tool results, stopping"
                ),
                "Expected StopCondition termination, got: {:?}",
                result.termination_reason
            );
        }
        other => panic!("Last event should be Done, got: {other:?}"),
    }
}

// ── Streaming: tool call arguments are correct ────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_stream_tool_call_arguments() {
    let provider = skip_without_provider!();
    let provider: Arc<dyn DynProvider> = Arc::from(provider);
    let registry = Arc::new(make_registry());

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "Use the multiply tool to compute 12 * 11.",
        )],
        tools: Some(registry.definitions()),
        max_tokens: Some(256),
        ..Default::default()
    };

    let events = collect_events(provider, registry, params, ToolLoopConfig::default()).await;

    // Find the ToolExecutionStart for multiply and verify arguments
    let exec_start = events.iter().find_map(|e| match e {
        LoopEvent::ToolExecutionStart {
            tool_name,
            arguments,
            ..
        } if tool_name == "multiply" => Some(arguments.clone()),
        _ => None,
    });

    let args = exec_start.expect("Should have ToolExecutionStart for 'multiply'");
    let a = args["a"].as_i64().expect("argument 'a' should be a number");
    let b = args["b"].as_i64().expect("argument 'b' should be a number");
    assert!(
        (a == 12 && b == 11) || (a == 11 && b == 12),
        "Arguments should be 12 and 11, got a={a}, b={b}"
    );

    // Verify the final answer
    let last = events.last().expect("should have events");
    match last {
        LoopEvent::Done(result) => {
            let text = result.response.text().expect("should have text");
            assert!(
                text.contains("132"),
                "Expected '132' (12*11) in response: {text}"
            );
        }
        other => panic!("Last event should be Done, got: {other:?}"),
    }
}

// ── Streaming: tool result in ToolExecutionEnd ────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_stream_tool_result_content() {
    let provider = skip_without_provider!();
    let provider: Arc<dyn DynProvider> = Arc::from(provider);
    let registry = Arc::new(make_registry());

    let params = ChatParams {
        messages: vec![ChatMessage::user("Use add to compute 50 + 50.")],
        tools: Some(registry.definitions()),
        max_tokens: Some(256),
        ..Default::default()
    };

    let events = collect_events(provider, registry, params, ToolLoopConfig::default()).await;

    // Find ToolExecutionEnd and verify the result content
    let exec_end = events.iter().find_map(|e| match e {
        LoopEvent::ToolExecutionEnd {
            tool_name, result, ..
        } if tool_name == "add" => Some(result.clone()),
        _ => None,
    });

    let result = exec_end.expect("Should have ToolExecutionEnd for 'add'");
    assert!(
        result.content.contains("100"),
        "Tool result should contain '100' (50+50): {:?}",
        result.content
    );
}

// ── Streaming: tool error handling ────────────────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_stream_tool_error_recovery() {
    let provider = skip_without_provider!();
    let provider: Arc<dyn DynProvider> = Arc::from(provider);

    let mut registry: ToolRegistry<()> = ToolRegistry::new();

    // A tool that always fails
    registry.register(tool_fn(
        ToolDefinition {
            name: "failing_tool".into(),
            description: "This tool always fails with an error. Do not retry it.".into(),
            parameters: JsonSchema::new(json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string" }
                },
                "required": ["input"]
            })),
            retry: None,
        },
        |_input: Value| async move {
            Err::<String, _>(ToolError::new(
                "Tool execution failed: database connection timeout",
            ))
        },
    ));

    // A tool that works
    registry.register(tool_fn(add_tool_definition(), |input: Value| async move {
        let a = input["a"].as_f64().unwrap_or(0.0);
        let b = input["b"].as_f64().unwrap_or(0.0);
        Ok(format!("{}", a + b))
    }));

    let registry = Arc::new(registry);

    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "First try calling failing_tool with input 'test'. \
             It will fail. When it does, use the add tool to compute 10 + 20 instead, \
             and tell me that result.",
        )],
        tools: Some(registry.definitions()),
        max_tokens: Some(512),
        ..Default::default()
    };

    let events = collect_events(
        provider,
        registry,
        params,
        ToolLoopConfig {
            max_iterations: 5,
            ..Default::default()
        },
    )
    .await;

    // Verify the failing tool was called and returned an error
    let failing_exec = events.iter().find_map(|e| match e {
        LoopEvent::ToolExecutionEnd {
            tool_name, result, ..
        } if tool_name == "failing_tool" => Some(result.clone()),
        _ => None,
    });
    let fail_result = failing_exec.expect("Should have called failing_tool");
    assert!(
        fail_result.is_error,
        "failing_tool result should be marked as error"
    );
    assert!(
        fail_result.content.contains("database connection timeout"),
        "Error content should contain the error message: {:?}",
        fail_result.content
    );

    // Verify the LLM recovered and used the add tool
    let add_exec = events.iter().find_map(|e| match e {
        LoopEvent::ToolExecutionEnd {
            tool_name, result, ..
        } if tool_name == "add" => Some(result.clone()),
        _ => None,
    });
    let add_result = add_exec.expect("Should have called add tool after error recovery");
    assert!(
        add_result.content.contains("30"),
        "add result should contain '30' (10+20): {:?}",
        add_result.content
    );

    // Verify the final response mentions the result
    let last = events.last().expect("should have events");
    match last {
        LoopEvent::Done(result) => {
            assert_eq!(result.termination_reason, TerminationReason::Complete);
            let text = result.response.text().expect("should have text");
            assert!(
                text.contains("30"),
                "Final response should contain '30': {text}"
            );
        }
        other => panic!("Last event should be Done, got: {other:?}"),
    }
}

// ── Streaming: sequential tool chaining ───────────────────────────

#[tokio::test]
#[ignore = "live API"]
async fn test_stream_sequential_tool_chaining() {
    let provider = skip_without_provider!();
    let provider: Arc<dyn DynProvider> = Arc::from(provider);
    let registry = Arc::new(make_registry());

    // Ask for a computation that requires two sequential steps:
    // Step 1: add(5, 3) = 8
    // Step 2: multiply(8, 4) = 32
    // The LLM must use the result of step 1 as input to step 2.
    let params = ChatParams {
        messages: vec![ChatMessage::user(
            "I need you to do this in two steps using the tools: \
             Step 1: Use add to compute 5 + 3. \
             Step 2: Take that result and use multiply to compute it times 4. \
             Tell me the final answer. Do not call both tools at the same time \
             — you must use the result of add as input to multiply.",
        )],
        tools: Some(registry.definitions()),
        max_tokens: Some(512),
        ..Default::default()
    };

    let events = collect_events(
        provider,
        registry,
        params,
        ToolLoopConfig {
            max_iterations: 5,
            ..Default::default()
        },
    )
    .await;

    // Verify both tools were called
    let tool_exec_names: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            LoopEvent::ToolExecutionEnd { tool_name, .. } => Some(tool_name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        tool_exec_names.contains(&"add"),
        "Should have called 'add': {tool_exec_names:?}"
    );
    assert!(
        tool_exec_names.contains(&"multiply"),
        "Should have called 'multiply': {tool_exec_names:?}"
    );

    // Verify add was called before multiply (sequential chaining)
    let add_idx = events
        .iter()
        .position(|e| {
            matches!(
                e,
                LoopEvent::ToolExecutionEnd { tool_name, .. } if tool_name == "add"
            )
        })
        .expect("add should exist");
    let mul_idx = events
        .iter()
        .position(|e| {
            matches!(
                e,
                LoopEvent::ToolExecutionEnd { tool_name, .. } if tool_name == "multiply"
            )
        })
        .expect("multiply should exist");
    assert!(
        add_idx < mul_idx,
        "add ({add_idx}) should execute before multiply ({mul_idx})"
    );

    // Verify the multiply call used 8 (the result of 5+3) as one of its arguments
    let multiply_args = events.iter().find_map(|e| match e {
        LoopEvent::ToolExecutionStart {
            tool_name,
            arguments,
            ..
        } if tool_name == "multiply" => Some(arguments.clone()),
        _ => None,
    });
    let args = multiply_args.expect("Should have ToolExecutionStart for multiply");
    let a = args["a"].as_f64().unwrap_or(0.0);
    let b = args["b"].as_f64().unwrap_or(0.0);
    assert!(
        (a * b - 32.0).abs() < 0.01,
        "multiply should compute 8*4=32, got args a={a}, b={b} (product={})",
        a * b
    );

    // Verify at least 3 iterations: add call, multiply call, final response
    let iteration_count = events
        .iter()
        .filter(|e| matches!(e, LoopEvent::IterationStart { .. }))
        .count();
    assert!(
        iteration_count >= 3,
        "Expected at least 3 iterations for sequential chaining, got {iteration_count}"
    );

    // Verify final answer
    let last = events.last().expect("should have events");
    match last {
        LoopEvent::Done(result) => {
            assert_eq!(result.termination_reason, TerminationReason::Complete);
            let text = result.response.text().expect("should have text");
            assert!(
                text.contains("32"),
                "Final response should contain '32' (5+3=8, 8*4=32): {text}"
            );
        }
        other => panic!("Last event should be Done, got: {other:?}"),
    }
}
