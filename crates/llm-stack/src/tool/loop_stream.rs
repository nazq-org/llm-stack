//! Streaming tool loop implementation.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;

use crate::chat::{ChatMessage, ChatResponse, ContentBlock, StopReason, ToolCall, ToolResult};
use crate::error::LlmError;
use crate::provider::{ChatParams, DynProvider};
use crate::stream::{ChatStream, StreamEvent};
use crate::usage::Usage;

use super::LoopDepth;
use super::ToolError;
use super::ToolRegistry;
use super::approval::approve_calls;
use super::config::{StopContext, StopDecision, TerminationReason, ToolLoopConfig, ToolLoopEvent};
use super::execution::execute_with_events;
use super::loop_detection::{IterationSnapshot, LoopDetectionState, handle_loop_detection};
use super::loop_sync::emit_event;

/// Streaming variant of [`tool_loop`](super::tool_loop).
///
/// Yields [`StreamEvent`]s from each iteration. Between iterations
/// (when executing tools), no events are emitted. The final
/// [`StreamEvent::Done`] carries the stop reason from the last
/// iteration.
///
/// # Depth Tracking
///
/// If `Ctx` implements [`LoopDepth`], nested calls are tracked automatically.
/// When `config.max_depth` is set and the context's depth exceeds the limit,
/// yields `Err(LlmError::MaxDepthExceeded)` immediately.
///
/// # Events
///
/// If `config.on_event` is set, the callback will be invoked with
/// [`ToolLoopEvent`]s at key points during execution, same as [`tool_loop`](super::tool_loop).
///
/// Uses `Arc` for provider, registry, and context since they must outlive
/// the returned stream.
#[allow(clippy::needless_pass_by_value)] // ctx is consumed to create nested_ctx
pub fn tool_loop_stream<Ctx: LoopDepth + Send + Sync + 'static>(
    provider: Arc<dyn DynProvider>,
    registry: Arc<ToolRegistry<Ctx>>,
    params: ChatParams,
    config: ToolLoopConfig,
    ctx: Arc<Ctx>,
) -> ChatStream {
    // Check depth limit at entry
    let current_depth = ctx.loop_depth();
    if let Some(max_depth) = config.max_depth {
        if current_depth >= max_depth {
            // Return a stream that immediately yields the depth error
            return Box::pin(futures::stream::once(async move {
                Err(LlmError::MaxDepthExceeded {
                    current: current_depth,
                    limit: max_depth,
                })
            }));
        }
    }

    // Create nested context with incremented depth
    let nested_ctx = Arc::new(ctx.with_depth(current_depth + 1));

    let stream = futures::stream::unfold(
        ToolLoopStreamState::new(provider, registry, params, config, nested_ctx),
        |mut state| async move {
            loop {
                match std::mem::replace(&mut state.phase, StreamPhase::Done) {
                    StreamPhase::Done => return None,
                    StreamPhase::StartIteration => match phase_start_iteration(&mut state).await {
                        PhaseResult::Yield(event, next) => {
                            state.phase = next;
                            return Some((event, state));
                        }
                        PhaseResult::Continue(next) => state.phase = next,
                    },
                    StreamPhase::Streaming(stream) => {
                        match phase_streaming(&mut state, stream).await {
                            PhaseResult::Yield(event, next) => {
                                state.phase = next;
                                return Some((event, state));
                            }
                            PhaseResult::Continue(next) => state.phase = next,
                        }
                    }
                    StreamPhase::ExecutingTools => {
                        state.phase = phase_executing_tools(&mut state).await;
                    }
                }
            }
        },
    );
    Box::pin(stream)
}

/// Result of processing a stream phase.
enum PhaseResult {
    /// Yield an event and transition to the next phase.
    Yield(Result<StreamEvent, LlmError>, StreamPhase),
    /// Transition to the next phase without yielding.
    Continue(StreamPhase),
}

/// Handle the `StartIteration` phase: emit event, check limits, start LLM stream.
async fn phase_start_iteration<Ctx: LoopDepth + Send + Sync + 'static>(
    state: &mut ToolLoopStreamState<Ctx>,
) -> PhaseResult {
    // Check timeout at start of each iteration
    if let Some(limit) = state.timeout_limit {
        if state.start_time.elapsed() >= limit {
            let err = LlmError::ToolExecution {
                tool_name: String::new(),
                source: Box::new(ToolError::new(format!(
                    "Tool loop exceeded timeout of {limit:?}",
                ))),
            };
            return PhaseResult::Yield(Err(err), StreamPhase::Done);
        }
    }

    state.iterations += 1;

    let iterations = state.iterations;
    let msg_count = state.params.messages.len();
    emit_event(&state.config, || ToolLoopEvent::IterationStart {
        iteration: iterations,
        message_count: msg_count,
    });

    if state.iterations > state.config.max_iterations {
        let err = LlmError::ToolExecution {
            tool_name: String::new(),
            source: Box::new(ToolError::new(format!(
                "Tool loop exceeded {} iterations",
                state.config.max_iterations,
            ))),
        };
        return PhaseResult::Yield(Err(err), StreamPhase::Done);
    }

    match state.provider.stream_boxed(&state.params).await {
        Ok(s) => {
            state.current_tool_calls.clear();
            state.current_text.clear();
            PhaseResult::Continue(StreamPhase::Streaming(s))
        }
        Err(e) => PhaseResult::Yield(Err(e), StreamPhase::Done),
    }
}

/// Handle the `Streaming` phase: pull events from the LLM stream.
async fn phase_streaming<Ctx: LoopDepth + Send + Sync + 'static>(
    state: &mut ToolLoopStreamState<Ctx>,
    mut stream: ChatStream,
) -> PhaseResult {
    match stream.next().await {
        Some(Ok(event)) => {
            if let StreamEvent::TextDelta(ref text) = event {
                state.current_text.push_str(text);
            }
            if let StreamEvent::ToolCallComplete { ref call, .. } = event {
                state.current_tool_calls.push(call.clone());
            }
            if let StreamEvent::Usage(ref u) = event {
                state.total_usage += u;
            }
            if let StreamEvent::Done { stop_reason } = &event {
                let iterations = state.iterations;
                let has_tool_calls = !state.current_tool_calls.is_empty();
                let text_length = state.current_text.len();
                emit_event(&state.config, || ToolLoopEvent::LlmResponseReceived {
                    iteration: iterations,
                    has_tool_calls,
                    text_length,
                });

                // Check stop condition before deciding next phase
                if let Some(ref stop_fn) = state.config.stop_when {
                    // Construct a ChatResponse from accumulated state for the stop condition
                    let response = build_response_from_stream_state(state, *stop_reason);
                    let ctx = StopContext {
                        iteration: state.iterations,
                        response: &response,
                        total_usage: &state.total_usage,
                        tool_calls_executed: state.tool_calls_executed,
                        last_tool_results: &state.last_tool_results,
                    };
                    match stop_fn(&ctx) {
                        StopDecision::Continue => {}
                        StopDecision::Stop | StopDecision::StopWithReason(_) => {
                            // Stop early - yield Done and terminate
                            return PhaseResult::Yield(Ok(event), StreamPhase::Done);
                        }
                    }
                }

                if *stop_reason == StopReason::ToolUse && !state.current_tool_calls.is_empty() {
                    // Check for loop detection before executing tools
                    let response = build_response_from_stream_state(state, *stop_reason);
                    let call_refs: Vec<&ToolCall> = state.current_tool_calls.iter().collect();
                    let snap = IterationSnapshot {
                        response: &response,
                        call_refs: &call_refs,
                        iterations: state.iterations,
                        total_usage: &state.total_usage,
                        tool_calls_executed: state.tool_calls_executed,
                        last_tool_results: &state.last_tool_results,
                        config: &state.config,
                    };
                    if let Some(result) = handle_loop_detection(
                        &mut state.loop_state,
                        &snap,
                        &mut state.params.messages,
                    ) {
                        // Convert termination reason to error for streaming
                        let err = match result.termination_reason {
                            TerminationReason::LoopDetected {
                                ref tool_name,
                                count,
                            } => LlmError::ToolExecution {
                                tool_name: tool_name.clone(),
                                source: Box::new(ToolError::new(format!(
                                    "Tool loop detected: '{tool_name}' called {count} \
                                         consecutive times with identical arguments"
                                ))),
                            },
                            _ => LlmError::ToolExecution {
                                tool_name: String::new(),
                                source: Box::new(ToolError::new("Unexpected termination")),
                            },
                        };
                        return PhaseResult::Yield(Err(err), StreamPhase::Done);
                    }
                    // Yield the Done event, then transition to ExecutingTools
                    return PhaseResult::Yield(Ok(event), StreamPhase::ExecutingTools);
                }
            }
            PhaseResult::Yield(Ok(event), StreamPhase::Streaming(stream))
        }
        Some(Err(e)) => PhaseResult::Yield(Err(e), StreamPhase::Done),
        // Stream exhausted — this is the clean termination path after Done event
        None => PhaseResult::Continue(StreamPhase::Done),
    }
}

/// Build a `ChatResponse` from accumulated stream state (for stop condition checks).
fn build_response_from_stream_state<Ctx: LoopDepth + Send + Sync + 'static>(
    state: &ToolLoopStreamState<Ctx>,
    stop_reason: StopReason,
) -> ChatResponse {
    let mut content = Vec::new();
    if !state.current_text.is_empty() {
        content.push(ContentBlock::Text(state.current_text.clone()));
    }
    for call in &state.current_tool_calls {
        content.push(ContentBlock::ToolCall(call.clone()));
    }

    ChatResponse {
        content,
        usage: state.total_usage.clone(),
        stop_reason,
        model: String::new(), // Not available in stream state
        metadata: std::collections::HashMap::new(),
    }
}

/// Handle the `ExecutingTools` phase: run tools, update messages, return next phase.
async fn phase_executing_tools<Ctx: LoopDepth + Send + Sync + 'static>(
    state: &mut ToolLoopStreamState<Ctx>,
) -> StreamPhase {
    // Take ownership of accumulated tool calls; clone for assistant message content
    let calls = std::mem::take(&mut state.current_tool_calls);
    let assistant_calls: Vec<ContentBlock> =
        calls.iter().cloned().map(ContentBlock::ToolCall).collect();
    let (approved, denied) = approve_calls(calls, &state.config);

    let results = execute_with_events(
        &state.registry,
        approved,
        denied,
        state.config.parallel_tool_execution,
        &state.config,
        &state.ctx,
    )
    .await;

    // Track executed tool calls for stop condition
    state.tool_calls_executed += results.len();
    state.last_tool_results.clone_from(&results);

    let mut assistant_content: Vec<ContentBlock> = Vec::new();
    if !state.current_text.is_empty() {
        assistant_content.push(ContentBlock::Text(std::mem::take(&mut state.current_text)));
    }
    assistant_content.extend(assistant_calls);
    state.params.messages.push(ChatMessage {
        role: crate::chat::ChatRole::Assistant,
        content: assistant_content,
    });
    for result in results {
        state
            .params
            .messages
            .push(ChatMessage::tool_result_full(result));
    }

    StreamPhase::StartIteration
}

/// Internal state for the streaming tool loop.
struct ToolLoopStreamState<Ctx: LoopDepth + Send + Sync + 'static> {
    provider: Arc<dyn DynProvider>,
    registry: Arc<ToolRegistry<Ctx>>,
    params: ChatParams,
    config: ToolLoopConfig,
    ctx: Arc<Ctx>,
    iterations: u32,
    total_usage: Usage,
    tool_calls_executed: usize,
    last_tool_results: Vec<ToolResult>,
    current_tool_calls: Vec<ToolCall>,
    current_text: String,
    phase: StreamPhase,
    loop_state: LoopDetectionState,
    /// Start time for timeout tracking.
    start_time: Instant,
    /// Cached timeout limit from config.
    timeout_limit: Option<Duration>,
}

enum StreamPhase {
    StartIteration,
    Streaming(ChatStream),
    ExecutingTools,
    /// Terminal state — unfold returns `None` on next poll.
    Done,
}

impl<Ctx: LoopDepth + Send + Sync + 'static> ToolLoopStreamState<Ctx> {
    fn new(
        provider: Arc<dyn DynProvider>,
        registry: Arc<ToolRegistry<Ctx>>,
        params: ChatParams,
        config: ToolLoopConfig,
        ctx: Arc<Ctx>,
    ) -> Self {
        let timeout_limit = config.timeout;
        Self {
            provider,
            registry,
            params,
            config,
            ctx,
            iterations: 0,
            total_usage: Usage::default(),
            tool_calls_executed: 0,
            last_tool_results: Vec::new(),
            current_tool_calls: Vec::new(),
            current_text: String::new(),
            phase: StreamPhase::StartIteration,
            loop_state: LoopDetectionState::default(),
            start_time: Instant::now(),
            timeout_limit,
        }
    }
}
