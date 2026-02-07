//! Streaming tool loop implementation.
//!
//! Returns a [`LoopStream`] — a unified stream of [`LoopEvent`]s that
//! includes both LLM streaming events (text deltas, tool call fragments)
//! and loop-level lifecycle events (iteration boundaries, tool execution
//! progress). Terminates with [`LoopEvent::Done`] carrying the final
//! [`ToolLoopResult`].
//!
//! Between iterations (when executing tools), the stream emits
//! `ToolExecutionStart` and `ToolExecutionEnd` events. No LLM deltas
//! are emitted during this phase.

use std::collections::VecDeque;
use std::sync::Arc;

use futures::StreamExt;

use crate::chat::{ChatResponse, ContentBlock, StopReason, ToolCall};
use crate::error::LlmError;
use crate::provider::{ChatParams, DynProvider};
use crate::stream::{ChatStream, StreamEvent};
use crate::usage::Usage;

use super::LoopDepth;
use super::ToolRegistry;
use super::config::{LoopEvent, LoopStream, ToolLoopConfig};
use super::loop_core::{IterationOutcome, LoopCore, StartOutcome};

/// Streaming variant of [`tool_loop`](super::tool_loop).
///
/// Yields [`LoopEvent`]s from each iteration. LLM streaming events
/// (text deltas, tool call fragments) are interleaved with loop-level
/// events (iteration start, tool execution start/end). The stream
/// terminates with [`LoopEvent::Done`] carrying the final
/// [`ToolLoopResult`](super::ToolLoopResult).
///
/// # Depth Tracking
///
/// If `Ctx` implements [`LoopDepth`], nested calls are tracked automatically.
/// When `config.max_depth` is set and the context's depth exceeds the limit,
/// yields `Err(LlmError::MaxDepthExceeded)` immediately.
///
/// Uses `Arc` for provider, registry, and context since they must outlive
/// the returned stream.
#[allow(clippy::needless_pass_by_value)] // ctx Arc is consumed into LoopCore
pub fn tool_loop_stream<Ctx: LoopDepth + Send + Sync + 'static>(
    provider: Arc<dyn DynProvider>,
    registry: Arc<ToolRegistry<Ctx>>,
    params: ChatParams,
    config: ToolLoopConfig,
    ctx: Arc<Ctx>,
) -> LoopStream {
    let core = LoopCore::new(params, config, &*ctx);

    let state = UnfoldState {
        core,
        provider,
        registry,
        phase: StreamPhase::StartIteration,
        current_text: String::new(),
        current_tool_calls: Vec::new(),
        current_usage: Usage::default(),
        pending_events: VecDeque::new(),
    };

    let stream = futures::stream::unfold(state, |mut state| async move {
        loop {
            // First, drain any pending events (from LoopCore's event buffer)
            if let Some(event) = state.pending_events.pop_front() {
                return Some((event, state));
            }

            match std::mem::replace(&mut state.phase, StreamPhase::Done) {
                StreamPhase::Done => return None,

                StreamPhase::StartIteration => {
                    match state.core.start_iteration(&*state.provider).await {
                        StartOutcome::Stream(s) => {
                            state.current_text.clear();
                            state.current_tool_calls.clear();
                            state.current_usage = Usage::default();
                            // Drain IterationStart event from core
                            state.load_core_events();
                            state.phase = StreamPhase::Streaming(s);
                        }
                        StartOutcome::Terminal(outcome) => {
                            // Drain any events (e.g., Done from finish())
                            state.load_core_events();
                            if let Some(event) = outcome_to_error(*outcome) {
                                state.phase = StreamPhase::Done;
                                // Push error, then let pending_events drain
                                state.pending_events.push_back(event);
                            }
                            // Continue loop to drain pending_events
                        }
                    }
                }

                StreamPhase::Streaming(mut stream) => match stream.next().await {
                    Some(Ok(event)) => {
                        // Accumulate for finish_iteration
                        if let StreamEvent::TextDelta(ref t) = event {
                            state.current_text.push_str(t);
                        }
                        if let StreamEvent::ToolCallComplete { ref call, .. } = event {
                            state.current_tool_calls.push(call.clone());
                        }
                        if let StreamEvent::Usage(ref u) = event {
                            state.current_usage += u;
                        }

                        let is_done = matches!(&event, StreamEvent::Done { .. });
                        let loop_event = translate_stream_event(event);

                        if is_done {
                            // Provider stream done — move to tool execution
                            state.phase = StreamPhase::ExecutingTools;
                        } else {
                            state.phase = StreamPhase::Streaming(stream);
                        }

                        // Don't forward provider-level Done — it's not the loop being done
                        if let Some(le) = loop_event {
                            return Some((Ok(le), state));
                        }
                        // If we filtered out Done, continue loop
                    }
                    Some(Err(e)) => {
                        state.phase = StreamPhase::Done;
                        return Some((Err(e), state));
                    }
                    None => {
                        // Stream exhausted without Done — clean end
                        return None;
                    }
                },

                StreamPhase::ExecutingTools => {
                    let response = build_response(
                        &state.current_text,
                        &state.current_tool_calls,
                        std::mem::take(&mut state.current_usage),
                    );
                    let outcome = state.core.finish_iteration(response, &state.registry).await;

                    // Drain tool execution events + possible Done from core
                    state.load_core_events();

                    match outcome {
                        IterationOutcome::ToolsExecuted { .. } => {
                            state.phase = StreamPhase::StartIteration;
                        }
                        IterationOutcome::Completed(_) => {
                            // Done event already in pending_events from finish()
                            state.phase = StreamPhase::Done;
                        }
                        IterationOutcome::Error(data) => {
                            state.phase = StreamPhase::Done;
                            state.pending_events.push_back(Err(data.error));
                        }
                    }
                    // Continue loop to drain pending_events
                }
            }
        }
    });

    Box::pin(stream)
}

/// Phases of the streaming state machine.
enum StreamPhase {
    StartIteration,
    Streaming(ChatStream),
    ExecutingTools,
    Done,
}

/// State carried through the unfold.
struct UnfoldState<Ctx: LoopDepth + Send + Sync + 'static> {
    core: LoopCore<Ctx>,
    provider: Arc<dyn DynProvider>,
    registry: Arc<ToolRegistry<Ctx>>,
    phase: StreamPhase,
    current_text: String,
    current_tool_calls: Vec<ToolCall>,
    current_usage: Usage,
    /// Events waiting to be yielded (FIFO).
    pending_events: VecDeque<Result<LoopEvent, LlmError>>,
}

impl<Ctx: LoopDepth + Send + Sync + 'static> UnfoldState<Ctx> {
    /// Drain events from `LoopCore`'s buffer into our pending queue (FIFO).
    fn load_core_events(&mut self) {
        for event in self.core.drain_events() {
            self.pending_events.push_back(Ok(event));
        }
    }
}

/// Translate a provider `StreamEvent` into a `LoopEvent`.
///
/// Returns `None` for `StreamEvent::Done` — the provider's "done" is not
/// the loop's "done". The loop continues with tool execution.
fn translate_stream_event(event: StreamEvent) -> Option<LoopEvent> {
    match event {
        StreamEvent::TextDelta(t) => Some(LoopEvent::TextDelta(t)),
        StreamEvent::ReasoningDelta(t) => Some(LoopEvent::ReasoningDelta(t)),
        StreamEvent::ToolCallStart { index, id, name } => {
            Some(LoopEvent::ToolCallStart { index, id, name })
        }
        StreamEvent::ToolCallDelta { index, json_chunk } => {
            Some(LoopEvent::ToolCallDelta { index, json_chunk })
        }
        StreamEvent::ToolCallComplete { index, call } => {
            Some(LoopEvent::ToolCallComplete { index, call })
        }
        StreamEvent::Usage(u) => Some(LoopEvent::Usage(u)),
        StreamEvent::Done { .. } => None, // Filtered — not the loop's done
    }
}

/// Build a `ChatResponse` from accumulated stream data.
fn build_response(text: &str, tool_calls: &[ToolCall], usage: Usage) -> ChatResponse {
    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(ContentBlock::Text(text.to_owned()));
    }
    for call in tool_calls {
        content.push(ContentBlock::ToolCall(call.clone()));
    }

    let stop_reason = if tool_calls.is_empty() {
        StopReason::EndTurn
    } else {
        StopReason::ToolUse
    };

    ChatResponse {
        content,
        usage,
        stop_reason,
        model: String::new(),
        metadata: std::collections::HashMap::new(),
    }
}

/// Convert a terminal `IterationOutcome` into an error event.
///
/// `Completed` outcomes are NOT converted to errors — they produce
/// `LoopEvent::Done` via the core's event buffer. Only `Error` outcomes
/// become `Err` items in the stream.
fn outcome_to_error(outcome: IterationOutcome) -> Option<Result<LoopEvent, LlmError>> {
    match outcome {
        IterationOutcome::Error(data) => Some(Err(data.error)),
        // Completed outcomes push Done into the core's event buffer
        IterationOutcome::Completed(_) | IterationOutcome::ToolsExecuted { .. } => None,
    }
}
