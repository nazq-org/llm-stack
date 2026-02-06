//! Synchronous (non-streaming) tool loop implementation.

use std::time::Instant;

use crate::chat::{ChatMessage, ChatResponse, ContentBlock, StopReason, ToolCall, ToolResult};
use crate::error::LlmError;
use crate::provider::{ChatParams, DynProvider};
use crate::usage::Usage;

use super::LoopDepth;
use super::ToolRegistry;
use super::approval::approve_calls;
use super::config::{
    StopContext, StopDecision, TerminationReason, ToolLoopConfig, ToolLoopEvent, ToolLoopResult,
};
use super::execution::execute_with_events;
use super::loop_detection::{IterationSnapshot, LoopDetectionState, handle_loop_detection};

/// Runs the LLM in a tool-calling loop until completion.
///
/// Each iteration:
/// 1. Calls `provider.generate_boxed()` with the current messages
/// 2. If the response contains tool calls, executes them via the registry
/// 3. Appends tool results as messages and repeats
/// 4. Stops when the LLM returns without tool calls, or max iterations
///    is reached
///
/// # Depth Tracking
///
/// If `Ctx` implements [`LoopDepth`], nested calls are tracked automatically.
/// When `config.max_depth` is set and the context's depth exceeds the limit,
/// returns `Err(LlmError::MaxDepthExceeded)`.
///
/// # Events
///
/// If `config.on_event` is set, the callback will be invoked with
/// [`ToolLoopEvent`]s at key points during execution:
/// - [`ToolLoopEvent::IterationStart`] at the beginning of each iteration
/// - [`ToolLoopEvent::LlmResponseReceived`] after the LLM responds
/// - [`ToolLoopEvent::ToolExecutionStart`] before each tool executes
/// - [`ToolLoopEvent::ToolExecutionEnd`] after each tool completes
///
/// # Errors
///
/// Returns `LlmError` if:
/// - The provider returns an error
/// - Max depth is exceeded (returns `LlmError::MaxDepthExceeded`)
/// - Max iterations is exceeded (returns in result with `TerminationReason::MaxIterations`)
pub async fn tool_loop<Ctx: LoopDepth + Send + Sync + 'static>(
    provider: &dyn DynProvider,
    registry: &ToolRegistry<Ctx>,
    mut params: ChatParams,
    config: ToolLoopConfig,
    ctx: &Ctx,
) -> Result<ToolLoopResult, LlmError> {
    // Check depth limit at entry
    let current_depth = ctx.loop_depth();
    if let Some(max_depth) = config.max_depth {
        if current_depth >= max_depth {
            return Err(LlmError::MaxDepthExceeded {
                current: current_depth,
                limit: max_depth,
            });
        }
    }

    // Create context with incremented depth for tool execution
    let nested_ctx = ctx.with_depth(current_depth + 1);

    let mut total_usage = Usage::default();
    let mut iterations = 0u32;
    let mut tool_calls_executed = 0usize;
    let mut last_tool_results: Vec<ToolResult> = Vec::new();
    let mut loop_state = LoopDetectionState::default();

    // Track start time for timeout
    let start_time = Instant::now();
    let timeout_limit = config.timeout;

    loop {
        // Check timeout at the start of each iteration
        if let Some(limit) = timeout_limit {
            if start_time.elapsed() >= limit {
                // Build a minimal response for timeout case
                return Ok(ToolLoopResult {
                    response: ChatResponse::empty(),
                    iterations,
                    total_usage,
                    termination_reason: TerminationReason::Timeout { limit },
                });
            }
        }

        iterations += 1;

        // Emit iteration start event
        let msg_count = params.messages.len();
        emit_event(&config, || ToolLoopEvent::IterationStart {
            iteration: iterations,
            message_count: msg_count,
        });

        let response = provider.generate_boxed(&params).await?;
        total_usage += &response.usage;

        // Get references for checks before potentially consuming response
        let call_refs: Vec<&ToolCall> = response.tool_calls();
        let text_length = response.text().map_or(0, str::len);
        let has_tool_calls = !call_refs.is_empty();

        // Emit response received event
        emit_event(&config, || ToolLoopEvent::LlmResponseReceived {
            iteration: iterations,
            has_tool_calls,
            text_length,
        });

        // Build the iteration snapshot once for both checks
        let snap = IterationSnapshot {
            response: &response,
            call_refs: &call_refs,
            iterations,
            total_usage: &total_usage,
            tool_calls_executed,
            last_tool_results: &last_tool_results,
            config: &config,
        };

        // Check stop condition and natural termination
        if let Some(result) = check_stop_condition(&snap) {
            return Ok(result);
        }

        if iterations > config.max_iterations {
            return Ok(ToolLoopResult {
                response,
                iterations,
                total_usage,
                termination_reason: TerminationReason::MaxIterations {
                    limit: config.max_iterations,
                },
            });
        }

        // Check for loop detection before executing tools
        if let Some(result) = handle_loop_detection(&mut loop_state, &snap, &mut params.messages) {
            return Ok(result);
        }

        // Now consume response and extract tool calls (no clone needed)
        let (calls, other_content) = response.partition_content();

        // Apply approval callback and execute with events
        // Tools receive nested_ctx with incremented depth
        let (approved_calls, denied_results) = approve_calls(calls, &config);
        let results = execute_with_events(
            registry,
            approved_calls,
            denied_results,
            config.parallel_tool_execution,
            &config,
            &nested_ctx,
        )
        .await;

        // Track executed tool calls
        tool_calls_executed += results.len();
        last_tool_results.clone_from(&results);

        // Append assistant response + tool results to message history
        // other_content already excludes ToolResult blocks (filtered by partition_content)
        params.messages.push(ChatMessage {
            role: crate::chat::ChatRole::Assistant,
            content: other_content,
        });

        for result in results {
            params.messages.push(ChatMessage::tool_result_full(result));
        }
    }
}

/// Emit an event if the callback is configured.
///
/// Takes a closure that produces the event, avoiding allocation when no callback is set.
#[inline]
pub(crate) fn emit_event<F>(config: &ToolLoopConfig, event_fn: F)
where
    F: FnOnce() -> ToolLoopEvent,
{
    if let Some(ref callback) = config.on_event {
        callback(event_fn());
    }
}

/// Check stop condition and natural termination, returning result if should stop.
fn check_stop_condition(snap: &IterationSnapshot<'_>) -> Option<ToolLoopResult> {
    // Check custom stop condition
    if let Some(ref stop_fn) = snap.config.stop_when {
        let ctx = StopContext {
            iteration: snap.iterations,
            response: snap.response,
            total_usage: snap.total_usage,
            tool_calls_executed: snap.tool_calls_executed,
            last_tool_results: snap.last_tool_results,
        };
        match stop_fn(&ctx) {
            StopDecision::Continue => {}
            StopDecision::Stop => {
                return Some(ToolLoopResult {
                    response: snap.response.clone(),
                    iterations: snap.iterations,
                    total_usage: snap.total_usage.clone(),
                    termination_reason: TerminationReason::StopCondition { reason: None },
                });
            }
            StopDecision::StopWithReason(reason) => {
                return Some(ToolLoopResult {
                    response: snap.response.clone(),
                    iterations: snap.iterations,
                    total_usage: snap.total_usage.clone(),
                    termination_reason: TerminationReason::StopCondition {
                        reason: Some(reason),
                    },
                });
            }
        }
    }

    // Check natural termination (no tool calls)
    if snap.call_refs.is_empty() || snap.response.stop_reason != StopReason::ToolUse {
        return Some(ToolLoopResult {
            response: snap.response.clone(),
            iterations: snap.iterations,
            total_usage: snap.total_usage.clone(),
            termination_reason: TerminationReason::Complete,
        });
    }

    None
}

// ── ChatMessage helper ──────────────────────────────────────────────

impl ChatMessage {
    /// Creates a tool result message from a [`ToolResult`].
    pub fn tool_result_full(result: ToolResult) -> Self {
        Self {
            role: crate::chat::ChatRole::Tool,
            content: vec![ContentBlock::ToolResult(result)],
        }
    }
}
