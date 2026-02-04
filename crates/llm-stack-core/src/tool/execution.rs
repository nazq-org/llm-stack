//! Tool execution with event emission.

use std::time::{Duration, Instant};

use futures::{StreamExt, stream};

use crate::chat::{ToolCall, ToolResult};

use super::ToolRegistry;
use super::config::{ToolLoopConfig, ToolLoopEvent};
use super::loop_sync::emit_event;

/// Emit end event for a tool execution.
#[inline]
fn emit_tool_end(
    config: &ToolLoopConfig,
    call_id: String,
    tool_name: String,
    result: &ToolResult,
    duration: Duration,
) {
    if config.on_event.is_some() {
        emit_event(config, || ToolLoopEvent::ToolExecutionEnd {
            call_id,
            tool_name,
            result: result.clone(),
            duration,
        });
    }
}

/// Execute tool calls with start/end events.
///
/// Uses streams for unified parallel/sequential execution:
/// - Parallel: `buffer_unordered` for concurrent execution
/// - Sequential: `then` for ordered execution
pub(crate) async fn execute_with_events<Ctx: Send + Sync + 'static>(
    registry: &ToolRegistry<Ctx>,
    calls: &[ToolCall],
    denied_results: Vec<ToolResult>,
    parallel: bool,
    config: &ToolLoopConfig,
    ctx: &Ctx,
) -> Vec<ToolResult> {
    if calls.is_empty() {
        return denied_results;
    }

    // Setup execution closure
    let execute_one = |call: &ToolCall| {
        let call_id = call.id.clone();
        let tool_name = call.name.clone();
        let arguments = call.arguments.clone();
        async move {
            // Emit start event
            if config.on_event.is_some() {
                emit_event(config, || ToolLoopEvent::ToolExecutionStart {
                    call_id: call_id.clone(),
                    tool_name: tool_name.clone(),
                    arguments: arguments.clone(),
                });
            }

            let start = Instant::now();
            let result = registry
                .execute_by_name(&tool_name, &call_id, arguments, ctx)
                .await;
            let duration = start.elapsed();

            // Emit end event
            emit_tool_end(config, call_id, tool_name, &result, duration);
            result
        }
    };

    // Execute calls in parallel or sequentially, notice the call to .buffer_unordered for parallelism
    let mut results: Vec<ToolResult> = if parallel && calls.len() > 1 {
        stream::iter(calls)
            .map(execute_one)
            .buffer_unordered(calls.len())
            .collect()
            .await
    } else {
        stream::iter(calls).then(execute_one).collect().await
    };

    results.extend(denied_results);
    results
}
