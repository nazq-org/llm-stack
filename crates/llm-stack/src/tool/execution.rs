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
/// Accepts owned `Vec<ToolCall>` to avoid deep-cloning `serde_json::Value`
/// arguments. Uses streams for unified parallel/sequential execution:
/// - Parallel: `buffer_unordered` for concurrent execution
/// - Sequential: `then` for ordered execution
pub(crate) async fn execute_with_events<Ctx: Send + Sync + 'static>(
    registry: &ToolRegistry<Ctx>,
    calls: Vec<ToolCall>,
    denied_results: Vec<ToolResult>,
    parallel: bool,
    config: &ToolLoopConfig,
    ctx: &Ctx,
) -> Vec<ToolResult> {
    if calls.is_empty() {
        return denied_results;
    }

    let has_event_cb = config.on_event.is_some();
    let call_count = calls.len();

    // Setup execution closure — moves owned ToolCall, no deep-clone of arguments
    let execute_one = |call: ToolCall| {
        let ToolCall {
            id: call_id,
            name: tool_name,
            arguments,
        } = call;
        async move {
            // Emit start event (only clone arguments when callback is set)
            if has_event_cb {
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

    // Execute calls in parallel or sequentially
    let mut results: Vec<ToolResult> = if parallel && call_count > 1 {
        stream::iter(calls)
            .map(execute_one)
            .buffer_unordered(call_count)
            .collect()
            .await
    } else {
        stream::iter(calls).then(execute_one).collect().await
    };

    results.extend(denied_results);
    results
}
