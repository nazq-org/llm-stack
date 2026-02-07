//! Tool execution with event collection.

use std::time::Instant;

use futures::{StreamExt, stream};

use crate::chat::{ToolCall, ToolResult};

use super::ToolRegistry;
use super::config::LoopEvent;

/// Result of executing tool calls: the tool results plus any events generated.
pub(crate) struct ExecutionResult {
    pub results: Vec<ToolResult>,
    pub events: Vec<LoopEvent>,
}

/// Execute tool calls, collecting start/end events.
///
/// Accepts owned `Vec<ToolCall>` to avoid deep-cloning `serde_json::Value`
/// arguments. Uses streams for unified parallel/sequential execution:
/// - Parallel: `buffer_unordered` for concurrent execution (completion order)
/// - Sequential: `then` for ordered execution (call order)
///
/// **Event ordering**: When `parallel` is true, `ToolExecutionStart`/`End`
/// event pairs are emitted in completion order, not call order. Use
/// `call_id` to correlate events across parallel calls.
pub(crate) async fn execute_with_events<Ctx: Send + Sync + 'static>(
    registry: &ToolRegistry<Ctx>,
    calls: Vec<ToolCall>,
    denied_results: Vec<ToolResult>,
    parallel: bool,
    ctx: &Ctx,
) -> ExecutionResult {
    if calls.is_empty() {
        return ExecutionResult {
            results: denied_results,
            events: Vec::new(),
        };
    }

    let call_count = calls.len();
    let mut events = Vec::with_capacity(call_count * 2);

    // Setup execution closure — moves owned ToolCall, no deep-clone of arguments
    let execute_one = |call: ToolCall| {
        let ToolCall {
            id: call_id,
            name: tool_name,
            arguments,
        } = call;
        async move {
            let start_event = LoopEvent::ToolExecutionStart {
                call_id: call_id.clone(),
                tool_name: tool_name.clone(),
                arguments: arguments.clone(),
            };

            let start = Instant::now();
            let result = registry
                .execute_by_name(&tool_name, &call_id, arguments, ctx)
                .await;
            let duration = start.elapsed();

            let end_event = LoopEvent::ToolExecutionEnd {
                call_id,
                tool_name,
                result: result.clone(),
                duration,
            };
            (result, start_event, end_event)
        }
    };

    // Execute calls in parallel or sequentially
    let outcomes: Vec<(ToolResult, LoopEvent, LoopEvent)> = if parallel && call_count > 1 {
        stream::iter(calls)
            .map(execute_one)
            .buffer_unordered(call_count)
            .collect()
            .await
    } else {
        stream::iter(calls).then(execute_one).collect().await
    };

    let mut results = Vec::with_capacity(outcomes.len() + denied_results.len());
    for (result, start_event, end_event) in outcomes {
        events.push(start_event);
        events.push(end_event);
        results.push(result);
    }

    results.extend(denied_results);
    ExecutionResult { results, events }
}
