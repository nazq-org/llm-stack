//! Synchronous (non-streaming) tool loop implementation.
//!
//! Thin wrapper around [`ToolLoopHandle`](super::ToolLoopHandle) that
//! auto-continues on every `Yielded` result, running the loop to completion.

use crate::error::LlmError;
use crate::provider::{ChatParams, DynProvider};

use super::LoopDepth;
use super::ToolRegistry;
use super::config::{ToolLoopConfig, ToolLoopResult};
use super::loop_resumable::{ToolLoopHandle, TurnResult};

/// Runs the LLM in a tool-calling loop until completion.
///
/// Each iteration:
/// 1. Calls `provider.stream_boxed()` with the current messages
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
/// # Errors
///
/// Returns `LlmError` if:
/// - The provider returns an error
/// - Max depth is exceeded (returns `LlmError::MaxDepthExceeded`)
/// - Max iterations is exceeded (returns in result with `TerminationReason::MaxIterations`)
pub async fn tool_loop<Ctx: LoopDepth + Send + Sync + 'static>(
    provider: &dyn DynProvider,
    registry: &ToolRegistry<Ctx>,
    params: ChatParams,
    config: ToolLoopConfig,
    ctx: &Ctx,
) -> Result<ToolLoopResult, LlmError> {
    let mut handle = ToolLoopHandle::new(provider, registry, params, config, ctx);
    loop {
        match handle.next_turn().await {
            TurnResult::Yielded(turn) => turn.continue_loop(),
            TurnResult::Completed(done) => {
                return Ok(ToolLoopResult {
                    response: done.response,
                    iterations: done.iterations,
                    total_usage: done.total_usage,
                    termination_reason: done.termination_reason,
                });
            }
            TurnResult::Error(err) => return Err(err.error),
        }
    }
}
