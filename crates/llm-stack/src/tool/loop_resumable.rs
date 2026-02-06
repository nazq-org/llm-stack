//! Resumable tool loop with caller-controlled iteration.
//!
//! Unlike [`tool_loop`](super::tool_loop) which runs to completion autonomously,
//! the resumable loop yields control to the caller after each tool execution
//! round. The caller can then decide to continue, inject messages, or stop.
//!
//! This enables orchestration patterns like:
//! - Multi-agent systems where a master inspects tool results between iterations
//! - External event injection (user follow-ups, worker completions)
//! - Context compaction between iterations
//! - Custom routing logic based on which tools were called
//!
//! # Example
//!
//! ```rust,no_run
//! use llm_stack::tool::{ToolLoopConfig, ToolRegistry, ToolLoopHandle, LoopEvent, LoopCommand};
//! use llm_stack::{ChatParams, ChatMessage};
//!
//! # async fn example(provider: &dyn llm_stack::DynProvider) -> Result<(), llm_stack::LlmError> {
//! let registry: ToolRegistry<()> = ToolRegistry::new();
//! let params = ChatParams {
//!     messages: vec![ChatMessage::user("Hello")],
//!     ..Default::default()
//! };
//!
//! let mut handle = ToolLoopHandle::new(
//!     provider,
//!     &registry,
//!     params,
//!     ToolLoopConfig::default(),
//!     &(),
//! );
//!
//! loop {
//!     match handle.next_event().await {
//!         LoopEvent::ToolsExecuted { ref results, .. } => {
//!             // Inspect results, decide what to do
//!             handle.resume(LoopCommand::Continue);
//!         }
//!         LoopEvent::Completed { .. } => break,
//!         LoopEvent::Error { .. } => break,
//!     }
//! }
//!
//! let result = handle.into_result();
//! # Ok(())
//! # }
//! ```

use std::time::Instant;

use crate::chat::{ChatMessage, ChatResponse, StopReason, ToolCall, ToolResult};
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
use super::loop_sync::emit_event;

/// Events yielded by the resumable tool loop to the caller.
///
/// After receiving a `ToolsExecuted` event, the caller must call
/// [`ToolLoopHandle::resume()`] with a [`LoopCommand`] before calling
/// [`next_event()`](ToolLoopHandle::next_event) again.
///
/// `Completed` and `Error` are terminal — calling `next_event()` after
/// receiving either will return the same terminal event.
#[derive(Debug)]
pub enum LoopEvent {
    /// Tools were executed and results are available.
    ///
    /// The caller must call [`ToolLoopHandle::resume()`] to proceed.
    ToolsExecuted {
        /// The tool calls that the LLM requested.
        tool_calls: Vec<ToolCall>,
        /// Results from executing the tool calls.
        results: Vec<ToolResult>,
        /// Current iteration number (1-indexed).
        iteration: u32,
        /// Accumulated usage across all iterations so far.
        total_usage: Usage,
    },

    /// The loop completed (LLM returned without tool calls, or a stop
    /// condition was met).
    Completed {
        /// The final LLM response.
        response: ChatResponse,
        /// Why the loop terminated.
        termination_reason: TerminationReason,
        /// Total iterations performed.
        iterations: u32,
        /// Accumulated usage across all iterations.
        total_usage: Usage,
    },

    /// An error occurred during the loop.
    Error {
        /// The error that occurred.
        error: LlmError,
        /// Iterations completed before the error.
        iterations: u32,
        /// Usage accumulated before the error.
        total_usage: Usage,
    },
}

/// Commands sent by the caller to control the resumable loop.
///
/// Passed to [`ToolLoopHandle::resume()`] after receiving a
/// [`LoopEvent::ToolsExecuted`].
#[derive(Debug)]
pub enum LoopCommand {
    /// Continue to the next LLM iteration normally.
    Continue,

    /// Inject additional messages before the next LLM call.
    ///
    /// The injected messages are appended after the tool results from
    /// the current round. Use this to provide additional context
    /// (e.g., worker agent results, user follow-ups).
    InjectMessages(Vec<ChatMessage>),

    /// Stop the loop immediately.
    ///
    /// Returns a `Completed` event with `TerminationReason::StopCondition`.
    Stop(Option<String>),
}

/// Caller-driven resumable tool loop.
///
/// Unlike [`tool_loop`](super::tool_loop) which runs autonomously, this struct
/// gives the caller control between each tool execution round. Call
/// [`next_event()`](Self::next_event) to advance the loop, inspect the result,
/// then [`resume()`](Self::resume) to control what happens next.
///
/// # No spawning required
///
/// This is a direct state machine — no background tasks, no channels. The
/// caller drives it by calling `next_event()` which performs one iteration
/// (LLM call + tool execution) and returns.
///
/// # Lifecycle
///
/// 1. Create with [`new()`](Self::new)
/// 2. Call [`next_event()`](Self::next_event) to get the first event
/// 3. If `ToolsExecuted`, call [`resume()`](Self::resume), then `next_event()` again
/// 4. Repeat until `Completed` or `Error`
/// 5. Optionally call [`into_result()`](Self::into_result) for a `ToolLoopResult`
pub struct ToolLoopHandle<'a, Ctx: LoopDepth + Send + Sync + 'static> {
    provider: &'a dyn DynProvider,
    registry: &'a ToolRegistry<Ctx>,
    params: ChatParams,
    config: ToolLoopConfig,
    nested_ctx: Ctx,
    // Loop state
    total_usage: Usage,
    iterations: u32,
    tool_calls_executed: usize,
    last_tool_results: Vec<ToolResult>,
    loop_state: LoopDetectionState,
    start_time: Instant,
    // Whether we've finished (terminal event returned)
    finished: bool,
    // Pending command from the caller (set by resume())
    pending_command: Option<LoopCommand>,
    // Cached final result
    final_result: Option<ToolLoopResult>,
    // Depth error to return on first next_event() call
    depth_error: Option<LlmError>,
}

impl<'a, Ctx: LoopDepth + Send + Sync + 'static> ToolLoopHandle<'a, Ctx> {
    /// Create a new resumable tool loop.
    ///
    /// Does not start execution — call [`next_event()`](Self::next_event) to
    /// begin the first iteration.
    ///
    /// # Depth Tracking
    ///
    /// Same as [`tool_loop`](super::tool_loop) — if `Ctx` implements [`LoopDepth`],
    /// nested calls are tracked and `max_depth` is enforced. If the depth limit
    /// is already exceeded, the first call to `next_event()` returns `Error`.
    pub fn new(
        provider: &'a dyn DynProvider,
        registry: &'a ToolRegistry<Ctx>,
        params: ChatParams,
        config: ToolLoopConfig,
        ctx: &Ctx,
    ) -> Self {
        let current_depth = ctx.loop_depth();
        let depth_error = config.max_depth.and_then(|max_depth| {
            if current_depth >= max_depth {
                Some(LlmError::MaxDepthExceeded {
                    current: current_depth,
                    limit: max_depth,
                })
            } else {
                None
            }
        });

        let nested_ctx = ctx.with_depth(current_depth + 1);

        Self {
            provider,
            registry,
            params,
            config,
            nested_ctx,
            total_usage: Usage::default(),
            iterations: 0,
            tool_calls_executed: 0,
            last_tool_results: Vec::new(),
            loop_state: LoopDetectionState::default(),
            start_time: Instant::now(),
            finished: false,
            pending_command: None,
            final_result: None,
            depth_error,
        }
    }

    /// Advance the loop and return the next event.
    ///
    /// Each call performs one iteration: LLM generation, tool execution (if
    /// applicable), and returns the result. If the LLM doesn't request tools,
    /// returns `Completed`. If tools are executed, returns `ToolsExecuted` and
    /// waits for [`resume()`](Self::resume) before the next call.
    ///
    /// After returning `Completed` or `Error`, all subsequent calls return
    /// the same terminal event.
    pub async fn next_event(&mut self) -> LoopEvent {
        // Phase 1: Pre-iteration guards
        if let Some(event) = self.check_preconditions() {
            return event;
        }

        self.iterations += 1;

        // Emit iteration start event
        let msg_count = self.params.messages.len();
        let iterations = self.iterations;
        emit_event(&self.config, || ToolLoopEvent::IterationStart {
            iteration: iterations,
            message_count: msg_count,
        });

        // Phase 2: LLM call
        let response = match self.provider.generate_boxed(&self.params).await {
            Ok(r) => r,
            Err(e) => return self.finish_error(e),
        };
        self.total_usage += &response.usage;

        // Emit response received event
        let call_refs: Vec<&ToolCall> = response.tool_calls();
        let text_length = response.text().map_or(0, str::len);
        let has_tool_calls = !call_refs.is_empty();
        let iterations = self.iterations;
        emit_event(&self.config, || ToolLoopEvent::LlmResponseReceived {
            iteration: iterations,
            has_tool_calls,
            text_length,
        });

        // Phase 3: Termination checks
        if let Some(event) = self.check_termination(&response, &call_refs) {
            return event;
        }

        // Phase 4: Execute tools and yield
        self.execute_and_yield(response).await
    }

    /// Tell the loop how to proceed after a `ToolsExecuted` event.
    ///
    /// Must be called before the next [`next_event()`](Self::next_event) call.
    /// Has no effect after `Completed` or `Error`.
    pub fn resume(&mut self, command: LoopCommand) {
        if !self.finished {
            self.pending_command = Some(command);
        }
    }

    /// Get a snapshot of the current conversation messages.
    ///
    /// Useful for context window management or debugging.
    pub fn messages(&self) -> &[ChatMessage] {
        &self.params.messages
    }

    /// Get a mutable reference to the conversation messages.
    ///
    /// Allows direct manipulation of the message history between iterations
    /// (e.g., for context compaction/summarization).
    pub fn messages_mut(&mut self) -> &mut Vec<ChatMessage> {
        &mut self.params.messages
    }

    /// Get the accumulated usage across all iterations so far.
    pub fn total_usage(&self) -> &Usage {
        &self.total_usage
    }

    /// Get the current iteration count.
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    /// Whether the loop has finished (returned Completed or Error).
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Consume the handle and return a `ToolLoopResult`.
    ///
    /// If the loop hasn't completed yet, returns a result with current
    /// iteration count and `TerminationReason::Complete`.
    pub fn into_result(self) -> ToolLoopResult {
        self.final_result.unwrap_or_else(|| ToolLoopResult {
            response: ChatResponse::empty(),
            iterations: self.iterations,
            total_usage: self.total_usage,
            termination_reason: TerminationReason::Complete,
        })
    }

    // ── Pre-iteration guards ────────────────────────────────────────

    /// Handle depth errors, already-finished state, pending commands, and timeout.
    ///
    /// Returns `Some(event)` if the loop should not proceed to an LLM call.
    fn check_preconditions(&mut self) -> Option<LoopEvent> {
        // Depth error deferred from new()
        if let Some(error) = self.depth_error.take() {
            return Some(self.finish_error(error));
        }

        // Already finished — return cached terminal event
        if self.finished {
            return Some(self.make_terminal_event());
        }

        // Apply pending command from previous resume() call
        if let Some(command) = self.pending_command.take() {
            match command {
                LoopCommand::Continue => {}
                LoopCommand::InjectMessages(messages) => {
                    self.params.messages.extend(messages);
                }
                LoopCommand::Stop(reason) => {
                    return Some(self.finish(
                        ChatResponse::empty(),
                        TerminationReason::StopCondition { reason },
                    ));
                }
            }
        }

        // Timeout
        if let Some(limit) = self.config.timeout {
            if self.start_time.elapsed() >= limit {
                return Some(
                    self.finish(ChatResponse::empty(), TerminationReason::Timeout { limit }),
                );
            }
        }

        None
    }

    // ── Post-response termination checks ────────────────────────────

    /// Check stop condition, natural completion, max iterations, and loop detection.
    ///
    /// Returns `Some(event)` if the loop should terminate after this response.
    fn check_termination(
        &mut self,
        response: &ChatResponse,
        call_refs: &[&ToolCall],
    ) -> Option<LoopEvent> {
        // Custom stop condition
        if let Some(ref stop_fn) = self.config.stop_when {
            let ctx = StopContext {
                iteration: self.iterations,
                response,
                total_usage: &self.total_usage,
                tool_calls_executed: self.tool_calls_executed,
                last_tool_results: &self.last_tool_results,
            };
            match stop_fn(&ctx) {
                StopDecision::Continue => {}
                StopDecision::Stop => {
                    return Some(self.finish(
                        response.clone(),
                        TerminationReason::StopCondition { reason: None },
                    ));
                }
                StopDecision::StopWithReason(reason) => {
                    return Some(self.finish(
                        response.clone(),
                        TerminationReason::StopCondition {
                            reason: Some(reason),
                        },
                    ));
                }
            }
        }

        // Natural completion (no tool calls)
        if call_refs.is_empty() || response.stop_reason != StopReason::ToolUse {
            return Some(self.finish(response.clone(), TerminationReason::Complete));
        }

        // Max iterations
        if self.iterations > self.config.max_iterations {
            return Some(self.finish(
                response.clone(),
                TerminationReason::MaxIterations {
                    limit: self.config.max_iterations,
                },
            ));
        }

        // Loop detection
        let snap = IterationSnapshot {
            response,
            call_refs,
            iterations: self.iterations,
            total_usage: &self.total_usage,
            tool_calls_executed: self.tool_calls_executed,
            last_tool_results: &self.last_tool_results,
            config: &self.config,
        };
        if let Some(result) =
            handle_loop_detection(&mut self.loop_state, &snap, &mut self.params.messages)
        {
            return Some(self.finish(result.response, result.termination_reason));
        }

        None
    }

    // ── Tool execution ──────────────────────────────────────────────

    /// Extract tool calls, execute them, append results, and yield to caller.
    async fn execute_and_yield(&mut self, response: ChatResponse) -> LoopEvent {
        let (calls, other_content) = response.partition_content();

        // Clone calls for the event return; originals move into approval/execution
        let event_calls = calls.clone();
        let (approved_calls, denied_results) = approve_calls(calls, &self.config);
        let results = execute_with_events(
            self.registry,
            approved_calls,
            denied_results,
            self.config.parallel_tool_execution,
            &self.config,
            &self.nested_ctx,
        )
        .await;

        self.tool_calls_executed += results.len();
        self.last_tool_results.clone_from(&results);

        // Append assistant message with tool calls to conversation
        self.params.messages.push(ChatMessage {
            role: crate::chat::ChatRole::Assistant,
            content: other_content,
        });

        // Append tool results to conversation
        for result in &results {
            self.params
                .messages
                .push(ChatMessage::tool_result_full(result.clone()));
        }

        LoopEvent::ToolsExecuted {
            tool_calls: event_calls,
            results,
            iteration: self.iterations,
            total_usage: self.total_usage.clone(),
        }
    }

    // ── Terminal event helpers ───────────────────────────────────────

    /// Mark the loop as finished and return a `Completed` event.
    fn finish(
        &mut self,
        response: ChatResponse,
        termination_reason: TerminationReason,
    ) -> LoopEvent {
        self.finished = true;
        self.final_result = Some(ToolLoopResult {
            response: response.clone(),
            iterations: self.iterations,
            total_usage: self.total_usage.clone(),
            termination_reason: termination_reason.clone(),
        });
        LoopEvent::Completed {
            response,
            termination_reason,
            iterations: self.iterations,
            total_usage: self.total_usage.clone(),
        }
    }

    /// Mark the loop as finished and return an `Error` event.
    fn finish_error(&mut self, error: LlmError) -> LoopEvent {
        self.finished = true;
        self.final_result = Some(ToolLoopResult {
            response: ChatResponse::empty(),
            iterations: self.iterations,
            total_usage: self.total_usage.clone(),
            termination_reason: TerminationReason::Complete,
        });
        LoopEvent::Error {
            error,
            iterations: self.iterations,
            total_usage: self.total_usage.clone(),
        }
    }

    /// Build a terminal event from cached state (for repeated calls after finish).
    fn make_terminal_event(&self) -> LoopEvent {
        if let Some(ref result) = self.final_result {
            LoopEvent::Completed {
                response: result.response.clone(),
                termination_reason: result.termination_reason.clone(),
                iterations: result.iterations,
                total_usage: result.total_usage.clone(),
            }
        } else {
            LoopEvent::Completed {
                response: ChatResponse::empty(),
                termination_reason: TerminationReason::Complete,
                iterations: self.iterations,
                total_usage: self.total_usage.clone(),
            }
        }
    }
}

impl<Ctx: LoopDepth + Send + Sync + 'static> std::fmt::Debug for ToolLoopHandle<'_, Ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolLoopHandle")
            .field("iterations", &self.iterations)
            .field("tool_calls_executed", &self.tool_calls_executed)
            .field("finished", &self.finished)
            .field("has_pending_command", &self.pending_command.is_some())
            .finish()
    }
}

/// Convenience function to create a resumable tool loop.
///
/// Equivalent to [`ToolLoopHandle::new()`].
pub fn tool_loop_resumable<'a, Ctx: LoopDepth + Send + Sync + 'static>(
    provider: &'a dyn DynProvider,
    registry: &'a ToolRegistry<Ctx>,
    params: ChatParams,
    config: ToolLoopConfig,
    ctx: &Ctx,
) -> ToolLoopHandle<'a, Ctx> {
    ToolLoopHandle::new(provider, registry, params, config, ctx)
}
