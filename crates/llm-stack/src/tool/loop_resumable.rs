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
//! use llm_stack::tool::{ToolLoopConfig, ToolRegistry, ToolLoopHandle, TurnResult, LoopCommand};
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
//!     match handle.next_turn().await {
//!         TurnResult::Yielded(turn) => {
//!             // Text from this turn is directly available
//!             if let Some(text) = turn.assistant_text() {
//!                 println!("LLM said: {text}");
//!             }
//!             // Inspect results, decide what to do
//!             turn.continue_loop();
//!         }
//!         TurnResult::Completed(done) => {
//!             println!("Done: {:?}", done.response.text());
//!             break;
//!         }
//!         TurnResult::Error(err) => {
//!             eprintln!("Error: {}", err.error);
//!             break;
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```

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
use super::loop_sync::emit_event;

/// Result of one turn of the tool loop.
///
/// Match on this to determine what happened and what you can do next.
/// Each variant carries the data from the turn AND (for `Yielded`) a handle
/// scoped to valid operations for that state.
///
/// This follows the same pattern as [`std::collections::hash_map::Entry`] —
/// the variant gives you exactly the methods that make sense for that state.
#[must_use = "a TurnResult must be matched — Yielded requires resume() to continue"]
pub enum TurnResult<'a, 'h, Ctx: LoopDepth + Send + Sync + 'static> {
    /// Tools were executed. The caller MUST consume this via `resume()`,
    /// `continue_loop()`, `inject_and_continue()`, or `stop()`.
    ///
    /// While this variant exists, the `ToolLoopHandle` is mutably borrowed
    /// and cannot be used directly. Consuming the `Yielded` releases the
    /// borrow.
    Yielded(Yielded<'a, 'h, Ctx>),

    /// The loop completed (no tool calls, stop condition, max iterations, or timeout).
    Completed(Completed),

    /// An unrecoverable error occurred.
    Error(TurnError),
}

/// Handle returned when tools were executed. Borrows the [`ToolLoopHandle`]
/// mutably, so the caller cannot call `next_turn()` again until this is
/// consumed via `resume()`, `continue_loop()`, `inject_and_continue()`, or
/// `stop()`.
///
/// The text content the LLM produced alongside tool calls is available
/// directly via [`assistant_content`](Self::assistant_content) and
/// [`assistant_text()`](Self::assistant_text) — no need to scan
/// `messages()`.
#[must_use = "must call .resume(), .continue_loop(), .inject_and_continue(), or .stop() to continue"]
pub struct Yielded<'a, 'h, Ctx: LoopDepth + Send + Sync + 'static> {
    handle: &'h mut ToolLoopHandle<'a, Ctx>,

    /// The tool calls the LLM requested.
    pub tool_calls: Vec<ToolCall>,

    /// Results from executing those tool calls.
    pub results: Vec<ToolResult>,

    /// Text content from the LLM's response alongside the tool calls.
    ///
    /// This is the `other_content` from `partition_content()` — `Text`,
    /// `Reasoning`, `Image`, etc. — everything that isn't a `ToolCall` or
    /// `ToolResult`. Previously only accessible by scanning `messages()`.
    pub assistant_content: Vec<ContentBlock>,

    /// Current iteration number (1-indexed).
    pub iteration: u32,

    /// Accumulated usage across all iterations so far.
    pub total_usage: Usage,
}

impl<Ctx: LoopDepth + Send + Sync + 'static> Yielded<'_, '_, Ctx> {
    /// Continue with the given command.
    pub fn resume(self, command: LoopCommand) {
        self.handle.resume(command);
    }

    /// Convenience: continue to the next LLM iteration with no injected messages.
    pub fn continue_loop(self) {
        self.resume(LoopCommand::Continue);
    }

    /// Convenience: inject messages and continue.
    pub fn inject_and_continue(self, messages: Vec<ChatMessage>) {
        self.resume(LoopCommand::InjectMessages(messages));
    }

    /// Convenience: stop the loop.
    pub fn stop(self, reason: Option<String>) {
        self.resume(LoopCommand::Stop(reason));
    }

    /// Extract text from `assistant_content` blocks.
    ///
    /// Returns the LLM's "thinking aloud" text emitted alongside tool calls,
    /// or `None` if there were no text blocks.
    pub fn assistant_text(&self) -> Option<String> {
        let text: String = self
            .assistant_content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if text.is_empty() { None } else { Some(text) }
    }

    /// Access the full message history (read-only).
    pub fn messages(&self) -> &[ChatMessage] {
        self.handle.messages()
    }

    /// Access the full message history (mutable, for context compaction).
    pub fn messages_mut(&mut self) -> &mut Vec<ChatMessage> {
        self.handle.messages_mut()
    }
}

/// Terminal: the loop completed successfully.
pub struct Completed {
    /// The final LLM response. Use `.text()` to get the response text.
    pub response: ChatResponse,
    /// Why the loop terminated.
    pub termination_reason: TerminationReason,
    /// Total iterations performed.
    pub iterations: u32,
    /// Accumulated usage.
    pub total_usage: Usage,
}

/// Terminal: the loop errored.
pub struct TurnError {
    /// The error.
    pub error: LlmError,
    /// Iterations completed before the error.
    pub iterations: u32,
    /// Usage accumulated before the error.
    pub total_usage: Usage,
}

/// Commands sent by the caller to control the resumable loop.
///
/// Passed to [`Yielded::resume()`] after receiving a
/// [`TurnResult::Yielded`].
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

// ── Internal: owned data from one iteration ──────────────────────────

/// Intermediate result from `do_iteration`. Holds owned data (no borrows on
/// the handle) so that `next_turn` can construct `TurnResult` with a fresh
/// `&mut self` afterwards.
enum IterationOutcome {
    ToolsExecuted {
        tool_calls: Vec<ToolCall>,
        results: Vec<ToolResult>,
        assistant_content: Vec<ContentBlock>,
        iteration: u32,
        total_usage: Usage,
    },
    Completed(Completed),
    Error(TurnError),
}

/// Caller-driven resumable tool loop.
///
/// Unlike [`tool_loop`](super::tool_loop) which runs autonomously, this struct
/// gives the caller control between each tool execution round. Call
/// [`next_turn()`](Self::next_turn) to advance the loop, inspect the result,
/// then consume the [`Yielded`] handle to control what happens next.
///
/// # No spawning required
///
/// This is a direct state machine — no background tasks, no channels. The
/// caller drives it by calling `next_turn()` which performs one iteration
/// (LLM call + tool execution) and returns.
///
/// # Lifecycle
///
/// 1. Create with [`new()`](Self::new)
/// 2. Call [`next_turn()`](Self::next_turn) to get the first result
/// 3. If `Yielded`, consume via `resume()` / `continue_loop()` / etc., then
///    call `next_turn()` again
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
    // Depth error to return on first next_turn() call
    depth_error: Option<LlmError>,
}

impl<'a, Ctx: LoopDepth + Send + Sync + 'static> ToolLoopHandle<'a, Ctx> {
    /// Create a new resumable tool loop.
    ///
    /// Does not start execution — call [`next_turn()`](Self::next_turn) to
    /// begin the first iteration.
    ///
    /// # Depth Tracking
    ///
    /// Same as [`tool_loop`](super::tool_loop) — if `Ctx` implements [`LoopDepth`],
    /// nested calls are tracked and `max_depth` is enforced. If the depth limit
    /// is already exceeded, the first call to `next_turn()` returns `Error`.
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

    /// Advance the loop and return the result of this turn.
    ///
    /// Each call performs one iteration: LLM generation, tool execution (if
    /// applicable), and returns the result.
    ///
    /// Returns a [`TurnResult`] that must be matched:
    /// - [`TurnResult::Yielded`] — tools ran, consume via `resume()` /
    ///   `continue_loop()` / `inject_and_continue()` / `stop()` to continue
    /// - [`TurnResult::Completed`] — loop is done, read `.response`
    /// - [`TurnResult::Error`] — loop failed, read `.error`
    ///
    /// After `Completed` or `Error`, all subsequent calls return the same
    /// terminal result.
    pub async fn next_turn(&mut self) -> TurnResult<'a, '_, Ctx> {
        let outcome = self.do_iteration().await;
        match outcome {
            IterationOutcome::ToolsExecuted {
                tool_calls,
                results,
                assistant_content,
                iteration,
                total_usage,
            } => TurnResult::Yielded(Yielded {
                handle: self,
                tool_calls,
                results,
                assistant_content,
                iteration,
                total_usage,
            }),
            IterationOutcome::Completed(c) => TurnResult::Completed(c),
            IterationOutcome::Error(e) => TurnResult::Error(e),
        }
    }

    /// Tell the loop how to proceed before the next [`next_turn()`](Self::next_turn) call.
    ///
    /// When using [`TurnResult::Yielded`], prefer the convenience methods on
    /// [`Yielded`] (`continue_loop()`, `inject_and_continue()`, `stop()`),
    /// which consume the yielded handle and call this internally.
    ///
    /// This method is useful when you need to set a command on the handle
    /// directly — for example, when driving the handle from an external
    /// event loop that receives the command asynchronously after the
    /// `Yielded` has already been consumed.
    ///
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

    // ── Core iteration logic ────────────────────────────────────────

    /// Perform one iteration and return owned data. Does NOT borrow `&mut self`
    /// beyond this call — the returned `IterationOutcome` is fully owned.
    async fn do_iteration(&mut self) -> IterationOutcome {
        // Phase 1: Pre-iteration guards
        if let Some(outcome) = self.check_preconditions() {
            return outcome;
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
        if let Some(outcome) = self.check_termination(&response, &call_refs) {
            return outcome;
        }

        // Phase 4: Execute tools and build outcome
        self.execute_tools(response).await
    }

    // ── Pre-iteration guards ────────────────────────────────────────

    /// Handle depth errors, already-finished state, pending commands, and timeout.
    ///
    /// Returns `Some(outcome)` if the loop should not proceed to an LLM call.
    fn check_preconditions(&mut self) -> Option<IterationOutcome> {
        // Depth error deferred from new()
        if let Some(error) = self.depth_error.take() {
            return Some(self.finish_error(error));
        }

        // Already finished — return cached terminal event
        if self.finished {
            return Some(self.make_terminal_outcome());
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
    /// Returns `Some(outcome)` if the loop should terminate after this response.
    fn check_termination(
        &mut self,
        response: &ChatResponse,
        call_refs: &[&ToolCall],
    ) -> Option<IterationOutcome> {
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

    /// Extract tool calls, execute them, append results, and return owned outcome.
    async fn execute_tools(&mut self, response: ChatResponse) -> IterationOutcome {
        let (calls, other_content) = response.partition_content();

        // Clone calls for the outcome return; originals move into approval/execution
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

        // Append assistant message with non-tool content to conversation
        self.params.messages.push(ChatMessage {
            role: crate::chat::ChatRole::Assistant,
            content: other_content.clone(),
        });

        // Append tool results to conversation
        for result in &results {
            self.params
                .messages
                .push(ChatMessage::tool_result_full(result.clone()));
        }

        let iteration = self.iterations;
        let total_usage = self.total_usage.clone();

        IterationOutcome::ToolsExecuted {
            tool_calls: event_calls,
            results,
            assistant_content: other_content,
            iteration,
            total_usage,
        }
    }

    // ── Terminal outcome helpers ─────────────────────────────────────

    /// Mark the loop as finished and return a `Completed` outcome.
    fn finish(
        &mut self,
        response: ChatResponse,
        termination_reason: TerminationReason,
    ) -> IterationOutcome {
        self.finished = true;
        self.final_result = Some(ToolLoopResult {
            response: response.clone(),
            iterations: self.iterations,
            total_usage: self.total_usage.clone(),
            termination_reason: termination_reason.clone(),
        });
        IterationOutcome::Completed(Completed {
            response,
            termination_reason,
            iterations: self.iterations,
            total_usage: self.total_usage.clone(),
        })
    }

    /// Mark the loop as finished and return an `Error` outcome.
    fn finish_error(&mut self, error: LlmError) -> IterationOutcome {
        self.finished = true;
        self.final_result = Some(ToolLoopResult {
            response: ChatResponse::empty(),
            iterations: self.iterations,
            total_usage: self.total_usage.clone(),
            termination_reason: TerminationReason::Complete,
        });
        IterationOutcome::Error(TurnError {
            error,
            iterations: self.iterations,
            total_usage: self.total_usage.clone(),
        })
    }

    /// Build a terminal outcome from cached state (for repeated calls after finish).
    fn make_terminal_outcome(&self) -> IterationOutcome {
        if let Some(ref result) = self.final_result {
            IterationOutcome::Completed(Completed {
                response: result.response.clone(),
                termination_reason: result.termination_reason.clone(),
                iterations: result.iterations,
                total_usage: result.total_usage.clone(),
            })
        } else {
            IterationOutcome::Completed(Completed {
                response: ChatResponse::empty(),
                termination_reason: TerminationReason::Complete,
                iterations: self.iterations,
                total_usage: self.total_usage.clone(),
            })
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
