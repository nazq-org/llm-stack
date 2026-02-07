//! Shared iteration engine for all tool loop variants.
//!
//! `LoopCore<Ctx>` holds all mutable state for a tool loop and provides the
//! iteration methods. The caller provides `&dyn DynProvider` and
//! `&ToolRegistry<Ctx>` for each operation — `LoopCore` has no opinion on
//! ownership (borrowed vs `Arc`).
//!
//! # Streaming-first design
//!
//! The fundamental LLM call uses `stream_boxed()`. Non-streaming callers
//! collect the stream internally via [`do_iteration`](LoopCore::do_iteration).
//! Streaming callers use [`start_iteration`](LoopCore::start_iteration) to
//! obtain the raw `ChatStream`, drive it themselves (yielding events to the
//! caller), then hand the accumulated response back via
//! [`finish_iteration`](LoopCore::finish_iteration).
//!
//! # Event buffer
//!
//! `LoopCore` accumulates [`LoopEvent`]s in an internal buffer as it performs
//! each phase. Callers drain the buffer via [`drain_events`](LoopCore::drain_events)
//! to forward events to consumers. This replaces the old `on_event` callback.

use std::collections::HashMap;
use std::time::Instant;

use futures::StreamExt;

use crate::chat::{ChatMessage, ChatResponse, ContentBlock, StopReason, ToolCall, ToolResult};
use crate::error::LlmError;
use crate::provider::{ChatParams, DynProvider};
use crate::stream::{ChatStream, StreamEvent};
use crate::usage::Usage;

use super::LoopDepth;
use super::ToolRegistry;
use super::approval::approve_calls;
use super::config::{
    LoopEvent, StopContext, StopDecision, TerminationReason, ToolLoopConfig, ToolLoopResult,
};
use super::execution::execute_with_events;
use super::loop_detection::{IterationSnapshot, LoopDetectionState, handle_loop_detection};
use super::loop_resumable::LoopCommand;

// ── IterationOutcome ────────────────────────────────────────────────

/// Intermediate result from `do_iteration` / `finish_iteration`. Holds
/// owned data (no borrows on the core) so the caller can construct its
/// own `TurnResult` with a fresh `&mut self` afterwards.
pub(crate) enum IterationOutcome {
    ToolsExecuted {
        tool_calls: Vec<ToolCall>,
        results: Vec<ToolResult>,
        assistant_content: Vec<ContentBlock>,
        iteration: u32,
        total_usage: Usage,
    },
    Completed(CompletedData),
    Error(ErrorData),
}

/// Owned data for a completed iteration.
pub(crate) struct CompletedData {
    pub response: ChatResponse,
    pub termination_reason: TerminationReason,
    pub iterations: u32,
    pub total_usage: Usage,
}

/// Owned data for a failed iteration.
pub(crate) struct ErrorData {
    pub error: LlmError,
    pub iterations: u32,
    pub total_usage: Usage,
}

// ── StartOutcome ────────────────────────────────────────────────────

/// Result of [`LoopCore::start_iteration`].
///
/// Either the LLM stream is ready to be consumed, or the iteration ended
/// early (precondition failure, max iterations, timeout, etc.).
pub(crate) enum StartOutcome {
    /// The LLM stream is ready. The caller should drive it to completion,
    /// accumulate the response, then call `finish_iteration`.
    Stream(ChatStream),
    /// The iteration ended before the LLM was called (depth error,
    /// already finished, timeout, stop command, max iterations).
    Terminal(Box<IterationOutcome>),
}

// ── LoopCore ────────────────────────────────────────────────────────

/// Shared mutable state for all tool loop variants.
///
/// Contains the iteration logic used by all loop implementations.
/// Methods that need the LLM provider or tool registry accept them by
/// reference, so this struct works with both borrowed and Arc-owned
/// callers.
///
/// # Two-phase iteration
///
/// 1. [`start_iteration`](Self::start_iteration) — pre-checks, opens the
///    LLM stream via `stream_boxed()`.
/// 2. [`finish_iteration`](Self::finish_iteration) — given the accumulated
///    `ChatResponse`, runs termination checks and tool execution.
///
/// For callers that don't need streaming, [`do_iteration`](Self::do_iteration)
/// composes both phases, collecting the stream internally.
pub(crate) struct LoopCore<Ctx: LoopDepth + Send + Sync + 'static> {
    pub(crate) params: ChatParams,
    config: ToolLoopConfig,
    nested_ctx: Ctx,
    total_usage: Usage,
    iterations: u32,
    tool_calls_executed: usize,
    last_tool_results: Vec<ToolResult>,
    loop_state: LoopDetectionState,
    start_time: Instant,
    finished: bool,
    pending_command: Option<LoopCommand>,
    final_result: Option<ToolLoopResult>,
    depth_error: Option<LlmError>,
    events: Vec<LoopEvent>,
}

impl<Ctx: LoopDepth + Send + Sync + 'static> LoopCore<Ctx> {
    /// Create a new `LoopCore` with the given params, config, and context.
    ///
    /// If the context's depth already exceeds `config.max_depth`, the depth
    /// error is stored and returned on the first iteration call.
    pub(crate) fn new(params: ChatParams, config: ToolLoopConfig, ctx: &Ctx) -> Self {
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
            events: Vec::new(),
        }
    }

    // ── Two-phase iteration (streaming-first) ────────────────────

    /// Phase 1: Pre-checks and LLM stream creation.
    ///
    /// Runs precondition guards (depth, finished, pending command, timeout,
    /// max iterations), then calls `provider.stream_boxed()` to start the
    /// LLM generation.
    ///
    /// Returns `StartOutcome::Stream` with the raw `ChatStream` if the LLM
    /// call succeeded, or `StartOutcome::Terminal` if the iteration ended
    /// early.
    ///
    /// Pushes `LoopEvent::IterationStart` into the event buffer on success.
    ///
    /// After obtaining the stream, the caller should:
    /// 1. Consume all `StreamEvent`s (optionally forwarding them)
    /// 2. Accumulate text, tool calls, and usage into a `ChatResponse`
    /// 3. Call [`finish_iteration`](Self::finish_iteration) with the result
    pub(crate) async fn start_iteration(&mut self, provider: &dyn DynProvider) -> StartOutcome {
        // Phase 1: Pre-iteration guards
        if let Some(outcome) = self.check_preconditions() {
            return StartOutcome::Terminal(Box::new(outcome));
        }

        self.iterations += 1;

        // Push iteration start event
        self.events.push(LoopEvent::IterationStart {
            iteration: self.iterations,
            message_count: self.params.messages.len(),
        });

        // Max iterations check (after increment, before LLM call)
        if self.iterations > self.config.max_iterations {
            return StartOutcome::Terminal(Box::new(self.finish(
                ChatResponse::empty(),
                TerminationReason::MaxIterations {
                    limit: self.config.max_iterations,
                },
            )));
        }

        // Start LLM stream
        match provider.stream_boxed(&self.params).await {
            Ok(stream) => StartOutcome::Stream(stream),
            Err(e) => StartOutcome::Terminal(Box::new(self.finish_error(e))),
        }
    }

    /// Phase 2: Post-stream termination checks and tool execution.
    ///
    /// Called after the caller has fully consumed the `ChatStream` from
    /// `start_iteration` and assembled the accumulated `ChatResponse`.
    ///
    /// Runs: usage accounting → stop condition → natural completion →
    /// loop detection → tool execution.
    ///
    /// Tool execution events (`ToolExecutionStart`, `ToolExecutionEnd`)
    /// are pushed into the event buffer. Terminal outcomes push
    /// `LoopEvent::Done`.
    pub(crate) async fn finish_iteration(
        &mut self,
        response: ChatResponse,
        registry: &ToolRegistry<Ctx>,
    ) -> IterationOutcome {
        self.total_usage += &response.usage;

        // Termination checks
        let call_refs: Vec<&ToolCall> = response.tool_calls();
        if let Some(outcome) = self.check_termination(&response, &call_refs) {
            return outcome;
        }

        // Execute tools
        self.execute_tools(registry, response).await
    }

    // ── Composed iteration (non-streaming) ───────────────────────

    /// Perform one full iteration: start → collect stream → finish.
    ///
    /// This is the non-streaming path. It calls `start_iteration`, collects
    /// the entire stream into a `ChatResponse`, then calls `finish_iteration`.
    pub(crate) async fn do_iteration(
        &mut self,
        provider: &dyn DynProvider,
        registry: &ToolRegistry<Ctx>,
    ) -> IterationOutcome {
        let stream = match self.start_iteration(provider).await {
            StartOutcome::Stream(s) => s,
            StartOutcome::Terminal(outcome) => return *outcome,
        };

        let response = collect_stream(stream).await;
        match response {
            Ok(resp) => self.finish_iteration(resp, registry).await,
            Err(e) => self.finish_error(e),
        }
    }

    // ── Event buffer ────────────────────────────────────────────

    /// Drain all buffered events, returning them and clearing the buffer.
    pub(crate) fn drain_events(&mut self) -> Vec<LoopEvent> {
        std::mem::take(&mut self.events)
    }

    // ── Pre-iteration guards ────────────────────────────────────

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

    // ── Post-response termination checks ────────────────────────

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

        // Max iterations (second check — covers edge cases where start_iteration
        // incremented but the check was at the boundary)
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
            config: &self.config,
        };
        if let Some(result) = handle_loop_detection(
            &mut self.loop_state,
            &snap,
            &mut self.params.messages,
            &mut self.events,
        ) {
            return Some(self.finish(result.response, result.termination_reason));
        }

        None
    }

    // ── Tool execution ──────────────────────────────────────────

    async fn execute_tools(
        &mut self,
        registry: &ToolRegistry<Ctx>,
        response: ChatResponse,
    ) -> IterationOutcome {
        let (calls, other_content) = response.partition_content();

        // Clone calls once: we need them for the outcome AND the approval pipeline.
        // The assistant message is built from refs to avoid a second clone.
        let outcome_calls = calls.clone();

        // Build the assistant message with text + tool-call blocks.
        // Anthropic (and others) require tool_use blocks in the assistant message
        // so that subsequent tool_result messages can reference them by ID.
        let mut msg_content = other_content.clone();
        msg_content.extend(calls.iter().map(|c| ContentBlock::ToolCall(c.clone())));
        self.params.messages.push(ChatMessage {
            role: crate::chat::ChatRole::Assistant,
            content: msg_content,
        });

        // Approve and execute tools (consumes owned calls)
        let (approved_calls, denied_results) = approve_calls(calls, &self.config);
        let exec_result = execute_with_events(
            registry,
            approved_calls,
            denied_results,
            self.config.parallel_tool_execution,
            &self.nested_ctx,
        )
        .await;

        self.events.extend(exec_result.events);

        let results = exec_result.results;
        self.tool_calls_executed += results.len();
        self.last_tool_results.clone_from(&results);

        // Append tool results to conversation
        for result in &results {
            self.params
                .messages
                .push(ChatMessage::tool_result_full(result.clone()));
        }

        IterationOutcome::ToolsExecuted {
            tool_calls: outcome_calls,
            results,
            assistant_content: other_content,
            iteration: self.iterations,
            total_usage: self.total_usage.clone(),
        }
    }

    // ── Terminal outcome helpers ─────────────────────────────────

    /// Mark the loop as finished and return a `Completed` outcome.
    ///
    /// Also pushes `LoopEvent::Done` into the event buffer.
    fn finish(
        &mut self,
        response: ChatResponse,
        termination_reason: TerminationReason,
    ) -> IterationOutcome {
        self.finished = true;
        let usage = self.total_usage.clone();
        let result = ToolLoopResult {
            response: response.clone(),
            iterations: self.iterations,
            total_usage: usage.clone(),
            termination_reason: termination_reason.clone(),
        };
        self.final_result = Some(result.clone());
        self.events.push(LoopEvent::Done(result));

        IterationOutcome::Completed(CompletedData {
            response,
            termination_reason,
            iterations: self.iterations,
            total_usage: usage,
        })
    }

    /// Mark the loop as finished and return an `Error` outcome.
    pub(crate) fn finish_error(&mut self, error: LlmError) -> IterationOutcome {
        self.finished = true;
        let usage = self.total_usage.clone();
        self.final_result = Some(ToolLoopResult {
            response: ChatResponse::empty(),
            iterations: self.iterations,
            total_usage: usage.clone(),
            termination_reason: TerminationReason::Complete,
        });
        IterationOutcome::Error(ErrorData {
            error,
            iterations: self.iterations,
            total_usage: usage,
        })
    }

    /// Build a terminal outcome from cached state (for repeated calls after finish).
    fn make_terminal_outcome(&self) -> IterationOutcome {
        if let Some(ref result) = self.final_result {
            IterationOutcome::Completed(CompletedData {
                response: result.response.clone(),
                termination_reason: result.termination_reason.clone(),
                iterations: result.iterations,
                total_usage: result.total_usage.clone(),
            })
        } else {
            IterationOutcome::Completed(CompletedData {
                response: ChatResponse::empty(),
                termination_reason: TerminationReason::Complete,
                iterations: self.iterations,
                total_usage: self.total_usage.clone(),
            })
        }
    }

    // ── Accessors ───────────────────────────────────────────────

    /// Set a pending command for the next iteration.
    pub(crate) fn resume(&mut self, command: LoopCommand) {
        if !self.finished {
            self.pending_command = Some(command);
        }
    }

    /// Read-only access to conversation messages.
    pub(crate) fn messages(&self) -> &[ChatMessage] {
        &self.params.messages
    }

    /// Mutable access to conversation messages.
    pub(crate) fn messages_mut(&mut self) -> &mut Vec<ChatMessage> {
        &mut self.params.messages
    }

    /// Accumulated usage across all iterations.
    pub(crate) fn total_usage(&self) -> &Usage {
        &self.total_usage
    }

    /// Current iteration count.
    pub(crate) fn iterations(&self) -> u32 {
        self.iterations
    }

    /// Whether the loop has finished.
    pub(crate) fn is_finished(&self) -> bool {
        self.finished
    }

    /// Consume the core and return a `ToolLoopResult`.
    pub(crate) fn into_result(self) -> ToolLoopResult {
        self.final_result.unwrap_or_else(|| ToolLoopResult {
            response: ChatResponse::empty(),
            iterations: self.iterations,
            total_usage: self.total_usage,
            termination_reason: TerminationReason::Complete,
        })
    }
}

// ── Stream collector ────────────────────────────────────────────────

/// Collect a `ChatStream` into a `ChatResponse`.
///
/// Consumes all events from the stream, accumulating text, tool calls,
/// and usage into a single response. Used by `do_iteration` to bridge
/// the streaming-first core to non-streaming callers.
pub(crate) async fn collect_stream(mut stream: ChatStream) -> Result<ChatResponse, LlmError> {
    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut usage = Usage::default();
    let mut stop_reason = StopReason::EndTurn;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta(t) => text.push_str(&t),
            StreamEvent::ToolCallComplete { call, .. } => tool_calls.push(call),
            StreamEvent::Usage(u) => usage += &u,
            StreamEvent::Done { stop_reason: sr } => stop_reason = sr,
            // ToolCallStart, ToolCallDelta, ReasoningDelta — not needed for response
            _ => {}
        }
    }

    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(ContentBlock::Text(text));
    }
    for call in tool_calls {
        content.push(ContentBlock::ToolCall(call));
    }

    Ok(ChatResponse {
        content,
        usage,
        stop_reason,
        model: String::new(),
        metadata: HashMap::new(),
    })
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

impl<Ctx: LoopDepth + Send + Sync + 'static> std::fmt::Debug for LoopCore<Ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopCore")
            .field("iterations", &self.iterations)
            .field("tool_calls_executed", &self.tool_calls_executed)
            .field("finished", &self.finished)
            .field("has_pending_command", &self.pending_command.is_some())
            .field("buffered_events", &self.events.len())
            .finish_non_exhaustive()
    }
}
