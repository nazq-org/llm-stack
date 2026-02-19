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
    /// Maps message index → iteration number for tool result messages.
    /// Used by observation masking to determine which results are old.
    tool_result_meta: Vec<ToolResultMeta>,
}

/// Metadata for a tool result message in the conversation.
struct ToolResultMeta {
    /// Index of this message in `params.messages`.
    message_index: usize,
    /// Iteration in which this tool result was added.
    iteration: u32,
    /// Whether this result has already been masked.
    masked: bool,
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
            tool_result_meta: Vec::new(),
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

        // Observation masking: replace old tool results with placeholders
        self.mask_old_observations();

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

        let mut results = exec_result.results;
        self.tool_calls_executed += results.len();

        // Post-process, extract, and optionally cache tool results
        self.postprocess_results(&mut results, &outcome_calls).await;

        self.last_tool_results.clone_from(&results);

        // Append tool results to conversation and record metadata for masking
        for result in &results {
            let idx = self.params.messages.len();
            self.params
                .messages
                .push(ChatMessage::tool_result_full(result.clone()));
            self.tool_result_meta.push(ToolResultMeta {
                message_index: idx,
                iteration: self.iterations,
                masked: false,
            });
        }

        IterationOutcome::ToolsExecuted {
            tool_calls: outcome_calls,
            results,
            assistant_content: other_content,
            iteration: self.iterations,
            total_usage: self.total_usage.clone(),
        }
    }

    // ── Result post-processing pipeline ────────────────────────

    /// Three-stage post-processing pipeline for tool results.
    ///
    /// 1. **Structural pruning** — sync `ToolResultProcessor`
    /// 2. **Semantic extraction** — async `ToolResultExtractor` (for large results)
    /// 3. **Cache overflow** — sync `ToolResultCacher` (for still-large results)
    async fn postprocess_results(&mut self, results: &mut [ToolResult], calls: &[ToolCall]) {
        let has_processor = self.config.result_processor.is_some();
        let has_extractor = self.config.result_extractor.is_some();
        let has_cacher = self.config.result_cacher.is_some();

        if !has_processor && !has_extractor && !has_cacher {
            return;
        }

        // Build call_id → tool_name lookup from the original calls
        let call_id_to_name: HashMap<&str, &str> = calls
            .iter()
            .map(|c| (c.id.as_str(), c.name.as_str()))
            .collect();

        // Extract the last user message for relevance-guided extraction
        let user_query: String = self
            .params
            .messages
            .iter()
            .rev()
            .find_map(|m| {
                if m.role == crate::chat::ChatRole::User {
                    m.content.iter().find_map(|b| match b {
                        ContentBlock::Text(t) => Some(t.clone()),
                        _ => None,
                    })
                } else {
                    None
                }
            })
            .unwrap_or_default();

        for result in results.iter_mut() {
            let tool_name = call_id_to_name
                .get(result.tool_call_id.as_str())
                .copied()
                .unwrap_or("unknown");

            if result.is_error {
                continue;
            }

            // Stage 1: structural pruning via processor
            if let Some(ref processor) = self.config.result_processor {
                let processed = processor.process(tool_name, &result.content);
                if processed.was_processed {
                    self.events.push(LoopEvent::ToolResultProcessed {
                        tool_name: tool_name.to_string(),
                        original_tokens: processed.original_tokens_est,
                        processed_tokens: processed.processed_tokens_est,
                    });
                    result.content = processed.content;
                }
            }

            // Stage 2: semantic extraction via async extractor
            if let Some(ref extractor) = self.config.result_extractor {
                let tokens = crate::context::estimate_tokens(&result.content);
                if tokens > extractor.extraction_threshold() {
                    if let Some(extracted) = extractor
                        .extract(tool_name, &result.content, &user_query)
                        .await
                    {
                        self.events.push(LoopEvent::ToolResultExtracted {
                            tool_name: tool_name.to_string(),
                            original_tokens: extracted.original_tokens_est,
                            extracted_tokens: extracted.extracted_tokens_est,
                        });
                        result.content = extracted.content;
                    }
                }
            }

            // Stage 3: cache overflow — store externally if still too large
            if let Some(ref cacher) = self.config.result_cacher {
                let tokens = crate::context::estimate_tokens(&result.content);
                if tokens > cacher.inline_threshold() {
                    if let Some(cached) = cacher.cache(tool_name, &result.content) {
                        self.events.push(LoopEvent::ToolResultCached {
                            tool_name: tool_name.to_string(),
                            original_tokens: cached.original_tokens_est,
                            summary_tokens: cached.summary_tokens_est,
                        });
                        result.content = cached.summary;
                    }
                }
            }
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

    // ── Observation masking ──────────────────────────────────────

    /// Replace old tool results with compact placeholders.
    ///
    /// Scans `tool_result_meta` for results from iterations older than
    /// `config.masking.max_iterations_to_keep` and larger than
    /// `config.masking.min_tokens_to_mask`, replacing their content
    /// with a one-line placeholder.
    fn mask_old_observations(&mut self) {
        let Some(masking_config) = self.config.masking else {
            return;
        };

        // Collect force-mask iterations (agent-directed via context_release)
        let force_mask = self
            .config
            .force_mask_iterations
            .as_ref()
            .and_then(|fm| fm.lock().ok())
            .map(|set| set.clone());

        // Only mask starting from iteration 3+ (need enough history),
        // unless there are force-masked iterations to process
        let has_force_masks = force_mask.as_ref().is_some_and(|s| !s.is_empty());
        if !has_force_masks && self.iterations <= masking_config.max_iterations_to_keep {
            return;
        }

        let cutoff = self
            .iterations
            .saturating_sub(masking_config.max_iterations_to_keep);
        let mut masked_count: usize = 0;
        let mut tokens_saved: u32 = 0;

        for meta in &mut self.tool_result_meta {
            if meta.masked {
                continue;
            }

            // Check: age-based OR force-masked
            let is_old = meta.iteration <= cutoff;
            let is_forced = force_mask
                .as_ref()
                .is_some_and(|s| s.contains(&meta.iteration));

            if !is_old && !is_forced {
                continue;
            }

            let msg = &self.params.messages[meta.message_index];

            // Extract the tool result content and estimate tokens
            let (tool_call_id, content, is_error) = match msg.content.first() {
                Some(ContentBlock::ToolResult(tr)) => {
                    (tr.tool_call_id.clone(), &tr.content, tr.is_error)
                }
                _ => continue,
            };

            // Don't mask errors (they're usually small and informative)
            if is_error {
                continue;
            }

            let content_tokens = crate::context::estimate_tokens(content);
            if content_tokens < masking_config.min_tokens_to_mask {
                continue;
            }

            // Build placeholder
            let placeholder = format!(
                "[Masked — tool result from iteration {iter}, ~{content_tokens} tokens. \
                 Use result_cache tool if available, or re-invoke tool.]",
                iter = meta.iteration,
            );
            let placeholder_tokens = crate::context::estimate_tokens(&placeholder);

            // Replace the message content
            self.params.messages[meta.message_index] = ChatMessage::tool_result_full(ToolResult {
                tool_call_id,
                content: placeholder,
                is_error: false,
            });

            meta.masked = true;
            masked_count += 1;
            tokens_saved += content_tokens.saturating_sub(placeholder_tokens);
        }

        if masked_count > 0 {
            self.events.push(LoopEvent::ObservationsMasked {
                masked_count,
                tokens_saved,
            });
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
