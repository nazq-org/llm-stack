//! Context window management with token budgeting.
//!
//! This module provides [`ContextWindow`], a token-aware message buffer that
//! tracks conversation history and signals when compaction is needed.
//!
//! # Design Philosophy
//!
//! The library doesn't tokenize text (that requires model-specific tokenizers).
//! Instead:
//! - Token counts are fed from provider-reported [`Usage`](crate::Usage) after each call
//! - [`estimate_tokens`] provides a rough heuristic for pre-call estimation
//! - Compaction is the caller's responsibility — the library signals when to
//!   compact and returns messages to summarize, but summarization is an LLM
//!   call the application controls
//!
//! # Example
//!
//! ```rust
//! use llm_stack_core::context::ContextWindow;
//! use llm_stack_core::ChatMessage;
//!
//! // 8K context window, reserve 1K for output
//! let mut window = ContextWindow::new(8000, 1000);
//!
//! // Add messages with their token counts (from provider usage)
//! window.push(ChatMessage::system("You are helpful."), 10);
//! window.push(ChatMessage::user("Hello!"), 5);
//! window.push(ChatMessage::assistant("Hi there!"), 8);
//!
//! // Check available space
//! assert_eq!(window.available(), 8000 - 1000 - 10 - 5 - 8);
//!
//! // Protect recent messages from compaction
//! window.protect_recent(2);
//!
//! // Check if compaction is needed (e.g., when 80% full)
//! if window.needs_compaction(0.8) {
//!     let old_messages = window.compact();
//!     // Summarize old_messages with an LLM call, then:
//!     // window.push(ChatMessage::system("Summary: ..."), summary_tokens);
//! }
//! ```

use crate::chat::ChatMessage;

/// A token-budgeted message buffer for managing conversation context.
///
/// Tracks messages with their token counts and provides compaction signals
/// when the context approaches capacity.
#[derive(Debug)]
pub struct ContextWindow {
    /// Maximum tokens the model's context window can hold.
    max_tokens: u32,
    /// Tokens reserved for model output (not counted against input budget).
    reserved_for_output: u32,
    /// Messages with metadata.
    messages: Vec<TrackedMessage>,
}

/// A message with tracking metadata.
#[derive(Debug, Clone)]
struct TrackedMessage {
    /// The actual message.
    message: ChatMessage,
    /// Token count for this message.
    token_count: u32,
    /// Whether this message can be removed during compaction.
    compactable: bool,
}

impl ContextWindow {
    /// Creates a new context window.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum tokens the model can handle (e.g., 128000 for GPT-4)
    /// * `reserved_for_output` - Tokens to reserve for model response (e.g., 4096)
    ///
    /// # Panics
    ///
    /// Panics if `reserved_for_output >= max_tokens`.
    pub fn new(max_tokens: u32, reserved_for_output: u32) -> Self {
        assert!(
            reserved_for_output < max_tokens,
            "reserved_for_output ({reserved_for_output}) must be less than max_tokens ({max_tokens})"
        );
        Self {
            max_tokens,
            reserved_for_output,
            messages: Vec::new(),
        }
    }

    /// Adds a message with its token count.
    ///
    /// New messages are compactable by default. Use [`protect_recent`](Self::protect_recent)
    /// to mark recent messages as non-compactable.
    ///
    /// # Arguments
    ///
    /// * `message` - The chat message to add
    /// * `tokens` - Token count for this message (from provider usage or estimation)
    pub fn push(&mut self, message: ChatMessage, tokens: u32) {
        self.messages.push(TrackedMessage {
            message,
            token_count: tokens,
            compactable: true,
        });
    }

    /// Returns the number of tokens available for new content.
    ///
    /// This is `max_tokens - reserved_for_output - total_tokens()`.
    pub fn available(&self) -> u32 {
        let input_budget = self.max_tokens.saturating_sub(self.reserved_for_output);
        input_budget.saturating_sub(self.total_tokens())
    }

    /// Returns an iterator over the current messages.
    ///
    /// Prefer this over [`messages`](Self::messages) to avoid allocation.
    pub fn iter(&self) -> impl Iterator<Item = &ChatMessage> {
        self.messages.iter().map(|t| &t.message)
    }

    /// Returns the current messages as a vector of references.
    ///
    /// For iteration without allocation, use [`iter`](Self::iter) instead.
    pub fn messages(&self) -> Vec<&ChatMessage> {
        self.messages.iter().map(|t| &t.message).collect()
    }

    /// Returns owned copies of the current messages.
    ///
    /// Use this when you need to pass messages to a provider that takes ownership.
    pub fn messages_owned(&self) -> Vec<ChatMessage> {
        self.messages.iter().map(|t| t.message.clone()).collect()
    }

    /// Returns the total tokens currently in the window.
    pub fn total_tokens(&self) -> u32 {
        self.messages
            .iter()
            .map(|t| t.token_count)
            .fold(0, u32::saturating_add)
    }

    /// Returns the number of messages in the window.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns true if the window contains no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Checks if compaction is needed based on a threshold.
    ///
    /// Returns `true` if the window is more than `threshold` percent full.
    ///
    /// # Arguments
    ///
    /// * `threshold` - A value between 0.0 and 1.0 (e.g., 0.8 for 80%)
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_stack_core::context::ContextWindow;
    /// use llm_stack_core::ChatMessage;
    ///
    /// let mut window = ContextWindow::new(1000, 200);
    /// window.push(ChatMessage::user("Hello"), 700);
    ///
    /// // 700 / (1000 - 200) = 87.5% full
    /// assert!(window.needs_compaction(0.8));
    /// assert!(!window.needs_compaction(0.9));
    /// ```
    #[allow(clippy::cast_precision_loss)]
    pub fn needs_compaction(&self, threshold: f32) -> bool {
        let input_budget = self.max_tokens.saturating_sub(self.reserved_for_output);
        if input_budget == 0 {
            return false;
        }
        // Precision loss is acceptable — f32 is precise to ~16M tokens,
        // far exceeding practical context windows (2025 models cap ~2M).
        let usage_ratio = self.total_tokens() as f32 / input_budget as f32;
        usage_ratio > threshold
    }

    /// Removes and returns compactable messages.
    ///
    /// Messages marked as non-compactable (via [`protect_recent`](Self::protect_recent)
    /// or system messages) are retained. Returns the removed messages so the
    /// caller can summarize them.
    ///
    /// # Returns
    ///
    /// A vector of removed messages, in their original order.
    pub fn compact(&mut self) -> Vec<ChatMessage> {
        let mut removed = Vec::new();
        let mut retained = Vec::new();

        for tracked in self.messages.drain(..) {
            if tracked.compactable {
                removed.push(tracked.message);
            } else {
                retained.push(tracked);
            }
        }

        self.messages = retained;
        removed
    }

    /// Marks the most recent `n` messages as non-compactable.
    ///
    /// This protects recent context from being removed during compaction.
    /// Call this after adding messages that should be preserved.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of recent messages to protect (from the end). If `n`
    ///   exceeds the window length, all messages are protected.
    pub fn protect_recent(&mut self, n: usize) {
        let len = self.messages.len();
        let start = len.saturating_sub(n);
        for msg in &mut self.messages[start..] {
            msg.compactable = false;
        }
    }

    /// Marks a message at the given index as non-compactable.
    ///
    /// Useful for protecting specific messages like system prompts.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len()`.
    pub fn protect(&mut self, index: usize) {
        self.messages[index].compactable = false;
    }

    /// Marks a message at the given index as compactable.
    ///
    /// Reverses the effect of [`protect`](Self::protect).
    ///
    /// # Panics
    ///
    /// Panics if `index >= len()`.
    pub fn unprotect(&mut self, index: usize) {
        self.messages[index].compactable = true;
    }

    /// Returns whether the message at `index` is protected from compaction.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len()`.
    pub fn is_protected(&self, index: usize) -> bool {
        !self.messages[index].compactable
    }

    /// Returns the input budget (`max_tokens - reserved_for_output`).
    pub fn input_budget(&self) -> u32 {
        self.max_tokens.saturating_sub(self.reserved_for_output)
    }

    /// Returns the maximum tokens this window was configured with.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Returns the tokens reserved for output.
    pub fn reserved_for_output(&self) -> u32 {
        self.reserved_for_output
    }

    /// Clears all messages from the window.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Returns the token count for the message at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len()`.
    pub fn token_count(&self, index: usize) -> u32 {
        self.messages[index].token_count
    }

    /// Updates the token count for the message at the given index.
    ///
    /// Useful when you get accurate token counts from the provider after
    /// initially using estimates.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len()`.
    pub fn update_token_count(&mut self, index: usize, tokens: u32) {
        self.messages[index].token_count = tokens;
    }
}

// ── Token estimation ────────────────────────────────────────────────

/// Estimates the token count for a string.
///
/// This is a rough heuristic (approximately 4 characters per token for English).
/// Use provider-reported token counts when available for accuracy.
///
/// # Arguments
///
/// * `text` - The text to estimate tokens for
///
/// # Returns
///
/// Estimated token count (always at least 1 for non-empty text).
#[allow(clippy::cast_possible_truncation)]
pub fn estimate_tokens(text: &str) -> u32 {
    if text.is_empty() {
        return 0;
    }
    // Rough heuristic: ~4 chars per token for English
    // Clamp to u32::MAX for very long strings (unlikely in practice)
    let len = text.len().min(u32::MAX as usize) as u32;
    len.div_ceil(4).max(1)
}

/// Estimates tokens for a chat message.
///
/// Includes a small overhead for role markers and message structure.
pub fn estimate_message_tokens(message: &ChatMessage) -> u32 {
    use crate::chat::ContentBlock;

    let content_tokens: u32 = message
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Text(text) => estimate_tokens(text),
            // Image tokens vary widely by model and resolution (85 = low-res baseline)
            ContentBlock::Image { .. } => 85,
            ContentBlock::ToolCall(tc) => {
                estimate_tokens(&tc.name) + estimate_tokens(&tc.arguments.to_string())
            }
            // Tool result content + ~10 tokens for structure (tool_call_id, is_error, etc.)
            ContentBlock::ToolResult(tr) => estimate_tokens(&tr.content) + 10,
            ContentBlock::Reasoning { content } => estimate_tokens(content),
        })
        .sum();

    // Add overhead for message structure (role, separators, etc.)
    content_tokens + 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ChatRole;

    fn user_msg(text: &str) -> ChatMessage {
        ChatMessage::user(text)
    }

    fn assistant_msg(text: &str) -> ChatMessage {
        ChatMessage::assistant(text)
    }

    fn system_msg(text: &str) -> ChatMessage {
        ChatMessage::system(text)
    }

    // ── ContextWindow basic tests ──────────────────────────────────

    #[test]
    fn test_new_context_window() {
        let window = ContextWindow::new(8000, 1000);
        assert_eq!(window.max_tokens(), 8000);
        assert_eq!(window.reserved_for_output(), 1000);
        assert_eq!(window.input_budget(), 7000);
        assert!(window.is_empty());
        assert_eq!(window.len(), 0);
    }

    #[test]
    #[should_panic(expected = "reserved_for_output")]
    fn test_new_invalid_reserved() {
        ContextWindow::new(1000, 1000);
    }

    #[test]
    #[should_panic(expected = "reserved_for_output")]
    fn test_new_reserved_exceeds_max() {
        ContextWindow::new(1000, 2000);
    }

    #[test]
    fn test_push_and_len() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("Hello"), 10);
        window.push(assistant_msg("Hi"), 8);

        assert_eq!(window.len(), 2);
        assert!(!window.is_empty());
    }

    #[test]
    fn test_total_tokens() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("Hello"), 10);
        window.push(assistant_msg("Hi"), 8);
        window.push(user_msg("How are you?"), 15);

        assert_eq!(window.total_tokens(), 33);
    }

    #[test]
    fn test_available_tokens() {
        let mut window = ContextWindow::new(8000, 1000);
        // Input budget = 8000 - 1000 = 7000
        assert_eq!(window.available(), 7000);

        window.push(user_msg("Hello"), 100);
        assert_eq!(window.available(), 6900);

        window.push(assistant_msg("Hi"), 50);
        assert_eq!(window.available(), 6850);
    }

    #[test]
    fn test_available_saturates() {
        let mut window = ContextWindow::new(1000, 100);
        // Input budget = 900
        window.push(user_msg("Large message"), 1000);
        // Would be negative, but saturates to 0
        assert_eq!(window.available(), 0);
    }

    #[test]
    fn test_messages() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("Hello"), 10);
        window.push(assistant_msg("Hi"), 8);

        let messages = window.messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, ChatRole::User);
        assert_eq!(messages[1].role, ChatRole::Assistant);
    }

    #[test]
    fn test_messages_owned() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("Hello"), 10);

        let messages = window.messages_owned();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, ChatRole::User);
    }

    // ── Compaction tests ───────────────────────────────────────────

    #[test]
    fn test_needs_compaction_below_threshold() {
        let mut window = ContextWindow::new(1000, 200);
        // Input budget = 800
        window.push(user_msg("Hello"), 400);
        // 400/800 = 50%, below 80% threshold
        assert!(!window.needs_compaction(0.8));
    }

    #[test]
    fn test_needs_compaction_above_threshold() {
        let mut window = ContextWindow::new(1000, 200);
        // Input budget = 800
        window.push(user_msg("Hello"), 700);
        // 700/800 = 87.5%, above 80% threshold
        assert!(window.needs_compaction(0.8));
    }

    #[test]
    fn test_needs_compaction_at_threshold() {
        let mut window = ContextWindow::new(1000, 200);
        // Input budget = 800
        window.push(user_msg("Hello"), 640);
        // 640/800 = 80%, not above threshold
        assert!(!window.needs_compaction(0.8));
    }

    #[test]
    fn test_needs_compaction_zero_budget() {
        let window = ContextWindow::new(100, 99);
        // Input budget = 1
        // Edge case: very small budget
        assert!(!window.needs_compaction(0.8));
    }

    #[test]
    fn test_compact_all_compactable() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("Hello"), 10);
        window.push(assistant_msg("Hi"), 8);
        window.push(user_msg("Bye"), 5);

        let removed = window.compact();

        assert_eq!(removed.len(), 3);
        assert!(window.is_empty());
        assert_eq!(window.total_tokens(), 0);
    }

    #[test]
    fn test_compact_with_protected() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(system_msg("System"), 20);
        window.push(user_msg("Hello"), 10);
        window.push(assistant_msg("Hi"), 8);
        window.push(user_msg("Question"), 15);

        // Protect system message and last 2 messages
        window.protect(0);
        window.protect_recent(2);

        let removed = window.compact();

        // Only the second message (user "Hello") should be removed
        assert_eq!(removed.len(), 1);
        assert_eq!(window.len(), 3);
        assert_eq!(window.total_tokens(), 20 + 8 + 15);
    }

    #[test]
    fn test_compact_none_compactable() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(system_msg("System"), 20);
        window.push(user_msg("Hello"), 10);

        // Protect all
        window.protect_recent(2);

        let removed = window.compact();

        assert!(removed.is_empty());
        assert_eq!(window.len(), 2);
    }

    #[test]
    fn test_protect_recent() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("1"), 10);
        window.push(user_msg("2"), 10);
        window.push(user_msg("3"), 10);
        window.push(user_msg("4"), 10);

        window.protect_recent(2);

        let removed = window.compact();

        // Messages 1 and 2 removed, 3 and 4 protected
        assert_eq!(removed.len(), 2);
        assert_eq!(window.len(), 2);
    }

    #[test]
    fn test_protect_recent_more_than_len() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("1"), 10);
        window.push(user_msg("2"), 10);

        window.protect_recent(10); // More than we have

        let removed = window.compact();

        assert!(removed.is_empty());
        assert_eq!(window.len(), 2);
    }

    #[test]
    fn test_protect_index() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("1"), 10);
        window.push(user_msg("2"), 10);
        window.push(user_msg("3"), 10);

        window.protect(1); // Protect middle message

        let removed = window.compact();

        assert_eq!(removed.len(), 2);
        assert_eq!(window.len(), 1);
    }

    #[test]
    fn test_unprotect() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("1"), 10);
        window.push(user_msg("2"), 10);

        window.protect(0);
        assert!(window.is_protected(0));

        window.unprotect(0);
        assert!(!window.is_protected(0));

        let removed = window.compact();
        assert_eq!(removed.len(), 2);
    }

    #[test]
    fn test_is_protected() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("1"), 10);
        window.push(user_msg("2"), 10);

        // New messages are not protected by default
        assert!(!window.is_protected(0));
        assert!(!window.is_protected(1));

        window.protect(0);
        assert!(window.is_protected(0));
        assert!(!window.is_protected(1));
    }

    #[test]
    fn test_iter() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("Hello"), 10);
        window.push(assistant_msg("Hi"), 8);

        let collected: Vec<_> = window.iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].role, ChatRole::User);
        assert_eq!(collected[1].role, ChatRole::Assistant);
    }

    // ── Token count management tests ───────────────────────────────

    #[test]
    fn test_token_count() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("Hello"), 42);

        assert_eq!(window.token_count(0), 42);
    }

    #[test]
    fn test_update_token_count() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("Hello"), 10);

        assert_eq!(window.total_tokens(), 10);

        window.update_token_count(0, 15);

        assert_eq!(window.token_count(0), 15);
        assert_eq!(window.total_tokens(), 15);
    }

    #[test]
    fn test_clear() {
        let mut window = ContextWindow::new(8000, 1000);
        window.push(user_msg("Hello"), 10);
        window.push(assistant_msg("Hi"), 8);

        window.clear();

        assert!(window.is_empty());
        assert_eq!(window.total_tokens(), 0);
        assert_eq!(window.available(), 7000);
    }

    // ── Token estimation tests ─────────────────────────────────────

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_short() {
        // "Hi" = 2 chars, (2 + 3) / 4 = 1
        assert_eq!(estimate_tokens("Hi"), 1);
    }

    #[test]
    fn test_estimate_tokens_medium() {
        // "Hello world" = 11 chars, (11 + 3) / 4 = 3
        assert_eq!(estimate_tokens("Hello world"), 3);
    }

    #[test]
    fn test_estimate_tokens_exact_multiple() {
        // 16 chars, (16 + 3) / 4 = 4
        assert_eq!(estimate_tokens("1234567890123456"), 4);
    }

    #[test]
    fn test_estimate_tokens_minimum() {
        // Single char should be at least 1 token
        assert_eq!(estimate_tokens("a"), 1);
    }

    #[test]
    fn test_estimate_message_tokens() {
        let msg = user_msg("Hello world");
        let estimate = estimate_message_tokens(&msg);
        // Content: 3 tokens + 4 overhead = 7
        assert_eq!(estimate, 7);
    }

    #[test]
    fn test_estimate_message_tokens_empty() {
        let msg = ChatMessage {
            role: ChatRole::User,
            content: vec![],
        };
        let estimate = estimate_message_tokens(&msg);
        // No content, just 4 overhead
        assert_eq!(estimate, 4);
    }

    // ── Debug and trait tests ──────────────────────────────────────

    #[test]
    fn test_context_window_debug() {
        let window = ContextWindow::new(8000, 1000);
        let debug = format!("{window:?}");
        assert!(debug.contains("ContextWindow"));
        assert!(debug.contains("8000"));
    }

    #[test]
    fn test_context_window_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ContextWindow>();
    }

    // ── Integration-style tests ────────────────────────────────────

    #[test]
    fn test_typical_conversation_flow() {
        let mut window = ContextWindow::new(4000, 500);
        // Input budget = 3500

        // System message (always protected)
        window.push(system_msg("You are a helpful assistant."), 15);
        window.protect(0);

        // Conversation
        window.push(user_msg("What is 2+2?"), 20);
        window.push(assistant_msg("2+2 equals 4."), 25);
        window.push(user_msg("What about 3+3?"), 22);
        window.push(assistant_msg("3+3 equals 6."), 25);

        assert_eq!(window.len(), 5);
        assert_eq!(window.total_tokens(), 107);
        assert_eq!(window.available(), 3500 - 107);

        // Not yet at compaction threshold
        assert!(!window.needs_compaction(0.8));

        // Add more messages to approach threshold
        for i in 0..50 {
            window.push(user_msg(&format!("Question {i}")), 30);
            window.push(assistant_msg(&format!("Answer {i}")), 30);
        }

        // Now should need compaction
        assert!(window.needs_compaction(0.8));

        // Protect recent context
        window.protect_recent(4);

        // Compact
        let removed = window.compact();

        // Should have removed old messages but kept system and recent 4
        assert!(!removed.is_empty());
        assert!(window.len() <= 5); // system + 4 recent
        assert!(window.messages()[0].role == ChatRole::System);
    }

    #[test]
    fn test_compact_then_add_summary() {
        let mut window = ContextWindow::new(1000, 100);
        // Input budget = 900

        window.push(system_msg("System"), 20);
        window.protect(0);

        // Fill with messages
        for _ in 0..10 {
            window.push(user_msg("Message"), 80);
        }

        // Compact
        let removed = window.compact();
        assert_eq!(removed.len(), 10);
        assert_eq!(window.len(), 1); // Only system remains

        // Add a summary message
        window.push(
            ChatMessage::system("Summary of previous conversation..."),
            50,
        );

        assert_eq!(window.len(), 2);
        assert_eq!(window.total_tokens(), 70);
    }
}
