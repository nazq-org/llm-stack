//! Post-execution processing of tool results.
//!
//! The [`ToolResultProcessor`] trait allows callers to transform tool
//! output **after** execution but **before** it enters the conversation
//! context. Common uses:
//!
//! - Structural pruning (strip HTML, truncate large results)
//! - Token budget enforcement (cap results at N tokens)
//! - Format normalization (convert tables to markdown)
//!
//! # Example
//!
//! ```rust
//! use llm_stack::tool::{ToolResultProcessor, ProcessedResult};
//! use llm_stack::context::estimate_tokens;
//!
//! struct TruncateProcessor {
//!     max_chars: usize,
//! }
//!
//! impl ToolResultProcessor for TruncateProcessor {
//!     fn process(&self, _tool_name: &str, output: &str) -> ProcessedResult {
//!         if output.len() <= self.max_chars {
//!             return ProcessedResult::unchanged();
//!         }
//!         let truncated = &output[..self.max_chars];
//!         ProcessedResult {
//!             content: format!("{truncated}\n[truncated — original was {} chars]", output.len()),
//!             was_processed: true,
//!             original_tokens_est: estimate_tokens(output),
//!             processed_tokens_est: estimate_tokens(truncated) + 10,
//!         }
//!     }
//! }
//! ```

/// Processes tool results before they enter the conversation context.
///
/// Implementations receive the tool name (for per-tool dispatch) and the
/// raw output string. They return a [`ProcessedResult`] indicating whether
/// the content was modified and providing token estimates for observability.
///
/// The processor runs synchronously. For heavyweight transformations (e.g.,
/// calling another LLM for semantic extraction), consider doing the work
/// inside the tool handler itself and using the processor only for
/// structural operations.
pub trait ToolResultProcessor: Send + Sync {
    /// Process a tool result, optionally transforming its content.
    ///
    /// # Arguments
    ///
    /// * `tool_name` — The name of the tool that produced this result.
    /// * `output` — The raw output string from tool execution.
    ///
    /// Return [`ProcessedResult::unchanged()`] to pass through unmodified.
    fn process(&self, tool_name: &str, output: &str) -> ProcessedResult;
}

/// The result of processing a tool's output.
#[derive(Debug, Clone)]
pub struct ProcessedResult {
    /// The (possibly transformed) content to use in the conversation.
    pub content: String,
    /// Whether the content was modified by the processor.
    pub was_processed: bool,
    /// Estimated token count of the original output.
    pub original_tokens_est: u32,
    /// Estimated token count of the processed output.
    pub processed_tokens_est: u32,
}

impl ProcessedResult {
    /// Create a result indicating no processing was performed.
    ///
    /// The content field is left empty — callers should use the original
    /// output when `was_processed` is false.
    pub fn unchanged() -> Self {
        Self {
            content: String::new(),
            was_processed: false,
            original_tokens_est: 0,
            processed_tokens_est: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct UpperCaseProcessor;

    impl ToolResultProcessor for UpperCaseProcessor {
        fn process(&self, _tool_name: &str, output: &str) -> ProcessedResult {
            ProcessedResult {
                content: output.to_uppercase(),
                was_processed: true,
                original_tokens_est: crate::context::estimate_tokens(output),
                processed_tokens_est: crate::context::estimate_tokens(output),
            }
        }
    }

    struct SelectiveProcessor;

    impl ToolResultProcessor for SelectiveProcessor {
        fn process(&self, tool_name: &str, output: &str) -> ProcessedResult {
            if tool_name == "web_search" && output.len() > 100 {
                ProcessedResult {
                    content: output[..100].to_string(),
                    was_processed: true,
                    original_tokens_est: crate::context::estimate_tokens(output),
                    processed_tokens_est: 25,
                }
            } else {
                ProcessedResult::unchanged()
            }
        }
    }

    #[test]
    fn test_unchanged_result() {
        let result = ProcessedResult::unchanged();
        assert!(!result.was_processed);
        assert!(result.content.is_empty());
        assert_eq!(result.original_tokens_est, 0);
        assert_eq!(result.processed_tokens_est, 0);
    }

    #[test]
    fn test_processor_transforms_output() {
        let processor = UpperCaseProcessor;
        let result = processor.process("any_tool", "hello world");
        assert!(result.was_processed);
        assert_eq!(result.content, "HELLO WORLD");
    }

    #[test]
    fn test_selective_processor_skips_non_matching() {
        let processor = SelectiveProcessor;
        let result = processor.process("calculator", "42");
        assert!(!result.was_processed);
    }

    #[test]
    fn test_selective_processor_truncates_matching() {
        let processor = SelectiveProcessor;
        let long_output = "x".repeat(200);
        let result = processor.process("web_search", &long_output);
        assert!(result.was_processed);
        assert_eq!(result.content.len(), 100);
    }

    #[test]
    fn test_selective_processor_skips_short_matching() {
        let processor = SelectiveProcessor;
        let result = processor.process("web_search", "short");
        assert!(!result.was_processed);
    }

    #[test]
    fn test_processor_is_object_safe() {
        // Verify the trait works as a trait object
        let processor: Box<dyn ToolResultProcessor> = Box::new(UpperCaseProcessor);
        let result = processor.process("tool", "test");
        assert_eq!(result.content, "TEST");
    }

    #[test]
    fn test_processor_token_estimates() {
        let processor = UpperCaseProcessor;
        let result = processor.process("tool", "Hello world!!");
        // 13 chars → ceil(13/4) = 4 tokens
        assert_eq!(result.original_tokens_est, 4);
        assert_eq!(result.processed_tokens_est, 4);
    }
}
