//! Async semantic extraction for large tool results.
//!
//! The [`ToolResultExtractor`] trait provides an async post-processing stage
//! that runs **after** structural pruning ([`ToolResultProcessor`]) but
//! **before** out-of-context caching ([`ToolResultCacher`]).
//!
//! Use this for heavyweight transformations that require async work, such as
//! calling a fast/cheap LLM (Haiku-class) to extract task-relevant information
//! from large tool results.
//!
//! The extractor receives the tool name, the (already-pruned) output, and the
//! last user message for relevance-guided extraction. It returns an
//! [`ExtractedResult`] with the condensed content.
//!
//! # Pipeline position
//!
//! ```text
//!   Tool executes
//!        │
//!   Stage 1: ToolResultProcessor::process()   (sync, structural)
//!        │
//!   Stage 2: ToolResultExtractor::extract()   (async, semantic)
//!        │
//!   Stage 3: ToolResultCacher::cache()        (sync, overflow)
//!        │
//!   Result enters conversation context
//! ```
//!
//! # Example
//!
//! ```rust
//! use llm_stack::tool::{ToolResultExtractor, ExtractedResult};
//! use llm_stack::context::estimate_tokens;
//! use std::future::Future;
//! use std::pin::Pin;
//!
//! struct KeywordExtractor;
//!
//! impl ToolResultExtractor for KeywordExtractor {
//!     fn extract<'a>(
//!         &'a self,
//!         tool_name: &'a str,
//!         output: &'a str,
//!         user_query: &'a str,
//!     ) -> Pin<Box<dyn Future<Output = Option<ExtractedResult>> + Send + 'a>> {
//!         Box::pin(async move {
//!             if tool_name != "web_search" || output.len() < 10_000 {
//!                 return None;
//!             }
//!             let extracted = format!("Extracted relevant info about: {user_query}");
//!             Some(ExtractedResult {
//!                 content: extracted.clone(),
//!                 original_tokens_est: estimate_tokens(output),
//!                 extracted_tokens_est: estimate_tokens(&extracted),
//!             })
//!         })
//!     }
//!
//!     fn extraction_threshold(&self) -> u32 {
//!         15_000
//!     }
//! }
//! ```

use std::future::Future;
use std::pin::Pin;

/// Async semantic extractor for oversized tool results.
///
/// Implementations run after structural pruning and can perform heavyweight
/// async work (e.g., calling a fast LLM) to condense large results into
/// task-relevant summaries.
///
/// The extractor receives the last user message as context to guide
/// relevance-based extraction.
pub trait ToolResultExtractor: Send + Sync {
    /// Extract task-relevant information from a tool result.
    ///
    /// # Arguments
    ///
    /// * `tool_name` — The name of the tool that produced this result.
    /// * `output` — The (already structurally pruned) output string.
    /// * `user_query` — The most recent user message, for relevance guidance.
    ///
    /// Return `None` to skip extraction (keep the structurally-pruned content).
    /// The extractor is only called for results exceeding
    /// [`extraction_threshold`](Self::extraction_threshold) tokens.
    fn extract<'a>(
        &'a self,
        tool_name: &'a str,
        output: &'a str,
        user_query: &'a str,
    ) -> Pin<Box<dyn Future<Output = Option<ExtractedResult>> + Send + 'a>>;

    /// Token threshold above which results are offered to the extractor.
    ///
    /// Results at or below this size skip semantic extraction entirely.
    /// Default: 15 000 tokens (~60 000 chars).
    fn extraction_threshold(&self) -> u32 {
        15_000
    }
}

/// The result of semantic extraction.
#[derive(Debug, Clone)]
pub struct ExtractedResult {
    /// The condensed, task-relevant content.
    pub content: String,
    /// Estimated token count of the pre-extraction content.
    pub original_tokens_est: u32,
    /// Estimated token count of the extracted content.
    pub extracted_tokens_est: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::estimate_tokens;

    struct NoopExtractor;

    impl ToolResultExtractor for NoopExtractor {
        fn extract<'a>(
            &'a self,
            _tool_name: &'a str,
            _output: &'a str,
            _user_query: &'a str,
        ) -> Pin<Box<dyn Future<Output = Option<ExtractedResult>> + Send + 'a>> {
            Box::pin(async { None })
        }
    }

    struct TestExtractor {
        threshold: u32,
    }

    impl ToolResultExtractor for TestExtractor {
        fn extract<'a>(
            &'a self,
            tool_name: &'a str,
            output: &'a str,
            user_query: &'a str,
        ) -> Pin<Box<dyn Future<Output = Option<ExtractedResult>> + Send + 'a>> {
            Box::pin(async move {
                let extracted = format!(
                    "[Extracted from {tool_name} for query: {user_query}] \
                     Summary of {} chars",
                    output.len()
                );
                Some(ExtractedResult {
                    content: extracted.clone(),
                    original_tokens_est: estimate_tokens(output),
                    extracted_tokens_est: estimate_tokens(&extracted),
                })
            })
        }

        fn extraction_threshold(&self) -> u32 {
            self.threshold
        }
    }

    #[test]
    fn test_extracted_result_debug_clone() {
        let result = ExtractedResult {
            content: "test".into(),
            original_tokens_est: 100,
            extracted_tokens_est: 10,
        };
        let cloned = result.clone();
        assert_eq!(cloned.content, "test");
        assert_eq!(format!("{result:?}").len(), format!("{cloned:?}").len());
    }

    #[test]
    fn test_default_threshold() {
        let extractor = NoopExtractor;
        assert_eq!(extractor.extraction_threshold(), 15_000);
    }

    #[test]
    fn test_custom_threshold() {
        let extractor = TestExtractor { threshold: 5_000 };
        assert_eq!(extractor.extraction_threshold(), 5_000);
    }

    #[tokio::test]
    async fn test_noop_extractor_returns_none() {
        let extractor = NoopExtractor;
        let result = extractor.extract("web_search", "content", "query").await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_extractor_returns_condensed_content() {
        let extractor = TestExtractor { threshold: 10 };
        let output = "a".repeat(1000);
        let result = extractor
            .extract("web_search", &output, "weather in Tybee")
            .await;
        assert!(result.is_some());
        let extracted = result.unwrap();
        assert!(extracted.content.contains("web_search"));
        assert!(extracted.content.contains("weather in Tybee"));
        assert!(extracted.extracted_tokens_est < extracted.original_tokens_est);
    }

    #[test]
    fn test_extractor_is_object_safe() {
        let extractor: Box<dyn ToolResultExtractor> = Box::new(NoopExtractor);
        assert_eq!(extractor.extraction_threshold(), 15_000);
    }

    #[tokio::test]
    async fn test_extractor_object_safe_extract() {
        let extractor: Box<dyn ToolResultExtractor> = Box::new(TestExtractor { threshold: 100 });
        let result = extractor.extract("tool", "data", "query").await;
        assert!(result.is_some());
    }
}
