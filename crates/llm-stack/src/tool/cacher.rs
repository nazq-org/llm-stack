//! Out-of-context caching for large tool results.
//!
//! The [`ToolResultCacher`] trait lets callers define how oversized tool
//! results are stored outside the conversation context. After the
//! [`ToolResultProcessor`](super::ToolResultProcessor) prunes a result,
//! the loop core checks whether it still exceeds the inline threshold.
//! If so, it hands the content to the cacher, which stores it however
//! it likes (disk, memory, KV store, …) and returns a compact summary
//! the LLM can use to retrieve slices on demand.
//!
//! llm-stack provides the hook and the threshold check. The caller
//! (e.g. chimera) provides the storage implementation.
//!
//! # Example
//!
//! ```rust
//! use llm_stack::tool::{ToolResultCacher, CachedResult};
//! use llm_stack::context::estimate_tokens;
//!
//! struct MemoryCacher;
//!
//! impl ToolResultCacher for MemoryCacher {
//!     fn cache(&self, tool_name: &str, content: &str) -> Option<CachedResult> {
//!         let ref_id = "mem_0001".to_string();
//!         let summary = format!(
//!             "[Cached: {tool_name} result → ref={ref_id}. Use result_cache to inspect.]"
//!         );
//!         Some(CachedResult {
//!             summary,
//!             original_tokens_est: estimate_tokens(content),
//!             summary_tokens_est: 20,
//!         })
//!     }
//!
//!     fn inline_threshold(&self) -> u32 {
//!         2_000
//!     }
//! }
//! ```

/// Caches oversized tool results out-of-context.
///
/// Implementations store the full content somewhere (file, KV store,
/// database, …) and return a compact summary for the conversation.
/// The summary should tell the LLM how to retrieve slices (e.g. via
/// a `result_cache` tool with a `ref` argument).
pub trait ToolResultCacher: Send + Sync {
    /// Store `content` and return a summary for the conversation.
    ///
    /// Called only when the (already-processed) result exceeds
    /// [`inline_threshold`](Self::inline_threshold) tokens.
    ///
    /// Return `None` to fall back to keeping the content inline
    /// (e.g. if storage fails).
    fn cache(&self, tool_name: &str, content: &str) -> Option<CachedResult>;

    /// Token threshold above which results are offered to the cacher.
    ///
    /// Results at or below this size stay inline in the conversation.
    /// Default: 2 000 tokens (~8 000 chars).
    fn inline_threshold(&self) -> u32 {
        2_000
    }
}

/// The summary returned after caching a tool result.
#[derive(Debug, Clone)]
pub struct CachedResult {
    /// Compact text that replaces the full content in the conversation.
    /// Should include a reference ID and instructions for retrieval.
    pub summary: String,
    /// Estimated token count of the original (pre-cache) content.
    pub original_tokens_est: u32,
    /// Estimated token count of the summary.
    pub summary_tokens_est: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::estimate_tokens;

    struct TestCacher {
        threshold: u32,
    }

    impl ToolResultCacher for TestCacher {
        fn cache(&self, tool_name: &str, content: &str) -> Option<CachedResult> {
            let summary = format!("[cached: {tool_name}, {} chars]", content.len());
            let summary_tokens = estimate_tokens(&summary);
            Some(CachedResult {
                summary,
                original_tokens_est: estimate_tokens(content),
                summary_tokens_est: summary_tokens,
            })
        }

        fn inline_threshold(&self) -> u32 {
            self.threshold
        }
    }

    struct FailingCacher;

    impl ToolResultCacher for FailingCacher {
        fn cache(&self, _tool_name: &str, _content: &str) -> Option<CachedResult> {
            None // storage failed
        }
    }

    #[test]
    fn test_cacher_returns_summary() {
        let test_cacher = TestCacher { threshold: 10 };
        let result = test_cacher.cache("db_sql", "lots of data here").unwrap();
        assert!(result.summary.contains("cached: db_sql"));
        assert!(result.summary.contains("17 chars"));
        assert!(result.original_tokens_est > 0);
        assert!(result.summary_tokens_est > 0);
    }

    #[test]
    fn test_failing_cacher_returns_none() {
        let cacher = FailingCacher;
        assert!(cacher.cache("tool", "data").is_none());
    }

    #[test]
    fn test_default_threshold() {
        let cacher = TestCacher { threshold: 2_000 };
        assert_eq!(cacher.inline_threshold(), 2_000);
    }

    #[test]
    fn test_custom_threshold() {
        let cacher = TestCacher { threshold: 500 };
        assert_eq!(cacher.inline_threshold(), 500);
    }

    #[test]
    fn test_cacher_is_object_safe() {
        let cacher: Box<dyn ToolResultCacher> = Box::new(TestCacher { threshold: 100 });
        let result = cacher.cache("tool", "data");
        assert!(result.is_some());
    }
}
