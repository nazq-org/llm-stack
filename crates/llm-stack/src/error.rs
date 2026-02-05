//! Unified error type for all LLM operations.
//!
//! Every provider maps its native errors into [`LlmError`], giving
//! callers a single type to match against regardless of which backend
//! is in use. Variants carry enough context for retry logic, user-facing
//! messages, and diagnostics.
//!
//! # Retryability
//!
//! Several variants include a `retryable` flag that providers set based
//! on the upstream response (e.g. HTTP 429 or 503). Middleware layers
//! can inspect this flag to decide whether to retry automatically:
//!
//! ```rust
//! use llm_stack::LlmError;
//!
//! fn should_retry(err: &LlmError) -> bool {
//!     match err {
//!         LlmError::Http { retryable, .. } => *retryable,
//!         LlmError::Provider { retryable, .. } => *retryable,
//!         LlmError::Timeout { .. } => true,
//!         _ => false,
//!     }
//! }
//! ```

use serde_json::Value;

/// The unified error type returned by all provider operations.
///
/// Variants are `#[non_exhaustive]` — new error kinds may be added in
/// minor releases without breaking downstream matches (always include a
/// wildcard arm).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LlmError {
    /// An HTTP-level failure (transport error, unexpected status code).
    ///
    /// `status` is `None` when the request never received a response
    /// (e.g. DNS failure, connection reset).
    #[error("HTTP error (status={status:?}): {message}")]
    Http {
        /// The HTTP status code, if one was received.
        status: Option<http::StatusCode>,
        /// A human-readable description of the failure.
        message: String,
        /// Whether the caller should retry this request.
        retryable: bool,
    },

    /// The API key or token was rejected.
    #[error("Authentication error: {0}")]
    Auth(String),

    /// The request was malformed (missing fields, invalid parameters).
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// A provider-specific error that doesn't map to another variant.
    #[error("Provider error ({code}): {message}")]
    Provider {
        /// Provider-defined error code (e.g. `"overloaded"`).
        code: String,
        /// Human-readable error description.
        message: String,
        /// Whether the caller should retry this request.
        retryable: bool,
    },

    /// The response body could not be parsed.
    #[error("Response format error: {message}")]
    ResponseFormat {
        /// What went wrong during parsing.
        message: String,
        /// The raw response body, for diagnostics.
        raw: String,
    },

    /// A structured-output response failed JSON Schema validation.
    #[error("Schema validation error: {message}")]
    SchemaValidation {
        /// Concatenated validation error messages.
        message: String,
        /// The schema the value was validated against.
        schema: Value,
        /// The value that failed validation.
        actual: Value,
    },

    /// A tool invocation raised an error.
    #[error("Tool execution error ({tool_name}): {source}")]
    ToolExecution {
        /// The name of the tool that failed.
        tool_name: String,
        /// The underlying error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A retry policy exhausted its budget without a successful response.
    #[error("Retry exhausted after {attempts} attempts: {last_error}")]
    RetryExhausted {
        /// How many attempts were made.
        attempts: u32,
        /// The error from the final attempt.
        #[source]
        last_error: Box<LlmError>,
    },

    /// The operation exceeded its deadline.
    #[error("Operation timed out after {elapsed_ms}ms")]
    Timeout {
        /// Milliseconds elapsed before the timeout fired.
        elapsed_ms: u64,
    },

    /// A nested tool loop exceeded the maximum allowed depth.
    ///
    /// This occurs when `tool_loop` is called recursively (e.g., a tool
    /// spawning a sub-agent) and the nesting depth exceeds `max_depth`
    /// in [`ToolLoopConfig`](crate::tool::ToolLoopConfig).
    #[error("max nesting depth exceeded (current: {current}, limit: {limit})")]
    MaxDepthExceeded {
        /// The depth at which the error was raised.
        current: u32,
        /// The configured maximum depth.
        limit: u32,
    },
}

impl LlmError {
    /// Returns `true` if the error is transient and the request may succeed on retry.
    ///
    /// Useful for retry interceptors. This checks the `retryable` flag
    /// on applicable variants and treats timeouts as always retryable.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_stack::LlmError;
    ///
    /// let err = LlmError::Timeout { elapsed_ms: 5000 };
    /// assert!(err.is_retryable());
    ///
    /// let err = LlmError::Auth("bad key".into());
    /// assert!(!err.is_retryable());
    /// ```
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Http { retryable, .. } | Self::Provider { retryable, .. } => *retryable,
            Self::Timeout { .. } => true,
            _ => false,
        }
    }
}

impl From<serde_json::Error> for LlmError {
    fn from(err: serde_json::Error) -> Self {
        Self::ResponseFormat {
            message: err.to_string(),
            raw: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_http() {
        let err = LlmError::Http {
            status: Some(http::StatusCode::TOO_MANY_REQUESTS),
            message: "rate limited".into(),
            retryable: true,
        };
        let display = format!("{err}");
        assert!(display.contains("429"));
        assert!(display.contains("rate limited"));
    }

    #[test]
    fn test_error_display_auth() {
        let err = LlmError::Auth("bad key".into());
        assert!(format!("{err}").contains("bad key"));
    }

    #[test]
    fn test_error_display_invalid_request() {
        let err = LlmError::InvalidRequest("missing model".into());
        assert!(format!("{err}").contains("missing model"));
    }

    #[test]
    fn test_error_display_provider() {
        let err = LlmError::Provider {
            code: "overloaded".into(),
            message: "server busy".into(),
            retryable: true,
        };
        let display = format!("{err}");
        assert!(display.contains("overloaded"));
        assert!(display.contains("server busy"));
    }

    #[test]
    fn test_error_display_response_format() {
        let err = LlmError::ResponseFormat {
            message: "not json".into(),
            raw: "hello".into(),
        };
        assert!(format!("{err}").contains("not json"));
    }

    #[test]
    fn test_error_display_schema_validation() {
        let err = LlmError::SchemaValidation {
            message: "missing field".into(),
            schema: serde_json::json!({"type": "object"}),
            actual: serde_json::json!({}),
        };
        assert!(format!("{err}").contains("missing field"));
    }

    #[test]
    fn test_error_display_tool_execution() {
        let err = LlmError::ToolExecution {
            tool_name: "calculator".into(),
            source: Box::new(std::io::Error::other("boom")),
        };
        let display = format!("{err}");
        assert!(display.contains("calculator"));
        assert!(display.contains("boom"));
    }

    #[test]
    fn test_error_display_retry_exhausted() {
        let inner = LlmError::Http {
            status: Some(http::StatusCode::INTERNAL_SERVER_ERROR),
            message: "server error".into(),
            retryable: true,
        };
        let err = LlmError::RetryExhausted {
            attempts: 3,
            last_error: Box::new(inner),
        };
        let display = format!("{err}");
        assert!(display.contains('3'));
        assert!(display.contains("server error"));
    }

    #[test]
    fn test_error_display_timeout() {
        let err = LlmError::Timeout { elapsed_ms: 5000 };
        assert!(format!("{err}").contains("5000"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LlmError>();
    }

    #[test]
    fn test_error_retryable_http() {
        let err = LlmError::Http {
            status: Some(http::StatusCode::TOO_MANY_REQUESTS),
            message: "rate limited".into(),
            retryable: true,
        };
        assert!(matches!(
            err,
            LlmError::Http {
                retryable: true,
                ..
            }
        ));
    }

    #[test]
    fn test_error_retryable_provider() {
        let err = LlmError::Provider {
            code: "bad_request".into(),
            message: "invalid".into(),
            retryable: false,
        };
        assert!(matches!(
            err,
            LlmError::Provider {
                retryable: false,
                ..
            }
        ));
    }

    #[test]
    fn test_error_retry_exhausted_nests() {
        let inner = LlmError::Auth("expired".into());
        let err = LlmError::RetryExhausted {
            attempts: 2,
            last_error: Box::new(inner),
        };
        assert!(matches!(
            &err,
            LlmError::RetryExhausted { last_error, .. }
                if matches!(last_error.as_ref(), LlmError::Auth(_))
        ));
    }

    #[test]
    fn test_error_retry_exhausted_source_chain() {
        use std::error::Error;
        let inner = LlmError::Auth("expired".into());
        let err = LlmError::RetryExhausted {
            attempts: 3,
            last_error: Box::new(inner),
        };
        let source = err.source().expect("RetryExhausted should have a source");
        assert!(format!("{source}").contains("expired"));
    }

    #[test]
    fn test_error_source_trait() {
        use std::error::Error;
        let err = LlmError::ToolExecution {
            tool_name: "test".into(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "gone")),
        };
        assert!(err.source().is_some());
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("not valid json").unwrap_err();
        let llm_err: LlmError = json_err.into();
        assert!(matches!(llm_err, LlmError::ResponseFormat { .. }));
    }

    #[test]
    fn test_error_display_max_depth_exceeded() {
        let err = LlmError::MaxDepthExceeded {
            current: 3,
            limit: 3,
        };
        let display = format!("{err}");
        assert!(display.contains("max nesting depth exceeded"));
        assert!(display.contains("current: 3"));
        assert!(display.contains("limit: 3"));
    }

    #[test]
    fn test_error_max_depth_not_retryable() {
        let err = LlmError::MaxDepthExceeded {
            current: 2,
            limit: 2,
        };
        assert!(!err.is_retryable());
    }
}
