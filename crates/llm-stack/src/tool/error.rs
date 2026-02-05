//! Tool error types.

/// Error returned by tool execution.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ToolError {
    /// Human-readable error description.
    pub message: String,
}

impl ToolError {
    /// Creates a new tool error with the given message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
