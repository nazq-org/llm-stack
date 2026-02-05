//! Tool output types.

/// Output returned by a tool handler.
///
/// Contains the content string that will be sent back to the LLM.
/// For application-level metadata (metrics, request IDs, etc.), use
/// the context parameter to write to shared state instead.
///
/// # Example
///
/// ```rust
/// use llm_stack::tool::ToolOutput;
///
/// let output = ToolOutput::new("Result: 42");
/// ```
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    /// The content to return to the LLM.
    pub content: String,
}

impl ToolOutput {
    /// Creates a new tool output with the given content.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }
}

impl From<String> for ToolOutput {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for ToolOutput {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}
