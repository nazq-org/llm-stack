//! MCP (Model Context Protocol) integration for `ToolRegistry`.
//!
//! This module defines the [`McpService`] trait - a minimal contract for MCP-like
//! tool sources. Users implement this trait for their chosen MCP client library
//! (e.g., `rmcp`), keeping version coupling in their code rather than in llm-core.
//!
//! # Design Philosophy
//!
//! Rather than depending on a specific MCP client library, llm-core defines a
//! simple trait that any MCP implementation can satisfy. This means:
//!
//! - No forced dependency upgrades when MCP libraries release new versions
//! - Users choose their preferred MCP client and version
//! - Easy to mock for testing
//! - Simple enough to implement (~50 lines for rmcp)
//!
//! # Example: Implementing for rmcp
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use rmcp::service::RunningService;
//! use rmcp::handler::client::ClientHandler;
//! use rmcp::RoleClient;
//! use llm_stack::{ToolDefinition, JsonSchema};
//! use llm_stack::mcp::{McpService, McpError};
//!
//! /// Adapter: wraps rmcp RunningService to implement llm-core's McpService
//! pub struct RmcpAdapter<S: ClientHandler> {
//!     service: Arc<RunningService<RoleClient, S>>,
//! }
//!
//! impl<S: ClientHandler> RmcpAdapter<S> {
//!     pub fn new(service: Arc<RunningService<RoleClient, S>>) -> Self {
//!         Self { service }
//!     }
//! }
//!
//! impl<S: ClientHandler> McpService for RmcpAdapter<S> {
//!     async fn list_tools(&self) -> Result<Vec<ToolDefinition>, McpError> {
//!         let tools = self.service
//!             .list_all_tools()
//!             .await
//!             .map_err(|e| McpError::Protocol(e.to_string()))?;
//!
//!         Ok(tools.into_iter().map(|t| ToolDefinition {
//!             name: t.name.to_string(),
//!             description: t.description.map(|d| d.to_string()).unwrap_or_default(),
//!             parameters: JsonSchema::new(
//!                 serde_json::to_value(&*t.input_schema).unwrap_or_default()
//!             ),
//!         }).collect())
//!     }
//!
//!     async fn call_tool(&self, name: &str, args: serde_json::Value) -> Result<String, McpError> {
//!         use rmcp::model::{CallToolRequestParams, RawContent};
//!
//!         let params = CallToolRequestParams {
//!             meta: None,
//!             name: name.to_string().into(),
//!             arguments: args.as_object().cloned(),
//!             task: None,
//!         };
//!
//!         let result = self.service
//!             .call_tool(params)
//!             .await
//!             .map_err(|e| McpError::ToolExecution(e.to_string()))?;
//!
//!         if result.is_error.unwrap_or(false) {
//!             return Err(McpError::ToolExecution(extract_text(&result.content)));
//!         }
//!
//!         Ok(extract_text(&result.content))
//!     }
//! }
//!
//! fn extract_text(content: &[rmcp::model::Content]) -> String {
//!     use rmcp::model::RawContent;
//!     content.iter().map(|c| match &c.raw {
//!         RawContent::Text(t) => t.text.clone(),
//!         _ => "[non-text]".into(),
//!     }).collect::<Vec<_>>().join("\n")
//! }
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use llm_stack::ToolRegistry;
//! use llm_stack::mcp::McpRegistryExt;
//!
//! // Create your MCP service (using rmcp or any other library)
//! let mcp_service = Arc::new(RmcpAdapter::new(rmcp_client));
//!
//! // Register tools with llm-core
//! let mut registry = ToolRegistry::new();
//! registry.register_mcp_service(&mcp_service).await?;
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::tool::{ToolError, ToolHandler};
use crate::{ToolDefinition, ToolRegistry};

/// Error type for MCP operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum McpError {
    /// Error during MCP protocol communication.
    #[error("MCP protocol error: {0}")]
    Protocol(String),

    /// Error during tool execution.
    #[error("MCP tool execution error: {0}")]
    ToolExecution(String),
}

/// Minimal contract for MCP-like tool sources.
///
/// Implement this trait to bridge any MCP client library with llm-core's
/// tool system. The trait requires only two operations:
///
/// 1. List available tools (with their schemas)
/// 2. Call a tool by name with JSON arguments
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to allow use across async tasks.
/// Typically achieved by wrapping the underlying client in `Arc`.
///
/// # Object Safety
///
/// This trait is object-safe (`dyn McpService`) to allow storing different
/// MCP service implementations in the same registry.
pub trait McpService: Send + Sync {
    /// Lists all tools available from the MCP server.
    fn list_tools(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ToolDefinition>, McpError>> + Send + '_>>;

    /// Calls a tool on the MCP server.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the tool to call
    /// * `args` - JSON arguments matching the tool's input schema
    ///
    /// # Returns
    ///
    /// The tool's text output, or an error if execution failed.
    fn call_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, McpError>> + Send + '_>>;
}

/// A [`ToolHandler`] that delegates execution to an [`McpService`].
struct McpToolHandler {
    service: Arc<dyn McpService>,
    definition: ToolDefinition,
}

impl McpToolHandler {
    fn new(service: Arc<dyn McpService>, definition: ToolDefinition) -> Self {
        Self {
            service,
            definition,
        }
    }
}

impl std::fmt::Debug for McpToolHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpToolHandler")
            .field("tool", &self.definition.name)
            .finish_non_exhaustive()
    }
}

impl ToolHandler<()> for McpToolHandler {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn execute<'a>(
        &'a self,
        input: serde_json::Value,
        _ctx: &'a (),
    ) -> Pin<Box<dyn Future<Output = Result<crate::tool::ToolOutput, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            self.service
                .call_tool(&self.definition.name, input)
                .await
                .map(crate::tool::ToolOutput::new)
                .map_err(|e| ToolError::new(e.to_string()))
        })
    }
}

/// Extension trait for registering MCP services with a [`ToolRegistry`].
pub trait McpRegistryExt {
    /// Registers all tools from an MCP service.
    ///
    /// Discovers tools from the service and registers each one as a
    /// `ToolHandler` that delegates execution to the service.
    ///
    /// # Returns
    ///
    /// The number of tools registered.
    fn register_mcp_service<S: McpService + 'static>(
        &mut self,
        service: &Arc<S>,
    ) -> impl Future<Output = Result<usize, McpError>> + Send;

    /// Registers specific tools from an MCP service by name.
    ///
    /// Only registers tools whose names are in the provided list.
    /// Tools not found on the server are silently skipped.
    ///
    /// # Returns
    ///
    /// The number of tools actually registered.
    fn register_mcp_tools_by_name<S: McpService + 'static>(
        &mut self,
        service: &Arc<S>,
        tool_names: &[&str],
    ) -> impl Future<Output = Result<usize, McpError>> + Send;
}

impl McpRegistryExt for ToolRegistry<()> {
    async fn register_mcp_service<S: McpService + 'static>(
        &mut self,
        service: &Arc<S>,
    ) -> Result<usize, McpError> {
        let tools = service.list_tools().await?;
        let count = tools.len();

        for definition in tools {
            let handler =
                McpToolHandler::new(Arc::clone(service) as Arc<dyn McpService>, definition);
            self.register(handler);
        }

        Ok(count)
    }

    async fn register_mcp_tools_by_name<S: McpService + 'static>(
        &mut self,
        service: &Arc<S>,
        tool_names: &[&str],
    ) -> Result<usize, McpError> {
        let tools = service.list_tools().await?;
        let mut count = 0;

        for definition in tools {
            if tool_names.contains(&definition.name.as_str()) {
                let handler =
                    McpToolHandler::new(Arc::clone(service) as Arc<dyn McpService>, definition);
                self.register(handler);
                count += 1;
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JsonSchema;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock MCP service for testing
    struct MockMcpService {
        tools: Vec<ToolDefinition>,
        call_count: AtomicUsize,
    }

    impl MockMcpService {
        fn new(tools: Vec<ToolDefinition>) -> Self {
            Self {
                tools,
                call_count: AtomicUsize::new(0),
            }
        }
    }

    impl McpService for MockMcpService {
        fn list_tools(
            &self,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<ToolDefinition>, McpError>> + Send + '_>>
        {
            let tools = self.tools.clone();
            Box::pin(async move { Ok(tools) })
        }

        fn call_tool(
            &self,
            name: &str,
            _args: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<String, McpError>> + Send + '_>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let result = format!("Called {name}");
            Box::pin(async move { Ok(result) })
        }
    }

    fn test_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("{name} description"),
            parameters: JsonSchema::new(serde_json::json!({"type": "object"})),
            retry: None,
        }
    }

    #[test]
    fn test_trait_is_object_safe() {
        fn assert_object_safe(_: &dyn McpService) {}
        let mock = MockMcpService::new(vec![]);
        assert_object_safe(&mock);
    }

    #[tokio::test]
    async fn test_register_mcp_service() {
        let service = Arc::new(MockMcpService::new(vec![
            test_tool("tool_a"),
            test_tool("tool_b"),
        ]));

        let mut registry = ToolRegistry::new();
        let count = registry.register_mcp_service(&service).await.unwrap();

        assert_eq!(count, 2);
        assert_eq!(registry.len(), 2);
        assert!(registry.get("tool_a").is_some());
        assert!(registry.get("tool_b").is_some());
    }

    #[tokio::test]
    async fn test_register_mcp_tools_by_name() {
        let service = Arc::new(MockMcpService::new(vec![
            test_tool("tool_a"),
            test_tool("tool_b"),
            test_tool("tool_c"),
        ]));

        let mut registry = ToolRegistry::new();
        let count = registry
            .register_mcp_tools_by_name(&service, &["tool_a", "tool_c"])
            .await
            .unwrap();

        assert_eq!(count, 2);
        assert!(registry.get("tool_a").is_some());
        assert!(registry.get("tool_b").is_none());
        assert!(registry.get("tool_c").is_some());
    }

    #[tokio::test]
    async fn test_mcp_tool_execution() {
        let service = Arc::new(MockMcpService::new(vec![test_tool("my_tool")]));

        let mut registry = ToolRegistry::new();
        registry.register_mcp_service(&service).await.unwrap();

        let handler = registry.get("my_tool").unwrap();
        let result = handler.execute(serde_json::json!({}), &()).await.unwrap();

        assert_eq!(result.content, "Called my_tool");
        assert_eq!(service.call_count.load(Ordering::SeqCst), 1);
    }
}
