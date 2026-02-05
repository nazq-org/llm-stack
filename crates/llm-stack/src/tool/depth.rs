//! Loop depth tracking for nested tool loops.
//!
//! When tools spawn sub-agents (nested `tool_loop` calls), depth tracking
//! prevents runaway recursion. Implement [`LoopDepth`] on your context type
//! to enable automatic depth management.
//!
//! # Example
//!
//! ```rust
//! use llm_stack::tool::LoopDepth;
//!
//! #[derive(Clone)]
//! struct AgentContext {
//!     user_id: String,
//!     depth: u32,
//! }
//!
//! impl LoopDepth for AgentContext {
//!     fn loop_depth(&self) -> u32 {
//!         self.depth
//!     }
//!
//!     fn with_depth(&self, depth: u32) -> Self {
//!         Self {
//!             depth,
//!             ..self.clone()
//!         }
//!     }
//! }
//! ```

/// Trait for contexts that support automatic depth tracking in nested tool loops.
///
/// When `tool_loop` executes tools, it passes a context with incremented depth
/// so that nested loops can enforce depth limits via `max_depth` in
/// [`ToolLoopConfig`](super::ToolLoopConfig).
///
/// # Blanket Implementation
///
/// The unit type `()` has a blanket implementation that always returns depth 0.
/// Use this for simple cases where depth tracking isn't needed:
///
/// ```rust
/// use llm_stack::tool::LoopDepth;
///
/// // () always returns 0, ignores depth changes
/// assert_eq!(().loop_depth(), 0);
/// assert_eq!(().with_depth(5), ());
/// ```
///
/// # Custom Implementation
///
/// For agent systems with nesting, implement this on your context type:
///
/// ```rust
/// use llm_stack::tool::LoopDepth;
///
/// #[derive(Clone)]
/// struct MyContext {
///     session_id: String,
///     loop_depth: u32,
/// }
///
/// impl LoopDepth for MyContext {
///     fn loop_depth(&self) -> u32 {
///         self.loop_depth
///     }
///
///     fn with_depth(&self, depth: u32) -> Self {
///         Self {
///             loop_depth: depth,
///             ..self.clone()
///         }
///     }
/// }
/// ```
pub trait LoopDepth: Clone + Send + Sync {
    /// Returns the current nesting depth.
    ///
    /// A depth of 0 means this is the top-level loop (not nested).
    fn loop_depth(&self) -> u32;

    /// Returns a new context with the specified depth.
    ///
    /// Called by `tool_loop` when passing context to tool handlers,
    /// incrementing depth for any nested loops.
    #[must_use]
    fn with_depth(&self, depth: u32) -> Self;
}

/// Blanket implementation for unit type — always depth 0, no tracking.
///
/// This allows simple use cases to work without implementing the trait:
///
/// ```rust
/// use llm_stack::tool::{ToolLoopConfig, ToolRegistry};
///
/// // Works with () context, no depth tracking
/// let registry: ToolRegistry<()> = ToolRegistry::new();
/// ```
impl LoopDepth for () {
    fn loop_depth(&self) -> u32 {
        0
    }

    fn with_depth(&self, _depth: u32) -> Self {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_loop_depth() {
        assert_eq!(().loop_depth(), 0);
    }

    #[test]
    #[allow(clippy::let_unit_value)]
    fn test_unit_with_depth_ignores_value() {
        // with_depth on () returns (), which still has depth 0
        let nested = ().with_depth(5);
        // Even after "nesting", unit context still reports depth 0
        assert_eq!(nested.loop_depth(), 0);
    }

    #[derive(Clone)]
    struct TestContext {
        name: String,
        depth: u32,
    }

    impl LoopDepth for TestContext {
        fn loop_depth(&self) -> u32 {
            self.depth
        }

        fn with_depth(&self, depth: u32) -> Self {
            Self {
                depth,
                ..self.clone()
            }
        }
    }

    #[test]
    fn test_custom_context_depth() {
        let ctx = TestContext {
            name: "test".into(),
            depth: 0,
        };
        assert_eq!(ctx.loop_depth(), 0);

        let nested = ctx.with_depth(1);
        assert_eq!(nested.loop_depth(), 1);
        assert_eq!(nested.name, "test"); // Other fields preserved
    }

    #[test]
    fn test_depth_increments() {
        let ctx = TestContext {
            name: "agent".into(),
            depth: 0,
        };

        let level1 = ctx.with_depth(ctx.loop_depth() + 1);
        assert_eq!(level1.loop_depth(), 1);

        let level2 = level1.with_depth(level1.loop_depth() + 1);
        assert_eq!(level2.loop_depth(), 2);
    }
}
