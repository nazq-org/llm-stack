//! Loop depth tracking for nested tool loops.
//!
//! When tools spawn sub-agents (nested `tool_loop` calls), depth tracking
//! prevents runaway recursion. Use [`LoopContext`] for built-in depth
//! management, or implement [`LoopDepth`] manually on your own type.
//!
//! # Using `LoopContext` (recommended)
//!
//! ```rust
//! use llm_stack::tool::LoopContext;
//!
//! // Wrap your application state — depth tracking is automatic
//! let ctx = LoopContext::new(MyState { user_id: "u123".into() });
//! # #[derive(Clone)] struct MyState { user_id: String }
//! ```
//!
//! # Manual implementation
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

// ── LoopContext ──────────────────────────────────────────────────────

/// Generic context wrapper with built-in depth tracking.
///
/// Wraps any `Clone + Send + Sync` state and automatically implements
/// [`LoopDepth`], eliminating the boilerplate of storing a `depth` field
/// and writing the trait impl yourself.
///
/// # Examples
///
/// ```rust
/// use llm_stack::tool::{LoopContext, LoopDepth, ToolRegistry};
///
/// #[derive(Clone)]
/// struct AppState {
///     user_id: String,
///     api_key: String,
/// }
///
/// let ctx = LoopContext::new(AppState {
///     user_id: "user_123".into(),
///     api_key: "sk-secret".into(),
/// });
///
/// assert_eq!(ctx.loop_depth(), 0);
/// assert_eq!(ctx.state.user_id, "user_123");
///
/// // Use with a typed registry
/// let registry: ToolRegistry<LoopContext<AppState>> = ToolRegistry::new();
/// ```
///
/// For the zero-state case, use `LoopContext<()>`:
///
/// ```rust
/// use llm_stack::tool::{LoopContext, LoopDepth};
///
/// let ctx = LoopContext::empty();
/// assert_eq!(ctx.loop_depth(), 0);
///
/// let nested = ctx.with_depth(1);
/// assert_eq!(nested.loop_depth(), 1);
/// ```
#[derive(Clone, Debug)]
pub struct LoopContext<T: Clone + Send + Sync = ()> {
    /// The application state accessible from tool handlers.
    pub state: T,
    depth: u32,
}

impl<T: Clone + Send + Sync> LoopContext<T> {
    /// Create a new context wrapping the given state at depth 0.
    pub fn new(state: T) -> Self {
        Self { state, depth: 0 }
    }
}

impl LoopContext<()> {
    /// Create a stateless context at depth 0.
    pub fn empty() -> Self {
        Self {
            state: (),
            depth: 0,
        }
    }
}

impl<T: Clone + Send + Sync> LoopDepth for LoopContext<T> {
    fn loop_depth(&self) -> u32 {
        self.depth
    }

    fn with_depth(&self, depth: u32) -> Self {
        Self {
            state: self.state.clone(),
            depth,
        }
    }
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
        let nested = ().with_depth(5);
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
        assert_eq!(nested.name, "test");
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

    #[test]
    fn test_loop_context_new() {
        #[derive(Clone, Debug, PartialEq)]
        struct State {
            name: String,
        }

        let ctx = LoopContext::new(State {
            name: "test".into(),
        });
        assert_eq!(ctx.loop_depth(), 0);
        assert_eq!(ctx.state.name, "test");
    }

    #[test]
    fn test_loop_context_with_depth_preserves_state() {
        #[derive(Clone, Debug, PartialEq)]
        struct State {
            user_id: String,
            api_key: String,
        }

        let ctx = LoopContext::new(State {
            user_id: "u1".into(),
            api_key: "k1".into(),
        });

        let nested = ctx.with_depth(3);
        assert_eq!(nested.loop_depth(), 3);
        assert_eq!(nested.state.user_id, "u1");
        assert_eq!(nested.state.api_key, "k1");
    }

    #[test]
    fn test_loop_context_empty() {
        let ctx = LoopContext::empty();
        assert_eq!(ctx.loop_depth(), 0);

        let nested = ctx.with_depth(2);
        assert_eq!(nested.loop_depth(), 2);
    }

    #[test]
    fn test_loop_context_depth_chain() {
        let ctx = LoopContext::new("agent");
        let l1 = ctx.with_depth(ctx.loop_depth() + 1);
        let l2 = l1.with_depth(l1.loop_depth() + 1);
        let l3 = l2.with_depth(l2.loop_depth() + 1);
        assert_eq!(l3.loop_depth(), 3);
        assert_eq!(l3.state, "agent");
    }
}
