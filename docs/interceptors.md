# Interceptors

The interceptor system provides composable middleware for LLM calls and tool executions. It's a unified abstraction—learn it once, apply it everywhere.

## Overview

```rust
use llm_stack::ToolRegistry;
use llm_stack::intercept::{InterceptorStack, Retry, Timeout, ToolExec};
use std::time::Duration;

// Tool execution with interceptors (integrated with ToolRegistry)
let registry: ToolRegistry<()> = ToolRegistry::new()
    .with_interceptors(
        InterceptorStack::<ToolExec<()>>::new()
            .with(Retry::default())
            .with(Timeout::new(Duration::from_secs(30)))
    );
```

## Core Concepts

### Interceptable

A marker trait that defines what can be intercepted:

```rust
pub trait Interceptable: Send + Sync + 'static {
    type Input: Send;
    type Output: Send;
}
```

Two domains are provided:
- `ToolExec<Ctx>` — Input: `ToolRequest`, Output: `ToolResponse` (**integrated with `ToolRegistry`**)
- `LlmCall` — Input: `ChatParams`, Output: `Result<ChatResponse, LlmError>` (reserved, not yet integrated)

### Interceptor

The core trait. Interceptors wrap operations and can:
- Pass through unchanged
- Modify input before continuing
- Short-circuit with an early return
- Retry by calling `next` multiple times
- Transform the output after completion

```rust
pub trait Interceptor<T: Interceptable>: Send + Sync {
    fn intercept<'a>(
        &'a self,
        input: &'a T::Input,
        next: Next<'a, T>,
    ) -> Pin<Box<dyn Future<Output = T::Output> + Send + 'a>>;
}
```

### Next

A handle to continue the chain. It's `Copy`, so you can call it multiple times (essential for retry):

```rust
impl<T: Interceptable> Next<'_, T> {
    pub async fn run(self, input: &T::Input) -> T::Output;
}
```

### InterceptorStack

Chains interceptors together. First added = outermost (runs first on request, last on response):

```rust
let stack = InterceptorStack::<ToolExec<()>>::new()
    .with(Logging::default())  // Logs each attempt
    .with(Retry::default());   // Retries (each retry gets logged)
```

## Built-in Interceptors

### Retry

Exponential backoff retry for transient failures:

```rust
use llm_stack::intercept::Retry;

// Defaults: 3 attempts, 500ms initial delay, 2x multiplier
let retry = Retry::default();

// Custom configuration
let retry = Retry {
    max_attempts: 5,
    initial_delay: Duration::from_millis(100),
    max_delay: Duration::from_secs(60),
    multiplier: 1.5,
};
```

Works with any `Interceptable` where `Output: Retryable`.

### Timeout

Wraps operations with a deadline:

```rust
use llm_stack::intercept::Timeout;

let timeout = Timeout::new(Duration::from_secs(30));
```

Works with any `Interceptable` where `Output: Timeoutable`.

### Logging (feature: `tracing`)

Logs operation start/completion via `tracing`:

```rust
use llm_stack::intercept::{Logging, LogLevel};

let logging = Logging::new(LogLevel::Debug);
```

Log levels:
- `Info` — Duration only
- `Debug` — Duration + success/failure
- `Trace` — Includes input description

### Approval (tool-only)

Gate tool calls with custom logic:

```rust
use llm_stack::intercept::{Approval, ApprovalDecision};

let approval = Approval::new(|req| {
    if req.name == "delete_file" {
        ApprovalDecision::Deny("Destructive operations disabled".into())
    } else if req.name == "write_file" {
        // Sanitize arguments
        let mut args = req.arguments.clone();
        args["path"] = sanitize_path(&args["path"]);
        ApprovalDecision::Modify(args)
    } else {
        ApprovalDecision::Allow
    }
});
```

## Behavior Traits

Generic interceptors use behavior traits to work across domains:

### Retryable

```rust
pub trait Retryable {
    fn should_retry(&self) -> bool;
}
```

Implemented for:
- `Result<ChatResponse, LlmError>` — Returns `error.is_retryable()`
- `ToolResponse` — Returns `false` (tools don't auto-retry by default)

### Timeoutable

```rust
pub trait Timeoutable: Sized {
    fn timeout_error(duration: Duration) -> Self;
}
```

Implemented for:
- `Result<ChatResponse, LlmError>` — Returns `Err(LlmError::Timeout { .. })`
- `ToolResponse` — Returns error response with timeout message

### Loggable

```rust
pub trait Loggable {
    fn log_description(&self) -> String;
}
```

Implemented for:
- `ChatParams` — Returns message/tool counts
- `ToolRequest` — Returns tool name and call ID

### Outcome

```rust
pub trait Outcome {
    fn is_success(&self) -> bool;
}
```

Implemented for:
- `Result<ChatResponse, LlmError>` — Returns `is_ok()`
- `ToolResponse` — Returns `!is_error`

Used by `Logging` to report success/failure. Separate from `Retryable` because
logging success != retry decision (a successful response might still be retryable
for partial results, and a failed response might not be retryable for auth errors).

## Integration with ToolRegistry

```rust
use llm_stack::{ToolRegistry, tool_fn};
use llm_stack::intercept::{InterceptorStack, ToolExec, Approval, ApprovalDecision, Retry};

let registry: ToolRegistry<MyContext> = ToolRegistry::new()
    .with_interceptors(
        InterceptorStack::<ToolExec<MyContext>>::new()
            .with(Approval::new(|req| {
                // Your approval logic
                ApprovalDecision::Allow
            }))
            .with(Retry::default())
    );

registry.register(my_tool);
```

## Writing Custom Interceptors

### Basic Pattern

```rust
use llm_stack::intercept::{Interceptable, Interceptor, Next};
use std::future::Future;
use std::pin::Pin;

struct MyInterceptor;

impl<T> Interceptor<T> for MyInterceptor
where
    T: Interceptable,
    T::Input: Sync,
{
    fn intercept<'a>(
        &'a self,
        input: &'a T::Input,
        next: Next<'a, T>,
    ) -> Pin<Box<dyn Future<Output = T::Output> + Send + 'a>> {
        Box::pin(async move {
            // Before
            let result = next.run(input).await;
            // After
            result
        })
    }
}
```

### Domain-Specific Interceptor

```rust
use llm_stack::intercept::{Interceptor, Next, ToolExec, ToolRequest, ToolResponse};

struct RateLimiter {
    permits: Arc<Semaphore>,
}

impl<Ctx: Send + Sync + 'static> Interceptor<ToolExec<Ctx>> for RateLimiter {
    fn intercept<'a>(
        &'a self,
        input: &'a ToolRequest,
        next: Next<'a, ToolExec<Ctx>>,
    ) -> Pin<Box<dyn Future<Output = ToolResponse> + Send + 'a>> {
        Box::pin(async move {
            let _permit = self.permits.acquire().await.unwrap();
            next.run(input).await
        })
    }
}
```

### Short-Circuit Example

```rust
impl<Ctx: Send + Sync + 'static> Interceptor<ToolExec<Ctx>> for BlockDangerous {
    fn intercept<'a>(
        &'a self,
        input: &'a ToolRequest,
        next: Next<'a, ToolExec<Ctx>>,
    ) -> Pin<Box<dyn Future<Output = ToolResponse> + Send + 'a>> {
        Box::pin(async move {
            if input.name.starts_with("dangerous_") {
                return ToolResponse::error("Blocked by policy");
            }
            next.run(input).await
        })
    }
}
```

### Input Modification

```rust
impl<Ctx: Send + Sync + 'static> Interceptor<ToolExec<Ctx>> for AddMetadata {
    fn intercept<'a>(
        &'a self,
        input: &'a ToolRequest,
        next: Next<'a, ToolExec<Ctx>>,
    ) -> Pin<Box<dyn Future<Output = ToolResponse> + Send + 'a>> {
        Box::pin(async move {
            let mut modified = input.clone();
            modified.arguments["_timestamp"] = json!(Utc::now().to_rfc3339());
            next.run(&modified).await
        })
    }
}
```

## Execution Order

Interceptors execute in onion layers:

```
Request → [Logging] → [Retry] → [Timeout] → Operation
                                              ↓
Response ← [Logging] ← [Retry] ← [Timeout] ← Result
```

With this stack:
```rust
let stack = InterceptorStack::new()
    .with(Logging)   // First added = outermost
    .with(Retry)
    .with(Timeout);  // Last added = innermost
```

1. Logging sees the request first
2. Retry wraps everything inside (each retry goes through Timeout)
3. Timeout is closest to the actual operation

## Sharing Interceptors

Use `Arc` for interceptors that need to share state across stacks:

```rust
let metrics = Arc::new(MetricsInterceptor::new());

let tool_stack = InterceptorStack::<ToolExec<()>>::new()
    .with_shared(metrics.clone() as Arc<dyn Interceptor<ToolExec<()>>>);

let llm_stack = InterceptorStack::<LlmCall>::new()
    .with_shared(metrics.clone() as Arc<dyn Interceptor<LlmCall>>);
```

## Module Structure

```
llm_stack::intercept
├── Interceptable          // Core trait for interceptable operations
├── Interceptor            // Core trait for interceptors
├── InterceptorStack       // Builder for chaining interceptors
├── Next                   // Handle to continue the chain
├── Operation              // Trait for final operations
├── FnOperation            // Closure wrapper
│
├── domain                 // Domain markers
│   ├── LlmCall           // Marker for provider calls
│   ├── ToolExec<Ctx>     // Marker for tool execution
│   ├── ToolRequest       // Input for tool execution
│   └── ToolResponse      // Output from tool execution
│
├── behavior              // Behavior traits
│   ├── Retryable
│   ├── Timeoutable
│   ├── Loggable
│   └── Outcome
│
├── interceptors          // Built-in generic interceptors
│   ├── Retry
│   ├── Timeout
│   ├── NoOp
│   ├── Logging           // (feature: tracing)
│   └── LogLevel          // (feature: tracing)
│
└── tool_interceptors     // Tool-specific interceptors
    ├── Approval
    └── ApprovalDecision
```

## Testing Interceptors

Use `NoOp` as a placeholder and custom test operations:

```rust
#[tokio::test]
async fn test_my_interceptor() {
    struct EchoOp;

    impl Operation<TestDomain> for EchoOp {
        fn execute<'a>(&'a self, input: &'a String)
            -> Pin<Box<dyn Future<Output = String> + Send + 'a>>
        {
            Box::pin(async move { format!("echo: {}", input) })
        }
    }

    let stack = InterceptorStack::<TestDomain>::new()
        .with(MyInterceptor);

    let result = stack.execute(&"test".to_string(), &EchoOp).await;
    assert_eq!(result, "echo: test");
}
```
