# Architecture

## Overview

llm-stack is a workspace of focused crates. Application code depends on `llm-stack-core` for types and traits, plus whichever provider crates it needs. The interceptor system handles cross-cutting concerns like retry, logging, and approval gates.

```
 Application
     |
     v
 InterceptorStack          (retry, logging, approval)
     |
     +--- LlmCall domain     (wraps Provider calls)
     +--- ToolExec domain    (wraps tool executions)
     |
     v
 Provider trait            (llm-stack-core)
     |
     +--- AnthropicProvider  (llm-stack-anthropic)
     +--- OpenAiProvider     (llm-stack-openai)
     +--- OllamaProvider     (llm-stack-ollama)
```

Adding a new provider means implementing two async methods and one metadata method. No registration, no feature flags, no changes to the core crate.

## Provider trait design

### Why two traits?

Rust 2024's async-fn-in-traits (AFIT) lets us write `Provider` as plain async methods -- no `#[async_trait]` macro. But AFIT returns `impl Future`, which is not object-safe. Code that needs `dyn` dispatch (collections of providers, interceptor chains) uses `DynProvider` instead.

A blanket impl bridges the two:

```text
Provider              (AFIT, not object-safe, zero overhead)
    |
    | blanket impl
    v
DynProvider           (boxed futures, object-safe, tiny heap alloc per call)
```

Provider authors implement `Provider`. The blanket impl gives them `DynProvider` for free. Application code uses whichever fits:

- **Generic code** (`fn foo(p: &impl Provider)`) -- zero-cost, monomorphized.
- **Dynamic code** (`fn foo(p: &dyn DynProvider)`) -- one Box per call, works with heterogeneous collections.

### The three methods

```rust
trait Provider: Send + Sync {
    async fn generate(&self, params: &ChatParams) -> Result<ChatResponse, LlmError>;
    async fn stream(&self, params: &ChatParams) -> Result<ChatStream, LlmError>;
    fn metadata(&self) -> ProviderMetadata;
}
```

`generate` returns the complete response. `stream` returns a pinned stream of incremental events. `metadata` returns static information (name, model, context window, capabilities).

Everything else -- retry, logging, tool execution, structured output -- is handled by the interceptor system or library functions that compose over `Provider`. This keeps individual provider implementations small and testable.

## Content model

Every message carries `Vec<ContentBlock>` rather than a plain `String`. A single enum covers all content types:

```rust
enum ContentBlock {
    Text(String),
    Image { media_type, data },
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    Reasoning { content },
}
```

This uniform representation avoids the combinatorial explosion of having separate message types for "text only", "tool calls only", "mixed", etc. Convenience constructors (`ChatMessage::user("hello")`) keep the simple case simple.

## Streaming

Streaming responses arrive as a sequence of `StreamEvent`s:

```text
TextDelta("Hello") -> TextDelta(" world") -> Usage(...) -> Done(EndTurn)
```

Tool calls stream in three phases:

```text
ToolCallStart { id, name }
ToolCallDelta { json_chunk }    (one or more)
ToolCallComplete { call }       (fully parsed)
```

The `index` field on each event identifies which tool call it belongs to when the model invokes multiple tools in parallel.

`ChatStream` is a type alias for `Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>`. Consumers use `futures::StreamExt`.

## Error design

`LlmError` is a single enum covering every failure mode. Key design decisions:

- **`retryable` is on the error**, not decided externally. The provider knows whether a 429 is retryable; the retry interceptor just reads the flag.
- **`#[non_exhaustive]`** on all public enums so new variants don't break downstream.
- **`#[source]`** on nested errors for proper error chain propagation.
- **`From<serde_json::Error>`** for ergonomic `?` in response parsing.

## Cost tracking

Costs use integer microdollars (1 USD = 1,000,000 microdollars) to avoid floating-point accumulation errors across thousands of API calls. The `Cost` type enforces `total == input + output` at construction time through private fields and a checked constructor.

## Module map (`llm-stack-core`)

```
src/
  lib.rs          Crate root, re-exports, #![warn(missing_docs)]
  chat.rs         ChatMessage, ContentBlock, ChatResponse, StopReason
  provider.rs     Provider, DynProvider, ChatParams, ToolDefinition, JsonSchema
  stream.rs       ChatStream, StreamEvent
  usage.rs        Usage, Cost, UsageTracker
  error.rs        LlmError
  context.rs      ContextWindow, token estimation
  mcp.rs          McpService trait, McpRegistryExt
  registry.rs     ProviderRegistry, ProviderFactory, ProviderConfig
  structured.rs   generate_object, stream_object (feature-gated)
  mock.rs         MockProvider, MockError (behind test-utils feature)
  intercept.rs    Interceptor system (LlmCall, ToolExec domains)
  tool/
    mod.rs          Re-exports, module docs
    error.rs        ToolError
    output.rs       ToolOutput
    handler.rs      ToolHandler trait, FnToolHandler, NoCtxToolHandler
    helpers.rs      tool_fn, tool_fn_with_ctx
    registry.rs     ToolRegistry (Clone, without, only), execute logic
    config.rs       ToolLoopConfig (max_depth), ToolLoopEvent, TerminationReason
    depth.rs        LoopDepth trait, blanket impl for ()
    loop_sync.rs    tool_loop() with depth checking
    loop_stream.rs  tool_loop_stream() with depth checking
    loop_channel.rs tool_loop_channel() for backpressure
    loop_detection.rs Loop detection logic
    execution.rs    Tool execution with events
    approval.rs     ToolApproval handling
```
