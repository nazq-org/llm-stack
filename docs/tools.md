# Tool Use

Tools let LLMs call functions you define. The model decides when to use a tool, generates arguments, and your code executes it. The SDK handles the back-and-forth conversation automatically.

## Quick example

```rust
use llm_stack::{
    ChatMessage, ChatParams, ToolDefinition, JsonSchema,
    tool_fn, ToolRegistry, tool_loop, ToolLoopConfig,
};
use serde_json::json;

// 1. Define and implement a tool using tool_fn helper
let weather_tool = tool_fn(
    ToolDefinition {
        name: "get_weather".into(),
        description: "Get current weather for a location".into(),
        parameters: JsonSchema::new(json!({
            "type": "object",
            "properties": {
                "location": { "type": "string", "description": "City name" }
            },
            "required": ["location"]
        })),
        retry: None,
    },
    |args| async move {
        let location = args["location"].as_str().unwrap_or("unknown");
        Ok(format!("Weather in {location}: 72°F, sunny"))
    },
);

// 2. Build a registry
let mut registry: ToolRegistry<()> = ToolRegistry::new();
registry.register(weather_tool);

// 3. Run the tool loop
let params = ChatParams {
    messages: vec![ChatMessage::user("What's the weather in Tokyo?")],
    tools: Some(registry.definitions()),
    ..Default::default()
};

let result = tool_loop(provider, &registry, params, ToolLoopConfig::default(), &()).await?;
println!("{}", result.response.text().unwrap_or_default());
// "The weather in Tokyo is 72°F and sunny."
```

## How it works

```text
User: "What's the weather in Tokyo?"
    ↓
Model: ToolCall { name: "get_weather", args: {"location": "Tokyo"} }
    ↓
SDK: Execute handler → "Weather in Tokyo: 72°F, sunny"
    ↓
Model: "The weather in Tokyo is 72°F and sunny."
    ↓
SDK: Return final response
```

The `tool_loop` function handles this entire conversation automatically.

## Defining tools

Tools have a name, description, JSON Schema for arguments, and optional retry config:

```rust
use llm_stack::{ToolDefinition, JsonSchema};
use serde_json::json;

let calculator = ToolDefinition {
    name: "calculator".into(),
    description: "Perform basic arithmetic".into(),
    parameters: JsonSchema::new(json!({
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression like '2 + 2' or '10 * 5'"
            }
        },
        "required": ["expression"]
    })),
    retry: None,  // Or Some(ToolRetryConfig::default())
};
```

The JSON Schema is sent to the model, which uses it to generate valid arguments.

## Implementing handlers

There are two ways to create tool handlers: the `tool_fn` helper or implementing `ToolHandler` directly.

### Using tool_fn (recommended)

The simplest way to create a tool:

```rust
use llm_stack::{tool_fn, ToolDefinition, JsonSchema};
use serde_json::json;

let handler = tool_fn(
    ToolDefinition {
        name: "greet".into(),
        description: "Greet someone".into(),
        parameters: JsonSchema::new(json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"]
        })),
        retry: None,
    },
    |args| async move {
        let name = args["name"].as_str().unwrap_or("stranger");
        Ok(format!("Hello, {name}!"))
    },
);
```

### Using tool_fn_with_ctx (for context access)

When tools need access to shared state (database connections, user identity, etc.):

```rust
use llm_stack::{tool_fn_with_ctx, ToolDefinition, JsonSchema, ToolOutput, ToolRegistry, tool_loop, ToolLoopConfig, ChatParams, ChatMessage};
use serde_json::{json, Value};

struct AppContext {
    user_id: String,
    api_key: String,
}

let handler = tool_fn_with_ctx(
    ToolDefinition {
        name: "get_orders".into(),
        description: "Fetch orders for the current user".into(),
        parameters: JsonSchema::new(json!({
            "type": "object",
            "properties": {
                "limit": { "type": "number", "description": "Max orders to return" }
            },
            "required": ["limit"]
        })),
        retry: None,
    },
    |args: Value, ctx: &AppContext| {
        let limit = args["limit"].as_u64().unwrap_or(10);
        let user_id = ctx.user_id.clone();
        let api_key = ctx.api_key.clone();
        async move {
            // Use user_id, api_key, and limit to fetch orders...
            Ok(ToolOutput::new(format!("Found {limit} orders for user {user_id}")))
        }
    },
);

// Create registry and context
let mut registry: ToolRegistry<AppContext> = ToolRegistry::new();
registry.register(handler);

let ctx = AppContext {
    user_id: "user_123".into(),
    api_key: "sk-secret".into(),
};

// Run tool loop with context
let params = ChatParams {
    messages: vec![ChatMessage::user("Show my last 5 orders")],
    tools: Some(registry.definitions()),
    ..Default::default()
};

let result = tool_loop(provider, &registry, params, ToolLoopConfig::default(), &ctx).await?;
```

**Note**: Clone data from context before the async block — the closure can't borrow across await points.

### Implementing ToolHandler directly

For more control, implement the trait:

```rust
use llm_stack::{ToolHandler, ToolError, ToolOutput, ToolDefinition, JsonSchema};
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;

struct Calculator;

impl ToolHandler<()> for Calculator {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "calculator".into(),
            description: "Evaluate math expressions".into(),
            parameters: JsonSchema::new(json!({
                "type": "object",
                "properties": { "expr": { "type": "string" } },
                "required": ["expr"]
            })),
            retry: None,
        }
    }

    fn execute<'a>(
        &'a self,
        input: Value,
        _ctx: &'a (),
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let expr = input["expr"].as_str()
                .ok_or_else(|| ToolError::new("missing expr"))?;
            // Evaluate expression...
            Ok(ToolOutput::new("42"))
        })
    }
}
```

## The tool registry

`ToolRegistry<Ctx>` manages tools and their handlers:

```rust
use llm_stack::ToolRegistry;

// Registry with no context
let mut registry: ToolRegistry<()> = ToolRegistry::new();
registry.register(weather_handler);
registry.register(calculator);

// Registry with custom context
let mut registry: ToolRegistry<AppContext> = ToolRegistry::new();
registry.register(context_aware_handler);

// Get definitions for ChatParams
let params = ChatParams {
    tools: Some(registry.definitions()),
    ..Default::default()
};

// Execute a tool call manually
let result = registry.execute(&tool_call, &ctx).await;
```

With the `schema` feature enabled, the registry validates arguments against the JSON Schema before calling handlers.

## Tool loop configuration

Control how the tool loop behaves:

```rust
use llm_stack::ToolLoopConfig;
use std::time::Duration;

let config = ToolLoopConfig {
    max_iterations: 10,              // Max tool call rounds (default: 10)
    parallel_tool_execution: true,   // Execute multiple tools in parallel
    on_tool_call: None,              // Optional approval callback
    on_event: None,                  // Optional observability callback
    stop_when: None,                 // Custom stop condition
    loop_detection: None,            // Detect stuck agents
    timeout: Some(Duration::from_secs(30)),  // Wall-clock timeout
};
```

### Limiting iterations

Prevent infinite loops with `max_iterations`:

```rust
let config = ToolLoopConfig {
    max_iterations: 5,  // Stop after 5 rounds of tool calls
    ..Default::default()
};
```

### Wall-clock timeout

Enforce a time budget for the entire tool loop:

```rust
use std::time::Duration;

let config = ToolLoopConfig {
    timeout: Some(Duration::from_secs(30)),
    ..Default::default()
};

let result = tool_loop(provider, &registry, params, config, &ctx).await?;

// Check if we timed out
match result.termination_reason {
    TerminationReason::Timeout { limit } => {
        println!("Exceeded {limit:?} timeout");
    }
    TerminationReason::Complete => {
        println!("Completed normally");
    }
    // ... other reasons
}
```

### Parallel execution

When the model calls multiple tools at once, execute them in parallel:

```rust
let config = ToolLoopConfig {
    parallel_tool_execution: true,  // Default
    ..Default::default()
};
```

Set to `false` to execute tools sequentially.

### Approval hooks

Require approval before executing tools:

```rust
use llm_stack::ToolApproval;
use std::sync::Arc;

let config = ToolLoopConfig {
    on_tool_call: Some(Arc::new(|tool_call| {
        if tool_call.name == "delete_file" {
            ToolApproval::Deny("Destructive operations not allowed".into())
        } else {
            ToolApproval::Approve
        }
    })),
    ..Default::default()
};
```

You can also modify arguments:

```rust
ToolApproval::Modify(json!({"sanitized": true}))
```

### Loop detection

Detect when an agent is stuck calling the same tool repeatedly:

```rust
use llm_stack::{LoopDetectionConfig, LoopAction};

let config = ToolLoopConfig {
    loop_detection: Some(LoopDetectionConfig {
        threshold: 3,  // Trigger after 3 identical consecutive calls
        action: LoopAction::InjectWarning,  // Tell the agent it's looping
    }),
    ..Default::default()
};
```

Actions:
- `LoopAction::Warn` — Emit event, continue
- `LoopAction::Stop` — Terminate with error
- `LoopAction::InjectWarning` — Add message to conversation

### Custom stop conditions

Stop the loop based on custom logic:

```rust
use llm_stack::{StopDecision, StopContext};
use std::sync::Arc;

let config = ToolLoopConfig {
    stop_when: Some(Arc::new(|ctx: &StopContext| {
        // Stop if we've used too many tokens
        if ctx.total_usage.input_tokens > 10000 {
            StopDecision::StopWithReason("Token budget exceeded".into())
        }
        // Stop if a specific tool was called
        else if ctx.last_tool_results.iter().any(|r| r.tool_call_id.contains("final")) {
            StopDecision::Stop
        } else {
            StopDecision::Continue
        }
    })),
    ..Default::default()
};
```

### Observability events

Get real-time events during tool loop execution:

```rust
use llm_stack::{ToolLoopConfig, ToolLoopEvent};
use std::sync::Arc;

fn log_event(event: &ToolLoopEvent) {
    match event {
        ToolLoopEvent::IterationStart { iteration, message_count } => {
            println!("Starting iteration {iteration} ({message_count} messages)");
        }
        ToolLoopEvent::ToolExecutionStart { tool_name, .. } => {
            println!("Calling {tool_name}...");
        }
        ToolLoopEvent::ToolExecutionEnd { tool_name, duration, result, .. } => {
            let status = if result.is_error { "failed" } else { "completed" };
            println!("{tool_name} {status} in {duration:?}");
        }
        ToolLoopEvent::LlmResponseReceived { iteration, has_tool_calls, .. } => {
            if *has_tool_calls {
                println!("Iteration {iteration}: model requested tools");
            } else {
                println!("Iteration {iteration}: model finished");
            }
        }
        ToolLoopEvent::LoopDetected { tool_name, consecutive_count, .. } => {
            println!("Loop detected: {tool_name} called {consecutive_count} times");
        }
    }
}

let config = ToolLoopConfig {
    on_event: Some(Arc::new(log_event)),
    ..Default::default()
};
```

## Streaming tool loops

For real-time output, use `tool_loop_stream`:

```rust
use llm_stack::tool_loop_stream;
use futures::StreamExt;
use std::sync::Arc;

let provider = Arc::new(provider);
let registry = Arc::new(registry);
let ctx = Arc::new(());

let stream = tool_loop_stream(provider, registry, params, config, ctx);

futures::pin_mut!(stream);
while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta(text) => print!("{text}"),
        StreamEvent::ToolCallStart { name, .. } => println!("\n[Calling {name}...]"),
        StreamEvent::Done { .. } => break,
        _ => {}
    }
}
```

### Channel-based streaming with backpressure

For slow consumers, use `tool_loop_channel` to prevent unbounded memory growth:

```rust
use llm_stack::tool_loop_channel;

let (mut rx, handle) = tool_loop_channel(
    provider,
    registry,
    params,
    config,
    ctx,
    32,  // Buffer size
);

// Consume events at your own pace
while let Some(event) = rx.recv().await {
    match event? {
        StreamEvent::TextDelta(text) => {
            // Slow processing is fine — producer blocks when buffer is full
            send_to_client(text).await;
        }
        StreamEvent::Done { .. } => break,
        _ => {}
    }
}

// Get the final result
let result = handle.await?;
```

## Resumable tool loop

The standard `tool_loop` runs to completion autonomously. For orchestration patterns where you need control between iterations — multi-agent systems, external event injection, context compaction — use `ToolLoopHandle`:

```rust
use llm_stack::{
    ChatMessage, ChatParams, ToolRegistry, ToolLoopConfig,
    ToolLoopHandle, TurnResult,
};

let registry: ToolRegistry<()> = ToolRegistry::new();
// ... register tools ...

let params = ChatParams {
    messages: vec![ChatMessage::user("Analyze this dataset")],
    tools: Some(registry.definitions()),
    ..Default::default()
};

let mut handle = ToolLoopHandle::new(
    provider, &registry, params, ToolLoopConfig::default(), &(),
);

loop {
    match handle.next_turn().await {
        TurnResult::Yielded(turn) => {
            println!("Iteration {}: {} tools executed",
                turn.iteration, turn.tool_calls.len());

            // Text the LLM produced alongside tool calls is directly available
            if let Some(text) = turn.assistant_text() {
                println!("LLM said: {text}");
            }

            // Inspect results and decide what to do
            if turn.results.iter().any(|r| r.content.contains("DONE")) {
                turn.stop(Some("Task complete".into()));
            } else {
                turn.continue_loop();
            }
        }
        TurnResult::Completed(done) => {
            println!("Done: {:?}", done.termination_reason);
            break;
        }
        TurnResult::Error(err) => {
            eprintln!("Error: {}", err.error);
            break;
        }
    }
}

let result = handle.into_result();
```

### How it works

`ToolLoopHandle` is a direct state machine — no background tasks, no channels, no spawning. The caller drives it:

1. **`next_turn()`** — performs one iteration (LLM call + tool execution) and returns a `TurnResult`
2. **Match on the result** — `Yielded` gives you tools, results, and text; `Completed` and `Error` are terminal
3. **Consume `Yielded`** — call `.continue_loop()`, `.inject_and_continue(msgs)`, `.stop(reason)`, or `.resume(command)` to proceed
4. Repeat until `Completed` or `Error`

The `Yielded` variant borrows the handle mutably — you literally cannot call `next_turn()` again until you consume it. This prevents the common bug of forgetting to resume.

### TurnResult variants

| Variant | What happened | What to do |
|---------|---------------|------------|
| `Yielded(turn)` | Tools were executed | Consume via `continue_loop()`, `inject_and_continue()`, `stop()`, or `resume()` |
| `Completed(done)` | Loop finished | Read `done.response`, `done.termination_reason` |
| `Error(err)` | LLM call failed | Read `err.error` |

### Yielded — the interesting one

`Yielded` carries everything from the turn:

- `turn.tool_calls` — what the LLM requested
- `turn.results` — what the tools returned
- `turn.assistant_content` — text/reasoning/images the LLM produced alongside tool calls
- `turn.assistant_text()` — convenience to extract just the text
- `turn.iteration` — which iteration this was
- `turn.total_usage` — accumulated token usage
- `turn.messages()` / `turn.messages_mut()` — full conversation history

### Injecting messages

Add context between iterations — worker results, user follow-ups, system guidance:

```rust
TurnResult::Yielded(turn) => {
    let worker_result = run_worker_agent().await;
    turn.inject_and_continue(vec![
        ChatMessage::user(format!("Worker completed: {worker_result}")),
    ]);
}
```

### Context compaction

Access and modify the conversation history through the `Yielded` handle:

```rust
TurnResult::Yielded(mut turn) => {
    let messages = turn.messages_mut();
    if messages.len() > 50 {
        // Summarize and compact old messages
        let summary = summarize(&messages[..messages.len() - 10]).await;
        messages.drain(1..messages.len() - 10);
        messages.insert(1, ChatMessage::user(summary));
    }
    turn.continue_loop();
}
```

### Inspecting state

The handle exposes loop state between iterations:

- `handle.messages()` — current conversation
- `handle.messages_mut()` — mutable conversation access
- `handle.total_usage()` — accumulated token usage
- `handle.iterations()` — current iteration count
- `handle.is_finished()` — whether a terminal event was returned
- `handle.into_result()` — consume handle into a `ToolLoopResult`

### Convenience constructor

`tool_loop_resumable()` is an alias for `ToolLoopHandle::new()`:

```rust
use llm_stack::tool_loop_resumable;

let mut handle = tool_loop_resumable(provider, &registry, params, config, &ctx);
```

All `ToolLoopConfig` options (max_iterations, timeout, stop_when, loop_detection, approval hooks, on_event) work identically to `tool_loop`.

## Tool results

The `tool_loop` returns comprehensive results:

```rust
let result = tool_loop(provider, &registry, params, config, &ctx).await?;

// The model's final text response
println!("{}", result.response.text().unwrap_or_default());

// Total token usage across all iterations
println!("Total tokens: {}", result.total_usage.input_tokens + result.total_usage.output_tokens);

// How many tool call rounds occurred
println!("Iterations: {}", result.iterations);

// Why the loop terminated
match result.termination_reason {
    TerminationReason::Complete => println!("Normal completion"),
    TerminationReason::MaxIterations { limit } => println!("Hit {limit} iteration limit"),
    TerminationReason::Timeout { limit } => println!("Exceeded {limit:?} timeout"),
    TerminationReason::StopCondition { reason } => println!("Custom stop: {reason:?}"),
    TerminationReason::LoopDetected { tool_name, count } => {
        println!("Stuck calling {tool_name} {count} times");
    }
}
```

## Per-tool retry configuration

Configure automatic retries for unreliable tools (HTTP calls, flaky APIs):

```rust
use llm_stack::{ToolDefinition, ToolRetryConfig, JsonSchema};
use std::time::Duration;
use std::sync::Arc;

let http_tool = ToolDefinition {
    name: "fetch_url".into(),
    description: "Fetch content from a URL".into(),
    parameters: JsonSchema::new(json!({
        "type": "object",
        "properties": { "url": { "type": "string" } },
        "required": ["url"]
    })),
    retry: Some(ToolRetryConfig {
        max_retries: 3,
        initial_backoff: Duration::from_millis(100),
        max_backoff: Duration::from_secs(5),
        backoff_multiplier: 2.0,
        jitter: 0.5,
        // Only retry timeout errors
        retry_if: Some(Arc::new(|err_msg| err_msg.contains("timeout"))),
    }),
};
```

Defaults: 3 retries, 100ms initial backoff, 5s max, 2x multiplier, 0.5 jitter.

## Error handling

Tool execution errors are returned to the model, which can retry or explain the failure:

```rust
use llm_stack::ToolError;

// In your handler
Err(ToolError::new("Database connection failed"))
```

The model might respond: "I encountered an error accessing the database. Let me try a different approach..."

## MCP (Model Context Protocol) integration

MCP servers expose tools over a standard protocol. Register MCP tools alongside native tools using the `McpService` trait.

### The design

llm-stack defines a minimal trait — you implement it for your MCP client:

```rust
use llm_stack::{McpService, McpError, ToolDefinition};
use std::pin::Pin;
use std::future::Future;

pub trait McpService: Send + Sync {
    fn list_tools(&self) -> Pin<Box<dyn Future<Output = Result<Vec<ToolDefinition>, McpError>> + Send + '_>>;
    fn call_tool(&self, name: &str, args: serde_json::Value) -> Pin<Box<dyn Future<Output = Result<String, McpError>> + Send + '_>>;
}
```

This keeps llm-stack dependency-free. You choose your MCP client library and version.

### Registering MCP tools

Once you have an `McpService` implementation:

```rust
use std::sync::Arc;
use llm_stack::{ToolRegistry, McpRegistryExt};

let mcp_service = Arc::new(my_mcp_adapter);

let mut registry: ToolRegistry<()> = ToolRegistry::new();
registry.register_mcp_service(&mcp_service).await?;

// Or register specific tools only
registry.register_mcp_tools_by_name(&mcp_service, &["read_file", "write_file"]).await?;

// Mix with native tools
registry.register(my_native_tool);
```

## Nested tool loops (Agent hierarchies)

Tools can call `tool_loop` internally, enabling Master/Worker patterns where one agent delegates to sub-agents. The SDK provides primitives for this:

### Depth tracking with LoopDepth

Implement `LoopDepth` on your context type to enable automatic depth tracking:

```rust
use llm_stack::tool::LoopDepth;

#[derive(Clone)]
struct AgentContext {
    user_id: String,
    db: Arc<DatabasePool>,
    depth: u32,  // Tracks nesting level
}

impl LoopDepth for AgentContext {
    fn loop_depth(&self) -> u32 { self.depth }
    fn with_depth(&self, depth: u32) -> Self {
        Self { depth, ..self.clone() }
    }
}
```

The `tool_loop` automatically increments depth when passing context to tool handlers.

### Depth limits

Prevent runaway recursion with `max_depth`:

```rust
let config = ToolLoopConfig {
    max_depth: Some(2),  // Master=0, Worker=1, no grandchildren
    ..Default::default()
};

// At depth 2+, tool_loop returns Err(LlmError::MaxDepthExceeded)
```

Default is `Some(3)`. Set to `None` for unlimited nesting (dangerous).

### Registry scoping

Create limited registries for sub-agents:

```rust
// Master has all tools
let mut master_registry: ToolRegistry<AgentContext> = ToolRegistry::new();
master_registry.register(search_tool);
master_registry.register(code_tool);
master_registry.register(spawn_task_tool);

// Workers can't spawn (no spawn_task)
let worker_registry = master_registry.without(["spawn_task"]);

// Or create a minimal read-only registry
let reader_registry = master_registry.only(["search_docs", "read_file"]);
```

### Master/Worker pattern

A tool that spawns a focused sub-agent:

```rust
struct SpawnTaskTool {
    provider: Arc<dyn DynProvider>,
    worker_registry: Arc<ToolRegistry<AgentContext>>,
}

impl ToolHandler<AgentContext> for SpawnTaskTool {
    fn execute<'a>(
        &'a self,
        input: Value,
        ctx: &'a AgentContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let task = input["task"].as_str()
                .ok_or_else(|| ToolError::new("missing task"))?;

            let params = ChatParams {
                system: Some("You are a focused worker. Complete the task.".into()),
                messages: vec![ChatMessage::user(task)],
                tools: Some(self.worker_registry.definitions()),
                ..Default::default()
            };

            // Workers have stricter limits
            let config = ToolLoopConfig {
                max_iterations: 10,
                max_depth: Some(2),  // Workers can't spawn grandchildren
                ..Default::default()
            };

            // Context depth is incremented automatically
            let result = tool_loop(&*self.provider, &self.worker_registry, params, config, ctx)
                .await
                .map_err(|e| ToolError::new(e.to_string()))?;

            Ok(ToolOutput::new(result.response.text().unwrap_or("Done").to_string()))
        })
    }
}
```

### Simple contexts

If you don't need depth tracking, use `()` as context — it has a blanket `LoopDepth` impl that always returns 0:

```rust
// Works without implementing LoopDepth
let result = tool_loop(provider, &registry, params, config, &()).await?;
```

## Best practices

1. **Write clear descriptions** — The model uses them to decide when to call tools
2. **Use `tool_fn`** — Simpler than implementing the trait directly
3. **Clone context data** — Before async blocks when using `tool_fn_with_ctx`
4. **Validate arguments** — Even with schema validation, check for edge cases
5. **Return structured data** — JSON or clear text the model can parse
6. **Handle errors gracefully** — Return helpful error messages via `ToolError`
7. **Set reasonable limits** — Use `max_iterations` and `timeout` to prevent runaway loops
8. **Use approval hooks** — For tools with side effects (file writes, API calls)
9. **Configure retries** — For unreliable external services
10. **Monitor with events** — Use `on_event` for logging and metrics
