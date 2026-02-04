//! Ollama Chat API request and response types.
//!
//! These types mirror Ollama's wire format and are not part of the
//! public API. Conversion to/from `llm-core` types happens in
//! [`convert`](crate::convert).

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Request types ──────────────────────────────────────────────────

/// Top-level request body for `POST /api/chat`.
#[derive(Debug, Serialize)]
pub(crate) struct Request {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Options>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
}

/// A single message in the conversation.
#[derive(Debug, Serialize)]
pub(crate) struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallRequest>>,
}

/// Generation options (temperature, `top_p`, etc.).
#[derive(Debug, Serialize)]
pub(crate) struct Options {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
}

/// Tool call in an assistant message (outgoing).
#[derive(Debug, Serialize)]
pub(crate) struct ToolCallRequest {
    pub function: FunctionCallRequest,
}

/// Function call details for outgoing messages.
#[derive(Debug, Serialize)]
pub(crate) struct FunctionCallRequest {
    pub name: String,
    pub arguments: Value,
}

/// Tool definition sent in the request.
#[derive(Debug, Serialize)]
pub(crate) struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

/// Function tool definition.
#[derive(Debug, Serialize)]
pub(crate) struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

// ── Response types ─────────────────────────────────────────────────

/// Top-level response from `POST /api/chat` (non-streaming).
#[derive(Debug, Deserialize)]
pub(crate) struct Response {
    pub message: Option<ResponseMessage>,
    pub model: Option<String>,
    /// Reason the generation stopped (e.g. `"stop"`, `"length"`).
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
}

/// Message within a response.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponseMessage {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallResponse>>,
}

/// Tool call in a response.
#[derive(Debug, Deserialize)]
pub(crate) struct ToolCallResponse {
    pub function: FunctionCallResponse,
}

/// Function call details in a response.
#[derive(Debug, Deserialize)]
pub(crate) struct FunctionCallResponse {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

// ── Error types ────────────────────────────────────────────────────

/// Error response body from the API.
#[derive(Debug, Deserialize)]
pub(crate) struct ErrorResponse {
    pub error: String,
}

// ── Streaming types ────────────────────────────────────────────────

/// A single JSON line from the streaming API.
///
/// In Ollama streaming, each line is a standalone JSON object
/// (JSON Lines format, not SSE).
#[derive(Debug, Deserialize)]
pub(crate) struct StreamChunk {
    pub message: Option<ResponseMessage>,
    pub done: Option<bool>,
    /// Reason the generation stopped (e.g. `"stop"`, `"length"`).
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization_minimal() {
        let req = Request {
            model: "llama3.2".into(),
            messages: vec![Message {
                role: "user".into(),
                content: "Hello".into(),
                images: None,
                tool_calls: None,
            }],
            stream: false,
            options: None,
            tools: None,
            format: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "llama3.2");
        assert_eq!(json["stream"], false);
        assert!(json.get("options").is_none());
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn test_request_with_options() {
        let req = Request {
            model: "llama3.2".into(),
            messages: vec![],
            stream: true,
            options: Some(Options {
                temperature: Some(0.7),
                num_predict: Some(100),
            }),
            tools: None,
            format: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        let temp = json["options"]["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.001, "Expected ~0.7, got {temp}");
        assert_eq!(json["options"]["num_predict"], 100);
    }

    #[test]
    fn test_request_with_tools() {
        let req = Request {
            model: "llama3.2".into(),
            messages: vec![],
            stream: false,
            options: None,
            tools: Some(vec![Tool {
                tool_type: "function".into(),
                function: FunctionDef {
                    name: "get_weather".into(),
                    description: "Get weather for a city".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"]
                    }),
                },
            }]),
            format: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn test_response_deserialization() {
        let json = serde_json::json!({
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "Hello!"
            },
            "done": true,
            "prompt_eval_count": 10,
            "eval_count": 5
        });
        let resp: Response = serde_json::from_value(json).unwrap();
        assert_eq!(
            resp.message.as_ref().unwrap().content.as_deref(),
            Some("Hello!")
        );
        assert_eq!(resp.prompt_eval_count, Some(10));
        assert_eq!(resp.eval_count, Some(5));
    }

    #[test]
    fn test_response_with_tool_calls() {
        let json = serde_json::json!({
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Tokyo"}
                    }
                }]
            },
            "done": true
        });
        let resp: Response = serde_json::from_value(json).unwrap();
        let tc = &resp.message.unwrap().tool_calls.unwrap()[0];
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(tc.function.arguments["city"], "Tokyo");
    }

    #[test]
    fn test_error_response_deserialization() {
        let json = serde_json::json!({
            "error": "model not found"
        });
        let err: ErrorResponse = serde_json::from_value(json).unwrap();
        assert_eq!(err.error, "model not found");
    }

    #[test]
    fn test_stream_chunk_deserialization() {
        let json = serde_json::json!({
            "message": { "content": "Hello" },
            "done": false
        });
        let chunk: StreamChunk = serde_json::from_value(json).unwrap();
        assert_eq!(
            chunk.message.as_ref().unwrap().content.as_deref(),
            Some("Hello")
        );
        assert_eq!(chunk.done, Some(false));
    }

    #[test]
    fn test_stream_chunk_final() {
        let json = serde_json::json!({
            "message": { "content": "" },
            "done": true,
            "prompt_eval_count": 42,
            "eval_count": 10
        });
        let chunk: StreamChunk = serde_json::from_value(json).unwrap();
        assert_eq!(chunk.done, Some(true));
        assert_eq!(chunk.prompt_eval_count, Some(42));
        assert_eq!(chunk.eval_count, Some(10));
    }

    #[test]
    fn test_message_with_images() {
        let msg = Message {
            role: "user".into(),
            content: "What's in this image?".into(),
            images: Some(vec!["base64data...".into()]),
            tool_calls: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["images"][0], "base64data...");
    }

    #[test]
    fn test_request_with_format() {
        let req = Request {
            model: "llama3.2".into(),
            messages: vec![],
            stream: false,
            options: None,
            tools: None,
            format: Some(serde_json::json!("json")),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["format"], "json");
    }
}
