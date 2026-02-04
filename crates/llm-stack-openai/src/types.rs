//! `OpenAI` Chat Completions API request and response types.
//!
//! These types mirror `OpenAI`'s wire format and are not part of the
//! public API. Conversion to/from `llm-core` types happens in
//! [`convert`](crate::convert).

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Request types ──────────────────────────────────────────────────

/// Top-level request body for `POST /chat/completions`.
#[derive(Debug, Serialize)]
pub(crate) struct Request<'a> {
    pub model: &'a str,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat<'a>>,
}

/// A single message in the conversation.
#[derive(Debug, Serialize)]
pub(crate) struct Message {
    pub role: &'static str,
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Message content — either a simple string or an array of content parts.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Array of typed content parts (text, images, etc.).
    Parts(Vec<ContentPart>),
}

/// A typed content part within a message.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(crate) enum ContentPart {
    /// Plain text.
    #[serde(rename = "text")]
    Text { text: String },
    /// Image via URL or base64 data URL.
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// Image URL for the `OpenAI` API.
#[derive(Debug, Serialize)]
pub(crate) struct ImageUrl {
    pub url: String,
}

/// Tool call in an assistant message (outgoing).
#[derive(Debug, Serialize)]
pub(crate) struct ToolCallRequest {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: &'static str,
    pub function: FunctionCallRequest,
}

/// Function call details.
#[derive(Debug, Serialize)]
pub(crate) struct FunctionCallRequest {
    pub name: String,
    /// JSON string of the arguments.
    pub arguments: String,
}

/// Tool definition sent in the request.
#[derive(Debug, Serialize)]
pub(crate) struct Tool<'a> {
    #[serde(rename = "type")]
    pub tool_type: &'static str,
    pub function: FunctionDef<'a>,
}

/// Function tool definition.
#[derive(Debug, Serialize)]
pub(crate) struct FunctionDef<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub parameters: &'a Value,
}

/// Stream options to request usage in the final chunk.
#[derive(Debug, Serialize)]
pub(crate) struct StreamOptions {
    pub include_usage: bool,
}

/// Response format for structured output.
#[derive(Debug, Serialize)]
pub(crate) struct ResponseFormat<'a> {
    #[serde(rename = "type")]
    pub format_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchemaFormat<'a>>,
}

/// JSON schema format for structured output.
#[derive(Debug, Serialize)]
pub(crate) struct JsonSchemaFormat<'a> {
    pub name: &'static str,
    pub schema: &'a Value,
    pub strict: bool,
}

// ── Response types ─────────────────────────────────────────────────

/// Top-level response from `POST /chat/completions`.
#[derive(Debug, Deserialize)]
pub(crate) struct Response {
    pub choices: Vec<Choice>,
    pub model: String,
    pub usage: Option<ResponseUsage>,
}

/// A single choice in the response.
#[derive(Debug, Deserialize)]
pub(crate) struct Choice {
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

/// Message within a response choice.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponseMessage {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallResponse>>,
}

/// Tool call in a response.
#[derive(Debug, Deserialize)]
pub(crate) struct ToolCallResponse {
    pub id: String,
    pub function: FunctionCallResponse,
}

/// Function call details in a response.
#[derive(Debug, Deserialize)]
pub(crate) struct FunctionCallResponse {
    pub name: String,
    pub arguments: String,
}

/// Token usage in the response.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponseUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Detailed breakdown of completion tokens.
#[derive(Debug, Deserialize)]
pub(crate) struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

// ── Error types ────────────────────────────────────────────────────

/// Error response body from the API.
#[derive(Debug, Deserialize)]
pub(crate) struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error detail within an error response.
#[derive(Debug, Deserialize)]
pub(crate) struct ErrorDetail {
    pub message: String,
}

// ── Streaming types ────────────────────────────────────────────────

/// A single SSE chunk from the streaming API.
#[derive(Debug, Deserialize)]
pub(crate) struct StreamChunk {
    pub choices: Vec<StreamChoice>,
    pub usage: Option<ResponseUsage>,
}

/// A choice within a streaming chunk.
#[derive(Debug, Deserialize)]
pub(crate) struct StreamChoice {
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

/// Delta content within a streaming chunk.
#[derive(Debug, Deserialize)]
pub(crate) struct StreamDelta {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<StreamToolCall>>,
}

/// Tool call delta in a streaming chunk.
#[derive(Debug, Deserialize)]
pub(crate) struct StreamToolCall {
    pub index: u32,
    pub id: Option<String>,
    pub function: Option<StreamFunctionCall>,
}

/// Function call delta in a streaming chunk.
#[derive(Debug, Deserialize)]
pub(crate) struct StreamFunctionCall {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization_minimal() {
        let req = Request {
            model: "gpt-4o",
            messages: vec![Message {
                role: "user",
                content: Some(MessageContent::Text("Hello".into())),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_completion_tokens: None,
            stream: None,
            stream_options: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "gpt-4o");
        assert!(json.get("temperature").is_none());
        assert!(json.get("tools").is_none());
        assert!(json.get("stream").is_none());
    }

    #[test]
    fn test_request_with_tools() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        });
        let req = Request {
            model: "gpt-4o",
            messages: vec![],
            temperature: Some(0.7),
            max_completion_tokens: Some(1024),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            tools: Some(vec![Tool {
                tool_type: "function",
                function: FunctionDef {
                    name: "get_weather",
                    description: "Get weather for a city",
                    parameters: &schema,
                },
            }]),
            tool_choice: Some(serde_json::json!("auto")),
            response_format: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(json["stream_options"]["include_usage"], true);
    }

    #[test]
    fn test_response_deserialization() {
        let json = serde_json::json!({
            "id": "chatcmpl-123",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "model": "gpt-4o-2024-08-06",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });
        let resp: Response = serde_json::from_value(json).unwrap();
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.as_ref().unwrap().prompt_tokens, 10);
    }

    #[test]
    fn test_response_with_tool_calls() {
        let json = serde_json::json!({
            "id": "chatcmpl-456",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Tokyo\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70
            }
        });
        let resp: Response = serde_json::from_value(json).unwrap();
        let tc = &resp.choices[0].message.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tc.id, "call_abc");
        assert_eq!(tc.function.name, "get_weather");
    }

    #[test]
    fn test_response_with_reasoning_tokens() {
        let json = serde_json::json!({
            "id": "chatcmpl-789",
            "choices": [{
                "message": { "role": "assistant", "content": "42" },
                "finish_reason": "stop"
            }],
            "model": "o1-mini",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 100,
                "total_tokens": 110,
                "completion_tokens_details": {
                    "reasoning_tokens": 80
                }
            }
        });
        let resp: Response = serde_json::from_value(json).unwrap();
        let usage = resp.usage.unwrap();
        assert_eq!(
            usage.completion_tokens_details.unwrap().reasoning_tokens,
            Some(80)
        );
    }

    #[test]
    fn test_error_response_deserialization() {
        let json = serde_json::json!({
            "error": {
                "message": "Invalid API key",
                "type": "invalid_api_key",
                "code": "invalid_api_key"
            }
        });
        let err: ErrorResponse = serde_json::from_value(json).unwrap();
        assert_eq!(err.error.message, "Invalid API key");
    }

    #[test]
    fn test_stream_chunk_deserialization() {
        let json = serde_json::json!({
            "id": "chatcmpl-123",
            "choices": [{
                "delta": { "content": "Hello" },
                "finish_reason": null
            }]
        });
        let chunk: StreamChunk = serde_json::from_value(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn test_stream_tool_call_deserialization() {
        let json = serde_json::json!({
            "id": "chatcmpl-456",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": ""
                        }
                    }]
                },
                "finish_reason": null
            }]
        });
        let chunk: StreamChunk = serde_json::from_value(json).unwrap();
        let tc = &chunk.choices[0].delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tc.index, 0);
        assert_eq!(tc.id.as_deref(), Some("call_abc"));
    }

    #[test]
    fn test_message_content_text_serialization() {
        let msg = Message {
            role: "user",
            content: Some(MessageContent::Text("Hello".into())),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["content"], "Hello");
    }

    #[test]
    fn test_message_content_parts_serialization() {
        let msg = Message {
            role: "user",
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "What's in this image?".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,abc123".into(),
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][1]["type"], "image_url");
    }

    #[test]
    fn test_response_format_json_schema() {
        let schema = serde_json::json!({"type": "object"});
        let rf = ResponseFormat {
            format_type: "json_schema",
            json_schema: Some(JsonSchemaFormat {
                name: "output",
                schema: &schema,
                strict: true,
            }),
        };
        let json = serde_json::to_value(&rf).unwrap();
        assert_eq!(json["type"], "json_schema");
        assert_eq!(json["json_schema"]["strict"], true);
    }
}
