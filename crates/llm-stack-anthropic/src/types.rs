//! Anthropic Messages API request and response types.
//!
//! These types mirror Anthropic's wire format and are not part of the
//! public API. Conversion to/from `llm-core` types happens in
//! [`convert`](crate::convert).

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Request types ──────────────────────────────────────────────────

/// Top-level request body for `POST /v1/messages`.
#[derive(Debug, Serialize)]
pub(crate) struct Request<'a> {
    pub model: &'a str,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

/// A single message in the conversation.
#[derive(Debug, Serialize)]
pub(crate) struct Message {
    pub role: &'static str,
    pub content: Vec<ContentBlock>,
}

/// A content block within a message.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(crate) enum ContentBlock {
    /// Plain text content.
    #[serde(rename = "text")]
    Text { text: String },
    /// Inline image (base64).
    #[serde(rename = "image")]
    Image { source: ImageSource },
    /// A tool invocation (sent in assistant messages).
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// A tool result (sent in user messages).
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

/// Base64-encoded image source for the API.
#[derive(Debug, Serialize)]
pub(crate) struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: &'static str,
    pub media_type: String,
    pub data: String,
}

/// Tool definition sent in the request.
#[derive(Debug, Serialize)]
pub(crate) struct Tool<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub input_schema: &'a Value,
}

/// Tool choice parameter.
#[derive(Debug, Serialize)]
pub(crate) struct ToolChoiceParam {
    #[serde(rename = "type")]
    pub choice_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Extended thinking configuration.
#[derive(Debug, Serialize)]
pub(crate) struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: &'static str,
    pub budget_tokens: u32,
}

// ── Response types ─────────────────────────────────────────────────

/// Top-level response from `POST /v1/messages`.
#[derive(Debug, Deserialize)]
pub(crate) struct Response {
    pub content: Vec<ResponseContent>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: ResponseUsage,
}

/// A content block in the response.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponseContent {
    #[serde(rename = "type")]
    pub content_type: String,
    /// Text content (for `type: "text"`).
    pub text: Option<String>,
    /// Thinking content (for `type: "thinking"`).
    pub thinking: Option<String>,
    /// Tool use ID (for `type: "tool_use"`).
    pub id: Option<String>,
    /// Tool name (for `type: "tool_use"`).
    pub name: Option<String>,
    /// Tool input JSON (for `type: "tool_use"`).
    pub input: Option<Value>,
}

/// Token usage in the response.
///
/// Field names match Anthropic API exactly — cannot rename.
#[derive(Debug, Deserialize)]
#[allow(clippy::struct_field_names)]
pub(crate) struct ResponseUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
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

/// A single SSE event from the streaming API.
#[derive(Debug, Deserialize)]
pub(crate) struct StreamResponse {
    #[serde(rename = "type")]
    pub event_type: String,
    /// Content block index (for `content_block_*` events).
    pub index: Option<u32>,
    /// Content block (for `content_block_start`).
    pub content_block: Option<StreamContentBlock>,
    /// Delta (for `content_block_delta` and `message_delta`).
    pub delta: Option<StreamDelta>,
    /// Usage info (for `message_start`).
    pub message: Option<StreamMessage>,
    /// Usage info (for `message_delta`).
    pub usage: Option<ResponseUsage>,
}

/// Content block within a `content_block_start` event.
#[derive(Debug, Deserialize)]
pub(crate) struct StreamContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub id: Option<String>,
    pub name: Option<String>,
}

/// Delta content within streaming events.
#[derive(Debug, Deserialize)]
pub(crate) struct StreamDelta {
    #[serde(rename = "type")]
    pub delta_type: Option<String>,
    pub text: Option<String>,
    pub thinking: Option<String>,
    pub partial_json: Option<String>,
    pub stop_reason: Option<String>,
}

/// Message metadata from `message_start` events.
#[derive(Debug, Deserialize)]
pub(crate) struct StreamMessage {
    pub usage: Option<ResponseUsage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization_minimal() {
        let req = Request {
            model: "claude-3-5-haiku-20241022",
            messages: vec![Message {
                role: "user",
                content: vec![ContentBlock::Text {
                    text: "Hello".into(),
                }],
            }],
            max_tokens: 1024,
            temperature: None,
            system: None,
            stream: None,
            tools: None,
            tool_choice: None,
            thinking: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "claude-3-5-haiku-20241022");
        assert_eq!(json["max_tokens"], 1024);
        assert!(json.get("temperature").is_none());
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn test_request_serialization_with_tools() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        });
        let req = Request {
            model: "claude-3-5-haiku-20241022",
            messages: vec![],
            max_tokens: 1024,
            temperature: Some(0.7),
            system: Some("You are helpful."),
            stream: Some(false),
            tools: Some(vec![Tool {
                name: "get_weather",
                description: "Get weather for a city",
                input_schema: &schema,
            }]),
            tool_choice: Some(ToolChoiceParam {
                choice_type: "auto".into(),
                name: None,
            }),
            thinking: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["tools"][0]["name"], "get_weather");
        assert_eq!(json["tool_choice"]["type"], "auto");
    }

    #[test]
    fn test_response_deserialization() {
        let json = serde_json::json!({
            "id": "msg_123",
            "content": [
                { "type": "text", "text": "Hello!" }
            ],
            "model": "claude-3-5-haiku-20241022",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        });
        let resp: Response = serde_json::from_value(json).unwrap();
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.content[0].text.as_deref(), Some("Hello!"));
        assert_eq!(resp.model, "claude-3-5-haiku-20241022");
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
    }

    #[test]
    fn test_response_with_tool_use() {
        let json = serde_json::json!({
            "id": "msg_456",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01ABC",
                    "name": "get_weather",
                    "input": { "city": "Tokyo" }
                }
            ],
            "model": "claude-3-5-haiku-20241022",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 50,
                "output_tokens": 30
            }
        });
        let resp: Response = serde_json::from_value(json).unwrap();
        assert_eq!(resp.content[0].content_type, "tool_use");
        assert_eq!(resp.content[0].name.as_deref(), Some("get_weather"));
        assert_eq!(resp.content[0].input.as_ref().unwrap()["city"], "Tokyo");
    }

    #[test]
    fn test_response_with_thinking() {
        let json = serde_json::json!({
            "id": "msg_789",
            "content": [
                { "type": "thinking", "thinking": "Let me reason about this..." },
                { "type": "text", "text": "The answer is 42." }
            ],
            "model": "claude-3-5-haiku-20241022",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            }
        });
        let resp: Response = serde_json::from_value(json).unwrap();
        assert_eq!(resp.content.len(), 2);
        assert_eq!(resp.content[0].content_type, "thinking");
        assert_eq!(
            resp.content[0].thinking.as_deref(),
            Some("Let me reason about this...")
        );
    }

    #[test]
    fn test_response_with_cache_tokens() {
        let json = serde_json::json!({
            "id": "msg_cache",
            "content": [{ "type": "text", "text": "ok" }],
            "model": "claude-3-5-haiku-20241022",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 30
            }
        });
        let resp: Response = serde_json::from_value(json).unwrap();
        assert_eq!(resp.usage.cache_creation_input_tokens, Some(50));
        assert_eq!(resp.usage.cache_read_input_tokens, Some(30));
    }

    #[test]
    fn test_error_response_deserialization() {
        let json = serde_json::json!({
            "error": {
                "type": "authentication_error",
                "message": "Invalid API key"
            }
        });
        let err: ErrorResponse = serde_json::from_value(json).unwrap();
        assert_eq!(err.error.message, "Invalid API key");
    }

    #[test]
    fn test_tool_result_content_block_serialization() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "toolu_01ABC".into(),
            content: "Weather: sunny, 25C".into(),
            is_error: false,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_result");
        assert_eq!(json["tool_use_id"], "toolu_01ABC");
        assert!(json.get("is_error").is_none()); // skipped when false
    }

    #[test]
    fn test_tool_result_error_serialization() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "toolu_01ABC".into(),
            content: "Connection refused".into(),
            is_error: true,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["is_error"], true);
    }

    #[test]
    fn test_content_block_text_serialization() {
        let block = ContentBlock::Text {
            text: "Hello".into(),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "Hello");
    }

    #[test]
    fn test_content_block_tool_use_serialization() {
        let block = ContentBlock::ToolUse {
            id: "toolu_01".into(),
            name: "search".into(),
            input: serde_json::json!({"query": "rust"}),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_use");
        assert_eq!(json["id"], "toolu_01");
        assert_eq!(json["name"], "search");
        assert_eq!(json["input"]["query"], "rust");
    }
}
