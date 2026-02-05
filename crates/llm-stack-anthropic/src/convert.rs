//! Bidirectional conversion between `llm-core` types and Anthropic API types.
//!
//! This module is internal — callers interact only with `llm-core` types.
//! The provider implementation uses these functions to build requests and
//! parse responses.

use llm_stack::chat::{
    ChatMessage, ChatResponse, ChatRole, ContentBlock as CoreContent, ImageSource as CoreImage,
    StopReason, ToolCall,
};
use llm_stack::error::LlmError;
use llm_stack::provider::{ChatParams, ToolChoice};
use llm_stack::usage::Usage;
use serde_json::Value;
use std::collections::HashMap;

use crate::config::AnthropicConfig;
use crate::types::{
    ContentBlock, ErrorResponse, ImageSource, Message, Request, ThinkingConfig, Tool,
    ToolChoiceParam,
};

// ── Request conversion ───────────────────────────────────────────────

/// Build an Anthropic API request from `ChatParams` and provider config.
pub(crate) fn build_request<'a>(
    params: &'a ChatParams,
    config: &'a AnthropicConfig,
    stream: bool,
) -> Result<Request<'a>, LlmError> {
    let messages = convert_messages(&params.messages)?;
    let system = params.system.as_deref();
    let max_tokens = params.max_tokens.unwrap_or(config.max_tokens);
    let tools = params.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|t| Tool {
                name: &t.name,
                description: &t.description,
                input_schema: t.parameters.as_value(),
            })
            .collect()
    });
    let tool_choice = params.tool_choice.as_ref().map(convert_tool_choice);
    let thinking = params.reasoning_budget.map(|budget| ThinkingConfig {
        thinking_type: "enabled",
        budget_tokens: budget,
    });

    Ok(Request {
        model: &config.model,
        messages,
        max_tokens,
        temperature: params.temperature,
        system,
        stream: if stream { Some(true) } else { None },
        tools,
        tool_choice,
        thinking,
    })
}

/// Convert core messages to Anthropic message format.
///
/// System messages are handled separately via the top-level `system` param,
/// so they are filtered out here. Tool role messages are mapped to "user"
/// role with `ToolResult` content blocks.
fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<Message>, LlmError> {
    messages
        .iter()
        .filter(|m| m.role != ChatRole::System)
        .map(|m| {
            let content: Result<Vec<ContentBlock>, LlmError> =
                m.content.iter().map(try_convert_content_block).collect();
            Ok(Message {
                role: match m.role {
                    ChatRole::Assistant => "assistant",
                    // System is filtered above; Tool and any future roles map to "user"
                    _ => "user",
                },
                content: content?,
            })
        })
        .collect()
}

/// Convert a single core content block to an Anthropic content block.
///
/// Returns `Err` if the content block cannot be represented in the Anthropic
/// API (e.g. URL-based images, which require pre-fetching by the caller).
fn try_convert_content_block(block: &CoreContent) -> Result<ContentBlock, LlmError> {
    match block {
        CoreContent::Text(text) => Ok(ContentBlock::Text { text: text.clone() }),
        CoreContent::Image { media_type, data } => match data {
            CoreImage::Base64(b64) => Ok(ContentBlock::Image {
                source: ImageSource {
                    source_type: "base64",
                    media_type: media_type.clone(),
                    data: b64.clone(),
                },
            }),
            CoreImage::Url(url) => Err(LlmError::InvalidRequest(format!(
                "Anthropic does not support URL-based images directly. \
                 Pre-fetch and base64-encode the image at: {url}"
            ))),
            _ => Err(LlmError::InvalidRequest(
                "Unsupported image source type for Anthropic provider".into(),
            )),
        },
        CoreContent::ToolCall(call) => Ok(ContentBlock::ToolUse {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.arguments.clone(),
        }),
        CoreContent::ToolResult(result) => Ok(ContentBlock::ToolResult {
            tool_use_id: result.tool_call_id.clone(),
            content: result.content.clone(),
            is_error: result.is_error,
        }),
        // Reasoning blocks in input messages are sent as text —
        // Anthropic's API doesn't accept reasoning blocks in requests,
        // but their textual content is still useful context.
        CoreContent::Reasoning { content } => Ok(ContentBlock::Text {
            text: content.clone(),
        }),
        _ => Ok(ContentBlock::Text {
            text: String::new(),
        }),
    }
}

/// Convert core tool choice to Anthropic's `tool_choice` parameter.
fn convert_tool_choice(choice: &ToolChoice) -> ToolChoiceParam {
    match choice {
        ToolChoice::None => ToolChoiceParam {
            choice_type: "none".into(),
            name: None,
        },
        ToolChoice::Required => ToolChoiceParam {
            choice_type: "any".into(),
            name: None,
        },
        ToolChoice::Specific(name) => ToolChoiceParam {
            choice_type: "tool".into(),
            name: Some(name.clone()),
        },
        // Auto and any future variants default to "auto"
        _ => ToolChoiceParam {
            choice_type: "auto".into(),
            name: None,
        },
    }
}

// ── Response conversion ──────────────────────────────────────────────

/// Convert an Anthropic API response to a `ChatResponse`.
pub(crate) fn convert_response(resp: crate::types::Response) -> ChatResponse {
    let content = resp
        .content
        .into_iter()
        .filter_map(|block| match block.content_type.as_str() {
            "text" => block.text.map(CoreContent::Text),
            "thinking" => block
                .thinking
                .map(|t| CoreContent::Reasoning { content: t }),
            "tool_use" => {
                let id = block.id.unwrap_or_default();
                let name = block.name.unwrap_or_default();
                let arguments = block
                    .input
                    .unwrap_or(Value::Object(serde_json::Map::default()));
                Some(CoreContent::ToolCall(ToolCall {
                    id,
                    name,
                    arguments,
                }))
            }
            _ => None,
        })
        .collect();

    let usage = Usage {
        input_tokens: resp.usage.input_tokens,
        output_tokens: resp.usage.output_tokens,
        reasoning_tokens: None,
        cache_read_tokens: resp.usage.cache_read_input_tokens,
        cache_write_tokens: resp.usage.cache_creation_input_tokens,
    };

    let stop_reason = resp
        .stop_reason
        .as_deref()
        .map_or(StopReason::EndTurn, convert_stop_reason);

    ChatResponse {
        content,
        usage,
        stop_reason,
        model: resp.model,
        metadata: HashMap::new(),
    }
}

/// Map Anthropic `stop_reason` strings to `StopReason`.
fn convert_stop_reason(reason: &str) -> StopReason {
    match reason {
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        // "end_turn" and unknown values default to EndTurn
        _ => StopReason::EndTurn,
    }
}

// ── Error conversion ─────────────────────────────────────────────────

/// Convert an HTTP status + optional error body into an `LlmError`.
pub(crate) fn convert_error(status: http::StatusCode, body: &str) -> LlmError {
    // Try to parse as Anthropic error response
    let message = serde_json::from_str::<ErrorResponse>(body)
        .map_or_else(|_| body.to_string(), |e| e.error.message);

    if status == http::StatusCode::UNAUTHORIZED || status == http::StatusCode::FORBIDDEN {
        return LlmError::Auth(message);
    }

    if status == http::StatusCode::BAD_REQUEST {
        return LlmError::InvalidRequest(message);
    }

    let retryable = matches!(status.as_u16(), 429 | 500 | 502 | 503 | 529);

    LlmError::Http {
        status: Some(status),
        message,
        retryable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_stack::chat::{ChatMessage, ChatRole, ImageSource as CoreImage};
    use llm_stack::provider::{JsonSchema, ToolDefinition};

    #[test]
    fn test_build_request_minimal() {
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hello")],
            ..Default::default()
        };
        let config = AnthropicConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        assert_eq!(req.model, "claude-sonnet-4-20250514");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.max_tokens, 4096);
        assert!(req.temperature.is_none());
        assert!(req.system.is_none());
        assert!(req.stream.is_none());
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());
        assert!(req.thinking.is_none());
    }

    #[test]
    fn test_build_request_with_system() {
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hi")],
            system: Some("You are helpful.".into()),
            ..Default::default()
        };
        let config = AnthropicConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        assert_eq!(req.system, Some("You are helpful."));
    }

    #[test]
    fn test_build_request_streaming() {
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hi")],
            ..Default::default()
        };
        let config = AnthropicConfig::default();
        let req = build_request(&params, &config, true).unwrap();

        assert_eq!(req.stream, Some(true));
    }

    #[test]
    fn test_build_request_with_tools() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        });
        let params = ChatParams {
            messages: vec![ChatMessage::user("Weather?")],
            tools: Some(vec![ToolDefinition {
                name: "get_weather".into(),
                description: "Get weather".into(),
                parameters: JsonSchema::new(schema.clone()),
                retry: None,
            }]),
            tool_choice: Some(ToolChoice::Auto),
            ..Default::default()
        };
        let config = AnthropicConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        let tc = req.tool_choice.unwrap();
        assert_eq!(tc.choice_type, "auto");
        assert!(tc.name.is_none());
    }

    #[test]
    fn test_build_request_with_thinking() {
        let params = ChatParams {
            messages: vec![ChatMessage::user("Think hard")],
            reasoning_budget: Some(8192),
            ..Default::default()
        };
        let config = AnthropicConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        let thinking = req.thinking.unwrap();
        assert_eq!(thinking.thinking_type, "enabled");
        assert_eq!(thinking.budget_tokens, 8192);
    }

    #[test]
    fn test_system_messages_filtered_from_messages() {
        let params = ChatParams {
            messages: vec![
                ChatMessage::system("System prompt"),
                ChatMessage::user("Hello"),
            ],
            ..Default::default()
        };
        let config = AnthropicConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        // System messages are NOT in the messages array
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
    }

    #[test]
    fn test_tool_result_message_conversion() {
        let msg = ChatMessage::tool_result("toolu_01", "sunny, 25C");
        let messages = convert_messages(&[msg]).unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert!(matches!(
            &messages[0].content[0],
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } if tool_use_id == "toolu_01" && content == "sunny, 25C" && !is_error
        ));
    }

    #[test]
    fn test_tool_error_message_conversion() {
        let msg = ChatMessage::tool_error("toolu_01", "connection refused");
        let messages = convert_messages(&[msg]).unwrap();

        assert!(matches!(
            &messages[0].content[0],
            ContentBlock::ToolResult { is_error, .. } if *is_error
        ));
    }

    #[test]
    fn test_tool_choice_conversions() {
        assert_eq!(convert_tool_choice(&ToolChoice::Auto).choice_type, "auto");
        assert_eq!(convert_tool_choice(&ToolChoice::None).choice_type, "none");
        assert_eq!(
            convert_tool_choice(&ToolChoice::Required).choice_type,
            "any"
        );

        let specific = convert_tool_choice(&ToolChoice::Specific("search".into()));
        assert_eq!(specific.choice_type, "tool");
        assert_eq!(specific.name.as_deref(), Some("search"));
    }

    #[test]
    fn test_convert_response_text() {
        let resp = crate::types::Response {
            content: vec![crate::types::ResponseContent {
                content_type: "text".into(),
                text: Some("Hello!".into()),
                thinking: None,
                id: None,
                name: None,
                input: None,
            }],
            model: "claude-3-5-haiku-20241022".into(),
            stop_reason: Some("end_turn".into()),
            usage: crate::types::ResponseUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        let chat = convert_response(resp);

        assert_eq!(chat.text(), Some("Hello!"));
        assert_eq!(chat.usage.input_tokens, 10);
        assert_eq!(chat.usage.output_tokens, 5);
        assert_eq!(chat.stop_reason, StopReason::EndTurn);
        assert_eq!(chat.model, "claude-3-5-haiku-20241022");
    }

    #[test]
    fn test_convert_response_tool_use() {
        let resp = crate::types::Response {
            content: vec![crate::types::ResponseContent {
                content_type: "tool_use".into(),
                text: None,
                thinking: None,
                id: Some("toolu_01".into()),
                name: Some("get_weather".into()),
                input: Some(serde_json::json!({"city": "Tokyo"})),
            }],
            model: "claude-3-5-haiku-20241022".into(),
            stop_reason: Some("tool_use".into()),
            usage: crate::types::ResponseUsage {
                input_tokens: 50,
                output_tokens: 30,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        let chat = convert_response(resp);

        assert_eq!(chat.stop_reason, StopReason::ToolUse);
        let calls = chat.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], "Tokyo");
    }

    #[test]
    fn test_convert_response_thinking() {
        let resp = crate::types::Response {
            content: vec![
                crate::types::ResponseContent {
                    content_type: "thinking".into(),
                    text: None,
                    thinking: Some("Let me reason...".into()),
                    id: None,
                    name: None,
                    input: None,
                },
                crate::types::ResponseContent {
                    content_type: "text".into(),
                    text: Some("The answer is 42.".into()),
                    thinking: None,
                    id: None,
                    name: None,
                    input: None,
                },
            ],
            model: "claude-3-5-haiku-20241022".into(),
            stop_reason: Some("end_turn".into()),
            usage: crate::types::ResponseUsage {
                input_tokens: 10,
                output_tokens: 20,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        let chat = convert_response(resp);

        assert_eq!(chat.content.len(), 2);
        assert!(matches!(
            &chat.content[0],
            CoreContent::Reasoning { content } if content == "Let me reason..."
        ));
        assert_eq!(chat.text(), Some("The answer is 42."));
    }

    #[test]
    fn test_convert_response_with_cache_tokens() {
        let resp = crate::types::Response {
            content: vec![crate::types::ResponseContent {
                content_type: "text".into(),
                text: Some("ok".into()),
                thinking: None,
                id: None,
                name: None,
                input: None,
            }],
            model: "claude-3-5-haiku-20241022".into(),
            stop_reason: Some("end_turn".into()),
            usage: crate::types::ResponseUsage {
                input_tokens: 100,
                output_tokens: 10,
                cache_creation_input_tokens: Some(50),
                cache_read_input_tokens: Some(30),
            },
        };
        let chat = convert_response(resp);

        assert_eq!(chat.usage.cache_write_tokens, Some(50));
        assert_eq!(chat.usage.cache_read_tokens, Some(30));
    }

    #[test]
    fn test_convert_error_auth() {
        let err = convert_error(
            http::StatusCode::UNAUTHORIZED,
            r#"{"error":{"type":"authentication_error","message":"Invalid API key"}}"#,
        );
        assert!(matches!(err, LlmError::Auth(msg) if msg == "Invalid API key"));
    }

    #[test]
    fn test_convert_error_forbidden() {
        let err = convert_error(
            http::StatusCode::FORBIDDEN,
            r#"{"error":{"type":"permission_error","message":"Forbidden"}}"#,
        );
        assert!(matches!(err, LlmError::Auth(msg) if msg == "Forbidden"));
    }

    #[test]
    fn test_convert_error_bad_request() {
        let err = convert_error(
            http::StatusCode::BAD_REQUEST,
            r#"{"error":{"type":"invalid_request_error","message":"max_tokens required"}}"#,
        );
        assert!(matches!(
            err,
            LlmError::InvalidRequest(msg) if msg == "max_tokens required"
        ));
    }

    #[test]
    fn test_convert_error_rate_limit() {
        let err = convert_error(
            http::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"type":"rate_limit_error","message":"Rate limited"}}"#,
        );
        assert!(matches!(
            err,
            LlmError::Http {
                retryable: true,
                ..
            }
        ));
    }

    #[test]
    fn test_convert_error_overloaded() {
        // 529 is Anthropic's "overloaded" status
        let status = http::StatusCode::from_u16(529).unwrap();
        let err = convert_error(
            status,
            r#"{"error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        );
        assert!(matches!(
            err,
            LlmError::Http {
                retryable: true,
                ..
            }
        ));
    }

    #[test]
    fn test_convert_error_server_error() {
        let err = convert_error(
            http::StatusCode::INTERNAL_SERVER_ERROR,
            "Internal Server Error",
        );
        assert!(matches!(
            err,
            LlmError::Http { retryable: true, message, .. } if message == "Internal Server Error"
        ));
    }

    #[test]
    fn test_convert_error_not_found_not_retryable() {
        let err = convert_error(http::StatusCode::NOT_FOUND, "Not Found");
        assert!(matches!(
            err,
            LlmError::Http {
                retryable: false,
                ..
            }
        ));
    }

    #[test]
    fn test_stop_reason_mapping() {
        assert_eq!(convert_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(convert_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(convert_stop_reason("max_tokens"), StopReason::MaxTokens);
        assert_eq!(
            convert_stop_reason("stop_sequence"),
            StopReason::StopSequence
        );
        // Unknown falls back to EndTurn
        assert_eq!(convert_stop_reason("unknown"), StopReason::EndTurn);
    }

    #[test]
    fn test_max_tokens_from_config_default() {
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hi")],
            ..Default::default()
        };
        let config = AnthropicConfig {
            max_tokens: 2048,
            ..Default::default()
        };
        let req = build_request(&params, &config, false).unwrap();
        assert_eq!(req.max_tokens, 2048);
    }

    #[test]
    fn test_max_tokens_from_params_overrides_config() {
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hi")],
            max_tokens: Some(512),
            ..Default::default()
        };
        let config = AnthropicConfig {
            max_tokens: 2048,
            ..Default::default()
        };
        let req = build_request(&params, &config, false).unwrap();
        assert_eq!(req.max_tokens, 512);
    }

    #[test]
    fn test_assistant_message_conversion() {
        let messages = convert_messages(&[
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
            ChatMessage::user("How are you?"),
        ])
        .unwrap();

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[2].role, "user");
    }

    #[test]
    fn test_url_image_returns_error() {
        let url = reqwest::Url::parse("https://example.com/image.png").unwrap();
        let msg = ChatMessage {
            role: ChatRole::User,
            content: vec![CoreContent::Image {
                media_type: "image/png".into(),
                data: CoreImage::Url(url),
            }],
        };
        let err = convert_messages(&[msg]).unwrap_err();
        assert!(
            matches!(err, LlmError::InvalidRequest(ref msg) if msg.contains("URL-based images")),
            "Expected InvalidRequest for URL images, got: {err:?}"
        );
    }

    #[test]
    fn test_base64_image_accepted() {
        let msg = ChatMessage {
            role: ChatRole::User,
            content: vec![CoreContent::Image {
                media_type: "image/png".into(),
                data: CoreImage::Base64("aGVsbG8=".into()),
            }],
        };
        let messages = convert_messages(&[msg]).unwrap();
        assert_eq!(messages.len(), 1);
        assert!(matches!(
            &messages[0].content[0],
            ContentBlock::Image { source } if source.data == "aGVsbG8="
        ));
    }

    #[test]
    fn test_tool_use_content_in_assistant_message() {
        let mut msg = ChatMessage::assistant("");
        msg.content = vec![CoreContent::ToolCall(ToolCall {
            id: "toolu_01".into(),
            name: "search".into(),
            arguments: serde_json::json!({"q": "rust"}),
        })];
        let messages = convert_messages(&[msg]).unwrap();

        assert_eq!(messages[0].role, "assistant");
        assert!(matches!(
            &messages[0].content[0],
            ContentBlock::ToolUse { name, .. } if name == "search"
        ));
    }
}
