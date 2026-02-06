//! Bidirectional conversion between `llm-core` types and `OpenAI` API types.

use llm_stack::chat::{
    ChatMessage, ChatResponse, ChatRole, ContentBlock as CoreContent, ImageSource as CoreImage,
    StopReason, ToolCall,
};
use llm_stack::error::LlmError;
use llm_stack::provider::{ChatParams, ToolChoice};
use llm_stack::usage::Usage;
use serde_json::Value;
use std::collections::HashMap;

use crate::config::OpenAiConfig;
use crate::types::{
    ContentPart, ErrorResponse, FunctionCallRequest, FunctionDef, ImageUrl, JsonSchemaFormat,
    Message, MessageContent, Request, ResponseFormat, StreamOptions, Tool, ToolCallRequest,
};

// ── Request conversion ───────────────────────────────────────────────

/// Build an `OpenAI` API request from `ChatParams` and provider config.
pub(crate) fn build_request<'a>(
    params: &'a ChatParams,
    config: &'a OpenAiConfig,
    stream: bool,
) -> Result<Request<'a>, LlmError> {
    // Build messages with optional system prompt prepended (avoids O(n) insert(0))
    let converted = convert_messages(&params.messages)?;
    let messages = if let Some(system) = &params.system {
        let mut msgs = Vec::with_capacity(1 + converted.len());
        msgs.push(Message {
            role: "system",
            content: Some(MessageContent::Text(system.clone())),
            tool_calls: None,
            tool_call_id: None,
        });
        msgs.extend(converted);
        msgs
    } else {
        converted
    };

    let tools = params.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|t| Tool {
                tool_type: "function",
                function: FunctionDef {
                    name: &t.name,
                    description: &t.description,
                    parameters: t.parameters.as_value(),
                },
            })
            .collect()
    });
    let tool_choice = params.tool_choice.as_ref().map(convert_tool_choice);
    let response_format = params
        .structured_output
        .as_ref()
        .map(|schema| ResponseFormat {
            format_type: "json_schema",
            json_schema: Some(JsonSchemaFormat {
                name: "output",
                schema: schema.as_value(),
                strict: true,
            }),
        });

    Ok(Request {
        model: &config.model,
        messages,
        temperature: params.temperature,
        max_completion_tokens: params.max_tokens,
        stream: if stream { Some(true) } else { None },
        stream_options: if stream {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        },
        tools,
        tool_choice,
        response_format,
    })
}

/// Convert core messages to `OpenAI` message format.
fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<Message>, LlmError> {
    messages.iter().map(convert_message).collect()
}

/// Extract the first text content from a message's content blocks.
fn extract_text(content: &[CoreContent]) -> Option<String> {
    content.iter().find_map(|b| {
        if let CoreContent::Text(t) = b {
            Some(t.clone())
        } else {
            None
        }
    })
}

/// Convert a single core message to an `OpenAI` message.
fn convert_message(msg: &ChatMessage) -> Result<Message, LlmError> {
    match msg.role {
        ChatRole::System => Ok(Message {
            role: "system",
            content: Some(MessageContent::Text(
                extract_text(&msg.content).unwrap_or_default(),
            )),
            tool_calls: None,
            tool_call_id: None,
        }),
        ChatRole::User => {
            let content = convert_user_content(&msg.content)?;
            Ok(Message {
                role: "user",
                content: Some(content),
                tool_calls: None,
                tool_call_id: None,
            })
        }
        ChatRole::Assistant => {
            let text = extract_text(&msg.content);
            let tool_calls: Vec<_> = msg
                .content
                .iter()
                .filter_map(|b| {
                    if let CoreContent::ToolCall(call) = b {
                        Some(ToolCallRequest {
                            id: call.id.clone(),
                            call_type: "function",
                            function: FunctionCallRequest {
                                name: call.name.clone(),
                                arguments: call.arguments.to_string(),
                            },
                        })
                    } else {
                        None
                    }
                })
                .collect();
            Ok(Message {
                role: "assistant",
                content: text.map(MessageContent::Text),
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                tool_call_id: None,
            })
        }
        ChatRole::Tool => {
            let tool_result = msg.content.iter().find_map(|b| {
                if let CoreContent::ToolResult(r) = b {
                    Some(r)
                } else {
                    None
                }
            });
            let (content, tool_call_id) = if let Some(result) = tool_result {
                (result.content.clone(), Some(result.tool_call_id.clone()))
            } else {
                (extract_text(&msg.content).unwrap_or_default(), None)
            };
            Ok(Message {
                role: "tool",
                content: Some(MessageContent::Text(content)),
                tool_calls: None,
                tool_call_id,
            })
        }
        _ => Ok(Message {
            role: "user",
            content: Some(MessageContent::Text(
                extract_text(&msg.content).unwrap_or_default(),
            )),
            tool_calls: None,
            tool_call_id: None,
        }),
    }
}

/// Convert user message content blocks to `OpenAI` message content.
fn convert_user_content(blocks: &[CoreContent]) -> Result<MessageContent, LlmError> {
    // If it's just a single text block, use the simple string format
    if blocks.len() == 1 {
        if let CoreContent::Text(text) = &blocks[0] {
            return Ok(MessageContent::Text(text.clone()));
        }
    }

    let parts: Result<Vec<ContentPart>, LlmError> = blocks
        .iter()
        .filter_map(|block| match block {
            CoreContent::Text(text) => Some(Ok(ContentPart::Text { text: text.clone() })),
            CoreContent::Image { media_type, data } => Some(match data {
                CoreImage::Base64(b64) => Ok(ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: format!("data:{media_type};base64,{b64}"),
                    },
                }),
                CoreImage::Url(url) => Ok(ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: url.to_string(),
                    },
                }),
                _ => Err(LlmError::InvalidRequest(
                    "Unsupported image source type for `OpenAI` provider".into(),
                )),
            }),
            // Reasoning blocks in input are sent as text
            CoreContent::Reasoning { content } => Some(Ok(ContentPart::Text {
                text: content.clone(),
            })),
            // ToolCall/ToolResult blocks don't belong in user messages; skip them
            _ => None,
        })
        .collect();

    Ok(MessageContent::Parts(parts?))
}

/// Convert core tool choice to `OpenAI`'s `tool_choice` parameter.
fn convert_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::None => Value::String("none".into()),
        ToolChoice::Required => Value::String("required".into()),
        ToolChoice::Specific(name) => serde_json::json!({
            "type": "function",
            "function": { "name": name }
        }),
        _ => Value::String("auto".into()),
    }
}

// ── Response conversion ──────────────────────────────────────────────

/// Convert an `OpenAI` API response to a `ChatResponse`.
pub(crate) fn convert_response(resp: crate::types::Response) -> ChatResponse {
    let choice = resp.choices.into_iter().next();

    let mut content = Vec::new();
    if let Some(choice) = &choice {
        if let Some(text) = &choice.message.content {
            if !text.is_empty() {
                content.push(CoreContent::Text(text.clone()));
            }
        }
        if let Some(tool_calls) = &choice.message.tool_calls {
            for tc in tool_calls {
                let arguments: Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                content.push(CoreContent::ToolCall(ToolCall {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments,
                }));
            }
        }
    }

    let usage = resp.usage.map_or_else(Usage::default, |u| Usage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
        reasoning_tokens: u.completion_tokens_details.and_then(|d| d.reasoning_tokens),
        cache_read_tokens: None,
        cache_write_tokens: None,
    });

    let stop_reason = choice
        .and_then(|c| c.finish_reason)
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

/// Map `OpenAI` `finish_reason` strings to `StopReason`.
pub(crate) fn convert_stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::EndTurn,
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        other => {
            tracing::warn!(finish_reason = other, "Unexpected OpenAI finish_reason");
            StopReason::EndTurn
        }
    }
}

// ── Error conversion ─────────────────────────────────────────────────

/// Convert an HTTP status + optional error body into an `LlmError`.
pub(crate) fn convert_error(status: http::StatusCode, body: &str) -> LlmError {
    let message = serde_json::from_str::<ErrorResponse>(body)
        .map_or_else(|_| body.to_string(), |e| e.error.message);

    if status == http::StatusCode::UNAUTHORIZED || status == http::StatusCode::FORBIDDEN {
        return LlmError::Auth(message);
    }

    if status == http::StatusCode::BAD_REQUEST {
        return LlmError::InvalidRequest(message);
    }

    let retryable = matches!(status.as_u16(), 429 | 500 | 502 | 503);

    LlmError::Http {
        status: Some(status),
        message,
        retryable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_stack::chat::ChatMessage;
    use llm_stack::provider::{JsonSchema, ToolDefinition};

    #[test]
    fn test_build_request_minimal() {
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hello")],
            ..Default::default()
        };
        let config = OpenAiConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        assert_eq!(req.model, "gpt-4o");
        assert_eq!(req.messages.len(), 1);
        assert!(req.temperature.is_none());
        assert!(req.stream.is_none());
        assert!(req.tools.is_none());
    }

    #[test]
    fn test_build_request_streaming() {
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hi")],
            ..Default::default()
        };
        let config = OpenAiConfig::default();
        let req = build_request(&params, &config, true).unwrap();

        assert_eq!(req.stream, Some(true));
        assert!(req.stream_options.is_some());
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
                parameters: JsonSchema::new(schema),
                retry: None,
            }]),
            tool_choice: Some(ToolChoice::Auto),
            ..Default::default()
        };
        let config = OpenAiConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
    }

    #[test]
    fn test_system_message_conversion() {
        let messages = convert_messages(&[
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ])
        .unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
    }

    #[test]
    fn test_tool_result_message_conversion() {
        let msg = ChatMessage::tool_result("call_abc", "sunny, 25C");
        let messages = convert_messages(&[msg]).unwrap();

        assert_eq!(messages[0].role, "tool");
        assert_eq!(messages[0].tool_call_id.as_deref(), Some("call_abc"));
    }

    #[test]
    fn test_tool_choice_conversions() {
        assert_eq!(convert_tool_choice(&ToolChoice::Auto), "auto");
        assert_eq!(convert_tool_choice(&ToolChoice::None), "none");
        assert_eq!(convert_tool_choice(&ToolChoice::Required), "required");

        let specific = convert_tool_choice(&ToolChoice::Specific("search".into()));
        assert_eq!(specific["function"]["name"], "search");
    }

    #[test]
    fn test_convert_response_text() {
        let resp = crate::types::Response {
            choices: vec![crate::types::Choice {
                message: crate::types::ResponseMessage {
                    content: Some("Hello!".into()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
            }],
            model: "gpt-4o".into(),
            usage: Some(crate::types::ResponseUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                completion_tokens_details: None,
            }),
        };
        let chat = convert_response(resp);

        assert_eq!(chat.text(), Some("Hello!"));
        assert_eq!(chat.usage.input_tokens, 10);
        assert_eq!(chat.usage.output_tokens, 5);
        assert_eq!(chat.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn test_convert_response_tool_calls() {
        let resp = crate::types::Response {
            choices: vec![crate::types::Choice {
                message: crate::types::ResponseMessage {
                    content: None,
                    tool_calls: Some(vec![crate::types::ToolCallResponse {
                        id: "call_abc".into(),
                        function: crate::types::FunctionCallResponse {
                            name: "get_weather".into(),
                            arguments: r#"{"city":"Tokyo"}"#.into(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".into()),
            }],
            model: "gpt-4o".into(),
            usage: None,
        };
        let chat = convert_response(resp);

        assert_eq!(chat.stop_reason, StopReason::ToolUse);
        let calls = chat.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], "Tokyo");
    }

    #[test]
    fn test_stop_reason_mapping() {
        assert_eq!(convert_stop_reason("stop"), StopReason::EndTurn);
        assert_eq!(convert_stop_reason("tool_calls"), StopReason::ToolUse);
        assert_eq!(convert_stop_reason("length"), StopReason::MaxTokens);
        assert_eq!(convert_stop_reason("unknown"), StopReason::EndTurn);
    }

    #[test]
    fn test_convert_error_auth() {
        let err = convert_error(
            http::StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"Invalid API key","type":"invalid_api_key"}}"#,
        );
        assert!(matches!(err, LlmError::Auth(ref msg) if msg == "Invalid API key"));
    }

    #[test]
    fn test_convert_error_rate_limit() {
        let err = convert_error(
            http::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"message":"Rate limited"}}"#,
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
    fn test_convert_error_not_retryable() {
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
    fn test_structured_output_in_request() {
        let schema = serde_json::json!({"type": "object"});
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hello")],
            structured_output: Some(JsonSchema::new(schema)),
            ..Default::default()
        };
        let config = OpenAiConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        let rf = req.response_format.unwrap();
        assert_eq!(rf.format_type, "json_schema");
    }

    // ── Review fix tests ────────────────────────────────────────────

    #[test]
    fn test_system_prompt_prepended_via_params(/* R1 */) {
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hello")],
            system: Some("You are helpful".into()),
            ..Default::default()
        };
        let config = OpenAiConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[1].role, "user");
    }

    #[test]
    fn test_system_prompt_not_duplicated_with_existing_system(/* R1 */) {
        let params = ChatParams {
            messages: vec![
                ChatMessage::system("Existing system"),
                ChatMessage::user("Hello"),
            ],
            system: Some("Prepended system".into()),
            ..Default::default()
        };
        let config = OpenAiConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        // params.system prepended + existing system + user = 3
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[1].role, "system");
        assert_eq!(req.messages[2].role, "user");
    }

    #[test]
    fn test_unknown_content_blocks_skipped_in_user_message(/* R17 */) {
        let content = vec![
            CoreContent::Text("Hello".into()),
            CoreContent::ToolCall(ToolCall {
                id: "call_abc".into(),
                name: "search".into(),
                arguments: serde_json::json!({}),
            }),
        ];
        let result = convert_user_content(&content).unwrap();
        // ToolCall in user content should be skipped, leaving only text
        // With 2 blocks but only 1 valid, we get Parts format
        match result {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 1);
                assert!(matches!(&parts[0], ContentPart::Text { text } if text == "Hello"));
            }
            MessageContent::Text(_) => panic!("Expected Parts format"),
        }
    }

    #[test]
    fn test_convert_stop_reason_unknown_falls_back(/* R3 */) {
        // content_filter and other unknown reasons should map to EndTurn
        assert_eq!(convert_stop_reason("content_filter"), StopReason::EndTurn);
    }
}
