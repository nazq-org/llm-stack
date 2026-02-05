//! Bidirectional conversion between `llm-core` types and Ollama API types.

use llm_stack::chat::{
    ChatMessage, ChatResponse, ChatRole, ContentBlock as CoreContent, ImageSource as CoreImage,
    StopReason, ToolCall,
};
use llm_stack::error::LlmError;
use llm_stack::provider::{ChatParams, ToolChoice};
use llm_stack::usage::Usage;
use std::collections::HashMap;

use crate::config::OllamaConfig;
use crate::types::{
    FunctionCallRequest, FunctionDef, Message, Options, Request, Tool, ToolCallRequest,
};

// ── Request conversion ───────────────────────────────────────────────

/// Build an Ollama API request from `ChatParams` and provider config.
pub(crate) fn build_request(
    params: &ChatParams,
    config: &OllamaConfig,
    stream: bool,
) -> Result<Request, LlmError> {
    let mut messages = convert_messages(&params.messages)?;

    // Prepend system prompt as a system message if provided
    if let Some(system) = &params.system {
        messages.insert(
            0,
            Message {
                role: "system".into(),
                content: system.clone(),
                images: None,
                tool_calls: None,
            },
        );
    }

    // ToolChoice::None means "don't use tools" — omit them from the request.
    let tools_disabled = matches!(params.tool_choice, Some(ToolChoice::None));
    let tools = if tools_disabled {
        None
    } else {
        params.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| Tool {
                    tool_type: "function".into(),
                    function: FunctionDef {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.as_value().clone(),
                    },
                })
                .collect()
        })
    };

    let options = if params.temperature.is_some() || params.max_tokens.is_some() {
        Some(Options {
            temperature: params.temperature,
            num_predict: params.max_tokens,
        })
    } else {
        None
    };

    let format = params
        .structured_output
        .as_ref()
        .map(|schema| schema.as_value().clone());

    Ok(Request {
        model: config.model.clone(),
        messages,
        stream,
        options,
        tools,
        format,
    })
}

/// Convert core messages to Ollama message format.
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

/// Convert a single core message to an Ollama message.
fn convert_message(msg: &ChatMessage) -> Result<Message, LlmError> {
    match msg.role {
        ChatRole::System => Ok(Message {
            role: "system".into(),
            content: extract_text(&msg.content).unwrap_or_default(),
            images: None,
            tool_calls: None,
        }),
        ChatRole::User => {
            let text = extract_text(&msg.content).unwrap_or_default();
            let images = extract_images(&msg.content)?;
            Ok(Message {
                role: "user".into(),
                content: text,
                images: if images.is_empty() {
                    None
                } else {
                    Some(images)
                },
                tool_calls: None,
            })
        }
        ChatRole::Assistant => {
            let text = extract_text(&msg.content).unwrap_or_default();
            let tool_calls: Vec<_> = msg
                .content
                .iter()
                .filter_map(|b| {
                    if let CoreContent::ToolCall(call) = b {
                        Some(ToolCallRequest {
                            function: FunctionCallRequest {
                                name: call.name.clone(),
                                arguments: call.arguments.clone(),
                            },
                        })
                    } else {
                        None
                    }
                })
                .collect();
            Ok(Message {
                role: "assistant".into(),
                content: text,
                images: None,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            })
        }
        ChatRole::Tool => {
            let content = msg
                .content
                .iter()
                .find_map(|b| {
                    if let CoreContent::ToolResult(r) = b {
                        Some(r.content.clone())
                    } else {
                        None
                    }
                })
                .or_else(|| extract_text(&msg.content))
                .unwrap_or_default();
            Ok(Message {
                role: "tool".into(),
                content,
                images: None,
                tool_calls: None,
            })
        }
        _ => Ok(Message {
            role: "user".into(),
            content: extract_text(&msg.content).unwrap_or_default(),
            images: None,
            tool_calls: None,
        }),
    }
}

/// Extract base64 image data from content blocks.
///
/// Ollama expects images as raw base64 strings (no data URI prefix).
/// URL images are not supported and return an error.
fn extract_images(content: &[CoreContent]) -> Result<Vec<String>, LlmError> {
    let mut images = Vec::new();
    for block in content {
        if let CoreContent::Image { data, .. } = block {
            match data {
                CoreImage::Base64(b64) => images.push(b64.clone()),
                CoreImage::Url(_) => {
                    return Err(LlmError::InvalidRequest(
                        "Ollama does not support URL images; use base64 encoding instead".into(),
                    ));
                }
                _ => {
                    return Err(LlmError::InvalidRequest(
                        "Unsupported image source type for Ollama provider".into(),
                    ));
                }
            }
        }
    }
    Ok(images)
}

// ── Response conversion ──────────────────────────────────────────────

/// Convert an Ollama API response to a `ChatResponse`.
pub(crate) fn convert_response(resp: crate::types::Response) -> ChatResponse {
    let mut content = Vec::new();
    let mut has_tool_calls = false;

    if let Some(message) = &resp.message {
        if let Some(text) = &message.content {
            if !text.is_empty() {
                content.push(CoreContent::Text(text.clone()));
            }
        }
        if let Some(tool_calls) = &message.tool_calls {
            for (i, tc) in tool_calls.iter().enumerate() {
                has_tool_calls = true;
                content.push(CoreContent::ToolCall(ToolCall {
                    // Ollama doesn't provide tool call IDs — synthesize one
                    // with index to avoid collisions when the same tool is
                    // called multiple times.
                    id: format!("call_{}_{i}", tc.function.name),
                    name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                }));
            }
        }
    }

    let usage = Usage {
        input_tokens: resp.prompt_eval_count.unwrap_or(0),
        output_tokens: resp.eval_count.unwrap_or(0),
        reasoning_tokens: None,
        cache_read_tokens: None,
        cache_write_tokens: None,
    };

    let stop_reason = if has_tool_calls {
        StopReason::ToolUse
    } else {
        match resp.done_reason.as_deref() {
            Some("length") => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        }
    };

    ChatResponse {
        content,
        usage,
        stop_reason,
        model: resp.model.unwrap_or_default(),
        metadata: HashMap::new(),
    }
}

/// Convert an HTTP status + optional error body into an `LlmError`.
pub(crate) fn convert_error(status: http::StatusCode, body: &str) -> LlmError {
    let message = serde_json::from_str::<crate::types::ErrorResponse>(body)
        .map_or_else(|_| body.to_string(), |e| e.error);

    if status == http::StatusCode::NOT_FOUND {
        return LlmError::InvalidRequest(format!("Model not found: {message}"));
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
        let config = OllamaConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        assert_eq!(req.model, "llama3.2");
        assert_eq!(req.messages.len(), 1);
        assert!(!req.stream);
        assert!(req.options.is_none());
        assert!(req.tools.is_none());
    }

    #[test]
    fn test_build_request_streaming() {
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hi")],
            ..Default::default()
        };
        let config = OllamaConfig::default();
        let req = build_request(&params, &config, true).unwrap();
        assert!(req.stream);
    }

    #[test]
    fn test_build_request_with_options() {
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hi")],
            temperature: Some(0.7),
            max_tokens: Some(100),
            ..Default::default()
        };
        let config = OllamaConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        let opts = req.options.unwrap();
        assert_eq!(opts.temperature, Some(0.7));
        assert_eq!(opts.num_predict, Some(100));
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
            ..Default::default()
        };
        let config = OllamaConfig::default();
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
        assert_eq!(messages[0].content, "You are helpful");
        assert_eq!(messages[1].role, "user");
    }

    #[test]
    fn test_tool_result_message_conversion() {
        let msg = ChatMessage::tool_result("call_search", "42");
        let messages = convert_messages(&[msg]).unwrap();

        assert_eq!(messages[0].role, "tool");
        assert_eq!(messages[0].content, "42");
    }

    #[test]
    fn test_assistant_with_tool_calls() {
        let msg = ChatMessage {
            role: ChatRole::Assistant,
            content: vec![CoreContent::ToolCall(ToolCall {
                id: "call_search".into(),
                name: "search".into(),
                arguments: serde_json::json!({"q": "rust"}),
            })],
        };
        let messages = convert_messages(&[msg]).unwrap();

        assert_eq!(messages[0].role, "assistant");
        let tc = messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].function.name, "search");
    }

    #[test]
    fn test_image_extraction_base64() {
        let content = vec![
            CoreContent::Text("What's this?".into()),
            CoreContent::Image {
                media_type: "image/png".into(),
                data: CoreImage::Base64("abc123".into()),
            },
        ];
        let images = extract_images(&content).unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0], "abc123");
    }

    #[test]
    fn test_image_extraction_url_rejected() {
        let content = vec![CoreContent::Image {
            media_type: "image/png".into(),
            data: CoreImage::from_url("https://example.com/img.png").unwrap(),
        }];
        let err = extract_images(&content).unwrap_err();
        assert!(matches!(err, LlmError::InvalidRequest(ref msg) if msg.contains("URL")));
    }

    #[test]
    fn test_convert_response_text() {
        let resp = crate::types::Response {
            message: Some(crate::types::ResponseMessage {
                content: Some("Hello!".into()),
                tool_calls: None,
            }),
            model: Some("llama3.2".into()),
            done_reason: None,
            prompt_eval_count: Some(10),
            eval_count: Some(5),
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
            message: Some(crate::types::ResponseMessage {
                content: Some(String::new()),
                tool_calls: Some(vec![crate::types::ToolCallResponse {
                    function: crate::types::FunctionCallResponse {
                        name: "get_weather".into(),
                        arguments: serde_json::json!({"city": "Tokyo"}),
                    },
                }]),
            }),
            model: Some("llama3.2".into()),
            done_reason: None,
            prompt_eval_count: None,
            eval_count: None,
        };
        let chat = convert_response(resp);

        assert_eq!(chat.stop_reason, StopReason::ToolUse);
        let calls = chat.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].id, "call_get_weather_0");
        assert_eq!(calls[0].arguments["city"], "Tokyo");
    }

    #[test]
    fn test_convert_error_not_found() {
        let err = convert_error(
            http::StatusCode::NOT_FOUND,
            r#"{"error":"model 'nomodel' not found"}"#,
        );
        assert!(
            matches!(err, LlmError::InvalidRequest(ref msg) if msg.contains("Model not found"))
        );
    }

    #[test]
    fn test_convert_error_retryable() {
        let err = convert_error(http::StatusCode::INTERNAL_SERVER_ERROR, "internal error");
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
        let err = convert_error(http::StatusCode::BAD_REQUEST, "bad request");
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
            structured_output: Some(JsonSchema::new(schema.clone())),
            ..Default::default()
        };
        let config = OllamaConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        assert_eq!(req.format.unwrap(), schema);
    }

    // ── Review fix tests ────────────────────────────────────────────

    #[test]
    fn test_system_prompt_prepended_via_params(/* R1 */) {
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hello")],
            system: Some("You are helpful".into()),
            ..Default::default()
        };
        let config = OllamaConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[0].content, "You are helpful");
        assert_eq!(req.messages[1].role, "user");
    }

    #[test]
    fn test_tool_call_id_includes_index(/* R2 */) {
        let resp = crate::types::Response {
            message: Some(crate::types::ResponseMessage {
                content: Some(String::new()),
                tool_calls: Some(vec![
                    crate::types::ToolCallResponse {
                        function: crate::types::FunctionCallResponse {
                            name: "search".into(),
                            arguments: serde_json::json!({"q": "a"}),
                        },
                    },
                    crate::types::ToolCallResponse {
                        function: crate::types::FunctionCallResponse {
                            name: "search".into(),
                            arguments: serde_json::json!({"q": "b"}),
                        },
                    },
                ]),
            }),
            model: Some("llama3.2".into()),
            done_reason: None,
            prompt_eval_count: None,
            eval_count: None,
        };
        let chat = convert_response(resp);
        let calls = chat.tool_calls();

        // Same function name but different indices → unique IDs
        assert_eq!(calls[0].id, "call_search_0");
        assert_eq!(calls[1].id, "call_search_1");
    }

    #[test]
    fn test_done_reason_length_maps_to_max_tokens(/* R4 */) {
        let resp = crate::types::Response {
            message: Some(crate::types::ResponseMessage {
                content: Some("truncated".into()),
                tool_calls: None,
            }),
            model: Some("llama3.2".into()),
            done_reason: Some("length".into()),
            prompt_eval_count: None,
            eval_count: None,
        };
        let chat = convert_response(resp);
        assert_eq!(chat.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn test_tool_choice_none_omits_tools(/* R5 */) {
        let schema = serde_json::json!({"type": "object"});
        let params = ChatParams {
            messages: vec![ChatMessage::user("Hello")],
            tools: Some(vec![ToolDefinition {
                name: "search".into(),
                description: "Search".into(),
                parameters: JsonSchema::new(schema),
                retry: None,
            }]),
            tool_choice: Some(ToolChoice::None),
            ..Default::default()
        };
        let config = OllamaConfig::default();
        let req = build_request(&params, &config, false).unwrap();

        // Tools should be omitted when tool_choice is None
        assert!(req.tools.is_none());
    }
}
