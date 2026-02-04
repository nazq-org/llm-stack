//! JSON Lines stream parser for the Ollama Chat API.
//!
//! Converts a raw `reqwest::Response` byte stream into a `ChatStream`
//! of [`StreamEvent`]s. Ollama uses JSON Lines (one JSON object per
//! line), not SSE. Each line is a complete chunk with `done: true/false`.

use futures::stream::StreamExt;
use llm_stack_core::chat::{StopReason, ToolCall};
use llm_stack_core::error::LlmError;
use llm_stack_core::stream::{ChatStream, StreamEvent};
use llm_stack_core::usage::Usage;

use crate::types::StreamChunk;

/// Maximum size for buffers before we abort the stream.
const MAX_BUF: usize = 16 * 1024 * 1024; // 16 MiB

/// Convert a reqwest response into a `ChatStream` using JSON Lines parsing.
pub(crate) fn into_stream(response: reqwest::Response) -> ChatStream {
    let stream = response
        .bytes_stream()
        .scan(
            (String::new(), Vec::<u8>::new()),
            move |(buffer, utf8_buf), chunk| {
                let result = match chunk {
                    Ok(bytes) => {
                        utf8_buf.extend_from_slice(&bytes);

                        if utf8_buf.len() > MAX_BUF || buffer.len() > MAX_BUF {
                            utf8_buf.clear();
                            buffer.clear();
                            Some(vec![Err(LlmError::ResponseFormat {
                                message: "Stream buffer exceeded 16 MiB".into(),
                                raw: String::new(),
                            })])
                        } else {
                            match std::str::from_utf8(utf8_buf) {
                                Ok(text) => {
                                    buffer.push_str(text);
                                    utf8_buf.clear();
                                }
                                Err(e) => {
                                    let valid_up_to = e.valid_up_to();
                                    if valid_up_to > 0 {
                                        // SAFETY: `from_utf8` validated bytes
                                        // up to this index are valid UTF-8.
                                        let valid = unsafe {
                                            std::str::from_utf8_unchecked(&utf8_buf[..valid_up_to])
                                        };
                                        buffer.push_str(valid);
                                    }
                                    // Skip past permanently invalid bytes
                                    let skip = valid_up_to + e.error_len().unwrap_or(1);
                                    utf8_buf.drain(..skip);
                                }
                            }

                            let mut results = Vec::new();
                            while let Some(pos) = buffer.find('\n') {
                                let line = buffer[..pos].trim_end_matches('\r').to_string();
                                buffer.drain(..=pos);

                                if line.is_empty() {
                                    continue;
                                }

                                results.extend(parse_json_line(&line).into_iter().map(Ok));
                            }

                            Some(results)
                        }
                    }
                    Err(e) => Some(vec![Err(LlmError::Http {
                        status: None,
                        message: format!("Stream read error: {e}"),
                        retryable: true,
                    })]),
                };

                async move { result }
            },
        )
        .flat_map(futures::stream::iter);

    Box::pin(stream)
}

/// Parse a single JSON line into zero or more `StreamEvent`s.
fn parse_json_line(line: &str) -> Vec<StreamEvent> {
    let Ok(chunk) = serde_json::from_str::<StreamChunk>(line) else {
        return vec![];
    };

    let mut events = Vec::new();

    if let Some(message) = &chunk.message {
        // Text delta
        if let Some(text) = &message.content {
            if !text.is_empty() {
                events.push(StreamEvent::TextDelta(text.clone()));
            }
        }

        // Tool calls (Ollama delivers complete tool calls in the final chunk)
        if let Some(tool_calls) = &message.tool_calls {
            for (index, tc) in tool_calls.iter().enumerate() {
                let index = u32::try_from(index).unwrap_or(0);
                let id = format!("call_{}_{index}", tc.function.name);
                let name = tc.function.name.clone();

                events.push(StreamEvent::ToolCallStart {
                    index,
                    id: id.clone(),
                    name: name.clone(),
                });

                if !tc.function.arguments.is_null() {
                    let args_str = tc.function.arguments.to_string();
                    events.push(StreamEvent::ToolCallDelta {
                        index,
                        json_chunk: args_str,
                    });
                }

                events.push(StreamEvent::ToolCallComplete {
                    index,
                    call: ToolCall {
                        id,
                        name,
                        arguments: tc.function.arguments.clone(),
                    },
                });
            }
        }
    }

    // Final chunk with done=true
    if chunk.done == Some(true) {
        // Emit usage if available
        let input_tokens = chunk.prompt_eval_count.unwrap_or(0);
        let output_tokens = chunk.eval_count.unwrap_or(0);
        if input_tokens > 0 || output_tokens > 0 {
            events.push(StreamEvent::Usage(Usage {
                input_tokens,
                output_tokens,
                reasoning_tokens: None,
                cache_read_tokens: None,
                cache_write_tokens: None,
            }));
        }

        let has_tool_calls = chunk
            .message
            .as_ref()
            .and_then(|m| m.tool_calls.as_ref())
            .is_some_and(|tc| !tc.is_empty());

        let stop_reason = if has_tool_calls {
            StopReason::ToolUse
        } else {
            match chunk.done_reason.as_deref() {
                Some("length") => StopReason::MaxTokens,
                _ => StopReason::EndTurn,
            }
        };

        events.push(StreamEvent::Done { stop_reason });
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_text_delta() {
        let line = r#"{"message":{"content":"Hello"},"done":false}"#;
        let events = parse_json_line(line);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0], StreamEvent::TextDelta("Hello".into()));
    }

    #[test]
    fn test_parse_empty_content_ignored() {
        let line = r#"{"message":{"content":""},"done":false}"#;
        let events = parse_json_line(line);
        assert!(events.is_empty());
    }

    #[test]
    fn test_parse_done_event() {
        let line =
            r#"{"message":{"content":""},"done":true,"prompt_eval_count":42,"eval_count":10}"#;
        let events = parse_json_line(line);

        assert_eq!(events.len(), 2);
        assert!(
            matches!(&events[0], StreamEvent::Usage(u) if u.input_tokens == 42 && u.output_tokens == 10)
        );
        assert_eq!(
            events[1],
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            }
        );
    }

    #[test]
    fn test_parse_done_without_usage() {
        let line = r#"{"message":{"content":""},"done":true}"#;
        let events = parse_json_line(line);

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            }
        );
    }

    #[test]
    fn test_parse_tool_call_in_final_chunk() {
        let line = r#"{"message":{"content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Tokyo"}}}]},"done":true}"#;
        let events = parse_json_line(line);

        // ToolCallStart + ToolCallDelta + ToolCallComplete + Done
        assert_eq!(events.len(), 4);
        assert!(matches!(
            &events[0],
            StreamEvent::ToolCallStart { index: 0, id, name }
                if id == "call_get_weather_0" && name == "get_weather"
        ));
        assert!(matches!(
            &events[1],
            StreamEvent::ToolCallDelta { index: 0, .. }
        ));
        assert!(matches!(
            &events[2],
            StreamEvent::ToolCallComplete { index: 0, call }
                if call.name == "get_weather" && call.arguments["city"] == "Tokyo"
        ));
        assert_eq!(
            events[3],
            StreamEvent::Done {
                stop_reason: StopReason::ToolUse,
            }
        );
    }

    #[test]
    fn test_parse_unparseable_line_ignored() {
        let events = parse_json_line("not-json");
        assert!(events.is_empty());
    }

    #[test]
    fn test_parse_text_with_final() {
        // Last chunk often has content + done=true
        let line =
            r#"{"message":{"content":"end."},"done":true,"prompt_eval_count":5,"eval_count":3}"#;
        let events = parse_json_line(line);

        // TextDelta + Usage + Done
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], StreamEvent::TextDelta("end.".into()));
        assert!(matches!(&events[1], StreamEvent::Usage(u) if u.input_tokens == 5));
        assert_eq!(
            events[2],
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            }
        );
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let line = r#"{"message":{"content":"","tool_calls":[{"function":{"name":"search","arguments":{"q":"rust"}}},{"function":{"name":"calc","arguments":{"expr":"2+2"}}}]},"done":true}"#;
        let events = parse_json_line(line);

        // 2 tools * 3 events each + Done = 7
        assert_eq!(events.len(), 7);
        assert!(matches!(
            &events[0],
            StreamEvent::ToolCallStart { index: 0, name, .. } if name == "search"
        ));
        assert!(matches!(
            &events[3],
            StreamEvent::ToolCallStart { index: 1, name, .. } if name == "calc"
        ));
        assert_eq!(
            events[6],
            StreamEvent::Done {
                stop_reason: StopReason::ToolUse,
            }
        );
    }

    // ── Review fix tests ────────────────────────────────────────────

    #[test]
    fn test_done_reason_length_in_stream(/* R4 */) {
        let line = r#"{"message":{"content":""},"done":true,"done_reason":"length","prompt_eval_count":10,"eval_count":50}"#;
        let events = parse_json_line(line);

        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Usage(_)));
        assert_eq!(
            events[1],
            StreamEvent::Done {
                stop_reason: StopReason::MaxTokens,
            }
        );
    }

    #[test]
    fn test_null_arguments_no_delta(/* R18 */) {
        let line = r#"{"message":{"content":"","tool_calls":[{"function":{"name":"no_args","arguments":null}}]},"done":true}"#;
        let events = parse_json_line(line);

        // ToolCallStart + ToolCallComplete (no Delta for null args) + Done
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], StreamEvent::ToolCallStart { .. }));
        assert!(
            matches!(&events[1], StreamEvent::ToolCallComplete { call, .. } if call.arguments.is_null())
        );
        assert!(matches!(
            &events[2],
            StreamEvent::Done {
                stop_reason: StopReason::ToolUse,
            }
        ));
    }
}
