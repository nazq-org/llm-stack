//! SSE stream parser for the `OpenAI` Chat Completions API.
//!
//! Converts a raw `reqwest::Response` byte stream into a `ChatStream`
//! of [`StreamEvent`]s. Handles UTF-8 boundary splitting, tool-call
//! accumulation across multiple delta events, and usage extraction.

use std::collections::HashMap;

use futures::stream::StreamExt;
use llm_stack::chat::{StopReason, ToolCall};
use llm_stack::error::LlmError;
use llm_stack::stream::{ChatStream, StreamEvent};
use llm_stack::usage::Usage;
use serde_json::Value;

use crate::convert::convert_stop_reason;
use crate::types::{ResponseUsage, StreamChunk};

/// Maximum size for buffers before we abort the stream.
const MAX_BUF: usize = 16 * 1024 * 1024; // 16 MiB

/// State tracked per in-flight tool call during streaming.
#[derive(Debug)]
struct ToolCallState {
    id: String,
    name: String,
    arguments_buffer: String,
}

/// Convert a reqwest SSE response into a `ChatStream`.
pub(crate) fn into_stream(response: reqwest::Response) -> ChatStream {
    let stream = response
        .bytes_stream()
        .scan(
            (
                String::new(),
                Vec::<u8>::new(),
                HashMap::<u32, ToolCallState>::new(),
            ),
            move |(buffer, utf8_buf, tool_states), chunk| {
                let result = match chunk {
                    Ok(bytes) => {
                        utf8_buf.extend_from_slice(&bytes);

                        if utf8_buf.len() > MAX_BUF || buffer.len() > MAX_BUF {
                            utf8_buf.clear();
                            buffer.clear();
                            Some(vec![Err(LlmError::ResponseFormat {
                                message: "SSE stream buffer exceeded 16 MiB".into(),
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
                            while let Some(pos) = buffer.find("\n\n") {
                                let event_text = buffer[..pos + 2].to_string();
                                buffer.drain(..pos + 2);

                                results.extend(
                                    parse_sse_event(&event_text, tool_states)
                                        .into_iter()
                                        .map(Ok),
                                );
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

/// Parse a single SSE event into zero or more `StreamEvent`s.
fn parse_sse_event(
    event_text: &str,
    tool_states: &mut HashMap<u32, ToolCallState>,
) -> Vec<StreamEvent> {
    let Some(data) = extract_data_line(event_text) else {
        return vec![];
    };

    if data == "[DONE]" {
        return flush_pending_tools(tool_states);
    }

    let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) else {
        return vec![];
    };

    let mut events = Vec::new();

    // Process choices
    if let Some(choice) = chunk.choices.first() {
        // Text delta
        if let Some(text) = &choice.delta.content {
            if !text.is_empty() {
                events.push(StreamEvent::TextDelta(text.clone()));
            }
        }

        // Tool call deltas
        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tc in tool_calls {
                let index = tc.index;

                // First chunk for this tool call — has id and name
                if let Some(id) = &tc.id {
                    let name = tc
                        .function
                        .as_ref()
                        .and_then(|f| f.name.as_ref())
                        .cloned()
                        .unwrap_or_default();

                    tool_states.insert(
                        index,
                        ToolCallState {
                            id: id.clone(),
                            name: name.clone(),
                            arguments_buffer: String::new(),
                        },
                    );

                    events.push(StreamEvent::ToolCallStart {
                        index,
                        id: id.clone(),
                        name,
                    });
                }

                // Accumulate arguments
                if let Some(func) = &tc.function {
                    if let Some(args) = &func.arguments {
                        if !args.is_empty() {
                            if let Some(state) = tool_states.get_mut(&index) {
                                state.arguments_buffer.push_str(args);
                            }
                            events.push(StreamEvent::ToolCallDelta {
                                index,
                                json_chunk: args.clone(),
                            });
                        }
                    }
                }
            }
        }

        // Finish reason
        if let Some(reason) = &choice.finish_reason {
            let stop_reason = convert_stop_reason(reason);

            // If finishing with tool_calls, emit tool completions
            if stop_reason == StopReason::ToolUse {
                events.extend(flush_pending_tools(tool_states));
            }

            events.push(StreamEvent::Done { stop_reason });
        }
    }

    // Usage from final chunk
    if let Some(usage) = &chunk.usage {
        events.push(StreamEvent::Usage(convert_usage(usage)));
    }

    events
}

/// Flush all pending tool call states as `ToolCallComplete` events.
fn flush_pending_tools(tool_states: &mut HashMap<u32, ToolCallState>) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    let mut indices: Vec<u32> = tool_states.keys().copied().collect();
    indices.sort_unstable();

    for index in indices {
        if let Some(state) = tool_states.remove(&index) {
            let json_str = if state.arguments_buffer.is_empty() {
                "{}".to_string()
            } else {
                state.arguments_buffer
            };
            let arguments: Value = serde_json::from_str(&json_str).unwrap_or_default();

            events.push(StreamEvent::ToolCallComplete {
                index,
                call: ToolCall {
                    id: state.id,
                    name: state.name,
                    arguments,
                },
            });
        }
    }

    events
}

/// Extract the `data: ` payload from an SSE event text block.
fn extract_data_line(event_text: &str) -> Option<&str> {
    for line in event_text.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(data) = line.strip_prefix("data: ") {
            return Some(data);
        }
    }
    None
}

/// Convert `OpenAI` `ResponseUsage` to core `Usage`.
fn convert_usage(usage: &ResponseUsage) -> Usage {
    Usage {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
        reasoning_tokens: usage
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens),
        cache_read_tokens: None,
        cache_write_tokens: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_data_line_basic() {
        let event = "data: {\"choices\":[]}\n\n";
        assert_eq!(extract_data_line(event), Some("{\"choices\":[]}"));
    }

    #[test]
    fn test_extract_data_line_done() {
        assert_eq!(extract_data_line("data: [DONE]\n\n"), Some("[DONE]"));
    }

    #[test]
    fn test_extract_data_line_no_data() {
        assert_eq!(extract_data_line("event: ping\n\n"), None);
    }

    #[test]
    fn test_parse_text_delta() {
        let event =
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n";
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0], StreamEvent::TextDelta("Hello".into()));
    }

    #[test]
    fn test_parse_done_sentinel() {
        let event = "data: [DONE]\n\n";
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);
        assert!(events.is_empty());
    }

    #[test]
    fn test_parse_finish_reason_stop() {
        let event = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            }
        );
    }

    #[test]
    fn test_parse_finish_reason_length() {
        let event = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n";
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);

        assert!(matches!(
            &events[0],
            StreamEvent::Done {
                stop_reason: StopReason::MaxTokens,
            }
        ));
    }

    #[test]
    fn test_parse_usage_event() {
        let event =
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":10}}\n\n";
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);

        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], StreamEvent::Usage(u) if u.input_tokens == 42 && u.output_tokens == 10)
        );
    }

    #[test]
    fn test_parse_tool_call_lifecycle() {
        let mut tool_states = HashMap::new();

        // 1. Start — first chunk with id, name
        let start = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

"#;
        let events = parse_sse_event(start, &mut tool_states);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            StreamEvent::ToolCallStart { index: 0, id, name }
                if id == "call_abc" && name == "get_weather"
        ));

        // 2. Delta — arguments fragment
        let delta1 = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}

"#;
        let events = parse_sse_event(delta1, &mut tool_states);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            StreamEvent::ToolCallDelta { index: 0, json_chunk }
                if json_chunk == r#"{"city":"#
        ));

        // 3. Delta — more arguments
        let delta2 = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Tokyo\"}"}}]},"finish_reason":null}]}

"#;
        let events = parse_sse_event(delta2, &mut tool_states);
        assert_eq!(events.len(), 1);

        // 4. Finish — emits tool complete + done
        let finish = r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

"#;
        let events = parse_sse_event(finish, &mut tool_states);
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            StreamEvent::ToolCallComplete { index: 0, call }
                if call.name == "get_weather" && call.arguments["city"] == "Tokyo"
        ));
        assert!(matches!(
            &events[1],
            StreamEvent::Done {
                stop_reason: StopReason::ToolUse
            }
        ));
    }

    #[test]
    fn test_flush_pending_tools_on_done() {
        let mut tool_states = HashMap::new();

        // Start a tool call
        let start = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"no_args_tool","arguments":""}}]},"finish_reason":null}]}

"#;
        parse_sse_event(start, &mut tool_states);
        assert!(!tool_states.is_empty());

        // [DONE] should flush
        let events = parse_sse_event("data: [DONE]\n\n", &mut tool_states);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            StreamEvent::ToolCallComplete { call, .. }
                if call.name == "no_args_tool" && call.arguments == serde_json::json!({})
        ));
    }

    #[test]
    fn test_convert_usage() {
        let api_usage = ResponseUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            completion_tokens_details: Some(crate::types::CompletionTokensDetails {
                reasoning_tokens: Some(20),
            }),
        };
        let usage = convert_usage(&api_usage);

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.reasoning_tokens, Some(20));
        assert!(usage.cache_read_tokens.is_none());
    }

    #[test]
    fn test_parse_unparseable_event_ignored() {
        let event = "data: not-json\n\n";
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);
        assert!(events.is_empty());
    }

    #[test]
    fn test_empty_text_delta_ignored() {
        let event =
            "data: {\"choices\":[{\"delta\":{\"content\":\"\"},\"finish_reason\":null}]}\n\n";
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);
        assert!(events.is_empty());
    }

    // ── Review fix tests ────────────────────────────────────────────

    #[test]
    fn test_finish_reason_content_filter(/* R3 */) {
        let event = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"content_filter\"}]}\n\n";
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);

        assert_eq!(events.len(), 1);
        // content_filter maps to EndTurn (with a warning log)
        assert_eq!(
            events[0],
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            }
        );
    }

    #[test]
    fn test_extract_data_with_carriage_return() {
        let event = "data: {\"choices\":[]}\r\n\r\n";
        assert_eq!(extract_data_line(event), Some("{\"choices\":[]}"));
    }
}
