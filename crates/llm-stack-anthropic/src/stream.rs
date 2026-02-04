//! SSE stream parser for the Anthropic Messages API.
//!
//! Converts a raw `reqwest::Response` byte stream into a `ChatStream`
//! of [`StreamEvent`]s. Handles UTF-8 boundary splitting, tool-call
//! accumulation across multiple delta events, and usage extraction.

use std::collections::HashMap;

use futures::stream::StreamExt;
use llm_stack_core::chat::{StopReason, ToolCall};
use llm_stack_core::error::LlmError;
use llm_stack_core::stream::{ChatStream, StreamEvent};
use llm_stack_core::usage::Usage;
use serde_json::Value;

use crate::types::{ResponseUsage, StreamResponse};

/// Maximum size for the UTF-8 accumulation buffer before we abort the stream.
const MAX_UTF8_BUF: usize = 16 * 1024 * 1024; // 16 MiB

/// State tracked per in-flight tool-use block during streaming.
#[derive(Debug)]
struct ToolUseState {
    id: String,
    name: String,
    json_buffer: String,
}

/// Convert a reqwest SSE response into a `ChatStream`.
///
/// The response must have been initiated with `stream: true`.
/// This function does not consume the full body upfront — it
/// processes chunks as they arrive from the network.
pub(crate) fn into_stream(response: reqwest::Response) -> ChatStream {
    let stream = response
        .bytes_stream()
        .scan(
            (
                String::new(),
                Vec::<u8>::new(),
                HashMap::<u32, ToolUseState>::new(),
            ),
            move |(buffer, utf8_buf, tool_states), chunk| {
                let result = match chunk {
                    Ok(bytes) => {
                        // Accumulate bytes and decode valid UTF-8 prefix
                        utf8_buf.extend_from_slice(&bytes);

                        // Guard against unbounded buffer growth from malformed data
                        if utf8_buf.len() > MAX_UTF8_BUF {
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

                            // Extract complete SSE events (delimited by \n\n)
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

/// Parse a single SSE event (one `data: ...` payload) into zero or more `StreamEvent`s.
///
/// Returns an empty vec for events we don't care about (pings, `message_stop`, etc.).
fn parse_sse_event(
    event_text: &str,
    tool_states: &mut HashMap<u32, ToolUseState>,
) -> Vec<StreamEvent> {
    let Some(data) = extract_data_line(event_text) else {
        return vec![];
    };

    // Skip the [DONE] sentinel
    if data == "[DONE]" {
        return vec![];
    }

    let Ok(response) = serde_json::from_str::<StreamResponse>(data) else {
        // Skip unparseable events (e.g. ping with empty object)
        return vec![];
    };

    match response.event_type.as_str() {
        "message_start" => handle_message_start(&response),
        "content_block_start" => handle_block_start(&response, tool_states),
        "content_block_delta" => handle_block_delta(&response, tool_states),
        "content_block_stop" => handle_block_stop(&response, tool_states),
        "message_delta" => handle_message_delta(&response),
        _ => vec![],
    }
}

fn handle_message_start(response: &StreamResponse) -> Vec<StreamEvent> {
    let Some(msg) = &response.message else {
        return vec![];
    };
    let Some(usage) = &msg.usage else {
        return vec![];
    };
    vec![StreamEvent::Usage(convert_usage(usage))]
}

fn handle_block_start(
    response: &StreamResponse,
    tool_states: &mut HashMap<u32, ToolUseState>,
) -> Vec<StreamEvent> {
    let (Some(index), Some(block)) = (response.index, &response.content_block) else {
        return vec![];
    };

    if block.block_type != "tool_use" {
        return vec![];
    }

    let id = block.id.clone().unwrap_or_default();
    let name = block.name.clone().unwrap_or_default();

    tool_states.insert(
        index,
        ToolUseState {
            id: id.clone(),
            name: name.clone(),
            json_buffer: String::new(),
        },
    );

    vec![StreamEvent::ToolCallStart { index, id, name }]
}

fn handle_block_delta(
    response: &StreamResponse,
    tool_states: &mut HashMap<u32, ToolUseState>,
) -> Vec<StreamEvent> {
    let (Some(index), Some(delta)) = (response.index, &response.delta) else {
        return vec![];
    };

    match delta.delta_type.as_deref() {
        Some("text_delta") => delta
            .text
            .as_ref()
            .map(|t| StreamEvent::TextDelta(t.clone()))
            .into_iter()
            .collect(),
        Some("thinking_delta") => delta
            .thinking
            .as_ref()
            .map(|t| StreamEvent::ReasoningDelta(t.clone()))
            .into_iter()
            .collect(),
        Some("input_json_delta") => {
            let Some(partial_json) = &delta.partial_json else {
                return vec![];
            };
            if let Some(state) = tool_states.get_mut(&index) {
                state.json_buffer.push_str(partial_json);
            }
            vec![StreamEvent::ToolCallDelta {
                index,
                json_chunk: partial_json.clone(),
            }]
        }
        _ => vec![],
    }
}

fn handle_block_stop(
    response: &StreamResponse,
    tool_states: &mut HashMap<u32, ToolUseState>,
) -> Vec<StreamEvent> {
    let Some(index) = response.index else {
        return vec![];
    };
    let Some(state) = tool_states.remove(&index) else {
        return vec![];
    };

    // Empty input → default to empty JSON object (Anthropic API requirement)
    let json_str = if state.json_buffer.is_empty() {
        "{}".to_string()
    } else {
        state.json_buffer
    };
    let arguments: Value = serde_json::from_str(&json_str).unwrap_or_default();

    vec![StreamEvent::ToolCallComplete {
        index,
        call: ToolCall {
            id: state.id,
            name: state.name,
            arguments,
        },
    }]
}

fn handle_message_delta(response: &StreamResponse) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    if let Some(delta) = &response.delta {
        if let Some(reason) = &delta.stop_reason {
            let stop_reason = match reason.as_str() {
                "tool_use" => StopReason::ToolUse,
                "max_tokens" => StopReason::MaxTokens,
                "stop_sequence" => StopReason::StopSequence,
                _ => StopReason::EndTurn,
            };
            events.push(StreamEvent::Done { stop_reason });
        }
    }
    if let Some(usage) = &response.usage {
        events.push(StreamEvent::Usage(convert_usage(usage)));
    }

    events
}

/// Extract the `data: ` payload from an SSE event text block.
fn extract_data_line(event_text: &str) -> Option<&str> {
    for line in event_text.lines() {
        // Only strip trailing \r (from \r\n line endings), not leading whitespace
        let line = line.trim_end_matches('\r');
        if let Some(data) = line.strip_prefix("data: ") {
            return Some(data);
        }
    }
    None
}

/// Convert Anthropic `ResponseUsage` to core `Usage`.
fn convert_usage(usage: &ResponseUsage) -> Usage {
    Usage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        reasoning_tokens: None,
        cache_read_tokens: usage.cache_read_input_tokens,
        cache_write_tokens: usage.cache_creation_input_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_data_line_basic() {
        let event = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\n";
        assert_eq!(
            extract_data_line(event),
            Some("{\"type\":\"content_block_delta\"}")
        );
    }

    #[test]
    fn test_extract_data_line_no_data() {
        assert_eq!(extract_data_line("event: ping\n\n"), None);
    }

    #[test]
    fn test_extract_data_line_done() {
        let event = "data: [DONE]\n\n";
        assert_eq!(extract_data_line(event), Some("[DONE]"));
    }

    #[test]
    fn test_parse_text_delta() {
        let event = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

"#;
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0], StreamEvent::TextDelta("Hello".into()));
    }

    #[test]
    fn test_parse_thinking_delta() {
        let event = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think..."}}

"#;
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            StreamEvent::ReasoningDelta("Let me think...".into())
        );
    }

    #[test]
    fn test_parse_tool_use_lifecycle() {
        let mut tool_states = HashMap::new();

        // 1. content_block_start with tool_use
        let start_event = r#"event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01", "name": "get_weather"}}

"#;
        let events = parse_sse_event(start_event, &mut tool_states);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            StreamEvent::ToolCallStart { index: 1, id, name }
                if id == "toolu_01" && name == "get_weather"
        ));
        assert!(tool_states.contains_key(&1));

        // 2. content_block_delta with input_json_delta
        let delta_event = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}

"#;
        let events = parse_sse_event(delta_event, &mut tool_states);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            StreamEvent::ToolCallDelta { index: 1, json_chunk }
                if json_chunk == "{\"city\":"
        ));

        // 3. Another delta
        let delta_event2 = r#"event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"Tokyo\"}"}}

"#;
        let events = parse_sse_event(delta_event2, &mut tool_states);
        assert_eq!(events.len(), 1);

        // 4. content_block_stop → tool call complete
        let stop_event = r#"event: content_block_stop
data: {"type": "content_block_stop", "index": 1}

"#;
        let events = parse_sse_event(stop_event, &mut tool_states);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            StreamEvent::ToolCallComplete { index: 1, call }
                if call.name == "get_weather" && call.arguments["city"] == "Tokyo"
        ));
        assert!(!tool_states.contains_key(&1));
    }

    #[test]
    fn test_parse_tool_use_empty_arguments() {
        let mut tool_states = HashMap::new();

        // Start a tool with no parameters
        let start = r#"event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_02", "name": "no_args_tool"}}

"#;
        parse_sse_event(start, &mut tool_states);

        // Immediately stop (no deltas)
        let stop = r#"event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

"#;
        let events = parse_sse_event(stop, &mut tool_states);
        assert_eq!(events.len(), 1);

        // Empty args should produce empty JSON object
        if let StreamEvent::ToolCallComplete { call, .. } = &events[0] {
            assert_eq!(call.arguments, serde_json::json!({}));
        } else {
            panic!("Expected ToolCallComplete");
        }
    }

    #[test]
    fn test_parse_message_delta_done() {
        let event = r#"event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 0, "output_tokens": 15}}

"#;
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);

        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0],
            StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
            }
        );
        assert!(matches!(&events[1], StreamEvent::Usage(u) if u.output_tokens == 15));
    }

    #[test]
    fn test_parse_message_delta_tool_use_stop() {
        let event = r#"event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"input_tokens": 0, "output_tokens": 30}}

"#;
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);

        assert!(matches!(
            &events[0],
            StreamEvent::Done {
                stop_reason: StopReason::ToolUse,
            }
        ));
    }

    #[test]
    fn test_parse_message_start_usage() {
        let event = r#"event: message_start
data: {"type": "message_start", "message": {"usage": {"input_tokens": 42, "output_tokens": 0}}}

"#;
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);

        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::Usage(u) if u.input_tokens == 42));
    }

    #[test]
    fn test_parse_ping_ignored() {
        let event = "event: ping\ndata: {}\n\n";
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);
        assert!(events.is_empty());
    }

    #[test]
    fn test_parse_done_sentinel_ignored() {
        let event = "data: [DONE]\n\n";
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);
        assert!(events.is_empty());
    }

    #[test]
    fn test_parse_message_stop_ignored() {
        let event = r#"event: message_stop
data: {"type": "message_stop"}

"#;
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);
        assert!(events.is_empty());
    }

    #[test]
    fn test_content_block_stop_without_tool_state_is_noop() {
        let event = r#"event: content_block_stop
data: {"type": "content_block_stop", "index": 5}

"#;
        let mut tool_states = HashMap::new();
        let events = parse_sse_event(event, &mut tool_states);
        assert!(events.is_empty());
    }

    #[test]
    fn test_convert_usage() {
        let api_usage = ResponseUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: Some(20),
            cache_read_input_tokens: Some(10),
        };
        let usage = convert_usage(&api_usage);

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_write_tokens, Some(20));
        assert_eq!(usage.cache_read_tokens, Some(10));
        assert!(usage.reasoning_tokens.is_none());
    }
}
