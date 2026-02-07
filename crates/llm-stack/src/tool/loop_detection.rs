//! Loop detection for tool loops.

use serde_json::Value;

use crate::chat::{ChatMessage, ChatResponse, ToolCall};
use crate::usage::Usage;

use super::config::{
    LoopAction, LoopDetectionConfig, LoopEvent, TerminationReason, ToolLoopConfig, ToolLoopResult,
};

/// Read-only snapshot of the current iteration state.
///
/// Bundles the data that both stop-condition checks and loop-detection checks
/// need, avoiding long argument lists across the three loop implementations
/// (sync, streaming, resumable).
pub(crate) struct IterationSnapshot<'a> {
    pub response: &'a ChatResponse,
    pub call_refs: &'a [&'a ToolCall],
    pub iterations: u32,
    pub total_usage: &'a Usage,
    pub config: &'a ToolLoopConfig,
}

/// Tracks consecutive identical tool calls for loop detection.
#[derive(Debug, Default)]
pub(crate) struct LoopDetectionState {
    /// Hash of the last tool calls (name + args combined).
    last_hash: Option<u64>,
    /// Tool name(s) from the last call, for error reporting.
    last_tool_name: String,
    /// Count of consecutive identical calls.
    consecutive_count: u32,
}

impl LoopDetectionState {
    /// Update state with new tool calls and return loop info if threshold hit.
    ///
    /// Returns `Some((tool_name, count))` if the threshold was reached.
    #[cfg(test)]
    pub(crate) fn update(&mut self, calls: &[ToolCall], threshold: u32) -> Option<(String, u32)> {
        // Convert to refs and delegate
        let refs: Vec<&ToolCall> = calls.iter().collect();
        self.update_refs(&refs, threshold)
    }

    /// Update state with tool call references (more efficient when refs are already available).
    pub(crate) fn update_refs(
        &mut self,
        calls: &[&ToolCall],
        threshold: u32,
    ) -> Option<(String, u32)> {
        // Compute hash signature from the tool calls
        let (hash, tool_name) = compute_tool_calls_hash(calls);

        if self.last_hash == Some(hash) {
            self.consecutive_count += 1;
            if self.consecutive_count >= threshold {
                return Some((self.last_tool_name.clone(), self.consecutive_count));
            }
        } else {
            self.last_hash = Some(hash);
            self.last_tool_name = tool_name;
            self.consecutive_count = 1;
        }
        None
    }

    /// Reset the detection state (e.g., after injecting a warning).
    pub(crate) fn reset(&mut self) {
        self.last_hash = None;
        self.last_tool_name.clear();
        self.consecutive_count = 0;
    }
}

/// Compute a hash-based signature for tool calls.
///
/// Returns `(hash, tool_name)` where `tool_name` is for error reporting.
/// Uses hash for efficient comparison without string allocations.
fn compute_tool_calls_hash(calls: &[&ToolCall]) -> (u64, String) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    if calls.is_empty() {
        return (0, String::new());
    }

    let mut hasher = DefaultHasher::new();

    // Hash all tool names and arguments
    for call in calls {
        call.name.hash(&mut hasher);
        hash_json_value(&call.arguments, &mut hasher);
    }

    // Build tool name(s) for error reporting
    let tool_name = if calls.len() == 1 {
        calls[0].name.clone()
    } else {
        calls
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join("+")
    };

    (hasher.finish(), tool_name)
}

/// Hash a JSON value in a deterministic way.
///
/// This hashes the structure and values without allocating strings.
fn hash_json_value<H: std::hash::Hasher>(value: &Value, hasher: &mut H) {
    use std::hash::Hash;

    match value {
        Value::Null => 0u8.hash(hasher),
        Value::Bool(b) => {
            1u8.hash(hasher);
            b.hash(hasher);
        }
        Value::Number(n) => {
            2u8.hash(hasher);
            // Hash the canonical string representation of the number
            n.to_string().hash(hasher);
        }
        Value::String(s) => {
            3u8.hash(hasher);
            s.hash(hasher);
        }
        Value::Array(arr) => {
            4u8.hash(hasher);
            arr.len().hash(hasher);
            for item in arr {
                hash_json_value(item, hasher);
            }
        }
        Value::Object(obj) => {
            5u8.hash(hasher);
            obj.len().hash(hasher);
            // Sort keys for deterministic ordering
            let mut keys: Vec<_> = obj.keys().collect();
            keys.sort();
            for key in keys {
                key.hash(hasher);
                hash_json_value(&obj[key], hasher);
            }
        }
    }
}

// Test-only string-based signature for backwards compatibility
#[cfg(test)]
pub(crate) fn compute_tool_calls_signature(calls: &[ToolCall]) -> (String, String) {
    if calls.is_empty() {
        return (String::new(), String::new());
    }

    if calls.len() == 1 {
        let call = &calls[0];
        let args = serde_json::to_string(&call.arguments).unwrap_or_default();
        return (call.name.clone(), args);
    }

    let mut names = Vec::with_capacity(calls.len());
    let mut args_parts = Vec::with_capacity(calls.len());
    for call in calls {
        names.push(call.name.as_str());
        args_parts.push(serde_json::to_string(&call.arguments).unwrap_or_default());
    }
    (names.join("+"), args_parts.join("|"))
}

/// Result of checking loop detection.
pub(crate) enum LoopCheckResult {
    /// No loop detected, continue normally.
    Continue,
    /// Loop detected, stop with error.
    Stop { tool_name: String, count: u32 },
    /// Loop detected, inject warning message.
    InjectWarning { tool_name: String, count: u32 },
}

/// Check for loop and determine action.
///
/// If a loop is detected, pushes a `LoopEvent::LoopDetected` into
/// the provided event buffer.
pub(crate) fn check_loop_detection_refs(
    state: &mut LoopDetectionState,
    calls: &[&ToolCall],
    config: Option<&LoopDetectionConfig>,
    events: &mut Vec<LoopEvent>,
) -> LoopCheckResult {
    let Some(detection) = config else {
        return LoopCheckResult::Continue;
    };

    if let Some((tool_name, count)) = state.update_refs(calls, detection.threshold) {
        let action = detection.action;
        events.push(LoopEvent::LoopDetected {
            tool_name: tool_name.clone(),
            consecutive_count: count,
            action,
        });

        match detection.action {
            LoopAction::Warn => LoopCheckResult::Continue,
            LoopAction::Stop => LoopCheckResult::Stop { tool_name, count },
            LoopAction::InjectWarning => LoopCheckResult::InjectWarning { tool_name, count },
        }
    } else {
        LoopCheckResult::Continue
    }
}

/// Create a warning message to inject when a loop is detected.
pub(crate) fn create_loop_warning_message(tool_name: &str, count: u32) -> ChatMessage {
    ChatMessage::system(format!(
        "Warning: You have called the tool '{tool_name}' with identical arguments {count} times in a row. \
         This appears to be a loop. Please try a different approach or tool."
    ))
}

/// Handle loop detection result, returning a termination result if the loop should stop.
///
/// Mutates `messages` if the action is `InjectWarning`.
/// Pushes `LoopEvent::LoopDetected` into `events` when a loop is detected.
pub(crate) fn handle_loop_detection(
    state: &mut LoopDetectionState,
    snap: &IterationSnapshot<'_>,
    messages: &mut Vec<ChatMessage>,
    events: &mut Vec<LoopEvent>,
) -> Option<ToolLoopResult> {
    match check_loop_detection_refs(
        state,
        snap.call_refs,
        snap.config.loop_detection.as_ref(),
        events,
    ) {
        LoopCheckResult::Continue => None,
        LoopCheckResult::Stop { tool_name, count } => Some(ToolLoopResult {
            response: snap.response.clone(),
            iterations: snap.iterations,
            total_usage: snap.total_usage.clone(),
            termination_reason: TerminationReason::LoopDetected { tool_name, count },
        }),
        LoopCheckResult::InjectWarning { tool_name, count } => {
            messages.push(create_loop_warning_message(&tool_name, count));
            state.reset();
            None
        }
    }
}
