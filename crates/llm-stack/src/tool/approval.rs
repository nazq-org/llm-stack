//! Tool call approval handling.

use crate::chat::{ToolCall, ToolResult};

use super::config::{ToolApproval, ToolLoopConfig};

/// Partition tool calls into approved (possibly modified) and denied results.
///
/// Accepts owned calls to avoid deep-cloning `serde_json::Value` arguments.
/// Callers typically get owned `Vec<ToolCall>` from `partition_content()`.
pub(crate) fn approve_calls(
    calls: Vec<ToolCall>,
    config: &ToolLoopConfig,
) -> (Vec<ToolCall>, Vec<ToolResult>) {
    let mut approved = Vec::with_capacity(calls.len());
    let mut denied = Vec::new();

    for call in calls {
        let approval = config
            .on_tool_call
            .as_ref()
            .map_or(ToolApproval::Approve, |f| f(&call));

        match approval {
            ToolApproval::Approve => approved.push(call),
            ToolApproval::Deny(reason) => {
                denied.push(ToolResult {
                    tool_call_id: call.id,
                    content: reason,
                    is_error: true,
                });
            }
            ToolApproval::Modify(new_args) => {
                approved.push(ToolCall {
                    id: call.id,
                    name: call.name,
                    arguments: new_args,
                });
            }
        }
    }

    (approved, denied)
}
