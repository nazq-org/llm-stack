//! Tool call approval handling.

use crate::chat::{ToolCall, ToolResult};

use super::config::{ToolApproval, ToolLoopConfig};

/// Partition tool calls into approved (possibly modified) and denied results.
pub(crate) fn approve_calls(
    calls: &[ToolCall],
    config: &ToolLoopConfig,
) -> (Vec<ToolCall>, Vec<ToolResult>) {
    let mut approved = Vec::with_capacity(calls.len());
    let mut denied = Vec::new();

    for call in calls {
        let approval = config
            .on_tool_call
            .as_ref()
            .map_or(ToolApproval::Approve, |f| f(call));

        match approval {
            ToolApproval::Approve => approved.push(call.clone()),
            ToolApproval::Deny(reason) => {
                denied.push(ToolResult {
                    tool_call_id: call.id.clone(),
                    content: reason,
                    is_error: true,
                });
            }
            ToolApproval::Modify(new_args) => {
                let mut modified = call.clone();
                modified.arguments = new_args;
                approved.push(modified);
            }
        }
    }

    (approved, denied)
}
