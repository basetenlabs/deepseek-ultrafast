use crate::chat::{self, PreparedPrompt};
use anyhow::{Result, ensure};
use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::anthropic::messages::response::MessagesStreamEvent;
use deepseek_recipe::request::{ConversationRequest, ConversionOptions, ProtocolRequest, WebSearchBehavior};
use deepseek_recipe::stream::{InferenceChunk, StreamError, StreamProcessor};
use deepseek_recipe_core::tools::ToolChoice;
use serde_json::Value;

const MAX_MESSAGES: usize = 4096;

pub fn prepare(mut body: Value, model: &str) -> Result<PreparedPrompt> {
    let _ = crate::output_options::validate(&body)?;
    crate::thinking::native(&mut body)?;
    validate(&body)?;
    let object = body.as_object_mut().expect("validated object");
    object.remove("context_management");
    object.entry("model").or_insert_with(|| model.into());
    if let Some(tools) = object.get_mut("tools").and_then(Value::as_array_mut) {
        for tool in tools {
            if tool.get("type").and_then(Value::as_str) == Some("custom") {
                tool.as_object_mut().unwrap().remove("type");
            }
        }
    }
    chat::prepare_conversation(body, |body| {
        let required = body.get("tool_choice").and_then(|v| v.get("type")).and_then(Value::as_str) == Some("any");
        let request: MessagesRequest = serde_json::from_value(body)?;
        let mut request = request.convert(ConversionOptions::default().with_messages_web_search(WebSearchBehavior::Reject))?;
        if required {
            ensure!(!request.conversation.tools.is_empty(), "tool_choice any requires tools");
            request.conversation.tool_choice = ToolChoice::Required;
            request.parsing_options.tool_call_initial_stage = true;
        }
        Ok(request)
    })
}

fn content(value: Option<&Value>) -> Result<()> {
    if let Some(blocks) = value.and_then(Value::as_array) {
        for block in blocks {
            ensure!(block.get("type").and_then(Value::as_str) != Some("document"), "document blocks are not supported");
            if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                content(block.get("content"))?;
            }
            ensure!(block.get("citations").is_none_or(|v| v.is_null() || v.as_array().is_some_and(Vec::is_empty)), "citations are not supported");
        }
    }
    Ok(())
}

fn clears_nothing(context_management: &Value) -> bool {
    context_management.get("edits").and_then(Value::as_array).is_some_and(|edits| {
        edits.iter().all(|e| e.get("type").and_then(Value::as_str).is_some_and(|t| t.starts_with("clear_thinking_")) && e.get("keep").and_then(Value::as_str) == Some("all"))
    })
}

fn validate(body: &Value) -> Result<()> {
    ensure!(body.get("max_tokens").and_then(Value::as_u64).is_some_and(|n| n > 0), "max_tokens must be a positive integer");
    ensure!(body.get("n").is_none_or(|v| v.is_null() || v.as_u64() == Some(1)), "only n = 1 is supported");
    ensure!(
        body.get("messages").and_then(Value::as_array).is_some_and(|v| !v.is_empty() && v.len() <= MAX_MESSAGES),
        "messages must contain 1..=4096 full messages"
    );
    for field in [
        "previous_response_id",
        "conversation",
        "container",
        "mcp_servers",
        "top_k",
        "max_completion_tokens",
        "max_output_tokens",
        "stop",
        "stream_options",
    ] {
        ensure!(body.get(field).is_none_or(Value::is_null), "{field} is not supported on stateless Messages");
    }
    for field in ["store", "background"] {
        ensure!(body.get(field).is_none_or(|v| v.is_null() || v == false), "{field}=true is not supported on stateless Messages");
    }
    ensure!(
        body.get("context_management").is_none_or(|v| v.is_null() || clears_nothing(v)),
        "context_management edits that clear content are not supported on stateless Messages"
    );
    ensure!(
        body.get("thinking").and_then(|v| v.get("budget_tokens")).is_none_or(Value::is_null),
        "thinking.budget_tokens is not enforced; use thinking.type enabled/adaptive/disabled and max_tokens"
    );
    ensure!(
        body.get("tool_choice").and_then(|v| v.get("disable_parallel_tool_use")).is_none_or(|v| v.is_null() || v == false),
        "disable_parallel_tool_use=true is not supported"
    );
    if let Some(temperature) = body.get("temperature").filter(|v| !v.is_null()) {
        ensure!(temperature.as_f64() == Some(1.0), "temperature must be 1");
    }
    if let Some(top_p) = body.get("top_p").filter(|v| !v.is_null()) {
        ensure!(top_p.as_f64() == Some(1.0), "top_p must be 1");
    }
    content(body.get("system"))?;
    for message in body["messages"].as_array().unwrap() {
        content(message.get("content"))?;
    }
    Ok(())
}

pub fn events(
    request: ConversationRequest,
    inference: impl futures::Stream<Item = InferenceChunk> + Send,
    id: String,
    model: String,
) -> impl futures::Stream<Item = Result<MessagesStreamEvent, StreamError>> + Send {
    let generator = MessagesRequest::chunk_generator(&request, id, model);
    StreamProcessor::new(generator, request.parsing_options).process(inference)
}

#[cfg(test)]
mod context_management_tests {
    use super::clears_nothing;
    use serde_json::json;

    #[test]
    fn only_edits_that_keep_everything() {
        assert!(clears_nothing(&json!({"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]})));
        assert!(clears_nothing(&json!({"edits": []})));
        assert!(!clears_nothing(&json!({"edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 1}}]})));
        assert!(!clears_nothing(&json!({"edits": [{"type": "clear_tool_uses_20250919"}]})));
        assert!(!clears_nothing(&json!({})));
    }
}
