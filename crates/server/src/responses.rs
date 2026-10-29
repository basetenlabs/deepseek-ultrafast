use crate::chat;
use anyhow::{Result, anyhow, ensure};
use deepseek_ultrafast_model::sampling::{DEFAULT_TEMPERATURE, DEFAULT_TOP_P};
use deepseek_recipe::openai::ResponsesRequest;
use deepseek_recipe::openai::responses::response::{ResponsesOutputItem, ResponsesStreamEvent};
use deepseek_recipe::request::{ConversationRequest, ConversionOptions, ProtocolRequest, WebSearchBehavior};
use deepseek_recipe::stream::{InferenceChunk, StreamError, StreamProcessor};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

pub struct PreparedResponses {
    pub prompt: String,
    pub literal_breaks: Vec<usize>,
    pub image_offsets: Vec<usize>,
    pub image_sources: Vec<deepseek_recipe_core::multimodal::ImageSource>,
    pub request: ConversationRequest,
    pub custom_tools: HashSet<String>,
    pub seed: Option<i64>,
    pub response_options: Value,
}

pub fn prepare(mut body: Value, model: &str) -> Result<PreparedResponses> {
    crate::thinking::responses(&mut body)?;
    validate(&body)?;
    let value_or = |key: &str, default: Value| body.get(key).filter(|v| !v.is_null()).cloned().unwrap_or(default);
    let mut tools = value_or("tools", json!([]));
    if let Some(tools) = tools.as_array_mut() {
        for tool in tools {
            if tool["type"] == "namespace" && tool.get("description").is_none_or(Value::is_null) {
                tool["description"] = "".into();
            }
        }
    }
    let response_options = json!({
        "store": false, "background": false, "parallel_tool_calls": true,
        "tool_choice": value_or("tool_choice", json!("auto")), "tools": tools,
        "temperature": value_or("temperature", json!(DEFAULT_TEMPERATURE)), "top_p": value_or("top_p", json!(DEFAULT_TOP_P)),
        "max_output_tokens": body.get("max_output_tokens"), "instructions": body.get("instructions"),
        "reasoning": value_or("reasoning", json!({"effort":"high"})), "text": {"format":{"type":"text"}},
        "metadata": value_or("metadata", json!({})), "user": body.get("user"),
        "previous_response_id": null, "truncation": "disabled"
    });
    let requires_tools = body.get("tool_choice").is_some_and(|v| !v.is_null() && v != "auto" && v != "none");
    let object = body.as_object_mut().ok_or_else(|| anyhow!("request must be an object"))?;
    object.entry("model").or_insert_with(|| model.into());
    if let Some(choice) = object.get_mut("tool_choice").and_then(Value::as_object_mut)
        && let Some(namespace) = choice.get("namespace").filter(|v| !v.is_null())
    {
        ensure!(choice.get("type").and_then(Value::as_str) == Some("function"), "only function tool selectors support namespaces");
        let namespace =
            namespace.as_str().filter(|s| !s.is_empty()).ok_or_else(|| anyhow!("tool_choice.namespace must be a nonempty string"))?;
        let name = choice.get("name").and_then(Value::as_str).ok_or_else(|| anyhow!("tool_choice.name is required"))?;
        choice.insert("name".into(), format!("{namespace}::{name}").into());
        choice.remove("namespace");
    }
    let seed = object
        .get("seed")
        .filter(|v| !v.is_null())
        .map(|v| v.as_i64().ok_or_else(|| anyhow!("seed must be a signed 64-bit integer")))
        .transpose()?;
    if let Some(items) = object.get_mut("input").and_then(Value::as_array_mut) {
        for item in items {
            if item.get("role").and_then(Value::as_str) == Some("developer") {
                item["role"] = "system".into();
            }
        }
    }
    let mut custom_tools = HashSet::new();
    let prompt = chat::prepare_conversation(body, |body| {
        let request: ResponsesRequest = serde_json::from_value(body)?;
        custom_tools = request.custom_tool_names();
        let request = request.convert(ConversionOptions::default().with_responses_web_search(WebSearchBehavior::Reject))?;
        ensure!(!requires_tools || !request.conversation.tools.is_empty(), "tool_choice requires at least one matching tool");
        Ok(request)
    })?;
    Ok(PreparedResponses { prompt: prompt.prompt, literal_breaks: prompt.literal_breaks, image_offsets: prompt.image_offsets, image_sources: prompt.image_sources,
        request: prompt.request, custom_tools, seed, response_options })
}

fn validate(body: &Value) -> Result<()> {
    ensure!(body.is_object(), "request must be an object");
    if let Some(metadata) = body.get("metadata").filter(|v| !v.is_null()) {
        ensure!(metadata.as_object().is_some_and(|m| m.values().all(Value::is_string)), "metadata must be an object of string values");
    }
    ensure!(body.get("text").and_then(|v| v.get("verbosity")).is_none_or(Value::is_null), "text.verbosity is not supported");
    for field in ["previous_response_id", "conversation"] {
        ensure!(
            body.get(field).is_none_or(Value::is_null),
            "{field} is not supported: resend the full input history (stateless Responses)"
        );
    }
    for field in ["store", "background"] {
        ensure!(body.get(field).is_none_or(|v| v.is_null() || v == false), "{field}=true is not supported; use {field}=false");
    }
    ensure!(body.get("truncation").is_none_or(|v| v.is_null() || v == "disabled"), "automatic truncation is not supported");
    ensure!(body.get("parallel_tool_calls").is_none_or(|v| v.is_null() || v == true), "parallel_tool_calls=false is not supported");
    ensure!(
        body.get("include").is_none_or(|v| v.is_null() || v.as_array().is_some_and(Vec::is_empty)),
        "include extensions (including encrypted reasoning and logprobs) are not supported"
    );
    for field in [
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "n",
        "stream_options",
        "thinking",
        "reasoning_effort",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "top_logprobs",
    ] {
        ensure!(
            body.get(field).is_none_or(Value::is_null),
            "{field} is not supported on Responses; use max_output_tokens and reasoning.effort"
        );
    }
    ensure!(body.get("logprobs").is_none_or(|v| v.is_null() || v == false), "logprobs are not supported");
    if let Some(reasoning) = body.get("reasoning").filter(|v| !v.is_null()) {
        for field in ["enabled", "summary", "generate_summary"] {
            ensure!(
                reasoning.get(field).is_none_or(Value::is_null),
                "reasoning.{field} is not supported; use reasoning.effort (plaintext reasoning is returned)"
            );
        }
    }
    if let Some(format) = body.get("text").and_then(|v| v.get("format")).filter(|v| !v.is_null()) {
        ensure!(format.get("type").and_then(Value::as_str) == Some("text"), "text.format other than text is not supported");
    }
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for tool in tools {
            validate_tool(tool, false)?;
        }
    }
    if let Some(items) = body.get("input").and_then(Value::as_array) {
        for item in items {
            match item.get("type").and_then(Value::as_str) {
                None | Some("message") => validate_content(item.get("content"))?,
                Some("function_call" | "custom_tool_call") => {}
                Some("function_call_output" | "custom_tool_call_output") => validate_content(item.get("output"))?,
                Some("reasoning") => {
                    ensure!(
                        item.get("encrypted_content").is_none_or(Value::is_null),
                        "encrypted reasoning recovery is not supported; resend plaintext content"
                    );
                    ensure!(
                        item.get("summary").is_none_or(|v| v.is_null() || v.as_array().is_some_and(Vec::is_empty)),
                        "reasoning summary input is not supported; resend plaintext content"
                    );
                }
                Some(kind) => anyhow::bail!("unsupported Responses input item: {kind}"),
            }
        }
    }
    Ok(())
}

fn validate_tool(tool: &Value, inside_namespace: bool) -> Result<()> {
    match tool.get("type").and_then(Value::as_str) {
        Some("function") => {
            ensure!(
                tool.get("namespace").is_none_or(Value::is_null),
                "declare functions inside a namespace tool instead of using a function.namespace field"
            );
        }
        Some("namespace") if !inside_namespace => {
            if let Some(tools) = tool.get("tools").and_then(Value::as_array) {
                for tool in tools {
                    validate_tool(tool, true)?;
                }
            }
        }
        Some("custom") if !inside_namespace => {
            ensure!(
                tool.get("format").is_none_or(|v| v.is_null() || v.get("type").and_then(Value::as_str) == Some("text")),
                "custom tool format constraints are not supported"
            );
        }
        _ => anyhow::bail!("unsupported tool type (only client function, namespace and apply_patch custom tools are supported)"),
    }
    Ok(())
}

fn validate_content(content: Option<&Value>) -> Result<()> {
    if let Some(blocks) = content.and_then(Value::as_array) {
        for block in blocks {
            ensure!(
                !matches!(block.get("type").and_then(Value::as_str), Some("input_file")),
                "file inputs are not supported"
            );
        }
    }
    Ok(())
}

pub fn events(
    request: ConversationRequest,
    custom_tools: HashSet<String>,
    inference: impl futures::Stream<Item = InferenceChunk> + Send,
    id: String,
    model: String,
) -> impl futures::Stream<Item = Result<ResponsesStreamEvent, StreamError>> + Send {
    let generator = ResponsesRequest::chunk_generator(&request, id, model).with_custom_tool_names(custom_tools);
    StreamProcessor::new(generator, request.parsing_options).process(inference)
}

pub fn wire_response(response: impl serde::Serialize, options: &Value) -> Value {
    let mut value = serde_json::to_value(response).expect("recipe response is JSON");
    add_options(&mut value, options);
    value
}

pub struct WireEvents {
    options: Value,
    function_names: HashMap<String, String>,
}

impl WireEvents {
    pub fn new(options: Value) -> Self {
        Self { options, function_names: HashMap::new() }
    }

    pub fn event(&mut self, event: ResponsesStreamEvent) -> Value {
        let name = match &event {
            ResponsesStreamEvent::OutputItemAdded { item: ResponsesOutputItem::FunctionCall { id, name, .. }, .. } => {
                self.function_names.insert(id.clone(), name.clone());
                None
            }
            ResponsesStreamEvent::FunctionCallArgumentsDone { item_id, .. } => {
                Some(self.function_names.remove(item_id).expect("recipe function call was added"))
            }
            _ => None,
        };
        let mut value = serde_json::to_value(event).expect("recipe event is JSON");
        if let Some(name) = name {
            value["name"] = name.into();
        }
        if let Some(response) = value.get_mut("response") {
            add_options(response, &self.options);
        }
        value
    }
}

fn add_options(response: &mut Value, options: &Value) {
    response.as_object_mut().unwrap().extend(options.as_object().unwrap().clone());
    if let Some(details) = response.get_mut("usage").and_then(|u| u.get_mut("input_tokens_details")) {
        details["cache_write_tokens"] = 0.into();
    }
}
