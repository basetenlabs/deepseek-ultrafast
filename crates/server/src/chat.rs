use anyhow::{Result, anyhow, ensure};
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::openai::chat_completion::response::ChatCompletionChunk;
use deepseek_recipe::request::{ConversationRequest, ConversionOptions, ProtocolRequest};
use deepseek_recipe::stream::{InferenceChunk, StreamError, StreamProcessor};
use deepseek_recipe_core::multimodal::{IMAGE_SPECIAL_TOKEN, ImageSource};
use deepseek_recipe_core::conversation::Conversation;
use deepseek_recipe_encoding::{PromptEncoding, dsv41::DeepseekV41Encoding};
use serde_json::Value;
use tokenizers::Tokenizer;

pub use deepseek_recipe_encoding::v4::EOS_TOKEN as EOS;

pub struct PreparedPrompt {
    pub request: ConversationRequest,
    pub prompt: String,
    pub literal_breaks: Vec<usize>,
    pub image_offsets: Vec<usize>,
    pub image_sources: Vec<ImageSource>,
}

fn normalize(body: &mut Value, model: &str) -> Result<Option<u64>> {
    crate::thinking::native(body)?;
    let body = body.as_object_mut().ok_or_else(|| anyhow!("request must be an object"))?;
    body.entry("model").or_insert_with(|| model.into());
    if let Some(value) = body.get("max_completion_tokens").filter(|v| !v.is_null()).cloned() {
        body.insert("max_tokens".into(), value);
    }
    body.remove("seed");
    if body.get("reasoning_effort").is_none_or(Value::is_null)
        && let Some(effort) = body.get("chat_template_kwargs").and_then(|k| k.get("reasoning_effort")).cloned()
    {
        body.insert("reasoning_effort".into(), effort);
    }
    let numeric_effort = if let Some(Value::Number(n)) = body.get("reasoning_effort") {
        let effort = n.as_u64().filter(|e| (1..=100).contains(e)).ok_or_else(|| anyhow!("reasoning_effort must be an integer in [1, 100]"))?;
        body.insert("reasoning_effort".into(), "high".into());
        Some(effort)
    } else {
        None
    };
    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            if message.get("role").and_then(Value::as_str) == Some("developer") {
                message["role"] = "system".into();
            }
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    reject_namespace(call)?;
                }
            }
        }
    }
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for tool in tools {
            reject_namespace(tool)?;
        }
    }
    ensure!(body.get("logprobs").is_none_or(|v| matches!(v, Value::Null | Value::Bool(false))), "logprobs are not supported");
    ensure!(body.get("top_logprobs").is_none_or(Value::is_null), "top_logprobs are not supported");
    ensure!(
        body.get("response_format").is_none_or(|v| v.is_null() || v.get("type").and_then(Value::as_str) == Some("text")),
        "response_format other than text is not supported"
    );
    Ok(numeric_effort)
}

fn reject_namespace(tool: &Value) -> Result<()> {
    ensure!(
        tool.get("namespace").is_none_or(Value::is_null) && tool.get("function").and_then(|f| f.get("namespace")).is_none_or(Value::is_null),
        "namespace fields are not supported for Chat Completions tools"
    );
    Ok(())
}

fn convert(body: Value) -> Result<ConversationRequest> {
    let request: ChatCompletionRequest = serde_json::from_value(body)?;
    Ok(request.convert(ConversionOptions::default())?)
}

pub fn prepare(mut body: Value, model: &str) -> Result<PreparedPrompt> {
    let numeric_effort = normalize(&mut body, model)?;
    let mut prepared = prepare_conversation(body, convert)?;
    if let Some(effort) = numeric_effort.filter(|_| prepared.request.conversation.thinking_mode) {
        const HEADER: &str = "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75";
        let rest = prepared.prompt.strip_prefix(HEADER).ok_or_else(|| anyhow!("recipe reasoning effort header changed"))?;
        prepared.prompt = format!("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {effort}{rest}");
        let digits = effort.to_string().len();
        for offset in prepared.literal_breaks.iter_mut().chain(&mut prepared.image_offsets) {
            *offset = *offset - 2 + digits;
        }
    }
    Ok(prepared)
}

pub(crate) fn prepare_conversation(body: Value, mut convert: impl FnMut(Value) -> Result<ConversationRequest>) -> Result<PreparedPrompt> {
    let encoding = DeepseekV41Encoding::new();
    let render = |conversation: &Conversation| encoding.render_conversation(conversation);
    let converted = convert(body.clone());
    let mut protected = false;
    let mut request = match converted {
        Ok(request) => request,
        Err(error) => match protect_image_text(&body) {
            Some(body) => {
                protected = true;
                convert(body).map_err(|e| anyhow!(restore_image_text(&e.to_string())))?
            }
            None => return Err(error),
        },
    };
    let mut rendered = render(&request.conversation);
    if !protected
        && rendered.prompt.matches(IMAGE_SPECIAL_TOKEN).count() > rendered.image_sources.len()
        && let Some(body) = protect_image_text(&body)
    {
        protected = true;
        request = convert(body).map_err(|e| anyhow!(restore_image_text(&e.to_string())))?;
        rendered = render(&request.conversation);
    }
    let literal = crate::literal_text::protect(&request.conversation);
    if let Some(conversation) = &literal {
        rendered = render(conversation);
    }
    let mut prompt = String::new();
    let mut literal_breaks = Vec::new();
    let append = |text: &str, prompt: &mut String, breaks: &mut Vec<usize>| -> Result<()> {
        let text = if protected { std::borrow::Cow::Owned(restore_image_text(text)) } else { std::borrow::Cow::Borrowed(text) };
        if literal.is_some() {
            crate::literal_text::append(&text, prompt, breaks)?;
        } else {
            prompt.push_str(&text);
        }
        Ok(())
    };
    let mut image_offsets = Vec::new();
    let mut at = 0;
    for (offset, _) in rendered.prompt.match_indices(IMAGE_SPECIAL_TOKEN) {
        let text = &rendered.prompt[at..offset];
        append(text, &mut prompt, &mut literal_breaks)?;
        image_offsets.push(prompt.len());
        prompt.push_str(IMAGE_SPECIAL_TOKEN);
        at = offset + IMAGE_SPECIAL_TOKEN.len();
    }
    let text = &rendered.prompt[at..];
    append(text, &mut prompt, &mut literal_breaks)?;
    ensure!(image_offsets.len() == rendered.image_sources.len(), "image source/placeholder mismatch");
    if protected {
        for tool in &mut request.conversation.tools {
            tool.parameters = serde_json::from_str(&restore_image_text(&tool.parameters.to_string()))?;
        }
    }
    Ok(PreparedPrompt { request, prompt, literal_breaks, image_offsets, image_sources: rendered.image_sources })
}

pub fn responses(
    request: ConversationRequest,
    inference: impl futures::Stream<Item = InferenceChunk> + Send,
    id: String,
    model: String,
) -> impl futures::Stream<Item = Result<ChatCompletionChunk, StreamError>> + Send {
    let generator = ChatCompletionRequest::chunk_generator(&request, id, model);
    StreamProcessor::new(generator, request.parsing_options).process(inference)
}

const LITERAL_MARKER: &str = "<_literal_image_>";

fn protect_image_text(body: &Value) -> Option<Value> {
    fn visit(value: &Value, text: bool, data: bool, changed: &mut bool) -> Value {
        match value {
            Value::String(s) if text || data => {
                *changed |= s.contains(IMAGE_SPECIAL_TOKEN);
                s.replace(LITERAL_MARKER, &format!("{LITERAL_MARKER}0")).replace(IMAGE_SPECIAL_TOKEN, &format!("{LITERAL_MARKER}1")).into()
            }
            Value::Array(items) => items.iter().map(|v| visit(v, false, data, changed)).collect(),
            Value::Object(object) => {
                let mut result = serde_json::Map::new();
                for (key, item) in object {
                    let key_out = if data { visit(&Value::String(key.clone()), false, true, changed).as_str().unwrap().to_owned() } else { key.clone() };
                    let arguments = if key == "arguments" && !data {
                        item.as_str().and_then(|s| serde_json::from_str::<Value>(s).ok()).filter(Value::is_object)
                    } else {
                        None
                    };
                    let item = if let Some(arguments) = arguments {
                        Value::String(visit(&arguments, false, true, changed).to_string())
                    } else {
                        visit(
                            item,
                            matches!(key.as_str(), "content" | "text" | "reasoning_content" | "description" | "arguments" | "name" | "instructions" | "input" | "output" | "system" | "thinking"),
                            data || matches!(key.as_str(), "parameters" | "input_schema")
                                || (key == "input" && object.get("type").and_then(Value::as_str) == Some("tool_use")),
                            changed,
                        )
                    };
                    result.insert(key_out, item);
                }
                Value::Object(result)
            }
            _ => value.clone(),
        }
    }
    let mut changed = false;
    let protected = visit(body, false, false, &mut changed);
    changed.then_some(protected)
}

fn restore_image_text(text: &str) -> String {
    text.replace(&format!("{LITERAL_MARKER}1"), IMAGE_SPECIAL_TOKEN).replace(&format!("{LITERAL_MARKER}0"), LITERAL_MARKER)
}

const IMAGE_SPELLINGS: [&str; 2] = [IMAGE_SPECIAL_TOKEN, "<｜deepseek_image｜>"];

pub fn tokenize_prompt(tokenizer: &Tokenizer, prompt: &str, literal_breaks: &[usize]) -> Result<Vec<i32>> {
    tokenize_segment(tokenizer, prompt, literal_breaks, 0..prompt.len())
}

pub(crate) fn tokenize_segment(tokenizer: &Tokenizer, prompt: &str, literal_breaks: &[usize], range: std::ops::Range<usize>) -> Result<Vec<i32>> {
    let start = literal_breaks.partition_point(|&offset| offset <= range.start);
    let end = start + literal_breaks[start..].partition_point(|&offset| offset < range.end);
    if start == end {
        return tokenize_raw_prompt(tokenizer, prompt.get(range).ok_or_else(|| anyhow!("invalid prompt range"))?);
    }
    let mut at = range.start;
    let mut tokens = Vec::new();
    for &end in literal_breaks[start..end].iter().chain(std::iter::once(&range.end)) {
        let text = prompt.get(at..end).ok_or_else(|| anyhow!("invalid literal text boundary"))?;
        tokens.extend(tokenize_raw_prompt(tokenizer, text)?);
        at = end;
    }
    Ok(tokens)
}

pub fn tokenize_raw_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<i32>> {
    let encode = |text: &str| -> Result<Vec<i32>> {
        Ok(crate::token_cache::encode(tokenizer, text)?.into_iter().map(|t| t as i32).collect())
    };
    let image_ids: Vec<_> = IMAGE_SPELLINGS.iter().filter_map(|s| tokenizer.token_to_id(s).map(|id| id as i32)).collect();
    let mut literals = Vec::new();
    for spelling in IMAGE_SPELLINGS {
        if !prompt.contains(spelling) || tokenizer.token_to_id(spelling).is_none() {
            continue;
        }
        let mut literal = encode(&spelling[..1])?;
        literal.extend(encode(&spelling[1..])?);
        ensure!(!literal.iter().any(|id| image_ids.contains(id)), "tokenizer cannot encode the literal image marker as text");
        literals.push((spelling, literal));
    }
    if literals.is_empty() {
        return encode(prompt);
    }
    let mut tokens = Vec::new();
    let mut rest = prompt;
    while let Some((offset, spelling, literal)) =
        literals.iter().filter_map(|(spelling, ids)| rest.find(spelling).map(|at| (at, *spelling, ids))).min_by_key(|(at, _, _)| *at)
    {
        if offset > 0 {
            tokens.extend(encode(&rest[..offset])?);
        }
        tokens.extend_from_slice(literal);
        rest = &rest[offset + spelling.len()..];
    }
    if !rest.is_empty() {
        tokens.extend(encode(rest)?);
    }
    Ok(tokens)
}

#[cfg(test)]
#[path = "chat_tests.rs"]
mod tests;
