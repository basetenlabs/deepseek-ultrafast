use anyhow::{Result, anyhow};
use deepseek_recipe_core::{conversation::Conversation, messages::InputMessage};
use deepseek_recipe_encoding::v4;
use serde_json::Value;

const TOKENS: &[&str] = &[
    v4::BOS_TOKEN,
    v4::EOS_TOKEN,
    v4::SYSTEM_SP_TOKEN,
    v4::USER_SP_TOKEN,
    v4::ASSISTANT_SP_TOKEN,
    v4::LATEST_REMINDER_SP_TOKEN,
    v4::THINKING_START_TOKEN,
    v4::THINKING_END_TOKEN,
    v4::DSML_SP_TOKEN,
];
const MARKER: &str = "<_deepseek_ultrafast_literal_>";

fn contains(text: &str) -> bool {
    TOKENS.iter().any(|token| text.contains(token))
}

fn json_contains(value: &Value) -> bool {
    match value {
        Value::String(s) => contains(s),
        Value::Array(items) => items.iter().any(json_contains),
        Value::Object(items) => items
            .iter()
            .any(|(key, value)| contains(key) || json_contains(value)),
        _ => false,
    }
}

fn arguments_contain(text: &str) -> bool {
    contains(text)
        || (text.contains('\\')
            && serde_json::from_str::<Value>(text)
                .ok()
                .filter(Value::is_object)
                .as_ref()
                .is_some_and(json_contains))
}

fn needs_protection(conversation: &Conversation) -> bool {
    conversation.messages.iter().any(|message| match message {
        InputMessage::Assistant {
            content,
            reasoning_content,
            tool_calls,
        } => {
            contains(content)
                || reasoning_content.as_deref().is_some_and(contains)
                || tool_calls.as_ref().is_some_and(|calls| {
                    calls
                        .iter()
                        .any(|call| contains(&call.name) || arguments_contain(&call.arguments))
                })
        }
        InputMessage::System { content }
        | InputMessage::User { content, .. }
        | InputMessage::Tool { content, .. }
        | InputMessage::LatestReminder { content } => contains(content),
    }) || conversation.tools.iter().any(|tool| {
        contains(&tool.name)
            || tool.description.as_deref().is_some_and(contains)
            || json_contains(&tool.parameters)
    })
}

fn protect_string(text: &mut String) {
    if text.contains(MARKER) {
        *text = text.replace(MARKER, &format!("{MARKER}0"));
    }
    for (index, token) in TOKENS.iter().enumerate() {
        if text.contains(token) {
            *text = text.replace(
                token,
                &format!("{MARKER}{}", char::from(b'a' + index as u8)),
            );
        }
    }
}

fn protect_json(value: &mut Value) {
    match value {
        Value::String(s) => protect_string(s),
        Value::Array(items) => items.iter_mut().for_each(protect_json),
        Value::Object(items) => {
            *items = std::mem::take(items)
                .into_iter()
                .map(|(mut key, mut value)| {
                    protect_string(&mut key);
                    protect_json(&mut value);
                    (key, value)
                })
                .collect();
        }
        _ => {}
    }
}

fn protect_arguments(text: &mut String) {
    if let Ok(mut value @ Value::Object(_)) = serde_json::from_str::<Value>(text) {
        protect_json(&mut value);
        *text = value.to_string();
    } else {
        protect_string(text);
    }
}

pub(crate) fn protect(conversation: &Conversation) -> Option<Conversation> {
    if !needs_protection(conversation) {
        return None;
    }
    let mut protected = conversation.clone();
    for message in &mut protected.messages {
        match message {
            InputMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => {
                protect_string(content);
                if let Some(reasoning) = reasoning_content {
                    protect_string(reasoning);
                }
                if let Some(calls) = tool_calls {
                    for call in calls {
                        protect_string(&mut call.name);
                        protect_arguments(&mut call.arguments);
                    }
                }
            }
            InputMessage::System { content }
            | InputMessage::User { content, .. }
            | InputMessage::Tool { content, .. }
            | InputMessage::LatestReminder { content } => protect_string(content),
        }
    }
    for tool in &mut protected.tools {
        protect_string(&mut tool.name);
        if let Some(description) = &mut tool.description {
            protect_string(description);
        }
        protect_json(&mut tool.parameters);
    }
    Some(protected)
}

pub(crate) fn append(text: &str, prompt: &mut String, breaks: &mut Vec<usize>) -> Result<()> {
    let mut rest = text;
    while let Some(at) = rest.find(MARKER) {
        prompt.push_str(&rest[..at]);
        rest = &rest[at + MARKER.len()..];
        let code = *rest
            .as_bytes()
            .first()
            .ok_or_else(|| anyhow!("incomplete literal text marker"))?;
        if code == b'0' {
            prompt.push_str(MARKER);
        } else {
            let token = code
                .checked_sub(b'a')
                .and_then(|index| TOKENS.get(index as usize))
                .ok_or_else(|| anyhow!("invalid literal text marker"))?;
            breaks.push(prompt.len() + token.chars().next().unwrap().len_utf8());
            prompt.push_str(token);
        }
        rest = &rest[1..];
    }
    prompt.push_str(rest);
    Ok(())
}
