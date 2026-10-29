use anyhow::{Result, anyhow};
use serde_json::{Map, Value, json};

fn flag(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Number(n) if n.is_i64() || n.is_u64() => Some(n.as_f64()? != 0.0),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Some(true),
            "false" | "0" | "no" | "off" | "" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn template(body: &Map<String, Value>) -> Option<bool> {
    ["chat_template_args", "chat_template_kwargs"].iter().find_map(|key| {
        let args = body.get(*key)?.as_object()?;
        args.get("enable_thinking").or_else(|| args.get("thinking")).and_then(flag)
    })
}

pub(crate) fn native(body: &mut Value) -> Result<()> {
    let body = body.as_object_mut().ok_or_else(|| anyhow!("request must be an object"))?;
    if let Some(Value::Bool(enabled)) = body.get("thinking") {
        body.insert("thinking".into(), json!({"type": if *enabled { "enabled" } else { "disabled" }}));
    }
    if body.get("thinking").is_none_or(Value::is_null) {
        let enabled = body.get("reasoning").and_then(|r| r.get("enabled")).and_then(flag).or_else(|| template(body));
        if let Some(enabled) = enabled {
            body.insert("thinking".into(), json!({"type": if enabled { "enabled" } else { "disabled" }}));
        }
    }
    Ok(())
}

pub(crate) fn responses(body: &mut Value) -> Result<()> {
    let object = body.as_object_mut().ok_or_else(|| anyhow!("request must be an object"))?;
    if object.get("reasoning").and_then(|r| r.get("effort")).is_some_and(|v| !v.is_null()) {
        return Ok(());
    }
    if let Some(enabled) = template(object) {
        let reasoning = object.entry("reasoning").or_insert_with(|| json!({}));
        if reasoning.is_null() {
            *reasoning = json!({});
        }
        reasoning.as_object_mut().ok_or_else(|| anyhow!("reasoning must be an object"))?.insert("effort".into(), if enabled { "high" } else { "none" }.into());
    }
    Ok(())
}
