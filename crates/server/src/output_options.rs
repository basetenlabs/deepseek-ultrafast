use anyhow::{Result, bail, ensure};
use serde_json::Value;

fn disabled(value: Option<&Value>) -> bool {
    value.is_none_or(|v| v.is_null() || v == false)
}

fn text_format(value: &Value) -> bool {
    value
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind == "text")
        && ["schema", "json_schema"]
            .iter()
            .all(|key| value.get(*key).is_none_or(Value::is_null))
}

fn strict_tool(tool: &Value) -> bool {
    tool.get("strict") == Some(&Value::Bool(true))
        || tool.pointer("/function/strict") == Some(&Value::Bool(true))
        || tool
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|tools| tools.iter().any(strict_tool))
}

pub(crate) fn validate(body: &Value) -> Result<()> {
    ensure!(body.is_object(), "request must be an object");
    for (field, value) in [("temperature", body.get("temperature")), ("top_p", body.get("top_p"))] {
        ensure!(
            value.is_none_or(|v| v.is_null() || v.as_f64() == Some(1.0)),
            "{field} must be 1"
        );
    }
    ensure!(disabled(body.get("logprobs")), "logprobs are not supported");
    for field in ["top_logprobs", "prompt_logprobs"] {
        ensure!(body.get(field).is_none_or(Value::is_null), "{field} are not supported");
    }
    if let Some(include) = body.get("include").and_then(Value::as_array) {
        ensure!(
            !include
                .iter()
                .any(|v| matches!(v.as_str(), Some("message.output_text.logprobs" | "logprobs" | "prompt_logprobs"))),
            "logprobs are not supported (include)"
        );
    }
    let object = body.as_object().unwrap();
    ensure!(
        body.get("structured_outputs").is_none_or(Value::is_null)
            && !object
                .iter()
                .any(|(key, value)| key.starts_with("guided_") && !disabled(Some(value))),
        "structured output controls are not supported"
    );
    for (value, field) in [
        (body.get("response_format"), "response_format"),
        (body.get("text").and_then(|v| v.get("format")), "text.format"),
        (body.get("output_config").and_then(|v| v.get("format")), "output_config.format"),
        (body.get("output_format"), "output_format"),
    ] {
        if let Some(value) = value.filter(|v| !v.is_null()) {
            if !text_format(value) {
                bail!("{field} other than text is not supported");
            }
        }
    }
    if body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| tools.iter().any(strict_tool))
    {
        bail!("strict tools are not supported");
    }
    Ok(())
}
