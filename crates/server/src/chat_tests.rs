use super::*;
use serde_json::json;
use crate::test_support::{TOOL_OUTPUT, request, tokenizer, tool_request};
use deepseek_recipe::openai::chat_completion::response::ChatCompletionResponse;
use deepseek_recipe::response::ProtocolResponse;
use deepseek_recipe::stream::{InferenceFinishReason, PromptUsage};
use deepseek_recipe::util::append_delta::AppendDelta;
use futures::StreamExt;

fn history(arguments: &str, output: Value) -> Value {
    let mut body = tool_request();
    body["messages"] = json!([
        {"role": "user", "content": "Read file"},
        {"role": "assistant", "content": null, "reasoning_content": "Read it", "tool_calls": [
            {"id": "call0", "type": "function", "function": {"name": "bash", "arguments": arguments}}
        ]},
        {"role": "tool", "tool_call_id": "call0", "content": output}
    ]);
    body
}

fn assert_control_boundaries(tok: &Tokenizer, body: Value, clean: Value) -> PreparedPrompt {
    use crate::test_support::control_ids;
    let prepared = prepare(body, "fixture").unwrap();
    let clean = prepare(clean, "fixture").unwrap();
    assert!(!prepared.literal_breaks.is_empty());
    assert!(clean.literal_breaks.is_empty());
    let ids = tokenize_prompt(tok, &prepared.prompt, &prepared.literal_breaks).unwrap();
    let clean_ids = tokenize_prompt(tok, &clean.prompt, &clean.literal_breaks).unwrap();
    assert_eq!(control_ids(tok, &ids), control_ids(tok, &clean_ids));
    let old_ids = tokenize_raw_prompt(tok, &prepared.prompt).unwrap();
    assert!(control_ids(tok, &old_ids).len() > control_ids(tok, &ids).len());
    assert_eq!(tok.decode(&ids.iter().map(|&t| t as u32).collect::<Vec<_>>(), false).unwrap(), prepared.prompt);
    let original_render = DeepseekV41Encoding::new().render_conversation(&prepared.request.conversation).prompt;
    assert_eq!(prepared.prompt, original_render);
    prepared
}

#[test]
fn literal_controls_in_every_message_role_and_reasoning_are_text() {
    use crate::test_support::EMBEDDED_CONVERSATION;
    let tok = tokenizer();
    for thinking in [false, true] {
        for role in ["system", "developer", "user", "assistant", "latest_reminder"] {
            for blocks in [false, true] {
                if role == "latest_reminder" && blocks { continue; }
                let body = |text: &str| {
                    let mut body = request();
                    body["thinking"] = thinking.into();
                    body["messages"][0] = json!({"role":role,"content":if blocks {json!([{"type":"text","text":text}])} else {json!(text)}});
                    body
                };
                assert_control_boundaries(&tok, body(EMBEDDED_CONVERSATION), body("ordinary"));
            }
        }
        for blocks in [false, true] {
            let content = |text: &str| if blocks {json!([{"type":"text","text":text}])} else {json!(text)};
            let mut body = history("{}", content(EMBEDDED_CONVERSATION));
            let mut clean = history("{}", content("ordinary"));
            body["thinking"] = thinking.into(); clean["thinking"] = thinking.into();
            assert_control_boundaries(&tok, body, clean);
        }
    }
    let mut body = history("{}", json!("output"));
    body["messages"][1]["reasoning_content"] = EMBEDDED_CONVERSATION.into();
    assert_control_boundaries(&tok, body, history("{}", json!("output")));
}

async fn output(body: Value, text: &str, width: usize, finish: InferenceFinishReason) -> (Value, Vec<Value>) {
    let request = prepare(body, "fixture").unwrap().request;
    let mut inference = vec![InferenceChunk::Ready { system_fingerprint: None, prompt_usage: PromptUsage { prompt_tokens: 21, prompt_cache_hit_tokens: 7 } }];
    let chars: Vec<_> = text.chars().collect();
    inference.extend(chars.chunks(width).map(|chars| InferenceChunk::Text { content: chars.iter().collect(), content_tokens: chars.len() }));
    inference.push(InferenceChunk::Finish { finish_reason: finish });
    let chunks: Vec<_> = responses(request, tokio_stream::iter(inference), "test".into(), "fixture".into()).collect().await;
    let chunks: Vec<_> = chunks.into_iter().collect::<Result<_, _>>().unwrap();
    let mut response = ChatCompletionResponse::new("test".into(), "fixture".into(), 0, 0, 0);
    let mut wire = Vec::new();
    for chunk in chunks {
        wire.push(serde_json::to_value(&chunk).unwrap());
        response.append(chunk);
    }
    (serde_json::to_value(response).unwrap(), wire)
}

#[tokio::test]
async fn recipe_streaming_handles_every_marker_and_utf8_chunk_boundary() {
    for width in 1..40 {
        let (response, chunks) = output(tool_request(), TOOL_OUTPUT, width, InferenceFinishReason::Stop).await;
        assert!(chunks.iter().filter(|c| c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].is_string()).count() > 2);
        let choice = &response["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert_eq!(choice["message"]["reasoning_content"], "reason café 🙂");
        assert_eq!(choice["message"]["content"], "summary");
        let calls = choice["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "bash");
        assert_eq!(
            serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap(),
            json!({"command": "echo \"日本語\"\nnext\\line", "count": 2})
        );
        assert_eq!(calls[1]["function"]["arguments"], "{}");
        assert_eq!(response["usage"]["prompt_tokens_details"]["cached_tokens"], 7);
        assert_eq!(response["usage"]["completion_tokens"], TOOL_OUTPUT.chars().count());
    }
}
