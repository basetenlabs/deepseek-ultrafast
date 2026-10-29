use crate::{chat, responses};
use crate::request_log::{Output, Trace};
use crate::generation::{self, Event, Finish, Job};
use crate::stream::{Detok, StopMatcher};
use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Extension, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use deepseek_recipe::openai::chat_completion::response::{ChatCompletionResponse, ChatCompletionFinishReason};
use deepseek_recipe::openai::responses::response::ResponsesResponse;
use deepseek_recipe::response::ProtocolResponse;
use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, PromptUsage};
use deepseek_recipe::util::append_delta::AppendDelta;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

pub struct App {
    pub tok: Arc<Tokenizer>,
    pub jobs: std::sync::mpsc::Sender<Job>,
    pub model: String,
    pub max_seq: usize,
    pub eos: i32,
}

const MAX_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;

pub fn router_with_readiness(app: Arc<App>, ready: crate::readiness::Readiness) -> Router {
    let health = move || {
        let ready = ready.clone();
        async move {
            if ready.is_ready() { (StatusCode::OK, "ok") }
            else { (StatusCode::SERVICE_UNAVAILABLE, "starting or stopping") }
        }
    };
    Router::new()
        .route("/health", get(health.clone()))
        .route("/readyz", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(create_response))
        .route("/v1/messages", post(messages_api::create_message))
        .route("/v1/completions", post(completions))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .layer(axum::middleware::from_fn(crate::request_log::middleware))
        .with_state(app)
}

#[derive(Debug)]
pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = if self.0 == StatusCode::TOO_MANY_REQUESTS { "rate_limit_error" }
            else if self.0.is_server_error() { "server_error" } else { "invalid_request_error" };
        (self.0, axum::Json(json!({ "error": { "message": self.1, "type": kind, "param": null, "code": null } }))).into_response()
    }
}

fn bad(msg: impl std::fmt::Display) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.to_string())
}
fn request_body(body: &[u8]) -> Result<Value, ApiError> {
    let value: Value = serde_json::from_slice(body).map_err(|e| bad(format!("invalid request: {e}")))?;
    crate::output_options::validate(&value).map_err(bad)?;
    Ok(value)
}

fn image_error(error: anyhow::Error) -> ApiError {
    if error.is::<crate::images::Busy>() { ApiError(StatusCode::TOO_MANY_REQUESTS, error.to_string()) }
    else { bad(error) }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Ingress {
    pub session: Option<String>,
}
pub(crate) fn ingress(headers: &HeaderMap) -> Result<Ingress, ApiError> {
    Ok(Ingress { session: session_id(headers)? })
}

const SESSION_HEADERS: [&str; 4] = ["x-session-id", "session-id", "x-claude-code-session-id", "x-deepseek-harness-session-id"];

pub(crate) fn session_id(headers: &HeaderMap) -> Result<Option<String>, ApiError> {
    let values: Vec<_> = SESSION_HEADERS.iter().flat_map(|h| headers.get_all(*h).iter()).collect();
    if values.is_empty() { return Ok(None); }
    if values.len() != 1 { return Err(bad("at most one session ID header (X-Session-ID, Session-ID, X-Claude-Code-Session-ID or X-DeepSeek-Harness-Session-ID) is allowed")); }
    let id = values[0].to_str().map_err(|_| bad("invalid session ID header"))?;
    if id.trim().is_empty() || id.len() > 1024 { return Err(bad("session ID must contain 1..1024 nonblank ASCII bytes")); }
    Ok(Some(id.to_string()))
}

const MULTI_TURN_SESSION_ERROR: &str = "multi-turn requests require X-Session-ID";

fn conversation_items(value: &Value) -> bool {
    value.as_array().is_some_and(|items| {
        items.iter().filter(|item| {
            !matches!(item.get("role").and_then(Value::as_str), Some("system" | "developer"))
        }).count() > 1
    })
}

fn require_session_for_multi_turn(ingress: &Ingress, multi_turn: bool) -> Result<(), ApiError> {
    if multi_turn && ingress.session.is_none() { return Err(bad(MULTI_TURN_SESSION_ERROR)); }
    Ok(())
}

fn session_identity(explicit: Option<String>) -> (String, &'static str) {
    match explicit {
        Some(id) => (id, "explicit"),
        None => (rand_id("session-"), "generated"),
    }
}

fn session_headers(g: &Gen) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("x-session-id", g.session.parse().expect("validated ASCII session ID"));
    headers.insert("x-session-id-source", g.session_source.parse().unwrap());
    headers
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn rand_id(prefix: &str) -> String {
    format!("{prefix}{:024x}", rand::random::<u128>() >> 32)
}

const MODEL_ALIASES: [&str; 2] = [
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "deepseek-ai/DeepSeek-V4.1-Flash-Fast",
];

async fn models(State(app): State<Arc<App>>) -> axum::Json<Value> {
    let created = now();
    let names = std::iter::once(app.model.as_str())
        .chain(MODEL_ALIASES.into_iter().filter(|name| *name != app.model));
    let data: Vec<_> = names.map(|id| json!({
        "id": id, "object": "model", "created": created, "owned_by": "deepseek-ultrafast", "max_model_len": app.max_seq,
    })).collect();
    axum::Json(json!({"object": "list", "data": data}))
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize, Default)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize, Default)]
struct Sampling {
    seed: Option<i64>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    stop: Option<Stop>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    n: Option<u32>,
    logprobs: Option<Value>,
}

struct Gen {
    trace: Trace,
    output: Output,
    session: String,
    session_source: &'static str,
    rx: mpsc::UnboundedReceiver<Event>,
    cancel: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    detok: Detok,
    stop: StopMatcher,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    finish: Option<&'static str>,
    done: bool,
}

impl Gen {
    async fn next(&mut self) -> Option<String> {
        while !self.done {
            let ev = self.rx.recv().await;
            match ev {
                Some(Event::Started { prompt_tokens, cached_tokens }) => {
                    self.prompt_tokens = prompt_tokens;
                    self.cached_tokens = cached_tokens;
                    self.trace.started(prompt_tokens, cached_tokens);
                }
                Some(Event::Tokens(ids)) => {
                    self.output.batch(ids.len(), self.trace.elapsed());
                    self.completion_tokens += ids.len();
                    let t = self.detok.push(&ids);
                    let s = self.stop.push(&t);
                    if self.stop.hit {
                        self.cancel.store(true, Ordering::Relaxed);
                        self.finish = Some("stop");
                        self.done = true;
                    }
                    if !s.is_empty() {
                        return Some(s);
                    }
                }
                Some(Event::Done { finish, completion_tokens }) => {
                    self.completion_tokens = completion_tokens;
                    self.done = true;
                    self.finish = Some(if finish == Finish::Length { "length" } else { "stop" });
                    let t = self.detok.finish();
                    let mut s = self.stop.push(&t);
                    if !self.stop.hit {
                        s += &self.stop.finish();
                    }
                    if !s.is_empty() {
                        return Some(s);
                    }
                }
                None => {
                    self.done = true;
                    self.trace.reason("error");
                    self.finish.get_or_insert("stop");
                }
            }
        }
        None
    }

    fn into_inference(self) -> impl futures::Stream<Item = InferenceChunk> + Send {
        futures::stream::unfold((self, None), |(mut g, pending_finish)| async move {
            if let Some(finish_reason) = pending_finish {
                g.done = true;
                return Some((InferenceChunk::Finish { finish_reason }, (g, None)));
            }
            if g.done {
                return None;
            }
            let Some(event) = g.rx.recv().await else {
                g.failed.store(true, Ordering::Relaxed);
                g.trace.reason("error");
                return None;
            };
            let chunk = match event {
                Event::Started { prompt_tokens, cached_tokens } => {
                    g.trace.started(prompt_tokens, cached_tokens);
                    InferenceChunk::Ready { system_fingerprint: None, prompt_usage: PromptUsage { prompt_tokens, prompt_cache_hit_tokens: cached_tokens } }
                }
                Event::Tokens(ids) => {
                    g.output.batch(ids.len(), g.trace.elapsed());
                    g.completion_tokens += ids.len();
                    InferenceChunk::Text { content: g.detok.push(&ids), content_tokens: ids.len() }
                }
                Event::Done { finish, completion_tokens } => {
                    if finish == Finish::Cancelled { g.failed.store(true, Ordering::Relaxed); }
                    let finish_reason = if finish == Finish::Length { InferenceFinishReason::Length } else { InferenceFinishReason::Stop };
                    let chunk = InferenceChunk::Text { content: g.detok.finish(), content_tokens: completion_tokens.saturating_sub(g.completion_tokens) };
                    return Some((chunk, (g, Some(finish_reason))));
                }
            };
            Some((chunk, (g, None)))
        })
    }

    fn usage(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "prompt_tokens_details": { "cached_tokens": self.cached_tokens },
        })
    }
}

impl Drop for Gen {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.trace.output(std::mem::take(&mut self.output));
    }
}

fn log_chat_finish(trace: &Trace, reason: Option<&ChatCompletionFinishReason>) {
    if let Some(reason) = reason {
        trace.reason(match reason {
            ChatCompletionFinishReason::Stop => "stop",
            ChatCompletionFinishReason::Length => "length",
            ChatCompletionFinishReason::ToolCalls => "tool_calls",
            ChatCompletionFinishReason::ContentFilter => "content_filter",
            ChatCompletionFinishReason::InsufficientSystemResource => "insufficient_system_resource",
            ChatCompletionFinishReason::Aborted => "cancelled",
        });
    }
}

struct RequestContext { trace: Trace, ingress: Ingress }

fn output_budget(requested: Option<usize>, default: Option<usize>, limit: usize) -> Result<usize, ApiError> {
    match requested {
        Some(0) => Err(bad("max_tokens must be at least 1")),
        Some(m) => Ok(m.min(limit)),
        None => Ok(default.unwrap_or(limit).min(limit)),
    }
}

fn submit(app: &Arc<App>, context: RequestContext, tokens: Vec<i32>, s: &Sampling, default_max: Option<usize>, images: Option<Arc<crate::images::Images>>) -> Result<Gen, ApiError> {
    let RequestContext { trace, ingress: Ingress { session } } = context;
    if images.is_none() && tokens.contains(&deepseek_ultrafast_model::vision::IMAGE_TOKEN) {
        return Err(bad("image token IDs require image content on Chat Completions, Responses or Messages"));
    }
    if tokens.is_empty() {
        return Err(bad("the prompt is empty"));
    }
    if s.n.is_some_and(|n| n != 1) {
        return Err(bad("only n = 1 is supported"));
    }
    if s.logprobs.as_ref().is_some_and(|l| !matches!(l, Value::Null | Value::Bool(false))) {
        return Err(bad("logprobs are not supported"));
    }
    let limit = generation::max_new_limit(tokens.len(), app.max_seq);
    if limit == 0 {
        return Err(bad(format!("the prompt ({} tokens) does not fit the context ({} tokens)", tokens.len(), app.max_seq)));
    }
    let max_new = output_budget(s.max_completion_tokens.or(s.max_tokens), default_max, limit)?;
    let stops = match &s.stop {
        None => vec![],
        Some(Stop::One(x)) => vec![x.clone()],
        Some(Stop::Many(v)) => v.clone(),
    };
    let seed = match s.seed {
        Some(x) => x as u64,
        None => rand::random(),
    };
    let (session, session_source) = session_identity(session);
    trace.session(&session, session_source);
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let job = Job { trace: trace.clone(), session: session.clone(), images, tokens: Arc::new(tokens), max_new, seed, stop_ids: Arc::new(vec![app.eos]), cancel: cancel.clone(), events: tx };
    app.jobs.send(job).map_err(|_| ApiError(StatusCode::SERVICE_UNAVAILABLE, "engine stopped".into()))?;
    Ok(Gen {
        trace, output: Output::default(),
        session, session_source,
        rx,
        cancel,
        failed: Arc::new(AtomicBool::new(false)),
        detok: Detok::new(app.tok.clone()),
        stop: StopMatcher::new(stops),
        prompt_tokens: 0,
        cached_tokens: 0,
        completion_tokens: 0,
        finish: None,
        done: false,
    })
}

async fn tokenize(app: &Arc<App>, text: String) -> Result<Vec<i32>, ApiError> {
    let tok = app.tok.clone();
    let enc = tokio::task::spawn_blocking(move || chat::tokenize_raw_prompt(&tok, &text))
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    enc.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("tokenizer: {e}")))
}

fn sse_stream(rx: mpsc::Receiver<Value>) -> Response {
    use futures::StreamExt;
    let s = tokio_stream::wrappers::ReceiverStream::new(rx).map(|v| {
        let data = if v.is_null() { "[DONE]".to_string() } else { v.to_string() };
        Ok::<_, Infallible>(SseEvent::default().data(data))
    });
    Sse::new(s).keep_alive(KeepAlive::default()).into_response()
}

async fn chat_completions(State(app): State<Arc<App>>, Extension(trace): Extension<Trace>, headers: HeaderMap, body: Bytes) -> Result<Response, ApiError> {
    let ingress = ingress(&headers)?;
    let value = request_body(&body)?;
    let sampling: Sampling = serde_json::from_value(value.clone()).map_err(bad)?;
    require_session_for_multi_turn(&ingress, conversation_items(&value["messages"]))?;
    let model = app.model.clone();
    let prepared = tokio::task::spawn_blocking(move || chat::prepare(value, &model))
        .await.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?.map_err(bad)?;
    let (tokens, images) = crate::images::prepare(app.tok.clone(), prepared.prompt, prepared.literal_breaks, prepared.image_offsets, prepared.image_sources, app.max_seq).await.map_err(image_error)?;
    let request = prepared.request;
    trace.parsed(tokens.len());
    let g = submit(&app, RequestContext { trace: trace.clone(), ingress }, tokens, &sampling, None, images)?;
    let headers = session_headers(&g);
    let (id, created, model) = (rand_id("chatcmpl-"), now(), app.model.clone());
    let chunks = chat::responses(request, g.into_inference(), id.clone(), model.clone());

    if sampling.stream {
        let include_usage = sampling.stream_options.as_ref().is_some_and(|o| o.include_usage);
        let (tx, rx) = mpsc::channel::<Value>(64);
        tokio::spawn(async move {
            let mut chunks = std::pin::pin!(chunks);
            loop {
                let next = tokio::select! {
                    _ = tx.closed() => return,
                    next = chunks.next() => next,
                };
                let Some(chunk) = next else { break };
                let mut chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        trace.reason("error");
                        let _ = tx.send(json!({"error": {"message": error.to_string(), "type": "server_error"}})).await;
                        return;
                    }
                };
                let usage = chunk.usage.take().flatten();
                log_chat_finish(&trace, chunk.choices.first().and_then(|c| c.finish_reason.as_ref()));
                if tx.send(serde_json::to_value(chunk).expect("recipe chunk is JSON")).await.is_err() {
                    return;
                }
                if include_usage
                    && let Some(usage) = usage
                    && tx
                        .send(json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": usage}))
                        .await
                        .is_err()
                {
                    return;
                }
            }
            let _ = tx.send(Value::Null).await;
        });
        return Ok((headers, sse_stream(rx)).into_response());
    }

    let mut response = ChatCompletionResponse::new(id, model, created, 0, 0);
    let mut chunks = std::pin::pin!(chunks);
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        log_chat_finish(&trace, chunk.choices.first().and_then(|c| c.finish_reason.as_ref()));
        response.append(chunk);
    }
    Ok((headers, axum::Json(response)).into_response())
}

#[path = "messages_api.rs"]
mod messages_api;

async fn create_response(State(app): State<Arc<App>>, Extension(trace): Extension<Trace>, headers: HeaderMap, body: Bytes) -> Result<Response, ApiError> {
    let ingress = ingress(&headers)?;
    let value = request_body(&body)?;
    require_session_for_multi_turn(&ingress, conversation_items(&value["input"]))?;
    let model = app.model.clone();
    let prepared = tokio::task::spawn_blocking(move || responses::prepare(value, &model))
        .await.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?.map_err(bad)?;
    let request = prepared.request;
    let response_options = prepared.response_options;
    let sampling = Sampling {
        max_tokens: request.inference_options.max_tokens.map(|n| n as usize),
        stream: request.stream,
        seed: prepared.seed,
        ..Default::default()
    };
    let (tokens, images) = crate::images::prepare(app.tok.clone(), prepared.prompt, prepared.literal_breaks, prepared.image_offsets, prepared.image_sources, app.max_seq).await.map_err(image_error)?;
    trace.parsed(tokens.len());
    let g = submit(&app, RequestContext { trace: trace.clone(), ingress }, tokens, &sampling, None, images).map_err(|mut e| {
        e.1 = e.1.replace("max_tokens", "max_output_tokens");
        e
    })?;
    let headers = session_headers(&g);
    let failed = g.failed.clone();
    let (id, model) = (rand_id("resp_"), app.model.clone());
    let chunks = responses::events(request, prepared.custom_tools, g.into_inference(), id.clone(), model.clone());
    if sampling.stream {
        let (tx, rx) = mpsc::channel::<Value>(64);
        tokio::spawn(async move {
            let mut chunks = std::pin::pin!(chunks);
            let mut sequence_number = 0;
            let mut wire = responses::WireEvents::new(response_options);
            loop {
                let next = tokio::select! {
                    _ = tx.closed() => return,
                    next = chunks.next() => next,
                };
                let Some(chunk) = next else { break };
                let chunk = if failed.load(Ordering::Relaxed) {
                    Err("engine stopped before completing the response".to_string())
                } else { chunk.map_err(|e| e.to_string()) };
                match chunk {
                    Ok(chunk) => {
                        let event = wire.event(chunk);
                        if let Some(response) = event.get("response") { trace.responses_reason(response); }
                        if tx.send(event).await.is_err() { return; }
                        sequence_number += 1;
                    }
                    Err(message) => {
                        trace.reason("error");
                        let _ = tx.send(json!({"type": "error", "code": "server_error", "message": message,
                            "param": null, "sequence_number": sequence_number})).await;
                        return;
                    }
                }
            }
        });
        let events = tokio_stream::wrappers::ReceiverStream::new(rx).map(|value| {
            Ok::<_, Infallible>(SseEvent::default().event(value["type"].as_str().expect("Responses event type")).data(value.to_string()))
        });
        return Ok((headers, Sse::new(events).keep_alive(KeepAlive::default())).into_response());
    }
    let mut response = ResponsesResponse::new(id, model, now(), 0, 0);
    let mut chunks = std::pin::pin!(chunks);
    while let Some(chunk) = chunks.next().await {
        if failed.load(Ordering::Relaxed) {
            return Err(ApiError(StatusCode::INTERNAL_SERVER_ERROR, "engine stopped before completing the response".into()));
        }
        response.append(chunk.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?);
    }
    let response = responses::wire_response(response, &response_options);
    trace.responses_reason(&response);
    Ok((headers, axum::Json(response)).into_response())
}

#[derive(Deserialize)]
struct CompletionReq {
    prompt: Value,
    #[serde(flatten)]
    s: Sampling,
}

async fn completions(State(app): State<Arc<App>>, Extension(trace): Extension<Trace>, headers: HeaderMap, body: Bytes) -> Result<Response, ApiError> {
    let ingress = ingress(&headers)?;
    let value = request_body(&body)?;
    let req: CompletionReq = serde_json::from_value(value).map_err(|e| bad(format!("invalid request: {e}")))?;
    let ids = |a: &[Value]| -> Option<Vec<i32>> { a.iter().map(|x| x.as_i64().and_then(|t| i32::try_from(t).ok())).collect() };
    let tokens = match &req.prompt {
        Value::String(s) => tokenize(&app, s.clone()).await?,
        Value::Array(a) if a.len() == 1 && a[0].is_string() => tokenize(&app, a[0].as_str().unwrap().to_string()).await?,
        Value::Array(a) if a.len() == 1 && a[0].is_array() => ids(a[0].as_array().unwrap()).ok_or_else(|| bad("prompt: expected token ids"))?,
        Value::Array(a) => ids(a).ok_or_else(|| bad("prompt: a string, a list of token ids, or a one-element list of either (batches are not supported)"))?,
        _ => return Err(bad("prompt must be a string or token ids")),
    };
    let vocab = app.tok.get_vocab_size(true) as i32;
    if tokens.iter().any(|&t| t < 0 || t >= vocab) {
        return Err(bad("prompt: token id out of range"));
    }
    trace.parsed(tokens.len());
    let mut g = submit(&app, RequestContext { trace: trace.clone(), ingress }, tokens, &req.s, Some(16), None)?;
    let headers = session_headers(&g);
    let (id, created, model) = (rand_id("cmpl-"), now(), app.model.clone());
    if req.s.stream {
        let include_usage = req.s.stream_options.as_ref().is_some_and(|o| o.include_usage);
        let (tx, rx) = mpsc::channel::<Value>(64);
        tokio::spawn(async move {
            let chunk = |text: &str, finish: Option<&str>| json!({ "id": id, "object": "text_completion", "created": created, "model": model, "choices": [{ "index": 0, "text": text, "logprobs": null, "finish_reason": finish }] });
            loop {
                let text = tokio::select! { _ = tx.closed() => return, text = g.next() => text };
                let Some(text) = text else { break };
                if tx.send(chunk(&text, None)).await.is_err() { return; }
            }
            if let Some(reason) = g.finish { trace.reason(reason); }
            let _ = tx.send(chunk("", Some(g.finish.unwrap_or("stop")))).await;
            if include_usage {
                let _ = tx.send(json!({ "id": id, "object": "text_completion", "created": created, "model": model, "choices": [], "usage": g.usage() })).await;
            }
            let _ = tx.send(Value::Null).await;
        });
        return Ok((headers, sse_stream(rx)).into_response());
    }
    let mut text = String::new();
    while let Some(t) = g.next().await {
        text += &t;
    }
    if let Some(reason) = g.finish { trace.reason(reason); }
    Ok((headers, axum::Json(json!({
        "id": id, "object": "text_completion", "created": created, "model": model,
        "choices": [{ "index": 0, "text": text, "logprobs": null, "finish_reason": g.finish.unwrap_or("stop") }],
        "usage": g.usage(),
    })))
    .into_response())
}

#[cfg(test)]
mod output_budget_tests {
    use super::output_budget;

    #[test]
    fn capped_at_the_context() {
        assert_eq!(output_budget(Some(256_000), None, 4_000).unwrap(), 4_000);
        assert_eq!(output_budget(Some(256_000), None, 256_000).unwrap(), 256_000);
        assert_eq!(output_budget(Some(100), None, 4_000).unwrap(), 100);
        assert_eq!(output_budget(None, Some(16), 4_000).unwrap(), 16);
        assert_eq!(output_budget(None, None, 4_000).unwrap(), 4_000);
        assert!(output_budget(Some(0), None, 4_000).is_err());
    }
}

#[cfg(test)]
mod session_id_tests {
    use super::session_id;
    use axum::http::{HeaderMap, HeaderName};

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.append(HeaderName::from_bytes(k.as_bytes()).unwrap(), v.parse().unwrap());
        }
        h
    }

    #[test]
    fn accepted_headers() {
        for name in ["X-Session-ID", "session-id", "X-Claude-Code-Session-Id", "X-DeepSeek-Harness-Session-ID", "x-deepseek-harness-session-id"] {
            let h = headers(&[(name, "s1")]);
            assert_eq!(session_id(&h).unwrap().as_deref(), Some("s1"), "{name}");
        }
        assert_eq!(session_id(&HeaderMap::new()).unwrap(), None);
    }

    #[test]
    fn ambiguous_or_blank() {
        assert!(session_id(&headers(&[("x-session-id", "a"), ("x-deepseek-harness-session-id", "b")])).is_err());
        assert!(session_id(&headers(&[("x-claude-code-session-id", "a"), ("session-id", "a")])).is_err());
        assert!(session_id(&headers(&[("x-deepseek-harness-session-id", "a"), ("x-deepseek-harness-session-id", "a")])).is_err());
        assert!(session_id(&headers(&[("x-deepseek-harness-session-id", " ")])).is_err());
    }
}
