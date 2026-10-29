use axum::{
    body::{Body, Bytes},
    extract::Request,
    middleware::Next,
    response::Response,
};
use deepseek_ultrafast_logging::{RequestEnd, Ticket, ms};
use http_body::{Body as HttpBody, Frame, SizeHint};
use std::{
    pin::Pin,
    sync::{Arc, Mutex, OnceLock},
    task::{Context, Poll},
    time::{Duration, Instant},
};

#[derive(Clone)]
pub struct Trace(Arc<Inner>);
struct Inner {
    start: Instant,
    endpoint: &'static str,
    session: OnceLock<(Arc<str>, &'static str)>,
    state: Mutex<State>,
}
#[derive(Default)]
struct State {
    ticket: Option<Ticket>,
    input: Option<usize>,
    cached: Option<usize>,
    output: Output,
    parse_end: Option<Duration>,
    fetch_end: Option<Duration>,
    prefill_end: Option<Duration>,
    admission: Option<Duration>,
    decode: Option<Decode>,
    engine_reason: Option<&'static str>,
    http_reason: Option<&'static str>,
    http_status: Option<u16>,
    http_end: Option<Duration>,
    request_mode: Option<&'static str>,
    streaming: Option<bool>,
    terminal_error: Option<deepseek_ultrafast_logging::TerminalError>,
}
#[derive(Default, Clone)]
pub struct Output {
    pub tokens: usize,
    pub first: Option<Duration>,
    pub last: Option<Duration>,
}
impl Output {
    pub fn batch(&mut self, count: usize, elapsed: Duration) {
        if count > 0 {
            self.first.get_or_insert(elapsed);
            self.last = Some(elapsed);
            self.tokens += count;
        }
    }
}
#[derive(Clone, Copy)]
pub struct Decode {
    pub elapsed: Duration,
    pub iterations: usize,
    pub accepted_drafts: usize,
}
impl Default for Trace {
    fn default() -> Self {
        Self::new(Instant::now(), "fixture", Ticket::default())
    }
}
impl Trace {
    fn new(
        start: Instant,
        endpoint: &'static str,
        ticket: Ticket,
    ) -> Self {
        Self(Arc::new(Inner {
            start,
            endpoint,
            session: OnceLock::new(),
            state: Mutex::new(State {
                ticket: Some(ticket),
                ..State::default()
            }),
        }))
    }
    pub fn elapsed(&self) -> Duration {
        self.0.start.elapsed()
    }
    pub fn session(&self, id: &str, source: &'static str) {
        self.0.session.get_or_init(|| {
            (
                Arc::from(id),
                if source == "explicit" {
                    "user_provided"
                } else {
                    "generated"
                },
            )
        });
    }
    pub fn session_label(&self) -> Option<Arc<str>> {
        self.0.session.get().map(|(id, _)| id.clone())
    }
    pub fn parsed(&self, tokens: usize) {
        let mut s = self.0.state.lock().unwrap();
        s.input = Some(tokens);
        s.parse_end = Some(self.elapsed());
    }
    pub fn fetched(&self) {
        self.0.state.lock().unwrap().fetch_end = Some(self.elapsed());
    }
    pub fn admitted(&self) {
        self.0.state.lock().unwrap().admission = Some(self.elapsed());
    }
    pub fn prefilled(&self, cached: usize) {
        let mut s = self.0.state.lock().unwrap();
        s.cached = Some(cached);
        s.prefill_end = Some(self.elapsed());
    }
    pub fn started(&self, input: usize, cached: usize) {
        let mut s = self.0.state.lock().unwrap();
        s.input = Some(input);
        s.cached = Some(cached);
    }
    pub fn decoded(&self, stats: Decode) {
        self.0.state.lock().unwrap().decode = Some(stats);
    }
    pub fn engine_finished(&self, reason: &'static str) {
        self.0.state.lock().unwrap().engine_reason = Some(reason);
    }
    pub fn output(&self, output: Output) {
        self.0.state.lock().unwrap().output = output;
    }
    pub fn reason(&self, reason: &'static str) {
        let mut state = self.0.state.lock().unwrap();
        if !matches!(state.http_reason, Some("error" | "cancelled")) {
            state.http_reason = Some(reason);
        }
    }
    pub fn mode(&self, mode: &'static str, streaming: bool) {
        let mut state = self.0.state.lock().unwrap();
        state.request_mode = Some(mode);
        state.streaming = Some(streaming);
    }
    pub fn error(&self, category: &'static str) {
        let mut state = self.0.state.lock().unwrap();
        if state.http_reason != Some("cancelled") { state.http_reason = Some("error"); }
        state.terminal_error.get_or_insert(deepseek_ultrafast_logging::TerminalError { category, parser_category: None, line: None, column: None });
    }
    pub fn status(&self, status: u16) {
        self.0.state.lock().unwrap().http_status = Some(status);
    }
    fn http_end(&self, completed: bool) {
        let mut s = self.0.state.lock().unwrap();
        if s.http_end.is_none() {
            s.http_end = Some(self.elapsed());
            if std::thread::panicking() {
                s.http_reason = Some("error");
            } else if !completed && s.http_reason != Some("error") {
                s.http_reason = Some("cancelled");
            } else if s.http_status.is_some_and(|code| code >= 400) {
                s.http_reason = Some("error");
            }
        }
    }
    pub fn responses_reason(&self, value: &serde_json::Value) {
        match value.get("status").and_then(|v| v.as_str()) {
            Some("incomplete") => self.reason("length"),
            Some("completed") => {
                let tools = value
                    .get("output")
                    .and_then(|v| v.as_array())
                    .is_some_and(|items| {
                        items.iter().any(|item| {
                            matches!(
                                item.get("type").and_then(|v| v.as_str()),
                                Some("function_call" | "custom_tool_call")
                            )
                        })
                    });
                self.reason(if tools { "tool_calls" } else { "stop" });
            }
            Some("failed") => self.reason("error"),
            _ => {}
        }
    }
}
fn ratio(value: f64, count: usize) -> Option<f64> {
    (count > 0).then(|| value / count as f64)
}
impl Inner {
    fn report(&self, state: &State) -> RequestEnd {
        let n = state.output.tokens.saturating_sub(1);
        let decode = state.decode;
        let decode_ms = decode.map(|d| ms(d.elapsed));
        let stop_reason = state.http_reason.or(state.engine_reason).unwrap_or(
            if state.http_status.is_some_and(|s| s >= 400) { "error" } else { "none" },
        );
        RequestEnd {
            endpoint: self.endpoint,
            session_id: self.session.get().map(|(id, _)| id.clone()),
            session_origin: self.session.get().map(|(_, origin)| *origin),
            input_tokens: state.input,
            output_tokens: state.output.tokens,
            cached_tokens: state.cached,
            stop_reason,
            request_mode: state.request_mode.unwrap_or("ordinary"),
            streaming: state.streaming,
            terminal_error: state.terminal_error.or_else(|| (stop_reason == "error").then_some(deepseek_ultrafast_logging::TerminalError {
                category: "unclassified", parser_category: None, line: None, column: None,
            })),
            tpot_ms: decode_ms.and_then(|d| ratio(d, n)),
            itl_ms: state
                .output
                .first
                .zip(state.output.last)
                .and_then(|(first, last)| ratio(ms(last.saturating_sub(first)), n)),
            average_accept_length: decode
                .and_then(|d| ratio((d.accepted_drafts + d.iterations) as f64, d.iterations)),
            parse_tokenize_ms: state.parse_end.map(ms),
            kv_fetch_end_ms: state.fetch_end.map(ms),
            prefill_end_ms: state.prefill_end.map(ms),
            e2e_ms: ms(state.http_end.unwrap_or_else(|| self.start.elapsed())),
            ttft_ms: state.output.first.map(ms),
            average_iteration_ms: decode.and_then(|d| ratio(ms(d.elapsed), d.iterations)),
            decode_tps: decode
                .filter(|d| n > 0 && !d.elapsed.is_zero())
                .map(|d| n as f64 / d.elapsed.as_secs_f64()),
            queue_ms: state.admission.map(ms),
            http_status_code: state.http_status,
        }
    }
}
impl Drop for Inner {
    fn drop(&mut self) {
        let mut state = std::mem::take(self.state.get_mut().unwrap_or_else(|e| e.into_inner()));
        let ticket = state.ticket.take();
        let report = self.report(&state);
        if let Some(ticket) = ticket {
            ticket.finish(report);
        }
    }
}

struct Pending(Option<Trace>);
impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(trace) = &self.0 {
            trace.http_end(false);
        }
    }
}
struct LoggedBody {
    inner: Body,
    trace: Trace,
    completed: bool,
}
impl HttpBody for LoggedBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let result = Pin::new(&mut self.inner).poll_frame(cx);
        match &result {
            Poll::Ready(Some(Err(_))) => {
                self.trace.reason("error");
                self.trace.http_end(true);
                self.completed = true;
            }
            Poll::Ready(None) => {
                self.trace.http_end(true);
                self.completed = true;
            }
            Poll::Ready(Some(Ok(_))) if self.inner.is_end_stream() => {
                self.trace.http_end(true);
                self.completed = true;
            }
            _ => {}
        }
        result
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}
impl Drop for LoggedBody {
    fn drop(&mut self) {
        self.trace
            .http_end(self.completed || self.inner.is_end_stream());
    }
}

pub async fn middleware(mut request: Request, next: Next) -> Response {
    let endpoint = match request.uri().path() {
        "/health" | "/readyz" => return next.run(request).await,
        "/v1/chat/completions" => "/v1/chat/completions",
        "/v1/responses" => "/v1/responses",
        "/v1/messages" => "/v1/messages",
        "/v1/completions" => "/v1/completions",
        "/v1/models" => "/v1/models",
        _ => "unmatched",
    };
    let start = Instant::now();
    let trace = Trace::new(start, endpoint, Ticket::default());
    let mut guard = Pending(Some(trace.clone()));
    if let Ok(Some(id)) = crate::api::session_id(request.headers()) {
        trace.session(&id, "explicit");
    }
    let ticket = Ticket::reserve().await;
    trace.0.state.lock().unwrap().ticket = Some(ticket);
    request.extensions_mut().insert(trace.clone());
    let response = next.run(request).await;
    drop(guard.0.take());
    trace.status(response.status().as_u16());
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(LoggedBody {
            inner: body,
            trace,
            completed: false,
        }),
    )
}
