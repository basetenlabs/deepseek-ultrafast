use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::fd::{AsFd, AsRawFd};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::{self, OwnedPermit, Sender};

const CAPACITY: usize = 4096;
static SENDER: OnceLock<Sender<Message>> = OnceLock::new();
static OUTPUT: OnceLock<Mutex<File>> = OnceLock::new();
static ERRORS: OnceLock<Mutex<File>> = OnceLock::new();

#[derive(Debug, Serialize)]
pub struct RequestEnd {
    pub endpoint: &'static str,
    pub session_id: Option<Arc<str>>,
    pub session_origin: Option<&'static str>,
    pub input_tokens: Option<usize>,
    pub output_tokens: usize,
    pub cached_tokens: Option<usize>,
    pub stop_reason: &'static str,
    pub request_mode: &'static str,
    pub streaming: Option<bool>,
    pub terminal_error: Option<TerminalError>,
    pub tpot_ms: Option<f64>,
    pub itl_ms: Option<f64>,
    pub average_accept_length: Option<f64>,
    pub parse_tokenize_ms: Option<f64>,
    pub kv_fetch_end_ms: Option<f64>,
    pub prefill_end_ms: Option<f64>,
    pub e2e_ms: f64,
    pub ttft_ms: Option<f64>,
    pub average_iteration_ms: Option<f64>,
    pub decode_tps: Option<f64>,
    pub queue_ms: Option<f64>,
    pub http_status_code: Option<u16>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct TerminalError {
    pub category: &'static str,
    pub parser_category: Option<&'static str>,
    pub line: Option<usize>,
    pub column: Option<usize>,
}

#[derive(Debug, Serialize)]
#[expect(
    clippy::large_enum_variant,
    reason = "Inline records avoid a completion-path allocation; the queue is bounded"
)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    ModelLoadingStart,
    ModelLoadingEnd {
        success: bool,
        elapsed_ms: f64,
    },
    CudaGraphCaptureStart,
    CudaGraphCaptureEnd {
        success: bool,
        elapsed_ms: f64,
    },
    HealthReady,
    RedisConnectionFailure {
        component: &'static str,
        operation: &'static str,
    },
    KvTransferConnectionFailure {
        operation: &'static str,
    },
    KvEvicted {
        session_id: Arc<str>,
        reason: EvictionReason,
        idle_age_ms: f64,
    },
    RequestEnd(RequestEnd),
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionReason {
    OtherReplica,
    Lru,
    OwnershipInvalidated,
}

#[derive(Serialize)]
struct Line {
    timestamp_unix_ms: u128,
    #[serde(flatten)]
    event: Event,
}
impl Line {
    fn new(event: Event) -> Self {
        Self {
            timestamp_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            event,
        }
    }
}
#[expect(
    clippy::large_enum_variant,
    reason = "Bounded inline records; no heap allocation per completion"
)]
enum Message {
    Line(Line),
    Flush(std::sync::mpsc::SyncSender<()>),
    Stop,
}

pub fn serving() -> bool {
    SENDER.get().is_some()
}

#[macro_export]
macro_rules! diagnostic {
    ($($arg:tt)*) => { if !$crate::serving() { eprintln!($($arg)*); } };
}

pub struct Guard(Option<std::thread::JoinHandle<()>>);
pub fn init() -> io::Result<Guard> {
    let output = File::from(io::stdout().as_fd().try_clone_to_owned()?);
    OUTPUT
        .set(Mutex::new(output))
        .map_err(|_| io::Error::other("logging initialized twice"))?;
    let errors = File::from(io::stderr().as_fd().try_clone_to_owned()?);
    let _ = ERRORS.set(Mutex::new(errors));
    let null = OpenOptions::new().write(true).open("/dev/null")?;
    for fd in [libc::STDOUT_FILENO, libc::STDERR_FILENO] {
        if unsafe { libc::dup2(null.as_raw_fd(), fd) } < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let (tx, mut rx) = mpsc::channel(CAPACITY);
    SENDER
        .set(tx)
        .map_err(|_| io::Error::other("logging initialized twice"))?;
    let worker = std::thread::Builder::new()
        .name("json-log-writer".into())
        .spawn(move || {
            let mut bytes = Vec::with_capacity(8192);
            while let Some(message) = rx.blocking_recv() {
                match message {
                    Message::Line(line) => write_line(&line, &mut bytes),
                    Message::Flush(done) => {
                        let _ = done.send(());
                    }
                    Message::Stop => break,
                }
            }
        })?;
    Ok(Guard(Some(worker)))
}
impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(sender) = SENDER.get() {
            let _ = sender.blocking_send(Message::Stop);
        }
        if let Some(worker) = self.0.take() {
            let _ = worker.join();
        }
    }
}
fn write_line(line: &Line, bytes: &mut Vec<u8>) {
    bytes.clear();
    if serde_json::to_writer(&mut *bytes, line).is_ok() {
        bytes.push(b'\n');
        if let Some(output) = OUTPUT.get() {
            let _ = output
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .write_all(bytes);
        }
    }
}

pub fn emit(event: Event) {
    if let Some(sender) = SENDER.get() {
        let line = Line::new(event);
        if let Err(error) = sender.try_send(Message::Line(line))
            && let Message::Line(line) = error.into_inner()
        {
            write_line(&line, &mut Vec::with_capacity(1024));
        }
    }
}

#[derive(Default)]
pub struct Ticket(Option<OwnedPermit<Message>>);
impl Ticket {
    pub async fn reserve() -> Self {
        match SENDER.get() {
            Some(sender) => Self(sender.clone().reserve_owned().await.ok()),
            None => Self::default(),
        }
    }
    pub fn finish(self, request: RequestEnd) {
        if let Some(permit) = self.0 {
            permit.send(Message::Line(Line::new(Event::RequestEnd(request))));
        } else {
            emit(Event::RequestEnd(request));
        }
    }
}

pub fn flush_fatal() {
    if let Some(sender) = SENDER.get() {
        let (done, ack) = std::sync::mpsc::sync_channel(1);
        if sender.try_send(Message::Flush(done)).is_ok() {
            let _ = ack.recv_timeout(Duration::from_secs(1));
        }
    }
}

pub fn redis_failure(component: &'static str, operation: &'static str) {
    emit(Event::RedisConnectionFailure {
        component,
        operation,
    });
}
pub fn kv_failure(operation: &'static str) {
    emit(Event::KvTransferConnectionFailure { operation });
}
pub fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

pub fn termination(reason: &str) {
    let line = format!("error: {reason}\n");
    match ERRORS.get() {
        Some(errors) => {
            let _ = errors.lock().unwrap_or_else(|e| e.into_inner()).write_all(line.as_bytes());
        }
        None => eprint!("{line}"),
    }
    #[derive(Serialize)]
    struct Failure<'a> {
        error: &'a str,
    }
    let end = reason
        .char_indices()
        .nth(512)
        .map_or(reason.len(), |(i, _)| i);
    if let Ok(bytes) = serde_json::to_vec(&Failure {
        error: &reason[..end],
    }) {
        let _ = std::fs::write("/dev/termination-log", bytes);
    }
}
