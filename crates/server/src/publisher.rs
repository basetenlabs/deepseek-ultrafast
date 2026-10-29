use anyhow::Result;
use deepseek_ultrafast_engine::kvstore::ExportSlot;
use deepseek_ultrafast_engine::model::Rank;
use deepseek_ultrafast_kv_transfer::{ImageSpan, PeerCache, Revision, Snapshot, is_rdma_required};
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, TryLockError, Weak, mpsc};
use std::time::Duration;

pub type Fence = Arc<Mutex<Option<Revision>>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capture {
    Prefill,
    Complete(usize),
}
const PROMPT_REWIND: usize = 256;

impl Capture {
    fn input_tokens(self, total: usize) -> Result<usize> {
        let generated = match self { Self::Prefill => 0, Self::Complete(n) => n };
        total.checked_sub(generated).filter(|&n| n > 0)
            .ok_or_else(|| anyhow::anyhow!("invalid capture input boundary"))
    }
    fn tail(self) -> usize {
        let generated = match self {
            Self::Prefill => 0,
            Self::Complete(n) => n,
        };
        generated
            .saturating_add(deepseek_ultrafast_kv_transfer::TAIL_READ + PROMPT_REWIND)
            .min(deepseek_ultrafast_kv_transfer::SAVED_TAIL)
    }
}

fn take_slot<T>(slots: &mut Vec<T>, phase: Capture) -> Option<T> {
    if phase == Capture::Prefill && slots.len() < 2 {
        None
    } else {
        slots.pop()
    }
}

struct FenceRef {
    origin: Option<Revision>,
    fence: Weak<Mutex<Option<Revision>>>,
}
#[derive(Default)]
struct Fences(HashMap<String, FenceRef>);
impl Fences {
    fn begin(&mut self, session: &str, observed: Option<Revision>, local: bool) -> Fence {
        self.0.retain(|_, entry| entry.fence.strong_count() > 0);
        if local
            && let Some(observed) = observed
            && let Some(fence) = self
                .0
                .get(session)
                .filter(|entry| entry.origin.is_some_and(|r| r.epoch == observed.epoch))
                .and_then(|entry| entry.fence.upgrade())
        {
            let current = match fence.try_lock() {
                Ok(revision) => revision
                    .is_some_and(|r| r.epoch == observed.epoch && r.sequence >= observed.sequence),
                Err(TryLockError::WouldBlock) => true,
                Err(TryLockError::Poisoned(_)) => panic!("KV publication fence poisoned"),
            };
            if current {
                return fence;
            }
        }
        let fence = Arc::new(Mutex::new(observed));
        self.0.insert(
            session.into(),
            FenceRef {
                origin: observed,
                fence: Arc::downgrade(&fence),
            },
        );
        fence
    }
}

#[derive(Default)]
pub struct Pending {
    sessions: Mutex<HashMap<String, usize>>,
    done: Condvar,
}
impl Pending {
    fn add(&self, session: &str) {
        *self.sessions.lock().unwrap().entry(session.into()).or_default() += 1;
    }
    fn finish(&self, session: &str) {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some(n) = sessions.get_mut(session) {
            *n -= 1;
            if *n == 0 {
                sessions.remove(session);
            }
        }
        self.done.notify_all();
    }
    pub fn wait(&self, session: &str, timeout: Duration) -> bool {
        let sessions = self.sessions.lock().unwrap();
        let (sessions, _) = self.done.wait_timeout_while(sessions, timeout, |s| s.contains_key(session)).unwrap();
        !sessions.contains_key(session)
    }
}

struct Work {
    session: String,
    session_id: Option<Arc<str>>,
    revision: Fence,
    input_tokens: usize,
    finish: Box<dyn FnOnce() -> Result<Arc<Snapshot>> + Send>,
}

fn drain(
    recv: mpsc::Receiver<Work>,
    pending: &Pending,
    publish: impl Fn(&str, Revision, Arc<Snapshot>, usize, Option<Arc<str>>) -> Result<Revision>,
) -> Result<()> {
    while let Ok(work) = recv.recv() {
        let snapshot = (work.finish)()?;
        let mut revision = work.revision.lock().unwrap();
        if let Some(expected) = *revision {
            match publish(&work.session, expected, snapshot, work.input_tokens, work.session_id) {
                Ok(next) => *revision = Some(next),
                Err(e) if is_rdma_required(&e) => return Err(e),
                Err(e) => {
                    *revision = None;
                    deepseek_ultrafast_logging::diagnostic!("KV background publication fenced: {e:#}");
                }
            }
        }
        drop(revision);
        pending.finish(&work.session);
    }
    Ok(())
}

pub struct Publisher {
    slots: Arc<Mutex<Vec<ExportSlot>>>,
    send: Option<mpsc::SyncSender<Work>>,
    worker: Option<std::thread::JoinHandle<()>>,
    fingerprint: String,
    fences: Fences,
    pending: Arc<Pending>,
}
impl Publisher {
    pub fn new(rank: &Rank, peer: Arc<PeerCache>, runtime: tokio::runtime::Handle) -> Result<Self> {
        let slots = Arc::new(Mutex::new(vec![
            ExportSlot::new(rank, peer.max_bytes())?,
            ExportSlot::new(rank, peer.max_bytes())?,
        ]));
        let fingerprint = peer.fingerprint().to_string();
        let (send, recv) = mpsc::sync_channel::<Work>(2);
        let pending = Arc::new(Pending::default());
        let drained = pending.clone();
        let worker = std::thread::Builder::new()
            .name("kv-publisher".into())
            .spawn(move || {
                if let Err(e) = drain(recv, &drained, |session, revision, snapshot, input_tokens, session_id| {
                    runtime.block_on(peer.publish_with_input_named(session, revision, snapshot, input_tokens, session_id))
                }) {
                    deepseek_ultrafast_logging::termination(&format!("kv publisher: {e:#}"));
                    deepseek_ultrafast_logging::flush_fatal();
                    std::process::exit(1);
                }
            })?;
        Ok(Self {
            slots,
            send: Some(send),
            worker: Some(worker),
            fingerprint,
            fences: Fences::default(),
            pending,
        })
    }
    pub fn pending(&self) -> Arc<Pending> {
        self.pending.clone()
    }
    pub fn fence(&mut self, session: &str, observed: Option<Revision>, local: bool) -> Fence {
        self.fences.begin(session, observed, local)
    }
    pub fn capture(
        &self,
        rank: &Rank,
        session: &str,
        session_id: Option<Arc<str>>,
        tokens: &[i32],
        images: &[ImageSpan],
        revision: Fence,
        phase: Capture,
    ) -> Result<()> {
        let input_tokens = phase.input_tokens(tokens.len())?;
        let Some(slot) = take_slot(&mut self.slots.lock().unwrap(), phase) else {
            return Ok(());
        };
        if !slot.fits(rank, tokens.len(), phase.tail()) {
            self.slots.lock().unwrap().push(slot);
            return Ok(());
        }
        let image = slot.capture(rank, tokens, images, &self.fingerprint, phase.tail())?;
        let returned = self.slots.clone();
        let finish = Box::new(move || {
            let (snapshot, slot) = image.finish()?;
            returned.lock().unwrap().push(slot);
            Ok(Arc::new(snapshot))
        });
        self.pending.add(session);
        self.send
            .as_ref()
            .unwrap()
            .try_send(Work {
                session: session.into(),
                session_id,
                revision,
                input_tokens,
                finish,
            })
            .map_err(|_| {
                self.pending.finish(session);
                anyhow::anyhow!("KV publication worker unavailable")
            })?;
        Ok(())
    }
}
impl Drop for Publisher {
    fn drop(&mut self) {
        self.send.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod pending_tests {
    use super::*;

    #[test]
    fn waits_for_its_own_session_only() {
        let pending = Arc::new(Pending::default());
        assert!(pending.wait("a", Duration::ZERO));
        pending.add("a");
        pending.add("a");
        assert!(pending.wait("b", Duration::ZERO), "another session's publication");
        assert!(!pending.wait("a", Duration::from_millis(20)), "in flight: the wait times out");
        let p = pending.clone();
        let t = std::thread::spawn(move || {
            p.finish("a");
            std::thread::sleep(Duration::from_millis(20));
            p.finish("a");
        });
        assert!(pending.wait("a", Duration::from_secs(5)), "both publications finished");
        t.join().unwrap();
    }
}
