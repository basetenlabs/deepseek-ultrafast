use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

struct Job {
    id: usize,
    cancel: Arc<AtomicBool>,
    discarded: Arc<Mutex<Vec<usize>>>,
}
impl Request for Job {
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
    fn discard(self) {
        self.discarded.lock().unwrap().push(self.id);
    }
}
fn job(id: usize) -> Job {
    Job {
        id,
        cancel: Default::default(),
        discarded: Default::default(),
    }
}
fn ready(q: &Queue<Job, usize>, id: usize) {
    q.push(job(id));
    let j = q.prepare().unwrap();
    assert_eq!(j.id, id);
    q.prepared(j, id);
}
fn next(q: &Queue<Job, usize>) -> usize {
    let (j, payload) = q.pop().unwrap();
    assert_eq!(j.id, payload);
    j.id
}

#[test]
fn fifo_admission_does_not_preempt_active() {
    let q = Queue::default();
    ready(&q, 0);
    let active = q.pop().unwrap();
    ready(&q, 1);
    ready(&q, 2);
    ready(&q, 3);
    assert_eq!(active.0.id, 0);
    drop(active);
    assert_eq!([next(&q), next(&q), next(&q)], [1, 2, 3]);
    q.close();
    assert!(q.pop().is_none());
    assert!(q.prepare().is_none());
}

#[test]
fn cancellation_prunes_pending_ready_and_completed_transfers_once() {
    let q: Queue<Job, usize> = Queue::default();
    let j = job(1);
    let cancelled = j.cancel.clone();
    let discarded = j.discarded.clone();
    q.push(j);
    cancelled.store(true, Ordering::Relaxed);
    let j = job(2);
    let cancel_ready = j.cancel.clone();
    let discarded_ready = j.discarded.clone();
    q.push(j);
    let j = q.prepare().unwrap();
    assert_eq!(j.id, 2);
    q.prepared(j, 2);
    cancel_ready.store(true, Ordering::Relaxed);
    let j = job(3);
    let cancel_fetching = j.cancel.clone();
    let discarded_fetching = j.discarded.clone();
    q.push(j);
    let j = q.prepare().unwrap();
    assert_eq!(j.id, 3);
    cancel_fetching.store(true, Ordering::Relaxed);
    q.prepared(j, 3);
    q.close();
    assert!(q.pop().is_none());
    assert!(q.prepare().is_none());
    assert_eq!(*discarded.lock().unwrap(), [1]);
    assert_eq!(*discarded_ready.lock().unwrap(), [2]);
    assert_eq!(*discarded_fetching.lock().unwrap(), [3]);
}
