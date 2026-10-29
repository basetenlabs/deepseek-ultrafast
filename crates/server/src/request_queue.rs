use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex},
};

pub const MAX_PREPARED_REQUESTS: usize = 6;

pub trait Request {
    fn cancelled(&self) -> bool;
    fn discard(self);
}

struct State<J, P> {
    pending: VecDeque<J>,
    ready: VecDeque<(J, P)>,
    preparing: bool,
    retiring: usize,
    closed: bool,
}
impl<J: Request, P> State<J, P> {
    fn prune(&mut self) -> (Vec<J>, Vec<(J, P)>) {
        let mut pending = Vec::new();
        let mut ready = Vec::new();
        for _ in 0..self.pending.len() {
            let job = self.pending.pop_front().unwrap();
            if job.cancelled() {
                pending.push(job);
            } else {
                self.pending.push_back(job);
            }
        }
        for _ in 0..self.ready.len() {
            let item = self.ready.pop_front().unwrap();
            if item.0.cancelled() {
                self.retiring += 1;
                ready.push(item);
            } else {
                self.ready.push_back(item);
            }
        }
        (pending, ready)
    }
}

pub struct Queue<J, P> {
    state: Mutex<State<J, P>>,
    changed: Condvar,
}
impl<J: Request, P> Default for Queue<J, P> {
    fn default() -> Self {
        Self {
            state: Mutex::new(State {
                pending: VecDeque::new(),
                ready: VecDeque::new(),
                preparing: false,
                retiring: 0,
                closed: false,
            }),
            changed: Condvar::new(),
        }
    }
}
impl<J: Request, P> Queue<J, P> {
    pub fn push(&self, job: J) {
        let mut s = self.state.lock().unwrap();
        assert!(!s.closed, "request submitted after ingress closed");
        s.pending.push_back(job);
        self.changed.notify_all();
    }
    pub fn close(&self) {
        self.state.lock().unwrap().closed = true;
        self.changed.notify_all();
    }
    fn retire(&self, job: J, prepared: P) {
        drop(prepared);
        job.discard();
        self.state.lock().unwrap().retiring -= 1;
        self.changed.notify_all();
    }
    fn discard(&self, pending: Vec<J>, ready: Vec<(J, P)>) {
        for job in pending {
            job.discard();
        }
        for (job, prepared) in ready {
            self.retire(job, prepared);
        }
    }
    pub fn prepare(&self) -> Option<J> {
        let mut s = self.state.lock().unwrap();
        loop {
            let (pending, ready) = s.prune();
            if !pending.is_empty() || !ready.is_empty() {
                self.changed.notify_all();
                drop(s);
                self.discard(pending, ready);
                s = self.state.lock().unwrap();
                continue;
            }
            if !s.preparing {
                if s.ready.len() + s.retiring < MAX_PREPARED_REQUESTS
                    && let Some(job) = s.pending.pop_front()
                {
                    s.preparing = true;
                    return Some(job);
                }
                if s.closed && s.pending.is_empty() {
                    return None;
                }
            }
            s = self.changed.wait(s).unwrap();
        }
    }
    pub fn prepared(&self, job: J, prepared: P) {
        let mut s = self.state.lock().unwrap();
        assert!(s.preparing);
        s.preparing = false;
        if job.cancelled() {
            s.retiring += 1;
            drop(s);
            self.retire(job, prepared);
        } else {
            s.ready.push_back((job, prepared));
            self.changed.notify_all();
        }
    }
    pub fn pop(&self) -> Option<(J, P)> {
        let mut s = self.state.lock().unwrap();
        loop {
            let (pending, ready) = s.prune();
            if !pending.is_empty() || !ready.is_empty() {
                self.changed.notify_all();
                drop(s);
                self.discard(pending, ready);
                s = self.state.lock().unwrap();
                continue;
            }
            if let Some(item) = s.ready.pop_front() {
                self.changed.notify_all();
                return Some(item);
            }
            if s.closed && !s.preparing && s.pending.is_empty() && s.retiring == 0 {
                return None;
            }
            s = self.changed.wait(s).unwrap();
        }
    }
}

#[cfg(test)]
#[path = "request_queue_tests.rs"]
mod tests;
