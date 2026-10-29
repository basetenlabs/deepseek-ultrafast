use deepseek_ultrafast_model::limits::{MAX_M, OUT_CAP};
use std::sync::{Arc, atomic::AtomicBool};
use tokio::sync::mpsc::UnboundedSender;

#[derive(Clone)]
pub struct Job {
    pub trace: crate::request_log::Trace,
    pub session: String,
    pub images: Option<Arc<crate::image_identity::Images>>,
    pub tokens: Arc<Vec<i32>>,
    pub max_new: usize,
    pub seed: u64,
    pub stop_ids: Arc<Vec<i32>>,
    pub cancel: Arc<AtomicBool>,
    pub events: UnboundedSender<Event>,
}
impl Job {
    pub fn prompt(&self) -> deepseek_ultrafast_kv_transfer::Prompt<'_> {
        deepseek_ultrafast_kv_transfer::Prompt::new(&self.tokens, self.spans())
    }
    pub fn spans(&self) -> &[deepseek_ultrafast_kv_transfer::ImageSpan] {
        self.images.as_ref().map_or(&[], |v| &v.spans)
    }
    pub fn done(&self, finish: Finish, completion_tokens: usize) {
        self.trace.engine_finished(match finish { Finish::Stop => "stop", Finish::Length => "length", Finish::Cancelled => "cancelled" });
        let _ = self.events.send(Event::Done {
            finish,
            completion_tokens,
        });
    }
}
impl crate::request_queue::Request for Job {
    fn cancelled(&self) -> bool { self.cancel.load(std::sync::atomic::Ordering::Relaxed) }
    fn discard(self) { self.done(Finish::Cancelled, 0); }
}
#[derive(Debug)]
pub enum Event {
    Started {
        prompt_tokens: usize,
        cached_tokens: usize,
    },
    Tokens(Vec<i32>),
    Done {
        finish: Finish,
        completion_tokens: usize,
    },
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Finish {
    Stop,
    Length,
    Cancelled,
}

pub fn max_new_limit(n: usize, max_seq: usize) -> usize {
    max_seq
        .saturating_sub(n + 4 * MAX_M)
        .min(OUT_CAP - 4 * MAX_M)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_budget_reserves_speculative_headroom() {
        let context = 1048576;
        let headroom = 4 * MAX_M;
        assert_eq!(max_new_limit(1, context), 256000);
        let full_budget_prompt = context - 256000 - headroom;
        assert_eq!(max_new_limit(full_budget_prompt, context), 256000);
        assert_eq!(max_new_limit(full_budget_prompt + 1, context), 255999);
        assert_eq!(max_new_limit(4096, 8192), 4096 - headroom);
        assert_eq!(max_new_limit(context - headroom - 1, context), 1);
        assert_eq!(max_new_limit(context - headroom, context), 0);
        assert_eq!(max_new_limit(context, context), 0);
        assert_eq!(max_new_limit(context + 1, context), 0);
    }
}
