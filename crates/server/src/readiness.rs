use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Default)]
pub struct Readiness(Arc<AtomicU8>);
impl Readiness {
    pub fn ready() -> Self {
        let value = Self::default();
        value.set_ready();
        value
    }
    pub fn is_ready(&self) -> bool {
        self.0.load(Ordering::Acquire) == 1
    }
    pub fn set_ready(&self) {
        if self.0.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            deepseek_ultrafast_logging::emit(deepseek_ultrafast_logging::Event::HealthReady);
        }
    }
    pub fn stop(&self) {
        self.0.store(2, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shutdown_is_terminal() {
        let r = Readiness::default();
        assert!(!r.is_ready());
        r.set_ready();
        assert!(r.is_ready());
        r.stop();
        r.set_ready();
        assert!(!r.is_ready());
    }
}
