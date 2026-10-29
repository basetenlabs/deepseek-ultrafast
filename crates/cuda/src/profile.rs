use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Profile {
    Sm103Tp4,
    Sm103Tp8,
    Base,
}

static PROFILE: OnceLock<(Profile, usize)> = OnceLock::new();
static READ_EARLY: AtomicBool = AtomicBool::new(false);

pub fn init(cc: (i32, i32), tp: usize) {
    assert!(!READ_EARLY.load(Ordering::Acquire), "the profile was read before profile::init");
    let p = match (cc, tp) {
        ((10, 3), 4) => Profile::Sm103Tp4,
        ((10, 3), 8) => Profile::Sm103Tp8,
        _ => Profile::Base,
    };
    let _ = PROFILE.set((p, tp));
}

fn get() -> (Profile, usize) {
    match PROFILE.get() {
        Some(&p) => p,
        None => {
            READ_EARLY.store(true, Ordering::Release);
            (Profile::Base, 0)
        }
    }
}

pub fn profile() -> Profile {
    get().0
}

pub fn sm103() -> bool {
    matches!(profile(), Profile::Sm103Tp4 | Profile::Sm103Tp8)
}

pub fn sm103_tp4() -> bool {
    profile() == Profile::Sm103Tp4
}

pub fn sm103_tp8() -> bool {
    profile() == Profile::Sm103Tp8
}

pub fn vadapt() -> f32 {
    match get() {
        (Profile::Sm103Tp4, _) => 0.3,
        (Profile::Sm103Tp8, _) => 0.2,
        (Profile::Base, tp) if tp >= 8 => 0.2,
        _ => 0.3,
    }
}

pub fn dspark_block() -> Option<usize> {
    match get() {
        (Profile::Sm103Tp4 | Profile::Sm103Tp8, _) => Some(6),
        (Profile::Base, tp) if tp >= 8 => Some(7),
        _ => None,
    }
}
