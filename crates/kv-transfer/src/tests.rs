use super::*;
fn config() -> Config {
    Config {
        redis_url: DEFAULT_REDIS_URL.into(),
        directory_prefix: DEFAULT_DIRECTORY_PREFIX.into(),
        fingerprint: "p".into(),
        bind: "127.0.0.1:0".parse().unwrap(),
        max_bytes: 8,
        max_snapshot_bytes: 8,
        rdma_bytes: 16 << 20,
        max_seq: 10,
        timeout: Duration::from_secs(1),
        token_text: None,
    }
}
fn revision(n: u64) -> Revision {
    Revision {
        epoch: Uuid::nil(),
        sequence: n,
    }
}
fn image(byte: u8) -> Arc<Stored> {
    let snapshot = Arc::new(Snapshot::new(
        "p".into(),
        vec![1, 2],
        0,
        [0, 2, 2],
        vec![8],
        vec![byte; 8].into(),
    ));
    let staged = rdma::heap_stage(snapshot.data.clone());
    Arc::new(Stored { snapshot, staged })
}
#[test]
fn lease_pins_registered_generation_across_invalidation_replacement_and_lru() {
    let mut state = State::default();
    let cfg = config();
    let old = image(1);
    let weak = Arc::downgrade(&old);
    state.insert("s".into(), revision(10), old, cfg.max_bytes);
    let Reply::Open { lease, handle, .. } = state.serve(
        Fetch::Open {
            key: "s".into(),
            revision: revision(10),
        },
        &cfg,
    ) else {
        panic!()
    };
    state.invalidate("s", revision(9));
    state.invalidate("s", revision(10));
    assert_eq!(state.entries.len(), 1);
    state.invalidate("s", revision(11));
    assert!(state.entries.is_empty());
    state.insert("s".into(), revision(12), image(2), cfg.max_bytes);
    assert_eq!(state.leases[&lease].stored.staged.handle, handle);
    assert_eq!(&state.leases[&lease].stored.snapshot.data[..], &[1; 8]);
    assert!(weak.upgrade().is_some());
    state.serve(Fetch::Close { lease }, &cfg);
    assert!(
        weak.upgrade().is_none(),
        "closing final lease releases old staging"
    );
    assert_eq!(state.entries["s"].revision, revision(12));
    state.insert("other".into(), revision(13), image(3), cfg.max_bytes);
    assert!(!state.entries.contains_key("s"));
    assert_eq!(state.entries.len(), 1);
}

#[test]
fn entries_serve_identical_images_and_stop_at_a_changed_one() {
    const IM: i32 = 129264;
    let span = |start: usize, len: usize, pixels: u8| snapshot::ImageSpan {
        start, end: start + len, height: 3, width: 3, digest: [pixels; 32],
    };
    let stored = |tokens: Vec<i32>, images: Vec<snapshot::ImageSpan>| {
        let n = tokens.len();
        let snapshot = Arc::new(Snapshot::encoded("p".into(), tokens, images, 0, [0, n, n], vec![Segment::raw(8)], vec![1u8; 8].into()));
        let staged = rdma::heap_stage(snapshot.data.clone());
        Arc::new(Stored { snapshot, staged })
    };
    let prompt = vec![1, 2, IM, IM, IM, 3, 4];
    let mut full = prompt.clone();
    full.extend([7, 8]);
    let a = span(2, 3, 1);
    let entry = Entry {
        stored: stored(full, vec![a]),
        checkpoint: Some(stored(prompt.clone(), vec![a])),
        revision: revision(1), used: 0, last_used: Instant::now(), session_id: None,
    };
    let next = full_with(&entry, &[9, IM, IM, IM, 5, 6]);
    let b = span(10, 3, 2);
    let selected = entry.select(Prompt::new(&next, &[a, b])).expect("hit");
    assert!(Arc::ptr_eq(selected, &entry.stored));
    assert_eq!(selected.snapshot.manifest.reusable_prefix(Prompt::new(&next, &[a, b])), 9);
    let changed = span(2, 3, 9);
    let selected = entry.select(Prompt::new(&next, &[changed, b])).expect("the leading text");
    assert_eq!(selected.snapshot.manifest.reusable_prefix(Prompt::new(&next, &[changed, b])), 2);
    assert_eq!(entry.stored.snapshot.manifest.reusable_prefix(Prompt::text(&next)), 2);
    let mut rewrite = prompt.clone();
    rewrite.extend([70, 80]);
    let selected = entry.select(Prompt::new(&rewrite, &[a])).expect("input");
    assert_eq!(selected.snapshot.manifest.reusable_prefix(Prompt::new(&rewrite, &[a])), 7);
    assert_eq!(entry.checkpoint.as_ref().unwrap().snapshot.manifest.reusable_prefix(Prompt::new(&rewrite, &[a])), 7);
    let lead = vec![IM, IM, IM, 3, 4, 7, 8];
    let front = span(0, 3, 1);
    let entry = Entry {
        stored: stored(lead.clone(), vec![front]), checkpoint: None,
        revision: revision(1), used: 0, last_used: Instant::now(), session_id: None,
    };
    let later = |images: Vec<snapshot::ImageSpan>| {
        let mut m = stored(lead.clone(), images).snapshot.manifest.clone();
        m.tail_start = 1;
        m
    };
    assert!(entry.checkpoint_for(&later(vec![front]), 5).is_some());
    assert!(entry.checkpoint_for(&later(vec![span(0, 3, 9)]), 5).is_none());
    assert!(entry.select(Prompt::new(&lead, &[front])).is_some());
    assert!(entry.select(Prompt::new(&lead, &[span(0, 3, 9)])).is_none());
    let other = span(0, 3, 9);
    let entry = Entry {
        stored: stored(lead.clone(), vec![other]), checkpoint: Some(stored(lead.clone(), vec![front])),
        revision: revision(1), used: 0, last_used: Instant::now(), session_id: None,
    };
    let selected = entry.select(Prompt::new(&lead, &[front])).expect("the checkpoint's pixels");
    assert!(Arc::ptr_eq(selected, entry.checkpoint.as_ref().unwrap()));
    let selected = entry.select(Prompt::new(&lead, &[other])).expect("the current snapshot's pixels");
    assert!(Arc::ptr_eq(selected, &entry.stored));
    assert!(entry.select(Prompt::new(&lead, &[span(0, 3, 5)])).is_none());
}
fn full_with(entry: &Entry, tail: &[i32]) -> Vec<i32> {
    let mut next = entry.stored.snapshot.manifest.tokens.clone();
    next.extend_from_slice(tail);
    next
}
