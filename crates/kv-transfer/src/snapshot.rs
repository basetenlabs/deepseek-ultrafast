use crate::Segment;
use anyhow::{Context, Result, ensure};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const VERSION: u32 = 3;
pub const TAIL_READ: usize = 256;
pub const SAVED_TAIL: usize = 8192;

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn payload_digest(bytes: &[u8]) -> String {
    if bytes.len() < 1 << 20 {
        return blake3::hash(bytes).to_hex().to_string();
    }
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    let pool = POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(std::thread::available_parallelism().map_or(1, |n| n.get().min(4)))
            .thread_name(|i| format!("kv-checksum-{i}"))
            .build()
            .expect("KV checksum pool")
    });
    pool.install(|| {
        let mut hash = blake3::Hasher::new();
        hash.update_rayon(bytes);
        hash.finalize().to_hex().to_string()
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSpan {
    pub start: usize,
    pub end: usize,
    pub height: usize,
    pub width: usize,
    pub digest: [u8; 32],
}
pub const MAX_IMAGES: usize = 1024;

#[derive(Clone, Copy, Debug)]
pub struct Prompt<'a> {
    pub tokens: &'a [i32],
    pub images: &'a [ImageSpan],
}
impl<'a> Prompt<'a> {
    pub fn text(tokens: &'a [i32]) -> Self { Self { tokens, images: &[] } }
    pub fn new(tokens: &'a [i32], images: &'a [ImageSpan]) -> Self { Self { tokens, images } }
}

pub fn image_prefix(mut prefix: usize, stored: &[ImageSpan], requested: &[ImageSpan]) -> usize {
    for i in 0..stored.len().max(requested.len()) {
        match (stored.get(i), requested.get(i)) {
            (Some(a), Some(b)) if a == b => {}
            (a, b) => {
                for im in a.into_iter().chain(b) { prefix = prefix.min(im.start); }
                break;
            }
        }
    }
    prefix
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub fingerprint: String,
    pub tokens: Vec<i32>,
    pub tail_start: usize,
    pub encoder_tail: [usize; 3],
    pub segments: Vec<Segment>,
    pub bytes: usize,
    pub blake3: String,
    #[serde(default)]
    pub images: Vec<ImageSpan>,
}

impl Manifest {
    pub fn validate(&self, fingerprint: &str, max_seq: usize, max_bytes: usize) -> Result<()> {
        ensure!(self.version == VERSION, "unsupported KV wire version");
        ensure!(self.fingerprint == fingerprint, "incompatible KV profile");
        ensure!(
            !self.tokens.is_empty() && self.tokens.len() <= max_seq,
            "invalid KV token count"
        );
        ensure!(self.tokens.iter().all(|t| *t >= 0), "invalid KV token id");
        ensure!(
            self.tail_start <= self.tokens.len(),
            "invalid KV tail start"
        );
        ensure!(self.images.len() <= MAX_IMAGES, "invalid KV image count");
        let mut end = 0;
        for im in &self.images {
            ensure!(
                im.start >= end && im.start < im.end && im.end <= self.tokens.len()
                    && im.height > 0 && im.width > 0,
                "invalid KV image span"
            );
            end = im.end;
        }
        ensure!(
            !self.segments.is_empty() && self.segments.len() <= 1024,
            "invalid KV segment count"
        );
        let sum = self.segments.iter().try_fold(0usize, |n, segment| {
            n.checked_add(segment.stored_bytes().ok()?)
        });
        ensure!(
            sum == Some(self.bytes) && self.bytes > 0 && self.bytes <= max_bytes,
            "invalid KV byte count"
        );
        ensure!(
            self.blake3.len() == 64 && self.blake3.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid KV checksum"
        );
        Ok(())
    }

    pub fn validate_geometry(&self, max_seq: usize, vocab: usize, window: usize) -> Result<()> {
        let n = self.tokens.len();
        ensure!(
            n > 0 && n <= max_seq && self.tail_start <= n,
            "invalid KV range"
        );
        ensure!(
            self.tail_start == 0 || n - self.tail_start >= TAIL_READ,
            "insufficient KV tail"
        );
        ensure!(
            self.tokens.iter().all(|&t| t >= 0 && (t as usize) < vocab),
            "invalid KV tokens"
        );
        let [start, len, end] = self.encoder_tail;
        ensure!(
            start < window && len <= window && len <= end && end <= n,
            "invalid encoder ring"
        );
        Ok(())
    }

    pub(crate) fn restore_start(&self) -> usize {
        if self.tail_start == 0 { 0 } else { self.tail_start.saturating_add(TAIL_READ) }
    }

    pub fn covers_prefix(&self, prefix: usize) -> bool {
        prefix <= self.tokens.len() && prefix >= self.restore_start()
    }

    pub fn reusable_prefix(&self, prompt: Prompt) -> usize {
        let n = image_prefix(self.token_prefix(prompt.tokens), &self.images, prompt.images);
        if self.covers_prefix(n) {
            n
        } else {
            0
        }
    }

    pub(crate) fn token_prefix(&self, tokens: &[i32]) -> usize {
        self.tokens
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(tokens.len().saturating_sub(1))
    }

    pub fn prefix(&self, n: usize) -> Prompt<'_> {
        let images = self.images.iter().take_while(|im| im.end <= n).count();
        Prompt { tokens: &self.tokens[..n.min(self.tokens.len())], images: &self.images[..images] }
    }
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub manifest: Manifest,
    pub data: Bytes,
}

impl Snapshot {
    pub fn new(
        fingerprint: String,
        tokens: Vec<i32>,
        tail_start: usize,
        encoder_tail: [usize; 3],
        segments: Vec<usize>,
        data: Bytes,
    ) -> Self {
        Self::encoded(
            fingerprint,
            tokens,
            vec![],
            tail_start,
            encoder_tail,
            segments.into_iter().map(Segment::raw).collect(),
            data,
        )
    }

    pub fn encoded(
        fingerprint: String,
        tokens: Vec<i32>,
        images: Vec<ImageSpan>,
        tail_start: usize,
        encoder_tail: [usize; 3],
        segments: Vec<Segment>,
        data: Bytes,
    ) -> Self {
        Self {
            manifest: Manifest {
                version: VERSION,
                fingerprint,
                tokens,
                tail_start,
                encoder_tail,
                segments,
                bytes: data.len(),
                blake3: payload_digest(&data),
                images,
            },
            data,
        }
    }

    pub fn validate(&self, fingerprint: &str, max_seq: usize, max_bytes: usize) -> Result<()> {
        self.manifest.validate(fingerprint, max_seq, max_bytes)?;
        ensure!(
            self.data.len() == self.manifest.bytes
                && payload_digest(&self.data) == self.manifest.blake3,
            "corrupt KV payload"
        );
        self.validate_encoding()
    }

    pub fn validate_encoding(&self) -> Result<()> {
        let mut offset = 0usize;
        for &segment in &self.manifest.segments {
            let end = offset
                .checked_add(segment.stored_bytes()?)
                .context("KV segment offset overflow")?;
            segment
                .validate_payload(self.data.get(offset..end).context("truncated KV segment")?)?;
            offset = end;
        }
        ensure!(offset == self.data.len(), "trailing KV segment bytes");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_is_exact_at_prompt_tail_and_divergence_boundaries() {
        for tail_start in [0, 1, 128, 600] {
            let s = Snapshot::new("p".into(), (0..1000).collect(), tail_start,
                [0, 128, 1000], vec![1], Bytes::from_static(b"x"));
            for n in 0usize..=1001 {
                let prompt: Vec<i32> = (0..n as i32).collect();
                let common = 1000.min(n.saturating_sub(1));
                let covered = tail_start == 0 || common >= tail_start + TAIL_READ;
                assert_eq!(s.manifest.reusable_prefix(Prompt::text(&prompt)), if covered { common } else { 0 });
                if n > 1 {
                    let mut changed = prompt;
                    changed[n - 2] = 9999;
                    let common = (n - 2).min(1000);
                    let covered = tail_start == 0 || common >= tail_start + TAIL_READ;
                    assert_eq!(s.manifest.reusable_prefix(Prompt::text(&changed)), if covered { common } else { 0 });
                }
            }
            assert!(!s.manifest.covers_prefix(1001));
        }
    }

    #[test]
    fn engine_geometry_rejects_invalid_ranges_tokens_and_ring_before_copy() {
        let s = Snapshot::new(
            "p".into(),
            (0..1000).collect(),
            744,
            [0, 128, 1000],
            vec![1],
            Bytes::from_static(b"x"),
        );
        s.manifest.validate_geometry(2000, 2000, 128).unwrap();
        for ring in [
            [128, 128, 1000],
            [0, 129, 1000],
            [0, 128, 127],
            [0, 128, 1001],
        ] {
            let mut m = s.manifest.clone();
            m.encoder_tail = ring;
            assert!(m.validate_geometry(2000, 2000, 128).is_err());
        }
        assert!(s.manifest.validate_geometry(999, 2000, 128).is_err());
        assert!(s.manifest.validate_geometry(2000, 999, 128).is_err());
        for tail in [745, 1001, usize::MAX] {
            let mut m = s.manifest.clone();
            m.tail_start = tail;
            assert!(m.validate_geometry(2000, 2000, 128).is_err());
        }
    }

    fn span(start: usize, len: usize, pixels: u8) -> ImageSpan {
        ImageSpan { start, end: start + len, height: 3, width: 3, digest: [pixels; 32] }
    }

    #[test]
    fn image_identity_clips_token_prefix_before_changed_pixels() {
        let stored = [span(100, 10, 1), span(500, 20, 2)];
        let mut tokens: Vec<i32> = (0..1000).collect();
        for im in &stored { tokens[im.start..im.end].fill(129264); }
        let s = Snapshot::new("p".into(), tokens.clone(), 0, [0, 128, 1000], vec![1], Bytes::from_static(b"x"));
        let mut m = s.manifest.clone();
        m.images = stored.to_vec();
        m.validate("p", 2000, 10).unwrap();
        assert_eq!(m.reusable_prefix(Prompt::new(&tokens, &stored)), 999);
        assert_eq!(m.reusable_prefix(Prompt::new(&tokens, &[span(100, 10, 1), span(500, 20, 3)])), 500);
        assert_eq!(m.reusable_prefix(Prompt::new(&tokens, &[span(100, 10, 1), ImageSpan { width: 4, ..span(500, 20, 2) }])), 500);
        assert_eq!(m.reusable_prefix(Prompt::new(&tokens, &[span(100, 10, 1), span(490, 20, 2)])), 490);
        assert_eq!(m.reusable_prefix(Prompt::new(&tokens, &[span(100, 10, 1)])), 500);
        assert_eq!(m.reusable_prefix(Prompt::new(&tokens, &[span(100, 10, 1), span(300, 5, 4), span(500, 20, 2)])), 300);
        assert_eq!(m.reusable_prefix(Prompt::new(&tokens, &[span(100, 10, 3), span(500, 20, 2)])), 100);
        assert_eq!(m.reusable_prefix(Prompt::text(&tokens)), 100);
        let mut text = s.manifest.clone();
        text.images.clear();
        assert_eq!(text.reusable_prefix(Prompt::new(&tokens, &stored)), 100);
        assert_eq!(m.reusable_prefix(Prompt::new(&tokens[..400], &[span(100, 10, 1)])), 399);
        let mut longer = tokens.clone();
        longer.extend(std::iter::repeat_n(129264, 10));
        longer.push(7);
        assert_eq!(m.reusable_prefix(Prompt::new(&longer, &[span(100, 10, 1), span(500, 20, 2), span(1000, 10, 5)])), 1000);
        assert_eq!(m.prefix(505).images.len(), 1);
        assert_eq!(m.prefix(520).images.len(), 2);
        assert_eq!(m.prefix(520).tokens.len(), 520);
        let mut tail = m.clone();
        tail.tail_start = 400;
        assert_eq!(tail.reusable_prefix(Prompt::new(&tokens, &stored)), 999);
        assert_eq!(tail.reusable_prefix(Prompt::new(&tokens, &[span(100, 10, 1), span(500, 20, 3)])), 0);
    }

    #[test]
    fn image_spans_validate_order_bounds_and_count() {
        let s = Snapshot::new("p".into(), (0..1000).collect(), 0, [0, 128, 1000], vec![1], Bytes::from_static(b"x"));
        for images in [
            vec![span(100, 10, 1), span(105, 10, 2)],
            vec![span(500, 20, 2), span(100, 10, 1)],
            vec![span(995, 10, 1)],
            vec![ImageSpan { end: 100, ..span(100, 10, 1) }],
            vec![ImageSpan { height: 0, ..span(100, 10, 1) }],
        ] {
            let mut m = s.manifest.clone();
            m.images = images;
            assert!(m.validate("p", 2000, 10).is_err());
        }
        let mut m = s.manifest.clone();
        m.images = vec![span(0, 10, 1), span(10, 10, 2), span(990, 10, 3)];
        m.validate("p", 2000, 10).unwrap();
        let long = Snapshot::new("p".into(), (0..2000).collect(), 0, [0, 128, 2000], vec![1], Bytes::from_static(b"x"));
        let mut m = long.manifest.clone();
        m.images = (0..MAX_IMAGES).map(|i| span(i, 1, 1)).collect();
        m.validate("p", 2000, 10).unwrap();
        m.images.push(span(MAX_IMAGES, 1, 1));
        assert!(m.validate("p", 2000, 10).is_err());
    }

    #[test]
    fn validates_bounds_layout_version_and_checksum() {
        let s = Snapshot::new(
            "p".into(),
            vec![1, 2],
            0,
            [0, 2, 2],
            vec![3],
            Bytes::from_static(b"abc"),
        );
        s.validate("p", 10, 10).unwrap();
        assert!(s.validate("other", 10, 10).is_err());
        assert!(s.validate("p", 1, 10).is_err());
        assert!(s.validate("p", 10, 2).is_err());
        let mut bad = s.clone();
        bad.data = Bytes::from_static(b"abd");
        assert!(bad.validate("p", 10, 10).is_err());
        bad.manifest.segments = vec![Segment::raw(usize::MAX), Segment::raw(1)];
        assert!(bad.validate("p", 10, 10).is_err());
        bad = s;
        bad.manifest.version += 1;
        assert!(bad.validate("p", 10, 10).is_err());
    }
}
