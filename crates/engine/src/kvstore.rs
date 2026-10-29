use crate::model::{EncoderTail, Rank};
use anyhow::{Result, ensure};
use deepseek_ultrafast_cuda::{self as cu, DevPtr};

pub use deepseek_ultrafast_kv_transfer::TAIL_READ;
use deepseek_ultrafast_kv_transfer::{Encoding, ImageSpan, SAVED_TAIL, Segment, Snapshot};

pub struct ExportSlot {
    image: cu::Buffer,
    staging: cu::PinnedHost,
    stream: cu::Stream,
    ready: cu::Event,
    device: i32,
    fp4: bool,
    valid: cu::Buffer,
}

pub struct PendingExport {
    slot: ExportSlot,
    tokens: Vec<i32>,
    images: Vec<ImageSpan>,
    fingerprint: String,
    tail_start: usize,
    encoder_tail: [usize; 3],
    segments: Vec<Segment>,
}

impl ExportSlot {
    pub fn new(rk: &Rank, budget: usize) -> Result<Self> {
        Self::with_format(rk, budget, true)
    }

    pub fn with_format(rk: &Rank, budget: usize, fp4: bool) -> Result<Self> {
        let bytes = image_bytes(rk, rk.max_seq, SAVED_TAIL, fp4)?.min(budget);
        ensure!(bytes >= 4, "KV capture budget is too small");
        Ok(Self {
            image: cu::Buffer::alloc(bytes)?,
            staging: cu::PinnedHost::alloc(bytes.min(8 << 20))?,
            stream: cu::Stream::new()?,
            ready: cu::Event::new()?,
            device: rk.dev.ordinal,
            fp4,
            valid: cu::Buffer::alloc(4)?,
        })
    }

    pub fn fits(&self, rk: &Rank, n: usize, tail: usize) -> bool {
        image_bytes(rk, n, tail, self.fp4).is_ok_and(|bytes| bytes <= self.image.bytes)
    }

    pub fn capture(
        self,
        rk: &Rank,
        tokens: &[i32],
        images: &[ImageSpan],
        fingerprint: &str,
        tail: usize,
    ) -> Result<PendingExport> {
        ensure!(
            (TAIL_READ + 1..=SAVED_TAIL).contains(&tail),
            "invalid snapshot rollback window"
        );
        ensure!(
            !tokens.is_empty() && tokens.len() <= rk.max_seq && self.fits(rk, tokens.len(), tail),
            "invalid asynchronous snapshot size"
        );
        ensure!(images.iter().all(|im| im.end <= tokens.len()), "image spans outside the snapshot");
        let tail_start = tokens.len().saturating_sub(tail);
        let segs = encoded_segments(rk, tokens.len(), tail_start, false, self.fp4);
        let tail = rk.encoder_tail_state();
        let pending = PendingExport {
            slot: self,
            tokens: tokens.to_vec(),
            images: images.to_vec(),
            fingerprint: fingerprint.into(),
            tail_start,
            encoder_tail: [tail.start, tail.len, tail.end],
            segments: segs.iter().map(|s| s.1).collect(),
        };
        let queued = (|| -> Result<()> {
            cu::memset_d32_async(pending.slot.valid.ptr, 1, 1, &rk.stream)?;
            let mut offset = 0;
            for (ptr, segment) in segs {
                let dst = pending.slot.image.ptr + offset;
                if segment.encoding == Encoding::Raw {
                    cu::memcpy_dtod_async(dst, ptr, segment.decoded_bytes, &rk.stream)?;
                } else {
                    crate::fp4::pack(
                        &rk.k,
                        &rk.stream,
                        segment.encoding,
                        ptr,
                        dst,
                        pending.slot.valid.ptr,
                        segment.decoded_bytes / 2,
                    )?;
                }
                offset += segment.stored_bytes()? as u64;
            }
            pending.slot.ready.record(&rk.stream)
        })();
        if queued.is_err() {
            rk.stream.synchronize()?;
        }
        queued?;
        Ok(pending)
    }
}

impl PendingExport {
    pub fn finish(mut self) -> Result<(Snapshot, ExportSlot)> {
        let device = cu::Device::new(self.slot.device)?;
        device.bind()?;
        self.slot.stream.wait(&self.slot.ready)?;
        let status = &mut self.slot.staging.as_mut_slice::<u8>()[..4];
        cu::memcpy_dtoh_async(status, self.slot.valid.ptr, &self.slot.stream)?;
        self.slot.stream.synchronize()?;
        ensure!(
            u32::from_ne_bytes(status.try_into()?) == 1,
            "invalid FP4 capture: nonfinite or out-of-range KV"
        );
        let bytes = self
            .segments
            .iter()
            .try_fold(0usize, |n, s| Ok::<_, anyhow::Error>(n + s.stored_bytes()?))?;
        let mut data = Vec::new();
        data.try_reserve_exact(bytes)?;
        for offset in (0..bytes).step_by(self.slot.staging.bytes) {
            let n = (bytes - offset).min(self.slot.staging.bytes);
            let buf = &mut self.slot.staging.as_mut_slice::<u8>()[..n];
            cu::memcpy_dtoh_async(buf, self.slot.image.ptr + offset as u64, &self.slot.stream)?;
            self.slot.stream.synchronize()?;
            data.extend_from_slice(buf);
        }
        let snapshot = Snapshot::encoded(
            self.fingerprint,
            self.tokens,
            self.images,
            self.tail_start,
            self.encoder_tail,
            self.segments,
            data.into(),
        );
        snapshot.validate_encoding()?;
        Ok((snapshot, self.slot))
    }
}

impl Drop for ExportSlot {
    fn drop(&mut self) {
        if let Ok(device) = cu::Device::new(self.device) {
            let _ = device.bind();
            let _ = self.ready.synchronize();
            let _ = self.stream.synchronize();
        }
    }
}

pub fn validate_snapshot(rk: &Rank, snapshot: &Snapshot) -> Result<()> {
    let m = &snapshot.manifest;
    m.validate_geometry(rk.max_seq, rk.cfg.vocab_size, rk.cfg.window_size)?;
    ensure!(
        m.version == deepseek_ultrafast_kv_transfer::VERSION,
        "unsupported KV wire version"
    );
    let segs = encoded_segments(rk, m.tokens.len(), m.tail_start, false, true);
    ensure!(
        segs.len() == m.segments.len()
            && segs.iter().zip(&m.segments).all(|((_, expected), actual)| {
                expected.decoded_bytes == actual.decoded_bytes
                    && (actual.encoding == Encoding::Raw || actual.encoding == expected.encoding)
            }),
        "KV segment layout mismatch"
    );
    ensure!(m.bytes == snapshot.data.len(), "KV payload length mismatch");
    snapshot.validate_encoding()
}

pub struct RestoreSlot {
    host: cu::PinnedHost,
    device: cu::Buffer,
}

impl RestoreSlot {
    pub fn new() -> Result<Self> {
        Ok(Self {
            host: cu::PinnedHost::alloc(8 << 20)?,
            device: cu::Buffer::alloc(8 << 20)?,
        })
    }

    pub fn import_prefix(&mut self, rk: &Rank, snapshot: &Snapshot, prefix: usize) -> Result<()> {
        validate_snapshot(rk, snapshot)?;
        let m = &snapshot.manifest;
        ensure!(prefix > 0 && prefix <= m.tokens.len() && m.covers_prefix(prefix),
                "KV truncation outside saved coverage");
        let result = self.copy(rk, snapshot, prefix);
        if result.is_err() {
            let _ = rk.stream.synchronize();
        }
        result
    }

    fn copy(&mut self, rk: &Rank, snapshot: &Snapshot, prefix: usize) -> Result<()> {
        let m = &snapshot.manifest;
        let [start, len, end] = m.encoder_tail;
        let tail = EncoderTail { start, len, end }.truncate(prefix);
        let full = encoded_layout(rk, m.tokens.len(), m.tail_start, false, true);
        let mut clipped = encoded_layout(rk, prefix, m.tail_start, false, true);
        if tail.len == 0 {
            for (_, segment) in clipped.iter_mut().rev().take(2) { segment.decoded_bytes = 0; }
        }
        let plan = restore_plan(&full, &clipped, &m.segments)?;
        for (ptr, segment, decoded_bytes, off) in plan {
            let source = &snapshot.data[off..off + segment.stored_bytes()?];
            if let Some(group) = segment.encoding.group() {
                let groups = decoded_bytes / (2 * group);
                let per_chunk = self.host.bytes / (group / 2 + 1);
                for first in (0..groups).step_by(per_chunk) {
                    let count = (groups - first).min(per_chunk);
                    let data_bytes = count * group / 2;
                    let scales_start = segment.decoded_bytes / 4 + first;
                    let buf = &mut self.host.as_mut_slice::<u8>()[..data_bytes + count];
                    buf[..data_bytes].copy_from_slice(
                        &source[first * group / 2..first * group / 2 + data_bytes],
                    );
                    buf[data_bytes..].copy_from_slice(&source[scales_start..scales_start + count]);
                    cu::memcpy_htod_async(self.device.ptr, buf, &rk.stream)?;
                    crate::fp4::unpack(
                        &rk.k,
                        &rk.stream,
                        segment.encoding,
                        self.device.ptr,
                        ptr + (first * group * 2) as u64,
                        count * group,
                    )?;
                    rk.stream.synchronize()?;
                }
            } else {
                for offset in (0..decoded_bytes).step_by(self.host.bytes) {
                    let n = (decoded_bytes - offset).min(self.host.bytes);
                    let buf = &mut self.host.as_mut_slice::<u8>()[..n];
                    buf.copy_from_slice(&source[offset..offset + n]);
                    cu::memcpy_htod_async(ptr + offset as u64, buf, &rk.stream)?;
                    rk.stream.synchronize()?;
                }
            }
        }
        rk.restore_encoder_tail_state(tail);
        Ok(())
    }
}

fn restore_plan(
    full: &[(DevPtr, Segment)],
    clipped: &[(DevPtr, Segment)],
    wire: &[Segment],
) -> Result<Vec<(DevPtr, Segment, usize, usize)>> {
    ensure!(full.len() == clipped.len(), "KV truncation layout mismatch");
    let mut plan = Vec::with_capacity(wire.len());
    let mut segments = wire.iter();
    let mut offset = 0;
    for (&(ptr, source), &(dst, target)) in full.iter().zip(clipped) {
        ensure!(ptr == dst && target.decoded_bytes <= source.decoded_bytes,
                "invalid KV truncation range");
        if source.decoded_bytes == 0 { continue; }
        let &segment = segments.next().ok_or_else(|| anyhow::anyhow!("missing KV segment"))?;
        ensure!(segment.decoded_bytes == source.decoded_bytes, "KV source layout mismatch");
        Segment { decoded_bytes: target.decoded_bytes, ..segment }.stored_bytes()?;
        if target.decoded_bytes > 0 {
            plan.push((ptr, segment, target.decoded_bytes, offset));
        }
        offset += segment.stored_bytes()?;
    }
    ensure!(segments.next().is_none(), "trailing KV segments");
    Ok(plan)
}

pub fn image_bytes(rk: &Rank, n: usize, tail: usize, fp4: bool) -> Result<usize> {
    ensure!(n <= rk.max_seq, "invalid snapshot length");
    encoded_segments(rk, n, n.saturating_sub(tail), false, fp4)
        .iter()
        .try_fold(0usize, |bytes, (_, s)| {
            bytes
                .checked_add(s.stored_bytes()?)
                .ok_or_else(|| anyhow::anyhow!("KV image size overflow"))
        })
}

pub fn segments(rk: &Rank, n: usize, t0: usize) -> Vec<(DevPtr, usize)> {
    segments_for(rk, n, t0, true)
}

fn segments_for(
    rk: &Rank,
    n: usize,
    t0: usize,
    include_decoder_windows: bool,
) -> Vec<(DevPtr, usize)> {
    encoded_segments(rk, n, t0, include_decoder_windows, false)
        .into_iter()
        .map(|(ptr, s)| (ptr, s.decoded_bytes))
        .collect()
}

fn encoded_segments(
    rk: &Rank, n: usize, t0: usize, include_decoder_windows: bool, fp4: bool,
) -> Vec<(DevPtr, Segment)> {
    encoded_layout(rk, n, t0, include_decoder_windows, fp4)
        .into_iter().filter(|(_, s)| s.decoded_bytes > 0).collect()
}

fn encoded_layout(
    rk: &Rank, n: usize, t0: usize, include_decoder_windows: bool, fp4: bool,
) -> Vec<(DevPtr, Segment)> {
    let c = &rk.cfg;
    let dec0 = *c.kv_source_layers.iter().max().unwrap();
    let (hd, ihd) = (c.head_dim, c.index_head_dim);
    let mut v = Vec::new();
    let mut add = |p: DevPtr, off: usize, len: usize, encoding: Encoding| {
        v.push((
            p + off as u64,
            Segment {
                decoded_bytes: len,
                encoding: if fp4 { encoding } else { Encoding::Raw },
            },
        ));
    };
    for (l, lc) in rk.caches.iter().chain(rk.mtp_caches.iter()).enumerate() {
        if include_decoder_windows || l < dec0 {
            add(lc.win, t0 * hd * 2, (n - t0) * hd * 2, Encoding::Raw);
        }
        let ratio = if l < c.n_layers {
            c.compress_ratios[l]
        } else {
            0
        };
        if ratio == 0 {
            continue;
        }
        let rows = n / ratio;
        if let Some(p) = lc.comp {
            add(p, 0, rows * hd * 2, Encoding::Fp4E4m3);
        }
        if let Some(p) = lc.idx_k {
            add(p, 0, rows * ihd * 2, Encoding::Fp4E8m0);
        }
        if let Some((p, s)) = lc.idx_k4 {
            add(p, 0, rows * 64, Encoding::Raw);
            add(s, 0, rows.div_ceil(128) * 512, Encoding::Raw);
        }
        for h in [lc.kv_hist, lc.sc_hist].into_iter().flatten() {
            add(h, t0 * hd * 4, (n - t0) * hd * 4, Encoding::Raw);
        }
    }
    add(rk.cmp_hist, 0, n * 4, Encoding::Raw);
    v.extend(
        rk.encoder_tail_segments()
            .into_iter()
            .map(|(ptr, n)| (ptr, Segment::raw(n))),
    );
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_plan_preserves_source_offsets_when_groups_disappear() {
        for encoding in [Encoding::Raw, Encoding::Fp4E4m3, Encoding::Fp4E8m0] {
            let segment = Segment { decoded_bytes: 256, encoding };
            let full = [(100, Segment::raw(10)), (200, segment), (300, Segment::raw(0)), (400, segment)];
            let wire = [full[0].1, segment, segment];
            for kept in [0, 64, 128, 256] {
                let clipped = [(100, Segment::raw(1)), (200, Segment { decoded_bytes: kept, encoding }),
                    (300, Segment::raw(0)), (400, Segment { decoded_bytes: 64, encoding })];
                let plan = restore_plan(&full, &clipped, &wire).unwrap();
                assert_eq!(plan[0], (100, Segment::raw(10), 1, 0));
                assert_eq!(*plan.last().unwrap(), (400, segment, 64, 10 + segment.stored_bytes().unwrap()));
                assert_eq!(plan.len(), if kept == 0 { 2 } else { 3 });
                if kept != 0 { assert_eq!(plan[1], (200, segment, kept, 10)); }
            }
        }
        let full = [(1, Segment::raw(64))];
        assert!(restore_plan(&full, &[(2, Segment::raw(32))], &[full[0].1]).is_err());
        assert!(restore_plan(&full, &[(1, Segment::raw(65))], &[full[0].1]).is_err());
        assert!(restore_plan(&full, &full, &[]).is_err());
        assert!(restore_plan(&full, &full, &[Segment::raw(1)]).is_err());
    }
}
