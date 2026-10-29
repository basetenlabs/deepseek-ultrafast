use crate::formats::FP4_STEP;

#[derive(Clone, Copy)]
pub struct Fp4Matrix<'a> {
    pub w: &'a [u8],
    pub s: &'a [u8],
    pub n: usize,
    pub k: usize,
}

impl Fp4Matrix<'_> {
    fn validate(self) {
        assert!(self.k % 32 == 0);
        assert_eq!(self.w.len(), self.n * self.k / 2);
        assert_eq!(self.s.len(), self.n * self.k / 32);
    }
}

#[inline]
fn pair(a: &[u8; 32], b: &[u8; 32], out: &mut [u8; 64]) {
    for t in 0..4 {
        for j in 0..4 {
            let i = t * 16 + j * 4;
            out[i] = a[j * 8 + t];
            out[i + 1] = a[j * 8 + 4 + t];
            out[i + 2] = b[j * 8 + t];
            out[i + 3] = b[j * 8 + 4 + t];
        }
    }
}

#[inline]
fn group(src: Fp4Matrix<'_>, row: usize, col: usize, valid: usize, weights: &mut [u8], scales: &mut [u8]) {
    for g in 0..8 {
        let (r0, r1) = (row + g, row + g + 8);
        let (a, b) = (r0 * (src.k / 2) + col / 2, r1 * (src.k / 2) + col / 2);
        let out = (&mut weights[g * 64..g * 64 + 64]).try_into().unwrap();
        if valid == 64 {
            pair(src.w[a..a + 32].try_into().unwrap(), src.w[b..b + 32].try_into().unwrap(), out);
        } else {
            let (mut x, mut y) = ([0u8; 32], [0u8; 32]);
            x[..valid / 2].copy_from_slice(&src.w[a..a + valid / 2]);
            y[..valid / 2].copy_from_slice(&src.w[b..b + valid / 2]);
            pair(&x, &y, out);
        }
        let (a, b) = (r0 * (src.k / 32) + col / 32, r1 * (src.k / 32) + col / 32);
        scales[g * 4..g * 4 + 4].copy_from_slice(&[
            src.s[a], if valid == 64 { src.s[a + 1] } else { 0 },
            src.s[b], if valid == 64 { src.s[b + 1] } else { 0 },
        ]);
    }
}

pub fn expert_shard(gate: Fp4Matrix<'_>, up: Fp4Matrix<'_>, down: Fp4Matrix<'_>, rank: usize, tp: usize,
                    out13: &mut [u8], out2: &mut [u8]) {
    expert_shard_impl::<false>(gate, up, down, rank, tp, out13, out2);
}

pub fn expert_shard_rowmajor(gate: Fp4Matrix<'_>, up: Fp4Matrix<'_>, down: Fp4Matrix<'_>, rank: usize, tp: usize,
                             out13: &mut [u8], out2: &mut [u8]) {
    expert_shard_impl::<true>(gate, up, down, rank, tp, out13, out2);
}

pub fn w2_rowmajor_bytes(dim: usize, inter: usize) -> usize {
    dim * inter.div_ceil(128) * 128 / 2 + (dim / 128) * inter.div_ceil(128) * 512
}

pub fn sgl_expert_bytes(dim: usize, inter: usize) -> (usize, usize) {
    let pad = inter.div_ceil(128) * 128;
    (2 * pad * dim / 2 + (2 * pad / 128) * (dim / 128) * 512, dim * sgl_w2_row_bytes(inter) + (dim / 128) * (pad / 128) * 512)
}

pub fn sgl_w2_row_bytes(inter: usize) -> usize {
    inter.div_ceil(64) * 64 / 2
}

pub fn sgl_unshuffled_row(n: usize) -> usize {
    (n & !31) | ((n & 7) << 2) | ((n >> 3) & 3)
}

pub fn sgl_sf_index(row: usize, kb: usize, kt: usize) -> usize {
    (row / 128 * kt + kb / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + kb % 4
}

pub fn expert_shard_sgl(gate: Fp4Matrix<'_>, up: Fp4Matrix<'_>, down: Fp4Matrix<'_>, rank: usize, tp: usize,
                        out13: &mut [u8], out2: &mut [u8]) {
    for src in [gate, up, down] { src.validate(); }
    assert!(tp > 0 && rank < tp && gate.n % tp == 0);
    let (dim, inter) = (gate.k, gate.n / tp);
    assert!(dim % 128 == 0 && inter % 32 == 0);
    assert_eq!((up.n, up.k, down.n, down.k), (gate.n, dim, dim, gate.n));
    let pad = inter.div_ceil(128) * 128;
    let (b13, b2) = sgl_expert_bytes(dim, inter);
    assert_eq!((out13.len(), out2.len()), (b13, b2));
    let (kb13, kt13) = (dim / 32, dim / 128);
    {
        let (w, s) = out13.split_at_mut(2 * pad * dim / 2);
        s.fill(127);
        for n in 0..2 * pad {
            let o = sgl_unshuffled_row(n);
            let (j, is_gate) = (o / 2, o % 2 == 1);
            let dst = &mut w[n * dim / 2..(n + 1) * dim / 2];
            if j >= inter {
                dst.fill(0);
                continue;
            }
            let src = if is_gate { gate } else { up };
            let sr = rank * inter + j;
            dst.copy_from_slice(&src.w[sr * dim / 2..(sr + 1) * dim / 2]);
            for kb in 0..kb13 {
                s[sgl_sf_index(n, kb, kt13)] = src.s[sr * kb13 + kb];
            }
        }
    }
    {
        let rb = sgl_w2_row_bytes(inter);
        let (w, s) = out2.split_at_mut(dim * rb);
        s.fill(127);
        w.fill(0);
        let (dk, dkb, kt2) = (down.k / 2, down.k / 32, pad / 128);
        for n in 0..dim {
            let o = sgl_unshuffled_row(n);
            w[n * rb..n * rb + inter / 2].copy_from_slice(&down.w[o * dk + rank * inter / 2..o * dk + (rank + 1) * inter / 2]);
            for kb in 0..inter / 32 {
                s[sgl_sf_index(n, kb, kt2)] = down.s[o * dkb + rank * inter / 32 + kb];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn shared_shard_sgl(w1: &[u8], s1: &[u8], w3: &[u8], s3: &[u8], w2: &[u8], s2: &[u8], n_inter: usize, dim: usize,
                        rank: usize, tp: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(n_inter % tp == 0 && dim % 128 == 0);
    let inter = n_inter / tp;
    assert!(inter % 32 == 0);
    let pad = inter.div_ceil(128) * 128;
    let (kb13, kt13) = (dim / 32, dim / 128);
    let mut o13 = vec![0u8; 2 * pad * dim + (2 * pad / 128) * kt13 * 512];
    {
        let (w, s) = o13.split_at_mut(2 * pad * dim);
        s.fill(127);
        for n in 0..2 * pad {
            let o = sgl_unshuffled_row(n);
            let (j, is_gate) = (o / 2, o % 2 == 1);
            if j >= inter {
                continue;
            }
            let (src, ss) = if is_gate { (w1, s1) } else { (w3, s3) };
            let sr = rank * inter + j;
            w[n * dim..(n + 1) * dim].copy_from_slice(&src[sr * dim..(sr + 1) * dim]);
            for kb in 0..kb13 {
                s[sgl_sf_index(n, kb, kt13)] = ss[(sr / 32) * kb13 + kb];
            }
        }
    }
    let kt2 = pad / 128;
    let mut o2 = vec![0u8; dim * pad + (dim / 128) * kt2 * 512];
    {
        let (w, s) = o2.split_at_mut(dim * pad);
        s.fill(127);
        let (dkb, c0) = (n_inter / 32, rank * inter);
        for n in 0..dim {
            let o = sgl_unshuffled_row(n);
            w[n * pad..n * pad + inter].copy_from_slice(&w2[o * n_inter + c0..o * n_inter + c0 + inter]);
            for kb in 0..inter / 32 {
                s[sgl_sf_index(n, kb, kt2)] = s2[(o / 32) * dkb + c0 / 32 + kb];
            }
        }
    }
    (o13, o2)
}

pub fn expert_shard_tc(gate: Fp4Matrix<'_>, up: Fp4Matrix<'_>, down: Fp4Matrix<'_>, rank: usize, tp: usize,
                       out13: &mut [u8], out2: &mut [u8]) {
    for src in [gate, up, down] { src.validate(); }
    assert!(tp > 0 && rank < tp && gate.n % tp == 0);
    let (dim, inter) = (gate.k, gate.n / tp);
    assert!(dim % 128 == 0 && (2 * inter) % 128 == 0 && inter % 32 == 0);
    assert_eq!((up.n, up.k, down.n, down.k), (gate.n, dim, dim, gate.n));
    assert_eq!(out13.len(), 2 * inter * dim / 2 + (2 * inter / 128) * (dim / 128) * 512);
    assert_eq!(out2.len(), w2_rowmajor_bytes(dim, inter));
    {
        let (w, s) = out13.split_at_mut(inter * dim);
        let kt = dim / 128;
        for row in 0..2 * inter {
            let src = if row / 32 % 2 == 0 { gate } else { up };
            let src_row = rank * inter + row / 64 * 32 + row % 32;
            w[row * dim / 2..(row + 1) * dim / 2].copy_from_slice(&src.w[src_row * dim / 2..(src_row + 1) * dim / 2]);
            for kb in 0..dim / 32 {
                s[(row / 128 * kt + kb / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + kb % 4] = src.s[src_row * (dim / 32) + kb];
            }
        }
    }
    let kp = inter.div_ceil(128) * 128;
    let (w, s) = out2.split_at_mut(dim * kp / 2);
    let kt = inter.div_ceil(128);
    s.fill(127);
    w.fill(0);
    let (dk, dkb) = (down.k / 2, down.k / 32);
    for row in 0..dim {
        w[row * kp / 2..row * kp / 2 + inter / 2].copy_from_slice(&down.w[row * dk + rank * inter / 2..row * dk + (rank + 1) * inter / 2]);
        for kb in 0..inter / 32 {
            s[(row / 128 * kt + kb / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + kb % 4] = down.s[row * dkb + rank * inter / 32 + kb];
        }
    }
}

fn expert_shard_impl<const RM: bool>(gate: Fp4Matrix<'_>, up: Fp4Matrix<'_>, down: Fp4Matrix<'_>, rank: usize, tp: usize,
                                    out13: &mut [u8], out2: &mut [u8]) {
    for src in [gate, up, down] { src.validate(); }
    assert!(tp > 0 && rank < tp && gate.n % tp == 0);
    let (dim, inter) = (gate.k, gate.n / tp);
    assert!(dim % 64 == 0 && inter > 0 && inter % 32 == 0);
    assert_eq!((up.n, up.k, down.n, down.k), (gate.n, dim, dim, gate.n));
    let kp = inter.div_ceil(64) * 64;
    assert_eq!(out13.len(), 2 * inter / 64 * (dim / 64) * FP4_STEP);
    assert_eq!(out2.len(), dim / 64 * (kp / 64) * FP4_STEP);
    if RM {
        assert!(2 * inter % 128 == 0 && dim % 128 == 0);
        let (w, s) = out13.split_at_mut(inter * dim);
        let kt = dim / 128;
        for row in 0..2 * inter {
            let src = if row / 32 % 2 == 0 { gate } else { up };
            let src_row = rank * inter + row / 64 * 32 + row % 32;
            w[row * dim / 2..(row + 1) * dim / 2]
                .copy_from_slice(&src.w[src_row * dim / 2..(src_row + 1) * dim / 2]);
            for kb in 0..dim / 32 {
                let dst = (row / 128 * kt + kb / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + kb % 4;
                s[dst] = src.s[src_row * (dim / 32) + kb];
            }
        }
    } else {
    for (quad, dst) in out13.chunks_exact_mut(dim / 64 * FP4_STEP).enumerate() {
        for (step, dst) in dst.chunks_exact_mut(FP4_STEP).enumerate() {
            let (w, s) = dst.split_at_mut(2048);
            for rg in 0..4 {
                let src = if rg < 2 { gate } else { up };
                group(src, rank * inter + quad * 32 + (rg % 2) * 16, step * 64, 64,
                      &mut w[rg * 512..(rg + 1) * 512], &mut s[rg * 32..(rg + 1) * 32]);
            }
        }
    }
    }
    for (quad, dst) in out2.chunks_exact_mut(kp / 64 * FP4_STEP).enumerate() {
        for (step, dst) in dst.chunks_exact_mut(FP4_STEP).enumerate() {
            let (w, s) = dst.split_at_mut(2048);
            for rg in 0..4 {
                group(down, quad * 64 + rg * 16, rank * inter + step * 64, (inter - step * 64).min(64),
                      &mut w[rg * 512..(rg + 1) * 512], &mut s[rg * 32..(rg + 1) * 32]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::tile_fp4;

    #[test]
    fn fused_shards_equal_original_layout_including_tp8_padding() {
        let mut state = 17u32;
        for (tp, inter, dim) in [(1, 32, 64), (1, 64, 128), (2, 96, 128), (2, 128, 512), (4, 576, 256), (8, 288, 128)] {
            let full = tp * inter;
            let mut bytes = |n| -> Vec<u8> { (0..n).map(|_| { state = state.wrapping_mul(1664525).wrapping_add(1013904223); (state >> 24) as u8 }).collect() };
            let w = [bytes(full * dim / 2), bytes(full * dim / 2), bytes(full * dim / 2)];
            let s = [bytes(full * dim / 32), bytes(full * dim / 32), bytes(full * dim / 32)];
            let mats = [Fp4Matrix { w: &w[0], s: &s[0], n: full, k: dim },
                        Fp4Matrix { w: &w[1], s: &s[1], n: full, k: dim },
                        Fp4Matrix { w: &w[2], s: &s[2], n: dim, k: full }];
            for rank in 0..tp {
                let (mut a, mut b) = (vec![], vec![]);
                for block in 0..inter / 32 {
                    let r = rank * inter + block * 32;
                    for m in &mats[..2] {
                        a.extend_from_slice(&m.w[r * dim / 2..(r + 32) * dim / 2]);
                        b.extend_from_slice(&m.s[r * dim / 32..(r + 32) * dim / 32]);
                    }
                }
                let expected13 = tile_fp4(&a, &b, 2 * inter, dim);
                let expected_rm = (2 * inter % 128 == 0 && dim % 128 == 0)
                    .then(|| crate::formats::rowmajor_fp4(&a, &b, 2 * inter, dim));
                let kp = inter.div_ceil(64) * 64;
                let (mut a, mut b) = (vec![0; dim * kp / 2], vec![0; dim * kp / 32]);
                for row in 0..dim {
                    a[row * kp / 2..row * kp / 2 + inter / 2].copy_from_slice(&w[2][(row * full + rank * inter) / 2..(row * full + (rank + 1) * inter) / 2]);
                    b[row * kp / 32..row * kp / 32 + inter / 32].copy_from_slice(&s[2][(row * full + rank * inter) / 32..(row * full + (rank + 1) * inter) / 32]);
                }
                let expected2 = tile_fp4(&a, &b, dim, kp);
                let (mut got13, mut got2) = (vec![0xa5; expected13.len()], vec![0xa5; expected2.len()]);
                expert_shard(mats[0], mats[1], mats[2], rank, tp, &mut got13, &mut got2);
                assert_eq!(got13, expected13, "TP{tp} rank{rank} gate/up");
                assert_eq!(got2, expected2, "TP{tp} rank{rank} down");
                if let Some(expected_rm) = expected_rm {
                    got13.fill(0xa5);
                    got2.fill(0xa5);
                    expert_shard_rowmajor(mats[0], mats[1], mats[2], rank, tp, &mut got13, &mut got2);
                    assert_eq!(got13, expected_rm, "TP{tp} rank{rank} row-major gate/up");
                    assert_eq!(got2, expected2, "TP{tp} rank{rank} unchanged down");
                }
            }
        }
    }

    #[test]
    fn sgl_w2_compact_rows() {
        assert_eq!(sgl_w2_row_bytes(576), 288);
        assert_eq!(sgl_w2_row_bytes(288), 160);
        for inter in [576usize, 288] {
            let (pad, rb) = (inter.div_ceil(128) * 128, sgl_w2_row_bytes(inter));
            assert!(rb % 32 == 0 && rb >= inter / 2 && rb <= pad / 2, "inter {inter}");
        }
        let (dim, tp, inter) = (128usize, 8usize, 288usize);
        let full = tp * inter;
        let mut state = 7u32;
        let mut bytes = |n| -> Vec<u8> { (0..n).map(|_| { state = state.wrapping_mul(1664525).wrapping_add(1013904223); (state >> 24) as u8 }).collect() };
        let w = [bytes(full * dim / 2), bytes(full * dim / 2), bytes(full * dim / 2)];
        let s = [bytes(full * dim / 32), bytes(full * dim / 32), bytes(full * dim / 32)];
        let mats = [Fp4Matrix { w: &w[0], s: &s[0], n: full, k: dim },
                    Fp4Matrix { w: &w[1], s: &s[1], n: full, k: dim },
                    Fp4Matrix { w: &w[2], s: &s[2], n: dim, k: full }];
        let (b13, b2) = sgl_expert_bytes(dim, inter);
        let pad = inter.div_ceil(128) * 128;
        assert_eq!(b2, dim * 160 + (dim / 128) * (pad / 128) * 512);
        let rank = 3;
        let (mut o13, mut o2) = (vec![0xa5; b13], vec![0xa5; b2]);
        expert_shard_sgl(mats[0], mats[1], mats[2], rank, tp, &mut o13, &mut o2);
        let rb = sgl_w2_row_bytes(inter);
        for n in 0..dim {
            let o = sgl_unshuffled_row(n);
            assert_eq!(&o2[n * rb..n * rb + inter / 2], &w[2][o * full / 2 + rank * inter / 2..o * full / 2 + (rank + 1) * inter / 2], "row {n}");
            assert!(o2[n * rb + inter / 2..(n + 1) * rb].iter().all(|&b| b == 0), "row {n}: the gap is zero");
            for kb in 0..inter / 32 {
                assert_eq!(o2[dim * rb + sgl_sf_index(n, kb, pad / 128)], s[2][o * full / 32 + rank * inter / 32 + kb]);
            }
        }
    }
}
