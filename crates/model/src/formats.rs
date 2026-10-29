use half::bf16;
use rayon::prelude::*;

pub fn e4m3_to_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    if e == 0xf && (b & 7) == 7 {
        return f32::NAN;
    }
    if e == 0 { s * m / 8.0 * 2f32.powi(-6) } else { s * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

pub fn e8m0_to_f32(b: u8) -> f32 {
    f32::from_bits((b as u32) << 23)
}

pub fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let s: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs().min(448.0);
    let mut best = 0u8;
    let mut bd = f32::INFINITY;
    for c in 0u8..=0x7e {
        let v = e4m3_to_f32(c);
        let d = (v - a).abs();
        if d < bd || (d == bd && c & 1 == 0) {
            bd = d;
            best = c;
        }
    }
    s | best
}

pub fn f32_to_e4m3_fast(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let s: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    if a >= 448.0 {
        return s | 0x7e;
    }
    if a < 0.015625 {
        let q = a * 512.0;
        let f = q.floor();
        let d = q - f;
        let m = if d > 0.5 || (d == 0.5 && (f as u32) & 1 == 1) { f as u32 + 1 } else { f as u32 };
        return s | m as u8;
    }
    let b = a.to_bits();
    let r = (b + 0x7ffff + ((b >> 20) & 1)) >> 20;
    let e = (r >> 3) as i32 - 127 + 7;
    s | ((e as u8) << 3) | (r & 7) as u8
}

pub fn quant_fp8_b32(w: &[u16], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(n % 32 == 0 && k % 32 == 0 && w.len() == n * k);
    let kb = k / 32;
    let mut q = vec![0u8; n * k];
    let mut sc = vec![0u8; (n / 32) * kb];
    q.par_chunks_mut(32 * k).zip(sc.par_chunks_mut(kb)).enumerate().for_each(|(rb, (q, sc))| {
        let f = |r: usize, c: usize| bf16::from_bits(w[(rb * 32 + r) * k + c]).to_f32();
        for b in 0..kb {
            let mut amax = 0f32;
            for r in 0..32 {
                for c in b * 32..b * 32 + 32 {
                    amax = amax.max(f(r, c).abs());
                }
            }
            let s = round_scale_pow2(amax.max(1e-30), 1.0 / 448.0);
            sc[b] = (s.to_bits() >> 23) as u8;
            let inv = 1.0 / s;
            for r in 0..32 {
                for c in b * 32..b * 32 + 32 {
                    q[r * k + c] = f32_to_e4m3_fast(f(r, c) * inv);
                }
            }
        }
    });
    (q, sc)
}

pub fn round_scale_pow2(amax: f32, qmax_inv: f32) -> f32 {
    let bits = (amax * qmax_inv).to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127 + if bits & 0x7fffff != 0 { 1 } else { 0 };
    f32::from_bits(((e + 127) as u32) << 23)
}

fn tile_off(rg: usize, ks: usize, steps: usize, step_bytes: usize) -> usize {
    ((rg / 4) * steps + ks) * step_bytes + (rg % 4) * 512
}

pub fn tile_fp8(w: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert!(n % 64 == 0 && k % 32 == 0 && w.len() == n * k);
    let mut out = vec![0u8; n * k];
    if out.is_empty() { return out; }
    let steps = k / 32;
    out.par_chunks_mut(64 * k).enumerate().for_each(|(q, out)| {
    for rg in 0..4 {
        for ks in 0..steps {
            let base = tile_off(rg, ks, steps, 2048);
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                let kk = [2 * t, 2 * t + 1, 8 + 2 * t, 9 + 2 * t, 16 + 2 * t, 17 + 2 * t, 24 + 2 * t, 25 + 2 * t];
                for (h, r) in [g, g + 8].into_iter().enumerate() {
                    for (j, &c) in kk.iter().enumerate() {
                        out[base + lane * 16 + h * 8 + j] = w[(q * 64 + rg * 16 + r) * k + ks * 32 + c];
                    }
                }
            }
        }
    }
    });
    out
}

pub const FP8_STEP64: usize = 4096 + 16;

pub fn tile_fp8_k64(wt: &[u8], s: &[u8], n: usize, k: usize, kpad: usize) -> Vec<u8> {
    assert!(n % 64 == 0 && k % 32 == 0 && kpad % 64 == 0 && kpad >= k && wt.len() == n * k && s.len() == n / 32 * (k / 32));
    let (s32, steps) = (k / 32, kpad / 64);
    let mut out = vec![0u8; n / 64 * steps * FP8_STEP64];
    for q in 0..n / 64 {
        for j in 0..steps {
            let o = (q * steps + j) * FP8_STEP64;
            for h in 0..2 {
                let ks = 2 * j + h;
                if ks >= s32 {
                    continue;
                }
                let src = (q * s32 + ks) * 2048;
                out[o + h * 2048..o + h * 2048 + 2048].copy_from_slice(&wt[src..src + 2048]);
                for sr in 0..2 {
                    out[o + 4096 + sr * 2 + h] = s[(2 * q + sr) * s32 + ks];
                }
            }
        }
    }
    out
}

pub const FP4_STEP: usize = 2048 + 128;

pub fn tile_fp4(w: &[u8], s: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert!(n % 64 == 0 && k % 64 == 0 && w.len() == n * k / 2 && s.len() == n * k / 32);
    let steps = k / 64;
    let mut out = vec![0u8; (n / 64) * steps * FP4_STEP];
    let byte = |r: usize, kk: usize| w[r * (k / 2) + kk / 2];
    let sk = k / 32;
    for rg in 0..n / 16 {
        for ks in 0..steps {
            let base = tile_off(rg, ks, steps, FP4_STEP);
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                let (r0, r1) = (rg * 16 + g, rg * 16 + g + 8);
                for j in 0..4 {
                    let kb = ks * 64 + j * 16;
                    let o = base + lane * 16 + j * 4;
                    out[o] = byte(r0, kb + 2 * t);
                    out[o + 1] = byte(r0, kb + 8 + 2 * t);
                    out[o + 2] = byte(r1, kb + 2 * t);
                    out[o + 3] = byte(r1, kb + 8 + 2 * t);
                }
            }
            let sbase = ((rg / 4) * steps + ks) * FP4_STEP + 2048 + (rg % 4) * 32;
            for g in 0..8 {
                let (r0, r1) = (rg * 16 + g, rg * 16 + g + 8);
                let o = sbase + g * 4;
                out[o] = s[r0 * sk + ks * 2];
                out[o + 1] = s[r0 * sk + ks * 2 + 1];
                out[o + 2] = s[r1 * sk + ks * 2];
                out[o + 3] = s[r1 * sk + ks * 2 + 1];
            }
        }
    }
    out
}

pub fn tile_bf16(w: &[u16], n: usize, k: usize) -> Vec<u16> {
    assert!(n % 64 == 0 && k % 16 == 0 && w.len() == n * k);
    let mut out = vec![0u16; n * k];
    if out.is_empty() { return out; }
    let steps = k / 16;
    out.par_chunks_mut(64 * k).enumerate().for_each(|(q, out)| {
    for rg in 0..4 {
        for ks in 0..steps {
            let base = tile_off(rg, ks, steps, 2048) / 2;
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                let (r0, r1) = (rg * 16 + g, rg * 16 + g + 8);
                let (r0, r1) = (q * 64 + r0, q * 64 + r1);
                let kb = ks * 16;
                let src = [(r0, 2 * t), (r1, 2 * t), (r0, 8 + 2 * t), (r1, 8 + 2 * t)];
                for (j, (r, c)) in src.into_iter().enumerate() {
                    out[base + lane * 8 + j * 2] = w[r * k + kb + c];
                    out[base + lane * 8 + j * 2 + 1] = w[r * k + kb + c + 1];
                }
            }
        }
    }
    });
    out
}

pub fn rowmajor_fp4(w: &[u8], s: &[u8], n: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * k / 2 + n * k / 32);
    out.extend_from_slice(w);
    out.extend(tile_sf_blocks(s, n, k));
    debug_assert_eq!(out.len(), (n / 64) * (k / 64) * FP4_STEP);
    out
}

pub fn tile_sf_blocks(s: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert!(n % 128 == 0 && k % 128 == 0 && s.len() == n * k / 32);
    let (rt, kt) = (n / 128, k / 128);
    let mut out = vec![0u8; rt * kt * 512];
    for r in 0..n {
        for kb in 0..k / 32 {
            let (t, rr) = (r / 128, r % 128);
            let (kti, kc) = (kb / 4, kb % 4);
            out[(t * kt + kti) * 512 + (rr % 32) * 16 + (rr / 32) * 4 + kc] = s[r * (k / 32) + kb];
        }
    }
    out
}

#[cfg(test)]
mod fp8_quant_tests {
    use super::*;

    #[test]
    fn e4m3_fast_matches_reference_on_every_bf16() {
        for bits in 0u32..=0xffff {
            let x = bf16::from_bits(bits as u16).to_f32();
            for v in [x, f32::from_bits(x.to_bits().wrapping_add(0x8000)), f32::from_bits(x.to_bits().wrapping_add(0x4321))] {
                assert_eq!(f32_to_e4m3_fast(v), f32_to_e4m3(v), "x = {v:e} ({:#x})", v.to_bits());
            }
        }
        for c in 0u8..0x7e {
            let (a, b) = (e4m3_to_f32(c), e4m3_to_f32(c + 1));
            let mid = (a + b) / 2.0;
            assert_eq!(f32_to_e4m3_fast(mid), f32_to_e4m3(mid), "mid {mid:e}");
            assert_eq!(f32_to_e4m3_fast(-mid), f32_to_e4m3(-mid), "mid {mid:e}");
        }
    }

    #[test]
    fn quant_fp8_b32_roundtrip() {
        let (n, k) = (64, 96);
        let mut seed = 0x9e3779b97f4a7c15u64;
        let w: Vec<u16> = (0..n * k)
            .map(|i| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let v = ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * if i % 97 == 0 { 40.0 } else { 0.1 };
                bf16::from_f32(v).to_bits()
            })
            .collect();
        let (q, s) = quant_fp8_b32(&w, n, k);
        for r in 0..n {
            for c in 0..k {
                let x = bf16::from_bits(w[r * k + c]).to_f32();
                let sc = e8m0_to_f32(s[(r / 32) * (k / 32) + c / 32]);
                let y = e4m3_to_f32(q[r * k + c]) * sc;
                assert!((x - y).abs() <= x.abs() / 16.0 + sc * 2f32.powi(-10), "{r},{c}: {x} vs {y}");
            }
        }
    }
}
