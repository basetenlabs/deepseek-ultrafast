use crate::ops::{WFmt, Weight};
use crate::weight_staging::ExpertStaging;
use anyhow::{Result, ensure};
use deepseek_ultrafast_cuda::{Buffer, DevPtr};
use deepseek_ultrafast_model::config::Config;
use deepseek_ultrafast_model::formats::{e4m3_to_f32, e8m0_to_f32, tile_bf16, tile_fp8, tile_fp8_k64, tile_sf_blocks, FP4_STEP};
use half::bf16;
use deepseek_ultrafast_model::st::Checkpoint;
use deepseek_ultrafast_model::packing::{Fp4Matrix, expert_shard, expert_shard_sgl, expert_shard_tc, sgl_expert_bytes, shared_shard_sgl, w2_rowmajor_bytes};
use rayon::prelude::*;
use std::cell::RefCell;
use std::borrow::Cow;
use std::collections::HashMap;

#[derive(Default)]
pub struct Arena {
    pub bufs: Vec<Buffer>,
    pub bytes: usize,
}

impl Arena {
    pub(crate) fn alloc(&mut self, bytes: usize) -> Result<DevPtr> {
        let b = Buffer::alloc(bytes)?;
        let p = b.ptr;
        self.bytes += b.bytes;
        self.bufs.push(b);
        Ok(p)
    }
    pub fn put<T: Copy>(&mut self, data: &[T]) -> Result<DevPtr> {
        let b = Buffer::from_host(data)?;
        let p = b.ptr;
        self.bytes += b.bytes;
        self.bufs.push(b);
        Ok(p)
    }
    fn put_padded(&mut self, data: &[u8], bytes: usize, pad: u8) -> Result<DevPtr> {
        ensure!(data.len() <= bytes);
        let b = Buffer::alloc(bytes)?;
        b.upload(data)?;
        if data.len() < bytes {
            deepseek_ultrafast_cuda::check(unsafe {
                deepseek_ultrafast_cuda::sys::cuMemsetD8_v2(b.ptr + data.len() as u64, pad, bytes - data.len())
            }, "weight padding")?;
        }
        let p = b.ptr;
        self.bytes += b.bytes;
        self.bufs.push(b);
        Ok(p)
    }
    pub fn zeroed(&mut self, bytes: usize) -> Result<DevPtr> {
        let b = Buffer::zeroed(bytes)?;
        let p = b.ptr;
        self.bytes += b.bytes;
        self.bufs.push(b);
        Ok(p)
    }
}

pub struct Compressor {
    pub wkv: Weight,
    pub wgate: Option<Weight>,
    pub wkvg: Option<Weight>,
    pub norm: DevPtr,
}

pub struct Indexer {
    pub wq_b: Weight,
    pub wq_b_cat: Weight,
    pub weights_proj: Weight,
    pub wk: Option<Weight>,
    pub k_norm: DevPtr,
}

pub struct Engram {
    pub table: DevPtr,
    pub scales: DevPtr,
    pub shard_rows: usize,
    pub wkv: Weight,
    pub qk: DevPtr,
}

pub struct Experts {
    pub n: usize,
    pub w13: DevPtr,
    pub w2: DevPtr,
    pub w13_stride: usize,
    pub w2_stride: usize,
    pub inter: usize,
    pub kpad: usize,
    pub rm: bool,
    pub sgl: bool,
}

pub struct Layer {
    pub id: usize,
    pub attn_norm: DevPtr,
    pub ffn_norm: DevPtr,
    pub hc_attn: (DevPtr, DevPtr, DevPtr),
    pub hc_ffn: (DevPtr, DevPtr, DevPtr),
    pub wq_a: Weight,
    pub wqkv: Weight,
    pub q_norm: DevPtr,
    pub wq_b: Weight,
    pub wkv: Weight,
    pub kv_norm: DevPtr,
    pub wo_a: Weight,
    pub wo_b: Weight,
    pub attn_sink: DevPtr,
    pub compressor: Option<Compressor>,
    pub indexer: Option<Indexer>,
    pub gate: Weight,
    pub gate_bias: DevPtr,
    pub experts: Experts,
    pub shared_w13: Weight,
    pub shared_w2: Weight,
    pub shared_k64: (DevPtr, DevPtr),
    pub shared_sgl: Option<(DevPtr, DevPtr)>,
    pub engram: Option<Engram>,
    pub main_proj: Option<Weight>,
    pub main_norm: Option<DevPtr>,
    pub final_norm: Option<DevPtr>,
    pub markov_embed: Option<DevPtr>,
    pub markov_head: Option<Weight>,
    pub confidence: Option<DevPtr>,
}

pub struct HostFp8 {
    pub tiles: Vec<u8>,
    pub scales: Vec<u8>,
    pub n: usize,
    pub k: usize,
}

pub struct Weights {
    pub embed: DevPtr,
    pub head: Weight,
    pub head_fp8: Option<Weight>,
    pub head_fp8_host: Option<HostFp8>,
    pub norm: DevPtr,
    pub layers: Vec<Layer>,
    pub mtp: Vec<Layer>,
    pub mtp_wkv: Option<Weight>,
    pub tc: HashMap<DevPtr, TcWeight>,
}

#[derive(Clone, Copy, Debug)]
pub struct TcWeight {
    pub w: DevPtr,
    pub sf: DevPtr,
    pub tmap: DevPtr,
    pub tmap128: DevPtr,
    pub nt: usize,
    pub bf16: bool,
    pub kp: usize,
}

pub struct Loader<'a> {
    pub ck: &'a Checkpoint,
    pub cfg: &'a Config,
    pub rank: usize,
    pub tp: usize,
    pub tc: RefCell<HashMap<DevPtr, TcWeight>>,
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn rows(b: &[u8], cols: usize, esz: usize, r0: usize, r1: usize) -> &[u8] {
    &b[r0 * cols * esz..r1 * cols * esz]
}

fn cols(b: &[u8], nrows: usize, ncols: usize, c0: usize, c1: usize) -> Vec<u8> {
    let w = c1 - c0;
    let mut out = vec![0u8; nrows * w];
    out.par_chunks_mut(w).enumerate().for_each(|(r, o)| o.copy_from_slice(&b[r * ncols + c0..r * ncols + c1]));
    out
}

impl Loader<'_> {
    fn tc_copy(&self, a: &mut Arena, key: DevPtr, w: &[u8], s: &[u8], n: usize, k: usize) -> Result<()> {
        if n % 64 != 0 || k % 32 != 0 {
            return Ok(());
        }
        let nt = if n % 256 == 0 { 256 } else if n % 128 == 0 { 128 } else { 64 };
        let kp = k.div_ceil(128) * 128;
        let (kb, kbp) = (k / 32, kp / 32);
        let s_rows: Vec<u8> = (0..n).flat_map(|r| (0..kbp).map(move |b| if b < kb { s[(r / 32) * kb + b] } else { 127 })).collect();
        let sf = if nt == 64 {
            let mut v = Vec::new();
            for t in 0..n / 64 {
                let mut rows = s_rows[t * 64 * kbp..(t + 1) * 64 * kbp].to_vec();
                rows.extend(std::iter::repeat(127u8).take(64 * kbp));
                v.extend(tile_sf_blocks(&rows, 128, kp));
            }
            v
        } else {
            tile_sf_blocks(&s_rows, n, kp)
        };
        let wpad: Cow<'_, [u8]> = if kp == k { Cow::Borrowed(w) } else {
            Cow::Owned(w.chunks(k).flat_map(|row| row.iter().copied().chain(std::iter::repeat(0u8).take(kp - k))).collect())
        };
        let wp = a.put(&wpad)?;
        let tmap = deepseek_ultrafast_cuda::tensor_map_u8_sw128(wp, kp as u64, n as u64, kp as u64, nt as u32)?;
        let tmap = a.put(&tmap)?;
        let tmap128 = match nt {
            256 => a.put(&deepseek_ultrafast_cuda::tensor_map_u8_sw128(wp, kp as u64, n as u64, kp as u64, 128)?)?,
            128 => tmap,
            _ => 0,
        };
        let t = TcWeight { w: wp, sf: a.put(&sf)?, tmap, tmap128, nt, bf16: false, kp };
        self.tc.borrow_mut().insert(key, t);
        Ok(())
    }

    fn tc_copy_bf16(&self, a: &mut Arena, key: DevPtr, w: &[u16], n: usize, k: usize) -> Result<()> {
        if n % 64 != 0 || k % 64 != 0 || n > 8192 {
            return Ok(());
        }
        let nt = if n % 256 == 0 { 256 } else if n % 128 == 0 { 128 } else { 64 };
        let wp = a.put(w)?;
        let tmap = deepseek_ultrafast_cuda::tensor_map_bf16_sw128(wp, k as u64, n as u64, (k * 2) as u64, nt as u32)?;
        if crate::ops::tcgemv2_on() {
            let tmap = a.put(&tmap)?;
            let tmap128 = match nt {
                256 => a.put(&deepseek_ultrafast_cuda::tensor_map_bf16_sw128(wp, k as u64, n as u64, (k * 2) as u64, 128)?)?,
                128 => tmap,
                _ => 0,
            };
            self.tc.borrow_mut().insert(key, TcWeight { w: wp, sf: 0, tmap, tmap128, nt, bf16: true, kp: k });
            return Ok(());
        }
        self.tc.borrow_mut().insert(key, TcWeight { w: wp, sf: 0, tmap: a.put(&tmap)?, tmap128: 0, nt, bf16: true, kp: k });
        Ok(())
    }

    fn get(&self, name: &str) -> Result<(&[u8], Vec<usize>, String)> {
        let t = self.ck.info(name)?;
        Ok((self.ck.bytes(name)?, t.shape.clone(), t.dtype.clone()))
    }

    fn vec_bf16(&self, a: &mut Arena, name: &str) -> Result<DevPtr> {
        let (b, _, d) = self.get(name)?;
        ensure!(d == "BF16", "{name} is {d}");
        a.put(b)
    }

    fn vec_f32(&self, a: &mut Arena, name: &str) -> Result<DevPtr> {
        let (b, _, d) = self.get(name)?;
        ensure!(d == "F32", "{name} is {d}");
        a.put(b)
    }

    fn fp8(&self, a: &mut Arena, name: &str, r: Option<(usize, usize)>, fmt: WFmt) -> Result<Weight> {
        let (w, shape, d) = self.get(&format!("{name}.weight"))?;
        ensure!(d == "F8_E4M3", "{name}: {d}");
        let (s, sshape, _) = self.get(&format!("{name}.scale"))?;
        let (n, k) = (shape[0], shape[1]);
        let (r0, r1) = r.unwrap_or((0, n));
        ensure!(r0 % 64 == 0 && r1 % 64 == 0);
        let wr = rows(w, k, 1, r0, r1);
        let wt = tile_fp8(&wr, r1 - r0, k);
        let st = rows(s, sshape[1], 1, r0 / 32, r1 / 32);
        let out = Weight { fmt, w: a.put(&wt)?, s: a.put(&st)?, n: r1 - r0, k };
        if fmt == WFmt::Fp8 {
            self.tc_copy(a, out.w, &wr, &st, r1 - r0, k)?;
        } else if fmt == WFmt::Fp8Bf16Act {
            let (n, ks) = (r1 - r0, k / 32);
            let wd: Vec<u16> = (0..n * k).into_par_iter().map(|i| {
                let (r, c) = (i / k, i % k);
                bf16::from_f32(e4m3_to_f32(wr[i]) * e8m0_to_f32(st[(r / 32) * ks + c / 32])).to_bits()
            }).collect();
            self.tc_copy_bf16(a, out.w, &wd, n, k)?;
        }
        Ok(out)
    }

    fn fp8_merged(&self, a: &mut Arena, names: &[&str]) -> Result<(Weight, Vec<Weight>)> {
        let (mut wt, mut st, mut dims, mut wr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for name in names {
            let (w, shape, d) = self.get(&format!("{name}.weight"))?;
            ensure!(d == "F8_E4M3", "{name}: {d}");
            let (s, _, _) = self.get(&format!("{name}.scale"))?;
            let (n, k) = (shape[0], shape[1]);
            ensure!(n % 64 == 0 && dims.iter().all(|&(_, kk)| kk == k));
            wt.extend_from_slice(&tile_fp8(w, n, k));
            wr.extend_from_slice(w);
            st.extend_from_slice(s);
            dims.push((n, k));
        }
        let (wp, sp) = (a.put(&wt)?, a.put(&st)?);
        let k = dims[0].1;
        let ntot: usize = dims.iter().map(|d| d.0).sum();
        self.tc_copy(a, wp, &wr, &st, ntot, k)?;
        let (mut views, mut row) = (Vec::new(), 0usize);
        for &(n, _) in &dims {
            views.push(Weight { fmt: WFmt::Fp8, w: wp + (row * k) as u64, s: sp + (row / 32 * k / 32) as u64, n, k });
            row += n;
        }
        Ok((Weight { fmt: WFmt::Fp8, w: wp, s: sp, n: row, k }, views))
    }

    fn fp8_cat(&self, a: &mut Arena, parts: &[(&str, Option<(usize, usize)>)]) -> Result<Weight> {
        let (mut wt, mut st, mut n, mut k0) = (Vec::new(), Vec::new(), 0usize, 0usize);
        let tg2 = crate::ops::tcgemv2_on();
        let mut wr = Vec::new();
        for &(name, r) in parts {
            let (w, shape, d) = self.get(&format!("{name}.weight"))?;
            ensure!(d == "F8_E4M3", "{name}: {d}");
            let (s, sshape, _) = self.get(&format!("{name}.scale"))?;
            let k = shape[1];
            ensure!(k0 == 0 || k == k0);
            k0 = k;
            let (r0, r1) = r.unwrap_or((0, shape[0]));
            ensure!(r0 % 64 == 0 && r1 % 64 == 0);
            wt.extend_from_slice(&tile_fp8(&rows(w, k, 1, r0, r1), r1 - r0, k));
            st.extend_from_slice(&rows(s, sshape[1], 1, r0 / 32, r1 / 32));
            if tg2 {
                wr.extend_from_slice(rows(w, k, 1, r0, r1));
            }
            n += r1 - r0;
        }
        let out = Weight { fmt: WFmt::Fp8, w: a.put(&wt)?, s: a.put(&st)?, n, k: k0 };
        if tg2 {
            self.tc_copy(a, out.w, &wr, &st, n, k0)?;
        }
        Ok(out)
    }

    fn fp8_cols(&self, a: &mut Arena, name: &str, c0: usize, c1: usize) -> Result<Weight> {
        let (w, shape, _) = self.get(&format!("{name}.weight"))?;
        let (s, sshape, _) = self.get(&format!("{name}.scale"))?;
        let (n, k) = (shape[0], shape[1]);
        let ws = cols(w, n, k, c0, c1);
        let ss = cols(s, sshape[0], sshape[1], c0 / 32, c1 / 32);
        let wt = tile_fp8(&ws, n, c1 - c0);
        let out = Weight { fmt: WFmt::Fp8, w: a.put(&wt)?, s: a.put(&ss)?, n, k: c1 - c0 };
        self.tc_copy(a, out.w, &ws, &ss, n, c1 - c0)?;
        Ok(out)
    }

    fn bf16(&self, a: &mut Arena, name: &str, r: Option<(usize, usize)>) -> Result<Weight> {
        let (w, shape, d) = self.get(name)?;
        ensure!(d == "BF16", "{name}: {d}");
        let (n, k) = (shape[0], shape[1]);
        let (r0, r1) = r.unwrap_or((0, n));
        let mut v = u16s(&rows(w, k, 2, r0, r1));
        let np = (r1 - r0).div_ceil(64) * 64;
        v.resize(np * k, 0);
        let wt = tile_bf16(&v, np, k);
        let out = Weight { fmt: WFmt::Bf16, w: a.put(&wt)?, s: 0, n: np, k };
        self.tc_copy_bf16(a, out.w, &v, np, k)?;
        Ok(out)
    }

    fn fp8_quant(&self, name: &str, r: Option<(usize, usize)>) -> Result<HostFp8> {
        let (w, shape, d) = self.get(name)?;
        ensure!(d == "BF16", "{name}: {d}");
        let (n, k) = (shape[0], shape[1]);
        ensure!(k % 32 == 0, "{name}: k {k}");
        let (r0, r1) = r.unwrap_or((0, n));
        let mut v = u16s(&rows(w, k, 2, r0, r1));
        let np = (r1 - r0).div_ceil(64) * 64;
        v.resize(np * k, 0);
        let (q, sc) = deepseek_ultrafast_model::formats::quant_fp8_b32(&v, np, k);
        Ok(HostFp8 { tiles: tile_fp8(&q, np, k), scales: sc, n: np, k })
    }

    fn bf16_cat(&self, a: &mut Arena, names: &[&str]) -> Result<Weight> {
        let (mut v, mut n, mut k0) = (Vec::new(), 0usize, 0usize);
        for name in names {
            let (w, shape, d) = self.get(name)?;
            ensure!(d == "BF16" && shape[0] % 64 == 0, "{name}: {d}");
            ensure!(k0 == 0 || shape[1] == k0);
            k0 = shape[1];
            v.extend(u16s(&rows(w, k0, 2, 0, shape[0])));
            n += shape[0];
        }
        Ok(Weight { fmt: WFmt::Bf16, w: a.put(&tile_bf16(&v, n, k0))?, s: 0, n, k: k0 })
    }

    fn layer(&self, a: &mut Arena, staging: &mut ExpertStaging, prefix: &str, id: usize) -> Result<Layer> {
        let c = self.cfg;
        let (r, tp) = (self.rank, self.tp);
        let p = |s: &str| format!("{prefix}.{s}");
        let heads = c.n_heads / tp;
        let hd = c.head_dim;
        let groups = c.o_groups / tp;
        let ol = c.o_lora_rank;
        let is_mtp = id >= c.n_layers;

        let wq_b = self.fp8(a, &p("attn.wq_b"), Some((r * heads * hd, (r + 1) * heads * hd)), WFmt::Fp8)?;
        let wo_a = self.fp8(a, &p("attn.wo_a"), Some((r * groups * ol, (r + 1) * groups * ol)), WFmt::Fp8Bf16Act)?;
        let in_b = c.o_groups * ol / tp;
        let wo_b = self.fp8_cols(a, &p("attn.wo_b"), r * in_b, (r + 1) * in_b)?;
        let sink = self.ck.f32s(&p("attn.attn_sink"))?;
        let attn_sink = a.put(&sink[r * heads..(r + 1) * heads])?;

        let compressor = if c.is_kv_source(id) {
            Some(Compressor {
                wkv: self.bf16(a, &p("attn.compressor.wkv.weight"), None)?,
                wgate: if c.compress_ratios[id] > 1 { Some(self.bf16(a, &p("attn.compressor.wgate.weight"), None)?) } else { None },
                wkvg: if c.compress_ratios[id] > 1 { Some(self.bf16_cat(a, &[&p("attn.compressor.wkv.weight"), &p("attn.compressor.wgate.weight")])?) } else { None },
                norm: self.vec_bf16(a, &p("attn.compressor.norm.weight"))?,
            })
        } else {
            None
        };
        let indexer = if c.is_index_source(id) {
            Some(Indexer {
                wq_b: self.fp8(a, &p("attn.indexer.wq_b"), None, WFmt::Fp8)?,
                wq_b_cat: self.fp8_cat(a, &[(&p("attn.wq_b"), Some((r * heads * hd, (r + 1) * heads * hd))), (&p("attn.indexer.wq_b"), None)])?,
                weights_proj: self.bf16(a, &p("attn.indexer.weights_proj.weight"), None)?,
                wk: if c.is_kv_source(id) { Some(self.bf16(a, &p("attn.indexer.wk.weight"), None)?) } else { None },
                k_norm: if c.is_kv_source(id) { self.vec_bf16(a, &p("attn.indexer.k_norm.weight"))? } else { 0 },
            })
        } else {
            None
        };

        let n_exp = if is_mtp { c.dspark_n_routed_experts } else { c.n_routed_experts };
        let inter = c.moe_inter_dim / tp;
        let kpad = inter.div_ceil(64) * 64;
        let dim = c.dim;
        let sgl = crate::ops::sgl_moe_on() && dim == 5120 && matches!((tp, inter), (4, 576) | (8, 288))
            && if is_mtp { crate::ops::sgl_mtp_on() && n_exp == 128 && c.dspark_n_activated_experts == 3 } else { n_exp == 384 && c.n_activated_experts == 6 };
        let rm = !sgl && crate::ops::tmoe_on() && (2 * inter) % 128 == 0 && dim % 128 == 0 && inter % 32 == 0;
        let w13_bytes = if sgl { sgl_expert_bytes(dim, inter).0 } else { (2 * inter / 64) * (dim / 64) * FP4_STEP };
        let w2_bytes = if sgl { sgl_expert_bytes(dim, inter).1 } else if rm { w2_rowmajor_bytes(dim, inter) } else { (dim / 64) * (kpad / 64) * FP4_STEP };
        let (all13, all2) = staging.buffers(n_exp * w13_bytes, n_exp * w2_bytes)?;
        all13.par_chunks_mut(w13_bytes).zip(all2.par_chunks_mut(w2_bytes)).enumerate()
            .try_for_each(|(e, (out13, out2))| -> Result<()> {
                let ep = |s: &str| format!("{prefix}.ffn.experts.{e}.{s}");
                let matrix = |name: &str, n, k| -> Result<Fp4Matrix<'_>> {
                    Ok(Fp4Matrix { w: self.ck.bytes(&ep(&format!("{name}.weight")))?,
                                   s: self.ck.bytes(&ep(&format!("{name}.scale")))?, n, k })
                };
                let pack = if sgl { expert_shard_sgl } else if rm { expert_shard_tc } else { expert_shard };
                pack(matrix("w1", c.moe_inter_dim, dim)?, matrix("w3", c.moe_inter_dim, dim)?,
                     matrix("w2", dim, c.moe_inter_dim)?, r, tp, out13, out2);
                Ok(())
            })?;
        let (w13, w2) = staging.upload(a, n_exp * w13_bytes, n_exp * w2_bytes)?;
        let experts = Experts { n: n_exp, w13, w2, w13_stride: w13_bytes, w2_stride: w2_bytes, inter, kpad, rm, sgl };
        let shared_sgl = if sgl {
            let (w1, _, _) = self.get(&p("ffn.shared_experts.w1.weight"))?;
            let (s1, _, _) = self.get(&p("ffn.shared_experts.w1.scale"))?;
            let (w3, _, _) = self.get(&p("ffn.shared_experts.w3.weight"))?;
            let (s3, _, _) = self.get(&p("ffn.shared_experts.w3.scale"))?;
            let (w2s, shape, _) = self.get(&p("ffn.shared_experts.w2.weight"))?;
            let (s2s, _, _) = self.get(&p("ffn.shared_experts.w2.scale"))?;
            anyhow::ensure!(shape[0] == dim, "sgl moe: shared w2 {:?}", shape);
            let (o13, o2) = shared_shard_sgl(w1, s1, w3, s3, w2s, s2s, shape[1], dim, r, tp);
            Some((a.put(&o13)?, a.put(&o2)?))
        } else {
            None
        };

        let shared_w13 = {
            let (w1, _, _) = self.get(&p("ffn.shared_experts.w1.weight"))?;
            let (s1, _, _) = self.get(&p("ffn.shared_experts.w1.scale"))?;
            let (w3, _, _) = self.get(&p("ffn.shared_experts.w3.weight"))?;
            let (s3, _, _) = self.get(&p("ffn.shared_experts.w3.scale"))?;
            let ks = dim / 32;
            let mut w = Vec::with_capacity(2 * inter * dim);
            let mut s = Vec::with_capacity(2 * inter / 32 * ks);
            for blk in 0..inter / 32 {
                let r0 = r * inter + blk * 32;
                w.extend_from_slice(&w1[r0 * dim..(r0 + 32) * dim]);
                s.extend_from_slice(&s1[(r0 / 32) * ks..(r0 / 32 + 1) * ks]);
                w.extend_from_slice(&w3[r0 * dim..(r0 + 32) * dim]);
                s.extend_from_slice(&s3[(r0 / 32) * ks..(r0 / 32 + 1) * ks]);
            }
            let wt = tile_fp8(&w, 2 * inter, dim);
            let out = Weight { fmt: WFmt::Fp8, w: a.put(&wt)?, s: a.put(&s)?, n: 2 * inter, k: dim };
            self.tc_copy(a, out.w, &w, &s, 2 * inter, dim)?;
            (out, a.put(&tile_fp8_k64(&wt, &s, 2 * inter, dim, dim))?)
        };
        let (shared_w13, sh13) = shared_w13;
        let shared_w2 = self.fp8_cols(a, &p("ffn.shared_experts.w2"), r * inter, (r + 1) * inter)?;
        let sh2 = {
            let (w, shape, _) = self.get(&p("ffn.shared_experts.w2.weight"))?;
            let (sc, sshape, _) = self.get(&p("ffn.shared_experts.w2.scale"))?;
            let ws = cols(w, shape[0], shape[1], r * inter, (r + 1) * inter);
            let ss = cols(sc, sshape[0], sshape[1], r * inter / 32, (r + 1) * inter / 32);
            a.put(&tile_fp8_k64(&tile_fp8(&ws, dim, inter), &ss, dim, inter, kpad))?
        };

        let engram = if let Some(ei) = c.engram_index(id) {
            let rows_total = c.engram_num_embeddings[ei];
            let shard = rows_total.div_ceil(tp);
            let (t, _, _) = self.get(&p("engram.embed.weight"))?;
            let (ts, _, _) = self.get(&p("engram.embed.scale"))?;
            let hd = c.engram_head_dim;
            let r0 = (r * shard).min(rows_total);
            let r1 = ((r + 1) * shard).min(rows_total);
            self.ck.prefault(&p("engram.embed.weight"), r0 * hd..r1 * hd)?;
            self.ck.prefault(&p("engram.embed.scale"), r0 * hd / 32..r1 * hd / 32)?;
            let tw = &t[r0 * hd..r1 * hd];
            let tsc = &ts[r0 * hd / 32..r1 * hd / 32];
            let nk = c.dim * (c.hc_mult + 1);
            let wkv = self.fp8(a, &p("engram.wkv"), Some((r * nk / tp, (r + 1) * nk / tp)), WFmt::Fp8)?;
            let q = self.ck.f32s(&p("engram.q_weight"))?;
            let k = self.ck.f32s(&p("engram.k_weight"))?;
            let qk: Vec<f32> = q.iter().zip(&k).map(|(a, b)| a * b).collect();
            Some(Engram { table: a.put_padded(tw, shard * hd, 0)?,
                          scales: a.put_padded(tsc, shard * hd / 32, 127)?,
                          shard_rows: shard, wkv, qk: a.put(&qk)? })
        } else {
            None
        };

        let has = |s: &str| self.ck.has(&p(s));
        let (wqkv, qkv) = self.fp8_merged(a, &[&p("attn.wq_a"), &p("attn.wkv")])?;
        Ok(Layer {
            id,
            wqkv,
            attn_norm: self.vec_bf16(a, &p("attn_norm.weight"))?,
            ffn_norm: self.vec_bf16(a, &p("ffn_norm.weight"))?,
            hc_attn: (self.vec_f32(a, &p("hc_attn_fn"))?, self.vec_f32(a, &p("hc_attn_base"))?, self.vec_f32(a, &p("hc_attn_scale"))?),
            hc_ffn: (self.vec_f32(a, &p("hc_ffn_fn"))?, self.vec_f32(a, &p("hc_ffn_base"))?, self.vec_f32(a, &p("hc_ffn_scale"))?),
            wq_a: qkv[0],
            q_norm: self.vec_bf16(a, &p("attn.q_norm.weight"))?,
            wq_b,
            wkv: qkv[1],
            kv_norm: self.vec_bf16(a, &p("attn.kv_norm.weight"))?,
            wo_a,
            wo_b,
            attn_sink,
            compressor,
            indexer,
            gate: self.bf16(a, &p("ffn.gate.weight"), None)?,
            gate_bias: self.vec_f32(a, &p("ffn.gate.bias"))?,
            experts,
            shared_w13,
            shared_w2,
            shared_k64: (sh13, sh2),
            shared_sgl,
            engram,
            main_proj: if has("main_proj.weight") { Some(self.fp8(a, &p("main_proj"), Some((r * dim / tp, (r + 1) * dim / tp)), WFmt::Fp8)?) } else { None },
            main_norm: if has("main_norm.weight") { Some(self.vec_bf16(a, &p("main_norm.weight"))?) } else { None },
            final_norm: if has("norm.weight") { Some(self.vec_bf16(a, &p("norm.weight"))?) } else { None },
            markov_embed: if has("markov_head.embed.weight") { Some(self.vec_bf16(a, &p("markov_head.embed.weight"))?) } else { None },
            markov_head: if has("markov_head.head.weight") {
                let v = c.vocab_size / tp;
                Some(self.bf16(a, &p("markov_head.head.weight"), Some((r * v, (r + 1) * v)))?)
            } else {
                None
            },
            confidence: if has("confidence_head.proj.weight") { Some(self.vec_bf16(a, &p("confidence_head.proj.weight"))?) } else { None },
        })
    }

    pub fn load(&self, a: &mut Arena, n_layers: usize, with_mtp: bool) -> Result<Weights> {
        let c = self.cfg;
        let inter = c.moe_inter_dim / self.tp;
        let experts = c.n_routed_experts.max(if with_mtp { c.dspark_n_routed_experts } else { 0 });
        let (s13, s2) = if crate::ops::sgl_moe_on() && c.dim % 128 == 0 { sgl_expert_bytes(c.dim, inter) } else { (0, 0) };
        let mut staging = ExpertStaging::new(
            experts * ((2 * inter / 64) * (c.dim / 64) * FP4_STEP).max(s13),
            experts * ((c.dim / 64) * inter.div_ceil(64) * FP4_STEP).max(w2_rowmajor_bytes(c.dim, inter)).max(s2),
        )?;
        let v = c.vocab_size / self.tp;
        let embed = self.vec_bf16(a, "embed.weight")?;
        let head = self.bf16(a, "head.weight", Some((self.rank * v, (self.rank + 1) * v)))?;
        let head_fp8_host = if with_mtp && deepseek_ultrafast_cuda::profile::sm103() { Some(self.fp8_quant("head.weight", Some((self.rank * v, (self.rank + 1) * v)))?) } else { None };
        let norm = self.vec_bf16(a, "norm.weight")?;
        let mut layers = Vec::new();
        for l in 0..n_layers {
            layers.push(self.layer(a, &mut staging, &format!("layers.{l}"), l)?);
        }
        let mut mtp = Vec::new();
        if with_mtp {
            for i in 0..c.n_mtp_layers {
                mtp.push(self.layer(a, &mut staging, &format!("mtp.{i}"), c.n_layers + i)?);
            }
        }
        let mtp_wkv = if mtp.is_empty() {
            None
        } else {
            let names: Vec<String> = (0..mtp.len()).map(|i| format!("mtp.{i}.attn.wkv")).collect();
            let parts: Vec<(&str, Option<(usize, usize)>)> = names.iter().map(|n| (n.as_str(), None)).collect();
            Some(self.fp8_cat(a, &parts)?)
        };
        staging.finish()?;
        drop(staging);
        deepseek_ultrafast_cuda::check(unsafe { deepseek_ultrafast_cuda::sys::cuStreamSynchronize(std::ptr::null_mut()) }, "finish weight uploads")?;
        Ok(Weights { embed, head, head_fp8: None, head_fp8_host, norm, layers, mtp, mtp_wkv, tc: self.tc.take() })
    }
}
