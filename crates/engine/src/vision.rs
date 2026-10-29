use anyhow::{Result, ensure};
use deepseek_ultrafast_cuda::{self as cu, Args, Buffer, DevPtr, Launch, Stream};
use deepseek_ultrafast_kernels::Kernels;
use deepseek_ultrafast_model::{st::Checkpoint, vision::{Image, VisionConfig, MAX_PATCHES}};
use cudarc::cublas::sys as blas;
use std::{collections::VecDeque, sync::Arc};

fn finish_uploads() -> Result<()> {
    cu::check(unsafe { cu::sys::cuStreamSynchronize(std::ptr::null_mut()) }, "finish vision uploads")
}
fn checked(s: blas::cublasStatus_t) -> Result<()> {
    ensure!(s == blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS, "vision cuBLAS: {s:?}"); Ok(())
}
struct Blas(blas::cublasHandle_t);
impl Blas {
    fn new(stream: &Stream) -> Result<Self> {
        let mut h = std::ptr::null_mut();
        checked(unsafe { blas::cublasCreate_v2(&mut h) })?;
        let b = Self(h);
        checked(unsafe { blas::cublasSetStream_v2(h, stream.raw as _) })?;
        Ok(b)
    }
    fn linear(&self, x: DevPtr, w: &Linear, out: DevPtr, rows: usize) -> Result<()> {
        use blas::{cublasOperation_t::*, cudaDataType_t::*, cublasComputeType_t::*, cublasGemmAlgo_t::*};
        let (one, zero) = (1f32, 0f32);
        checked(unsafe { blas::cublasGemmEx(self.0, CUBLAS_OP_T, CUBLAS_OP_N,
            w.n as i32, rows as i32, w.k as i32, (&one as *const f32).cast(),
            w.w.ptr as _, CUDA_R_16BF, w.k as i32, x as _, CUDA_R_16BF, w.k as i32,
            (&zero as *const f32).cast(), out as _, CUDA_R_32F, w.n as i32, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DFALT) })
    }
}
impl Drop for Blas { fn drop(&mut self) { unsafe { blas::cublasDestroy_v2(self.0); } } }

struct Linear { w: Buffer, bias: Option<Buffer>, n: usize, k: usize }
impl Linear {
    fn load(ck: &Checkpoint, key: &str, n: usize, k: usize, bias: bool) -> Result<Self> {
        Ok(Self { w: Buffer::from_host(ck.get(&format!("{key}.weight"), "BF16", &[n,k])?)?,
            bias: if bias { Some(Buffer::from_host(ck.get(&format!("{key}.bias"), "BF16", &[n])?)?) } else { None }, n, k })
    }
}
struct Block { norm1: Buffer, qkv: Linear, o: Linear, norm2: Buffer, w1: Linear, w2: Linear }
struct Scratch {
    capacity: usize,
    patches: Buffer, x: Buffer, norm: Buffer, tmp: Buffer, act: Buffer, attn: Buffer,
    accum: Buffer, unfolded: Buffer, align: Buffer, cos_sin: Buffer,
}
impl Scratch {
    fn new(n: usize, dim: usize) -> Result<Self> {
        let cap = n.next_power_of_two().min(MAX_PATCHES);
        let llm = 1024;
        Ok(Self { capacity: cap, patches: Buffer::alloc(cap * 588 * 2)?, x: Buffer::alloc(cap * 1024 * 2)?,
            norm: Buffer::alloc(cap * 1024 * 2)?, tmp: Buffer::alloc(cap * 5632 * 2)?,
            act: Buffer::alloc(cap * 2816 * 2)?, attn: Buffer::alloc(cap * 1024 * 2)?,
            accum: Buffer::alloc((cap * 5632).max(llm * dim) * 4)?,
            unfolded: Buffer::alloc(llm * 9216 * 2)?, align: Buffer::alloc(llm * dim * 2)?,
            cos_sin: Buffer::alloc(cap * 64 * 4)? })
    }
}

pub struct Vision {
    config: VisionConfig, dim: usize, blas: Blas, attention: crate::vision_attention::Attention, patch: Linear, blocks: Vec<Block>, norm: Buffer,
    align1: Linear, align2: Linear, start: Buffer, newline: Buffer, end: Buffer,
    pub biases: Vec<Buffer>,
    scratch: Option<Scratch>,
    cache: VecDeque<([u8;32], Arc<Buffer>)>, cache_bytes: usize,
    active: Vec<(usize, usize, Arc<Buffer>)>,
    spans: Option<Buffer>,
    pub span_count: usize,
    pub encodes: usize,
    pub hits: usize,
}
impl Vision {
    pub fn load(ck: &Checkpoint, dim: usize, layers: usize, stream: &Stream) -> Result<Self> {
        let c = VisionConfig::load(&ck.dir.join("inference/config.json"))?;
        let norm = |key: &str, len: usize| -> Result<Buffer> {
            let v = ck.f32s(key)?; ensure!(v.len() == len, "vision norm/bias shape: {key}"); Buffer::from_host(&v)
        };
        let mut blocks = Vec::new();
        for i in 0..c.vision_n_layers {
            let p = format!("vision.blocks.{i}");
            blocks.push(Block { norm1: norm(&format!("{p}.norm1.weight"), 1024)?,
                qkv: Linear::load(ck, &format!("{p}.attn.wqkv"), 3072, 1024, true)?,
                o: Linear::load(ck, &format!("{p}.attn.wo"), 1024, 1024, true)?,
                norm2: norm(&format!("{p}.norm2.weight"), 1024)?,
                w1: Linear::load(ck, &format!("{p}.mlp.w1"), 5632, 1024, false)?,
                w2: Linear::load(ck, &format!("{p}.mlp.w2"), 1024, 2816, false)? });
        }
        let biases = (0..layers).map(|i| norm(&format!("layers.{i}.ffn.gate.bias_vl"), 384)).collect::<Result<_>>()?;
        let vision = Self { config:c, dim, blas:Blas::new(stream)?, attention:crate::vision_attention::Attention::new(stream)?, patch:Linear::load(ck, "vision.patch_embed.proj", 1024, 588, true)?,
            blocks, norm:norm("vision.norm.weight", 1024)?, align1:Linear::load(ck, "aligner.w1", dim, 9216, true)?,
            align2:Linear::load(ck, "aligner.w2", dim, dim, true)?,
            start:Buffer::from_host(ck.get("image_start", "BF16", &[dim])?)?,
            newline:Buffer::from_host(ck.get("image_newline", "BF16", &[dim])?)?,
            end:Buffer::from_host(ck.get("image_end", "BF16", &[dim])?)?,
            biases, scratch:None, cache:VecDeque::new(), cache_bytes:0, active:Vec::new(), spans:None, span_count:0, encodes:0, hits:0 };
        finish_uploads()?;
        Ok(vision)
    }
    fn kernel(k: &Kernels, stream: &Stream, name: &str, grid: usize, args: &mut Args) -> Result<()> {
        cu::launch(k.get(name)?, &Launch::new(grid as u32, 256), stream, args)
    }
    pub fn encode(&mut self, k: &Kernels, stream: &Stream, image: &Image) -> Result<Arc<Buffer>> {
        let n = image.height * image.width;
        let (h,w) = (image.height.div_ceil(3), image.width.div_ceil(3));
        ensure!(n > 0 && n <= MAX_PATCHES && h*w <= 1024 && image.patches.len() == n*588, "invalid vision grid");
        if let Some(at) = self.cache.iter().position(|(key, _)| key == &image.digest) {
            let item = self.cache.remove(at).unwrap(); let out = item.1.clone(); self.cache.push_back(item); self.hits += 1; return Ok(out);
        }
        stream.synchronize()?;
        if self.scratch.as_ref().is_none_or(|s| s.capacity < n) { self.scratch = Some(Scratch::new(n, self.dim)?); }
        let s = self.scratch.as_ref().unwrap();
        cu::upload(s.patches.ptr, &image.patches)?;
        let mut cs = Vec::with_capacity(n*64);
        let theta = self.config.vision_rope_theta;
        for y in 0..image.height { for x in 0..image.width {
            let a: Vec<f32> = [y,x].iter().flat_map(|&p| (0..16).map(move |i| p as f32 / theta.powf((2*i) as f32/32.))).collect();
            cs.extend(a.iter().map(|a| a.cos())); cs.extend(a.iter().map(|a| a.sin()));
        }}
        cu::upload(s.cos_sin.ptr, &cs)?;
        finish_uploads()?;
        let lin = |x: DevPtr, w: &Linear, y: DevPtr, rows: usize, residual: DevPtr, gelu: bool| -> Result<()> {
            self.blas.linear(x, w, s.accum.ptr, rows)?;
            Self::kernel(k, stream, "vision_linear_finish", (rows*w.n).div_ceil(256), Args::new().ptr(s.accum.ptr)
                .ptr(w.bias.as_ref().map_or(0, |b| b.ptr)).ptr(y).ptr(residual).i32((rows*w.n) as i32).i32(w.n as i32).i32(gelu as i32))
        };
        lin(s.patches.ptr, &self.patch, s.x.ptr, n, 0, false)?;
        for b in &self.blocks {
            Self::kernel(k, stream, "vision_norm", n, Args::new().ptr(s.x.ptr).ptr(b.norm1.ptr).ptr(s.norm.ptr).i32(1024))?;
            lin(s.norm.ptr, &b.qkv, s.tmp.ptr, n, 0, false)?;
            Self::kernel(k, stream, "vision_rope", (n*1024).div_ceil(256), Args::new().ptr(s.tmp.ptr).ptr(s.cos_sin.ptr).i32((n*1024) as i32).i32(16))?;
            self.attention.run(n, s.tmp.ptr, s.attn.ptr)?;
            lin(s.attn.ptr, &b.o, s.x.ptr, n, s.x.ptr, false)?;
            Self::kernel(k, stream, "vision_norm", n, Args::new().ptr(s.x.ptr).ptr(b.norm2.ptr).ptr(s.norm.ptr).i32(1024))?;
            lin(s.norm.ptr, &b.w1, s.tmp.ptr, n, 0, false)?;
            Self::kernel(k, stream, "vision_swiglu", (n*2816).div_ceil(256), Args::new().ptr(s.tmp.ptr).ptr(s.act.ptr).i32((n*2816) as i32).i32(2816))?;
            lin(s.act.ptr, &b.w2, s.x.ptr, n, s.x.ptr, false)?;
        }
        Self::kernel(k, stream, "vision_norm", n, Args::new().ptr(s.x.ptr).ptr(self.norm.ptr).ptr(s.norm.ptr).i32(1024))?;
        Self::kernel(k, stream, "vision_unfold", (h*w*9216).div_ceil(256), Args::new().ptr(s.norm.ptr).ptr(s.unfolded.ptr).i32(image.height as i32).i32(image.width as i32).i32(1024))?;
        lin(s.unfolded.ptr, &self.align1, s.align.ptr, h*w, 0, true)?;
        lin(s.align.ptr, &self.align2, s.align.ptr, h*w, 0, false)?;
        let out = Arc::new(Buffer::alloc(image.len()*self.dim*2)?);
        Self::kernel(k, stream, "vision_pack", (image.len()*self.dim).div_ceil(256), Args::new().ptr(s.align.ptr).ptr(out.ptr)
            .ptr(self.start.ptr).ptr(self.newline.ptr).ptr(self.end.ptr).i32(h as i32).i32(w as i32).i32(self.dim as i32))?;
        self.encodes += 1;
        const CACHE_BYTES: usize = 64 << 20;
        while self.cache_bytes + out.bytes > CACHE_BYTES && !self.cache.is_empty() {
            let (_, old) = self.cache.pop_front().unwrap(); self.cache_bytes -= old.bytes;
        }
        if out.bytes <= CACHE_BYTES { self.cache_bytes += out.bytes; self.cache.push_back((image.digest, out.clone())); }
        Ok(out)
    }
    pub fn warm_up(&mut self, k: &Kernels, stream: &Stream) -> Result<()> {
        let (height, width) = (3, 3);
        let image = Image { start: 0, height, width, digest: [0xff; 32], patches: vec![0u16; height * width * 588].into() };
        self.encode(k, stream, &image)?;
        stream.synchronize()?;
        self.cache.retain(|(key, _)| key != &image.digest);
        self.cache_bytes = self.cache.iter().map(|(_, b)| b.bytes).sum();
        self.encodes -= 1;
        Ok(())
    }
    pub fn stage(&mut self, k: &Kernels, stream: &Stream, images: &[Image], tokens: &[i32], from: usize) -> Result<()> {
        if images.is_empty() && self.span_count == 0 { return Ok(()); }
        stream.synchronize()?;
        self.active.clear(); self.spans = None; self.span_count = images.len();
        let mut spans = Vec::new(); let mut end = 0;
        for im in images {
            im.validate(tokens)?;
            ensure!(im.start >= end, "overlapping/out-of-order image spans"); end = im.end();
            spans.extend([im.start as i32, im.end() as i32]);
            if im.end() > from.min(tokens.len().saturating_sub(128)) {
                let rows = self.encode(k, stream, im)?;
                self.active.push((im.start, im.end(), rows));
            }
        }
        if !spans.is_empty() { self.spans = Some(Buffer::from_host(&spans)?); finish_uploads()?; }
        Ok(())
    }
    pub fn spans_ptr(&self) -> DevPtr { self.spans.as_ref().map_or(0, |b| b.ptr) }
    pub fn overlaps(&self, pos: usize, n: usize) -> bool {
        self.active.iter().any(|&(s,e,_)| s < pos+n && e > pos)
    }
    pub fn embed(&self, k: &Kernels, stream: &Stream, h: DevPtr, cmp_hist: DevPtr, pos: usize, n: usize, hc: usize) -> Result<()> {
        for (start,end,rows) in &self.active {
            let lo = pos.max(*start); let hi = (pos+n).min(*end);
            if lo < hi { Self::kernel(k, stream, "vision_embed", ((hi-lo)*self.dim*hc).div_ceil(256), Args::new()
                .ptr(rows.ptr + ((lo-start)*self.dim*2) as u64).ptr(h + ((lo-pos)*self.dim*hc*2) as u64)
                .ptr(if cmp_hist == 0 { 0 } else { cmp_hist + (lo*4) as u64 }).i32((hi-lo) as i32).i32(self.dim as i32).i32(hc as i32))?; }
        }
        Ok(())
    }
}
