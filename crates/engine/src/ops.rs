use anyhow::{Result, bail, ensure};
use deepseek_ultrafast_cuda::{self as cu, Args, DevPtr, Launch, Stream};
use deepseek_ultrafast_kernels::Kernels;
use deepseek_ultrafast_model::formats::FP8_STEP64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WFmt {
    Fp8,
    Fp4,
    Bf16,
    Fp8Bf16Act,
}

impl WFmt {
    pub fn tile_k(self) -> usize {
        match self {
            WFmt::Fp8 | WFmt::Fp8Bf16Act => 32,
            WFmt::Fp4 => 64,
            WFmt::Bf16 => 16,
        }
    }
    fn name(self) -> &'static str {
        match self {
            WFmt::Fp8 => "fp8",
            WFmt::Fp4 => "fp4",
            WFmt::Bf16 => "bf16",
            WFmt::Fp8Bf16Act => "fp8b",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Weight {
    pub fmt: WFmt,
    pub w: DevPtr,
    pub s: DevPtr,
    pub n: usize,
    pub k: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RopeParams {
    pub x: u64,
    pub ldx: i32,
    pub n: i32,
    pub rows_per_token: i32,
    pub rope_off: i32,
    pub cos: u64,
    pub sin: u64,
    pub inverse: i32,
    pub quant: i32,
    pub ratio: i32,
    pub dst: u64,
    pub pos_offset: i32,
    pub norm_w: u64,
    pub norm_eps: f32,
    pub dst_by_row: i32,
    pub q4: u64,
    pub sf4: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemvParams {
    w: u64,
    ws: u64,
    x: u64,
    out: u64,
    m: i32,
    n: i32,
    k: i32,
    ldx: i32,
    ldo: i32,
    out_f32: i32,
    x_group_rows: i32,
    alpha: f32,
    stage_steps: i32,
    plan: u64,
    w_stride: u64,
    mode: i32,
    swiglu_limit: f32,
    prenorm_w: u64,
    prenorm_eps: f32,
    norm_part: u64,
    norm_nslices: i32,
    ksplit_slots: i32,
    xq: u64,
    xsq: u64,
    oq: u64,
    osq: u64,
    sched: u64,
    out2: u64,
    ldo2: i32,
    n_split: i32,
    rope_cos: u64,
    rope_sin: u64,
    rope_pos: u64,
    rope_head: i32,
    rope_off: i32,
    w_sh: u64,
    route: RouteArgs,
    kvr: RopeParams,
    kvr_on: i32,
    _pad_kvr: i32,
    out_pos: u64,
    iq_head: i32,
    iq_off: i32,
    sp: u64,
    spart: u64,
    sp_vloc: i32,
    sp_off: i32,
    xshift: i32,
    _padx: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemvParamsXb {
    p: GemvParams,
    xb: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RouteArgs {
    pub scores: u64,
    pub bias: u64,
    pub ids: u64,
    pub w: u64,
    pub n_exp: i32,
    pub topk: i32,
    pub scale: f32,
    pub nsum: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Epilogue {
    Store,
    SwiGlu(f32),
    SqrtSoftplus,
    AtomicAdd,
    Partial,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MoeQuant {
    pub x: Option<(DevPtr, DevPtr)>,
    pub out: Option<(DevPtr, DevPtr)>,
}

const SPLAN_BYTES: usize = 10240;

#[derive(Clone, Copy, Debug, Default)]
pub struct Moe {
    pub plan: DevPtr,
    pub w_stride: usize,
    pub max_slots: usize,
    pub split_hint: usize,
    pub sched: DevPtr,
    pub w_sh: DevPtr,
    pub route: RouteArgs,
}

#[derive(Clone, Copy)]
pub struct GemvOpts {
    pub out_f32: bool,
    pub x_group_rows: usize,
    pub alpha: f32,
    pub ksplit: Option<usize>,
    pub pdl: bool,
    pub epilogue: Epilogue,
    pub moe: Option<Moe>,
    pub prenorm: Option<(DevPtr, f32)>,
    pub norm_part: Option<(DevPtr, usize)>,
    pub split: Option<(usize, DevPtr, usize)>,
    pub rope: Option<(DevPtr, DevPtr, DevPtr, usize, usize)>,
    pub xq_out: Option<(DevPtr, DevPtr)>,
    pub xb_out: Option<DevPtr>,
    pub kv_rope: Option<RopeParams>,
    pub out_pos: Option<DevPtr>,
    pub iq: Option<(usize, usize)>,
    pub persist: bool,
    pub samp_rec: Option<(DevPtr, DevPtr, usize, usize)>,
    pub xshift: u32,
    pub ph: Option<usize>,
}

impl Default for GemvOpts {
    fn default() -> Self {
        GemvOpts { out_f32: false, x_group_rows: 0, alpha: 1.0, ksplit: None, pdl: true, epilogue: Epilogue::Store, moe: None, prenorm: None, norm_part: None, split: None, xq_out: None, xb_out: None, rope: None, kv_rope: None, out_pos: None, iq: None, persist: false, samp_rec: None, xshift: 0, ph: None }
    }
}

pub struct Hw {
    pub sms: usize,
    pub smem_optin: usize,
    pub two_cta_smem: usize,
    pub comm_ctas_per_sm: usize,
    pub max_threads_per_sm: usize,
    pub max_ctas_per_sm: usize,
    pub cc: (i32, i32),
}

pub fn hw() -> &'static Hw {
    static H: std::sync::OnceLock<Hw> = std::sync::OnceLock::new();
    H.get_or_init(|| {
        use cu::sys::CUdevice_attribute as A;
        let cc = (cu::current_attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(10), cu::current_attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(0));
        let (sms, smem_optin, two_cta_smem, comm_ctas_per_sm) = match cc {
            (10, 0) | (10, 3) => (cu::current_sm_count().map(|n| n as usize).unwrap_or(148), 227 * 1024, 110 * 1024, 18),
            _ => {
                let sms = cu::current_attr(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).expect("sm count") as usize;
                let optin = cu::current_attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN).expect("smem optin") as usize;
                let per_sm = cu::current_attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR).expect("smem per sm") as usize;
                let max_blocks = cu::current_attr(A::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR).expect("max blocks per sm") as usize;
                (sms, optin, (per_sm.min(optin + 1024) / 2 - 3 * 1024) / 1024 * 1024, max_blocks.min(18))
            }
        };
        let ov = String::from("none");
        let max_threads_per_sm = cu::current_attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR).map(|v| v as usize).unwrap_or(0);
        let max_ctas_per_sm = cu::current_attr(A::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR).map(|v| v as usize).unwrap_or(0);
        eprintln!(
            "hw: cc {}.{} sms {sms} smem_optin {}K two_cta {}K comm_ctas {comm_ctas_per_sm} threads/sm {max_threads_per_sm} ctas/sm {max_ctas_per_sm} (overrides: {ov})",
            cc.0,
            cc.1,
            smem_optin / 1024,
            two_cta_smem / 1024
        );
        Hw { sms, smem_optin, two_cta_smem, comm_ctas_per_sm, max_threads_per_sm, max_ctas_per_sm, cc }
    })
}

pub fn variant_cc() -> bool {
    hw().cc == (10, 3)
}

pub fn attn_fused_kernel() -> &'static str {
    static A: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    A.get_or_init(|| {
        attn_aud_kernel().unwrap_or("attn_fused")
    })
}

pub fn attn_aud_kernel() -> Option<&'static str> {
    static A: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *A.get_or_init(|| {
        deepseek_ultrafast_cuda::profile::sm103().then_some("variant_zzzzzaud/attn_aud_mqx")
    })
}

pub fn topk_cand_on() -> bool {
    deepseek_ultrafast_cuda::profile::sm103()
}

pub fn topk_cl2() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| {
        deepseek_ultrafast_cuda::profile::sm103()
    })
}
pub const TOPK_CL2_KERNEL: &str = "variant_topk_cand/topk_clc2_8_1024";

pub fn topk_cand_smem(ncand: usize) -> usize {
    let slots = ((ncand * 8).div_ceil(8) + 7) & !7;
    slots * 4
}

pub fn sm_count() -> usize {
    hw().sms
}

pub fn attn_smem_for(kernel: &str, r: usize, sp: usize, nh: usize) -> u32 {
    let bulk = if kernel.starts_with("variant_zzzzzaud/") { (sp - 1) * 16 * 2 * 4 } else { 0 };
    let full = 16 * 520 * 2 + r * 520 * 2 + 16 * r * 4 + 16 * (r + 8) * 2 + nh * 512 * 4 + 16 * 2 * 4 + r * 4 + 16 + (sp * 32 + nh * 512) * 4 + bulk;
    let src = (sp * 32 + nh * 512) * 4;
    let alias = kernel.starts_with("variant_zzzzzaud/attn_aud_m") && (16 + r) * 520 * 2 + 16 * r * 4 >= src;
    static PAD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let pad = *PAD.get_or_init(|| if deepseek_ultrafast_cuda::profile::sm103_tp4() { 120832 } else { 0 });
    (if alias { (full - src).max(pad) } else { full }) as u32
}

const STAGES: usize = 4;

pub(crate) fn pick_ph(steps: usize) -> usize {
    if steps >= 48 { 4 } else if steps >= 12 { 2 } else { 1 }
}

impl WFmt {
    fn step_bytes(self) -> usize {
        match self {
            WFmt::Fp4 => 2048 + 128,
            _ => 2048,
        }
    }
}

fn pick_cluster(w: &Weight, quads: usize) -> usize {
    let unit = w.fmt.tile_k().max(32);
    let mut best = 1;
    let cands: &[usize] = &[1, 2, 4, 8];
    for &cs in cands {
        if w.k % (cs * unit) == 0 && quads * cs <= 2 * sm_count() {
            best = cs;
        }
    }
    best
}

fn gemv_smem(w: &Weight, m: usize, o: &GemvOpts, cs: usize) -> usize {
    let mt = if m <= 8 { 8 } else { 16 };
    let ks = w.k / cs;
    let nsteps = ks / w.fmt.tile_k();
    let ss = nsteps.div_ceil(STAGES).min(12);
    STAGES * ss * w.fmt.step_bytes() + 2 * STAGES * 8 + mt * (ks + 32) * 2 + mt * (ks / 32) * 4 + o.ph.unwrap_or_else(|| pick_ph(nsteps)) * mt * 64 * 4 + if o.prenorm.is_some() { ks * 2 } else { 0 }
        + if cs > 1 { (mt * 64 + 4 * cs) * 4 } else { 0 }
}

pub fn gemv_cluster(w: &Weight, m: usize, o: &GemvOpts) -> usize {
    let quads = w.n / 64;
    let unit = w.fmt.tile_k().max(32);
    let hint = o.moe.map(|m| m.split_hint).unwrap_or(1);
    let mut cs = o.ksplit.unwrap_or_else(|| pick_cluster(w, quads * hint));
    while gemv_smem(w, m, o, cs) > hw().smem_optin && cs < 16 && w.k % (2 * cs * unit) == 0 {
        cs *= 2;
    }
    cs
}

#[allow(clippy::too_many_arguments)]
pub fn gemv(
    k: &Kernels,
    s: &Stream,
    w: &Weight,
    x: DevPtr,
    ldx: usize,
    m: usize,
    out: DevPtr,
    ldo: usize,
    o: &GemvOpts,
) -> Result<()> {
    ensure!((1..=16).contains(&m), "gemv supports 1..=16 rows, got {m}");
    ensure!(w.n % 64 == 0, "gemv needs n % 64 == 0 (n = {})", w.n);
    let nt = if m <= 8 { 1 } else { 2 };
    let quads = w.n / 64;
    let cs = gemv_cluster(w, m, o);
    let smem_for = |cs: usize| gemv_smem(w, m, o, cs);
    let ks = w.k / cs;
    if w.k % cs != 0 || ks % w.fmt.tile_k() != 0 || ks % 32 != 0 {
        bail!("bad split {cs} for k {}", w.k);
    }
    let ss = (ks / w.fmt.tile_k()).div_ceil(STAGES).min(12);
    let smem = smem_for(cs);
    ensure!(smem <= hw().smem_optin, "gemv smem {smem} too large");
    let ph = o.ph.unwrap_or_else(|| pick_ph(ks / w.fmt.tile_k()));
    let f = if o.xb_out.is_some() {
        ensure!(w.fmt == WFmt::Bf16 && nt == 1 && o.moe.is_none() && !o.persist && o.kv_rope.is_none(), "xb_out: dense BF16 GEMV of <= 8 rows");
        k.get(&format!("gemv_bf16_xb_n1_p{ph}"))?
    } else {
        k.get(&format!("gemv_{}_n{nt}_p{ph}", w.fmt.name()))?
    };
    f.set_max_dynamic_smem(smem as u32)?;
    if cs > 8 {
        f.set_non_portable_cluster()?;
    }
    let p = GemvParams {
        w: w.w,
        ws: w.s,
        x,
        out,
        m: m as i32,
        n: w.n as i32,
        k: w.k as i32,
        ldx: ldx as i32,
        ldo: ldo as i32,
        out_f32: o.out_f32 as i32,
        x_group_rows: o.x_group_rows as i32,
        alpha: o.alpha,
        stage_steps: ss as i32,
        plan: o.moe.map(|m| m.plan).unwrap_or(0),
        w_stride: o.moe.map(|m| m.w_stride as u64).unwrap_or(0),
        mode: match o.epilogue {
            Epilogue::Store => 0,
            Epilogue::SwiGlu(_) => 1,
            Epilogue::AtomicAdd => 2,
            Epilogue::SqrtSoftplus => 3,
            Epilogue::Partial => 4,
        },
        swiglu_limit: match o.epilogue {
            Epilogue::SwiGlu(l) => l,
            _ => 0.0,
        },
        prenorm_w: o.prenorm.map(|p| p.0).unwrap_or(0),
        prenorm_eps: o.prenorm.map(|p| p.1).unwrap_or(0.0),
        norm_part: o.norm_part.map(|p| p.0).unwrap_or(0),
        norm_nslices: o.norm_part.map(|p| p.1 as i32).unwrap_or(0),
        ksplit_slots: 0,
        xq: 0,
        xsq: 0,
        oq: o.xq_out.map(|v| v.0).unwrap_or(0),
        osq: o.xq_out.map(|v| v.1).unwrap_or(0),
        sched: 0,
        out2: o.split.map(|v| v.1).unwrap_or(0),
        ldo2: o.split.map(|v| v.2 as i32).unwrap_or(0),
        n_split: o.split.map(|v| v.0 as i32).unwrap_or(0),
        rope_cos: o.rope.map(|v| v.0).unwrap_or(0),
        rope_sin: o.rope.map(|v| v.1).unwrap_or(0),
        rope_pos: o.rope.map(|v| v.2).unwrap_or(0),
        rope_head: o.rope.map(|v| v.3 as i32).unwrap_or(0),
        rope_off: o.rope.map(|v| v.4 as i32).unwrap_or(0),
        w_sh: 0,
        route: RouteArgs::default(),
        kvr: RopeParams::default(),
        kvr_on: 0,
        _pad_kvr: 0,
        out_pos: 0,
        iq_head: 0,
        iq_off: 0,
        sp: o.samp_rec.map(|v| v.0).unwrap_or(0),
        spart: o.samp_rec.map(|v| v.1).unwrap_or(0),
        sp_vloc: o.samp_rec.map(|v| v.2 as i32).unwrap_or(0),
        sp_off: o.samp_rec.map(|v| v.3 as i32).unwrap_or(0),
        xshift: o.xshift as i32,
        _padx: 0,
    };
    let slots = o.moe.map(|m| m.max_slots).unwrap_or(1) as u32;
    let mut p = p;
    let mut gx = quads as u32;
    if let Some((h, off)) = o.iq {
        ensure!(o.split.is_some() && o.rope.is_some() && !o.out_f32 && o.alpha == 1.0, "iq: grouped bf16 store with the rope tables");
        ensure!(o.split.unwrap().0 % 64 == 0 && h % 32 == 0 && off % 32 == 0);
        ensure!((m * 64).div_ceil(cs) % 32 == 0, "iq: per-CTA epilogue ranges must be whole warps (m {m}, cs {cs})");
        p.iq_head = h as i32;
        p.iq_off = off as i32;
    }
    if let Some(op) = o.out_pos {
        ensure!(o.moe.is_none() && o.rope.is_none(), "out_pos: dense store only");
        p.out_pos = op;
    }
    if let Some(r) = o.kv_rope {
        ensure!(o.rope.is_some() && o.moe.is_none(), "kv_rope needs the fused rope (its meta) on a dense GEMV");
        p.kvr = r;
        p.kvr_on = 1;
        gx += 1;
    }
    ensure!(o.samp_rec.is_none() || o.persist, "samp_rec needs the persistent head");
    if o.persist && w.fmt == WFmt::Bf16 && cs == 1 && m <= 8 && ph == 4 && o.out_f32 && o.epilogue == Epilogue::Store && o.moe.is_none() && o.split.is_none()
        && o.rope.is_none() && o.kv_rope.is_none() && o.out_pos.is_none() && o.x_group_rows == 0 && o.xq_out.is_none()
    {
        let pf = k.get("gemv_bf16_persist_p4")?;
        let psmem = STAGES * ss * w.fmt.step_bytes() + 2 * STAGES * 8 + 8 * (w.k + 8) * 2 + 4 * 8 * 64 * 4 + w.k * 2;
        pf.set_max_dynamic_smem(psmem as u32)?;
        let g = quads.min(sm_count()) as u32;
        return cu::launch(pf, &Launch::new(g, (17 * 32) as u32).smem(psmem as u32).pdl(o.pdl), s, Args::new().push(p));
    }
    if o.persist && w.fmt == WFmt::Fp8Bf16Act && cs == 1 && m <= 8 && ph == 4 && o.out_f32 && o.epilogue == Epilogue::Store && o.moe.is_none() && o.split.is_none()
        && o.rope.is_none() && o.kv_rope.is_none() && o.out_pos.is_none() && o.x_group_rows == 0 && o.xq_out.is_none()
    {
        let pf = k.get("gemv_fp8b_persist_p4")?;
        let g = quads.min(sm_count());
        let nqc = quads.div_ceil(g);
        let psmem = STAGES * ss * w.fmt.step_bytes() + 2 * STAGES * 8 + 8 * (w.k + 8) * 2 + 4 * 8 * 64 * 4 + w.k * 2 + nqc * 2 * (w.k / 32);
        ensure!(psmem <= hw().smem_optin, "fp8 head smem {psmem} too large");
        pf.set_max_dynamic_smem(psmem as u32)?;
        return cu::launch(pf, &Launch::new(g as u32, (17 * 32) as u32).smem(psmem as u32).pdl(o.pdl), s, Args::new().push(GemvParamsXb { p, xb: 0 }));
    }
    ensure!(o.samp_rec.is_none(), "samp_rec: the persistent head did not apply");
    let mut cfg = Launch::new(gx, ((4 * ph + 1) * 32) as u32).grid3((gx, cs as u32, slots)).smem(smem as u32).pdl(o.pdl);
    cfg.cluster = Some((1, cs as u32, 1));
    if let Some(xb) = o.xb_out {
        return cu::launch(f, &cfg, s, Args::new().push(GemvParamsXb { p, xb }));
    }
    cu::launch(f, &cfg, s, Args::new().push(p))
}

pub fn verify_head_fp8() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(deepseek_ultrafast_cuda::profile::sm103_tp4)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_gemv(
    k: &Kernels,
    s: &Stream,
    w: &Weight,
    x: DevPtr,
    ldx: usize,
    m: usize,
    out: DevPtr,
    ldo: usize,
    moe: &Moe,
    epilogue: Epilogue,
    prenorm: Option<(DevPtr, f32)>,
    norm_part: Option<(DevPtr, usize)>,
    q: MoeQuant,
    pdl: bool,
) -> Result<()> {
    ensure!(w.fmt == WFmt::Fp4 && w.n % 64 == 0);
    let nt = if m <= 8 { 1 } else { 2 };
    let mt = 8 * nt;
    let quads = w.n / 64;
    let nsteps = w.k / 64;
    let swiglu = matches!(epilogue, Epilogue::SwiGlu(_));
    if !swiglu && q.x.is_some() {
        let qb = (nsteps * 2176).max(nsteps.div_ceil(2) * FP8_STEP64);
        let smem = 4 * qb + mt * (w.k + 32) * 2 + (w.k / 32) * mt * 4;
        ensure!(w.k % 64 == 0 && smem <= hw().smem_optin);
        let f = k.get(&format!("moe_w2_n{nt}"))?;
        f.set_max_dynamic_smem(smem as u32)?;
        let per_sm = if smem <= hw().two_cta_smem { 2 } else { 1 };
        let items = quads.div_ceil(2) * moe.max_slots + if moe.w_sh != 0 { quads } else { 0 };
        let gx = (per_sm * sm_count()).min(items);
        let p = GemvParams {
            w: w.w,
            ws: 0,
            x,
            out,
            m: m as i32,
            n: w.n as i32,
            k: w.k as i32,
            ldx: ldx as i32,
            ldo: ldo as i32,
            out_f32: 1,
            x_group_rows: 0,
            alpha: 1.0,
            stage_steps: nsteps as i32,
            plan: moe.plan,
            w_stride: moe.w_stride as u64,
            mode: 0,
            swiglu_limit: 0.0,
            prenorm_w: 0,
            prenorm_eps: 0.0,
            norm_part: 0,
            norm_nslices: 0,
            ksplit_slots: 0,
            xq: q.x.map(|v| v.0).unwrap_or(0),
            xsq: q.x.map(|v| v.1).unwrap_or(0),
            oq: 0,
            osq: 0,
            sched: 0,
            out2: 0,
            ldo2: 0,
            n_split: 0,
        rope_cos: 0,
        rope_sin: 0,
        rope_pos: 0,
        rope_head: 0,
        rope_off: 0,
        w_sh: moe.w_sh,
        route: RouteArgs::default(),
            kvr: RopeParams::default(),
            kvr_on: 0,
        _pad_kvr: 0,
        out_pos: 0,
        iq_head: 0,
        iq_off: 0,
        sp: 0,
        spart: 0,
        sp_vloc: 0,
        sp_off: 0,
        xshift: 0,
        _padx: 0,
        };
        let cfg = Launch::new(gx as u32, 9 * 32).smem(smem as u32).pdl(pdl);
        return cu::launch(f, &cfg, s, Args::new().push(p));
    }
    let ph = if nsteps >= 48 { 4 } else { 2 };
    let fixed = mt * (w.k + 32) * 2 + mt * (w.k / 32) * 4 + 4 * mt * 64 * 4 + if swiglu { w.k * 2 + SPLAN_BYTES } else { 0 } + 2 * STAGES * 8 + 256;
    let budget = hw().smem_optin - fixed;
    let ss = nsteps.min(12).min(budget / (STAGES * 2176)).max(1);
    let smem = fixed - if swiglu { SPLAN_BYTES } else { 0 } + STAGES * ss * 2176;
    ensure!(moe.w_sh == 0 || (swiglu && ss * 2176 >= FP8_STEP64), "shared-expert slot needs a stage of at least one FP8 64-k step");
    let per_sm = if swiglu { 1 } else { 2 };
    let gx = (per_sm * sm_count()).min(quads * (moe.max_slots + (moe.w_sh != 0) as usize));
    let (name, mode) = if swiglu { (format!("moe_fp4_swiglu_n{nt}_p4"), 1) } else { (format!("moe_fp4_store_n{nt}_p{ph}"), 0) };
    let f = k.get(&name)?;
    f.set_max_dynamic_smem(smem as u32)?;
    let p = GemvParams {
        w: w.w,
        ws: 0,
        x,
        out,
        m: m as i32,
        n: w.n as i32,
        k: w.k as i32,
        ldx: ldx as i32,
        ldo: ldo as i32,
        out_f32: (mode == 0) as i32,
        x_group_rows: 0,
        alpha: 1.0,
        stage_steps: ss as i32,
        plan: moe.plan,
        w_stride: moe.w_stride as u64,
        mode,
        swiglu_limit: match epilogue {
            Epilogue::SwiGlu(l) => l,
            _ => 0.0,
        },
        prenorm_w: prenorm.map(|p| p.0).unwrap_or(0),
        prenorm_eps: prenorm.map(|p| p.1).unwrap_or(0.0),
        norm_part: norm_part.map(|p| p.0).unwrap_or(0),
        norm_nslices: norm_part.map(|p| p.1 as i32).unwrap_or(0),
        ksplit_slots: 0,
        xq: q.x.map(|v| v.0).unwrap_or(0),
        xsq: q.x.map(|v| v.1).unwrap_or(0),
        oq: q.out.map(|v| v.0).unwrap_or(0),
        osq: q.out.map(|v| v.1).unwrap_or(0),
        sched: moe.sched,
        out2: 0,
        ldo2: 0,
        n_split: 0,
        rope_cos: 0,
        rope_sin: 0,
        rope_pos: 0,
        rope_head: 0,
        rope_off: 0,
        w_sh: moe.w_sh,
        route: if swiglu { moe.route } else { RouteArgs::default() },
        kvr: RopeParams::default(),
        kvr_on: 0,
        _pad_kvr: 0,
        out_pos: 0,
        iq_head: 0,
        iq_off: 0,
        sp: 0,
        spart: 0,
        sp_vloc: 0,
        sp_off: 0,
        xshift: 0,
        _padx: 0,
    };
    let cfg = Launch::new(gx as u32, ((4 * ph + 1) * 32) as u32).smem(smem as u32).pdl(pdl);
    cu::launch(f, &cfg, s, Args::new().push(p))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct W2Params {
    w: u64,
    w_stride: u64,
    w_sh: u64,
    out: u64,
    ldo: i32,
    n: i32,
    k: i32,
    alpha: f32,
    done: u64,
    sq_ld: i32,
    qps: i32,
    w2x: i32,
    flags: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GateParams {
    w: u64,
    part: u64,
    ctr: u64,
    xq: u64,
    xsq: u64,
    nks: i32,
    quads: i32,
    ss: i32,
    ph: i32,
    mixf: u64,
    dots_h: u64,
    dots_fn: u64,
    dots_ctr: u64,
    dots_z: i32,
    _pad2: i32,
    mx: HcMixParams,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct HcMixParams {
    pub part: u64,
    pub nslices: i32,
    pub _pad: i32,
    pub base: u64,
    pub scale: u64,
    pub pre: u64,
    pub post: u64,
    pub comb: u64,
    pub dim: i32,
    pub iters: i32,
    pub norm_eps: f32,
    pub hc_eps: f32,
}

#[derive(Clone, Copy)]
pub struct DotsFused {
    pub h: DevPtr,
    pub fn_: DevPtr,
    pub ctr: DevPtr,
    pub z: usize,
    pub mx: HcMixParams,
}

#[derive(Clone, Copy)]
pub struct GateFused<'a> {
    pub w: &'a Weight,
    pub dots: Option<DotsFused>,
    pub part: DevPtr,
    pub ctr: DevPtr,
    pub xq: DevPtr,
    pub xsq: DevPtr,
    pub x: DevPtr,
    pub ldx: usize,
    pub norm_w: DevPtr,
    pub norm_eps: f32,
    pub norm_part: DevPtr,
    pub nslices: usize,
    pub mixf: DevPtr,
}

pub fn gemv_split_params(w: &Weight, m: usize, o: &GemvOpts) -> (usize, usize, usize) {
    let cs = gemv_cluster(w, m, o);
    let ks = w.k / cs;
    let ss = (ks / w.fmt.tile_k()).div_ceil(STAGES).min(12);
    (cs, ss, o.ph.unwrap_or_else(|| pick_ph(ks / w.fmt.tile_k())))
}

pub const GATE_PH: usize = 2;

fn moe_gate_act_bytes(k2: usize) -> usize {
    2 * 8 * (k2 + 32) * 2 + 2 * 8 * moe_fused_sq_ld(k2) * 4
}

pub fn moe_gate_extra(n_exp: usize, k2: usize) -> usize {
    (8 * n_exp * 4).saturating_sub(moe_gate_act_bytes(k2)).next_multiple_of(128)
}

pub fn moe_gate_fits(gate: &Weight, m: usize, k13: usize, k2: usize) -> bool {
    let Some(shape) = moe_fused_cfg(k13, k2) else { return false };
    let ss = shape.ss;
    let (cs, _, _) = gemv_split_params(gate, m, &GemvOpts { ph: Some(GATE_PH), ..Default::default() });
    let ksg = gate.k / cs;
    let ss16 = ss * 2176 / 2048;
    let ntg = ksg / 16;
    let mt = 8;
    let act = moe_gate_act_bytes(k2);
    let stat = SPLAN_BYTES + 2 * 4 * 32 * 4 * 4 + 512;
    let extra = moe_gate_extra(gate.n, k2);
    gate.fmt == WFmt::Bf16 && gate.k == k13 && gate.n % 64 == 0 && cs <= 8 && ksg % 16 == 0 && ntg.div_ceil(ss16) <= STAGES
        && mt * (ksg + 8) * 2 <= act && shape.smem + extra + stat <= hw().smem_optin && m <= mt
}

pub fn moe_dots_fit(k13: usize, z: usize) -> bool {
    z >= 1 && 24 % z == 0 && (24 / z) * 4 * 256 * 4 + 4 * 24 * 65 * 4 <= 8 * (k13 + 32) * 2
}

pub fn prep_head() -> bool {
    deepseek_ultrafast_cuda::profile::sm103()
}

pub fn moe_fused_shape(k13: usize, k2: usize) -> Option<(usize, usize, usize)> {
    moe_fused_cfg(k13, k2).map(|f| (f.ss, f.smem, f.qps))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedShape {
    pub ss: usize,
    pub smem: usize,
    pub qps: usize,
}

pub fn fused_shape_for(k13: usize, k2: usize, optin: usize) -> Option<FusedShape> {
    let mt = 8;
    let sq_ld = moe_fused_sq_ld(k2);
    let ns2 = k2 / 64;
    let fixed = mt * (k13 + 32) * 2 + mt * (k13 / 32) * 4 + 4 * mt * 64 * 4 + 2 * mt * (k2 + 32) * 2 + 2 * mt * sq_ld * 4 + 2 * (STAGES + 4) * 8;
    let stat = SPLAN_BYTES + 2 * 4 * 32 * 4 * 4 + 512;
    let holds = |ss: usize| ss * 2176 >= FP8_STEP64 && ss * 2176 >= 2 * ns2 * 2176 && ss * 2176 >= ns2 * FP8_STEP64;
    let ss = (k13 / 64).min(12).min((optin - fixed - stat) / (STAGES * 2176));
    if holds(ss) {
        return Some(FusedShape { ss, smem: fixed + STAGES * ss * 2176, qps: 2 });
    }
    let stage = ss * 2176;
    if stage < FP8_STEP64 {
        return None;
    }
    let dead = mt * (k13 + 32) * 2 + mt * (k13 / 32) * 4 + 4 * mt * 64 * 4;
    let w2stage = (ns2 * 2176).max(ns2.div_ceil(2) * FP8_STEP64);
    (stage >= ns2 * 2176 && stage >= ns2.div_ceil(2) * FP8_STEP64 && ns2 >= 2 && 4 * w2stage <= dead).then_some(FusedShape { ss, smem: fixed + STAGES * stage, qps: 1 })
}

pub fn moe_fused_cfg(k13: usize, k2: usize) -> Option<FusedShape> {
    fused_shape_for(k13, k2, hw().smem_optin)
}

pub fn tmoe_on() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        deepseek_ultrafast_cuda::profile::sm103_tp4()
    })
}

pub const TMOE_KERNEL: &str = "variant_zzzzzzzzzzmoe/tmoe_fused";
pub fn sgl_moe_on() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        deepseek_ultrafast_cuda::profile::sm103()
    })
}

pub fn sgl_mtp_on() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| sgl_moe_on() && deepseek_ultrafast_cuda::profile::sm103_tp8())
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct SgmParams {
    pub tm_a: u64,
    pub tm_b: u64,
    pub tm_sh: u64,
    pub sfa_sh: u64,
    pub sh_clamp: f32,
    pub has_sh: i32,
    pub m_rows: i32,
    pub _pad0: i32,
    pub sfa: u64,
    pub sfa_stride: i64,
    pub sfb: u64,
    pub out: u64,
    pub hs: u64,
    pub record: u64,
    pub clamp: u64,
    pub rows: i32,
    pub cta_words: i32,
    pub trigger_at_start: i32,
    pub _pad: i32,
}
const _: () = assert!(std::mem::size_of::<SgmParams>() == 120);

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct SgrParams {
    pub x: u64,
    pub ldx: i64,
    pub norm_part: u64,
    pub nslices: i32,
    pub eps: f32,
    pub norm_w: u64,
    pub w: u64,
    pub bias: u64,
    pub slots: u64,
    pub counter: u64,
    pub out_w: u64,
    pub out_i: u64,
    pub permuted: u64,
    pub record: u64,
    pub xq: u64,
    pub xs: u64,
    pub scale: f32,
    pub _pad: i32,
}
const _: () = assert!(std::mem::size_of::<SgrParams>() == 128);

pub fn attn_sds_on() -> bool {
    deepseek_ultrafast_cuda::profile::sm103_tp8()
}

pub fn attn_sds_smem(h: usize) -> u32 {
    let kv = 80 * 1024;
    let q = h * 1024;
    let pt = (h * 88 * 2).next_multiple_of(16);
    let s = 3 * h * 80 * 4 + 80 * 4 + h * 8;
    let stage = (16 * h * 32 * 4).next_multiple_of(16);
    let recv = 8 * h * 64 * 4 + 8 * h * 8 + 8;
    ((kv + q + pt + s + stage + recv + 1024).next_multiple_of(128) + 128) as u32
}

pub const SGL_GEMM1_SMEM: u32 = 6 * 36864 + 1024;

pub const SGL_GEMM2_SMEM: u32 = 10 * 18432 + 1024;
pub const SGL_GEMM2_SMEM_TP8: u32 = 4 * 3 * 18432 + 1024;

pub fn sgl_moe_launch(k: &Kernels, s: &Stream, m: usize, pad: usize, n_exp: usize, r: SgrParams, g1: SgmParams, g2: SgmParams) -> Result<()> {
    ensure!((1..=8).contains(&m), "sgl_moe: {m} rows");
    let rt = match n_exp {
        384 => "sgl_router",
        128 => "sgl_router_d",
        _ => anyhow::bail!("sgl_moe: no router for {n_exp} experts"),
    };
    let sfx = match pad {
        640 => "",
        384 => "_tp8",
        _ => anyhow::bail!("sgl_moe: no GEMMs for a padded intermediate of {pad}"),
    };
    let f = k.get(&format!("variant_zzzzzzzzzzsgl/{rt}_{m}"))?;
    cu::launch(f, &Launch::new(384 / 3, 320).pdl(true), s, Args::new().push(r))?;
    let grid = sm_count() as u32;
    let f1 = k.get(&format!("variant_zzzzzzzzzzsgl/sgl_gemm1{sfx}"))?;
    f1.set_max_dynamic_smem(SGL_GEMM1_SMEM)?;
    cu::launch(f1, &Launch::new(grid, 256).smem(SGL_GEMM1_SMEM).pdl(true), s, Args::new().push(g1))?;
    let f2 = k.get(&format!("variant_zzzzzzzzzzsgl/sgl_gemm2{sfx}"))?;
    let sm2 = if pad == 384 { SGL_GEMM2_SMEM_TP8 } else { SGL_GEMM2_SMEM };
    f2.set_max_dynamic_smem(sm2)?;
    cu::launch(f2, &Launch::new(grid, 256).smem(sm2).pdl(true), s, Args::new().push(g2))
}

pub const TMOE_STAGE_BYTES: usize = 3 * 4 * (16384 + 512 + 1024) + 5 * 512;
pub const TMOE_IMG: usize = 1536;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct TmoeParams {
    pub w13_map: u64,
    pub w13: u64,
    pub w13_stride: i64,
    pub w2_map: u64,
    pub w2: u64,
    pub w2_stride: i64,
    pub sh13_map: u64,
    pub sh13_sf: u64,
    pub sh2_map: u64,
    pub sh2_sf: u64,
    pub ximg: u64,
    pub aimg: u64,
    pub out: u64,
    pub done: u64,
    pub part: u64,
    pub tcnt: u64,
    pub sctr: u64,
    pub r13: i32,
    pub k2: i32,
    pub n2: i32,
    pub flags: i32,
    pub kq: i32,
    pub _pad: i32,
}
const _: () = assert!(std::mem::size_of::<TmoeParams>() == 160);

pub const TMOE_KQ: usize = 2;

pub const TMOE_KQ_MAX: usize = 8;

pub fn moe_fused_sq_ld(kpad: usize) -> usize {
    (kpad / 32).next_multiple_of(4)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_fused(
    k: &Kernels,
    s: &Stream,
    w13: &Weight,
    w2: &Weight,
    w2_stride: usize,
    w2_sh: DevPtr,
    x: DevPtr,
    ldx: usize,
    m: usize,
    act: DevPtr,
    ldact: usize,
    out2: DevPtr,
    ldo2: usize,
    moe: &Moe,
    swiglu_limit: f32,
    q: MoeQuant,
    done: DevPtr,
    pdl: bool,
    gate: Option<GateFused>,
    tm: Option<TmoeParams>,
) -> Result<()> {
    ensure!(m <= 8 && w13.fmt == WFmt::Fp4 && w2.fmt == WFmt::Fp4 && w13.n % 64 == 0 && w2.n % 256 == 0 && w2.k % 64 == 0);
    ensure!(tm.is_none() || gate.is_some(), "tmoe_fused: the gate runs inside the launch");
    ensure!(moe.sched != 0 && (moe.route.scores != 0 || gate.is_some()) && (q.x.is_some() || gate.is_some()) && q.out.is_some());
    let sq_ld = moe_fused_sq_ld(w2.k);
    let Some(shape) = moe_fused_cfg(w13.k, w2.k) else {
        bail!("fused MoE: ring stage too small for w2 K {}", w2.k)
    };
    let (ss, qps) = (shape.ss, shape.qps);
    let smem = shape.smem + gate.map(|g| moe_gate_extra(g.w.n, w2.k)).unwrap_or(0);
    let flags = 0;
    let (name, warps) = if tm.is_some() { (TMOE_KERNEL, 18) } else if gate.is_some() { ("moe_fp4_fused_gate_n1", 18) } else { ("moe_fp4_fused_n1", 18) };
    let f = k.get(name)?;
    let smem = if tm.is_some() { smem.max(TMOE_STAGE_BYTES) + 1024 } else { smem };
    f.set_max_dynamic_smem(smem as u32)?;
    let free = 5usize;
    let grid_for = || match gate { Some(g) if g.dots.is_some() => sm_count(), Some(_) => sm_count() - free, None => sm_count() - 1 };
    let gp = match gate {
        Some(gt) => {
            ensure!(moe_gate_fits(gt.w, m, w13.k, w2.k) && gt.w.n as i32 == moe.route.n_exp, "fused MoE: the gate does not fit the launch");
            let (cs, gss, gph) = gemv_split_params(gt.w, m, &GemvOpts { ph: Some(GATE_PH), ..Default::default() });
            let d = gt.dots;
            let need = (gt.w.n / 64) * cs + cs + d.map(|d| d.mx.nslices as usize * d.z + 1).unwrap_or(0);
            if need > grid_for() {
                deepseek_ultrafast_logging::diagnostic!("fused MoE: the gate, x slice{} items need {need} CTAs, the grid has {}", if d.is_some() { ", dots and mix" } else { "" }, grid_for());
                bail!("fused MoE: the gate, x slice{} items need {need} CTAs, the grid has {}", if d.is_some() { ", dots and mix" } else { "" }, grid_for());
            }
            if let Some(d) = d {
                ensure!(moe_dots_fit(w13.k, d.z), "fused MoE: a dots item's mixing weights and partials do not fit the x rows (z {})", d.z);
            }
            GateParams {
                w: gt.w.w, part: gt.part, ctr: gt.ctr, xq: gt.xq, xsq: gt.xsq, nks: cs as i32, quads: (gt.w.n / 64) as i32, ss: gss as i32, ph: gph as i32, mixf: gt.mixf,
                dots_h: d.map(|d| d.h).unwrap_or(0), dots_fn: d.map(|d| d.fn_).unwrap_or(0), dots_ctr: d.map(|d| d.ctr).unwrap_or(0), dots_z: d.map(|d| d.z as i32).unwrap_or(0), _pad2: 0,
                mx: d.map(|d| d.mx).unwrap_or_default(),
            }
        }
        None => GateParams::default(),
    };
    let p = GemvParams {
        w: w13.w,
        ws: 0,
        x: gate.map(|g| g.x).unwrap_or(x),
        out: act,
        m: m as i32,
        n: w13.n as i32,
        k: w13.k as i32,
        ldx: gate.map(|g| g.ldx).unwrap_or(ldx) as i32,
        ldo: ldact as i32,
        out_f32: 0,
        x_group_rows: 0,
        alpha: 1.0,
        stage_steps: ss as i32,
        plan: moe.plan,
        w_stride: moe.w_stride as u64,
        mode: 1,
        swiglu_limit,
        prenorm_w: gate.map(|g| g.norm_w).unwrap_or(0),
        prenorm_eps: gate.map(|g| g.norm_eps).unwrap_or(0.0),
        norm_part: gate.map(|g| g.norm_part).unwrap_or(0),
        norm_nslices: gate.map(|g| g.nslices).unwrap_or(0) as i32,
        ksplit_slots: 0,
        xq: if gate.is_some() { 0 } else { q.x.map(|v| v.0).unwrap_or(0) },
        xsq: if gate.is_some() { 0 } else { q.x.map(|v| v.1).unwrap_or(0) },
        oq: q.out.map(|v| v.0).unwrap_or(0),
        osq: q.out.map(|v| v.1).unwrap_or(0),
        sched: moe.sched,
        out2: 0,
        ldo2: 0,
        n_split: 0,
        rope_cos: 0,
        rope_sin: 0,
        rope_pos: 0,
        rope_head: 0,
        rope_off: 0,
        w_sh: moe.w_sh,
        route: moe.route,
        kvr: RopeParams::default(),
        kvr_on: 0,
        _pad_kvr: 0,
        out_pos: 0,
        iq_head: 0,
        iq_off: 0,
        sp: 0,
        spart: 0,
        sp_vloc: 0,
        sp_off: 0,
        xshift: 0,
        _padx: 0,
    };
    let q2 = W2Params { w: w2.w, w_stride: w2_stride as u64, w_sh: w2_sh, out: out2, ldo: ldo2 as i32, n: w2.n as i32, k: w2.k as i32, alpha: 1.0, done, sq_ld: sq_ld as i32, qps: qps as i32, w2x: if qps == 1 { 4 } else { 0 }, flags };
    let grid = grid_for();
    let cfg = Launch::new(grid as u32, warps * 32).smem(smem as u32).pdl(pdl);
    let mut args = Args::new();
    if let Some(t) = tm {
        args.push(p).push(gp).push(t);
    } else {
        args.push(p).push(q2);
        if gate.is_some() {
            args.push(gp);
        }
    }
    cu::launch(f, &cfg, s, &mut args)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TvParams {
    tmap: u64,
    sfa: u64,
    x: u64,
    prenorm_w: u64,
    norm_part: u64,
    out: u64,
    out2: u64,
    rope_cos: u64,
    rope_sin: u64,
    meta: u64,
    alpha: f32,
    eps: f32,
    m: i32,
    n: i32,
    k: i32,
    kpt: i32,
    ldx: i32,
    ldo: i32,
    ldo2: i32,
    n_split: i32,
    out_f32: i32,
    nslices: i32,
    rope_head: i32,
    rope_off: i32,
    partial: i32,
    group_rows: i32,
    kvr: RopeParams,
    kvr_on: i32,
    _pad2: i32,
}

pub fn tc_gemv_cluster(n: usize, k: usize) -> Option<usize> {
    [8usize, 5, 4, 2, 1].into_iter().find(|&c| k % (128 * c) == 0 && k / c <= 1024 && (n / 128) * c <= sm_count()).filter(|&c| (n / 128) * c >= 96)
}

pub fn tc_partial_cluster(n: usize, k: usize) -> Option<usize> {
    [2usize, 4, 5, 8].into_iter().find(|&c| k % (128 * c) == 0 && k / c <= 1024 && (n / 128) * c >= 64 && (n / 128) * c <= sm_count())
}

pub fn tc_push_shard(m: usize, cs: usize) -> usize {
    ((m * 128).div_ceil(cs) + 3) & !3
}

#[allow(clippy::too_many_arguments)]
pub fn tc_gemv(k: &Kernels, s: &Stream, tmap128: DevPtr, sf: DevPtr, w: &Weight, x: DevPtr, ldx: usize, m: usize, out: DevPtr, ldo: usize, o: &GemvOpts, meta: DevPtr) -> Result<usize> {
    ensure!(m <= 8 && w.n % 128 == 0);
    let cs = (if o.epilogue == Epilogue::Partial { tc_partial_cluster(w.n, w.k) } else { tc_gemv_cluster(w.n, w.k) }).ok_or_else(|| anyhow::anyhow!("tc_gemv: no split for k {}", w.k))?;
    let ks = w.k / cs;
    let kt = ks / 128;
    let smem = kt * (16384 + 1024 + 512 + 512) + 8 * ks * 2 + ks * 2 + 8 * 128 * 4 + 1040 * 4 + 1024;
    let f = k.get("tc_gemv_fp8")?;
    f.set_max_dynamic_smem(smem as u32)?;
    let p = TvParams {
        tmap: tmap128,
        sfa: sf,
        x,
        prenorm_w: o.prenorm.map(|v| v.0).unwrap_or(0),
        norm_part: o.norm_part.map(|v| v.0).unwrap_or(0),
        out,
        out2: o.split.map(|v| v.1).unwrap_or(0),
        rope_cos: o.rope.map(|v| v.0).unwrap_or(0),
        rope_sin: o.rope.map(|v| v.1).unwrap_or(0),
        meta,
        alpha: o.alpha,
        eps: o.prenorm.map(|v| v.1).unwrap_or(0.0),
        m: m as i32,
        n: w.n as i32,
        k: w.k as i32,
        kpt: (w.k / 128) as i32,
        ldx: ldx as i32,
        ldo: ldo as i32,
        ldo2: o.split.map(|v| v.2 as i32).unwrap_or(0),
        n_split: o.split.map(|v| v.0 as i32).unwrap_or(0),
        out_f32: (o.out_f32 || o.epilogue == Epilogue::Partial) as i32,
        nslices: o.norm_part.map(|v| v.1 as i32).unwrap_or(0),
        rope_head: o.rope.map(|v| v.3 as i32).unwrap_or(0),
        rope_off: o.rope.map(|v| v.4 as i32).unwrap_or(0),
        partial: (o.epilogue == Epilogue::Partial) as i32,
        group_rows: 0,
        kvr: o.kv_rope.unwrap_or_default(),
        kvr_on: o.kv_rope.is_some() as i32,
        _pad2: 0,
    };
    ensure!(o.kv_rope.is_none() || o.rope.is_some(), "kv_rope needs the fused rope (its meta)");
    let gx = (w.n / 128) as u32 + o.kv_rope.is_some() as u32;
    let mut cfg = Launch::new(gx, 128).grid3((gx, cs as u32, 1)).smem(smem as u32).pdl(o.pdl);
    cfg.cluster = Some((1, cs as u32, 1));
    cu::launch(f, &cfg, s, Args::new().push(p))?;
    Ok(cs)
}

const _: () = assert!(std::mem::size_of::<TvParams>() == 256);
const _: () = assert!(std::mem::size_of::<GemvParams>() == 432 && std::mem::size_of::<W2Params>() == 72);

pub fn tcgemv2_on() -> bool {
    deepseek_ultrafast_cuda::profile::sm103()
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Tg2Params {
    pub tmap: u64,
    pub sfa: u64,
    pub x: u64,
    pub gamma: u64,
    pub bimg: u64,
    pub rs_part: u64,
    pub out: u64,
    pub out2: u64,
    pub rope_cos: u64,
    pub rope_sin: u64,
    pub meta: u64,
    pub qimg: u64,
    pub qgamma: u64,
    pub qssq: u64,
    pub kvr: RopeParams,
    pub alpha: f32,
    pub eps: f32,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub kpt: i32,
    pub ldx: i32,
    pub ldo: i32,
    pub ldo2: i32,
    pub n_split: i32,
    pub out_f32: i32,
    pub bsrc: i32,
    pub rs_n: i32,
    pub rs_stride: i32,
    pub rs_rowstride: i32,
    pub rs_off: i32,
    pub rope_head: i32,
    pub rope_off: i32,
    pub iq_head: i32,
    pub iq_off: i32,
    pub partial: i32,
    pub group_rows: i32,
    pub qrows: i32,
    pub kvr_on: i32,
    pub xq: u64,
    pub xsq: u64,
    pub ssp: i32,
    pub rs_pre: i32,
    pub nacc: i32,
    pub bload: i32,
    pub flags: i32,
}
const _: () = assert!(std::mem::size_of::<Tg2Params>() == 352);

pub fn tg2_gate_cluster(n: usize, k: usize) -> Option<usize> {
    tg2_cluster_bf16(n, k, 10)
}

pub fn tg2_carveout(name: &str) -> Option<i32> {
    static C: std::sync::OnceLock<(Option<i32>, Vec<String>)> = std::sync::OnceLock::new();
    let (p, only) = C.get_or_init(|| {
        if !deepseek_ultrafast_cuda::profile::sm103() {
            return (None, Vec::new());
        }
        (Some(100), vec!["qb".to_string()])
    });
    p.filter(|_| only.is_empty() || only.iter().any(|o| name.contains(o.as_str())))
}

pub fn tcg3_on() -> bool {
    deepseek_ultrafast_cuda::profile::sm103()
}

pub fn tcg3_entry(name: &'static str) -> &'static str {
    if !tcg3_on() {
        return name;
    }
    match name {
        "tc_gemv2_qkv" => "variant_zzzzzy3/tc_gemv3b_qkv",
        "tc_gemv2_qb" => "variant_zzzzzy3/tc_gemv3b_qb",
        "tc_gemv2_part" => "variant_zzzzzy3/tc_gemv3b_part",
        "tc_gemv2_oa" => "variant_zzzzzy3/tc_gemv3b_oa",
        _ => name,
    }
}

pub fn tg2_img_bytes(k: usize) -> usize {
    k / 128 * 1536
}

pub fn tg2_cluster_fp8(n: usize, k: usize, extra: usize) -> Option<usize> {
    if n % 128 != 0 {
        return None;
    }
    [8usize, 5, 4, 2, 1].into_iter().find(|&c| k % (128 * c) == 0 && k / c <= 1024 && (n / 128 + extra) * c <= sm_count()).or_else(|| {
        let c = [8usize, 5, 4, 2].into_iter().find(|&c| k % (128 * c) == 0 && k / c <= 640 && (n / 128 + extra) * c <= 2 * sm_count())?;
        let mut d = TG2_2SM.lock().unwrap();
        if !d.contains(&(n, k, c)) {
            d.push((n, k, c));
        }
        Some(c)
    })
}

static TG2_2SM: std::sync::Mutex<Vec<(usize, usize, usize)>> = std::sync::Mutex::new(Vec::new());

pub fn tg2_2sm_resolved() -> String {
    let mut d = TG2_2SM.lock().unwrap().clone();
    d.sort();
    if d.is_empty() {
        return "-".into();
    }
    d.iter().map(|(n, k, c)| format!("{n}x{k}/{c}")).collect::<Vec<_>>().join(",")
}

pub fn tg2_cluster_bf16(n: usize, k: usize, max_boxes: usize) -> Option<usize> {
    if n % 128 != 0 {
        return None;
    }
    [8usize, 4, 2, 1].into_iter().find(|&c| k % (64 * c) == 0 && k / c <= 64 * max_boxes.min(10) && (n / 128) * c <= sm_count())
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AaPush {
    pub recv: [u64; 4],
    pub state: u64,
    pub meta: u64,
    pub cap: i32,
    pub rank: i32,
}
const _: () = assert!(std::mem::size_of::<AaPush>() == 56);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AaPush8 {
    pub recv: [u64; 8],
    pub state: u64,
    pub meta: u64,
    pub cap: i32,
    pub rank: i32,
}
const _: () = assert!(std::mem::size_of::<AaPush8>() == 88);

#[derive(Clone, Copy, Debug)]
pub enum AaArm {
    Push(AaPush),
    Push8(AaPush8),
}

pub fn tc_gemv2_aa(k: &Kernels, s: &Stream, fp8: bool, mut p: Tg2Params, cs: usize, pdl: bool, aa: Option<AaArm>) -> Result<(usize, bool)> {
    let (n, kk, m) = (p.n as usize, p.k as usize, p.m as usize);
    ensure!((1..=8).contains(&m) && n % 128 == 0 && kk % (cs * if fp8 { 128 } else { 64 }) == 0, "tc_gemv2: m {m}, n {n}, k {kk}, cs {cs}");
    let ks = kk / cs;
    let kt = ks / if fp8 { 128 } else { 64 };
    ensure!(kt >= 1 && kt <= if fp8 { 8 } else { 10 } && cs <= 16, "tc_gemv2: {kt} weight boxes per CTA (cs {cs})");
    ensure!(p.rs_pre == 0 || (!fp8 && p.gamma != 0 && p.rs_part != 0 && p.rs_pre <= 2), "tc_gemv2: rs_pre is the BF16 prenorm staging");
    ensure!(p.xq == 0 || (!fp8 && p.xsq != 0 && p.group_rows == 0), "tc_gemv2: the MoE side output is BF16 (dense)");
    ensure!(p.qimg == 0 || (p.qrows as usize % 128 == 0 && p.qrows > 0), "tc_gemv2: qrows {} (whole 128-row tiles)", p.qrows);
    ensure!(p.n_split as usize % 32 == 0 && (p.rope_cos == 0 || (p.rope_head % 32 == 0 && p.rope_off % 32 == 0 && p.meta != 0)), "tc_gemv2: split / rope granularity");
    ensure!(p.iq_head == 0 || (p.iq_head % 32 == 0 && p.iq_off % 32 == 0 && p.rope_cos != 0 && p.n_split > 0), "tc_gemv2: iq rows");
    ensure!(p.rs_part == 0 || p.rs_n <= 64, "tc_gemv2: at most 64 row-statistic partials");
    ensure!(p.partial == 0 || (p.rs_part == 0 && p.qimg == 0 && p.n_split == 0 && p.rope_cos == 0), "tc_gemv2: partial rows have no epilogue");
    p.kpt = (kk / 128) as i32;
    p.nacc = 1;
    let z = |v: u64| v == 0;
    let common = p.ssp == 0 && p.out_f32 == 0 && p.out != 0 && cs > 1 && cs <= 8;
    let name = if fp8 && p.partial == 1 && z(p.rs_part) && z(p.qimg) && p.n_split == 0 && z(p.rope_cos) && p.kvr_on == 0 && p.bsrc == 1 && p.out != 0 {
        "tc_gemv2_part"
    } else if fp8 && common && p.partial == 0 && p.bsrc == 0 && !z(p.qimg) && p.n_split > 0 && z(p.rope_cos) && p.iq_head == 0 && p.kvr_on == 0 {
        "tc_gemv2_qkv"
    } else if fp8 && common && p.partial == 0 && p.bsrc == 1 && z(p.qimg) && z(p.gamma) && (z(p.rope_cos) || (p.rope_head as u32).is_power_of_two()) && (p.iq_head == 0 || (p.iq_head as u32).is_power_of_two()) {
        "tc_gemv2_qb"
    } else if !fp8 && common && !z(p.qimg) && z(p.qgamma) && z(p.qssq) && z(p.gamma) && z(p.rs_part) && p.n_split == 0 && z(p.rope_cos) && z(p.xq) {
        "tc_gemv2_oa"
    } else if !fp8 && p.ssp == 1 && p.out != 0 && cs > 1 && cs <= 8 && !z(p.gamma) && p.rs_pre >= 1 && z(p.qimg) && p.n_split == 0 && p.group_rows == 0 && z(p.rope_cos) {
        "tc_gemv2_gate"
    } else if fp8 {
        "tc_gemv2_fp8"
    } else {
        "tc_gemv2_bf16"
    };
    let aa = aa.filter(|_| {
        (cs == 4 || (cs == 2 && (p.m as usize) <= 6) || (cs == 1 && matches!(aa, Some(AaArm::Push8(_))))) && tcg3_entry(name) == "variant_zzzzzy3/tc_gemv3b_part"
    });
    let aa_push = aa.is_some();
    let recv = if cs > 1 && (p.partial == 0 || aa_push) { (32usize).div_ceil(cs) * cs * 128 } else { 0 };
    let xraw = if !fp8 && p.rs_pre == 2 && p.xq != 0 { 8 * ks * 2 } else { 0 };
    let smem = 1024 + kt * 16384 + if fp8 { kt * 1024 + kt * 1024 } else { ks * 16 } + if p.gamma != 0 { ks * 2 } else { 0 } + xraw + recv;
    ensure!(smem + 8192 <= hw().smem_optin, "tc_gemv2: {smem} B of shared memory");
    let name = tcg3_entry(name);
    let gx_all = (n / 128 + p.kvr_on as usize) * cs;
    let name = if gx_all > sm_count() && name == "variant_zzzzzy3/tc_gemv3b_qb" { "variant_zzzzzy3/tc_gemv3yr_qb" } else { name };
    let name = match aa {
        None => name,
        Some(AaArm::Push(_)) => "variant_zzzzzzaa/tc_gemv3yp_part",
        Some(AaArm::Push8(_)) => "variant_zzzzzzaa/tc_gemv3yp8_part",
    };
    if name.starts_with("variant_zzzzzy3/tc_gemv3yr") || name.starts_with("variant_zzzzzzaa/tc_gemv3y") {
        let fits = (334_848 / (smem + 8192 + 1024)).min(2048 / 128);
        let cols = if fp8 { 128 } else { 32 };
        ensure!(fits * cols <= 512, "tc_gemv2: {name}: {fits} CTAs per SM x {cols} TMEM columns > 512 (TG3_REL could deadlock in tcgen05.alloc)");
    }
    let f = k.get(name)?;
    f.set_max_dynamic_smem(smem as u32)?;
    if let Some(c) = tg2_carveout(name) {
        f.set_carveout(c)?;
    } else if tcg3_on() && (name.starts_with("variant_zzzzzy3/") || name.starts_with("variant_zzzzzzaa/")) && ["_qkv", "_qb", "_oa", "_part"].iter().any(|u| name.ends_with(u)) {
        f.set_carveout(100)?;
    }
    if cs > 8 {
        f.set_non_portable_cluster()?;
    }
    let gx = (n / 128) as u32 + p.kvr_on as u32;
    let mut cfg = Launch::new(gx, 128).grid3((gx, cs as u32, 1)).smem(smem as u32).pdl(pdl);
    cfg.cluster = Some((1, cs as u32, 1));
    let mut a = Args::new();
    a.push(p);
    match aa {
        Some(AaArm::Push(q)) => {
            a.push(q);
        }
        Some(AaArm::Push8(q)) => {
            a.push(q);
        }
        _ => {}
    }
    cu::launch(f, &cfg, s, &mut a)?;
    Ok((cs, aa.is_some()))
}

pub const WOA64_CS: usize = 4;
pub fn tc_gemv2_oa64(k: &Kernels, s: &Stream, mut p: Tg2Params, tmap64: u64, pdl: bool) -> Result<()> {
    let (n, kk, m) = (p.n as usize, p.k as usize, p.m as usize);
    let cs = WOA64_CS;
    let ks = kk / cs;
    let kt = ks / 64;
    ensure!((1..=8).contains(&m) && n % 64 == 0 && kk % (cs * 64) == 0 && (1..=16).contains(&kt), "tc_gemv3m_oa: m {m}, n {n}, k {kk}");
    ensure!(p.group_rows == 0 || p.group_rows % 64 == 0, "tc_gemv3m_oa: group rows {}", p.group_rows);
    ensure!(p.qimg == 0 || (p.qrows as usize % 64 == 0 && p.qrows > 0), "tc_gemv3m_oa: qrows {}", p.qrows);
    ensure!(p.gamma == 0 && p.rs_part == 0 && p.qgamma == 0 && p.qssq == 0 && p.n_split == 0 && p.rope_cos == 0 && p.xq == 0 && p.partial == 0 && p.bimg == 0, "tc_gemv3m_oa: wo_a options only");
    p.tmap = tmap64;
    p.kpt = (kk / 128) as i32;
    p.nacc = 1;
    let recv = (16usize).div_ceil(cs) * cs * 128;
    let smem = 1024 + kt * 8192 + ks * 16 + recv;
    ensure!(smem + 8192 <= hw().smem_optin, "tc_gemv3m_oa: {smem} B of shared memory");
    let f = k.get("variant_zzzztg3/tc_gemv3m_oa")?;
    f.set_max_dynamic_smem(smem as u32)?;
    let gx = (n / 64) as u32;
    let mut cfg = Launch::new(gx, 128).grid3((gx, cs as u32, 1)).smem(smem as u32).pdl(pdl);
    cfg.cluster = Some((1, cs as u32, 1));
    cu::launch(f, &cfg, s, Args::new().push(p))
}

pub fn woa64_on(k: &Kernels, n: usize, kk: usize, cs128: usize) -> bool {
    let key = (n, kk, cs128);
    if let Some(&(_, v)) = WOA64_DECISIONS.lock().unwrap().iter().find(|(c, _)| *c == key) {
        return v;
    }
    let v = woa64_decide(k, n, kk, cs128);
    let mut d = WOA64_DECISIONS.lock().unwrap();
    if let Some(&(_, r)) = d.iter().find(|(c, _)| *c == key) {
        return r;
    }
    d.push((key, v));
    v
}

pub fn woa64_resolved() -> String {
    woa64_profile(WOA64_DECISIONS.lock().unwrap().clone())
}

fn woa64_profile(mut d: Vec<((usize, usize, usize), bool)>) -> String {
    d.sort();
    if d.is_empty() {
        return "-".into();
    }
    d.iter().map(|((n, kk, cs), v)| format!("{n}x{kk}/{cs}:{}", *v as u8)).collect::<Vec<_>>().join(",")
}

static WOA64_DECISIONS: std::sync::Mutex<Vec<((usize, usize, usize), bool)>> = std::sync::Mutex::new(Vec::new());

fn woa64_decide(k: &Kernels, n: usize, kk: usize, cs128: usize) -> bool {
    if n % 128 != 0 || cs128 == 0 || kk % (WOA64_CS * 64) != 0 || kk / WOA64_CS / 64 > 16 {
        return false;
    }
    let fits = |name: &str, cs: usize, smem: usize| -> Option<usize> {
        let f = k.get(name).ok()?;
        f.set_max_dynamic_smem(smem as u32).ok()?;
        let mut cfg = Launch::new(1, 128).grid3((1, cs as u32, 1)).smem(smem as u32);
        cfg.cluster = Some((1, cs as u32, 1));
        cu::max_active_clusters(f, &cfg).ok().map(|v| v as usize)
    };
    let smem128 = 1024 + (kk / cs128 / 64) * 16384 + (kk / cs128) * 16 + (32usize).div_ceil(cs128) * cs128 * 128;
    let smem64 = 1024 + (kk / WOA64_CS / 64) * 8192 + (kk / WOA64_CS) * 16 + (16usize).div_ceil(WOA64_CS) * WOA64_CS * 128;
    let v = match (fits(tcg3_entry("tc_gemv2_oa"), cs128, smem128), fits("variant_zzzztg3/tc_gemv3m_oa", WOA64_CS, smem64)) {
        (Some(f128), Some(f64_)) => f128 < n / 128 && f64_ >= n / 64,
        _ => false,
    };
    deepseek_ultrafast_logging::diagnostic!("wo_a {n}x{kk}: 64-row tiles {}", if v { "on" } else { "off" });
    v
}

const _: () = assert!(std::mem::size_of::<GemvParamsXb>() == 440 && std::mem::size_of::<TmRmParams>() == 136);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MpParams {
    pub ids: u64,
    pub w: u64,
    pub m: i32,
    pub topk: i32,
    pub n_exp: i32,
    pub _pad: i32,
    pub row_pair: u64,
    pub row_w: u64,
    pub pair_row: u64,
    pub row_tile: u64,
    pub tiles: u64,
    pub ntiles: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MqParams {
    pub x: u64,
    pub ldx: i32,
    pub m: i32,
    pub k: i32,
    pub topk: i32,
    pub pw: u64,
    pub norm_part: u64,
    pub nslices: i32,
    pub eps: f32,
    pub pair_row: u64,
    pub row_tile: u64,
    pub xq: u64,
    pub sf: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TmParams {
    pub w: u64,
    pub w_stride: u64,
    pub k: i32,
    pub rt: i32,
    pub tb: u64,
    pub sfb: u64,
    pub tiles: u64,
    pub ntiles: u64,
    pub row_pair: u64,
    pub row_w: u64,
    pub aq: u64,
    pub asf: u64,
    pub ldaq: i32,
    pub akt: i32,
    pub limit: f32,
    pub nq: i32,
    pub out: u64,
    pub ldo: i32,
    pub flags: i32,
    pub sf_off: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TmRmParams {
    pub p: TmParams,
    pub wmap: u64,
}

pub const TM_SMEM: u32 = (4 * 16384 + 2 * (16 * 2080 + 4096) + 3 * 8 * 2176 + 40 * 512 + 2 * 32 * 33 * 4 + 1024) as u32;
pub const TM_NT: usize = 128;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TgParams {
    pub ta: u64,
    pub tb: u64,
    pub sfa: u64,
    pub sfb: u64,
    pub out: u64,
    pub ldo: i32,
    pub out_f32: i32,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub nt: i32,
    pub alpha: f32,
    pub _pad: i32,
    pub out2: u64,
    pub ldo2: i32,
    pub n_split: i32,
    pub swiglu: i32,
    pub swiglu_limit: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TqParams {
    pub x: u64,
    pub ldx: i32,
    pub m: i32,
    pub k: i32,
    pub _pad: i32,
    pub pw: u64,
    pub norm_part: u64,
    pub nslices: i32,
    pub eps: f32,
    pub aq: u64,
    pub sfa: u64,
}

pub const TGP_SMEM: u32 = 6 * 34816 + 1024;

pub fn tg_smem(nt: usize) -> u32 {
    let sb = (16384 + nt * 128 + 512 + (nt / 128).max(1) * 512 + 1023) & !1023;
    (4 * sb + 1024) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TbParams {
    pub ta: u64,
    pub tb: u64,
    pub out: u64,
    pub ldo: i32,
    pub out_f32: i32,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub nt: i32,
    pub alpha: f32,
    pub epi: i32,
    pub a_group_rows: i32,
    pub a_group_k: i32,
}

pub fn tb_smem(nt: usize) -> u32 {
    let sb = (16384 + nt * 128 + 1023) & !1023;
    (4 * sb + 1024) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct IxParams {
    pub tk: u64,
    pub tq: u64,
    pub w: u64,
    pub ldw: i32,
    pub _pad: i32,
    pub scores: u64,
    pub ld: i32,
    pub ratio: i32,
    pub tmax: i32,
    pub pos: i32,
    pub m: i32,
    pub _pad2: i32,
    pub cand: u64,
    pub ld_cand: i32,
    pub cand_block: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Ix4Params {
    pub tk: u64,
    pub ksf: u64,
    pub tq: u64,
    pub qsf: u64,
    pub w: u64,
    pub ldw: i32,
    pub _pad: i32,
    pub scores: u64,
    pub ld: i32,
    pub ratio: i32,
    pub tmax: i32,
    pub pos: i32,
    pub m: i32,
    pub flags: i32,
    pub cand: u64,
    pub ld_cand: i32,
    pub cand_block: i32,
}
pub const IX4B_SMEM: u32 = 1024 + 20 * 8704 + 16 * 4 * 32 * 4 + 16 * 4 * 4;
pub const IX4B_THREADS: u32 = 18 * 32;

pub const IXD_SMEM: u32 = 65536 + 3 * 32768 + 8 * 32 * 4 + 1024;

#[cfg(test)]
mod staging_tests {
    fn e4m3(b: u8) -> Option<f64> {
        let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
        let e = ((b >> 3) & 0xf) as i32;
        let m = (b & 7) as f64;
        if e == 15 && m == 7.0 { return None; }
        Some(if e == 0 { s * m / 8.0 * 2f64.powi(-6) } else { s * (1.0 + m / 8.0) * 2f64.powi(e - 7) })
    }

    #[test]
    fn dequantized_staging_contract() {
        for b in 0..=255u8 {
            let Some(v) = e4m3(b) else { continue };
            for k in -30..=15i32 {
                let sc = 2f64.powi(k);
                let exact = v * sc;
                let staged = half::f16::from_f32((v as f32) * (sc as f32)).to_f64();
                if (-15..=7).contains(&k) {
                    assert_eq!(staged, exact, "e4m3 {b:#x} x 2^{k}: {staged} != {exact}");
                } else if k < -15 {
                    assert!((staged - exact).abs() <= 2f64.powi(-25), "e4m3 {b:#x} x 2^{k}: error {}", (staged - exact).abs());
                } else if exact.abs() <= 65504.0 {
                    assert_eq!(staged, exact);
                } else {
                    assert!(staged.is_infinite() && (staged > 0.0) == (exact > 0.0), "e4m3 {b:#x} x 2^{k}: {staged}");
                }
            }
        }
    }

    #[test]
    fn block_partial_scales_exactly() {
        let w = [0.75f32, -1.5, 0.5, 6.0, -0.5, 1.0, 1.5, -6.0, 0.5, 0.5, -1.5, 4.0, 0.0, 2.0, -0.5, 3.0];
        let x = [448.0f32, -0.0625, 1.0, -2.0, 0.125, 104.0, -7.5, 0.5, 0.25, -3.0, 9.0, 0.03125, 1.0, -1.0, 5.5, -0.25];
        for k in -15..=7i32 {
            let sc = 2f32.powi(k);
            let (mut a, mut b) = (0f32, 0f32);
            for i in 0..16 {
                a = a + w[i] * x[i];
                b = b + w[i] * (x[i] * sc);
            }
            assert_eq!(b, a * sc, "k {k}");
        }
    }
}

#[cfg(test)]
mod tc_gemv_tests {
    use super::*;

    #[test]
    fn woa64_profile_lists_the_resolved_tilings() {
        assert_eq!(woa64_profile(vec![]), "-");
        assert_eq!(woa64_profile(vec![((2048, 4096, 8), true)]), "2048x4096/8:1");
        assert_eq!(woa64_profile(vec![((2048, 4096, 8), false)]), "2048x4096/8:0");
        assert_eq!(woa64_profile(vec![((4096, 4096, 8), false), ((2048, 4096, 8), true)]), "2048x4096/8:1,4096x4096/8:0");
    }

    #[test]
    fn push_shards_are_float4_multiples_within_the_recv_buffer() {
        for m in 1..=8usize {
            for cs in [1usize, 2, 4, 5, 8] {
                let per = tc_push_shard(m, cs);
                assert_eq!(per % 4, 0, "m {m} cs {cs}");
                assert!(per * cs >= m * 128, "m {m} cs {cs}: the shards cover every output");
                assert!(per * cs <= 1040, "m {m} cs {cs}: recv [cs][per] overflows");
                assert_eq!((cs - 1) * (per / 4) * 16, (cs - 1) * per * 4);
            }
        }
        assert_eq!(tc_push_shard(8, 5), 208);
    }

    #[test]
    fn partial_selector_covers_tp4_wo_b() {
        assert_eq!(tc_partial_cluster(5120, 2048), Some(2));
        assert_eq!(tc_gemv_cluster(5120, 2048), None);
        assert_eq!(tc_gemv_cluster(1792, 5120), Some(8));
        assert_eq!(tc_gemv_cluster(8192, 1280), Some(2));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KB: usize = 1024;

    fn shape_before(k13: usize, k2: usize, optin: usize) -> Option<(usize, usize)> {
        let mt = 8;
        let sq_ld = moe_fused_sq_ld(k2);
        let ns2 = k2 / 64;
        let fixed = mt * (k13 + 32) * 2 + mt * (k13 / 32) * 4 + 4 * mt * 64 * 4 + 2 * mt * (k2 + 32) * 2 + 2 * mt * sq_ld * 4 + 2 * (STAGES + 4) * 8;
        let stat = SPLAN_BYTES + 2 * 4 * 32 * 4 * 4 + 512;
        let ss = (k13 / 64).min(12).min((optin - fixed - stat) / (STAGES * 2176));
        (ss * 2176 >= FP8_STEP64 && ss * 2176 >= 2 * ns2 * 2176 && ss * 2176 >= ns2 * FP8_STEP64).then_some((ss, fixed + STAGES * ss * 2176))
    }

    fn budget_underflows(k13: usize, k2: usize, optin: usize) -> bool {
        let mt = 8;
        let fixed = mt * (k13 + 32) * 2 + mt * (k13 / 32) * 4 + 4 * mt * 64 * 4 + 2 * mt * (k2 + 32) * 2 + 2 * mt * moe_fused_sq_ld(k2) * 4 + 2 * (STAGES + 4) * 8;
        fixed + SPLAN_BYTES + 2 * 4 * 32 * 4 * 4 + 512 > optin
    }

    #[test]
    fn fused_shape_unchanged() {
        for k13 in [2048, 4096, 5120, 7168] {
            for k2 in (64..=2304).step_by(64) {
                for optin in [160 * KB, 200 * KB, 227 * KB] {
                    if cfg!(debug_assertions) && budget_underflows(k13, k2, optin) {
                        continue;
                    }
                    let old = shape_before(k13, k2, optin);
                    let new = fused_shape_for(k13, k2, optin);
                    match old {
                        Some(_) => assert_eq!(old, new.map(|s| (s.ss, s.smem)), "k13 {k13} k2 {k2} optin {optin}"),
                        None => assert!(new.is_none_or(|s| s.qps == 1), "k13 {k13} k2 {k2} optin {optin}"),
                    }
                    assert!(new.is_none_or(|s| (s.qps == 2) == old.is_some()));
                }
            }
        }
    }
}
