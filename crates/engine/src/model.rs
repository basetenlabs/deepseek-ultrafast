use crate::ops::{self, Epilogue, GemvOpts, Moe, RopeParams, WFmt, Weight};
use crate::sampling::{SampIo, SampParams};
use crate::weights::{Arena, Layer, Weights};
use anyhow::{Result, ensure};
use deepseek_ultrafast_cuda::{self as cu, Args, DevPtr, Launch, Stream};
use deepseek_ultrafast_kernels::Kernels;
use deepseek_ultrafast_model::config::{Config, rope_table};
use deepseek_ultrafast_model::packing::sgl_w2_row_bytes;

pub use deepseek_ultrafast_model::limits::{MAX_M, OUT_CAP};
const MOE_PLAN_SLOTS: usize = 97;

pub const PMAX: usize = 8192;
const _: () = assert!(PMAX % MAX_M == 0);
pub const MAX_TP: usize = 8;

pub fn prefill_capacity(chunk: usize, window: usize) -> usize {
    chunk.max(window).clamp(MAX_M, PMAX).next_multiple_of(MAX_M)
}

pub fn meta_slices(pmax: usize) -> usize {
    pmax.div_ceil(MAX_M)
}
const COMM_MAX_PER_TOKEN: usize = 6400;
const COMM_MAX_BLOCKS: usize = (if PMAX > MAX_M { PMAX } else { MAX_M }) * COMM_MAX_PER_TOKEN / 256;
const COMM_SHORT_ROWS: usize = 512;
const COMM_SHORT_CAP: usize = COMM_SHORT_ROWS * 6400;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Meta {
    pub pos: i32,
    pub m: i32,
    pub tokens: [i32; 16],
}

pub struct CommRank {
    pub recv: DevPtr,
    pub state: DevPtr,
    pub state2: DevPtr,
    pub cap: usize,
    pub off_b: u64,
    pub cap_b: usize,
    pub state_b: DevPtr,
    pub dir_off: u64,
    short_rs: u64,
    short_ag: u64,
    short_state: DevPtr,
}

impl CommRank {
    fn rs_off(tp: usize, cap: usize) -> u64 { (3 * tp * cap * 4) as u64 }
    fn ag_off(tp: usize, cap: usize) -> u64 { (2 * 3 * tp * cap * 4) as u64 }
    fn dg_off(tp: usize, cap: usize) -> u64 { (Self::ag_off(tp, cap) + (3 * cap * 2) as u64).next_multiple_of(256) }
    fn samp_off(tp: usize, cap: usize) -> u64 { (Self::dg_off(tp, cap) + 512).next_multiple_of(256) }
    const SAMP_SO_S2: usize = 2 * 8 * 8 * 99 * 4;
    const SAMP_SO_FR: usize = Self::SAMP_SO_S2 + 128;
    const SAMP_SO_BV: usize = Self::SAMP_SO_FR + 3 * 8 * 8 * 256 * 4 + 8 * 8 * 2048 * 4;
    fn samp_bytes(_vpad: usize) -> u64 { ((Self::SAMP_SO_BV + 2 * 8 * 8 * 3 * 4) * 4) as u64 }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct EgPeers {
    table: [u64; MAX_TP],
    scales: [u64; MAX_TP],
    part: [u64; MAX_TP],
    flag: [u64; MAX_TP],
    epoch: u64,
    shard: i64,
    tp: i32,
    nloc: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CommParams {
    recv: [u64; MAX_TP],
    state: u64,
    tp: i32,
    rank: i32,
    cap: i32,
    src: u64,
    ld_src: i32,
    n_per_token: i32,
    mode: i32,
    round_bf16: i32,
    out: u64,
    ld_out: i32,
    out_f32: i32,
    res: u64,
    post: u64,
    comb: u64,
    nsum: i32,
    src2: u64,
    epi: i32,
    e: HcEpi,
    mx: HcMix,
    mx_on: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NvlsParams {
    c: CommParams,
    mixf: u64,
    sg_perm: u64,
    sg_w: u64,
    sg_rec: u64,
    sg_gamma: u64,
    sg_img: u64,
    sg_early: i32,
    _sgpad: i32,
}

const NVLS_CAP: usize = MAX_M * 6400 + 1024;
const NVLS_FLAGS: usize = 4096;

#[derive(Clone, Copy)]
pub struct NvlsBufs {
    pub state: DevPtr,
    pub recv: [u64; MAX_TP],
}

static NVLS_UC: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());

const FFN_LP_ROWS: usize = 8 * 8;

fn ffn_lp_kernel(n: usize, blocks: usize) -> Option<(String, usize, u32)> {
    (matches!(n, 3 | 6) && blocks % 2 == 0).then(|| (format!("variant_zzzzzzzzzlp/comm_lp_ffn_{n}_2zp"), blocks / 2, 160))
}

pub fn setup_nvls(rk: &mut Rank, gpus: &[i32], barrier: &std::sync::Barrier) -> Result<()> {
    if !deepseek_ultrafast_cuda::profile::sm103() || rk.tp < 2 {
        return Ok(());
    }
    ensure!(gpus.len() == rk.tp, "setup_nvls: {} gpus for TP {}", gpus.len(), rk.tp);
    let fill = rk.k.get("comm_fill_u32")?;
    let rbytes = 3 * rk.tp * NVLS_CAP * 4;
    let buf = cu::Buffer::zeroed(rbytes + NVLS_FLAGS * 16)?;
    cu::launch(fill, &Launch::new(1024, 256), &rk.stream, Args::new().ptr(buf.ptr).push((3 * rk.tp * NVLS_CAP) as u64).u32(0x8000_0000))?;
    rk.stream.synchronize()?;
    if rk.rank == 0 {
        *NVLS_UC.lock().unwrap() = vec![0; rk.tp];
    }
    barrier.wait();
    NVLS_UC.lock().unwrap()[rk.rank] = buf.ptr;
    barrier.wait();
    let mut recv = [0u64; MAX_TP];
    for (i, &a) in NVLS_UC.lock().unwrap().iter().enumerate() {
        recv[i] = a;
    }
    rk.nvls = Some(NvlsBufs { state: buf.ptr + rbytes as u64, recv });
    rk.nvls_hold = Some(buf);
    barrier.wait();
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MkParams {
    w: u64,
    nquads: i32,
    qpc: i32,
    vloc: i32,
    vpad: i32,
    offset: i32,
    _pad0: i32,
    table: u64,
    logits: u64,
    drafts: u64,
    blk: i32,
    _pad: i32,
    cand: u64,
    bar: u64,
    recv: [u64; MAX_TP],
    tp: i32,
    rank: i32,
    seq: u64,
    cx: u64,
    cw: u64,
    ldx: i32,
    dim: i32,
    xpart: u64,
    conf: u64,
    theta: f32,
    _pad1: i32,
    hcond: u64,
    samp: u64,
    vq: u64,
    qstat: u64,
    theta_r: [f32; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Comm2Params {
    rs: [u64; MAX_TP],
    ag: [u64; MAX_TP],
    state: u64,
    tp: i32,
    rank: i32,
    cap: i32,
    src: u64,
    ld_src: i32,
    n_per_token: i32,
    mode: i32,
    round_bf16: i32,
    out: u64,
    ld_out: i32,
    out_f32: i32,
    res: u64,
    post: u64,
    comb: u64,
    nsum: i32,
    src2: u64,
    epi: i32,
    e: HcEpi,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HcEpi {
    fn_: u64,
    pre: u64,
    x_raw: u64,
    part: u64,
    dim: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HcMix {
    part: u64,
    nslices: i32,
    base: u64,
    scale: u64,
    pre: u64,
    post: u64,
    comb: u64,
    dim: i32,
    iters: i32,
    norm_eps: f32,
    hc_eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnFParams {
    q: u64,
    win: u64,
    comp: u64,
    topk: u64,
    ktop: i32,
    scale: f32,
    sink: u64,
    out: u64,
    cos: u64,
    sin: u64,
    r: i32,
    fixed_window: i32,
    nh: i32,
    pre: i32,
    win_floor: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct IdxParams {
    q: u64,
    w: u64,
    ldw: i32,
    kc: u64,
    scores: u64,
    ld: i32,
    ratio: i32,
    tmax: i32,
    cand: u64,
    ld_cand: i32,
    cand_block: i32,
    bf16_ref: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TopkParams {
    x: u64,
    ld: i32,
    ratio: i32,
    block: i32,
    k: i32,
    out: u64,
    mask: u64,
    ld_mask: i32,
    drop_neg_inf: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TopkPartParams {
    t: TopkParams,
    nsplit: i32,
    cap: i32,
    cv: u64,
    ci: u64,
    cn: u64,
    kth: u64,
}

const TK_S: usize = 4096;

pub fn topk_cluster_smem(n_cap: usize, ratio: usize, block: usize, cl: usize) -> u32 {
    let cached = (131_072 / ratio.max(1)).div_ceil(block.max(1)).min(n_cap);
    ((cached.div_ceil(cl.max(1)) + 3) & !3) as u32 * 4
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EngramHash {
    mult: u64,
    primes: u64,
    offsets: u64,
    max_ngram: i32,
    heads: i32,
    pad_id: i32,
}

const Q_NONE: i32 = 0;
const Q_FP8: i32 = 1;
const Q_FP4_E8M0: i32 = 2;
const Q_FP4_E4M3_16: i32 = 3;
const CM_SUM: i32 = 0;
const CM_GATHER: i32 = 1;
const CM_SUM_HCPOST: i32 = 2;
const CM_SRC16: i32 = 0x80;

fn shard_rows(rank: usize, m: usize, tp: usize) -> (usize, usize) {
    let b = |r: usize| if r == tp { m } else { (r * m / tp) & !3 };
    (b(rank), b(rank + 1))
}

pub struct LayerCache {
    pub win: DevPtr,
    pub comp: Option<DevPtr>,
    pub idx_k: Option<DevPtr>,
    pub idx_k4: Option<(DevPtr, DevPtr)>,
    pub kv_hist: Option<DevPtr>,
    pub sc_hist: Option<DevPtr>,
}

pub struct EngramTables {
    pub token_map: DevPtr,
    pub mult: DevPtr,
    pub primes: DevPtr,
    pub offsets: DevPtr,
    pub pad_id: i32,
}

#[derive(Clone, Copy)]
struct Bufs {
    h: [DevPtr; 2],
    x: DevPtr,
    dots: DevPtr,
    pre_a: DevPtr,
    post_a: DevPtr,
    comb_a: DevPtr,
    pre_f: DevPtr,
    post_f: DevPtr,
    comb_f: DevPtr,
    q_lat: DevPtr,
    qr: DevPtr,
    q: DevPtr,
    kv: DevPtr,
    kv_all: DevPtr,
    o: DevPtr,
    oa: DevPtr,
    ob: DevPtr,
    ob_part: DevPtr,
    iq: DevPtr,
    iq4: (DevPtr, DevPtr),
    iw: DevPtr,
    scores: DevPtr,
    bscores: DevPtr,
    cand: DevPtr,
    cand_ids: DevPtr,
    topk: DevPtr,
    ckv: DevPtr,
    csc: DevPtr,
    latent: DevPtr,
    ik: DevPtr,
    glog: DevPtr,
    glog_part: DevPtr,
    route_ids: DevPtr,
    route_w: DevPtr,
    plan: DevPtr,
    shared_plan: DevPtr,
    act: DevPtr,
    sact: DevPtr,
    xq: DevPtr,
    xsq: DevPtr,
    sched: DevPtr,
    moe_done: DevPtr,
    dg_seq: DevPtr,
    mk_cand: DevPtr,
    mk_bar: DevPtr,
    actq: DevPtr,
    asq: DevPtr,
    y: DevPtr,
    ypair: DevPtr,
    eg: DevPtr,
    ex: DevPtr,
    ekv_part: DevPtr,
    ekv: DevPtr,
    logits: DevPtr,
    amax_v: DevPtr,
    amax_i: DevPtr,
    amax_pack: DevPtr,
    amax_all: DevPtr,
    next: DevPtr,
    main_hidden: DevPtr,
    out_tokens: DevPtr,
    out_count: DevPtr,
    main_x: DevPtr,
    x_raw: DevPtr,
    part: DevPtr,
    block_kv: DevPtr,
    block_idx: DevPtr,
    drafts: DevPtr,
    spec_state: DevPtr,
    idx_okeys: DevPtr,
}

struct SglMoe {
    slots: DevPtr,
    counter: DevPtr,
    out_w: DevPtr,
    out_i: DevPtr,
    permuted: DevPtr,
    record: DevPtr,
    xq: DevPtr,
    xs: DevPtr,
    h: DevPtr,
    hs: DevPtr,
    clamp: DevPtr,
    clamp_v: f32,
    c: DevPtr,
    tm_xq: DevPtr,
    tm_h: DevPtr,
    metas: DevPtr,
    img: DevPtr,
    maps: std::collections::HashMap<DevPtr, [DevPtr; 4]>,
    pad: usize,
}
const SGL_TILES: usize = 8 * 6 + 1;
const SGL_C_ROWS: usize = 8 * SGL_TILES;

struct TcW13 {
    maps: std::collections::HashMap<DevPtr, DevPtr>,
    w2maps: std::collections::HashMap<DevPtr, DevPtr>,
    ximg: DevPtr,
    aimg: DevPtr,
    part: DevPtr,
    tcnt: DevPtr,
    sctr: DevPtr,
}

#[derive(Clone, Copy)]
struct Tg2Bufs {
    qimg: DevPtr,
    qssq: DevPtr,
    oaimg: DevPtr,
}

#[derive(Clone, Copy)]
struct PrefillMoe {
    row_pair: DevPtr,
    row_w: DevPtr,
    pair_row: DevPtr,
    row_tile: DevPtr,
    tiles: DevPtr,
    ntiles: DevPtr,
    xq: DevPtr,
    sf: DevPtr,
    aq: DevPtr,
    asf: DevPtr,
    tb13: DevPtr,
    tb2: DevPtr,
    max_tiles: usize,
    ldaq: usize,
    akt: usize,
    sms: u32,
}

pub struct Rank {
    pub rank: usize,
    pub tp: usize,
    sms: u32,
    pub pmax: usize,
    pub cfg: Config,
    pub stream: Stream,
    pub k: Kernels,
    pub w: Weights,
    pub vision: Option<crate::vision::Vision>,
    image_prefill: bool,
    pub arena: Arena,
    pub meta: DevPtr,
    metas: DevPtr,
    meta_step: DevPtr,
    ptoks: DevPtr,
    tk_scratch: (DevPtr, DevPtr, DevPtr, usize),
    tk_scratch_b: (DevPtr, DevPtr, DevPtr, usize),
    tk_gather: (DevPtr, DevPtr),
    hc_fnb: std::collections::HashMap<DevPtr, DevPtr>,
    pub max_seq: usize,
    pub caches: Vec<LayerCache>,
    pub mtp_caches: Vec<LayerCache>,
    pub cmp_hist: DevPtr,
    rope_base: (DevPtr, DevPtr),
    rope_comp: (DevPtr, DevPtr),
    pub engram: EngramTables,
    pub comm: CommRank,
    pub peers: Vec<DevPtr>,
    pub nvls: Option<NvlsBufs>,
    nvls_hold: Option<cu::Buffer>,
    aa_pending: std::cell::Cell<bool>,
    ffn_lp_buf: Option<cu::Buffer>,
    lp_pending: std::cell::Cell<bool>,
    lp_mixf: DevPtr,
    lp_mixf_on: std::cell::Cell<bool>,
    sgl: Option<SglMoe>,
    sgl_pending: std::cell::Cell<bool>,
    sg_next: std::cell::Cell<(u64, u64, i32)>,
    qkv_img: std::cell::Cell<(u64, usize)>,
    b: Bufs,
    pm: Option<PrefillMoe>,
    tcw13: Option<TcW13>,
    tg2: Option<Tg2Bufs>,
    woa64: std::collections::HashMap<DevPtr, DevPtr>,
    tca: Option<[(DevPtr, DevPtr, DevPtr, DevPtr); 2]>,
    sact_tc: DevPtr,
    tca_maps: std::cell::RefCell<std::collections::HashMap<(DevPtr, usize), DevPtr>>,
    tcb_maps: std::cell::RefCell<std::collections::HashMap<(DevPtr, usize, usize), DevPtr>>,
    tmaps: std::cell::RefCell<std::collections::HashMap<(DevPtr, usize, usize, usize, u32), DevPtr>>,
    hcur: usize,
    host_pos_end: usize,
    shared_m: usize,
    cur_m: usize,
    short_append: bool,
    short_ready: bool,
    head_rows: usize,
    cur_layer: std::cell::Cell<usize>,
    side: Stream,
    ev_fork: cu::Event,
    ev_join: cu::Event,
    ev_mean: cu::Event,
    spec_snap: std::cell::RefCell<Vec<(DevPtr, cu::Event)>>,
    spec_snap_buf: DevPtr,
    par: Stream,
    ev_pfork: cu::Event,
    ev_pjoin: cu::Event,
    par_on: std::cell::Cell<bool>,
    ixs: Stream,
    ev_ixf: cu::Event,
    ev_ixj: cu::Event,
    ix_on: std::cell::Cell<bool>,
    eng: Stream,
    eng_on: std::cell::Cell<bool>,
    vadapt: std::cell::Cell<Option<(u64, f32)>>,
    ev_efork: cu::Event,
    ev_ejoin: Vec<cu::Event>,
    eng_bufs: Vec<(DevPtr, DevPtr, DevPtr, DevPtr)>,
    eng_pending: std::cell::Cell<u32>,
    eng_joined: std::cell::Cell<u32>,
    cand_pending: std::cell::Cell<bool>,
    conf_buf: DevPtr,
    samp: (DevPtr, DevPtr, DevPtr, DevPtr, DevPtr, DevPtr),
    samp_rec: DevPtr,
    mk_kmax: DevPtr,
    mk_grid: std::cell::Cell<usize>,
    blk: usize,
    block_idx_by: std::cell::RefCell<std::collections::HashMap<usize, DevPtr>>,
    eng_flags: DevPtr,
    eng_epoch: DevPtr,
    eng_peers: std::cell::RefCell<Vec<EgPeers>>,
    spec_prep_par: std::cell::Cell<bool>,
    hc_late_pending: std::cell::Cell<Option<(DevPtr, DevPtr, DevPtr)>>,
    prep_gather_pending: std::cell::Cell<bool>,
    prep_proj_pending: std::cell::Cell<bool>,
    ev_prep: cu::Event,
    pub mk_dump: std::cell::Cell<DevPtr>,
    win_floor: std::cell::Cell<i32>,
    swa_replay: std::cell::Cell<bool>,
    encoder_readonly: std::cell::Cell<bool>,
    tail: TailBufs,
    pub dev: cu::Device,
}

struct TailBufs {
    h: DevPtr,
    pre: DevPtr,
    state: std::cell::Cell<EncoderTail>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EncoderTail {
    pub(crate) start: usize,
    pub(crate) len: usize,
    pub(crate) end: usize,
}

impl EncoderTail {
    fn retain(self, lo: usize, hi: usize, w: usize) -> Option<Self> {
        if lo >= hi || lo < self.end.saturating_sub(self.len) || hi > self.end {
            return None;
        }
        Some(Self { start: (self.start + lo - (self.end - self.len)) % w, len: hi - lo, end: hi })
    }
    pub(crate) fn truncate(self, prefix: usize) -> Self {
        let end = self.end.min(prefix);
        let len = end.saturating_sub(self.end - self.len);
        if len == 0 { Self::default() } else { Self { len, end, ..self } }
    }
}

fn short_append_eligible(tail: EncoderTail, from: usize, n: usize, w: usize, warmed: bool) -> bool {
    warmed && w > 0 && from >= w && tail.end == from && tail.len == w
        && n.checked_sub(from).is_some_and(|append| (1..1000).contains(&append))
}

fn short_cold_eligible(from: usize, n: usize) -> bool {
    from == 0 && n <= COMM_SHORT_ROWS
}

fn tail_append(start: usize, len: usize, m: usize, w: usize) -> (usize, usize, usize, usize, usize) {
    if m >= w {
        return (0, w, 0, m - w, w);
    }
    let drop = (len + m).saturating_sub(w);
    let start = (start + drop) % w;
    let len = len - drop + m;
    (start, len, (start + len - m) % w, 0, m)
}

fn grid(x: u32, y: u32, block: u32) -> Launch {
    Launch::new(x, block).grid3((x, y, 1))
}

fn init_hc_split(k: &Kernels, stream: &Stream, source: DevPtr, dim: usize, output: DevPtr, bytes: usize) -> Result<()> {
    cu::memset_d8_async(output, 0, bytes, stream)?;
    cu::launch(k.get("hc_fn_split")?, &Launch::new(1024, 256), stream, Args::new().ptr(source).i32(dim as i32).ptr(output))
}

impl Rank {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rank: usize,
        tp: usize,
        cfg: Config,
        dev: cu::Device,
        mut w: Weights,
        mut arena: Arena,
        max_seq: usize,
        pmax: usize,
        engram: EngramTables,
    ) -> Result<Self> {
        let c = &cfg;
        let k = Kernels::load()?;
        let pmax = prefill_capacity(pmax, c.window_size);
        let (plo, phi) = Stream::priority_range()?;
        let (pmain, pside, ppar) = match deepseek_ultrafast_cuda::profile::profile() {
            deepseek_ultrafast_cuda::profile::Profile::Sm103Tp4 => (-4, -4, -5),
            deepseek_ultrafast_cuda::profile::Profile::Sm103Tp8 => (-4, -5, -5),
            deepseek_ultrafast_cuda::profile::Profile::Base => (0, 0, 0),
        };
        let (pmain, pside, ppar) = (i32::clamp(pmain, phi, plo), i32::clamp(pside, phi, plo), i32::clamp(ppar, phi, plo));
        let stream = Stream::with_priority(pmain)?;
        let dim = c.dim;
        let hd = c.head_dim;
        let a = &mut arena;
        let f32b = 4usize;
        let bf = 2usize;
        let mut caches = Vec::new();
        for l in 0..c.n_layers + c.n_mtp_layers {
            let ratio = c.compress_ratios[l];
            let src = c.is_kv_source(l);
            caches.push(LayerCache {
                win: a.zeroed(max_seq * hd * bf)?,
                comp: if src { Some(a.zeroed(max_seq / ratio * hd * bf)?) } else { None },
                idx_k: if src { Some(a.zeroed(max_seq / ratio * c.index_head_dim * bf)?) } else { None },
                idx_k4: if src { Some((a.zeroed(max_seq / ratio * 64)?, a.zeroed((max_seq / ratio).div_ceil(128) * 512)?)) } else { None },
                kv_hist: if src && ratio > 1 { Some(a.zeroed(max_seq * hd * f32b)?) } else { None },
                sc_hist: if src && ratio > 1 { Some(a.zeroed(max_seq * hd * f32b)?) } else { None },
            });
        }
        let mtp_caches = caches.split_off(c.n_layers);
        let cmp_hist = a.zeroed(max_seq * 4)?;
        let rd = c.rope_head_dim;
        let (cb, sb) = rope_table(rd, max_seq, 0, c.rope_theta, c.rope_factor, c.beta_fast, c.beta_slow);
        let (cc, scomp) = rope_table(rd, max_seq, c.original_seq_len, c.compress_rope_theta, c.rope_factor, c.beta_fast, c.beta_slow);
        let rope_base = (a.put(&cb)?, a.put(&sb)?);
        let rope_comp = (a.put(&cc)?, a.put(&scomp)?);
        let ecols = (c.engram_max_ngram_size - 1) * c.engram_n_heads * c.engram_head_dim;
        let per_tok = c.dim.max(5 * c.dim / (2 * tp)).max(ecols).next_multiple_of(256);
        ensure!(per_tok <= COMM_MAX_PER_TOKEN, "comm payload per token {per_tok} exceeds COMM_MAX_PER_TOKEN");
        let cap = pmax * per_tok + 1024;
        let dspark_blk = c.dspark_block_size;
        let conf_buf = a.zeroed(256)?;
        let samp = (a.zeroed(64)?, a.zeroed(64)?, a.zeroed(8 * 200 * 4)?, a.zeroed(8 * w.head.n * 4)?, a.zeroed(8 * 1024 * 8)?, a.zeroed(64)?);
        let samp_rec = a.zeroed(8 * (w.head.n / 64) * 20 * 4)?;
        let spec_snap_buf = a.zeroed(32)?;
        let ne = c.engram_layer_ids.len();
        let ev_ejoin: Vec<cu::Event> = (0..ne).map(|_| cu::Event::new()).collect::<Result<_>>()?;
        let eng_bufs: Vec<(DevPtr, DevPtr, DevPtr, DevPtr)> = (0..ne)
            .map(|_| -> Result<_> { Ok((a.zeroed(MAX_M * ecols * 4)?, a.zeroed(MAX_M * ecols * 2)?, a.zeroed(MAX_M * 5 * c.dim / tp * 4)?, a.zeroed(MAX_M * 5 * c.dim * 2)?)) })
            .collect::<Result<_>>()?;
        let (off_b, cap_b) = ((CommRank::samp_off(tp, cap) + CommRank::samp_bytes(w.head.n)).next_multiple_of(256), MAX_M * per_tok + 1024);
        let dir_off = (off_b + (3 * tp * cap_b * 4) as u64).next_multiple_of(256);
        let short_rs = (dir_off + (ne.max(1) * 32) as u64).next_multiple_of(256);
        let short_ag = short_rs + (3 * tp * COMM_SHORT_CAP * 4) as u64;
        let short_enabled = tp == 2 || tp == 4 || (tp == 8 && ops::variant_cc());
        let comm_bytes = if short_enabled { short_ag as usize + 3 * COMM_SHORT_CAP * 2 } else { short_rs as usize };
        let comm = CommRank {
            recv: a.zeroed(comm_bytes)?,
            state: a.zeroed(COMM_MAX_BLOCKS * 16)?,
            state2: a.zeroed(COMM_MAX_BLOCKS * 16)?,
            cap,
            off_b,
            cap_b,
            state_b: a.zeroed(COMM_MAX_BLOCKS * 16)?,
            dir_off,
            short_rs,
            short_ag,
            short_state: if short_enabled { a.zeroed(COMM_SHORT_CAP / 256 * 16)? } else { 0 },
        };
        let (eng_flags, eng_epoch) = (a.zeroed(ne.max(1) * 4)?, a.zeroed(ne.max(1) * 4)?);
        for (ei, &l) in c.engram_layer_ids.iter().enumerate() {
            if let Some(eg) = w.layers.get(l).and_then(|ly| ly.engram.as_ref()) {
                cu::upload(comm.recv + dir_off + (ei * 32) as u64, &[eg.table, eg.scales, eng_bufs[ei].2, eng_flags + (ei * 4) as u64])?;
            }
        }
        {
            dev.synchronize()?;
            let n = (2 * 3 * tp * cap) as u64;
            cu::launch(k.get("comm_init")?, &Launch::new(1024, 256), &stream, Args::new().ptr(comm.recv).push(n))?;
            cu::launch(k.get("comm_init")?, &Launch::new(1024, 256), &stream, Args::new().ptr(comm.recv + off_b).push((3 * tp * cap_b) as u64))?;
            let n2 = (3 * cap / 2) as u64;
            cu::launch(k.get("comm_init2")?, &Launch::new(1024, 256), &stream, Args::new().ptr(comm.recv + CommRank::ag_off(tp, cap)).push(n2))?;
            if short_enabled {
                cu::launch(k.get("comm_init")?, &Launch::new(1024, 256), &stream, Args::new().ptr(comm.recv + short_rs).push((3 * tp * COMM_SHORT_CAP) as u64))?;
                cu::launch(k.get("comm_init2")?, &Launch::new(1024, 256), &stream, Args::new().ptr(comm.recv + short_ag).push((3 * COMM_SHORT_CAP / 2) as u64))?;
            }
            stream.synchronize()?;
        }
        let m = pmax;
        let heads = c.n_heads / tp;
        let tcap = max_seq;
        let n_exp = c.n_routed_experts.max(c.dspark_n_routed_experts);
        let inter = c.moe_inter_dim / tp;
        let kpad = inter.div_ceil(64) * 64;
        let b = Bufs {
            h: [a.zeroed(m * 4 * dim * bf)?, a.zeroed(m * 4 * dim * bf)?],
            x: a.zeroed(m * dim * bf)?,
            dots: a.zeroed(m * 32 * f32b)?,
            pre_a: a.zeroed(m * 4 * f32b)?,
            post_a: a.zeroed(m * 4 * f32b)?,
            comb_a: a.zeroed(m * 16 * f32b)?,
            pre_f: a.zeroed(m * 4 * f32b)?,
            post_f: a.zeroed(m * 4 * f32b)?,
            comb_f: a.zeroed(m * 16 * f32b)?,
            q_lat: a.zeroed(m * c.q_lora_rank * bf)?,
            qr: a.zeroed(m * c.q_lora_rank * bf)?,
            q: a.zeroed(m * heads * hd * bf)?,
            kv: a.zeroed(m * hd * bf)?,
            kv_all: a.zeroed(m * 3 * hd * bf)?,
            o: a.zeroed(m * heads * hd * bf)?,
            oa: a.zeroed(m * c.o_groups / tp * c.o_lora_rank * bf)?,
            ob: a.zeroed(m * dim * f32b)?,
            ob_part: a.zeroed(MAX_M * 16 * dim * f32b)?,
            iq: a.zeroed(m * c.index_n_heads * c.index_head_dim * bf)?,
            iq4: (a.zeroed(m * c.index_n_heads * 64)?, a.zeroed((m * c.index_n_heads).div_ceil(128) * 512)?),
            iw: a.zeroed(m * 64 * bf)?,
            scores: a.zeroed(Self::score_rows(pmax, tp) * tcap * f32b)?,
            bscores: a.zeroed(Self::score_rows(pmax, tp) * (tcap / 8 + 8) * f32b)?,
            cand: a.zeroed(m * (tcap / 8 + 8))?,
            cand_ids: a.zeroed(m * c.candidate_topk_blocks * 4)?,
            topk: a.zeroed(m * c.index_topk * 4)?,
            ckv: a.zeroed(m * hd * f32b)?,
            csc: a.zeroed(m * hd * f32b)?,
            latent: a.zeroed(m * hd * bf)?,
            ik: a.zeroed(m * c.index_head_dim * bf)?,
            glog: a.zeroed(m * n_exp * f32b)?,
            glog_part: a.zeroed(MAX_M * 16 * n_exp * f32b)?,
            route_ids: a.zeroed(m * 8 * 4)?,
            route_w: a.zeroed(m * 8 * f32b)?,
            plan: a.zeroed(64 << 10)?,
            shared_plan: {
                let ms = 97usize;
                let words = 1 + 2 * ms + 3 * ms * 16;
                let mut all = vec![0i32; MAX_M * words];
                for mm in 1..=MAX_M {
                    let pl = &mut all[(mm - 1) * words..mm * words];
                    pl[0] = 1;
                    pl[1] = 0;
                    pl[1 + ms] = mm as i32;
                    let (tok, pair) = (1 + 2 * ms, 1 + 2 * ms + ms * 16);
                    let w = pair + ms * 16;
                    for i in 0..mm {
                        pl[tok + i] = i as i32;
                        pl[pair + i] = i as i32;
                        pl[w + i] = 1.0f32.to_bits() as i32;
                    }
                }
                a.put(&all)?
            },
            act: a.zeroed(m * 8 * inter * bf)?,
            sact: a.zeroed(m * inter * bf)?,
            xq: a.zeroed(m * (dim + 32) * bf)?,
            sched: a.zeroed(256)?,
            xsq: a.zeroed(dim / 32 * m * f32b)?,
            actq: a.zeroed(m * 8 * (kpad + 32) * bf)?,
            asq: a.zeroed(m * 8 * ops::moe_fused_sq_ld(kpad) * f32b)?,
            moe_done: a.zeroed(4 * 128)?,
            dg_seq: a.zeroed(16)?,
            mk_cand: a.zeroed(2 * 256 * 8 + 256 * 8 * 4)?,
            mk_bar: a.zeroed(16)?,
            y: a.zeroed(m * dim * f32b)?,
            ypair: a.zeroed(m * 8 * dim * f32b)?,
            eg: a.zeroed(m * 24 * 256 * f32b)?,
            ex: a.zeroed(m * 24 * 256 * bf)?,
            ekv_part: a.zeroed(m * 5 * dim / tp * f32b)?,
            ekv: a.zeroed(m * 5 * dim * bf)?,
            logits: a.zeroed(MAX_M * w.head.n * f32b)?,
            amax_v: a.zeroed(m * f32b)?,
            amax_i: a.zeroed(m * 4)?,
            amax_pack: a.zeroed(m * 4 * f32b)?,
            amax_all: a.zeroed(m * 4 * tp * f32b)?,
            next: a.zeroed(m * 4)?,
            main_hidden: a.zeroed(m * 3 * dim * bf)?,
            out_tokens: a.zeroed(OUT_CAP * 4)?,
            out_count: a.zeroed(64)?,
            main_x: a.zeroed(m * dim * bf)?,
            x_raw: a.zeroed(m * dim * bf)?,
            part: a.zeroed(m * (dim / 256) * 26 * f32b)?,
            block_kv: a.zeroed(m * hd * bf)?,
            block_idx: {
                let blk = c.dspark_block_size;
                let t: Vec<i32> = (0..m * blk).map(|i| (i % blk) as i32).collect();
                a.put(&t)?
            },
            drafts: a.zeroed(64)?,
            spec_state: a.zeroed(64)?,
            idx_okeys: if ops::variant_cc() { a.zeroed(MAX_M * c.index_topk * 4)? } else { 0 },
        };
        let pm = if pmax > MAX_M && (2 * inter) % 64 == 0 && kpad % 64 == 0 {
            let topk = c.n_activated_experts;
            let rows = pmax * topk;
            let slack = rows + ops::TM_NT;
            let max_tiles = rows.div_ceil(ops::TM_NT) + c.n_routed_experts;
            anyhow::ensure!(max_tiles <= 1024, "prefill MoE tile list exceeds TM_MAX_TILES");
            let ldaq = kpad.div_ceil(128) * 128;
            let akt = kpad.div_ceil(128);
            let xq = a.zeroed(slack * dim)?;
            let aq = a.zeroed(slack * ldaq)?;
            let maps = |base: DevPtr, cols: usize| -> Result<Vec<u64>> {
                let mut v = Vec::new();
                for nb in 1..=8u32 {
                    v.extend_from_slice(&cu::tensor_map_u8_sw128(base, cols as u64, slack as u64, cols as u64, 16 * nb)?);
                }
                Ok(v)
            };
            let (tb13, tb2) = (maps(xq, dim)?, maps(aq, ldaq)?);
            for f in ["tc_moe_w13", "tc_moe_w2"] {
                k.get(f)?.set_max_dynamic_smem(ops::TM_SMEM)?;
            }
            Some(PrefillMoe {
                row_pair: a.zeroed(rows * 4)?,
                row_w: a.zeroed(rows * 4)?,
                pair_row: a.zeroed(rows * 4)?,
                row_tile: a.zeroed(rows * 4)?,
                tiles: a.zeroed(max_tiles * 16)?,
                ntiles: a.zeroed(16)?,
                xq,
                sf: a.zeroed(max_tiles * (dim / 128) * 512)?,
                aq,
                asf: a.zeroed(max_tiles * akt * 512)?,
                tb13: a.put(&tb13)?,
                tb2: a.put(&tb2)?,
                max_tiles,
                ldaq,
                akt,
                sms: dev.sm_count()? as u32,
            })
        } else {
            None
        };
        let tca = if pmax > MAX_M && !w.tc.is_empty() {
            k.get("tc_gemm_fp8")?.set_max_dynamic_smem(ops::tg_smem(256))?;
            k.get("tc_gemm_fp8p")?.set_max_dynamic_smem(ops::TGP_SMEM)?;
            k.get("tc_gemm_bf16")?.set_max_dynamic_smem(ops::tb_smem(256))?;
            let mut scratch = [(0, 0, 0, 0); 2];
            for s in &mut scratch {
                *s = (a.zeroed(pmax * 15360)?, a.zeroed(pmax.div_ceil(128) * 120 * 512)?, a.zeroed(pmax * 26 * f32b)?, a.zeroed(pmax * 8192 * bf)?);
            }
            Some(scratch)
        } else {
            None
        };
        let sact_tc = if tca.is_some() { a.zeroed(pmax * inter.div_ceil(128) * 128 * bf)? } else { 0 };
        let meta = a.zeroed(std::mem::size_of::<Meta>())?;
        let metas = a.zeroed((meta_slices(pmax) + 3) * std::mem::size_of::<Meta>())?;
        let ptoks = a.zeroed(pmax * 4)?;
        k.get("topk_merge")?.set_max_dynamic_smem(200 * 1024)?;
        let tk_per_row = (max_seq.div_ceil(TK_S) * c.index_topk).max((max_seq / 8).div_ceil(TK_S) * c.candidate_topk_blocks);
        let tk_rows = Self::score_rows(pmax, tp);
        let tk_scratch = (a.zeroed(tk_rows * tk_per_row * 4)?, a.zeroed(tk_rows * tk_per_row * 4)?, a.zeroed(2 * tk_rows * 64 * 4)?, tk_per_row);
        let tk_scratch_b = (a.zeroed(tk_rows * tk_per_row * 4)?, a.zeroed(tk_rows * tk_per_row * 4)?, a.zeroed(2 * tk_rows * 64 * 4)?, tk_per_row);
        let tk_gather = (a.zeroed(pmax * c.index_topk * 4)?, a.zeroed(pmax * c.index_topk * 4)?);
        let mut hc_fnb = std::collections::HashMap::new();
        for ly in &w.layers {
            for f in [ly.hc_attn.0, ly.hc_ffn.0] {
                let nb = (c.dim / 256) * 2 * 24 * (4 * 256 + 8) * 2;
                let b = a.alloc(nb)?;
                init_hc_split(&k, &stream, f, c.dim, b, nb)?;
                hc_fnb.insert(f, b);
            }
        }
        stream.synchronize()?;
        let tail = TailBufs { h: a.zeroed(c.window_size * 4 * dim * bf)?, pre: a.zeroed(c.window_size * 4 * f32b)?, state: std::cell::Cell::new(EncoderTail::default()) };
        let tcw13 = Self::tcw13_init(&k, &w, a, dim)?;
        let sgl = Self::sgl_init(&w, a, dim, cfg.swiglu_limit)?;
        let tg2 = if ops::tcgemv2_on() {
            let gl = c.o_groups / tp * c.o_lora_rank;
            Some(Tg2Bufs { qimg: a.zeroed(ops::tg2_img_bytes(c.q_lora_rank))?, qssq: a.zeroed(8 * c.q_lora_rank / 32 * 4)?, oaimg: a.zeroed(ops::tg2_img_bytes(gl))? })
        } else {
            None
        };
        let mut woa64 = std::collections::HashMap::new();
        if tg2.is_some() {
            for l in w.layers.iter().chain(w.mtp.iter()) {
                if let Some(tw) = w.tc.get(&l.wo_a.w).filter(|tw| tw.bf16 && tw.kp == l.wo_a.k && l.wo_a.n % 64 == 0) {
                    let t = cu::tensor_map_bf16_sw128(tw.w, l.wo_a.k as u64, l.wo_a.n as u64, (l.wo_a.k * 2) as u64, 64)?;
                    let buf = cu::upload_synced(&t)?;
                    woa64.insert(l.wo_a.w, buf.ptr);
                    std::mem::forget(buf);
                }
            }
        }
        if let Some(h) = w.head_fp8_host.take() {
            w.head_fp8 = Some(ops::Weight { fmt: ops::WFmt::Fp8Bf16Act, w: a.put(&h.tiles)?, s: a.put(&h.scales)?, n: h.n, k: h.k });
        }
        let mk_kmax = if deepseek_ultrafast_cuda::profile::sm103() { a.zeroed(8 * 8)? } else { 0 };
        let lp_mixf = if deepseek_ultrafast_cuda::profile::sm103_tp4() { a.zeroed(64)? } else { 0 };
        let ffn_lp_buf = if deepseek_ultrafast_cuda::profile::sm103_tp4() {
            let b = cu::Buffer::zeroed(FFN_LP_ROWS * cfg.dim * 4)?;
            cu::memset_d32_async(b.ptr, 0x8000_0000, FFN_LP_ROWS * cfg.dim, &stream)?;
            Some(b)
        } else {
            None
        };
        dev.synchronize()?;
        let mut k = k;
        k.load_variants()?;
        let rk = Rank {
            sms: dev.sm_count()? as u32,
            pmax,
            metas,
            meta_step: meta,
            ptoks,
            tk_scratch,
            tk_scratch_b,
            tk_gather,
            hc_fnb,
            rank,
            tp,
            cfg,
            dev,
            stream,
            k,
            w,
            arena,
            meta,
            max_seq,
            caches,
            mtp_caches,
            cmp_hist,
            rope_base,
            rope_comp,
            engram,
            comm,
            peers: Vec::new(),
            nvls: None,
            nvls_hold: None,
            aa_pending: std::cell::Cell::new(false),
            lp_mixf,
            ffn_lp_buf,
            lp_pending: std::cell::Cell::new(false),
            lp_mixf_on: std::cell::Cell::new(false),
            sgl,
            sgl_pending: std::cell::Cell::new(false),
            sg_next: Default::default(),
            qkv_img: Default::default(),
            b,
            pm,
            tca,
            sact_tc,
            tca_maps: Default::default(),
            tcb_maps: Default::default(),
            tmaps: Default::default(),
            hcur: 0,
            vision: None,
            image_prefill: false,
            host_pos_end: 0,
            shared_m: 0,
            cur_m: 1,
            short_append: false,
            short_ready: false,
            head_rows: 1,
            cur_layer: std::cell::Cell::new(usize::MAX),
            side: Stream::with_priority(pside)?,
            ev_fork: cu::Event::new()?,
            ev_join: cu::Event::new()?,
            ev_mean: cu::Event::new()?,
            spec_snap: std::cell::RefCell::new(Vec::new()),
            spec_snap_buf,
            par: Stream::with_priority(ppar)?,
            ev_pfork: cu::Event::new()?,
            ev_pjoin: cu::Event::new()?,
            par_on: std::cell::Cell::new(false),
            ixs: Stream::with_priority(ppar)?,
            ev_ixf: cu::Event::new()?,
            ev_ixj: cu::Event::new()?,
            ix_on: std::cell::Cell::new(false),
            eng: Stream::with_priority(0)?,
            eng_on: std::cell::Cell::new(false),
            vadapt: std::cell::Cell::new(None),
            ev_efork: cu::Event::new()?,
            ev_ejoin,
            eng_bufs,
            eng_pending: std::cell::Cell::new(0),
            eng_joined: std::cell::Cell::new(0),
            cand_pending: std::cell::Cell::new(false),
            conf_buf,
            samp,
            samp_rec,
            mk_kmax,
            mk_grid: std::cell::Cell::new(0),
            blk: dspark_blk,
            block_idx_by: Default::default(),
            eng_flags,
            eng_epoch,
            eng_peers: Default::default(),
            spec_prep_par: std::cell::Cell::new(false),
            hc_late_pending: std::cell::Cell::new(None),
            prep_gather_pending: std::cell::Cell::new(false),
            prep_proj_pending: std::cell::Cell::new(false),
            ev_prep: cu::Event::new()?,
            mk_dump: std::cell::Cell::new(0),
            win_floor: std::cell::Cell::new(0),
            swa_replay: std::cell::Cell::new(false),
            encoder_readonly: std::cell::Cell::new(false),
            tail,
            tcw13,
            tg2,
            woa64,
        };
        rk.index_tmaps()?;
        Ok(rk)
    }

    fn index_tmaps(&self) -> Result<()> {
        let c = &self.cfg;
        let (ihd, inh) = (c.index_head_dim, c.index_n_heads);
        if self.b.iq == 0 || ihd == 0 {
            return Ok(());
        }
        for (l, cache) in self.caches.iter().enumerate() {
            let ratio = c.compress_ratios[l];
            if let (Some(kc), true) = (cache.idx_k, ratio > 0 && self.max_seq / ratio > 0) {
                self.tmap_bf16(kc, ihd, self.max_seq / ratio, ihd * 2, 128)?;
            }
        }
        self.tmap_bf16(self.b.iq, ihd, self.pmax * inh, ihd * 2, 256)?;
        Ok(())
    }

    fn tcw13_init(_k: &Kernels, w: &Weights, a: &mut Arena, dim: usize) -> Result<Option<TcW13>> {
        let mut maps = std::collections::HashMap::new();
        let mut w2maps = std::collections::HashMap::new();
        let mut kt2 = 0;
        for l in w.layers.iter().chain(w.mtp.iter()).filter(|l| l.experts.rm) {
            let e = &l.experts;
            let r = 2 * e.inter;
            let tm = cu::tensor_map_fp4_unpack_strided(e.w13, dim as u64, r as u64, e.n as u64, e.w13_stride as u64, 128)?;
            maps.insert(e.w13, a.put(&tm)?);
            let t2 = cu::tensor_map_fp4_unpack_strided(e.w2, e.inter.div_ceil(128) as u64 * 128, dim as u64, e.n as u64, e.w2_stride as u64, 128)?;
            w2maps.insert(e.w2, a.put(&t2)?);
            kt2 = kt2.max(e.inter.div_ceil(128));
        }
        if maps.is_empty() {
            return Ok(None);
        }
        let ximg = a.zeroed((dim / 128) * ops::TMOE_IMG)?;
        let aimg = a.zeroed(MOE_PLAN_SLOTS * kt2 * ops::TMOE_IMG)?;
        let rt13 = w.layers.iter().chain(w.mtp.iter()).filter(|l| l.experts.rm).map(|l| 2 * l.experts.inter / 128).max().unwrap_or(0);
        ensure!((dim / 128) % ops::TMOE_KQ == 0, "TMOE_KQ {} does not divide {} K chunks", ops::TMOE_KQ, dim / 128);
        let part = a.zeroed(MOE_PLAN_SLOTS * rt13 * ops::TMOE_KQ_MAX * 128 * 8 * 4)?;
        let tcnt = a.zeroed(MOE_PLAN_SLOTS * rt13 * 4)?;
        let sctr = a.zeroed(MOE_PLAN_SLOTS * 4)?;
        deepseek_ultrafast_logging::diagnostic!("tmoe: {} layers' routed experts in the tcgen05 layout (decode tmoe_fused, prefill tc_moe_w13_rm2 / tc_moe_w2_rm)", maps.len());
        Ok(Some(TcW13 { maps, w2maps, ximg, aimg, part, tcnt, sctr }))
    }

    fn sgl_init(w: &Weights, a: &mut Arena, dim: usize, swiglu_limit: f32) -> Result<Option<SglMoe>> {
        let ls: Vec<&Layer> = w.layers.iter().chain(w.mtp.iter()).filter(|l| l.experts.sgl).collect();
        if ls.is_empty() {
            return Ok(None);
        }
        ensure!(dim == 5120, "sgl moe: dim {dim}");
        let mut maps = std::collections::HashMap::new();
        let pad = ls[0].experts.inter.div_ceil(128) * 128;
        ensure!(ls.iter().all(|l| l.experts.inter.div_ceil(128) * 128 == pad), "sgl moe: layers of different expert widths");
        for l in &ls {
            let e = &l.experts;
            let (sh13, sh2) = l.shared_sgl.ok_or_else(|| anyhow::anyhow!("sgl moe: no shared expert copy (layer {})", l.id))?;
            let pad = e.inter.div_ceil(128) * 128;
            let m13 = cu::tensor_map_fp4_unpack_extent(e.w13, dim as u64, (2 * e.inter) as u64, e.n as u64, (dim / 2) as u64, e.w13_stride as u64, 128)?;
            let m2 = cu::tensor_map_fp4_unpack_extent(e.w2, pad as u64, dim as u64, e.n as u64, sgl_w2_row_bytes(e.inter) as u64, e.w2_stride as u64, 128)?;
            let s13 = cu::tensor_map_u8_sw128(sh13, dim as u64, (2 * pad) as u64, dim as u64, 128)?;
            let s2 = cu::tensor_map_u8_sw128(sh2, pad as u64, dim as u64, pad as u64, 128)?;
            maps.insert(e.w13, [a.put(&m13)?, a.put(&m2)?, a.put(&s13)?, a.put(&s2)?]);
        }
        let slots = a.put(&vec![u64::MAX; 8 * 384])?;
        let counter = a.zeroed(64)?;
        let (out_w, out_i, permuted, record) = (a.zeroed(8 * 8 * 4)?, a.zeroed(8 * 8 * 4)?, a.zeroed(8 * 8 * 4)?, a.zeroed(1024 * 4)?);
        let (xq, xs) = (a.zeroed(8 * dim)?, a.zeroed(8 * dim / 32)?);
        let (h, hs) = (a.zeroed(SGL_C_ROWS * 640)?, a.zeroed(SGL_TILES * 2560)?);
        let clamp_v = if swiglu_limit > 0.0 { swiglu_limit } else { f32::INFINITY };
        let clamp = a.put(&vec![clamp_v; 384])?;
        let c = a.put(&vec![0x8000u16; SGL_C_ROWS * dim])?;
        let tm_xq = a.put(&cu::tensor_map_u8_sw128(xq, dim as u64, 8, dim as u64, 8)?)?;
        let tm_h = a.put(&cu::tensor_map_u8_sw128(h, pad as u64, SGL_C_ROWS as u64, pad as u64, 8)?)?;
        deepseek_ultrafast_logging::diagnostic!("sgl moe: {} layers on SGLang's low-batch MoE chain", ls.len());
        let metas = a.zeroed(4 * std::mem::size_of::<Meta>())?;
        let img = a.zeroed(dim / 128 * 1536)?;
        Ok(Some(SglMoe { slots, counter, out_w, out_i, permuted, record, xq, xs, h, hs, clamp, clamp_v, c, tm_xq, tm_h, metas, img, maps, pad }))
    }

    fn ffn_sgl(&mut self, ly: &'static Layer, m: usize, topk: usize) -> Result<()> {
        let c = self.cfg.clone();
        let dim = c.dim;
        ensure!(m >= 1 && m <= 8 && self.cur_m == m, "sgl moe: {m} rows (cur {}), top-{topk}", self.cur_m);
        let st = self.sgl.as_ref().ok_or_else(|| anyhow::anyhow!("sgl moe: no state"))?;
        let e = &ly.experts;
        let mp = *st.maps.get(&e.w13).ok_or_else(|| anyhow::anyhow!("sgl moe: layer {} has no maps", ly.id))?;
        let (sh13, sh2) = ly.shared_sgl.ok_or_else(|| anyhow::anyhow!("sgl moe: no shared expert copy"))?;
        let gate_rm = self.w.tc.get(&ly.gate.w).filter(|t| t.bf16 && t.kp == dim).map(|t| t.w).ok_or_else(|| anyhow::anyhow!("sgl moe: no row-major gate copy"))?;
        ensure!(ly.gate.n == e.n && matches!((e.n, topk), (384, 6) | (128, 3)), "sgl moe: {} experts, top-{topk}", e.n);
        let pad = e.inter.div_ceil(128) * 128;
        let cw = ((m * topk + 3) / 4 * 4) as i32;
        let r = ops::SgrParams {
            x: self.b.x_raw,
            ldx: dim as i64,
            norm_part: self.b.part,
            nslices: (dim / 256) as i32,
            eps: c.norm_eps,
            norm_w: ly.ffn_norm,
            w: gate_rm,
            bias: ly.gate_bias,
            slots: st.slots,
            counter: st.counter,
            out_w: st.out_w,
            out_i: st.out_i,
            permuted: st.permuted,
            record: st.record,
            xq: st.xq,
            xs: st.xs,
            scale: c.route_scale,
            _pad: 0,
        };
        let g1 = ops::SgmParams {
            tm_a: mp[0],
            tm_b: st.tm_xq,
            tm_sh: mp[2],
            sfa_sh: sh13 + (2 * pad * dim) as u64,
            sh_clamp: st.clamp_v,
            has_sh: 1,
            m_rows: m as i32,
            _pad0: 0,
            sfa: e.w13 + (2 * pad * dim / 2) as u64,
            sfa_stride: e.w13_stride as i64,
            sfb: st.xs,
            out: st.h,
            hs: st.hs,
            record: st.record,
            clamp: st.clamp,
            rows: m as i32,
            cta_words: cw,
            trigger_at_start: 1,
            _pad: 0,
        };
        let g2 = ops::SgmParams {
            tm_a: mp[1],
            tm_b: st.tm_h,
            tm_sh: mp[3],
            sfa_sh: sh2 + (dim * pad) as u64,
            sfa: e.w2 + (dim * sgl_w2_row_bytes(e.inter)) as u64,
            sfa_stride: e.w2_stride as i64,
            sfb: st.hs,
            out: st.c,
            hs: 0,
            rows: SGL_C_ROWS as i32,
            trigger_at_start: 1,
            ..g1
        };
        let cbuf = st.c;
        let spad = st.pad;
        ensure!(spad == pad, "sgl moe: layer {} pads to {pad}, the chain to {spad}", ly.id);
        let ne = e.n;
        self.timed("moe sgl", || ops::sgl_moe_launch(&self.k, &self.stream, m, spad, ne, r, g1, g2))?;
        let (hin, hout) = (self.b.h[self.hcur], self.b.h[1 - self.hcur]);
        self.hc_join()?;
        if let Some(ei) = c.engram_index(ly.id + 1) {
            if self.eng_pending.get() & (1 << ei) != 0 {
                self.eng_pending.set(self.eng_pending.get() & !(1 << ei));
                self.eng_joined.set(self.eng_joined.get() | (1 << ei));
                self.stream.wait(&self.ev_ejoin[ei])?;
            }
        }
        self.sgl_pending.set(true);
        if deepseek_ultrafast_cuda::profile::sm103() && self.tg2_step(m).is_some() {
            if let Some(nx) = self.w.layers.get(ly.id + 1).filter(|nx| nx.engram.is_none()) {
                let plain = nx.indexer.is_none() && nx.compressor.is_none() && c.engram_index(nx.id).is_none() && c.engram_index(nx.id + 1).is_none();
                self.sg_next.set((nx.attn_norm, st.img, plain as i32));
                self.qkv_img.set((st.img, m));
            }
        }
        self.comm_ex(cbuf, dim, dim, CM_SUM_HCPOST, false, hout, dim, false, Some((hin, self.b.post_a, self.b.comb_a)), topk, 0, Some(self.epi_nodots()))?;
        ensure!(!self.sgl_pending.get(), "sgl moe: GEMM2's rows were not consumed");
        self.hcur = 1 - self.hcur;
        Ok(())
    }

    fn tmoe_params(&self, ly: &Layer, out: DevPtr) -> Result<ops::TmoeParams> {
        let t = self.tcw13.as_ref().ok_or_else(|| anyhow::anyhow!("tmoe: no tcgen05 expert state"))?;
        let e = &ly.experts;
        let sh13 = *self.w.tc.get(&ly.shared_w13.w).ok_or_else(|| anyhow::anyhow!("tmoe: no tensor-core copy of the shared w13"))?;
        let sh2 = *self.w.tc.get(&ly.shared_w2.w).ok_or_else(|| anyhow::anyhow!("tmoe: no tensor-core copy of the shared w2"))?;
        ensure!(!sh13.bf16 && !sh2.bf16 && sh13.tmap128 != 0 && sh2.tmap128 != 0 && sh13.kp == self.cfg.dim && sh2.kp == e.inter.div_ceil(128) * 128,
            "tmoe: the shared expert's tensor-core copies (kp {} / {})", sh13.kp, sh2.kp);
        ensure!(2 * e.inter == ly.shared_w13.n && ly.shared_w2.n == self.cfg.dim, "tmoe: shared expert shapes");
        Ok(ops::TmoeParams {
            w13_map: t.maps[&e.w13],
            w13: e.w13,
            w13_stride: e.w13_stride as i64,
            w2_map: t.w2maps[&e.w2],
            w2: e.w2,
            w2_stride: e.w2_stride as i64,
            sh13_map: sh13.tmap128,
            sh13_sf: sh13.sf,
            sh2_map: sh2.tmap128,
            sh2_sf: sh2.sf,
            ximg: t.ximg,
            aimg: t.aimg,
            out,
            done: self.b.moe_done,
            part: t.part,
            tcnt: t.tcnt,
            sctr: t.sctr,
            kq: ops::TMOE_KQ as i32,
            _pad: 0,
            r13: (2 * e.inter) as i32,
            k2: e.inter as i32,
            n2: self.cfg.dim as i32,
            flags: 0,
        })
    }

    fn st(&self) -> &Stream {
        if self.eng_on.get() { &self.eng } else if self.par_on.get() { &self.par } else if self.ix_on.get() { &self.ixs } else { &self.stream }
    }

    fn on_par<T>(&self, f: impl FnOnce(&Self) -> Result<T>) -> Result<T> {
        self.par_on.set(true);
        let r = f(self);
        self.par_on.set(false);
        r
    }

    fn launch(&self, name: &str, mut cfg: Launch, args: &mut Args) -> Result<()> {
        cfg.pdl = true;
        self.timed(name, || cu::launch(self.k.get(name)?, &cfg, self.st(), args))
    }

    fn timed(&self, name: &str, f: impl FnOnce() -> Result<()>) -> Result<()> {
        let _ = name;
        f()
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_slices(&self, par: bool, w: &Weight, x: DevPtr, ldx: usize, m: usize, out: DevPtr, ldo: usize, o: GemvOpts) -> Result<()> {
        let ob = if o.out_f32 || matches!(o.epilogue, Epilogue::SqrtSoftplus) { 4 } else { 2 };
        for off in (0..m).step_by(MAX_M) {
            let rows = MAX_M.min(m - off);
            let u = off as u64;
            let mut so = o;
            so.norm_part = o.norm_part.map(|(p, n)| (p + u * (n * 26 * 4) as u64, n));
            so.split = o.split.map(|(n, p, ld)| (n, p + u * (ld * 2) as u64, ld));
            so.xq_out = o.xq_out.map(|(q, sq)| (q + u * ((w.k + 32) * 2) as u64, sq + u * ((w.k / 32) * 4) as u64));
            if let Some(mo) = o.moe {
                so.moe = Some(Moe { plan: self.shared_plan(rows), ..mo });
            }
            let (xs, os) = (x + u * (ldx * 2) as u64, out + u * (ldo * ob) as u64);
            if par {
                self.gemv_par(w, xs, ldx, rows, os, ldo, so)?;
            } else {
                self.gemv(w, xs, ldx, rows, os, ldo, so)?;
            }
        }
        Ok(())
    }

    fn tc_scratch(&self) -> (DevPtr, DevPtr, DevPtr, DevPtr) {
        debug_assert!(!self.eng_on.get());
        self.tca.unwrap()[usize::from(self.par_on.get())]
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_tc(&self, tw: crate::weights::TcWeight, w: &Weight, x: DevPtr, ldx: usize, m: usize, out: DevPtr, ldo: usize, o: GemvOpts) -> Result<()> {
        let (aq, sfa, part, _) = self.tc_scratch();
        let (n, k) = (w.n, tw.kp);
        ensure!(m <= self.pmax && k % 128 == 0 && (k == w.k || o.prenorm.is_none()));
        let (pw, np, ns, eps) = match (o.prenorm, o.norm_part) {
            (Some((pw, eps)), Some((np, ns))) => (pw, np, ns as i32, eps),
            (Some((pw, eps)), None) => {
                self.launch("tc_row_sumsq", Launch::new((m as u32).div_ceil(8), 256), Args::new().ptr(x).i32(ldx as i32).i32(m as i32).i32(k as i32).ptr(part))?;
                (pw, part, 1, eps)
            }
            (None, _) => (0, 0, 0, 0.0),
        };
        let qp = ops::TqParams { x, ldx: ldx as i32, m: m as i32, k: k as i32, _pad: 0, pw, norm_part: np, nslices: ns, eps, aq, sfa };
        let qt = ((k / 32).div_ceil(32) * 32).min(256) as u32;
        self.launch("tc_quant_rows", Launch::new(m as u32, qt), Args::new().push(qp))?;
        let ta = {
            let mut maps = self.tca_maps.borrow_mut();
            let key = (aq, k);
            match maps.get(&key) {
                Some(&t) => t,
                None => {
                    ensure!(!self.st().is_capturing()?, "tensor map for the tc activation image (k {k}) created inside a capture");
                    let t = cu::tensor_map_u8_sw128(aq, k as u64, self.pmax as u64, k as u64, 128)?;
                    let b = cu::upload_synced(&t)?;
                    let ptr = b.ptr;
                    std::mem::forget(b);
                    maps.insert(key, ptr);
                    ptr
                }
            }
        };
        let (out2, ldo2, n_split) = match o.split { Some((ns, o2, l2)) => (o2, l2 as i32, ns as i32), None => (0, 0, n as i32) };
        let (swiglu, swiglu_limit) = match o.epilogue { Epilogue::SwiGlu(l) => (1, l), _ => (0, 0.0) };
        let gp = ops::TgParams { ta, tb: tw.tmap, sfa, sfb: tw.sf, out, ldo: ldo as i32, out_f32: o.out_f32 as i32, m: m as i32, n: n as i32, k: k as i32, nt: tw.nt as i32, alpha: o.alpha, _pad: 0, out2, ldo2, n_split, swiglu, swiglu_limit };
        let gx = (n / tw.nt) as u32;
        let (mt, sms) = ((m as u32).div_ceil(128), self.sms);
        if tw.tmap128 != 0 && !(gx * mt >= sms * 2 / 3 && gx * mt <= sms) {
            let tiles = (n as u32 / 128) * mt;
            let gp = ops::TgParams { tb: tw.tmap128, ..gp };
            return self.launch("tc_gemm_fp8p", Launch::new(tiles.min(sms), 256).smem(ops::TGP_SMEM), Args::new().push(gp));
        }
        self.launch("tc_gemm_fp8", Launch::new(gx, 128).grid3((gx, (m as u32).div_ceil(128), 1)).smem(ops::tg_smem(tw.nt)), Args::new().push(gp))
    }

    fn score_rows(pmax: usize, tp: usize) -> usize {
        if tp == 1 { pmax } else { (pmax / tp + 8).max(4 * tp + MAX_M) }
    }

    fn idx_rows(&self, m: usize) -> Option<(usize, usize, DevPtr)> {
        if m <= MAX_M || self.tp == 1 || self.meta != self.metas {
            return None;
        }
        if (0..self.tp).any(|r| { let (a, b) = shard_rows(r, m, self.tp); a == b }) {
            return None;
        }
        let (r0, r1) = shard_rows(self.rank, m, self.tp);
        Some((r0, r1, self.metas + ((meta_slices(self.pmax) + 1) * std::mem::size_of::<Meta>()) as u64))
    }

    fn topk_m(&self, t: TopkParams, m: usize, n_cap: usize, meta: DevPtr) -> Result<()> {
        self.topk_mc(t, m, n_cap, meta, None)
    }

    fn topk_mc(&self, t: TopkParams, m: usize, n_cap: usize, meta: DevPtr, cand: Option<(DevPtr, usize)>) -> Result<()> {
        let nsplit = n_cap.div_ceil(TK_S).max(1);
        let cap = t.k as usize;
        let (cv, ci, cn, per_row) = if t.k as usize == self.cfg.candidate_topk_blocks { self.tk_scratch_b } else { self.tk_scratch };
        let two_stage_ok = nsplit * cap <= per_row && nsplit <= 64 && nsplit * cap * 8 <= 200 * 1024;
        let n_max = self.host_pos_end / t.ratio.max(1) as usize / t.block.max(1) as usize;
        if m > MAX_M && n_max >= 24576 && t.k as usize <= 2048 {
            const TF_SMEM: u32 = 4096 * 8 + 2048 * 4;
            self.k.get("topk_fast")?.set_max_dynamic_smem(TF_SMEM)?;
            return self.launch("topk_fast", Launch::new(m as u32, 512).smem(TF_SMEM), Args::new().ptr(meta).push(t));
        }
        let sm = topk_cluster_smem(n_cap, t.ratio.max(1) as usize, t.block.max(1) as usize, 8);
        if m <= MAX_M && t.x % 16 == 0 && t.ld % 4 == 0 && sm <= 160 << 10 {
            let cand = cand.filter(|&(_, nc)| t.mask == 0 && sm.max(ops::topk_cand_smem(nc) as u32) <= 160 << 10 && ops::topk_cand_on());
            let repl = if let Some((ids, nc)) = cand {
                Some((ops::TOPK_CL2_KERNEL, ids, nc as i32, sm.max(ops::topk_cand_smem(nc) as u32)))
            } else if ops::topk_cl2() {
                Some((ops::TOPK_CL2_KERNEL, 0, 0, sm))
            } else {
                None
            };
            if let Some((kn, ids, ncl, smc)) = repl {
                let f = self.k.get(kn)?;
                f.set_max_dynamic_smem(smc)?;
                let mut l = Launch::new(8, 1024).grid3((8, m as u32, 1)).smem(smc);
                l.cluster = Some((8, 1, 1));
                return self.launch(kn, l, Args::new().ptr(meta).push(t).ptr(ids).i32(ncl));
            }
            let f = self.k.get("topk_cl_8_1024")?;
            f.set_max_dynamic_smem(sm)?;
            let mut l = Launch::new(8, 1024).grid3((8, m as u32, 1)).smem(sm);
            l.cluster = Some((8, 1, 1));
            return self.launch("topk_cl_8_1024", l, Args::new().ptr(meta).push(t));
        }
        if !two_stage_ok {
            return self.launch("topk_select", Launch::new(m as u32, 1024), Args::new().ptr(meta).push(t));
        }
        let p = TopkPartParams { t, nsplit: nsplit as i32, cap: cap as i32, cv, ci, cn, kth: cn + (Self::score_rows(self.pmax, self.tp) * 64 * 4) as u64 };
        self.launch("topk_kth", Launch::new(nsplit as u32, 512).grid3((nsplit as u32, m as u32, 1)), Args::new().ptr(meta).push(p))?;
        self.launch("topk_part", Launch::new(nsplit as u32, 512).grid3((nsplit as u32, m as u32, 1)), Args::new().ptr(meta).push(p))?;
        let smem = (nsplit * cap * 8) as u32;
        self.launch("topk_merge", Launch::new(m as u32, 1024).smem(smem), Args::new().ptr(meta).push(p))?;
        Ok(())
    }

    fn tmap_fp4(&self, base: DevPtr, rows: usize) -> Result<DevPtr> {
        let key = (base, usize::MAX, rows, 64, 128);
        if let Some(&t) = self.tmaps.borrow().get(&key) {
            return Ok(t);
        }
        ensure!(!self.stream.is_capturing()? && !self.side.is_capturing()?, "fp4 tensor map ({rows} rows) created inside a capture");
        let t = cu::tensor_map_fp4_rows(base, rows as u64, 128)?;
        let b = cu::upload_synced(&t)?;
        let ptr = b.ptr;
        std::mem::forget(b);
        self.tmaps.borrow_mut().insert(key, ptr);
        Ok(ptr)
    }

    fn tmap_bf16(&self, base: DevPtr, cols: usize, rows: usize, ld_bytes: usize, box_rows: u32) -> Result<DevPtr> {
        let key = (base, cols, rows, ld_bytes, box_rows);
        if let Some(&t) = self.tmaps.borrow().get(&key) {
            return Ok(t);
        }
        ensure!(!self.stream.is_capturing()? && !self.side.is_capturing()?, "bf16 tensor map ({rows} x {cols}, box {box_rows}) created inside a capture");
        let t = cu::tensor_map_bf16_sw128(base, cols as u64, rows as u64, ld_bytes as u64, box_rows)?;
        let b = cu::upload_synced(&t)?;
        let ptr = b.ptr;
        std::mem::forget(b);
        self.tmaps.borrow_mut().insert(key, ptr);
        Ok(ptr)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_tc_bf16(&self, tw: crate::weights::TcWeight, w: &Weight, x: DevPtr, ldx: usize, m: usize, out: DevPtr, ldo: usize, o: GemvOpts) -> Result<()> {
        let (_, _, part, xn) = self.tc_scratch();
        let (n, k) = (w.n, w.k);
        ensure!(m <= self.pmax && k % 64 == 0);
        let (a, lda, acols) = match o.prenorm {
            Some((pw, eps)) => {
                let (np, ns) = match o.norm_part {
                    Some((np, ns)) => (np, ns as i32),
                    None => {
                        self.launch("tc_row_sumsq", Launch::new((m as u32).div_ceil(8), 256), Args::new().ptr(x).i32(ldx as i32).i32(m as i32).i32(k as i32).ptr(part))?;
                        (part, 1)
                    }
                };
                self.launch(
                    "tc_norm_rows",
                    Launch::new(m as u32, ((k / 8).div_ceil(32) * 32).min(256) as u32),
                    Args::new().ptr(x).i32(ldx as i32).i32(m as i32).i32(k as i32).ptr(pw).ptr(np).i32(ns).f32(eps).ptr(xn),
                )?;
                (xn, k, k)
            }
            None => (x, ldx, ldx),
        };
        let ta = {
            let mut maps = self.tcb_maps.borrow_mut();
            let key = (a, acols, lda);
            match maps.get(&key) {
                Some(&t) => t,
                None => {
                    ensure!(!self.st().is_capturing()?, "tensor map for the bf16 GEMM activations ({acols} cols) created inside a capture");
                    let t = cu::tensor_map_bf16_sw128(a, acols as u64, self.pmax as u64, (lda * 2) as u64, 128)?;
                    let b = cu::upload_synced(&t)?;
                    let ptr = b.ptr;
                    std::mem::forget(b);
                    maps.insert(key, ptr);
                    ptr
                }
            }
        };
        let epi = matches!(o.epilogue, Epilogue::SqrtSoftplus) as i32;
        let gp = ops::TbParams {
            ta,
            tb: tw.tmap,
            out,
            ldo: ldo as i32,
            out_f32: (o.out_f32 || epi == 1) as i32,
            m: m as i32,
            n: n as i32,
            k: k as i32,
            nt: tw.nt as i32,
            alpha: o.alpha,
            epi,
            a_group_rows: o.x_group_rows as i32,
            a_group_k: k as i32,
        };
        let gx = (n / tw.nt) as u32;
        self.launch("tc_gemm_bf16", Launch::new(gx, 128).grid3((gx, (m as u32).div_ceil(128), 1)).smem(ops::tb_smem(tw.nt)), Args::new().push(gp))
    }

    fn tc_cluster_for(&self, w: &Weight, m: usize, o: &GemvOpts) -> Option<usize> {
        let on = self.tp == 4;
        let shape = on && m <= MAX_M.min(8) && w.fmt == WFmt::Fp8 && o.moe.is_none() && o.iq.is_none() && o.out_pos.is_none() && o.x_group_rows == 0
            && o.xq_out.is_none() && matches!(o.epilogue, Epilogue::Store | Epilogue::Partial) && w.n % 128 == 0 && !o.persist && o.samp_rec.is_none()
            && self.w.tc.get(&w.w).is_some_and(|t| t.tmap128 != 0 && t.kp == w.k && !t.bf16);
        if !shape { return None; }
        if o.epilogue == Epilogue::Partial { ops::tc_partial_cluster(w.n, w.k) } else { ops::tc_gemv_cluster(w.n, w.k) }
    }

    fn gemv(&self, w: &Weight, x: DevPtr, ldx: usize, m: usize, out: DevPtr, ldo: usize, o: GemvOpts) -> Result<()> {
        if m > MAX_M {
            if let Some(tw) = self.w.tc.get(&w.w) {
                if tw.bf16 && o.moe.is_none() && o.xq_out.is_none() && o.split.is_none() && matches!(o.epilogue, Epilogue::Store | Epilogue::SqrtSoftplus) && self.tca.is_some() && w.k <= 8192 {
                    return self.gemm_tc_bf16(*tw, w, x, ldx, m, out, ldo, o);
                }
                if w.fmt == WFmt::Fp8 && tw.kp == w.k && o.x_group_rows == 0 && o.moe.is_none() && o.xq_out.is_none() && matches!(o.epilogue, Epilogue::Store) && self.tca.is_some() && w.k <= 15360 {
                    return self.gemm_tc(*tw, w, x, ldx, m, out, ldo, o);
                }
            }
            return self.gemv_slices(false, w, x, ldx, m, out, ldo, o);
        }
        if let Some(cs) = self.tc_cluster_for(w, m, &o) {
            if let Some(tw) = self.w.tc.get(&w.w).filter(|t| t.tmap128 != 0 && t.kp == w.k && !t.bf16) {
                {
                    let _ = cs;
                    let name = format!("tcgemv {}x{}", w.n, w.k);
                    let (tmap, sf) = (tw.tmap128, tw.sf);
                    return self.timed(&name, || ops::tc_gemv(&self.k, self.st(), tmap, sf, w, x, ldx, m, out, ldo, &GemvOpts { pdl: true, ..o }, self.meta).map(|_| ()));
                }
            }
        }
        let name = format!("gemv {:?} {}x{}{}", w.fmt, w.n, w.k, if o.moe.is_some() { " moe" } else { "" });
        self.timed(&name, || ops::gemv(&self.k, self.st(), w, x, ldx, m, out, ldo, &GemvOpts { pdl: true, ..o }))
    }

    fn tg2_fp8(&self, w: &Weight) -> Option<(DevPtr, DevPtr)> {
        if w.fmt != WFmt::Fp8 {
            return None;
        }
        self.w.tc.get(&w.w).filter(|t| t.tmap128 != 0 && t.kp == w.k && !t.bf16).map(|t| (t.tmap128, t.sf))
    }

    fn tg2_step(&self, m: usize) -> Option<Tg2Bufs> {
        self.tg2.filter(|_| m <= 8 && !self.par_on.get() && !self.eng_on.get())
    }

    fn tg2_wqkv(&self, ly: &Layer, x: DevPtr, m: usize) -> Result<Option<Tg2Bufs>> {
        let qimg = match self.qkv_img.replace((0, 0)) {
            (img, mm) if img != 0 && mm == m && x == self.b.x_raw => img,
            _ => 0,
        };
        let c = &self.cfg;
        let (dim, qlr, hd) = (c.dim, c.q_lora_rank, c.head_dim);
        let Some((t, (tmap, sf), cs)) = self.tg2_step(m).filter(|_| true && ly.wqkv.n == qlr + hd).and_then(|t| Some((t, self.tg2_fp8(&ly.wqkv)?, ops::tg2_cluster_fp8(ly.wqkv.n, dim, 0)?))) else {
            return Ok(None);
        };
        let nsl = dim / 256;
        let p = ops::Tg2Params {
            tmap,
            sfa: sf,
            x,
            gamma: if qimg != 0 { 0 } else { ly.attn_norm },
            bimg: qimg,
            rs_part: self.b.part,
            out: self.b.q_lat,
            out2: self.b.kv,
            qimg: t.qimg,
            qgamma: ly.q_norm,
            qssq: t.qssq,
            alpha: 1.0,
            eps: c.norm_eps,
            m: m as i32,
            n: ly.wqkv.n as i32,
            k: dim as i32,
            ldx: dim as i32,
            ldo: qlr as i32,
            ldo2: hd as i32,
            n_split: qlr as i32,
            rs_n: nsl as i32,
            rs_stride: 26,
            rs_rowstride: (nsl * 26) as i32,
            rs_off: 25,
            qrows: qlr as i32,
            ..Default::default()
        };
        self.tg2_launch(&format!("tcg2 {}x{}", ly.wqkv.n, dim), true, p, cs)?;
        Ok(Some(t))
    }

    #[allow(clippy::too_many_arguments)]
    fn tg2_wqb(&self, t: Option<Tg2Bufs>, w: &Weight, m: usize, qrope: Option<(DevPtr, DevPtr, DevPtr, usize, usize)>, kv_rope: Option<RopeParams>, iq_rows: Option<(usize, Option<(usize, usize)>)>) -> Result<bool> {
        let c = &self.cfg;
        let (qlr, nqh) = (c.q_lora_rank, c.n_heads / self.tp * c.head_dim);
        let kv_tiles = if kv_rope.is_some() { 2 } else { 0 };
        let Some((t, (tmap, sf), cs)) = t.filter(|_| true && w.k == qlr).and_then(|t| Some((t, self.tg2_fp8(w)?, ops::tg2_cluster_fp8(w.n, qlr, kv_tiles)?))) else {
            return Ok(false);
        };
        let mut p = ops::Tg2Params {
            tmap,
            sfa: sf,
            bimg: t.qimg,
            rs_part: t.qssq,
            out: self.b.q,
            meta: self.meta,
            alpha: 1.0,
            eps: c.norm_eps,
            m: m as i32,
            n: w.n as i32,
            k: qlr as i32,
            ldo: nqh as i32,
            bsrc: 1,
            rs_n: (qlr / 32) as i32,
            rs_stride: 1,
            rs_rowstride: (qlr / 32) as i32,
            ..Default::default()
        };
        if let Some((cos, sin, meta, h, off)) = qrope {
            (p.rope_cos, p.rope_sin, p.meta, p.rope_head, p.rope_off) = (cos, sin, meta, h as i32, off as i32);
        }
        if let Some(kvp) = kv_rope {
            p.kvr = kvp;
            p.kvr_on = kv_tiles as i32;
        }
        if let Some((nq, iq)) = iq_rows {
            ensure!(w.n == nqh + nq);
            (p.out2, p.ldo2, p.n_split) = (self.b.iq, nq as i32, nqh as i32);
            if let Some((h, off)) = iq {
                (p.iq_head, p.iq_off) = (h as i32, off as i32);
            }
        } else {
            ensure!(w.n == nqh);
        }
        self.tg2_launch(&format!("tcg2 {}x{}", w.n, qlr), true, p, cs)?;
        Ok(true)
    }

    fn tg2_wo(&self, ly: &Layer, m: usize) -> Result<Option<usize>> {
        let c = &self.cfg;
        let (dim, heads, hd) = (c.dim, c.n_heads / self.tp, c.head_dim);
        let gl = c.o_groups / self.tp * c.o_lora_rank;
        let Some((t, ta, csa, (tmap, sf), csb)) = self
            .tg2_step(m)
            .filter(|_| true && c.o_lora_rank % 128 == 0 && matches!(ly.wo_a.fmt, WFmt::Fp8Bf16Act | WFmt::Bf16) && ly.wo_a.n == gl && ly.wo_b.k == gl && ly.wo_b.n == dim)
            .and_then(|t| {
                let ta = self.w.tc.get(&ly.wo_a.w).filter(|tw| tw.tmap128 != 0 && tw.bf16 && tw.kp == ly.wo_a.k)?.tmap128;
                Some((t, ta, ops::tg2_cluster_bf16(ly.wo_a.n, ly.wo_a.k, 8)?, self.tg2_fp8(&ly.wo_b)?, if deepseek_ultrafast_cuda::profile::sm103_tp8() { 1 } else { ops::tg2_cluster_fp8(ly.wo_b.n, gl, 0)? }))
            })
        else {
            return Ok(None);
        };
        let p = ops::Tg2Params {
            tmap: ta,
            x: self.b.o,
            bimg: 0,
            out: self.b.oa,
            qimg: t.oaimg,
            alpha: 1.0,
            m: m as i32,
            n: gl as i32,
            k: ly.wo_a.k as i32,
            ldx: (heads * hd) as i32,
            ldo: gl as i32,
            group_rows: c.o_lora_rank as i32,
            qrows: gl as i32,
            ..Default::default()
        };
        match self.woa64.get(&ly.wo_a.w).copied().filter(|_| p.bimg == 0 && ops::woa64_on(&self.k, gl, ly.wo_a.k as usize, csa)) {
            Some(map64) => {
                self.timed(&format!("tcg2b {}x{}", gl, ly.wo_a.k), || ops::tc_gemv2_oa64(&self.k, self.st(), p, map64, true))?
            }
            None => {
                self.tg2_launch(&format!("tcg2b {}x{}", gl, ly.wo_a.k), false, p, csa)?;
            }
        }
        let p = ops::Tg2Params { tmap, sfa: sf, bimg: t.oaimg, out: self.b.ob_part, alpha: 1.0, m: m as i32, n: dim as i32, k: gl as i32, ldo: dim as i32, bsrc: 1, partial: 1, ..Default::default() };
        let aa = self.aa_arm(m);
        anyhow::ensure!(!self.aa_pending.get(), "ar push: a wo_b arm is still pending");
        let (csb, applied) = self.tg2_launch_aa(&format!("tcg2 {}x{}", dim, gl), true, p, csb, aa)?;
        if applied {
            self.aa_pending.set(aa.is_some());
        }
        Ok(Some(csb))
    }

    fn tg2_launch(&self, name: &str, fp8: bool, p: ops::Tg2Params, cs: usize) -> Result<usize> {
        Ok(self.tg2_launch_aa(name, fp8, p, cs, None)?.0)
    }

    fn tg2_launch_aa(&self, name: &str, fp8: bool, p: ops::Tg2Params, cs: usize, aa: Option<ops::AaArm>) -> Result<(usize, bool)> {
        let (mut used, mut applied) = (cs, false);
        self.timed(name, || {
            (used, applied) = ops::tc_gemv2_aa(&self.k, self.st(), fp8, p, cs, true, aa)?;
            Ok(())
        })?;
        Ok((used, applied))
    }

    fn aa_arm(&self, m: usize) -> Option<ops::AaArm> {
        let nv = self.nvls?;
        let dim = self.cfg.dim;
        let ok = (self.tp == 4 || self.tp == 8)
            && m == self.cur_m
            && m <= MAX_M
            && dim % 256 == 0
            && dim * m <= NVLS_CAP
            && (m * dim / 4).div_ceil(64) <= NVLS_FLAGS.min(COMM_MAX_BLOCKS);
        if !ok {
            return None;
        }
        if self.tp == 8 {
            return Some(ops::AaArm::Push8(ops::AaPush8 { recv: nv.recv, state: nv.state, meta: self.meta, cap: NVLS_CAP as i32, rank: self.rank as i32 }));
        }
        Some(ops::AaArm::Push(ops::AaPush { recv: [nv.recv[0], nv.recv[1], nv.recv[2], nv.recv[3]], state: nv.state, meta: self.meta, cap: NVLS_CAP as i32, rank: self.rank as i32 }))
    }

    fn comm(&self, src: DevPtr, ld_src: usize, n_per_token: usize, mode: i32, round_bf16: bool, out: DevPtr, ld_out: usize, out_f32: bool, hc: Option<(DevPtr, DevPtr, DevPtr)>) -> Result<()> {
        self.comm_ex(src, ld_src, n_per_token, mode, round_bf16, out, ld_out, out_f32, hc, 1, 0, None)
    }

    fn epi(&self, fn_: DevPtr, identity: bool) -> HcEpi {
        HcEpi { fn_, pre: if identity { 0 } else { self.b.pre_a }, x_raw: self.b.x_raw, part: self.b.part, dim: self.cfg.dim as i32 }
    }

    fn epi_nodots(&self) -> HcEpi {
        self.epi(0, false)
    }

    fn xin(&self, w: DevPtr) -> GemvOpts {
        GemvOpts { prenorm: Some((w, self.cfg.norm_eps)), norm_part: Some((self.b.part, self.cfg.dim / 256)), ..Default::default() }
    }

    fn prep_side_dots(&self) -> bool {
        self.cur_m > MAX_M
    }

    fn hc_prep(&self, m: usize, e: HcEpi) -> Result<()> {
        let e = if self.prep_side_dots() { HcEpi { fn_: 0, ..e } } else { e };
        let smem = if e.fn_ != 0 { 24 * 4 * 256 * 4 } else { 0 };
        let ns = (self.cfg.dim / 256) as u32;
        let name = match deepseek_ultrafast_cuda::profile::sm103() && m <= MAX_M && e.fn_ != 0 {
            true if e.pre != 0 => "variant_zzhc/hc_prep_pd",
            true => "variant_zzhc/hc_prep_d",
            false => "hc_prep",
        };
        let f = self.k.get(name)?;
        f.set_max_dynamic_smem(smem)?;
        self.launch(name, Launch::new(ns, 64).grid3((ns, m as u32, 1)).smem(smem), Args::new().ptr(self.meta).ptr(self.hbuf()).push(e))
    }

    fn embed_prep(&self, m: usize, fn_: DevPtr, cmp_hist: DevPtr) -> Result<bool> {
        if !deepseek_ultrafast_cuda::profile::sm103() || m > MAX_M || fn_ == 0 || self.prep_side_dots() {
            return Ok(false);
        }
        let name = "variant_zzhc/embed_prep";
        let smem = 24 * 4 * 256 * 4;
        self.k.get(name)?.set_max_dynamic_smem(smem)?;
        let e = self.epi(fn_, true);
        let ns = (self.cfg.dim / 256) as u32;
        let cfg = Launch::new(ns, 64).grid3((ns, m as u32, 1));
        self.launch(name, cfg.smem(smem), Args::new().ptr(self.meta).ptr(self.w.embed).ptr(self.hbuf()).ptr(self.engram.token_map).ptr(cmp_hist).push(e))?;
        Ok(true)
    }

    fn embed_meta(&self, m: usize, cmp_hist: DevPtr) -> Result<()> {
        let name = "embed_hc";
        let c = &self.cfg;
        self.launch(name, Launch::new(m as u32, 256), Args::new().ptr(self.meta).ptr(self.w.embed).ptr(self.hbuf()).i32(c.dim as i32).i32(c.hc_mult as i32).ptr(self.engram.token_map).ptr(cmp_hist).ptr(0))
    }

    fn hc_fork_dots(&self, hc: (DevPtr, DevPtr, DevPtr), dots: bool, sub: &'static str) -> Result<()> {
        ensure!(self.hc_late_pending.get().is_none(), "hc side branch: deferred hc_mix not forked");
        let late = sub == "attn" && self.cur_m <= MAX_M && deepseek_ultrafast_cuda::profile::sm103_tp4();
        if !dots {
            if late {
                self.hc_late_pending.set(Some(hc));
                return Ok(());
            }
            return self.hc_fork(hc);
        }
        let c = &self.cfg;
        self.ev_fork.record(&self.stream)?;
        self.side.wait(&self.ev_fork)?;
        let smem = 24 * 4 * 256 * 4;
        if self.cur_m > MAX_M {
            if let Some(&fnb) = self.hc_fnb.get(&hc.0) {
                let f = self.k.get("hc_side_dots_tc")?;
                let smem = (2 * 24 * (4 * 256 + 8) * 2) as u32;
                f.set_max_dynamic_smem(smem)?;
                let ns = c.dim / 256;
                let tile = 128;
                let gy = self.cur_m.div_ceil(tile) as u32;
                cu::launch(f, &Launch::new(ns as u32, 256).grid3((ns as u32, gy, 1)).smem(smem), &self.side, Args::new().ptr(self.meta).ptr(self.hbuf()).ptr(fnb).ptr(self.b.part).i32(c.dim as i32).i32(tile as i32))?;
                return self.hc_mix_on_side(hc);
            }
        }
        let name = if self.cur_m > MAX_M { "hc_side_dots_p" } else { "hc_side_dots" };
        let f = self.k.get(name)?;
        f.set_max_dynamic_smem(smem)?;
        if name == "hc_side_dots_p" {
            let ns = c.dim / 256;
            let gy_max = (2 * ops::sm_count() / ns).max(1);
            let tile = self.cur_m.div_ceil(gy_max).div_ceil(64) * 64;
            let gy = self.cur_m.div_ceil(tile) as u32;
            cu::launch(f, &Launch::new(ns as u32, 128).grid3((ns as u32, gy, 1)).smem(smem), &self.side, Args::new().ptr(self.meta).ptr(self.hbuf()).ptr(hc.0).ptr(self.b.part).i32(c.dim as i32).i32(tile as i32))?;
        } else {
            let gy = self.cur_m.div_ceil(64) as u32;
            let gz: usize = if self.cur_m <= MAX_M { 2 } else { 1 };
            ensure!(24 % gz == 0);
            f.set_max_dynamic_smem(smem)?;
            let cfg = Launch::new((c.dim / 256) as u32, 256).grid3(((c.dim / 256) as u32, gy, gz as u32)).smem((smem as usize / gz) as u32);
            cu::launch(f, &cfg, &self.side, Args::new().ptr(self.meta).ptr(self.hbuf()).ptr(hc.0).ptr(self.b.part).i32(c.dim as i32))?;
        }
        if late {
            self.hc_late_pending.set(Some(hc));
            return Ok(());
        }
        self.hc_mix_on_side(hc)
    }

    fn hc_late_fork(&self) -> Result<()> {
        let Some(hc) = self.hc_late_pending.take() else { return Ok(()) };
        self.hc_fork(hc)
    }

    fn hc_mix_launch(&self, p: HcMix) -> Result<()> {
        let name = if self.cur_m <= MAX_M && ops::variant_cc() { "variant_zzzzzzzzhcm/hc_mix" } else { "hc_mix" };
        let f = self.k.get(name)?;
        if self.meta != self.metas || self.cur_m <= MAX_M {
            return cu::launch(f, &Launch::new(1, 512), &self.side, Args::new().ptr(self.meta).push(p));
        }
        cu::launch(f, &Launch::new(self.cur_m.div_ceil(16) as u32, 512), &self.side, Args::new().ptr(self.metas).push(p))
    }

    fn hc_mix_params(&self, hc: (DevPtr, DevPtr, DevPtr)) -> HcMix {
        let c = &self.cfg;
        HcMix {
            part: self.b.part,
            nslices: (c.dim / 256) as i32,
            base: hc.1,
            scale: hc.2,
            pre: self.b.pre_a,
            post: self.b.post_a,
            comb: self.b.comb_a,
            dim: c.dim as i32,
            iters: c.hc_sinkhorn_iters as i32,
            norm_eps: c.norm_eps,
            hc_eps: c.hc_eps,
        }
    }

    fn hc_mix_on_side(&self, hc: (DevPtr, DevPtr, DevPtr)) -> Result<()> {
        self.hc_mix_launch(self.hc_mix_params(hc))?;
        self.ev_join.record(&self.side)
    }

    fn hc_fork(&self, hc: (DevPtr, DevPtr, DevPtr)) -> Result<()> {
        self.ev_fork.record(&self.stream)?;
        self.side.wait(&self.ev_fork)?;
        self.hc_mix_launch(self.hc_mix_params(hc))?;
        self.ev_join.record(&self.side)
    }

    fn hc_join(&self) -> Result<()> {
        ensure!(self.hc_late_pending.get().is_none(), "hc side branch joined before its deferred hc_mix was forked");
        self.stream.wait(&self.ev_join)
    }

    fn par_fork(&self) -> Result<()> {
        self.ev_pfork.record(&self.stream)?;
        self.par.wait(&self.ev_pfork)
    }

    fn par_join(&self) -> Result<()> {
        self.ev_pjoin.record(&self.par)?;
        self.stream.wait(&self.ev_pjoin)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_par(&self, w: &Weight, x: DevPtr, ldx: usize, m: usize, out: DevPtr, ldo: usize, o: GemvOpts) -> Result<()> {
        if m > MAX_M {
            return self.gemv_slices(true, w, x, ldx, m, out, ldo, o);
        }
        ops::gemv(&self.k, &self.par, w, x, ldx, m, out, ldo, &GemvOpts { pdl: true, ..o })
    }

    #[allow(clippy::too_many_arguments)]
    fn comm_ex(&self, src: DevPtr, ld_src: usize, n_per_token: usize, mode: i32, round_bf16: bool, out: DevPtr, ld_out: usize, out_f32: bool, hc: Option<(DevPtr, DevPtr, DevPtr)>, nsum: usize, src2: DevPtr, epi: Option<HcEpi>) -> Result<()> {
        ensure!(n_per_token * self.cur_m <= self.comm.cap && n_per_token % 4 == 0, "comm payload {n_per_token} x {} rows exceeds the channel's {} elements", self.cur_m, self.comm.cap);
        let aa = self.aa_pending.replace(false);
        let lp = self.lp_pending.replace(false);
        let base_mode = mode & 0x3f;
        ensure!(mode & CM_SRC16 == 0 || (self.cur_m > MAX_M && self.tp > 1), "bf16 comm sources need the two-shot path");
        if self.cur_m > MAX_M && self.tp > 1 && (base_mode == CM_SUM_HCPOST || (base_mode == CM_SUM && !out_f32)) && n_per_token % 256 == 0 {
            let short = self.short_append && self.comm.short_state != 0 && self.cur_m <= COMM_SHORT_ROWS && n_per_token * self.cur_m <= COMM_SHORT_CAP
                && epi.is_none_or(|e| e.fn_ == 0);
            let (ro, ao, cap, state, slices) = if short {
                (self.comm.short_rs, self.comm.short_ag, COMM_SHORT_CAP, self.comm.short_state, 1)
            } else {
                (CommRank::rs_off(self.tp, self.comm.cap), CommRank::ag_off(self.tp, self.comm.cap), self.comm.cap, self.comm.state2, 16)
            };
            let mut p = Comm2Params {
                rs: [0; MAX_TP],
                ag: [0; MAX_TP],
                state,
                tp: self.tp as i32,
                rank: self.rank as i32,
                cap: cap as i32,
                src,
                ld_src: ld_src as i32,
                n_per_token: n_per_token as i32,
                mode: mode | 0x40,
                round_bf16: round_bf16 as i32,
                out,
                ld_out: ld_out as i32,
                out_f32: out_f32 as i32,
                res: hc.map(|h| h.0).unwrap_or(0),
                post: hc.map(|h| h.1).unwrap_or(0),
                comb: hc.map(|h| h.2).unwrap_or(0),
                nsum: nsum as i32,
                src2,
                epi: epi.is_some() as i32,
                e: epi.unwrap_or_default(),
            };
            for (i, r) in self.peers.iter().enumerate() {
                p.rs[i] = *r + ro;
                p.ag[i] = *r + ao;
            }
            let virtual_blocks = (self.cur_m * n_per_token).div_ceil(256 * slices);
            let blocks = virtual_blocks.min(ops::hw().comm_ctas_per_sm * self.sms as usize);
            ensure!(blocks <= COMM_MAX_BLOCKS && virtual_blocks <= blocks * 8);
            let smem = if epi.is_some_and(|e| e.fn_ != 0) { 24 * 4 * 256 * 4 } else { 0 };
            ensure!(!aa, "ar push: a prefill all-reduce after the push producer");
            ensure!(!lp, "ffn lp: a prefill all-reduce after the Lamport-armed MoE rows");
            ensure!(!self.sgl_pending.get(), "sgl moe: a two-shot all-reduce after GEMM2");
            let name = match (short, self.tp) {
                (true, 8) => "variant_zzzzzzzztp8/comm_push2_k1_tp8".to_string(),
                (true, tp) => format!("comm_push2_k1_tp{tp}"),
                (false, tp) => format!("comm_push2_tp{tp}"),
            };
            self.k.get(&name)?.set_max_dynamic_smem(smem)?;
            return self.launch(&name, Launch::new(blocks as u32, 64).smem(smem), Args::new().ptr(self.meta).push(p));
        }
        let mut p = CommParams {
            recv: [0; MAX_TP],
            state: self.comm.state,
            tp: self.tp as i32,
            rank: self.rank as i32,
            cap: self.comm.cap as i32,
            src,
            ld_src: ld_src as i32,
            n_per_token: n_per_token as i32,
            mode,
            round_bf16: round_bf16 as i32,
            out,
            ld_out: ld_out as i32,
            out_f32: out_f32 as i32,
            res: hc.map(|h| h.0).unwrap_or(0),
            post: hc.map(|h| h.1).unwrap_or(0),
            comb: hc.map(|h| h.2).unwrap_or(0),
            nsum: nsum as i32,
            src2,
            epi: epi.is_some() as i32,
            e: epi.unwrap_or_default(),
            mx: HcMix::default(),
            mx_on: 0,
        };
        for (i, r) in self.peers.iter().enumerate() {
            p.recv[i] = *r;
        }
        if let (Some(nv), CM_SUM_HCPOST) = (self.nvls, base_mode) {
            if mode == base_mode && n_per_token * self.cur_m <= NVLS_CAP {
                let mut c = p;
                c.state = nv.state;
                c.cap = NVLS_CAP as i32;
                c.recv = nv.recv;
                let mixf = if lp && self.lp_mixf_on.replace(false) { self.lp_mixf } else { 0 };
                let sg = self.sgl_pending.replace(false);
                let (sg_perm, sg_w, sg_rec) = match (&self.sgl, sg) {
                    (Some(st), true) => (st.permuted, st.out_w, st.record),
                    _ => (0, 0, 0),
                };
                let (sg_gamma, sg_img, sg_early) = if sg { self.sg_next.replace((0, 0, 0)) } else { (0, 0, 0) };
                let q = NvlsParams { c, mixf, sg_perm, sg_w, sg_rec, sg_gamma, sg_img, sg_early, _sgpad: 0 };
                let blocks = (self.cur_m * n_per_token / 4).div_ceil(64).max(1);
                ensure!(blocks <= COMM_MAX_BLOCKS && blocks <= NVLS_FLAGS);
                let smem = if epi.is_some_and(|e| e.fn_ != 0) { 24 * 4 * 256 * 4 } else { 0 };
                let dec_ok = (self.tp == 4 || self.tp == 8) && epi.is_some_and(|e| e.fn_ == 0) && c.mx_on == 0 && nsum <= 8 && n_per_token % 256 == 0;
                let dm = if self.tp == 8 { "variant_zzzzzzzztp8/comm_dec8" } else { "variant_zcomm_dec/comm_dec" };
                let name: std::borrow::Cow<'static, str> = if dec_ok && round_bf16 && src2 == 0 {
                    format!("{dm}_attn_{}", nsum.max(1)).into()
                } else if dec_ok && !round_bf16 && src2 != 0 {
                    format!("{dm}_ffn_{}", nsum.max(1)).into()
                } else if dec_ok {
                    format!("{dm}_{}", if round_bf16 { "x16" } else { "f32" }).into()
                } else {
                    "comm_nvls_ucpush".into()
                };
                ensure!(!aa || (dec_ok && round_bf16 && src2 == 0 && smem == 0 && mode == base_mode), "ar push: the all-reduce after wo_b's push arm is not comm_dec_attn");
                let name: std::borrow::Cow<'static, str> = match (aa, self.tp) {
                    (true, 8) => "variant_zzzzzzaa/comm_dec8_attn_pp".into(),
                    (true, _) => "variant_zzzzzzaa/comm_dec_attn_pp".into(),
                    _ => name,
                };
                let (name, grid, threads): (std::borrow::Cow<'static, str>, usize, u32) = if lp {
                    let k = ffn_lp_kernel(nsum, blocks).filter(|_| !aa && dec_ok && !round_bf16 && src2 != 0 && smem == 0 && mode == base_mode && name.starts_with("variant_zcomm_dec/comm_dec_ffn_"));
                    let (n, g, t) = k.ok_or_else(|| anyhow::anyhow!("ffn lp: the all-reduce after the Lamport-armed MoE rows is not comm_dec_ffn ({name})"))?;
                    (n.into(), g, t)
                } else if sg {
                    ensure!(!aa && !round_bf16 && smem == 0 && mode == base_mode && matches!(nsum, 3 | 6) && blocks % 2 == 0 && n_per_token % 256 == 0,
                        "sgl moe: the FFN all-reduce is not on comm_dec's channel ({name})");
                    let k = format!("variant_zzzzzzzzzlp/comm_lp_sgl{}_{nsum}_2", if self.tp == 8 { "8" } else { "" });
                    (k.into(), blocks / 2, 160)
                } else {
                    (name, blocks, 64)
                };
                self.k.get(&name)?.set_max_dynamic_smem(smem)?;
                return self.launch(&name, Launch::new(grid as u32, threads).smem(smem), Args::new().ptr(self.meta).push(q));
            }
        }
        ensure!(!aa, "ar push: the all-reduce after wo_b's push arm left the NVLS channel");
        ensure!(!lp, "ffn lp: the all-reduce after the Lamport-armed MoE rows left the NVLS channel");
        ensure!(!self.sgl_pending.get(), "sgl moe: the FFN all-reduce left the NVLS channel");
        let blocks = (self.cur_m * n_per_token / 4).div_ceil(64).max(1);
        ensure!(blocks <= COMM_MAX_BLOCKS);
        let smem = if epi.is_some_and(|e| e.fn_ != 0) { 24 * 4 * 256 * 4 } else { 0 };
        let name = "comm_push";
        self.k.get(name)?.set_max_dynamic_smem(smem)?;
        self.launch(name, Launch::new(blocks as u32, 64).smem(smem), Args::new().ptr(self.meta).push(p))
    }

    #[allow(clippy::too_many_arguments)]
    fn rope(&self, x: DevPtr, ldx: usize, n: usize, rows_per_token: usize, m: usize, table: (DevPtr, DevPtr), inverse: bool, quant: i32, ratio: usize, dst: DevPtr) -> Result<()> {
        self.rope_n(x, ldx, n, rows_per_token, m, table, inverse, quant, ratio, dst, 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn rope_n(&self, x: DevPtr, ldx: usize, n: usize, rows_per_token: usize, m: usize, table: (DevPtr, DevPtr), inverse: bool, quant: i32, ratio: usize, dst: DevPtr, norm_w: DevPtr) -> Result<()> {
        let p = self.rope_params(x, ldx, n, rows_per_token, table, inverse, quant, ratio, dst, norm_w);
        self.rope_launch(p, m * rows_per_token)
    }

    fn rope_launch(&self, p: RopeParams, rows: usize) -> Result<()> {
        if rows >= 1024 {
            return self.launch("rope_quant4", Launch::new(rows.div_ceil(4) as u32, 256), Args::new().ptr(self.meta).push(p));
        }
        self.launch("rope_quant", Launch::new(rows as u32, 64), Args::new().ptr(self.meta).push(p))
    }

    fn rope_pair(&self, m: usize, a: RopeParams, ra: usize, b: RopeParams, rb: usize) -> Result<()> {
        let n1 = (m * ra) as i32;
        self.launch("rope_quant2", Launch::new((m * (ra + rb)) as u32, 64), Args::new().ptr(self.meta).push(a).push(b).i32(n1))
    }

    #[allow(clippy::too_many_arguments)]
    fn rope_params(&self, x: DevPtr, ldx: usize, n: usize, rows_per_token: usize, table: (DevPtr, DevPtr), inverse: bool, quant: i32, ratio: usize, dst: DevPtr, norm_w: DevPtr) -> RopeParams {
        RopeParams {
            x,
            ldx: ldx as i32,
            n: n as i32,
            rows_per_token: rows_per_token as i32,
            rope_off: (n - self.cfg.rope_head_dim) as i32,
            cos: table.0,
            sin: table.1,
            inverse: inverse as i32,
            quant,
            ratio: ratio as i32,
            dst,
            pos_offset: 0,
            norm_w,
            norm_eps: self.cfg.norm_eps,
            dst_by_row: 0,
            q4: 0,
            sf4: 0,
        }
    }

    pub fn set_prefill(&mut self, pos: usize, tokens: &[i32]) -> Result<()> {
        let n = tokens.len();
        ensure!(n >= 1 && n <= self.pmax);
        ensure!(pos + n <= self.max_seq, "sequence exceeds max_seq {}", self.max_seq);
        self.stream.synchronize()?;
        let mut all = vec![Meta { pos: pos as i32, m: n as i32, tokens: [0; 16] }];
        for (k, ch) in tokens.chunks(MAX_M).enumerate() {
            let mut mt = Meta { pos: (pos + k * MAX_M) as i32, m: ch.len() as i32, tokens: [0; 16] };
            mt.tokens[..ch.len()].copy_from_slice(ch);
            all.push(mt);
        }
        if n <= MAX_M {
            all[0].tokens[..n].copy_from_slice(tokens);
        }
        let bytes = unsafe { std::slice::from_raw_parts(all.as_ptr() as *const u8, all.len() * std::mem::size_of::<Meta>()) };
        cu::memcpy_htod_async(self.metas, bytes, &self.stream)?;
        let (r0, r1) = shard_rows(self.rank, n, self.tp);
        let sm = [Meta { pos: (pos + r0) as i32, m: (r1 - r0) as i32, tokens: [0; 16] }];
        let sb = unsafe { std::slice::from_raw_parts(sm.as_ptr() as *const u8, std::mem::size_of::<Meta>()) };
        cu::memcpy_htod_async(self.metas + ((meta_slices(self.pmax) + 1) * std::mem::size_of::<Meta>()) as u64, sb, &self.stream)?;
        cu::memcpy_htod_async(self.ptoks, tokens, &self.stream)?;
        self.meta = self.metas;
        self.host_pos_end = pos + n;
        self.cur_m = n;
        Ok(())
    }

    fn row_view(&self, off: usize) -> Bufs {
        let c = &self.cfg;
        let (dim, hd, tp) = (c.dim, c.head_dim, self.tp);
        let heads = c.n_heads / tp;
        let inter = c.moe_inter_dim / tp;
        let kpad = inter.div_ceil(64) * 64;
        let n_exp = c.n_routed_experts.max(c.dspark_n_routed_experts);
        let (f, bf) = (4u64, 2u64);
        let o = off as u64;
        let tk = c.n_activated_experts as u64;
        let b = self.b;
        let r = |p: DevPtr, row: u64| p + o * row;
        Bufs {
            h: [r(b.h[0], 4 * dim as u64 * bf), r(b.h[1], 4 * dim as u64 * bf)],
            x: r(b.x, dim as u64 * bf),
            dots: r(b.dots, 32 * f),
            pre_a: r(b.pre_a, 4 * f),
            post_a: r(b.post_a, 4 * f),
            comb_a: r(b.comb_a, 16 * f),
            pre_f: r(b.pre_f, 4 * f),
            post_f: r(b.post_f, 4 * f),
            comb_f: r(b.comb_f, 16 * f),
            q_lat: r(b.q_lat, c.q_lora_rank as u64 * bf),
            qr: r(b.qr, c.q_lora_rank as u64 * bf),
            q: r(b.q, (heads * hd) as u64 * bf),
            kv: r(b.kv, hd as u64 * bf),
            o: r(b.o, (heads * hd) as u64 * bf),
            oa: r(b.oa, (c.o_groups / tp * c.o_lora_rank) as u64 * bf),
            ob: r(b.ob, dim as u64 * f),
            iq: r(b.iq, (c.index_n_heads * c.index_head_dim) as u64 * bf),
            iq4: b.iq4,
            iw: r(b.iw, 64 * bf),
            ckv: r(b.ckv, hd as u64 * f),
            csc: r(b.csc, hd as u64 * f),
            latent: r(b.latent, hd as u64 * bf),
            ik: r(b.ik, c.index_head_dim as u64 * bf),
            glog: r(b.glog, n_exp as u64 * f),
            route_ids: r(b.route_ids, tk * 4),
            route_w: r(b.route_w, tk * f),
            act: r(b.act, tk * inter as u64 * bf),
            sact: r(b.sact, inter as u64 * bf),
            xq: r(b.xq, (dim + 32) as u64 * bf),
            xsq: r(b.xsq, (dim / 32) as u64 * f),
            actq: r(b.actq, tk * (kpad + 32) as u64 * bf),
            asq: r(b.asq, tk * (kpad / 32) as u64 * f),
            y: r(b.y, dim as u64 * f),
            ypair: r(b.ypair, tk * dim as u64 * f),
            eg: r(b.eg, 24 * 256 * f),
            ex: r(b.ex, 24 * 256 * bf),
            ekv_part: r(b.ekv_part, (5 * dim / tp) as u64 * f),
            ekv: r(b.ekv, 5 * dim as u64 * bf),
            main_hidden: r(b.main_hidden, 3 * dim as u64 * bf),
            main_x: r(b.main_x, dim as u64 * bf),
            x_raw: r(b.x_raw, dim as u64 * bf),
            part: r(b.part, (dim / 256 * 26) as u64 * f),
            amax_v: r(b.amax_v, f),
            amax_i: r(b.amax_i, 4),
            amax_pack: r(b.amax_pack, 4 * f),
            amax_all: r(b.amax_all, 4 * tp as u64 * f),
            next: r(b.next, 4),
            ..b
        }
    }

    fn sliced<F: FnMut(&mut Self, usize) -> Result<()>>(&mut self, m: usize, mut f: F) -> Result<()> {
        if m <= MAX_M || self.meta != self.metas {
            return f(self, m);
        }
        let b0 = self.b;
        let r = self.sub16(m, |me, off, rows| {
            me.b = me.row_view(off);
            let r = f(me, rows);
            me.b = b0;
            r
        });
        self.b = b0;
        r
    }

    pub fn end_prefill(&mut self) {
        self.meta = self.meta_step;
    }

    fn stream_kv_source(&self, l: usize, m: usize) -> Result<()> {
        self.attn_side(l, m)
    }

    fn stash_tail(&mut self, m: usize) -> Result<()> {
        let w = self.cfg.window_size;
        let state = self.tail.state.get();
        let (start, len, first, src, k) = tail_append(state.start, state.len, m, w);
        let (hrow, prow) = (4 * self.cfg.dim * 2, 4 * 4);
        let (hsrc, psrc) = (self.hbuf(), self.b.pre_a);
        let run = (w - first).min(k);
        cu::memcpy_dtod_async(self.tail.h + (first * hrow) as u64, hsrc + (src * hrow) as u64, run * hrow, &self.stream)?;
        cu::memcpy_dtod_async(self.tail.pre + (first * prow) as u64, psrc + (src * prow) as u64, run * prow, &self.stream)?;
        if run < k {
            cu::memcpy_dtod_async(self.tail.h, hsrc + ((src + run) * hrow) as u64, (k - run) * hrow, &self.stream)?;
            cu::memcpy_dtod_async(self.tail.pre, psrc + ((src + run) * prow) as u64, (k - run) * prow, &self.stream)?;
        }
        self.tail.state.set(EncoderTail { start, len, end: self.host_pos_end });
        Ok(())
    }

    pub(crate) fn encoder_tail_state(&self) -> EncoderTail { self.tail.state.get() }

    pub(crate) fn restore_encoder_tail_state(&self, state: EncoderTail) { self.tail.state.set(state); }

    pub(crate) fn encoder_tail_segments(&self) -> [(DevPtr, usize); 2] {
        [(self.tail.h, self.cfg.window_size * 4 * self.cfg.dim * 2), (self.tail.pre, self.cfg.window_size * 4 * 4)]
    }

    fn replay_decoder(&mut self, toks: &[i32]) -> Result<i32> {
        let c = self.cfg.clone();
        let n = toks.len();
        let w = c.window_size;
        let s = n.saturating_sub(w);
        let m = n - s;
        let tail = self.tail.state.get();
        ensure!(m >= 1 && tail.len >= m && tail.end == n, "encoder tail {tail:?} does not cover replay [{s}, {n})");
        let (hrow, prow) = (4 * c.dim * 2, 4 * 4);
        let first = (tail.start + tail.len - m) % w;
        let run = (w - first).min(m);
        self.hcur = 0;
        cu::memcpy_dtod_async(self.hbuf(), self.tail.h + (first * hrow) as u64, run * hrow, &self.stream)?;
        cu::memcpy_dtod_async(self.b.pre_a, self.tail.pre + (first * prow) as u64, run * prow, &self.stream)?;
        if run < m {
            cu::memcpy_dtod_async(self.hbuf() + (run * hrow) as u64, self.tail.h, (m - run) * hrow, &self.stream)?;
            cu::memcpy_dtod_async(self.b.pre_a + (run * prow) as u64, self.tail.pre, (m - run) * prow, &self.stream)?;
        }
        if m > MAX_M {
            self.set_prefill(s, &toks[s..])?;
        } else {
            let mut meta = Meta { pos: s as i32, m: m as i32, tokens: [0; 16] };
            meta.tokens[..m].copy_from_slice(&toks[s..]);
            self.set_meta(&meta)?;
        }
        let dec0 = *c.kv_source_layers.iter().max().unwrap();
        self.win_floor.set(s as i32);
        self.swa_replay.set(true);
        let r = self.forward_impl(m, dec0, c.n_layers, false, true, None, None);
        self.win_floor.set(0);
        self.swa_replay.set(false);
        r?;
        if m > MAX_M {
            self.dspark_prep_tail(m, 0)?;
        } else {
            self.dspark_prep(m)?;
        }
        let next = self.read_next(m)?;
        Ok(next[m - 1])
    }

    pub fn prefill(&mut self, toks: &[i32], from: usize, chunk: usize) -> Result<i32> {
        self.short_append = (short_append_eligible(self.tail.state.get(), from, toks.len(), self.cfg.window_size, self.short_ready)
            || short_cold_eligible(from, toks.len()))
            && self.vision.as_ref().is_none_or(|v| !v.overlaps(from, toks.len().saturating_sub(from)));
        self.image_prefill = true;
        let result = self.prefill_inner(toks, from, chunk);
        self.image_prefill = false;
        self.short_append = false;
        if result.is_ok() && toks.len() >= self.cfg.window_size { self.short_ready = true; }
        result
    }

    pub fn stage_images(&mut self, images: &[deepseek_ultrafast_model::vision::Image], toks: &[i32], from: usize) -> Result<()> {
        if let Some(v) = &mut self.vision {
            v.stage(&self.k, &self.stream, images, toks, from)
        } else {
            ensure!(images.is_empty(), "vision weights were not loaded");
            Ok(())
        }
    }

    fn image_rows(&self, m: usize) -> bool {
        self.image_prefill && self.vision.as_ref().is_some_and(|v| v.overlaps(self.host_pos_end.saturating_sub(m), m))
    }

    fn prefill_inner(&mut self, toks: &[i32], from: usize, chunk: usize) -> Result<i32> {
        let n = toks.len();
        ensure!(from < n && n <= self.max_seq, "prefill requires from < tokens.len() <= max_seq");
        ensure!(chunk > 0 && chunk <= self.pmax, "prefill chunk must be in 1..={} (this rank's buffers)", self.pmax);
        let dec0 = self.cfg.n_layers / 2;
        ensure!(dec0 > 0 && self.cfg.window_size == 128
            && self.cfg.kv_source_layers.iter().copied().max() == Some(dec0)
            && self.cfg.compress_ratios[dec0] == 1
            && self.cfg.engram_layer_ids.iter().all(|&l| l < dec0)
            && self.cfg.dspark_target_layer_ids.iter().all(|&l| l >= dec0),
            "CED requires the V4.1 encoder/decoder KV-source layout and a 128-token window");
        let s = n.saturating_sub(self.cfg.window_size);
        let kept = self.tail.state.get().retain(s, from, self.cfg.window_size);
        self.tail.state.set(kept.unwrap_or_default());
        if s < from && kept.is_none() {
            self.swa_replay.set(true);
            self.encoder_readonly.set(true);
            let r = (|| -> Result<()> {
                self.set_prefill(s, &toks[s..from])?;
                self.forward_impl(from - s, 0, dec0, true, false, None, None)?;
                self.stash_tail(from - s)
            })();
            self.swa_replay.set(false);
            self.encoder_readonly.set(false);
            r?;
        }
        let mut pos = from;
        while pos < n {
            let m = (n - pos).min(chunk);
            if m > MAX_M {
                self.set_prefill(pos, &toks[pos..pos + m])?;
            } else {
                let mut meta = Meta { pos: pos as i32, m: m as i32, tokens: [0; 16] };
                meta.tokens[..m].copy_from_slice(&toks[pos..pos + m]);
                self.set_meta(&meta)?;
            }
            self.forward_impl(m, 0, dec0, true, false, None, None)?;
            self.stream_kv_source(dec0, m)?;
            self.stash_tail(m)?;
            pos += m;
        }
        let tok = self.replay_decoder(toks)?;
        self.end_prefill();
        Ok(tok)
    }

    fn sub16<F: FnMut(&mut Self, usize, usize) -> Result<()>>(&mut self, m: usize, mut f: F) -> Result<()> {
        if m <= MAX_M || self.meta != self.metas {
            return f(self, 0, m);
        }
        let (meta0, cur0) = (self.meta, self.cur_m);
        for k in 0..m.div_ceil(MAX_M) {
            let rows = MAX_M.min(m - k * MAX_M);
            self.meta = self.metas + ((1 + k) * std::mem::size_of::<Meta>()) as u64;
            self.cur_m = rows;
            let r = f(self, k * MAX_M, rows);
            if r.is_err() {
                self.meta = meta0;
                self.cur_m = cur0;
                return r;
            }
        }
        self.meta = meta0;
        self.cur_m = cur0;
        Ok(())
    }

    pub fn set_meta(&mut self, meta: &Meta) -> Result<()> {
        ensure!(meta.m >= 1 && meta.m as usize <= MAX_M);
        ensure!((meta.pos + meta.m) as usize <= self.max_seq, "sequence exceeds max_seq {}", self.max_seq);
        self.meta = self.meta_step;
        self.host_pos_end = (meta.pos + meta.m) as usize;
        self.cur_m = meta.m as usize;
        let m = meta.m as usize;
        if false && self.shared_m != m {
            let mut plan = vec![0i32; (64 << 10) / 4];
            let ms = 97usize;
            plan[0] = 1;
            plan[1] = 0;
            plan[1 + ms] = m as i32;
            let tok = 1 + 2 * ms;
            let pair = tok + ms * 16;
            let w = pair + ms * 16;
            for i in 0..m {
                plan[tok + i] = i as i32;
                plan[pair + i] = i as i32;
                plan[w + i] = 1.0f32.to_bits() as i32;
            }
            cu::memcpy_htod_async(self.b.shared_plan, &plan, &self.stream)?;
            self.stream.synchronize()?;
            self.shared_m = m;
        }
        cu::memcpy_htod_async(self.meta, std::slice::from_ref(meta), &self.stream)?;
        self.stream.synchronize()
    }

    fn shared_plan(&self, m: usize) -> DevPtr {
        let words = 1 + 2 * 97 + 3 * 97 * 16;
        self.b.shared_plan + ((m - 1) * words * 4) as u64
    }

    pub fn read_next(&self, m: usize) -> Result<Vec<i32>> {
        self.stream.synchronize()?;
        cu::download(self.b.next, m)
    }

    pub fn read_h(&self, m: usize) -> Result<Vec<u16>> {
        self.stream.synchronize()?;
        cu::download(self.hbuf(), m * 4 * self.cfg.dim)
    }

    pub fn read_last_logits(&self) -> Result<Vec<f32>> {
        let v = self.cfg.vocab_size / self.tp;
        let all = self.read_logits(self.head_rows)?;
        Ok(all[(self.head_rows - 1) * v..].to_vec())
    }

    pub fn read_logits(&self, m: usize) -> Result<Vec<f32>> {
        self.stream.synchronize()?;
        let (v, ld) = (self.cfg.vocab_size / self.tp, self.w.head.n);
        let all: Vec<f32> = cu::download(self.b.logits, m * ld)?;
        Ok((0..m).flat_map(|i| all[i * ld..i * ld + v].iter().copied()).collect())
    }

    fn hbuf(&self) -> DevPtr {
        self.b.h[self.hcur]
    }

    pub fn read_out(&self, n: usize) -> Result<Vec<i32>> {
        self.stream.synchronize()?;
        cu::download(self.b.out_tokens, n)
    }

    pub fn inject(&mut self, h: &[u16], pre: &[f32]) -> Result<()> {
        self.stream.synchronize()?;
        cu::memcpy_htod_async(self.hbuf(), h, &self.stream)?;
        cu::memcpy_htod_async(self.b.pre_a, pre, &self.stream)?;
        self.stream.synchronize()
    }

    pub fn forward(&mut self, m: usize, n_layers: usize, mut trace: Option<&mut Vec<Vec<u16>>>) -> Result<()> {
        self.moe_rows_ok(m)?;
        self.forward_impl(m, 0, n_layers, true, true, trace.as_deref_mut(), None)
    }

    fn moe_rows_ok(&self, rows: usize) -> Result<()> {
        if rows <= 8 {
            return Ok(());
        }
        ensure!(
            rows > MAX_M || !self.w.layers.iter().any(|l| l.experts.rm),
            "a {rows}-row decode step (the verify of a {}-token draft block) is not supported: the tensor-core MoE takes at most 8 rows. Use a draft block <= 7",
            rows - 1
        );
        ensure!(
            rows > MAX_M || self.tg2.is_none(),
            "a {rows}-row decode step (the verify of a {}-token draft block) is not supported: the tcgen05 decode GEMVs (wqkv, wq_b, wo_a/wo_b, gate) serve steps of at most 8 rows, so this step would take the mma.sync GEMVs while plain decode takes the tcgen05 ones (different numerics: spec != plain). Use a draft block <= 7",
            rows - 1
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_impl(
        &mut self,
        m: usize,
        lo: usize,
        hi: usize,
        embed: bool,
        head: bool,
        mut trace: Option<&mut Vec<Vec<u16>>>,
        inject: Option<&dyn Fn(usize) -> Option<(Vec<u16>, Vec<f32>)>>,
    ) -> Result<()> {
        let c = self.cfg.clone();
        let dim = c.dim;
        self.hcur = 0;
        let hist = if self.encoder_readonly.get() { 0 } else { self.cmp_hist };
        let fused = embed && lo == 0 && !(m > MAX_M && self.meta == self.metas) && !self.image_rows(m) && self.embed_prep(m, self.w.layers[lo].hc_attn.0, hist)?;
        if embed && !fused {
            if m > MAX_M && self.meta == self.metas {
                self.launch(
                    "embed_hc",
                    Launch::new(m as u32, 256),
                    Args::new().ptr(self.meta).ptr(self.w.embed).ptr(self.hbuf()).i32(dim as i32).i32(c.hc_mult as i32).ptr(self.engram.token_map).ptr(hist).ptr(self.ptoks),
                )?;
            } else {
                self.embed_meta(m, hist)?;
            }
        }
        if embed && self.image_rows(m) {
            self.vision.as_ref().unwrap().embed(&self.k, &self.stream, self.hbuf(), hist, self.host_pos_end - m, m, c.hc_mult)?;
        }
        if !fused {
            self.hc_prep(m, self.epi(self.w.layers[lo].hc_attn.0, lo == 0))?;
        }
        self.engram_fork(m, lo, hi)?;
        for l in lo..hi {
            if let Some(f) = inject {
                if let Some((h, pre)) = f(l) {
                    self.inject(&h, &pre)?;
                    self.hc_prep(m, self.epi(self.w.layers[l].hc_attn.0, false))?;
                }
            }
            let r = self.layer(l, m);
            r?;
            if l + 1 == hi && self.cand_pending.replace(false) {
                self.par_join()?;
            }
            if let Some(t) = trace.as_deref_mut() {
                t.push(self.read_h(m)?);
            }
        }
        if !head {
            return Ok(());
        }
        if m > MAX_M && self.meta == self.metas {
            let last = (m - 1) / MAX_M;
            let (meta0, b0, cur0) = (self.meta, self.b, self.cur_m);
            self.meta = self.metas + ((1 + last) * std::mem::size_of::<Meta>()) as u64;
            self.b = self.row_view(last * MAX_M);
            self.cur_m = m - last * MAX_M;
            let r = self.head(self.cur_m);
            self.meta = meta0;
            self.b = b0;
            self.cur_m = cur0;
            return r;
        }
        if self.spec_prep_par.get() {
            if self.prep_proj_pending.replace(false) {
                self.par_join()?;
                self.prep_gather(m)?;
                self.par_fork()?;
                self.on_par(|me| me.prep_kv(m))?;
            } else {
                self.par_fork()?;
                self.par_on.set(true);
                self.prep_gather_pending.set(true);
                let r = self.dspark_prep(m);
                self.par_on.set(false);
                r?;
            }
        }
        let r = self.head(m);
        r
    }

    fn head(&mut self, m: usize) -> Result<()> {
        ensure!(m <= MAX_M, "head: {m} rows (the logits buffer holds MAX_M)");
        self.head_rows = m;
        let c = self.cfg.clone();
        let dim = c.dim;
        let (vloc, vpad) = (c.vocab_size / self.tp, self.w.head.n);
        let srec = (m <= 8).then_some((self.samp.0, self.samp_rec, vloc, self.rank * vloc));
        let head = self.w.head_fp8.as_ref().filter(|_| ops::verify_head_fp8() && m <= 8).unwrap_or(&self.w.head);
        self.gemv(head, self.b.x_raw, dim, m, self.b.logits, vpad, GemvOpts { out_f32: true, persist: true, samp_rec: srec, ..self.xin(self.w.norm) })?;
        self.pick_head(m, srec.is_some())
    }

    fn pick_head(&mut self, m: usize, records: bool) -> Result<()> {
        let (vloc, vpad) = (self.cfg.vocab_size / self.tp, self.w.head.n);

        ensure!(m <= 8 || !self.spec_prep_par.get(), "sampling verify head: at most 8 rows");
        let mut q = SampIo {
            logits: self.b.logits,
            vloc: vloc as i32,
            vpad: vpad as i32,
            offset: (self.rank * vloc) as i32,
            qn: (self.mk_grid.get() * 4) as i32,
            drafts: self.b.drafts,
            vq: self.samp.3,
            qstat: self.samp.4,
            sp: self.samp.0,
            recv: [0; MAX_TP],
            tp: self.tp as i32,
            rank: self.rank as i32,
            seq: self.samp.1,
            rowres: self.samp.2,
            out: self.samp.5,
            next: self.b.next,
            force_fb: 0,
            last_only: !self.spec_prep_par.get() as i32,
            spart: if records { self.samp_rec } else { 0 },
            nquads: (vpad / 64) as i32,
            so_s2: CommRank::SAMP_SO_S2 as i32,
            so_fr: CommRank::SAMP_SO_FR as i32,
            leps: 1e-3f32.ln(),
            so_bv: CommRank::SAMP_SO_BV as i32,
            block: 1,
        };
        let off = CommRank::samp_off(self.tp, self.comm.cap);
        for (r, pb) in self.peers.iter().enumerate() {
            q.recv[r] = *pb + off;
        }
        self.prep_gather_pending.set(false);
        let sc = if q.block != 0 && q.last_only == 0 && ops::variant_cc() { 4 } else { 1 };
        let mut cfg = Launch::new((m * sc) as u32, 1024);
        if sc > 1 {
            cfg.cluster = Some((sc as u32, 1, 1));
        }
        return self.launch(if sc > 1 { "variant_zzzzzzzsamp/samp_stats" } else { "samp_stats" }, cfg, Args::new().ptr(self.meta).push(q));
    }

    fn layer(&mut self, l: usize, m: usize) -> Result<()> {
        self.cur_layer.set(l);
        let c = self.cfg.clone();
        let dim = c.dim;
        let hd = c.head_dim;
        let tp = self.tp;
        let heads = c.n_heads / tp;
        let ly: &Layer = unsafe { &*(&self.w.layers[l] as *const Layer) };
        let ratio = c.compress_ratios[l];
        let table = if ratio > 0 { self.rope_comp } else { self.rope_base };

        if ly.engram.is_some() {
            self.engram_block(l, m)?;
        }
        if let Some(t) = c.dspark_target_layer_ids.iter().position(|&x| x == l) {
            let mean_side = m <= MAX_M && deepseek_ultrafast_cuda::profile::sm103();
            let margs = || {
                let mut a = Args::new();
                a.ptr(self.meta).ptr(self.hbuf()).ptr(self.b.main_hidden + (t * dim * 2) as u64).i32(dim as i32).i32((3 * dim) as i32);
                a
            };
            if mean_side {
                self.ev_fork.record(&self.stream)?;
                self.side.wait(&self.ev_fork)?;
                let cfg = grid((dim / 256) as u32, m as u32, 256);
                cu::launch(self.k.get("hc_mean")?, &cfg, &self.side, &mut margs())?;
            } else {
                self.launch("hc_mean", grid((dim / 256) as u32, m as u32, 256), &mut margs())?;
            }
            if t + 1 == c.dspark_target_layer_ids.len() && self.spec_prep_par.get() && m <= MAX_M && !ops::prep_head() {
                self.par_fork()?;
                if mean_side {
                    self.ev_mean.record(&self.side)?;
                    self.par.wait(&self.ev_mean)?;
                }
                self.on_par(|me| me.prep_proj(m))?;
                self.prep_proj_pending.set(true);
            }
        }

        self.hc_fork_dots(ly.hc_attn, (l != 0 && ly.engram.is_none()) || self.prep_side_dots(), "attn")?;
        let x = self.b.x_raw;
        let xa = self.xin(ly.attn_norm);
        let qlr = c.q_lora_rank;
        let side_real = ly.compressor.is_some() || ly.indexer.is_some();
        let kv_wqb = m <= MAX_M;
        let par = side_real || !kv_wqb;
        let ix_late = m <= MAX_M && ly.indexer.is_some() && deepseek_ultrafast_cuda::profile::sm103() && self.wproj_main(ly, m);
        if ix_late {
            self.ev_ixf.record(&self.stream)?;
        }
        if par {
            self.par_fork()?;
            self.on_par(|me| me.attn_side(l, m))?;
        }
        let qkv2 = self.tg2_wqkv(ly, x, m)?;
        if qkv2.is_none() {
            self.gemv(&ly.wqkv, x, dim, m, self.b.q_lat, qlr, GemvOpts { split: Some((qlr, self.b.kv, hd)), ..xa })?;
        }
        if ix_late {
            self.ixs.wait(&self.ev_ixf)?;
            self.ix_on.set(true);
            let r = self.idx_weights(ly, m);
            self.ix_on.set(false);
            r?;
            self.ev_ixj.record(&self.ixs)?;
        }
        let cache_win = if self.encoder_readonly.get() { 0 } else { self.caches[l].win };
        let kvp = self.rope_params(self.b.kv, hd, hd, 1, table, false, Q_FP8, 0, cache_win, ly.kv_norm);
        if par && !kv_wqb {
            self.par_fork()?;
            self.on_par(|me| me.rope_n(me.b.kv, hd, hd, 1, m, table, false, Q_FP8, 0, cache_win, ly.kv_norm))?;
        }
        let qnorm = Some((ly.q_norm, c.norm_eps));
        let fuse_rope = (par || kv_wqb) && m <= MAX_M;
        let qrope = if fuse_rope { Some((table.0, table.1, self.meta, hd, hd - c.rope_head_dim)) } else { None };
        let kv_rope = if kv_wqb { Some(kvp) } else { None };
        let nq_i = c.index_n_heads * c.index_head_dim;
        let idx_cat = m <= MAX_M && ly.indexer.is_some() && (qnorm_split(heads * hd) == qnorm_split(nq_i) || deepseek_ultrafast_cuda::profile::sm103());
        let qb_w = if idx_cat { &ly.indexer.as_ref().unwrap().wq_b_cat } else { &ly.wq_b };
        let iq_rows = idx_cat.then(|| (nq_i, if qrope.is_some() { Some((c.index_head_dim, c.index_head_dim - c.rope_head_dim)) } else { None }));
        if self.tg2_wqb(qkv2, qb_w, m, qrope, kv_rope, iq_rows)? {
        } else if idx_cat {
            let ix = ly.indexer.as_ref().unwrap();
            let nq = c.index_n_heads * c.index_head_dim;
            let iq = if qrope.is_some() { Some((c.index_head_dim, c.index_head_dim - c.rope_head_dim)) } else { None };
            self.gemv(&ix.wq_b_cat, self.b.q_lat, qlr, m, self.b.q, heads * hd, GemvOpts { prenorm: qnorm, ksplit: Some(if deepseek_ultrafast_cuda::profile::sm103() { 1 } else { qnorm_split(heads * hd).unwrap_or(1) }), rope: qrope, kv_rope, split: Some((heads * hd, self.b.iq, nq)), iq, ..Default::default() })?;
        } else {
            self.gemv(&ly.wq_b, self.b.q_lat, qlr, m, self.b.q, heads * hd, GemvOpts { prenorm: qnorm, ksplit: qnorm_split(heads * hd), rope: qrope, kv_rope, ..Default::default() })?;
        }
        if par {
            if !fuse_rope {
                self.rope(self.b.q, hd, hd, heads, m, table, false, Q_NONE, 0, 0)?;
            }
        } else if kv_wqb {
        } else {
            let qp = self.rope_params(self.b.q, hd, hd, heads, table, false, Q_NONE, 0, 0, 0);
            self.rope_pair(m, qp, heads, kvp, 1)?;
            self.attn_side(l, m)?;
        }

        let idxq_early = m <= MAX_M && ratio > 0 && ly.indexer.is_some() && !idx_cat;
        if idxq_early {
            let ix = ly.indexer.as_ref().unwrap();
            let (ihd, inh) = (c.index_head_dim, c.index_n_heads);
            self.gemv(&ix.wq_b, self.b.q_lat, qlr, m, self.b.iq, inh * ihd, GemvOpts { prenorm: Some((ly.q_norm, c.norm_eps)), ksplit: qnorm_split(inh * ihd), ..Default::default() })?;
            self.rope(self.b.iq, ihd, ihd, inh, m, table, false, Q_FP4_E8M0, 0, 0)?;
        }
        if ix_late {
            self.stream.wait(&self.ev_ixj)?;
        } else if self.wproj_main(ly, m) {
            self.idx_weights(ly, m)?;
        }
        let mut ktop = 0usize;
        let q_ready = idxq_early || (idx_cat && fuse_rope);
        let idx_split = self.idx_split_ok(l, m, par, q_ready);
        if idx_split {
            self.par_fork()?;
            self.on_par(|me| me.idx_split_new(l))?;
            self.idx_split_old(l, m)?;
        }
        if ratio > 0 {
            let src = c.kv_source_of(l).unwrap();
            if par {
                self.par_join()?;
            }
            if let Some(ix) = &ly.indexer {
                let ihd = c.index_head_dim;
                let inh = c.index_n_heads;
                if !idx_cat && !idxq_early {
                    self.gemv(&ix.wq_b, self.b.q_lat, qlr, m, self.b.iq, inh * ihd, GemvOpts { prenorm: Some((ly.q_norm, c.norm_eps)), ksplit: qnorm_split(inh * ihd), ..Default::default() })?;
                }
                if m > MAX_M {
                    let p = RopeParams { q4: self.b.iq4.0, sf4: self.b.iq4.1, ..self.rope_params(self.b.iq, ihd, ihd, inh, table, false, Q_FP4_E8M0, 0, 0, 0) };
                    self.rope_launch(p, m * inh)?;
                } else if !(idx_cat && fuse_rope) && !idxq_early {
                    self.rope(self.b.iq, ihd, ihd, inh, m, table, false, Q_FP4_E8M0, 0, 0)?;
                }

                let tmax = self.max_seq / ratio;
                let shard = self.idx_rows(m);
                let uses_cand = c.candidate_source_layer < l;
                if uses_cand && self.cand_pending.replace(false) && !par {
                    self.par_join()?;
                }
                let is_cand_src = c.candidate_source_layer == l;
                let ip = IdxParams {
                    q: self.b.iq,
                    w: self.b.iw,
                    ldw: 64,
                    kc: self.caches[src].idx_k.unwrap(),
                    scores: self.b.scores,
                    ld: self.max_seq as i32,
                    ratio: ratio as i32,
                    tmax: tmax as i32,
                    cand: if uses_cand { self.b.cand } else { 0 },
                    ld_cand: (self.max_seq / 8 + 8) as i32,
                    cand_block: c.candidate_block_size as i32,
                    bf16_ref: 0,
                };
                if tmax > 0 {
                    if m > MAX_M {
                        let pos0 = self.host_pos_end - m;
                        let (r0, r1) = shard.map(|s| (s.0, s.1)).unwrap_or((0, m));
                        let rows = r1 - r0;
                        let reach = (pos0 + r1) / ratio;
                        let tk = self.tmap_bf16(ip.kc, ihd, tmax, ihd * 2, 128)?;
                        let tq = self.tmap_bf16(self.b.iq + (r0 * inh * ihd * 2) as u64, ihd, (self.pmax - r0) * inh, ihd * 2, 256)?;
                        self.tmap_bf16(self.b.iq, ihd, self.pmax * inh, ihd * 2, 256)?;
                        let xp = ops::IxParams {
                            tk,
                            tq,
                            w: self.b.iw + (r0 * 64 * 2) as u64,
                            ldw: 64,
                            _pad: 0,
                            scores: self.b.scores,
                            ld: self.max_seq as i32,
                            ratio: ratio as i32,
                            tmax: reach as i32,
                            pos: (pos0 + r0) as i32,
                            m: rows as i32,
                            _pad2: 0,
                            cand: if ip.cand != 0 { ip.cand + (r0 * ip.ld_cand as usize) as u64 } else { 0 },
                            ld_cand: ip.ld_cand,
                            cand_block: ip.cand_block,
                        };
                        {
                            let (k4, ks) = self.caches[src].idx_k4.unwrap();
                            ensure!(r0 % 4 == 0);
                            let x4 = ops::Ix4Params {
                                tk: self.tmap_fp4(k4, tmax)?,
                                ksf: ks,
                                tq: self.tmap_fp4(self.b.iq4.0 + (r0 * inh * 64) as u64, (self.pmax - r0) * inh)?,
                                qsf: self.b.iq4.1 + (r0 * inh / 128 * 512) as u64,
                                w: xp.w,
                                ldw: 64,
                                _pad: 0,
                                scores: xp.scores,
                                ld: xp.ld,
                                ratio: xp.ratio,
                                tmax: xp.tmax,
                                pos: xp.pos,
                                m: xp.m,
                                flags: 0,
                                cand: xp.cand,
                                ld_cand: xp.ld_cand,
                                cand_block: xp.cand_block,
                            };
                            self.k.get("idx_score_tc4b")?.set_max_dynamic_smem(ops::IX4B_SMEM)?;
                            self.launch("idx_score_tc4b", Launch::new(self.sms, ops::IX4B_THREADS).smem(ops::IX4B_SMEM), Args::new().push(x4))?;
                        }
                    } else if idx_split {
                    } else if m <= 8 && self.tca.is_some() {
                        let tk = self.tmap_bf16(ip.kc, ihd, tmax, ihd * 2, 128)?;
                        let tq = self.tmap_bf16(self.b.iq, ihd, self.pmax * inh, ihd * 2, 256)?;
                        let xp = ops::IxParams {
                            tk,
                            tq,
                            w: self.b.iw,
                            ldw: 64,
                            _pad: 0,
                            scores: self.b.scores,
                            ld: self.max_seq as i32,
                            ratio: ratio as i32,
                            tmax: tmax as i32,
                            pos: 0,
                            m: m as i32,
                            _pad2: 0,
                            cand: ip.cand,
                            ld_cand: ip.ld_cand,
                            cand_block: ip.cand_block,
                        };
                        let gx = (tmax.div_ceil(128) as u32).min(self.sms);
                        let (kn, ksm) = ("idx_score_tc_dec", ops::IXD_SMEM);
                        self.k.get(kn)?.set_max_dynamic_smem(ksm)?;
                        self.launch(kn, Launch::new(gx, 320).smem(ksm), Args::new().ptr(self.meta).push(xp))?;
                    } else {
                        let gx = ((tmax as u32).div_ceil(128)).min((2 * self.sms / m as u32).max(1));
                        self.launch("idx_score", grid(gx, m as u32, 128), Args::new().ptr(self.meta).push(ip))?;
                    }
                    let (r0, rows, smeta) = shard.map(|s| (s.0, s.1 - s.0, s.2)).unwrap_or((0, m, self.meta));
                    let cand_par = is_cand_src && m <= MAX_M && shard.is_none();
                    if cand_par {
                        self.par_fork()?;
                        self.par_on.set(true);
                        self.cand_pending.set(true);
                    }
                    if is_cand_src {
                        let bs = c.candidate_block_size;
                        let nb = tmax.div_ceil(bs);
                        let ldb = self.max_seq / 8 + 8;
                        self.launch(
                            "block_max",
                            Launch::new(1, 256).grid3((((nb as u32).div_ceil(256)).min(64), rows as u32, 1)),
                            Args::new().ptr(smeta).ptr(self.b.scores).i32(self.max_seq as i32).i32(ratio as i32).i32(bs as i32).ptr(self.b.bscores).i32(ldb as i32),
                        )?;
                        let tpb = TopkParams {
                            x: self.b.bscores,
                            ld: ldb as i32,
                            ratio: ratio as i32,
                            block: bs as i32,
                            k: c.candidate_topk_blocks as i32,
                            out: self.b.cand_ids + (r0 * c.candidate_topk_blocks * 4) as u64,
                            mask: self.b.cand + (r0 * ldb) as u64,
                            ld_mask: ldb as i32,
                            drop_neg_inf: 1,
                        };
                        let r = match ops::variant_cc().then_some(2u32) {
                            Some(cl) if m <= MAX_M && shard.is_none() && ops::topk_cl2() && topk_cluster_smem(nb, ratio, bs, cl as usize) <= 160 << 10 => {
                                let kn = "variant_zzzzzzzzixc/topk_clc2_2_1024";
                                let sm = topk_cluster_smem(nb, ratio, bs, cl as usize);
                                self.k.get(kn)?.set_max_dynamic_smem(sm)?;
                                let mut lc = Launch::new(cl, 1024).grid3((cl, rows as u32, 1)).smem(sm);
                                lc.cluster = Some((cl, 1, 1));
                                self.launch(kn, lc, Args::new().ptr(smeta).push(tpb).ptr(0).i32(0))
                            }
                            _ => self.topk_m(tpb, rows, nb, smeta),
                        };
                        self.par_on.set(false);
                        r?;
                    }
                    let tp_ = TopkParams {
                        x: self.b.scores,
                        ld: self.max_seq as i32,
                        ratio: ratio as i32,
                        block: 1,
                        k: c.index_topk as i32,
                        out: self.b.topk + (r0 * c.index_topk * 4) as u64,
                        mask: 0,
                        ld_mask: 0,
                        drop_neg_inf: 0,
                    };
                    let cand = (uses_cand && shard.is_none() && c.candidate_block_size == 8).then_some((self.b.cand_ids, c.candidate_topk_blocks));
                    if idx_split {
                        self.idx_split_merge(l, m)?;
                    } else {
                        self.topk_mc(tp_, rows, tmax, smeta, cand)?;
                    }
                    if let Some((r0, r1, _)) = shard {
                        let (g0, g1) = self.tk_gather;
                        let kk = c.index_topk;
                        let n = m * kk;
                        self.launch("rows_i2f", Launch::new((n as u32).div_ceil(256), 256), Args::new().ptr(self.b.topk).ptr(g0).i32(m as i32).i32(kk as i32).i32(r0 as i32).i32(r1 as i32))?;
                        self.comm_ex(g0, kk, kk, CM_SUM, false, g1, kk, true, None, 1, 0, None)?;
                        self.launch("rows_f2i", Launch::new((n as u32).div_ceil(256), 256), Args::new().ptr(g1).ptr(self.b.topk).push(n as i64))?;
                    }
                } else {
                    cu::memset_d32_async(self.b.topk, u32::MAX, m * c.index_topk, &self.stream)?;
                }
            }
            ktop = c.index_topk;
            if ly.indexer.is_some() {
            }
            let _ = src;
        }
        if ratio == 0 && par {
            self.par_join()?;
        }
        let comp = if ratio > 0 { self.caches[c.kv_source_of(l).unwrap()].comp.unwrap() } else { 0 };
        {
            let (sp, r) = if ktop > 0 { (16u32, 48usize) } else { (8u32, 16usize) };
            let p = AttnFParams {
                q: self.b.q,
                win: self.caches[l].win,
                comp,
                topk: if ktop > 0 { self.b.topk } else { 0 },
                ktop: ktop as i32,
                scale: (hd as f32).powf(-0.5),
                sink: ly.attn_sink,
                out: self.b.o,
                cos: table.0,
                sin: table.1,
                r: r as i32,
                fixed_window: 0,
                nh: (c.n_heads / self.tp) as i32,
                pre: 1 | if ratio > 0 && ly.indexer.is_none() { 2 } else { 0 },
                win_floor: self.win_floor.get(),
            };
            let hgroups = heads.div_ceil(16) as u32;
            let fused = if hgroups > 1 { "attn_fused_hg" } else { "attn_fused" };
            let af = if hgroups > 1 { fused } else { ops::attn_fused_kernel() };
            let smem = ops::attn_smem_for(af, r, sp as usize, heads.min(16));
            if m > MAX_M {
                let (pc, nb, name) = (16usize, 2usize, "attn_prefill_ks16");
                let smem = (nb * pc * 520 * 2 + 4 * 16 * pc * 4 + 16 * (pc + 8) * 2) as u32;
                self.k.get(name)?.set_max_dynamic_smem(smem)?;
                self.launch(name, Launch::new(m as u32, 128).grid3((m as u32, 1, hgroups)).smem(smem), Args::new().ptr(self.meta).push(p))?;
            } else if ktop > 0 && ops::attn_sds_on() && heads == 8 && 128 + ktop <= 640 {
                let name = "variant_zzzzzzzzzzsds/attn_sds_8";
                let smem = ops::attn_sds_smem(heads);
                self.k.get(name)?.set_max_dynamic_smem(smem)?;
                let mut cfg = Launch::new(8, 512).grid3((8, m as u32, 1)).smem(smem);
                cfg.cluster = Some((8, 1, 1));
                self.launch(name, cfg, Args::new().ptr(self.meta).push(p))?;
            } else {
                self.k.get(af)?.set_max_dynamic_smem(smem)?;
                let mut cfg = Launch::new(sp, 128).grid3((sp, m as u32, hgroups)).smem(smem);
                cfg.cluster = Some((sp, 1, 1));
                self.launch(af, cfg, Args::new().ptr(self.meta).push(p))?;
            }
        }
        self.hc_late_fork()?;
        let gl = c.o_groups / tp * c.o_lora_rank;
        let ob16 = m > MAX_M && self.tp > 1;
        let src16 = if ob16 { CM_SRC16 } else { 0 };
        let ob_part = m <= MAX_M;
        let o2 = if ob_part { self.tg2_wo(ly, m)? } else { None };
        if o2.is_none() {
            self.gemv(&ly.wo_a, self.b.o, heads * hd, m, self.b.oa, gl, GemvOpts { x_group_rows: c.o_lora_rank, ..Default::default() })?;
        }
        let (ob, ob_ns) = if let Some(ns) = o2 {
            (self.b.ob_part, ns)
        } else if ob_part {
            let o = GemvOpts { out_f32: true, epilogue: Epilogue::Partial, xshift: 4, ..Default::default() };
            let ns = self.tc_cluster_for(&ly.wo_b, m, &o).unwrap_or_else(|| ops::gemv_cluster(&ly.wo_b, m, &o));
            self.gemv(&ly.wo_b, self.b.oa, gl, m, self.b.ob_part, dim, o)?;
            (self.b.ob_part, ns)
        } else {
            self.gemv(&ly.wo_b, self.b.oa, gl, m, self.b.ob, dim, GemvOpts { out_f32: !ob16, xshift: 4, ..Default::default() })?;
            (self.b.ob, 1)
        };
        let (hin, hout) = (self.b.h[self.hcur], self.b.h[1 - self.hcur]);
        self.hc_join()?;
        let _ = (hin, hout);
        let (hin, hout) = (self.b.h[self.hcur], self.b.h[1 - self.hcur]);
        self.comm_ex(ob, dim, dim, CM_SUM_HCPOST | src16, true, hout, dim, false, Some((hin, self.b.post_a, self.b.comb_a)), ob_ns, 0, Some(self.epi_nodots()))?;
        self.hcur = 1 - self.hcur;

        let ly: &Layer = &self.w.layers[l];
        let side_in = self.ffn_dots_in(ly, m) && !self.image_rows(m) && !(m > MAX_M && self.pm.is_some());
        if !side_in {
            self.hc_fork_dots(ly.hc_ffn, true, "ffn")?;
        }
        let lyp = ly as *const Layer;
        let topk = c.n_activated_experts;
        let h0 = self.hcur;
        let rm_mid = ly.experts.rm && m <= MAX_M && !self.ffn_gate_in(ly, m);
        ensure!(!rm_mid || self.image_prefill || m <= 8, "tmoe: decode steps of more than 8 rows (draft block > 7) are not supported");
        ensure!(!rm_mid || self.pm.is_some(), "tmoe: a {m}-row MoE without the prefill MoE buffers");
        if unsafe { &*lyp }.experts.sgl {
            let b0 = self.b;
            let (meta0, cur0) = (self.meta, self.cur_m);
            let chain = self.sgl_chain_ok() && !self.image_rows(m);
            let r = if m <= 8 && chain {
                self.hcur = h0;
                self.ffn_sgl(unsafe { &*lyp }, m, topk)
            } else if self.pm.is_some() {
                self.ffn_prefill(lyp, m, topk)
            } else if chain {
                self.sgl_slices(lyp, m, topk, h0)
            } else {
                Err(anyhow::anyhow!("sgl moe: a {m}-row FFN with neither the decode chain nor the prefill MoE buffers"))
            };
            self.b = b0;
            self.meta = meta0;
            self.cur_m = cur0;
            r?;
            self.hcur = 1 - h0;
        } else if self.image_rows(m) || ((m > MAX_M || rm_mid) && self.pm.is_some()) {
            self.ffn_prefill(lyp, m, topk)?;
        } else {
            self.sliced(m, |me, mm| {
                me.hcur = h0;
                me.ffn(lyp, mm, topk)
            })?;
            self.hcur = 1 - h0;
        }
        Ok(())
    }

    fn sgl_chain_ok(&self) -> bool {
        self.sgl.is_some()
            && (self.tp == 4 || self.tp == 8)
            && self.nvls.is_some()
    }

    fn sgl_slices(&mut self, ly: *const Layer, m: usize, topk: usize, h0: usize) -> Result<()> {
        let ly: &Layer = unsafe { &*ly };
        let slot0 = self.sgl.as_ref().map(|s| s.metas).ok_or_else(|| anyhow::anyhow!("sgl moe: no state"))?;
        let b0 = self.b;
        let pos0 = self.host_pos_end.saturating_sub(m);
        for k in 0..m.div_ceil(8) {
            let rows = 8.min(m - k * 8);
            let slot = slot0 + ((k % 2) * std::mem::size_of::<Meta>()) as u64;
            let mt = [Meta { pos: (pos0 + k * 8) as i32, m: rows as i32, tokens: [0; 16] }];
            let mb = unsafe { std::slice::from_raw_parts(mt.as_ptr() as *const u8, std::mem::size_of::<Meta>()) };
            cu::memcpy_htod_async(slot, mb, &self.stream)?;
            self.meta = slot;
            self.cur_m = rows;
            self.b = self.row_view(k * 8);
            self.hcur = h0;
            self.ffn_sgl(ly, rows, topk)?;
            self.b = b0;
        }
        Ok(())
    }

    fn ffn_prefill(&mut self, ly: *const Layer, m: usize, topk: usize) -> Result<()> {
        let ly: &Layer = unsafe { &*ly };
        let c = self.cfg.clone();
        let dim = c.dim;
        let n_exp = ly.experts.n;
        let inter = ly.experts.inter;
        let h0 = self.hcur;
        let (k13, k2) = (Some(4), Some(1));
        self.par_fork()?;
        let tc_shared = match (self.w.tc.get(&ly.shared_w13.w), self.w.tc.get(&ly.shared_w2.w)) {
            (Some(t13), Some(t2)) if self.tca.is_some() && !t13.bf16 && !t2.bf16 && t13.kp == dim => Some((*t13, *t2)),
            _ => None,
        };
        let y16 = m > MAX_M && self.tp > 1 && tc_shared.is_some();
        let src16 = if y16 { CM_SRC16 } else { 0 };
        if let Some((t13, t2)) = tc_shared {
            let xf = self.xin(ly.ffn_norm);
            let kp = t2.kp;
            self.on_par(|me| {
                me.gemm_tc(t13, &ly.shared_w13, me.b.x_raw, dim, m, me.sact_tc, kp, GemvOpts { epilogue: Epilogue::SwiGlu(c.swiglu_limit), ..xf })?;
                me.gemm_tc(t2, &ly.shared_w2, me.sact_tc, kp, m, me.b.y, dim, GemvOpts { out_f32: !y16, ..Default::default() })
            })?;
        } else {
        self.sliced(m, |me, mm| {
            let xf = me.xin(ly.ffn_norm);
            let sp = Moe { plan: me.shared_plan(mm), w_stride: 0, max_slots: 1, split_hint: 1, sched: 0, w_sh: 0, ..Default::default() };
            me.gemv_par(&ly.shared_w13, me.b.x_raw, dim, mm, me.b.sact, inter, GemvOpts { epilogue: Epilogue::SwiGlu(c.swiglu_limit), moe: Some(sp), ksplit: k13, ..xf })?;
            me.gemv_par(&ly.shared_w2, me.b.sact, inter, mm, me.b.y, dim, GemvOpts { out_f32: true, moe: Some(sp), ksplit: k2, ..Default::default() })
        })?;
        }
        let xf = self.xin(ly.ffn_norm);
        self.gemv(&ly.gate, self.b.x_raw, dim, m, self.b.glog, n_exp, GemvOpts { out_f32: true, epilogue: Epilogue::SqrtSoftplus, ..xf })?;
        let mut route = Args::new();
        route.i32(m as i32).ptr(self.b.glog).ptr(ly.gate_bias).i32(n_exp as i32).i32(topk as i32)
            .f32(c.route_scale).ptr(self.b.route_ids).ptr(self.b.route_w);
        let kernel = if self.image_rows(m) {
            let v = self.vision.as_ref().unwrap();
            route.ptr(v.biases[ly.id].ptr).ptr(v.spans_ptr()).i32(v.span_count as i32).i32((self.host_pos_end - m) as i32);
            "vision_route"
        } else { "moe_route_tokens" };
        self.launch(kernel, Launch::new((m as u32).div_ceil(8), 256), &mut route)?;
        let pm = self.pm.unwrap();
        let b = self.b;
        let mp = ops::MpParams { ids: b.route_ids, w: b.route_w, m: m as i32, topk: topk as i32, n_exp: n_exp as i32, _pad: 0, row_pair: pm.row_pair, row_w: pm.row_w, pair_row: pm.pair_row, row_tile: pm.row_tile, tiles: pm.tiles, ntiles: pm.ntiles };
        self.launch("moe_perm", Launch::new(1, 1024), Args::new().push(mp))?;
        let mq = ops::MqParams {
            x: b.x_raw,
            ldx: dim as i32,
            m: m as i32,
            k: dim as i32,
            topk: topk as i32,
            pw: ly.ffn_norm,
            norm_part: b.part,
            nslices: (dim / 256) as i32,
            eps: c.norm_eps,
            pair_row: pm.pair_row,
            row_tile: pm.row_tile,
            xq: pm.xq,
            sf: pm.sf,
        };
        let rm = ly.experts.rm || ly.experts.sgl;
        self.launch(if rm { "moe_gather_quant_nat" } else { "moe_gather_quant" }, Launch::new(((m * dim / 32) as u32).div_ceil(256), 256), Args::new().push(mq))?;
        let p13 = ops::TmParams {
            w: ly.experts.w13,
            w_stride: ly.experts.w13_stride as u64,
            k: dim as i32,
            rt: (2 * inter).div_ceil(128) as i32,
            tb: pm.tb13,
            sfb: pm.sf,
            tiles: pm.tiles,
            ntiles: pm.ntiles,
            row_pair: pm.row_pair,
            row_w: pm.row_w,
            aq: pm.aq,
            asf: pm.asf,
            ldaq: pm.ldaq as i32,
            akt: pm.akt as i32,
            limit: c.swiglu_limit,
            nq: (2 * inter / 64) as i32,
            out: 0,
            ldo: 0,
            flags: if ly.experts.sgl { 32 } else { 0 },
            sf_off: if ly.experts.sgl { (2 * inter.div_ceil(128) * 128 * dim / 2) as u64 } else { 0 },
        };
        let p2 = ops::TmParams { w: ly.experts.w2, w_stride: ly.experts.w2_stride as u64, k: ly.experts.kpad as i32, rt: (dim / 128) as i32, nq: (dim / 64) as i32, tb: pm.tb2, sfb: pm.asf, out: b.ypair, ldo: dim as i32, flags: (if y16 { 16 } else { 0 }) | (if ly.experts.sgl { 32 } else { 0 }), sf_off: if ly.experts.sgl { (dim * sgl_w2_row_bytes(inter)) as u64 } else { 0 }, ..p13 };
        let _ = pm.max_tiles;
        if rm {
            let (m13, m2) = if ly.experts.sgl {
                let st = self.sgl.as_ref().ok_or_else(|| anyhow::anyhow!("sgl moe: no state"))?;
                let mp = st.maps[&ly.experts.w13];
                (mp[0], mp[1])
            } else {
                let t = self.tcw13.as_ref().ok_or_else(|| anyhow::anyhow!("tmoe: no tcgen05 expert state"))?;
                (t.maps[&ly.experts.w13], t.w2maps[&ly.experts.w2])
            };
            let p13 = ops::TmRmParams { p: p13, wmap: m13 };
            let p2 = ops::TmRmParams { p: p2, wmap: m2 };
            for (name, prm) in [("variant_zzzzzzzzzzmoe/tc_moe_w13_rm2", p13), ("variant_zzzzzzzzzzmoe/tc_moe_w2_rm", p2)] {
                self.k.get(name)?.set_max_dynamic_smem(ops::TM_SMEM)?;
                self.launch(name, Launch::new(pm.sms, 512).smem(ops::TM_SMEM), Args::new().push(prm))?;
            }
        } else {
            self.launch("tc_moe_w13", Launch::new(pm.sms, 512).smem(ops::TM_SMEM), Args::new().push(p13))?;
            self.launch("tc_moe_w2", Launch::new(pm.sms, 512).smem(ops::TM_SMEM), Args::new().push(p2))?;
        }
        self.par_join()?;
        self.hc_join()?;
        let (hin, hout) = (self.b.h[h0], self.b.h[1 - h0]);
        self.comm_ex(self.b.ypair, dim, dim, CM_SUM_HCPOST | src16, false, hout, dim, false, Some((hin, self.b.post_a, self.b.comb_a)), topk, self.b.y, Some(self.epi_nodots()))?;
        self.hcur = 1 - h0;
        Ok(())
    }

    fn engram_pre(&self, l: usize, m: usize, bufs: (DevPtr, DevPtr, DevPtr, DevPtr)) -> Result<()> {
        let c = &self.cfg;
        let dim = c.dim;
        let ly: &Layer = &self.w.layers[l];
        let eg = ly.engram.as_ref().unwrap();
        let ei = c.engram_index(l).unwrap();
        let ng = c.engram_max_ngram_size;
        let nh = c.engram_n_heads;
        let hp = EngramHash {
            mult: self.engram.mult + (ei * ng * 8) as u64,
            primes: self.engram.primes + (ei * (ng - 1) * nh * 8) as u64,
            offsets: self.engram.offsets + (ei * (ng - 1) * nh * 8) as u64,
            max_ngram: ng as i32,
            heads: nh as i32,
            pad_id: self.engram.pad_id,
        };
        let cols = (ng - 1) * nh;
        let ehd = c.engram_head_dim;
        let (b_eg, b_ex, b_part, b_ekv) = bufs;
        self.launch(
            "engram_gather",
            grid(cols as u32, m as u32, 64),
            Args::new()
                .ptr(self.meta)
                .ptr(self.cmp_hist)
                .push(hp)
                .ptr(eg.table)
                .ptr(eg.scales)
                .push((self.rank * eg.shard_rows) as i64)
                .push(eg.shard_rows as i64)
                .ptr(b_eg)
                .i32(ehd as i32),
        )?;
        let ecols = cols * ehd;
        self.comm(b_eg, ecols, ecols, CM_SUM, false, b_ex, ecols, false, None)?;
        let nloc = eg.wkv.n;
        ensure!(nloc % 8 == 0);
        self.gemv(&eg.wkv, b_ex, ecols, m, b_part, nloc, GemvOpts { xshift: 4, ..Default::default() })?;
        self.comm(b_part, nloc / 2, nloc / 2, CM_GATHER, false, b_ekv, 5 * dim / 2, true, None)
    }

    fn eng_load_peers(&self) -> Result<()> {
        if !self.eng_peers.borrow().is_empty() || self.peers.len() != self.tp || self.stream.is_capturing()? {
            return Ok(());
        }
        let c = &self.cfg;
        let mut v = Vec::new();
        for (ei, &l) in c.engram_layer_ids.iter().enumerate() {
            let Some(eg) = self.w.layers.get(l).and_then(|ly| ly.engram.as_ref()) else { return Ok(()) };
            ensure!(c.dim <= 5120 && c.dim % 4 == 0 && eg.wkv.n % 4 == 0, "engram_apply_peer: dim <= 5120, float4 parts");
            let mut q = EgPeers { epoch: self.eng_epoch + (ei * 4) as u64, shard: eg.shard_rows as i64, tp: self.tp as i32, nloc: eg.wkv.n as i32, ..Default::default() };
            for (r, &pb) in self.peers.iter().enumerate() {
                let d: Vec<u64> = cu::download(pb + self.comm.dir_off + (ei * 32) as u64, 4)?;
                ensure!(d.iter().all(|&x| x != 0), "engram directory of rank {r} not set");
                (q.table[r], q.scales[r], q.part[r], q.flag[r]) = (d[0], d[1], d[2], d[3]);
            }
            v.push(q);
        }
        *self.eng_peers.borrow_mut() = v;
        Ok(())
    }

    fn engram_fork(&mut self, m: usize, lo: usize, hi: usize) -> Result<()> {
        self.eng_pending.set(0);
        self.eng_joined.set(0);
        self.eng_load_peers()?;
        if m > MAX_M || self.eng_peers.borrow().is_empty() {
            return Ok(());
        }
        let ids: Vec<usize> = self.cfg.engram_layer_ids.iter().copied().filter(|&l| l >= lo && l < hi).collect();
        if ids.is_empty() {
            return Ok(());
        }
        self.ev_efork.record(&self.stream)?;
        self.eng.wait(&self.ev_efork)?;
        self.eng_on.set(true);
        let mut r = Ok(());
        for &l in &ids {
            let ei = self.cfg.engram_index(l).unwrap();
            r = self.engram_pre_peer(l, m, ei).and_then(|_| self.ev_ejoin[ei].record(&self.eng));
            if r.is_err() {
                break;
            }
            self.eng_pending.set(self.eng_pending.get() | 1 << ei);
        }
        self.eng_on.set(false);
        r
    }

    fn engram_hash(&self, l: usize) -> EngramHash {
        let c = &self.cfg;
        let ei = c.engram_index(l).unwrap();
        let (ng, nh) = (c.engram_max_ngram_size, c.engram_n_heads);
        EngramHash {
            mult: self.engram.mult + (ei * ng * 8) as u64,
            primes: self.engram.primes + (ei * (ng - 1) * nh * 8) as u64,
            offsets: self.engram.offsets + (ei * (ng - 1) * nh * 8) as u64,
            max_ngram: ng as i32,
            heads: nh as i32,
            pad_id: self.engram.pad_id,
        }
    }

    fn engram_pre_peer(&self, l: usize, m: usize, ei: usize) -> Result<()> {
        let c = &self.cfg;
        let eg = self.w.layers[l].engram.as_ref().unwrap();
        let q = self.eng_peers.borrow()[ei];
        let cols = (c.engram_max_ngram_size - 1) * c.engram_n_heads;
        let ehd = c.engram_head_dim;
        let (_, b_ex, b_part, _) = self.eng_bufs[ei];
        self.launch("engram_gather_peer", grid(cols as u32, m as u32, 64), Args::new().ptr(self.meta).ptr(self.cmp_hist).push(self.engram_hash(l)).push(q).ptr(b_ex).i32(ehd as i32))?;
        self.gemv(&eg.wkv, b_ex, cols * ehd, m, b_part, eg.wkv.n, GemvOpts { out_f32: true, xshift: 4, ..Default::default() })?;
        self.launch("engram_signal", Launch::new(1, 32), Args::new().ptr(self.eng_epoch + (ei * 4) as u64).ptr(self.eng_flags + (ei * 4) as u64))
    }

    fn engram_block(&mut self, l: usize, m: usize) -> Result<()> {
        let c = self.cfg.clone();
        let dim = c.dim;
        let ly: &Layer = unsafe { &*(&self.w.layers[l] as *const Layer) };
        if let Some(eg) = &ly.engram {
            let ei = c.engram_index(l).unwrap();
            let joined_early = self.eng_joined.get() & (1 << ei) != 0;
            if self.eng_pending.get() & (1 << ei) != 0 || joined_early {
                self.eng_joined.set(self.eng_joined.get() & !(1 << ei));
                if !joined_early {
                    self.eng_pending.set(self.eng_pending.get() & !(1 << ei));
                    self.stream.wait(&self.ev_ejoin[ei])?;
                }
                let q = self.eng_peers.borrow()[ei];
                ensure!(dim <= 20 * 256, "engram_apply_peer: dim <= 5120");
                self.launch("engram_apply_peer", grid(4, m as u32, 256), Args::new().ptr(self.meta).ptr(self.cmp_hist).ptr(self.hbuf()).push(q).ptr(eg.qk).i32(dim as i32).f32(c.norm_eps))?;
            } else {
                self.engram_pre(l, m, (self.b.eg, self.b.ex, self.b.ekv_part, self.b.ekv))?;
                self.launch("engram_apply", grid(4, m as u32, 256), Args::new().ptr(self.meta).ptr(self.cmp_hist).ptr(self.hbuf()).ptr(self.b.ekv).ptr(eg.qk).i32(dim as i32).f32(c.norm_eps))?;
            }
            self.hc_prep(m, self.epi(ly.hc_attn.0, false))?;
        }
        Ok(())
    }

    fn attn_side(&self, l: usize, m: usize) -> Result<()> {
        let c = self.cfg.clone();
        let dim = c.dim;
        let hd = c.head_dim;
        let ly: &Layer = &self.w.layers[l];
        let ratio = c.compress_ratios[l];
        let table = if ratio > 0 { self.rope_comp } else { self.rope_base };
        let x = self.b.x_raw;
        let xa = self.xin(ly.attn_norm);
        let cache = &self.caches[l];
        if ratio > 0 && !self.swa_replay.get() {
            if let Some(cp) = &ly.compressor {
                ensure!(ratio <= 4 && hd <= 512, "compress_pool: ratio <= 4, head dim <= 512 (compress ratio {ratio}, head dim {hd})");
                if ratio > 1 && m <= MAX_M && cp.wkvg.is_some() {
                    let (kvh, sch) = (cache.kv_hist.unwrap(), cache.sc_hist.unwrap());
                    self.gemv(cp.wkvg.as_ref().unwrap(), x, dim, m, kvh, hd, GemvOpts { out_f32: true, split: Some((hd, sch, hd)), out_pos: Some(self.meta), ..xa })?;
                    let (pool, nt) = ("compress_pool", 128);
                    self.launch(
                        pool,
                        Launch::new(m as u32, nt),
                        Args::new().ptr(self.meta).ptr(kvh).ptr(sch).i32(ratio as i32).ptr(cp.norm).f32(c.norm_eps).ptr(self.b.latent).i32(hd as i32),
                    )?;
                } else if ratio > 1 {
                    self.gemv(&cp.wkv, x, dim, m, self.b.ckv, hd, GemvOpts { out_f32: true, ..xa })?;
                    self.gemv(cp.wgate.as_ref().unwrap(), x, dim, m, self.b.csc, hd, GemvOpts { out_f32: true, ..xa })?;
                    for (srcb, hist) in [(self.b.ckv, cache.kv_hist.unwrap()), (self.b.csc, cache.sc_hist.unwrap())] {
                        self.launch(
                            "store_hist_f32",
                            grid((hd / 128) as u32, m as u32, 128),
                            Args::new().ptr(self.meta).ptr(srcb).i32(hd as i32).ptr(hist).i32(hd as i32),
                        )?;
                    }
                    self.launch(
                        "compress_pool",
                        Launch::new(m as u32, 128),
                        Args::new().ptr(self.meta).ptr(cache.kv_hist.unwrap()).ptr(cache.sc_hist.unwrap()).i32(ratio as i32).ptr(cp.norm).f32(c.norm_eps).ptr(self.b.latent).i32(hd as i32),
                    )?;
                } else {
                    self.gemv(&cp.wkv, x, dim, m, self.b.latent, hd, xa)?;
                }
                let lnorm = if ratio > 1 { 0 } else { cp.norm };
                let ix = ly.indexer.as_ref().unwrap();
                let wk = ix.wk.as_ref().unwrap();
                let ihd = c.index_head_dim;
                let pre = if ratio > 1 { None } else { Some((cp.norm, c.norm_eps)) };
                self.gemv(wk, self.b.latent, hd, m, self.b.ik, ihd, GemvOpts { prenorm: pre, ..Default::default() })?;
                let (k4, ks) = cache.idx_k4.unwrap();
                let pk = RopeParams { q4: k4, sf4: ks, ..self.rope_params(self.b.ik, ihd, ihd, 1, table, false, Q_FP4_E8M0, ratio, cache.idx_k.unwrap(), ix.k_norm) };
                let pc = self.rope_params(self.b.latent, hd, hd, 1, table, false, Q_FP4_E4M3_16, ratio, cache.comp.unwrap(), lnorm);
                if m <= MAX_M {
                    self.rope_pair(m, pk, 1, pc, 1)?;
                } else {
                    self.rope_launch(pk, m)?;
                    self.rope_launch(pc, m)?;
                }
            }
        }
        if ly.indexer.is_some() && !self.wproj_main(ly, m) {
            self.idx_weights(ly, m)?;
        }
        Ok(())
    }

    fn wproj_main(&self, ly: &Layer, m: usize) -> bool {
        m <= MAX_M && ly.compressor.is_some() && ly.indexer.is_some()
    }

    fn idx_split_ok(&self, l: usize, m: usize, par: bool, q_ready: bool) -> bool {
        let c = &self.cfg;
        let ly = &self.w.layers[l];
        let ratio = c.compress_ratios[l];
        if !ops::variant_cc() || m > 8 || !par || !q_ready || ratio == 0 || ly.indexer.is_none() || ly.compressor.is_none() || c.kv_source_of(l) != Some(l) || !self.wproj_main(ly, m) {
            return false;
        }
        if c.candidate_source_layer == l {
            return false;
        }
        let tmax = self.max_seq / ratio;
        let k = c.index_topk;
        self.tca.is_some()
            && self.idx_rows(m).is_none()
            && c.candidate_source_layer >= l
            && !self.swa_replay.get()
            && tmax > 0
            && k + 16 <= 1024
            && self.b.idx_okeys != 0
            && self.max_seq % 4 == 0
            && topk_cluster_smem(tmax, ratio, 1, 8) <= 160 << 10
    }

    fn idx_split_xp(&self, l: usize, pass: i32) -> Result<ops::IxParams> {
        let c = &self.cfg;
        let ratio = c.compress_ratios[l];
        let (ihd, inh) = (c.index_head_dim, c.index_n_heads);
        let tmax = self.max_seq / ratio;
        let kc = self.caches[l].idx_k.unwrap();
        Ok(ops::IxParams {
            tk: self.tmap_bf16(kc, ihd, tmax, ihd * 2, 128)?,
            tq: self.tmap_bf16(self.b.iq, ihd, self.pmax * inh, ihd * 2, 256)?,
            w: self.b.iw,
            ldw: 64,
            _pad: 0,
            scores: self.b.scores,
            ld: self.max_seq as i32,
            ratio: ratio as i32,
            tmax: tmax as i32,
            pos: 0,
            m: 0,
            _pad2: pass,
            cand: 0,
            ld_cand: (self.max_seq / 8 + 8) as i32,
            cand_block: c.candidate_block_size as i32,
        })
    }

    fn idx_split_old(&self, l: usize, m: usize) -> Result<()> {
        let c = &self.cfg;
        let ratio = c.compress_ratios[l];
        let tmax = self.max_seq / ratio;
        let xp = self.idx_split_xp(l, 1)?;
        const SC: &str = "variant_zzzzzzzzixs/idx_score_tc_dec";
        self.k.get(SC)?.set_max_dynamic_smem(ops::IXD_SMEM)?;
        let gx = (tmax.div_ceil(128) as u32).min(self.sms.saturating_sub(32).max(1));
        self.launch(SC, Launch::new(gx, 320).smem(ops::IXD_SMEM), Args::new().ptr(self.meta).push(xp))?;
        let tp = TopkParams { x: self.b.scores, ld: self.max_seq as i32, ratio: ratio as i32, block: 1, k: c.index_topk as i32, out: self.b.topk, mask: 0, ld_mask: 0, drop_neg_inf: 0 };
        const TK: &str = "variant_zzzzzzzzixt/topk_old_8_1024";
        let sm = topk_cluster_smem(tmax, ratio, 1, 8);
        self.k.get(TK)?.set_max_dynamic_smem(sm)?;
        let mut lc = Launch::new(8, 1024).grid3((8, m as u32, 1)).smem(sm);
        lc.cluster = Some((8, 1, 1));
        self.launch(TK, lc, Args::new().ptr(self.meta).push(tp).ptr(self.b.idx_okeys))
    }

    fn idx_split_new(&self, l: usize) -> Result<()> {
        let xp = self.idx_split_xp(l, 2)?;
        const SC: &str = "variant_zzzzzzzzixs/idx_score_tc_dec";
        self.k.get(SC)?.set_max_dynamic_smem(ops::IXD_SMEM)?;
        self.launch(SC, Launch::new(2, 320).smem(ops::IXD_SMEM), Args::new().ptr(self.meta).push(xp))
    }

    fn idx_split_merge(&self, l: usize, m: usize) -> Result<()> {
        let c = &self.cfg;
        let ratio = c.compress_ratios[l];
        const MG: &str = "variant_zzzzzzzzixs/idx_merge";
        self.launch(
            MG,
            Launch::new(m as u32, 1024),
            Args::new().ptr(self.meta).ptr(self.b.scores).i32(self.max_seq as i32).i32(ratio as i32).ptr(self.b.topk).ptr(self.b.idx_okeys).i32(c.index_topk as i32).ptr(self.b.topk),
        )
    }

    fn idx_weights(&self, ly: &Layer, m: usize) -> Result<()> {
        let c = &self.cfg;
        let ix = ly.indexer.as_ref().unwrap();
        let (ihd, inh) = (c.index_head_dim, c.index_n_heads);
        let wscale = (ihd as f32).powf(-0.5) * (inh as f32).powf(-0.5);
        let xa = self.xin(ly.attn_norm);
        let pow2 = wscale.to_bits() & 0x007f_ffff == 0;
        if pow2 {
            self.gemv(&ix.weights_proj, self.b.x_raw, c.dim, m, self.b.iw, 64, GemvOpts { alpha: wscale, ..xa })
        } else {
            self.gemv(&ix.weights_proj, self.b.x_raw, c.dim, m, self.b.iw, 64, xa)?;
            self.launch("scale_bf16", Launch::new(((m * 64) as u32).div_ceil(256), 256), Args::new().ptr(self.meta).ptr(self.b.iw).i32(64).f32(wscale))
        }
    }

    fn ffn_gate_in(&self, ly: &Layer, m: usize) -> bool {
        let c = &self.cfg;
        let fused_ok = m <= 8 && ops::moe_fused_shape(c.dim, ly.experts.kpad).is_some();
        let min_m = 2;
        fused_ok && m >= min_m && ops::moe_gate_fits(&ly.gate, m, c.dim, ly.experts.kpad)
    }

    fn ffn_dots_in(&self, ly: &Layer, m: usize) -> bool {
        if ly.experts.sgl {
            return false;
        }
        let z = 2;
        self.ffn_gate_in(ly, m) && (ly.experts.n >= 256) && ops::moe_dots_fit(self.cfg.dim, z)
    }

    fn ffn(&mut self, ly: *const Layer, m: usize, topk: usize) -> Result<()> {
        let ly: &Layer = unsafe { &*ly };
        let c = self.cfg.clone();
        let dim = c.dim;

        let n_exp = ly.experts.n;
        let xf = self.xin(ly.ffn_norm);
        let fused_ok = m <= 8 && ops::moe_fused_shape(dim, ly.experts.kpad).is_some();
        let lp = fused_ok && self.ffn_lp_ok(m, topk);
        let ypair = if lp { self.ffn_lp_buf.as_ref().map(|b| b.ptr).unwrap_or(0) } else { self.b.ypair };
        ensure!(ypair != 0, "ffn lp: no Lamport-armed MoE output buffer");
        let ysh = ypair + (m * topk * dim * 4) as u64;
        let gate_in = self.ffn_gate_in(ly, m);
        debug_assert!(!gate_in || fused_ok);
        let gph = (m <= MAX_M).then_some(ops::GATE_PH);
        let tm = if ly.experts.rm {
            ensure!(fused_ok && gate_in, "tmoe: a {m}-row decode MoE without the gate inside the launch (layer {}): the prefill MoE serves it", ly.id);
            ensure!(n_exp <= 384 && m * topk <= 96, "tmoe_fused: {n_exp} experts, {} pairs", m * topk);
            Some(self.tmoe_params(ly, ypair)?)
        } else {
            None
        };
        let go = GemvOpts { out_f32: true, epilogue: Epilogue::SqrtSoftplus, xq_out: Some((self.b.xq, self.b.xsq)), ph: gph, ..xf };
        let gate_in_any = (2..=MAX_M.min(8)).any(|mm| self.ffn_gate_in(ly, mm));
        let gate2 = self.tg2_step(m).filter(|_| !gate_in && !gate_in_any && true && ly.gate.fmt == WFmt::Bf16 && go.xq_out == Some((self.b.xq, self.b.xsq))).and_then(|_| {
            let tw = self.w.tc.get(&ly.gate.w).filter(|tw| tw.tmap128 != 0 && tw.bf16 && tw.kp == ly.gate.k)?;
            Some((tw.tmap128, ops::tg2_gate_cluster(ly.gate.n, ly.gate.k)?))
        });
        if gate_in {
        } else if let Some((tmap, cs)) = gate2 {
            let nsl = dim / 256;
            let p = ops::Tg2Params {
                tmap,
                x: self.b.x_raw,
                gamma: ly.ffn_norm,
                rs_part: self.b.part,
                out: self.b.glog,
                xq: self.b.xq,
                xsq: self.b.xsq,
                alpha: 1.0,
                eps: c.norm_eps,
                m: m as i32,
                n: n_exp as i32,
                k: dim as i32,
                ldx: dim as i32,
                ldo: n_exp as i32,
                out_f32: 1,
                ssp: 1,
                rs_pre: 1,
                rs_n: nsl as i32,
                rs_stride: 26,
                rs_rowstride: (nsl * 26) as i32,
                rs_off: 25,
                ..Default::default()
            };
            self.tg2_launch(&format!("tcg2b {}x{}", n_exp, dim), false, p, cs)?;
        } else {
            self.gemv(&ly.gate, self.b.x_raw, dim, m, self.b.glog, n_exp, go)?;
        }
        let route = ops::RouteArgs {
            scores: if gate_in { 0 } else { self.b.glog },
            bias: ly.gate_bias,
            ids: self.b.route_ids,
            w: self.b.route_w,
            n_exp: n_exp as i32,
            topk: topk as i32,
            scale: c.route_scale,
            nsum: 1,
        };
        let inter = ly.experts.inter;
        let w13 = Weight { fmt: WFmt::Fp4, w: ly.experts.w13, s: 0, n: 2 * inter, k: dim };
        let w2 = Weight { fmt: WFmt::Fp4, w: ly.experts.w2, s: 0, n: dim, k: ly.experts.kpad };
        let slots = (m * topk).min(n_exp);
        let moe13 = Moe { plan: self.b.plan, w_stride: ly.experts.w13_stride, max_slots: slots, split_hint: topk, sched: self.b.sched, w_sh: ly.shared_k64.0, route };
        let moe2 = Moe { plan: self.b.plan, w_stride: ly.experts.w2_stride, max_slots: slots, split_hint: topk, sched: 0, w_sh: ly.shared_k64.1, ..Default::default() };
        {
            let pdl = true;
            let (q13, q2) = (ops::MoeQuant { x: Some((self.b.xq, self.b.xsq)), out: Some((self.b.actq, self.b.asq)) }, ops::MoeQuant { x: Some((self.b.actq, self.b.asq)), out: None });
            if fused_ok {
                let dots = self.ffn_dots_in(ly, m).then(|| {
                    let hc = ly.hc_ffn;
                    let z = 2;
                    ops::DotsFused {
                        h: self.b.h[self.hcur],
                        fn_: hc.0,
                        ctr: self.b.sched + 40,
                        z,
                        mx: ops::HcMixParams { part: self.b.part, nslices: (dim / 256) as i32, _pad: 0, base: hc.1, scale: hc.2, pre: self.b.pre_a, post: self.b.post_a, comb: self.b.comb_a, dim: dim as i32, iters: c.hc_sinkhorn_iters as i32, norm_eps: c.norm_eps, hc_eps: c.hc_eps },
                    }
                });
                let mixf = if lp && dots.is_some() { self.lp_mixf } else { 0 };
                self.lp_mixf_on.set(mixf != 0 && gate_in);
                let gate = gate_in.then(|| ops::GateFused { w: &ly.gate, dots, part: self.b.glog_part, ctr: self.b.sched + 32, xq: self.b.xq, xsq: self.b.xsq, x: self.b.x_raw, ldx: dim, norm_w: ly.ffn_norm, norm_eps: c.norm_eps, norm_part: self.b.part, nslices: dim / 256, mixf });
                let q13 = if gate_in { ops::MoeQuant { x: None, ..q13 } } else { q13 };
                self.timed("moe fused", || ops::moe_fused(&self.k, &self.stream, &w13, &w2, ly.experts.w2_stride, moe2.w_sh, self.b.x_raw, dim, m, self.b.act, inter, ypair, dim, &moe13, c.swiglu_limit, q13, self.b.moe_done, pdl, gate, tm))?;
            } else {
                ensure!(!lp, "ffn lp: the decode MoE is not the fused kernel");
                self.timed("moe w13", || ops::moe_gemv(&self.k, &self.stream, &w13, self.b.x_raw, dim, m, self.b.act, inter, &moe13, Epilogue::SwiGlu(c.swiglu_limit), xf.prenorm, xf.norm_part, q13, pdl))?;
                self.timed("moe w2", || ops::moe_gemv(&self.k, &self.stream, &w2, self.b.act, inter, m, self.b.ypair, dim, &moe2, Epilogue::Store, None, None, q2, pdl))?;
            }
        }
        let (hin, hout) = (self.b.h[self.hcur], self.b.h[1 - self.hcur]);
        if !self.ffn_dots_in(ly, m) {
            self.hc_join()?;
        }
        if let Some(ei) = c.engram_index(ly.id + 1) {
            if self.eng_pending.get() & (1 << ei) != 0 {
                self.eng_pending.set(self.eng_pending.get() & !(1 << ei));
                self.eng_joined.set(self.eng_joined.get() | (1 << ei));
                self.stream.wait(&self.ev_ejoin[ei])?;
            }
        }
        self.lp_pending.set(lp);
        self.comm_ex(ypair, dim, dim, CM_SUM_HCPOST, false, hout, dim, false, Some((hin, self.b.post_a, self.b.comb_a)), topk, ysh, Some(self.epi_nodots()))?;
        ensure!(!self.lp_pending.get(), "ffn lp: the MoE's Lamport-armed rows were not consumed");
        self.hcur = 1 - self.hcur;
        Ok(())
    }

    fn ffn_lp_ok(&self, m: usize, topk: usize) -> bool {
        if self.ffn_lp_buf.is_none() {
            return false;
        }
        let dim = self.cfg.dim;
        m <= 8
            && self.cur_m == m
            && matches!(topk, 3 | 6)
            && m * (topk + 1) <= FFN_LP_ROWS
            && self.tp == 4
            && dim % 256 == 0
            && m * dim <= NVLS_CAP
            && self.nvls.is_some()
            && ffn_lp_kernel(topk, m * dim / 256).is_some()
    }

    fn ffn_lp_rearm(&self) -> Result<()> {
        if let Some(b) = &self.ffn_lp_buf {
            cu::memset_d32_async(b.ptr, 0x8000_0000, FFN_LP_ROWS * self.cfg.dim, &self.stream)?;
        }
        if let Some(st) = &self.sgl {
            cu::memset_d32_async(st.c, 0x8000_8000, SGL_C_ROWS * self.cfg.dim / 2, &self.stream)?;
            cu::memset_d32_async(st.img + (self.cfg.dim / 128 * 1024) as u64, 0x7f7f_7f7f, self.cfg.dim / 128 * 512 / 4, &self.stream)?;
        }
        Ok(())
    }
}

impl Rank {
    pub fn dspark_prep_tail(&mut self, m: usize, row0: usize) -> Result<()> {
        ensure!(row0 < m && m > MAX_M && self.meta == self.metas);
        let k = m - row0;
        let pos0 = self.host_pos_end - m;
        let slot = self.metas + ((meta_slices(self.pmax) + 2) * std::mem::size_of::<Meta>()) as u64;
        let mt = [Meta { pos: (pos0 + row0) as i32, m: k as i32, tokens: [0; 16] }];
        let mb = unsafe { std::slice::from_raw_parts(mt.as_ptr() as *const u8, std::mem::size_of::<Meta>()) };
        cu::memcpy_htod_async(slot, mb, &self.stream)?;
        let (meta, cur_m, mh) = (self.meta, self.cur_m, self.b.main_hidden);
        self.meta = slot;
        self.cur_m = k;
        self.b.main_hidden = mh + (row0 * 3 * self.cfg.dim * 2) as u64;
        let r = self.dspark_prep(k);
        self.meta = meta;
        self.cur_m = cur_m;
        self.b.main_hidden = mh;
        r
    }

    pub fn dspark_prep(&mut self, m: usize) -> Result<()> {
        let c = self.cfg.clone();
        let dim = c.dim;
        let hd = c.head_dim;
        let _ = (dim, hd);
        self.prep_proj(m)?;
        self.prep_gather(m)?;
        if self.prep_gather_pending.get() {
            self.ev_prep.record(&self.par)?;
        }
        self.prep_kv(m)
    }

    fn prep_proj(&self, m: usize) -> Result<()> {
        let dim = self.cfg.dim;
        let mp = self.w.mtp[0].main_proj.unwrap();
        self.gemv(&mp, self.b.main_hidden, 3 * dim, m, self.b.ekv_part, mp.n, GemvOpts { out_f32: true, xshift: 6, ..Default::default() })
    }

    fn prep_gather(&self, m: usize) -> Result<()> {
        let _ = m;
        let nloc = self.w.mtp[0].main_proj.unwrap().n;
        self.comm(self.b.ekv_part, nloc, nloc, CM_GATHER, false, self.b.main_x, self.cfg.dim, false, None)
    }

    fn prep_kv(&self, m: usize) -> Result<()> {
        let c = &self.cfg;
        let (dim, hd) = (c.dim, c.head_dim);
        let mn = Some((self.w.mtp[0].main_norm.unwrap(), c.norm_eps));
        let nm = self.w.mtp.len();
        if let (Some(wg), true) = (self.w.mtp_wkv, m <= MAX_M && nm == 3) {
            let o1 = GemvOpts { prenorm: mn, ..Default::default() };
            let cs = ops::gemv_cluster(&self.w.mtp[0].wkv, m, &o1);
            let og = GemvOpts { prenorm: mn, ksplit: Some(cs), ..Default::default() };
            ensure!(ops::gemv_cluster(&wg, m, &og) == cs && wg.n == nm * hd, "prep_kv: grouped wkv split");
            self.gemv(&wg, self.b.main_x, dim, m, self.b.kv_all, nm * hd, og)?;
            let rp = |i: usize| self.rope_params(self.b.kv_all + (i * hd * 2) as u64, nm * hd, hd, 1, self.rope_base, false, Q_FP8, 0, self.mtp_caches[i].win, self.w.mtp[i].kv_norm);
            self.rope_pair(m, rp(0), 1, rp(1), 1)?;
            return self.rope_launch(rp(2), m);
        }
        for i in 0..self.w.mtp.len() {
            let ly = &self.w.mtp[i];
            self.gemv(&ly.wkv, self.b.main_x, dim, m, self.b.kv, hd, GemvOpts { prenorm: mn, ..Default::default() })?;
            self.rope_n(self.b.kv, hd, hd, 1, m, self.rope_base, false, Q_FP8, 0, self.mtp_caches[i].win, ly.kv_norm)?;
        }
        Ok(())
    }

    fn mtp_block(&mut self, i: usize, m: usize) -> Result<()> {
        let c = self.cfg.clone();
        let dim = c.dim;
        let hd = c.head_dim;
        let heads = c.n_heads / self.tp;
        let table = self.rope_base;
        let ly: &Layer = &self.w.mtp[i];
        self.hc_fork_dots(ly.hc_attn, i != 0, "mtp attn")?;
        let x = self.b.x_raw;
        let xa = self.xin(ly.attn_norm);
        let qlr = c.q_lora_rank;
        let mtp2 = true;
        let qkv2 = if mtp2 { self.tg2_wqkv(ly, x, m)? } else { None };
        if qkv2.is_none() {
            self.gemv(&ly.wqkv, x, dim, m, self.b.q_lat, qlr, GemvOpts { split: Some((qlr, self.b.kv, hd)), ..xa })?;
        }
        let p = RopeParams {
            x: self.b.kv,
            ldx: hd as i32,
            n: hd as i32,
            rows_per_token: 1,
            rope_off: (hd - c.rope_head_dim) as i32,
            cos: table.0,
            sin: table.1,
            inverse: 0,
            quant: Q_FP8,
            ratio: 0,
            dst: self.b.block_kv,
            pos_offset: 0,
            norm_w: ly.kv_norm,
            norm_eps: c.norm_eps,
            dst_by_row: 1,
            q4: 0,
            sf4: 0,
        };
        let fuse = m <= MAX_M;
        let kv_wqb = fuse;
        if !kv_wqb {
            self.par_fork()?;
            self.on_par(|me| me.rope_launch(p, m))?;
        }
        let qrope = if fuse { Some((table.0, table.1, self.meta, hd, hd - c.rope_head_dim)) } else { None };
        let kv_rope = if kv_wqb { Some(p) } else { None };
        if !(mtp2 && self.tg2_wqb(qkv2, &ly.wq_b, m, qrope, kv_rope, None)?) {
            self.gemv(&ly.wq_b, self.b.q_lat, qlr, m, self.b.q, heads * hd, GemvOpts { prenorm: Some((ly.q_norm, c.norm_eps)), ksplit: qnorm_split(heads * hd), rope: qrope, kv_rope, ..Default::default() })?;
        }
        if !fuse {
            self.rope(self.b.q, hd, hd, heads, m, table, false, Q_NONE, 0, 0)?;
        }
        if !kv_wqb {
            self.par_join()?;
        }

        let sp: u32 = if deepseek_ultrafast_cuda::profile::sm103() { 8 } else { 16 };
        let r = (128 + self.blk).div_ceil(sp as usize).next_multiple_of(16);
        let ap = AttnFParams {
            q: self.b.q,
            win: self.mtp_caches[i].win,
            comp: self.b.block_kv,
            topk: self.block_idx_for(self.blk)?,
            ktop: self.blk as i32,
            scale: (hd as f32).powf(-0.5),
            sink: ly.attn_sink,
            out: self.b.o,
            cos: table.0,
            sin: table.1,
            r: r as i32,
            fixed_window: 1,
            nh: (c.n_heads / self.tp) as i32,
            pre: 0,
            win_floor: 0,
        };
        let hgroups = heads.div_ceil(16) as u32;
        let fused = if hgroups > 1 { "attn_fused_hg" } else { "attn_fused" };
        let af = if hgroups > 1 { fused } else { ops::attn_fused_kernel() };
        let smem = ops::attn_smem_for(af, r, sp as usize, heads.min(16));
        self.k.get(af)?.set_max_dynamic_smem(smem)?;
        let mut cfg = Launch::new(sp, 128).grid3((sp, m as u32, hgroups)).smem(smem);
        cfg.cluster = Some((sp, 1, 1));
        self.launch(af, cfg, Args::new().ptr(self.meta).push(ap))?;
        let gl = c.o_groups / self.tp * c.o_lora_rank;
        let ob_part = m <= MAX_M;
        let o2 = if ob_part && mtp2 { self.tg2_wo(ly, m)? } else { None };
        if o2.is_none() {
            self.gemv(&ly.wo_a, self.b.o, heads * hd, m, self.b.oa, gl, GemvOpts { x_group_rows: c.o_lora_rank, ..Default::default() })?;
        }
        let (ob, ob_ns) = if let Some(ns) = o2 {
            (self.b.ob_part, ns)
        } else if ob_part {
            let o = GemvOpts { out_f32: true, epilogue: Epilogue::Partial, xshift: 4, ..Default::default() };
            let ns = self.tc_cluster_for(&ly.wo_b, m, &o).unwrap_or_else(|| ops::gemv_cluster(&ly.wo_b, m, &o));
            self.gemv(&ly.wo_b, self.b.oa, gl, m, self.b.ob_part, dim, o)?;
            (self.b.ob_part, ns)
        } else {
            self.gemv(&ly.wo_b, self.b.oa, gl, m, self.b.ob, dim, GemvOpts { out_f32: true, xshift: 4, ..Default::default() })?;
            (self.b.ob, 1)
        };
        let (hin, hout) = (self.b.h[self.hcur], self.b.h[1 - self.hcur]);
        self.hc_join()?;
        self.comm_ex(ob, dim, dim, CM_SUM_HCPOST, true, hout, dim, false, Some((hin, self.b.post_a, self.b.comb_a)), ob_ns, 0, Some(self.epi_nodots()))?;
        self.hcur = 1 - self.hcur;
        let ly: &Layer = &self.w.mtp[i];
        if !self.ffn_dots_in(ly, m) {
            self.hc_fork_dots(ly.hc_ffn, true, "mtp ffn")?;
        }
        if ly.experts.sgl {
            ensure!(m <= 8 && self.sgl_chain_ok(), "sgl mtp: a {m}-row drafter FFN off the decode chain");
            let ly: &'static Layer = unsafe { &*(ly as *const Layer) };
            self.ffn_sgl(ly, m, c.dspark_n_activated_experts)?;
        } else {
            self.ffn(ly as *const Layer, m, c.dspark_n_activated_experts)?;
        }
        Ok(())
    }

    pub fn dspark_draft(&mut self) -> Result<()> {
        let blk = self.blk;
        self.hcur = 0;
        if !self.embed_prep(blk, self.w.mtp[0].hc_attn.0, 0)? {
            self.embed_meta(blk, 0)?;
            self.hc_prep(blk, self.epi(self.w.mtp[0].hc_attn.0, true))?;
        }
        for i in 0..self.w.mtp.len() {
            let r = self.mtp_block(i, blk);
            r?;
        }
        let r = self.dspark_draft_tail();
        r
    }

    fn dspark_draft_tail(&mut self) -> Result<()> {
        let c = self.cfg.clone();
        let dim = c.dim;
        let blk = self.blk;
        let last = self.w.mtp.last().unwrap();
        let (vloc, vpad) = (c.vocab_size / self.tp, self.w.head.n);
        let head = self.w.head_fp8.as_ref().unwrap_or(&self.w.head);
        self.gemv(head, self.b.x_raw, dim, blk, self.b.logits, vpad, GemvOpts { out_f32: true, persist: true, ..self.xin(last.final_norm.unwrap()) })?;
        let mh = last.markov_head.unwrap();
        let me = last.markov_embed.unwrap();
        ensure!(mh.fmt == WFmt::Bf16 && mh.k == 256 && mh.n % 64 == 0 && blk <= 8 && c.dspark_markov_rank == 256);
        let nquads = mh.n / 64;
        let cap = 256.min(2 * self.sms as usize);
        let qpc = nquads.div_ceil(cap).max(2).next_multiple_of(2);
        let grid = nquads.div_ceil(qpc);
        ensure!(grid <= cap, "markov_rounds: {grid} CTAs cannot be co-resident");
        let mut q = MkParams {
            w: mh.w,
            nquads: nquads as i32,
            qpc: qpc as i32,
            vloc: vloc as i32,
            vpad: vpad as i32,
            offset: (self.rank * vloc) as i32,
            _pad0: 0,
            table: me,
            logits: self.b.logits,
            drafts: self.b.drafts,
            blk: blk as i32,
            _pad: 0,
            cand: self.b.mk_cand,
            bar: self.b.mk_bar,
            recv: [0; MAX_TP],
            tp: self.tp as i32,
            rank: self.rank as i32,
            seq: self.b.dg_seq,
            cx: 0,
            cw: 0,
            ldx: 0,
            dim: 0,
            xpart: 0,
            conf: 0,
            theta: 0.0,
            _pad1: 0,
            hcond: 0,
            samp: self.samp.0,
            vq: self.samp.3,
            qstat: self.samp.4,
            theta_r: [0.0; 8],
        };
        self.mk_grid.set(grid);
        if let Some((h, theta)) = self.vadapt.get() {
            q.cx = self.b.x_raw;
            q.cw = last.confidence.unwrap();
            q.ldx = dim as i32;
            q.dim = dim as i32;
            q.xpart = self.b.mk_cand + 2 * 256 * 8;
            q.conf = self.conf_buf;
            q.theta = theta;
            q.theta_r = [theta; 8];
            if blk >= 3 {
                q.theta_r[blk - 2] = theta * 0.5;
                q.theta_r[blk - 1] = theta * 0.25;
            }
            q.hcond = h;
        }
        let off = CommRank::dg_off(self.tp, self.comm.cap);
        for (r, pb) in self.peers.iter().enumerate() {
            q.recv[r] = *pb + off;
        }
        let smem = 2 * 16 * 2048;
        self.k.get("markov_rounds")?.set_max_dynamic_smem(smem)?;
        if self.mk_kmax != 0 && qpc == 2 {
            self.k.get("markov_rounds_ld")?.set_max_dynamic_smem(smem)?;
            self.launch("markov_rounds_ld", Launch::new(grid as u32, 512).smem(smem), Args::new().ptr(self.meta).push(q).ptr(self.mk_kmax))?;
        } else {
            self.launch("markov_rounds", Launch::new(grid as u32, 512).smem(smem), Args::new().ptr(self.meta).push(q))?;
        }
        if self.mk_dump.get() != 0 {
            cu::memcpy_dtod_async(self.mk_dump.get(), self.b.logits, blk * vpad * 4, &self.stream)?;
        }
        Ok(())
    }
}

impl Rank {
    pub fn verify_prep(&mut self, m: usize) -> Result<()> {
        let n = self.cfg.n_layers;
        self.spec_prep_par.set(true);
        let r = self.forward(m, n, None);
        self.spec_prep_par.set(false);
        r?;
        self.par_join()
    }

    pub fn capture_spec(&mut self) -> Result<cu::GraphExec> {
        self.capture_spec_vadapt(deepseek_ultrafast_cuda::profile::vadapt())
    }

    fn block_idx_for(&self, blk: usize) -> Result<DevPtr> {
        if blk == self.cfg.dspark_block_size {
            return Ok(self.b.block_idx);
        }
        if let Some(&p) = self.block_idx_by.borrow().get(&blk) {
            return Ok(p);
        }
        ensure!(!self.stream.is_capturing()?, "block index table for {blk} created inside a capture");
        let t: Vec<i32> = (0..MAX_M * blk).map(|i| (i % blk) as i32).collect();
        let b = cu::upload_synced(&t)?;
        let p = b.ptr;
        std::mem::forget(b);
        self.block_idx_by.borrow_mut().insert(blk, p);
        Ok(p)
    }

    pub fn capture_spec_blk(&mut self, blk: usize) -> Result<cu::GraphExec> {
        let c = self.cfg.clone();
        ensure!(blk >= 1 && blk <= c.dspark_block_size && blk + 1 <= MAX_M, "draft block {blk}");
        self.moe_rows_ok(blk + 1)?;
        self.block_idx_for(blk)?;
        self.blk = blk;
        self.ffn_lp_rearm()?;
        self.stream.synchronize()?;
        self.stream.begin_capture()?;
        let r = (|| -> Result<()> {
            self.launch("spec_draft_meta", Launch::new(1, 32), Args::new().ptr(self.meta).ptr(self.b.spec_state).i32(blk as i32).i32(c.dspark_noise_token_id as i32))?;
            self.cur_m = blk;
            let r = self.dspark_draft();
            r?;
            self.launch("spec_verify_meta", Launch::new(1, 32), Args::new().ptr(self.meta).ptr(self.b.drafts).i32(blk as i32 + 1))?;
            self.cur_m = blk + 1;
            let r = self.verify_prep(blk + 1);
            r?;
            self.launch(
                "spec_accept",
                Launch::new(1, 32),
                Args::new().ptr(self.meta).ptr(self.b.drafts).ptr(self.b.next).i32(blk as i32).ptr(self.b.out_tokens).ptr(self.b.out_count).i32(OUT_CAP as i32).ptr(self.b.spec_state).i32(0).i32(0).ptr(self.samp.5).ptr(self.spec_snap_buf),
            )
        })();
        let g = self.stream.end_capture();
        r?;
        let g = g?;
        let n = g.node_count()?;
        let e = g.instantiate()?;
        e.upload(&self.stream)?;
        if self.rank == 0 {
            deepseek_ultrafast_logging::diagnostic!("captured spec graph: {n} nodes");
        }
        Ok(e)
    }

    fn spec_verify_body(&mut self, l: usize, blk: usize) -> Result<()> {
        let c = self.cfg.clone();
        self.cur_m = l + 1;
        let r = self.verify_prep(l + 1);
        r?;
        self.launch(
            "spec_accept",
            Launch::new(1, 32),
            Args::new().ptr(self.meta).ptr(self.b.drafts).ptr(self.b.next).i32(l as i32).ptr(self.b.out_tokens).ptr(self.b.out_count).i32(OUT_CAP as i32).ptr(self.b.spec_state).i32(blk as i32).i32(c.dspark_noise_token_id as i32).ptr(self.samp.5).ptr(self.spec_snap_buf),
        )
    }

    pub fn capture_spec_vadapt(&mut self, theta: f32) -> Result<cu::GraphExec> {
        let c = self.cfg.clone();
        let blk = c.dspark_block_size;
        ensure!(self.w.mtp.last().is_some_and(|l| l.confidence.is_some()) && blk + 1 <= MAX_M, "vadapt: confidence head, block {blk}");
        self.moe_rows_ok(blk + 1)?;
        self.block_idx_for(blk)?;
        self.blk = blk;
        self.stream.synchronize()?;
        let g = cu::Graph::empty()?;
        let h = g.cond_handle_flags((blk - 1) as u32, true)?;
        self.vadapt.set(Some((h, theta)));
        self.ffn_lp_rearm()?;
        self.stream.synchronize()?;
        self.stream.begin_capture_into(g.raw)?;
        let r = (|| -> Result<()> {
            self.cur_m = blk;
            let r = self.dspark_draft();
            r?;
            self.vadapt.set(None);
            Ok(())
        })();
        self.vadapt.set(None);
        let e = self.stream.end_capture_into();
        r?;
        e?;
        let sinks = g.sinks()?;
        let bodies = g.add_switch(&sinks, h, blk as u32)?;
        for (i, &b) in bodies.iter().enumerate() {
            self.stream.begin_capture_into(b)?;
            let r = self.spec_verify_body(i + 1, blk);
            let e = self.stream.end_capture_into();
            r?;
            e?;
        }
        let n = g.node_count()?;
        let e = g.instantiate()?;
        e.upload(&self.stream)?;
        if self.rank == 0 {
            deepseek_ultrafast_logging::diagnostic!("captured adaptive spec graph: {n} top-level nodes, {blk} verify bodies");
        }
        Ok(e)
    }

    pub fn set_sampling(&mut self, seed: u64) -> Result<()> {
        let sp = SampParams { temp: 1.0, top_p: 1.0, seed, step: 0, _pad: 0 };
        let b = unsafe { std::slice::from_raw_parts(&sp as *const SampParams as *const u8, std::mem::size_of::<SampParams>()) };
        cu::memcpy_htod_async(self.samp.0, b, &self.stream)?;
        Ok(())
    }

    pub fn spec_begin(&mut self, anchor: i32, pos: usize) -> Result<()> {
        let blk = self.cfg.dspark_block_size;
        let mut tokens = [0i32; 16];
        tokens[0] = anchor;
        for t in tokens.iter_mut().take(blk).skip(1) {
            *t = self.cfg.dspark_noise_token_id as i32;
        }
        let meta = Meta { pos: pos as i32, m: blk as i32, tokens };
        self.set_meta(&meta)?;
        cu::memcpy_htod_async(self.b.spec_state, &[anchor, 0, 0, 0], &self.stream)?;
        cu::memset_d32_async(self.b.out_count, 0, 1, &self.stream)?;
        self.stream.synchronize()
    }

    pub fn spec_snapshot(&self, slot: usize) -> Result<()> {
        let mut sn = self.spec_snap.borrow_mut();
        if sn.is_empty() {
            for i in 0..2 {
                sn.push((self.spec_snap_buf + (i * 16) as u64, cu::Event::new()?));
            }
        }
        sn[slot].1.record(&self.stream)
    }

    pub fn spec_snapshot_read(&self, slot: usize) -> Result<(i32, i32, i32)> {
        let sn = self.spec_snap.borrow();
        sn[slot].1.synchronize()?;
        let v: Vec<i32> = cu::download(sn[slot].0, 4)?;
        Ok((v[1], v[2], v[3]))
    }

    pub fn read_out_nosync(&self, n: usize) -> Result<Vec<i32>> {
        cu::download(self.b.out_tokens, n)
    }

    pub fn finish_decode_cache_with_stats(&mut self, prompt: &[i32], first: i32, emitted: usize, chunk: usize) -> Result<(Vec<i32>, [i32; 3])> {
        let (steps, accepted_drafts, committed) = self.spec_stats()?;
        ensure!(committed >= 0, "negative speculative commit count");
        let (end, forwarded) = crate::decode_cache::boundary(prompt.len(), emitted, committed as usize, self.max_seq)?;
        let accepted = if emitted > 1 { self.read_out_nosync(emitted - 1)? } else { vec![] };
        let tokens = crate::decode_cache::tokens(prompt, first, emitted, &accepted)?;
        if forwarded < end {
            self.prefill(&tokens, forwarded, chunk)?;
        }
        Ok((tokens, [steps, accepted_drafts, committed]))
    }

    pub fn spec_stats(&self) -> Result<(i32, i32, i32)> {
        self.stream.synchronize()?;
        let v: Vec<i32> = cu::download(self.b.spec_state, 4)?;
        Ok((v[1], v[2], v[3]))
    }
}

fn qnorm_split(n: usize) -> Option<usize> {
    let quads = n / 64;
    Some([8, 4, 2, 1].into_iter().find(|cs| quads * cs <= ops::sm_count()).unwrap_or(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_capacity_covers_chunk_window_and_slices() {
        let w = 128;
        for chunk in [1, 15, 16, 17, 64, 127, 128, 129, 255, 4096, 4097, 8192, 100_000, usize::MAX - 1, usize::MAX] {
            let cap = prefill_capacity(chunk, w);
            assert!(cap >= chunk.min(PMAX) && cap >= w && cap >= MAX_M, "chunk {chunk}: {cap}");
            assert!(cap <= PMAX && cap % MAX_M == 0, "chunk {chunk}: {cap}");
            let slices = meta_slices(cap);
            assert!(slices * MAX_M >= cap && (slices - 1) * MAX_M < cap);
            for n in [1, MAX_M, MAX_M + 1, cap - 1, cap] {
                let last_slice = 1 + n.div_ceil(MAX_M) - 1;
                assert!(last_slice < 1 + slices, "n {n} of {cap}");
            }
        }
        assert_eq!(prefill_capacity(16, 128), 128);
        assert_eq!(prefill_capacity(129, 128), 144);
        assert_eq!(prefill_capacity(8192, 128), 8192);
        assert_eq!(prefill_capacity(usize::MAX, 128), PMAX);
        assert_eq!(meta_slices(144), 9);
        assert_eq!(meta_slices(129), 9);
        assert_eq!(meta_slices(128), 8);
    }

    #[test]
    fn score_rows_cover_the_shard_and_tiny_chunks() {
        for tp in [2, 4, 8] {
            for pmax in [128, 144, 4096, 8192] {
                let rows = Rank::score_rows(pmax, tp);
                for m in (MAX_M + 1..=pmax).step_by(7).chain([pmax]) {
                    for r in 0..tp {
                        let (r0, r1) = shard_rows(r, m, tp);
                        assert!(r1 - r0 <= rows, "tp {tp} pmax {pmax} m {m} rank {r}: {} > {rows}", r1 - r0);
                    }
                }
                assert!(rows >= 4 * tp + MAX_M);
            }
        }
    }

    #[test]
    fn ffn_lp_entries() {
        assert_eq!(ffn_lp_kernel(6, 120), Some(("variant_zzzzzzzzzlp/comm_lp_ffn_6_2zp".into(), 60, 160)));
        assert_eq!(ffn_lp_kernel(3, 20), Some(("variant_zzzzzzzzzlp/comm_lp_ffn_3_2zp".into(), 10, 160)));
        assert_eq!(ffn_lp_kernel(6, 19), None);
        assert_eq!(ffn_lp_kernel(7, 20), None);
    }
}
