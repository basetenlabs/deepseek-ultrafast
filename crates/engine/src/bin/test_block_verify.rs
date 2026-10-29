use anyhow::{Result, ensure};
use deepseek_ultrafast_cuda::{self as cu, Args, Buffer, GraphExec, Launch, Module, Stream};
use deepseek_ultrafast_engine::{model::Meta, sampling::{SampIo, SampParams}};

const RRW: usize = 8 + 3 * 64;
const SO_S2: usize = 2 * 8 * 8 * 99 * 4;
const SO_FR: usize = SO_S2 + 128;
const SO_BV: usize = SO_FR + 3 * 8 * 8 * 256 * 4 + 8 * 8 * 2048 * 4;
const RECV_WORDS: usize = SO_BV + 2 * 8 * 8 * 3 * 4;
const MAX_LOCAL: usize = 32320;
const SEED: u64 = 240310444;

struct TestRank {
    module: Module,
    stream: Stream,
    io: SampIo,
    meta: Buffer,
    logits: Buffer,
    sp: Buffer,
    recv: Buffer,
    seq: Buffer,
    rowres: Buffer,
    out: Buffer,
    next: Buffer,
    records: Buffer,
    drafts: Buffer,
    vq: Buffer,
    qstat: Buffer,
    dev: cu::Device,
}
impl Drop for TestRank {
    fn drop(&mut self) {
        let _ = self.dev.bind();
    }
}

impl TestRank {
    fn new(rank: usize, tp: usize, image: &[u8], block: bool) -> Result<Self> {
        let dev = cu::Device::new(rank as i32)?;
        let module = Module::load(image)?;
        let stream = Stream::new()?;
        let meta = Buffer::from_host(&[Meta { m: 2, ..Default::default() }])?;
        let logits = Buffer::zeroed(8 * MAX_LOCAL * 4)?;
        let sp = Buffer::zeroed(std::mem::size_of::<SampParams>())?;
        let recv = Buffer::zeroed(RECV_WORDS * 4)?;
        let seq = Buffer::zeroed(4)?;
        let rowres = Buffer::zeroed(8 * RRW * 4)?;
        let out = Buffer::zeroed(64)?;
        let next = Buffer::zeroed(32)?;
        let records = Buffer::zeroed(8 * MAX_LOCAL.div_ceil(64) * 20 * 4)?;
        let drafts = Buffer::from_host(&[0i32; 8])?;
        let vq = Buffer::zeroed(8 * MAX_LOCAL * 4)?;
        let qstat = Buffer::zeroed(8 * 2 * 4)?;
        let io = SampIo {
            logits: logits.ptr, vloc: 0, vpad: 0, offset: 0, qn: 1,
            drafts: drafts.ptr, vq: vq.ptr, qstat: qstat.ptr, sp: sp.ptr, recv: [0; 8],
            tp: tp as i32, rank: rank as i32, seq: seq.ptr, rowres: rowres.ptr, out: out.ptr, next: next.ptr,
            force_fb: 0, last_only: 0, spart: records.ptr, nquads: 0,
            so_s2: SO_S2 as i32, so_fr: SO_FR as i32, leps: (1e-3f32).ln(), so_bv: SO_BV as i32, block: block as i32,
        };
        Ok(Self { module, stream, io, meta, logits, sp, recv, seq, rowres, out, next, records, drafts, vq, qstat, dev })
    }

    fn prepare(&mut self, z: &[Vec<f32>], q: &[Vec<f32>], drafts: &[i32], top_p: f32, step: u64) -> Result<()> {
        self.dev.bind()?;
        let rows = z.len();
        let vocab = z[0].len();
        let tp = self.io.tp as usize;
        let n = vocab / tp;
        let vpad = n.next_multiple_of(64);
        ensure!(vocab % tp == 0 && vpad <= MAX_LOCAL && vpad / 64 * 8 <= 4096 && rows <= 8);
        let off = n * self.io.rank as usize;
        let mut zz = Vec::new();
        let mut qq = Vec::new();
        let mut rec = Vec::new();
        let mut qs = Vec::new();
        for row in 0..rows {
            let mut zr = z[row][off..off + n].to_vec();
            zr.resize(vpad, f32::NEG_INFINITY);
            let mut qr = if row < q.len() { q[row][off..off + n].to_vec() } else { vec![f32::NEG_INFINITY; n] };
            qr.resize(vpad, f32::NEG_INFINITY);
            let qm = qr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let qe = if qm.is_finite() { qr.iter().map(|&v| ((v - qm) as f64).exp()).sum::<f64>() as f32 } else { 0. };
            qs.extend([qm, qe]);
            for (quad, values) in zr.chunks(64).enumerate() {
                let mut sorted: Vec<_> = values.iter().enumerate().map(|(i, v)| (*v, off + quad * 64 + i)).collect();
                sorted.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                let mx = sorted[0].0;
                let sum = if mx.is_finite() { sorted.iter().map(|v| ((v.0 - mx) as f64).exp()).sum::<f64>() as f32 } else { 0. };
                rec.extend([mx, sum, sorted[8].0]);
                rec.extend(sorted[..8].iter().map(|v| v.0));
                rec.extend(sorted[..8].iter().map(|v| f32::from_bits(v.1 as u32)));
                rec.push(0.);
            }
            zz.extend(zr);
            qq.extend(qr);
        }
        self.logits.upload(&zz)?;
        self.vq.upload(&qq)?;
        self.records.upload(&rec)?;
        self.qstat.upload(&qs)?;
        self.sp.upload(&[SampParams { temp: 1., top_p, seed: SEED, step, ..Default::default() }])?;
        self.seq.upload(&[step as u32])?;
        self.rowres.upload(&vec![-1234f32; 8 * RRW])?;
        self.out.upload(&[0i32; 16])?;
        self.next.upload(&[-1i32; 8])?;
        let mut d = [0i32; 8];
        d[..drafts.len()].copy_from_slice(drafts);
        self.drafts.upload(&d)?;
        self.meta.upload(&[Meta { m: rows as i32, ..Default::default() }])?;
        self.io.vloc = n as i32;
        self.io.vpad = vpad as i32;
        self.io.offset = off as i32;
        self.io.nquads = (vpad / 64) as i32;
        Ok(())
    }

    fn launch(&self) -> Result<()> {
        let s = cluster();
        let mut l = Launch::new(8 * s, 1024).pdl(true);
        if s > 1 {
            l.cluster = Some((s, 1, 1));
        }
        cu::launch(&self.module.function("samp_stats")?, &l, &self.stream, Args::new().ptr(self.meta.ptr).push(self.io))
    }
    fn capture(&self) -> Result<GraphExec> {
        self.dev.bind()?;
        self.stream.begin_capture()?;
        self.launch()?;
        self.stream.end_capture()?.instantiate()
    }
}

fn rng_bits(step: u64, purpose: u32, a: u32, b: u32) -> u32 {
    let mut c = [a, (b & 0xffffff) | (purpose << 24), step as u32, (step >> 32) as u32];
    let mut k = [SEED as u32, (SEED >> 32) as u32];
    for _ in 0..10 {
        let x = 0xD2511F53u64 * c[0] as u64;
        let y = 0xCD9E8D57u64 * c[2] as u64;
        c = [(y >> 32) as u32 ^ c[1] ^ k[0], y as u32, (x >> 32) as u32 ^ c[3] ^ k[1], x as u32];
        k[0] = k[0].wrapping_add(0x9E3779B9);
        k[1] = k[1].wrapping_add(0xBB67AE85);
    }
    c[0]
}
fn rng_u01(x: u32) -> f64 { ((x >> 9) as f64 + 0.5) / 8388608. }
fn accept_coin(step: u64, i: usize) -> f64 { rng_u01(rng_bits(step, 2, i as u32 - 1, 0)) }

struct Reference { w: Vec<f64>, mass: Vec<f64>, n: usize, margin: f64 }
fn reference(p: &[Vec<f64>], q: &[Vec<f64>], drafts: &[usize], step: u64, block: bool) -> Reference {
    let l = drafts.len();
    let mut w = vec![1f64; l + 1];
    for i in 0..l {
        w[i + 1] = (w[i] * p[i][drafts[i]] / q[i][drafts[i]]).min(1.);
    }
    let mass: Vec<f64> = (0..l).map(|i| {
        let wi = if block { w[i] } else { 1. };
        (0..p[i].len()).map(|v| (wi * p[i][v] - q[i][v]).max(0.)).sum()
    }).collect();
    let mut n = 0;
    let mut margin = f64::INFINITY;
    if block {
        for i in 1..=l {
            let h = if i == l { w[l] } else if w[i] >= 1. { 1. } else { mass[i] / (mass[i] + 1. - w[i]) };
            let u = accept_coin(step, i);
            margin = margin.min((u - h).abs());
            if u < h { n = i; }
        }
    } else {
        for i in 1..=l {
            let (pd, qd) = (p[i - 1][drafts[i - 1]], q[i - 1][drafts[i - 1]]);
            let u = accept_coin(step, i);
            margin = margin.min((u * qd - pd).abs());
            if u * qd < pd { n = i; } else { break; }
        }
    }
    Reference { w, mass, n, margin }
}

struct Chain { p: Vec<Vec<f64>>, q: Vec<Vec<f64>>, live: Vec<usize> }
impl Chain {
    fn vocab(&self) -> usize { self.p[0].len() }
    fn start(&self) -> usize { self.p.len() - 1 }
    fn row(&self, tok: usize) -> usize { self.live.iter().position(|&t| t == tok).expect("a live token") }
}
fn logs(v: &[f64]) -> Vec<f32> { v.iter().map(|&x| if x > 0. { x.ln() as f32 } else { f32::NEG_INFINITY }).collect() }
fn nucleus(p: &[f64], top_p: f64) -> Vec<f64> {
    if top_p >= 1. { return p.to_vec(); }
    let mut ord: Vec<usize> = (0..p.len()).collect();
    ord.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
    let (mut cum, mut k) = (0., 0);
    while k < ord.len() && cum < top_p { cum += p[ord[k]]; k += 1; }
    while k < ord.len() && p[ord[k]] == p[ord[k - 1]] { cum += p[ord[k]]; k += 1; }
    let mut out = vec![0f64; p.len()];
    for &i in &ord[..k] { out[i] = p[i] / cum; }
    out
}
fn draw(dist: &[f64], u: f64) -> usize {
    let mut c = 0.;
    for (i, &v) in dist.iter().enumerate() {
        c += v;
        if u < c { return i; }
    }
    dist.len() - 1
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn law(ranks: &mut [TestRank], chain: &Chain, steps: usize, top_p: f64, trials: usize, graph: bool, name: &str) -> Result<f64> {
    let block = ranks[0].io.block != 0;
    let v = chain.vocab();
    let mut rng = Rng(0x9E3779B97F4A7C15 ^ trials as u64);
    let mut hist = vec![0usize; 27];
    let mut accepted = 0usize;
    let (mut mismatches, mut ambiguous) = (0usize, 0usize);
    let graphs: Vec<GraphExec> = if graph { ranks.iter().map(|r| r.capture()).collect::<Result<_>>()? } else { vec![] };
    let pt: Vec<Vec<f64>> = chain.p.iter().map(|r| nucleus(r, top_p)).collect();
    for t in 0..trials {
        let mut state = chain.start();
        let (mut p, mut q, mut drafts, mut z) = (vec![], vec![], vec![], vec![]);
        for _ in 0..steps {
            p.push(pt[state].clone());
            z.push(logs(&chain.p[state]));
            q.push(chain.q[state].clone());
            let tok = draw(&chain.q[state], rng.next());
            drafts.push(tok);
            state = chain.row(tok);
        }
        p.push(pt[state].clone());
        z.push(logs(&chain.p[state]));
        let step = 1000 + t as u64;
        let lq: Vec<Vec<f32>> = q.iter().map(|r| logs(r)).collect();
        let mut d: Vec<i32> = vec![0];
        d.extend(drafts.iter().map(|&x| x as i32));
        for r in ranks.iter_mut() { r.prepare(&z, &lq, &d, top_p as f32, step)?; }
        for (i, r) in ranks.iter().enumerate() {
            r.dev.bind()?;
            if graph { graphs[i].launch(&r.stream)?; } else { r.launch()?; }
        }
        let mut first: Option<Vec<i32>> = None;
        for r in ranks.iter() {
            r.dev.bind()?;
            r.stream.synchronize()?;
            let out: Vec<i32> = r.out.download(6)?;
            let pick = out[..2].to_vec();
            match &first { Some(f) => ensure!(*f == pick, "{name}: TP ranks disagree: {f:?} vs {pick:?}"), None => first = Some(pick) }
            if r.io.rank == 0 && top_p < 1. { ensure!(out[4] == steps as i32 + 1 && out[5] == 0, "{name}: {} of {} rows took the fallback", out[5], out[4]); }
        }
        let pick = first.unwrap();
        let (n, tok) = (pick[0] as usize, pick[1] as usize);
        ensure!(n <= steps && tok < v, "{name}: invalid pick {pick:?}");
        let rf = reference(&p, &q, &drafts, step, block);
        if rf.margin < 1e-5 { ambiguous += 1; } else if rf.n != n { mismatches += 1; }
        if n < steps {
            let wn = if block { rf.w[n] } else { 1. };
            let ok = wn * p[n][tok] - q[n][tok] > 0. || (rf.mass[n] <= 1e-9 && p[n][tok] > 0.);
            ensure!(ok, "{name}: correction token {tok} outside the residual of row {n} (w {:.4}, p {:.4}, q {:.4})", rf.w[n], p[n][tok], q[n][tok]);
        } else {
            ensure!(p[steps][tok] > 0., "{name}: bonus token outside the target");
        }
        accepted += n;
        let mut out: Vec<usize> = drafts[..n].iter().map(|&t| chain.row(t)).collect();
        out.push(chain.row(tok));
        while out.len() < 3 {
            let s = *out.last().unwrap();
            out.push(chain.row(draw(&pt[s], rng.next())));
        }
        hist[out[0] * 9 + out[1] * 3 + out[2]] += 1;
    }
    ensure!(mismatches == 0, "{name}: {mismatches} acceptance lengths differ from the fp64 reference ({ambiguous} ambiguous)");
    let mut worst = 0f64;
    let t = (2. * 27. / 1e-9f64).ln();
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let e = pt[chain.start()][chain.live[a]] * pt[a][chain.live[b]] * pt[b][chain.live[c]];
                let got = hist[a * 9 + b * 3 + c] as f64 / trials as f64;
                if e == 0. { ensure!(got == 0., "{name}: impossible sequence {a}{b}{c} observed"); continue; }
                let bound = (2. * e * (1. - e) * t / trials as f64).sqrt() + 2. * t / (3. * trials as f64);
                worst = worst.max((got - e).abs() / bound);
                ensure!((got - e).abs() <= bound, "{name}: sequence {a}{b}{c}: {got:.4} vs {e:.4} (bound {bound:.4})");
            }
        }
    }
    let mean = accepted as f64 / trials as f64;
    println!("PASS {name} {} TP={} graph={graph} top_p={top_p} trials={trials}: mean accepted {mean:.3} of {steps}, reference-ambiguous {ambiguous}, worst deviation {worst:.2} of the bound",
             if block { "block" } else { "rejection" }, ranks.len());
    Ok(mean)
}

fn replay(ranks: &mut [TestRank], chain: &Chain, steps: usize, k: usize, name: &str) -> Result<()> {
    let block = ranks[0].io.block != 0;
    let mut rng = Rng(0xD1B54A32D192ED03);
    let mut state = chain.start();
    let (mut p, mut q, mut drafts, mut z) = (vec![], vec![], vec![], vec![]);
    for _ in 0..steps {
        p.push(chain.p[state].clone());
        z.push(logs(&chain.p[state]));
        q.push(chain.q[state].clone());
        let tok = draw(&chain.q[state], rng.next());
        drafts.push(tok);
        state = chain.row(tok);
    }
    p.push(chain.p[state].clone());
    z.push(logs(&chain.p[state]));
    let lq: Vec<Vec<f32>> = q.iter().map(|r| logs(r)).collect();
    let mut d: Vec<i32> = vec![0];
    d.extend(drafts.iter().map(|&x| x as i32));
    let step0 = 5000u64;
    for r in ranks.iter_mut() { r.prepare(&z, &lq, &d, 1., step0)?; }
    let graphs: Vec<GraphExec> = ranks.iter().map(|r| r.capture()).collect::<Result<_>>()?;
    let (mut mismatches, mut ambiguous, mut lengths) = (0usize, 0usize, vec![]);
    for i in 0..k {
        for (r, g) in ranks.iter().zip(&graphs) {
            r.dev.bind()?;
            g.launch(&r.stream)?;
        }
        let mut first: Option<Vec<i32>> = None;
        for r in ranks.iter() {
            r.dev.bind()?;
            r.stream.synchronize()?;
            let out: Vec<i32> = r.out.download(10)?;
            ensure!(out[8] == 0 && out[9] == 0, "{name}: replay {i}: the counters were not reset: {:?}", &out[8..10]);
            let seq: Vec<u32> = r.seq.download(1)?;
            let sp: Vec<SampParams> = r.sp.download(1)?;
            ensure!(seq[0] as u64 == step0 + i as u64 + 1 && sp[0].step == step0 + i as u64 + 1, "{name}: replay {i}: call counter {} / step {} not advanced", seq[0], sp[0].step);
            let pick = out[..2].to_vec();
            match &first { Some(f) => ensure!(*f == pick, "{name}: replay {i}: TP ranks disagree"), None => first = Some(pick) }
        }
        let pick = first.unwrap();
        let rf = reference(&p, &q, &drafts, step0 + i as u64, block);
        if rf.margin < 1e-5 { ambiguous += 1; } else if rf.n != pick[0] as usize { mismatches += 1; }
        lengths.push(pick[0]);
    }
    ensure!(mismatches == 0, "{name}: {mismatches} replays differ from the reference at their step ({ambiguous} ambiguous)");
    println!("PASS {name} {} TP={} replay x{k}: counters reset, step / call counter advanced, lengths {:?}", if block { "block" } else { "rejection" }, ranks.len(), &lengths[..k.min(12)]);
    Ok(())
}

static CLUSTER: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
fn cluster() -> u32 {
    *CLUSTER.get().unwrap_or(&1)
}

fn main() -> Result<()> {
    let (mut tp, mut trials, mut mode, mut cl) = (1usize, 10000usize, String::from("both"), 1u32);
    let argv: Vec<String> = std::env::args().collect();
    for w in argv[1..].chunks(2) {
        let v = w.get(1).ok_or_else(|| anyhow::anyhow!("{} needs a value", w[0]))?;
        match w[0].as_str() {
            "--tp" => tp = v.parse()?,
            "--trials" => trials = v.parse()?,
            "--mode" => mode = v.clone(),
            "--cluster" => cl = v.parse()?,
            x => anyhow::bail!("unknown argument {x} (--tp, --trials, --mode, --cluster)"),
        }
    }
    CLUSTER.set(cl).unwrap();
    cu::init()?;
    ensure!(tp == 1 || tp == 4 || tp == 8, "--tp must be 1, 4 or 8");
    let module = if cluster() > 1 { "variant_zzzzzzzsamp" } else { "sample" };
    let image = deepseek_ultrafast_kernels::CUBINS.iter().find(|(name, _)| *name == module).unwrap().1.to_vec();
    if cluster() > 1 {
        println!("samp_stats from {module}: clusters of {} CTAs per row", cluster());
    }
    let vocab = 256 * tp;
    let live = [0usize, 65, vocab - 1];
    let spread = |rows: &[[f64; 3]]| -> Vec<Vec<f64>> {
        rows.iter().map(|r| { let mut d = vec![0f64; vocab]; for (k, &i) in live.iter().enumerate() { d[i] = r[k]; } d }).collect()
    };
    let chain = |p: Vec<Vec<f64>>, q: Vec<Vec<f64>>| Chain { p, q, live: live.to_vec() };
    let chains = [
        ("iid-scaled-residual", chain(spread(&[[0.6, 0.3, 0.1]; 4]), spread(&[[0.1, 0.1, 0.8]; 4])), 5),
        ("different-support", chain(spread(&[[0.6, 0.4, 0.0]; 4]), spread(&[[0.0, 0.4, 0.6]; 4])), 2),
        ("autoregressive", chain(spread(&[[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.5, 0.5, 0.0], [0.25, 0.75, 0.0]]),
                                 spread(&[[0.3, 0.7, 0.0], [0.6, 0.4, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])), 2),
        ("close-drafter", chain(spread(&[[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5], [0.4, 0.4, 0.2]]),
                                spread(&[[0.45, 0.35, 0.2], [0.25, 0.45, 0.3], [0.3, 0.25, 0.45], [0.35, 0.45, 0.2]])), 7),
        ("one-draft", chain(spread(&[[0.6, 0.3, 0.1]; 4]), spread(&[[0.3, 0.3, 0.4]; 4])), 1),
    ];
    let top_ps = [1.];
    let mut means = vec![];
    let modes: Vec<bool> = match mode.as_str() { "block" => vec![true], "rejection" => vec![false], _ => vec![true, false] };
    for block in modes {
        let mut ranks: Vec<_> = (0..tp).map(|r| TestRank::new(r, tp, &image, block)).collect::<Result<_>>()?;
        let mut peers = [0; 8];
        for (i, r) in ranks.iter().enumerate() {
            peers[i] = r.recv.ptr;
            for (j, q) in ranks.iter().enumerate() {
                if i != j { r.dev.enable_peer(&q.dev)?; }
            }
        }
        for r in &mut ranks { r.io.recv = peers; }
        for (name, chain, steps) in &chains {
            for &top_p in &top_ps {
                for graph in [false, true] {
                    let n = if graph { trials / 4 } else { trials };
                    let mean = law(&mut ranks, chain, *steps, top_p, n, graph, name)?;
                    if !graph { means.push((name.to_string(), block, mean)); }
                }
            }
            replay(&mut ranks, chain, *steps, 64, name)?;
        }
    }
    for (name, _, _) in &chains {
        let (Some(b), Some(r)) = (means.iter().find(|m| m.0 == *name && m.1).map(|m| m.2), means.iter().find(|m| m.0 == *name && !m.1).map(|m| m.2)) else { continue };
        println!("{name}: mean accepted drafts block {b:.3} vs rejection {r:.3} ({:+.1}%)", (b / r - 1.) * 100.);
        ensure!(b >= r * 0.98, "{name}: block verification accepted fewer drafts than rejection sampling");
    }
    Ok(())
}
