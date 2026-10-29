use anyhow::Result;
use deepseek_ultrafast_cuda as cu;
use deepseek_ultrafast_engine::model::{Meta, Rank};
use deepseek_ultrafast_engine::weights::{Arena, Loader};
use deepseek_ultrafast_model::config::Config;
use deepseek_ultrafast_model::st::Checkpoint;
use std::path::PathBuf;
use std::sync::{Barrier, Mutex};
use std::time::Instant;

const USAGE: &str = "run --tokens <json> [--ckpt DIR] [--gpus 0,1,..] [--decode N] [--max-seq N] \
[--chunk N] [--seed N] [--runs N] [--fixed-block N] [--dump-tokens FILE] [--samp-test N] [--score FILE]";

struct Args {
    ckpt: PathBuf,
    gpus: Vec<i32>,
    prompt: Vec<i32>,
    decode: usize,
    max_seq: usize,
    chunk: usize,
    seed: u64,
    dump_tokens: Option<PathBuf>,
    samp_test: Option<usize>,
    score: Option<Vec<i32>>,
    runs: usize,
    fixed_block: Option<usize>,
}

fn parse_args(v: &[String]) -> Result<Args> {
    let mut a = Args {
        ckpt: PathBuf::from("/models"),
        gpus: vec![0, 1, 2, 3],
        prompt: vec![],
        decode: 8,
        max_seq: 8192,
        chunk: 8192,
        seed: 0,
        dump_tokens: None,
        samp_test: None,
        score: None,
        runs: 1,
        fixed_block: None,
    };
    let ids = |v: &serde_json::Value| -> Vec<i32> { v.as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap() as i32).collect()).unwrap_or_default() };
    let mut i = 1;
    while i < v.len() {
        let val = || v.get(i + 1).cloned().ok_or_else(|| anyhow::anyhow!("{} needs a value\nusage: {USAGE}", v[i]));
        match v[i].as_str() {
            "--ckpt" => a.ckpt = val()?.into(),
            "--gpus" => a.gpus = val()?.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?,
            "--decode" => a.decode = val()?.parse()?,
            "--max-seq" => a.max_seq = val()?.parse()?,
            "--chunk" => a.chunk = val()?.parse()?,
            "--seed" => a.seed = val()?.parse()?,
            "--tokens" => a.prompt = serde_json::from_str(&std::fs::read_to_string(val()?)?)?,
            "--dump-tokens" => a.dump_tokens = Some(val()?.into()),
            "--samp-test" => a.samp_test = Some(val()?.parse()?),
            "--runs" => a.runs = val()?.parse()?,
            "--fixed-block" => a.fixed_block = Some(val()?.parse()?),
            "--score" => {
                let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(val()?)?)?;
                a.prompt = ids(&j["input_ids"]);
                a.score = Some(ids(&j["output_ids"]));
            }
            x => anyhow::bail!("unknown argument {x}\nusage: {USAGE}"),
        }
        i += 2;
    }
    anyhow::ensure!(!a.prompt.is_empty(), "no prompt (use --tokens or --score)\nusage: {USAGE}");
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args(&std::env::args().collect::<Vec<_>>())?;
    deepseek_ultrafast_model::st::require_checkpoint(&args.ckpt)?;
    cu::init()?;
    cu::profile::init(cu::device_cc(args.gpus[0])?, args.gpus.len());
    let _ = rayon::ThreadPoolBuilder::new().stack_size(64 << 20).build_global();
    let cfg = deepseek_ultrafast_engine::load_config(&args.ckpt)?;
    let t0 = Instant::now();
    let ck = Checkpoint::open(&args.ckpt)?;
    let engram = deepseek_ultrafast_engine::engram::EngramCell::new(&args.ckpt, &cfg);
    let tp = args.gpus.len();
    let prompt = &args.prompt;

    let barrier = Barrier::new(tp);
    let peers: Mutex<Vec<u64>> = Mutex::new(vec![0; tp]);
    let shards: Mutex<Vec<Vec<f32>>> = Mutex::new(vec![vec![]; tp]);
    let ctxs: Mutex<Vec<usize>> = Mutex::new(vec![0; tp]);
    std::thread::scope(|sc| -> Result<()> {
        sc.spawn(|| engram.get().map(|_| ()));
        let mut hs = Vec::new();
        for r in 0..tp {
            let (ck, engram, cfg, args, barrier, peers, ctxs, shards) = (&ck, &engram, &cfg, &args, &barrier, &peers, &ctxs, &shards);
            hs.push(std::thread::Builder::new().stack_size(256 << 20).spawn_scoped(sc, move || -> Result<()> {
                let res = (move || -> Result<()> {
                    let dev = cu::Device::new(args.gpus[r])?;
                    ctxs.lock().unwrap()[r] = dev.ctx as usize;
                    barrier.wait();
                    for (q, c) in ctxs.lock().unwrap().clone().into_iter().enumerate() {
                        if q != r {
                            let res = unsafe { cu::sys::cuCtxEnablePeerAccess(c as cu::sys::CUcontext, 0) };
                            if res != cu::sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
                                cu::check(res, "peer access")?;
                            }
                        }
                    }
                    let mut arena = Arena::default();
                    let lt = Instant::now();
                    let w = Loader { ck, cfg, rank: r, tp, tc: Default::default() }.load(&mut arena, cfg.n_layers, true)?;
                    let engram = engram.get()?.upload(&mut arena)?;
                    println!("rank {r}: weights {:.1} GB in {:.1}s", arena.bytes as f64 / 1e9, lt.elapsed().as_secs_f32());
                    let mut rk = Rank::new(r, tp, cfg.clone(), dev, w, arena, args.max_seq, args.chunk, engram)?;
                    rk.set_sampling(args.seed)?;
                    peers.lock().unwrap()[r] = rk.comm.recv;
                    barrier.wait();
                    rk.peers = peers.lock().unwrap().clone();
                    deepseek_ultrafast_engine::model::setup_nvls(&mut rk, &args.gpus, barrier)?;
                    let (free, _) = rk.dev.mem_info()?;
                    if r == 0 {
                        println!("ready after {:.1}s, {:.1} GB free per GPU", t0.elapsed().as_secs_f32(), free as f64 / 1e9);
                    }
                    barrier.wait();

                    let ptime = Instant::now();
                    let tok = rk.prefill(prompt, 0, args.chunk)?;
                    if let Some(cont) = &args.score {
                        return score(&mut rk, r, tp, cfg, barrier, shards, prompt.len(), cont);
                    }
                    if r == 0 {
                        println!("prefill {} tokens: {:.1} ms (first token {tok})", prompt.len(), ptime.elapsed().as_secs_f64() * 1e3);
                    }
                    if args.decode == 0 {
                        return Ok(());
                    }
                    if let Some(n) = args.samp_test {
                        return samp_test(&mut rk, r, cfg, barrier, shards, prompt.len(), tok, n, args.seed);
                    }
                    rk.spec_begin(tok, prompt.len())?;
                    let g = match args.fixed_block {
                        Some(b) => rk.capture_spec_blk(b)?,
                        None => rk.capture_spec()?,
                    };
                    let (mut all_steps, mut all_committed, mut all_dt) = (0u64, 0u64, 0f64);
                    for run in 0..args.runs {
                        let seed = args.seed + run as u64;
                        if run > 0 {
                            rk.set_sampling(seed)?;
                        }
                        rk.spec_begin(tok, prompt.len())?;
                        barrier.wait();
                        let st = Instant::now();
                        let mut launched = 0;
                        loop {
                            for _ in 0..8 {
                                g.launch(&rk.stream)?;
                            }
                            launched += 8;
                            let (_, _, committed) = rk.spec_stats()?;
                            if committed as usize + 1 >= args.decode || launched > 4 * args.decode {
                                break;
                            }
                        }
                        let dt = st.elapsed().as_secs_f64();
                        let (steps, acc, committed) = rk.spec_stats()?;
                        (all_steps, all_committed, all_dt) = (all_steps + steps as u64, all_committed + committed as u64, all_dt + dt);
                        if r != 0 {
                            continue;
                        }
                        let out = rk.read_out(committed as usize)?;
                        let mut generated = vec![tok];
                        generated.extend_from_slice(&out);
                        let n = generated.len();
                        let label = if args.runs > 1 { format!("graph spec decode (seed {seed})") } else { "graph spec decode".to_string() };
                        println!("{label}: {n} tokens in {steps} steps ({:.2} tokens/step, {:.2} accepted drafts/step), {:.1} ms ({:.1} tok/s, {:.3} ms/step)", committed as f64 / steps as f64, acc as f64 / steps as f64, dt * 1e3, (committed as f64) / dt, dt * 1e3 / steps as f64);
                        if run + 1 < args.runs {
                            continue;
                        }
                        println!("generated: {:?}", &generated[..n.min(40)]);

                        if let Some(path) = &args.dump_tokens {
                            std::fs::write(path, serde_json::to_string(&generated)?)?;
                        }
                        if args.runs > 1 {
                            println!("all {} runs: {all_committed} tokens in {all_steps} steps ({:.3} tokens/step), {:.1} tok/s, {:.4} ms/step", args.runs, all_committed as f64 / all_steps as f64, all_committed as f64 / all_dt, all_dt * 1e3 / all_steps as f64);
                        }
                    }
                    Ok(())
                })();
                if let Err(e) = &res {
                    eprintln!("[rank {r}] error: {e:#}");
                }
                res
            }));
        }
        for h in hs {
            h?.join().unwrap()?;
        }
        Ok(())
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn samp_test(rk: &mut Rank, r: usize, cfg: &Config, barrier: &Barrier, shards: &Mutex<Vec<Vec<f32>>>, plen: usize, tok: i32, n: usize, seed: u64) -> Result<()> {
    rk.spec_begin(tok, plen)?;
    let gs = rk.capture_spec()?;
    let mut spec: Vec<i32> = vec![];
    let mut stepsum = 0usize;
    for t in 0..n {
        rk.spec_begin(tok, plen)?;
        rk.set_sampling(seed * 1_000_003 + t as u64)?;
        gs.launch(&rk.stream)?;
        gs.launch(&rk.stream)?;
        let (_, acc, committed) = rk.spec_stats()?;
        stepsum += acc as usize;
        let o = rk.read_out_nosync(committed as usize)?;
        spec.push(o[0]);
    }
    let meta0 = Meta { pos: plen as i32, m: 1, tokens: { let mut t = [0i32; 16]; t[0] = tok; t } };
    rk.set_meta(&meta0)?;
    rk.forward(1, cfg.n_layers, None)?;
    shards.lock().unwrap()[r] = rk.read_logits(1)?;
    barrier.wait();
    if r == 0 {
        let z: Vec<f64> = shards.lock().unwrap().concat().iter().map(|&x| x as f64).collect();
        let mx = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = mx + z.iter().map(|&x| (x - mx).exp()).sum::<f64>().ln();
        let ex: Vec<f64> = z.iter().map(|&x| (x - lse).exp()).collect();
        let mut ord: Vec<usize> = (0..ex.len()).collect();
        ord.sort_by(|&a, &b| ex[b].partial_cmp(&ex[a]).unwrap());
        println!("exact: top probs {:?}", ord.iter().take(5).map(|&i| (i, (ex[i] * 1e4).round() / 1e4)).collect::<Vec<_>>());
        let mut h = vec![0usize; ex.len()];
        let mut bad = 0;
        for &a in &spec {
            if a >= 0 && (a as usize) < h.len() { h[a as usize] += 1; } else { bad += 1; }
        }
        let (mut chi, mut dof, mut re, mut ro) = (0f64, 0usize, 0f64, 0f64);
        let (mut te, mut to) = (0f64, 0usize);
        for i in 0..ex.len() {
            let e = ex[i] * n as f64;
            if e >= 5.0 { chi += (h[i] as f64 - e).powi(2) / e; dof += 1; } else { re += e; ro += h[i] as f64; }
            if ex[i] < 1e-3 { te += e; to += h[i]; }
        }
        if re > 0.0 { chi += (ro - re).powi(2) / re; dof += 1; }
        println!("spec vs exact (first token): chi2 {chi:.1} on {} dof (p > 0.01 needs < ~{:.0}); tokens with p < 1e-3: expected {te:.1}, got {to}; invalid ids: {bad}", dof.saturating_sub(1), dof as f64 + 2.33 * (2.0 * dof as f64).sqrt());
        println!("spec: {:.2} accepted drafts per step over {} trials of 2 steps", stepsum as f64 / (2 * n) as f64, n);
    }
    barrier.wait();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn score(rk: &mut Rank, r: usize, tp: usize, cfg: &Config, barrier: &Barrier, shards: &Mutex<Vec<Vec<f32>>>, plen: usize, cont: &[i32]) -> Result<()> {
    const BLOCK: usize = 8;
    let n = cont.len();
    anyhow::ensure!(n >= 1, "--score: empty continuation");
    let v = cfg.vocab_size / tp;
    let (mut nll_sum, mut agree) = (0f64, 0usize);
    let mut nlls: Vec<f64> = vec![];
    let st = Instant::now();
    let mut eval = |full: &[f32], tok: i32| {
        let mx = full.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let lse = mx + full.iter().map(|x| (*x as f64 - mx).exp()).sum::<f64>().ln();
        let am = full.iter().enumerate().fold((0usize, f32::MIN), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
        let nll = lse - full[tok as usize] as f64;
        nll_sum += nll;
        nlls.push(nll);
        agree += (am as i32 == tok) as usize;
    };
    let lg = rk.read_last_logits()?;
    shards.lock().unwrap()[r] = lg[..v].to_vec();
    barrier.wait();
    if r == 0 {
        let full: Vec<f32> = shards.lock().unwrap().concat();
        eval(&full, cont[0]);
    }
    barrier.wait();
    let mut j = 0usize;
    while j + 1 < n {
        let b = BLOCK.min(n - 1 - j);
        let mut meta = Meta { pos: (plen + j) as i32, m: b as i32, tokens: [0; 16] };
        meta.tokens[..b].copy_from_slice(&cont[j..j + b]);
        rk.set_meta(&meta)?;
        rk.forward(b, cfg.n_layers, None)?;
        shards.lock().unwrap()[r] = rk.read_logits(b)?;
        barrier.wait();
        if r == 0 {
            let sh = shards.lock().unwrap();
            for i in 0..b {
                let full: Vec<f32> = (0..tp).flat_map(|q| sh[q][i * v..(i + 1) * v].iter().copied()).collect();
                eval(&full, cont[j + i + 1]);
            }
        }
        barrier.wait();
        j += b;
    }
    if r == 0 {
        let cnt = nlls.len();
        let mean = nll_sum / cnt as f64;
        let sd = (nlls.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (cnt as f64 - 1.0).max(1.0)).sqrt();
        println!("score: {cnt} tokens ({:.1} s): mean NLL {mean:.5} (+- {:.5} s.e.), ppl {:.4}, argmax == recorded {agree}/{cnt}", st.elapsed().as_secs_f64(), sd / (cnt as f64).sqrt(), mean.exp());
    }
    Ok(())
}
