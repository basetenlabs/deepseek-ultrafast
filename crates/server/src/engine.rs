use anyhow::Result;
use deepseek_ultrafast_cuda as cu;
use deepseek_ultrafast_engine::kvstore;
use deepseek_ultrafast_engine::model::Rank;
use deepseek_ultrafast_engine::weights::{Arena, Loader};
use deepseek_ultrafast_kv_transfer::{Candidate, PeerCache, is_rdma_required};
use crate::publisher::{Capture, Fence, Publisher};
use deepseek_ultrafast_model::st::Checkpoint;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Barrier, Mutex, OnceLock};
use std::time::Instant;
use deepseek_ultrafast_logging::{Event as LogEvent, emit, ms};
use deepseek_ultrafast_server::request_log::Decode;
use deepseek_ultrafast_server::request_queue::Queue;

struct StartupLog(Mutex<Option<(bool, Instant)>>);
impl StartupLog {
    fn new() -> Self {
        emit(LogEvent::ModelLoadingStart);
        Self(Mutex::new(Some((false, Instant::now()))))
    }
    fn capture(&self) {
        self.end(true);
        emit(LogEvent::CudaGraphCaptureStart);
        *self.0.lock().unwrap() = Some((true, Instant::now()));
    }
    fn end(&self, success: bool) {
        let stage = self.0.lock().unwrap().take();
        if let Some((capture, start)) = stage {
            let elapsed_ms = ms(start.elapsed());
            emit(if capture { LogEvent::CudaGraphCaptureEnd { success, elapsed_ms } }
                 else { LogEvent::ModelLoadingEnd { success, elapsed_ms } });
        }
    }
}
impl Drop for StartupLog { fn drop(&mut self) { self.end(false); } }
pub use deepseek_ultrafast_server::generation::{Event, Finish, Job};

pub struct EngineArgs {
    pub ckpt: PathBuf,
    pub gpus: Vec<i32>,
    pub max_seq: usize,
    pub chunk: usize,
}

struct Shared {
    barrier: Barrier,
    go: [AtomicBool; 2],
    emitted: AtomicUsize,
    peer: OnceLock<Arc<PeerCache>>,
    runtime: tokio::runtime::Handle,
    candidate: Mutex<Option<Candidate>>,
    job: Mutex<Option<Job>>,
    restore_ok: AtomicBool,
    pending: OnceLock<Arc<crate::publisher::Pending>>,
    max_seq: usize,
}

const PUBLICATION_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

impl Shared {
    fn peer(&self) -> &Arc<PeerCache> {
        self.peer.get().expect("peer initialized after all-rank warmup and before readiness")
    }
    fn next_job(&self, rank: usize, admission: &Queue<Job, Option<Candidate>>) -> Result<Option<Job>> {
        if rank == 0 {
            let next = admission.pop();
            let (mut job, mut candidate) = next.map_or((None, None), |(j, c)| (Some(j), c));
            if let Some(job) = job.as_mut() {
                if let Some(pending) = self.pending.get() {
                    pending.wait(&job.session, PUBLICATION_WAIT);
                }
                candidate = match candidate {
                    Some(c) => match self.runtime.block_on(self.peer().revalidate(&job.session, job.prompt(), c)) {
                        Ok(c) => Some(align_job(job, c, self.max_seq)),
                        Err(e) if is_rdma_required(&e) => return Err(e),
                        Err(_) => None,
                    },
                    None => None,
                };
            }
            *self.candidate.lock().unwrap() = candidate;
            *self.job.lock().unwrap() = job;
        }
        self.barrier.wait();
        Ok(self.job.lock().unwrap().clone())
    }
    fn retired(&self, rank: usize) {
        self.barrier.wait();
        if rank == 0 { self.job.lock().unwrap().take(); }
    }
}

fn cpu_worker(task: impl FnOnce()) {
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(task)).is_err() {
        deepseek_ultrafast_logging::flush_fatal();
        std::process::exit(1);
    }
}

fn align_job(job: &mut Job, mut candidate: Candidate, max_seq: usize) -> Candidate {
    let Some(aligned) = candidate.aligned.take() else { return candidate };
    if deepseek_ultrafast_server::generation::max_new_limit(aligned.tokens.len(), max_seq) < job.max_new {
        return candidate;
    }
    let images = job.images.as_ref().map(|im| Arc::new(im.shifted(aligned.at, aligned.shift)));
    if images.as_ref().map_or(&[][..], |im| &im.spans[..]) != &aligned.images[..] {
        return candidate;
    }
    job.tokens = Arc::new(aligned.tokens);
    job.images = images;
    candidate
}

pub fn run(args: EngineArgs, incoming: Receiver<Job>, ready: std::sync::mpsc::Sender<Arc<PeerCache>>, startup: crate::peer::Options, runtime: tokio::runtime::Handle) -> Result<()> {
    let startup_log = StartupLog::new();
    let _ = rayon::ThreadPoolBuilder::new().stack_size(64 << 20).build_global();
    let tp = args.gpus.len();
    anyhow::ensure!(tp > 0, "at least one GPU is required");
    let cfg = deepseek_ultrafast_engine::load_config(&args.ckpt)?;
    let ck = Checkpoint::open(&args.ckpt)?;
    let engram = deepseek_ultrafast_engine::engram::EngramCell::new(&args.ckpt, &cfg);
    let shared = Shared { barrier: Barrier::new(tp), go: [AtomicBool::new(false), AtomicBool::new(false)], emitted: AtomicUsize::new(0), peer: OnceLock::new(), runtime, candidate: Mutex::new(None), job: Mutex::new(None), restore_ok: AtomicBool::new(true), pending: OnceLock::new(), max_seq: args.max_seq };
    let peers: Mutex<Vec<u64>> = Mutex::new(vec![0; tp]);
    let ctxs: Mutex<Vec<usize>> = Mutex::new(vec![0; tp]);
    let ready = Mutex::new(ready);
    let admission: Queue<Job, Option<Candidate>> = Queue::default();
    let (fetch_ready, fetch_start) = std::sync::mpsc::channel();
    std::thread::scope(|sc| {
        sc.spawn(|| engram.get().map(|_| ()));
        {
            let admission = &admission;
            sc.spawn(move || cpu_worker(|| {
                for job in incoming { admission.push(job); }
                admission.close();
            }));
        }
        {
            let (admission, shared, max_seq) = (&admission, &shared, args.max_seq);
            sc.spawn(move || cpu_worker(|| {
                if fetch_start.recv().is_err() { return; }
                while let Some(mut job) = admission.prepare() {
                    let candidate = if job.cancel.load(Ordering::Relaxed) { None } else {
                        let result = shared.runtime.block_on(shared.peer().lookup(&job.session, job.prompt()));
                        match result {
                            Ok(c) => Some(c),
                            Err(e) if is_rdma_required(&e) => {
                                deepseek_ultrafast_logging::termination(&format!("KV preparation: {e:#}"));
                                deepseek_ultrafast_logging::flush_fatal();
                                std::process::exit(1);
                            }
                            Err(_) => None,
                        }
                    };
                    let candidate = candidate.map(|c| align_job(&mut job, c, max_seq));
                    admission.prepared(job, candidate);
                }
            }));
        }
        for r in 0..tp {
            let (ck, engram, cfg, args, shared, peers, ctxs, ready, startup, startup_log, admission, fetch_ready) = (&ck, &engram, &cfg, &args, &shared, &peers, &ctxs, &ready, &startup, &startup_log, &admission, &fetch_ready);
            std::thread::Builder::new()
                .stack_size(256 << 20)
                .spawn_scoped(sc, move || {
                    let res = (move || -> Result<()> {
                        let dev = cu::Device::new(args.gpus[r])?;
                        ctxs.lock().unwrap()[r] = dev.ctx as usize;
                        shared.barrier.wait();
                        for (q, c) in ctxs.lock().unwrap().clone().into_iter().enumerate() {
                            if q != r {
                                let res = unsafe { cu::sys::cuCtxEnablePeerAccess(c as cu::sys::CUcontext, 0) };
                                if res != cu::sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
                                    cu::check(res, "peer access")?;
                                }
                            }
                        }
                        let mut arena = Arena::default();
                        let w = Loader { ck, cfg, rank: r, tp, tc: Default::default() }.load(&mut arena, cfg.n_layers, true)?;
                        let engram = engram.get()?.upload(&mut arena)?;
                        let mut rk = Rank::new(r, tp, cfg.clone(), dev, w, arena, args.max_seq, args.chunk, engram)?;
                        let mut vision = deepseek_ultrafast_engine::vision::Vision::load(ck, cfg.dim, cfg.n_layers, &rk.stream)?;
                        vision.warm_up(&rk.k, &rk.stream)?;
                        rk.vision = Some(vision);
                        peers.lock().unwrap()[r] = rk.comm.recv;
                        shared.barrier.wait();
                        rk.peers = peers.lock().unwrap().clone();
                        deepseek_ultrafast_engine::model::setup_nvls(&mut rk, &args.gpus, &shared.barrier)?;
                        if r == 0 { startup_log.capture(); }
                        shared.barrier.wait();
                        let mut w = Worker::new(rk, r, args.chunk, shared)?;
                        w.rk.dev.synchronize()?;
                        shared.barrier.wait();
                        if r == 0 {
                            startup_log.end(true);
                            let raw_max = kvstore::image_bytes(&w.rk, args.max_seq, deepseek_ultrafast_kv_transfer::SAVED_TAIL, false)?;
                            let peer = Arc::new(shared.runtime.block_on(startup.start(raw_max))?);
                            shared.peer.set(peer.clone()).map_err(|_| anyhow::anyhow!("peer initialized twice"))?;
                            let publisher = Publisher::new(&w.rk, peer.clone(), shared.runtime.clone())?;
                            shared.pending.set(publisher.pending()).map_err(|_| anyhow::anyhow!("publisher initialized twice"))?;
                            w.publisher = Some(publisher);
                            let _ = fetch_ready.send(());
                            let _ = ready.lock().unwrap().send(peer);
                        }
                        shared.barrier.wait();
                        while let Some(job) = shared.next_job(r, admission)? {
                            w.job(&job)?;
                            drop(job);
                            shared.retired(r);
                        }
                        w.rk.dev.synchronize()?;
                        shared.barrier.wait();
                        drop(w);
                        Ok(())
                    })();
                    if let Err(e) = res {
                        startup_log.end(false);
                        deepseek_ultrafast_logging::termination(&format!("rank {r}: {e:#}"));
                        deepseek_ultrafast_logging::flush_fatal();
                        std::process::exit(1);
                    }
                })
                .expect("spawn rank thread");
        }
    });
    Ok(())
}

struct Worker<'a> {
    publisher: Option<Publisher>,
    graph: cu::GraphExec,
    restore: kvstore::RestoreSlot,
    rk: Rank,
    r: usize,
    chunk: usize,
    shared: &'a Shared,
    cached: Vec<i32>,
    decisions: usize,
}

impl<'a> Worker<'a> {
    fn new(mut rk: Rank, r: usize, chunk: usize, shared: &'a Shared) -> Result<Self> {
        let first = prefill_from(&mut rk, chunk, &[0], 0)?;
        rk.spec_begin(first, 1)?;
        rk.set_sampling(0)?;
        let graph = rk.capture_spec()?;
        let restore = kvstore::RestoreSlot::new()?;
        Ok(Worker { publisher: None, rk, r, chunk, shared, graph, restore, cached: vec![], decisions: 0 })
    }

    fn publish(&self, job: &Job, tokens: &[i32], revision: Fence, phase: Capture) -> Result<()> {
        if let Some(publisher) = &self.publisher { publisher.capture(&self.rk, &job.session, job.trace.session_label(), tokens, job.spans(), revision, phase)?; }
        Ok(())
    }

    fn job(&mut self, job: &Job) -> Result<()> {
        let toks = &job.tokens[..];
        let rank0 = self.r == 0;
        if rank0 {
            self.shared.restore_ok.store(true, Ordering::Release);
        }
        self.shared.barrier.wait();
        let candidate = self.shared.candidate.lock().unwrap().clone();
        let local = candidate.as_ref().is_some_and(|c| c.is_local());
        let snapshot = candidate.as_ref().and_then(|c| c.snapshot.as_ref());
        let mut revision = candidate.as_ref().map(|c| c.revision);
        if let Some(snapshot) = snapshot
            && kvstore::validate_snapshot(&self.rk, snapshot).is_err() {
            self.shared.restore_ok.store(false, Ordering::Release);
        }
        self.shared.barrier.wait();
        let mut common = 0;
        if self.shared.restore_ok.load(Ordering::Acquire)
            && let Some(snapshot) = snapshot {
            common = snapshot.manifest.reusable_prefix(job.prompt());
            if common > 0 { self.restore.import_prefix(&self.rk, snapshot, common)?; }
        }
        self.shared.barrier.wait();
        if rank0 && common > 0 && candidate.as_ref().is_some_and(Candidate::needs_onboard) {
            match self.shared.runtime.block_on(self.shared.peer().onboard_named(&job.session, candidate.as_ref().unwrap(), job.trace.session_label())) {
                Ok(next) => revision = Some(next),
                Err(e) if is_rdma_required(&e) => return Err(e),
                Err(_) => { revision = None; }
            }
        }
        drop(candidate);
        if rank0 { *self.shared.candidate.lock().unwrap() = None; }
        self.shared.barrier.wait();
        let revision = if let Some(publisher) = &mut self.publisher {
            publisher.fence(&job.session, revision, common > 0 && local)
        } else {
            Arc::new(Mutex::new(None))
        };
        if rank0 { job.trace.fetched(); job.trace.admitted(); }
        self.rk.stage_images(job.images.as_ref().map_or(&[], |v| &v.items), toks, common)?;
        self.rk.set_sampling(job.seed)?;
        let tok = prefill_from(&mut self.rk, self.chunk, toks, common)?;
        if rank0 { job.trace.prefilled(common); }
        self.publish(job, toks, revision.clone(), Capture::Prefill)?;
        self.shared.barrier.wait();

        let mut emitted: Vec<i32> = vec![tok];
        let mut finish: Option<Finish> = None;
        let mut alive = true;
        if rank0 {
            alive &= job.events.send(Event::Started { prompt_tokens: toks.len(), cached_tokens: common }).is_ok();
            alive &= job.events.send(Event::Tokens(vec![tok])).is_ok();
        }
        if job.stop_ids.contains(&tok) {
            finish = Some(Finish::Stop);
        } else if job.max_new <= 1 {
            finish = Some(Finish::Length);
        }

        self.rk.spec_begin(tok, toks.len())?;
        let (mut launched, mut prev) = (0usize, 0i32);
        let t1 = Instant::now();
        loop {
            let want = rank0 && finish.is_none() && alive && !job.cancel.load(Ordering::Relaxed) && launched <= 4 * job.max_new + 8;
            let go = {
                let slot = self.decisions % 2;
                self.decisions += 1;
                if rank0 {
                    self.shared.go[slot].store(want, Ordering::Release);
                }
                self.shared.barrier.wait();
                self.shared.go[slot].load(Ordering::Acquire)
            };
            if !go {
                break;
            }
            self.graph.launch(&self.rk.stream)?;
            self.rk.spec_snapshot(launched % 2)?;
            launched += 1;
            if launched < 2 {
                continue;
            }
            let (_, _, committed) = self.rk.spec_snapshot_read(launched % 2)?;
            if committed > prev && rank0 && finish.is_none() {
                let all = self.rk.read_out_nosync(committed as usize)?;
                let mut new: Vec<i32> = vec![];
                for &t in &all[prev as usize..] {
                    new.push(t);
                    if job.stop_ids.contains(&t) {
                        finish = Some(Finish::Stop);
                        break;
                    }
                    if emitted.len() + new.len() >= job.max_new {
                        finish = Some(Finish::Length);
                        break;
                    }
                }
                emitted.extend_from_slice(&new);
                alive &= job.events.send(Event::Tokens(new)).is_ok();
            }
            prev = committed;
        }
        let finish = finish.unwrap_or(Finish::Cancelled);
        self.rk.stream.synchronize()?;
        let decode_elapsed = t1.elapsed();

        if rank0 {
            self.shared.emitted.store(emitted.len(), Ordering::Release);
        }
        self.shared.barrier.wait();
        let n = self.shared.emitted.load(Ordering::Acquire);
        self.shared.barrier.wait();
        let (steps, accepted, _) = self.rk.spec_stats()?;
        if rank0 && let (Ok(iterations), Ok(accepted_drafts)) = (usize::try_from(steps), usize::try_from(accepted)) {
            job.trace.decoded(Decode { elapsed: decode_elapsed, iterations, accepted_drafts });
        }
        if rank0 {
            job.done(finish, emitted.len());
        }
        self.complete(job, toks, tok, n, revision)
    }

    fn complete(&mut self, job: &Job, toks: &[i32], tok: i32, n: usize, revision: Fence) -> Result<()> {
        let (cache, _) = self.rk.finish_decode_cache_with_stats(toks, tok, n, self.chunk)?;
        self.cached = cache;
        self.publish(job, &self.cached, revision, Capture::Complete(n))?;
        self.shared.barrier.wait();
        self.cached.clear();
        Ok(())
    }
}

fn prefill_from(rk: &mut Rank, chunk: usize, toks: &[i32], from: usize) -> Result<i32> {
    rk.prefill(toks, from, chunk)
}
