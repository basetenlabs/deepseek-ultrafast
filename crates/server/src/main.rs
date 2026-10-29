use deepseek_ultrafast_server::{api, chat};
mod engine;
mod host_budget;
mod peer;
mod publisher;

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;

const USAGE: &str = "serve [--ckpt DIR] [--gpus 0,1,..] [--max-seq N] [--chunk N] [--host H] [--port P] [--model-name NAME]";

fn main() -> std::process::ExitCode {
    std::panic::set_hook(Box::new(|info| deepseek_ultrafast_logging::termination(&info.to_string())));
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            deepseek_ultrafast_logging::termination(&format!("{error:#}"));
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut ea = engine::EngineArgs {
        ckpt: PathBuf::from("/models"),
        gpus: (0..8).collect(),
        max_seq: 1048576,
        chunk: 8192,
    };
    let (mut host, mut port, mut model) = ("0.0.0.0".to_string(), 8000u16, "deepseek-v4.1-flash".to_string());
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < v.len() {
        let val = || v.get(i + 1).cloned().with_context(|| format!("{} needs a value\nusage: {USAGE}", v[i]));
        match v[i].as_str() {
            "--ckpt" => ea.ckpt = val()?.into(),
            "--gpus" => ea.gpus = val()?.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?,
            "--max-seq" => ea.max_seq = val()?.parse()?,
            "--chunk" => ea.chunk = val()?.parse()?,
            "--host" => host = val()?,
            "--port" => port = val()?.parse()?,
            "--model-name" => model = val()?,
            "-h" | "--help" => {
                println!("usage: {USAGE}");
                return Ok(());
            }
            x => anyhow::bail!("unknown arg {x}\nusage: {USAGE}"),
        }
        i += 2;
    }
    anyhow::ensure!(!ea.gpus.is_empty() && ea.gpus.iter().enumerate().all(|(i, gpu)| !ea.gpus[..i].contains(gpu)), "--gpus must name distinct GPUs");
    deepseek_ultrafast_model::st::require_checkpoint(&ea.ckpt)?;
    deepseek_ultrafast_cuda::init()?;
    deepseek_ultrafast_cuda::profile::init(deepseek_ultrafast_cuda::device_cc(ea.gpus[0] as i32)?, ea.gpus.len());
    unsafe {
        std::env::set_var("CUDNN_LOGLEVEL_DBG", "0");
        std::env::set_var("OPENCV_LOG_LEVEL", "SILENT");
    }
    let _logging = deepseek_ultrafast_logging::init()?;
    let rt = tokio::runtime::Runtime::new()?;
    let peer_options = peer::Options::new(&ea)?;
    let tok = Arc::new(tokenizers::Tokenizer::from_file(ea.ckpt.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?);
    deepseek_ultrafast_server::token_cache::register(tok.clone());
    let peer_options = peer_options.with_token_text(Arc::new(deepseek_ultrafast_server::token_text::token_text(&tok)));
    let eos = tok.token_to_id(chat::EOS).context("tokenizer has no end-of-sentence token")? as i32;
    let max_seq = ea.max_seq;

    let (tx, rx) = std::sync::mpsc::channel();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let runtime = rt.handle().clone();
    let eng = std::thread::spawn(move || engine::run(ea, rx, ready_tx, peer_options, runtime));
    let peer = match ready_rx.recv() {
        Ok(peer) => peer,
        Err(_) => return eng.join().map_err(|_| anyhow::anyhow!("engine thread panicked"))?,
    };

    let app = Arc::new(api::App { tok, jobs: tx, model, max_seq, eos });
    rt.block_on(async move {
        let ready = deepseek_ultrafast_server::readiness::Readiness::default();
        let result: Result<()> = async {
            let addr = format!("{host}:{port}");
            let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("bind {addr}"))?;
            ready.set_ready();
            let shutdown_ready = ready.clone();
            let result = axum::serve(listener, api::router_with_readiness(app, ready.clone())).with_graceful_shutdown(async move {
                let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
                shutdown_ready.stop();
            }).await;
            result?;
            Ok(())
        }.await;
        ready.stop();
        let engine_result = tokio::task::spawn_blocking(move || {
            eng.join().map_err(|_| anyhow::anyhow!("engine thread panicked"))?
        }).await;
        peer.shutdown().await;
        result?;
        engine_result??;
        Ok(())
    })
}
