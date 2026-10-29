use anyhow::Result;
use deepseek_ultrafast_kv_transfer::{Config, DEFAULT_DIRECTORY_PREFIX, DEFAULT_REDIS_URL, PeerCache, digest};
use std::time::Duration;
use crate::host_budget::{self, Settings, env};

pub struct Options {
    config: Config,
    memory: Settings,
    profile: String,
}

fn fingerprint(profile: &str, woa64: &str) -> String {
    digest(format!("{profile}\0woa64={woa64}").as_bytes())
}

fn decode_numerics() -> String {
    format!("{};tg2sm={}", deepseek_ultrafast_engine::ops::woa64_resolved(), deepseek_ultrafast_engine::ops::tg2_2sm_resolved())
}

impl Options {
    pub fn new(engine: &crate::engine::EngineArgs) -> Result<Self> {
        let bind = deepseek_ultrafast_kv_transfer::kv_address()?;
        let tg2 = deepseek_ultrafast_cuda::profile::sm103();
        let profile = format!(
            "deepseek-ultrafast-kv-snapshot-v3-image-v2\0tp={}\0max_seq={}\0config={}\0tokenizer={}\0tcgemv2={}",
            engine.gpus.len(), engine.max_seq,
            digest(&std::fs::read(engine.ckpt.join("inference/config.json"))?),
            digest(&std::fs::read(engine.ckpt.join("tokenizer.json"))?),
            tg2 as u8,
        );
        Ok(Self {
            profile,
            config: Config {
                redis_url: env("DSUF_REDIS_URL")?.unwrap_or_else(|| DEFAULT_REDIS_URL.into()),
                directory_prefix: DEFAULT_DIRECTORY_PREFIX.into(),
                fingerprint: String::new(),
                bind,
                max_bytes: 0,
                max_snapshot_bytes: 0,
                rdma_bytes: 0,
                max_seq: engine.max_seq,
                timeout: Duration::from_secs(30),
                token_text: None,
            },
            memory: Settings,
        })
    }

    pub fn with_token_text(mut self, text: std::sync::Arc<deepseek_ultrafast_kv_transfer::TokenText>) -> Self {
        self.config.token_text = Some(text);
        self
    }

    fn fingerprint(&self) -> String {
        fingerprint(&self.profile, &decode_numerics())
    }

    pub async fn start(&self, raw_image_max: usize) -> Result<PeerCache> {
        let available = host_budget::sample()?;
        let budget = self.memory.resolve(available.effective, raw_image_max)?;
        deepseek_ultrafast_logging::diagnostic!("KV host budget after ALL-rank CUDA graph warmup: host_available={} effective_available={} fraction={} gross={} payload_lru={} max_snapshot={} registered_limit={} heap_slack={} fp4={} cgroup_limits={:?}",
            available.host, available.effective, Settings::FRACTION, budget.gross, budget.cache,
            budget.snapshot, budget.registered, budget.heap_slack, true, available.limits);
        let mut config = self.config.clone();
        config.fingerprint = self.fingerprint();
        deepseek_ultrafast_logging::diagnostic!("KV snapshot profile: decode numerics {} (fingerprint {})", decode_numerics(), config.fingerprint);
        config.max_bytes = budget.cache;
        config.max_snapshot_bytes = budget.snapshot;
        config.rdma_bytes = budget.registered;
        PeerCache::start(config).await
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fingerprint_carries_the_resolved_wo_a_tiling() {
        let p = "deepseek-ultrafast-kv-snapshot-v3-image-v2\0ns\0tp=4";
        assert_ne!(super::fingerprint(p, "2048x4096/8:1"), super::fingerprint(p, "2048x4096/8:0"));
        assert_ne!(super::fingerprint(p, "2048x4096/8:1"), super::fingerprint(p, "-"));
        assert_eq!(super::fingerprint(p, "2048x4096/8:1"), super::fingerprint(p, "2048x4096/8:1"));
    }
}
