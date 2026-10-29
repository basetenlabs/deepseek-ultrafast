mod decode_cache;
pub mod engram;
pub mod fp4;
pub mod kvstore;
pub mod model;
pub mod ops;
pub mod sampling;
pub mod weights;
pub mod vision;
mod vision_attention;
mod weight_staging;

pub fn load_config(ckpt: &std::path::Path) -> anyhow::Result<deepseek_ultrafast_model::config::Config> {
    let mut cfg = deepseek_ultrafast_model::config::Config::load(&ckpt.join("inference/config.json"))?;
    if let Some(b) = deepseek_ultrafast_cuda::profile::dspark_block() {
        cfg.dspark_block_size = b;
    }
    Ok(cfg)
}
