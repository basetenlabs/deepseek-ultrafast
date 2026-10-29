use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub moe_inter_dim: usize,
    pub n_layers: usize,
    pub n_mtp_layers: usize,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: u32,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: usize,
    pub dspark_n_routed_experts: usize,
    pub dspark_n_activated_experts: usize,
    pub n_heads: usize,
    pub n_routed_experts: usize,
    pub n_activated_experts: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub q_lora_rank: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub norm_eps: f32,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub original_seq_len: usize,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_source_layer: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub engram_layer_ids: Vec<usize>,
    pub engram_num_embeddings: Vec<usize>,
    pub engram_vocab_size: u64,
    pub engram_compressed_vocab_size: usize,
    pub engram_pad_id: usize,
    pub engram_max_ngram_size: usize,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    pub compress_rope_theta: f32,
    pub compress_ratios: Vec<usize>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }

    pub fn is_kv_source(&self, l: usize) -> bool {
        l < self.n_layers && self.kv_source_layers.contains(&l)
    }
    pub fn is_index_source(&self, l: usize) -> bool {
        l < self.n_layers && self.index_source_layers.contains(&l)
    }
    pub fn kv_source_of(&self, l: usize) -> Option<usize> {
        if self.compress_ratios[l] == 0 {
            return None;
        }
        self.kv_source_layers.iter().copied().filter(|&s| s <= l).max()
    }
    pub fn engram_index(&self, l: usize) -> Option<usize> {
        self.engram_layer_ids.iter().position(|&x| x == l)
    }
}

pub fn rope_table(dim: usize, seqlen: usize, original_seq_len: usize, base: f32, factor: f32, beta_fast: f32, beta_slow: f32) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut freqs: Vec<f32> = (0..half).map(|i| 1.0 / base.powf((2 * i) as f32 / dim as f32)).collect();
    if original_seq_len > 0 {
        let corrected = |rot: f32| -> f64 {
            dim as f64 * ((original_seq_len as f64) / (rot as f64 * 2.0 * std::f64::consts::PI)).ln() / (2.0 * (base as f64).ln())
        };
        let low = (corrected(beta_fast).floor()).max(0.0) as f32;
        let high = (corrected(beta_slow).ceil()).min((dim - 1) as f64) as f32;
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }
    let mut c = vec![0f32; seqlen * half];
    let mut s = vec![0f32; seqlen * half];
    for p in 0..seqlen {
        for i in 0..half {
            let a = p as f32 * freqs[i];
            c[p * half + i] = a.cos();
            s[p * half + i] = a.sin();
        }
    }
    (c, s)
}
