use anyhow::{Result, ensure};
use serde::Deserialize;
use std::{path::Path, sync::Arc};

pub const IMAGE_TOKEN: i32 = 129264;
pub const MAX_PATCHES: usize = 9 * 1024;

#[derive(Clone, Debug, Deserialize)]
pub struct VisionConfig {
    #[serde(default)]
    pub vision_n_layers: usize,
    pub vision_dim: usize,
    pub vision_n_heads: usize,
    pub vision_inter_dim: usize,
    pub vision_patch_size: usize,
    pub vision_downsample_ratio: usize,
    pub vision_rope_theta: f32,
    pub image_token_id: i32,
}
impl VisionConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let c: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(c.vision_n_layers > 0 && c.vision_dim == 1024 && c.vision_n_heads == 16
            && c.vision_inter_dim == 2816 && c.vision_patch_size == 14 && c.vision_downsample_ratio == 3
            && c.image_token_id == IMAGE_TOKEN && c.vision_rope_theta == 10000.0,
            "unsupported vision geometry (requires the V4.1 vision tower)");
        Ok(c)
    }
}

#[derive(Clone, Debug)]
pub struct Image {
    pub start: usize,
    pub height: usize,
    pub width: usize,
    pub digest: [u8; 32],
    pub patches: Arc<[u16]>,
}
impl Image {
    pub fn len(&self) -> usize { self.height.div_ceil(3) * (self.width.div_ceil(3) + 1) + 2 }
    pub fn end(&self) -> usize { self.start + self.len() }
    pub fn validate(&self, tokens: &[i32]) -> Result<()> {
        let n = self.height.checked_mul(self.width).unwrap_or(usize::MAX);
        ensure!(n > 0 && n <= MAX_PATCHES && self.patches.len() == n * 3 * 14 * 14, "invalid image patches");
        ensure!(self.height.div_ceil(3) * self.width.div_ceil(3) <= 1024, "image aligner grid exceeds capacity");
        ensure!(self.start <= tokens.len() && self.len() <= tokens.len() - self.start, "invalid image span");
        ensure!(tokens[self.start..self.end()].iter().all(|&t| t == IMAGE_TOKEN), "image span/token mismatch");
        Ok(())
    }
}
