use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub file: usize,
    pub start: usize,
    pub end: usize,
}

pub struct Checkpoint {
    pub dir: PathBuf,
    maps: Vec<Mmap>,
    pub tensors: HashMap<String, TensorInfo>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        files.sort();
        Self::open_files(dir, files)
    }

    fn open_files(dir: &Path, files: Vec<PathBuf>) -> Result<Self> {
        let mut maps = Vec::new();
        let mut tensors = HashMap::new();
        for (fi, f) in files.iter().enumerate() {
            let file = File::open(f).with_context(|| format!("open {}", f.display()))?;
            let map = unsafe { Mmap::map(&file)? };
            let n = u64::from_le_bytes(map[0..8].try_into().unwrap()) as usize;
            let hdr: Value = serde_json::from_slice(&map[8..8 + n])?;
            for (name, v) in hdr.as_object().unwrap() {
                if name == "__metadata__" {
                    continue;
                }
                let off = v["data_offsets"].as_array().unwrap();
                tensors.insert(
                    name.clone(),
                    TensorInfo {
                        dtype: v["dtype"].as_str().unwrap().to_string(),
                        shape: v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(),
                        file: fi,
                        start: 8 + n + off[0].as_u64().unwrap() as usize,
                        end: 8 + n + off[1].as_u64().unwrap() as usize,
                    },
                );
            }
            maps.push(map);
        }
        Ok(Checkpoint { dir: dir.to_path_buf(), maps, tensors })
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("tensor {name} not in checkpoint"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.info(name)?;
        Ok(&self.maps[t.file][t.start..t.end])
    }

    pub fn prefault(&self, name: &str, range: std::ops::Range<usize>) -> Result<()> {
        let t = self.info(name)?;
        anyhow::ensure!(range.start <= range.end && range.end <= t.end - t.start, "invalid prefault range for {name}");
        if range.is_empty() { return Ok(()); }
        #[cfg(target_os = "linux")]
        if self.maps[t.file].advise_range(memmap2::Advice::PopulateRead, t.start + range.start, range.len()).is_ok() {
            return Ok(());
        }
        let bytes = &self.maps[t.file][t.start + range.start..t.start + range.end];
        for b in bytes.iter().step_by(4096) { std::hint::black_box(*b); }
        if let Some(b) = bytes.last() { std::hint::black_box(*b); }
        Ok(())
    }

    pub fn get(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<&[u8]> {
        let t = self.info(name)?;
        if t.dtype != dtype || t.shape != shape {
            bail!("{name}: expected {dtype}{shape:?}, found {}{:?}", t.dtype, t.shape);
        }
        self.bytes(name)
    }

    pub fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.info(name)?;
        let b = self.bytes(name)?;
        Ok(match t.dtype.as_str() {
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "BF16" => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            d => bail!("{name}: cannot read {d} as f32"),
        })
    }
}

pub fn require_checkpoint(dir: &Path) -> Result<()> {
    let shown = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    let get = "download DeepSeek-V4.1-Flash (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) into it, or pass the model's directory with --ckpt";
    if !dir.is_dir() {
        bail!("model directory {} not found: {get}", shown.display());
    }
    let weights = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).any(|e| e.path().extension().is_some_and(|x| x == "safetensors"));
    let missing: Vec<&str> = [("inference/config.json", dir.join("inference/config.json").is_file()), ("tokenizer.json", dir.join("tokenizer.json").is_file()), ("the weights (*.safetensors)", weights)]
        .into_iter()
        .filter_map(|(name, present)| (!present).then_some(name))
        .collect();
    if let Some((last, rest)) = missing.split_last() {
        let list = if rest.is_empty() { last.to_string() } else { format!("{} and {last}", rest.join(", ")) };
        bail!("model directory {} is missing {list}: {get}", shown.display());
    }
    Ok(())
}
