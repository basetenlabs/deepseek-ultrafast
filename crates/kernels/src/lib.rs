use anyhow::{Context, Result};
use deepseek_ultrafast_cuda::{Function, Module};
use std::collections::HashMap;

include!(concat!(env!("OUT_DIR"), "/cubins.rs"));

pub struct Kernels {
    _modules: Vec<Module>,
    variants: bool,
    funcs: HashMap<String, Function>,
}

impl Kernels {
    pub fn load() -> Result<Self> {
        let mut k = Kernels { _modules: Vec::new(), variants: false, funcs: HashMap::new() };
        k.load_modules(false)?;
        Ok(k)
    }

    pub fn load_variants(&mut self) -> Result<()> {
        if !self.variants {
            self.variants = true;
            self.load_modules(true)?;
        }
        Ok(())
    }

    fn load_modules(&mut self, variants: bool) -> Result<()> {
        for (name, image) in CUBINS {
            if name.starts_with("variant_") != variants {
                continue;
            }
            let m = Module::load(image).with_context(|| format!("loading module {name}"))?;
            for f in function_names(image) {
                let func = m.function(&f)?;
                let _ = func.set_non_portable_cluster();
                let key = if variants { format!("{name}/{f}") } else { f.clone() };
                if self.funcs.insert(key, func).is_some() {
                    anyhow::bail!("duplicate kernel name {f} (module {name})");
                }
            }
            self._modules.push(m);
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<&Function> {
        self.funcs.get(name).with_context(|| format!("no kernel named {name}"))
    }
}

fn function_names(image: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let rd16 = |o: usize| u16::from_le_bytes([image[o], image[o + 1]]) as usize;
    let rd32 = |o: usize| u32::from_le_bytes(image[o..o + 4].try_into().unwrap()) as usize;
    let rd64 = |o: usize| u64::from_le_bytes(image[o..o + 8].try_into().unwrap()) as usize;
    if image.len() < 64 || &image[0..4] != b"\x7fELF" {
        return out;
    }
    let shoff = rd64(0x28);
    let shentsize = rd16(0x3a);
    let shnum = rd16(0x3c);
    let shstrndx = rd16(0x3e);
    let sh = |i: usize| shoff + i * shentsize;
    let strtab_off = rd64(sh(shstrndx) + 0x18);
    for i in 0..shnum {
        let name_off = rd32(sh(i));
        let s = &image[strtab_off + name_off..];
        let end = s.iter().position(|&b| b == 0).unwrap_or(0);
        let name = std::str::from_utf8(&s[..end]).unwrap_or("");
        if let Some(f) = name.strip_prefix(".text.") {
            out.push(f.to_string());
        }
    }
    out
}
