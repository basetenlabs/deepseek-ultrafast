use anyhow::{Context, Result, anyhow, ensure};
use deepseek_ultrafast_model::config::Config;
use crate::model::EngramTables;
use crate::weights::Arena;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;
use tokenizers::normalizers::replace::ReplacePattern;
use tokenizers::normalizers::{Lowercase, NFD, NFKC, Replace, Sequence, StripAccents, Strip};
use tokenizers::{NormalizedString, Normalizer, NormalizerWrapper, Tokenizer};

#[derive(Debug, PartialEq)]
pub struct EngramConsts {
    pub token_map: Vec<i32>,
    pub primes: Vec<i64>,
    pub offsets: Vec<i64>,
    pub multipliers: Vec<i64>,
    pub pad_id: i32,
}

impl EngramConsts {
    pub fn compute(dir: &Path, cfg: &Config) -> Result<Self> {
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let (token_map, n) = compressed_token_map(&tok)?;
        ensure!(n == cfg.engram_compressed_vocab_size, "Engram: {n} compressed tokens, the config expects {}", cfg.engram_compressed_vocab_size);
        let pad_id = *token_map.get(cfg.engram_pad_id).context("Engram pad id outside the vocabulary")?;
        let (primes, offsets) = bucket_primes(cfg.engram_layer_ids.len(), cfg.engram_max_ngram_size, cfg.engram_n_heads, cfg.engram_vocab_size);
        let multipliers = cfg.engram_layer_ids.iter().flat_map(|&l| hash_multipliers(l as u64, cfg.engram_max_ngram_size, n as u64)).collect();
        Ok(EngramConsts { token_map, primes, offsets, multipliers, pad_id })
    }

    pub fn upload(&self, a: &mut Arena) -> Result<EngramTables> {
        Ok(EngramTables { token_map: a.put(&self.token_map)?, mult: a.put(&self.multipliers)?, primes: a.put(&self.primes)?, offsets: a.put(&self.offsets)?, pad_id: self.pad_id })
    }
}

pub struct EngramCell<'a> {
    dir: &'a Path,
    cfg: &'a Config,
    cell: OnceLock<Result<EngramConsts>>,
}

impl<'a> EngramCell<'a> {
    pub fn new(dir: &'a Path, cfg: &'a Config) -> Self {
        EngramCell { dir, cfg, cell: OnceLock::new() }
    }

    pub fn get(&self) -> Result<&EngramConsts> {
        self.cell.get_or_init(|| EngramConsts::compute(self.dir, self.cfg)).as_ref().map_err(|e| anyhow!("Engram constants: {e:#}"))
    }
}

fn compressed_token_map(tok: &Tokenizer) -> Result<(Vec<i32>, usize)> {
    let sentinel = "\u{e000}";
    let normalizers: Vec<NormalizerWrapper> = vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        Replace::new(ReplacePattern::Regex(r"[ \t\r\n]+".into()), " ").map_err(|e| anyhow!("{e}"))?.into(),
        Replace::new(ReplacePattern::Regex(r"^ $".into()), sentinel).map_err(|e| anyhow!("{e}"))?.into(),
        Strip::new(true, true).into(),
        Replace::new(sentinel, " ").map_err(|e| anyhow!("{e}"))?.into(),
    ];
    let norm = Sequence::new(normalizers);
    let key = |id: u32| -> Result<String> {
        let text = tok.decode(&[id], false).map_err(|e| anyhow!("decode token {id}: {e}"))?;
        if text.contains('\u{fffd}') {
            return tok.id_to_token(id).with_context(|| format!("token {id} has no string"));
        }
        let mut n = NormalizedString::from(text.as_str());
        norm.normalize(&mut n).map_err(|e| anyhow!("normalize token {id}: {e}"))?;
        Ok(if n.get().is_empty() { text } else { n.get().to_owned() })
    };
    let keys: Vec<String> = (0..tok.get_vocab_size(true) as u32).map(key).collect::<Result<_>>()?;
    let mut ids: HashMap<&str, i32> = HashMap::new();
    let map = keys.iter().map(|k| { let next = ids.len() as i32; *ids.entry(k).or_insert(next) }).collect();
    Ok((map, ids.len()))
}

fn bucket_primes(layers: usize, max_ngram: usize, heads: usize, vocab: u64) -> (Vec<i64>, Vec<i64>) {
    let mut seen = std::collections::HashSet::new();
    let (mut primes, mut offsets) = (Vec::new(), Vec::new());
    for _ in 0..layers {
        let mut start = 0i64;
        for _ in 1..max_ngram {
            let mut p = vocab - 1;
            for _ in 0..heads {
                p += 1;
                while !is_prime(p) || seen.contains(&p) {
                    p += 1;
                }
                seen.insert(p);
                primes.push(p as i64);
                offsets.push(start);
                start += p as i64;
            }
        }
    }
    (primes, offsets)
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

fn hash_multipliers(layer: u64, max_ngram: usize, vocab: u64) -> Vec<i64> {
    let bound = ((i64::MAX as u64 / vocab) / 2).max(1);
    let mut rng = Pcg64::from_seed_sequence(10007 * layer);
    (0..max_ngram).map(|_| (rng.bounded(bound) as i64) * 2 + 1).collect()
}

struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    const MULT: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

    fn from_seed_sequence(seed: u64) -> Self {
        let w = seed_sequence_state(seed, 8);
        let u = |i: usize| (w[2 * i] as u64) | ((w[2 * i + 1] as u64) << 32);
        let (initstate, initseq) = (((u(0) as u128) << 64) | u(1) as u128, ((u(2) as u128) << 64) | u(3) as u128);
        let mut r = Pcg64 { state: 0, inc: (initseq << 1) | 1 };
        r.step();
        r.state = r.state.wrapping_add(initstate);
        r.step();
        r
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(Self::MULT).wrapping_add(self.inc);
    }

    fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        ((s >> 64) as u64 ^ s as u64).rotate_right((s >> 122) as u32)
    }

    fn bounded(&mut self, n: u64) -> u64 {
        let mut m = self.next_u64() as u128 * n as u128;
        if (m as u64) < n {
            let threshold = n.wrapping_neg() % n;
            while (m as u64) < threshold {
                m = self.next_u64() as u128 * n as u128;
            }
        }
        (m >> 64) as u64
    }
}

fn seed_sequence_state(seed: u64, words: usize) -> Vec<u32> {
    const INIT_A: u32 = 0x43b0d7e5;
    const MULT_A: u32 = 0x931e8875;
    const INIT_B: u32 = 0x8b51f9dd;
    const MULT_B: u32 = 0x58f38ded;
    const MIX_L: u32 = 0xca01f9dd;
    const MIX_R: u32 = 0x4973f715;
    assert!(seed <= u32::MAX as u64, "seed sequence: one 32-bit entropy word");
    let mut hash = INIT_A;
    let mut hashmix = |mut v: u32| {
        v ^= hash;
        hash = hash.wrapping_mul(MULT_A);
        v = v.wrapping_mul(hash);
        v ^ (v >> 16)
    };
    let mix = |x: u32, y: u32| {
        let r = MIX_L.wrapping_mul(x).wrapping_sub(MIX_R.wrapping_mul(y));
        r ^ (r >> 16)
    };
    let entropy = [seed as u32];
    let mut pool = [0u32; 4];
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(entropy.get(i).copied().unwrap_or(0));
    }
    for src in 0..4 {
        for dst in 0..4 {
            if src != dst {
                let h = hashmix(pool[src]);
                pool[dst] = mix(pool[dst], h);
            }
        }
    }
    let mut hash = INIT_B;
    (0..words)
        .map(|i| {
            let mut v = pool[i % 4] ^ hash;
            hash = hash.wrapping_mul(MULT_B);
            v = v.wrapping_mul(hash);
            v ^ (v >> 16)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_streams() {
        assert_eq!(seed_sequence_state(10007, 4), vec![2329349383, 2010327604, 3091877797, 2571032073]);
        assert_eq!(hash_multipliers(1, 4, 99092), vec![76632096046245, 4839876093313, 35959672319349, 73987337458391]);
        assert_eq!(hash_multipliers(14, 4, 99092), vec![67716810739261, 51510806800915, 30921347202721, 82619226485591]);
    }

    #[test]
    fn primes() {
        assert!(is_prime(2) && is_prime(16000057) && !is_prime(16000000) && !is_prime(1));
        let (p, o) = bucket_primes(1, 3, 2, 10);
        assert_eq!(p, vec![11, 13, 17, 19]);
        assert_eq!(o, vec![0, 11, 24, 41]);
        let (p, o) = bucket_primes(2, 4, 8, 16_000_000);
        assert_eq!((p.len(), &p[..3], &p[45..]), (48, &[16000057, 16000079, 16000081][..], &[16000877, 16000879, 16000889][..]));
        assert_eq!(&o[45..], &[336014037, 352014914, 368015793]);
    }

    #[test]
    fn checkpoint_constants() -> Result<()> {
        let Ok(dir) = std::env::var("DSUF_TEST_CKPT") else { return Ok(()) };
        let dir = Path::new(&dir);
        let cfg = Config::load(&dir.join("inference/config.json"))?;
        let c = EngramConsts::compute(dir, &cfg)?;
        let sum = c.token_map.iter().enumerate().fold(0u64, |a, (i, &v)| a.wrapping_add((i as u64 + 1) * v as u64));
        assert_eq!((c.token_map.len(), sum, c.pad_id), (129280, 464308076285175, 2));
        assert_eq!(c.multipliers[..4], [76632096046245, 4839876093313, 35959672319349, 73987337458391]);
        Ok(())
    }
}
