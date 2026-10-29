use std::sync::Arc;
use tokenizers::Tokenizer;

pub struct Detok {
    tok: Arc<Tokenizer>,
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl Detok {
    pub fn new(tok: Arc<Tokenizer>) -> Self {
        Detok { tok, ids: vec![], prefix: 0, read: 0 }
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids, true).unwrap_or_default()
    }

    pub fn push(&mut self, ids: &[i32]) -> String {
        self.ids.extend(ids.iter().map(|&t| t as u32));
        self.emit(false)
    }

    pub fn finish(&mut self) -> String {
        self.emit(true)
    }

    fn emit(&mut self, force: bool) -> String {
        if self.read == self.ids.len() {
            return String::new();
        }
        let prev = self.decode(&self.ids[self.prefix..self.read]);
        let full = self.decode(&self.ids[self.prefix..]);
        if full.len() > prev.len() && (force || !full.ends_with('\u{FFFD}')) {
            let new = if full.starts_with(&prev) { full[prev.len()..].to_string() } else { full.get(prev.len()..).unwrap_or("").to_string() };
            self.prefix = self.read;
            self.read = self.ids.len();
            return new;
        }
        String::new()
    }
}

fn partial_suffix(s: &str, pat: &str) -> usize {
    let max = s.len().min(pat.len().saturating_sub(1));
    (1..=max).rev().find(|&k| s.is_char_boundary(s.len() - k) && pat.starts_with(&s[s.len() - k..])).unwrap_or(0)
}

pub struct StopMatcher {
    stops: Vec<String>,
    buf: String,
    pub hit: bool,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        StopMatcher { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), buf: String::new(), hit: false }
    }

    pub fn push(&mut self, text: &str) -> String {
        if self.hit {
            return String::new();
        }
        if self.stops.is_empty() {
            return text.to_string();
        }
        self.buf.push_str(text);
        if let Some(p) = self.stops.iter().filter_map(|s| self.buf.find(s.as_str())).min() {
            self.hit = true;
            let out = self.buf[..p].to_string();
            self.buf.clear();
            return out;
        }
        let hold = self.stops.iter().map(|s| partial_suffix(&self.buf, s)).max().unwrap_or(0);
        let out = self.buf[..self.buf.len() - hold].to_string();
        self.buf.drain(..self.buf.len() - hold);
        out
    }

    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }
}
