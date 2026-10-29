use anyhow::{Result, anyhow, ensure};
use deepseek_ultrafast_model::vision::{Image, IMAGE_TOKEN, MAX_PATCHES};
use deepseek_recipe_core::multimodal::{ImageInfo, ImageSource, IMAGE_SPECIAL_TOKEN};
use deepseek_recipe_image::{ImageResolver, ImageLimits, ImageQuota, ImageError, ImageFetcher,
    ImagePreprocessor, ImageByteBudget, PreprocessOptions, OpenCvImagePreprocessor, ReqwestImageFetcher};
use opencv::{core::Vector, imgcodecs, prelude::*};
use sha2::{Digest, Sha256};
use std::{collections::VecDeque, net::{IpAddr, SocketAddr}, sync::{Arc, Mutex, OnceLock}, time::Duration};
use tokio::sync::Semaphore;
pub use crate::image_identity::Images;
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct Busy;
impl std::fmt::Display for Busy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("image request capacity exhausted; retry later") }
}
impl std::error::Error for Busy {}

struct PublicDns;
impl reqwest::dns::Resolve for PublicDns {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        Box::pin(async move {
            let addresses: Vec<SocketAddr> = tokio::net::lookup_host((name.as_str(), 0)).await?.collect();
            if addresses.is_empty() || addresses.iter().any(|a| !public_ip(a.ip())) {
                return Err(std::io::Error::other("image URL resolves to a non-public address").into());
            }
            Ok(Box::new(addresses.into_iter()) as reqwest::dns::Addrs)
        })
    }
}
fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => {
            let [a,b,c,_] = v.octets();
            !(a == 0 || a == 10 || a == 127 || a >= 224 || (a == 100 && (64..128).contains(&b))
                || (a == 169 && b == 254) || (a == 172 && (16..32).contains(&b))
                || (a == 192 && (b == 168 || (b == 0 && (c == 0 || c == 2)) || (b == 88 && c == 99)))
                || (a == 198 && (b == 18 || b == 19 || (b == 51 && c == 100))) || (a == 203 && b == 0 && c == 113))
        }
        IpAddr::V6(v) => {
            if let Some(v4) = v.to_ipv4_mapped() { return public_ip(v4.into()); }
            let s = v.segments();
            (s[0] & 0xe000) == 0x2000 && s[0] != 0x2002 && s[0] != 0x3fff
                && !(s[0] == 0x2001 && (s[1] < 0x200 || s[1] == 0xdb8))
        }
    }
}
fn public_url(url: &reqwest::Url) -> bool {
    matches!(url.scheme(), "http" | "https") && url.username().is_empty() && url.password().is_none()
        && matches!(url.port_or_known_default(), Some(80 | 443))
        && match url.host() {
            Some(url::Host::Ipv4(ip)) => public_ip(ip.into()),
            Some(url::Host::Ipv6(ip)) => public_ip(ip.into()),
            Some(url::Host::Domain(_)) => true,
            None => false,
        }
}
struct Fetcher(ReqwestImageFetcher);
impl ImageFetcher for Fetcher {
    async fn fetch(&self, url: &str, budget: &ImageByteBudget) -> std::result::Result<Vec<u8>, ImageError> {
        let parsed = reqwest::Url::parse(url).map_err(|e| ImageError::Client(e.to_string()))?;
        if !public_url(&parsed) { return Err(ImageError::Unsupported("image URLs require public HTTP(S) destinations on ports 80/443")); }
        self.0.fetch(url, budget).await
    }
}
#[derive(Clone)]
struct Preprocessor {
    slots: Arc<Semaphore>,
    cache: Arc<Mutex<(usize, VecDeque<([u8;32], ImageInfo)>)>>,
}
impl ImagePreprocessor for Preprocessor {
    async fn preprocess(&self, data: Vec<u8>, options: PreprocessOptions) -> std::result::Result<ImageInfo, ImageError> {
        let permit = self.slots.clone().acquire_owned().await.map_err(|e| ImageError::Client(e.to_string()))?;
        let cache = self.cache.clone();
        tokio::spawn(async move {
            let _permit = permit;
            let (key, data) = tokio::task::spawn_blocking(move || {
                let mut h = Sha256::new(); h.update(&data);
                h.update([options.detail.is_low() as u8]);
                h.update(options.max_dimension_px.to_le_bytes()); h.update(options.low_detail_max_dimension_px.to_le_bytes());
                (<[u8;32]>::from(h.finalize()), data)
            }).await.map_err(|e| ImageError::Decode(e.to_string()))?;
            {
                let mut c = cache.lock().unwrap();
                if let Some(at) = c.1.iter().position(|(k,_)| k == &key) {
                    let entry = c.1.remove(at).unwrap(); let result = entry.1.clone(); c.1.push_back(entry); return Ok(result);
                }
            }
            let image = OpenCvImagePreprocessor.preprocess(data, options).await?;
            const BUDGET: usize = 32 << 20;
            let mut c = cache.lock().unwrap();
            while (c.0 + image.data.len() > BUDGET || c.1.len() >= 128) && !c.1.is_empty() {
                let (_, old) = c.1.pop_front().unwrap(); c.0 -= old.data.len();
            }
            if image.data.len() <= BUDGET { c.0 += image.data.len(); c.1.push_back((key, image.clone())); }
            Ok(image)
        }).await.map_err(|e| ImageError::Decode(e.to_string()))?
    }
}
struct Service { resolver: ImageResolver<Fetcher, Preprocessor>, jobs: Arc<Semaphore> }
impl Service {
    fn new() -> Result<Self> {
        let client = reqwest::Client::builder().no_proxy().dns_resolver(Arc::new(PublicDns))
            .connect_timeout(Duration::from_secs(5)).timeout(Duration::from_secs(20))
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.previous().len() >= 3 || !public_url(attempt.url()) { attempt.error("image redirect is not allowed") }
                else { attempt.follow() }
            })).build()?;
        let limits = ImageLimits { max_images:16, max_concurrent_sources:2, ..Default::default() };
        let preprocessor = Preprocessor { slots:Arc::new(Semaphore::new(2)), cache:Arc::new(Mutex::new((0,VecDeque::new()))) };
        Ok(Self { resolver:ImageResolver::new(Fetcher(ReqwestImageFetcher::from_client(client, limits.max_image_bytes)), preprocessor).with_limits(limits),
            jobs:Arc::new(Semaphore::new(8)) })
    }
}
fn service() -> Result<&'static Service> {
    static S: OnceLock<std::result::Result<Service,String>> = OnceLock::new();
    S.get_or_init(|| Service::new().map_err(|e| e.to_string())).as_ref().map_err(|e| anyhow!("image service: {e}"))
}

pub async fn prepare(tok: Arc<Tokenizer>, prompt: String, literal_breaks: Vec<usize>, offsets: Vec<usize>, sources: Vec<ImageSource>, max_seq: usize)
    -> Result<(Vec<i32>, Option<Arc<Images>>)> {
    if sources.is_empty() {
        ensure!(offsets.is_empty(), "image source/placeholder mismatch");
        return Ok((tokio::task::spawn_blocking(move || crate::chat::tokenize_prompt(&tok, &prompt, &literal_breaks)).await??, None));
    }
    ensure!(sources.len() == offsets.len(), "image source/placeholder mismatch");
    let svc = service()?;
    let permit = svc.jobs.clone().try_acquire_owned().map_err(|_| Busy)?;
    let rt = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let data = rt.block_on(svc.resolver.resolve(&sources, &mut ImageQuota::new()))?;
        let (tokens, items) = expand(&tok, &prompt, &literal_breaks, &offsets, &data.images, max_seq)?;
        Ok((tokens, Some(Arc::new(Images::new(items, permit)))))
    }).await?
}

fn expand(tok: &Tokenizer, prompt: &str, literal_breaks: &[usize], offsets: &[usize], images: &[ImageInfo], max_seq: usize) -> Result<(Vec<i32>, Vec<Image>)> {
    ensure!(offsets.len() == images.len(), "image source/placeholder mismatch");
    let mut tokens = Vec::new(); let mut items = Vec::new(); let mut at = 0;
    for (&offset, info) in offsets.iter().zip(images) {
        ensure!(offset >= at && prompt.get(offset..).is_some_and(|s| s.starts_with(IMAGE_SPECIAL_TOKEN)), "invalid image marker offset");
        tokens.extend(crate::chat::tokenize_segment(tok, prompt, literal_breaks, at..offset)?);
        let (w,h) = (info.width as usize, info.height as usize);
        ensure!(w > 0 && h > 0 && w % 14 == 0 && h % 14 == 0 && (w/14)*(h/14) <= MAX_PATCHES
            && (h/14).div_ceil(3)*(w/14).div_ceil(3) <= 1024, "invalid recipe image grid");
        let len = (h/14).div_ceil(3) * ((w/14).div_ceil(3)+1) + 2;
        ensure!(tokens.len().checked_add(len).is_some_and(|n| n < max_seq), "expanded image prompt exceeds context capacity");
        let rgb = imgcodecs::imdecode(&Vector::from_slice(&info.data), imgcodecs::IMREAD_COLOR)?;
        ensure!(rgb.cols() == w as i32 && rgb.rows() == h as i32 && rgb.is_continuous(), "resolved image size mismatch");
        let pixels = rgb.data_bytes()?;
        let mut patches = Vec::with_capacity(w*h*3);
        let values: Vec<u16> = (0..256).map(|i| half::bf16::from_f32((i as f32 / 255. - 0.5) / 0.5).to_bits()).collect();
        for py in 0..h/14 { for px in 0..w/14 { for channel in 0..3 { for yy in 0..14 { for xx in 0..14 {
            patches.push(values[pixels[((py*14+yy)*w + px*14+xx)*3 + 2-channel] as usize]);
        }}}}}
        let mut digest = Sha256::new(); digest.update(&info.data); digest.update(info.width.to_le_bytes()); digest.update(info.height.to_le_bytes());
        items.push(Image { start:tokens.len(), width:w/14, height:h/14, digest:digest.finalize().into(), patches:patches.into() });
        tokens.extend(std::iter::repeat_n(IMAGE_TOKEN, len)); at = offset + IMAGE_SPECIAL_TOKEN.len();
    }
    tokens.extend(crate::chat::tokenize_segment(tok, prompt, literal_breaks, at..prompt.len())?);
    ensure!(tokens.len() < max_seq, "expanded image prompt exceeds context capacity");
    Ok((tokens,items))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn network_policy_rejects_local_special_and_mapped_addresses() {
        for ip in ["127.0.0.1","10.0.0.1","169.254.169.254","100.64.0.1","172.31.0.1","192.168.1.1","198.18.1.1","0.1.2.3","224.0.0.1","::1","::ffff:127.0.0.1","fe80::1","fc00::1","2001:db8::1","2002:a00:1::"] {
            assert!(!public_ip(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["8.8.8.8","1.1.1.1","2606:4700::1111"] { assert!(public_ip(ip.parse().unwrap())); }
        for url in ["file:///etc/passwd","http://127.1/a","http://2130706433/a","http://[::ffff:127.0.0.1]/","http://user:pass@example.com/a","https://example.com:8080/a"] {
            assert!(!public_url(&reqwest::Url::parse(url).unwrap()), "{url}");
        }
    }
}
