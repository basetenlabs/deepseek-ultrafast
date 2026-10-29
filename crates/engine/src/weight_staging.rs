use anyhow::Result;
use deepseek_ultrafast_cuda::{self as cu, DevPtr, PinnedHost, Stream};
use crate::weights::Arena;

enum Host {
    Pinned(PinnedHost),
    Pageable(Vec<u8>),
}
impl Host {
    fn pinned(bytes: usize) -> Result<Self> {
        let p = PinnedHost::alloc(bytes)?;
        unsafe { std::ptr::write_bytes(p.ptr, 0, bytes) };
        Ok(Self::Pinned(p))
    }
    fn bytes(&mut self) -> &mut [u8] {
        match self { Self::Pinned(p) => p.as_mut_slice(), Self::Pageable(v) => v }
    }
}

pub(crate) struct ExpertStaging {
    a: Host,
    b: Host,
    stream: Option<Stream>,
}
impl ExpertStaging {
    pub fn new(a: usize, b: usize) -> Result<Self> {
        Self::try_pinned(a, b, Host::pinned)
    }

    fn try_pinned(a: usize, b: usize, mut allocate: impl FnMut(usize) -> Result<Host>) -> Result<Self> {
        let mut pinned = || -> Result<Self> {
            Ok(Self { a: allocate(a)?, b: allocate(b)?, stream: Some(Stream::new()?) })
        };
        match pinned() {
            Ok(p) => Ok(p),
            Err(e) => {
                deepseek_ultrafast_logging::diagnostic!("weight loading: pinned staging unavailable ({e}); using reusable pageable buffers");
                Self::pageable(a, b)
            }
        }
    }

    fn pageable(a: usize, b: usize) -> Result<Self> {
        let mut x = Vec::new();
        let mut y = Vec::new();
        x.try_reserve_exact(a)?;
        y.try_reserve_exact(b)?;
        x.resize(a, 0);
        y.resize(b, 0);
        Ok(Self { a: Host::Pageable(x), b: Host::Pageable(y), stream: None })
    }

    pub fn finish(&self) -> Result<()> {
        if let Some(s) = &self.stream { s.synchronize()?; }
        Ok(())
    }

    pub fn buffers(&mut self, a: usize, b: usize) -> Result<(&mut [u8], &mut [u8])> {
        self.finish()?;
        Ok((&mut self.a.bytes()[..a], &mut self.b.bytes()[..b]))
    }

    pub fn upload(&mut self, arena: &mut Arena, a: usize, b: usize) -> Result<(DevPtr, DevPtr)> {
        if let Some(s) = &self.stream {
            let x = arena.alloc(a)?;
            let y = arena.alloc(b)?;
            cu::memcpy_htod_async(x, &self.a.bytes()[..a], s)?;
            cu::memcpy_htod_async(y, &self.b.bytes()[..b], s)?;
            Ok((x, y))
        } else {
            Ok((arena.put(&self.a.bytes()[..a])?, arena.put(&self.b.bytes()[..b])?))
        }
    }
}

impl Drop for ExpertStaging {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}
