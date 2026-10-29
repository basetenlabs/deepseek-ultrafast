use anyhow::{Context, Result, ensure};
use std::path::{Component, Path, PathBuf};

const MIB: u64 = 1 << 20;
const FIXED_SLACK: u64 = 64 * MIB;

const CACHE_FRACTION: f64 = 0.75;

#[derive(Clone, Debug, Default)]
pub struct Settings;

#[derive(Debug)]
pub struct Available {
    pub host: u64,
    pub effective: u64,
    pub limits: Vec<(PathBuf, u64, u64)>,
}

#[derive(Debug)]
pub struct Budget {
    pub gross: u64,
    pub cache: usize,
    pub snapshot: usize,
    pub registered: u64,
    pub heap_slack: u64,
}

pub fn env(key: &str) -> Result<Option<String>> {
    match std::env::var(key) {
        Ok(v) => Ok(Some(v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(e).with_context(|| format!("invalid {key}")),
    }
}

impl Settings {
    pub const FRACTION: f64 = CACHE_FRACTION;

    pub fn resolve(&self, available: u64, raw_image_max: usize) -> Result<Budget> {
        ensure!(
            available > 0 && raw_image_max > 0,
            "no available host memory for offload"
        );
        ensure!(
            available <= 1u64 << 53,
            "host memory measurement is too large"
        );
        let gross = ((available as f64 * Self::FRACTION).floor() as u64).min(available);
        let evaluate = |cache: u64| -> Option<(u64, u64, u64)> {
            let snapshot = cache.min(raw_image_max as u64);
            let queued = deepseek_ultrafast_server::request_queue::MAX_PREPARED_REQUESTS as u64;
            let lease_and_work =
                (deepseek_ultrafast_kv_transfer::MAX_LEASES as u64 + queued + 2).checked_mul(snapshot)?;
            let receive = snapshot
                .max(2 * MIB)
                .checked_next_power_of_two()?
                .checked_mul(4)?;
            let heap_slack = lease_and_work.checked_add(FIXED_SLACK)?;
            let registered = cache.checked_add(lease_and_work)?.checked_add(receive)?;
            let total = cache.checked_add(heap_slack)?.checked_add(registered)?;
            (total <= gross)
                .then_some((snapshot, registered, heap_slack))
        };
        let mut lo = 0;
        let mut hi = gross / 2 / MIB + 1;
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if evaluate(mid * MIB).is_some() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        ensure!(
            lo > 0,
            "host offload budget is too small after staging/lease reserves"
        );
        let cache = lo * MIB;
        let (snapshot, registered, heap_slack) =
            evaluate(cache).context("invalid offload budget")?;
        Ok(Budget {
            gross,
            cache: usize::try_from(cache)?,
            snapshot: usize::try_from(snapshot)?,
            registered,
            heap_slack,
        })
    }
}

pub fn sample() -> Result<Available> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let host = mem_available(&meminfo)?;
    let cgroups = std::fs::read_to_string("/proc/self/cgroup")?;
    let mounts = std::fs::read_to_string("/proc/self/mountinfo")?;
    measure(
        host,
        &cgroups,
        &mounts,
        |path| match std::fs::read_to_string(path) {
            Ok(s) => Ok(Some(s)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        },
    )
}

fn mem_available(text: &str) -> Result<u64> {
    let line = text
        .lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .context("/proc/meminfo has no MemAvailable")?;
    let fields: Vec<_> = line.split_whitespace().collect();
    ensure!(
        fields.len() == 3 && fields[2] == "kB",
        "invalid MemAvailable units"
    );
    fields[1]
        .parse::<u64>()?
        .checked_mul(1024)
        .context("MemAvailable overflow")
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    V1,
    V2,
}

fn unescape(s: &str) -> String {
    s.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

fn paths(cgroups: &str, mounts: &str) -> Result<Vec<(PathBuf, Kind)>> {
    let mut out = Vec::new();
    for line in cgroups.lines() {
        let fields: Vec<_> = line.splitn(3, ':').collect();
        ensure!(fields.len() == 3, "invalid /proc/self/cgroup");
        let kind = if fields[1].is_empty() {
            Kind::V2
        } else if fields[1].split(',').any(|s| s == "memory") {
            Kind::V1
        } else {
            continue;
        };
        let current = Path::new(fields[2]);
        ensure!(
            current.is_absolute()
                && current
                    .components()
                    .all(|c| matches!(c, Component::RootDir | Component::Normal(_))),
            "invalid cgroup path"
        );
        let mut matched = false;
        for mount in mounts.lines() {
            let Some((left, right)) = mount.split_once(" - ") else {
                continue;
            };
            let a: Vec<_> = left.split_whitespace().collect();
            let b: Vec<_> = right.split_whitespace().collect();
            if a.len() < 5 || b.len() < 3 {
                continue;
            }
            let relevant = match kind {
                Kind::V2 => b[0] == "cgroup2",
                Kind::V1 => b[0] == "cgroup" && b[2].split(',').any(|s| s == "memory"),
            };
            if !relevant {
                continue;
            }
            let root = PathBuf::from(unescape(a[3]));
            let base = PathBuf::from(unescape(a[4]));
            let Ok(relative) = current.strip_prefix(&root) else {
                continue;
            };
            matched = true;
            let mut path = base.join(relative);
            loop {
                out.push((path.clone(), kind));
                if path == base {
                    break;
                }
                ensure!(
                    path.pop() && path.starts_with(&base),
                    "cgroup hierarchy escaped mount"
                );
            }
        }
        ensure!(matched, "cannot locate the process memory cgroup mount");
    }
    Ok(out)
}

fn measure(
    host: u64,
    cgroups: &str,
    mounts: &str,
    read: impl Fn(&Path) -> Result<Option<String>>,
) -> Result<Available> {
    let mut measured = Available {
        host,
        effective: host,
        limits: Vec::new(),
    };
    for (path, kind) in paths(cgroups, mounts)? {
        let (limit, current) = match kind {
            Kind::V2 => ("memory.max", "memory.current"),
            Kind::V1 => ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        };
        let (limit, current) = (read(&path.join(limit))?, read(&path.join(current))?);
        let (limit, current) = match (limit, current) {
            (None, None) => continue,
            (Some(l), Some(c)) => (l, c),
            _ => anyhow::bail!("incomplete memory accounting at {}", path.display()),
        };
        let current: u64 = current
            .trim()
            .parse()
            .context("invalid cgroup memory usage")?;
        let mut limits = vec![limit];
        if matches!(kind, Kind::V2)
            && let Some(high) = read(&path.join("memory.high"))?
        {
            limits.push(high);
        }
        for limit in limits {
            if limit.trim() == "max" {
                continue;
            }
            let limit: u64 = limit
                .trim()
                .parse()
                .context("invalid cgroup memory limit")?;
            measured.effective = measured.effective.min(limit.saturating_sub(current));
            measured.limits.push((path.clone(), limit, current));
        }
    }
    ensure!(
        measured.effective > 0,
        "no available memory after CUDA warmup"
    );
    Ok(measured)
}
