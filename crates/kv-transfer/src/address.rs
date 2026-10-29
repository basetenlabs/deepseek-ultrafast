use anyhow::{Context, Result};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

pub const DEFAULT_KV_PORT: u16 = 19500;

pub fn kv_address() -> Result<SocketAddr> {
    let var = |key: &str| std::env::var(key).ok().filter(|v| !v.trim().is_empty());
    let port = match var("DSUF_KV_PORT") {
        Some(v) => v.trim().parse().with_context(|| format!("DSUF_KV_PORT={v} is not a port"))?,
        None => DEFAULT_KV_PORT,
    };
    let host = match var("DSUF_KV_HOST") {
        Some(v) => v.trim().parse().with_context(|| format!("DSUF_KV_HOST={v} is not an IP address"))?,
        None => first_host_address().context("no network address found: set DSUF_KV_HOST to this machine's routable IP")?,
    };
    Ok(SocketAddr::new(host, port))
}

fn first_host_address() -> Option<IpAddr> {
    let mut list: *mut libc::ifaddrs = std::ptr::null_mut();
    if unsafe { libc::getifaddrs(&mut list) } != 0 {
        return None;
    }
    let (mut v4, mut v6) = (None, None);
    let mut cur = list;
    while !cur.is_null() {
        let ifa = unsafe { &*cur };
        cur = ifa.ifa_next;
        if ifa.ifa_addr.is_null() || ifa.ifa_flags & libc::IFF_UP as libc::c_uint == 0 {
            continue;
        }
        match unsafe { (*ifa.ifa_addr).sa_family } as libc::c_int {
            libc::AF_INET if v4.is_none() => {
                let a = unsafe { &*(ifa.ifa_addr as *const libc::sockaddr_in) };
                let ip = Ipv4Addr::from(u32::from_be(a.sin_addr.s_addr));
                if !ip.is_loopback() {
                    v4 = Some(IpAddr::V4(ip));
                }
            }
            libc::AF_INET6 if v6.is_none() => {
                let a = unsafe { &*(ifa.ifa_addr as *const libc::sockaddr_in6) };
                let ip = Ipv6Addr::from(a.sin6_addr.s6_addr);
                if !ip.is_loopback() && (ip.segments()[0] & 0xffc0) != 0xfe80 {
                    v6 = Some(IpAddr::V6(ip));
                }
            }
            _ => {}
        }
    }
    unsafe { libc::freeifaddrs(list) };
    v4.or(v6)
}

#[cfg(test)]
mod tests {
    #[test]
    fn host_address_matches_hostname() {
        let ours = super::first_host_address();
        let Ok(out) = std::process::Command::new("hostname").arg("-I").output() else { return };
        let first = String::from_utf8_lossy(&out.stdout).split_whitespace().next().map(|s| s.parse().unwrap());
        assert_eq!(ours, first);
    }
}
