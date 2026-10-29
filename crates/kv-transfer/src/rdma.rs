use anyhow::{Result, ensure};
use bytes::Bytes;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use velo::{DataHandle, PeerInfo, ShutdownPolicy, Velo};

#[derive(Debug)]
pub struct RdmaRequired(pub String);
impl std::fmt::Display for RdmaRequired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KV RDMA REQUIRED (TCP fallback forbidden): {}", self.0)
    }
}
impl std::error::Error for RdmaRequired {}

pub fn is_rdma_required(error: &anyhow::Error) -> bool {
    error.downcast_ref::<RdmaRequired>().is_some()
}

pub(crate) fn required(error: impl std::fmt::Display) -> anyhow::Error {
    RdmaRequired(error.to_string()).into()
}

#[cfg(any(test, all(target_os = "linux", feature = "rdma")))]
const DEFAULT_TLS: &str = "rc_verbs,ud_verbs";

#[cfg(any(test, all(target_os = "linux", feature = "rdma")))]
fn validate_env(tls: &str, devices: &str, disabled: &str, proto: &str) -> Result<()> {
    ensure!(
        !matches!(
            disabled.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        "VELO_RDMA_RENDEZVOUS_DISABLE disables RDMA"
    );
    ensure!(
        proto.is_empty()
            || matches!(
                proto.to_ascii_lowercase().as_str(),
                "y" | "yes" | "1" | "true" | "on"
            ),
        "UCX_PROTO_ENABLE must be enabled"
    );
    let mut transports: Vec<_> = tls.split(',').collect();
    transports.sort_unstable();
    ensure!(
        transports == ["rc_verbs", "ud_verbs"] || transports == ["rc_mlx5", "ud_mlx5"],
        "UCX_TLS must be rc_verbs,ud_verbs or rc_mlx5,ud_mlx5; software RMA is forbidden (got {tls:?})"
    );
    ensure!(
        !devices.is_empty(),
        "no ACTIVE RDMA port found (or set UCX_NET_DEVICES, e.g. mlx5_0:1)"
    );
    for device in devices.split(',') {
        let (name, port) = device.split_once(':').ok_or_else(|| {
            anyhow::anyhow!("UCX_NET_DEVICES requires device:port, got {device:?}")
        })?;
        ensure!(
            !name.is_empty()
                && name != "all"
                && name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
            "invalid RDMA device {name:?}"
        );
        ensure!(
            port.parse::<u16>().is_ok_and(|p| p > 0),
            "invalid RDMA port {port:?}"
        );
    }
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "rdma"))]
fn active_ports() -> Vec<String> {
    let mut ports = Vec::new();
    for device in std::fs::read_dir("/sys/class/infiniband").into_iter().flatten().flatten() {
        let name = device.file_name().to_string_lossy().into_owned();
        for port in std::fs::read_dir(device.path().join("ports")).into_iter().flatten().flatten() {
            let state = std::fs::read_to_string(port.path().join("state")).unwrap_or_default();
            if state.trim() == "4: ACTIVE" {
                ports.push(format!("{name}:{}", port.file_name().to_string_lossy()));
            }
        }
    }
    ports.sort();
    ports
}

#[cfg(all(target_os = "linux", feature = "rdma"))]
fn raise_memlock() {
    let mut limit = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    unsafe {
        if libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut limit) == 0 && limit.rlim_cur < limit.rlim_max {
            limit.rlim_cur = limit.rlim_max;
            libc::setrlimit(libc::RLIMIT_MEMLOCK, &limit);
        }
    }
}

#[cfg(all(target_os = "linux", feature = "rdma"))]
fn hardware_env() -> Result<(String, String)> {
    let tls = std::env::var("UCX_TLS").unwrap_or_else(|_| DEFAULT_TLS.into());
    let devices = std::env::var("UCX_NET_DEVICES").ok().filter(|d| !d.is_empty())
        .unwrap_or_else(|| active_ports().join(","));
    validate_env(
        &tls,
        &devices,
        &std::env::var("VELO_RDMA_RENDEZVOUS_DISABLE").unwrap_or_default(),
        &std::env::var("UCX_PROTO_ENABLE").unwrap_or_default(),
    )?;
    for device in devices.split(',') {
        let (name, port) = device.split_once(':').unwrap();
        let path = format!("/sys/class/infiniband/{name}/ports/{port}/state");
        let state = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("RDMA port {device} unavailable: {path}: {e}"))?;
        ensure!(
            state.trim() == "4: ACTIVE",
            "RDMA port {device} is not ACTIVE: {}",
            state.trim()
        );
    }
    Ok((tls, devices))
}

pub(crate) struct Staged {
    pub handle: DataHandle,
    store: Arc<velo::rendezvous::store::DataStore>,
    #[cfg(all(target_os = "linux", feature = "rdma"))]
    registration: Option<Registration>,
}
#[cfg(all(target_os = "linux", feature = "rdma"))]
struct Registration {
    region: velo::rendezvous::rdma::RegionGuard,
    runtime: tokio::runtime::Handle,
    timeout: Duration,
}
impl Drop for Staged {
    fn drop(&mut self) {
        let id = self.handle.unpack().1;
        self.store.ref_decrement(id);
        self.store.try_free(id);
        #[cfg(all(target_os = "linux", feature = "rdma"))]
        if let Some(registration) = self.registration.take() {
            registration.runtime.spawn(async move {
                if let Err(error) = registration.region.unregister(registration.timeout).await {
                    deepseek_ultrafast_logging::diagnostic!("KV RDMA deregistration unconfirmed; Velo retains the memory until shutdown: {error}");
                }
            });
        }
    }
}

pub(crate) struct DataPlane {
    pub velo: Arc<Velo>,
    #[cfg(all(target_os = "linux", feature = "rdma"))]
    cleanup_timeout: Duration,
}

impl DataPlane {
    pub async fn start(bind: SocketAddr, budget: u64, timeout: Duration) -> Result<Self> {
        #[cfg(not(all(target_os = "linux", feature = "rdma")))]
        {
            let _ = (bind, budget, timeout);
            Err(required(
                "this binary has no Linux Velo UCX support; build with the rdma feature (enabled by default)",
            ))
        }
        #[cfg(all(target_os = "linux", feature = "rdma"))]
        {
            raise_memlock();
            let (tls, devices) = hardware_env().map_err(required)?;
            let plane = Self::build(bind, budget, timeout, &tls, &devices)
                .await
                .map_err(required)?;
            let result = plane.probe(bind, budget, timeout, &tls, &devices).await;
            if let Err(error) = result {
                plane.shutdown().await;
                return Err(required(format!(
                    "startup RDMA registration/GET probe failed: {error:#}"
                )));
            }
            deepseek_ultrafast_logging::diagnostic!(
                "KV RDMA ready: UCX_TLS={tls} UCX_NET_DEVICES={devices}; bidirectional registered-memory GET probe passed"
            );
            Ok(plane)
        }
    }

    #[cfg(all(target_os = "linux", feature = "rdma"))]
    async fn build(
        bind: SocketAddr,
        budget: u64,
        timeout: Duration,
        tls: &str,
        devices: &str,
    ) -> Result<Self> {
        use velo::rendezvous::rdma::{RdmaConfig, RdmaPoolConfig, RdmaRendezvousConfig};
        use velo::transports::{tcp::TcpTransportBuilder, ucx::UcxTransportBuilder};
        let tcp = Arc::new(
            TcpTransportBuilder::new()
                .bind_addr(bind)
                .socket_buffers(None)
                .build()?,
        );
        let ucx = Arc::new(
            UcxTransportBuilder::new()
                .tls(tls)
                .net_devices(devices)
                .eager_endpoints(true)
                .ep_idle_timeout(None)
                .build()?,
        );
        let velo = Velo::builder()
            .add_transport(tcp)
            .add_ucx_transport(ucx)
            .stream_bind_addr(bind.ip())
            .rdma_config(RdmaConfig {
                pool: RdmaPoolConfig {
                    initial_arena_bytes: 2 << 20,
                    registered_bytes_budget: budget,
                    arena_reclaim_after: Some(Duration::from_secs(1)),
                    retain_arena_bytes: 2 << 20,
                    ..Default::default()
                },
                rendezvous: RdmaRendezvousConfig {
                    enabled: true,
                    rdma_min_bytes: 1,
                    lease_timeout: timeout.max(Duration::from_secs(5)),
                },
                ..Default::default()
            })
            .build()
            .await?;
        Ok(Self {
            velo,
            cleanup_timeout: timeout.saturating_mul(2).max(Duration::from_secs(30)),
        })
    }

    #[cfg(all(target_os = "linux", feature = "rdma"))]
    async fn probe(
        &self,
        bind: SocketAddr,
        budget: u64,
        timeout: Duration,
        tls: &str,
        devices: &str,
    ) -> Result<()> {
        let probe =
            Self::build(SocketAddr::new(bind.ip(), 0), budget, timeout, tls, devices).await?;
        let result = tokio::time::timeout(timeout, async {
            self.register_peer(probe.velo.peer_info())?;
            probe.register_peer(self.velo.peer_info())?;
            let data = vec![0xa7; 4096];
            let staged = self.stage(&data).await?;
            ensure!(
                probe.receive(staged.handle, data.len()).await? == data,
                "RDMA probe data mismatch"
            );
            let staged = probe.stage(&data).await?;
            ensure!(
                self.receive(staged.handle, data.len()).await? == data,
                "reverse RDMA probe data mismatch"
            );
            Ok::<_, anyhow::Error>(())
        })
        .await;
        probe.shutdown().await;
        result?
    }

    pub fn register_peer(&self, peer: PeerInfo) -> Result<()> {
        ensure!(
            peer.worker_address().get_entry("ucx")?.is_some(),
            RdmaRequired("peer advertises no UCX RDMA endpoint".into())
        );
        self.velo.register_peer(peer).map_err(required)
    }

    pub async fn stage(&self, data: &[u8]) -> Result<Arc<Staged>> {
        #[cfg(not(all(target_os = "linux", feature = "rdma")))]
        {
            let _ = data;
            Err(required("Velo UCX was not compiled in"))
        }
        #[cfg(all(target_os = "linux", feature = "rdma"))]
        {
            let mut owned = Vec::new();
            owned.try_reserve_exact(data.len()).map_err(required)?;
            owned.extend_from_slice(data);
            let region = self
                .velo
                .register_owned(owned.into_boxed_slice())
                .await
                .map_err(required)?;
            let handle = self
                .velo
                .rendezvous_manager()
                .register_data_in_region(&region, 0..data.len() as u64)
                .map_err(required)?;
            Ok(Arc::new(Staged {
                handle,
                store: self.velo.rendezvous_manager().data_store().clone(),
                registration: Some(Registration {
                    region,
                    runtime: tokio::runtime::Handle::current(),
                    timeout: self.cleanup_timeout,
                }),
            }))
        }
    }

    pub async fn receive(&self, handle: DataHandle, len: usize) -> Result<Bytes> {
        #[cfg(not(all(target_os = "linux", feature = "rdma")))]
        {
            let _ = (handle, len);
            Err(required("Velo UCX was not compiled in"))
        }
        #[cfg(all(target_os = "linux", feature = "rdma"))]
        {
            let metadata = self.velo.metadata(handle).await.map_err(required)?;
            ensure!(
                metadata.pinned && metadata.total_len == len as u64,
                RdmaRequired(
                    "donor is not registered or its handle length differs from the manifest".into()
                )
            );
            let capacity = len
                .checked_next_power_of_two()
                .ok_or_else(|| required("receive capacity overflow"))?;
            let writer = self
                .velo
                .alloc_pinned_writer(capacity)
                .await
                .map_err(required)?;
            let mut dest = StrictWriter(writer);
            let lease = self
                .velo
                .get_into(handle, &mut dest)
                .await
                .map_err(required)?;
            self.velo.detach(handle, lease).await.map_err(required)?;
            let data = Bytes::copy_from_slice(&dest.0.as_slice()[..len]);
            Ok(data)
        }
    }

    pub async fn shutdown(&self) {
        self.velo
            .shutdown(ShutdownPolicy::Timeout(Duration::from_secs(5)))
            .await;
    }
}

#[cfg(all(target_os = "linux", feature = "rdma"))]
struct StrictWriter(velo::PinnedWriter);
#[cfg(all(target_os = "linux", feature = "rdma"))]
impl velo::RendezvousWrite for StrictWriter {
    fn capacity(&self) -> usize {
        self.0.len()
    }
    fn rdma_destination(&mut self) -> Option<velo::RdmaDestination<'_>> {
        self.0.rdma_destination()
    }
    fn write_chunk(&mut self, _offset: usize, _data: &[u8]) -> Result<()> {
        Err(required(
            "Velo attempted a chunked/copy fallback instead of a direct RDMA GET",
        ))
    }
}

pub async fn check_rdma(bind: SocketAddr, budget: u64, timeout: Duration) -> Result<()> {
    let plane = DataPlane::start(bind, budget, timeout).await?;
    println!("rdma-check: ok");
    plane.shutdown().await;
    Ok(())
}

#[cfg(test)]
pub(crate) use tests::heap_stage;

#[cfg(test)]
mod tests {
    use super::*;
    pub(crate) fn heap_stage(data: Bytes) -> Arc<Staged> {
        let store = Arc::new(velo::rendezvous::store::DataStore::new());
        let id = store.register(data, None);
        Arc::new(Staged {
            handle: DataHandle::pack(velo::InstanceId::new_v4().worker_id(), id),
            store,
            #[cfg(all(target_os = "linux", feature = "rdma"))]
            registration: None,
        })
    }
}
