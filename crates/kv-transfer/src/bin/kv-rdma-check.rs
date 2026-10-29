use anyhow::Result;
use std::time::Duration;
#[tokio::main]
async fn main() -> Result<()> {
    let bind = std::net::SocketAddr::new(deepseek_ultrafast_kv_transfer::kv_address()?.ip(), 0);
    deepseek_ultrafast_kv_transfer::check_rdma(bind, 64 << 20, Duration::from_secs(30)).await
}
