use anyhow::{Result, ensure};
use redis::Script;
use redis::aio::ConnectionManager;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use std::time::Duration;
use uuid::Uuid;
use velo::PeerInfo;

pub const DEFAULT_REDIS_URL: &str = "redis://127.0.0.1:6379";
pub const DEFAULT_DIRECTORY_PREFIX: &str = "deepseek-ultrafast:kv:v1";
const RECORD_TTL: Duration = Duration::from_secs(86_400);
pub(crate) const MAX_CHECK: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Revision {
    pub epoch: Uuid,
    pub sequence: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Owner {
    pub revision: Revision,
    pub peer: Option<PeerInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Onboarded {
    pub key: String,
    pub epoch: Uuid,
    pub sequence: u64,
    pub peer: PeerInfo,
}
impl Onboarded {
    pub(crate) fn owner(&self) -> Owner {
        Owner {
            revision: Revision { epoch: self.epoch, sequence: self.sequence },
            peer: Some(self.peer.clone()),
        }
    }
}

const EPOCH: &str = r#"
local function epoch()
  redis.call('SET', KEYS[1], ARGV[1], 'NX')
  return redis.call('GET', KEYS[1])
end
local function next_sequence()
  return string.format('%.0f', redis.call('INCR', KEYS[2]))
end
"#;

static LOOKUP: LazyLock<Script> = LazyLock::new(|| Script::new(&format!(r#"{EPOCH}
local e = epoch()
local r = redis.call('HMGET', KEYS[3], 'epoch', 'seq', 'peer')
if r[1] == e then
  redis.call('PEXPIRE', KEYS[3], ARGV[2])
  return {{r[1], r[2], r[3]}}
end
local s = next_sequence()
redis.call('DEL', KEYS[3])
redis.call('HSET', KEYS[3], 'epoch', e, 'seq', s, 'peer', '')
redis.call('PEXPIRE', KEYS[3], ARGV[2])
return {{e, s, ''}}
"#)));

static CLAIM: LazyLock<Script> = LazyLock::new(|| Script::new(&format!(r#"{EPOCH}
local e = epoch()
local r = redis.call('HMGET', KEYS[3], 'epoch', 'seq')
if r[1] ~= e or r[1] ~= ARGV[3] or r[2] ~= ARGV[4] then
  return redis.error_reply('STALE ownership changed, expired or directory restarted')
end
local s = next_sequence()
redis.call('HSET', KEYS[3], 'seq', s, 'peer', ARGV[5])
redis.call('PEXPIRE', KEYS[3], ARGV[2])
redis.call('PUBLISH', ARGV[6], '{{"key":"' .. ARGV[7] .. '","epoch":"' .. e .. '","sequence":' .. s .. ',"peer":' .. ARGV[5] .. '}}')
return {{e, s, ARGV[5]}}
"#)));

static CHECK: LazyLock<Script> = LazyLock::new(|| Script::new(&format!(r#"{EPOCH}
local e = epoch()
local invalid = {{}}
for i = 3, #KEYS do
  local r = redis.call('HMGET', KEYS[i], 'epoch', 'seq')
  if r[1] == e and r[1] == ARGV[2 * i - 3] and r[2] == ARGV[2 * i - 2] then
    redis.call('PEXPIRE', KEYS[i], ARGV[2])
  else
    invalid[#invalid + 1] = i - 3
  end
end
return {{e, invalid}}
"#)));

static PING: LazyLock<Script> = LazyLock::new(|| Script::new(&format!("{EPOCH}\nreturn epoch()")));

pub(crate) fn validate_prefix(prefix: &str) -> Result<()> {
    ensure!(
        !prefix.is_empty()
            && prefix.len() <= 128
            && prefix.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b':' | b'.')),
        "invalid KV directory prefix"
    );
    Ok(())
}
fn hash_key(key: &str) -> bool {
    key.len() == 64 && key.bytes().all(|b| b.is_ascii_hexdigit())
}

#[derive(Clone)]
pub(crate) struct Directory {
    conn: ConnectionManager,
    tag: String,
    timeout: Duration,
}
impl Directory {
    pub(crate) async fn connect(client: &redis::Client, prefix: &str, timeout: Duration) -> Result<Self> {
        validate_prefix(prefix)?;
        let conn = tokio::time::timeout(timeout, ConnectionManager::new(client.clone())).await??;
        Ok(Self { conn, tag: format!("{{{prefix}}}"), timeout })
    }
    pub(crate) fn events(&self, scope: &str) -> String {
        format!("{}:onboarded:{scope}", self.tag)
    }
    fn record(&self, scope: &str, key: &str) -> String {
        format!("{}:owner:{scope}:{key}", self.tag)
    }
    fn invocation<'a>(&self, script: &'a Script) -> redis::ScriptInvocation<'a> {
        let mut call = script.prepare_invoke();
        call.key(format!("{}:epoch", self.tag))
            .key(format!("{}:sequence", self.tag))
            .arg(Uuid::new_v4().to_string());
        call
    }
    async fn run<T: redis::FromRedisValue>(&self, call: redis::ScriptInvocation<'_>) -> Result<T> {
        let mut conn = self.conn.clone();
        let value = tokio::time::timeout(self.timeout, call.invoke_async(&mut conn)).await??;
        Ok(value)
    }

    pub(crate) async fn epoch(&self) -> Result<Uuid> {
        let result = self.run::<String>(self.invocation(&PING)).await;
        Ok(Uuid::parse_str(&logged(result)?)?)
    }
    pub(crate) async fn lookup(&self, scope: &str, key: &str) -> Result<Owner> {
        ensure!(hash_key(scope) && hash_key(key), "invalid directory key");
        let mut call = self.invocation(&LOOKUP);
        call.key(self.record(scope, key)).arg(RECORD_TTL.as_millis() as u64);
        owner(logged(self.run(call).await)?)
    }
    pub(crate) async fn claim(&self, scope: &str, key: &str, expected: Revision, peer: &PeerInfo) -> Result<Owner> {
        ensure!(hash_key(scope) && hash_key(key), "invalid directory key");
        let mut call = self.invocation(&CLAIM);
        call.key(self.record(scope, key))
            .arg(RECORD_TTL.as_millis() as u64)
            .arg(expected.epoch.to_string())
            .arg(expected.sequence.to_string())
            .arg(serde_json::to_string(peer)?)
            .arg(self.events(scope))
            .arg(key);
        owner(self.run(call).await.map_err(|e| match e.downcast::<redis::RedisError>() {
            Ok(e) if e.code() == Some("STALE") => anyhow::anyhow!("KV directory: {}", e.detail().unwrap_or("claim rejected")),
            Ok(e) => e.into(),
            Err(e) => e,
        })?)
    }
    pub(crate) async fn check(&self, scope: &str, entries: &[(String, Revision)]) -> Result<(Uuid, Vec<usize>)> {
        ensure!(hash_key(scope) && entries.len() <= MAX_CHECK && entries.iter().all(|(k, _)| hash_key(k)), "invalid directory check");
        let mut call = self.invocation(&CHECK);
        call.arg(RECORD_TTL.as_millis() as u64);
        for (key, revision) in entries {
            call.key(self.record(scope, key)).arg(revision.epoch.to_string()).arg(revision.sequence.to_string());
        }
        let (epoch, invalid): (String, Vec<usize>) = logged(self.run(call).await)?;
        ensure!(
            invalid.iter().all(|&i| i < entries.len()) && invalid.windows(2).all(|w| w[0] < w[1]),
            "invalid directory check reply"
        );
        Ok((Uuid::parse_str(&epoch)?, invalid))
    }
}

fn owner((epoch, sequence, peer): (String, String, String)) -> Result<Owner> {
    Ok(Owner {
        revision: Revision { epoch: Uuid::parse_str(&epoch)?, sequence: sequence.parse()? },
        peer: if peer.is_empty() { None } else { Some(serde_json::from_str(&peer)?) },
    })
}
fn transport(error: &redis::RedisError) -> bool {
    error.is_io_error() || error.is_connection_dropped() || error.is_connection_refusal() || error.is_timeout()
}
fn logged<T>(result: Result<T>) -> Result<T> {
    if let Err(error) = &result { log_failure(error); }
    result
}
pub(crate) fn log_failure(error: &anyhow::Error) {
    if error.is::<tokio::time::error::Elapsed>() {
        deepseek_ultrafast_logging::redis_failure("directory", "request_timeout");
    } else if error.downcast_ref::<redis::RedisError>().is_some_and(transport) {
        deepseek_ultrafast_logging::redis_failure("directory", "request");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn client() -> Option<redis::Client> {
        let url = std::env::var("DSUF_TEST_REDIS_URL").ok();
        if url.is_none() {
            eprintln!("DSUF_TEST_REDIS_URL unset: skipped");
        }
        url.map(|u| redis::Client::open(u).unwrap())
    }
    fn peer(b: u8) -> PeerInfo {
        PeerInfo::new(velo::InstanceId::new_v4(), velo::WorkerAddress::from_encoded(vec![b; 16]))
    }
    fn hex(c: char) -> String {
        std::iter::repeat_n(c, 64).collect()
    }
    async fn clear(client: &redis::Client, prefix: &str) {
        let mut conn = client.get_multiplexed_async_connection().await.unwrap();
        let keys: Vec<String> = redis::cmd("KEYS").arg(format!("{{{prefix}}}*")).query_async(&mut conn).await.unwrap();
        if !keys.is_empty() {
            redis::cmd("DEL").arg(keys).query_async::<()>(&mut conn).await.unwrap();
        }
    }

    #[tokio::test]
    async fn ownership_cas_check_events_and_epoch_fencing() {
        let Some(client) = client() else { return };
        let prefix = format!("deepseek-ultrafast:test:{}", Uuid::new_v4().simple());
        let dir = Directory::connect(&client, &prefix, Duration::from_secs(5)).await.unwrap();
        let (scope, key, other) = (hex('a'), hex('b'), hex('c'));
        let epoch = dir.epoch().await.unwrap();
        assert_eq!(dir.epoch().await.unwrap(), epoch);
        let mut pubsub = client.get_async_pubsub().await.unwrap();
        pubsub.subscribe(dir.events(&scope)).await.unwrap();
        let mut events = pubsub.into_on_message();

        let first = dir.lookup(&scope, &key).await.unwrap();
        assert_eq!(first.revision.epoch, epoch);
        assert!(first.peer.is_none());
        assert_eq!(dir.lookup(&scope, &key).await.unwrap().revision, first.revision);

        let p = peer(1);
        let owned = dir.claim(&scope, &key, first.revision, &p).await.unwrap();
        assert!(owned.revision.sequence > first.revision.sequence);
        assert_eq!(owned.peer.as_ref().unwrap().instance_id(), p.instance_id());
        let message = tokio::time::timeout(Duration::from_secs(5), events.next()).await.unwrap().unwrap();
        let event: Onboarded = serde_json::from_slice(&message.get_payload::<Vec<u8>>().unwrap()).unwrap();
        assert_eq!(event.key, key);
        assert_eq!(event.owner().revision, owned.revision);
        assert_eq!(event.peer.instance_id(), p.instance_id());
        assert_eq!(event.peer.worker_address(), p.worker_address());

        let error = dir.claim(&scope, &key, first.revision, &peer(2)).await.unwrap_err();
        assert!(error.to_string().contains("ownership changed"), "{error:#}");
        let seen = dir.lookup(&scope, &key).await.unwrap();
        assert_eq!(seen.revision, owned.revision);
        assert_eq!(seen.peer.unwrap().instance_id(), p.instance_id());

        let entries = [(key.clone(), owned.revision), (key.clone(), first.revision), (other.clone(), owned.revision)];
        assert_eq!(dir.check(&scope, &entries).await.unwrap(), (epoch, vec![1, 2]));
        assert_eq!(dir.check(&scope, &[]).await.unwrap(), (epoch, vec![]));

        clear(&client, &prefix).await;
        let epoch2 = dir.epoch().await.unwrap();
        assert_ne!(epoch2, epoch);
        assert_eq!(dir.check(&scope, &entries[..1]).await.unwrap(), (epoch2, vec![0]));
        assert!(dir.claim(&scope, &key, owned.revision, &p).await.is_err());
        let fresh = dir.lookup(&scope, &key).await.unwrap();
        assert_eq!(fresh.revision.epoch, epoch2);
        assert!(fresh.peer.is_none());
        clear(&client, &prefix).await;
    }

    #[tokio::test]
    async fn rejects_malformed_keys_and_prefixes() {
        assert!(validate_prefix("deepseek-ultrafast:kv:v1").is_ok());
        assert!(validate_prefix("").is_err());
        assert!(validate_prefix("a b").is_err());
        assert!(validate_prefix("a{b}").is_err());
        let Some(client) = client() else { return };
        let dir = Directory::connect(&client, "deepseek-ultrafast:test:keys", Duration::from_secs(5)).await.unwrap();
        assert!(dir.lookup("scope", &hex('b')).await.is_err());
        assert!(dir.lookup(&hex('a'), "b*").await.is_err());
    }
}
