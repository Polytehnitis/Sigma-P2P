//! Private sync between the user's own paired devices.
//!
//! Private state (memories, reputations, learned weights, working state) never touches the
//! gossip swarm. It only travels over direct iroh connections, which are QUIC + TLS 1.3 keyed
//! by each device's Ed25519 identity: end-to-end encrypted (relays only see ciphertext) and
//! mutually authenticated, so `remote_id()` is the peer's proven identity. Each side serves a
//! connection only if that identity is one of its paired devices.
//!
//! Pairing: one device makes a single-use code (`sigma-pair:<endpoint id>:<secret>`); the other
//! dials it and both prove knowledge of the secret, bound to both identities, before either
//! records the other as paired. Paired-device lists then replicate like any other state, so
//! pairing a new device with one of them pairs it with all.

use anyhow::{bail, ensure, Context, Result};
use iroh::endpoint::Connection;
use iroh::protocol::{AcceptError, ProtocolHandler};
use iroh::{Endpoint, EndpointId};
use sigma_core::{crypto, FactorStore, SyncReport, SyncState};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

pub const DEVICE_SYNC_ALPN: &[u8] = b"/sigma/device-sync/1";
pub const PAIRING_ALPN: &[u8] = b"/sigma/pair/1";
/// Largest sync state accepted from a paired device.
pub const MAX_SYNC_BYTES: usize = 64 * 1024 * 1024;
/// How long a pairing code stays valid.
pub const PAIRING_TTL: Duration = Duration::from_secs(10 * 60);
/// How often a node syncs with each paired device in the background.
pub const SYNC_INTERVAL: Duration = Duration::from_secs(30);
const SYNC_TIMEOUT: Duration = Duration::from_secs(60);
/// Dialing an offline device gives up after this long.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const CODE_PREFIX: &str = "sigma-pair:";

/// A pending pairing secret and when it expires.
type Invite = ([u8; 32], Instant);

#[derive(Clone)]
pub(crate) struct DeviceSync {
    me: EndpointId,
    store: Arc<FactorStore>,
    reports: mpsc::UnboundedSender<SyncReport>,
    /// Pending pairing secrets and when they expire.
    invites: Arc<Mutex<Vec<Invite>>>,
}

impl std::fmt::Debug for DeviceSync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DeviceSync")
    }
}

/// Inbound side of `PAIRING_ALPN`.
#[derive(Debug, Clone)]
pub(crate) struct Pairing(pub(crate) DeviceSync);

fn proof(secret: &[u8; 32], role: &str, a: &EndpointId, b: &EndpointId) -> blake3::Hash {
    let mut h = blake3::Hasher::new_keyed(secret);
    h.update(b"sigma-pair ").update(role.as_bytes()).update(a.as_bytes()).update(b.as_bytes());
    h.finalize()
}

/// Splits a pairing code into the inviting device and the secret.
pub fn parse_pairing_code(code: &str) -> Result<(EndpointId, [u8; 32])> {
    let rest = code.trim().strip_prefix(CODE_PREFIX).context("not a sigma pairing code")?;
    let (id, secret) = rest.rsplit_once(':').context("malformed pairing code")?;
    let id: EndpointId = id.parse().context("bad device id in pairing code")?;
    let secret = crypto::pubkey_from_hex(secret).map_err(anyhow::Error::msg).context("bad pairing secret")?;
    Ok((id, secret))
}

impl DeviceSync {
    pub(crate) fn new(me: EndpointId, store: Arc<FactorStore>) -> (Self, mpsc::UnboundedReceiver<SyncReport>) {
        let (reports, rx) = mpsc::unbounded_channel();
        (Self { me, store, reports, invites: Default::default() }, rx)
    }

    pub(crate) fn pairing_code(&self) -> String {
        let secret: [u8; 32] = rand_secret();
        let mut invites = self.invites.lock().expect("invites poisoned");
        invites.retain(|(_, exp)| *exp > Instant::now());
        invites.push((secret, Instant::now() + PAIRING_TTL));
        format!("{CODE_PREFIX}{}:{}", self.me, crypto::to_hex(&secret))
    }

    /// Joins the device that issued `code`, then runs a first sync with it.
    pub(crate) async fn pair(&self, endpoint: &Endpoint, code: &str) -> Result<EndpointId> {
        let (inviter, secret) = parse_pairing_code(code)?;
        let me = endpoint.id();
        ensure!(inviter != me, "that pairing code is this device's own");
        let conn = endpoint.connect(inviter, PAIRING_ALPN).await?;
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(proof(&secret, "join", &me, &inviter).as_bytes()).await?;
        send.finish()?;
        let reply = recv.read_to_end(32).await?;
        conn.close(0u32.into(), b"done");
        // `remote_id` is TLS-authenticated, so a forged code naming someone else fails here too.
        ensure!(conn.remote_id() == inviter, "pairing peer identity mismatch");
        ensure!(
            reply.len() == 32
                && blake3::Hash::from_bytes(reply.try_into().unwrap()) == proof(&secret, "accept", &inviter, &me),
            "pairing code rejected (expired, already used, or wrong)"
        );
        self.store.pair_device(inviter.as_bytes()).map_err(|e| anyhow::anyhow!(e))?;
        self.sync_with(endpoint, inviter).await?;
        Ok(inviter)
    }

    /// One sync session: send our state, merge the device's state that comes back.
    pub(crate) async fn sync_with(&self, endpoint: &Endpoint, device: EndpointId) -> Result<()> {
        ensure!(self.store.is_paired(device.as_bytes()), "{} is not a paired device", device.fmt_short());
        let session = async {
            let conn = tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect(device, DEVICE_SYNC_ALPN))
                .await
                .context("device unreachable")??;
            ensure!(conn.remote_id() == device, "sync peer identity mismatch");
            let (mut send, mut recv) = conn.open_bi().await?;
            send.write_all(&self.export()?).await?;
            send.finish()?;
            let reply = recv.read_to_end(MAX_SYNC_BYTES).await?;
            conn.close(0u32.into(), b"done");
            self.merge(device, &reply)
        };
        tokio::time::timeout(SYNC_TIMEOUT, session).await.context("sync timed out")?
    }

    /// Syncs with every paired device that can be reached, in parallel, including devices
    /// first heard of during this round. Returns how many were reached.
    pub(crate) async fn sync_all(&self, endpoint: &Endpoint) -> usize {
        let mut tried = std::collections::BTreeSet::new();
        let mut reached = 0;
        loop {
            let mut round = tokio::task::JoinSet::new();
            for device in self.store.paired_devices().unwrap_or_default() {
                let Ok(id) = EndpointId::from_bytes(&device) else { continue };
                if tried.insert(device) {
                    let (this, ep) = (self.clone(), endpoint.clone());
                    // Offline devices are normal: they catch up the next time either side syncs.
                    round.spawn(async move { this.sync_with(&ep, id).await.is_ok() });
                }
            }
            if round.is_empty() {
                return reached;
            }
            reached += round.join_all().await.into_iter().filter(|ok| *ok).count();
        }
    }

    fn export(&self) -> Result<Vec<u8>> {
        let state = self.store.export_sync_state().map_err(|e| anyhow::anyhow!(e))?;
        Ok(postcard::to_allocvec(&state)?)
    }

    fn merge(&self, from: EndpointId, bytes: &[u8]) -> Result<()> {
        let state: SyncState = postcard::from_bytes(bytes).context("malformed sync state")?;
        let report = self.store.merge_sync_state(*from.as_bytes(), state).map_err(|e| anyhow::anyhow!(e))?;
        if !report.is_empty() {
            let _ = self.reports.send(report);
        }
        Ok(())
    }

    async fn serve_sync(&self, conn: &Connection) -> Result<()> {
        let peer = conn.remote_id();
        if !self.store.is_paired(peer.as_bytes()) {
            conn.close(1u32.into(), b"not paired");
            bail!("refused sync from unpaired peer {}", peer.fmt_short());
        }
        let (mut send, mut recv) = conn.accept_bi().await?;
        let bytes = recv.read_to_end(MAX_SYNC_BYTES).await?;
        self.merge(peer, &bytes)?;
        send.write_all(&self.export()?).await?;
        send.finish()?;
        conn.closed().await;
        Ok(())
    }

    async fn serve_pairing(&self, conn: &Connection) -> Result<()> {
        let (joiner, me) = (conn.remote_id(), self.me);
        let (mut send, mut recv) = conn.accept_bi().await?;
        let got = recv.read_to_end(32).await?;
        let secret = {
            let mut invites = self.invites.lock().expect("invites poisoned");
            invites.retain(|(_, exp)| *exp > Instant::now());
            let hit = (got.len() == 32)
                .then(|| blake3::Hash::from_bytes(got.as_slice().try_into().unwrap()))
                .and_then(|got| invites.iter().position(|(s, _)| proof(s, "join", &joiner, &me) == got));
            hit.map(|i| invites.swap_remove(i).0) // single use
        };
        let Some(secret) = secret else {
            send.finish()?;
            conn.closed().await;
            bail!("rejected pairing attempt from {}", joiner.fmt_short());
        };
        self.store.pair_device(joiner.as_bytes()).map_err(|e| anyhow::anyhow!(e))?;
        send.write_all(proof(&secret, "accept", &me, &joiner).as_bytes()).await?;
        send.finish()?;
        conn.closed().await;
        Ok(())
    }
}

fn rand_secret() -> [u8; 32] {
    // A fresh Ed25519 key is 32 bytes from the OS CSPRNG; avoids another dependency.
    sigma_core::NodeIdentity::generate().to_bytes()
}

fn accept_err(e: anyhow::Error) -> AcceptError {
    eprintln!("sigma-net: {e:#}");
    AcceptError::from_boxed(e.into())
}

impl ProtocolHandler for DeviceSync {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        self.serve_sync(&conn).await.map_err(accept_err)
    }
}

impl ProtocolHandler for Pairing {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        self.0.serve_pairing(&conn).await.map_err(accept_err)
    }
}
