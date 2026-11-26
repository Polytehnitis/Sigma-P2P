//! sigma-net: serverless skill exchange over iroh (QUIC + TLS 1.3) and iroh-gossip.
//!
//! Impasses are broadcast on a gossip topic derived from the swarm passphrase. Any peer
//! holding a skill that mentions the missing predicate dials the requester directly and
//! offers the signed package; the requester verifies it before it ever reaches the engine.

use anyhow::{Context, Result};
use iroh::endpoint::{presets, Builder, Connection};
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{Endpoint, SecretKey};
use iroh_gossip::{Event, Gossip, GossipSender, TopicId};
use n0_future::StreamExt;
use serde::{Deserialize, Serialize};
use sigma_core::{FactorStore, ImpasseEvent, SkillPackage, QUARANTINE};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

pub use iroh::EndpointId;

pub const SKILL_EXCHANGE_ALPN: &[u8] = b"/sigma/skill-exchange/1";
/// Largest skill message accepted from a peer.
pub const MAX_SKILL_BYTES: usize = 4 * 1024 * 1024;

/// Broadcast when the local engine reaches an impasse.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GossipImpasseQuery {
    pub requester: EndpointId,
    pub impasse: ImpasseEvent,
    /// Makes repeated queries distinct (gossip deduplicates identical messages).
    pub nonce: u64,
}

/// Sent over a direct QUIC stream on `SKILL_EXCHANGE_ALPN`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DirectSkillMessage {
    OfferSkill(SkillPackage),
}

/// Local skills offered to the swarm, keyed by sorted predicate signature.
type SkillIndex = Arc<Mutex<BTreeMap<String, SkillPackage>>>;

pub struct SkillMeshNode {
    endpoint: Endpoint,
    router: Router,
    sender: GossipSender,
    local_skills: SkillIndex,
    neighbors: Arc<AtomicUsize>,
    listener: tokio::task::JoinHandle<()>,
    /// Verified skills pushed to us by peers.
    pub downloaded_skills: mpsc::Receiver<SkillPackage>,
}

impl SkillMeshNode {
    /// Joins the swarm over n0's public relays and DNS discovery (`presets::N0`).
    pub async fn spawn(
        swarm_passphrase: &str,
        bootstrap_peers: Vec<EndpointId>,
        store: Arc<FactorStore>,
    ) -> Result<Self> {
        Self::spawn_with(Endpoint::builder(presets::N0), swarm_passphrase, bootstrap_peers, store).await
    }

    /// Like [`spawn`](Self::spawn) with a caller-configured endpoint builder (e.g. LAN-only or tests).
    /// The node's identity key from `store` always becomes the endpoint's secret key.
    pub async fn spawn_with(
        builder: Builder,
        swarm_passphrase: &str,
        bootstrap_peers: Vec<EndpointId>,
        store: Arc<FactorStore>,
    ) -> Result<Self> {
        let identity = store.load_or_create_identity().map_err(|e| anyhow::anyhow!(e))?;
        let endpoint = builder.secret_key(SecretKey::from_bytes(&identity.to_bytes())).bind().await?;
        let gossip = Gossip::builder().spawn(endpoint.clone());
        let (tx, downloaded_skills) = mpsc::channel(64);
        let router = Router::builder(endpoint.clone())
            .accept(iroh_gossip::ALPN, gossip.clone())
            .accept(SKILL_EXCHANGE_ALPN, SkillExchange { store: store.clone(), tx })
            .spawn();

        let topic = TopicId::from_bytes(*blake3::hash(format!("sigma-swarm/{swarm_passphrase}").as_bytes()).as_bytes());
        let (sender, mut receiver) = gossip.subscribe(topic, bootstrap_peers).await?.split();
        let local_skills = SkillIndex::default();
        let neighbors = Arc::new(AtomicUsize::new(0));
        let (ep, skills, count) = (endpoint.clone(), local_skills.clone(), neighbors.clone());
        let listener = tokio::spawn(async move {
            while let Some(Ok(event)) = receiver.next().await {
                match event {
                    Event::NeighborUp(_) => _ = count.fetch_add(1, Ordering::Relaxed),
                    Event::NeighborDown(_) => _ = count.fetch_sub(1, Ordering::Relaxed),
                    Event::Received(msg) => {
                        let Ok(query) = postcard::from_bytes::<GossipImpasseQuery>(&msg.content) else { continue };
                        if query.requester == ep.id()
                            || store.get_peer_reputation(query.requester.as_bytes()) < QUARANTINE
                        {
                            continue;
                        }
                        let offers = matching_skills(&skills, &query.impasse.missing_predicate);
                        if !offers.is_empty() {
                            tokio::spawn(offer_skills(ep.clone(), query.requester, offers));
                        }
                    }
                    _ => {}
                }
            }
        });
        Ok(Self { endpoint, router, sender, local_skills, neighbors, listener, downloaded_skills })
    }

    pub fn node_id(&self) -> EndpointId {
        self.endpoint.id()
    }

    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Number of current gossip neighbors.
    pub fn neighbor_count(&self) -> usize {
        self.neighbors.load(Ordering::Relaxed)
    }

    /// Offers a (validated) skill to peers that hit a matching impasse.
    pub fn publish_local_skill(&self, skill: SkillPackage) -> Result<()> {
        skill.validate().map_err(anyhow::Error::msg)?;
        let mut skills = self.local_skills.lock().expect("skill index poisoned");
        let key = skill.signature_key();
        if skills.get(&key).is_none_or(|old| old.version <= skill.version) {
            skills.insert(key, skill);
        }
        Ok(())
    }

    pub async fn broadcast_impasse(&self, impasse: ImpasseEvent) -> Result<()> {
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos() as u64;
        let query = GossipImpasseQuery { requester: self.node_id(), impasse, nonce };
        self.sender.broadcast(postcard::to_allocvec(&query)?.into()).await?;
        Ok(())
    }

    pub async fn shutdown(self) -> Result<()> {
        self.listener.abort();
        self.router.shutdown().await?;
        Ok(())
    }
}

fn matching_skills(skills: &SkillIndex, predicate: &str) -> Vec<SkillPackage> {
    let skills = skills.lock().expect("skill index poisoned");
    skills.iter().filter(|(key, _)| key.split(',').any(|p| p == predicate)).map(|(_, s)| s.clone()).collect()
}

/// Dials the requester over QUIC and pushes each skill on its own stream.
async fn offer_skills(endpoint: Endpoint, to: EndpointId, skills: Vec<SkillPackage>) {
    let push = async {
        let conn = endpoint.connect(to, SKILL_EXCHANGE_ALPN).await?;
        for skill in skills {
            let (mut send, mut recv) = conn.open_bi().await?;
            send.write_all(&postcard::to_allocvec(&DirectSkillMessage::OfferSkill(skill))?).await?;
            send.finish()?;
            recv.read_to_end(64).await?; // wait for the receiver's verdict
        }
        conn.close(0u32.into(), b"done");
        anyhow::Ok(())
    };
    if let Err(e) = push.await {
        eprintln!("sigma-net: offering skills to {} failed: {e:#}", to.fmt_short());
    }
}

/// Inbound side of `SKILL_EXCHANGE_ALPN`: bounded read, reputation gate, full validation.
struct SkillExchange {
    store: Arc<FactorStore>,
    tx: mpsc::Sender<SkillPackage>,
}

impl std::fmt::Debug for SkillExchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SkillExchange")
    }
}

impl SkillExchange {
    fn check(&self, peer: &EndpointId, bytes: &[u8]) -> Result<SkillPackage> {
        let DirectSkillMessage::OfferSkill(skill) = postcard::from_bytes(bytes).context("malformed skill message")?;
        for key in [peer.as_bytes(), &skill.author_pubkey] {
            anyhow::ensure!(self.store.get_peer_reputation(key) >= QUARANTINE, "quarantined peer");
        }
        skill.validate().map_err(anyhow::Error::msg)?;
        Ok(skill)
    }

    async fn serve(&self, conn: &Connection) -> Result<()> {
        let peer = conn.remote_id();
        while let Ok((mut send, mut recv)) = conn.accept_bi().await {
            let bytes = recv.read_to_end(MAX_SKILL_BYTES).await?;
            let verdict: &[u8] = match self.check(&peer, &bytes) {
                Ok(skill) => {
                    self.tx.send(skill).await?;
                    b"accepted"
                }
                Err(e) => {
                    eprintln!("sigma-net: rejected skill from {}: {e:#}", peer.fmt_short());
                    b"rejected"
                }
            };
            send.write_all(verdict).await?;
            send.finish()?;
        }
        Ok(())
    }
}

impl ProtocolHandler for SkillExchange {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        self.serve(&conn).await.map_err(|e| AcceptError::from_boxed(e.into()))
    }
}
