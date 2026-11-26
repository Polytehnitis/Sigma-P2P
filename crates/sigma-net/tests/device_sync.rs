//! Phone, PC and NAS pair up and keep private state in sync over iroh, offline-first, while a
//! stranger in the same gossip swarm can neither pair with a spent code nor pull any of it.

use iroh::address_lookup::memory::MemoryLookup;
use iroh::endpoint::presets;
use iroh::{Endpoint, RelayMode};
use sigma_core::sync::keys;
use sigma_core::{FactorStore, NodeIdentity, PredicateRole::*, PredicateSchema, SigmaEngine, SkillPackage};
use sigma_net::{EndpointId, SkillMeshNode};
use std::sync::Arc;
use std::time::Duration;

const SWARM: &str = "device-sync-test";

struct Device {
    node: SkillMeshNode,
    store: Arc<FactorStore>,
    id: NodeIdentity,
}

fn open_store() -> Arc<FactorStore> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("sigma-devsync-{}-{nanos}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Arc::new(FactorStore::open(dir.join("sigma_memory.redb")).unwrap())
}

async fn spawn(lookup: &MemoryLookup, store: Arc<FactorStore>, bootstrap: Vec<EndpointId>) -> SkillMeshNode {
    let builder = Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).address_lookup(lookup.clone());
    let node = SkillMeshNode::spawn_with(builder, SWARM, bootstrap, store).await.unwrap();
    lookup.add_endpoint_info(node.endpoint().addr());
    node
}

async fn device(lookup: &MemoryLookup, bootstrap: Vec<EndpointId>) -> Device {
    let store = open_store();
    let id = store.load_or_create_identity().unwrap();
    Device { node: spawn(lookup, store.clone(), bootstrap).await, store, id }
}

fn backup_skill(id: &NodeIdentity) -> SkillPackage {
    let mut s = SkillPackage {
        skill_name: "backup-needs-nas".into(),
        version: 1,
        author_pubkey: [0; 32],
        signature: vec![],
        predicates: vec![
            PredicateSchema::binary("GoalBackup", Goal),
            PredicateSchema::binary("ActionMountNAS", Action),
        ],
        factor_shape: vec![2, 2],
        factor_values: vec![1.0, 1.0, 1.0, 5.0],
    };
    id.sign_skill(&mut s);
    s
}

#[tokio::test(flavor = "multi_thread")]
async fn paired_devices_sync_offline_first_and_strangers_get_nothing() {
    let lookup = MemoryLookup::new();
    let pc = device(&lookup, vec![]).await;
    let phone = device(&lookup, vec![pc.node.node_id()]).await;
    let mut nas = device(&lookup, vec![pc.node.node_id()]).await;
    let stranger = device(&lookup, vec![pc.node.node_id()]).await;

    // Pairing: NAS and phone each join the PC with a single-use code.
    let code = pc.node.pairing_code();
    assert_eq!(nas.node.pair(&code).await.unwrap(), pc.node.node_id());
    assert!(stranger.node.pair(&code).await.is_err(), "a spent code cannot pair anyone else");
    phone.node.pair(&pc.node.pairing_code()).await.unwrap();
    assert!(pc.store.is_paired(nas.node.node_id().as_bytes()));
    // The phone learned of the NAS through the PC, without ever exchanging a code with it.
    assert!(phone.store.is_paired(nas.node.node_id().as_bytes()));
    assert_eq!(phone.store.get_peer_reputation(&nas.id.author_pubkey()), 1.0);

    // The phone learns while the NAS is offline.
    nas.node.shutdown().await.unwrap();
    let mut engine = SigmaEngine::new();
    let skill = backup_skill(&phone.id);
    engine.compile_skill(skill.clone()).unwrap();
    phone.store.save_skill(&skill).unwrap();
    phone.store.save_skill_weights("backup-needs-nas", &[1.0, 1.0, 1.0, 8.0]).unwrap();
    engine.push_goal("GoalBackup");
    engine.cognitive_cycle();
    engine.reinforce_factors(1.0);
    phone.store.save_memories(&engine).unwrap();
    phone.store.save_goal_stack(&engine.goal_stack).unwrap();
    let stranger_key = stranger.id.author_pubkey();
    phone.store.update_peer_reputation(&stranger_key, -1.0).unwrap();
    assert_eq!(phone.node.sync_now().await, 1, "only the PC is online");

    // The NAS comes back (same identity, same store) and catches up through the PC.
    nas.node = spawn(&lookup, nas.store.clone(), vec![pc.node.node_id()]).await;
    assert_eq!(nas.node.sync_now().await, 2, "phone and PC");
    let mut nas_engine = SigmaEngine::new();
    let first = tokio::time::timeout(Duration::from_secs(10), nas.node.synced.recv()).await.unwrap().unwrap();
    nas.store.refresh_engine(&mut nas_engine, &first).unwrap();
    while let Ok(report) = nas.node.synced.try_recv() {
        nas.store.refresh_engine(&mut nas_engine, &report).unwrap();
    }
    assert_eq!(nas_engine.skill_weights("backup-needs-nas").unwrap()[3], 8.0);
    assert_eq!(nas_engine.goal_stack, ["GoalBackup"]);
    assert_eq!(nas_engine.episodic.episodes.len(), 1);
    assert_eq!(nas_engine.episodic.episodes[0].reward, 1.0);
    assert_eq!(nas.store.get_peer_reputation(&stranger_key), phone.store.get_peer_reputation(&stranger_key));
    let state = |d: &Device| d.store.export_sync_state().unwrap();
    assert_eq!(state(&nas), state(&phone));
    assert_eq!(state(&nas), state(&pc));

    // The stranger shares the swarm, but cannot pull private state even if it lists the PC as
    // its own device: the PC refuses unpaired peers.
    stranger.store.pair_device(pc.node.node_id().as_bytes()).unwrap();
    assert!(stranger.node.sync_with(pc.node.node_id()).await.is_err());
    let leaked = state(&stranger);
    assert!(!leaked.registers.keys().any(|k| k.starts_with(keys::SKILL) || k == keys::GOALS));
    assert!(leaked.episodic.episodes.is_empty());
    assert!(!pc.store.is_paired(&stranger_key));
    assert_eq!(pc.store.get_peer_reputation(&stranger_key), phone.store.get_peer_reputation(&stranger_key));

    for n in [pc.node, phone.node, nas.node, stranger.node] {
        n.shutdown().await.unwrap();
    }
}
