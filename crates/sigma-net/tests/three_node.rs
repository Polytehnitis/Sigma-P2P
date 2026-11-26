//! Node A (requester) composes a two-skill chain downloaded from Node B and Node C
//! through two sequential gossip impasses.

use iroh::address_lookup::memory::MemoryLookup;
use iroh::endpoint::presets;
use iroh::{Endpoint, RelayMode};
use sigma_core::{FactorStore, NodeIdentity, PredicateRole::*, PredicateSchema, SigmaEngine, SkillPackage};
use sigma_net::{EndpointId, SkillMeshNode};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

const SWARM: &str = "three-node-test";

async fn node(lookup: &MemoryLookup, bootstrap: Vec<EndpointId>) -> (SkillMeshNode, NodeIdentity) {
    let dir = std::env::temp_dir().join(format!("sigma-net-{}", rand_suffix()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = Arc::new(FactorStore::open(dir.join("sigma_memory.redb")).unwrap());
    let identity = store.load_or_create_identity().unwrap();
    let builder = Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled).address_lookup(lookup.clone());
    let node = SkillMeshNode::spawn_with(builder, SWARM, bootstrap, store).await.unwrap();
    assert_eq!(node.node_id().as_bytes(), &identity.author_pubkey(), "one key for transport and authorship");
    lookup.add_endpoint_info(node.endpoint().addr());
    (node, identity)
}

fn rand_suffix() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
}

fn signed(id: &NodeIdentity, name: &str, predicates: Vec<PredicateSchema>, factor_values: Vec<f32>) -> SkillPackage {
    let mut s = SkillPackage {
        skill_name: name.into(),
        version: 1,
        author_pubkey: [0; 32],
        signature: vec![],
        factor_shape: vec![2; predicates.len()],
        predicates,
        factor_values,
    };
    id.sign_skill(&mut s);
    s
}

async fn wait_for(what: &str, cond: impl Fn() -> bool) {
    timeout(Duration::from_secs(20), async {
        while !cond() {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
}

/// Broadcasts the engine's impasse until a skill named `want` arrives, then compiles it.
async fn resolve_impasse(a: &mut SkillMeshNode, engine: &mut SigmaEngine, want: &str) {
    let impasse = engine.cognitive_cycle().impasse.expect("impasse expected");
    timeout(Duration::from_secs(30), async {
        loop {
            a.broadcast_impasse(impasse.clone()).await.unwrap();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
            while let Ok(Some(skill)) = tokio::time::timeout_at(deadline, a.downloaded_skills.recv()).await {
                let name = skill.skill_name.clone();
                engine.compile_skill(skill).unwrap();
                if name == want {
                    return;
                }
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("never received {want}"));
}

#[tokio::test(flavor = "multi_thread")]
async fn requester_composes_skills_from_two_peers() {
    let lookup = MemoryLookup::new();
    let (mut a, _) = node(&lookup, vec![]).await;
    let (b, b_id) = node(&lookup, vec![a.node_id()]).await;
    let (c, c_id) = node(&lookup, vec![a.node_id()]).await;

    let goal = PredicateSchema::binary("GoalBackup", Goal);
    let sub = PredicateSchema::binary("SubgoalMountNAS", Subgoal);
    let mount = PredicateSchema::binary("ActionMountNAS", Action);
    let sync = PredicateSchema::binary("ActionSyncFiles", Action);
    let skill1 = signed(
        &b_id,
        "backup-needs-nas",
        vec![goal, sub.clone(), mount],
        vec![1.0, 1.0, 1.0, 1.0, 0.1, 5.0, 10.0, 2.0],
    );
    let skill2 = signed(&c_id, "sync-on-mounted-nas", vec![sub, sync], vec![5.0, 0.1, 1.0, 10.0]);
    b.publish_local_skill(skill1).unwrap();
    c.publish_local_skill(skill2).unwrap();

    wait_for("gossip neighbors", || a.neighbor_count() >= 2).await;

    let mut engine = SigmaEngine::new();
    engine.push_goal("GoalBackup");

    // Impasse 1: nothing known about the goal -> Node B answers with Skill 1.
    assert_eq!(engine.detect_structural_impasse().unwrap().missing_predicate, "GoalBackup");
    resolve_impasse(&mut a, &mut engine, "backup-needs-nas").await;

    // Impasse 2: Skill 1 leaves the SubgoalMountNAS -> Action bridge missing -> Node C answers.
    assert_eq!(engine.detect_structural_impasse().unwrap().missing_predicate, "SubgoalMountNAS");
    resolve_impasse(&mut a, &mut engine, "sync-on-mounted-nas").await;

    // The downloaded chain solves the goal: mount first, then sync.
    let step = engine.cognitive_cycle();
    assert_eq!(step.impasse, None);
    assert_eq!(step.goal_stack, ["GoalBackup", "SubgoalMountNAS"]);
    assert_eq!(step.selected_action.as_deref(), Some("ActionMountNAS"));
    let authors: Vec<_> = step.provenance.iter().map(|p| p.1).collect();
    assert!(authors.contains(&b_id.author_pubkey()) && authors.contains(&c_id.author_pubkey()));
    engine.set_evidence("SubgoalMountNAS", 1).unwrap();
    assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("ActionSyncFiles"));

    for n in [a, b, c] {
        n.shutdown().await.unwrap();
    }
}
