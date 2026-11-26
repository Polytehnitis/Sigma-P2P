//! Two nodes find each other by EndpointId alone over n0's public relays and DNS
//! discovery (`presets::N0`), then exchange a skill through a gossip impasse.
//! Needs internet access, so it is opt-in: `cargo test -p sigma-net -- --ignored`.

use sigma_core::{FactorStore, NodeIdentity, PredicateRole::*, PredicateSchema, SigmaEngine, SkillPackage};
use sigma_net::SkillMeshNode;
use std::sync::Arc;
use std::time::Duration;

async fn node(bootstrap: Vec<sigma_net::EndpointId>, swarm: &str) -> (SkillMeshNode, NodeIdentity) {
    let dir = std::env::temp_dir().join(format!("sigma-relay-{}", rand_id()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = Arc::new(FactorStore::open(dir.join("sigma_memory.redb")).unwrap());
    let id = store.load_or_create_identity().unwrap();
    let node = SkillMeshNode::spawn(swarm, bootstrap, store).await.unwrap();
    node.endpoint().online().await;
    (node, id)
}

fn rand_id() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs internet access to n0 relays and DNS discovery"]
async fn skill_exchange_over_public_relays() {
    let swarm = format!("relay-test-{}", rand_id());
    let (b, b_id) = node(vec![], &swarm).await;
    let mut skill = SkillPackage {
        skill_name: "tidy".into(),
        version: 1,
        author_pubkey: [0; 32],
        signature: vec![],
        predicates: vec![PredicateSchema::binary("GoalTidy", Goal), PredicateSchema::binary("ActionSort", Action)],
        factor_shape: vec![2, 2],
        factor_values: vec![1.0, 1.0, 1.0, 10.0],
    };
    b_id.sign_skill(&mut skill);
    b.publish_local_skill(skill).unwrap();

    let (mut a, _) = node(vec![b.node_id()], &swarm).await;
    let mut engine = SigmaEngine::new();
    engine.push_goal("GoalTidy");
    let impasse = engine.cognitive_cycle().impasse.unwrap();

    let got = tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            if a.neighbor_count() > 0 {
                a.broadcast_impasse(impasse.clone()).await.unwrap();
            }
            if let Ok(Some(s)) = tokio::time::timeout(Duration::from_secs(2), a.downloaded_skills.recv()).await {
                return s;
            }
        }
    })
    .await
    .expect("no skill over the public relays within 60s");
    engine.compile_skill(got).unwrap();
    assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("ActionSort"));
    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
}
