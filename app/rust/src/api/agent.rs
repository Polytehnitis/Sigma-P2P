//! FFI surface for the Flutter client. One agent per process; all calls are blocking
//! from Rust's side and run on flutter_rust_bridge worker threads.

use anyhow::{anyhow, Context, Result};
use sigma_core::{
    crypto, FactorStore, NodeIdentity, PredicateRole, PredicateSchema, SigmaEngine, SkillPackage, QUARANTINE,
};
use sigma_net::{EndpointId, SkillMeshNode};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::runtime::Runtime;

struct Agent {
    rt: Runtime,
    store: Arc<FactorStore>,
    identity: NodeIdentity,
    engine: Arc<Mutex<SigmaEngine>>,
    node: Arc<SkillMeshNode>,
}

static AGENT: OnceLock<Agent> = OnceLock::new();

fn agent() -> Result<&'static Agent> {
    AGENT.get().ok_or_else(|| anyhow!("call init_agent first"))
}

fn err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{e}")
}

pub enum Role {
    Context,
    Goal,
    Subgoal,
    Action,
}

impl From<Role> for PredicateRole {
    fn from(r: Role) -> Self {
        match r {
            Role::Context => PredicateRole::Context,
            Role::Goal => PredicateRole::Goal,
            Role::Subgoal => PredicateRole::Subgoal,
            Role::Action => PredicateRole::Action,
        }
    }
}

pub struct AgentInfo {
    pub endpoint_id: String,
    pub author_pubkey: String,
    pub restored_skills: u32,
}

pub struct Evidence {
    pub predicate: String,
    pub value: u32,
}

pub struct Belief {
    pub predicate: String,
    pub distribution: Vec<f32>,
}

pub struct SkillProvenance {
    pub skill_name: String,
    pub author_pubkey: String,
    pub reputation: f32,
}

pub struct CognitiveStep {
    pub goal_stack: Vec<String>,
    pub selected_action: Option<String>,
    pub beliefs: Vec<Belief>,
    pub provenance: Vec<SkillProvenance>,
    /// The predicate the agent is missing a skill for (already broadcast to the swarm).
    pub impasse: Option<String>,
}

pub struct FeedbackResult {
    pub reputations: Vec<SkillProvenance>,
    pub evicted_skills: Vec<String>,
}

/// One predicate of a demonstration: its role and the value it took.
pub struct PredicateDemo {
    pub name: String,
    pub role: Role,
    pub value: u32,
}

/// Opens `<storage_dir>/sigma_memory.redb`, restores skills and joins the swarm.
pub fn init_agent(storage_dir: String, swarm_passphrase: String, bootstrap_peers: Vec<String>) -> Result<AgentInfo> {
    if let Some(a) = AGENT.get() {
        return Ok(info(a, 0));
    }
    let peers = bootstrap_peers
        .iter()
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.trim().parse::<EndpointId>().with_context(|| format!("bad peer id {p}")))
        .collect::<Result<Vec<_>>>()?;
    std::fs::create_dir_all(&storage_dir)?;
    let store = Arc::new(FactorStore::open(std::path::Path::new(&storage_dir).join("sigma_memory.redb")).map_err(err)?);
    let identity = store.load_or_create_identity().map_err(err)?;
    let mut engine = SigmaEngine::new();
    let restored = store.restore_engine(&mut engine).map_err(err)?;
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let mut node = rt.block_on(SkillMeshNode::spawn(&swarm_passphrase, peers, store.clone()))?;
    for skill in &restored {
        node.publish_local_skill(skill.clone())?;
    }
    let mut downloads = std::mem::replace(&mut node.downloaded_skills, tokio::sync::mpsc::channel(1).1);
    let (engine, node) = (Arc::new(Mutex::new(engine)), Arc::new(node));
    let (e, n, s) = (engine.clone(), node.clone(), store.clone());
    rt.spawn(async move {
        while let Some(skill) = downloads.recv().await {
            if e.lock().unwrap().compile_skill(skill.clone()).is_ok() && s.save_skill(&skill).is_ok() {
                let _ = n.publish_local_skill(skill);
            }
        }
    });
    let agent = AGENT.get_or_init(|| Agent { rt, store, identity, engine, node });
    Ok(info(agent, restored.len() as u32))
}

fn info(a: &Agent, restored_skills: u32) -> AgentInfo {
    AgentInfo {
        endpoint_id: a.node.node_id().to_string(),
        author_pubkey: crypto::to_hex(&a.identity.author_pubkey()),
        restored_skills,
    }
}

/// Applies observations, pushes goals and runs one cognitive cycle. An impasse is
/// broadcast to the swarm automatically; downloaded skills appear on later steps.
pub fn run_cognitive_step(evidence: Vec<Evidence>, active_goals: Vec<String>) -> Result<CognitiveStep> {
    let a = agent()?;
    let out = {
        let mut engine = a.engine.lock().unwrap();
        for ev in evidence {
            let p = PredicateSchema::binary(&ev.predicate, PredicateRole::Context);
            if engine.register_predicate(&p).is_ok() {
                let _ = a.store.save_predicate(&p);
            }
            engine.set_evidence(&ev.predicate, ev.value as usize).map_err(err)?;
        }
        for goal in active_goals.iter().filter(|g| !g.trim().is_empty()) {
            engine.push_goal(goal.trim());
        }
        engine.cognitive_cycle()
    };
    if let Some(impasse) = out.impasse.clone() {
        let node = a.node.clone();
        a.rt.spawn(async move { node.broadcast_impasse(impasse).await });
    }
    Ok(CognitiveStep {
        goal_stack: out.goal_stack,
        selected_action: out.selected_action,
        beliefs: out.beliefs.into_iter().map(|(predicate, distribution)| Belief { predicate, distribution }).collect(),
        provenance: out.provenance.into_iter().map(|(name, author)| provenance(a, name, author)).collect(),
        impasse: out.impasse.map(|i| i.missing_predicate),
    })
}

fn provenance(a: &Agent, skill_name: String, author: [u8; 32]) -> SkillProvenance {
    SkillProvenance {
        skill_name,
        author_pubkey: crypto::to_hex(&author),
        reputation: a.store.get_peer_reputation(&author),
    }
}

/// Reinforces the last decision's chain (reward in [-1, 1]), updates author reputation
/// and evicts skills whose author or utility fell below the quarantine threshold.
pub fn provide_feedback(reward: f32) -> Result<FeedbackResult> {
    let a = agent()?;
    let me = a.identity.author_pubkey();
    let mut engine = a.engine.lock().unwrap();
    let mut per_author = BTreeMap::new();
    for (author, delta) in engine.reinforce_factors(reward) {
        per_author.insert(author, delta);
    }
    let mut reputations = Vec::new();
    for (author, delta) in per_author {
        if author != me {
            a.store.update_peer_reputation(&author, delta).map_err(err)?;
        }
        reputations.push(provenance(a, String::new(), author));
    }
    for skill in engine.export_all_skills() {
        if let Some(w) = engine.skill_weights(&skill.skill_name) {
            let _ = a.store.save_skill_weights(&skill.skill_name, &w);
        }
    }
    a.store.save_memories(&engine).map_err(err)?;
    let evicted = engine
        .evict_skills(|f| f.utility_score < QUARANTINE || a.store.get_peer_reputation(&f.author_pubkey) < QUARANTINE);
    for skill in &evicted {
        a.store.delete_skill(&skill.skill_name).map_err(err)?;
    }
    Ok(FeedbackResult { reputations, evicted_skills: evicted.into_iter().map(|s| s.skill_name).collect() })
}

/// Turns a demonstration into a skill signed by this node, persists it and offers it to the swarm.
pub fn teach_and_share_skill(skill_name: String, predicates: Vec<PredicateDemo>) -> Result<String> {
    let a = agent()?;
    let demo: Vec<usize> = predicates.iter().map(|p| p.value as usize).collect();
    let schemas: Vec<PredicateSchema> =
        predicates.into_iter().map(|p| PredicateSchema::binary(&p.name, p.role.into())).collect();
    let skill: SkillPackage = a
        .engine
        .lock()
        .unwrap()
        .learn_new_skill_from_experience(&a.identity, &skill_name, schemas.clone(), &demo)
        .map_err(err)?;
    for p in &schemas {
        a.store.save_predicate(p).map_err(err)?;
    }
    a.store.save_skill(&skill).map_err(err)?;
    a.node.publish_local_skill(skill.clone())?;
    Ok(format!("{} v{}", skill.skill_name, skill.version))
}

/// Fully trusts the author key of one of the user's own devices.
pub fn trust_peer_author(hex_pubkey: String) -> Result<()> {
    let key = crypto::pubkey_from_hex(&hex_pubkey).map_err(err)?;
    agent()?.store.set_trusted_peer(&key).map_err(err)
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    flutter_rust_bridge::setup_default_user_utils();
}
