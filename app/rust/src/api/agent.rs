//! FFI surface for the Flutter client. One agent per process; all calls are blocking
//! from Rust's side and run on flutter_rust_bridge worker threads.

use anyhow::{anyhow, Context, Result};
use sigma_core::substate::offered_to_peers;
use sigma_core::{
    crypto, language, ActionManifest, ActionSpec, ConfirmationGate, Emotion as CoreEmotion, FactorStore, ImpasseKind,
    NodeIdentity, PredicateRole, PredicateSchema, SigmaEngine, SkillPackage, QUARANTINE,
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
    /// Holds the token of the action awaiting the user's confirmation.
    gate: Mutex<ConfirmationGate>,
    /// Phrases the user taught, this device's and synced ones.
    phrases: Arc<Mutex<language::PhraseBook>>,
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

/// How the agent felt about its last cycle; drives the desktop avatar.
pub enum Emotion {
    Neutral,
    Joy,
    Distress,
    Surprise,
    Hope,
    Fear,
}

impl From<CoreEmotion> for Emotion {
    fn from(e: CoreEmotion) -> Self {
        match e {
            CoreEmotion::Neutral => Emotion::Neutral,
            CoreEmotion::Joy => Emotion::Joy,
            CoreEmotion::Distress => Emotion::Distress,
            CoreEmotion::Surprise => Emotion::Surprise,
            CoreEmotion::Hope => Emotion::Hope,
            CoreEmotion::Fear => Emotion::Fear,
        }
    }
}

/// The cycle's appraisal: desirability in [-1, 1], the rest in [0, 1].
pub struct Appraisal {
    pub desirability: f32,
    pub surprise: f32,
    pub control: f32,
    pub familiarity: f32,
    pub emotion: Emotion,
}

pub enum DecisionImpasseKind {
    /// No candidate action for the open (sub)goal.
    NoCandidates,
    /// Several candidates the knowledge does not tell apart.
    Tie,
    /// The same action again while nothing changed.
    NoChange,
    /// The best action is both strongly supported and strongly opposed.
    Conflict,
}

pub struct DecisionImpasse {
    pub kind: DecisionImpasseKind,
    pub goal: String,
    pub candidates: Vec<String>,
    /// Reasoning in a substate (or episodic recall) settled it.
    pub resolved: bool,
    pub cycles: u32,
}

pub struct CognitiveStep {
    pub goal_stack: Vec<String>,
    pub selected_action: Option<String>,
    pub beliefs: Vec<Belief>,
    pub provenance: Vec<SkillProvenance>,
    /// The predicate the agent is missing a skill for (already broadcast to the swarm).
    pub impasse: Option<String>,
    pub decision_impasse: Option<DecisionImpasse>,
    pub appraisal: Appraisal,
    /// Skills that took part in elaboration when attention is limited (empty: all).
    pub attended_skills: Vec<String>,
    /// The selected action is irreversible and waits for the user. Only the UI holds the
    /// token; pass it to `confirm_action` when the user approves.
    pub awaiting_confirmation: Option<ConfirmationRequest>,
    /// Actions released to the actuator during this step (confirmed ones included).
    pub motor_commands: Vec<String>,
    /// Actions `undo_last_action` can reverse, oldest first.
    pub undo_stack: Vec<String>,
}

pub struct ConfirmationRequest {
    pub action: String,
    pub token: String,
}

/// Safety rule for one action: what reverses it, or that it needs confirmation every time.
pub struct ActionRule {
    pub action: String,
    pub reverse: Option<String>,
    pub irreversible: bool,
}

pub struct UndoResult {
    pub action: String,
    /// The reverse action sent to the actuator, or `None` if the action had not been taken yet.
    pub reverse_action: Option<String>,
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
    // Chunking: what reasoning in a substate concludes becomes a rule signed by this node.
    engine.chunk_author = Some(identity.clone());
    let restored = store.restore_engine(&mut engine).map_err(err)?;
    // Built-in knowledge (the smem-question-answering Soar agent's facts), signed by this node
    // at every start: never stored, synced or offered to peers.
    engine.install_starting_knowledge(&identity).map_err(err)?;
    engine.goal_stack = store.load_goal_stack().map_err(err)?;
    engine.preferences = store.load_preferences().map_err(err)?;
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let mut node = rt.block_on(SkillMeshNode::spawn(&swarm_passphrase, peers, store.clone()))?;
    for skill in restored.iter().filter(|s| offered_to_peers(s)) {
        node.publish_local_skill(skill.clone())?;
    }
    let mut downloads = std::mem::replace(&mut node.downloaded_skills, tokio::sync::mpsc::channel(1).1);
    let mut synced = std::mem::replace(&mut node.synced, tokio::sync::mpsc::unbounded_channel().1);
    let (engine, node) = (Arc::new(Mutex::new(engine)), Arc::new(node));
    let (e, n, s) = (engine.clone(), node.clone(), store.clone());
    rt.spawn(async move {
        while let Some(skill) = downloads.recv().await {
            if e.lock().unwrap().compile_skill(skill.clone()).is_ok() && s.save_skill(&skill).is_ok() {
                let _ = n.publish_local_skill(skill);
            }
        }
    });
    // Changes synced from the user's other devices go straight into the live engine.
    let phrases = Arc::new(Mutex::new(store.load_phrases().map_err(err)?));
    let (e, n, s, p) = (engine.clone(), node.clone(), store.clone(), phrases.clone());
    rt.spawn(async move {
        while let Some(report) = synced.recv().await {
            if report.changed.iter().any(|k| k.starts_with(sigma_core::sync::keys::PHRASE)) {
                if let Ok(book) = s.load_phrases() {
                    *p.lock().unwrap() = book;
                }
            }
            let compiled = s.refresh_engine(&mut e.lock().unwrap(), &report).unwrap_or_default();
            for skill in compiled.into_iter().filter(offered_to_peers) {
                let _ = n.publish_local_skill(skill);
            }
        }
    });
    let gate = Mutex::new(ConfirmationGate::new());
    let agent = AGENT.get_or_init(|| Agent { rt, store, identity, engine, node, gate, phrases });
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
        let out = engine.cognitive_cycle();
        // A learned chunk is kept and follows the user to their paired devices. A variablized
        // one names none of the user's own things, so it is offered to the swarm too; a ground
        // one describes the user's own situations and is not.
        if let Some(chunk) = &out.chunk {
            a.store.save_skill(chunk).map_err(err)?;
            if offered_to_peers(chunk) {
                a.node.publish_local_skill(chunk.clone()).map_err(err)?;
            }
        }
        // Persisted every step so paired devices see the latest memories and goals.
        a.store.save_memories(&engine).map_err(err)?;
        a.store.save_goal_stack(&engine.goal_stack).map_err(err)?;
        let mut gate = a.gate.lock().unwrap();
        // A fresh single-use token per step; the previous step's prompt is superseded.
        gate.revoke();
        let confirmation = gate.issue(&engine).map(|(p, token)| ConfirmationRequest { action: p.action, token });
        (out, confirmation, engine.take_motor_commands(), engine.undo_stack())
    };
    let (out, awaiting_confirmation, motor_commands, undo_stack) = out;
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
        decision_impasse: out.decision_impasse.map(|d| DecisionImpasse {
            kind: match d.kind {
                ImpasseKind::None => DecisionImpasseKind::NoCandidates,
                ImpasseKind::Tie => DecisionImpasseKind::Tie,
                ImpasseKind::NoChange => DecisionImpasseKind::NoChange,
                ImpasseKind::Conflict => DecisionImpasseKind::Conflict,
            },
            goal: d.goal,
            candidates: d.candidates,
            resolved: d.resolved,
            cycles: d.cycles,
        }),
        appraisal: Appraisal {
            desirability: out.appraisal.desirability,
            surprise: out.appraisal.surprise,
            control: out.appraisal.control,
            familiarity: out.appraisal.familiarity,
            emotion: out.appraisal.emotion.into(),
        },
        attended_skills: out.attended_skills,
        awaiting_confirmation,
        motor_commands,
        undo_stack,
    })
}

/// Releases the action awaiting confirmation. `token` must be the one the last step showed
/// the user; a token works once. Returns the action now sent to the actuator.
pub fn confirm_action(token: String) -> Result<String> {
    let a = agent()?;
    let mut engine = a.engine.lock().unwrap();
    let action = a.gate.lock().unwrap().confirm(&mut engine, &token).map_err(err)?;
    engine.take_motor_commands();
    Ok(action)
}

/// Drops the action awaiting confirmation; it is never taken.
pub fn cancel_pending_action() -> Result<Option<String>> {
    let a = agent()?;
    a.gate.lock().unwrap().revoke();
    Ok(a.engine.lock().unwrap().cancel_pending_action().map(|p| p.action))
}

/// Reverses the most recent undoable action: working memory and goals go back to when it
/// was chosen, and its reverse action is sent to the actuator.
pub fn undo_last_action() -> Result<UndoResult> {
    let a = agent()?;
    let mut engine = a.engine.lock().unwrap();
    let undone = engine.undo().map_err(err)?;
    engine.take_motor_commands();
    a.store.save_goal_stack(&engine.goal_stack).map_err(err)?;
    Ok(UndoResult { action: undone.action, reverse_action: undone.reverse_action })
}

/// This node's action safety rules.
pub fn action_rules() -> Result<Vec<ActionRule>> {
    let m = agent()?.store.load_action_manifest().map_err(err)?;
    Ok(m.map(|m| m.specs)
        .unwrap_or_default()
        .into_iter()
        .map(|s| ActionRule { action: s.action, reverse: s.reverse, irreversible: s.irreversible })
        .collect())
}

/// Replaces this node's action safety rules (signed by this node and persisted).
pub fn set_action_rules(rules: Vec<ActionRule>) -> Result<()> {
    let a = agent()?;
    let version = a.store.load_action_manifest().map_err(err)?.map_or(1, |m| m.version + 1);
    let specs = rules
        .into_iter()
        .map(|r| ActionSpec {
            action: r.action.trim().into(),
            reverse: r.reverse.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()),
            irreversible: r.irreversible,
        })
        .collect();
    let manifest = ActionManifest::signed(&a.identity, version, specs);
    a.engine.lock().unwrap().apply_action_manifest(&manifest).map_err(err)?;
    a.store.save_action_manifest(&manifest).map_err(err)
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

/// A single-use code (valid 10 minutes) that another of the user's devices can pair with.
pub fn create_pairing_code() -> Result<String> {
    Ok(agent()?.node.pairing_code())
}

/// Pairs with the device that showed `code` and syncs with it. Returns its author pubkey hex.
/// From then on memories, reputations, learned weights and goals stay in sync, end-to-end
/// encrypted and only between paired devices.
pub fn pair_device(code: String) -> Result<String> {
    let a = agent()?;
    let device = a.rt.block_on(a.node.pair(&code))?;
    Ok(crypto::to_hex(device.as_bytes()))
}

/// Syncs with every reachable paired device now (this also happens every 30 s).
/// Returns how many were reached.
pub fn sync_now() -> Result<u32> {
    let a = agent()?;
    Ok(a.rt.block_on(a.node.sync_now()) as u32)
}

/// Author pubkeys (hex) of the user's other paired devices.
pub fn paired_devices() -> Result<Vec<String>> {
    Ok(agent()?.store.paired_devices().map_err(err)?.iter().map(|d| crypto::to_hex(d)).collect())
}

/// Unpairs a device on all of the user's devices; it stops receiving private state.
pub fn unpair_device(hex_pubkey: String) -> Result<()> {
    let key = crypto::pubkey_from_hex(&hex_pubkey).map_err(err)?;
    agent()?.store.unpair_device(&key).map_err(err)
}

/// What an utterance asks for. Language never confirms an action: `ConfirmRequested` only
/// tells the UI to point at its own Confirm button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntentKind {
    SetGoal,
    Observe,
    Think,
    Feedback,
    Undo,
    Cancel,
    ConfirmRequested,
    Teach,
    Explain,
}

/// One reading of an utterance. `predicate` is set for `SetGoal` and `Observe`; `value` is the
/// observed value (`Observe`) or 1 for good and 0 for bad (`Feedback`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpokenIntent {
    pub kind: IntentKind,
    pub predicate: Option<String>,
    pub value: u32,
}

pub enum InterpretationRoute {
    /// A phrase the user taught.
    Taught,
    /// Command or predicate words matched directly.
    FastPath,
    /// The naive-Bayes posterior settled it.
    Inference,
    /// Nothing was confident enough; see `alternatives`.
    NoMatch,
}

pub struct Interpretation {
    pub normalized: String,
    pub intent: Option<SpokenIntent>,
    /// In [0, 1], the recognizer's confidence included.
    pub confidence: f32,
    pub route: InterpretationRoute,
    /// Other likely readings, best first, for "did you mean…?".
    pub alternatives: Vec<ScoredIntent>,
}

pub struct ScoredIntent {
    pub intent: SpokenIntent,
    pub score: f32,
}

pub struct TaughtPhrase {
    pub phrase: String,
    pub intent: SpokenIntent,
}

impl From<language::Intent> for SpokenIntent {
    fn from(i: language::Intent) -> Self {
        use language::Intent as I;
        let (kind, predicate, value) = match i {
            I::SetGoal(g) => (IntentKind::SetGoal, Some(g), 1),
            I::Observe { predicate, value } => (IntentKind::Observe, Some(predicate), value as u32),
            I::Think => (IntentKind::Think, None, 0),
            I::Feedback(good) => (IntentKind::Feedback, None, u32::from(good)),
            I::Undo => (IntentKind::Undo, None, 0),
            I::Cancel => (IntentKind::Cancel, None, 0),
            I::ConfirmRequested => (IntentKind::ConfirmRequested, None, 0),
            I::Teach => (IntentKind::Teach, None, 0),
            I::Explain => (IntentKind::Explain, None, 0),
        };
        SpokenIntent { kind, predicate, value }
    }
}

impl TryFrom<SpokenIntent> for language::Intent {
    type Error = anyhow::Error;

    fn try_from(i: SpokenIntent) -> Result<Self> {
        use language::Intent as I;
        let predicate =
            || i.predicate.clone().filter(|p| !p.trim().is_empty()).ok_or_else(|| anyhow!("predicate missing"));
        Ok(match i.kind {
            IntentKind::SetGoal => I::SetGoal(predicate()?),
            IntentKind::Observe if i.value < 2 => I::Observe { predicate: predicate()?, value: i.value as usize },
            IntentKind::Observe => return Err(anyhow!("observed value must be 0 or 1")),
            IntentKind::Think => I::Think,
            IntentKind::Feedback => I::Feedback(i.value > 0),
            IntentKind::Undo => I::Undo,
            IntentKind::Cancel => I::Cancel,
            IntentKind::ConfirmRequested => I::ConfirmRequested,
            IntentKind::Teach => I::Teach,
            IntentKind::Explain => I::Explain,
        })
    }
}

/// Interprets what the user said (or typed) against the agent's own predicates, on this
/// device. `recognition_confidence` is the speech recognizer's confidence; 1 for typed text.
/// Nothing is applied: the app acts on the returned intent.
pub fn interpret_utterance(text: String, recognition_confidence: f32) -> Result<Interpretation> {
    let a = agent()?;
    let predicates: Vec<PredicateSchema> = a
        .engine
        .lock()
        .unwrap()
        .variables()
        .iter()
        .map(|v| v.schema.clone())
        .filter(|p| !p.name.starts_with('@'))
        .collect();
    let i = language::interpret(&text, recognition_confidence, &predicates, &a.phrases.lock().unwrap());
    Ok(Interpretation {
        normalized: i.normalized,
        intent: i.intent.map(Into::into),
        confidence: i.confidence,
        route: match i.route {
            language::Route::Taught => InterpretationRoute::Taught,
            language::Route::FastPath => InterpretationRoute::FastPath,
            language::Route::Inference => InterpretationRoute::Inference,
            language::Route::NoMatch => InterpretationRoute::NoMatch,
        },
        alternatives: i
            .alternatives
            .into_iter()
            .map(|(intent, score)| ScoredIntent { intent: intent.into(), score })
            .collect(),
    })
}

/// Teaches a phrase so it means `intent` from now on, on all of the user's paired devices.
/// Returns the phrase as it will be matched (normalized).
pub fn teach_phrase(phrase: String, intent: SpokenIntent) -> Result<String> {
    let a = agent()?;
    let intent = language::Intent::try_from(intent)?;
    let key = a.store.save_phrase(&phrase, Some(&intent)).map_err(err)?;
    a.phrases.lock().unwrap().phrases.insert(key.clone(), intent);
    Ok(key)
}

/// Forgets a taught phrase, on all of the user's paired devices.
pub fn forget_phrase(phrase: String) -> Result<()> {
    let a = agent()?;
    a.store.save_phrase(&phrase, None).map_err(err)?;
    a.phrases.lock().unwrap().forget(&phrase);
    Ok(())
}

pub fn taught_phrases() -> Result<Vec<TaughtPhrase>> {
    let book = agent()?.phrases.lock().unwrap().clone();
    Ok(book.phrases.into_iter().map(|(phrase, intent)| TaughtPhrase { phrase, intent: intent.into() }).collect())
}

/// Uncertain evidence for the next cycle only: `predicate` is true with `probability`. Used
/// for what the user said when recognition was unsure, instead of a hard observation.
pub fn perceive(predicate: String, probability: f32) -> Result<()> {
    let a = agent()?;
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(anyhow!("probability must be in [0, 1]"));
    }
    let mut engine = a.engine.lock().unwrap();
    if engine.variables().iter().any(|v| v.schema.name == predicate && v.schema.role == PredicateRole::Action) {
        return Err(anyhow!("actions cannot be perceived"));
    }
    let p = PredicateSchema::binary(&predicate, PredicateRole::Context);
    if engine.register_predicate(&p).is_ok() {
        let _ = a.store.save_predicate(&p);
    }
    engine.perceive(&predicate, &[1.0 - probability, probability]).map_err(err)
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // flutter_rust_bridge turns on RUST_BACKTRACE, so every returned `anyhow::Error` would
    // capture a backtrace. On Linux the unwinder resolves to the copy bundled in
    // libflutter_linux_gtk.so, which segfaults walking these frames: any FFI error (a bad
    // pairing code, a malformed pubkey) killed the app. Panics keep their backtraces.
    if std::env::var_os("RUST_LIB_BACKTRACE").is_none() {
        std::env::set_var("RUST_LIB_BACKTRACE", "0");
    }
    flutter_rust_bridge::setup_default_user_utils();
}
