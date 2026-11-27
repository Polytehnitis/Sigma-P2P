//! State + action transition models, perception models and backed-up (Q-style) utilities.

use super::*;
use crate::graph_memory::{transition_name, LearnedModels};
use crate::tests::signed;
use std::collections::BTreeMap;
use PredicateRole::*;

fn bin(name: &str, role: PredicateRole) -> PredicateSchema {
    PredicateSchema::binary(name, role)
}

/// `Toggle` is the clear choice for `GoalLamp`; G needs the subgoal `Light`, and no skill says
/// how to bring `Light` about.
fn lamp(id: &NodeIdentity) -> SigmaEngine {
    let mut engine = SigmaEngine::new();
    let lamp = signed(id, "lamp", vec![bin("GoalLamp", Goal), bin("Toggle", Action)], vec![1.0, 1.0, 1.0, 10.0]);
    let need = signed(id, "g-needs-light", vec![bin("G", Goal), bin("Light", Subgoal)], vec![1.0, 1.0, 1.0, 10.0]);
    engine.compile_skill(lamp).unwrap();
    engine.compile_skill(need).unwrap();
    engine
}

/// Toggles the light `n` times from each state: it flips.
fn toggle_experience(engine: &mut SigmaEngine, n: usize) {
    for i in 0..2 * n {
        let before = i % 2;
        engine.set_evidence("Light", before).unwrap();
        engine.push_goal("GoalLamp");
        assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("Toggle"));
        engine.goal_stack.clear();
        engine.set_evidence("Light", 1 - before).unwrap();
        engine.cognitive_cycle();
    }
}

fn p_next(t: &[f32], now: usize) -> f32 {
    // Row-major [action, now, next]; the row of the action taken.
    let row = 4 + 2 * now;
    t[row + 1] / (t[row] + t[row + 1])
}

#[test]
fn transition_models_condition_on_the_current_state() {
    let id = NodeIdentity::generate();
    let mut engine = lamp(&id);
    toggle_experience(&mut engine, 4);

    // An action-only model cannot tell what Toggle does: half the time Light followed.
    let key = ("Toggle".to_string(), "Light".to_string());
    let e = &engine.models.effects[&key];
    assert!((e[3] / (e[2] + e[3]) - 0.5).abs() < 0.2, "{e:?}");
    // The state + action model can: Toggle flips Light.
    let t = &engine.models.transitions[&key];
    assert!(p_next(t, 0) > 0.8 && p_next(t, 1) < 0.2, "{t:?}");
    assert!(engine
        .memory_factors()
        .iter()
        .any(|f| engine.vars[*f.vars.last().unwrap()].schema.name == transition_name("Light", "Toggle")));

    // G needs Light, which is off, and no skill says how: a none impasse. Imagining Toggle
    // with the transition model shows Light coming on, so the substate selects it.
    engine.set_evidence("Light", 0).unwrap();
    engine.push_goal("G");
    let mut picked = None;
    for _ in 0..crate::substate::MAX_HORIZON {
        let out = engine.cognitive_cycle();
        assert_eq!(out.decision_impasse.as_ref().map(|d| d.kind), Some(ImpasseKind::None));
        if out.selected_action.is_some() {
            picked = out.selected_action;
            break;
        }
    }
    assert_eq!(picked.as_deref(), Some("Toggle"));
}

#[test]
fn perception_models_let_observations_inform_hidden_state() {
    let mut engine = SigmaEngine::new();
    for p in ["Raining", "WetGrass"] {
        engine.register_predicate(&bin(p, Context)).unwrap();
    }
    assert!(engine.declare_perception_model("WetGrass", "Nope").is_err());
    engine.declare_perception_model("WetGrass", "Raining").unwrap();
    // Untrained, it says nothing.
    engine.set_evidence("WetGrass", 1).unwrap();
    assert!((engine.posterior("Raining").unwrap()[1] - 0.5).abs() < 1e-4);

    // Experience: when it rains the grass is wet, otherwise mostly dry (once seen as a percept).
    for i in 0..12 {
        let rain = i % 2;
        engine.set_evidence("Raining", rain).unwrap();
        if i == 3 {
            engine.clear_evidence("WetGrass");
            engine.perceive("WetGrass", &[0.1, 0.9]).unwrap();
        } else {
            engine.set_evidence("WetGrass", rain).unwrap();
        }
        engine.cognitive_cycle();
    }
    let t = &engine.models.perceptions[&("WetGrass".to_string(), "Raining".to_string())];
    assert!(t[3] / (t[2] + t[3]) > 0.8 && t[1] / (t[0] + t[1]) < 0.2, "{t:?}");

    // Now the hidden state is inferred from the observation alone, through the graph.
    engine.clear_evidence("Raining");
    engine.set_evidence("WetGrass", 1).unwrap();
    assert!(engine.posterior("Raining").unwrap()[1] > 0.8);
    engine.set_evidence("WetGrass", 0).unwrap();
    assert!(engine.posterior("Raining").unwrap()[1] < 0.2);
    // A soft percept is weighed, not rounded.
    engine.clear_evidence("WetGrass");
    engine.perceive("WetGrass", &[0.5, 0.5]).unwrap();
    assert!((engine.posterior("Raining").unwrap()[1] - 0.5).abs() < 0.05);
}

/// G at stage 0 calls for A, at stage 1 for B; only finishing (B) is rewarded.
fn staged(id: &NodeIdentity) -> SigmaEngine {
    let mut engine = SigmaEngine::new();
    let at = |stage_one: bool| {
        // [G, Stage, X]: X is wanted with G at the given stage.
        let mut t = vec![1.0; 8];
        t[if stage_one { 7 } else { 5 }] = 10.0;
        t
    };
    let a = signed(id, "a", vec![bin("G", Goal), bin("Stage", Context), bin("A", Action)], at(false));
    let b = signed(id, "b", vec![bin("G", Goal), bin("Stage", Context), bin("B", Action)], at(true));
    engine.compile_skill(a).unwrap();
    engine.compile_skill(b).unwrap();
    engine
}

#[test]
fn utilities_are_backed_up_from_later_rewards() {
    let id = NodeIdentity::generate();
    let mut engine = staged(&id);
    let s0 = "G|Stage=0".to_string();
    let s1 = "G|Stage=1".to_string();
    for _ in 0..4 {
        engine.goal_stack = vec!["G".into()];
        engine.set_evidence("Stage", 0).unwrap();
        assert_eq!(engine.state_key(), s0);
        assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("A"));
        engine.set_evidence("Stage", 1).unwrap();
        assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("B"));
        engine.reinforce_factors(1.0);
        engine.goal_stack.clear();
        engine.cognitive_cycle(); // terminal: backs up B's reward
    }
    let q = |e: &SigmaEngine, a: &str, stage: usize| {
        let mut e = e.hypothetical();
        e.models = engine.models.clone();
        e.goal_stack = vec!["G".into()];
        e.set_evidence("Stage", stage).unwrap();
        e.q_value(a)
    };
    let (qa, qb) = (q(&engine, "A", 0), q(&engine, "B", 1));
    // B earned the reward directly; A, never rewarded itself, earned its discounted value.
    assert!(qb > 0.5, "Q(s1, B) = {qb}");
    assert!(qa > 0.05 && qa < qb, "Q(s0, A) = {qa}");
    // B was never taken at stage 0, so no exact-state utility was learned for it there.
    assert_eq!(engine.models.utility("B", &s0), 0.0);
    assert!(engine.models.utility("B", &s1) > 0.0);

    // In the graph, the backed-up utility raises A's posterior at stage 0.
    let ia = engine.index["A"];
    let without = {
        let mut e = staged(&id);
        e.goal_stack = vec!["G".into()];
        e.set_evidence("Stage", 0).unwrap();
        e.rebuild_memory_factors();
        e.infer(|v| (v == e.index["G"]).then_some(1))[ia][1]
    };
    engine.goal_stack = vec!["G".into()];
    engine.set_evidence("Stage", 0).unwrap();
    engine.rebuild_memory_factors();
    let with = engine.infer(|v| (v == engine.index["G"]).then_some(1))[ia][1];
    assert!(with > without + 0.01, "{with} vs {without}");
}

#[test]
fn new_models_sync_and_old_ones_still_load() {
    let id = NodeIdentity::generate();
    let open = || {
        let path = std::env::temp_dir().join(format!("sigma-models2-{}.redb", rand::random::<u64>()));
        let store = FactorStore::open(path).unwrap();
        let me = store.load_or_create_identity().unwrap().author_pubkey();
        (store, me)
    };
    let (phone, pc) = (open(), open());
    phone.0.pair_device(&pc.1).unwrap();
    pc.0.pair_device(&phone.1).unwrap();

    let mut on_phone = lamp(&id);
    on_phone.register_predicate(&bin("Bright", Context)).unwrap();
    on_phone.declare_perception_model("Bright", "Light").unwrap();
    toggle_experience(&mut on_phone, 3);
    on_phone.set_evidence("Light", 1).unwrap();
    on_phone.set_evidence("Bright", 1).unwrap();
    on_phone.cognitive_cycle();
    phone.0.save_memories(&on_phone).unwrap();

    let mut on_pc = lamp(&id);
    on_pc.register_predicate(&bin("Bright", Context)).unwrap();
    let report = pc.0.merge_sync_state(phone.1, phone.0.export_sync_state().unwrap()).unwrap();
    pc.0.refresh_engine(&mut on_pc, &report).unwrap();
    let key = ("Toggle".to_string(), "Light".to_string());
    assert_eq!(on_pc.models.transitions[&key], on_phone.models.transitions[&key]);
    let pkey = ("Bright".to_string(), "Light".to_string());
    assert_eq!(on_pc.models.perceptions[&pkey], on_phone.models.perceptions[&pkey]);
    let mut booted = SigmaEngine::new();
    pc.0.restore_engine(&mut booted).unwrap();
    assert_eq!(booted.models.transitions[&key], on_phone.models.transitions[&key]);
    assert_eq!(booted.models.perceptions[&pkey], on_phone.models.perceptions[&pkey]);

    // Models stored or synced before these existed still decode.
    #[derive(serde::Serialize)]
    struct Old {
        effects: BTreeMap<(String, String), Vec<f32>>,
        rewards: BTreeMap<String, Vec<f32>>,
    }
    let old = Old { effects: on_phone.models.effects.clone(), rewards: on_phone.models.rewards.clone() };
    let decoded = LearnedModels::decode(&postcard::to_allocvec(&old).unwrap()).unwrap();
    assert_eq!(decoded.effects, on_phone.models.effects);
    assert!(decoded.transitions.is_empty());
    let model = on_phone.models.action_model("Toggle");
    let old_action = postcard::to_allocvec(&(model.reward.clone(), model.effects.clone())).unwrap();
    let decoded = ActionModel::decode(&old_action).unwrap();
    assert_eq!((decoded.reward, decoded.effects), (model.reward.clone(), model.effects.clone()));
    // And a new model round-trips whole.
    assert_eq!(ActionModel::decode(&postcard::to_allocvec(&model).unwrap()).unwrap(), model);

    // Out-of-bounds transitions and utilities are rejected.
    let bad = ActionModel { transitions: [("Light".to_string(), vec![1.0; 6])].into(), ..Default::default() };
    assert!(!bad.is_valid());
    let bad = ActionModel { utilities: [("s".to_string(), f32::INFINITY)].into(), ..Default::default() };
    assert!(!bad.is_valid());
    assert!(!on_pc.adopt_perception_model("Bright", "@Episode", vec![1.0; 4]));
}
