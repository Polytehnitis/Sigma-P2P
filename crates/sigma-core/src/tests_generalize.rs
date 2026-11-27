//! Learned models that generalize: linear utilities over state features, transition models
//! conditioned on a learned parent, and discovered perception models.

use super::*;
use crate::graph_memory::transition_name;
use crate::tests::signed;
use PredicateRole::*;

fn bin(name: &str, role: PredicateRole) -> PredicateSchema {
    PredicateSchema::binary(name, role)
}

/// G calls for A; a context predicate `Noise` that the skill ignores varies the state.
fn noisy(id: &NodeIdentity) -> SigmaEngine {
    let mut engine = SigmaEngine::new();
    let a = signed(id, "a", vec![bin("G", Goal), bin("A", Action)], vec![1.0, 1.0, 1.0, 10.0]);
    let b = signed(id, "b", vec![bin("G", Goal), bin("B", Action)], vec![1.0, 1.0, 1.0, 4.0]);
    engine.compile_skill(a).unwrap();
    engine.compile_skill(b).unwrap();
    for p in ["Noise", "Other"] {
        engine.register_predicate(&bin(p, Context)).unwrap();
    }
    engine
}

#[test]
fn utilities_generalize_to_unseen_states() {
    let id = NodeIdentity::generate();
    let mut engine = noisy(&id);
    // A is rewarded in states where Noise is 0 or 1, but Other is always 0.
    for i in 0..8 {
        engine.goal_stack = vec!["G".into()];
        engine.set_evidence("Noise", i % 2).unwrap();
        engine.set_evidence("Other", 0).unwrap();
        assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("A"));
        engine.reinforce_factors(1.0);
        engine.goal_stack.clear();
        engine.cognitive_cycle();
    }
    // A state never seen (Other = 1): the table has nothing for it, the features do.
    engine.goal_stack = vec!["G".into()];
    engine.set_evidence("Other", 1).unwrap();
    let unseen = engine.state_key();
    assert_eq!(engine.models.utility("A", &unseen), 0.0);
    let qa = engine.q_value("A");
    assert!(qa > 0.2, "Q(unseen, A) = {qa}");
    assert!(engine.q_value("B").abs() < 1e-6);
    // And it reaches the graph: A's posterior is raised in the unseen state.
    let ia = engine.index["A"];
    let g = engine.index["G"];
    engine.rebuild_memory_factors();
    let with = engine.infer(|v| (v == g).then_some(1))[ia][1];
    let mut fresh = noisy(&id);
    fresh.goal_stack = vec!["G".into()];
    fresh.set_evidence("Other", 1).unwrap();
    fresh.rebuild_memory_factors();
    let without = fresh.infer(|v| (v == g).then_some(1))[ia][1];
    assert!(with > without + 0.01, "{with} vs {without}");
}

/// `Toggle` flips `Light` only when `Power` is on. `G` needs `Light`, and no skill says how.
fn switched(id: &NodeIdentity) -> SigmaEngine {
    let mut engine = SigmaEngine::new();
    let lamp = signed(id, "lamp", vec![bin("GoalLamp", Goal), bin("Toggle", Action)], vec![1.0, 1.0, 1.0, 10.0]);
    let need = signed(id, "g-needs-light", vec![bin("G", Goal), bin("Light", Subgoal)], vec![1.0, 1.0, 1.0, 10.0]);
    engine.compile_skill(lamp).unwrap();
    engine.compile_skill(need).unwrap();
    engine.register_predicate(&bin("Power", Context)).unwrap();
    engine
}

fn switch_experience(engine: &mut SigmaEngine, n: usize) {
    for i in 0..n {
        let (light, power) = (i % 2, (i / 2) % 2);
        engine.set_evidence("Light", light).unwrap();
        engine.set_evidence("Power", power).unwrap();
        engine.push_goal("GoalLamp");
        assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("Toggle"));
        engine.goal_stack.clear();
        engine.set_evidence("Light", if power == 1 { 1 - light } else { light }).unwrap();
        engine.cognitive_cycle();
    }
}

#[test]
fn transitions_learn_the_predicate_they_depend_on() {
    let id = NodeIdentity::generate();
    let mut engine = switched(&id);
    switch_experience(&mut engine, 24);
    let report = engine.learn_model_structure();
    assert!(report.parents.contains(&("Toggle".into(), "Light".into(), "Power".into())), "{:?}", report.parents);
    // The four-slot model is in the graph, [Toggle, Light, Power, @Trans(Light|Toggle)].
    let f = engine
        .memory_factors()
        .iter()
        .find(|f| engine.vars[*f.vars.last().unwrap()].schema.name == transition_name("Light", "Toggle"))
        .unwrap();
    assert_eq!(f.vars.len(), 4);
    assert_eq!(engine.vars[f.vars[2]].schema.name, "Power");
    // Reading it: with the power on, Toggle turns the light on; with it off, it does not.
    let t = &engine.models.contexts[&("Toggle".to_string(), "Light".to_string())].1;
    let p_on = |light: usize, power: usize| {
        let row = 8 + (light * 2 + power) * 2;
        t[row + 1] / (t[row] + t[row + 1])
    };
    assert!(p_on(0, 1) > 0.8 && p_on(0, 0) < 0.2, "{t:?}");
    // The plain model is a coin flip from Light = 0.
    let plain = &engine.models.transitions[&("Toggle".to_string(), "Light".to_string())];
    let row = 4;
    assert!((plain[row + 1] / (plain[row] + plain[row + 1]) - 0.5).abs() < 0.25, "{plain:?}");

    // G needs Light. With the power off, imagining Toggle shows nothing; with it on, Toggle.
    let decide = |power: usize| {
        let mut e = switched(&id);
        e.models = engine.models.clone();
        e.rebuild_memory_factors();
        e.set_evidence("Light", 0).unwrap();
        e.set_evidence("Power", power).unwrap();
        e.push_goal("G");
        (0..crate::substate::MAX_HORIZON).find_map(|_| e.cognitive_cycle().selected_action)
    };
    assert_eq!(decide(1).as_deref(), Some("Toggle"));
    assert_eq!(decide(0), None);
}

#[test]
fn context_models_sync_and_decode() {
    let id = NodeIdentity::generate();
    let mut engine = switched(&id);
    switch_experience(&mut engine, 16);
    engine.learn_model_structure();
    let model = engine.models.action_model("Toggle");
    assert!(model.contexts.contains_key("Light"));
    assert!(model.is_valid());
    assert_eq!(ActionModel::decode(&postcard::to_allocvec(&model).unwrap()).unwrap(), model);
    let mut other = switched(&id);
    assert!(other.adopt_action_model("Toggle", model.clone()));
    assert_eq!(other.models.contexts, engine.models.contexts);
    // A malformed context table, or one naming an internal parent, is refused.
    let mut bad = model.clone();
    bad.contexts.insert("Light".into(), ("Power".into(), vec![1.0; 5]));
    assert!(!bad.is_valid());
    let mut bad = model;
    bad.contexts.insert("Light".into(), ("@Episode".into(), vec![0.5; 16]));
    assert!(!bad.is_valid());
    // The whole store round-trips, and the layout before these fields still decodes.
    let bytes = postcard::to_allocvec(&engine.models).unwrap();
    assert_eq!(LearnedModels::decode(&bytes).unwrap(), engine.models);
    let m = &engine.models;
    let v2 = postcard::to_allocvec(&(&m.effects, &m.rewards, &m.transitions, &m.perceptions, &m.utilities)).unwrap();
    let decoded = LearnedModels::decode(&v2).unwrap();
    assert_eq!(decoded.transitions, m.transitions);
    assert!(decoded.contexts.is_empty() && decoded.q_features.is_empty());
}

#[test]
fn perception_models_are_discovered() {
    let mut engine = SigmaEngine::new();
    let id = NodeIdentity::generate();
    engine
        .compile_skill(signed(&id, "act", vec![bin("G", Goal), bin("Look", Action)], vec![1.0, 1.0, 1.0, 10.0]))
        .unwrap();
    for p in ["Raining", "WetGrass", "Unrelated"] {
        engine.register_predicate(&bin(p, Context)).unwrap();
    }
    for i in 0..20 {
        let rain = (i / 2) % 2;
        engine.set_evidence("WetGrass", rain).unwrap();
        engine.set_evidence("Unrelated", i % 2).unwrap();
        // Rain is mostly observed directly, sometimes not.
        if i % 5 == 4 {
            engine.clear_evidence("Raining");
        } else {
            engine.set_evidence("Raining", rain).unwrap();
        }
        engine.push_goal("G");
        engine.cognitive_cycle();
    }
    let report = engine.learn_model_structure();
    let wet = ("WetGrass".to_string(), "Raining".to_string());
    assert!(engine.models.perceptions.contains_key(&wet), "{report:?}");
    assert!(!engine.models.perceptions.keys().any(|(o, s)| o == "Unrelated" || s == "Unrelated"));
    // Discovered, the model lets the observation alone inform the hidden state.
    engine.goal_stack.clear();
    engine.clear_evidence("Raining");
    engine.set_evidence("WetGrass", 1).unwrap();
    assert!(engine.posterior("Raining").unwrap()[1] > 0.75);
    engine.set_evidence("WetGrass", 0).unwrap();
    assert!(engine.posterior("Raining").unwrap()[1] < 0.25);
}
