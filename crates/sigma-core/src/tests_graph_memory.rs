//! Memories and learned models inside the factor graph.

use super::*;
use crate::graph_memory::{next_name, CATEGORY, EPISODE};
use crate::tests::signed;
use PredicateRole::*;

fn bin(name: &str, role: PredicateRole) -> PredicateSchema {
    PredicateSchema::binary(name, role)
}

fn wm(v: &[(&str, usize)]) -> Vec<(String, usize)> {
    v.iter().map(|(n, x)| (n.to_string(), *x)).collect()
}

#[test]
fn episodes_are_recalled_by_message_passing() {
    let mut engine = SigmaEngine::new();
    for p in ["Door", "Light"] {
        engine.register_predicate(&bin(p, Context)).unwrap();
    }
    engine.episodic.record(vec![], wm(&[("Door", 1), ("Light", 0)]), Some("A".into()));
    engine.episodic.record(vec![], wm(&[("Door", 0), ("Light", 0)]), Some("B".into()));
    engine.episodic.record(vec![], wm(&[("Door", 1), ("Light", 1)]), Some("C".into()));
    engine.set_evidence("Door", 0).unwrap();
    engine.rebuild_memory_factors();
    assert!(engine.memory_factors().iter().any(|f| f.vars.contains(&engine.index[EPISODE])));
    let (e, confidence) = engine.recall_episode(false).unwrap();
    assert_eq!(e.action.as_deref(), Some("B"));
    assert!(confidence > 0.5);

    // A soft percept is a soft cue: the graph weighs it, it is not rounded to a value.
    engine.clear_evidence("Door");
    engine.perceive("Door", &[0.2, 0.8]).unwrap();
    engine.set_evidence("Light", 1).unwrap();
    engine.rebuild_memory_factors();
    assert_eq!(engine.recall_episode(false).unwrap().0.action.as_deref(), Some("C"));

    // Recall reads working memory but never writes it: Condition slots.
    let before = engine.posterior("Door").unwrap();
    assert!((before[1] - 0.8).abs() < 1e-5, "{before:?}");
    // Internal predicates stay out of the outcome.
    engine.push_goal("G");
    assert!(engine.cognitive_cycle().beliefs.iter().all(|(n, _)| !n.starts_with('@')));
}

#[test]
fn semantic_categories_live_in_the_graph() {
    let mut engine = SigmaEngine::new();
    for p in ["Barks", "Fetches"] {
        engine.register_predicate(&bin(p, Context)).unwrap();
    }
    for _ in 0..3 {
        engine.semantic.learn("dog", &wm(&[("Barks", 1), ("Fetches", 1)]));
        engine.semantic.learn("cat", &wm(&[("Barks", 0), ("Fetches", 0)]));
    }
    engine.set_evidence("Barks", 1).unwrap();
    engine.rebuild_memory_factors();
    assert!(engine.index.contains_key(CATEGORY));
    let cats = engine.categorize();
    assert_eq!(cats[0].0, "dog");
    // The category predicts the unobserved feature, through ordinary posteriors.
    // P(dog | Barks) = 0.8 and P(Fetches | dog) = 0.8 with add-one smoothing: 0.68.
    assert!((engine.posterior("Fetches").unwrap()[1] - 0.68).abs() < 1e-3);
    engine.set_evidence("Barks", 0).unwrap();
    assert!((engine.posterior("Fetches").unwrap()[1] - 0.32).abs() < 1e-3);
}

/// A lamp skill that makes `ActionA` the choice for `GoalLight`, and a goal G that needs
/// `Ready` with no skill linking `Ready` to any action.
fn lamp_and_ready(id: &NodeIdentity) -> SigmaEngine {
    let mut engine = SigmaEngine::new();
    engine
        .compile_skill(signed(
            id,
            "lamp",
            vec![bin("GoalLight", Goal), bin("ActionA", Action)],
            vec![1.0, 1.0, 1.0, 10.0],
        ))
        .unwrap();
    engine
        .compile_skill(signed(
            id,
            "g-needs-ready",
            vec![bin("G", Goal), bin("Ready", Subgoal)],
            vec![1.0, 1.0, 1.0, 10.0],
        ))
        .unwrap();
    engine
}

#[test]
fn action_models_are_learned_and_used_to_imagine() {
    let id = NodeIdentity::generate();
    let mut engine = lamp_and_ready(&id);
    // Experience: every time the agent turned the lamp on, Ready came about.
    for _ in 0..4 {
        engine.set_evidence("Ready", 0).unwrap();
        engine.push_goal("GoalLight");
        assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("ActionA"));
        engine.goal_stack.clear();
        engine.set_evidence("Ready", 1).unwrap();
        engine.cognitive_cycle(); // observes what followed, and learns
    }
    let t = &engine.models.effects[&("ActionA".to_string(), "Ready".to_string())];
    assert!(t[3] / (t[2] + t[3]) > 0.8, "P(Ready after | ActionA) = {t:?}");
    assert!(engine
        .memory_factors()
        .iter()
        .any(|f| engine.vars[*f.vars.last().unwrap()].schema.name == next_name("Ready", "ActionA")));

    // Now G needs Ready and no skill says how: a none impasse. Imagining ActionA with the
    // learned model shows Ready coming about, so the substate selects it.
    engine.clear_evidence("Ready");
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
    assert_eq!(picked.as_deref(), Some("ActionA"));

    // The models persist with the memories.
    let path = std::env::temp_dir().join(format!("sigma-models-{}.redb", rand::random::<u64>()));
    let store = FactorStore::open(&path).unwrap();
    store.save_memories(&engine).unwrap();
    let mut restored = SigmaEngine::new();
    store.restore_engine(&mut restored).unwrap();
    assert_eq!(restored.models, engine.models);
}

#[test]
fn reward_models_break_ties_in_substates() {
    // A and B are equally good means to G, except that A has been rewarded before.
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    let a = signed(&id, "a", vec![bin("G", Goal), bin("A", Action)], vec![1.0, 1.0, 1.0, 10.0]);
    let b = signed(&id, "b", vec![bin("G", Goal), bin("B", Action)], vec![1.0, 1.0, 1.0, 10.0]);
    // While Cue holds, A is the clear choice (so it gets selected and rewarded).
    let cue = signed(
        &id,
        "cue",
        vec![bin("G", Goal), bin("Cue", Context), bin("A", Action)],
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0],
    );
    for s in [a, b, cue] {
        engine.compile_skill(s).unwrap();
    }
    engine.push_goal("G");
    engine.set_evidence("Cue", 0).unwrap();
    let out = engine.cognitive_cycle();
    assert_eq!(out.decision_impasse.unwrap().kind, ImpasseKind::Tie, "symmetric without the cue");
    let initial: Vec<(String, Vec<f32>)> =
        ["a", "b", "cue"].iter().map(|n| (n.to_string(), engine.skill_weights(n).unwrap())).collect();
    engine.set_evidence("Cue", 1).unwrap();
    for _ in 0..3 {
        engine.goal_stack = vec!["G".into()];
        assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("A"));
        engine.reinforce_factors(1.0);
        // A fresh episode each time (no no-change impasse).
        engine.goal_stack.clear();
        engine.cognitive_cycle();
    }
    // Undo the Hebbian change to the skills, so only the reward model remembers.
    for (n, w) in &initial {
        assert!(engine.apply_weights(n, w));
    }
    engine.set_evidence("Cue", 0).unwrap();
    engine.goal_stack = vec!["G".into()];
    let out = engine.cognitive_cycle();
    let imp = out.decision_impasse.unwrap();
    assert_eq!((imp.kind, imp.resolved), (ImpasseKind::Tie, true));
    assert_eq!(out.selected_action.as_deref(), Some("A"), "{:?}", out.substate);
}

#[test]
fn skills_cannot_declare_internal_predicates() {
    let id = NodeIdentity::generate();
    let s = signed(&id, "x", vec![bin("@Episode", Context), bin("A", Action)], vec![1.0; 4]);
    assert_eq!(s.validate(), Err("reserved predicate name"));
}
