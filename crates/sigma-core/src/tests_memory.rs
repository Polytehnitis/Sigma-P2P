//! Perception and motor buffers, episodic and semantic memory.

use super::*;
use crate::tests::{signed, skill_backup, skill_sync};
use PredicateRole::*;

fn bin(name: &str, role: PredicateRole) -> PredicateSchema {
    PredicateSchema::binary(name, role)
}

#[test]
fn perception_lasts_one_cycle_and_motor_buffers_actions() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    // Umbrella only when it rains: (Rain, ActionUmbrella) plus a goal that wants to stay dry.
    let skill = signed(
        &id,
        "dry",
        vec![bin("GoalDry", Goal), bin("Rain", Context), bin("ActionUmbrella", Action)],
        vec![1.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 9.0],
    );
    engine.compile_skill(skill).unwrap();
    engine.push_goal("GoalDry");
    assert!(engine.perceive("Rain", &[0.5]).is_err());
    engine.perceive("Rain", &[0.0, 1.0]).unwrap();
    assert_eq!(engine.working_memory(), [("Rain".to_string(), 1)]);
    assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("ActionUmbrella"));
    assert!(engine.working_memory().is_empty(), "percepts are gone after the cycle");
    let dry = engine.cognitive_cycle();
    assert_eq!(dry.selected_action, None, "no rain percept, no umbrella");
    assert_eq!(engine.take_motor_commands(), ["ActionUmbrella"]);
    assert!(engine.take_motor_commands().is_empty());
}

#[test]
fn episodic_memory_resolves_a_none_impasse() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill_backup(&id)).unwrap();
    engine.compile_skill(skill_sync(&id)).unwrap();
    engine.push_goal("GoalBackup");
    engine.set_evidence("SubgoalMountNAS", 1).unwrap();
    assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("ActionSyncFiles"));
    engine.reinforce_factors(1.0);
    assert_eq!(engine.episodic.episodes.back().unwrap().reward, 1.0);

    // The sync skill is lost; in the same situation the episode still suggests what worked.
    engine.evict_skills(|f| f.skill.skill_name == "sync-on-mounted-nas");
    let out = engine.cognitive_cycle();
    let imp = out.decision_impasse.unwrap();
    assert_eq!((imp.kind, imp.resolved), (ImpasseKind::None, true));
    assert_eq!(out.selected_action.as_deref(), Some("ActionSyncFiles"));

    // Unrewarded or punished episodes are never replayed.
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill_sync(&id)).unwrap();
    engine.push_goal("GoalBackup");
    engine.set_evidence("SubgoalMountNAS", 1).unwrap();
    engine.cognitive_cycle();
    engine.reinforce_factors(-1.0);
    engine.evict_skills(|_| true);
    assert_eq!(engine.cognitive_cycle().selected_action, None);
}

#[test]
fn episodic_recall_prefers_best_match_then_recency() {
    let mut m = EpisodicMemory::default();
    let s = |v: &[(&str, usize)]| v.iter().map(|(n, x)| (n.to_string(), *x)).collect::<Vec<_>>();
    m.record(vec!["G".into()], s(&[("Door", 1), ("Light", 0)]), Some("A".into()));
    m.record(vec!["G".into()], s(&[("Door", 0), ("Light", 0)]), Some("B".into()));
    m.record(vec!["G".into()], s(&[("Door", 1), ("Light", 0)]), Some("C".into()));
    let r = m.recall(&["G".into()], &s(&[("Door", 1)]), |_| true).unwrap();
    assert_eq!(r.episode.action.as_deref(), Some("C"), "most recent of the equal matches");
    assert!(r.confidence > 0.45);
    let r = m.recall(&["G".into()], &s(&[("Door", 0)]), |_| true).unwrap();
    assert_eq!(r.episode.action.as_deref(), Some("B"));
}

#[test]
fn semantic_memory_classifies_and_completes_working_memory() {
    let f = |v: &[(&str, usize)]| v.iter().map(|(n, x)| (n.to_string(), *x)).collect::<Vec<_>>();
    let mut engine = SigmaEngine::new();
    for _ in 0..3 {
        engine.semantic.learn("dog", &f(&[("Barks", 1), ("Purrs", 0), ("Fetches", 1)]));
        engine.semantic.learn("cat", &f(&[("Barks", 0), ("Purrs", 1), ("Fetches", 0)]));
    }
    let r = engine.semantic.recall(&f(&[("Barks", 1)])).unwrap();
    assert_eq!(r.categories[0].0, "dog");
    let fetch = &r.predicted.iter().find(|p| p.0 == "Fetches").unwrap().1;
    assert!(fetch[1] > 0.6);

    // In a cycle, semantic memory fills in the unobserved feature, which drives the choice.
    let id = NodeIdentity::generate();
    let skill = signed(
        &id,
        "play",
        vec![bin("GoalPlay", Goal), bin("Fetches", Context), bin("ActionThrowBall", Action)],
        vec![1.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 9.0],
    );
    engine.compile_skill(skill).unwrap();
    engine.register_predicate(&bin("Barks", Context)).unwrap();
    engine.push_goal("GoalPlay");
    engine.set_evidence("Barks", 1).unwrap();
    assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("ActionThrowBall"));
    engine.set_evidence("Barks", 0).unwrap();
    assert_eq!(engine.cognitive_cycle().selected_action, None);

    let path = std::env::temp_dir().join(format!("sigma-mem-{}.redb", rand::random::<u64>()));
    let store = FactorStore::open(&path).unwrap();
    store.save_memories(&engine).unwrap();
    let mut restored = SigmaEngine::new();
    store.restore_engine(&mut restored).unwrap();
    assert_eq!(restored.semantic, engine.semantic);
    assert_eq!(restored.episodic, engine.episodic);
}
