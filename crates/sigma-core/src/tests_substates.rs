//! Reflective impasse substates: reasoning inside an impasse, across cycles and levels.

use super::*;
use crate::tests::signed;
use PredicateRole::*;

fn bin(name: &str, role: PredicateRole) -> PredicateSchema {
    PredicateSchema::binary(name, role)
}

/// Goal G needs subgoal Ready. Actions A and B are equally favoured while Ready is open, but
/// only A brings Ready about.
fn two_step(id: &NodeIdentity) -> SigmaEngine {
    let mut engine = SigmaEngine::new();
    for skill in [
        signed(id, "g-needs-ready", vec![bin("G", Goal), bin("Ready", Subgoal)], vec![1.0, 1.0, 1.0, 10.0]),
        signed(id, "a-readies", vec![bin("Ready", Subgoal), bin("A", Action)], vec![1.0, 4.0, 1.0, 40.0]),
        signed(id, "b-habit", vec![bin("Ready", Subgoal), bin("B", Action)], vec![1.0, 4.0, 1.0, 4.0]),
    ] {
        engine.compile_skill(skill).unwrap();
    }
    engine.push_goal("G");
    engine
}

#[test]
fn a_substate_keeps_reasoning_until_the_tie_is_settled() {
    let id = NodeIdentity::generate();
    let mut engine = two_step(&id);

    // Cycle 1: a one-step look cannot tell A from B (Ready is still open either way), so the
    // substate stays on the state stack and nothing is selected.
    let out = engine.cognitive_cycle();
    let imp = out.decision_impasse.clone().unwrap();
    assert_eq!((imp.kind, imp.resolved, imp.goal.as_str()), (ImpasseKind::Tie, false, "Ready"));
    assert_eq!(out.selected_action, None);
    let sub = out.substate.clone().unwrap();
    assert_eq!((sub.depth, sub.horizon, sub.cycles, sub.result.clone()), (1, 1, 1, None));
    assert_eq!(engine.substates(), &[sub]);

    // Cycle 2: the same impasse, so the substate goes on, one step further. Imagining A shows
    // Ready coming about; imagining B shows nothing changing. A is returned to the top state.
    let out = engine.cognitive_cycle();
    let sub = out.substate.unwrap();
    assert_eq!((sub.horizon, sub.cycles), (2, 2));
    assert_eq!(sub.result.as_deref(), Some("A"), "{:?}", sub.evaluations);
    let value = |n: &str| sub.evaluations.iter().find(|e| e.0 == n).unwrap().1;
    assert!(value("A") > value("B") + TIE_MARGIN);
    assert_eq!(out.selected_action.as_deref(), Some("A"));
    assert!(out.decision_impasse.unwrap().resolved);
    assert!(engine.substates().is_empty(), "a settled substate is removed");
}

#[test]
fn lower_states_are_deleted_when_the_impasse_changes() {
    let id = NodeIdentity::generate();
    let mut engine = two_step(&id);
    engine.cognitive_cycle();
    assert_eq!(engine.substates().len(), 1);
    // Ready now holds: the impasse is gone and so is its substate.
    engine.set_evidence("Ready", 1).unwrap();
    let out = engine.cognitive_cycle();
    assert!(engine.substates().is_empty());
    assert!(out.substate.as_ref().is_none_or(|s| s.goal != "Ready"));
    // Dropping every goal deletes them too.
    let mut engine = two_step(&id);
    engine.cognitive_cycle();
    engine.goal_stack.clear();
    engine.cognitive_cycle();
    assert!(engine.substates().is_empty());
}

#[test]
fn imagined_ties_open_deeper_substates() {
    // G needs Ready (A or B tie for it, as above) and Set, for which C and D tie. Only C
    // brings Set about. Imagining either useful action leads to a state where the other
    // subgoal's candidates tie again, which takes a deeper substate.
    let id = NodeIdentity::generate();
    let mut engine = two_step(&id);
    for skill in [
        // G is likely only with Ready and Set both true.
        signed(
            &id,
            "g-needs-set",
            vec![bin("G", Goal), bin("Ready", Subgoal), bin("Set", Subgoal)],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 30.0],
        ),
        signed(&id, "c-sets", vec![bin("Set", Subgoal), bin("C", Action)], vec![1.0, 4.0, 1.0, 40.0]),
        signed(&id, "d-habit", vec![bin("Set", Subgoal), bin("D", Action)], vec![1.0, 4.0, 1.0, 4.0]),
    ] {
        engine.compile_skill(skill).unwrap();
    }
    let mut last = None;
    for _ in 0..MAX_HORIZON_FOR_TEST {
        let out = engine.cognitive_cycle();
        if let Some(s) = out.substate.clone() {
            last = Some(s);
        }
        if out.selected_action.is_some() {
            break;
        }
    }
    let sub = last.unwrap();
    assert!(sub.deepest >= 2, "a tie inside the imagined state opened a deeper substate: {sub:?}");
    // Both A and C lead toward G (each brings about one of the subgoals it needs); the habits
    // B and D lead nowhere.
    assert!(matches!(sub.result.as_deref(), Some("A" | "C")), "{sub:?}");
    let value = |n: &str| sub.evaluations.iter().find(|e| e.0 == n).unwrap().1;
    assert!(value("A").max(value("C")) > value("B").max(value("D")) + TIE_MARGIN);
}

const MAX_HORIZON_FOR_TEST: usize = crate::substate::MAX_HORIZON + 1;

/// Goal G and a Danger context, both with an opinion about action A.
fn conflicted(goal_table: Vec<f32>) -> SigmaEngine {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed(&id, "want", vec![bin("G", Goal), bin("A", Action)], goal_table)).unwrap();
    // With Danger, A is strongly opposed.
    engine
        .compile_skill(signed(&id, "danger", vec![bin("Danger", Context), bin("A", Action)], vec![1.0, 1.0, 9.0, 1.0]))
        .unwrap();
    engine.set_evidence("Danger", 1).unwrap();
    engine.push_goal("G");
    engine
}

#[test]
fn conflicts_are_reasoned_about_in_a_substate() {
    // A is strongly wanted and really helps G: the substate settles the conflict in A's favour.
    let out = conflicted(vec![1.0, 1.0, 1.0, 19.0]).cognitive_cycle();
    let imp = out.decision_impasse.unwrap();
    assert_eq!((imp.kind, imp.resolved), (ImpasseKind::Conflict, true));
    assert_eq!(out.selected_action.as_deref(), Some("A"));

    // A is wanted only out of habit (G is no likelier with it): the conflict stays open and
    // nothing is selected.
    let mut engine = conflicted(vec![1.0, 19.0, 1.0, 19.0]);
    let out = engine.cognitive_cycle();
    let imp = out.decision_impasse.unwrap();
    assert_eq!((imp.kind, imp.resolved), (ImpasseKind::Conflict, false));
    assert_eq!(out.selected_action, None);
    assert_eq!(engine.substates()[0].impasse, ImpasseKind::Conflict);
}

#[test]
fn none_impasses_search_the_actions_connected_to_the_goal() {
    // Nothing favours A while Ready is open, but imagining it shows Ready coming about.
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine
        .compile_skill(signed(&id, "g", vec![bin("G", Goal), bin("Ready", Subgoal)], vec![1.0, 1.0, 1.0, 10.0]))
        .unwrap();
    engine
        .compile_skill(signed(&id, "a", vec![bin("Ready", Subgoal), bin("A", Action)], vec![4.0, 1.0, 1.0, 40.0]))
        .unwrap();
    engine.push_goal("G");
    let first = engine.cognitive_cycle();
    assert_eq!(first.decision_impasse.unwrap().kind, ImpasseKind::None);
    let mut picked = first.selected_action;
    for _ in 0..crate::substate::MAX_HORIZON {
        if picked.is_some() {
            break;
        }
        picked = engine.cognitive_cycle().selected_action;
    }
    assert_eq!(picked.as_deref(), Some("A"));
}
