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

/// [`two_step`], with a context predicate `Sunny` linked to `Ready` (it does not matter to the
/// choice, but it is part of the situation).
fn two_step_sunny(id: &NodeIdentity, sunny: usize) -> SigmaEngine {
    let mut engine = two_step(id);
    let weather = signed(id, "weather", vec![bin("Ready", Subgoal), bin("Sunny", Context)], vec![1.0; 4]);
    engine.compile_skill(weather).unwrap();
    engine.set_evidence("Sunny", sunny).unwrap();
    engine
}

#[test]
fn settled_substates_are_chunked_into_signed_rules() {
    let id = NodeIdentity::generate();
    let mut engine = two_step_sunny(&id, 1);
    engine.chunk_author = Some(id.clone());
    assert!(engine.cognitive_cycle().chunk.is_none(), "nothing settled yet");
    let out = engine.cognitive_cycle();
    assert_eq!(out.selected_action.as_deref(), Some("A"));
    let chunk = out.chunk.expect("the settled tie is chunked");
    assert!(chunk.skill_name.starts_with(crate::substate::CHUNK_PREFIX));
    assert_eq!(chunk.validate(), Ok(()));
    assert_eq!(chunk.author_pubkey, id.author_pubkey());
    // No atom has arguments, so the chunk is a ground table, and it is not offered to peers.
    assert!(!crate::substate::shareable_chunk(&chunk));
    let table = chunk.as_v1().expect("a ground chunk");
    let names: Vec<&str> = table.predicates.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["Ready", "Sunny", "A"]);
    // Its strong entry is the situation: Ready still open, Sunny observed, A taken.
    assert_eq!(table.factor_values[0b011], crate::substate::CHUNK_WEIGHT);

    // Next time the same situation is decided at once: no impasse, no substate.
    engine.goal_stack = vec!["G".into()];
    engine.cognitive_cycle(); // compiles the chunk; pushes Ready
    assert!(engine.export_skill(&chunk.skill_name).is_some());
    let mut fresh = two_step_sunny(&id, 1);
    fresh.compile_skill(chunk.clone()).unwrap();
    let out = fresh.cognitive_cycle();
    assert_eq!(out.selected_action.as_deref(), Some("A"));
    assert_eq!(out.decision_impasse, None);
    assert!(out.chunk.is_none());

    // In another situation the chunk does not apply, and the tie is reasoned about again.
    let mut other = two_step_sunny(&id, 0);
    other.compile_skill(chunk).unwrap();
    assert_eq!(other.cognitive_cycle().decision_impasse.map(|d| d.kind), Some(ImpasseKind::Tie));
}

#[test]
fn chunks_without_an_author_are_proposed_unsigned() {
    let id = NodeIdentity::generate();
    let mut engine = two_step_sunny(&id, 1);
    engine.cognitive_cycle();
    let chunk = engine.cognitive_cycle().chunk.expect("proposed");
    assert!(chunk.validate().is_err(), "unsigned");
    let mut fresh = two_step_sunny(&id, 1);
    let signed_chunk = fresh.learn_chunk(&id, chunk).unwrap();
    assert_eq!(signed_chunk.validate(), Ok(()));
    assert_eq!(fresh.cognitive_cycle().decision_impasse, None);
    assert!(fresh.learn_chunk(&id, crate::tests::skill_sync(&id)).is_err(), "only chunks");
}

/// [`two_step_sunny`] over devices: `G(d)` needs `Ready(d)`, and `A(d)` readies it.
fn devices(id: &NodeIdentity, goal: &str) -> SigmaEngine {
    use crate::program::*;
    let pred = |name: &str, role, args: &[&str]| PredicateDef {
        name: name.into(),
        role,
        args: args.iter().map(|a| a.to_string()).collect(),
        value_type: None,
    };
    let pat = |kind, predicate: &str, args: &[&str]| Pattern {
        kind,
        predicate: predicate.into(),
        args: args.iter().map(|a| Term::Var(a.to_string())).collect(),
    };
    let cond = |name: &str, patterns: Vec<Pattern>, table: Vec<f32>| Conditional {
        name: name.into(),
        patterns,
        function_vars: Vec::new(),
        function: Function::Table(table),
        learnable: false,
        normal: None,
    };
    use PatternKind::{Condact as C, Condition as R};
    let program = Program {
        types: vec![TypeDef::symbolic("Device", &["nas", "phone", "pc"])],
        predicates: vec![
            pred("G", Goal, &["Device"]),
            pred("Ready", Subgoal, &["Device"]),
            pred("A", Action, &["Device"]),
            pred("B", Action, &["Device"]),
            pred("Sunny", Context, &[]),
        ],
        conditionals: vec![
            cond("needs", vec![pat(C, "G", &["x"]), pat(C, "Ready", &["x"])], vec![1.0, 1.0, 1.0, 10.0]),
            cond("a", vec![pat(C, "Ready", &["x"]), pat(C, "A", &["x"])], vec![4.0, 1.0, 1.0, 4.0]),
            cond("b", vec![pat(C, "Ready", &["x"]), pat(C, "B", &["x"])], vec![4.0, 2.0, 1.0, 1.0]),
            cond("weather", vec![pat(R, "Ready", &["x"]), pat(R, "Sunny", &[])], vec![1.0; 4]),
        ],
    };
    let mut skill = Skill::program("devices", 1, program);
    id.sign(&mut skill);
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill).unwrap();
    engine.set_evidence("Sunny", 1).unwrap();
    engine.push_goal(goal);
    engine
}

#[test]
fn chunks_are_variablized_and_apply_to_other_constants() {
    let id = NodeIdentity::generate();
    let mut engine = devices(&id, "G(nas)");
    engine.cognitive_cycle();
    let out = engine.cognitive_cycle();
    assert_eq!(out.selected_action.as_deref(), Some("A(nas)"));
    let chunk = out.chunk.expect("chunked");
    // Learned for the NAS, the rule names no device: Ready(X1), G(X1), Sunny -> A(X1).
    let SkillBody::Program(p) = &chunk.body else { panic!("a variablized chunk is a program: {chunk:?}") };
    let c = &p.conditionals[0];
    let shown: Vec<(PatternKind, &str, &[Term])> =
        c.patterns.iter().map(|p| (p.kind, p.predicate.as_str(), p.args.as_slice())).collect();
    let x1 = [Term::Var("X1".into())];
    assert_eq!(
        shown,
        [
            (PatternKind::Condition, "Ready", &x1[..]),
            (PatternKind::Condition, "G", &x1[..]),
            (PatternKind::Condition, "Sunny", &[][..]),
            (PatternKind::Action, "A", &x1[..]),
        ]
    );
    assert!(chunk.skill_name.starts_with("chunk:Ready(X1)->A(X1)#"), "{}", chunk.skill_name);
    assert!(crate::substate::shareable_chunk(&chunk));
    assert!(crate::substate::offered_to_peers(&chunk));
    // A peer stuck on any device's goal finds it in the swarm.
    assert!(chunk.mentions("A(pc)") && chunk.mentions("Ready(phone)"));
    // It generalizes: a peer with the goal for the phone decides at once, with no impasse.
    let mut signed_chunk = chunk.clone();
    id.sign(&mut signed_chunk);
    assert_eq!(signed_chunk.validate(), Ok(()));
    let mut peer = devices(&id, "G(phone)");
    assert_eq!(peer.cognitive_cycle().decision_impasse.map(|d| d.kind), Some(ImpasseKind::None));
    let mut peer = devices(&id, "G(phone)");
    peer.compile_skill(signed_chunk.clone()).unwrap();
    let out = peer.cognitive_cycle();
    assert_eq!(out.selected_action.as_deref(), Some("A(phone)"));
    assert_eq!(out.decision_impasse, None);
    // But only where its condition holds.
    let mut cloudy = devices(&id, "G(pc)");
    cloudy.compile_skill(signed_chunk.clone()).unwrap();
    cloudy.set_evidence("Sunny", 0).unwrap();
    assert_eq!(cloudy.cognitive_cycle().decision_impasse.map(|d| d.kind), Some(ImpasseKind::None));
    // The same rule learned from another device is the same chunk.
    let mut again = devices(&id, "G(pc)");
    again.cognitive_cycle();
    assert_eq!(again.cognitive_cycle().chunk.map(|c| c.skill_name), Some(chunk.skill_name));
}
