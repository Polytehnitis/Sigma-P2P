//! Attention, appraisal/emotion and Theory of Mind.

use super::*;
use crate::tests::{signed, skill_backup, skill_sync};
use crate::tests_v2::signed_program;
use PredicateRole::*;

fn bin(name: &str, role: PredicateRole) -> PredicateSchema {
    PredicateSchema::binary(name, role)
}

fn backup_engine(id: &NodeIdentity) -> SigmaEngine {
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill_backup(id)).unwrap();
    engine.compile_skill(skill_sync(id)).unwrap();
    engine
}

#[test]
fn attention_budget_keeps_goal_relevant_factors() {
    let id = NodeIdentity::generate();
    let mut engine = backup_engine(&id);
    let noise = signed(&id, "noise", vec![bin("Birdsong", Context), bin("Sunny", Context)], vec![1.0, 2.0, 3.0, 4.0]);
    engine.compile_skill(noise).unwrap();
    engine.push_goal("GoalBackup");

    engine.attention_budget = Some(2);
    let out = engine.cognitive_cycle();
    assert_eq!(out.attended_skills, ["backup-needs-nas", "sync-on-mounted-nas"]);
    assert_eq!(out.selected_action.as_deref(), Some("ActionMountNAS"));

    engine.attention_budget = Some(1);
    let out = engine.cognitive_cycle();
    assert_eq!(out.attended_skills, ["backup-needs-nas"]);
    assert_eq!(out.selected_action.as_deref(), Some("ActionMountNAS"));

    engine.attention_budget = None;
    assert!(engine.cognitive_cycle().attended_skills.is_empty());
}

#[test]
fn appraisal_tracks_progress_surprise_control_and_reward() {
    let id = NodeIdentity::generate();
    let mut engine = backup_engine(&id);
    engine.push_goal("GoalBackup");
    let a = engine.cognitive_cycle().appraisal;
    assert!(a.control > 0.7 && a.desirability == 0.0 && a.surprise == 0.0);
    assert_eq!(a.emotion, Emotion::Hope);

    // The mount succeeds although the agent planned as if it had not: progress and surprise.
    engine.set_evidence("SubgoalMountNAS", 1).unwrap();
    let a = engine.cognitive_cycle().appraisal;
    assert!((a.desirability - 0.5).abs() < 1e-6, "one of two open goals done");
    assert!(a.surprise > 0.9);
    assert_eq!(a.emotion, Emotion::Joy);
    assert!(a.learning_rate(0.1) > 0.19);

    engine.reinforce_factors(-1.0);
    let a = engine.cognitive_cycle().appraisal;
    assert_eq!(a.emotion, Emotion::Distress);
    assert_eq!(a.surprise, 0.0, "nothing new was observed");
    assert!(a.familiarity > 0.99, "the same situation as before");

    let mut lost = SigmaEngine::new();
    lost.push_goal("GoalUnknown");
    assert_eq!(lost.cognitive_cycle().appraisal.emotion, Emotion::Fear);
}

#[test]
fn theory_of_mind_simulates_others_without_touching_our_state() {
    let id = NodeIdentity::generate();
    let mut engine = backup_engine(&id);
    engine.push_goal("GoalBackup");
    let bob = OtherAgent { name: "bob".into(), goals: vec!["GoalBackup".into()], ..OtherAgent::default() };
    assert_eq!(engine.model_agent(&bob, 1).unwrap().predicted_action.as_deref(), Some("ActionMountNAS"));
    let bob_mounted = OtherAgent { evidence: vec![("SubgoalMountNAS".into(), 1)], ..bob.clone() };
    let m = engine.model_agent(&bob_mounted, 1).unwrap();
    assert_eq!(m.predicted_action.as_deref(), Some("ActionSyncFiles"));
    assert!(m.action_beliefs.iter().any(|(a, p)| a == "ActionSyncFiles" && *p > 0.5));
    // Our own working memory and goals are untouched.
    assert_eq!(engine.variables().iter().filter(|v| v.evidence.is_some()).count(), 0);
    assert_eq!(engine.goal_stack, ["GoalBackup"]);

    // Someone with different knowledge decides differently.
    let tidy =
        signed(&id, "tidy", vec![bin("GoalBackup", Goal), bin("ActionSortPapers", Action)], vec![1.0, 1.0, 1.0, 9.0]);
    let clerk = OtherAgent { skills: Some(vec![tidy.into()]), ..bob };
    assert_eq!(engine.model_agent(&clerk, 1).unwrap().predicted_action.as_deref(), Some("ActionSortPapers"));
}

/// Bob goes where he expects Carol to go; Carol habitually goes right.
#[test]
fn recursive_theory_of_mind() {
    let id = NodeIdentity::generate();
    let c = |k: &str| Term::Const(k.into());
    let pat = |kind, p: &str, args| Pattern { kind, predicate: p.into(), args };
    let follow = |dir: &str| Conditional {
        name: format!("follow-{dir}"),
        patterns: vec![
            pat(PatternKind::Condition, "Coordinate", vec![]),
            pat(PatternKind::Condition, "Does", vec![c("carol"), c(dir)]),
            pat(PatternKind::Action, dir, vec![]),
        ],
        function_vars: vec![],
        function: Function::Table(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0]),
        learnable: false,
        normal: None,
    };
    let pred = |name: &str, role, args: &[&str]| PredicateDef {
        name: name.into(),
        role,
        args: args.iter().map(|a| a.to_string()).collect(),
        value_type: None,
    };
    let bob_skill = signed_program(
        &id,
        "follow-carol",
        Program {
            types: vec![TypeDef::symbolic("agent", &["carol"]), TypeDef::symbolic("act", &["GoLeft", "GoRight"])],
            predicates: vec![
                pred("Coordinate", Goal, &[]),
                pred("Does", Context, &["agent", "act"]),
                pred("GoLeft", Action, &[]),
                pred("GoRight", Action, &[]),
            ],
            conditionals: vec![follow("GoLeft"), follow("GoRight")],
        },
    );
    let habit =
        signed(&id, "carol-habit", vec![bin("Coordinate", Goal), bin("GoRight", Action)], vec![1.0, 1.0, 1.0, 9.0]);
    let carol = OtherAgent {
        name: "carol".into(),
        goals: vec!["Coordinate".into()],
        skills: Some(vec![habit.into()]),
        ..OtherAgent::default()
    };
    let bob = OtherAgent {
        name: "bob".into(),
        goals: vec!["Coordinate".into()],
        skills: Some(vec![bob_skill]),
        models: vec![carol],
        ..OtherAgent::default()
    };
    let me = SigmaEngine::new();
    let level2 = me.model_agent(&bob, 2).unwrap();
    assert_eq!(level2.nested[0].predicted_action.as_deref(), Some("GoRight"));
    assert_eq!(level2.predicted_action.as_deref(), Some("GoRight"));
    let level1 = me.model_agent(&bob, 1).unwrap();
    assert!(level1.nested.is_empty());
    assert_eq!(level1.predicted_action, None, "without modelling Carol, Bob is stuck in a tie");
}

/// Bob follows Carol, built as in `recursive_theory_of_mind`.
fn bob_and_carol(id: &NodeIdentity) -> (OtherAgent, OtherAgent) {
    let c = |k: &str| Term::Const(k.into());
    let pat = |kind, p: &str, args| Pattern { kind, predicate: p.into(), args };
    let follow = |dir: &str| Conditional {
        name: format!("follow-{dir}"),
        patterns: vec![
            pat(PatternKind::Condition, "Coordinate", vec![]),
            pat(PatternKind::Condition, "Does", vec![c("carol"), c(dir)]),
            pat(PatternKind::Action, dir, vec![]),
        ],
        function_vars: vec![],
        function: Function::Table(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0]),
        learnable: false,
        normal: None,
    };
    let pred = |name: &str, role, args: &[&str]| PredicateDef {
        name: name.into(),
        role,
        args: args.iter().map(|a| a.to_string()).collect(),
        value_type: None,
    };
    let bob_skill = signed_program(
        id,
        "follow-carol",
        Program {
            types: vec![TypeDef::symbolic("agent", &["carol"]), TypeDef::symbolic("act", &["GoLeft", "GoRight"])],
            predicates: vec![
                pred("Coordinate", Goal, &[]),
                pred("Does", Context, &["agent", "act"]),
                pred("GoLeft", Action, &[]),
                pred("GoRight", Action, &[]),
            ],
            conditionals: vec![follow("GoLeft"), follow("GoRight")],
        },
    );
    let habit =
        signed(id, "carol-habit", vec![bin("Coordinate", Goal), bin("GoRight", Action)], vec![1.0, 1.0, 1.0, 9.0]);
    let carol = OtherAgent {
        name: "carol".into(),
        goals: vec!["Coordinate".into()],
        skills: Some(vec![habit.into()]),
        ..OtherAgent::default()
    };
    let bob = OtherAgent {
        name: "bob".into(),
        goals: vec!["Coordinate".into()],
        skills: Some(vec![bob_skill]),
        ..OtherAgent::default()
    };
    (bob, carol)
}

#[test]
fn several_agents_are_modelled_in_one_graph() {
    let id = NodeIdentity::generate();
    let (bob, carol) = bob_and_carol(&id);
    let me = SigmaEngine::new();
    let joint = me.model_agents_jointly(&[bob.clone(), carol.clone()], &[], 5).unwrap();
    // Bob's `Does(carol,GoRight)` is tied to Carol's own action in the same graph (Carol's
    // skills have no GoLeft to tie to).
    assert_eq!(joint.links, 1);
    assert!(joint.converged);
    let [b, c] = &joint.agents[..] else { panic!("two agents") };
    assert_eq!((b.agent.as_str(), c.agent.as_str()), ("bob", "carol"));
    assert_eq!(c.predicted_action.as_deref(), Some("GoRight"));
    assert_eq!(b.predicted_action.as_deref(), Some("GoRight"), "Bob follows Carol's predicted move");
    assert!(b.action_beliefs.iter().all(|(a, _)| a == "GoLeft" || a == "GoRight"), "only Bob's own actions");
    // One round: Carol's move is not yet known when Bob decides, so he is stuck in a tie.
    let first = me.model_agents_jointly(&[bob.clone(), carol.clone()], &[], 1).unwrap();
    assert_eq!(first.agents[0].predicted_action, None);
    assert!(!first.converged);
    // The per-agent design with nesting agrees.
    let nested = OtherAgent { models: vec![carol.clone()], ..bob.clone() };
    assert_eq!(me.model_agent(&nested, 2).unwrap().predicted_action, b.predicted_action);
    assert!(me.model_agents_jointly(&[bob.clone(), bob], &[], 1).is_err(), "names must differ");
}

#[test]
fn agents_in_one_graph_share_the_world() {
    let id = NodeIdentity::generate();
    let engine = backup_engine(&id);
    let goal = vec!["GoalBackup".to_string()];
    // Alice sees that the NAS is mounted; Bob does not look.
    let alice = OtherAgent {
        name: "alice".into(),
        goals: goal.clone(),
        evidence: vec![("SubgoalMountNAS".into(), 1)],
        ..OtherAgent::default()
    };
    let bob = OtherAgent { name: "bob".into(), goals: goal, ..OtherAgent::default() };
    // Separately, each with its own view of the world.
    let apart = engine.model_agents_jointly(&[alice.clone(), bob.clone()], &[], 3).unwrap();
    let acts: Vec<_> = apart.agents.iter().map(|m| m.predicted_action.clone()).collect();
    assert_eq!(acts, [Some("ActionSyncFiles".into()), Some("ActionMountNAS".into())]);
    // One shared world: what Alice observes holds for Bob too.
    let shared = engine.model_agents_jointly(&[alice.clone(), bob.clone()], &["SubgoalMountNAS"], 3).unwrap();
    let acts: Vec<_> = shared.agents.iter().map(|m| m.predicted_action.clone()).collect();
    assert_eq!(acts, [Some("ActionSyncFiles".into()), Some("ActionSyncFiles".into())]);
    assert!(shared.variables < apart.variables, "one variable for both");
    // A single agent in the joint graph predicts what `model_agent` does.
    for a in [&alice, &bob] {
        let one = engine.model_agents_jointly(std::slice::from_ref(a), &[], 2).unwrap();
        assert_eq!(one.agents[0].predicted_action, engine.model_agent(a, 1).unwrap().predicted_action);
    }
}

#[test]
fn appraisals_live_in_the_graph_and_skills_can_use_them() {
    use crate::appraisal::{EMOTION, EMOTIONS, SURPRISED};
    let id = NodeIdentity::generate();
    let bin = |n: &str, r: PredicateRole| PredicateSchema::binary(n, r);
    let mut engine = SigmaEngine::new();
    // The door is usually shut. While the goal is on, carrying on is the habit, but a
    // surprised agent investigates first.
    let skills = [
        crate::tests::signed(&id, "door", vec![bin("Door", PredicateRole::Context)], vec![9.0, 1.0]),
        crate::tests::signed(
            &id,
            "carry",
            vec![bin("G", PredicateRole::Goal), bin("Carry", PredicateRole::Action)],
            vec![1.0, 1.0, 1.0, 4.0],
        ),
        crate::tests::signed(
            &id,
            "investigate",
            vec![
                bin("G", PredicateRole::Goal),
                bin(SURPRISED, PredicateRole::Context),
                bin("Investigate", PredicateRole::Action),
            ],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 40.0],
        ),
    ];
    for s in skills {
        assert_eq!(s.validate(), Ok(()), "skills may use appraisal predicates");
        engine.compile_skill(s).unwrap();
    }
    engine.push_goal("G");
    assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("Carry"));
    // The door is found open: surprising, and the surprise is in the graph this cycle.
    engine.set_evidence("Door", 1).unwrap();
    let out = engine.cognitive_cycle();
    assert!(out.appraisal.surprise > 0.8);
    assert_eq!(out.appraisal.emotion, Emotion::Surprise);
    assert_eq!(out.selected_action.as_deref(), Some("Investigate"));
    assert!(engine.posterior(SURPRISED).unwrap()[1] > 0.8);
    // The emotion is a variable of the graph too.
    let e = engine.posterior(EMOTION).unwrap();
    let best = e.iter().copied().fold(0.0, f32::max);
    assert_eq!(EMOTIONS[e.iter().position(|&x| x == best).unwrap()], Emotion::Surprise);
    // Nothing new next time: no surprise, back to the habit.
    let out = engine.cognitive_cycle();
    assert_eq!(out.appraisal.surprise, 0.0);
    assert_eq!(out.selected_action.as_deref(), Some("Carry"));
    // Internal names other than the appraisal predicates stay reserved, and an appraisal
    // predicate must be declared as the engine defines it.
    let bad = crate::tests::signed(
        &id,
        "x",
        vec![bin(EMOTION, PredicateRole::Context), bin("A", PredicateRole::Action)],
        vec![1.0; 4],
    );
    assert_eq!(bad.validate(), Err("reserved predicate name"));
}
