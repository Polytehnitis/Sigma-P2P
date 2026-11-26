//! Format v2 (conditionals with variables) and the decision cycle's impasses.

use super::*;
use PatternKind::{Condact, Condition};
use PredicateRole::*;

fn var(x: &str) -> Term {
    Term::Var(x.into())
}

fn pat(kind: PatternKind, predicate: &str, args: Vec<Term>) -> Pattern {
    Pattern { kind, predicate: predicate.into(), args }
}

fn pred(name: &str, role: PredicateRole, args: &[&str]) -> PredicateDef {
    PredicateDef { name: name.into(), role, args: args.iter().map(|a| a.to_string()).collect(), value_type: None }
}

fn cond(name: &str, patterns: Vec<Pattern>, function: Vec<f32>) -> Conditional {
    Conditional { name: name.into(), patterns, function_vars: vec![], function }
}

pub(crate) fn signed_program(id: &NodeIdentity, name: &str, program: Program) -> Skill {
    let mut skill = Skill::program(name, 1, program);
    id.sign(&mut skill);
    skill
}

/// The backup chain of `tests.rs`, but over any `device`, as two conditionals.
pub(crate) fn device_backup_program() -> Program {
    let d = || vec![var("d")];
    Program {
        types: vec![TypeDef { name: "device".into(), values: vec!["nas".into(), "usb".into()] }],
        predicates: vec![
            pred("Backup", Goal, &["device"]),
            pred("Mounted", Subgoal, &["device"]),
            pred("Mount", Action, &["device"]),
            pred("Sync", Action, &["device"]),
        ],
        conditionals: vec![
            cond(
                "backup-needs-mount",
                vec![
                    pat(Condition, "Backup", d()),
                    pat(Condact, "Mounted", d()),
                    pat(PatternKind::Action, "Mount", d()),
                ],
                vec![1.0, 1.0, 1.0, 1.0, 0.1, 5.0, 10.0, 2.0],
            ),
            cond(
                "sync-when-mounted",
                vec![pat(Condact, "Mounted", d()), pat(PatternKind::Action, "Sync", d())],
                vec![5.0, 0.1, 1.0, 10.0],
            ),
        ],
    }
}

#[test]
fn v2_grounds_signs_and_rejects_tampering() {
    let id = NodeIdentity::generate();
    let skill = signed_program(&id, "device-backup", device_backup_program());
    assert_eq!(skill.format_version(), 2);
    assert_eq!(skill.validate(), Ok(()));
    let g = skill.grounding().unwrap();
    assert_eq!(g.atoms.len(), 8);
    assert_eq!(g.factors.len(), 4, "two conditionals x two devices");
    assert_eq!(g.factors[1].predicates[2].name, "Mount(usb)");

    let mut tampered = skill.clone();
    let SkillBody::Program(p) = &mut tampered.body else { unreachable!() };
    p.conditionals[0].function[5] = 9.0;
    assert_eq!(tampered.validate(), Err("signature verification failed"));

    // A v1 package converted to `Skill` keeps its v1 signature.
    let v1 = Skill::from(crate::tests::skill_sync(&id));
    assert_eq!(v1.format_version(), 1);
    assert_eq!(v1.validate(), Ok(()));
    assert_eq!(crypto::any_skill_digest(&v1), crypto::skill_digest(&v1.as_v1().unwrap()));

    // Structural errors are caught before signatures.
    let mut bad = device_backup_program();
    bad.conditionals[0].patterns[0].args = vec![Term::Const("floppy".into())];
    assert_eq!(signed_program(&id, "bad", bad).validate(), Err("constant outside its type"));
    let mut bad = device_backup_program();
    bad.predicates[0].name = "Backup(x)".into();
    assert_eq!(signed_program(&id, "bad", bad).validate(), Err("invalid predicate name"));
    let mut bad = device_backup_program();
    bad.conditionals[1].function.pop();
    assert_eq!(signed_program(&id, "bad", bad).validate(), Err("function length does not match conditional"));
}

#[test]
fn v2_conditionals_chain_for_any_binding() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&id, "device-backup", device_backup_program())).unwrap();
    engine.push_goal("Backup(usb)");

    let step = engine.cognitive_cycle();
    assert_eq!(step.goal_stack, ["Backup(usb)", "Mounted(usb)"]);
    assert_eq!(step.selected_action.as_deref(), Some("Mount(usb)"));
    assert_eq!(step.impasse, None);

    engine.set_evidence("Mounted(usb)", 1).unwrap();
    assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("Sync(usb)"));
}

#[test]
fn condition_and_action_patterns_are_one_way() {
    let id = NodeIdentity::generate();
    let program = Program {
        types: vec![],
        predicates: vec![pred("Rain", Context, &[]), pred("Umbrella", Action, &[])],
        conditionals: vec![cond(
            "rain-umbrella",
            vec![pat(Condition, "Rain", vec![]), pat(PatternKind::Action, "Umbrella", vec![])],
            vec![1.0, 1.0, 1.0, 9.0],
        )],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&id, "rain", program)).unwrap();
    engine.set_evidence("Rain", 1).unwrap();
    let p = engine.infer(|_| None);
    assert!((p[1][1] - 0.9).abs() < 1e-4, "condition informs action");
    engine.clear_evidence("Rain");
    engine.set_evidence("Umbrella", 1).unwrap();
    let p = engine.infer(|_| None);
    assert!((p[0][1] - 0.5).abs() < 1e-6, "action does not inform condition");
}

#[test]
fn function_variables_give_each_binding_its_own_slice() {
    let id = NodeIdentity::generate();
    let program = Program {
        types: vec![TypeDef { name: "device".into(), values: vec!["nas".into(), "usb".into()] }],
        predicates: vec![pred("Backup", Goal, &[]), pred("Use", Action, &["device"])],
        conditionals: vec![Conditional {
            name: "prefer".into(),
            patterns: vec![pat(Condition, "Backup", vec![]), pat(PatternKind::Action, "Use", vec![var("d")])],
            function_vars: vec!["d".into()],
            function: vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 8.0],
        }],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&id, "prefer", program)).unwrap();
    engine.push_goal("Backup");
    let out = engine.cognitive_cycle();
    assert_eq!(out.selected_action.as_deref(), Some("Use(usb)"));
    assert_eq!(out.decision_impasse, None);
}

/// Goal G with two actions; `b_without_goal` sets f(G=0, B=1).
fn tie_engine(b_without_goal: f32) -> SigmaEngine {
    let id = NodeIdentity::generate();
    let bin = PredicateSchema::binary;
    let mut engine = SigmaEngine::new();
    let a = crate::tests::signed(&id, "a", vec![bin("G", Goal), bin("A", Action)], vec![1.0, 1.0, 1.0, 10.0]);
    let b =
        crate::tests::signed(&id, "b", vec![bin("G", Goal), bin("B", Action)], vec![1.0, b_without_goal, 1.0, 10.0]);
    engine.compile_skill(a).unwrap();
    engine.compile_skill(b).unwrap();
    engine.push_goal("G");
    engine
}

#[test]
fn tie_impasse_resolved_by_lookahead_or_left_open() {
    // B is just as likely given the goal, but also common without it: A is the better means.
    let out = tie_engine(5.0).cognitive_cycle();
    let imp = out.decision_impasse.unwrap();
    assert_eq!((imp.kind, imp.resolved), (ImpasseKind::Tie, true));
    assert_eq!(imp.candidates.len(), 2);
    assert_eq!(out.selected_action.as_deref(), Some("A"));

    // Fully symmetric knowledge: the tie stays open and nothing is selected.
    let out = tie_engine(1.0).cognitive_cycle();
    let imp = out.decision_impasse.unwrap();
    assert_eq!((imp.kind, imp.resolved, imp.goal.as_str()), (ImpasseKind::Tie, false, "G"));
    assert_eq!(out.selected_action, None);
}

#[test]
fn no_change_and_none_impasses() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(crate::tests::skill_backup(&id)).unwrap();
    engine.compile_skill(crate::tests::skill_sync(&id)).unwrap();
    engine.push_goal("GoalBackup");
    assert_eq!(engine.cognitive_cycle().decision_impasse, None);
    for n in 1..=2 {
        let out = engine.cognitive_cycle();
        let imp = out.decision_impasse.unwrap();
        assert_eq!((imp.kind, imp.cycles), (ImpasseKind::NoChange, n));
        assert_eq!(out.selected_action.as_deref(), Some("ActionMountNAS"), "the action stays selected");
    }
    engine.set_evidence("SubgoalMountNAS", 1).unwrap();
    assert_eq!(engine.cognitive_cycle().decision_impasse, None, "working memory changed");

    let mut engine = SigmaEngine::new();
    engine.push_goal("GoalUnknown");
    let imp = engine.cognitive_cycle().decision_impasse.unwrap();
    assert_eq!((imp.kind, imp.goal.as_str()), (ImpasseKind::None, "GoalUnknown"));
}

#[test]
fn v2_learning_ties_shared_functions_and_persists() {
    let id = NodeIdentity::generate();
    let skill = signed_program(&id, "device-backup", device_backup_program());
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill.clone()).unwrap();
    engine.push_goal("Backup(usb)");
    engine.cognitive_cycle();
    engine.reinforce_factors(1.0);
    let w = engine.skill_weights("device-backup").unwrap();
    assert_eq!(w.len(), 12);
    assert!(w[5] > 5.0, "the usb binding's reward reached the shared function");
    let tables: Vec<Vec<f32>> =
        engine.factors().iter().filter(|f| f.offset == 0).map(|f| f.table.iter().copied().collect()).collect();
    assert_eq!(tables.len(), 2);
    assert_eq!(tables[0], tables[1], "both device bindings share one function");

    let path = std::env::temp_dir().join(format!("sigma-v2-{}.redb", rand::random::<u64>()));
    let store = FactorStore::open(&path).unwrap();
    store.save_skill(&skill).unwrap();
    store.save_skill_weights("device-backup", &w).unwrap();
    let mut restored = SigmaEngine::new();
    assert_eq!(store.restore_engine(&mut restored).unwrap(), vec![skill]);
    assert_eq!(restored.skill_weights("device-backup").unwrap(), w);
    assert!(store.delete_skill("device-backup").unwrap());
}
