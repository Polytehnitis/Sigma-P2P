//! Predicate options: closed/open world, persistence, unique/multi-valued, exponential and
//! unnormalized beliefs, and format-v3 skills that carry them.

use super::*;
use crate::tests::signed;
use crate::tests_v2::{device_backup_program, signed_program};
use PredicateRole::*;

fn bin(name: &str, role: PredicateRole) -> PredicateSchema {
    PredicateSchema::binary(name, role)
}

fn opts(f: impl FnOnce(&mut PredicateOptions)) -> PredicateOptions {
    let mut o = PredicateOptions::default();
    f(&mut o);
    o
}

fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3)
}

#[test]
fn closed_world_predicates_default_to_false() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    // The alarm goes off when the door opens.
    engine
        .compile_skill(signed(
            &id,
            "alarm",
            vec![bin("Door", Context), bin("Alarm", Context)],
            vec![9.0, 1.0, 1.0, 9.0],
        ))
        .unwrap();
    engine.set_evidence("Alarm", 1).unwrap();
    // Open world (the default): the door is inferred to be open.
    assert!(engine.posterior("Door").unwrap()[1] > 0.85);
    // Closed world: what working memory does not hold is false.
    engine.set_predicate_options("Door", opts(|o| o.closed_world = true)).unwrap();
    assert!(close(&engine.posterior("Door").unwrap(), &[1.0, 0.0]));
    // Evidence and percepts still say otherwise.
    engine.perceive("Door", &[0.3, 0.7]).unwrap();
    assert!(engine.posterior("Door").unwrap()[1] > 0.9);
    engine.set_evidence("Door", 1).unwrap();
    assert!(close(&engine.posterior("Door").unwrap(), &[0.0, 1.0]));
}

#[test]
fn non_persistent_evidence_lasts_one_cycle() {
    let mut engine = SigmaEngine::new();
    for p in ["Doorbell", "Home"] {
        engine.register_predicate(&bin(p, Context)).unwrap();
    }
    engine.set_predicate_options("Doorbell", opts(|o| o.persistent = false)).unwrap();
    engine.set_evidence("Doorbell", 1).unwrap();
    engine.set_evidence("Home", 1).unwrap();
    engine.push_goal("G");
    engine.cognitive_cycle();
    // The episode saw the doorbell, but it is gone from working memory now.
    assert!(engine.episodic.episodes.back().unwrap().state.contains(&("Doorbell".to_string(), 1)));
    assert_eq!(engine.working_memory(), [("Home".to_string(), 1)]);
}

#[test]
fn multi_valued_predicates_hold_sets_and_are_maximized_over() {
    let id = NodeIdentity::generate();
    let colour = PredicateSchema { name: "Seen".into(), domain_size: 3, role: Context };
    let build = |unique: bool| {
        let mut engine = SigmaEngine::new();
        // Happy given the colour seen: red [1, 9], green [9, 1], blue [5, 5].
        let t = vec![1.0, 9.0, 9.0, 1.0, 5.0, 5.0];
        let mut skill = signed(&id, "mood", vec![colour.clone(), bin("Happy", Context)], t);
        skill.factor_shape = vec![3, 2];
        id.sign_skill(&mut skill);
        engine.compile_skill(skill).unwrap();
        engine.set_predicate_options("Seen", opts(|o| o.unique = unique)).unwrap();
        engine
    };
    let mut engine = build(true);
    assert_eq!(engine.set_evidence_values("Seen", &[0, 2]), Err("predicate is unique: use set_evidence"));
    // Unique: red or blue (one of them) sums: 14 : 6.
    engine.perceive("Seen", &[0.5, 0.0, 0.5]).unwrap();
    assert!(close(&engine.posterior("Happy").unwrap(), &[0.3, 0.7]));

    // Multi-valued: red *and* blue both hold. Maximized over: 9 : 5.
    let mut engine = build(false);
    engine.set_evidence_values("Seen", &[2, 0, 2]).unwrap();
    assert!(close(&engine.posterior("Happy").unwrap(), &[5.0 / 14.0, 9.0 / 14.0]));
    // The belief is a truth level per value.
    assert!(close(&engine.posterior("Seen").unwrap(), &[1.0, 0.0, 1.0]));
    let wm = engine.working_memory();
    assert!(wm.contains(&("Seen".to_string(), 0)) && wm.contains(&("Seen".to_string(), 2)));
    assert!(engine.set_evidence_values("Seen", &[3]).is_err());
    engine.set_evidence("Seen", 1).unwrap();
    assert_eq!(engine.working_memory(), [("Seen".to_string(), 1)]);
}

#[test]
fn exponential_and_unnormalized_beliefs() {
    let mut engine = SigmaEngine::new();
    engine.register_predicate(&bin("U", Context)).unwrap();
    engine.perceive("U", &[0.2, 0.8]).unwrap();
    engine.set_predicate_options("U", opts(|o| o.exponential = true)).unwrap();
    let (a, b) = (0.2f32.exp(), 0.8f32.exp());
    assert!(close(&engine.posterior("U").unwrap(), &[a / (a + b), b / (a + b)]));
    engine.set_predicate_options("U", opts(|o| o.normalize = false)).unwrap();
    assert!(close(&engine.posterior("U").unwrap(), &[0.25, 1.0]));
}

#[test]
fn v3_skills_carry_predicate_options() {
    let id = NodeIdentity::generate();
    let closed = opts(|o| o.closed_world = true);
    let mut skill =
        Skill::program_with_options("device-backup", 1, device_backup_program(), vec![("Mounted".into(), closed)]);
    id.sign(&mut skill);
    assert_eq!(skill.format_version(), 3);
    assert_eq!(skill.validate(), Ok(()));
    // Not the v2 signature: the formats never collide.
    let v2 = signed_program(&id, "device-backup", device_backup_program());
    assert_ne!(crypto::any_skill_digest(&skill), crypto::any_skill_digest(&v2));
    let mut tampered = skill.clone();
    let SkillBody::Extended(_, x) = &mut tampered.body else { unreachable!() };
    x.predicate_options[0].1.closed_world = false;
    assert!(tampered.validate().is_err());

    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill.clone()).unwrap();
    assert!(engine.predicate_options("Mounted(nas)").closed_world);
    assert!(close(&engine.posterior("Mounted(usb)").unwrap(), &[1.0, 0.0]));
    assert!(!engine.predicate_options("Backup(nas)").closed_world);
    // Another skill may not declare different options for the same predicate.
    let mut other = Skill::program_with_options(
        "other",
        1,
        device_backup_program(),
        vec![("Mounted".into(), PredicateOptions::default())],
    );
    id.sign(&mut other);
    assert_eq!(engine.compile_skill(other), Err("predicate option conflict"));

    // Options must name a declared predicate, once, and keep continuous predicates open, unique.
    let bad = |o: Vec<(String, PredicateOptions)>, program: Program| {
        let mut s = Skill::program_with_options("bad", 1, program, o);
        id.sign(&mut s);
        s.validate()
    };
    assert_eq!(bad(vec![("Nope".into(), closed)], device_backup_program()), Err("options for an unknown predicate"));
    assert_eq!(
        bad(vec![("Mounted".into(), closed), ("Mounted".into(), closed)], device_backup_program()),
        Err("duplicate predicate options")
    );
    let mut temp = device_backup_program();
    temp.types.push(TypeDef::continuous("celsius", 0.0, 40.0, 8));
    temp.predicates.push(PredicateDef {
        name: "Temp".into(),
        role: Context,
        args: vec![],
        value_type: Some("celsius".into()),
    });
    assert!(bad(vec![("Temp".into(), opts(|o| o.unique = false))], temp).is_err());

    // A v3 skill round-trips through the store.
    let path = std::env::temp_dir().join(format!("sigma-v3-{}.redb", rand::random::<u64>()));
    let store = FactorStore::open(&path).unwrap();
    store.save_skill(&skill).unwrap();
    let mut restored = SigmaEngine::new();
    store.restore_engine(&mut restored).unwrap();
    assert!(restored.predicate_options("Mounted(nas)").closed_world);
}
