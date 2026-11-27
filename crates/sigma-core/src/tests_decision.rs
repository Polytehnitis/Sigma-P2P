//! Operator preferences and selection policies.

use super::*;
use crate::decision::Preferred;
use crate::tests::signed;
use PredicateRole::*;

fn bin(name: &str, role: PredicateRole) -> PredicateSchema {
    PredicateSchema::binary(name, role)
}

/// A and B are equally good means to G (a tie). Cue is a context predicate.
fn tied(id: &NodeIdentity) -> SigmaEngine {
    let mut engine = SigmaEngine::new();
    let a = signed(id, "a", vec![bin("G", Goal), bin("A", Action)], vec![1.0, 1.0, 1.0, 10.0]);
    let b = signed(id, "b", vec![bin("G", Goal), bin("B", Action)], vec![1.0, 1.0, 1.0, 10.0]);
    engine.compile_skill(a).unwrap();
    engine.compile_skill(b).unwrap();
    engine.register_predicate(&bin("Cue", Context)).unwrap();
    engine.push_goal("G");
    engine
}

fn decide_with(id: &NodeIdentity, prefs: &[Preference]) -> CycleOutcome {
    let mut engine = tied(id);
    for p in prefs {
        engine.add_preference(p.clone()).unwrap();
    }
    engine.cognitive_cycle()
}

#[test]
fn preferences_settle_what_knowledge_leaves_tied() {
    let id = NodeIdentity::generate();
    let out = decide_with(&id, &[]);
    assert_eq!(out.decision_impasse.unwrap().kind, ImpasseKind::Tie);
    let ops: Vec<&str> = out.operators.iter().map(|o| o.0.as_str()).collect();
    assert_eq!(ops.len(), 2);
    assert!(out.operators.iter().all(|o| (o.1 - 0.5).abs() < 1e-3), "{:?}", out.operators);

    let picks = |prefs: &[Preference]| {
        let out = decide_with(&id, prefs);
        assert_eq!(out.decision_impasse, None, "{prefs:?}");
        out.selected_action.unwrap()
    };
    use PreferenceKind::*;
    assert_eq!(picks(&[Preference::new("B", Reject)]), "A");
    assert_eq!(picks(&[Preference::new("B", Better { other: "A".into() })]), "B");
    assert_eq!(picks(&[Preference::new("B", Best)]), "B");
    assert_eq!(picks(&[Preference::new("A", Worst)]), "B");
    // Worst is still chosen when nothing else is left.
    assert_eq!(picks(&[Preference::new("A", Worst), Preference::new("B", Reject)]), "A");
    // The rejected operator is gone from the operator distribution.
    let out = decide_with(&id, &[Preference::new("B", Reject)]);
    assert_eq!(out.operators, vec![("A".to_string(), 1.0)]);

    // A conditional preference holds only in its situation.
    let mut engine = tied(&id);
    engine.add_preference(Preference::new("A", Reject).when("Cue", 1)).unwrap();
    engine.set_evidence("Cue", 0).unwrap();
    assert_eq!(engine.cognitive_cycle().decision_impasse.unwrap().kind, ImpasseKind::Tie);
    engine.set_evidence("Cue", 1).unwrap();
    let out = engine.cognitive_cycle();
    assert_eq!((out.selected_action.as_deref(), out.decision_impasse), (Some("B"), None));
    // Goals on the stack count as conditions too.
    let out = decide_with(&id, &[Preference::new("A", Reject).when("G", 1)]);
    assert_eq!(out.selected_action.as_deref(), Some("B"));

    // Indifferent operators are chosen between without an impasse, reproducibly.
    let indifferent =
        [Preference::new("A", Indifferent { other: None }), Preference::new("B", Indifferent { other: None })];
    let first = picks(&indifferent);
    assert_eq!(picks(&indifferent), first);
    let pair = [Preference::new("A", Indifferent { other: Some("B".into()) })];
    assert!(["A", "B"].contains(&picks(&pair).as_str()));
    // Rejecting every candidate leaves a none impasse.
    let out = decide_with(&id, &[Preference::new("A", Reject), Preference::new("B", Reject)]);
    assert_eq!(out.decision_impasse.unwrap().kind, ImpasseKind::None);
    assert_eq!(out.selected_action, None);
}

/// How often each of `ranked` is selected in `n` decisions.
fn frequencies(engine: &mut SigmaEngine, ranked: &[(usize, f32)], n: usize) -> Vec<f32> {
    let p = Preferred { ranked: ranked.to_vec(), indifferent: Vec::new() };
    let mut counts = vec![0usize; ranked.len()];
    for _ in 0..n {
        let v = engine.select_operator(&p).unwrap();
        counts[ranked.iter().position(|c| c.0 == v).unwrap()] += 1;
    }
    counts.into_iter().map(|c| c as f32 / n as f32).collect()
}

#[test]
fn sampled_selection_follows_the_operator_distribution() {
    let id = NodeIdentity::generate();
    let mut engine = tied(&id);
    let (a, b) = (engine.index["A"], engine.index["B"]);
    let ranked = [(a, 0.9), (b, 0.6)];

    // Best: the most probable, always.
    assert_eq!(frequencies(&mut engine, &ranked, 50), [1.0, 0.0]);
    // Ties impasse under Best ...
    let tie = Preferred { ranked: vec![(a, 0.8), (b, 0.8)], indifferent: Vec::new() };
    assert_eq!(engine.select_operator(&tie), Err(vec![a, b]));
    // ... but never when sampling.
    engine.selection = Selection::ProbabilityMatching;
    assert!(engine.select_operator(&tie).is_ok());

    // Probability matching: in proportion to the probabilities (0.6 and 0.4).
    let f = frequencies(&mut engine, &ranked, 2000);
    assert!((f[0] - 0.6).abs() < 0.05, "{f:?}");
    assert!((engine.operator_distribution(&ranked).iter().map(|o| o.1).sum::<f32>() - 1.0).abs() < 1e-6);

    // Boltzmann: a cold temperature is nearly greedy, a hot one nearly uniform.
    engine.selection = Selection::Boltzmann { temperature: 0.02 };
    assert!(frequencies(&mut engine, &ranked, 500)[0] > 0.99);
    engine.selection = Selection::Boltzmann { temperature: 10.0 };
    let f = frequencies(&mut engine, &ranked, 2000);
    assert!((f[0] - 0.5).abs() < 0.06, "{f:?}");

    // Reproducible for a seed.
    let run = |seed: u64| {
        let mut e = tied(&id);
        e.selection = Selection::ProbabilityMatching;
        e.selection_seed = seed;
        let p = Preferred { ranked: ranked.to_vec(), indifferent: Vec::new() };
        (0..20).map(|_| e.select_operator(&p).unwrap()).collect::<Vec<_>>()
    };
    assert_eq!(run(7), run(7));

    // In a cycle, a sampled choice settles what would have been a tie.
    let mut engine = tied(&id);
    engine.selection = Selection::Boltzmann { temperature: 0.1 };
    let out = engine.cognitive_cycle();
    assert_eq!(out.decision_impasse, None);
    assert!(out.selected_action.is_some());
}

#[test]
fn preferences_follow_the_user_across_devices() {
    let open = || {
        let path = std::env::temp_dir().join(format!("sigma-prefs-{}.redb", rand::random::<u64>()));
        let store = FactorStore::open(path).unwrap();
        let me = store.load_or_create_identity().unwrap().author_pubkey();
        (store, me)
    };
    let (phone, pc) = (open(), open());
    phone.0.pair_device(&pc.1).unwrap();
    pc.0.pair_device(&phone.1).unwrap();
    let prefs = vec![Preference::new("B", PreferenceKind::Reject).when("Cue", 1)];
    phone.0.save_preferences(&prefs).unwrap();
    let mut engine = SigmaEngine::new();
    let report = pc.0.merge_sync_state(phone.1, phone.0.export_sync_state().unwrap()).unwrap();
    pc.0.refresh_engine(&mut engine, &report).unwrap();
    assert_eq!(engine.preferences, prefs);
    assert_eq!(pc.0.load_preferences().unwrap(), prefs);
}
