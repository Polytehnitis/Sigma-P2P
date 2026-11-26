//! Exact piecewise-linear message passing over continuous variables.

use super::*;
use crate::tests_v2::signed_program;

fn pred(name: &str, role: PredicateRole, value_type: Option<&str>) -> PredicateDef {
    PredicateDef { name: name.into(), role, args: vec![], value_type: value_type.map(Into::into) }
}

fn pat(kind: PatternKind, name: &str) -> Pattern {
    Pattern { kind, predicate: name.into(), args: vec![] }
}

fn engine_with(types: Vec<TypeDef>, predicates: Vec<PredicateDef>, patterns: Vec<Pattern>, f: Function) -> SigmaEngine {
    let program = Program {
        types,
        predicates,
        conditionals: vec![Conditional {
            name: "c".into(),
            patterns,
            function_vars: vec![],
            function: f,
            learnable: false,
            normal: None,
        }],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&NodeIdentity::generate(), "s", program)).unwrap();
    engine
}

/// A prior density over Temp in [0, 40) that rises linearly, f(t) = t, on only two regions.
fn rising(bins: usize) -> SigmaEngine {
    engine_with(
        vec![TypeDef::continuous("celsius", 0.0, 40.0, bins)],
        vec![pred("Temp", PredicateRole::Context, Some("celsius"))],
        vec![pat(PatternKind::Condact, "Temp")],
        Function::PiecewiseLinear {
            default: 0.0,
            pieces: vec![Piece { lo: vec![0.0], hi: vec![40.0], coef: vec![0.0, 1.0] }],
        },
    )
}

#[test]
fn integrals_are_exact_whatever_the_region_count() {
    let mut engine = rising(2);
    // ∫ t·t / ∫ t over [0, 40) = 2/3 · 40.
    let exact = engine.expected_value("Temp").unwrap();
    assert!((exact - 80.0 / 3.0).abs() < 1e-3, "E[Temp] = {exact}");
    // Region masses are exact too: ∫_0^20 t / ∫_0^40 t = 1/4.
    let post = engine.posterior("Temp").unwrap();
    assert!((post[0] - 0.25).abs() < 1e-5, "{post:?}");
    let density = engine.density("Temp").unwrap();
    assert_eq!(density.bounds[0], vec![0.0, 40.0], "one slice: the function never changes");
    assert!((density.eval(&[30.0]) - 30.0 / 800.0).abs() < 1e-9);

    engine.tabulate_continuous = true;
    let grid = engine.expected_value("Temp").unwrap();
    assert!((grid - 25.0).abs() < 1e-3, "the grid sees only region midpoints: {grid}");
}

#[test]
fn slices_follow_the_pieces_not_the_regions() {
    // Comfort ∈ {0,1}; a piece over Temp ∈ [0, 13.7) only, on a 4-region grid.
    let mut engine = engine_with(
        vec![TypeDef::continuous("celsius", 0.0, 40.0, 4)],
        vec![pred("Cold", PredicateRole::Context, None), pred("Temp", PredicateRole::Context, Some("celsius"))],
        vec![pat(PatternKind::Condact, "Cold"), pat(PatternKind::Condact, "Temp")],
        Function::PiecewiseLinear {
            default: 1.0,
            pieces: vec![Piece { lo: vec![0.5, 0.0], hi: vec![1.5, 13.7], coef: vec![9.0, 0.0, 0.0] }],
        },
    );
    engine.set_evidence("Cold", 1).unwrap();
    let d = engine.density("Temp").unwrap();
    assert_eq!(d.bounds[0], vec![0.0, 13.7_f32 as f64, 40.0]);
    // P(Temp < 13.7 | Cold) = 9·13.7 / (9·13.7 + 26.3).
    let want = 9.0 * 13.7 / (9.0 * 13.7 + 26.3);
    let (m0, _) = d.moments(0.0, 13.7_f32 as f64);
    assert!((m0 - want).abs() < 1e-5, "{m0} vs {want}");
    // Exact evidence flows back: a reading of 13.6 says Cold, one of 13.8 does not.
    engine.clear_evidence("Cold");
    engine.set_value("Temp", 13.6).unwrap();
    assert!((engine.posterior("Cold").unwrap()[1] - 0.9).abs() < 1e-5);
    engine.set_value("Temp", 13.8).unwrap();
    assert!((engine.posterior("Cold").unwrap()[1] - 0.5).abs() < 1e-5);
}

#[test]
fn chains_of_continuous_variables_stay_exact() {
    // Inside ~ N(5 + 0.5·Outside, 1), Outside a soft reading around 30.
    let mut engine = engine_with(
        vec![TypeDef::continuous("celsius", 0.0, 40.0, 4)],
        vec![
            pred("Outside", PredicateRole::Context, Some("celsius")),
            pred("Inside", PredicateRole::Context, Some("celsius")),
        ],
        vec![pat(PatternKind::Condact, "Outside"), pat(PatternKind::Condact, "Inside")],
        Function::Gaussian { target: 1, mean: vec![5.0, 0.5, 0.0], std: 1.0 },
    );
    engine.set_value("Inside", 21.0).unwrap();
    // Outside = 2·(21 - 5) = 32, both directions through the same factor.
    let outside = engine.expected_value("Outside").unwrap();
    assert!((outside - 32.0).abs() < 0.1, "E[Outside] = {outside}");
    // Learning still runs on the region tables, and scales the exact function.
    engine.clear_evidence("Inside");
    let before = engine.expected_value("Inside").unwrap();
    let w = engine.skill_weights("s").unwrap();
    let mut boosted = w.clone();
    // Favour Outside in [30, 40) (the last of 4 regions) for every Inside region.
    for i in 0..4 {
        boosted[3 * 4 + i] = (w[3 * 4 + i] * 50.0).min(MAX_FACTOR_VALUE);
    }
    assert!(engine.apply_weights("s", &boosted));
    let after = engine.expected_value("Inside").unwrap();
    assert!(after > before + 2.0, "{before} -> {after}");
}

#[test]
fn discrete_graphs_are_unchanged() {
    // The device-backup chain has no continuous variables: same cycle as on the grid.
    let id = NodeIdentity::generate();
    let mut a = SigmaEngine::new();
    a.compile_skill(signed_program(&id, "b", crate::tests_v2::device_backup_program())).unwrap();
    a.push_goal("Backup(nas)");
    let mut b = SigmaEngine::new();
    b.compile_skill(signed_program(&id, "b", crate::tests_v2::device_backup_program())).unwrap();
    b.push_goal("Backup(nas)");
    b.tabulate_continuous = true;
    let (x, y) = (a.cognitive_cycle(), b.cognitive_cycle());
    assert_eq!(x.selected_action, y.selected_action);
    assert_eq!(x.beliefs, y.beliefs);
}
