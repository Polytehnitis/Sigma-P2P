//! Continuous types as predicate arguments.

use super::*;
use crate::tests_v2::signed_program;
use PatternKind::Condact;

fn pat(predicate: &str, args: Vec<Term>) -> Pattern {
    Pattern { kind: Condact, predicate: predicate.into(), args }
}

fn conditional(name: &str, patterns: Vec<Pattern>, function_vars: &[&str], function: Function) -> Conditional {
    Conditional {
        name: name.into(),
        patterns,
        function_vars: function_vars.iter().map(|v| v.to_string()).collect(),
        function,
        learnable: false,
        normal: None,
    }
}

fn position() -> TypeDef {
    TypeDef::continuous("position", 0.0, 10.0, 5)
}

/// `Heat(x)` over a continuous position: the further along, the likelier it is hot
/// (`P(Heat(x)) ∝ x` against `10 - x`).
fn heat_field() -> Program {
    Program {
        types: vec![position()],
        predicates: vec![PredicateDef {
            name: "Heat".into(),
            role: PredicateRole::Context,
            args: vec!["position".into()],
            value_type: None,
        }],
        conditionals: vec![conditional(
            "rising",
            vec![pat("Heat", vec![Term::Var("x".into())])],
            &["x"],
            // Dimensions: x (its region's midpoint), then Heat's value.
            Function::PiecewiseLinear {
                default: 0.0,
                pieces: vec![
                    Piece { lo: vec![0.0, 0.0], hi: vec![10.0, 1.0], coef: vec![10.0, -1.0, 0.0] },
                    Piece { lo: vec![0.0, 1.0], hi: vec![10.0, 2.0], coef: vec![0.0, 1.0, 0.0] },
                ],
            },
        )],
    }
}

#[test]
fn continuous_arguments_ground_by_region() {
    let t = position();
    assert_eq!(t.argument_values(), ["0", "2", "4", "6", "8"]);
    assert_eq!(t.argument_of(7.3).as_deref(), Some("6"));
    assert_eq!(t.argument_of(99.0).as_deref(), Some("8"), "clamped");
    assert_eq!(TypeDef::symbolic("s", &["a"]).argument_of(1.0), None);

    let id = NodeIdentity::generate();
    let skill = signed_program(&id, "heat", heat_field());
    assert_eq!(skill.validate(), Ok(()));
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill).unwrap();
    // One atom per region, and the function sees each region's midpoint.
    for (atom, mid) in [("Heat(0)", 1.0), ("Heat(4)", 5.0), ("Heat(8)", 9.0)] {
        let p = engine.posterior(atom).unwrap();
        assert!((p[1] - mid / 10.0).abs() < 1e-4, "{atom}: {p:?}");
    }
    assert!(engine.posterior("Heat(10)").is_none());
}

#[test]
fn affine_transforms_shift_continuous_arguments() {
    // Heat spreads one region along: Heat(x) and Heat(x + 1 region) agree.
    let mut program = heat_field();
    program.conditionals.push(conditional(
        "spread",
        vec![pat("Heat", vec![Term::Var("x".into())]), pat("Heat", vec![Term::shift("x", 1, None)])],
        &[],
        Function::Table(vec![9.0, 1.0, 1.0, 9.0]),
    ));
    program.conditionals.remove(0);
    let id = NodeIdentity::generate();
    let g = signed_program(&id, "spread", program.clone()).grounding().unwrap();
    assert_eq!(g.factors.len(), 4, "five regions, four neighbouring pairs");
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&id, "spread", program)).unwrap();
    engine.set_evidence("Heat(0)", 1).unwrap();
    let (near, far) = (engine.posterior("Heat(2)").unwrap()[1], engine.posterior("Heat(8)").unwrap()[1]);
    assert!(near > 0.85 && far > 0.5 && far < near, "{near} {far}");
}

#[test]
fn continuous_arguments_with_continuous_values() {
    // Temperature along a corridor: a hybrid predicate whose argument and value are both
    // continuous. The reading at a position is near 20 + position.
    let program = Program {
        types: vec![position(), TypeDef::continuous("celsius", 10.0, 40.0, 6)],
        predicates: vec![PredicateDef {
            name: "Temp".into(),
            role: PredicateRole::Context,
            args: vec!["position".into()],
            value_type: Some("celsius".into()),
        }],
        conditionals: vec![conditional(
            "gradient",
            vec![pat("Temp", vec![Term::Var("x".into())])],
            &["x"],
            Function::Gaussian { target: 1, mean: vec![20.0, 1.0, 0.0], std: 2.0 },
        )],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&NodeIdentity::generate(), "corridor", program)).unwrap();
    let (start, end) = (engine.expected_value("Temp(0)").unwrap(), engine.expected_value("Temp(8)").unwrap());
    assert!((start - 21.0).abs() < 0.5 && (end - 29.0).abs() < 0.5, "{start} {end}");
    engine.set_value("Temp(4)", 33.0).unwrap();
    assert_eq!(engine.working_memory(), [("Temp(4)".to_string(), 4)]);
}
