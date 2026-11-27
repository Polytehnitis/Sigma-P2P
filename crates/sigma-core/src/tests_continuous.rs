//! Continuous and hybrid predicates: piecewise-linear and Gaussian functions, particles.

use super::*;
use crate::tests_v2::signed_program;
use rand::SeedableRng;

fn pred(name: &str, role: PredicateRole, value_type: Option<&str>) -> PredicateDef {
    PredicateDef { name: name.into(), role, args: vec![], value_type: value_type.map(Into::into) }
}

fn pat(kind: PatternKind, name: &str) -> Pattern {
    Pattern { kind, predicate: name.into(), args: vec![] }
}

fn conditional(patterns: Vec<Pattern>, function: Function) -> Conditional {
    Conditional { name: "c".into(), patterns, function_vars: vec![], function, learnable: false, normal: None }
}

/// Dimensions (Comfort, Temp in [0, 40) as 8 regions, Heater): heat when it's cold.
fn thermostat(pieces: Vec<Piece>) -> Program {
    Program {
        types: vec![TypeDef::continuous("celsius", 0.0, 40.0, 8)],
        predicates: vec![
            pred("Comfort", PredicateRole::Goal, None),
            pred("Temp", PredicateRole::Context, Some("celsius")),
            pred("Heater", PredicateRole::Action, None),
        ],
        conditionals: vec![conditional(
            vec![
                pat(PatternKind::Condition, "Comfort"),
                pat(PatternKind::Condition, "Temp"),
                pat(PatternKind::Action, "Heater"),
            ],
            Function::PiecewiseLinear { default: 1.0, pieces },
        )],
    }
}

fn cold_piece(slope: f32) -> Piece {
    Piece { lo: vec![0.5, 0.0, 0.5], hi: vec![1.5, 15.0, 1.5], coef: vec![9.0, 0.0, slope, 0.0] }
}

#[test]
fn piecewise_linear_hybrid_function_drives_decisions() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&id, "thermostat", thermostat(vec![cold_piece(-0.2)]))).unwrap();
    // Regions are 5 degrees wide; the cold piece covers centres 2.5, 7.5 and 12.5 with Heater = 1.
    let w = engine.skill_weights("thermostat").unwrap();
    assert_eq!(w.len(), 2 * 8 * 2);
    assert!((w[16 + 1] - 8.5).abs() < 1e-5 && (w[16 + 5] - 6.5).abs() < 1e-5 && w[16 + 7] == 1.0);

    engine.push_goal("Comfort");
    engine.set_value("Temp", 4.0).unwrap();
    assert_eq!(engine.variables()[engine.index["Temp"]].evidence, Some(0));
    assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("Heater"));
    engine.set_value("Temp", 31.0).unwrap();
    assert_eq!(engine.cognitive_cycle().selected_action, None);
    assert_eq!(engine.region_of("Temp", 99.0), Ok(7), "clamped to the range");
    assert!(engine.set_value("Comfort", 1.0).is_err(), "not continuous");
}

fn gaussian_program() -> Program {
    Program {
        types: vec![TypeDef::continuous("celsius", 0.0, 40.0, 40)],
        predicates: vec![
            pred("Outside", PredicateRole::Context, Some("celsius")),
            pred("Inside", PredicateRole::Context, Some("celsius")),
        ],
        conditionals: vec![conditional(
            vec![pat(PatternKind::Condition, "Outside"), pat(PatternKind::Condact, "Inside")],
            // Inside ~ N(5 + 0.5 * Outside, 2)
            Function::Gaussian { target: 1, mean: vec![5.0, 0.5, 0.0], std: 2.0 },
        )],
    }
}

#[test]
fn gaussian_conditional_and_particle_messages() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&id, "insulation", gaussian_program())).unwrap();
    engine.set_value("Outside", 20.0).unwrap();
    let inside = engine.expected_value("Inside").unwrap();
    assert!((inside - 15.25).abs() < 0.3, "E[Inside] = {inside}");

    // A noisy reading enters as a soft percept.
    engine.clear_evidence("Outside");
    engine.perceive_value("Outside", 30.0, 1.0).unwrap();
    let inside = engine.expected_value("Inside").unwrap();
    assert!((inside - 20.25).abs() < 0.4, "E[Inside] = {inside}");

    // Particles out: samples of the posterior agree with its mean.
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let particles = engine.sample_particles("Inside", 4000, &mut rng).unwrap();
    assert!((particles.mean() - inside).abs() < 0.2);
    assert!(particles.values.iter().all(|x| (0.0..40.0).contains(x)));

    // Particles in: a weighted sample message about Outside (all near 10).
    let msg = Particles { values: vec![9.6, 10.2, 10.4], weights: vec![1.0, 2.0, 1.0] };
    engine.perceive_particles("Outside", &msg).unwrap();
    assert!((engine.expected_value("Inside").unwrap() - 10.25).abs() < 0.4);
}

#[test]
fn continuous_types_are_validated() {
    let id = NodeIdentity::generate();
    let check = |p: Program| signed_program(&id, "x", p).validate();
    let mut p = thermostat(vec![cold_piece(0.0)]);
    p.predicates[2].args = vec!["nowhere".into()];
    assert_eq!(check(p), Err("unknown argument type"));
    let mut p = thermostat(vec![cold_piece(0.0)]);
    p.conditionals[0].function = Function::PiecewiseLinear {
        default: 1.0,
        pieces: vec![Piece { lo: vec![0.0], hi: vec![1.0], coef: vec![1.0] }],
    };
    assert_eq!(check(p), Err("invalid piecewise-linear function"));
    assert_eq!(check(thermostat(vec![cold_piece(-5.0)])), Err("factor value out of bounds"), "negative values");
    let mut p = gaussian_program();
    p.types[0] = TypeDef::continuous("celsius", 5.0, 5.0, 4);
    assert_eq!(check(p), Err("invalid continuous type"));
    // A continuous atom cannot be redefined with another range.
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&id, "a", gaussian_program())).unwrap();
    let mut p = gaussian_program();
    p.types[0] = TypeDef::continuous("celsius", 0.0, 80.0, 40);
    assert_eq!(engine.compile_skill(signed_program(&id, "b", p)), Err("predicate schema conflict"));
}
