//! Particle messages inside the graph (particle belief propagation).

use super::*;
use crate::tests_v2::signed_program;

fn pred(name: &str, role: PredicateRole, value_type: Option<&str>) -> PredicateDef {
    PredicateDef { name: name.into(), role, args: vec![], value_type: value_type.map(Into::into) }
}

fn pat(kind: PatternKind, name: &str) -> Pattern {
    Pattern { kind, predicate: name.into(), args: vec![] }
}

fn conditional(patterns: Vec<Pattern>, function: Function) -> Conditional {
    Conditional { name: "c".into(), patterns, function_vars: vec![], function, learnable: false, normal: None }
}

/// Inside ~ N(5 + 0.5 * Outside, 1) over [0, 40), with only `bins` regions.
fn insulation(bins: usize) -> SigmaEngine {
    let program = Program {
        types: vec![TypeDef::continuous("celsius", 0.0, 40.0, bins)],
        predicates: vec![
            pred("Outside", PredicateRole::Context, Some("celsius")),
            pred("Inside", PredicateRole::Context, Some("celsius")),
        ],
        conditionals: vec![conditional(
            vec![pat(PatternKind::Condition, "Outside"), pat(PatternKind::Condact, "Inside")],
            Function::Gaussian { target: 1, mean: vec![5.0, 0.5, 0.0], std: 1.0 },
        )],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&NodeIdentity::generate(), "insulation", program)).unwrap();
    engine
}

#[test]
fn particles_resolve_what_coarse_regions_cannot() {
    let mut engine = insulation(4);
    engine.set_value("Outside", 22.0).unwrap();
    let grid = engine.expected_value("Inside").unwrap();
    assert!((grid - 16.0).abs() > 0.7, "four 10-degree regions blur the answer: {grid}");

    engine.particle_count = Some(128);
    let belief = engine.particle_belief("Inside").unwrap();
    assert_eq!(belief.values.len(), 128);
    assert!((belief.mean() - 16.0).abs() < 0.3, "E[Inside] = {}", belief.mean());
    assert!((0.5..1.6).contains(&belief.variance()), "Var[Inside] = {}", belief.variance());
    assert!((engine.expected_value("Inside").unwrap() - belief.mean()).abs() < 1e-6);
    // The region view is the particle histogram: almost all mass in [10, 20).
    assert!(engine.posterior("Inside").unwrap()[1] > 0.9);
    // Reproducible for a given state and seed.
    assert_eq!(engine.particle_belief("Inside").unwrap(), belief);
}

#[test]
fn particle_messages_flow_through_the_graph() {
    let mut engine = insulation(4);
    engine.particle_count = Some(128);
    // An incoming particle message about Outside (a sensor ensemble around 12).
    let msg = Particles { values: vec![11.6, 11.9, 12.0, 12.1, 12.4], weights: vec![1.0; 5] };
    engine.perceive_particles("Outside", &msg).unwrap();
    let inside = engine.particle_belief("Inside").unwrap();
    assert!((inside.mean() - 11.0).abs() < 0.35, "E[Inside] = {}", inside.mean());
    // Evidence downstream flows back up: observing Inside informs Outside.
    let mut engine = insulation(4);
    engine.particle_count = Some(128);
    let rev = Program {
        types: vec![TypeDef::continuous("celsius", 0.0, 40.0, 4)],
        predicates: vec![
            pred("Outside", PredicateRole::Context, Some("celsius")),
            pred("Inside", PredicateRole::Context, Some("celsius")),
        ],
        conditionals: vec![conditional(
            vec![pat(PatternKind::Condact, "Outside"), pat(PatternKind::Condact, "Inside")],
            Function::Gaussian { target: 1, mean: vec![5.0, 0.5, 0.0], std: 1.0 },
        )],
    };
    engine.compile_skill(signed_program(&NodeIdentity::generate(), "insulation", rev)).unwrap();
    engine.set_value("Inside", 21.0).unwrap();
    let outside = engine.particle_belief("Outside").unwrap();
    assert!((outside.mean() - 32.0).abs() < 1.0, "E[Outside] = {}", outside.mean());
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);
    let drawn = engine.sample_particles("Outside", 500, &mut rng).unwrap();
    assert!((drawn.mean() - outside.mean()).abs() < 0.3);
}

#[test]
fn particle_decisions_use_exact_values() {
    // Heat below 13 degrees; regions are 5 degrees wide, so 14 sits in a region centred at 12.5.
    let program = Program {
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
            Function::PiecewiseLinear {
                default: 1.0,
                pieces: vec![Piece {
                    lo: vec![0.5, 0.0, 0.5],
                    hi: vec![1.5, 13.0, 1.5],
                    coef: vec![9.0, 0.0, 0.0, 0.0],
                }],
            },
        )],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&NodeIdentity::generate(), "thermostat", program)).unwrap();
    engine.push_goal("Comfort");
    engine.set_value("Temp", 14.0).unwrap();
    assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("Heater"), "grid: region midpoint 12.5");
    engine.particle_count = Some(64);
    assert_eq!(engine.cognitive_cycle().selected_action, None, "particles: exactly 14");
    engine.set_value("Temp", 9.0).unwrap();
    assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("Heater"));
}
