//! Gradient-descent learning over factor functions.

use super::*;
use crate::tests_v2::signed_program;

fn weather_program(learnable: bool) -> Program {
    let pred =
        |name: &str| PredicateDef { name: name.into(), role: PredicateRole::Context, args: vec![], value_type: None };
    let pat = |kind, name: &str| Pattern { kind, predicate: name.into(), args: vec![] };
    Program {
        types: vec![],
        predicates: vec![pred("Rain"), pred("Wet")],
        conditionals: vec![Conditional {
            name: "wet-given-rain".into(),
            patterns: vec![pat(PatternKind::Condition, "Rain"), pat(PatternKind::Condact, "Wet")],
            function_vars: vec![],
            function: Function::Table(vec![1.0, 1.0, 1.0, 1.0]),
            learnable,
            normal: Some(1),
        }],
    }
}

/// Feeds `rounds` x 10 observations: Wet in 9/10 rainy cases and 2/10 dry ones.
fn train(engine: &mut SigmaEngine, rounds: usize) -> Vec<f32> {
    let mut fit = Vec::new();
    for _ in 0..rounds {
        let mut total = 0.0;
        for i in 0..10 {
            for (rain, wet) in [(1, usize::from(i < 9)), (0, usize::from(i < 2))] {
                engine.set_evidence("Rain", rain).unwrap();
                engine.set_evidence("Wet", wet).unwrap();
                total += engine.learn_gradient(0.02);
            }
        }
        fit.push(total);
    }
    fit
}

#[test]
fn gradient_learning_fits_a_conditional_distribution() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&id, "weather", weather_program(true))).unwrap();
    let fit = train(&mut engine, 60);
    assert!(fit.last() > fit.first(), "likelihood proxy rises");

    // Rows (Rain=0) and (Rain=1) are distributions over Wet that kept their mass (2).
    let w = engine.skill_weights("weather").unwrap();
    assert!((w[0] + w[1] - 2.0).abs() < 1e-4 && (w[2] + w[3] - 2.0).abs() < 1e-4);
    assert!((w[3] / 2.0 - 0.9).abs() < 0.06, "P(Wet | Rain) = {}", w[3] / 2.0);
    assert!((w[1] / 2.0 - 0.2).abs() < 0.06, "P(Wet | no rain) = {}", w[1] / 2.0);

    // The learned function now predicts: rain makes wet likely.
    engine.clear_evidence("Wet");
    engine.set_evidence("Rain", 1).unwrap();
    assert!(engine.infer(|_| None)[1][1] > 0.8);

    // Functions not marked learnable stay as signed.
    let mut fixed = SigmaEngine::new();
    fixed.compile_skill(signed_program(&id, "weather", weather_program(false))).unwrap();
    train(&mut fixed, 5);
    assert_eq!(fixed.skill_weights("weather").unwrap(), [1.0; 4]);
}

#[test]
fn gradient_learning_works_through_hidden_variables_and_v1_tables() {
    // v1 skills are learnable over the whole table; partial evidence spreads the update.
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(crate::tests::skill_sync(&id)).unwrap();
    let before = engine.skill_weights("sync-on-mounted-nas").unwrap();
    for _ in 0..50 {
        engine.set_evidence("ActionSyncFiles", 0).unwrap();
        engine.learn_gradient(0.05);
    }
    let after = engine.skill_weights("sync-on-mounted-nas").unwrap();
    let mass = |w: &[f32]| w.iter().sum::<f32>();
    assert!((mass(&after) - mass(&before)).abs() < 1e-3);
    assert!(after[0] + after[2] > before[0] + before[2], "mass moved to ActionSyncFiles = 0");
}
