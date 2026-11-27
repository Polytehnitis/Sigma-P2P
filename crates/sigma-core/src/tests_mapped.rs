//! Affine maps on continuous values (`Function::Mapped`).

use super::*;
use crate::tests_v2::signed_program;

fn pred(name: &str, value_type: Option<&str>) -> PredicateDef {
    PredicateDef {
        name: name.into(),
        role: PredicateRole::Context,
        args: vec![],
        value_type: value_type.map(Into::into),
    }
}

fn pat(name: &str) -> Pattern {
    Pattern { kind: PatternKind::Condact, predicate: name.into(), args: vec![] }
}

fn program(types: Vec<TypeDef>, predicates: Vec<PredicateDef>, patterns: Vec<Pattern>, f: Function) -> Program {
    Program {
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
    }
}

fn compile(p: Program) -> Result<SigmaEngine, String> {
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&NodeIdentity::generate(), "s", p)).map_err(|e| e.to_string())?;
    Ok(engine)
}

/// "Comfortable is 18 to 24 °C", nine times likelier, read on a Fahrenheit thermometer.
fn comfort_in_fahrenheit() -> SigmaEngine {
    let celsius = Function::PiecewiseLinear {
        default: 1.0,
        pieces: vec![Piece { lo: vec![0.5, 18.0], hi: vec![1.5, 24.0], coef: vec![9.0, 0.0, 0.0] }],
    };
    compile(program(
        vec![TypeDef::continuous("fahrenheit", 32.0, 122.0, 5)],
        vec![pred("Comfy", None), pred("Temp", Some("fahrenheit"))],
        vec![pat("Comfy"), pat("Temp")],
        Function::mapped(celsius, 2, 1, 1.8, 32.0),
    ))
    .unwrap()
}

#[test]
fn pieces_move_to_the_mapped_interval() {
    let mut engine = comfort_in_fahrenheit();
    engine.set_evidence("Comfy", 1).unwrap();
    let d = engine.density("Temp").unwrap();
    let (lo, hi) = (18.0_f32 * 1.8 + 32.0, 24.0_f32 * 1.8 + 32.0);
    assert_eq!(d.bounds[0], vec![32.0, lo as f64, hi as f64, 122.0]);
    // P(64.4 ≤ Temp < 75.2 | Comfy) = 9·10.8 / (9·10.8 + 79.2).
    let width = (hi - lo) as f64;
    let want = 9.0 * width / (9.0 * width + (90.0 - width));
    let (m0, _) = d.moments(lo as f64, hi as f64);
    assert!((m0 - want).abs() < 1e-4, "{m0} vs {want}");
    // And back: 70 °F (21 °C) is comfortable, 80 °F (26.7 °C) is not.
    engine.clear_evidence("Comfy");
    engine.set_value("Temp", 70.0).unwrap();
    assert!((engine.posterior("Comfy").unwrap()[1] - 0.9).abs() < 1e-5);
    engine.set_value("Temp", 80.0).unwrap();
    assert!((engine.posterior("Comfy").unwrap()[1] - 0.5).abs() < 1e-5);
}

#[test]
fn a_mirror_flips_a_slope() {
    // f(u) = u on [0, 40), read through x = 40 - u: f(x) = 40 - x, so E[x] = 40/3.
    let rising = Function::PiecewiseLinear {
        default: 0.0,
        pieces: vec![Piece { lo: vec![0.0], hi: vec![40.0], coef: vec![0.0, 1.0] }],
    };
    let engine = compile(program(
        vec![TypeDef::continuous("celsius", 0.0, 40.0, 2)],
        vec![pred("Temp", Some("celsius"))],
        vec![pat("Temp")],
        Function::mapped(rising, 1, 0, -1.0, 40.0),
    ))
    .unwrap();
    let e = engine.expected_value("Temp").unwrap();
    assert!((e - 40.0 / 3.0).abs() < 1e-3, "E[Temp] = {e}");
}

/// A sensor that reads Celsius with 1 °C noise, reporting in Fahrenheit.
fn fahrenheit_sensor() -> SigmaEngine {
    // In Celsius: Reading ~ N(Temp, 1).
    let noise = Function::Gaussian { target: 1, mean: vec![0.0, 1.0, 0.0], std: 1.0 };
    compile(program(
        vec![TypeDef::continuous("celsius", 0.0, 40.0, 4), TypeDef::continuous("fahrenheit", 32.0, 104.0, 4)],
        vec![pred("Temp", Some("celsius")), pred("Reading", Some("fahrenheit"))],
        vec![pat("Temp"), pat("Reading")],
        Function::mapped(noise, 2, 1, 1.8, 32.0),
    ))
    .unwrap()
}

#[test]
fn gaussians_become_the_density_of_the_mapped_value() {
    let resolved =
        Function::mapped(Function::Gaussian { target: 1, mean: vec![0.0, 1.0, 0.0], std: 1.0 }, 2, 1, 1.8, 32.0)
            .resolved()
            .unwrap();
    assert_eq!(resolved, Function::Gaussian { target: 1, mean: vec![32.0, 1.8, 0.0], std: 1.8 });

    let mut engine = fahrenheit_sensor();
    engine.set_value("Reading", 77.0).unwrap();
    let t = engine.expected_value("Temp").unwrap();
    assert!((t - 25.0).abs() < 0.1, "77 °F reads as 25 °C: {t}");
    engine.clear_evidence("Reading");
    engine.set_value("Temp", 10.0).unwrap();
    let r = engine.expected_value("Reading").unwrap();
    assert!((r - 50.0).abs() < 0.2, "10 °C reads as 50 °F: {r}");
}

#[test]
fn particles_and_the_grid_see_the_same_map() {
    let mut engine = fahrenheit_sensor();
    engine.particle_count = Some(400);
    engine.set_value("Reading", 77.0).unwrap();
    let t = engine.expected_value("Temp").unwrap();
    assert!((t - 25.0).abs() < 1.0, "particles: {t}");

    let mut engine = comfort_in_fahrenheit();
    engine.tabulate_continuous = true;
    engine.set_value("Temp", 70.0).unwrap();
    // The 5-region grid puts 70 °F in [68, 86), whose midpoint 77 °F is 25 °C: not comfortable.
    assert!((engine.posterior("Comfy").unwrap()[1] - 0.5).abs() < 1e-5);
}

#[test]
fn bad_maps_are_rejected() {
    let piece = Function::PiecewiseLinear {
        default: 1.0,
        pieces: vec![Piece { lo: vec![0.0], hi: vec![10.0], coef: vec![2.0, 0.0] }],
    };
    let one = |f: Function| {
        compile(program(
            vec![TypeDef::continuous("celsius", 0.0, 40.0, 2)],
            vec![pred("Temp", Some("celsius"))],
            vec![pat("Temp")],
            f,
        ))
        .err()
        .unwrap_or_default()
    };
    assert!(one(Function::mapped(piece.clone(), 1, 0, 0.0, 1.0)).contains("invalid value map"));
    assert!(one(Function::mapped(piece.clone(), 1, 0, f32::NAN, 1.0)).contains("invalid value map"));
    assert!(one(Function::mapped(Function::Table(vec![1.0, 2.0]), 1, 0, 2.0, 0.0)).contains("must be analytic"));
    let nested = Function::mapped(Function::mapped(piece.clone(), 1, 0, 2.0, 0.0), 1, 0, 2.0, 0.0);
    assert!(one(nested).contains("nested value maps"));
    assert!(one(Function::mapped(piece.clone(), 2, 0, 2.0, 0.0)).contains("do not match"));
    // A map that leaves every value alone is accepted.
    assert!(one(Function::mapped(piece, 1, 0, 1.0, 0.0)).is_empty());
}

#[test]
fn mapped_functions_survive_the_wire() {
    let f = Function::mapped(Function::Gaussian { target: 0, mean: vec![1.0, 0.0], std: 2.0 }, 1, 0, -3.0, 4.0);
    let bytes = postcard::to_allocvec(&f).unwrap();
    assert_eq!(postcard::from_bytes::<Function>(&bytes).unwrap(), f);
    // Existing variants keep their encoding (Mapped is appended last).
    let t = postcard::to_allocvec(&Function::Table(vec![1.0])).unwrap();
    assert_eq!(t[0], 0);
}
