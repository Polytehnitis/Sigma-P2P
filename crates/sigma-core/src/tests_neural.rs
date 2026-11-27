//! PySigma's neural side: autograd, neural functions trained by backpropagation, learned
//! Gaussian parameters, and variational and expectation-propagation inference.

use super::*;
use crate::neural::{NeuralNet, Tape};
use crate::tests_v2::signed_program;

fn pred(name: &str, role: PredicateRole, value_type: Option<&str>) -> PredicateDef {
    PredicateDef { name: name.into(), role, args: vec![], value_type: value_type.map(Into::into) }
}

fn pat(kind: PatternKind, name: &str) -> Pattern {
    Pattern { kind, predicate: name.into(), args: vec![] }
}

fn learnable(patterns: Vec<Pattern>, function: Function, normal: Option<usize>) -> Conditional {
    Conditional { name: "c".into(), patterns, function_vars: vec![], function, learnable: true, normal }
}

#[test]
fn the_tape_differentiates_compositions() {
    // f(x, y) = tanh(x·y) + exp(x) - ln(y)
    let (x0, y0) = (0.7, 1.3);
    let mut t = Tape::new();
    let (x, y) = (t.leaf(x0), t.leaf(y0));
    let xy = t.mul(x, y);
    let a = t.tanh(xy);
    let b = t.exp(x);
    let c = t.ln(y);
    let ab = t.add(a, b);
    let f = t.sub(ab, c);
    let g = t.gradient(f);
    let sech2 = 1.0 - (x0 * y0).tanh().powi(2);
    assert!((g[0] - (sech2 * y0 + x0.exp())).abs() < 1e-9);
    assert!((g[1] - (sech2 * x0 - 1.0 / y0)).abs() < 1e-9);
    // log-sum-exp's gradient is the softmax.
    let mut t = Tape::new();
    let xs: Vec<_> = [1.0, 2.0, 3.0].iter().map(|&v| t.leaf(v)).collect();
    let l = t.log_sum_exp(&xs);
    let g = t.gradient(l);
    let z: f64 = [1.0f64, 2.0, 3.0].iter().map(|v| v.exp()).sum();
    for (i, v) in [1.0f64, 2.0, 3.0].iter().enumerate() {
        assert!((g[i] - v.exp() / z).abs() < 1e-9);
    }
}

/// Weather (sun, rain, snow) -> Activity (walk, read, ski), as a neural energy.
fn activities(id: &NodeIdentity) -> SigmaEngine {
    let program = Program {
        types: vec![
            TypeDef::symbolic("weather", &["sun", "rain", "snow"]),
            TypeDef::symbolic("activity", &["walk", "read", "ski"]),
        ],
        predicates: vec![
            pred("Weather", PredicateRole::Context, Some("weather")),
            pred("Activity", PredicateRole::Context, Some("activity")),
        ],
        conditionals: vec![learnable(
            vec![pat(PatternKind::Condition, "Weather"), pat(PatternKind::Condact, "Activity")],
            Function::Neural(NeuralNet::new(&[2, 8, 1], 7)),
            Some(1),
        )],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(id, "activities", program)).unwrap();
    engine
}

#[test]
fn neural_functions_learn_by_backpropagation() {
    let id = NodeIdentity::generate();
    let mut engine = activities(&id);
    // Untrained, the net says little.
    engine.set_evidence("Weather", 1).unwrap();
    let before = engine.posterior("Activity").unwrap();
    assert!(before.iter().all(|&p| p < 0.6), "{before:?}");
    for _ in 0..300 {
        for w in 0..3 {
            engine.set_evidence("Weather", w).unwrap();
            engine.set_evidence("Activity", w).unwrap();
            engine.learn_gradient(0.2);
        }
    }
    engine.clear_evidence("Activity");
    for w in 0..3 {
        engine.set_evidence("Weather", w).unwrap();
        let p = engine.posterior("Activity").unwrap();
        assert!(p[w] > 0.8, "weather {w}: {p:?}");
    }
    // The learned net is kept, and a fresh compile of the skill picks it up again.
    let learned = engine.models.functions.values().next().cloned().expect("learned");
    assert!(matches!(learned, Function::Neural(_)));
    let mut again = activities(&id);
    again.models = engine.models.clone();
    again.rebuild_memory_factors();
    again.set_evidence("Weather", 2).unwrap();
    assert!(again.posterior("Activity").unwrap()[2] > 0.8);
    // Malformed nets are refused when a skill is checked.
    let mut bad = NeuralNet::new(&[3, 4, 1], 1);
    assert!(bad.check(2).is_err(), "wrong input size");
    bad.layers[0].weights[0] = f32::NAN;
    assert!(bad.check(3).is_err());
}

/// Setting (low, high) -> Temp in [0, 40): a Gaussian whose mean depends on the setting.
fn heating(id: &NodeIdentity) -> SigmaEngine {
    let program = Program {
        types: vec![TypeDef::symbolic("setting", &["low", "high"]), TypeDef::continuous("celsius", 0.0, 40.0, 40)],
        predicates: vec![
            pred("Setting", PredicateRole::Context, Some("setting")),
            pred("Temp", PredicateRole::Context, Some("celsius")),
        ],
        conditionals: vec![learnable(
            vec![pat(PatternKind::Condition, "Setting"), pat(PatternKind::Condact, "Temp")],
            Function::Gaussian { target: 1, mean: vec![20.0, 0.0, 0.0], std: 8.0 },
            Some(1),
        )],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(id, "heating", program)).unwrap();
    engine
}

#[test]
fn gaussian_parameters_are_learned() {
    let id = NodeIdentity::generate();
    let mut engine = heating(&id);
    // Low gives about 15 °C, high about 25 °C.
    let readings = [(0, 14.2), (1, 25.6), (0, 15.9), (1, 24.1), (0, 15.3), (1, 25.2), (0, 14.6), (1, 24.9)];
    for _ in 0..60 {
        for &(s, t) in &readings {
            engine.set_evidence("Setting", s).unwrap();
            engine.set_value("Temp", t).unwrap();
            engine.learn_gradient(0.3);
        }
    }
    let Some(Function::Gaussian { mean, std, .. }) = engine.models.functions.values().next().cloned() else {
        panic!("a learned gaussian")
    };
    assert!((mean[0] - 15.0).abs() < 1.0 && (mean[1] - 10.0).abs() < 1.0, "{mean:?}");
    assert!(std < 2.5, "{std}");
    // The learned form is what inference uses.
    engine.clear_evidence("Temp");
    engine.set_evidence("Setting", 1).unwrap();
    let t = engine.expected_value("Temp").unwrap();
    assert!((t - 25.0).abs() < 1.0, "{t}");
    engine.set_evidence("Setting", 0).unwrap();
    assert!((engine.expected_value("Temp").unwrap() - 15.0).abs() < 1.0);
}

fn bin(name: &str) -> PredicateSchema {
    PredicateSchema::binary(name, PredicateRole::Context)
}

#[test]
fn variational_message_passing_is_mean_field() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    // A prior on A, and an attractive coupling of A and B.
    engine.compile_skill(crate::tests::signed(&id, "prior", vec![bin("A")], vec![3.0, 7.0])).unwrap();
    engine
        .compile_skill(crate::tests::signed(&id, "couple", vec![bin("A"), bin("B")], vec![4.0, 1.0, 1.0, 4.0]))
        .unwrap();
    let bp = engine.posterior("B").unwrap()[1];
    engine.inference = Inference::VariationalMessagePassing;
    let vmp = engine.posterior("B").unwrap()[1];
    // On a tree BP is exact (0.62); mean field is overconfident with attractive coupling.
    assert!((bp - 0.62).abs() < 1e-3, "{bp}");
    assert!(vmp > bp + 0.05, "{vmp} vs {bp}");
    // With A observed, there is nothing left to approximate: both agree.
    engine.set_evidence("A", 1).unwrap();
    let vmp = engine.posterior("B").unwrap()[1];
    engine.inference = Inference::SumProduct;
    assert!((engine.posterior("B").unwrap()[1] - vmp).abs() < 1e-4);
}

#[test]
fn expectation_propagation_moment_matches_continuous_messages() {
    let id = NodeIdentity::generate();
    // X in [0, 10): a broad bump around 5, and a bimodal likelihood with peaks at 3 and 7.
    let program = Program {
        types: vec![TypeDef::continuous("x", 0.0, 10.0, 40)],
        predicates: vec![pred("X", PredicateRole::Context, Some("x"))],
        conditionals: vec![
            learnable(
                vec![pat(PatternKind::Condact, "X")],
                Function::Gaussian { target: 0, mean: vec![5.0, 0.0], std: 3.0 },
                None,
            ),
            learnable(
                vec![pat(PatternKind::Condact, "X")],
                Function::PiecewiseLinear {
                    default: 0.05,
                    pieces: vec![
                        Piece { lo: vec![2.0], hi: vec![4.0], coef: vec![1.0, 0.0] },
                        Piece { lo: vec![6.5], hi: vec![7.5], coef: vec![2.0, 0.0] },
                    ],
                },
                None,
            ),
        ],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&id, "bimodal", program)).unwrap();
    engine.tabulate_continuous = true;
    let bp = engine.posterior("X").unwrap();
    engine.inference = Inference::ExpectationPropagation;
    let ep = engine.posterior("X").unwrap();
    let mean = |p: &[f32]| p.iter().enumerate().map(|(i, q)| q * (i as f32 + 0.5) * 0.25).sum::<f32>();
    let peaks = |p: &[f32]| (1..p.len() - 1).filter(|&i| p[i] > 0.01 && p[i] > p[i - 1] && p[i] >= p[i + 1]).count();
    // The exact (grid) posterior is bimodal; EP's is the single Gaussian with its moments.
    assert_eq!(peaks(&bp), 2, "{bp:?}");
    assert_eq!(peaks(&ep), 1, "{ep:?}");
    assert!((mean(&ep) - mean(&bp)).abs() < 0.3, "{} vs {}", mean(&ep), mean(&bp));
}

#[test]
fn learned_functions_sync_to_paired_devices() {
    let open = || {
        let path = std::env::temp_dir().join(format!("sigma-fn-sync-{}.redb", rand::random::<u64>()));
        let store = FactorStore::open(path).unwrap();
        let me = store.load_or_create_identity().unwrap().author_pubkey();
        (store, me)
    };
    let (phone, pc) = (open(), open());
    let id = NodeIdentity::generate();
    // The phone trains the net; the PC has the same skill, untrained.
    let mut trained = activities(&id);
    for _ in 0..300 {
        for w in 0..3 {
            trained.set_evidence("Weather", w).unwrap();
            trained.set_evidence("Activity", w).unwrap();
            trained.learn_gradient(0.2);
        }
    }
    trained.clear_evidence("Activity");
    phone.0.save_memories(&trained).unwrap();
    let state = phone.0.export_sync_state().unwrap();
    let key =
        state.registers.keys().find(|k| k.starts_with(sync::keys::FUNCTION)).cloned().expect("a function register");
    assert_eq!(sync::keys::function_slot(&key[sync::keys::FUNCTION.len()..]).unwrap().0, "activities");

    let mut untrained = activities(&id);
    untrained.set_evidence("Weather", 2).unwrap();
    assert!(untrained.posterior("Activity").unwrap()[2] < 0.6);
    let report = pc.0.merge_sync_state(phone.1, state).unwrap();
    assert!(report.changed.contains(&key));
    pc.0.refresh_engine(&mut untrained, &report).unwrap();
    assert!(untrained.posterior("Activity").unwrap()[2] > 0.8, "the PC uses the phone's net");
    assert_eq!(untrained.models.functions, trained.models.functions);
    // It is stored on the PC too, so a restart keeps it.
    let mut restarted = activities(&id);
    pc.0.restore_engine(&mut restarted).unwrap();
    restarted.set_evidence("Weather", 0).unwrap();
    assert!(restarted.posterior("Activity").unwrap()[0] > 0.8);

    // Malformed or mismatched functions are refused.
    let mut bad = NeuralNet::new(&[2, 4, 1], 3);
    bad.layers[1].bias[0] = f32::INFINITY;
    assert!(!crate::parametric::function_is_valid(&Function::Neural(bad.clone())));
    assert!(!untrained.adopt_learned_function("activities", 0, Function::Neural(bad)));
    let gaussian = Function::Gaussian { target: 0, mean: vec![0.0, 1.0], std: 0.0 };
    assert!(!crate::parametric::function_is_valid(&gaussian));
    // A well-formed net of the wrong shape is kept but not applied to the factor.
    let before = untrained.posterior("Activity").unwrap();
    assert!(untrained.adopt_learned_function("activities", 999, Function::Neural(NeuralNet::new(&[5, 1], 1))));
    assert_eq!(untrained.posterior("Activity").unwrap(), before);
    assert_eq!(sync::keys::function_slot("a#b"), None);
    assert_eq!(sync::keys::function_slot("#3"), None);
    assert_eq!(sync::keys::function_slot("x#y#4"), Some(("x#y".into(), 4)));
}

/// X → Y → Z, each a linear Gaussian over [0, 20): X ~ N(6, 2²), Y ~ N(X + 2, 1.5²),
/// Z ~ N(Y + 1, 1.5²).
fn chain(id: &NodeIdentity) -> SigmaEngine {
    let program = Program {
        types: vec![TypeDef::continuous("v", 0.0, 20.0, 40)],
        predicates: vec![
            pred("X", PredicateRole::Context, Some("v")),
            pred("Y", PredicateRole::Context, Some("v")),
            pred("Z", PredicateRole::Context, Some("v")),
        ],
        conditionals: vec![
            learnable(
                vec![pat(PatternKind::Condact, "X")],
                Function::Gaussian { target: 0, mean: vec![6.0, 0.0], std: 2.0 },
                None,
            ),
            learnable(
                vec![pat(PatternKind::Condact, "X"), pat(PatternKind::Condact, "Y")],
                Function::Gaussian { target: 1, mean: vec![2.0, 1.0, 0.0], std: 1.5 },
                None,
            ),
            learnable(
                vec![pat(PatternKind::Condact, "Y"), pat(PatternKind::Condact, "Z")],
                Function::Gaussian { target: 1, mean: vec![1.0, 1.0, 0.0], std: 1.5 },
                None,
            ),
        ],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(id, "chain", program)).unwrap();
    engine
}

#[test]
fn variational_message_passing_runs_on_exact_functions() {
    let id = NodeIdentity::generate();
    let mut engine = chain(&id);
    engine.set_value("Z", 14.0).unwrap();
    // Exact sum-product on piecewise-linear functions.
    let (bp_mean, bp_var) = engine.density("X").unwrap().mean_variance();
    engine.inference = Inference::VariationalMessagePassing;
    let vmp = engine.density("X").expect("VMP runs on the exact path");
    let (vmp_mean, vmp_var) = vmp.mean_variance();
    // The linear-Gaussian posterior mean: X | Z ~ N(6 + 4/8.5·(14 - 9), …) ≈ 8.35.
    assert!((bp_mean - 8.35).abs() < 0.3, "{bp_mean}");
    // Mean field finds the mean and underestimates the spread (Y is integrated out exactly by
    // BP but held as a separate factor by VMP).
    assert!((vmp_mean - bp_mean).abs() < 0.4, "{vmp_mean} vs {bp_mean}");
    assert!(vmp_var < bp_var * 0.95, "{vmp_var} vs {bp_var}");
    // Not the grid: its density has more pieces than the 40 regions' steps would make flat.
    assert!(vmp.bounds[0].len() > 41);
    let e = engine.expected_value("X").unwrap();
    assert!((e as f64 - vmp_mean).abs() < 0.05);
}

#[test]
fn expectation_propagation_runs_on_exact_functions() {
    let id = NodeIdentity::generate();
    let program = Program {
        types: vec![TypeDef::continuous("x", 0.0, 10.0, 40)],
        predicates: vec![pred("X", PredicateRole::Context, Some("x"))],
        conditionals: vec![
            learnable(
                vec![pat(PatternKind::Condact, "X")],
                Function::Gaussian { target: 0, mean: vec![5.0, 0.0], std: 3.0 },
                None,
            ),
            learnable(
                vec![pat(PatternKind::Condact, "X")],
                Function::PiecewiseLinear {
                    default: 0.05,
                    pieces: vec![
                        Piece { lo: vec![2.0], hi: vec![4.0], coef: vec![1.0, 0.0] },
                        Piece { lo: vec![6.5], hi: vec![7.5], coef: vec![2.0, 0.0] },
                    ],
                },
                None,
            ),
        ],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&id, "bimodal", program)).unwrap();
    let exact = engine.density("X").unwrap();
    engine.inference = Inference::ExpectationPropagation;
    let ep = engine.density("X").expect("EP runs on the exact path");
    let peaks = |p: &Plm| {
        let xs: Vec<f64> = (0..200).map(|i| 0.025 + i as f64 * 0.05).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| p.eval(&[x])).collect();
        let top = ys.iter().copied().fold(0.0, f64::max);
        // Plateaus count once: a peak is where the density stops rising.
        (1..ys.len() - 1).filter(|&i| ys[i] > 0.2 * top && ys[i] > ys[i - 1] + 1e-9 && ys[i] >= ys[i + 1]).count()
    };
    assert_eq!(peaks(&exact), 2);
    assert_eq!(peaks(&ep), 1);
    // The moments of the exact posterior are kept (EP's fixed point on one factor).
    let ((me, ve), (mp, vp)) = (exact.mean_variance(), ep.mean_variance());
    assert!((me - mp).abs() < 0.25, "{me} vs {mp}");
    assert!((ve - vp).abs() < 0.35 * ve, "{ve} vs {vp}");
}
