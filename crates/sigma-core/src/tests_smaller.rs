//! Temperature schedules, attention inferred in a graph, continuous arguments between regions,
//! exact max-product envelopes and unsupervised category learning.

use super::*;
use crate::tests::signed;
use crate::tests_v2::signed_program;
use PredicateRole::*;

fn bin(name: &str, role: PredicateRole) -> PredicateSchema {
    PredicateSchema::binary(name, role)
}

#[test]
fn annealed_selection_cools_with_every_decision() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed(&id, "a", vec![bin("G", Goal), bin("A", Action)], vec![1.0, 1.0, 1.0, 9.0])).unwrap();
    engine.compile_skill(signed(&id, "b", vec![bin("G", Goal), bin("B", Action)], vec![1.0, 1.0, 1.0, 3.0])).unwrap();
    engine.register_predicate(&bin("Tick", Context)).unwrap();
    engine.selection = Selection::Annealed { start: 2.0, decay: 0.5, floor: 0.05 };
    assert_eq!(engine.temperature(), Some(2.0));
    let mut shares = Vec::new();
    for i in 0..8 {
        engine.goal_stack = vec!["G".into()];
        engine.set_evidence("Tick", i % 2).unwrap(); // working memory changes: no no-change impasse
        let out = engine.cognitive_cycle();
        assert!(out.selected_action.is_some());
        shares.push(out.operators.iter().find(|o| o.0 == "A").unwrap().1);
    }
    // Hot at first (nearly even), then nearly greedy; the floor holds.
    assert!(shares[0] < 0.6 && shares[7] > 0.95, "{shares:?}");
    assert!(shares.windows(2).all(|w| w[1] >= w[0] - 1e-6), "{shares:?}");
    assert_eq!(engine.temperature(), Some(0.05));
    let t = Selection::Annealed { start: 1.0, decay: 0.9, floor: 0.0 }.temperature_after(2).unwrap();
    assert!((t - 0.81).abs() < 1e-6);
    assert_eq!(Selection::Best.temperature_after(3), None);
}

#[test]
fn attention_is_raised_by_surprise_through_the_graph() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine
        .compile_skill(signed(&id, "act", vec![bin("G", Goal), bin("Act", Action)], vec![1.0, 1.0, 1.0, 9.0]))
        .unwrap();
    let quiet = signed(&id, "quiet", vec![bin("N1", Context), bin("N2", Context)], vec![1.0; 4]);
    let loud = signed(&id, "loud", vec![bin("N3", Context), bin("N4", Context)], vec![1.0; 4]);
    engine.compile_skill(quiet).unwrap();
    engine.compile_skill(loud).unwrap();
    engine.attention_budget = Some(2);
    engine.push_goal("G");
    // Nothing surprising: the relevant skill, then the first of two equal ones.
    assert_eq!(engine.cognitive_cycle().attended_skills, ["act", "quiet"]);
    // N3 is observed against expectation: the factor it is in now wins the second place.
    engine.set_evidence("N3", 1).unwrap();
    engine.push_goal("G");
    assert_eq!(engine.cognitive_cycle().attended_skills, ["act", "loud"]);
}

fn field(id: &NodeIdentity) -> SigmaEngine {
    let program = Program {
        types: vec![TypeDef::continuous("position", 0.0, 10.0, 5), TypeDef::continuous("celsius", 0.0, 40.0, 40)],
        predicates: vec![
            PredicateDef { name: "Heat".into(), role: Context, args: vec!["position".into()], value_type: None },
            PredicateDef {
                name: "Temp".into(),
                role: Context,
                args: vec!["position".into()],
                value_type: Some("celsius".into()),
            },
        ],
        conditionals: vec![Conditional {
            name: "rising".into(),
            patterns: vec![Pattern {
                kind: PatternKind::Condact,
                predicate: "Heat".into(),
                args: vec![Term::Var("x".into())],
            }],
            function_vars: vec!["x".into()],
            function: Function::PiecewiseLinear {
                default: 0.0,
                pieces: vec![
                    Piece { lo: vec![0.0, 0.0], hi: vec![10.0, 1.0], coef: vec![10.0, -1.0, 0.0] },
                    Piece { lo: vec![0.0, 1.0], hi: vec![10.0, 2.0], coef: vec![0.0, 1.0, 0.0] },
                ],
            },
            learnable: false,
            normal: None,
        }],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(id, "field", program)).unwrap();
    engine
}

#[test]
fn continuous_arguments_are_read_and_written_between_regions() {
    let id = NodeIdentity::generate();
    let mut engine = field(&id);
    // P(Heat(x)) = x / 10 at region midpoints; between them the reading is interpolated, so
    // a linear field is read exactly where region resolution would round.
    let at = |e: &SigmaEngine, x: f32| e.value_at("Heat", x).unwrap();
    assert!((at(&engine, 3.0) - 0.3).abs() < 1e-3);
    assert!((at(&engine, 3.1) - 0.31).abs() < 1e-3, "{}", at(&engine, 3.1));
    assert!((at(&engine, 6.4) - 0.64).abs() < 1e-3);
    assert!((at(&engine, 0.2) - 0.1).abs() < 1e-3, "clamped to the outer midpoint");
    assert!(engine.value_at("Nope", 1.0).is_none());

    // A reading between two midpoints is shared by both regions.
    engine.perceive_value_at("Temp", 4.0, 21.0, 0.5).unwrap();
    for atom in ["Temp(2)", "Temp(4)"] {
        assert!((engine.expected_value(atom).unwrap() - 21.0).abs() < 0.6, "{atom}");
    }
    assert!((engine.expected_value("Temp(8)").unwrap() - 20.0).abs() < 0.6, "untouched: uniform");
    assert!((engine.value_at("Temp", 4.0).unwrap() - 21.0).abs() < 0.6);
    // A percept at a midpoint goes to that region alone.
    engine.perceive_at("Heat", 9.0, &[0.0, 1.0]).unwrap();
    assert!(engine.posterior("Heat(8)").unwrap()[1] > 0.99);
    assert!((engine.posterior("Heat(6)").unwrap()[1] - 0.7).abs() < 1e-3);
}

#[test]
fn fields_over_several_arguments_interpolate_bilinearly() {
    let id = NodeIdentity::generate();
    // P(Heat(x, y)) = (x + y) / 20 over a 5 × 5 grid of regions.
    let program = Program {
        types: vec![TypeDef::continuous("position", 0.0, 10.0, 5)],
        predicates: vec![PredicateDef {
            name: "Heat".into(),
            role: Context,
            args: vec!["position".into(), "position".into()],
            value_type: None,
        }],
        conditionals: vec![Conditional {
            name: "rising".into(),
            patterns: vec![Pattern {
                kind: PatternKind::Condact,
                predicate: "Heat".into(),
                args: vec![Term::Var("x".into()), Term::Var("y".into())],
            }],
            function_vars: vec!["x".into(), "y".into()],
            function: Function::PiecewiseLinear {
                default: 0.0,
                pieces: vec![
                    Piece { lo: vec![0.0, 0.0, 0.0], hi: vec![10.0, 10.0, 1.0], coef: vec![20.0, -1.0, -1.0, 0.0] },
                    Piece { lo: vec![0.0, 0.0, 1.0], hi: vec![10.0, 10.0, 2.0], coef: vec![0.0, 1.0, 1.0, 0.0] },
                ],
            },
            learnable: false,
            normal: None,
        }],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&id, "plane", program)).unwrap();
    let at = |e: &SigmaEngine, x: f32, y: f32| e.value_at_point("Heat", &[x, y]).unwrap();
    // A linear field is read exactly between the region midpoints, in both arguments.
    for (x, y) in [(3.0, 5.0), (3.1, 6.4), (4.0, 4.0), (8.9, 1.2)] {
        assert!((at(&engine, x, y) - (x + y) / 20.0).abs() < 1e-3, "({x}, {y}): {}", at(&engine, x, y));
    }
    assert!((at(&engine, 0.2, 9.9) - (1.0 + 9.0) / 20.0).abs() < 1e-3, "clamped to the outer midpoints");
    assert!(engine.value_at_point("Heat", &[1.0]).is_none(), "wrong number of arguments");
    // A percept at a point goes to the four regions around it, tempered by their weights.
    engine.perceive_at_point("Heat", &[4.0, 6.0], &[0.0, 1.0]).unwrap();
    for atom in ["Heat(2,4)", "Heat(4,4)", "Heat(2,6)", "Heat(4,6)"] {
        let p = engine.posterior(atom).unwrap()[1];
        let prior = {
            let args: Vec<f32> = atom[5..atom.len() - 1].split(',').map(|a| a.parse::<f32>().unwrap() + 1.0).collect();
            (args[0] + args[1]) / 20.0
        };
        assert!(p > prior + 0.05, "{atom}: {p} vs {prior}");
    }
    assert!((engine.posterior("Heat(8,8)").unwrap()[1] - 0.9).abs() < 1e-3, "untouched");
}

#[test]
fn max_product_takes_the_exact_envelope_between_slices() {
    // f(x, y) = y for x < 0.5 and 1 - y for x >= 0.5; max over x is max(y, 1 - y).
    let f = Plm { bounds: vec![vec![0.0, 0.5, 1.0], vec![0.0, 1.0]], coef: vec![0.0, 0.0, 1.0, 1.0, 0.0, -1.0] };
    let m = Plm::constant(vec![vec![0.0, 1.0]], 1.0);
    let out = f.mul_max_out(0, &m);
    for (y, want) in [(0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.7), (0.95, 0.95)] {
        assert!((out.eval(&[y]) - want).abs() < 1e-9, "max at {y}: {}", out.eval(&[y]));
    }
}

#[test]
fn max_product_finds_peaks_inside_slices() {
    // f(x, y) = x + y times the message m(x) = 1 - x peaks inside the slice, at
    // x* = (1 - y) / 2, where it is ((1 + y) / 2)². The ends alone would give y.
    let f = Plm { bounds: vec![vec![0.0, 1.0], vec![0.0, 1.0]], coef: vec![0.0, 1.0, 1.0] };
    let m = Plm { bounds: vec![vec![0.0, 1.0]], coef: vec![1.0, -1.0] };
    let out = f.mul_max_out(0, &m);
    for i in 0..=20 {
        let y = i as f64 / 20.0 * 0.999;
        let want = ((1.0 + y) / 2.0).powi(2);
        let got = out.eval(&[y]);
        assert!((got - want).abs() <= 1e-5 * want.max(1e-3), "y {y}: {got} vs {want}");
    }
    // Maximizing out the other dimension works too: f(y, x) with x second.
    let g = Plm { bounds: vec![vec![0.0, 1.0], vec![0.0, 1.0]], coef: vec![0.0, 1.0, 1.0] };
    let out = g.mul_max_out(1, &m);
    assert!((out.eval(&[0.0]) - 0.25).abs() < 1e-5, "{}", out.eval(&[0.0]));
    // With a flat message the end-point envelope is still used (and exact).
    let flat = Plm::constant(vec![vec![0.0, 1.0]], 1.0);
    let out = f.mul_max_out(0, &flat);
    assert!((out.eval(&[0.3]) - 1.3).abs() < 1e-6);
}

#[test]
fn categories_are_discovered_without_labels() {
    let mut engine = SigmaEngine::new();
    for p in ["Fur", "Wings", "Barks"] {
        engine.register_predicate(&bin(p, Context)).unwrap();
    }
    engine.discover_categories = true;
    let dog = [("Fur", 1), ("Wings", 0), ("Barks", 1)];
    let bird = [("Fur", 0), ("Wings", 1), ("Barks", 0)];
    for i in 0..10 {
        for (p, x) in if i % 2 == 0 { dog } else { bird } {
            engine.set_evidence(p, x).unwrap();
        }
        engine.cognitive_cycle();
    }
    let concepts = engine.semantic.concepts();
    assert_eq!(concepts.len(), 2, "{concepts:?}");
    assert!(concepts.iter().all(|c| c.1 == 5.0));
    // The discovered concepts work as any learned ones: a partial cue is categorized and
    // completed through the graph.
    for p in ["Fur", "Wings", "Barks"] {
        engine.clear_evidence(p);
    }
    engine.set_evidence("Wings", 1).unwrap();
    engine.rebuild_memory_factors();
    assert!(engine.categorize()[0].1 > 0.8);
    assert!(engine.posterior("Fur").unwrap()[0] > 0.7);
    // The same first object names the same concept on another device.
    let mut other = SemanticMemory::default();
    let first: Vec<(String, usize)> = dog.iter().map(|(p, x)| (p.to_string(), *x)).collect();
    assert!(concepts.iter().any(|c| c.0 == other.discover(&first)));
}
