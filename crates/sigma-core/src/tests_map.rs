//! Max-product message passing and MAP estimates.

use super::*;
use crate::tests::signed;
use crate::tests_v2::signed_program;
use PredicateRole::*;

fn bin(name: &str, role: PredicateRole) -> PredicateSchema {
    PredicateSchema::binary(name, role)
}

fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3)
}

/// P(x, y): (0,0) = 0.4, (0,1) = 0, (1,0) = 0.3, (1,1) = 0.3. Each variable's own most likely
/// value gives (1, 0), with probability 0.3; the jointly most likely assignment is (0, 0).
fn joint() -> SigmaEngine {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine
        .compile_skill(signed(&id, "xy", vec![bin("X", Context), bin("Y", Context)], vec![4.0, 0.0, 3.0, 3.0]))
        .unwrap();
    engine
}

#[test]
fn max_product_finds_the_joint_map_where_marginals_do_not() {
    let mut engine = joint();
    // Sum-product: marginals.
    assert!(close(&engine.posterior("X").unwrap(), &[0.4, 0.6]));
    assert!(close(&engine.posterior("Y").unwrap(), &[0.7, 0.3]));
    // Max-product: max-marginals, 0.4 : 0.3 for both.
    engine.inference = Inference::MaxProduct;
    assert!(close(&engine.posterior("X").unwrap(), &[4.0 / 7.0, 3.0 / 7.0]));
    assert!(close(&engine.posterior("Y").unwrap(), &[4.0 / 7.0, 3.0 / 7.0]));
    // The MAP estimate, whatever mode is set.
    engine.inference = Inference::SumProduct;
    let map = engine.map_estimate();
    assert_eq!(map.assignment, [("X".to_string(), 0), ("Y".to_string(), 0)]);
    // Evidence conditions it: given Y = 1, X must be 1.
    engine.set_evidence("Y", 1).unwrap();
    assert_eq!(engine.map_estimate().assignment[0], ("X".to_string(), 1));
}

#[test]
fn map_decoding_breaks_ties_consistently() {
    // X = Y, with both agreements equally likely: every max-marginal ties.
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine
        .compile_skill(signed(&id, "eq", vec![bin("X", Context), bin("Y", Context)], vec![1.0, 0.0, 0.0, 1.0]))
        .unwrap();
    let map = engine.map_estimate();
    let (x, y) = (map.assignment[0].1, map.assignment[1].1);
    assert_eq!(x, y, "a consistent assignment, not one value from each tied mode");
}

#[test]
fn max_product_on_exact_continuous_messages() {
    // f(t, hot): hot = 0 is 30 everywhere on [0, 40); hot = 1 is 40 on [30, 40) only. Summed
    // over t, hot = 0 is likelier (1200 : 400); at its best t, hot = 1 is (40 : 30).
    let program = Program {
        types: vec![TypeDef::continuous("celsius", 0.0, 40.0, 4)],
        predicates: vec![
            PredicateDef { name: "Temp".into(), role: Context, args: vec![], value_type: Some("celsius".into()) },
            PredicateDef { name: "Hot".into(), role: Context, args: vec![], value_type: None },
        ],
        conditionals: vec![Conditional {
            name: "c".into(),
            patterns: vec![
                Pattern { kind: PatternKind::Condact, predicate: "Temp".into(), args: vec![] },
                Pattern { kind: PatternKind::Condact, predicate: "Hot".into(), args: vec![] },
            ],
            function_vars: vec![],
            function: Function::PiecewiseLinear {
                default: 0.0,
                pieces: vec![
                    Piece { lo: vec![0.0, 0.0], hi: vec![40.0, 1.0], coef: vec![30.0, 0.0, 0.0] },
                    Piece { lo: vec![30.0, 1.0], hi: vec![40.0, 2.0], coef: vec![40.0, 0.0, 0.0] },
                ],
            },
            learnable: false,
            normal: None,
        }],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&NodeIdentity::generate(), "hot", program)).unwrap();
    assert!(close(&engine.posterior("Hot").unwrap(), &[0.75, 0.25]));
    engine.inference = Inference::MaxProduct;
    assert!(close(&engine.posterior("Hot").unwrap(), &[3.0 / 7.0, 4.0 / 7.0]), "{:?}", engine.posterior("Hot"));
    let map = engine.map_estimate();
    assert_eq!(map.assignment, [("Temp".to_string(), 3), ("Hot".to_string(), 1)]);
}

#[test]
fn decisions_run_on_max_product() {
    // The backup chain still decides the same way under max-product.
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.inference = Inference::MaxProduct;
    engine.compile_skill(crate::tests::skill_backup(&id)).unwrap();
    engine.compile_skill(crate::tests::skill_sync(&id)).unwrap();
    engine.push_goal("GoalBackup");
    let out = engine.cognitive_cycle();
    assert_eq!(out.goal_stack, ["GoalBackup", "SubgoalMountNAS"]);
    assert_eq!(out.selected_action.as_deref(), Some("ActionMountNAS"));
}
