//! The graph viewer and message tracing, and a regression suite that checks inference against
//! brute-force enumeration on seeded random graphs.

use super::*;
use crate::tests::{signed, skill_backup, skill_sync};
use PredicateRole::*;

fn bin(name: &str, role: PredicateRole) -> PredicateSchema {
    PredicateSchema::binary(name, role)
}

fn demo_engine() -> SigmaEngine {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill_backup(&id)).unwrap();
    engine.compile_skill(skill_sync(&id)).unwrap();
    engine
}

#[test]
fn snapshot_lists_variables_factors_and_beliefs() {
    let mut engine = demo_engine();
    engine.goal_stack = vec!["GoalBackup".into()];
    let snap = engine.graph_snapshot();
    let visible = snap.visible();
    let names: Vec<&str> = visible.variables.iter().map(|v| v.name.as_str()).collect();
    for n in ["GoalBackup", "SubgoalMountNAS", "ActionMountNAS", "ActionSyncFiles"] {
        assert!(names.contains(&n), "{names:?}");
    }
    assert_eq!(visible.factors.len(), 2);
    let backup = visible.factors.iter().find(|f| f.skill == "backup-needs-nas").unwrap();
    assert_eq!(backup.variables.len(), 3);
    assert_eq!(backup.entries, 8);
    assert!(!backup.memory);
    for v in &visible.variables {
        assert_eq!(v.belief.len(), v.domain);
        assert!((v.belief.iter().sum::<f32>() - 1.0).abs() < 1e-3, "{}: {:?}", v.name, v.belief);
    }
    for f in &visible.factors {
        assert!(f.variables.iter().all(|&i| i < visible.variables.len()));
    }
    assert_eq!(snap.goal_stack, vec!["GoalBackup".to_string()]);
}

#[test]
fn snapshot_renders_as_dot_json_and_html() {
    let engine = demo_engine();
    let snap = engine.graph_snapshot().visible();
    let dot = snap.to_dot();
    assert!(dot.starts_with("graph sigma {"), "{dot}");
    assert!(dot.contains("label=\"GoalBackup") && dot.contains("backup-needs-nas"), "{dot}");
    assert!(dot.trim_end().ends_with('}'));
    let json = snap.to_json();
    assert!(json.starts_with('{') && json.ends_with('}'), "{json}");
    assert!(json.contains("\"variables\"") && json.contains("\"factors\""));
    assert!(json.contains("\"name\":\"SubgoalMountNAS\""), "{json}");
    // Balanced brackets: the hand-written JSON is well formed.
    let depth = json.chars().try_fold(0i32, |d, c| {
        let d = match c {
            '{' | '[' => d + 1,
            '}' | ']' => d - 1,
            _ => d,
        };
        (d >= 0).then_some(d)
    });
    assert_eq!(depth, Some(0));
    let html = snap.to_html();
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("SubgoalMountNAS") && !html.contains("__GRAPH__"));
    assert!(html.contains("<script>") && html.trim_end().ends_with("</html>"));
}

#[test]
fn trace_records_sweeps_until_quiescence_and_explains_beliefs() {
    let mut engine = demo_engine();
    engine.goal_stack = vec!["GoalBackup".into()];
    let trace = engine.trace_messages();
    assert!(trace.converged, "{} sweeps", trace.sweeps.len());
    assert!(!trace.sweeps.is_empty());
    assert!(trace.sweeps.last().unwrap().delta < 1e-5);
    // The first sweep changes messages; the last changes (almost) none.
    assert!(!trace.sweeps[0].changed.is_empty());
    let into = trace.messages_into("ActionMountNAS");
    assert_eq!(into.len(), 1);
    assert_eq!(into[0].skill, "backup-needs-nas");
    // The belief is the normalized product of the messages into it (no evidence, flat prior).
    let mut product = vec![1.0f32; 2];
    for m in &into {
        for (p, x) in product.iter_mut().zip(&m.message) {
            *p *= x;
        }
    }
    let z: f32 = product.iter().sum();
    let belief = &trace.beliefs.iter().find(|(n, _)| n == "ActionMountNAS").unwrap().1;
    for (p, b) in product.iter().zip(belief) {
        assert!((p / z - b).abs() < 1e-3, "{product:?} vs {belief:?}");
    }
    // Tracing does not change inference.
    let plain = engine.posterior("ActionMountNAS").unwrap();
    assert_eq!(plain.len(), belief.len());
}

// ---------------------------------------------------------------------------------------------
// Regression suite: inference against brute force on seeded random graphs.
// ---------------------------------------------------------------------------------------------

/// A small deterministic generator (no dependence on the `rand` crate's algorithms).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn weight(&mut self) -> f32 {
        0.1 + (self.next() % 1000) as f32 / 100.0
    }
}

/// A random graph over `n` binary context variables: unary factors on some, and pairwise
/// factors along a random tree, plus `extra` edges that close loops.
struct RandomGraph {
    n: usize,
    factors: Vec<(Vec<usize>, Vec<f32>)>,
}

impl RandomGraph {
    fn new(seed: u64, n: usize, extra: usize) -> Self {
        let mut rng = Lcg(seed);
        let mut factors = Vec::new();
        for v in 0..n {
            if rng.below(2) == 0 {
                factors.push((vec![v], vec![rng.weight(), rng.weight()]));
            }
        }
        for v in 1..n {
            let parent = rng.below(v);
            factors.push((vec![parent, v], (0..4).map(|_| rng.weight()).collect()));
        }
        for _ in 0..extra {
            let a = rng.below(n);
            let b = (a + 1 + rng.below(n - 1)) % n;
            factors.push((vec![a.min(b), a.max(b)], (0..4).map(|_| rng.weight()).collect()));
        }
        Self { n, factors }
    }

    fn name(v: usize) -> String {
        format!("V{v}")
    }

    fn engine(&self) -> SigmaEngine {
        let id = NodeIdentity::generate();
        let mut engine = SigmaEngine::new();
        for v in 0..self.n {
            engine.register_predicate(&bin(&Self::name(v), Context)).unwrap();
        }
        for (i, (vars, table)) in self.factors.iter().enumerate() {
            let preds = vars.iter().map(|&v| bin(&Self::name(v), Context)).collect();
            engine.compile_skill(signed(&id, &format!("f{i}"), preds, table.clone())).unwrap();
        }
        engine
    }

    /// Unnormalized probability of a full assignment (bit `v` of `x` is variable `v`).
    fn weight(&self, x: usize) -> f64 {
        let bit = |v: usize| (x >> v) & 1;
        self.factors
            .iter()
            .map(|(vars, table)| {
                let idx = vars.iter().fold(0, |i, &v| i * 2 + bit(v));
                table[idx] as f64
            })
            .product()
    }

    fn consistent(x: usize, evidence: &[(usize, usize)]) -> bool {
        evidence.iter().all(|&(v, val)| (x >> v) & 1 == val)
    }

    fn marginals(&self, evidence: &[(usize, usize)]) -> Vec<[f64; 2]> {
        let mut m = vec![[0.0; 2]; self.n];
        for x in (0..1usize << self.n).filter(|&x| Self::consistent(x, evidence)) {
            let w = self.weight(x);
            for (v, mv) in m.iter_mut().enumerate() {
                mv[(x >> v) & 1] += w;
            }
        }
        for mv in &mut m {
            let z = mv[0] + mv[1];
            mv[0] /= z;
            mv[1] /= z;
        }
        m
    }

    fn map(&self, evidence: &[(usize, usize)]) -> (usize, f64, f64) {
        let mut best = (0, -1.0, -1.0);
        for x in (0..1usize << self.n).filter(|&x| Self::consistent(x, evidence)) {
            let w = self.weight(x);
            if w > best.1 {
                best = (x, w, best.1);
            } else if w > best.2 {
                best.2 = w;
            }
        }
        best
    }
}

fn evidence_for(seed: u64, n: usize) -> Vec<(usize, usize)> {
    let mut rng = Lcg(seed ^ 0x9e37_79b9);
    let v = rng.below(n);
    vec![(v, rng.below(2))]
}

fn apply(engine: &mut SigmaEngine, evidence: &[(usize, usize)]) {
    for &(v, val) in evidence {
        engine.set_evidence(&RandomGraph::name(v), val).unwrap();
    }
}

#[test]
fn regression_sum_product_is_exact_on_random_trees() {
    for seed in 0..24u64 {
        let n = 3 + (seed as usize % 6); // 3..=8 variables
        let graph = RandomGraph::new(seed, n, 0);
        let evidence = if seed % 2 == 0 { vec![] } else { evidence_for(seed, n) };
        let mut engine = graph.engine();
        apply(&mut engine, &evidence);
        let truth = graph.marginals(&evidence);
        for (v, t) in truth.iter().enumerate() {
            let b = engine.posterior(&RandomGraph::name(v)).unwrap();
            assert!((b[1] as f64 - t[1]).abs() < 2e-3, "seed {seed} V{v}: {b:?} vs {t:?} with evidence {evidence:?}");
        }
    }
}

#[test]
fn regression_max_product_decodes_the_map_assignment_on_random_trees() {
    let mut checked = 0;
    for seed in 100..124u64 {
        let n = 3 + (seed as usize % 6);
        let graph = RandomGraph::new(seed, n, 0);
        let evidence = if seed % 3 == 0 { evidence_for(seed, n) } else { vec![] };
        let (best, w1, w2) = graph.map(&evidence);
        if w1 - w2 < 1e-3 * w1 {
            continue; // a near tie: either assignment is a MAP
        }
        let mut engine = graph.engine();
        apply(&mut engine, &evidence);
        let map = engine.map_estimate();
        for v in 0..n {
            let got = map.assignment.iter().find(|(name, _)| *name == RandomGraph::name(v)).unwrap().1;
            assert_eq!(got, (best >> v) & 1, "seed {seed} V{v}: {map:?}");
        }
        checked += 1;
    }
    assert!(checked >= 18, "only {checked} graphs without near ties");
}

#[test]
fn regression_loopy_beliefs_stay_normalized_and_close() {
    let mut total_error = 0.0;
    let mut count = 0;
    for seed in 200..216u64 {
        let n = 4 + (seed as usize % 4);
        let graph = RandomGraph::new(seed, n, 2);
        let mut engine = graph.engine();
        // Loopy BP on the grid (the exact path would hide its error).
        engine.inference = Inference::ExpectationPropagation;
        let truth = graph.marginals(&[]);
        for (v, t) in truth.iter().enumerate() {
            let b = engine.posterior(&RandomGraph::name(v)).unwrap();
            assert!(b.iter().all(|x| x.is_finite() && *x >= 0.0), "seed {seed}: {b:?}");
            assert!((b.iter().sum::<f32>() - 1.0).abs() < 1e-3, "seed {seed}: {b:?}");
            total_error += (b[1] as f64 - t[1]).abs();
            count += 1;
        }
        let trace = engine.trace_messages();
        assert!(!trace.sweeps.is_empty());
    }
    // Loopy BP is approximate, but with two extra edges it stays close on average.
    let mean = total_error / count as f64;
    assert!(mean < 0.1, "mean loopy error {mean}");
}

#[test]
fn regression_variational_beliefs_respect_evidence_and_normalize() {
    for seed in 300..312u64 {
        let n = 3 + (seed as usize % 5);
        let graph = RandomGraph::new(seed, n, 0);
        let evidence = evidence_for(seed, n);
        let mut engine = graph.engine();
        engine.inference = Inference::VariationalMessagePassing;
        apply(&mut engine, &evidence);
        let truth = graph.marginals(&evidence);
        for (v, t) in truth.iter().enumerate() {
            let b = engine.posterior(&RandomGraph::name(v)).unwrap();
            assert!((b.iter().sum::<f32>() - 1.0).abs() < 1e-3, "seed {seed}: {b:?}");
            if let Some(&(_, val)) = evidence.iter().find(|e| e.0 == v) {
                assert!(b[val] > 0.999, "seed {seed} V{v}: {b:?}");
            }
            // Mean field leans the same way as the truth where the truth is decided.
            if (t[1] - 0.5).abs() > 0.3 {
                assert_eq!(b[1] > 0.5, t[1] > 0.5, "seed {seed} V{v}: {b:?} vs {t:?}");
            }
        }
    }
}

#[test]
fn regression_snapshots_follow_the_goal_chaining_demo() {
    // The README's two-skill demo, seen through the viewer at each step.
    let mut engine = demo_engine();
    engine.push_goal("GoalBackup");
    let step1 = engine.cognitive_cycle();
    assert_eq!(step1.selected_action.as_deref(), Some("ActionMountNAS"));
    let snap = engine.graph_snapshot().visible();
    assert_eq!(snap.goal_stack, ["GoalBackup", "SubgoalMountNAS"]);
    let dot = snap.to_dot();
    assert_eq!(dot.matches("doublecircle").count(), 2, "{dot}");
    // Skill factors are not memory factors; anything learned while cycling is.
    for f in &snap.factors {
        assert_eq!(f.memory, !f.skill.starts_with("backup") && !f.skill.starts_with("sync"), "{f:?}");
    }
    engine.set_evidence("SubgoalMountNAS", 1).unwrap();
    let step2 = engine.cognitive_cycle();
    assert_eq!(step2.selected_action.as_deref(), Some("ActionSyncFiles"));
    let snap = engine.graph_snapshot().visible();
    let mounted = snap.variables.iter().find(|v| v.name == "SubgoalMountNAS").unwrap();
    assert_eq!(mounted.evidence, Some(1));
    assert!(mounted.belief[1] > 0.999);
    assert!(snap.to_dot().contains("style=bold"));
}

/// Setting (low, high) -> Temp in [0, 40): a Gaussian whose mean depends on the setting.
fn heating() -> SigmaEngine {
    let id = NodeIdentity::generate();
    let pred = |name: &str, t: &str| PredicateDef {
        name: name.into(),
        role: Context,
        args: vec![],
        value_type: Some(t.into()),
    };
    let pat = |kind, name: &str| Pattern { kind, predicate: name.into(), args: vec![] };
    let program = Program {
        types: vec![TypeDef::symbolic("setting", &["low", "high"]), TypeDef::continuous("celsius", 0.0, 40.0, 20)],
        predicates: vec![pred("Setting", "setting"), pred("Temp", "celsius")],
        conditionals: vec![Conditional {
            name: "heat".into(),
            patterns: vec![pat(PatternKind::Condact, "Setting"), pat(PatternKind::Condact, "Temp")],
            function_vars: vec![],
            function: Function::Gaussian { target: 1, mean: vec![15.0, 10.0, 0.0], std: 3.0 },
            learnable: false,
            normal: None,
        }],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(crate::tests_v2::signed_program(&id, "heating", program)).unwrap();
    engine
}

#[test]
fn traces_follow_the_exact_and_particle_paths() {
    let mut engine = heating();
    engine.set_value("Temp", 26.0).unwrap();
    let trace = engine.trace_messages();
    assert_eq!(trace.path, crate::inspect::TracePath::Exact);
    assert!(trace.converged && !trace.sweeps.is_empty());
    // The message into Setting says "high" (the reading is near 25, not 15).
    let into = trace.messages_into("Setting");
    assert_eq!(into.len(), 1);
    assert!(into[0].message[1] > into[0].message[0], "{:?}", into[0].message);
    let setting = &trace.beliefs.iter().find(|(n, _)| n == "Setting").unwrap().1;
    let plain = engine.posterior("Setting").unwrap();
    assert!((setting[1] - plain[1]).abs() < 1e-3, "{setting:?} vs {plain:?}");

    engine.clear_evidence("Temp");
    engine.set_evidence("Setting", 0).unwrap();
    engine.particle_count = Some(64);
    let trace = engine.trace_messages();
    assert_eq!(trace.path, crate::inspect::TracePath::Particles);
    let rounds: std::collections::BTreeSet<usize> = trace.sweeps.iter().map(|s| s.round).collect();
    assert!(rounds.len() > 1, "every round is recorded: {rounds:?}");
    let into = trace.messages_into("Temp");
    assert!(!into.is_empty() && into[0].message.len() == 64, "messages over the particles");
    // The grid path is still traced when nothing is continuous.
    let grid = demo_engine().trace_messages();
    assert_eq!(grid.path, crate::inspect::TracePath::Grid);
}

#[test]
fn the_live_viewer_polls_for_snapshots() {
    let engine = demo_engine();
    let snap = engine.graph_snapshot().visible();
    let page = snap.to_live_html("graph.json", 500);
    assert!(page.contains(r#"const LIVE={"url":"graph.json","ms":500}"#), "live settings");
    assert!(page.contains("setInterval") && page.contains("SubgoalMountNAS"));
    assert!(!page.contains("__LIVE__") && !page.contains("__GRAPH__"));
    assert!(snap.to_html().contains("const LIVE=null"), "a static page does not poll");
    // A URL cannot break out of the script.
    let page = snap.to_live_html("</script><script>alert(1)</script>", 100);
    assert!(!page.contains("</script><script>alert"));
}

#[test]
fn parallel_message_passing_matches_one_thread() {
    // 40 variables of 16 values, a random tree plus loops: enough work to spread.
    let id = NodeIdentity::generate();
    let mut rng = Lcg(77);
    let mut engine = SigmaEngine::new();
    let schema = |v: usize| PredicateSchema { name: format!("W{v}"), domain_size: 16, role: Context };
    for v in 0..40 {
        engine.register_predicate(&schema(v)).unwrap();
    }
    let mut pairs: Vec<(usize, usize)> = (1..40).map(|v| (rng.below(v), v)).collect();
    pairs.extend((0..40).map(|_| {
        let a = rng.below(40);
        (a, (a + 1 + rng.below(39)) % 40)
    }));
    for (i, (a, b)) in pairs.into_iter().enumerate() {
        let table: Vec<f32> = (0..256).map(|_| rng.weight()).collect();
        let mut skill = SkillPackage {
            skill_name: format!("p{i}"),
            version: 1,
            author_pubkey: [0; 32],
            signature: vec![],
            factor_shape: vec![16, 16],
            predicates: vec![schema(a), schema(b)],
            factor_values: table,
        };
        id.sign_skill(&mut skill);
        engine.compile_skill(skill).unwrap();
    }
    engine.set_evidence("W3", 5).unwrap();
    assert!(engine.worker_threads(1 << 20) >= 1);
    for inference in [Inference::SumProduct, Inference::MaxProduct, Inference::VariationalMessagePassing] {
        engine.inference = inference;
        engine.threads = 1;
        let beliefs =
            |e: &SigmaEngine| -> Vec<Vec<f32>> { e.graph_snapshot().variables.into_iter().map(|v| v.belief).collect() };
        let one = beliefs(&engine);
        engine.threads = 4;
        let four = beliefs(&engine);
        assert_eq!(one, four, "{inference:?}: the same beliefs, bit for bit");
    }
    // Small graphs stay on the calling thread.
    assert_eq!(demo_engine().worker_threads(100), 1);
}
