//! The optional GPU backend (cargo feature `gpu`) against the CPU path. Without the feature,
//! or without an adapter, the GPU flag falls back to the CPU and the results are identical.
//! Set `SIGMA_REQUIRE_GPU=1` to fail when no GPU (or software Vulkan) adapter is found.

use super::*;
use rand::{Rng, SeedableRng};
use PredicateRole::*;

/// A loopy graph of variables with 2–4 values and factors of one to three slots.
fn random_engine(seed: u64) -> (SigmaEngine, Vec<String>) {
    let id = NodeIdentity::generate();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = 4 + (seed as usize % 4);
    let names: Vec<String> = (0..n).map(|v| format!("G{v}")).collect();
    let sizes: Vec<usize> = (0..n).map(|_| rng.gen_range(2..=4)).collect();
    let mut engine = SigmaEngine::new();
    let schema = |v: usize| PredicateSchema { name: names[v].clone(), domain_size: sizes[v], role: Context };
    for v in 0..n {
        engine.register_predicate(&schema(v)).unwrap();
    }
    for i in 0..n + 3 {
        let arity = 1 + i % 3;
        let mut vars: Vec<usize> = Vec::new();
        while vars.len() < arity {
            let v = rng.gen_range(0..n);
            if !vars.contains(&v) {
                vars.push(v);
            }
        }
        let shape: Vec<usize> = vars.iter().map(|&v| sizes[v]).collect();
        let values: Vec<f32> = (0..shape.iter().product::<usize>()).map(|_| rng.gen_range(0.1..5.0)).collect();
        let mut skill = SkillPackage {
            skill_name: format!("g{i}"),
            version: 1,
            author_pubkey: [0; 32],
            signature: vec![],
            factor_shape: shape,
            predicates: vars.iter().map(|&v| schema(v)).collect(),
            factor_values: values,
        };
        id.sign_skill(&mut skill);
        engine.compile_skill(skill).unwrap();
    }
    (engine, names)
}

#[test]
fn gpu_messages_match_the_cpu() {
    let required = std::env::var("SIGMA_REQUIRE_GPU").is_ok_and(|v| v == "1");
    if required {
        assert!(gpu::available(), "SIGMA_REQUIRE_GPU=1 but no GPU adapter was found");
    }
    let before = gpu::sweeps();
    for seed in 0..12u64 {
        let (mut engine, names) = random_engine(seed);
        if seed % 3 == 0 {
            engine.set_evidence(&names[0], 1).unwrap();
        }
        for mode in [Inference::SumProduct, Inference::MaxProduct] {
            engine.inference = mode;
            engine.gpu = false;
            let cpu: Vec<Vec<f32>> = names.iter().map(|n| engine.posterior(n).unwrap()).collect();
            engine.gpu = true;
            let gpu: Vec<Vec<f32>> = names.iter().map(|n| engine.posterior(n).unwrap()).collect();
            for (c, g) in cpu.iter().zip(&gpu) {
                let off = c.iter().zip(g).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert!(off < 1e-4, "seed {seed} {mode:?}: {c:?} vs {g:?}");
            }
        }
    }
    let used = gpu::sweeps() - before;
    if gpu::available() {
        assert!(used > 0, "an adapter was found, so sweeps ran on it");
    } else {
        assert_eq!(used, 0, "no adapter: everything stayed on the CPU");
    }
    eprintln!("GPU available: {}, sweeps on the GPU: {used}", gpu::available());
}

#[test]
fn modes_the_gpu_does_not_cover_fall_back_to_the_cpu() {
    let (mut engine, names) = random_engine(3);
    for mode in [Inference::VariationalMessagePassing, Inference::ExpectationPropagation] {
        engine.inference = mode;
        engine.gpu = false;
        let cpu = engine.posterior(&names[1]).unwrap();
        engine.gpu = true;
        assert_eq!(engine.posterior(&names[1]).unwrap(), cpu, "{mode:?} runs on the CPU either way");
    }
}
