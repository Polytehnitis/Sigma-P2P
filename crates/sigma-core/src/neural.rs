//! PySigma's neural side, small and dependency-free: reverse-mode automatic differentiation
//! ([`Tape`]) and multilayer perceptrons ([`NeuralNet`]) that a conditional can use as its
//! function ([`crate::Function::Neural`]).
//!
//! A neural function is an energy: the factor's value at a point is `exp(net(x))`, kept within
//! the engine's factor bounds, where `x` are the conditional's coordinates (function
//! variables, then each pattern's value: a symbol's index or a continuous value). It is
//! tabulated over the region grid like every analytic form, and evaluated at exact points by
//! the particle path. Gradient learning ([`crate::SigmaEngine::learn_gradient`]) trains it by
//! backpropagation: the observed assignment's probability, normalized over the conditional's
//! `normal` pattern (a softmax over that pattern's values), is raised by one step of gradient
//! ascent computed on the tape.

use crate::MAX_FACTOR_VALUE;
use serde::{Deserialize, Serialize};

/// Most parameters a neural function may carry.
pub const MAX_PARAMETERS: usize = 4096;
/// Most layers.
pub const MAX_LAYERS: usize = 8;

/// One value on a [`Tape`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(usize);

#[derive(Clone, Debug)]
struct Node {
    value: f64,
    /// Up to two parents with the local derivative of this node with respect to each.
    parents: [(usize, f64); 2],
    arity: usize,
}

/// A reverse-mode automatic differentiation tape: operations record their inputs and local
/// derivatives as they are computed; [`Tape::gradient`] then propagates the derivative of one
/// output back to every value.
#[derive(Clone, Debug, Default)]
pub struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, value: f64, parents: &[(usize, f64)]) -> Var {
        let mut p = [(0, 0.0); 2];
        p[..parents.len()].copy_from_slice(parents);
        self.nodes.push(Node { value, parents: p, arity: parents.len() });
        Var(self.nodes.len() - 1)
    }

    /// A leaf (an input or a parameter).
    pub fn leaf(&mut self, value: f64) -> Var {
        self.push(value, &[])
    }

    pub fn value(&self, v: Var) -> f64 {
        self.nodes[v.0].value
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        self.push(self.value(a) + self.value(b), &[(a.0, 1.0), (b.0, 1.0)])
    }

    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        self.push(self.value(a) - self.value(b), &[(a.0, 1.0), (b.0, -1.0)])
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let (x, y) = (self.value(a), self.value(b));
        self.push(x * y, &[(a.0, y), (b.0, x)])
    }

    pub fn scale(&mut self, a: Var, k: f64) -> Var {
        self.push(self.value(a) * k, &[(a.0, k)])
    }

    pub fn tanh(&mut self, a: Var) -> Var {
        let t = self.value(a).tanh();
        self.push(t, &[(a.0, 1.0 - t * t)])
    }

    pub fn relu(&mut self, a: Var) -> Var {
        let x = self.value(a);
        self.push(x.max(0.0), &[(a.0, if x > 0.0 { 1.0 } else { 0.0 })])
    }

    pub fn exp(&mut self, a: Var) -> Var {
        let e = self.value(a).exp();
        self.push(e, &[(a.0, e)])
    }

    pub fn ln(&mut self, a: Var) -> Var {
        let x = self.value(a).max(1e-300);
        self.push(x.ln(), &[(a.0, 1.0 / x)])
    }

    /// Sum of several values.
    pub fn sum(&mut self, xs: &[Var]) -> Var {
        let mut it = xs.iter();
        let Some(&first) = it.next() else { return self.leaf(0.0) };
        it.fold(first, |acc, &x| self.add(acc, x))
    }

    /// `log Σ exp(x_i)`, stably.
    pub fn log_sum_exp(&mut self, xs: &[Var]) -> Var {
        let m = xs.iter().map(|&x| self.value(x)).fold(f64::NEG_INFINITY, f64::max);
        let shift = self.leaf(if m.is_finite() { m } else { 0.0 });
        let exps: Vec<Var> = xs
            .iter()
            .map(|&x| {
                let d = self.sub(x, shift);
                self.exp(d)
            })
            .collect();
        let total = self.sum(&exps);
        let l = self.ln(total);
        self.add(l, shift)
    }

    /// The derivative of `output` with respect to every value on the tape.
    pub fn gradient(&self, output: Var) -> Vec<f64> {
        let mut g = vec![0.0; self.nodes.len()];
        g[output.0] = 1.0;
        for i in (0..=output.0).rev() {
            if g[i] == 0.0 {
                continue;
            }
            let n = &self.nodes[i];
            for &(p, d) in &n.parents[..n.arity] {
                g[p] += g[i] * d;
            }
        }
        g
    }
}

/// A layer's nonlinearity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    Linear,
    Tanh,
    Relu,
}

/// A fully connected layer: `outputs × inputs` weights (row-major) and `outputs` biases.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub inputs: usize,
    pub outputs: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub activation: Activation,
}

/// A multilayer perceptron with one output, used as a conditional's (log) function.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeuralNet {
    pub layers: Vec<Layer>,
}

/// Small deterministic generator for initial weights (so a skill's bytes are reproducible).
fn xorshift(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

impl NeuralNet {
    /// A net with layer sizes `sizes` (`[inputs, hidden…, 1]`), tanh hidden layers, a linear
    /// output and small seeded weights.
    pub fn new(sizes: &[usize], seed: u64) -> Self {
        let mut state = seed.max(1);
        let n = sizes.len().saturating_sub(1);
        let layers = sizes
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                let scale = 1.0 / (w[0].max(1) as f32).sqrt();
                Layer {
                    inputs: w[0],
                    outputs: w[1],
                    weights: (0..w[0] * w[1]).map(|_| xorshift(&mut state) * scale).collect(),
                    bias: vec![0.0; w[1]],
                    activation: if i + 1 == n { Activation::Linear } else { Activation::Tanh },
                }
            })
            .collect();
        NeuralNet { layers }
    }

    pub fn inputs(&self) -> usize {
        self.layers.first().map_or(0, |l| l.inputs)
    }

    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|l| l.weights.len() + l.bias.len()).sum()
    }

    /// Shape and bounds checks: a chain of layers ending in one output, finite parameters.
    pub fn check(&self, inputs: usize) -> Result<(), &'static str> {
        let chained = self.layers.windows(2).all(|w| w[0].outputs == w[1].inputs);
        let shaped = self.layers.iter().all(|l| l.weights.len() == l.inputs * l.outputs && l.bias.len() == l.outputs);
        let finite = self.layers.iter().flat_map(|l| l.weights.iter().chain(&l.bias)).all(|w| w.is_finite());
        if self.layers.is_empty()
            || self.layers.len() > MAX_LAYERS
            || !chained
            || !shaped
            || !finite
            || self.inputs() != inputs
            || self.layers.last().map(|l| l.outputs) != Some(1)
            || self.parameter_count() > MAX_PARAMETERS
        {
            return Err("invalid neural function");
        }
        Ok(())
    }

    /// The net's output at `x`.
    pub fn forward(&self, x: &[f32]) -> f32 {
        let mut h: Vec<f32> = x.to_vec();
        for l in &self.layers {
            h = (0..l.outputs)
                .map(|o| {
                    let z = l.bias[o] + (0..l.inputs).map(|i| l.weights[o * l.inputs + i] * h[i]).sum::<f32>();
                    match l.activation {
                        Activation::Linear => z,
                        Activation::Tanh => z.tanh(),
                        Activation::Relu => z.max(0.0),
                    }
                })
                .collect();
        }
        h[0]
    }

    /// The factor value at `x`: `exp(net(x))` within the factor bounds.
    pub fn value(&self, x: &[f32]) -> f32 {
        let y = self.forward(x);
        if y.is_finite() {
            y.min(MAX_FACTOR_VALUE.ln()).exp().clamp(0.0, MAX_FACTOR_VALUE)
        } else {
            0.0
        }
    }

    /// Records the net's parameters on `tape` as leaves (weights then biases, layer by layer).
    fn on_tape(&self, tape: &mut Tape) -> Vec<Var> {
        self.layers.iter().flat_map(|l| l.weights.iter().chain(&l.bias)).map(|&w| tape.leaf(w as f64)).collect()
    }

    fn forward_on(&self, tape: &mut Tape, params: &[Var], x: &[f32]) -> Var {
        let mut h: Vec<Var> = x.iter().map(|&v| tape.leaf(v as f64)).collect();
        let mut at = 0;
        for l in &self.layers {
            let (w, b) =
                (&params[at..at + l.weights.len()], &params[at + l.weights.len()..at + l.weights.len() + l.outputs]);
            at += l.weights.len() + l.outputs;
            h = (0..l.outputs)
                .map(|o| {
                    let terms: Vec<Var> = (0..l.inputs).map(|i| tape.mul(w[o * l.inputs + i], h[i])).collect();
                    let s = tape.sum(&terms);
                    let z = tape.add(s, b[o]);
                    match l.activation {
                        Activation::Linear => z,
                        Activation::Tanh => tape.tanh(z),
                        Activation::Relu => tape.relu(z),
                    }
                })
                .collect();
        }
        h[0]
    }

    /// One step of gradient ascent on `log softmax(net)[observed]` over the `candidates`
    /// points (the observed point is `candidates[observed]`), computed by backpropagation on a
    /// [`Tape`]. Returns the log-probability before the step.
    pub fn softmax_step(&mut self, candidates: &[Vec<f32>], observed: usize, rate: f32) -> f32 {
        if observed >= candidates.len() {
            return 0.0;
        }
        let mut tape = Tape::new();
        let params = self.on_tape(&mut tape);
        let outs: Vec<Var> = candidates.iter().map(|x| self.forward_on(&mut tape, &params, x)).collect();
        let norm = tape.log_sum_exp(&outs);
        let log_p = tape.sub(outs[observed], norm);
        let g = tape.gradient(log_p);
        let mut i = 0;
        for l in &mut self.layers {
            for w in l.weights.iter_mut().chain(l.bias.iter_mut()) {
                let step = (rate as f64 * g[params[i].0]).clamp(-1.0, 1.0) as f32;
                *w += step;
                i += 1;
            }
        }
        tape.value(log_p) as f32
    }
}
