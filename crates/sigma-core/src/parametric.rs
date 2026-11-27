//! Learning the parameters of analytic functions, as PySigma does for its parametric
//! distributions, rather than only rescaling their region tables.
//!
//! * **Gaussians.** A learnable conditional whose function is a [`Function::Gaussian`] learns
//!   its mean (the affine coefficients of the other coordinates, by normalized least mean
//!   squares) and its spread (by gradient ascent on the log-density) from each fully observed
//!   assignment, using the exact value behind continuous evidence.
//! * **Neural functions.** A learnable conditional whose function is a [`Function::Neural`] net
//!   is trained by backpropagation: one step of gradient ascent on the observed value's
//!   softmax probability over the `normal` pattern's values (see [`crate::neural`]).
//!
//! After a step the function is tabulated again over the region grid for every factor
//! grounded from the same function slice, so grid, exact and particle inference all see the
//! learned form. Learned functions are kept in [`crate::LearnedModels::functions`] (persisted
//! with the memories) and re-applied when the skill is compiled again, e.g. after a restart.
//! Each one is also a sync register (`function/<skill>#<offset>`), so a function learned on
//! one device reaches the user's paired devices; the newest write wins, and a function that
//! fails [`function_is_valid`] or does not fit the receiving skill's factor is ignored.

use crate::{Function, SigmaEngine, MAX_FACTOR_VALUE};
use std::sync::Arc;

/// Most coordinates a synced Gaussian may have.
const MAX_FUNCTION_DIMS: usize = 64;

/// Most learned functions kept (and accepted from paired devices).
pub const MAX_LEARNED_FUNCTIONS: usize = 256;

/// Whether a learned function received from elsewhere is well formed: a Gaussian with finite
/// parameters and a positive spread, or a neural net that passes its own checks. (Its fit to a
/// particular factor is checked when it is applied.)
pub fn function_is_valid(function: &Function) -> bool {
    match function {
        Function::Gaussian { target, mean, std } => {
            *target < MAX_FUNCTION_DIMS
                && !mean.is_empty()
                && mean.len() <= MAX_FUNCTION_DIMS + 1
                && mean.iter().all(|m| m.is_finite())
                && std.is_finite()
                && *std >= MIN_STD
        }
        Function::Neural(net) => net.check(net.inputs()).is_ok(),
        _ => false,
    }
}

/// Smallest standard deviation a learned Gaussian keeps.
const MIN_STD: f32 = 1e-3;

impl SigmaEngine {
    /// Coordinate and width of each value of variable `v`: a symbol's index (width 1) or a
    /// region's midpoint (width of the region).
    fn axis_of(&self, v: usize) -> (Vec<f32>, f32) {
        let d = self.vars[v].schema.domain_size;
        match self.vars[v].range {
            Some((lo, hi)) => {
                let w = (hi - lo) / d as f32;
                ((0..d).map(|i| lo + (i as f32 + 0.5) * w).collect(), w)
            }
            None => ((0..d).map(|i| i as f32).collect(), 1.0),
        }
    }

    /// The observed coordinate of every slot of factor `fi` (the exact value behind continuous
    /// evidence), with each slot's observed index, when every slot is observed.
    fn observed_point(&self, fi: usize) -> Option<(Vec<usize>, Vec<f32>)> {
        let f = &self.factors[fi];
        let mut idx = Vec::with_capacity(f.vars.len());
        let mut x = Vec::with_capacity(f.vars.len());
        for &v in &f.vars {
            let e = self.vars[v].evidence?;
            idx.push(e);
            x.push(match self.exact.get(&v) {
                Some(&value) => value,
                None => self.axis_of(v).0[e],
            });
        }
        Some((idx, x))
    }

    /// Tabulates factor `fi`'s analytic function over the region grid again.
    fn retabulate(&mut self, fi: usize) {
        let Some(a) = self.factors[fi].analytic.clone() else { return };
        let axes: Vec<(Vec<f32>, f32)> = self.factors[fi].vars.iter().map(|&v| self.axis_of(v)).collect();
        let widths: Vec<f32> = a.prefix.iter().map(|_| 1.0).chain(axes.iter().map(|a| a.1)).collect();
        for (idx, w) in self.factors[fi].table.indexed_iter_mut() {
            let point: Vec<f32> =
                a.prefix.iter().copied().chain(axes.iter().enumerate().map(|(s, ax)| ax.0[idx[s]])).collect();
            let value = a.function.eval(&point, &widths);
            *w = if value.is_finite() { value.clamp(0.0, MAX_FACTOR_VALUE) } else { 0.0 };
        }
    }

    /// Gives every factor grounded from the same function slice as `fi` the function `function`
    /// and tabulates it again; records it as learned.
    fn set_learned_function(&mut self, fi: usize, function: Function) {
        let (skill, offset) = (self.factors[fi].skill.skill_name.clone(), self.factors[fi].offset);
        for fj in 0..self.factors.len() {
            let f = &self.factors[fj];
            if f.skill.skill_name != skill || f.offset != offset || f.memory {
                continue;
            }
            let Some(a) = &f.analytic else { continue };
            let mut a = (**a).clone();
            a.function = function.clone();
            self.factors[fj].analytic = Some(Arc::new(a));
            self.retabulate(fj);
        }
        if self.models.functions.len() < MAX_LEARNED_FUNCTIONS
            || self.models.functions.contains_key(&(skill.clone(), offset))
        {
            self.models.functions.insert((skill, offset), function);
        }
    }

    /// Takes a function learned on one of the user's other devices for skill `skill`'s factors
    /// at weight offset `offset`, and applies it to those factors. Returns whether it was kept.
    pub fn adopt_learned_function(&mut self, skill: &str, offset: usize, function: Function) -> bool {
        let key = (skill.to_string(), offset);
        if skill.is_empty()
            || !function_is_valid(&function)
            || (!self.models.functions.contains_key(&key) && self.models.functions.len() >= MAX_LEARNED_FUNCTIONS)
        {
            return false;
        }
        self.models.functions.insert(key, function);
        self.apply_learned_functions();
        true
    }

    /// One learning step of a Gaussian or neural factor from the observed assignment, when
    /// every slot is observed. Returns the log-likelihood of the observation before the step,
    /// or `None` when the factor has no such function or is not fully observed.
    pub(crate) fn learn_analytic(&mut self, fi: usize, rate: f32) -> Option<f32> {
        let a = self.factors[fi].analytic.clone()?;
        if !matches!(a.function, Function::Gaussian { .. } | Function::Neural(_)) {
            return None;
        }
        let (idx, x) = self.observed_point(fi)?;
        let point: Vec<f32> = a.prefix.iter().copied().chain(x.iter().copied()).collect();
        let (function, log_likelihood) = match &a.function {
            Function::Gaussian { target, mean, std } => {
                let t = *target;
                let others = |i: &usize| *i != t;
                let mu = mean[0] + (0..point.len()).filter(others).map(|i| mean[i + 1] * point[i]).sum::<f32>();
                let e = point[t] - mu;
                let ll = -0.5 * (e / std).powi(2) - (std * (2.0 * std::f32::consts::PI).sqrt()).ln();
                let norm = 1.0 + (0..point.len()).filter(others).map(|i| point[i] * point[i]).sum::<f32>();
                let mut mean = mean.clone();
                mean[0] += rate * e / norm;
                for i in (0..point.len()).filter(others) {
                    mean[i + 1] += rate * e * point[i] / norm;
                }
                // The spread moves toward the size of the error, on a log scale.
                let ratio = ((e / std).powi(2) - 1.0).clamp(-4.0, 4.0);
                let std = (std * (0.5 * rate * ratio).exp()).max(MIN_STD);
                (Function::Gaussian { target: t, mean, std }, ll)
            }
            Function::Neural(net) => {
                let f = &self.factors[fi];
                let mut net = net.clone();
                // Softmax over the normal pattern's values (or over the whole table when small).
                let axes: Vec<Vec<f32>> = f.vars.iter().map(|&v| self.axis_of(v).0).collect();
                let (candidates, observed): (Vec<Vec<f32>>, usize) = match f.normal {
                    Some(n) => {
                        let c = (0..axes[n].len())
                            .map(|i| {
                                let mut p = point.clone();
                                p[a.prefix.len() + n] = axes[n][i];
                                p
                            })
                            .collect();
                        (c, idx[n])
                    }
                    None if f.table.len() <= 256 => {
                        let mut c = Vec::with_capacity(f.table.len());
                        let mut observed = 0;
                        for (k, (at, _)) in f.table.indexed_iter().enumerate() {
                            if (0..idx.len()).all(|s| at[s] == idx[s]) {
                                observed = k;
                            }
                            c.push(a.prefix.iter().copied().chain((0..idx.len()).map(|s| axes[s][at[s]])).collect());
                        }
                        (c, observed)
                    }
                    None => return None,
                };
                let ll = net.softmax_step(&candidates, observed, rate);
                (Function::Neural(net), ll)
            }
            _ => return None,
        };
        self.set_learned_function(fi, function);
        Some(log_likelihood)
    }

    /// Re-applies learned functions to the skill factors they were learned for (after a
    /// restart, or when the skill is compiled again).
    pub(crate) fn apply_learned_functions(&mut self) {
        if self.models.functions.is_empty() {
            return;
        }
        for fi in 0..self.factors.len() {
            let f = &self.factors[fi];
            if f.memory {
                continue;
            }
            let key = (f.skill.skill_name.clone(), f.offset);
            let (Some(learned), Some(a)) = (self.models.functions.get(&key), &f.analytic) else { continue };
            if a.function == *learned || std::mem::discriminant(&a.function) != std::mem::discriminant(learned) {
                continue;
            }
            let n = f.vars.len() + a.prefix.len();
            if learned.check_dims(n).is_err() {
                continue;
            }
            let mut a = (**a).clone();
            a.function = learned.clone();
            self.factors[fi].analytic = Some(Arc::new(a));
            self.retabulate(fi);
        }
    }

    /// The learned parameters of the Gaussian or neural function behind skill `skill`'s factors
    /// at weight offset `offset`, if any were learned.
    pub fn learned_function(&self, skill: &str, offset: usize) -> Option<&Function> {
        self.models.functions.get(&(skill.to_string(), offset))
    }
}
