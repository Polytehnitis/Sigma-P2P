//! Exact message passing over continuous variables with piecewise-linear functions.
//!
//! Unless `SigmaEngine::tabulate_continuous` is set (or particles are in use), a graph with
//! continuous variables is solved on [`Plm`]s, as in Sigma: every factor function becomes a
//! piecewise-linear function whose slices fall only where the function changes (a
//! piecewise-linear conditional keeps its own pieces, not the region grid), every message is
//! a one-dimensional piecewise-linear density, and each factor-to-variable message is the
//! exact integral of the factor times its incoming messages. Continuous evidence set with
//! `set_value` stays an exact point. A discrete variable is a dimension of unit slices, so on
//! a purely discrete graph this is ordinary sum-product.
//!
//! Beliefs still come out as region histograms (each region's exact mass) for the rest of the
//! cycle; [`SigmaEngine::density`] gives the exact posterior density, and
//! `expected_value` its exact mean. Gradient learning keeps updating the region tables, and
//! a learned change scales the exact function region by region.

use crate::plm::{merge_bounds, Plm, MAX_REGIONS};
use crate::program::{Analytic, Function};
use crate::{PatternKind, SigmaEngine, BP_ITERS};

/// Message changes (L1) below this end message passing.
const TOLERANCE: f64 = 1e-5;

/// What a variable's prior or belief is: a density, or an exact point (continuous evidence).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Belief {
    Density(Plm),
    Point(f64),
}

/// Priors and converged messages of one exact inference.
pub(crate) struct ExactState {
    prior: Vec<Belief>,
    /// Per factor, per slot: the factor-to-variable message.
    msgs: Vec<Vec<Plm>>,
}

impl SigmaEngine {
    /// Whether this graph is solved exactly: it has a continuous variable that a factor uses,
    /// and neither particles nor the fixed grid were asked for.
    pub(crate) fn uses_exact(&self) -> bool {
        self.particle_count.is_none()
            && !self.tabulate_continuous
            && self.factors.iter().any(|f| f.vars.iter().any(|&v| self.vars[v].range.is_some()))
    }

    /// The exact posterior density of a continuous predicate given working memory, as a
    /// piecewise-linear function over its range (`None` for a discrete predicate, or when the
    /// value is known exactly: see `expected_value`).
    pub fn density(&self, name: &str) -> Option<Plm> {
        let v = *self.index.get(name)?;
        self.vars[v].range?;
        let state = self.exact_state(&|_| None)?;
        match state.belief(self, v, None) {
            Belief::Density(p) => Some(p),
            Belief::Point(_) => None,
        }
    }

    /// The exact posterior mean of continuous predicate `v`, if exact inference applies.
    pub(crate) fn exact_mean(&self, v: usize) -> Option<f32> {
        let state = self.exact_state(&|_| None)?;
        Some(match state.belief(self, v, None) {
            Belief::Density(p) => p.mean() as f32,
            Belief::Point(x) => x as f32,
        })
    }

    /// Region beliefs of every variable by exact inference, or `None` when it does not apply.
    pub(crate) fn infer_exact(&self, clamp: &dyn Fn(usize) -> Option<usize>) -> Option<Vec<Vec<f32>>> {
        let state = self.exact_state(clamp)?;
        Some((0..self.vars.len()).map(|v| state.grid_belief(self, v, None)).collect())
    }

    /// Runs exact message passing to quiescence.
    pub(crate) fn exact_state(&self, clamp: &dyn Fn(usize) -> Option<usize>) -> Option<ExactState> {
        if !self.uses_exact() {
            return None;
        }
        let functions: Vec<Plm> = (0..self.factors.len()).map(|fi| self.factor_plm(fi)).collect::<Option<_>>()?;
        let prior: Vec<Belief> = (0..self.vars.len()).map(|v| self.exact_prior(v, clamp)).collect();
        let uniform = |v: usize| self.uniform(v);
        let mut state = ExactState {
            msgs: self.factors.iter().map(|f| f.vars.iter().map(|&v| uniform(v)).collect()).collect(),
            prior,
        };
        for _ in 0..BP_ITERS {
            let next: Vec<Vec<Plm>> = self
                .factors
                .iter()
                .enumerate()
                .map(|(fi, f)| {
                    if self.attended.as_ref().is_some_and(|mask| !mask[fi]) {
                        return f.vars.iter().map(|&v| uniform(v)).collect();
                    }
                    let inbound: Vec<Belief> = f
                        .vars
                        .iter()
                        .enumerate()
                        .map(|(s, &v)| match f.kinds[s] {
                            PatternKind::Action => Belief::Density(uniform(v)),
                            _ => state.belief(self, v, Some((fi, s))),
                        })
                        .collect();
                    (0..f.vars.len())
                        .map(|s| {
                            if f.kinds[s] == PatternKind::Condition {
                                return uniform(f.vars[s]);
                            }
                            let mut g = functions[fi].clone();
                            for j in (0..f.vars.len()).rev().filter(|&j| j != s) {
                                g = match &inbound[j] {
                                    Belief::Point(x) => g.substitute(j, *x),
                                    Belief::Density(m) => g.mul_sum_out(j, m),
                                };
                            }
                            nonnegative(g).simplified().normalized()
                        })
                        .collect()
                })
                .collect();
            let delta = next
                .iter()
                .flatten()
                .zip(state.msgs.iter().flatten())
                .fold(0.0f64, |m, (a, b)| m.max(a.l1_distance(b)));
            state.msgs = next;
            if delta < TOLERANCE {
                break;
            }
        }
        Some(state)
    }

    /// Slice boundaries of a variable's whole domain (unit slices for a discrete variable,
    /// its regions for a continuous one).
    fn grid_bounds(&self, v: usize) -> Vec<f64> {
        let d = self.vars[v].schema.domain_size;
        match self.vars[v].range {
            Some((lo, hi)) => (0..=d).map(|i| lo as f64 + (hi - lo) as f64 * i as f64 / d as f64).collect(),
            None => (0..=d).map(|i| i as f64).collect(),
        }
    }

    fn uniform(&self, v: usize) -> Plm {
        let b = self.grid_bounds(v);
        Plm::constant(vec![vec![b[0], b[b.len() - 1]]], 1.0).normalized()
    }

    /// Evidence or a clamp (a point for exact continuous evidence, else the region), else a
    /// percept, else uniform.
    fn exact_prior(&self, v: usize, clamp: &dyn Fn(usize) -> Option<usize>) -> Belief {
        let bounds = self.grid_bounds(v);
        let evidence = self.vars[v].evidence;
        if let Some(r) = evidence.or_else(|| clamp(v)) {
            if let (Some(_), Some(&x)) = (evidence, self.exact.get(&v)) {
                return Belief::Point(x as f64);
            }
            let values: Vec<f64> = (0..bounds.len() - 1).map(|i| f64::from(u8::from(i == r))).collect();
            return Belief::Density(Plm::steps(bounds, &values).simplified().normalized());
        }
        if let Some(dist) = self.percepts.get(&v) {
            let values: Vec<f64> = dist.iter().zip(bounds.windows(2)).map(|(&p, w)| p as f64 / (w[1] - w[0])).collect();
            return Belief::Density(Plm::steps(bounds, &values).simplified().normalized());
        }
        Belief::Density(self.uniform(v))
    }

    /// A factor's function as a piecewise-linear function over its slots' domains.
    fn factor_plm(&self, fi: usize) -> Option<Plm> {
        let f = &self.factors[fi];
        let grid: Vec<Vec<f64>> = f.vars.iter().map(|&v| self.grid_bounds(v)).collect();
        let live = Plm {
            coef: f.table.iter().flat_map(|&w| one_region(w as f64, grid.len())).collect(),
            bounds: grid.clone(),
        };
        let Some(a) = &f.analytic else { return Some(live.simplified()) };
        let continuous: Vec<Option<(f64, f64)>> =
            f.vars.iter().map(|&v| self.vars[v].range.map(|(lo, hi)| (lo as f64, hi as f64))).collect();
        let exact = match &a.function {
            Function::Table(_) => return Some(live.simplified()),
            Function::PiecewiseLinear { default, pieces } => {
                let k = a.prefix.len();
                let bounds: Vec<Vec<f64>> = (0..f.vars.len())
                    .map(|s| match continuous[s] {
                        None => grid[s].clone(),
                        Some((lo, hi)) => {
                            let cuts: Vec<f64> = pieces
                                .iter()
                                .flat_map(|p| [p.lo[k + s] as f64, p.hi[k + s] as f64])
                                .filter(|&x| x > lo && x < hi)
                                .collect();
                            merge_bounds(&[lo, hi], &cuts)
                        }
                    })
                    .collect();
                build(bounds, &continuous, |centre| {
                    let point: Vec<f32> = a.prefix.iter().copied().chain(centre.iter().map(|&x| x as f32)).collect();
                    let inside = |p: &&crate::program::Piece| {
                        point.iter().zip(p.lo.iter().zip(&p.hi)).all(|(v, (l, h))| l <= v && v < h)
                    };
                    match pieces.iter().find(inside) {
                        None => (*default as f64, vec![0.0; centre.len()]),
                        Some(p) => {
                            let c0 = p.coef[0] as f64
                                + p.coef[1..=k].iter().zip(&a.prefix).map(|(c, x)| (c * x) as f64).sum::<f64>();
                            let slopes: Vec<f64> = (0..centre.len()).map(|s| p.coef[1 + k + s] as f64).collect();
                            (c0, slopes)
                        }
                    }
                })?
            }
            Function::Gaussian { target, mean, std } => {
                // Slices a quarter of a standard deviation apart (along the target, and along
                // each dimension the mean moves with, scaled by how fast it moves).
                let k = a.prefix.len();
                let mut counts: Vec<usize> = (0..f.vars.len())
                    .map(|s| match continuous[s] {
                        None => grid[s].len() - 1,
                        Some((lo, hi)) => {
                            let speed = if k + s == *target { 1.0 } else { (mean[1 + k + s] as f64).abs() };
                            let width = (*std as f64 / 4.0) / speed.max(1e-6);
                            (((hi - lo) / width).ceil() as usize).clamp(grid[s].len() - 1, 4096)
                        }
                    })
                    .collect();
                while counts.iter().try_fold(1usize, |a, &c| a.checked_mul(c)).is_none_or(|n| n > MAX_REGIONS / 4) {
                    let (s, _) =
                        counts.iter().enumerate().filter(|&(s, _)| continuous[s].is_some()).max_by_key(|&(_, &c)| c)?;
                    counts[s] = (counts[s] / 2).max(1);
                }
                let bounds: Vec<Vec<f64>> = (0..f.vars.len())
                    .map(|s| match continuous[s] {
                        None => grid[s].clone(),
                        Some((lo, hi)) => {
                            (0..=counts[s]).map(|i| lo + (hi - lo) * i as f64 / counts[s] as f64).collect()
                        }
                    })
                    .collect();
                sampled(a, bounds, &continuous)?
            }
        };
        // Learning changes the region tables; scale the exact function by the change.
        let init = self.initial.get(&f.skill.skill_name)?;
        let ratios: Vec<f64> = f
            .table
            .iter()
            .enumerate()
            .map(|(i, &w)| match init.get(f.offset + i) {
                Some(&i0) if i0 > 0.0 => w as f64 / i0 as f64,
                _ => 1.0,
            })
            .collect();
        let exact = if ratios.iter().all(|r| (r - 1.0).abs() < 1e-6) {
            exact
        } else {
            let scale = Plm { coef: ratios.iter().flat_map(|&r| one_region(r, grid.len())).collect(), bounds: grid };
            times_steps(&exact, &scale)?
        };
        Some(nonnegative(exact).simplified())
    }
}

impl ExactState {
    /// The mass that factor `fi`'s message to its slot `s` puts on value `x` (discrete).
    pub(crate) fn message_mass(&self, fi: usize, s: usize, x: usize) -> f32 {
        self.msgs[fi][s].moments(x as f64, x as f64 + 1.0).0 as f32
    }

    /// Prior times every incoming message except `skip` (a factor and slot).
    pub(crate) fn belief(&self, e: &SigmaEngine, v: usize, skip: Option<(usize, usize)>) -> Belief {
        let mut b = match &self.prior[v] {
            Belief::Point(x) => return Belief::Point(*x),
            Belief::Density(p) => p.clone(),
        };
        for (fi, f) in e.factors.iter().enumerate() {
            for (s, _) in f.vars.iter().enumerate().filter(|&(s, &u)| u == v && skip != Some((fi, s))) {
                b = b.mul_1d(&self.msgs[fi][s]);
            }
        }
        Belief::Density(b.normalized())
    }

    /// A belief as region masses (a point falls in one region).
    pub(crate) fn grid_belief(&self, e: &SigmaEngine, v: usize, skip: Option<(usize, usize)>) -> Vec<f32> {
        let bounds = e.grid_bounds(v);
        match self.belief(e, v, skip) {
            Belief::Point(x) => {
                let k = bounds.partition_point(|&b| b <= x).clamp(1, bounds.len() - 1) - 1;
                (0..bounds.len() - 1).map(|i| f32::from(u8::from(i == k))).collect()
            }
            Belief::Density(p) => crate::normalize(p.masses(&bounds).into_iter().map(|m| m as f32).collect()),
        }
    }
}

fn one_region(value: f64, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n + 1];
    c[0] = value;
    c
}

fn region_count(bounds: &[Vec<f64>]) -> Option<usize> {
    bounds.iter().try_fold(1usize, |a, b| a.checked_mul(b.len() - 1)).filter(|&n| n <= MAX_REGIONS)
}

/// Builds a function region by region: `at(centre)` gives the region's constant term and its
/// slope along each dimension (slopes along discrete dimensions are folded into the constant,
/// since a discrete value is a point).
fn build(
    bounds: Vec<Vec<f64>>,
    continuous: &[Option<(f64, f64)>],
    at: impl Fn(&[f64]) -> (f64, Vec<f64>),
) -> Option<Plm> {
    let n = bounds.len();
    let total = region_count(&bounds)?;
    let shape: Vec<usize> = bounds.iter().map(|b| b.len() - 1).collect();
    let mut coef = Vec::with_capacity(total * (n + 1));
    let mut idx = vec![0usize; n];
    for _ in 0..total {
        let centre: Vec<f64> = (0..n)
            .map(|d| match continuous[d] {
                Some(_) => (bounds[d][idx[d]] + bounds[d][idx[d] + 1]) / 2.0,
                None => bounds[d][idx[d]],
            })
            .collect();
        let (mut c0, slopes) = at(&centre);
        let mut region = vec![0.0; n + 1];
        for d in 0..n {
            match continuous[d] {
                Some(_) => region[d + 1] = slopes[d],
                None => c0 += slopes[d] * centre[d],
            }
        }
        region[0] = c0;
        coef.extend(region);
        for d in (0..n).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Some(Plm { bounds, coef })
}

/// A smooth analytic function (a Gaussian) approximated by linear pieces: on each region, the
/// value at the centre and the secant slope across the region along each continuous dimension.
fn sampled(a: &Analytic, bounds: Vec<Vec<f64>>, continuous: &[Option<(f64, f64)>]) -> Option<Plm> {
    let widths: Vec<Vec<f64>> = bounds.iter().map(|b| b.windows(2).map(|w| w[1] - w[0]).collect()).collect();
    let eval = |x: &[f64]| -> f64 {
        let x: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        a.eval(&x) as f64
    };
    let bref = &bounds;
    build(bounds.clone(), continuous, |centre| {
        let value = eval(centre);
        let slopes: Vec<f64> = (0..centre.len())
            .map(|d| {
                if continuous[d].is_none() {
                    return 0.0;
                }
                let k = bref[d].partition_point(|&b| b <= centre[d]) - 1;
                let h = widths[d][k] / 2.0;
                let (mut lo, mut hi) = (centre.to_vec(), centre.to_vec());
                lo[d] -= h;
                hi[d] += h;
                (eval(&hi) - eval(&lo)) / (2.0 * h)
            })
            .collect();
        (value - slopes_dot(&slopes, centre, continuous), slopes)
    })
}

fn slopes_dot(slopes: &[f64], centre: &[f64], continuous: &[Option<(f64, f64)>]) -> f64 {
    slopes.iter().zip(centre).zip(continuous).filter(|(_, c)| c.is_some()).map(|((s, x), _)| s * x).sum()
}

/// `f` times a function that is constant on every region of its own grid (exact).
fn times_steps(f: &Plm, steps: &Plm) -> Option<Plm> {
    let n = f.dims();
    let bounds: Vec<Vec<f64>> = (0..n).map(|d| merge_bounds(&f.bounds[d], &steps.bounds[d])).collect();
    let continuous = vec![Some((0.0, 0.0)); n];
    let fb = &bounds;
    build(bounds.clone(), &continuous, |centre| {
        let s = steps.eval(centre);
        // f's region at this centre: value and slopes.
        let v = f.eval(centre);
        let slopes: Vec<f64> = (0..n)
            .map(|d| {
                let k = fb[d].partition_point(|&b| b <= centre[d]) - 1;
                let h = (fb[d][k + 1] - fb[d][k]) / 4.0;
                let (mut lo, mut hi) = (centre.to_vec(), centre.to_vec());
                lo[d] -= h;
                hi[d] += h;
                (f.eval(&hi) - f.eval(&lo)) / (2.0 * h) * s
            })
            .collect();
        (v * s - slopes.iter().zip(centre).map(|(a, b)| a * b).sum::<f64>(), slopes)
    })
}

/// Replaces every region that dips below zero by its (clamped) value at the centre.
fn nonnegative(mut f: Plm) -> Plm {
    let n = f.dims();
    let regions = f.regions();
    let shape: Vec<usize> = f.bounds.iter().map(|b| b.len() - 1).collect();
    let mut idx = vec![0usize; n];
    for r in 0..regions {
        let c = &mut f.coef[r * (n + 1)..(r + 1) * (n + 1)];
        let (mut lo, mut hi, mut mid) = (c[0], c[0], c[0]);
        for d in 0..n {
            let (a, b) = (f.bounds[d][idx[d]], f.bounds[d][idx[d] + 1]);
            lo += (c[d + 1] * a).min(c[d + 1] * b);
            hi += (c[d + 1] * a).max(c[d + 1] * b);
            mid += c[d + 1] * (a + b) / 2.0;
        }
        if lo < 0.0 || !hi.is_finite() || !mid.is_finite() {
            c.iter_mut().for_each(|x| *x = 0.0);
            c[0] = if mid.is_finite() { mid.max(0.0) } else { 0.0 };
        }
        for d in (0..n).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    f
}
