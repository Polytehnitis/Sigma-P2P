//! Particle-based message passing over continuous variables, in the style of PySigma.
//!
//! With `SigmaEngine::particle_count` set, each continuous variable is represented inside
//! the graph by a set of particles instead of its fixed regions. Factor functions are
//! evaluated at the particles' exact values (analytic piecewise-linear and Gaussian forms
//! directly, tables through their regions), messages are vectors over the particles, and
//! importance weights (prior density / proposal density) keep the sums over particles
//! unbiased estimates of the integrals that sum-product needs (particle belief propagation).
//! After each round, every variable's particles are resampled from its belief with a
//! reflected Gaussian kernel, which focuses them where the posterior mass is.
//!
//! Discrete variables and the rest of the cycle are unchanged: beliefs over continuous
//! variables are reported both as particles and as region histograms.

use crate::{normalize, Particles, SigmaEngine, MAX_FACTOR_VALUE, MAX_TABLE_LEN};
use ndarray::{ArrayD, IxDyn};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::{BTreeMap, HashMap};

/// Resampling rounds of particle message passing.
const ROUNDS: usize = 4;
/// Most particles per continuous variable.
pub const MAX_PARTICLES: usize = 512;
/// Particles used by `particle_belief` when `particle_count` is not set.
pub const DEFAULT_PARTICLES: usize = 64;

impl Particles {
    /// Draws `n` equally weighted particles in proportion to the weights.
    pub fn resample(&self, n: usize, rng: &mut impl Rng) -> Particles {
        let total: f32 = self.weights.iter().sum();
        let values = (0..n)
            .map(|_| {
                let mut u = rng.gen::<f32>() * total;
                let k = self.weights.iter().position(|w| {
                    u -= w;
                    u < 0.0
                });
                self.values[k.unwrap_or(self.values.len() - 1)]
            })
            .collect();
        Particles { values, weights: vec![1.0 / n.max(1) as f32; n] }
    }

    pub fn variance(&self) -> f32 {
        let z: f32 = self.weights.iter().sum();
        let m = self.mean();
        self.values.iter().zip(&self.weights).map(|(x, w)| w * (x - m).powi(2)).sum::<f32>() / z
    }
}

/// A continuous variable's particles during one inference, with the proposal density each
/// was drawn from. A delta domain is one exact value (continuous evidence).
struct Domain {
    points: Vec<f32>,
    proposal: Vec<f32>,
    delta: bool,
}

/// A weighted Gaussian kernel density on `[lo, hi)`, reflected at both ends.
struct Kde {
    points: Vec<f32>,
    weights: Vec<f32>,
    h: f32,
    lo: f32,
    hi: f32,
}

impl Kde {
    fn new(points: &[f32], weights: &[f32], (lo, hi): (f32, f32)) -> Self {
        let weights = normalize(weights.to_vec());
        let p = Particles { values: points.to_vec(), weights: weights.clone() };
        let n = points.len().max(1) as f32;
        let h = (1.06 * p.variance().sqrt() * n.powf(-0.2)).max((hi - lo) * 1e-3);
        Kde { points: points.to_vec(), weights, h, lo, hi }
    }

    fn density(&self, x: f32) -> f32 {
        let k = |d: f32| (-0.5 * (d / self.h).powi(2)).exp() / (self.h * (2.0 * std::f32::consts::PI).sqrt());
        self.points
            .iter()
            .zip(&self.weights)
            .map(|(&p, w)| w * (k(x - p) + k(2.0 * self.lo - x - p) + k(2.0 * self.hi - x - p)))
            .sum()
    }

    fn sample(&self, rng: &mut impl Rng) -> f32 {
        let k = Particles { values: self.points.clone(), weights: self.weights.clone() }.resample(1, rng).values[0];
        let mut x = k + self.h * normal(rng);
        if x < self.lo {
            x = 2.0 * self.lo - x;
        }
        if x >= self.hi {
            x = 2.0 * self.hi - x;
        }
        x.clamp(self.lo, self.hi - (self.hi - self.lo) * 1e-6)
    }
}

fn normal(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(f32::MIN_POSITIVE);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

impl SigmaEngine {
    /// The posterior of a continuous predicate as a particle message, given working memory
    /// alone. Uses `particle_count` particles (or `DEFAULT_PARTICLES`).
    pub fn particle_belief(&self, name: &str) -> Option<Particles> {
        let v = *self.index.get(name)?;
        let n = self.particle_count.unwrap_or(DEFAULT_PARTICLES);
        self.infer_particles(&|_| None, n)?.1.remove(&v)
    }

    /// Particle belief propagation. Returns region beliefs for every variable plus the
    /// particle beliefs of continuous ones, or `None` when there is nothing continuous to do
    /// (or a particle table would be too large), in which case the caller uses the grid.
    #[allow(clippy::type_complexity)]
    pub(crate) fn infer_particles(
        &self,
        clamp: &dyn Fn(usize) -> Option<usize>,
        n: usize,
    ) -> Option<(Vec<Vec<f32>>, HashMap<usize, Particles>)> {
        let n = n.clamp(1, MAX_PARTICLES);
        let cont: Vec<usize> = (0..self.vars.len())
            .filter(|&v| self.vars[v].range.is_some() && self.factors.iter().any(|f| f.vars.contains(&v)))
            .collect();
        if cont.is_empty() {
            return None;
        }
        let mut rng = StdRng::seed_from_u64(self.particle_seed);
        let grid = self.grid_prior(&clamp);
        let mut doms: BTreeMap<usize, Domain> =
            cont.iter().map(|&v| (v, self.initial_domain(v, clamp, n, &mut rng))).collect();
        let mut beliefs = Vec::new();
        for round in 0..ROUNDS {
            let prior: Vec<Vec<f32>> = (0..self.vars.len())
                .map(|v| match doms.get(&v) {
                    Some(d) if d.delta => vec![1.0],
                    Some(d) => d
                        .points
                        .iter()
                        .zip(&d.proposal)
                        .map(|(&x, &q)| self.prior_density(v, x, clamp) / q.max(1e-12))
                        .collect(),
                    None => grid[v].clone(),
                })
                .collect();
            let owned: Vec<Option<ArrayD<f32>>> = (0..self.factors.len())
                .map(|fi| {
                    if self.factors[fi].vars.iter().any(|v| doms.contains_key(v)) {
                        self.particle_table(fi, &doms).map(Some)
                    } else {
                        Some(None)
                    }
                })
                .collect::<Option<_>>()?;
            let tables: Vec<&ArrayD<f32>> =
                self.factors.iter().zip(&owned).map(|(f, t)| t.as_ref().unwrap_or(&f.table)).collect();
            let msgs = self.message_passing(&prior, &tables);
            beliefs = (0..self.vars.len()).map(|v| self.belief(v, &prior, &msgs, None)).collect();
            if round + 1 < ROUNDS {
                for (&v, d) in doms.iter_mut().filter(|(_, d)| !d.delta) {
                    let kde = Kde::new(&d.points, &beliefs[v], self.vars[v].range.expect("continuous"));
                    let points: Vec<f32> = (0..n).map(|_| kde.sample(&mut rng)).collect();
                    let proposal = points.iter().map(|&x| kde.density(x)).collect();
                    *d = Domain { points, proposal, delta: false };
                }
            }
        }
        let mut particles = HashMap::new();
        let regions = (0..self.vars.len())
            .map(|v| match doms.get(&v) {
                Some(d) => {
                    let mut hist = vec![0.0; self.vars[v].schema.domain_size];
                    d.points.iter().zip(&beliefs[v]).for_each(|(&x, w)| hist[self.region(v, x)] += w);
                    particles.insert(v, Particles { values: d.points.clone(), weights: beliefs[v].clone() });
                    normalize(hist)
                }
                None => beliefs[v].clone(),
            })
            .collect();
        Some((regions, particles))
    }

    fn region(&self, v: usize, x: f32) -> usize {
        let (lo, hi) = self.vars[v].range.expect("continuous");
        let bins = self.vars[v].schema.domain_size;
        (((x - lo) / (hi - lo) * bins as f32).floor().max(0.0) as usize).min(bins - 1)
    }

    /// First particles: the exact value, the evidenced region, the perceived particles or
    /// regions, or the whole range.
    fn initial_domain(&self, v: usize, clamp: &dyn Fn(usize) -> Option<usize>, n: usize, rng: &mut impl Rng) -> Domain {
        let (lo, hi) = self.vars[v].range.expect("continuous");
        let width = (hi - lo) / self.vars[v].schema.domain_size as f32;
        let in_region = |r: usize, rng: &mut StdRng| lo + (r as f32 + rng.gen::<f32>()) * width;
        let mut rng = StdRng::seed_from_u64(rng.gen());
        let evidence = self.vars[v].evidence;
        if let Some(r) = evidence.or_else(|| clamp(v)) {
            if let (Some(_), Some(&x)) = (evidence, self.exact.get(&v)) {
                return Domain { points: vec![x], proposal: vec![1.0], delta: true };
            }
            let points = (0..n).map(|_| in_region(r, &mut rng)).collect();
            return Domain { points, proposal: vec![1.0 / width; n], delta: false };
        }
        if let Some(p) = self.particle_percepts.get(&v) {
            let kde = Kde::new(&p.values, &p.weights, (lo, hi));
            let points: Vec<f32> = (0..n).map(|_| kde.sample(&mut rng)).collect();
            let proposal = points.iter().map(|&x| kde.density(x)).collect();
            return Domain { points, proposal, delta: false };
        }
        if let Some(dist) = self.percepts.get(&v) {
            let regions = Particles { values: (0..dist.len()).map(|r| r as f32).collect(), weights: dist.clone() };
            let points: Vec<f32> =
                regions.resample(n, &mut rng).values.iter().map(|&r| in_region(r as usize, &mut rng)).collect();
            let proposal = points.iter().map(|&x| dist[self.region(v, x)] / width).collect();
            return Domain { points, proposal, delta: false };
        }
        let points = (0..n).map(|_| lo + rng.gen::<f32>() * (hi - lo)).collect();
        Domain { points, proposal: vec![1.0 / (hi - lo); n], delta: false }
    }

    /// Density of what working memory says about `v` at `x`.
    fn prior_density(&self, v: usize, x: f32, clamp: &dyn Fn(usize) -> Option<usize>) -> f32 {
        let (lo, hi) = self.vars[v].range.expect("continuous");
        let width = (hi - lo) / self.vars[v].schema.domain_size as f32;
        if let Some(r) = self.vars[v].evidence.or_else(|| clamp(v)) {
            return if self.region(v, x) == r { 1.0 / width } else { 0.0 };
        }
        if let Some(p) = self.particle_percepts.get(&v) {
            return Kde::new(&p.values, &p.weights, (lo, hi)).density(x);
        }
        if let Some(dist) = self.percepts.get(&v) {
            return dist[self.region(v, x)] / width;
        }
        1.0 / (hi - lo)
    }

    /// A factor's table over the current particles: analytic functions at the exact values
    /// (scaled by any learning applied to their region), tables through their regions.
    fn particle_table(&self, fi: usize, doms: &BTreeMap<usize, Domain>) -> Option<ArrayD<f32>> {
        let f = &self.factors[fi];
        let dims: Vec<usize> =
            f.vars.iter().map(|v| doms.get(v).map_or(self.vars[*v].schema.domain_size, |d| d.points.len())).collect();
        dims.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).filter(|&n| n <= 4 * MAX_TABLE_LEN)?;
        let init = self.initial.get(&f.skill.skill_name)?;
        let shape = f.table.shape().to_vec();
        Some(ArrayD::from_shape_fn(IxDyn(&dims), |idx| {
            let point = |s: usize| doms.get(&f.vars[s]).map(|d| d.points[idx[s]]);
            let grid: Vec<usize> =
                (0..dims.len()).map(|s| point(s).map_or(idx[s], |x| self.region(f.vars[s], x))).collect();
            let live = f.table[IxDyn(&grid)];
            let value = match &f.analytic {
                None => live,
                Some(a) => {
                    let coords: Vec<f32> = (0..dims.len()).map(|s| point(s).unwrap_or(idx[s] as f32)).collect();
                    let flat = grid.iter().zip(&shape).fold(0, |acc, (&g, &d)| acc * d + g);
                    let i0 = init.get(f.weight_index(flat)).copied().unwrap_or(0.0);
                    if i0 > 0.0 {
                        a.eval(&coords) * live / i0
                    } else {
                        live
                    }
                }
            };
            value.clamp(0.0, MAX_FACTOR_VALUE)
        }))
    }
}
