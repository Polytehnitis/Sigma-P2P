//! Continuous arguments read and written between regions.
//!
//! A continuous argument grounds region by region (`Heat(0)`, `Heat(2)`, … over `[0, 10)` in
//! five regions; see [`crate::TypeDef::argument_values`]). These functions go beyond that
//! resolution by linear interpolation between region midpoints, as a piecewise-linear
//! function of the argument would: a query at `x = 3.1` reads the two regions whose midpoints
//! surround it, weighted by distance, and a reading at `x` is shared between them, each
//! trusted in proportion to its weight.
//!
//! Predicates over several continuous arguments (`Heat(x, y)`) interpolate multilinearly:
//! [`SigmaEngine::value_at_point`] reads the 2^k regions around a point, each weighted by the
//! product of its per-argument weights (bilinear for two arguments), and the `perceive_*_at_point`
//! functions share a reading among them in the same proportions.

use crate::SigmaEngine;

/// Readings whose interpolation weight on a region is below this are not given to it.
const MIN_SHARE: f32 = 0.05;

impl SigmaEngine {
    /// The ground atoms of `predicate` whose arguments are all numbers (regions of continuous
    /// arguments): each atom's variable and argument values.
    fn field_atoms(&self, predicate: &str) -> Vec<(usize, Vec<f32>)> {
        let prefix = format!("{predicate}(");
        self.vars
            .iter()
            .enumerate()
            .filter_map(|(v, n)| {
                let args = n.schema.name.strip_prefix(&prefix)?.strip_suffix(')')?;
                let point: Option<Vec<f32>> =
                    args.split(',').map(|a| a.trim().parse::<f32>().ok().filter(|x| x.is_finite())).collect();
                Some((v, point?))
            })
            .collect()
    }

    /// The atoms around `point`, with their multilinear interpolation weights (summing to 1).
    /// Along each argument the two region midpoints around the coordinate share it linearly;
    /// beyond the outer midpoints the outer region takes it all.
    fn bracket_point(&self, predicate: &str, point: &[f32]) -> Option<Vec<(usize, f32)>> {
        let atoms = self.field_atoms(predicate);
        let k = point.len();
        if k == 0 || point.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let atoms: Vec<&(usize, Vec<f32>)> = atoms.iter().filter(|a| a.1.len() == k).collect();
        // Per argument: the sorted region lower bounds, and their midpoints.
        let mut axes: Vec<Vec<f32>> = vec![Vec::new(); k];
        for (_, p) in &atoms {
            for (d, &x) in p.iter().enumerate() {
                axes[d].push(x);
            }
        }
        let mut per_axis: Vec<Vec<(f32, f32)>> = Vec::with_capacity(k);
        for (d, axis) in axes.iter_mut().enumerate() {
            axis.sort_by(f32::total_cmp);
            axis.dedup();
            if axis.len() < 2 {
                return None;
            }
            let width = axis[1] - axis[0];
            let mids: Vec<f32> = axis.iter().map(|a| a + width / 2.0).collect();
            let x = point[d];
            per_axis.push(if x <= mids[0] {
                vec![(axis[0], 1.0)]
            } else if x >= mids[mids.len() - 1] {
                vec![(axis[axis.len() - 1], 1.0)]
            } else {
                let i = mids.partition_point(|&m| m <= x) - 1;
                let t = (x - mids[i]) / (mids[i + 1] - mids[i]);
                vec![(axis[i], 1.0 - t), (axis[i + 1], t)]
            });
        }
        // Every corner: the product of its per-argument weights.
        let mut corners: Vec<(Vec<f32>, f32)> = vec![(Vec::new(), 1.0)];
        for options in &per_axis {
            corners = corners
                .into_iter()
                .flat_map(|(p, w)| {
                    options.iter().map(move |&(x, wx)| {
                        let mut q = p.clone();
                        q.push(x);
                        (q, w * wx)
                    })
                })
                .collect();
        }
        let mut out = Vec::new();
        for (corner, w) in corners {
            let (v, _) = atoms.iter().find(|a| a.1 == corner)?;
            out.push((*v, w));
        }
        Some(out)
    }

    /// The value of a field `predicate(x)` at any argument value `x`, between regions: the
    /// expected value of a continuous-valued predicate, or the probability of true of a
    /// boolean one, interpolated linearly between the surrounding regions' midpoints.
    pub fn value_at(&self, predicate: &str, x: f32) -> Option<f32> {
        self.value_at_point(predicate, &[x])
    }

    /// [`SigmaEngine::value_at`] for a predicate over several continuous arguments, at the
    /// argument values `point`: multilinear interpolation between the regions around it.
    pub fn value_at_point(&self, predicate: &str, point: &[f32]) -> Option<f32> {
        let around = self.bracket_point(predicate, point)?;
        let beliefs = self.infer(|_| None);
        around
            .iter()
            .map(|&(v, w)| {
                let name = &self.vars[v].schema.name;
                let value =
                    if self.vars[v].range.is_some() { self.expected_value(name)? } else { *beliefs[v].get(1)? };
                Some(w * value)
            })
            .sum()
    }

    /// A noisy reading `value ± std` of continuous-valued field `predicate` at argument value
    /// `x`, between regions: each surrounding region perceives it with the spread widened by
    /// how far `x` lies from its midpoint (`std / √weight`).
    pub fn perceive_value_at(&mut self, predicate: &str, x: f32, value: f32, std: f32) -> Result<(), &'static str> {
        self.perceive_value_at_point(predicate, &[x], value, std)
    }

    /// [`SigmaEngine::perceive_value_at`] at the argument values `point`.
    pub fn perceive_value_at_point(
        &mut self,
        predicate: &str,
        point: &[f32],
        value: f32,
        std: f32,
    ) -> Result<(), &'static str> {
        let around = self.bracket_point(predicate, point).ok_or("not a field over continuous arguments")?;
        for (v, w) in around.into_iter().filter(|a| a.1 >= MIN_SHARE) {
            let name = self.vars[v].schema.name.clone();
            self.perceive_value(&name, value, std / w.sqrt())?;
        }
        Ok(())
    }

    /// A percept `distribution` of discrete field `predicate` at argument value `x`, between
    /// regions: each surrounding region perceives it tempered by its weight (`p^weight`).
    pub fn perceive_at(&mut self, predicate: &str, x: f32, distribution: &[f32]) -> Result<(), &'static str> {
        self.perceive_at_point(predicate, &[x], distribution)
    }

    /// [`SigmaEngine::perceive_at`] at the argument values `point`.
    pub fn perceive_at_point(
        &mut self,
        predicate: &str,
        point: &[f32],
        distribution: &[f32],
    ) -> Result<(), &'static str> {
        let around = self.bracket_point(predicate, point).ok_or("not a field over continuous arguments")?;
        for (v, w) in around.into_iter().filter(|a| a.1 >= MIN_SHARE) {
            let name = self.vars[v].schema.name.clone();
            let tempered: Vec<f32> = distribution.iter().map(|p| p.max(0.0).powf(w)).collect();
            self.perceive(&name, &tempered)?;
        }
        Ok(())
    }
}
