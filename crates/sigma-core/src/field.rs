//! Continuous arguments read and written between regions.
//!
//! A continuous argument grounds region by region (`Heat(0)`, `Heat(2)`, … over `[0, 10)` in
//! five regions; see [`crate::TypeDef::argument_values`]). These functions go beyond that
//! resolution by linear interpolation between region midpoints, as a piecewise-linear
//! function of the argument would: a query at `x = 3.1` reads the two regions whose midpoints
//! surround it, weighted by distance, and a reading at `x` is shared between them, each
//! trusted in proportion to its weight.

use crate::SigmaEngine;

/// Readings whose interpolation weight on a region is below this are not given to it.
const MIN_SHARE: f32 = 0.05;

impl SigmaEngine {
    /// The ground atoms of one-argument predicate `predicate` over a continuous argument,
    /// with each region's lower bound, sorted along the argument.
    fn field_atoms(&self, predicate: &str) -> Vec<(usize, f32)> {
        let prefix = format!("{predicate}(");
        let mut atoms: Vec<(usize, f32)> = self
            .vars
            .iter()
            .enumerate()
            .filter_map(|(v, n)| {
                let arg = n.schema.name.strip_prefix(&prefix)?.strip_suffix(')')?;
                (!arg.contains(',')).then_some(())?;
                Some((v, arg.parse::<f32>().ok().filter(|x| x.is_finite())?))
            })
            .collect();
        atoms.sort_by(|a, b| a.1.total_cmp(&b.1));
        atoms
    }

    /// The two atoms around argument value `x`, with their interpolation weights (one atom
    /// with weight 1 beyond the outer midpoints).
    fn bracket(&self, predicate: &str, x: f32) -> Option<Vec<(usize, f32)>> {
        let atoms = self.field_atoms(predicate);
        if atoms.len() < 2 || !x.is_finite() {
            return None;
        }
        let width = atoms[1].1 - atoms[0].1;
        let mids: Vec<f32> = atoms.iter().map(|a| a.1 + width / 2.0).collect();
        if x <= mids[0] {
            return Some(vec![(atoms[0].0, 1.0)]);
        }
        if x >= mids[mids.len() - 1] {
            return Some(vec![(atoms[atoms.len() - 1].0, 1.0)]);
        }
        let i = mids.partition_point(|&m| m <= x) - 1;
        let t = (x - mids[i]) / (mids[i + 1] - mids[i]);
        Some(vec![(atoms[i].0, 1.0 - t), (atoms[i + 1].0, t)])
    }

    /// The value of a field `predicate(x)` at any argument value `x`, between regions: the
    /// expected value of a continuous-valued predicate, or the probability of true of a
    /// boolean one, interpolated linearly between the surrounding regions' midpoints.
    pub fn value_at(&self, predicate: &str, x: f32) -> Option<f32> {
        let around = self.bracket(predicate, x)?;
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
        let around = self.bracket(predicate, x).ok_or("not a field over a continuous argument")?;
        for (v, w) in around.into_iter().filter(|a| a.1 >= MIN_SHARE) {
            let name = self.vars[v].schema.name.clone();
            self.perceive_value(&name, value, std / w.sqrt())?;
        }
        Ok(())
    }

    /// A percept `distribution` of discrete field `predicate` at argument value `x`, between
    /// regions: each surrounding region perceives it tempered by its weight (`p^weight`).
    pub fn perceive_at(&mut self, predicate: &str, x: f32, distribution: &[f32]) -> Result<(), &'static str> {
        let around = self.bracket(predicate, x).ok_or("not a field over a continuous argument")?;
        for (v, w) in around.into_iter().filter(|a| a.1 >= MIN_SHARE) {
            let name = self.vars[v].schema.name.clone();
            let tempered: Vec<f32> = distribution.iter().map(|p| p.max(0.0).powf(w)).collect();
            self.perceive(&name, &tempered)?;
        }
        Ok(())
    }
}
