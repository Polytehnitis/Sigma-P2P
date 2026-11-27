//! Operator selection, as in Sigma's decision phase: preferences over the candidate
//! operators (the node's actions), and how one is selected from their distribution.
//!
//! * **Operators.** The candidates for the deepest open (sub)goal form the operator
//!   distribution: each candidate's posterior of being taken, after preferences, normalized.
//!   `CycleOutcome::operators` reports it.
//! * **Preferences** (Soar and Sigma style), each optionally conditioned on working memory:
//!   *reject* removes a candidate, *best* restricts the choice to the best candidates, *worst*
//!   keeps a candidate only when nothing else is left, *better(a, b)* removes `b` while `a` is a
//!   candidate, and *indifferent* lets tied candidates be chosen between without a tie impasse.
//! * **Selection.** [`Selection::Best`] takes the most probable operator (ties impasse, as
//!   before). [`Selection::ProbabilityMatching`] samples an operator in proportion to its
//!   probability, and [`Selection::Boltzmann`] in proportion to `exp(p / temperature)`; a
//!   sampled choice never ties. Sampling is seeded (`selection_seed`), so a run is
//!   reproducible.

use crate::{SigmaEngine, TIE_MARGIN};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Most preferences an engine keeps.
pub const MAX_PREFERENCES: usize = 1024;

/// How an operator is selected from the operator distribution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Selection {
    /// The most probable operator; candidates within `TIE_MARGIN` impasse.
    #[default]
    Best,
    /// An operator sampled in proportion to its probability.
    ProbabilityMatching,
    /// An operator sampled in proportion to `exp(p / temperature)`.
    Boltzmann { temperature: f32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreferenceKind {
    /// Never select the operator.
    Reject,
    /// Select among the best operators only.
    Best,
    /// Select the operator only when nothing else is left.
    Worst,
    /// While this operator is a candidate, `other` is not selected.
    Better { other: String },
    /// Ties between this operator and `other` (or, with `None`, any other indifferent
    /// operator) are not an impasse: one is chosen by the selection policy.
    Indifferent { other: Option<String> },
}

/// A preference about one operator, holding when every `(predicate, value)` of `when` does
/// (a predicate in working memory with that value, or a goal on the stack with value 1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Preference {
    pub operator: String,
    pub kind: PreferenceKind,
    pub when: Vec<(String, usize)>,
}

impl Preference {
    pub fn new(operator: &str, kind: PreferenceKind) -> Self {
        Self { operator: operator.into(), kind, when: Vec::new() }
    }

    /// The same preference, holding only when `predicate` has `value`.
    pub fn when(mut self, predicate: &str, value: usize) -> Self {
        self.when.push((predicate.into(), value));
        self
    }
}

/// What the preferences made of the candidates.
pub(crate) struct Preferred {
    /// Candidates left, best first.
    pub ranked: Vec<(usize, f32)>,
    /// Pairs of candidates that may be chosen between without a tie impasse.
    pub indifferent: Vec<(usize, usize)>,
}

impl SigmaEngine {
    /// Adds a preference (replacing an identical one).
    pub fn add_preference(&mut self, p: Preference) -> Result<(), &'static str> {
        if self.preferences.len() >= MAX_PREFERENCES {
            return Err("too many preferences");
        }
        if !self.preferences.contains(&p) {
            self.preferences.push(p);
        }
        Ok(())
    }

    /// The preferences that hold now.
    fn active_preferences(&self) -> Vec<&Preference> {
        let wm = self.working_memory();
        self.preferences
            .iter()
            .filter(|p| {
                p.when
                    .iter()
                    .all(|(n, x)| wm.iter().any(|(m, y)| m == n && y == x) || (*x == 1 && self.goal_stack.contains(n)))
            })
            .collect()
    }

    /// Applies the preferences that hold to ranked candidates.
    pub(crate) fn apply_preferences(&self, ranked: &[(usize, f32)]) -> Preferred {
        let prefs = self.active_preferences();
        let name = |v: usize| self.vars[v].schema.name.as_str();
        let has =
            |v: usize, f: &dyn Fn(&PreferenceKind) -> bool| prefs.iter().any(|p| p.operator == name(v) && f(&p.kind));
        let mut left: Vec<(usize, f32)> =
            ranked.iter().copied().filter(|&(v, _)| !has(v, &|k| *k == PreferenceKind::Reject)).collect();
        // Better(a, b): b goes while a is a candidate.
        let names: Vec<&str> = left.iter().map(|&(v, _)| name(v)).collect();
        let dominated: Vec<&str> = prefs
            .iter()
            .filter(|p| names.contains(&p.operator.as_str()))
            .filter_map(|p| match &p.kind {
                PreferenceKind::Better { other } if *other != p.operator => Some(other.as_str()),
                _ => None,
            })
            .collect();
        left.retain(|&(v, _)| !dominated.contains(&name(v)));
        if left.iter().any(|&(v, _)| has(v, &|k| *k == PreferenceKind::Best)) {
            left.retain(|&(v, _)| has(v, &|k| *k == PreferenceKind::Best));
        }
        if left.iter().any(|&(v, _)| !has(v, &|k| *k == PreferenceKind::Worst)) {
            left.retain(|&(v, _)| !has(v, &|k| *k == PreferenceKind::Worst));
        }
        let indifferent_to = |a: usize, b: usize| {
            prefs.iter().any(|p| match &p.kind {
                PreferenceKind::Indifferent { other: Some(o) } => {
                    (p.operator == name(a) && o == name(b)) || (p.operator == name(b) && o == name(a))
                }
                _ => false,
            }) || (has(a, &|k| *k == PreferenceKind::Indifferent { other: None })
                && has(b, &|k| *k == PreferenceKind::Indifferent { other: None }))
        };
        let mut indifferent = Vec::new();
        for (i, &(a, _)) in left.iter().enumerate() {
            for &(b, _) in &left[i + 1..] {
                if indifferent_to(a, b) {
                    indifferent.push((a, b));
                }
            }
        }
        Preferred { ranked: left, indifferent }
    }

    /// The operator distribution over ranked candidates under the selection policy.
    pub(crate) fn operator_distribution(&self, ranked: &[(usize, f32)]) -> Vec<(usize, f32)> {
        let weights: Vec<f32> = match self.selection {
            Selection::Boltzmann { temperature } => {
                let t = if temperature.is_finite() && temperature > 0.0 { temperature } else { 1e-3 };
                let max = ranked.iter().map(|c| c.1).fold(f32::MIN, f32::max);
                ranked.iter().map(|c| ((c.1 - max) / t).exp()).collect()
            }
            _ => ranked.iter().map(|c| c.1).collect(),
        };
        let z: f32 = weights.iter().sum();
        ranked.iter().zip(weights).map(|(c, w)| (c.0, if z > 0.0 { w / z } else { 0.0 })).collect()
    }

    /// Chooses among `ranked` (non-empty) by the selection policy. `None` under `Best` when the
    /// leaders tie and are not all indifferent to each other (a tie impasse); the tied leaders
    /// are then returned in `Err`.
    pub(crate) fn select_operator(&mut self, p: &Preferred) -> Result<usize, Vec<usize>> {
        let ranked = &p.ranked;
        let tied: Vec<usize> = ranked.iter().take_while(|c| ranked[0].1 - c.1 < TIE_MARGIN).map(|c| c.0).collect();
        let all_indifferent = tied.iter().enumerate().all(|(i, &a)| {
            tied[i + 1..].iter().all(|&b| p.indifferent.contains(&(a, b)) || p.indifferent.contains(&(b, a)))
        });
        let pool: Vec<(usize, f32)> = match self.selection {
            Selection::Best if tied.len() == 1 => return Ok(tied[0]),
            Selection::Best if !all_indifferent => return Err(tied),
            // Indifferent leaders: chosen between at random.
            Selection::Best => tied.iter().map(|&v| (v, 1.0)).collect(),
            _ => self.operator_distribution(ranked),
        };
        self.decisions += 1;
        let mut rng = StdRng::seed_from_u64(self.selection_seed ^ self.decisions.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let z: f32 = pool.iter().map(|c| c.1).sum();
        let mut u = rng.gen::<f32>() * z;
        for &(v, w) in &pool {
            u -= w;
            if u < 0.0 {
                return Ok(v);
            }
        }
        Ok(pool.last().expect("non-empty").0)
    }
}
