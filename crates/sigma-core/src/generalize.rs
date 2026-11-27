//! Learned models that generalize beyond the situations they were learned in.
//!
//! * **Linear utilities.** `Q(s, a)` is no longer only a table keyed by the exact state. It is
//!   the exact-state weight (what the table held) plus the weights of the state's *features*:
//!   a bias, each stacked goal (`goal:G`) and each value working memory holds (`P=x`). A
//!   temporal-difference backup moves the exact-state weight and the feature weights by equal
//!   halves of the same total step, so a state seen often converges as the table did, while a
//!   state never seen gets the utility its features earned elsewhere.
//! * **Context-conditioned transitions.** A transition model `P(p' | a, p)` misses effects
//!   that depend on another predicate (a switch that only works with the power on). The engine
//!   keeps the last [`EXPERIENCE_CAPACITY`] transitions it lived, and structure learning picks,
//!   for each action and predicate, the one other predicate whose value best predicts the
//!   next value (by likelihood gain over the plain model, less a BIC penalty). The model then
//!   becomes `P(p' | a, p, parent)`, a four-slot factor that keeps learning online.
//! * **Discovered perception models.** Pairs of predicates where one is sometimes hidden and,
//!   when both are known, strongly informative of each other (mutual information above
//!   [`PERCEPTION_MI`]) get a perception model `P(observation | state)` without being declared,
//!   initialized from the counts.
//!
//! Structure learning runs every [`STRUCTURE_INTERVAL`] transitions unless
//! `SigmaEngine::fixed_structure` is set, and on demand with
//! [`SigmaEngine::learn_model_structure`].

use crate::graph_memory::{
    internal, MAX_MODEL_FACTORS, MAX_PARENTS, MAX_PARENT_TABLE, MAX_PERCEPTION_MODELS, MAX_Q, MAX_Q_STATES,
    MODEL_SKILL, PERCEPTION_SKILL, Q_DISCOUNT, Q_RATE,
};
use crate::{PredicateRole, SigmaEngine};
use std::collections::HashMap;

/// Transitions kept for structure learning.
pub const EXPERIENCE_CAPACITY: usize = 256;
/// New transitions between automatic structure-learning passes.
pub const STRUCTURE_INTERVAL: usize = 8;
/// Fewest samples a structure decision is made from.
pub const MIN_STRUCTURE_SAMPLES: usize = 6;
/// Mutual information (nats) above which a perception model is discovered.
pub const PERCEPTION_MI: f32 = 0.2;
/// Fewest co-observations a perception model is discovered from.
pub const MIN_PERCEPTION_SAMPLES: usize = 8;
/// Largest domain structure learning considers.
const MAX_STRUCTURE_DOMAIN: usize = 8;
/// Share of a temporal-difference step that goes to the exact-state weight (the rest is split
/// over the state's features).
const EXACT_SHARE: f32 = 0.5;
/// Longest feature name kept.
const MAX_FEATURE: usize = 256;

/// One transition as lived: the action, and the evidence before and after it.
#[derive(Clone, Debug, PartialEq)]
pub struct Experience {
    pub action: usize,
    pub before: Vec<Option<usize>>,
    pub after: Vec<Option<usize>>,
}

/// What a structure-learning pass found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StructureReport {
    /// `(action, predicate, parent)` for every context-conditioned transition model now held.
    pub parents: Vec<(String, String, String)>,
    /// `(observation, state)` of perception models discovered in this pass.
    pub perceptions: Vec<(String, String)>,
}

impl SigmaEngine {
    /// The features of the current state that the linear part of a utility weighs: a bias
    /// (`""`), each stacked goal (`goal:G`) and each visible non-action value working memory
    /// holds (`P=x`).
    pub fn q_features(&self) -> Vec<String> {
        let mut out = vec![String::new()];
        out.extend(self.goal_stack.iter().filter(|g| !internal(g)).map(|g| format!("goal:{g}")));
        out.extend(
            self.working_memory()
                .into_iter()
                .filter(|(n, _)| {
                    !internal(n)
                        && self.index.get(n).is_some_and(|&v| self.vars[v].schema.role != PredicateRole::Action)
                })
                .map(|(n, x)| format!("{n}={x}")),
        );
        out.retain(|f| f.len() <= MAX_FEATURE);
        out
    }

    /// The backed-up utility `Q(s, a)` of action `action` in the current state.
    pub fn q_value(&self, action: &str) -> f32 {
        self.models.q_value(action, &self.state_key(), &self.q_features())
    }

    /// The backed-up utility of action variable `action` in the current state.
    pub(crate) fn utility(&self, action: usize) -> f32 {
        self.q_value(&self.vars[action].schema.name)
    }

    /// One temporal-difference backup of `Q(state, action)` after `reward`, now that the agent
    /// is in the state working memory describes (terminal when no goal is left). `features`
    /// are the features of the state the action was chosen in.
    pub(crate) fn back_up_utility(&mut self, state: &str, features: &[String], action: usize, reward: f32) {
        let an = self.vars[action].schema.name.clone();
        let future = if self.goal_stack.is_empty() {
            0.0
        } else {
            let (next, next_features) = (self.state_key(), self.q_features());
            (0..self.vars.len())
                .filter(|&b| self.vars[b].schema.role == PredicateRole::Action && !internal(&self.vars[b].schema.name))
                .map(|b| self.models.q_value(&self.vars[b].schema.name, &next, &next_features))
                .fold(0.0, f32::max)
        };
        let q = self.models.q_value(&an, state, features);
        let step = Q_RATE * (reward + Q_DISCOUNT * future - q);
        if step == 0.0 {
            return;
        }
        let per_action =
            |m: &std::collections::BTreeMap<(String, String), f32>| m.keys().filter(|(a, _)| *a == an).count();
        let clamp = |w: f32| w.clamp(-MAX_Q, MAX_Q);
        // The exact state takes half the step, the features share the other half.
        let key = (an.clone(), state.to_string());
        if self.models.utilities.contains_key(&key) || per_action(&self.models.utilities) < MAX_Q_STATES {
            let w = self.models.utilities.entry(key).or_insert(0.0);
            *w = clamp(*w + EXACT_SHARE * step);
        }
        let share = (1.0 - EXACT_SHARE) * step / features.len().max(1) as f32;
        for f in features {
            let key = (an.clone(), f.clone());
            if self.models.q_features.contains_key(&key) || per_action(&self.models.q_features) < MAX_Q_STATES {
                let w = self.models.q_features.entry(key).or_insert(0.0);
                *w = clamp(*w + share);
            }
        }
    }

    /// Keeps a lived transition for structure learning, and runs a pass every
    /// [`STRUCTURE_INTERVAL`] transitions unless `fixed_structure` is set.
    pub(crate) fn record_experience(&mut self, action: usize, before: &[Option<usize>]) {
        if self.experience.len() == EXPERIENCE_CAPACITY {
            self.experience.pop_front();
        }
        let after: Vec<Option<usize>> = self.vars.iter().map(|n| n.evidence).collect();
        self.experience.push_back(Experience { action, before: before.to_vec(), after });
        self.since_structure += 1;
        if !self.fixed_structure && self.since_structure >= STRUCTURE_INTERVAL {
            self.learn_model_structure();
        }
    }

    /// The transitions kept for structure learning, oldest first.
    pub fn experience(&self) -> impl Iterator<Item = &Experience> {
        self.experience.iter()
    }

    /// Predicates structure learning may use: visible, discrete, not actions, small domains.
    fn structure_candidates(&self) -> Vec<usize> {
        (0..self.vars.len())
            .filter(|&v| {
                let s = &self.vars[v].schema;
                s.role != PredicateRole::Action
                    && !internal(&s.name)
                    && !self.is_knowledge(&s.name)
                    && self.vars[v].range.is_none()
                    && s.domain_size <= MAX_STRUCTURE_DOMAIN
            })
            .collect()
    }

    /// Structure learning over the kept transitions: for each action and predicate, the parent
    /// that best predicts the predicate's next value (if any beats the plain model), and
    /// perception models for strongly coupled, sometimes hidden pairs. Rebuilds the model
    /// factors when anything changed.
    pub fn learn_model_structure(&mut self) -> StructureReport {
        self.since_structure = 0;
        self.save_models();
        let preds = self.structure_candidates();
        let actions: Vec<usize> = {
            let mut a: Vec<usize> = self.experience.iter().map(|e| e.action).collect();
            a.sort_unstable();
            a.dedup();
            a
        };
        let mut changed = false;
        for &a in &actions {
            let samples: Vec<&Experience> = self.experience.iter().filter(|e| e.action == a).collect();
            let an = self.vars[a].schema.name.clone();
            for &p in &preds {
                let key = (an.clone(), self.vars[p].schema.name.clone());
                let parents = self.select_parents(&samples, p, &preds);
                if parents.is_empty() {
                    // Enough evidence and no parent helps any more: back to the plain model.
                    let enough = samples
                        .iter()
                        .filter(|e| {
                            e.before.get(p).copied().flatten().is_some() && e.after.get(p).copied().flatten().is_some()
                        })
                        .count()
                        >= MIN_STRUCTURE_SAMPLES;
                    if enough {
                        changed |= self.models.contexts.remove(&key).is_some();
                        changed |= self.models.parent_sets.remove(&key).is_some();
                    }
                    continue;
                }
                let names: Vec<String> = parents.iter().map(|&q| self.vars[q].schema.name.clone()).collect();
                let held: Option<Vec<String>> =
                    match (self.models.contexts.get(&key), self.models.parent_sets.get(&key)) {
                        (Some((q, _)), _) => Some(vec![q.clone()]),
                        (_, Some((qs, _))) => Some(qs.clone()),
                        _ => None,
                    };
                if held.as_ref() == Some(&names) {
                    continue;
                }
                let total = self.models.contexts.len() + self.models.parent_sets.len();
                if held.is_none() && total >= MAX_MODEL_FACTORS {
                    continue;
                }
                let table = self.parent_table(&samples, p, &parents);
                self.models.contexts.remove(&key);
                self.models.parent_sets.remove(&key);
                if names.len() == 1 {
                    self.models.contexts.insert(key, (names[0].clone(), table));
                } else {
                    self.models.parent_sets.insert(key, (names, table));
                }
                changed = true;
            }
        }
        let perceptions = self.discover_perception_models(&preds);
        changed |= !perceptions.is_empty();
        if changed {
            // Drop the old model factors first, so the rebuild does not copy them back.
            self.factors.retain(|f| {
                !(f.memory && (f.skill.skill_name == MODEL_SKILL || f.skill.skill_name == PERCEPTION_SKILL))
            });
            self.rebuild_memory_factors();
        }
        let single = self.models.contexts.iter().map(|((a, p), (q, _))| (a.clone(), p.clone(), q.clone()));
        let several = self
            .models
            .parent_sets
            .iter()
            .flat_map(|((a, p), (qs, _))| qs.iter().map(move |q| (a.clone(), p.clone(), q.clone())));
        StructureReport { parents: single.chain(several).collect(), perceptions }
    }

    /// The parents of `p`'s transition under these samples, chosen greedily: each round adds
    /// the predicate whose gain in log-likelihood, over the samples where it and the parents so
    /// far are known, beats its BIC penalty by the most; it stops when none does, at
    /// [`MAX_PARENTS`], or when the table would grow past [`MAX_PARENT_TABLE`].
    fn select_parents(&self, samples: &[&Experience], p: usize, preds: &[usize]) -> Vec<usize> {
        let d = self.vars[p].schema.domain_size;
        let mut parents: Vec<usize> = Vec::new();
        while parents.len() < MAX_PARENTS {
            let size: usize = 2 * d * d * parents.iter().map(|&q| self.vars[q].schema.domain_size).product::<usize>();
            let best = preds
                .iter()
                .copied()
                .filter(|&q| q != p && !parents.contains(&q))
                .filter(|&q| size * self.vars[q].schema.domain_size <= MAX_PARENT_TABLE)
                .filter_map(|q| {
                    let mut with = parents.clone();
                    with.push(q);
                    let rows = self.parent_rows(samples, p, &with);
                    if rows.len() < MIN_STRUCTURE_SAMPLES {
                        return None;
                    }
                    let dq = self.vars[q].schema.domain_size;
                    // Rows keyed without q, then with it (q is the last, fastest-varying digit).
                    let before = log_likelihood(rows.iter().map(|(x, z, y)| (x * size + z / dq, *y)), d);
                    let after = log_likelihood(rows.iter().map(|(x, z, y)| (x * size * dq + z, *y)), d);
                    let configs = size / (2 * d * d) * d;
                    let penalty = 0.5 * (rows.len() as f32).ln() * (configs * (dq - 1) * (d - 1)) as f32;
                    let score = after - before - penalty;
                    (score > 0.0).then_some((q, score))
                })
                .max_by(|x, y| x.1.total_cmp(&y.1));
            match best {
                Some((q, _)) => parents.push(q),
                None => break,
            }
        }
        parents
    }

    /// `(p now, parents' joint value, p next)` for the samples where all are known; the
    /// parents' joint value is row-major over `parents`.
    fn parent_rows(&self, samples: &[&Experience], p: usize, parents: &[usize]) -> Vec<(usize, usize, usize)> {
        samples
            .iter()
            .filter_map(|e| {
                let x = e.before.get(p).copied()??;
                let y = e.after.get(p).copied()??;
                let mut z = 0;
                for &q in parents {
                    z = z * self.vars[q].schema.domain_size + e.before.get(q).copied()??;
                }
                Some((x, z, y))
            })
            .collect()
    }

    /// The transition table `[action, p now, parents…, p next]` estimated from the samples;
    /// only the row of the action taken is learned, the other stays uniform.
    fn parent_table(&self, samples: &[&Experience], p: usize, parents: &[usize]) -> Vec<f32> {
        let d = self.vars[p].schema.domain_size;
        let dz: usize = parents.iter().map(|&q| self.vars[q].schema.domain_size).product();
        let mut table = vec![1.0 / d as f32; 2 * d * dz * d];
        let mut counts = vec![0.5f32; d * dz * d];
        for (x, z, y) in self.parent_rows(samples, p, parents) {
            counts[(x * dz + z) * d + y] += 1.0;
        }
        for row in 0..d * dz {
            let total: f32 = counts[row * d..(row + 1) * d].iter().sum();
            for y in 0..d {
                table[d * dz * d + row * d + y] = counts[row * d + y] / total;
            }
        }
        table
    }

    /// Perception models for pairs of context predicates that are strongly coupled when both
    /// are known, where the state is sometimes hidden while the observation is seen.
    fn discover_perception_models(&mut self, preds: &[usize]) -> Vec<(String, String)> {
        let snapshots: Vec<&Vec<Option<usize>>> = self.experience.iter().map(|e| &e.after).collect();
        let context: Vec<usize> =
            preds.iter().copied().filter(|&v| self.vars[v].schema.role == PredicateRole::Context).collect();
        let mut found = Vec::new();
        for &s in &context {
            for &o in &context {
                if o == s || self.models.perceptions.len() >= MAX_PERCEPTION_MODELS {
                    continue;
                }
                let (on, sn) = (self.vars[o].schema.name.clone(), self.vars[s].schema.name.clone());
                if self.models.perceptions.contains_key(&(on.clone(), sn.clone()))
                    || self.models.perceptions.contains_key(&(sn.clone(), on.clone()))
                {
                    continue;
                }
                let at = |w: &Vec<Option<usize>>, v: usize| w.get(v).copied().flatten();
                let hidden = snapshots.iter().any(|w| at(w, o).is_some() && at(w, s).is_none());
                let pairs: Vec<(usize, usize)> =
                    snapshots.iter().filter_map(|w| Some((at(w, s)?, at(w, o)?))).collect();
                if !hidden || pairs.len() < MIN_PERCEPTION_SAMPLES {
                    continue;
                }
                let (ds, d_o) = (self.vars[s].schema.domain_size, self.vars[o].schema.domain_size);
                if mutual_information(&pairs, ds, d_o) < PERCEPTION_MI {
                    continue;
                }
                let mut counts = vec![0.5f32; ds * d_o];
                for &(x, y) in &pairs {
                    counts[x * d_o + y] += 1.0;
                }
                for x in 0..ds {
                    let total: f32 = counts[x * d_o..(x + 1) * d_o].iter().sum();
                    counts[x * d_o..(x + 1) * d_o].iter_mut().for_each(|c| *c /= total);
                }
                self.models.perceptions.insert((on.clone(), sn.clone()), counts);
                found.push((on, sn));
            }
        }
        found
    }
}

/// Log-likelihood of `(row, value)` samples under the row-wise frequencies (smoothed by ½).
fn log_likelihood(samples: impl Iterator<Item = (usize, usize)> + Clone, d: usize) -> f32 {
    let mut counts: HashMap<usize, Vec<f32>> = HashMap::new();
    for (r, y) in samples.clone() {
        counts.entry(r).or_insert_with(|| vec![0.5; d])[y] += 1.0;
    }
    samples
        .map(|(r, y)| {
            let c = &counts[&r];
            (c[y] / c.iter().sum::<f32>()).ln()
        })
        .sum()
}

/// Mutual information (nats) between the two coordinates of `pairs`.
fn mutual_information(pairs: &[(usize, usize)], da: usize, db: usize) -> f32 {
    let n = pairs.len() as f32;
    let mut joint = vec![0.0f32; da * db];
    for &(a, b) in pairs {
        joint[a * db + b] += 1.0 / n;
    }
    let pa: Vec<f32> = (0..da).map(|a| (0..db).map(|b| joint[a * db + b]).sum()).collect();
    let pb: Vec<f32> = (0..db).map(|b| (0..da).map(|a| joint[a * db + b]).sum()).collect();
    let mut mi = 0.0;
    for a in 0..da {
        for b in 0..db {
            let p = joint[a * db + b];
            if p > 0.0 {
                mi += p * (p / (pa[a] * pb[b])).ln();
            }
        }
    }
    mi
}
