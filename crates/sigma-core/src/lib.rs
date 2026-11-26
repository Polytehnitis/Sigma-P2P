//! sigma-core: a clean-room, declarative factor-graph cognitive engine.
//!
//! Skills are signed factor tables over typed predicates. Nothing in a skill is ever
//! executed: the engine runs sum-product message passing and *selects* one of the
//! node's own primitive actions (`PredicateRole::Action`).
//!
//! Truth convention: value `1` means "true" for a binary predicate. A `Goal` on the goal
//! stack is clamped to `1` (desired). A `Subgoal` names a state the goal needs; it is
//! satisfied once working memory holds it `1`, and until then it is assumed `0`
//! (closed world) when choosing an action.

pub mod crypto;
pub mod storage;

use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

pub use crypto::NodeIdentity;
pub use storage::FactorStore;

/// Largest factor value a skill may carry.
pub const MAX_FACTOR_VALUE: f32 = 100.0;
/// Largest factor table a skill may carry (entries).
pub const MAX_TABLE_LEN: usize = 1 << 16;
/// Posterior above which an unsatisfied subgoal is pushed onto the goal stack.
pub const SUBGOAL_PROMOTION: f32 = 0.65;
/// Authors and skills below this score are quarantined.
pub const QUARANTINE: f32 = 0.25;
const DEMO_WEIGHT: f32 = 10.0;
const MIN_WEIGHT: f32 = 1e-3;
const BP_ITERS: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PredicateRole {
    Context,
    Goal,
    Subgoal,
    Action,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateSchema {
    pub name: String,
    pub domain_size: usize,
    pub role: PredicateRole,
}

impl PredicateSchema {
    /// A boolean predicate (`domain_size == 2`).
    pub fn binary(name: &str, role: PredicateRole) -> Self {
        Self { name: name.into(), domain_size: 2, role }
    }
}

/// A purely declarative skill: a signed factor table over typed predicates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillPackage {
    pub skill_name: String,
    pub version: u64,
    pub author_pubkey: [u8; 32],
    pub signature: Vec<u8>,
    pub predicates: Vec<PredicateSchema>,
    pub factor_shape: Vec<usize>,
    pub factor_values: Vec<f32>,
}

impl SkillPackage {
    /// Structural, numerical and cryptographic checks. Must pass before compilation.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.predicates.is_empty() || self.factor_shape.len() != self.predicates.len() {
            return Err("factor rank does not match predicate count");
        }
        let len = self
            .factor_shape
            .iter()
            .try_fold(1usize, |acc, &d| if d < 2 { None } else { acc.checked_mul(d) })
            .filter(|&n| n <= MAX_TABLE_LEN)
            .ok_or("invalid or oversized factor shape")?;
        if len != self.factor_values.len() {
            return Err("factor length does not match shape");
        }
        if self.predicates.iter().zip(&self.factor_shape).any(|(p, &d)| p.domain_size != d) {
            return Err("factor shape does not match predicate domains");
        }
        let names: HashSet<_> = self.predicates.iter().map(|p| &p.name).collect();
        if names.len() != self.predicates.len() {
            return Err("duplicate predicate");
        }
        if !self.factor_values.iter().all(|v| v.is_finite() && (0.0..=MAX_FACTOR_VALUE).contains(v)) {
            return Err("factor value out of bounds");
        }
        NodeIdentity::verify_skill_signature(self)
    }

    /// Sorted predicate names, the key peers match impasse queries against.
    pub fn signature_key(&self) -> String {
        let mut names: Vec<&str> = self.predicates.iter().map(|p| p.name.as_str()).collect();
        names.sort_unstable();
        names.join(",")
    }
}

#[derive(Clone, Debug)]
pub struct VariableNode {
    pub schema: PredicateSchema,
    pub evidence: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct FactorNode {
    pub package: SkillPackage,
    pub vars: Vec<usize>,
    pub table: ArrayD<f32>,
    pub author_pubkey: [u8; 32],
    pub utility_score: f32,
}

/// A broken chain: `missing_predicate` (reached from `goal`) has no skill linking it to an action.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImpasseEvent {
    pub goal: String,
    pub missing_predicate: String,
    pub role: PredicateRole,
}

#[derive(Clone, Debug, Default)]
pub struct CycleOutcome {
    pub goal_stack: Vec<String>,
    pub selected_action: Option<String>,
    /// Posterior distribution of every predicate.
    pub beliefs: Vec<(String, Vec<f32>)>,
    /// `(skill_name, author_pubkey)` of every factor in the active chain.
    pub provenance: Vec<(String, [u8; 32])>,
    pub impasse: Option<ImpasseEvent>,
}

/// Factors and marginals of the last decision, kept for reinforcement.
#[derive(Debug, Default)]
struct Trace {
    factors: Vec<usize>,
    marginals: Vec<Vec<f32>>,
}

#[derive(Debug, Default)]
pub struct SigmaEngine {
    vars: Vec<VariableNode>,
    index: HashMap<String, usize>,
    factors: Vec<FactorNode>,
    pub goal_stack: Vec<String>,
    trace: Option<Trace>,
}

impl SigmaEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn factors(&self) -> &[FactorNode] {
        &self.factors
    }

    pub fn variables(&self) -> &[VariableNode] {
        &self.vars
    }

    /// Adds a predicate, or checks it agrees with the one already known.
    pub fn register_predicate(&mut self, p: &PredicateSchema) -> Result<usize, &'static str> {
        match self.index.get(&p.name) {
            Some(&v) if self.vars[v].schema != *p => Err("predicate schema conflict"),
            Some(&v) => Ok(v),
            None => {
                self.vars.push(VariableNode { schema: p.clone(), evidence: None });
                self.index.insert(p.name.clone(), self.vars.len() - 1);
                Ok(self.vars.len() - 1)
            }
        }
    }

    /// Verifies and links a skill into the factor graph, replacing older versions of it.
    pub fn compile_skill(&mut self, skill: SkillPackage) -> Result<(), &'static str> {
        skill.validate()?;
        if skill.predicates.iter().any(|p| self.index.get(&p.name).is_some_and(|&v| self.vars[v].schema != *p)) {
            return Err("predicate schema conflict");
        }
        if let Some(old) = self.factors.iter().find(|f| f.package.skill_name == skill.skill_name) {
            if old.package.version > skill.version {
                return Err("stale skill version");
            }
        }
        self.factors.retain(|f| f.package.skill_name != skill.skill_name);
        let vars = skill.predicates.iter().map(|p| self.register_predicate(p)).collect::<Result<_, _>>()?;
        let table = ArrayD::from_shape_vec(IxDyn(&skill.factor_shape), skill.factor_values.clone())
            .map_err(|_| "factor shape error")?;
        self.factors.push(FactorNode {
            author_pubkey: skill.author_pubkey,
            package: skill,
            vars,
            table,
            utility_score: 0.5,
        });
        self.trace = None;
        Ok(())
    }

    /// Replaces a compiled skill's live (learned) weights after bounds checks.
    pub fn apply_weights(&mut self, skill_name: &str, weights: &[f32]) -> bool {
        let Some(f) = self.factors.iter_mut().find(|f| f.package.skill_name == skill_name) else { return false };
        let ok = weights.len() == f.table.len()
            && weights.iter().all(|w| w.is_finite() && (0.0..=MAX_FACTOR_VALUE).contains(w));
        if ok {
            f.table.iter_mut().zip(weights).for_each(|(t, w)| *t = *w);
        }
        ok
    }

    pub fn skill_weights(&self, skill_name: &str) -> Option<Vec<f32>> {
        self.factor(skill_name).map(|f| f.table.iter().copied().collect())
    }

    pub fn set_evidence(&mut self, name: &str, value: usize) -> Result<(), &'static str> {
        let v = *self.index.get(name).ok_or("unknown predicate")?;
        if value >= self.vars[v].schema.domain_size {
            return Err("value outside predicate domain");
        }
        self.vars[v].evidence = Some(value);
        Ok(())
    }

    pub fn clear_evidence(&mut self, name: &str) {
        if let Some(&v) = self.index.get(name) {
            self.vars[v].evidence = None;
        }
    }

    /// Pushes a top-level goal (registering it as a binary `Goal` predicate if unknown).
    pub fn push_goal(&mut self, name: &str) {
        if !self.index.contains_key(name) {
            let _ = self.register_predicate(&PredicateSchema::binary(name, PredicateRole::Goal));
        }
        if !self.goal_stack.iter().any(|g| g == name) {
            self.goal_stack.push(name.into());
        }
    }

    /// Removes every goal/subgoal that working memory now holds true.
    pub fn pop_satisfied_goals(&mut self) -> Vec<String> {
        let (done, open) = std::mem::take(&mut self.goal_stack)
            .into_iter()
            .partition(|g| self.index.get(g).is_some_and(|&v| self.satisfied(v)));
        self.goal_stack = open;
        done
    }

    /// One perceive → infer → subgoal → select cycle.
    pub fn cognitive_cycle(&mut self) -> CycleOutcome {
        self.pop_satisfied_goals();
        let Some(root) = self.goal_stack.first().and_then(|g| self.index.get(g)).copied() else {
            self.trace = None;
            return CycleOutcome::default();
        };
        let stacked_goals: HashSet<usize> = self
            .goal_stack
            .iter()
            .filter_map(|g| self.index.get(g).copied())
            .filter(|&v| self.vars[v].schema.role == PredicateRole::Goal)
            .collect();

        // Phase 1: which subgoal states do the active goals call for?
        let p1 = self.infer(|v| if stacked_goals.contains(&v) { Some(1) } else { None });
        let (_, chain) = self.reachable(root);
        for v in chain {
            let s = &self.vars[v].schema;
            if s.role == PredicateRole::Subgoal
                && !self.satisfied(v)
                && p1[v].get(1).is_some_and(|&p| p > SUBGOAL_PROMOTION)
                && !self.goal_stack.contains(&s.name)
            {
                self.goal_stack.push(s.name.clone());
            }
        }

        // Phase 2: pick the prerequisite action for the deepest open (sub)goal.
        let p2 = self.infer(|v| match self.vars[v].schema.role {
            _ if stacked_goals.contains(&v) => Some(1),
            PredicateRole::Subgoal => Some(0),
            _ => None,
        });
        let top = self.index[self.goal_stack.last().expect("root is on the stack")];
        let (top_factors, candidates) = self.reachable(top);
        let selected = candidates
            .into_iter()
            .filter(|&v| self.vars[v].schema.role == PredicateRole::Action)
            .filter_map(|v| p2[v].get(1).map(|&p| (v, p)))
            .filter(|&(_, p)| p > 0.5)
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(v, _)| v);

        let mut active = self.reachable(root).0;
        for f in top_factors {
            if !active.contains(&f) {
                active.push(f);
            }
        }
        let mut marginals = p2.clone();
        for (v, m) in marginals.iter_mut().enumerate() {
            if self.vars[v].schema.role == PredicateRole::Action {
                *m = one_hot(usize::from(Some(v) == selected), m.len());
            }
        }
        let outcome = CycleOutcome {
            goal_stack: self.goal_stack.clone(),
            selected_action: selected.map(|v| self.vars[v].schema.name.clone()),
            beliefs: self
                .vars
                .iter()
                .enumerate()
                .map(|(v, n)| {
                    let p = if n.schema.role == PredicateRole::Subgoal { &p1[v] } else { &p2[v] };
                    (n.schema.name.clone(), p.clone())
                })
                .collect(),
            provenance: active
                .iter()
                .map(|&f| (self.factors[f].package.skill_name.clone(), self.factors[f].author_pubkey))
                .collect(),
            impasse: self.detect_structural_impasse(),
        };
        self.trace = Some(Trace { factors: active, marginals });
        outcome
    }

    /// Walks the chain from the root goal; reports the first open (sub)goal with no
    /// factor path onward to an `Action`.
    pub fn detect_structural_impasse(&self) -> Option<ImpasseEvent> {
        let goal = self.goal_stack.iter().find(|g| !self.index.get(*g).is_some_and(|&v| self.satisfied(v)))?;
        let missing = match self.index.get(goal) {
            None => {
                return Some(ImpasseEvent {
                    goal: goal.clone(),
                    missing_predicate: goal.clone(),
                    role: PredicateRole::Goal,
                })
            }
            Some(&root) => self.dead_end(root, None, &mut HashSet::new())?,
        };
        Some(ImpasseEvent {
            goal: goal.clone(),
            missing_predicate: self.vars[missing].schema.name.clone(),
            role: self.vars[missing].schema.role,
        })
    }

    fn dead_end(&self, v: usize, from: Option<usize>, seen: &mut HashSet<usize>) -> Option<usize> {
        let mut reaches_action = false;
        for fi in self.factors_of(v) {
            if Some(fi) == from {
                continue;
            }
            if !seen.insert(fi) {
                reaches_action = true; // already explored along another branch
                continue;
            }
            for &u in self.factors[fi].vars.iter().filter(|&&u| u != v) {
                match self.vars[u].schema.role {
                    PredicateRole::Context => {}
                    PredicateRole::Action => reaches_action = true,
                    _ if self.satisfied(u) => reaches_action = true,
                    _ => {
                        if let Some(dead) = self.dead_end(u, Some(fi), seen) {
                            return Some(dead);
                        }
                        reaches_action = true;
                    }
                }
            }
        }
        (!reaches_action).then_some(v)
    }

    /// Bounded multiplicative Hebbian update of every factor in the last decision's chain:
    /// each entry is scaled by `exp(reward · Π marginals)`. Returns per-factor author attributions.
    pub fn reinforce_factors(&mut self, reward: f32) -> Vec<([u8; 32], f32)> {
        let reward = if reward.is_finite() { reward.clamp(-1.0, 1.0) } else { 0.0 };
        let Some(trace) = self.trace.as_ref() else { return Vec::new() };
        let mut attributions = Vec::new();
        for &fi in &trace.factors {
            let f = &mut self.factors[fi];
            for (idx, w) in f.table.indexed_iter_mut() {
                let co: f32 = f.vars.iter().enumerate().map(|(j, &v)| trace.marginals[v][idx[j]]).product();
                *w = (*w * (reward * co).exp()).clamp(MIN_WEIGHT, MAX_FACTOR_VALUE);
            }
            f.utility_score = (f.utility_score + 0.25 * reward).clamp(0.0, 1.0);
            attributions.push((f.author_pubkey, reward));
        }
        attributions
    }

    /// Removes every skill matching `pred`, returning the removed packages.
    pub fn evict_skills(&mut self, pred: impl Fn(&FactorNode) -> bool) -> Vec<SkillPackage> {
        let (gone, keep) = std::mem::take(&mut self.factors).into_iter().partition(|f| pred(f));
        self.factors = keep;
        self.trace = None;
        gone.into_iter().map(|f: FactorNode| f.package).collect()
    }

    /// Turns a demonstrated assignment into a new signed skill and compiles it.
    pub fn learn_new_skill_from_experience(
        &mut self,
        identity: &NodeIdentity,
        skill_name: &str,
        predicates: Vec<PredicateSchema>,
        demonstration: &[usize],
    ) -> Result<SkillPackage, &'static str> {
        let factor_shape: Vec<usize> = predicates.iter().map(|p| p.domain_size).collect();
        if demonstration.len() != factor_shape.len() || demonstration.iter().zip(&factor_shape).any(|(x, d)| x >= d) {
            return Err("demonstration does not fit predicates");
        }
        let len = factor_shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).filter(|&n| n <= MAX_TABLE_LEN);
        let mut factor_values = vec![1.0; len.ok_or("factor too large")?];
        let flat = demonstration.iter().zip(&factor_shape).fold(0, |acc, (x, d)| acc * d + x);
        factor_values[flat] = DEMO_WEIGHT;
        let version = self.factor(skill_name).map_or(1, |f| f.package.version + 1);
        let mut skill = SkillPackage {
            skill_name: skill_name.into(),
            version,
            author_pubkey: [0; 32],
            signature: Vec::new(),
            predicates,
            factor_shape,
            factor_values,
        };
        identity.sign_skill(&mut skill);
        self.compile_skill(skill.clone())?;
        Ok(skill)
    }

    pub fn export_skill(&self, skill_name: &str) -> Option<SkillPackage> {
        self.factor(skill_name).map(|f| f.package.clone())
    }

    pub fn export_all_skills(&self) -> Vec<SkillPackage> {
        self.factors.iter().map(|f| f.package.clone()).collect()
    }

    fn factor(&self, skill_name: &str) -> Option<&FactorNode> {
        self.factors.iter().find(|f| f.package.skill_name == skill_name)
    }

    fn satisfied(&self, v: usize) -> bool {
        self.vars[v].evidence.is_some_and(|e| e > 0)
    }

    fn factors_of(&self, v: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.factors.len()).filter(move |&f| self.factors[f].vars.contains(&v))
    }

    /// Breadth-first factors and variables connected to `start` (variables in BFS order).
    fn reachable(&self, start: usize) -> (Vec<usize>, Vec<usize>) {
        let (mut fs, mut vs) = (Vec::new(), vec![start]);
        let mut queue = VecDeque::from([start]);
        while let Some(v) = queue.pop_front() {
            for f in self.factors_of(v) {
                if fs.contains(&f) {
                    continue;
                }
                fs.push(f);
                for &u in &self.factors[f].vars {
                    if !vs.contains(&u) {
                        vs.push(u);
                        queue.push_back(u);
                    }
                }
            }
        }
        (fs, vs)
    }

    /// Loopy sum-product. `clamp` adds assumptions on top of working-memory evidence.
    fn infer(&self, clamp: impl Fn(usize) -> Option<usize>) -> Vec<Vec<f32>> {
        let fixed: Vec<Option<usize>> =
            (0..self.vars.len()).map(|v| self.vars[v].evidence.or_else(|| clamp(v))).collect();
        let mut msgs: Vec<Vec<Vec<f32>>> = self
            .factors
            .iter()
            .map(|f| f.vars.iter().map(|&v| vec![1.0; self.vars[v].schema.domain_size]).collect())
            .collect();
        for _ in 0..BP_ITERS {
            msgs = self
                .factors
                .iter()
                .enumerate()
                .map(|(fi, f)| {
                    let inbound: Vec<Vec<f32>> =
                        f.vars.iter().enumerate().map(|(s, &v)| self.belief(v, &fixed, &msgs, Some((fi, s)))).collect();
                    (0..f.vars.len())
                        .map(|s| {
                            let mut out = vec![0.0; inbound[s].len()];
                            for (idx, &w) in f.table.indexed_iter() {
                                out[idx[s]] +=
                                    (0..inbound.len()).filter(|&j| j != s).fold(w, |acc, j| acc * inbound[j][idx[j]]);
                            }
                            normalize(out)
                        })
                        .collect()
                })
                .collect();
        }
        (0..self.vars.len()).map(|v| self.belief(v, &fixed, &msgs, None)).collect()
    }

    fn belief(
        &self,
        v: usize,
        fixed: &[Option<usize>],
        msgs: &[Vec<Vec<f32>>],
        skip: Option<(usize, usize)>,
    ) -> Vec<f32> {
        let d = self.vars[v].schema.domain_size;
        let mut b = fixed[v].map_or_else(|| vec![1.0; d], |x| one_hot(x, d));
        for (fi, f) in self.factors.iter().enumerate() {
            for (s, _) in f.vars.iter().enumerate().filter(|&(s, &u)| u == v && skip != Some((fi, s))) {
                b.iter_mut().zip(&msgs[fi][s]).for_each(|(x, m)| *x *= m);
            }
        }
        normalize(b)
    }
}

fn one_hot(x: usize, d: usize) -> Vec<f32> {
    (0..d).map(|i| if i == x { 1.0 } else { 0.0 }).collect()
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let sum: f32 = v.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        v.iter_mut().for_each(|x| *x /= sum);
    } else {
        let n = v.len() as f32;
        v.iter_mut().for_each(|x| *x = 1.0 / n);
    }
    v
}

#[cfg(test)]
mod tests;
