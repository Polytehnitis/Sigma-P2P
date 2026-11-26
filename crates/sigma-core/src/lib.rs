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
pub mod memory;
pub mod program;
pub mod storage;

use ndarray::{ArrayD, Dimension, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub use crypto::NodeIdentity;
pub use memory::{Episode, EpisodicMemory, SemanticMemory, SemanticRecall};
pub use program::{
    Conditional, Function, Pattern, PatternKind, Piece, PredicateDef, Program, Skill, SkillBody, Term, TypeDef,
    TypeKind,
};
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
/// Message passing runs to quiescence (largest message change below `BP_TOLERANCE`),
/// for at most `BP_ITERS` sweeps.
const BP_ITERS: usize = 50;
const BP_TOLERANCE: f32 = 1e-5;
/// Candidate actions whose posteriors differ by less than this are tied.
pub const TIE_MARGIN: f32 = 0.02;

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
    /// `[lo, hi)` for a continuous value, split into `domain_size` equal regions.
    pub range: Option<(f32, f32)>,
}

/// One (ground) factor. A v1 skill compiles to one; a v2 skill to one per conditional binding.
#[derive(Clone, Debug)]
pub struct FactorNode {
    pub skill: Arc<Skill>,
    pub vars: Vec<usize>,
    /// Message direction of each slot (v1 factors are all `Condact`).
    pub kinds: Vec<PatternKind>,
    pub table: ArrayD<f32>,
    /// Where `table` sits in the skill's flat weights. Factors grounded from the same
    /// conditional slice share it, so their learned tables are kept equal.
    pub offset: usize,
    pub author_pubkey: [u8; 32],
    pub utility_score: f32,
    /// Gradient learning may change this factor (v1: always; v2: per conditional).
    pub learnable: bool,
    /// Slot the table is a conditional distribution over (see `Conditional::normal`).
    pub normal: Option<usize>,
}

/// A broken chain: `missing_predicate` (reached from `goal`) has no skill linking it to an action.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImpasseEvent {
    pub goal: String,
    pub missing_predicate: String,
    pub role: PredicateRole,
}

/// Sigma's decision impasses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpasseKind {
    /// No candidate action for the deepest open (sub)goal.
    None,
    /// Several candidates that the available knowledge does not tell apart.
    Tie,
    /// The same action was selected again while working memory did not change.
    NoChange,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionImpasse {
    pub kind: ImpasseKind,
    /// The (sub)goal on top of the stack when the impasse arose.
    pub goal: String,
    pub candidates: Vec<String>,
    /// True when reflective processing inside the impasse (one-step lookahead) settled it.
    pub resolved: bool,
    /// Consecutive cycles this kind of impasse has persisted.
    pub cycles: u32,
}

#[derive(Clone, Debug, Default)]
pub struct CycleOutcome {
    pub goal_stack: Vec<String>,
    pub selected_action: Option<String>,
    /// Posterior distribution of every predicate.
    pub beliefs: Vec<(String, Vec<f32>)>,
    /// `(skill_name, author_pubkey)` of every factor in the active chain.
    pub provenance: Vec<(String, [u8; 32])>,
    /// Structural impasse (missing skill), broadcast to peers.
    pub impasse: Option<ImpasseEvent>,
    /// Decision impasse (none, tie or no-change) from this cycle's decision phase.
    pub decision_impasse: Option<DecisionImpasse>,
}

/// A particle message over a continuous predicate: weighted samples of its value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Particles {
    pub values: Vec<f32>,
    pub weights: Vec<f32>,
}

impl Particles {
    pub fn mean(&self) -> f32 {
        let z: f32 = self.weights.iter().sum();
        self.values.iter().zip(&self.weights).map(|(x, w)| x * w).sum::<f32>() / z
    }
}

/// Factors and marginals of the last decision, kept for reinforcement.
#[derive(Debug, Default)]
struct Trace {
    factors: Vec<usize>,
    marginals: Vec<Vec<f32>>,
}

/// What the previous decision saw, for no-change detection.
#[derive(Debug)]
struct LastDecision {
    action: Option<usize>,
    wm: Vec<Option<usize>>,
    goals: Vec<String>,
    impasse: Option<(ImpasseKind, u32)>,
}

#[derive(Debug, Default)]
pub struct SigmaEngine {
    vars: Vec<VariableNode>,
    index: HashMap<String, usize>,
    factors: Vec<FactorNode>,
    skills: Vec<Arc<Skill>>,
    /// Initial flat weights of each skill.
    initial: HashMap<String, Vec<f32>>,
    pub goal_stack: Vec<String>,
    trace: Option<Trace>,
    last: Option<LastDecision>,
    /// Perception buffer: soft evidence that lasts one decision cycle.
    percepts: HashMap<usize, Vec<f32>>,
    /// Motor buffer: selected actions not yet taken by an actuator.
    motor: Vec<String>,
    pub episodic: EpisodicMemory,
    pub semantic: SemanticMemory,
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

    pub fn skills(&self) -> &[Arc<Skill>] {
        &self.skills
    }

    /// Adds a predicate, or checks it agrees with the one already known.
    pub fn register_predicate(&mut self, p: &PredicateSchema) -> Result<usize, &'static str> {
        match self.index.get(&p.name) {
            Some(&v) if self.vars[v].schema != *p => Err("predicate schema conflict"),
            Some(&v) => Ok(v),
            None => {
                self.vars.push(VariableNode { schema: p.clone(), evidence: None, range: None });
                self.index.insert(p.name.clone(), self.vars.len() - 1);
                Ok(self.vars.len() - 1)
            }
        }
    }

    /// Verifies, grounds and links a skill (v1 or v2) into the factor graph, replacing
    /// older versions of it.
    pub fn compile_skill(&mut self, skill: impl Into<Skill>) -> Result<(), &'static str> {
        let skill = skill.into();
        skill.validate()?;
        let grounding = skill.grounding()?;
        let conflict = |p: &PredicateSchema| self.index.get(&p.name).is_some_and(|&v| self.vars[v].schema != *p);
        let range_conflict = |(n, lo, hi): &(String, f32, f32)| {
            self.index.get(n).is_some_and(|&v| self.vars[v].range.is_some_and(|r| r != (*lo, *hi)))
        };
        if grounding.atoms.iter().chain(grounding.factors.iter().flat_map(|f| &f.predicates)).any(conflict)
            || grounding.ranges.iter().any(range_conflict)
        {
            return Err("predicate schema conflict");
        }
        if self.skill(&skill.skill_name).is_some_and(|old| old.version > skill.version) {
            return Err("stale skill version");
        }
        self.remove_skill(&skill.skill_name);
        for p in &grounding.atoms {
            self.register_predicate(p)?;
        }
        for (n, lo, hi) in &grounding.ranges {
            self.vars[self.index[n]].range = Some((*lo, *hi));
        }
        self.initial.insert(skill.skill_name.clone(), grounding.weights);
        let skill = Arc::new(skill);
        for g in grounding.factors {
            let vars = g.predicates.iter().map(|p| self.register_predicate(p)).collect::<Result<_, _>>()?;
            let shape: Vec<usize> = g.predicates.iter().map(|p| p.domain_size).collect();
            let table = ArrayD::from_shape_vec(IxDyn(&shape), g.values).map_err(|_| "factor shape error")?;
            self.factors.push(FactorNode {
                author_pubkey: skill.author_pubkey,
                skill: skill.clone(),
                vars,
                kinds: g.kinds,
                table,
                offset: g.offset,
                utility_score: 0.5,
                learnable: g.learnable,
                normal: g.normal,
            });
        }
        self.skills.push(skill);
        self.trace = None;
        Ok(())
    }

    /// Replaces a compiled skill's live (learned) flat weights after bounds checks.
    pub fn apply_weights(&mut self, skill_name: &str, weights: &[f32]) -> bool {
        let Some(init) = self.initial.get(skill_name) else { return false };
        let ok = weights.len() == init.len()
            && weights.iter().all(|w| w.is_finite() && (0.0..=MAX_FACTOR_VALUE).contains(w));
        if ok {
            for f in self.factors.iter_mut().filter(|f| f.skill.skill_name == skill_name) {
                let n = f.table.len();
                f.table.iter_mut().zip(&weights[f.offset..f.offset + n]).for_each(|(t, w)| *t = *w);
            }
        }
        ok
    }

    /// A skill's live flat weights (v2: every conditional function, concatenated).
    pub fn skill_weights(&self, skill_name: &str) -> Option<Vec<f32>> {
        let init = self.initial.get(skill_name)?.clone();
        let (mut sum, mut count) = (vec![0.0; init.len()], vec![0u32; init.len()]);
        for f in self.factors.iter().filter(|f| f.skill.skill_name == skill_name) {
            for (i, w) in f.table.iter().enumerate() {
                sum[f.offset + i] += w;
                count[f.offset + i] += 1;
            }
        }
        Some(init.into_iter().zip(sum).zip(count).map(|((w, s), c)| if c == 0 { w } else { s / c as f32 }).collect())
    }

    /// Re-ties factors grounded from the same function slice to their average.
    fn tie_parameters(&mut self, names: &HashSet<String>) {
        for name in names {
            if let Some(w) = self.skill_weights(name) {
                self.apply_weights(name, &w);
            }
        }
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

    /// Puts a (possibly uncertain) percept in the perception buffer for the next cycle.
    pub fn perceive(&mut self, name: &str, distribution: &[f32]) -> Result<(), &'static str> {
        let v = *self.index.get(name).ok_or("unknown predicate")?;
        if distribution.len() != self.vars[v].schema.domain_size
            || !distribution.iter().all(|p| p.is_finite() && *p >= 0.0)
            || distribution.iter().sum::<f32>() <= 0.0
        {
            return Err("invalid percept distribution");
        }
        self.percepts.insert(v, normalize(distribution.to_vec()));
        Ok(())
    }

    /// The region of continuous predicate `name` that contains `x` (clamped to the range).
    pub fn region_of(&self, name: &str, x: f32) -> Result<usize, &'static str> {
        let v = &self.vars[*self.index.get(name).ok_or("unknown predicate")?];
        let (lo, hi) = v.range.ok_or("not a continuous predicate")?;
        if !x.is_finite() {
            return Err("value is not finite");
        }
        let bins = v.schema.domain_size;
        Ok((((x - lo) / (hi - lo) * bins as f32).floor().max(0.0) as usize).min(bins - 1))
    }

    /// Sets a continuous predicate to the region containing `x`.
    pub fn set_value(&mut self, name: &str, x: f32) -> Result<(), &'static str> {
        let r = self.region_of(name, x)?;
        self.set_evidence(name, r)
    }

    /// Perceives a noisy continuous reading: a normal percept centred on `x`.
    pub fn perceive_value(&mut self, name: &str, x: f32, std: f32) -> Result<(), &'static str> {
        let centres = self.centres(name)?;
        if !(x.is_finite() && std.is_finite() && std > 0.0) {
            return Err("invalid reading");
        }
        let dist: Vec<f32> = centres.iter().map(|c| (-0.5 * ((c - x) / std).powi(2)).exp()).collect();
        self.perceive(name, &dist)
    }

    /// Perceives a particle message (weighted samples), projected onto the regions.
    pub fn perceive_particles(&mut self, name: &str, particles: &Particles) -> Result<(), &'static str> {
        let mut dist = vec![0.0; self.centres(name)?.len()];
        for (&x, &w) in particles.values.iter().zip(&particles.weights) {
            if w.is_finite() && w > 0.0 {
                dist[self.region_of(name, x)?] += w;
            }
        }
        self.perceive(name, &dist)
    }

    /// Posterior of `name` given working memory alone (no goal assumptions).
    pub fn posterior(&self, name: &str) -> Option<Vec<f32>> {
        let v = *self.index.get(name)?;
        Some(self.infer(|_| None).swap_remove(v))
    }

    /// Posterior mean of a continuous predicate.
    pub fn expected_value(&self, name: &str) -> Option<f32> {
        let centres = self.centres(name).ok()?;
        Some(self.posterior(name)?.iter().zip(&centres).map(|(p, c)| p * c).sum())
    }

    /// Draws a particle message from the posterior of a continuous predicate: a region by its
    /// probability, then a uniform point inside it. Equal weights.
    pub fn sample_particles(&self, name: &str, n: usize, rng: &mut impl rand::Rng) -> Option<Particles> {
        let v = &self.vars[*self.index.get(name)?];
        let (lo, hi) = v.range?;
        let post = self.posterior(name)?;
        let width = (hi - lo) / post.len() as f32;
        let values = (0..n)
            .map(|_| {
                let mut u: f32 = rng.gen();
                let r = post.iter().position(|p| {
                    u -= p;
                    u < 0.0
                });
                let r = r.unwrap_or(post.len() - 1);
                lo + (r as f32 + rng.gen::<f32>()) * width
            })
            .collect();
        Some(Particles { values, weights: vec![1.0 / n.max(1) as f32; n] })
    }

    fn centres(&self, name: &str) -> Result<Vec<f32>, &'static str> {
        let v = &self.vars[*self.index.get(name).ok_or("unknown predicate")?];
        let (lo, hi) = v.range.ok_or("not a continuous predicate")?;
        let w = (hi - lo) / v.schema.domain_size as f32;
        Ok((0..v.schema.domain_size).map(|i| lo + (i as f32 + 0.5) * w).collect())
    }

    /// Drains the motor buffer: every action selected since the last call, oldest first.
    pub fn take_motor_commands(&mut self) -> Vec<String> {
        std::mem::take(&mut self.motor)
    }

    /// Working memory as `(predicate, value)`: evidence, then the most likely value of each percept.
    pub fn working_memory(&self) -> Vec<(String, usize)> {
        let mut wm: Vec<(String, usize)> =
            self.vars.iter().filter_map(|n| n.evidence.map(|x| (n.schema.name.clone(), x))).collect();
        for (&v, p) in &self.percepts {
            if self.vars[v].evidence.is_none() {
                let x = p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |m| m.0);
                wm.push((self.vars[v].schema.name.clone(), x));
            }
        }
        wm.sort();
        wm
    }

    /// Semantic memory joins elaboration: features it predicts for the current object, for
    /// predicates without evidence or percepts, enter the perception buffer.
    fn complete_from_semantic_memory(&mut self) {
        let Some(recall) = self.semantic.recall(&self.working_memory()) else { return };
        for (feature, dist) in recall.predicted {
            if let Some(&v) = self.index.get(&feature) {
                if self.vars[v].evidence.is_none() && dist.len() == self.vars[v].schema.domain_size {
                    self.percepts.entry(v).or_insert(dist);
                }
            }
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

    /// One decision cycle: elaboration (message passing to quiescence, subgoaling), then
    /// decision (select an action, or reach a none / tie / no-change impasse).
    pub fn cognitive_cycle(&mut self) -> CycleOutcome {
        self.pop_satisfied_goals();
        let Some(root) = self.goal_stack.first().and_then(|g| self.index.get(g)).copied() else {
            self.trace = None;
            self.last = None;
            self.percepts.clear();
            return CycleOutcome::default();
        };
        self.complete_from_semantic_memory();
        let stacked_goals: HashSet<usize> = self
            .goal_stack
            .iter()
            .filter_map(|g| self.index.get(g).copied())
            .filter(|&v| self.vars[v].schema.role == PredicateRole::Goal)
            .collect();

        // Elaboration 1: which subgoal states do the active goals call for?
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

        // Elaboration 2: action posteriors for the deepest open (sub)goal.
        let p2 = self.infer(|v| match self.vars[v].schema.role {
            _ if stacked_goals.contains(&v) => Some(1),
            PredicateRole::Subgoal => Some(0),
            _ => None,
        });
        let top = self.index[self.goal_stack.last().expect("root is on the stack")];
        let (top_factors, candidates) = self.reachable(top);
        let mut ranked: Vec<(usize, f32)> = candidates
            .into_iter()
            .filter(|&v| self.vars[v].schema.role == PredicateRole::Action)
            .filter_map(|v| p2[v].get(1).map(|&p| (v, p)))
            .filter(|&(_, p)| p > 0.5)
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let tied: Vec<usize> = ranked.iter().take_while(|c| ranked[0].1 - c.1 < TIE_MARGIN).map(|c| c.0).collect();

        // Decision.
        let (selected, mut impasse) = match tied.len() {
            // None impasse: fall back on a rewarded action from a similar episode.
            0 => match self.recall_action() {
                Some(a) => (Some(a), Some((ImpasseKind::None, vec![a], true))),
                None => (None, Some((ImpasseKind::None, Vec::new(), false))),
            },
            1 => (Some(tied[0]), None),
            _ => {
                // Tie impasse: reflect by looking one step ahead with each candidate.
                let mut scored: Vec<(usize, f32)> =
                    tied.iter().map(|&a| (a, self.lookahead(a, top, &tied, &stacked_goals))).collect();
                scored.sort_by(|a, b| b.1.total_cmp(&a.1));
                let resolved = scored[0].1 - scored[1].1 >= TIE_MARGIN;
                (resolved.then_some(scored[0].0), Some((ImpasseKind::Tie, tied.clone(), resolved)))
            }
        };
        let wm: Vec<Option<usize>> = self.vars.iter().map(|n| n.evidence).collect();
        if let (None, Some(last)) = (&impasse, &self.last) {
            if selected.is_some() && last.action == selected && last.wm == wm && last.goals == self.goal_stack {
                impasse = Some((ImpasseKind::NoChange, selected.into_iter().collect(), false));
            }
        }
        let decision_impasse = impasse.map(|(kind, cands, resolved)| DecisionImpasse {
            kind,
            goal: self.vars[top].schema.name.clone(),
            candidates: cands.iter().map(|&v| self.vars[v].schema.name.clone()).collect(),
            resolved,
            cycles: match self.last.as_ref().and_then(|l| l.impasse) {
                Some((k, n)) if k == kind => n + 1,
                _ => 1,
            },
        });
        self.last = Some(LastDecision {
            action: selected,
            wm,
            goals: self.goal_stack.clone(),
            impasse: decision_impasse.as_ref().map(|d| (d.kind, d.cycles)),
        });

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
                .map(|&f| (self.factors[f].skill.skill_name.clone(), self.factors[f].author_pubkey))
                .collect(),
            impasse: self.detect_structural_impasse(),
            decision_impasse,
        };
        self.trace = Some(Trace { factors: active, marginals });
        self.episodic.record(self.goal_stack.clone(), self.working_memory(), outcome.selected_action.clone());
        self.motor.extend(outcome.selected_action.clone());
        self.percepts.clear();
        outcome
    }

    /// The action of the best-matching past episode that was rewarded, if it is still one of
    /// this node's actions.
    fn recall_action(&self) -> Option<usize> {
        let is_action = |a: &str| self.index.get(a).filter(|&&v| self.vars[v].schema.role == PredicateRole::Action);
        let ok = |e: &Episode| e.reward > 0.0 && e.action.as_deref().is_some_and(|a| is_action(a).is_some());
        let recall = self.episodic.recall(&self.goal_stack, &self.working_memory(), ok)?;
        is_action(recall.episode.action.as_deref()?).copied()
    }

    /// Reflection inside a tie impasse: how likely is `top` to hold if `action` is taken
    /// (and the other tied candidates are not)?
    fn lookahead(&self, action: usize, top: usize, tied: &[usize], stacked: &HashSet<usize>) -> f32 {
        let p = self.infer(|v| match self.vars[v].schema.role {
            _ if v == top => None,
            _ if tied.contains(&v) => Some(usize::from(v == action)),
            _ if stacked.contains(&v) => Some(1),
            PredicateRole::Subgoal => Some(0),
            _ => None,
        });
        p[top].get(1).copied().unwrap_or(0.0)
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
        self.episodic.reward_last(reward);
        let Some(trace) = self.trace.as_ref() else { return Vec::new() };
        let mut attributions = Vec::new();
        let mut touched = HashSet::new();
        for &fi in &trace.factors {
            let f = &mut self.factors[fi];
            for (idx, w) in f.table.indexed_iter_mut() {
                let co: f32 = f.vars.iter().enumerate().map(|(j, &v)| trace.marginals[v][idx[j]]).product();
                *w = (*w * (reward * co).exp()).clamp(MIN_WEIGHT, MAX_FACTOR_VALUE);
            }
            f.utility_score = (f.utility_score + 0.25 * reward).clamp(0.0, 1.0);
            attributions.push((f.author_pubkey, reward));
            touched.insert(f.skill.skill_name.clone());
        }
        self.tie_parameters(&touched);
        attributions
    }

    /// One step of Sigma-style gradient descent on the likelihood of working memory.
    ///
    /// As in Sigma, a learnable factor's gradient is the normalized product of its incoming
    /// messages. The (normalized) function moves `rate` along it and is renormalized: per
    /// configuration of the other slots when the factor has a `normal` slot (a conditional
    /// distribution), else over the whole table. Each group keeps its original total weight,
    /// so only the shape of the function is learned. With full evidence this converges to the
    /// observed frequencies. Returns the summed log local partition functions before the step
    /// (a likelihood proxy that rises as the functions fit the data).
    pub fn learn_gradient(&mut self, rate: f32) -> f32 {
        let rate = if rate.is_finite() { rate.clamp(0.0, 1.0) } else { 0.0 };
        let (prior, msgs) = self.propagate(|_| None);
        let mut log_z = 0.0;
        let mut touched = HashSet::new();
        for fi in 0..self.factors.len() {
            if !self.factors[fi].learnable {
                continue;
            }
            let inbound: Vec<Vec<f32>> = self.factors[fi]
                .vars
                .iter()
                .enumerate()
                .map(|(s, &v)| self.belief(v, &prior, &msgs, Some((fi, s))))
                .collect();
            let f = &mut self.factors[fi];
            let groups = |idx: &[usize]| -> usize {
                // Rows that are normalized together: all slots except `normal`, or one group.
                f.normal
                    .map_or(0, |n| (0..idx.len()).filter(|&j| j != n).fold(0, |a, j| a * f.table.shape()[j] + idx[j]))
            };
            let n_groups = f.normal.map_or(1, |n| f.table.len() / f.table.shape()[n]);
            let (mut mass, mut z, mut q_sum) = (vec![0.0f32; n_groups], 0.0f32, 0.0f32);
            let q: Vec<f32> = f
                .table
                .indexed_iter()
                .map(|(idx, &w)| {
                    let q: f32 = idx.slice().iter().enumerate().map(|(j, &x)| inbound[j][x]).product();
                    mass[groups(idx.slice())] += w;
                    z += w * q;
                    q_sum += q;
                    q
                })
                .collect();
            if z <= 0.0 || !z.is_finite() {
                continue;
            }
            log_z += z.ln();
            // Step on p = w / mass, then renormalize each group and restore its mass.
            let mut stepped: Vec<(usize, f32)> = f
                .table
                .indexed_iter()
                .zip(&q)
                .map(|((idx, &w), &q)| {
                    let g = groups(idx.slice());
                    (g, w / mass[g] + rate * q / q_sum)
                })
                .collect();
            let mut sums = vec![0.0f32; n_groups];
            stepped.iter().for_each(|&(g, p)| sums[g] += p);
            stepped.iter_mut().for_each(|(g, p)| *p = *p / sums[*g] * mass[*g]);
            f.table.iter_mut().zip(stepped).for_each(|(w, (_, p))| *w = p.clamp(MIN_WEIGHT, MAX_FACTOR_VALUE));
            touched.insert(f.skill.skill_name.clone());
        }
        self.tie_parameters(&touched);
        self.trace = None;
        log_z
    }

    /// Removes every skill with a factor matching `pred`, returning the removed skills.
    pub fn evict_skills(&mut self, pred: impl Fn(&FactorNode) -> bool) -> Vec<Skill> {
        let names: HashSet<String> =
            self.factors.iter().filter(|f| pred(f)).map(|f| f.skill.skill_name.clone()).collect();
        let gone = self.skills.iter().filter(|s| names.contains(&s.skill_name)).map(|s| (**s).clone()).collect();
        names.iter().for_each(|n| self.remove_skill(n));
        self.trace = None;
        gone
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
        let version = self.skill(skill_name).map_or(1, |s| s.version + 1);
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

    pub fn export_skill(&self, skill_name: &str) -> Option<Skill> {
        self.skill(skill_name).map(|s| (**s).clone())
    }

    pub fn export_all_skills(&self) -> Vec<Skill> {
        self.skills.iter().map(|s| (**s).clone()).collect()
    }

    fn skill(&self, skill_name: &str) -> Option<&Arc<Skill>> {
        self.skills.iter().find(|s| s.skill_name == skill_name)
    }

    fn remove_skill(&mut self, skill_name: &str) {
        self.factors.retain(|f| f.skill.skill_name != skill_name);
        self.skills.retain(|s| s.skill_name != skill_name);
        self.initial.remove(skill_name);
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

    /// Loopy sum-product to quiescence. `clamp` adds assumptions on top of working-memory
    /// evidence. A `Condition` slot never receives messages from its factor, and an `Action`
    /// slot never sends any into it.
    fn infer(&self, clamp: impl Fn(usize) -> Option<usize>) -> Vec<Vec<f32>> {
        let (prior, msgs) = self.propagate(clamp);
        (0..self.vars.len()).map(|v| self.belief(v, &prior, &msgs, None)).collect()
    }

    /// Runs message passing; returns each variable's prior and the final factor-to-variable messages.
    #[allow(clippy::type_complexity)]
    fn propagate(&self, clamp: impl Fn(usize) -> Option<usize>) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
        let fixed: Vec<Vec<f32>> = (0..self.vars.len())
            .map(|v| {
                let d = self.vars[v].schema.domain_size;
                match self.vars[v].evidence.or_else(|| clamp(v)) {
                    Some(x) => one_hot(x, d),
                    None => self.percepts.get(&v).cloned().unwrap_or_else(|| vec![1.0; d]),
                }
            })
            .collect();
        let uniform = |v: usize| normalize(vec![1.0; self.vars[v].schema.domain_size]);
        let mut msgs: Vec<Vec<Vec<f32>>> =
            self.factors.iter().map(|f| f.vars.iter().map(|&v| uniform(v)).collect()).collect();
        for _ in 0..BP_ITERS {
            let next: Vec<Vec<Vec<f32>>> = self
                .factors
                .iter()
                .enumerate()
                .map(|(fi, f)| {
                    let inbound: Vec<Vec<f32>> = f
                        .vars
                        .iter()
                        .enumerate()
                        .map(|(s, &v)| match f.kinds[s] {
                            PatternKind::Action => uniform(v),
                            _ => self.belief(v, &fixed, &msgs, Some((fi, s))),
                        })
                        .collect();
                    (0..f.vars.len())
                        .map(|s| {
                            if f.kinds[s] == PatternKind::Condition {
                                return uniform(f.vars[s]);
                            }
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
            let delta = next
                .iter()
                .flatten()
                .flatten()
                .zip(msgs.iter().flatten().flatten())
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            msgs = next;
            if delta < BP_TOLERANCE {
                break;
            }
        }
        (fixed, msgs)
    }

    fn belief(&self, v: usize, prior: &[Vec<f32>], msgs: &[Vec<Vec<f32>>], skip: Option<(usize, usize)>) -> Vec<f32> {
        let mut b = prior[v].clone();
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

pub(crate) fn normalize(mut v: Vec<f32>) -> Vec<f32> {
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
#[cfg(test)]
mod tests_continuous;
#[cfg(test)]
mod tests_learning;
#[cfg(test)]
mod tests_memory;
#[cfg(test)]
mod tests_v2;
