//! Memories and learned models inside the factor graph, as in Sigma.
//!
//! The episodic and semantic stores ([`EpisodicMemory`], [`SemanticMemory`]) stay the record
//! that is persisted and synced between devices, but recall no longer happens beside the
//! graph: at the start of every cycle they are compiled into factors, and recall is message
//! passing like everything else.
//!
//! * **Episodic memory.** A hidden variable `@Episode` ranges over the most recent
//!   [`EPISODE_SLOTS`] episodes, with a recency prior. A cue factor links it to each predicate
//!   in working memory and to each stacked goal (read only, so a memory never overwrites what
//!   is perceived), and `@Rewarded` says whether the episode's decision was rewarded. Recall
//!   is the posterior over `@Episode` given the cue.
//! * **Semantic memory.** A hidden variable `@Category` over the learned concepts, with a prior
//!   from their counts and one naive-Bayes factor per feature, both ways: features suggest a
//!   category and the category predicts unobserved features.
//! * **Action models.** For each of the node's actions and each goal or subgoal, a factor
//!   `P(@Next(p|a) | a)` predicts the predicate's value after the action. It is learned from
//!   what is observed in the cycle after the action, with the same gradient rule as every
//!   other function, and reasoning in substates uses it to imagine actions.
//! * **Reward models.** For each action, `P(@Reward(a) | a)` is learned from the reward given
//!   for it (reinforcement learning), and reasoning in substates prefers rewarded actions.
//!
//! Internal predicates start with `@`. Skills may not declare such names, and they are left
//! out of cycle outcomes and of other agents' perspectives.

use crate::memory::{Episode, NOISE, RECENCY};
use crate::{
    normalize, FactorNode, PatternKind, PredicateRole, PredicateSchema, Program, SigmaEngine, Skill, MAX_FACTOR_VALUE,
};
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Most recent episodes that `@Episode` ranges over.
pub const EPISODE_SLOTS: usize = 256;
/// Most concepts that `@Category` ranges over (the most often seen).
pub const CATEGORY_SLOTS: usize = 64;
/// Most action and reward model factors.
pub const MAX_MODEL_FACTORS: usize = 512;
/// Largest predicate domain an action model is learned for.
const MAX_MODEL_DOMAIN: usize = 16;
/// Learning rate of action and reward models.
pub const MODEL_RATE: f32 = 0.3;
/// Weight of the learned reward model in a substate's evaluation of a candidate.
pub const REWARD_WEIGHT: f32 = 0.25;

pub const EPISODE: &str = "@Episode";
pub const REWARDED: &str = "@Rewarded";
pub const CATEGORY: &str = "@Category";
const EPISODIC_SKILL: &str = "@episodic";
const SEMANTIC_SKILL: &str = "@semantic";
const MODEL_SKILL: &str = "@models";

/// The predicate predicting `p`'s value after action `a`.
pub fn next_name(p: &str, a: &str) -> String {
    format!("@Next({p}|{a})")
}

/// The predicate predicting the reward for action `a`.
pub fn reward_name(a: &str) -> String {
    format!("@Reward({a})")
}

/// Whether `name` is an internal predicate.
pub fn internal(name: &str) -> bool {
    name.starts_with('@')
}

/// Learned action and reward models: conditional distributions, row-major over
/// `(action ∈ {0, 1}, value)`. Kept on the engine (and persisted with the memories) so that
/// learning carries over when the graph is rebuilt.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LearnedModels {
    /// `(action, predicate)` -> `P(predicate after | action)`, 2 × domain.
    pub effects: BTreeMap<(String, String), Vec<f32>>,
    /// `action` -> `P(rewarded after | action)`, 2 × 2.
    pub rewards: BTreeMap<String, Vec<f32>>,
}

/// Factors of one internal skill being built.
struct Builder {
    skill: Arc<Skill>,
    weights: Vec<f32>,
    factors: Vec<FactorNode>,
}

impl Builder {
    fn new(name: &str) -> Self {
        Builder {
            skill: Arc::new(Skill::program(name, 0, Program::default())),
            weights: Vec::new(),
            factors: Vec::new(),
        }
    }

    fn add(
        &mut self,
        vars: Vec<usize>,
        kinds: Vec<PatternKind>,
        shape: &[usize],
        values: Vec<f32>,
        normal: Option<usize>,
    ) {
        let values: Vec<f32> =
            values.into_iter().map(|v| if v.is_finite() { v.clamp(0.0, MAX_FACTOR_VALUE) } else { 0.0 }).collect();
        let table = ArrayD::from_shape_vec(IxDyn(shape), values.clone()).expect("shape matches values");
        self.factors.push(FactorNode {
            skill: self.skill.clone(),
            vars,
            kinds,
            table,
            offset: self.weights.len(),
            author_pubkey: [0; 32],
            utility_score: 0.5,
            learnable: false,
            normal,
            analytic: None,
            memory: true,
        });
        self.weights.extend(values);
    }
}

impl SigmaEngine {
    /// Recompiles the memory and model factors from the stores (run at the start of every
    /// cycle, so recall sees every episode recorded so far).
    pub fn rebuild_memory_factors(&mut self) {
        self.save_models();
        self.factors.retain(|f| !f.memory);
        for name in [EPISODIC_SKILL, SEMANTIC_SKILL, MODEL_SKILL] {
            self.initial.remove(name);
        }
        let builders = [self.episodic_factors(), self.semantic_factors(), self.model_factors()];
        for b in builders.into_iter().flatten() {
            self.initial.insert(b.skill.skill_name.clone(), b.weights);
            self.factors.extend(b.factors);
        }
        self.trace = None;
    }

    fn internal_var(&mut self, name: &str, domain: usize) -> Option<usize> {
        self.register_predicate(&PredicateSchema {
            name: name.into(),
            domain_size: domain,
            role: PredicateRole::Context,
        })
        .ok()
    }

    /// Visible (non-internal) predicates, by name.
    fn visible(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied().filter(|_| !internal(name))
    }

    /// The episodes `@Episode` ranges over, oldest first.
    fn episode_window(&self) -> Vec<&Episode> {
        let n = self.episodic.episodes.len();
        self.episodic.episodes.iter().skip(n.saturating_sub(EPISODE_SLOTS)).collect()
    }

    fn episodic_factors(&mut self) -> Option<Builder> {
        let window: Vec<Episode> = self.episode_window().into_iter().cloned().collect();
        if window.is_empty() {
            return None;
        }
        let ep = self.internal_var(EPISODE, EPISODE_SLOTS)?;
        let rewarded = self.internal_var(REWARDED, 2)?;
        let clock = window.last().map_or(0, |e| e.time);
        let mut b = Builder::new(EPISODIC_SKILL);
        // Recency prior; empty slots get no mass.
        let prior: Vec<f32> = (0..EPISODE_SLOTS)
            .map(|i| window.get(i).map_or(0.0, |e| (-RECENCY * clock.saturating_sub(e.time) as f32).exp()))
            .collect();
        b.add(vec![ep], vec![PatternKind::Condact], &[EPISODE_SLOTS], prior, None);
        // `@Rewarded`: was the episode's decision rewarded?
        let values = (0..EPISODE_SLOTS)
            .flat_map(|i| {
                let p = match window.get(i) {
                    Some(e) if e.reward > 0.0 => 1.0 - NOISE,
                    Some(_) => NOISE,
                    None => 0.5,
                };
                [1.0 - p, p]
            })
            .collect();
        b.add(
            vec![ep, rewarded],
            vec![PatternKind::Condact, PatternKind::Condact],
            &[EPISODE_SLOTS, 2],
            values,
            Some(1),
        );
        // Cue factors over working memory (read only).
        for (name, _) in self.working_memory() {
            let Some(v) = self.visible(&name) else { continue };
            let d = self.vars[v].schema.domain_size;
            let values = (0..EPISODE_SLOTS)
                .flat_map(|i| {
                    let stored = window.get(i).and_then(|e| e.state.iter().find(|(n, _)| *n == name).map(|kv| kv.1));
                    (0..d).map(move |x| match stored {
                        Some(y) if y == x => 1.0 - NOISE,
                        Some(_) => NOISE / (d - 1).max(1) as f32,
                        None => 1.0 / d as f32,
                    })
                })
                .collect();
            b.add(
                vec![ep, v],
                vec![PatternKind::Condact, PatternKind::Condition],
                &[EPISODE_SLOTS, d],
                values,
                Some(1),
            );
        }
        // Cue factors over the stacked goals (read only).
        for g in self.goal_stack.clone() {
            let Some(v) = self.visible(&g) else { continue };
            if self.vars[v].schema.domain_size != 2 {
                continue;
            }
            let values = (0..EPISODE_SLOTS)
                .flat_map(|i| {
                    let p = match window.get(i) {
                        Some(e) if e.goals.contains(&g) => 1.0 - NOISE,
                        Some(_) => NOISE,
                        None => 0.5,
                    };
                    [1.0 - p, p]
                })
                .collect();
            b.add(
                vec![ep, v],
                vec![PatternKind::Condact, PatternKind::Condition],
                &[EPISODE_SLOTS, 2],
                values,
                Some(1),
            );
        }
        Some(b)
    }

    fn semantic_factors(&mut self) -> Option<Builder> {
        let mut concepts = self.semantic.concepts();
        if concepts.is_empty() {
            return None;
        }
        concepts.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        concepts.truncate(CATEGORY_SLOTS);
        let cat = self.internal_var(CATEGORY, CATEGORY_SLOTS)?;
        let mut b = Builder::new(SEMANTIC_SKILL);
        let total: f32 = concepts.iter().map(|c| c.1).sum();
        let prior = (0..CATEGORY_SLOTS).map(|i| concepts.get(i).map_or(0.0, |c| c.1 / total)).collect();
        b.add(vec![cat], vec![PatternKind::Condact], &[CATEGORY_SLOTS], prior, None);
        for feature in self.semantic.features() {
            let Some(v) = self.visible(&feature) else { continue };
            let d = self.vars[v].schema.domain_size;
            let values = (0..CATEGORY_SLOTS)
                .flat_map(|i| {
                    let row: Vec<f32> = match concepts.get(i) {
                        Some((name, _)) => (0..d).map(|x| self.semantic.likelihood_of(name, &feature, x, d)).collect(),
                        None => vec![1.0 / d as f32; d],
                    };
                    row
                })
                .collect();
            b.add(
                vec![cat, v],
                vec![PatternKind::Condact, PatternKind::Condact],
                &[CATEGORY_SLOTS, d],
                values,
                Some(1),
            );
        }
        self.category_names = concepts.into_iter().map(|c| c.0).collect();
        Some(b)
    }

    fn model_factors(&mut self) -> Option<Builder> {
        let actions: Vec<usize> = (0..self.vars.len())
            .filter(|&v| self.vars[v].schema.role == PredicateRole::Action && !internal(&self.vars[v].schema.name))
            .collect();
        if actions.is_empty() {
            return None;
        }
        let modeled: Vec<usize> = (0..self.vars.len())
            .filter(|&v| {
                let s = &self.vars[v].schema;
                matches!(s.role, PredicateRole::Goal | PredicateRole::Subgoal)
                    && !internal(&s.name)
                    && self.vars[v].range.is_none()
                    && s.domain_size <= MAX_MODEL_DOMAIN
            })
            .collect();
        let mut b = Builder::new(MODEL_SKILL);
        for &a in &actions {
            let an = self.vars[a].schema.name.clone();
            if b.factors.len() >= MAX_MODEL_FACTORS {
                break;
            }
            let r = self.internal_var(&reward_name(&an), 2)?;
            let values = self.models.rewards.get(&an).cloned().unwrap_or_else(|| vec![0.5; 4]);
            b.add(vec![a, r], vec![PatternKind::Condition, PatternKind::Condact], &[2, 2], values, Some(1));
            for &p in &modeled {
                if b.factors.len() >= MAX_MODEL_FACTORS {
                    break;
                }
                let (pn, d) = (self.vars[p].schema.name.clone(), self.vars[p].schema.domain_size);
                let next = self.internal_var(&next_name(&pn, &an), d)?;
                let values = match self.models.effects.get(&(an.clone(), pn.clone())) {
                    Some(t) if t.len() == 2 * d => t.clone(),
                    _ => vec![1.0 / d as f32; 2 * d],
                };
                b.add(vec![a, next], vec![PatternKind::Condition, PatternKind::Condact], &[2, d], values, Some(1));
            }
        }
        Some(b)
    }

    /// Copies the model factors' learned tables back into [`LearnedModels`].
    fn save_models(&mut self) {
        let mut models = std::mem::take(&mut self.models);
        for f in self.factors.iter().filter(|f| f.memory && f.skill.skill_name == MODEL_SKILL) {
            let a = self.vars[f.vars[0]].schema.name.clone();
            let target = &self.vars[f.vars[1]].schema.name;
            let table: Vec<f32> = f.table.iter().copied().collect();
            if target.starts_with("@Reward(") {
                models.rewards.insert(a, table);
            } else if let Some(p) = target.strip_prefix("@Next(").and_then(|t| t.split('|').next()) {
                models.effects.insert((a, p.to_string()), table);
            }
        }
        self.models = models;
    }

    /// Learns the action models from what working memory holds now, after `action` was taken
    /// last cycle: for that action the row `action = 1`, for the others the row `action = 0`.
    pub(crate) fn learn_action_models(&mut self, action: usize) {
        let rate = MODEL_RATE;
        for fi in 0..self.factors.len() {
            let f = &self.factors[fi];
            if !(f.memory && f.skill.skill_name == MODEL_SKILL) {
                continue;
            }
            let target = self.vars[f.vars[1]].schema.name.clone();
            let Some(p) = target.strip_prefix("@Next(").and_then(|t| t.split('|').next()) else { continue };
            let Some(x) = self.index.get(p).and_then(|&v| self.vars[v].evidence) else { continue };
            let taken = usize::from(f.vars[0] == action);
            let d = f.table.shape()[1];
            let inbound = vec![crate::one_hot(taken, 2), crate::one_hot(x, d)];
            crate::gradient_step(&mut self.factors[fi], &inbound, rate);
        }
        self.save_models();
    }

    /// Learns the reward model of `action` from a reward given for it.
    pub(crate) fn learn_reward_model(&mut self, action: usize, reward: f32) {
        if reward == 0.0 {
            return;
        }
        let name = reward_name(&self.vars[action].schema.name);
        for fi in 0..self.factors.len() {
            let f = &self.factors[fi];
            if f.memory && f.vars[0] == action && self.vars[f.vars[1]].schema.name == name {
                let inbound = vec![crate::one_hot(1, 2), crate::one_hot(usize::from(reward > 0.0), 2)];
                crate::gradient_step(&mut self.factors[fi], &inbound, MODEL_RATE * reward.abs());
            }
        }
        self.save_models();
    }

    /// The learned chance that taking `action` is rewarded, minus the chance when it is not
    /// taken (0 until rewards were given). Read from the reward model's factor.
    pub(crate) fn reward_advantage(&self, action: usize) -> f32 {
        let name = reward_name(&self.vars[action].schema.name);
        self.factors
            .iter()
            .find(|f| f.memory && f.vars[0] == action && self.vars[f.vars[1]].schema.name == name)
            .map_or(0.0, |f| {
                let t: Vec<f32> = f.table.iter().copied().collect();
                t[3] / (t[2] + t[3]).max(1e-9) - t[1] / (t[0] + t[1]).max(1e-9)
            })
    }

    /// Recalls the episode that best explains working memory and the stacked goals, by message
    /// passing over the memory factors: the most probable value of `@Episode`, and its posterior.
    /// With `rewarded`, the recall is cued with `@Rewarded = 1` too.
    pub fn recall_episode(&self, rewarded: bool) -> Option<(Episode, f32)> {
        let ep = *self.index.get(EPISODE)?;
        let rw = self.index.get(REWARDED).copied();
        let window = self.episode_window();
        if window.is_empty() {
            return None;
        }
        let stacked: Vec<usize> = self.goal_stack.iter().filter_map(|g| self.index.get(g).copied()).collect();
        let p = self.infer(|v| if stacked.contains(&v) || (rewarded && Some(v) == rw) { Some(1) } else { None });
        let post = normalize(p[ep].iter().take(window.len()).copied().collect());
        // Most recent wins ties, so scan newest first.
        let (i, &best) = post.iter().enumerate().rev().max_by(|a, b| a.1.total_cmp(b.1))?;
        Some((window[i].clone(), best))
    }

    /// The posterior over learned concepts, from the semantic memory factors (most likely first).
    pub fn categorize(&self) -> Vec<(String, f32)> {
        let Some(&c) = self.index.get(CATEGORY) else { return Vec::new() };
        let p = self.infer(|_| None);
        let mut out: Vec<(String, f32)> =
            self.category_names.iter().enumerate().map(|(i, n)| (n.clone(), p[c][i])).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }
}
