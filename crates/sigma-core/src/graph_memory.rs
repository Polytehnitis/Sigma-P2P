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
//! * **Transition models.** For each action and each modeled predicate `p`,
//!   `P(@Trans(p|a) | a, p)` predicts `p`'s next value from the action *and its current value*
//!   (Sigma's state + action transition model), learned from consecutive cycles.
//! * **Perception models.** For each declared `(observation, state)` pair, `P(observation |
//!   state)` is learned from cycles in which both were known, and then lets an observation
//!   alone inform the (hidden) state, both ways through the graph.
//! * **Backed-up utilities.** A Q-style utility `Q(s, a)` per action and state is learned by
//!   temporal-difference backups, `Q(s, a) += α (r + γ max_b Q(s', b) - Q(s, a))`. In the graph
//!   each action gets a factor `[1, exp(Q(s, a))]` for the current state, so learned utility
//!   shapes action posteriors, and substates add it to a candidate's value.
//!
//! Internal predicates start with `@`. Skills may not declare such names, and they are left
//! out of cycle outcomes and of other agents' perspectives.

use crate::appraisal::APPRAISAL_SKILL;
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
/// Weight of the backed-up utility `Q(s, a)` in a substate's evaluation of a candidate.
pub const Q_WEIGHT: f32 = 0.25;
/// Discount of backed-up utilities.
pub const Q_DISCOUNT: f32 = 0.9;
/// Learning rate of backed-up utilities.
pub const Q_RATE: f32 = 0.3;
/// Largest magnitude a backed-up utility may hold.
pub const MAX_Q: f32 = 4.0;
/// Most states one action keeps a utility for.
pub const MAX_Q_STATES: usize = 1024;
/// Longest state key (longer ones are hashed).
const MAX_STATE_KEY: usize = 256;
/// Most declared perception models.
pub const MAX_PERCEPTION_MODELS: usize = 64;

pub const EPISODE: &str = "@Episode";
pub const REWARDED: &str = "@Rewarded";
pub const CATEGORY: &str = "@Category";
const EPISODIC_SKILL: &str = "@episodic";
const SEMANTIC_SKILL: &str = "@semantic";
pub(crate) const MODEL_SKILL: &str = "@models";
pub(crate) const PERCEPTION_SKILL: &str = "@perception";
const UTILITY_SKILL: &str = "@utility";

/// The predicate predicting `p`'s value after action `a`.
pub fn next_name(p: &str, a: &str) -> String {
    format!("@Next({p}|{a})")
}

/// The predicate predicting the reward for action `a`.
pub fn reward_name(a: &str) -> String {
    format!("@Reward({a})")
}

/// The predicate predicting `p`'s next value from action `a` and `p`'s current value.
pub fn transition_name(p: &str, a: &str) -> String {
    format!("@Trans({p}|{a})")
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
    /// `(action, predicate)` -> `P(predicate after | action, predicate now)`, 2 × d × d
    /// (row-major over action, current value, next value).
    pub transitions: BTreeMap<(String, String), Vec<f32>>,
    /// `(observation, state)` -> `P(observation | state)`, d_state × d_observation. Declared
    /// with [`SigmaEngine::declare_perception_model`].
    pub perceptions: BTreeMap<(String, String), Vec<f32>>,
    /// `(action, state key)` -> the exact-state part of the backed-up utility `Q(s, a)` (see
    /// [`SigmaEngine::state_key`]).
    pub utilities: BTreeMap<(String, String), f32>,
    /// `(action, feature)` -> weight of a state feature in the linear part of `Q(s, a)`, which
    /// generalizes to states never seen (see [`SigmaEngine::q_features`]).
    pub q_features: BTreeMap<(String, String), f32>,
    /// `(action, predicate)` -> `(parent, P(predicate after | action, predicate now, parent now))`,
    /// 2 × d × d_parent × d (row-major over action, current value, parent's value, next value):
    /// a transition model conditioned on a second predicate found by structure learning.
    pub contexts: BTreeMap<(String, String), (String, Vec<f32>)>,
}

/// The models as stored before linear utilities and context-conditioned transitions existed.
#[derive(Deserialize)]
struct ModelsV2 {
    effects: BTreeMap<(String, String), Vec<f32>>,
    rewards: BTreeMap<String, Vec<f32>>,
    transitions: BTreeMap<(String, String), Vec<f32>>,
    perceptions: BTreeMap<(String, String), Vec<f32>>,
    utilities: BTreeMap<(String, String), f32>,
}

/// An action's models as synced before linear utilities and context-conditioned transitions.
#[derive(Deserialize)]
struct ActionModelV2 {
    reward: Option<Vec<f32>>,
    effects: BTreeMap<String, Vec<f32>>,
    transitions: BTreeMap<String, Vec<f32>>,
    utilities: BTreeMap<String, f32>,
}

/// The models as stored before transition, perception and utility models existed.
#[derive(Deserialize)]
struct LegacyModels {
    effects: BTreeMap<(String, String), Vec<f32>>,
    rewards: BTreeMap<String, Vec<f32>>,
}

/// An action's models as synced before transition and utility models existed.
#[derive(Deserialize)]
struct LegacyActionModel {
    reward: Option<Vec<f32>>,
    effects: BTreeMap<String, Vec<f32>>,
}

/// Everything learned about one action: its reward model and its effect on each modeled
/// predicate. The unit that syncs between the user's devices (one register per action, the
/// newest write winning), since an action's models are learned together.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionModel {
    pub reward: Option<Vec<f32>>,
    pub effects: BTreeMap<String, Vec<f32>>,
    /// Predicate -> `P(predicate after | action, predicate now)`, 2 × d × d.
    pub transitions: BTreeMap<String, Vec<f32>>,
    /// State key -> the exact-state part of the action's backed-up utility in that state.
    pub utilities: BTreeMap<String, f32>,
    /// State feature -> its weight in the linear part of the action's utility.
    pub features: BTreeMap<String, f32>,
    /// Predicate -> `(parent, P(predicate after | action, predicate now, parent now))`.
    pub contexts: BTreeMap<String, (String, Vec<f32>)>,
}

/// Longest action or predicate name a synced model may carry.
const MAX_MODEL_NAME: usize = 128;

impl ActionModel {
    /// Bounds checks for a model received from another device.
    pub fn is_valid(&self) -> bool {
        let table_ok = |t: &Vec<f32>| t.iter().all(|v| v.is_finite() && (0.0..=MAX_FACTOR_VALUE).contains(v));
        let name_ok = |p: &String| !p.is_empty() && p.len() <= MAX_MODEL_NAME && !internal(p);
        let square = |n: usize| (2..=MAX_MODEL_DOMAIN).any(|d| 2 * d * d == n);
        self.reward.as_ref().is_none_or(|t| t.len() == 4 && table_ok(t))
            && self.effects.len() <= MAX_MODEL_FACTORS
            && self.effects.iter().all(|(p, t)| {
                name_ok(p) && t.len() % 2 == 0 && (4..=2 * MAX_MODEL_DOMAIN).contains(&t.len()) && table_ok(t)
            })
            && self.transitions.len() <= MAX_MODEL_FACTORS
            && self.transitions.iter().all(|(p, t)| name_ok(p) && square(t.len()) && table_ok(t))
            && self.utilities.len() <= MAX_Q_STATES
            && self.utilities.iter().all(|(s, q)| s.len() <= MAX_STATE_KEY && q.is_finite() && q.abs() <= MAX_Q)
            && self.features.len() <= MAX_Q_STATES
            && self.features.iter().all(|(f, w)| f.len() <= MAX_STATE_KEY && w.is_finite() && w.abs() <= MAX_Q)
            && self.contexts.len() <= MAX_MODEL_FACTORS
            && self.contexts.iter().all(|(p, (q, t))| {
                name_ok(p)
                    && name_ok(q)
                    && p != q
                    && (2..=MAX_MODEL_DOMAIN).any(|d| (2..=MAX_MODEL_DOMAIN).any(|dq| 2 * d * d * dq == t.len()))
                    && table_ok(t)
            })
    }

    /// Decodes a synced model, including one written before transition and utility models
    /// existed.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        postcard::from_bytes::<Self>(bytes).ok().or_else(|| {
            if let Ok(v2) = postcard::from_bytes::<ActionModelV2>(bytes) {
                return Some(ActionModel {
                    reward: v2.reward,
                    effects: v2.effects,
                    transitions: v2.transitions,
                    utilities: v2.utilities,
                    ..Default::default()
                });
            }
            let old = postcard::from_bytes::<LegacyActionModel>(bytes).ok()?;
            Some(ActionModel { reward: old.reward, effects: old.effects, ..Default::default() })
        })
    }
}

/// One gradient step of a conditional table (row-major, `shape`, normalized over slot
/// `normal`) toward the observed assignment `at`, as [`crate::gradient_step`] does for a factor.
pub(crate) fn learn_table(table: &mut Vec<f32>, shape: &[usize], at: &[usize], normal: usize, rate: f32) {
    let Ok(t) = ArrayD::from_shape_vec(IxDyn(shape), std::mem::take(table)) else { return };
    let mut b = Builder::new("@scratch");
    let n = shape.len();
    b.add((0..n).collect(), vec![PatternKind::Condact; n], shape, t.iter().copied().collect(), Some(normal));
    let inbound: Vec<Vec<f32>> = at.iter().zip(shape).map(|(&x, &d)| crate::one_hot(x, d)).collect();
    let f = &mut b.factors[0];
    crate::gradient_step(f, &inbound, rate);
    *table = f.table.iter().copied().collect();
}

/// Bounds checks for a perception model received from another device.
pub fn perception_is_valid(table: &[f32]) -> bool {
    (4..=MAX_MODEL_DOMAIN * MAX_MODEL_DOMAIN).contains(&table.len())
        && table.iter().all(|v| v.is_finite() && (0.0..=MAX_FACTOR_VALUE).contains(v))
}

impl LearnedModels {
    /// The actions with any learned model.
    pub fn actions(&self) -> std::collections::BTreeSet<String> {
        self.rewards
            .keys()
            .chain(self.effects.keys().map(|(a, _)| a))
            .chain(self.transitions.keys().map(|(a, _)| a))
            .chain(self.utilities.keys().map(|(a, _)| a))
            .chain(self.q_features.keys().map(|(a, _)| a))
            .chain(self.contexts.keys().map(|(a, _)| a))
            .cloned()
            .collect()
    }

    /// Decodes stored models, including ones stored before transition, perception and
    /// utility models existed.
    pub fn decode(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes::<Self>(bytes).or_else(|e| {
            if let Ok(v2) = postcard::from_bytes::<ModelsV2>(bytes) {
                return Ok(LearnedModels {
                    effects: v2.effects,
                    rewards: v2.rewards,
                    transitions: v2.transitions,
                    perceptions: v2.perceptions,
                    utilities: v2.utilities,
                    ..Default::default()
                });
            }
            let old = postcard::from_bytes::<LegacyModels>(bytes).map_err(|_| e)?;
            Ok(LearnedModels { effects: old.effects, rewards: old.rewards, ..Default::default() })
        })
    }

    /// The exact-state part of the backed-up utility of `action` in state `state` (0 until
    /// learned). The whole utility adds the weights of the state's features, see
    /// [`LearnedModels::q_value`].
    pub fn utility(&self, action: &str, state: &str) -> f32 {
        self.utilities.get(&(action.to_string(), state.to_string())).copied().unwrap_or(0.0)
    }

    /// The backed-up utility `Q(s, a)` of `action` in the state with key `state` and features
    /// `features`: its exact-state part plus the weights of the features, so a state never
    /// seen still gets the utility its features earned elsewhere.
    pub fn q_value(&self, action: &str, state: &str, features: &[String]) -> f32 {
        let shared: f32 = features.iter().filter_map(|f| self.q_features.get(&(action.to_string(), f.clone()))).sum();
        (self.utility(action, state) + shared).clamp(-MAX_Q, MAX_Q)
    }

    /// Everything learned about `action`.
    pub fn action_model(&self, action: &str) -> ActionModel {
        ActionModel {
            reward: self.rewards.get(action).cloned(),
            effects: self
                .effects
                .iter()
                .filter(|((a, _), _)| a == action)
                .map(|((_, p), t)| (p.clone(), t.clone()))
                .collect(),
            transitions: self
                .transitions
                .iter()
                .filter(|((a, _), _)| a == action)
                .map(|((_, p), t)| (p.clone(), t.clone()))
                .collect(),
            utilities: self
                .utilities
                .iter()
                .filter(|((a, _), _)| a == action)
                .map(|((_, s), q)| (s.clone(), *q))
                .collect(),
            features: self
                .q_features
                .iter()
                .filter(|((a, _), _)| a == action)
                .map(|((_, f), w)| (f.clone(), *w))
                .collect(),
            contexts: self
                .contexts
                .iter()
                .filter(|((a, _), _)| a == action)
                .map(|((_, p), c)| (p.clone(), c.clone()))
                .collect(),
        }
    }

    /// Replaces everything learned about `action` with `model`.
    pub fn set_action_model(&mut self, action: &str, model: ActionModel) {
        self.effects.retain(|(a, _), _| a != action);
        self.transitions.retain(|(a, _), _| a != action);
        self.utilities.retain(|(a, _), _| a != action);
        self.q_features.retain(|(a, _), _| a != action);
        self.contexts.retain(|(a, _), _| a != action);
        self.rewards.remove(action);
        if let Some(t) = model.reward {
            self.rewards.insert(action.to_string(), t);
        }
        for (p, t) in model.effects {
            self.effects.insert((action.to_string(), p), t);
        }
        for (p, t) in model.transitions {
            self.transitions.insert((action.to_string(), p), t);
        }
        for (s, q) in model.utilities {
            self.utilities.insert((action.to_string(), s), q);
        }
        for (f, w) in model.features {
            self.q_features.insert((action.to_string(), f), w);
        }
        for (p, c) in model.contexts {
            self.contexts.insert((action.to_string(), p), c);
        }
    }
}

/// Factors of one internal skill being built.
pub(crate) struct Builder {
    skill: Arc<Skill>,
    weights: Vec<f32>,
    pub(crate) factors: Vec<FactorNode>,
}

impl Builder {
    pub(crate) fn new(name: &str) -> Self {
        Builder {
            skill: Arc::new(Skill::program(name, 0, Program::default())),
            weights: Vec::new(),
            factors: Vec::new(),
        }
    }

    pub(crate) fn add(
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
            layout: None,
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
        for name in [EPISODIC_SKILL, SEMANTIC_SKILL, MODEL_SKILL, PERCEPTION_SKILL, UTILITY_SKILL, APPRAISAL_SKILL] {
            self.initial.remove(name);
        }
        let appraisal = Some(self.appraisal_factor());
        let builders = [
            self.episodic_factors(),
            self.semantic_factors(),
            self.model_factors(),
            self.perception_factors(),
            self.utility_factors(),
            appraisal,
        ];
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
        // Transition models: the next value from the action and the current value, for every
        // discrete goal, subgoal and context predicate (within the factor budget).
        let states: Vec<usize> = (0..self.vars.len())
            .filter(|&v| {
                let s = &self.vars[v].schema;
                s.role != PredicateRole::Action
                    && !internal(&s.name)
                    && self.vars[v].range.is_none()
                    && s.domain_size <= MAX_MODEL_DOMAIN
            })
            .collect();
        'actions: for &a in &actions {
            let an = self.vars[a].schema.name.clone();
            for &p in &states {
                if b.factors.len() >= MAX_MODEL_FACTORS {
                    break 'actions;
                }
                let (pn, d) = (self.vars[p].schema.name.clone(), self.vars[p].schema.domain_size);
                let next = self.internal_var(&transition_name(&pn, &an), d)?;
                // Structure learning found a second predicate the transition depends on: the
                // model conditions on it too, [action, p now, parent now, p next].
                if let Some((parent, t)) = self.models.contexts.get(&(an.clone(), pn.clone())) {
                    if let Some(q) = self.visible(parent).filter(|&q| self.vars[q].range.is_none()) {
                        let dq = self.vars[q].schema.domain_size;
                        if t.len() == 2 * d * dq * d {
                            let kinds = vec![
                                PatternKind::Condition,
                                PatternKind::Condition,
                                PatternKind::Condition,
                                PatternKind::Condact,
                            ];
                            b.add(vec![a, p, q, next], kinds, &[2, d, dq, d], t.clone(), Some(3));
                            continue;
                        }
                    }
                }
                let values = match self.models.transitions.get(&(an.clone(), pn.clone())) {
                    Some(t) if t.len() == 2 * d * d => t.clone(),
                    _ => vec![1.0 / d as f32; 2 * d * d],
                };
                let kinds = vec![PatternKind::Condition, PatternKind::Condition, PatternKind::Condact];
                b.add(vec![a, p, next], kinds, &[2, d, d], values, Some(2));
            }
        }
        Some(b)
    }

    /// Perception models `P(observation | state)` for the declared pairs, both ways: an
    /// observation informs the state it is evidence of.
    fn perception_factors(&mut self) -> Option<Builder> {
        let mut b = Builder::new(PERCEPTION_SKILL);
        for ((obs, state), table) in self.models.perceptions.clone() {
            let (Some(o), Some(s)) = (self.visible(&obs), self.visible(&state)) else { continue };
            let (ds, d_o) = (self.vars[s].schema.domain_size, self.vars[o].schema.domain_size);
            let values = if table.len() == ds * d_o { table } else { vec![1.0 / d_o as f32; ds * d_o] };
            b.add(vec![s, o], vec![PatternKind::Condact, PatternKind::Condact], &[ds, d_o], values, Some(1));
        }
        (!b.factors.is_empty()).then_some(b)
    }

    /// Backed-up utilities of the current state: a factor `[1, exp(Q(s, a))]` on each action
    /// with a utility here (learned for this state, or carried by its features).
    fn utility_factors(&mut self) -> Option<Builder> {
        let state = self.state_key();
        let features = self.q_features();
        let mut b = Builder::new(UTILITY_SKILL);
        for v in 0..self.vars.len() {
            let s = &self.vars[v].schema;
            if s.role != PredicateRole::Action || internal(&s.name) {
                continue;
            }
            let q = self.models.q_value(&s.name, &state, &features);
            if q != 0.0 {
                b.add(vec![v], vec![PatternKind::Condact], &[2], vec![1.0, q.exp()], None);
            }
        }
        (!b.factors.is_empty()).then_some(b)
    }

    /// The state a backed-up utility is learned for: the stacked goals and what working memory
    /// holds of the visible non-action predicates (hashed when long).
    pub fn state_key(&self) -> String {
        let goals: Vec<&str> = self.goal_stack.iter().map(String::as_str).collect();
        let wm: Vec<String> = self
            .working_memory()
            .into_iter()
            .filter(|(n, _)| {
                !internal(n) && self.index.get(n).is_some_and(|&v| self.vars[v].schema.role != PredicateRole::Action)
            })
            .map(|(n, x)| format!("{n}={x}"))
            .collect();
        let key = format!("{}|{}", goals.join(","), wm.join(","));
        if key.len() <= MAX_STATE_KEY {
            key
        } else {
            format!("#{}", blake3::hash(key.as_bytes()).to_hex())
        }
    }

    /// Declares that `observation` is evidence of `state`, so a perception model
    /// `P(observation | state)` is learned for them and joins the graph.
    pub fn declare_perception_model(&mut self, observation: &str, state: &str) -> Result<(), &'static str> {
        let (o, s) =
            (self.visible(observation).ok_or("unknown predicate")?, self.visible(state).ok_or("unknown predicate")?);
        if o == s || self.vars[o].range.is_some() || self.vars[s].range.is_some() {
            return Err("invalid perception model");
        }
        let (ds, d_o) = (self.vars[s].schema.domain_size, self.vars[o].schema.domain_size);
        if ds > MAX_MODEL_DOMAIN || d_o > MAX_MODEL_DOMAIN {
            return Err("predicate domain too large for a model");
        }
        let key = (observation.to_string(), state.to_string());
        if !self.models.perceptions.contains_key(&key) && self.models.perceptions.len() >= MAX_PERCEPTION_MODELS {
            return Err("too many perception models");
        }
        self.save_models();
        self.models.perceptions.entry(key).or_insert_with(|| vec![1.0 / d_o as f32; ds * d_o]);
        self.rebuild_memory_factors();
        Ok(())
    }

    /// Adopts what another of the user's devices learned about `action`, replacing this
    /// device's models of it, and rebuilds the model factors from them.
    pub fn adopt_action_model(&mut self, action: &str, model: ActionModel) -> bool {
        if action.is_empty() || action.len() > MAX_MODEL_NAME || internal(action) || !model.is_valid() {
            return false;
        }
        self.save_models();
        self.models.set_action_model(action, model);
        // Drop the old model factors first, so the rebuild does not copy them back.
        self.factors.retain(|f| !(f.memory && f.skill.skill_name == MODEL_SKILL));
        self.rebuild_memory_factors();
        true
    }

    /// Copies the model factors' learned tables back into [`LearnedModels`].
    pub(crate) fn save_models(&mut self) {
        let mut models = std::mem::take(&mut self.models);
        for f in self.factors.iter().filter(|f| f.memory && f.skill.skill_name == MODEL_SKILL) {
            let a = self.vars[f.vars[0]].schema.name.clone();
            let target = &self.vars[*f.vars.last().expect("model factors have slots")].schema.name;
            let table: Vec<f32> = f.table.iter().copied().collect();
            if target.starts_with("@Reward(") {
                models.rewards.insert(a, table);
            } else if let Some(p) = target.strip_prefix("@Next(").and_then(|t| t.split('|').next()) {
                models.effects.insert((a, p.to_string()), table);
            } else if target.starts_with("@Trans(") && f.vars.len() == 4 {
                let key = (a, self.vars[f.vars[1]].schema.name.clone());
                if let Some((_, t)) = models.contexts.get_mut(&key) {
                    *t = table;
                }
            } else if target.starts_with("@Trans(") {
                // Saved only once learned, so untouched (uniform) transitions are not synced.
                let key = (a, self.vars[f.vars[1]].schema.name.clone());
                if let Some(t) = models.transitions.get_mut(&key) {
                    *t = table;
                }
            }
        }
        for f in self.factors.iter().filter(|f| f.memory && f.skill.skill_name == PERCEPTION_SKILL) {
            let key = (self.vars[f.vars[1]].schema.name.clone(), self.vars[f.vars[0]].schema.name.clone());
            if models.perceptions.contains_key(&key) {
                models.perceptions.insert(key, f.table.iter().copied().collect());
            }
        }
        self.models = models;
    }

    /// Learns the action models from what working memory holds now, after `action` was taken
    /// last cycle: for that action the row `action = 1`, for the others the row `action = 0`.
    /// The transition models learn from the value each predicate had when `action` was chosen
    /// (`before`, the evidence then) and the value it has now.
    pub(crate) fn learn_action_models(&mut self, action: usize, before: &[Option<usize>]) {
        let rate = MODEL_RATE;
        for fi in 0..self.factors.len() {
            let f = &self.factors[fi];
            if !(f.memory && f.skill.skill_name == MODEL_SKILL) {
                continue;
            }
            let taken = usize::from(f.vars[0] == action);
            let target = self.vars[*f.vars.last().expect("model factors have slots")].schema.name.clone();
            if target.starts_with("@Trans(") {
                // Only the row of the action taken: a transition is what *this* action does.
                let p = f.vars[1];
                let (Some(x0), Some(x)) = (before.get(p).copied().flatten(), self.vars[p].evidence) else { continue };
                if taken == 0 {
                    continue;
                }
                let d = f.table.shape()[1];
                let key = (self.vars[action].schema.name.clone(), self.vars[p].schema.name.clone());
                if f.vars.len() == 4 {
                    // Conditioned on a parent too: learns from the parent's value then.
                    let q = f.vars[2];
                    let dq = f.table.shape()[2];
                    if let Some(xq) = before.get(q).copied().flatten() {
                        let inbound = vec![
                            crate::one_hot(1, 2),
                            crate::one_hot(x0, d),
                            crate::one_hot(xq, dq),
                            crate::one_hot(x, d),
                        ];
                        crate::gradient_step(&mut self.factors[fi], &inbound, rate);
                    }
                    // The plain transition model keeps learning beside it.
                    let mut plain =
                        self.models.transitions.get(&key).cloned().unwrap_or_else(|| vec![1.0 / d as f32; 2 * d * d]);
                    if plain.len() == 2 * d * d {
                        learn_table(&mut plain, &[2, d, d], &[1, x0, x], 2, rate);
                        self.models.transitions.insert(key, plain);
                    }
                    continue;
                }
                let inbound = vec![crate::one_hot(1, 2), crate::one_hot(x0, d), crate::one_hot(x, d)];
                crate::gradient_step(&mut self.factors[fi], &inbound, rate);
                let f = &self.factors[fi];
                self.models.transitions.insert(key, f.table.iter().copied().collect());
                continue;
            }
            let Some(p) = target.strip_prefix("@Next(").and_then(|t| t.split('|').next()) else { continue };
            let Some(x) = self.index.get(p).and_then(|&v| self.vars[v].evidence) else { continue };
            let d = f.table.shape()[1];
            let inbound = vec![crate::one_hot(taken, 2), crate::one_hot(x, d)];
            crate::gradient_step(&mut self.factors[fi], &inbound, rate);
        }
        self.save_models();
    }

    /// Learns the perception models from a cycle in which both the state (as evidence) and
    /// the observation (as evidence or a percept) are known.
    pub(crate) fn learn_perception_models(&mut self) {
        for fi in 0..self.factors.len() {
            let f = &self.factors[fi];
            if !(f.memory && f.skill.skill_name == PERCEPTION_SKILL) {
                continue;
            }
            let (s, o) = (f.vars[0], f.vars[1]);
            let Some(x) = self.vars[s].evidence else { continue };
            let d_o = self.vars[o].schema.domain_size;
            let seen = match self.vars[o].evidence {
                Some(y) => crate::one_hot(y, d_o),
                None => match self.percepts.get(&o) {
                    Some(p) => p.clone(),
                    None => continue,
                },
            };
            let inbound = vec![crate::one_hot(x, self.vars[s].schema.domain_size), seen];
            crate::gradient_step(&mut self.factors[fi], &inbound, MODEL_RATE);
        }
        self.save_models();
    }

    /// Adopts a perception model learned on another of the user's devices.
    pub fn adopt_perception_model(&mut self, observation: &str, state: &str, table: Vec<f32>) -> bool {
        let names_ok = |n: &str| !n.is_empty() && n.len() <= MAX_MODEL_NAME && !internal(n);
        if !(names_ok(observation) && names_ok(state) && perception_is_valid(&table)) {
            return false;
        }
        self.save_models();
        let key = (observation.to_string(), state.to_string());
        if !self.models.perceptions.contains_key(&key) && self.models.perceptions.len() >= MAX_PERCEPTION_MODELS {
            return false;
        }
        self.models.perceptions.insert(key, table);
        self.factors.retain(|f| !(f.memory && f.skill.skill_name == PERCEPTION_SKILL));
        self.rebuild_memory_factors();
        true
    }

    /// Learns the reward model of `action` from a reward given for it.
    pub(crate) fn learn_reward_model(&mut self, action: usize, reward: f32) {
        if reward == 0.0 {
            return;
        }
        let name = reward_name(&self.vars[action].schema.name);
        for fi in 0..self.factors.len() {
            let f = &self.factors[fi];
            if f.memory && f.vars.len() == 2 && f.vars[0] == action && self.vars[f.vars[1]].schema.name == name {
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
            .find(|f| f.memory && f.vars.len() == 2 && f.vars[0] == action && self.vars[f.vars[1]].schema.name == name)
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
