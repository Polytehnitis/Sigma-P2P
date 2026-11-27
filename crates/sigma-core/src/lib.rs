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

pub mod actions;
pub mod appraisal;
pub mod crypto;
pub mod decision;
pub mod demos;
mod exact;
pub mod generalize;
pub mod graph_memory;
pub mod language;
pub mod map;
pub mod memory;
pub mod mind;
pub mod neural;
pub mod parametric;
pub mod particles;
pub mod plm;
pub mod program;
pub mod storage;
pub mod substate;
pub mod sync;

use ndarray::{ArrayD, Dimension, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub use actions::{ActionManifest, ActionSpec, ConfirmationGate, PendingAction, Undone};
pub use crypto::NodeIdentity;
pub use decision::{Preference, PreferenceKind, Selection};
pub use graph_memory::{ActionModel, LearnedModels};
pub use language::{Intent, Interpretation, PhraseBook, Route};
pub use map::MapEstimate;
pub use memory::{Episode, EpisodicMemory, SemanticMemory, SemanticRecall};
pub use mind::{Appraisal, Emotion, MindModel, OtherAgent};
pub use plm::Plm;
pub use program::{
    Conditional, Extensions, Function, Pattern, PatternKind, Piece, PredicateDef, PredicateOptions, Program, Skill,
    SkillBody, Term, TypeDef, TypeKind, ValueMap,
};
pub use storage::FactorStore;
pub use substate::Substate;
pub use sync::{SyncReport, SyncState};

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
pub(crate) const BP_ITERS: usize = 50;
const BP_TOLERANCE: f32 = 1e-5;
/// Candidate actions whose posteriors differ by less than this are tied.
pub const TIE_MARGIN: f32 = 0.02;

/// How messages summarize out the other variables of a factor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Inference {
    /// Sum-product: beliefs are marginal probabilities (the default).
    #[default]
    SumProduct,
    /// Max-product: beliefs are max-marginals, the probability of the best joint assignment
    /// that gives the variable each value, so their argmaxes form a MAP estimate.
    MaxProduct,
    /// Variational message passing (mean field), as PySigma declares: a factor sends each
    /// variable `exp(E_q[log f])`, the expectation over the other variables' current beliefs,
    /// so beliefs are a fully factorized approximation that minimizes `KL(q || p)`. Runs on
    /// the region grid.
    VariationalMessagePassing,
    /// Expectation propagation with a Gaussian family for continuous variables, as PySigma
    /// declares: a factor's message into a continuous variable is the moment-matched Gaussian
    /// of the tilted belief (cavity times the exact message) divided by the cavity. Discrete
    /// variables get sum-product messages (EP with a categorical family). Runs on the region
    /// grid.
    ExpectationPropagation,
}

impl Inference {
    /// Whether this kind of inference runs only on the region grid (not on exact
    /// piecewise-linear functions or particles).
    pub fn grid_only(self) -> bool {
        matches!(self, Inference::VariationalMessagePassing | Inference::ExpectationPropagation)
    }
}

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
        if self
            .predicates
            .iter()
            .any(|p| graph_memory::internal(&p.name) && appraisal::appraisal_schema(&p.name).as_ref() != Some(p))
        {
            return Err("reserved predicate name");
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
    /// Analytic (piecewise-linear or Gaussian) form, evaluated at particle points.
    pub analytic: Option<Arc<program::Analytic>>,
    /// Compiled from a memory or a learned model (see [`graph_memory`]), not from a skill.
    pub memory: bool,
    /// Where each table entry (row-major) sits in the skill's flat weights, when the entries
    /// are not one contiguous run from `offset` (a pattern padded by an affine transform).
    pub layout: Option<Arc<Vec<usize>>>,
}

impl FactorNode {
    /// Position of table entry `i` (row-major) in the skill's flat weights.
    pub fn weight_index(&self, i: usize) -> usize {
        match &self.layout {
            Some(l) => l[i],
            None => self.offset + i,
        }
    }
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
    /// The best candidate is both strongly supported and strongly opposed by different
    /// factors (Sigma's operator that is both accepted and rejected).
    Conflict,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionImpasse {
    pub kind: ImpasseKind,
    /// The (sub)goal on top of the stack when the impasse arose.
    pub goal: String,
    pub candidates: Vec<String>,
    /// True when reasoning in a substate (or, for a none impasse, episodic recall) settled it.
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
    /// Decision impasse (none, tie, conflict or no-change) from this cycle's decision phase.
    pub decision_impasse: Option<DecisionImpasse>,
    /// The substate that reasoned about this cycle's impasse: settled (with its `result`) or
    /// still open on the state stack (see `SigmaEngine::substates`).
    pub substate: Option<Substate>,
    pub appraisal: Appraisal,
    /// Skills that took part in elaboration, when an attention budget is set.
    pub attended_skills: Vec<String>,
    /// The selected action needs the user's confirmation: it is held here instead of
    /// entering the motor buffer (see [`ConfirmationGate`]).
    pub awaiting_confirmation: Option<PendingAction>,
    /// The chunk a substate's result was compiled into this cycle (see [`SigmaEngine::chunk`]):
    /// signed and compiled when `chunk_author` is set, else unsigned, for the host to adopt
    /// with [`SigmaEngine::learn_chunk`].
    pub chunk: Option<Skill>,
    /// The operator distribution of this cycle's decision: each candidate action, after
    /// preferences, with its selection probability under the selection policy.
    pub operators: Vec<(String, f32)>,
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
    /// The state the action was chosen in, and its features, for backing up its utility.
    state: String,
    features: Vec<String>,
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
    /// Most factors that may take part in one cycle's elaboration (`None`: all).
    pub attention_budget: Option<usize>,
    attended: Option<Vec<bool>>,
    /// Last cycle's beliefs, the predictions this cycle's observations are appraised against.
    prediction: Vec<Vec<f32>>,
    last_reward: f32,
    /// The last cycle's appraisal.
    pub appraisal: Appraisal,
    /// Particles per continuous variable for particle-based message passing (`None`: messages
    /// over continuous variables stay piecewise-constant over regions).
    pub particle_count: Option<usize>,
    /// Seed for particle sampling, so inference is reproducible for a given state.
    pub particle_seed: u64,
    /// Solve continuous variables on the fixed region grid (functions tabulated at region
    /// midpoints) instead of exactly with piecewise-linear functions (the default; see
    /// [`plm`]). Particles, when `particle_count` is set, take precedence over both.
    pub tabulate_continuous: bool,
    /// Exact values behind continuous evidence (`set_value`).
    exact: HashMap<usize, f32>,
    /// Particle messages in the perception buffer.
    particle_percepts: HashMap<usize, Particles>,
    /// Actions without an [`ActionSpec`] need confirmation too (default: only actions
    /// marked `irreversible` do).
    pub confirm_unlisted_actions: bool,
    /// Action specs, the undo stack and the action awaiting confirmation.
    safety: actions::ActionSafety,
    /// The state stack below the top state: substates still reasoning about an impasse.
    substates: Vec<Substate>,
    /// Learned action and reward models (compiled into the graph each cycle).
    pub models: LearnedModels,
    /// The concepts `@Category` ranges over, in slot order.
    category_names: Vec<String>,
    /// Chunking: when set, what a substate concludes is compiled at once into a rule signed
    /// by this identity (see [`SigmaEngine::chunk`]).
    pub chunk_author: Option<NodeIdentity>,
    /// Signed chunks waiting to be compiled at the start of the next cycle.
    pending_chunks: Vec<Skill>,
    /// How an operator is selected from the operator distribution.
    pub selection: Selection,
    /// Seed of sampled selections, so a run is reproducible.
    pub selection_seed: u64,
    /// Sampled selections so far (varies the sample from one decision to the next).
    decisions: u64,
    /// Preferences over operators (see [`decision`]).
    pub preferences: Vec<Preference>,
    /// Predicate options, by predicate name (see [`PredicateOptions`]).
    options: HashMap<String, PredicateOptions>,
    /// Evidence of multi-valued predicates: the set of values that hold.
    multi: HashMap<usize, Vec<usize>>,
    /// Sum-product (marginals) or max-product (MAP) message passing.
    pub inference: Inference,
    /// Soft evidence on the appraisal predicates from the last cycle's appraisal.
    appraisal_priors: HashMap<usize, Vec<f32>>,
    /// Transitions lived, for structure learning (see [`generalize`]).
    experience: VecDeque<generalize::Experience>,
    /// Transitions since the last structure-learning pass.
    since_structure: usize,
    /// Turns off automatic structure learning of transition parents and perception models
    /// (it still runs on demand with [`SigmaEngine::learn_model_structure`]).
    pub fixed_structure: bool,
}

impl SigmaEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// The factors compiled from skills.
    pub fn factors(&self) -> &[FactorNode] {
        &self.factors[..self.factors.iter().position(|f| f.memory).unwrap_or(self.factors.len())]
    }

    /// The factors compiled from memories and learned models (see [`graph_memory`]).
    pub fn memory_factors(&self) -> &[FactorNode] {
        &self.factors[self.factors().len()..]
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
        for (name, o) in &grounding.options {
            if self.options.get(name).is_some_and(|old| old != o) {
                return Err("predicate option conflict");
            }
        }
        self.remove_skill(&skill.skill_name);
        for (name, o) in &grounding.options {
            self.options.insert(name.clone(), *o);
        }
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
                analytic: g.analytic,
                memory: false,
                layout: g.layout.map(Arc::new),
            });
        }
        self.skills.push(skill);
        // Memory factors stay after the skill factors.
        self.factors.sort_by_key(|f| f.memory);
        self.apply_learned_functions();
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
                let at: Vec<usize> = (0..f.table.len()).map(|i| f.weight_index(i)).collect();
                f.table.iter_mut().zip(at).for_each(|(t, i)| *t = weights[i]);
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
                sum[f.weight_index(i)] += w;
                count[f.weight_index(i)] += 1;
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
        self.exact.remove(&v);
        self.multi.remove(&v);
        Ok(())
    }

    pub fn clear_evidence(&mut self, name: &str) {
        if let Some(&v) = self.index.get(name) {
            self.vars[v].evidence = None;
            self.exact.remove(&v);
            self.multi.remove(&v);
        }
    }

    /// Sets the options of predicate `name` (a predicate name, applying to all its ground
    /// atoms), as a v3 skill's extensions do.
    pub fn set_predicate_options(&mut self, name: &str, options: PredicateOptions) -> Result<(), &'static str> {
        let continuous = self.vars.iter().any(|n| atom_base(&n.schema.name) == name && n.range.is_some());
        if continuous && (options.closed_world || !options.unique) {
            return Err("a continuous predicate is open world and unique");
        }
        self.options.insert(name.to_string(), options);
        Ok(())
    }

    /// The options of the predicate behind atom `name` (the defaults when none were set).
    pub fn predicate_options(&self, name: &str) -> PredicateOptions {
        if self.options.is_empty() {
            return PredicateOptions::default();
        }
        self.options.get(atom_base(name)).copied().unwrap_or_default()
    }

    fn options_of(&self, v: usize) -> PredicateOptions {
        self.predicate_options(&self.vars[v].schema.name)
    }

    /// Evidence for a multi-valued predicate: every value in `values` holds (and no other).
    pub fn set_evidence_values(&mut self, name: &str, values: &[usize]) -> Result<(), &'static str> {
        let v = *self.index.get(name).ok_or("unknown predicate")?;
        if self.options_of(v).unique {
            return Err("predicate is unique: use set_evidence");
        }
        if values.is_empty() || values.iter().any(|&x| x >= self.vars[v].schema.domain_size) {
            return Err("value outside predicate domain");
        }
        let mut set = values.to_vec();
        set.sort_unstable();
        set.dedup();
        self.vars[v].evidence = None;
        self.exact.remove(&v);
        self.multi.insert(v, set);
        Ok(())
    }

    /// Ends one cycle's hold on non-persistent evidence.
    fn forget_transient(&mut self) {
        for v in 0..self.vars.len() {
            if !self.options_of(v).persistent {
                self.vars[v].evidence = None;
                self.exact.remove(&v);
                self.multi.remove(&v);
            }
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
        self.particle_percepts.remove(&v);
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

    /// Sets a continuous predicate to `x` (its region, and the exact value for particles).
    pub fn set_value(&mut self, name: &str, x: f32) -> Result<(), &'static str> {
        let r = self.region_of(name, x)?;
        self.set_evidence(name, r)?;
        self.exact.insert(self.index[name], x);
        Ok(())
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

    /// Perceives a particle message (weighted samples). With `particle_count` set it enters
    /// message passing as particles; otherwise it is projected onto the regions.
    pub fn perceive_particles(&mut self, name: &str, particles: &Particles) -> Result<(), &'static str> {
        let mut dist = vec![0.0; self.centres(name)?.len()];
        if particles.values.len() != particles.weights.len() {
            return Err("particle values and weights differ in length");
        }
        for (&x, &w) in particles.values.iter().zip(&particles.weights) {
            if w.is_finite() && w > 0.0 {
                dist[self.region_of(name, x)?] += w;
            }
        }
        self.perceive(name, &dist)?;
        self.particle_percepts.insert(self.index[name], particles.clone());
        Ok(())
    }

    /// Posterior of `name` given working memory alone (no goal assumptions).
    pub fn posterior(&self, name: &str) -> Option<Vec<f32>> {
        let v = *self.index.get(name)?;
        Some(self.infer(|_| None).swap_remove(v))
    }

    /// Posterior mean of a continuous predicate.
    pub fn expected_value(&self, name: &str) -> Option<f32> {
        if self.particle_count.is_some() {
            return self.particle_belief(name).map(|p| p.mean());
        }
        let v = *self.index.get(name)?;
        if self.vars[v].range.is_some() {
            if let Some(mean) = self.exact_mean(v) {
                return Some(mean);
            }
        }
        let centres = self.centres(name).ok()?;
        Some(self.posterior(name)?.iter().zip(&centres).map(|(p, c)| p * c).sum())
    }

    /// Draws a particle message from the posterior of a continuous predicate: a region by its
    /// probability, then a uniform point inside it. Equal weights.
    pub fn sample_particles(&self, name: &str, n: usize, rng: &mut impl rand::Rng) -> Option<Particles> {
        let v = &self.vars[*self.index.get(name)?];
        let (lo, hi) = v.range?;
        if self.particle_count.is_some() {
            return Some(self.particle_belief(name)?.resample(n, rng));
        }
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

    /// Drains the motor buffer: every action selected (and, where needed, confirmed) since
    /// the last call, oldest first. Actuators take actions only from here.
    pub fn take_motor_commands(&mut self) -> Vec<String> {
        std::mem::take(&mut self.motor)
    }

    /// Working memory as `(predicate, value)`: evidence, then the most likely value of each percept.
    pub fn working_memory(&self) -> Vec<(String, usize)> {
        let mut wm: Vec<(String, usize)> =
            self.vars.iter().filter_map(|n| n.evidence.map(|x| (n.schema.name.clone(), x))).collect();
        for (&v, p) in &self.percepts {
            if self.vars[v].evidence.is_none() && !self.multi.contains_key(&v) {
                let x = p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |m| m.0);
                wm.push((self.vars[v].schema.name.clone(), x));
            }
        }
        for (&v, set) in &self.multi {
            wm.extend(set.iter().map(|&x| (self.vars[v].schema.name.clone(), x)));
        }
        wm.sort();
        wm
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
        for chunk in std::mem::take(&mut self.pending_chunks) {
            let _ = self.compile_skill(chunk);
        }
        // The last action's utility is backed up from the reward since and the state it led to.
        if let Some((a, state, features)) =
            self.last.as_ref().and_then(|l| Some((l.action?, l.state.clone(), l.features.clone())))
        {
            self.back_up_utility(&state, &features, a, self.last_reward);
        }
        // Memories and models join the graph; the last action's models learn what followed it.
        self.rebuild_memory_factors();
        if let Some((a, before)) = self.last.as_ref().and_then(|l| Some((l.action?, l.wm.clone()))) {
            self.learn_action_models(a, &before);
            self.record_experience(a, &before);
        }
        self.learn_perception_models();
        // Appraise what changed since the last cycle, before elaborating on it.
        let open = self.goal_stack.len();
        let progress = self.pop_satisfied_goals().len() as f32 / open.max(1) as f32;
        let surprise = self.surprise();
        let mut appraisal = Appraisal {
            desirability: (progress + std::mem::take(&mut self.last_reward)).clamp(-1.0, 1.0),
            surprise: surprise.iter().copied().fold(0.0, f32::max),
            ..Appraisal::default()
        };
        let Some(root) = self.goal_stack.first().and_then(|g| self.index.get(g)).copied() else {
            self.trace = None;
            self.last = None;
            self.substates.clear();
            self.percepts.clear();
            self.particle_percepts.clear();
            self.forget_transient();
            self.appraisal = self.appraise(appraisal, false);
            return CycleOutcome { appraisal: self.appraisal.clone(), ..CycleOutcome::default() };
        };
        // Appraisals of the situation enter the graph before elaboration, so skills can
        // condition on them this cycle (control and being stuck come from the last decision).
        appraisal.familiarity = self.familiarity();
        self.set_appraisal_priors(&appraisal, None);
        self.attended = self.attend(&surprise);
        let elaboration = self.elaborate(root);
        let wm: Vec<Option<usize>> = self.vars.iter().map(|n| n.evidence).collect();
        let last = self.last.as_ref().map(|l| (l.action, l.wm.clone(), l.goals.clone()));
        let goals = self.goal_stack.clone();
        let repeats = |a: usize| {
            last.as_ref().is_some_and(|(action, w, g)| *action == Some(a) && same_evidence(w, &wm) && *g == goals)
        };
        let substate::Decision { selected, impasse, substate, operators } = self.decide(&elaboration, repeats);
        let substate::Elaboration { p1, p2, top, top_factors, .. } = elaboration;
        // Chunking: a substate that settled the impasse leaves a rule behind.
        let chunk = match (&substate, selected) {
            // Signed now, compiled at the start of the next cycle (so this cycle's trace,
            // which reinforcement reads, keeps its factor indices).
            (Some(s), Some(a)) if s.result.is_some() => self.chunk(top, a).map(|mut c| {
                if let Some(author) = &self.chunk_author {
                    c.version = self.export_skill(&c.skill_name).map_or(1, |s| s.version + 1);
                    author.sign(&mut c);
                    self.pending_chunks.push(c.clone());
                }
                c
            }),
            _ => None,
        };
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
            state: self.state_key(),
            features: self.q_features(),
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
        appraisal.control = selected.map_or(0.0, |v| p2[v][1]);
        let stuck = decision_impasse.as_ref().is_some_and(|d| !d.resolved && d.kind != ImpasseKind::NoChange);
        self.appraisal = self.appraise(appraisal, stuck);
        let attended_skills = match &self.attended {
            Some(mask) => {
                let mut names: Vec<String> = (0..self.factors.len())
                    .filter(|&f| mask[f] && !self.factors[f].memory)
                    .map(|f| self.factors[f].skill.skill_name.clone())
                    .collect();
                names.sort();
                names.dedup();
                names
            }
            None => Vec::new(),
        };
        let mut outcome = CycleOutcome {
            appraisal: self.appraisal.clone(),
            attended_skills,
            goal_stack: self.goal_stack.clone(),
            selected_action: selected.map(|v| self.vars[v].schema.name.clone()),
            beliefs: self
                .vars
                .iter()
                .enumerate()
                .filter(|(_, n)| !graph_memory::internal(&n.schema.name))
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
            substate,
            awaiting_confirmation: None,
            chunk,
            operators: operators.into_iter().map(|(v, p)| (self.vars[v].schema.name.clone(), p)).collect(),
        };
        self.trace = Some(Trace { factors: active, marginals });
        self.prediction = p2;
        self.attended = None;
        self.episodic.record(self.goal_stack.clone(), self.working_memory(), outcome.selected_action.clone());
        if let Some(action) = outcome.selected_action.clone() {
            outcome.awaiting_confirmation = self.dispatch(action);
        }
        self.percepts.clear();
        self.particle_percepts.clear();
        self.forget_transient();
        outcome
    }

    /// Per variable: how unexpected its newly observed value is (1 - last cycle's prediction).
    fn surprise(&self) -> Vec<f32> {
        let prev = self.last.as_ref().map(|l| &l.wm);
        (0..self.vars.len())
            .map(|v| {
                let now = self.vars[v].evidence.or_else(|| {
                    self.percepts
                        .get(&v)
                        .and_then(|p| p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|m| m.0))
                });
                match (now, self.prediction.get(v)) {
                    (Some(x), Some(p)) if prev.and_then(|w| w.get(v).copied().flatten()) != Some(x) => {
                        1.0 - p.get(x).copied().unwrap_or(0.0)
                    }
                    _ => 0.0,
                }
            })
            .collect()
    }

    /// Fraction of working memory that the recalled episode shares.
    fn familiarity(&self) -> f32 {
        let wm = self.working_memory();
        let Some((episode, _)) = self.recall_episode(false) else { return 0.0 };
        if wm.is_empty() {
            return 0.0;
        }
        wm.iter().filter(|kv| episode.state.contains(kv)).count() as f32 / wm.len() as f32
    }

    /// The action of the episode recalled for this situation with a reward as part of the cue,
    /// if that episode was in fact rewarded and its action is still one of this node's actions.
    fn recall_action(&self) -> Option<usize> {
        let (episode, _) = self.recall_episode(true)?;
        let a = *self.index.get(episode.action.as_deref()?)?;
        (episode.reward > 0.0 && self.vars[a].schema.role == PredicateRole::Action).then_some(a)
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
        self.last_reward = reward;
        if let Some(a) = self.last.as_ref().and_then(|l| l.action) {
            self.learn_reward_model(a, reward);
        }
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
        let exact = self.exact_state(&|_| None);
        let (prior, msgs) = if exact.is_some() { (Vec::new(), Vec::new()) } else { self.propagate(|_| None) };
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
                .map(|(s, &v)| match &exact {
                    Some(state) => state.grid_belief(self, v, Some((fi, s))),
                    None => self.belief(v, &prior, &msgs, Some((fi, s))),
                })
                .collect();
            // Gaussian and neural functions learn their parameters, not just their tables.
            if let Some(ll) = self.learn_analytic(fi, rate) {
                log_z += ll;
                continue;
            }
            let Some(z) = gradient_step(&mut self.factors[fi], &inbound, rate) else { continue };
            log_z += z.ln();
            touched.insert(self.factors[fi].skill.skill_name.clone());
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
        (0..self.factors.len()).filter(move |&f| !self.factors[f].memory && self.factors[f].vars.contains(&v))
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
        let mut beliefs = self.infer_raw(clamp);
        if !self.options.is_empty() {
            for (v, b) in beliefs.iter_mut().enumerate() {
                *b = self.shape_belief(v, std::mem::take(b));
            }
        }
        beliefs
    }

    fn infer_raw(&self, clamp: impl Fn(usize) -> Option<usize>) -> Vec<Vec<f32>> {
        if self.inference.grid_only() {
            let (prior, msgs) = self.propagate(clamp);
            return (0..self.vars.len()).map(|v| self.belief(v, &prior, &msgs, None)).collect();
        }
        if let Some(n) = self.particle_count {
            if let Some((beliefs, _)) = self.infer_particles(&clamp, n) {
                return beliefs;
            }
        }
        if let Some(beliefs) = self.infer_exact(&clamp) {
            return beliefs;
        }
        let (prior, msgs) = self.propagate(clamp);
        (0..self.vars.len()).map(|v| self.belief(v, &prior, &msgs, None)).collect()
    }

    /// A predicate's options applied to its belief (a distribution): exponential, then
    /// normalized to sum 1 or, for a multi-valued or unnormalized predicate, scaled so the
    /// largest value is 1.
    fn shape_belief(&self, v: usize, mut b: Vec<f32>) -> Vec<f32> {
        let o = self.options_of(v);
        if o.exponential {
            b.iter_mut().for_each(|x| *x = x.exp());
        }
        if o.normalize && o.unique {
            return normalize(b);
        }
        let max = b.iter().copied().fold(0.0f32, f32::max);
        if max > 0.0 && max.is_finite() {
            b.iter_mut().for_each(|x| *x /= max);
        }
        b
    }

    /// Runs message passing; returns each variable's prior and the final factor-to-variable messages.
    #[allow(clippy::type_complexity)]
    fn propagate(&self, clamp: impl Fn(usize) -> Option<usize>) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
        let prior = self.grid_prior(&clamp);
        let tables: Vec<&ArrayD<f32>> = self.factors.iter().map(|f| &f.table).collect();
        let msgs = self.message_passing(&prior, &tables);
        (prior, msgs)
    }

    /// Each variable's prior over its regions/values: evidence or a clamp, else a percept.
    fn grid_prior(&self, clamp: &impl Fn(usize) -> Option<usize>) -> Vec<Vec<f32>> {
        (0..self.vars.len())
            .map(|v| {
                let d = self.vars[v].schema.domain_size;
                match self.vars[v].evidence.or_else(|| clamp(v)) {
                    Some(x) => one_hot(x, d),
                    None => self.open_prior(v),
                }
            })
            .collect()
    }

    /// What working memory says of `v` without evidence or a clamp: the set of values of a
    /// multi-valued predicate, a percept, the default value of a closed-world predicate, or
    /// nothing (uniform).
    pub(crate) fn open_prior(&self, v: usize) -> Vec<f32> {
        let d = self.vars[v].schema.domain_size;
        if let Some(set) = self.multi.get(&v) {
            return (0..d).map(|x| if set.contains(&x) { 1.0 } else { 0.0 }).collect();
        }
        if let Some(p) = self.percepts.get(&v).or_else(|| self.appraisal_priors.get(&v)) {
            return p.clone();
        }
        if self.vars[v].range.is_none() && self.options_of(v).closed_world {
            return one_hot(0, d);
        }
        vec![1.0; d]
    }

    /// Loopy sum-product over `tables` (one per factor, indexed like `prior`'s domains).
    fn message_passing(&self, prior: &[Vec<f32>], tables: &[&ArrayD<f32>]) -> Vec<Vec<Vec<f32>>> {
        let fixed = prior;
        let uniform = |v: usize| normalize(vec![1.0; prior[v].len()]);
        let mut msgs: Vec<Vec<Vec<f32>>> =
            self.factors.iter().map(|f| f.vars.iter().map(|&v| uniform(v)).collect()).collect();
        // Slots over multi-valued predicates, which are maximized over (Sigma's universal).
        let universal: Vec<Vec<bool>> =
            self.factors.iter().map(|f| f.vars.iter().map(|&v| !self.options_of(v).unique).collect()).collect();
        let maximize = self.inference == Inference::MaxProduct;
        let variational = self.inference == Inference::VariationalMessagePassing;
        let ep = self.inference == Inference::ExpectationPropagation;
        for _ in 0..BP_ITERS {
            let next: Vec<Vec<Vec<f32>>> = self
                .factors
                .iter()
                .enumerate()
                .map(|(fi, f)| {
                    if self.attended.as_ref().is_some_and(|mask| !mask[fi]) {
                        return f.vars.iter().map(|&v| uniform(v)).collect();
                    }
                    let inbound: Vec<Vec<f32>> = f
                        .vars
                        .iter()
                        .enumerate()
                        .map(|(s, &v)| match f.kinds[s] {
                            PatternKind::Action => uniform(v),
                            // Mean field: the other variables' whole beliefs, not cavities.
                            _ if variational => self.belief(v, fixed, &msgs, None),
                            _ => self.belief(v, fixed, &msgs, Some((fi, s))),
                        })
                        .collect();
                    (0..f.vars.len())
                        .map(|s| {
                            if f.kinds[s] == PatternKind::Condition {
                                return uniform(f.vars[s]);
                            }
                            if variational {
                                // exp(E_q[log f]) over the other slots.
                                let mut out = vec![0.0f32; inbound[s].len()];
                                for (idx, &w) in tables[fi].indexed_iter() {
                                    let q = (0..inbound.len())
                                        .filter(|&j| j != s)
                                        .fold(1.0, |acc, j| acc * inbound[j][idx[j]]);
                                    out[idx[s]] += q * w.max(1e-30).ln();
                                }
                                let top = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                                return normalize(out.into_iter().map(|l| (l - top).exp()).collect());
                            }
                            if ep && self.vars[f.vars[s]].range.is_some() {
                                let mut out = vec![0.0; inbound[s].len()];
                                for (idx, &w) in tables[fi].indexed_iter() {
                                    out[idx[s]] += (0..inbound.len())
                                        .filter(|&j| j != s)
                                        .fold(w, |acc, j| acc * inbound[j][idx[j]]);
                                }
                                return self.ep_project(f.vars[s], &inbound[s], normalize(out));
                            }
                            if maximize {
                                // Max-product: every other slot is maximized over.
                                let mut out = vec![0.0f32; inbound[s].len()];
                                for (idx, &w) in tables[fi].indexed_iter() {
                                    let p = (0..inbound.len())
                                        .filter(|&j| j != s)
                                        .fold(w, |acc, j| acc * inbound[j][idx[j]]);
                                    out[idx[s]] = out[idx[s]].max(p);
                                }
                                return normalize(out);
                            }
                            if !universal[fi].iter().enumerate().any(|(j, &u)| u && j != s) {
                                let mut out = vec![0.0; inbound[s].len()];
                                for (idx, &w) in tables[fi].indexed_iter() {
                                    out[idx[s]] += (0..inbound.len())
                                        .filter(|&j| j != s)
                                        .fold(w, |acc, j| acc * inbound[j][idx[j]]);
                                }
                                return normalize(out);
                            }
                            // Multi-valued slots are maximized over, the others summed (within
                            // each assignment of the multi-valued ones).
                            let mut groups: HashMap<(usize, Vec<usize>), f32> = HashMap::new();
                            for (idx, &w) in tables[fi].indexed_iter() {
                                let key: Vec<usize> = (0..inbound.len())
                                    .filter(|&j| j != s && universal[fi][j])
                                    .map(|j| idx[j])
                                    .collect();
                                let p =
                                    (0..inbound.len()).filter(|&j| j != s).fold(w, |acc, j| acc * inbound[j][idx[j]]);
                                *groups.entry((idx[s], key)).or_default() += p;
                            }
                            let mut out = vec![0.0f32; inbound[s].len()];
                            for ((x, _), p) in groups {
                                out[x] = out[x].max(p);
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
        msgs
    }

    /// Expectation propagation's projection of a factor's message `msg` into continuous
    /// variable `v`, given the cavity (its belief without this factor): the tilted belief
    /// `cavity · msg` is moment-matched by a Gaussian (its mean and variance, plus the region
    /// width's own spread), and the message is that Gaussian divided by the cavity.
    fn ep_project(&self, v: usize, cavity: &[f32], msg: Vec<f32>) -> Vec<f32> {
        let Some((lo, hi)) = self.vars[v].range else { return msg };
        if self.vars[v].evidence.is_some() {
            return msg;
        }
        let d = msg.len();
        let w = (hi - lo) / d as f32;
        let centres: Vec<f32> = (0..d).map(|i| lo + (i as f32 + 0.5) * w).collect();
        let tilted = normalize(cavity.iter().zip(&msg).map(|(c, m)| c * m).collect());
        let mean: f32 = tilted.iter().zip(&centres).map(|(p, x)| p * x).sum();
        let var = tilted.iter().zip(&centres).map(|(p, x)| p * (x - mean).powi(2)).sum::<f32>() + w * w / 12.0;
        if !(var.is_finite() && var > 0.0) {
            return msg;
        }
        let q: Vec<f32> = centres.iter().map(|x| (-0.5 * (x - mean).powi(2) / var).exp()).collect();
        let out: Vec<f32> =
            q.iter().zip(cavity).map(|(q, c)| if *c > 1e-12 { (q / c).min(1e12) } else { *q }).collect();
        normalize(out)
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

/// One gradient step on factor `f` given its incoming messages `inbound` (one per slot): the
/// normalized function moves `rate` along the normalized product of the messages, then each
/// group of entries normalized together (per row of the `normal` slot, else the whole table)
/// is renormalized and keeps its total weight. Returns the local partition function before the
/// step, or `None` when it is zero.
pub(crate) fn gradient_step(f: &mut FactorNode, inbound: &[Vec<f32>], rate: f32) -> Option<f32> {
    let groups = |idx: &[usize]| -> usize {
        // Rows that are normalized together: all slots except `normal`, or one group.
        f.normal.map_or(0, |n| (0..idx.len()).filter(|&j| j != n).fold(0, |a, j| a * f.table.shape()[j] + idx[j]))
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
        return None;
    }
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
    Some(z)
}

/// The predicate an atom name (`Pred` or `Pred(a,b)`) is a ground instance of.
pub(crate) fn atom_base(atom: &str) -> &str {
    atom.split('(').next().unwrap_or(atom)
}

/// Whether two evidence snapshots agree (variables registered since count as unobserved).
fn same_evidence(a: &[Option<usize>], b: &[Option<usize>]) -> bool {
    let n = a.len().max(b.len());
    (0..n).all(|i| a.get(i).copied().flatten() == b.get(i).copied().flatten())
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
mod tests_actions;
#[cfg(test)]
mod tests_affine;
#[cfg(test)]
mod tests_continuous;
#[cfg(test)]
mod tests_continuous_args;
#[cfg(test)]
mod tests_decision;
#[cfg(test)]
mod tests_exact;
#[cfg(test)]
mod tests_generalize;
#[cfg(test)]
mod tests_graph_memory;
#[cfg(test)]
mod tests_language;
#[cfg(test)]
mod tests_learning;
#[cfg(test)]
mod tests_map;
#[cfg(test)]
mod tests_mapped;
#[cfg(test)]
mod tests_memory;
#[cfg(test)]
mod tests_mind;
#[cfg(test)]
mod tests_models;
#[cfg(test)]
mod tests_neural;
#[cfg(test)]
mod tests_options;
#[cfg(test)]
mod tests_particles;
#[cfg(test)]
mod tests_substates;
#[cfg(test)]
mod tests_sync;
#[cfg(test)]
mod tests_v2;
