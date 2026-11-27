//! Reflective impasse substates, as in Sigma.
//!
//! When the decision phase reaches an impasse (none, tie, conflict or no-change), the engine
//! does not just look one step ahead: it opens a *substate* one level below the state that
//! impassed. The substate starts from a copy of that state's working memory and goals
//! (Sigma's impasse-copy-state) and problem-solves in it: for each candidate it imagines
//! taking the action, predicts what that would bring about, and keeps deciding inside the
//! imagined state. A decision inside the imagined state can impasse in turn, which opens a
//! deeper substate (up to [`MAX_SUBSTATE_DEPTH`]). Candidates are valued by the discounted
//! probability of reaching the (sub)goal the impasse arose for.
//!
//! A substate that settles the impasse returns its result, the selected action, to the state
//! above and is removed. One that does not stays on the state stack, and keeps reasoning in
//! the following cycles, looking one step further each time (up to [`MAX_HORIZON`]). When the
//! impasse goes away or changes (other goal, other candidates, other kind), every lower state
//! is deleted.

use crate::graph_memory::{next_name, transition_name, Q_WEIGHT, REWARD_WEIGHT};
use crate::{
    ImpasseKind, NodeIdentity, PatternKind, PredicateRole, PredicateSchema, SigmaEngine, SkillPackage, MAX_TABLE_LEN,
    SUBGOAL_PROMOTION, TIE_MARGIN,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Deepest level of nested substates (the top state is level 0).
pub const MAX_SUBSTATE_DEPTH: usize = 3;
/// Furthest a substate looks ahead (it starts at 1 and deepens each cycle it persists).
pub const MAX_HORIZON: usize = 4;
/// Discount per imagined step.
pub const DISCOUNT: f32 = 0.9;
/// Most imagined decisions per cycle, across all substates.
const SIMULATION_BUDGET: usize = 256;
/// A factor message this sure an action should (or should not) be taken counts as a vote.
const CONFLICT_SURE: f32 = 0.8;
/// Most context conditions a chunk tests.
pub const MAX_CHUNK_CONDITIONS: usize = 6;
/// Weight a chunk gives its action in the situation it was learned in (as a demonstration).
pub const CHUNK_WEIGHT: f32 = 10.0;
/// Skill names of chunks start with this.
pub const CHUNK_PREFIX: &str = "chunk:";

/// One level of the state stack below the top state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Substate {
    /// 1 for a substate of the top state, 2 for one opened inside it, and so on.
    pub depth: usize,
    /// The impasse that opened it.
    pub impasse: ImpasseKind,
    /// The (sub)goal on top of the stack when the impasse arose.
    pub goal: String,
    /// The candidate actions it reasons about.
    pub candidates: Vec<String>,
    /// What reasoning in the substate concluded: each candidate's value (discounted chance of
    /// reaching the root goal), plus `"(nothing)"` for doing nothing where that is an option.
    pub evaluations: Vec<(String, f32)>,
    /// How many steps ahead it has looked.
    pub horizon: usize,
    /// Deepest substate opened while imagining (`depth` when none was needed).
    pub deepest: usize,
    /// The action it returned to the state above, once the impasse is settled.
    pub result: Option<String>,
    /// Consecutive cycles it has been open.
    pub cycles: u32,
}

/// What elaboration found for the decision phase.
pub(crate) struct Elaboration {
    /// Posteriors with the active goals clamped (subgoaling).
    pub p1: Vec<Vec<f32>>,
    /// Posteriors with goals desired and open subgoals assumed false (action choice).
    pub p2: Vec<Vec<f32>>,
    /// The deepest open (sub)goal.
    pub top: usize,
    pub top_factors: Vec<usize>,
    /// Candidate actions for `top` with `P(true) > 0.5`, best first.
    pub ranked: Vec<(usize, f32)>,
    /// Stacked goals (role `Goal`), clamped to desired.
    pub stacked: HashSet<usize>,
}

/// The outcome of the decision phase.
pub(crate) struct Decision {
    pub selected: Option<usize>,
    pub impasse: Option<(ImpasseKind, Vec<usize>, bool)>,
    /// The substate that worked on this cycle's impasse, if any.
    pub substate: Option<Substate>,
}

/// Imagined decisions left this cycle, and the deepest substate reached.
struct Search {
    budget: usize,
    deepest: usize,
}

impl SigmaEngine {
    /// The state stack below the top state: substates still working on an impasse.
    pub fn substates(&self) -> &[Substate] {
        &self.substates
    }

    /// Elaboration: subgoals the active goals call for are pushed, then the candidate actions
    /// for the deepest open (sub)goal are ranked.
    pub(crate) fn elaborate(&mut self, root: usize) -> Elaboration {
        let stacked: HashSet<usize> = self
            .goal_stack
            .iter()
            .filter_map(|g| self.index.get(g).copied())
            .filter(|&v| self.vars[v].schema.role == PredicateRole::Goal)
            .collect();

        // Elaboration 1: which subgoal states do the active goals call for? A subgoal observed
        // false is still needed, so its observation does not hide the need.
        let goals = |v: usize| if stacked.contains(&v) { Some(1) } else { None };
        let observed_false: Vec<usize> = (0..self.vars.len())
            .filter(|&v| self.vars[v].schema.role == PredicateRole::Subgoal && self.vars[v].evidence == Some(0))
            .collect();
        let p1 = if observed_false.is_empty() {
            self.infer(goals)
        } else {
            let mut needs = self.hypothetical();
            observed_false.iter().for_each(|&v| needs.vars[v].evidence = None);
            needs.infer(goals)
        };
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
            _ if stacked.contains(&v) => Some(1),
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
        Elaboration { p1, p2, top, top_factors, ranked, stacked }
    }

    /// The decision phase: select the best candidate, or reach an impasse and reflect on it
    /// in a substate. `repeats` tells whether selecting an action would repeat the last
    /// decision with nothing changed (a no-change impasse).
    pub(crate) fn decide(&mut self, e: &Elaboration, repeats: impl Fn(usize) -> bool) -> Decision {
        let tied: Vec<usize> = e.ranked.iter().take_while(|c| e.ranked[0].1 - c.1 < TIE_MARGIN).map(|c| c.0).collect();
        let others: Vec<usize> = e.ranked.iter().map(|c| c.0).collect();
        let (kind, candidates, stuck) = match tied.len() {
            0 => {
                // None impasse: a rewarded action from a similar episode settles it at once.
                if let Some(a) = self.recall_action() {
                    self.substates.clear();
                    return Decision {
                        selected: Some(a),
                        impasse: Some((ImpasseKind::None, vec![a], true)),
                        substate: None,
                    };
                }
                // Otherwise reason about every action connected to the goal.
                // Otherwise reason about every action connected to the goal, by a skill or by a
                // learned action model that predicts an effect on an open (sub)goal.
                let (_, reach) = self.reachable(e.top);
                let mut actions: Vec<usize> =
                    reach.into_iter().filter(|&v| self.vars[v].schema.role == PredicateRole::Action).collect();
                for a in self.modeled_means() {
                    if !actions.contains(&a) {
                        actions.push(a);
                    }
                }
                (ImpasseKind::None, actions, None)
            }
            1 if self.conflicted(tied[0], e) => (ImpasseKind::Conflict, others, None),
            1 if repeats(tied[0]) => (ImpasseKind::NoChange, others, Some(tied[0])),
            1 => {
                self.substates.clear();
                return Decision { selected: Some(tied[0]), impasse: None, substate: None };
            }
            _ => (ImpasseKind::Tie, tied, None),
        };
        let (selected, resolved, substate) = self.reflect(kind, e, &candidates, stuck);
        let reported = match kind {
            ImpasseKind::None if !resolved => Vec::new(),
            ImpasseKind::NoChange => stuck.into_iter().collect(),
            _ => candidates,
        };
        Decision { selected, impasse: Some((kind, reported, resolved)), substate: Some(substate) }
    }

    /// Opens (or continues) the substate for an impasse and reasons in it.
    fn reflect(
        &mut self,
        kind: ImpasseKind,
        e: &Elaboration,
        candidates: &[usize],
        stuck: Option<usize>,
    ) -> (Option<usize>, bool, Substate) {
        let goal = self.vars[e.top].schema.name.clone();
        // Candidates are valued by how well they lead to the root goal of the stack.
        let root = self.index[&self.goal_stack[0]];
        let names: Vec<String> = candidates.iter().map(|&v| self.vars[v].schema.name.clone()).collect();
        // The same impasse as last cycle: keep reasoning in its substate, a step further.
        // Anything else: delete the lower states and open a new one.
        let (horizon, cycles) = match self.substates.first() {
            Some(s) if s.impasse == kind && s.goal == goal && s.candidates == names => {
                ((s.horizon + 1).min(MAX_HORIZON), s.cycles + 1)
            }
            _ => (1, 1),
        };
        self.substates.clear();

        let state = self.hypothetical();
        let mut search = Search { budget: SIMULATION_BUDGET, deepest: 1 };
        let mut evaluations: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&a| {
                // The repeated action has just been seen not to change anything.
                let v = if Some(a) == stuck {
                    0.0
                } else {
                    state.imagine(a, root, candidates, horizon, 1, &[], &mut search)
                };
                // Reinforcement learning: the reward model's advantage for the action, and its
                // backed-up utility in this state.
                (a, (v + REWARD_WEIGHT * self.reward_advantage(a) + Q_WEIGHT * self.utility(a)).max(0.0))
            })
            .collect();
        evaluations.sort_by(|a, b| b.1.total_cmp(&a.1));
        // Doing nothing is an option unless the candidates tie with each other.
        let idle = (kind != ImpasseKind::Tie).then(|| state.idle_value(root, candidates));
        let best = evaluations.first().copied();
        let runner_up = match (evaluations.get(1), idle) {
            (Some(&(_, v)), Some(i)) => v.max(i),
            (Some(&(_, v)), None) => v,
            (None, Some(i)) => i,
            (None, None) => 0.0,
        };
        let resolved = best.is_some_and(|(_, v)| v > 0.0 && v - runner_up >= TIE_MARGIN);
        let selected = match (resolved, kind) {
            (true, _) => best.map(|b| b.0),
            // An unsettled no-change keeps its action selected; anything else selects nothing.
            (false, ImpasseKind::NoChange) => stuck,
            (false, _) => None,
        };
        let mut evaluations: Vec<(String, f32)> =
            evaluations.into_iter().map(|(a, v)| (self.vars[a].schema.name.clone(), v)).collect();
        if let Some(i) = idle {
            evaluations.push(("(nothing)".into(), i));
        }
        let substate = Substate {
            depth: 1,
            impasse: kind,
            goal,
            candidates: names,
            evaluations,
            horizon,
            deepest: search.deepest,
            result: resolved.then(|| selected.map(|a| self.vars[a].schema.name.clone())).flatten(),
            cycles,
        };
        if !resolved {
            self.substates.push(substate.clone());
        }
        (selected, resolved, substate)
    }

    /// A copy of this state's knowledge, working memory and goals to reason in (no memories,
    /// action safety or buffers beyond the current percepts).
    fn hypothetical(&self) -> SigmaEngine {
        SigmaEngine {
            vars: self.vars.clone(),
            index: self.index.clone(),
            factors: self.factors.clone(),
            skills: self.skills.clone(),
            initial: self.initial.clone(),
            goal_stack: self.goal_stack.clone(),
            percepts: self.percepts.clone(),
            exact: self.exact.clone(),
            particle_percepts: self.particle_percepts.clone(),
            particle_count: self.particle_count,
            particle_seed: self.particle_seed,
            tabulate_continuous: self.tabulate_continuous,
            attended: self.attended.clone(),
            ..SigmaEngine::default()
        }
    }

    /// How likely `goal` holds if `action` is taken and the other candidates are not: the
    /// one-step value, with stacked goals desired and open subgoals assumed false.
    fn step_value(&self, action: Option<usize>, goal: usize, candidates: &[usize]) -> f32 {
        let stacked: HashSet<usize> = self
            .goal_stack
            .iter()
            .filter_map(|g| self.index.get(g).copied())
            .filter(|&v| self.vars[v].schema.role == PredicateRole::Goal)
            .collect();
        let p = self.infer(|v| match self.vars[v].schema.role {
            _ if v == goal => None,
            _ if Some(v) == action => Some(1),
            _ if candidates.contains(&v) => Some(0),
            _ if stacked.contains(&v) => Some(1),
            PredicateRole::Subgoal => Some(0),
            _ => None,
        });
        p[goal].get(1).copied().unwrap_or(0.0)
    }

    /// The value of taking no candidate at all.
    fn idle_value(&self, goal: usize, candidates: &[usize]) -> f32 {
        DISCOUNT * self.step_value(None, goal, candidates)
    }

    /// Imagines taking `action` in this (hypothetical) state and keeps deciding inside the
    /// imagined state for up to `horizon` steps. Returns the discounted probability that `goal`
    /// is reached.
    /// `taken` lists the actions already imagined on this path, which are not taken again (a
    /// repeat would change nothing the imagination can see).
    #[allow(clippy::too_many_arguments)]
    fn imagine(
        &self,
        action: usize,
        goal: usize,
        candidates: &[usize],
        horizon: usize,
        depth: usize,
        taken: &[usize],
        search: &mut Search,
    ) -> f32 {
        if search.budget == 0 {
            return 0.0;
        }
        search.budget -= 1;
        let now = DISCOUNT * self.step_value(Some(action), goal, candidates);
        if horizon <= 1 {
            return now;
        }
        // Predicted effects: open (sub)goals that the action makes likely, and likelier than
        // doing nothing would, become true. Subgoals are predicted with the goals not
        // assumed (their pull on subgoals is a need, not an effect).
        let goals: HashSet<usize> = self
            .vars
            .iter()
            .enumerate()
            .filter(|(_, n)| n.schema.role == PredicateRole::Goal)
            .map(|(v, _)| v)
            .collect();
        let predict = |act: bool, goals_free: bool| {
            self.infer(|v| {
                if v == action {
                    Some(usize::from(act))
                } else if candidates.contains(&v)
                    || (!goals_free && goals.contains(&v) && self.vars[v].evidence.is_none())
                {
                    Some(0)
                } else {
                    None
                }
            })
        };
        let (sub, sub_idle) = (predict(true, false), predict(false, false));
        let (top, top_idle) = (predict(true, true), predict(false, true));
        let mut next = self.hypothetical();
        let (mut confidence, mut changed) = (1.0, false);
        for g in &self.goal_stack {
            let v = self.index[g];
            let (p, idle) = if goals.contains(&v) { (&top, &top_idle) } else { (&sub, &sub_idle) };
            let (mut q, mut q0) = (p[v].get(1).copied().unwrap_or(0.0), idle[v].get(1).copied().unwrap_or(0.0));
            // The learned action and transition models' predictions, where they say more. The
            // transition model reads the predicate's current value; doing nothing leaves it.
            let an = &self.vars[action].schema.name;
            if let Some(&m) = self.index.get(&next_name(g, an)) {
                let (mq, mq0) = (sub[m].get(1).copied().unwrap_or(0.0), sub_idle[m].get(1).copied().unwrap_or(0.0));
                if mq - mq0 > q - q0 {
                    (q, q0) = (mq, mq0);
                }
            }
            if let Some(&m) = self.index.get(&transition_name(g, an)) {
                let mq = sub[m].get(1).copied().unwrap_or(0.0);
                let mq0 = if self.satisfied(v) { 1.0 } else { 0.0 };
                // An unlearned (uniform) transition model never passes this.
                if mq > SUBGOAL_PROMOTION && mq - mq0 > q - q0 {
                    (q, q0) = (mq, mq0);
                }
            }
            if !self.satisfied(v) && q > SUBGOAL_PROMOTION && q - q0 >= TIE_MARGIN {
                confidence *= q;
                next.vars[v].evidence = Some(1);
                changed = true;
            }
        }
        if !changed {
            return now; // nothing predicted to change
        }
        if next.satisfied(goal) {
            return now.max(DISCOUNT * confidence);
        }
        next.pop_satisfied_goals();
        let Some(root) = next.goal_stack.first().map(|g| next.index[g]) else {
            return now.max(DISCOUNT * confidence);
        };
        // Decide inside the imagined state; a tie there opens a deeper substate.
        let path: Vec<usize> = taken.iter().copied().chain([action]).collect();
        let e = next.elaborate(root);
        let ranked: Vec<(usize, f32)> = e.ranked.into_iter().filter(|c| !path.contains(&c.0)).collect();
        let tied: Vec<usize> = ranked.iter().take_while(|c| ranked[0].1 - c.1 < TIE_MARGIN).map(|c| c.0).collect();
        let tail = match tied.len() {
            // Nothing more to do in the imagined state: it is worth what it already achieves.
            0 => DISCOUNT * next.step_value(None, goal, &[]),
            1 => next.imagine(tied[0], goal, &[tied[0]], horizon - 1, depth, &path, search),
            _ if depth < MAX_SUBSTATE_DEPTH => {
                search.deepest = search.deepest.max(depth + 1);
                tied.iter()
                    .map(|&b| next.imagine(b, goal, &tied, horizon - 1, depth + 1, &path, search))
                    .fold(0.0, f32::max)
            }
            _ => next.imagine(tied[0], goal, &tied, horizon - 1, depth, &path, search),
        };
        now.max(DISCOUNT * confidence * tail)
    }

    /// Chunking, as in Sigma: compiles what a substate concluded into a rule, so the same
    /// situation is decided at once next time, without an impasse. The rule is a table skill
    /// over the (sub)goal the impasse arose for, the context working memory holds that the
    /// decision depended on (observed context predicates linked to that goal, to the root goal
    /// or to the action, at most [`MAX_CHUNK_CONDITIONS`]), and the action. Its one strong
    /// entry is that situation with the action taken. Returned unsigned.
    pub fn chunk(&self, top: usize, action: usize) -> Option<SkillPackage> {
        let root = self.index.get(self.goal_stack.first()?).copied()?;
        let mut conditions: Vec<usize> = Vec::new();
        for start in [top, root, action] {
            for v in self.reachable(start).1 {
                let n = &self.vars[v];
                if n.schema.role == PredicateRole::Context
                    && n.evidence.is_some()
                    && n.range.is_none()
                    && !crate::graph_memory::internal(&n.schema.name)
                    && !conditions.contains(&v)
                {
                    conditions.push(v);
                }
            }
        }
        conditions.sort_by(|&a, &b| self.vars[a].schema.name.cmp(&self.vars[b].schema.name));
        conditions.truncate(MAX_CHUNK_CONDITIONS);
        let slots: Vec<usize> = [top].into_iter().chain(conditions.iter().copied()).chain([action]).collect();
        let predicates: Vec<PredicateSchema> = slots.iter().map(|&v| self.vars[v].schema.clone()).collect();
        let factor_shape: Vec<usize> = predicates.iter().map(|p| p.domain_size).collect();
        let len = factor_shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).filter(|&n| n <= MAX_TABLE_LEN)?;
        // A goal is decided on as desired, an open subgoal as not yet holding.
        let goal_value = usize::from(self.vars[top].schema.role == PredicateRole::Goal);
        let values: Vec<usize> = [goal_value]
            .into_iter()
            .chain(conditions.iter().map(|&v| self.vars[v].evidence.expect("observed")))
            .chain([1])
            .collect();
        let mut factor_values = vec![1.0; len];
        let flat = values.iter().zip(&factor_shape).fold(0, |acc, (x, d)| acc * d + x);
        factor_values[flat] = CHUNK_WEIGHT;
        let situation: Vec<String> =
            slots.iter().zip(&values).map(|(&v, x)| format!("{}={x}", self.vars[v].schema.name)).collect();
        let tag = blake3::hash(situation.join(",").as_bytes()).to_hex();
        Some(SkillPackage {
            skill_name: format!(
                "{CHUNK_PREFIX}{}->{}#{}",
                self.vars[top].schema.name,
                self.vars[action].schema.name,
                &tag[..12]
            ),
            version: 1,
            author_pubkey: [0; 32],
            signature: Vec::new(),
            predicates,
            factor_shape,
            factor_values,
        })
    }

    /// Signs a chunk with `author` and compiles it as a skill.
    pub fn learn_chunk(
        &mut self,
        author: &NodeIdentity,
        mut chunk: SkillPackage,
    ) -> Result<SkillPackage, &'static str> {
        if !chunk.skill_name.starts_with(CHUNK_PREFIX) {
            return Err("not a chunk");
        }
        chunk.version = self.export_skill(&chunk.skill_name).map_or(1, |s| s.version + 1);
        author.sign_skill(&mut chunk);
        self.compile_skill(chunk.clone())?;
        Ok(chunk)
    }

    /// Actions whose learned model predicts that they bring about an open (sub)goal.
    fn modeled_means(&self) -> Vec<usize> {
        let open: Vec<&String> = self.goal_stack.iter().filter(|g| !self.satisfied(self.index[*g])).collect();
        let effective = |f: &crate::FactorNode, a: usize| {
            let an = &self.vars[a].schema.name;
            if f.vars[0] != a
                || f.table.len() != 4
                || !open.iter().any(|g| self.vars[f.vars[1]].schema.name == next_name(g, an))
            {
                return false;
            }
            let t: Vec<f32> = f.table.iter().copied().collect();
            let (q, q0) = (t[3] / (t[2] + t[3]), t[1] / (t[0] + t[1]));
            q > SUBGOAL_PROMOTION && q - q0 >= TIE_MARGIN
        };
        // A transition model, read at the (sub)goal's current value: binary, [a, g, @Trans].
        let transition = |f: &crate::FactorNode, a: usize| {
            let an = &self.vars[a].schema.name;
            if f.vars.len() != 3 || f.vars[0] != a || f.table.len() != 8 {
                return false;
            }
            let g = &self.vars[f.vars[1]].schema.name;
            if !open.contains(&g) || self.vars[f.vars[2]].schema.name != transition_name(g, an) {
                return false;
            }
            let Some(x0) = self.vars[f.vars[1]].evidence else { return false };
            let t: Vec<f32> = f.table.iter().copied().collect();
            let row = 4 + 2 * x0; // action taken, current value x0
            t[row + 1] / (t[row] + t[row + 1]) > SUBGOAL_PROMOTION
        };
        (0..self.vars.len())
            .filter(|&a| self.vars[a].schema.role == PredicateRole::Action)
            .filter(|&a| self.memory_factors().iter().any(|f| effective(f, a) || transition(f, a)))
            .collect()
    }

    /// A conflict: among the factors that send `action` a message, some are sure it should be
    /// taken and others sure it should not (Sigma's operator both accepted and rejected).
    fn conflicted(&self, action: usize, e: &Elaboration) -> bool {
        let clamp = |v: usize| match self.vars[v].schema.role {
            _ if e.stacked.contains(&v) => Some(1),
            PredicateRole::Subgoal => Some(0),
            _ => None,
        };
        let votes = self.support(action, &clamp);
        votes.iter().any(|&p| p >= CONFLICT_SURE) && votes.iter().any(|&p| p <= 1.0 - CONFLICT_SURE)
    }

    /// P(true) of each factor-to-variable message into binary `v` (grid or exact messages).
    fn support(&self, v: usize, clamp: &dyn Fn(usize) -> Option<usize>) -> Vec<f32> {
        if self.vars[v].schema.domain_size != 2 || self.particle_count.is_some() {
            return Vec::new();
        }
        let slots: Vec<(usize, usize)> = self
            .factors
            .iter()
            .enumerate()
            .flat_map(|(fi, f)| {
                f.vars
                    .iter()
                    .enumerate()
                    .filter(move |&(s, &u)| u == v && f.kinds[s] != PatternKind::Condition)
                    .map(move |(s, _)| (fi, s))
            })
            .collect();
        if let Some(state) = self.exact_state(clamp) {
            return slots.iter().map(|&(fi, s)| state.message_mass(fi, s, 1)).collect();
        }
        let (_, msgs) = self.propagate(clamp);
        slots.iter().map(|&(fi, s)| msgs[fi][s][1]).collect()
    }
}
