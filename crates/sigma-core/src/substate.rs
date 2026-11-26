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

use crate::graph_memory::{next_name, REWARD_WEIGHT};
use crate::{ImpasseKind, PatternKind, PredicateRole, SigmaEngine, SUBGOAL_PROMOTION, TIE_MARGIN};
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

        // Elaboration 1: which subgoal states do the active goals call for?
        let p1 = self.infer(|v| if stacked.contains(&v) { Some(1) } else { None });
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
                // Reinforcement learning: the reward model's advantage for the action.
                (a, (v + REWARD_WEIGHT * self.reward_advantage(a)).max(0.0))
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
            // The learned action model's prediction, where it says more.
            if let Some(&m) = self.index.get(&next_name(g, &self.vars[action].schema.name)) {
                let (mq, mq0) = (sub[m].get(1).copied().unwrap_or(0.0), sub_idle[m].get(1).copied().unwrap_or(0.0));
                if mq - mq0 > q - q0 {
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
        (0..self.vars.len())
            .filter(|&a| self.vars[a].schema.role == PredicateRole::Action)
            .filter(|&a| self.memory_factors().iter().any(|f| effective(f, a)))
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
