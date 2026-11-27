//! Higher functions on top of the decision cycle: attention, appraisal/emotion and
//! Theory of Mind. Each one reuses the same graph and inference; none adds new mechanisms.

use crate::{PatternKind, PredicateRole, PredicateSchema, SigmaEngine, Skill};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// How the agent appraises its current cycle (each in `[0, 1]`, desirability in `[-1, 1]`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Appraisal {
    /// Progress toward goals plus the latest reward.
    pub desirability: f32,
    /// How unexpected the new observations were, given last cycle's predictions.
    pub surprise: f32,
    /// Coping potential: confidence in the selected action (0 when nothing is selected).
    pub control: f32,
    /// How much of the current situation matches the closest remembered episode.
    pub familiarity: f32,
    /// The MAP value of `@Emotion` in the graph (see [`crate::appraisal`]).
    pub emotion: Emotion,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Emotion {
    #[default]
    Neutral,
    /// Things went well.
    Joy,
    /// Things went badly.
    Distress,
    /// Something unexpected happened.
    Surprise,
    /// A confident way forward.
    Hope,
    /// Stuck, with no way forward.
    Fear,
}

impl Appraisal {
    /// Affect modulates learning: surprising outcomes are learned faster.
    pub fn learning_rate(&self, base: f32) -> f32 {
        base * (1.0 + self.surprise)
    }
}

/// Another agent as this agent believes it to be.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OtherAgent {
    pub name: String,
    pub goals: Vec<String>,
    /// What we believe the other agent observes.
    pub evidence: Vec<(String, usize)>,
    /// The other's skills; `None` assumes it knows what we know (simulation theory).
    pub skills: Option<Vec<Skill>>,
    /// Agents the other agent itself reasons about (recursive Theory of Mind).
    pub models: Vec<OtherAgent>,
}

/// The predicted decision of a modelled agent.
#[derive(Clone, Debug, PartialEq)]
pub struct MindModel {
    pub agent: String,
    pub predicted_action: Option<String>,
    /// `(action, P(action = true))` for every action the model considered.
    pub action_beliefs: Vec<(String, f32)>,
    pub nested: Vec<MindModel>,
}

/// The predicted decisions of several agents modelled in one factor graph
/// ([`SigmaEngine::model_agents_jointly`]).
#[derive(Clone, Debug, PartialEq)]
pub struct JointModel {
    /// One prediction per agent, in the order given.
    pub agents: Vec<MindModel>,
    /// Best-response rounds run.
    pub rounds: usize,
    /// No agent's prediction changed in the last round.
    pub converged: bool,
    /// Variables in the joint graph.
    pub variables: usize,
    /// `Does(other,a)` atoms tied to the other agent's action.
    pub links: usize,
}

impl SigmaEngine {
    /// Predicts what `other` will do by running a decision cycle from its perspective: its
    /// goals, observations and skills (ours unless given). Agents it models in turn are
    /// predicted first, up to `depth` levels; each nested prediction `a` of agent `n` becomes
    /// evidence `Does(n,a) = 1` from the other's perspective, when that atom exists there.
    pub fn model_agent(&self, other: &OtherAgent, depth: usize) -> Result<MindModel, &'static str> {
        let mut view = match &other.skills {
            None => self.perspective(),
            Some(skills) => {
                let mut e = SigmaEngine::new();
                for s in skills {
                    e.compile_skill(s.clone())?;
                }
                e
            }
        };
        let nested = if depth > 1 {
            other.models.iter().map(|m| view.model_agent(m, depth - 1)).collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        for m in &nested {
            if let Some(a) = &m.predicted_action {
                let _ = view.set_evidence(&format!("Does({},{a})", m.agent), 1);
            }
        }
        for (name, x) in &other.evidence {
            if !view.index.contains_key(name) {
                view.register_predicate(&PredicateSchema::binary(name, PredicateRole::Context))?;
            }
            view.set_evidence(name, *x)?;
        }
        other.goals.iter().for_each(|g| view.push_goal(g));
        let out = view.cognitive_cycle();
        let actions: HashSet<&str> = view
            .vars
            .iter()
            .filter(|v| v.schema.role == PredicateRole::Action)
            .map(|v| v.schema.name.as_str())
            .collect();
        Ok(MindModel {
            agent: other.name.clone(),
            predicted_action: out.selected_action,
            action_beliefs: out
                .beliefs
                .into_iter()
                .filter(|(n, _)| actions.contains(n.as_str()))
                .map(|(n, p)| (n, p.get(1).copied().unwrap_or(0.0)))
                .collect(),
            nested,
        })
    }

    /// Predicts what several agents will do in **one** factor graph, as an option beside
    /// [`SigmaEngine::model_agent`] (which runs one separate graph per agent).
    ///
    /// Every agent's view (our knowledge, or its own skills) is copied into a joint engine with
    /// its variables renamed `agent.name`, so each keeps its own beliefs, goals and
    /// observations. Variables named in `shared` (the world the agents act in) and internal
    /// `@` variables keep their names and are one variable for all. An agent's atom
    /// `Does(other,a)` is tied by a near-hard factor to the other agent's action `a`, so what
    /// one agent believes about another follows from the other's own model in the same graph.
    ///
    /// Each agent then decides on the joint graph with its own goals and observations (and every
    /// agent's observations of shared variables), choosing only among its own actions, while
    /// the others' predicted actions stand as evidence. This repeats
    /// (best response) until no prediction changes or `rounds` rounds have run. Nested
    /// [`OtherAgent::models`] are not used: the joint graph takes their place.
    pub fn model_agents_jointly(
        &self,
        others: &[OtherAgent],
        shared: &[&str],
        rounds: usize,
    ) -> Result<JointModel, &'static str> {
        let names: HashSet<&str> = others.iter().map(|o| o.name.as_str()).collect();
        if names.len() != others.len() || names.iter().any(|n| n.is_empty() || n.contains('.')) {
            return Err("agents need distinct names without dots");
        }
        let mut joint = SigmaEngine::new();
        // Per agent: its own variable names and their indices in the joint engine.
        let mut own: Vec<HashMap<String, usize>> = Vec::with_capacity(others.len());
        let is_shared = |n: &str| shared.contains(&n) || crate::graph_memory::internal(n);
        for other in others {
            let mut view = match &other.skills {
                None => self.perspective(),
                Some(skills) => {
                    let mut e = SigmaEngine::new();
                    for s in skills {
                        e.compile_skill(s.clone())?;
                    }
                    e
                }
            };
            for (name, _) in &other.evidence {
                if !view.index.contains_key(name) {
                    view.register_predicate(&PredicateSchema::binary(name, PredicateRole::Context))?;
                }
            }
            for g in &other.goals {
                view.push_goal(g);
            }
            let mut map = Vec::with_capacity(view.vars.len());
            let mut mine = HashMap::new();
            for v in &view.vars {
                let name = if is_shared(&v.schema.name) {
                    v.schema.name.clone()
                } else {
                    format!("{}.{}", other.name, v.schema.name)
                };
                let schema = PredicateSchema { name: name.clone(), ..v.schema.clone() };
                let j = match joint.index.get(&name) {
                    // A shared variable keeps its first role; only its domain must agree.
                    Some(&j) if joint.vars[j].schema.domain_size != schema.domain_size => {
                        return Err("shared variable domains differ");
                    }
                    Some(&j) => j,
                    None => joint.register_predicate(&schema)?,
                };
                if joint.vars[j].range.is_none() {
                    joint.vars[j].range = v.range;
                }
                mine.insert(v.schema.name.clone(), j);
                map.push(j);
            }
            for f in view.factors.iter().filter(|f| !f.memory) {
                let mut f = f.clone();
                f.vars = f.vars.iter().map(|&v| map[v]).collect();
                joint.factors.push(f);
            }
            for s in &view.skills {
                if !joint.skills.iter().any(|k| k.skill_name == s.skill_name) {
                    joint.skills.push(s.clone());
                }
            }
            for (k, w) in &view.initial {
                joint.initial.entry(k.clone()).or_insert_with(|| w.clone());
            }
            own.push(mine);
        }
        // `Does(other,a)` in one agent's view is the other agent's action `a`.
        let mut links = crate::graph_memory::Builder::new("@joint");
        let c = PatternKind::Condact;
        for (i, mine) in own.iter().enumerate() {
            for (name, &v) in mine {
                let Some((who, action)) =
                    name.strip_prefix("Does(").and_then(|r| r.strip_suffix(')')).and_then(|r| r.split_once(','))
                else {
                    continue;
                };
                let Some(k) = others.iter().position(|o| o.name == who).filter(|&k| k != i) else { continue };
                let Some(&a) = own[k].get(action) else { continue };
                if joint.vars[a].schema.role == PredicateRole::Action && joint.vars[v].schema.domain_size == 2 {
                    links.add(vec![v, a], vec![c, c], &[2, 2], vec![0.999, 0.001, 0.001, 0.999], None);
                }
            }
        }
        let tied = links.factors.len();
        for mut f in links.factors {
            f.memory = false;
            f.learnable = false;
            joint.factors.push(f);
        }
        // Each agent's actions, as (joint index, its own name).
        let actions: Vec<Vec<(usize, String)>> = own
            .iter()
            .map(|mine| {
                let mut a: Vec<(usize, String)> = mine
                    .iter()
                    .filter(|&(_, &j)| joint.vars[j].schema.role == PredicateRole::Action)
                    .map(|(n, &j)| (j, n.clone()))
                    .collect();
                a.sort();
                a
            })
            .collect();
        let mut predicted: Vec<Option<String>> = vec![None; others.len()];
        let mut models: Vec<MindModel> = Vec::new();
        let (mut done, mut converged) = (0, false);
        while done < rounds.max(1) {
            done += 1;
            let mut next = predicted.clone();
            models.clear();
            for (i, other) in others.iter().enumerate() {
                let mut e = SigmaEngine {
                    vars: joint.vars.clone(),
                    index: joint.index.clone(),
                    factors: joint.factors.clone(),
                    skills: joint.skills.clone(),
                    initial: joint.initial.clone(),
                    ..SigmaEngine::default()
                };
                for (k, acts) in actions.iter().enumerate().filter(|&(k, _)| k != i) {
                    for (j, name) in acts {
                        // Another agent's actions are not ours to choose; its prediction is evidence.
                        e.vars[*j].schema.role = PredicateRole::Context;
                        if let Some(p) = &predicted[k] {
                            e.vars[*j].evidence = Some(usize::from(p == name));
                        }
                    }
                }
                // What any agent observes of the shared world holds for all; the rest is its own.
                for (k, o) in others.iter().enumerate() {
                    for (name, x) in o.evidence.iter().filter(|(n, _)| k == i || is_shared(n)) {
                        e.vars[own[k][name]].evidence = Some(*x);
                    }
                }
                for g in &other.goals {
                    let name = e.vars[own[i][g]].schema.name.clone();
                    e.push_goal(&name);
                }
                let out = e.cognitive_cycle();
                let prefix = format!("{}.", other.name);
                let unprefix = |n: &str| n.strip_prefix(&prefix).unwrap_or(n).to_string();
                let mine: HashSet<String> =
                    actions[i].iter().map(|(j, _)| joint.vars[*j].schema.name.clone()).collect();
                next[i] = out.selected_action.as_deref().filter(|a| mine.contains(*a)).map(unprefix);
                models.push(MindModel {
                    agent: other.name.clone(),
                    predicted_action: next[i].clone(),
                    action_beliefs: out
                        .beliefs
                        .into_iter()
                        .filter(|(n, _)| mine.contains(n))
                        .map(|(n, p)| (unprefix(&n), p.get(1).copied().unwrap_or(0.0)))
                        .collect(),
                    nested: Vec::new(),
                });
            }
            let same = next == predicted;
            predicted = next;
            if same {
                converged = true;
                break;
            }
        }
        Ok(JointModel { agents: models, rounds: done, converged, variables: joint.vars.len(), links: tied })
    }

    /// Our knowledge without our working memory, goals, memories or learned models.
    fn perspective(&self) -> SigmaEngine {
        let mut e = SigmaEngine {
            vars: self.vars.clone(),
            index: self.index.clone(),
            factors: self.factors.iter().filter(|f| !f.memory).cloned().collect(),
            skills: self.skills.clone(),
            initial: self.initial.clone(),
            ..SigmaEngine::default()
        };
        e.vars.iter_mut().for_each(|v| v.evidence = None);
        e
    }

    /// Attention: when a budget is set, only the most salient factors take part in this
    /// cycle's elaboration. Salience is the posterior of `@Attend(f)` in an attention graph
    /// solved by the engine's message passing: evidence from the factor's utility and its
    /// relevance (on a path from a stacked goal or not), raised by the surprise of its
    /// variables, which factors sharing a variable share.
    pub(crate) fn attend(&self, surprise: &[f32]) -> Option<Vec<bool>> {
        let k = self.attention_budget?;
        let relevant: HashSet<usize> =
            self.goal_stack.iter().filter_map(|g| self.index.get(g)).flat_map(|&v| self.reachable(v).0).collect();
        // Attention is inferred by message passing, in a graph of its own: `@Attend(f)` for each
        // skill factor, with its utility and its relevance to the stacked goals as evidence, and
        // `@Novel(v)` for each surprising variable, which raises the attention of every factor
        // that touches it.
        let mut g = SigmaEngine::new();
        let mut b = crate::graph_memory::Builder::new("@attention");
        let var = |g: &mut SigmaEngine, name: String| {
            g.register_predicate(&crate::PredicateSchema { name, domain_size: 2, role: crate::PredicateRole::Context })
                .expect("fresh names")
        };
        let c = crate::PatternKind::Condact;
        let mut attend: Vec<(usize, usize)> = Vec::new();
        let mut novel: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for (fi, f) in self.factors.iter().enumerate().filter(|(_, f)| !f.memory) {
            let a = var(&mut g, format!("@Attend({fi})"));
            attend.push((fi, a));
            let u = f.utility_score.clamp(0.01, 0.99);
            b.add(vec![a], vec![c], &[2], vec![1.0 - u, u], None);
            let r = if relevant.contains(&fi) { 0.8 } else { 0.2 };
            b.add(vec![a], vec![c], &[2], vec![1.0 - r, r], None);
            for &v in &f.vars {
                let s = surprise.get(v).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                if s <= 0.0 {
                    continue;
                }
                let n = *novel.entry(v).or_insert_with(|| {
                    let n = var(&mut g, format!("@Novel({v})"));
                    b.add(vec![n], vec![c], &[2], vec![1.0 - s, s], None);
                    n
                });
                // A novel variable doubles the odds of attending to the factors it is in.
                b.add(vec![n, a], vec![c, c], &[2, 2], vec![1.0, 1.0, 1.0, 2.0], None);
            }
        }
        g.factors.extend(b.factors);
        let p = g.infer(|_| None);
        let mut ranked: Vec<(usize, f32)> = attend.iter().map(|&(fi, a)| (fi, p[a][1])).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        // Memories and models are always attended and take no part of the budget.
        let mut mask: Vec<bool> = self.factors.iter().map(|f| f.memory).collect();
        ranked.iter().take(k).for_each(|&(fi, _)| mask[fi] = true);
        Some(mask)
    }
}
