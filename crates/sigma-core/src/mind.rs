//! Higher functions on top of the decision cycle: attention, appraisal/emotion and
//! Theory of Mind. Each one reuses the same graph and inference; none adds new mechanisms.

use crate::{PredicateRole, PredicateSchema, SigmaEngine, Skill};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

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
