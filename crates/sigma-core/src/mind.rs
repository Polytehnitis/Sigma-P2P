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
    pub(crate) fn label(mut self, stuck: bool) -> Self {
        self.emotion = match () {
            _ if self.desirability > 0.3 => Emotion::Joy,
            _ if self.desirability < -0.3 => Emotion::Distress,
            _ if self.surprise > 0.5 => Emotion::Surprise,
            _ if stuck => Emotion::Fear,
            _ if self.control > 0.7 => Emotion::Hope,
            _ => Emotion::Neutral,
        };
        self
    }

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

    /// Our knowledge without our working memory, goals or memories.
    fn perspective(&self) -> SigmaEngine {
        let mut e = SigmaEngine {
            vars: self.vars.clone(),
            index: self.index.clone(),
            factors: self.factors.clone(),
            skills: self.skills.clone(),
            initial: self.initial.clone(),
            ..SigmaEngine::default()
        };
        e.vars.iter_mut().for_each(|v| v.evidence = None);
        e
    }

    /// Attention: when a budget is set, only the most salient factors take part in this
    /// cycle's elaboration. Salience is utility, times relevance (on a path from a stacked
    /// goal: 1, otherwise 0.25), times one plus the surprise of the factor's variables.
    pub(crate) fn attend(&self, surprise: &[f32]) -> Option<Vec<bool>> {
        let k = self.attention_budget?;
        let relevant: HashSet<usize> =
            self.goal_stack.iter().filter_map(|g| self.index.get(g)).flat_map(|&v| self.reachable(v).0).collect();
        let mut ranked: Vec<(usize, f32)> = self
            .factors
            .iter()
            .enumerate()
            .map(|(fi, f)| {
                let relevance = if relevant.contains(&fi) { 1.0 } else { 0.25 };
                let novelty = 1.0 + f.vars.iter().map(|&v| surprise[v]).fold(0.0, f32::max);
                (fi, f.utility_score.max(0.01) * relevance * novelty)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut mask = vec![false; self.factors.len()];
        ranked.iter().take(k).for_each(|&(fi, _)| mask[fi] = true);
        Some(mask)
    }
}
