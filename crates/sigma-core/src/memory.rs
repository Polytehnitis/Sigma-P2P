//! Long-term memories beside the skill graph: episodic (what happened, when) and semantic
//! (what kinds of things there are). Both are learned from working memory, and both are
//! recalled by Bayesian partial matching against a cue, as in Sigma.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// Episodes kept before the oldest is forgotten.
pub const EPISODIC_CAPACITY: usize = 1024;
/// Chance that a stored value is misremembered (keeps partial matches alive).
const NOISE: f32 = 0.05;
/// Per-cycle decay of the recency prior.
const RECENCY: f32 = 0.01;

/// One decision cycle as the agent lived it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    pub time: u64,
    pub goals: Vec<String>,
    /// Working-memory values (evidence and perceived values) during the cycle.
    pub state: Vec<(String, usize)>,
    pub action: Option<String>,
    /// Reward given for this cycle's decision (0 until feedback arrives).
    pub reward: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EpisodeRecall {
    pub episode: Episode,
    /// Posterior of the best match among all stored episodes.
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub episodes: VecDeque<Episode>,
    clock: u64,
}

impl EpisodicMemory {
    pub fn record(&mut self, goals: Vec<String>, state: Vec<(String, usize)>, action: Option<String>) {
        if self.episodes.len() == EPISODIC_CAPACITY {
            self.episodes.pop_front();
        }
        self.clock += 1;
        self.episodes.push_back(Episode { time: self.clock, goals, state, action, reward: 0.0 });
    }

    /// Attaches feedback to the most recent episode.
    pub fn reward_last(&mut self, reward: f32) {
        if let Some(e) = self.episodes.back_mut() {
            e.reward = reward;
        }
    }

    /// The episode that best explains `cue` (goals plus values), with recency as the prior.
    /// `filter` restricts which episodes may be recalled.
    pub fn recall(
        &self,
        goals: &[String],
        cue: &[(String, usize)],
        filter: impl Fn(&Episode) -> bool,
    ) -> Option<EpisodeRecall> {
        let scores: Vec<f32> = self
            .episodes
            .iter()
            .map(|e| {
                if !filter(e) {
                    return 0.0;
                }
                let age = (self.clock - e.time) as f32;
                let goal_match = goals.iter().map(|g| if e.goals.contains(g) { 1.0 - NOISE } else { NOISE });
                let value_match = cue.iter().map(|(n, x)| match e.state.iter().find(|(m, _)| m == n) {
                    Some((_, y)) if y == x => 1.0 - NOISE,
                    Some(_) => NOISE,
                    None => 0.5,
                });
                goal_match.chain(value_match).product::<f32>() * (-RECENCY * age).exp()
            })
            .collect();
        let total: f32 = scores.iter().sum();
        // Most recent wins ties, so scan newest first.
        let (i, best) = scores.iter().enumerate().rev().max_by(|a, b| a.1.total_cmp(b.1))?;
        (*best > 0.0).then(|| EpisodeRecall { episode: self.episodes[i].clone(), confidence: best / total })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct Concept {
    count: f32,
    /// feature -> count of each value
    features: BTreeMap<String, Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticRecall {
    /// Posterior over categories, most likely first.
    pub categories: Vec<(String, f32)>,
    /// Feature not in the cue -> predicted distribution over its values.
    pub predicted: Vec<(String, Vec<f32>)>,
}

/// A naive-Bayes concept memory: category -> independent feature distributions.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SemanticMemory {
    concepts: BTreeMap<String, Concept>,
    /// Domain size of every feature seen.
    domains: BTreeMap<String, usize>,
}

impl SemanticMemory {
    /// Learns one observed object of `category`.
    pub fn learn(&mut self, category: &str, features: &[(String, usize)]) {
        let c = self.concepts.entry(category.into()).or_default();
        c.count += 1.0;
        for (f, x) in features {
            let d = self.domains.entry(f.clone()).or_insert(0);
            *d = (*d).max(x + 1).max(2);
            let counts = c.features.entry(f.clone()).or_default();
            if counts.len() <= *x {
                counts.resize(x + 1, 0.0);
            }
            counts[*x] += 1.0;
        }
    }

    /// P(value | category) with add-one smoothing.
    fn likelihood(&self, c: &Concept, feature: &str, x: usize) -> f32 {
        let d = self.domains.get(feature).copied().unwrap_or(2).max(x + 1) as f32;
        let seen = c.features.get(feature).and_then(|v| v.get(x)).copied().unwrap_or(0.0);
        (seen + 1.0) / (c.count + d)
    }

    /// Classifies a partial cue and completes the features it lacks.
    pub fn recall(&self, cue: &[(String, usize)]) -> Option<SemanticRecall> {
        let total: f32 = self.concepts.values().map(|c| c.count).sum();
        if total == 0.0 {
            return None;
        }
        let mut post: Vec<(String, f32)> = self
            .concepts
            .iter()
            .map(|(name, c)| {
                let p = cue.iter().fold(c.count / total, |p, (f, x)| p * self.likelihood(c, f, *x));
                (name.clone(), p)
            })
            .collect();
        let z: f32 = post.iter().map(|p| p.1).sum();
        post.iter_mut().for_each(|p| p.1 /= z.max(f32::MIN_POSITIVE));
        post.sort_by(|a, b| b.1.total_cmp(&a.1));
        let predicted = self
            .domains
            .iter()
            .filter(|(f, _)| !cue.iter().any(|(g, _)| g == *f))
            .map(|(f, &d)| {
                let dist = (0..d)
                    .map(|x| post.iter().map(|(n, p)| p * self.likelihood(&self.concepts[n], f, x)).sum::<f32>())
                    .collect();
                (f.clone(), crate::normalize(dist))
            })
            .collect();
        Some(SemanticRecall { categories: post, predicted })
    }
}
