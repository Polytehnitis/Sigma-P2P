//! Long-term memories beside the skill graph: episodic (what happened, when) and semantic
//! (what kinds of things there are). Both are learned from working memory, and both are
//! recalled by Bayesian partial matching against a cue, as in Sigma.

use serde::{Deserialize, Serialize};
use std::collections::btree_map::Entry;
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

/// Episodes are timestamped with a Lamport clock, so replicas of one memory on several of the
/// user's devices can be merged (see [`merge`](Self::merge)) and still order sensibly.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub episodes: VecDeque<Episode>,
    clock: u64,
    /// Identity of the episode this replica recorded last (not persisted), so feedback never
    /// lands on an episode merged in from another device.
    #[serde(skip)]
    last: Option<Vec<u8>>,
}

impl PartialEq for EpisodicMemory {
    fn eq(&self, other: &Self) -> bool {
        self.episodes == other.episodes && self.clock == other.clock
    }
}

/// What makes two episodes the same one: everything but the (later attached) reward.
fn identity(e: &Episode) -> Vec<u8> {
    postcard::to_allocvec(&(e.time, &e.goals, &e.state, &e.action)).expect("in-memory encoding cannot fail")
}

/// Between two copies of one episode, the rewarded one (then the larger reward) wins.
fn preferred(a: Episode, b: Episode) -> Episode {
    let rewarded = |e: &Episode| e.reward != 0.0;
    match rewarded(&a).cmp(&rewarded(&b)).then(a.reward.total_cmp(&b.reward)) {
        std::cmp::Ordering::Less => b,
        _ => a,
    }
}

impl EpisodicMemory {
    pub fn record(&mut self, goals: Vec<String>, state: Vec<(String, usize)>, action: Option<String>) {
        if self.episodes.len() == EPISODIC_CAPACITY {
            self.episodes.pop_front();
        }
        self.clock += 1;
        let episode = Episode { time: self.clock, goals, state, action, reward: 0.0 };
        self.last = Some(identity(&episode));
        self.episodes.push_back(episode);
    }

    /// Attaches feedback to the episode this replica recorded last (after a restart: the newest).
    pub fn reward_last(&mut self, reward: f32) {
        let e = match &self.last {
            Some(id) => self.episodes.iter_mut().rev().find(|e| identity(e) == *id),
            None => self.episodes.back_mut(),
        };
        if let Some(e) = e {
            e.reward = reward;
        }
    }

    /// CRDT join with another replica: the union of both, newest [`EPISODIC_CAPACITY`] kept.
    /// Commutative, associative and idempotent, so devices converge in any sync order.
    /// Returns whether `self` changed.
    pub fn merge(&mut self, other: &EpisodicMemory) -> bool {
        let mut all: BTreeMap<(u64, Vec<u8>), Episode> = BTreeMap::new();
        for e in self.episodes.iter().chain(&other.episodes).cloned() {
            match all.entry((e.time, identity(&e))) {
                Entry::Vacant(v) => _ = v.insert(e),
                Entry::Occupied(mut o) => {
                    let old = o.get().clone();
                    o.insert(preferred(old, e));
                }
            }
        }
        let skip = all.len().saturating_sub(EPISODIC_CAPACITY);
        let episodes: VecDeque<Episode> = all.into_values().skip(skip).collect();
        let clock = self.clock.max(other.clock);
        let changed = episodes != self.episodes || clock != self.clock;
        self.episodes = episodes;
        self.clock = clock;
        changed
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

    /// CRDT join with another replica: every count becomes the larger of the two (a max-register
    /// per count). Re-syncing never double-counts; learning of the same concept on two devices
    /// while they are apart keeps the larger tally. Returns whether `self` changed.
    pub fn merge(&mut self, other: &SemanticMemory) -> bool {
        let before = self.clone();
        for (name, theirs) in &other.concepts {
            let mine = self.concepts.entry(name.clone()).or_default();
            mine.count = mine.count.max(theirs.count);
            for (f, counts) in &theirs.features {
                let own = mine.features.entry(f.clone()).or_default();
                if own.len() < counts.len() {
                    own.resize(counts.len(), 0.0);
                }
                own.iter_mut().zip(counts).for_each(|(a, b)| *a = a.max(*b));
            }
        }
        for (f, d) in &other.domains {
            let own = self.domains.entry(f.clone()).or_insert(0);
            *own = (*own).max(*d);
        }
        *self != before
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
