//! Language understanding: what the user says (spoken or typed) becomes Sigma goals,
//! observations and commands.
//!
//! Interpretation runs entirely on the device and in two stages:
//!
//! 1. **Fast path.** Phrases the user has taught ([`PhraseBook`]) match exactly. Then fixed
//!    command phrases ("undo that", "what next?") and the engine's own predicates are matched
//!    word by word. A predicate's words come from its name (`SubgoalMountNAS` -> "mount nas"),
//!    with light stemming, a few synonyms and an edit-distance tolerance for recognition slips.
//! 2. **Inference.** When the fast path is not sure, a naive-Bayes classifier (the same model
//!    as Sigma's semantic memory) weighs every word against everything it has seen: taught
//!    phrases, command phrases and predicate names. It returns a posterior over intents, and
//!    the best one is used only if it is confident enough. Otherwise the alternatives are
//!    returned so the UI can ask "did you mean…?", and the answer can be taught.
//!
//! Safety. Language never selects an action and never confirms one. Action predicates are
//! never a target, and [`Intent::ConfirmRequested`] carries nothing: the UI answers it by
//! pointing at its own Confirm button, which alone holds the confirmation token.

use crate::{PredicateRole, PredicateSchema};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Longest utterance considered (characters); the rest is ignored.
pub const MAX_UTTERANCE: usize = 500;
/// Longest phrase that can be taught (characters, after normalization).
pub const MAX_PHRASE: usize = 120;
/// Most taught phrases kept.
pub const MAX_PHRASES: usize = 512;
/// Fast-path matches at or above this score are taken without inference.
pub const FAST_PATH_SCORE: f32 = 0.75;
/// The fast path also needs this lead over the runner-up.
pub const FAST_PATH_MARGIN: f32 = 0.15;
/// Posterior an inferred intent needs to be used.
pub const INFERENCE_POSTERIOR: f32 = 0.55;
/// Added to a predicate's match when the parse's core alone matches it.
const CORE_BONUS: f32 = 0.15;
/// Below this word similarity, two words do not match.
const WORD_MATCH: f32 = 0.75;

/// What the user wants done.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Intent {
    /// Push a top-level goal.
    SetGoal(String),
    /// The user reports that a predicate holds (`value` 1) or not (0).
    Observe { predicate: String, value: usize },
    /// Run a decision cycle.
    Think,
    /// Reward for the last decision: `true` is good (+1), `false` bad (-1).
    Feedback(bool),
    /// Undo the last undoable action.
    Undo,
    /// Drop the action awaiting confirmation.
    Cancel,
    /// The user said "yes, do it". Language cannot confirm: the UI shows its Confirm button.
    ConfirmRequested,
    /// Open the demonstration dialog.
    Teach,
    /// Say what Sigma is doing and why.
    Explain,
}

/// How an interpretation was reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Route {
    /// A phrase the user taught.
    Taught,
    /// Command or predicate words matched directly.
    FastPath,
    /// The naive-Bayes posterior settled it.
    Inference,
    /// Nothing was confident enough; see `alternatives`.
    NoMatch,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Interpretation {
    /// The normalized words that were interpreted.
    pub normalized: String,
    pub intent: Option<Intent>,
    /// Confidence in `intent` (in [0, 1]), recognition confidence included.
    pub confidence: f32,
    pub route: Route,
    /// Other likely readings, best first (at most 3), for "did you mean…?".
    pub alternatives: Vec<(Intent, f32)>,
}

/// Phrases the user taught, normalized phrase -> intent. Persisted and synced between the
/// user's paired devices (see `FactorStore::save_phrase`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PhraseBook {
    pub phrases: BTreeMap<String, Intent>,
}

impl PhraseBook {
    /// Adds or replaces a phrase. Returns its normalized form, or `None` if it is empty, too
    /// long, or the book is full.
    pub fn teach(&mut self, phrase: &str, intent: Intent) -> Option<String> {
        let key = normalize(phrase);
        if key.is_empty() || key.len() > MAX_PHRASE {
            return None;
        }
        if !self.phrases.contains_key(&key) && self.phrases.len() >= MAX_PHRASES {
            return None;
        }
        self.phrases.insert(key.clone(), intent);
        Some(key)
    }

    pub fn forget(&mut self, phrase: &str) -> bool {
        self.phrases.remove(&normalize(phrase)).is_some()
    }
}

/// Lowercase words, contractions expanded ("isn't" -> "is not"), punctuation dropped.
pub fn normalize(text: &str) -> String {
    let text: String = text.chars().take(MAX_UTTERANCE).collect::<String>().to_lowercase();
    let text = text.replace(['’', '‘'], "'").replace("can't", "can not").replace("won't", "will not");
    let text = text.replace("n't", " not").replace("'re", " are").replace("'m", " am").replace("'ll", " will");
    let text = text.replace("let's", "let us").replace("'s", "").replace("'ve", " have").replace("'d", " would");
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ")
}

/// Light suffix stripping, enough to tie "mounted", "mounting" and "mounts" to "mount", then
/// a few synonyms.
fn stem(word: &str) -> String {
    let base = strip_suffix(word);
    synonym(&base).map_or(base, String::from)
}

fn strip_suffix(word: &str) -> String {
    let w = word;
    let strip = |suffix: &str, min: usize| w.len() >= min + suffix.len() && w.ends_with(suffix);
    let plural_es = w.ends_with("ses") || w.ends_with("xes") || w.ends_with("ches");
    let base = if strip("ing", 3) {
        &w[..w.len() - 3]
    } else if strip("ed", 3) || (strip("es", 3) && plural_es) {
        &w[..w.len() - 2]
    } else if strip("s", 3) && !w.ends_with("ss") && !w.ends_with("us") && !w.ends_with("is") {
        &w[..w.len() - 1]
    } else {
        w
    };
    // "stopped" -> "stopp" -> "stop"
    let b = base.as_bytes();
    let base = if b.len() > 3 && b[b.len() - 1] == b[b.len() - 2] && !b"lsz".contains(&b[b.len() - 1]) {
        &base[..base.len() - 1]
    } else {
        base
    };
    base.to_string()
}

/// A few everyday words for what predicate names usually say.
fn synonym(word: &str) -> Option<&'static str> {
    Some(match word {
        "connect" | "attach" | "plug" => "mount",
        "disconnect" | "detach" | "unplug" | "eject" => "unmount",
        "server" => "nas",
        "switch" | "turn" => "power",
        "complete" | "finish" | "finished" | "done" => "done",
        _ => return None,
    })
}

/// An utterance that starts like this is a question ("is the NAS mounted"), not a report.
const QUESTION_STARTS: &[&str] = &["is", "are", "was", "were", "did", "does", "do", "has", "have", "can", "will"];
const NEGATIONS: &[&str] = &["not", "no", "never", "without", "nothing"];
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "my", "your", "our", "to", "of", "for", "and", "or", "it", "this", "that", "on", "in", "at",
    "is", "are", "was", "were", "be", "been", "am", "i", "you", "we", "me", "us", "please", "can", "could", "would",
    "will", "should", "do", "does", "did", "have", "has", "had", "now", "already", "just", "want", "need", "like",
    "let", "go", "some", "all", "so", "then", "there", "here", "with", "from", "up", "hey", "sigma", "ok", "okay",
];
/// Words that make an utterance a request ("I want to back up").
const GOAL_CUES: &[&str] =
    &["want", "need", "please", "let", "goal", "should", "could", "can", "would", "like", "make", "start", "try"];
/// Words that make it a report ("the NAS is mounted").
const OBSERVATION_CUES: &[&str] =
    &["is", "are", "was", "were", "has", "have", "been", "already", "now", "done", "ready", "finished", "got"];

/// Fixed command phrases, normalized.
const COMMANDS: &[(&str, Cmd)] = &[
    ("undo", Cmd::Undo),
    ("undo that", Cmd::Undo),
    ("undo it", Cmd::Undo),
    ("take that back", Cmd::Undo),
    ("take it back", Cmd::Undo),
    ("revert", Cmd::Undo),
    ("revert that", Cmd::Undo),
    ("reverse that", Cmd::Undo),
    ("put it back", Cmd::Undo),
    ("cancel", Cmd::Cancel),
    ("cancel that", Cmd::Cancel),
    ("never mind", Cmd::Cancel),
    ("nevermind", Cmd::Cancel),
    ("do not do that", Cmd::Cancel),
    ("do not do it", Cmd::Cancel),
    ("no do not", Cmd::Cancel),
    ("stop", Cmd::Cancel),
    ("abort", Cmd::Cancel),
    ("forget it", Cmd::Cancel),
    ("yes", Cmd::Confirm),
    ("yeah", Cmd::Confirm),
    ("yep", Cmd::Confirm),
    ("sure", Cmd::Confirm),
    ("confirm", Cmd::Confirm),
    ("confirm that", Cmd::Confirm),
    ("do it", Cmd::Confirm),
    ("go ahead", Cmd::Confirm),
    ("approve", Cmd::Confirm),
    ("yes do it", Cmd::Confirm),
    ("ok do it", Cmd::Confirm),
    ("think", Cmd::Think),
    ("think about it", Cmd::Think),
    ("what next", Cmd::Think),
    ("what now", Cmd::Think),
    ("next", Cmd::Think),
    ("next step", Cmd::Think),
    ("what should i do", Cmd::Think),
    ("what should you do", Cmd::Think),
    ("what would you do", Cmd::Think),
    ("continue", Cmd::Think),
    ("carry on", Cmd::Think),
    ("keep going", Cmd::Think),
    ("good", Cmd::Good),
    ("good job", Cmd::Good),
    ("good one", Cmd::Good),
    ("well done", Cmd::Good),
    ("thanks", Cmd::Good),
    ("thank you", Cmd::Good),
    ("great", Cmd::Good),
    ("nice", Cmd::Good),
    ("perfect", Cmd::Good),
    ("correct", Cmd::Good),
    ("that was right", Cmd::Good),
    ("that is right", Cmd::Good),
    ("bad", Cmd::Bad),
    ("bad idea", Cmd::Bad),
    ("wrong", Cmd::Bad),
    ("that is wrong", Cmd::Bad),
    ("that was wrong", Cmd::Bad),
    ("not good", Cmd::Bad),
    ("no", Cmd::Bad),
    ("incorrect", Cmd::Bad),
    ("do not do that again", Cmd::Bad),
    ("teach", Cmd::Teach),
    ("let me teach you", Cmd::Teach),
    ("let us teach you", Cmd::Teach),
    ("i will teach you", Cmd::Teach),
    ("let me show you", Cmd::Teach),
    ("i will show you", Cmd::Teach),
    ("learn this", Cmd::Teach),
    ("learn a skill", Cmd::Teach),
    ("why", Cmd::Explain),
    ("explain", Cmd::Explain),
    ("status", Cmd::Explain),
    ("what are you doing", Cmd::Explain),
    ("what are you thinking", Cmd::Explain),
    ("what is going on", Cmd::Explain),
    ("what is happening", Cmd::Explain),
    ("how are you", Cmd::Explain),
    ("why did you do that", Cmd::Explain),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cmd {
    Undo,
    Cancel,
    Confirm,
    Think,
    Good,
    Bad,
    Teach,
    Explain,
}

impl Cmd {
    fn intent(self) -> Intent {
        match self {
            Cmd::Undo => Intent::Undo,
            Cmd::Cancel => Intent::Cancel,
            Cmd::Confirm => Intent::ConfirmRequested,
            Cmd::Think => Intent::Think,
            Cmd::Good => Intent::Feedback(true),
            Cmd::Bad => Intent::Feedback(false),
            Cmd::Teach => Intent::Teach,
            Cmd::Explain => Intent::Explain,
        }
    }
}

/// Splits a predicate name into lowercase words: `SubgoalMountNAS` -> ["subgoal", "mount", "nas"].
pub fn name_words(name: &str) -> Vec<String> {
    let chars: Vec<char> = name.chars().collect();
    let mut words = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if !c.is_alphanumeric() {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            continue;
        }
        let prev = i.checked_sub(1).map(|j| chars[j]);
        let next = chars.get(i + 1);
        let boundary = match prev {
            Some(p) if p.is_alphanumeric() => {
                (c.is_uppercase() && p.is_lowercase())
                    || (c.is_uppercase() && p.is_uppercase() && next.is_some_and(|n| n.is_lowercase()))
                    || (c.is_ascii_digit() != p.is_ascii_digit())
            }
            _ => false,
        };
        if boundary && !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
        }
        cur.extend(c.to_lowercase());
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

/// The content words of a predicate name, its role prefix dropped (`GoalBackup` -> ["backup"]).
fn predicate_words(p: &PredicateSchema) -> Vec<String> {
    content_words(p).iter().map(|w| stem(w)).collect()
}

/// [`predicate_words`] before stemming, as a person would say them.
fn content_words(p: &PredicateSchema) -> Vec<String> {
    let mut words = name_words(&p.name);
    let prefix = match p.role {
        PredicateRole::Goal => &["goal"][..],
        PredicateRole::Subgoal => &["subgoal", "sub"][..],
        PredicateRole::Context => &["context", "ctx", "is", "has"][..],
        PredicateRole::Action => &["action"][..],
    };
    if words.len() > 1 && prefix.contains(&words[0].as_str()) {
        words.remove(0);
    }
    if words.len() > 1 && words[0] == "goal" {
        words.remove(0); // "SubgoalGoal…" is unusual, but harmless
    }
    words
}

/// Longest [`recognition_hint`], in bytes. whisper.cpp keeps at most about 224 prompt tokens.
pub const MAX_HINT: usize = 400;

/// A hint for the speech recognizer, as a short sentence of what the user is likely to say:
/// Sigma's command words, then the user's taught phrases and the engine's own predicates. It
/// biases recognition towards them (whisper.cpp's initial prompt), so "stop" or "mount the NAS"
/// are heard as such. Never longer than [`MAX_HINT`].
pub fn recognition_hint(predicates: &[PredicateSchema], phrases: &PhraseBook) -> String {
    const SPOKEN_COMMANDS: &[&str] =
        &["stop", "cancel", "undo that", "yes", "no", "think", "what next", "teach", "explain", "good", "bad"];
    let mut seen = std::collections::BTreeSet::new();
    let mut hint = String::from("Sigma");
    let taught = phrases.phrases.keys().map(String::as_str).map(str::to_owned);
    let named = targets(predicates).map(|p| content_words(p).join(" "));
    for item in SPOKEN_COMMANDS.iter().map(|c| c.to_string()).chain(taught).chain(named) {
        if item.is_empty() || !seen.insert(item.clone()) {
            continue;
        }
        if hint.len() + item.len() + 3 > MAX_HINT {
            break;
        }
        hint.push_str(", ");
        hint.push_str(&item);
    }
    hint.push('.');
    hint
}

/// Similarity of two stems in [0, 1]: equal, prefix of each other, or few edits apart.
fn similarity(a: &str, b: &str) -> f32 {
    if a == b {
        return 1.0;
    }
    let (la, lb) = (a.chars().count(), b.chars().count());
    if la.min(lb) >= 4 && (a.starts_with(b) || b.starts_with(a)) {
        return 0.85;
    }
    if la.max(lb) < 4 {
        return 0.0;
    }
    1.0 - levenshtein(a, b) as f32 / la.max(lb) as f32
}

fn levenshtein(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diag = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let next = (diag + usize::from(ca != cb)).min(row[j] + 1).min(row[j + 1] + 1);
            diag = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
}

/// An utterance prepared for matching.
struct Utterance {
    words: Vec<String>,
    /// Stems of every word, plus joined neighbours ("back up" -> "backup").
    stems: Vec<String>,
    negated: bool,
    goal_cue: bool,
    observation_cue: bool,
    question: bool,
}

impl Utterance {
    fn new(normalized: &str) -> Self {
        let words: Vec<String> = normalized.split(' ').filter(|w| !w.is_empty()).map(String::from).collect();
        let mut stems: Vec<String> = words.iter().map(|w| stem(w)).collect();
        for pair in words.windows(2) {
            stems.push(stem(&format!("{}{}", pair[0], pair[1])));
            stems.push(stem(&format!("{}{}", strip_suffix(&pair[0]), strip_suffix(&pair[1]))));
        }
        let has = |set: &[&str]| words.iter().any(|w| set.contains(&w.as_str()));
        // A past participle reports a state: "I mounted the NAS".
        let participle = words.iter().any(|w| w.len() > 4 && w.ends_with("ed"));
        Self {
            negated: has(NEGATIONS),
            goal_cue: has(GOAL_CUES),
            observation_cue: has(OBSERVATION_CUES) || participle,
            question: words.first().is_some_and(|w| QUESTION_STARTS.contains(&w.as_str())),
            stems,
            words,
        }
    }

    /// Words that carry meaning (stopwords, negations and cue words dropped).
    fn content(&self) -> Vec<String> {
        self.words
            .iter()
            .filter(|w| !STOPWORDS.contains(&w.as_str()) && !NEGATIONS.contains(&w.as_str()))
            .map(|w| stem(w))
            .collect()
    }

    /// How well the predicate's words are covered (0 when any word is missing), and whether a
    /// word only matched through an "un" prefix ("unmounted" for `Mount…`).
    fn match_words(&self, words: &[String]) -> (f32, bool) {
        if words.is_empty() {
            return (0.0, false);
        }
        let mut total = 0.0;
        let mut flipped = false;
        for w in words {
            let direct = self.stems.iter().map(|s| similarity(s, w)).fold(0.0, f32::max);
            let un = self
                .stems
                .iter()
                .filter_map(|s| s.strip_prefix("un").or_else(|| s.strip_prefix("dis")))
                .map(|s| similarity(&stem(s), w))
                .fold(0.0, f32::max);
            let best = direct.max(un);
            if best < WORD_MATCH {
                return (0.0, false);
            }
            flipped |= un > direct;
            total += best;
        }
        let recall = total / words.len() as f32;
        // Extra content words lower the score a little ("back up my photos to the NAS").
        let content = self.content().len().max(words.len());
        let precision = words.len() as f32 / content as f32;
        (recall * (0.8 + 0.2 * precision), flipped)
    }
}

/// Reads one predicate match as a goal or an observation.
fn predicate_intent(p: &PredicateSchema, u: &Utterance, flipped: bool) -> Option<Intent> {
    let negated = u.negated ^ flipped;
    let value = usize::from(!negated);
    match p.role {
        PredicateRole::Action => None,
        PredicateRole::Goal if u.observation_cue && !u.goal_cue => {
            Some(Intent::Observe { predicate: p.name.clone(), value })
        }
        // "I don't want to back up" is not a request for a backup.
        PredicateRole::Goal if negated => None,
        PredicateRole::Goal => Some(Intent::SetGoal(p.name.clone())),
        _ => Some(Intent::Observe { predicate: p.name.clone(), value }),
    }
}

/// Every predicate language may refer to: binary, and never an action.
fn targets(predicates: &[PredicateSchema]) -> impl Iterator<Item = &PredicateSchema> {
    predicates.iter().filter(|p| p.role != PredicateRole::Action && p.domain_size == 2)
}

/// Keeps only intents the current predicates allow (a taught phrase may outlive its predicate).
fn allowed(intent: &Intent, predicates: &[PredicateSchema]) -> bool {
    let known = |name: &str| targets(predicates).any(|p| p.name == name);
    match intent {
        Intent::SetGoal(g) => known(g),
        Intent::Observe { predicate, value } => *value < 2 && known(predicate),
        _ => true,
    }
}

/// Interprets one utterance. `recognition_confidence` (in [0, 1]) is what the speech
/// recognizer reported; pass 1 for typed text.
pub fn interpret(
    text: &str,
    recognition_confidence: f32,
    predicates: &[PredicateSchema],
    phrases: &PhraseBook,
) -> Interpretation {
    let normalized = normalize(text);
    let rc = if recognition_confidence.is_finite() { recognition_confidence.clamp(0.0, 1.0) } else { 0.0 };
    let done = |intent: Intent, confidence: f32, route: Route, alternatives: Vec<(Intent, f32)>| Interpretation {
        normalized: normalized.clone(),
        intent: Some(intent),
        confidence: confidence * rc,
        route,
        alternatives,
    };
    if normalized.is_empty() {
        return Interpretation {
            normalized,
            intent: None,
            confidence: 0.0,
            route: Route::NoMatch,
            alternatives: vec![],
        };
    }
    if let Some(intent) = phrases.phrases.get(&normalized).filter(|i| allowed(i, predicates)) {
        return done(intent.clone(), 1.0, Route::Taught, vec![]);
    }

    let u = Utterance::new(&normalized);
    let mut scored: BTreeMap<Intent, f32> = BTreeMap::new();
    let mut add = |intent: Intent, score: f32| {
        let s = scored.entry(intent).or_insert(0.0);
        *s = s.max(score);
    };
    // Commands: the whole utterance, or a short utterance that contains the phrase.
    for (phrase, cmd) in COMMANDS {
        let p: Vec<&str> = phrase.split(' ').collect();
        let w: Vec<&str> = u.words.iter().map(String::as_str).collect();
        let score = if p == w {
            1.0
        } else if w.len() <= p.len() + 3 && w.windows(p.len()).any(|win| win == p.as_slice()) {
            0.55 + 0.4 * p.len() as f32 / w.len() as f32
        } else {
            0.0
        };
        if score > 0.0 {
            add(cmd.intent(), score);
        }
    }
    for p in targets(predicates) {
        let (mut score, flipped) = u.match_words(&predicate_words(p));
        if u.question && p.role != PredicateRole::Goal {
            // Never take a question for a fact; at most offer it as a reading.
            score -= 0.3;
        }
        if score > 0.0 {
            if let Some(intent) = predicate_intent(p, &u, flipped) {
                add(intent, score);
            }
        }
    }
    let mut ranked: Vec<(Intent, f32)> = scored.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if let Some((best, score)) = ranked.first().cloned() {
        let runner_up = ranked.get(1).map_or(0.0, |r| r.1);
        if score >= FAST_PATH_SCORE && score - runner_up >= FAST_PATH_MARGIN {
            return done(best, score, Route::FastPath, ranked.into_iter().skip(1).take(3).collect());
        }
    }

    // Parsing in the graph: the sentence's core (its root and content dependents) is matched
    // too, so filler around a command does not hide it.
    let core = crate::parse::Grammar::for_predicates(predicates)
        .parse(&u.words)
        .map(|p| p.core())
        .filter(|c| !c.is_empty() && c.len() < u.words.len())
        .map(|c| {
            let mut cu = Utterance::new(&c.join(" "));
            cu.negated |= u.negated;
            cu.goal_cue |= u.goal_cue;
            cu.observation_cue |= u.observation_cue;
            cu.question |= u.question;
            cu
        });
    if let Some(cu) = &core {
        for p in targets(predicates) {
            let (mut score, flipped) = cu.match_words(&predicate_words(p));
            // Matching the command's core is stronger evidence than matching anywhere.
            if score > 0.0 {
                score = (score + CORE_BONUS).min(1.0);
            }
            if cu.question && p.role != PredicateRole::Goal {
                score -= 0.3;
            }
            if score > 0.0 {
                if let Some(intent) = predicate_intent(p, cu, flipped) {
                    match ranked.iter_mut().find(|r| r.0 == intent) {
                        Some(r) => r.1 = r.1.max(score),
                        None => ranked.push((intent, score)),
                    }
                }
            }
        }
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    }

    // Inference: naive Bayes over everything known, blended with the direct matches.
    let classifier = Classifier::new(predicates, phrases);
    let mut posterior = classifier.posterior(&u.content());
    if posterior.is_empty() {
        if let Some(cu) = &core {
            posterior = classifier.posterior(&cu.content());
        }
    }
    let mut blended: BTreeMap<Intent, f32> = posterior.iter().cloned().collect();
    for (intent, score) in &ranked {
        let p = blended.entry(intent.clone()).or_insert(0.0);
        *p = 0.5 * *p + 0.5 * score;
    }
    let mut blended: Vec<(Intent, f32)> = blended
        .into_iter()
        .filter(|(i, _)| allowed(i, predicates))
        // A question is never taken for the fact it asks about.
        .map(|(i, p)| if u.question && matches!(i, Intent::Observe { .. }) { (i, p * 0.5) } else { (i, p) })
        .collect();
    blended.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    match blended.first().cloned() {
        Some((best, p)) if p >= INFERENCE_POSTERIOR => {
            done(best, p, Route::Inference, blended.into_iter().skip(1).take(3).collect())
        }
        _ => Interpretation {
            normalized: normalized.clone(),
            intent: None,
            confidence: 0.0,
            route: Route::NoMatch,
            alternatives: blended.into_iter().filter(|a| a.1 > 0.05).take(3).collect(),
        },
    }
}

/// Multinomial naive Bayes, intent <- words. Examples are taught phrases (weighted most),
/// command phrases and predicate names.
struct Classifier {
    counts: BTreeMap<Intent, (f32, BTreeMap<String, f32>)>,
    vocabulary: BTreeSet<String>,
}

impl Classifier {
    fn new(predicates: &[PredicateSchema], phrases: &PhraseBook) -> Self {
        let mut c = Self { counts: BTreeMap::new(), vocabulary: BTreeSet::new() };
        for (phrase, intent) in &phrases.phrases {
            if allowed(intent, predicates) {
                c.learn(intent.clone(), &Utterance::new(phrase).content(), 3.0);
            }
        }
        for (phrase, cmd) in COMMANDS {
            let u = Utterance::new(phrase);
            let words = if u.content().is_empty() { u.words.iter().map(|w| stem(w)).collect() } else { u.content() };
            c.learn(cmd.intent(), &words, 1.0);
        }
        for p in targets(predicates) {
            let words = predicate_words(p);
            let positive = Utterance { words: words.clone(), stems: words.clone(), ..Utterance::new("") };
            if let Some(intent) = predicate_intent(p, &positive, false) {
                c.learn(intent, &words, 1.0);
            }
        }
        c
    }

    fn learn(&mut self, intent: Intent, words: &[String], weight: f32) {
        let (n, counts) = self.counts.entry(intent).or_default();
        *n += weight;
        for w in words {
            *counts.entry(w.clone()).or_insert(0.0) += weight;
            self.vocabulary.insert(w.clone());
        }
    }

    /// Posterior over intents for the utterance's content words. Words never seen count
    /// against nobody; an utterance with no known word gets no posterior at all.
    fn posterior(&self, words: &[String]) -> Vec<(Intent, f32)> {
        // Unknown words are matched to the closest known one, so recognition slips still count.
        let known: Vec<&String> = words
            .iter()
            .filter_map(|w| {
                self.vocabulary
                    .iter()
                    .map(|v| (v, similarity(v, w)))
                    .filter(|(_, s)| *s >= WORD_MATCH)
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(v, _)| v)
            })
            .collect();
        if known.len() * 2 <= words.len() {
            return vec![];
        }
        let v = self.vocabulary.len() as f32;
        let total: f32 = self.counts.values().map(|c| c.0).sum();
        let logs: Vec<(Intent, f32)> = self
            .counts
            .iter()
            .map(|(intent, (n, counts))| {
                let words_in_class: f32 = counts.values().sum();
                let log = known.iter().fold((n / total).ln(), |acc, w| {
                    acc + ((counts.get(*w).copied().unwrap_or(0.0) + 0.1) / (words_in_class + 0.1 * v)).ln()
                });
                (intent.clone(), log)
            })
            .collect();
        let max = logs.iter().map(|l| l.1).fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = logs.iter().map(|l| (l.1 - max).exp()).sum();
        logs.into_iter().map(|(i, l)| (i, (l - max).exp() / z)).collect()
    }
}
