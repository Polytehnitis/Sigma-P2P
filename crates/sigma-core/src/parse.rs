//! Parsing inside the architecture, as Sigma's language work does: a sentence becomes a factor
//! graph that the engine's own message passing solves.
//!
//! For a sentence of `n` words the graph has, per position `i`, a part-of-speech variable
//! `Tag(i)` and a head variable `Head(i)` over the other positions plus the root:
//!
//! * a lexical factor on `Tag(i)`: how likely each tag is to produce the word,
//! * a transition factor on `Tag(i), Tag(i+1)` (and a start factor on `Tag(0)`),
//! * for every other position `j`, an attachment factor on `Head(i), Tag(i), Tag(j)` that
//!   scores the arc `j -> i` (by the two tags, its direction and its length) when `Head(i) = j`
//!   and is neutral otherwise, plus a root factor on `Head(i), Tag(i)`.
//!
//! Max-product message passing ([`SigmaEngine::map_estimate`]) decodes the most probable tags
//! and heads together: an arc-factored dependency parse, solved by loopy max-product rather
//! than a chart. The single root is enforced when decoding. A [`Grammar`] holds the counts
//! behind the factors and learns from parses it is shown ([`Grammar::learn`]).
//!
//! [`crate::language::interpret`] uses the parse to find the core of a command (the root and
//! its content dependents), so "could you please go ahead and mount my nas for me" is
//! understood as "mount nas".

use crate::graph_memory::Builder;
use crate::{Inference, PatternKind, PredicateRole, PredicateSchema, SigmaEngine};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The parts of speech.
pub const TAGS: [&str; 8] = ["VERB", "NOUN", "DET", "PRON", "ADJ", "PART", "PREP", "ADV"];
const T: usize = TAGS.len();
const VERB: usize = 0;
const NOUN: usize = 1;
const DET: usize = 2;
const PRON: usize = 3;
const ADJ: usize = 4;
const PART: usize = 5;
const PREP: usize = 6;
const ADV: usize = 7;
/// The root, as a head "tag".
const ROOT: usize = T;
/// Longest sentence parsed (words).
pub const MAX_PARSE_WORDS: usize = 12;
/// Arc length penalty per extra word of distance.
const DISTANCE: f32 = 0.35;

/// A dependency parse: a tag per word, and each word's head (`None` for the root).
#[derive(Clone, Debug, PartialEq)]
pub struct Parse {
    pub words: Vec<String>,
    pub tags: Vec<usize>,
    pub heads: Vec<Option<usize>>,
}

impl Parse {
    /// The tag of word `i`, by name.
    pub fn tag(&self, i: usize) -> &'static str {
        TAGS[self.tags[i]]
    }

    /// The root word's position.
    pub fn root(&self) -> Option<usize> {
        self.heads.iter().position(Option::is_none)
    }

    /// Positions whose head is `i`.
    pub fn dependents(&self, i: usize) -> Vec<usize> {
        (0..self.words.len()).filter(|&j| self.heads[j] == Some(i)).collect()
    }

    /// The core of the sentence: its command verb (the root, or the verb a light verb such as
    /// "could", "go" or "want" leads on to) and, recursively, the nouns and adjectives that
    /// depend on it, also through prepositions, in sentence order. Determiners, pronouns,
    /// particles, adverbs, prepositions and other clauses are left out.
    pub fn core(&self) -> Vec<String> {
        let Some(mut verb) = self.root() else { return Vec::new() };
        let mut guard = 0;
        while LIGHT_VERBS.contains(&self.words[verb].as_str()) && guard < self.words.len() {
            guard += 1;
            match self.dependents(verb).into_iter().rev().find(|&j| j > verb && self.tags[j] == VERB) {
                Some(next) => verb = next,
                None => break,
            }
        }
        let mut keep = vec![false; self.words.len()];
        let mut stack = vec![verb];
        while let Some(i) = stack.pop() {
            if keep[i] {
                continue;
            }
            keep[i] = true;
            for j in self.dependents(i) {
                if matches!(self.tags[j], NOUN | ADJ | PREP) {
                    stack.push(j);
                }
            }
        }
        (0..self.words.len()).filter(|&i| keep[i] && self.tags[i] != PREP).map(|i| self.words[i].clone()).collect()
    }
}

/// Verbs that lead on to the command verb (their last verb dependent) rather than being it.
const LIGHT_VERBS: &[&str] = &["could", "can", "would", "will", "go", "want", "need", "please", "try", "lets"];

/// Counts behind the parsing factors (add-one smoothed when used).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grammar {
    /// Word -> count per tag.
    pub lexicon: BTreeMap<String, [f32; T]>,
    /// Tags of unknown words.
    pub unknown: [f32; T],
    /// Row `T` is the start; `[from][to]`.
    pub transitions: [[f32; T]; T + 1],
    /// `[dependent tag][head tag or ROOT][head to the left, head to the right]`.
    pub attachments: [[[f32; 2]; T + 1]; T],
}

fn seed<const N: usize>(pairs: &[(usize, f32)]) -> [f32; N] {
    let mut a = [0.1; N];
    for &(i, w) in pairs {
        a[i] = w;
    }
    a
}

impl Grammar {
    /// A small English grammar for spoken commands and reports.
    pub fn english() -> Self {
        let mut lexicon = BTreeMap::new();
        let words: &[(&str, usize)] = &[
            ("the", DET),
            ("a", DET),
            ("an", DET),
            ("my", DET),
            ("your", DET),
            ("this", DET),
            ("that", DET),
            ("some", DET),
            ("i", PRON),
            ("you", PRON),
            ("me", PRON),
            ("it", PRON),
            ("we", PRON),
            ("us", PRON),
            ("please", PART),
            ("to", PART),
            ("and", PART),
            ("not", PART),
            ("could", VERB),
            ("can", VERB),
            ("would", VERB),
            ("will", VERB),
            ("go", VERB),
            ("want", VERB),
            ("need", VERB),
            ("make", VERB),
            ("is", VERB),
            ("are", VERB),
            ("for", PREP),
            ("on", PREP),
            ("in", PREP),
            ("with", PREP),
            ("at", PREP),
            ("of", PREP),
            ("from", PREP),
            ("now", ADV),
            ("ahead", ADV),
            ("just", ADV),
            ("again", ADV),
            ("quickly", ADV),
            ("new", ADJ),
            ("old", ADJ),
            ("big", ADJ),
            ("external", ADJ),
        ];
        for &(w, t) in words {
            let mut c = [0.0; T];
            c[t] = 5.0;
            lexicon.insert(w.to_string(), c);
        }
        let mut transitions = [[1.0; T]; T + 1];
        transitions[T] = seed(&[(VERB, 4.0), (PART, 3.0), (PRON, 3.0), (DET, 2.0), (NOUN, 2.0)]);
        transitions[DET] = seed(&[(NOUN, 6.0), (ADJ, 3.0)]);
        transitions[ADJ] = seed(&[(NOUN, 6.0), (ADJ, 1.0)]);
        transitions[PART] = seed(&[(VERB, 6.0), (PRON, 1.0), (ADV, 1.0)]);
        transitions[VERB] =
            seed(&[(DET, 4.0), (NOUN, 4.0), (PRON, 2.0), (VERB, 2.0), (PART, 2.0), (ADV, 2.0), (PREP, 2.0)]);
        transitions[NOUN] = seed(&[(PREP, 3.0), (NOUN, 2.0), (PART, 2.0), (VERB, 1.0)]);
        transitions[PRON] = seed(&[(VERB, 5.0), (PREP, 1.0), (PART, 1.0)]);
        transitions[PREP] = seed(&[(DET, 4.0), (NOUN, 4.0), (PRON, 3.0)]);
        transitions[ADV] = seed(&[(VERB, 3.0), (PART, 2.0), (DET, 1.0), (NOUN, 1.0)]);
        let mut attachments = [[[0.1; 2]; T + 1]; T];
        let mut arc = |dep: usize, head: usize, left: f32, right: f32| attachments[dep][head] = [left, right];
        arc(VERB, ROOT, 6.0, 6.0);
        arc(VERB, VERB, 3.0, 0.5); // "go ahead and mount": the later verb hangs off the first
        arc(NOUN, VERB, 5.0, 2.0); // object (head to the left) or subject
        arc(NOUN, NOUN, 1.0, 2.0); // compounds: "backup drive"
        arc(NOUN, PREP, 4.0, 0.2);
        arc(DET, NOUN, 0.2, 6.0);
        arc(ADJ, NOUN, 0.2, 6.0);
        arc(PRON, VERB, 3.0, 4.0);
        arc(PRON, PREP, 4.0, 0.2);
        arc(PART, VERB, 1.0, 5.0);
        arc(PREP, VERB, 3.0, 0.5);
        arc(PREP, NOUN, 2.0, 0.2);
        arc(ADV, VERB, 3.0, 2.0);
        arc(NOUN, ROOT, 0.5, 0.5);
        Grammar { lexicon, unknown: seed(&[(NOUN, 4.0), (VERB, 3.0), (ADJ, 1.0)]), transitions, attachments }
    }

    /// The English grammar, with the words of the engine's predicate names added: the first
    /// word of an action, goal or subgoal is likely a verb (`MountNas` -> "mount"), the rest
    /// nouns.
    pub fn for_predicates(predicates: &[PredicateSchema]) -> Self {
        let mut g = Self::english();
        for p in predicates {
            let words = crate::language::name_words(&p.name);
            for (k, w) in words.iter().enumerate() {
                if g.lexicon.get(w).is_some_and(|c| c.iter().sum::<f32>() >= 5.0) {
                    continue; // a function word stays what it is
                }
                let c = g.lexicon.entry(w.clone()).or_insert([0.0; T]);
                let verbish = k == 0 && p.role != PredicateRole::Context && words.len() > 1;
                c[if verbish { VERB } else { NOUN }] += 2.0;
            }
        }
        g
    }

    fn emission(&self, word: &str) -> [f32; T] {
        let c = self.lexicon.get(word).unwrap_or(&self.unknown);
        let total: f32 = c.iter().sum::<f32>() + 0.1 * T as f32;
        let mut e = [0.0; T];
        for t in 0..T {
            e[t] = (c[t] + 0.1) / total;
        }
        e
    }

    fn transition(&self, from: usize) -> [f32; T] {
        let row = &self.transitions[from];
        let total: f32 = row.iter().sum();
        let mut p = [0.0; T];
        for t in 0..T {
            p[t] = row[t] / total;
        }
        p
    }

    fn attachment(&self, dep: usize, head: usize, right: bool) -> f32 {
        self.attachments[dep][head][usize::from(right)]
    }

    /// Parses `text` (split on whitespace, lower-cased).
    pub fn parse_text(&self, text: &str) -> Option<Parse> {
        let words: Vec<String> = text.split_whitespace().map(|w| w.to_lowercase()).collect();
        self.parse(&words)
    }

    /// Builds the sentence's factor graph and decodes it by max-product message passing.
    pub fn parse(&self, words: &[String]) -> Option<Parse> {
        let n = words.len();
        if n == 0 || n > MAX_PARSE_WORDS {
            return None;
        }
        let mut engine = SigmaEngine::new();
        engine.inference = Inference::MaxProduct;
        let var = |e: &mut SigmaEngine, name: String, d: usize| {
            e.register_predicate(&PredicateSchema { name, domain_size: d, role: PredicateRole::Context }).ok()
        };
        let tags: Vec<usize> = (0..n).map(|i| var(&mut engine, format!("Tag({i})"), T)).collect::<Option<_>>()?;
        // Head(i) ranges over the positions, then the root (value n).
        let heads: Vec<usize> = (0..n).map(|i| var(&mut engine, format!("Head({i})"), n + 1)).collect::<Option<_>>()?;
        let mut b = Builder::new("@parse");
        let c = PatternKind::Condact;
        for (i, w) in words.iter().enumerate() {
            b.add(vec![tags[i]], vec![c], &[T], self.emission(w).to_vec(), None);
        }
        b.add(vec![tags[0]], vec![c], &[T], self.transition(T).to_vec(), None);
        for i in 0..n - 1 {
            let values: Vec<f32> = (0..T).flat_map(|from| self.transition(from)).collect();
            b.add(vec![tags[i], tags[i + 1]], vec![c, c], &[T, T], values, Some(1));
        }
        for i in 0..n {
            // Root factor [Head(i), Tag(i)]: the root arc, and no word heads itself.
            let values: Vec<f32> = (0..=n)
                .flat_map(|h| {
                    (0..T).map(move |t| match h {
                        _ if h == i => 0.0,
                        _ if h == n => self.attachment(t, ROOT, false),
                        _ => 1.0,
                    })
                })
                .collect();
            b.add(vec![heads[i], tags[i]], vec![c, c], &[n + 1, T], values, None);
            for j in (0..n).filter(|&j| j != i) {
                let length = (i as f32 - j as f32).abs() - 1.0;
                let scale = (-DISTANCE * length).exp();
                let values: Vec<f32> = (0..=n)
                    .flat_map(|h| {
                        (0..T).flat_map(move |ti| {
                            (0..T).map(move |tj| if h == j { self.attachment(ti, tj, j > i) * scale } else { 1.0 })
                        })
                    })
                    .collect();
                b.add(vec![heads[i], tags[i], tags[j]], vec![c, c, c], &[n + 1, T, T], values, None);
            }
        }
        engine.factors.extend(b.factors);
        let map = engine.map_estimate();
        let value = |name: &str| map.assignment.iter().find(|(m, _)| m == name).map(|kv| kv.1);
        let tag: Vec<usize> = (0..n).map(|i| value(&format!("Tag({i})"))).collect::<Option<_>>()?;
        let mut head: Vec<Option<usize>> =
            (0..n).map(|i| value(&format!("Head({i})")).map(|h| (h < n).then_some(h))).collect::<Option<_>>()?;
        // One root: the most verb-like of the words decoded as roots (or, if none, the first
        // verb); the other roots attach to it.
        let roots: Vec<usize> = (0..n).filter(|&i| head[i].is_none()).collect();
        let root = roots
            .iter()
            .copied()
            .max_by(|&a, &b| {
                self.attachment(tag[a], ROOT, false).total_cmp(&self.attachment(tag[b], ROOT, false)).then(b.cmp(&a))
            })
            .or_else(|| tag.iter().position(|&t| t == VERB))
            .unwrap_or(0);
        for (i, h) in head.iter_mut().enumerate() {
            if i == root {
                *h = None;
            } else if h.is_none() {
                *h = Some(root);
            }
        }
        // Break any cycle by attaching its first word to the root.
        for i in 0..n {
            let mut seen = vec![false; n];
            let mut at = i;
            while let Some(h) = head[at] {
                if seen[at] {
                    head[at] = Some(root).filter(|&r| r != at);
                    break;
                }
                seen[at] = true;
                at = h;
            }
        }
        Some(Parse { words: words.to_vec(), tags: tag, heads: head })
    }

    /// Learns from a parse it is shown (an annotated sentence): its words' tags, tag
    /// transitions and arcs are counted.
    pub fn learn(&mut self, parse: &Parse) {
        let n = parse.words.len();
        for i in 0..n {
            self.lexicon.entry(parse.words[i].clone()).or_insert([0.0; T])[parse.tags[i]] += 1.0;
            let from = if i == 0 { T } else { parse.tags[i - 1] };
            self.transitions[from][parse.tags[i]] += 1.0;
            let (head, right) = match parse.heads[i] {
                None => (ROOT, false),
                Some(h) => (parse.tags[h], h > i),
            };
            self.attachments[parse.tags[i]][head][usize::from(right)] += 1.0;
        }
    }
}

/// The tag index of a tag name.
pub fn tag_index(name: &str) -> Option<usize> {
    TAGS.iter().position(|t| *t == name)
}
