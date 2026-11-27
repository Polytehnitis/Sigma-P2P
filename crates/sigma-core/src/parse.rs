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
//! [`Grammar::parse_in_graph`] builds that graph and decodes it with the engine's max-product
//! message passing ([`SigmaEngine::map_estimate`]); on a graph this loopy that is approximate,
//! and the graph grows with the square of the sentence, so it takes at most 12 words.
//!
//! [`Grammar::parse`] finds the exact maximum of the same model over projective trees with a
//! single root: Eisner's chart algorithm, with each chart item also carrying the tags of its
//! two end words. Every pair of neighbouring words meets exactly once where two spans join,
//! which is where the tag transition between them is scored, so tags and arcs are decoded
//! jointly and exactly in O(n³·T³). It takes sentences of up to [`MAX_PARSE_WORDS`] words.
//! [`Grammar::log_score`] scores any parse under the model, so the two can be compared.
//!
//! A [`Grammar`] holds the counts behind the factors and learns from parses it is shown
//! ([`Grammar::learn`]).
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
/// Longest sentence parsed exactly (words).
pub const MAX_PARSE_WORDS: usize = 48;
/// Longest sentence parsed in the factor graph (words).
pub const MAX_GRAPH_PARSE_WORDS: usize = 12;
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

    /// The model's log score of a parse: emissions, tag transitions, arcs (with their length
    /// penalty) and the root arc. `None` if the parse is not a tree with one root.
    pub fn log_score(&self, parse: &Parse) -> Option<f32> {
        let n = parse.words.len();
        if n == 0
            || parse.tags.len() != n
            || parse.heads.len() != n
            || parse.heads.iter().filter(|h| h.is_none()).count() != 1
        {
            return None;
        }
        let mut score = 0.0;
        for i in 0..n {
            let t = *parse.tags.get(i).filter(|&&t| t < T)?;
            score += self.emission(&parse.words[i])[t].ln();
            let from = if i == 0 { T } else { parse.tags[i - 1] };
            score += self.transition(from)[t].ln();
            score += match parse.heads[i] {
                None => self.attachment(t, ROOT, false).ln(),
                Some(h) if h < n && h != i => {
                    let length = (i as f32 - h as f32).abs() - 1.0;
                    self.attachment(t, parse.tags[h], h > i).ln() - DISTANCE * length
                }
                Some(_) => return None,
            };
        }
        Some(score)
    }

    /// The exact best parse: Eisner's algorithm over (span, direction, end tags). Projective
    /// trees with a single root.
    pub fn parse(&self, words: &[String]) -> Option<Parse> {
        let n = words.len();
        if n == 0 || n > MAX_PARSE_WORDS {
            return None;
        }
        Some(Chart::new(self, words).decode())
    }

    /// Builds the sentence's factor graph and decodes it by the engine's loopy max-product
    /// message passing (approximate; at most [`MAX_GRAPH_PARSE_WORDS`] words).
    pub fn parse_in_graph(&self, words: &[String]) -> Option<Parse> {
        let n = words.len();
        if n == 0 || n > MAX_GRAPH_PARSE_WORDS {
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

/// Eisner's chart, with the tags of each span's two end words.
///
/// `complete[d]` and `incomplete[d]` are indexed by `(i, j, tag of i, tag of j)` for spans
/// `i <= j`; direction `d = 1` has its head at `i`, `d = 0` at `j`. An incomplete span holds
/// the arc between `i` and `j`; a complete one is a head with all its dependents on that side.
struct Chart<'a> {
    grammar: &'a Grammar,
    words: &'a [String],
    n: usize,
    emission: Vec<[f32; T]>,
    transition: [[f32; T]; T + 1],
    complete: [Vec<f32>; 2],
    incomplete: [Vec<f32>; 2],
    /// Where each incomplete span was split: `(r, tag of r, tag of r + 1)`.
    split: Vec<(usize, usize, usize)>,
    /// Where each complete span was split: `(r, tag of r)`.
    joined: [Vec<(usize, usize)>; 2],
}

impl<'a> Chart<'a> {
    fn new(grammar: &'a Grammar, words: &'a [String]) -> Self {
        let n = words.len();
        let size = n * n * T * T;
        let emission: Vec<[f32; T]> = words.iter().map(|w| grammar.emission(w).map(f32::ln)).collect();
        let mut transition = [[0.0; T]; T + 1];
        for (from, row) in transition.iter_mut().enumerate() {
            *row = grammar.transition(from).map(f32::ln);
        }
        let mut chart = Chart {
            grammar,
            words,
            n,
            emission,
            transition,
            complete: [vec![f32::NEG_INFINITY; size], vec![f32::NEG_INFINITY; size]],
            incomplete: [vec![f32::NEG_INFINITY; size], vec![f32::NEG_INFINITY; size]],
            split: vec![(0, 0, 0); size],
            joined: [vec![(0, 0); size], vec![(0, 0); size]],
        };
        chart.fill();
        chart
    }

    fn at(&self, i: usize, j: usize, ti: usize, tj: usize) -> usize {
        ((i * self.n + j) * T + ti) * T + tj
    }

    fn arc(&self, dep: usize, head: usize, head_right: bool, length: usize) -> f32 {
        self.grammar.attachment(dep, head, head_right).ln() - DISTANCE * (length as f32 - 1.0)
    }

    fn fill(&mut self) {
        let n = self.n;
        for i in 0..n {
            for t in 0..T {
                // Every word is the bottom of exactly one left and one right complete span, so
                // its emission is scored on the left one only.
                let k = self.at(i, i, t, t);
                self.complete[0][k] = self.emission[i][t];
                self.complete[1][k] = 0.0;
            }
        }
        for len in 1..n {
            for i in 0..n - len {
                let j = i + len;
                // Incomplete spans: C→[i, r] and C←[r + 1, j] joined over the neighbours r, r + 1.
                let mut best = vec![(f32::NEG_INFINITY, (0usize, 0usize, 0usize)); T * T];
                for r in i..j {
                    for ti in 0..T {
                        // Over the tag of r, for each tag of r + 1.
                        let mut through = [(f32::NEG_INFINITY, 0usize); T];
                        for tr in 0..T {
                            let left = self.complete[1][self.at(i, r, ti, tr)];
                            if left == f32::NEG_INFINITY {
                                continue;
                            }
                            for (tr1, slot) in through.iter_mut().enumerate() {
                                let v = left + self.transition[tr][tr1];
                                if v > slot.0 {
                                    *slot = (v, tr);
                                }
                            }
                        }
                        for tj in 0..T {
                            for (tr1, &(v, tr)) in through.iter().enumerate() {
                                let right = self.complete[0][self.at(r + 1, j, tr1, tj)];
                                let total = v + right;
                                if total > best[ti * T + tj].0 {
                                    best[ti * T + tj] = (total, (r, tr, tr1));
                                }
                            }
                        }
                    }
                }
                for ti in 0..T {
                    for tj in 0..T {
                        let (v, at) = best[ti * T + tj];
                        let k = self.at(i, j, ti, tj);
                        self.split[k] = at;
                        // Head i, dependent j (the head is to the dependent's left) ...
                        self.incomplete[1][k] = v + self.arc(tj, ti, false, len);
                        // ... or head j, dependent i.
                        self.incomplete[0][k] = v + self.arc(ti, tj, true, len);
                    }
                }
                // Complete spans.
                for ti in 0..T {
                    for tj in 0..T {
                        let k = self.at(i, j, ti, tj);
                        // Head i: I→[i, r] then C→[r, j].
                        let mut right = (f32::NEG_INFINITY, (0, 0));
                        for r in i + 1..=j {
                            for tr in 0..T {
                                let v =
                                    self.incomplete[1][self.at(i, r, ti, tr)] + self.complete[1][self.at(r, j, tr, tj)];
                                if v > right.0 {
                                    right = (v, (r, tr));
                                }
                            }
                        }
                        self.complete[1][k] = right.0;
                        self.joined[1][k] = right.1;
                        // Head j: C←[i, r] then I←[r, j].
                        let mut left = (f32::NEG_INFINITY, (0, 0));
                        for r in i..j {
                            for tr in 0..T {
                                let v =
                                    self.complete[0][self.at(i, r, ti, tr)] + self.incomplete[0][self.at(r, j, tr, tj)];
                                if v > left.0 {
                                    left = (v, (r, tr));
                                }
                            }
                        }
                        self.complete[0][k] = left.0;
                        self.joined[0][k] = left.1;
                    }
                }
            }
        }
    }

    /// The best single-root tree: the root word `r` heads C←[0, r] and C→[r, n - 1].
    fn decode(&self) -> Parse {
        let n = self.n;
        let mut best = (f32::NEG_INFINITY, (0, 0, 0, 0));
        for r in 0..n {
            for tr in 0..T {
                let root = self.grammar.attachment(tr, ROOT, false).ln();
                for t0 in 0..T {
                    let left = self.transition[T][t0] + self.complete[0][self.at(0, r, t0, tr)];
                    if left == f32::NEG_INFINITY {
                        continue;
                    }
                    for tn in 0..T {
                        let v = left + root + self.complete[1][self.at(r, n - 1, tr, tn)];
                        if v > best.0 {
                            best = (v, (r, tr, t0, tn));
                        }
                    }
                }
            }
        }
        let (r, tr, t0, tn) = best.1;
        let mut tags = vec![0; n];
        let mut heads = vec![None; n];
        self.complete_span(0, r, 0, t0, tr, &mut tags, &mut heads);
        self.complete_span(r, n - 1, 1, tr, tn, &mut tags, &mut heads);
        Parse { words: self.words.to_vec(), tags, heads }
    }

    #[allow(clippy::too_many_arguments)]
    fn complete_span(
        &self,
        i: usize,
        j: usize,
        d: usize,
        ti: usize,
        tj: usize,
        tags: &mut [usize],
        heads: &mut [Option<usize>],
    ) {
        tags[i] = ti;
        tags[j] = tj;
        if i == j {
            return;
        }
        let (r, tr) = self.joined[d][self.at(i, j, ti, tj)];
        if d == 1 {
            self.incomplete_span(i, r, 1, ti, tr, tags, heads);
            self.complete_span(r, j, 1, tr, tj, tags, heads);
        } else {
            self.complete_span(i, r, 0, ti, tr, tags, heads);
            self.incomplete_span(r, j, 0, tr, tj, tags, heads);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn incomplete_span(
        &self,
        i: usize,
        j: usize,
        d: usize,
        ti: usize,
        tj: usize,
        tags: &mut [usize],
        heads: &mut [Option<usize>],
    ) {
        tags[i] = ti;
        tags[j] = tj;
        if d == 1 {
            heads[j] = Some(i);
        } else {
            heads[i] = Some(j);
        }
        let (r, tr, tr1) = self.split[self.at(i, j, ti, tj)];
        self.complete_span(i, r, 1, ti, tr, tags, heads);
        self.complete_span(r + 1, j, 0, tr1, tj, tags, heads);
    }
}

/// The tag index of a tag name.
pub fn tag_index(name: &str) -> Option<usize> {
    TAGS.iter().position(|t| *t == name)
}
