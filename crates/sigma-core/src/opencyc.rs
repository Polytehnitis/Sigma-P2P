//! Starting knowledge from [OpenCyc 4.0](https://github.com/Polytehnitis/Cyc-4.0-OWL), Cycorp's
//! common-sense ontology (Creative Commons Attribution 3.0, © 2001-2012 Cycorp, Inc.).
//!
//! The OWL export holds about 117,000 classes, far more than a node should carry in its graph,
//! and most of it is specialist (molecules, military units, product categories). A curated
//! subset is bundled instead: 232 everyday concepts (animals, plants, foods and drinks,
//! furniture and kitchenware, tools, devices, vehicles, clothing, buildings and rooms, places,
//! people and occupations, body parts) and the general concepts above them (mammal,
//! vertebrate, living thing, artifact, device, container, place, …). Each concept's parents
//! are the kept concepts OpenCyc says it is a subcollection of (`genls`), followed through the
//! whole export and reduced to the nearest ones; nothing is added that OpenCyc does not state.
//! `knowledge/opencyc/curate.py` rebuilds the table from the OWL files.
//!
//! The concepts are one knowledge base, `opencyc-everyday`, with one attribute: `genls`, the
//! concept's parents. Reasoning over it is done by retrievals, as Cyc's `genls` is transitive:
//! a concept's ancestors are retrieved one link at a time, and its kinds are the concepts
//! retrieved with it as the `genls` cue. [`SigmaEngine::answer_opencyc`] answers English
//! questions about them: "what is a dog?", "is a whale a mammal?", "is a spider an insect?",
//! "what kinds of furniture do you know?".

use crate::knowledge::{KnowledgeBase, SmemGraph};
use crate::SigmaEngine;
use std::collections::{BTreeSet, HashMap, VecDeque};

pub const TAXONOMY_SOURCE: &str = include_str!("../knowledge/opencyc/everyday-taxonomy.tsv");
/// The knowledge base of OpenCyc's everyday concepts.
pub const EVERYDAY: &str = "opencyc-everyday";
/// The attribute that holds a concept's parents.
pub const GENLS: &str = "genls";
/// Most concepts visited when walking up or down the taxonomy.
const MAX_VISITS: usize = 256;
/// Longest concept name, in words ("person with an occupation").
const MAX_WORDS: usize = 4;

/// Reads the bundled table: `word<TAB>OpenCyc constant<TAB>OpenCyc id<TAB>parents`, the
/// parents separated by commas. Every parent must be a concept of the table.
pub fn parse_taxonomy(source: &str) -> Result<SmemGraph, String> {
    let mut graph = SmemGraph::new();
    let mut parents: Vec<(usize, String)> = Vec::new();
    for (n, line) in source.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let [word, _constant, _id, genls] = cols[..] else {
            return Err(format!("line {}: expected 4 columns", n + 1));
        };
        let word = word.trim().to_string();
        if word.is_empty() || graph.contains_key(&word) {
            return Err(format!("line {}: empty or repeated concept {word:?}", n + 1));
        }
        let node = graph.entry(word).or_default();
        for p in genls.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            node.entry(GENLS.into()).or_default().insert(p.to_string());
            parents.push((n + 1, p.to_string()));
        }
    }
    if let Some((n, p)) = parents.iter().find(|(_, p)| !graph.contains_key(p)) {
        return Err(format!("line {n}: unknown parent {p:?}"));
    }
    Ok(graph)
}

/// The OpenCyc knowledge base.
pub fn opencyc_knowledge() -> KnowledgeBase {
    let bundled = "bundled knowledge is valid";
    let mut kb =
        KnowledgeBase::from_graph(EVERYDAY, "Cyc", parse_taxonomy(TAXONOMY_SOURCE).expect(bundled)).expect(bundled);
    // Questions about it go through `answer_opencyc`, which knows what `genls` means.
    kb.english = false;
    kb
}

/// Concepts that take no article.
const MASS: &[&str] = &[
    "water",
    "milk",
    "coffee",
    "tea",
    "juice",
    "beer",
    "wine",
    "rice",
    "bread",
    "cheese",
    "butter",
    "yogurt",
    "pasta",
    "soup",
    "chocolate",
    "beef",
    "pork",
    "chicken meat",
    "food",
    "meat",
    "furniture",
    "clothing",
    "paper",
    "grass",
    "metal",
    "kitchenware",
];

/// Plurals that adding or dropping an `s` does not reach.
const IRREGULAR: &[(&str, &str)] = &[
    ("men", "man"),
    ("women", "woman"),
    ("children", "child"),
    ("people", "person"),
    ("persons", "person"),
    ("mice", "mouse"),
    ("feet", "foot"),
    ("knives", "knife"),
    ("wolves", "wolf"),
    ("shelves", "shelf"),
    ("leaves", "leaf"),
    ("police officers", "police officer"),
];

fn with_article(w: &str) -> String {
    if MASS.contains(&w) {
        w.to_string()
    } else if w.starts_with(['a', 'e', 'i', 'o', 'u']) {
        format!("an {w}")
    } else {
        format!("a {w}")
    }
}

fn capitalize(w: &str) -> String {
    let mut c = w.chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

fn list(items: &[String]) -> String {
    match items.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, init)) => format!("{} and {last}", init.join(", ")),
        None => String::new(),
    }
}

/// A concept named in a question, and whether it was named in the plural.
struct Mention {
    concept: String,
    plural: bool,
}

impl SigmaEngine {
    /// The parents of a concept (`dog` -> mammal), by retrieval.
    pub fn cyc_parents(&self, concept: &str) -> Vec<String> {
        self.ask(EVERYDAY, &[("@node".into(), concept.into())]).map_or_else(Vec::new, |r| r.get(GENLS).to_vec())
    }

    /// The concepts whose parents include `concept` (`mammal` -> bat, bear, cat, …), by
    /// retrieval with it as the cue.
    pub fn cyc_kinds(&self, concept: &str) -> Vec<String> {
        let mut out: Vec<String> = self
            .ask(EVERYDAY, &[(GENLS.into(), concept.into())])
            .map_or_else(Vec::new, |r| r.matches.into_iter().map(|m| m.0).collect());
        out.sort();
        out
    }

    /// Every ancestor of a concept, nearest first, each with the concept it was reached from:
    /// parents retrieved one link at a time, as `genls` is transitive.
    pub fn cyc_ancestors(&self, concept: &str) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        let mut queue = VecDeque::from([concept.to_string()]);
        while let Some(c) = queue.pop_front() {
            if out.len() >= MAX_VISITS {
                break;
            }
            for p in self.cyc_parents(&c) {
                if p != concept && !out.iter().any(|(a, _)| *a == p) {
                    out.push((p.clone(), c.clone()));
                    queue.push_back(p);
                }
            }
        }
        out
    }

    /// Whether `concept` is `ancestor` or one of its kinds; with the chain of parents that
    /// shows it (`whale` is a `mammal` -> [whale, mammal]).
    pub fn cyc_is_a(&self, concept: &str, ancestor: &str) -> Option<Vec<String>> {
        if concept == ancestor {
            return Some(vec![concept.into()]);
        }
        let found = self.cyc_ancestors(concept);
        let from: HashMap<&str, &str> = found.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        let mut chain = vec![ancestor.to_string()];
        let mut at = *from.get(ancestor)?;
        while at != concept {
            chain.push(at.to_string());
            at = from[at];
        }
        chain.push(concept.into());
        chain.reverse();
        Some(chain)
    }

    /// Every concept below `concept`, by retrievals one level at a time.
    pub fn cyc_descendants(&self, concept: &str) -> BTreeSet<String> {
        let mut found = BTreeSet::new();
        let mut queue = vec![concept.to_string()];
        while let Some(c) = queue.pop() {
            if found.len() >= MAX_VISITS {
                break;
            }
            for k in self.cyc_kinds(&c) {
                if found.insert(k.clone()) {
                    queue.push(k);
                }
            }
        }
        found
    }

    /// The concepts a question names, longest names first, in order.
    fn cyc_mentions(&self, text: &str) -> Option<(Vec<Mention>, Vec<String>)> {
        let kb = self.knowledge_bases().into_iter().find(|k| k.name == EVERYDAY)?;
        let words: Vec<String> = crate::language::normalize(text)
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(String::from)
            .collect();
        let known = |w: &str| kb.graph.contains_key(w).then(|| w.to_string());
        let singular = |w: &str| -> Option<String> {
            if let Some((_, s)) = IRREGULAR.iter().find(|(p, _)| *p == w) {
                return known(s);
            }
            let stems = [
                w.strip_suffix("ies").map(|s| format!("{s}y")),
                w.strip_suffix("es").map(String::from),
                w.strip_suffix('s').map(String::from),
            ];
            stems.into_iter().flatten().find_map(|s| known(&s))
        };
        let (mut mentions, mut rest) = (Vec::new(), Vec::new());
        let mut i = 0;
        while i < words.len() {
            let hit = (i + 1..=words.len().min(i + MAX_WORDS)).rev().find_map(|j| {
                let phrase = words[i..j].join(" ");
                known(&phrase)
                    .map(|c| (j, Mention { concept: c, plural: false }))
                    .or_else(|| singular(&phrase).map(|c| (j, Mention { concept: c, plural: true })))
            });
            match hit {
                Some((j, m)) => {
                    mentions.push(m);
                    i = j;
                }
                None => {
                    rest.push(words[i].clone());
                    i += 1;
                }
            }
        }
        Some((mentions, rest))
    }

    /// Answers an English question about OpenCyc's everyday concepts, or `None`:
    ///
    /// * "what is a dog?": its parents and every concept above them;
    /// * "is a whale a mammal?", "are spiders insects?": whether one is a kind of the other,
    ///   and the chain of parents that shows it;
    /// * "what kinds of furniture do you know?", "name some birds": the kinds below a concept.
    ///
    /// With `defer` set, questions that only name categories Rosie knows (and do not ask for
    /// kinds) are left to Rosie, whose ontology has the affordances of those things.
    pub fn answer_opencyc(&self, text: &str, defer: bool) -> Option<String> {
        let (mentions, rest) = self.cyc_mentions(text)?;
        let has = |ws: &[&str]| rest.iter().any(|w| ws.contains(&w.as_str()));
        // Where, who and when are about something specific, not about a concept.
        if has(&["where", "who", "whose", "when"]) {
            return None;
        }
        let normalized = crate::language::normalize(text);
        let first = normalized.split(|c: char| !c.is_alphanumeric()).find(|w| !w.is_empty()).unwrap_or("");
        let asks_is_a = matches!(first, "is" | "are") || has(&["kind", "type", "sort"]);
        let asks_what = matches!(first, "what" | "tell" | "describe" | "define") || has(&["about"]);
        let listing = has(&["kinds", "types", "sorts", "examples", "name", "list"])
            || (has(&["know", "there"]) && mentions.len() == 1 && mentions[0].plural);
        if defer && !listing {
            let rosie = self.knowledge_bases().into_iter().find(|k| k.name == crate::rosie::CATEGORIES);
            let words: BTreeSet<String> = rosie
                .and_then(|k| k.attribute("word"))
                .map_or_else(BTreeSet::new, |a| a.values.iter().map(|v| v.replace('-', " ")).collect());
            if mentions.iter().all(|m| words.contains(&m.concept)) {
                return None;
            }
        }
        match mentions.as_slice() {
            [m] if listing => {
                let c = &m.concept;
                let kinds = self.cyc_kinds(c);
                if kinds.is_empty() {
                    return Some(format!("I know no kinds of {c}."));
                }
                let all = self.cyc_descendants(c).len();
                let more = if all > kinds.len() { format!(" ({all} counting their own kinds)") } else { String::new() };
                Some(format!("Kinds of {c} I know{more}: {}.", list(&kinds)))
            }
            [x, y] if asks_is_a => {
                let (x, y) = (&x.concept, &y.concept);
                let subject = capitalize(&with_article(x));
                Some(match self.cyc_is_a(x, y) {
                    Some(chain) if chain.len() <= 2 => format!("Yes, {} is {}.", with_article(x), with_article(y)),
                    Some(chain) => {
                        let steps: Vec<String> = chain[1..].iter().map(|c| with_article(c)).collect();
                        format!("Yes: {subject} is {}.", steps.join(", which is "))
                    }
                    None => format!("No, {} is not {} that I know of.", with_article(x), with_article(y)),
                })
            }
            [m] if asks_what => {
                let c = &m.concept;
                let parents = self.cyc_parents(c);
                if parents.is_empty() {
                    return Some(format!("{} is the most general concept I know.", capitalize(c)));
                }
                let direct: Vec<String> = parents.iter().map(|p| with_article(p)).collect();
                let mut out = format!("{} is {}.", capitalize(&with_article(c)), list(&direct));
                let above: Vec<String> = self
                    .cyc_ancestors(c)
                    .into_iter()
                    .filter(|(a, _)| !parents.contains(a))
                    .map(|(a, _)| with_article(&a))
                    .collect();
                if !above.is_empty() {
                    out.push_str(&format!(" So it is also {}.", list(&above)));
                }
                Some(out)
            }
            _ => None,
        }
    }
}
