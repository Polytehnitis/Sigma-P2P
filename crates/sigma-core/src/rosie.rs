//! Starting knowledge from [Rosie](https://github.com/SoarGroup/rosie), the Soar agent that
//! learns tasks from instruction (BSD 3-clause, University of Michigan Soar Lab).
//!
//! Rosie keeps its everyday world knowledge as semantic-memory *templates*
//! (`agent/init-smem/*.txt`): tables whose rows its configurator expands into `smem --add`
//! facts. Two of them are bundled here, verbatim:
//!
//! * `category-ontology.txt`: about 95 categories of things (fork, mug, fridge, kitchen, …),
//!   each with its word, its supercategory (a fork is a utensil, a utensil is kitchenware,
//!   kitchenware is an object …) and the affordances given for it (grabbable, fillable,
//!   openable, activatable, surface, receptacle, …).
//! * `smem-common.txt`: the property words (colors, shapes, sizes, weights, temperatures,
//!   modifiers, concepts).
//!
//! [`parse_rosie_templates`] reads such tables into a semantic memory, and they become two
//! knowledge bases (`rosie-categories`, `rosie-properties`). Rosie's category reasoning is
//! rebuilt over retrievals by message passing, as its rules do it: the supercategories of a
//! category are retrieved one link at a time up to the root, and a category has the
//! affordances of every category on that chain (Rosie collects `^affordance1` from every
//! category it retrieved for an object). [`SigmaEngine::answer_rosie`] answers English
//! questions about them: "what is a fork?", "is a mug a kind of kitchenware?", "can I fill a
//! mug?", "what can I open?", "is red a color?".

use crate::knowledge::{KnowledgeBase, SmemGraph};
use crate::SigmaEngine;
use std::collections::BTreeSet;

pub const CATEGORY_SOURCE: &str = include_str!("../knowledge/rosie/category-ontology.txt");
pub const COMMON_SOURCE: &str = include_str!("../knowledge/rosie/smem-common.txt");
/// The knowledge base of Rosie's categories.
pub const CATEGORIES: &str = "rosie-categories";
/// The knowledge base of Rosie's property words.
pub const PROPERTIES: &str = "rosie-properties";
/// The supercategory of Rosie's root category.
const ROOT: &str = "none1";
/// Longest supercategory chain followed.
const MAX_DEPTH: usize = 16;

/// Reads Rosie's smem templates: `template <name> _HANDLE_ _COLUMN_ … _LAST_* { rows }`. Each
/// row is one identifier (its handle), with `^handle`, `^property <name>` and one attribute per
/// column (`_WORD_` is `word`, `_SUPER_` is `supercategory`, `_AFFORDANCE_` is `affordance`,
/// `_TYPE_` and `_WORDTYPE_` are `word-type`); a starred last column takes the rest of the row.
pub fn parse_rosie_templates(source: &str) -> Result<SmemGraph, String> {
    let mut graph = SmemGraph::new();
    let mut current: Option<(String, Vec<String>, bool)> = None;
    for (n, raw) in source.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let err = |m: &str| format!("line {}: {m}", n + 1);
        if let Some(rest) = line.strip_prefix("template ") {
            if current.is_some() {
                return Err(err("template inside a template"));
            }
            let head = rest.strip_suffix('{').ok_or_else(|| err("expected `{`"))?;
            let mut words = head.split_whitespace();
            let name = words.next().ok_or_else(|| err("template without a name"))?.to_string();
            let cols: Vec<&str> = words.collect();
            if cols.first() != Some(&"_HANDLE_") {
                return Err(err("the first column must be _HANDLE_"));
            }
            let starred = cols.last().is_some_and(|c| c.ends_with('*'));
            let cols = cols[1..]
                .iter()
                .map(|c| column(c.trim_end_matches('*')))
                .collect::<Result<_, _>>()
                .map_err(|e| err(&e))?;
            current = Some((name, cols, starred));
            continue;
        }
        if line == "}" {
            current.take().ok_or_else(|| err("`}` outside a template"))?;
            continue;
        }
        let (name, cols, starred) = current.as_ref().ok_or_else(|| err("row outside a template"))?;
        let mut cells = line.split_whitespace();
        let handle = cells.next().expect("the line is not empty").to_string();
        let node = graph.entry(handle.clone()).or_default();
        node.entry("handle".into()).or_default().insert(handle.clone());
        node.entry("property".into()).or_default().insert(name.clone());
        let cells: Vec<&str> = cells.collect();
        let fixed = cols.len() - usize::from(*starred);
        if cells.len() < fixed || (!starred && cells.len() > fixed) {
            return Err(err(&format!("{handle}: expected {} values", cols.len())));
        }
        for (i, cell) in cells.iter().enumerate() {
            let attr = &cols[i.min(cols.len() - 1)];
            node.entry(attr.clone()).or_default().insert(cell.to_string());
        }
    }
    if current.is_some() {
        return Err("unterminated template".into());
    }
    Ok(graph)
}

fn column(placeholder: &str) -> Result<String, String> {
    let inner = placeholder
        .strip_prefix('_')
        .and_then(|c| c.strip_suffix('_'))
        .filter(|c| !c.is_empty())
        .ok_or_else(|| format!("bad column {placeholder}"))?;
    Ok(match inner {
        "WORD" => "word".into(),
        "SUPER" => "supercategory".into(),
        "AFFORDANCE" => "affordance".into(),
        "TYPE" | "WORDTYPE" => "word-type".into(),
        other => other.to_lowercase(),
    })
}

/// Rosie's two knowledge bases: its categories and its property words.
pub fn rosie_knowledge() -> Vec<KnowledgeBase> {
    let bundled = "bundled knowledge is valid";
    let categories = parse_rosie_templates(CATEGORY_SOURCE).expect(bundled);
    let mut properties = parse_rosie_templates(COMMON_SOURCE).expect(bundled);
    // A handle in both (none here) would be a category.
    properties.retain(|h, _| !categories.contains_key(h));
    let mut kbs = vec![
        KnowledgeBase::from_graph(CATEGORIES, "RosieCategory", categories).expect(bundled),
        KnowledgeBase::from_graph(PROPERTIES, "RosieProperty", properties).expect(bundled),
    ];
    // Questions about them go through `answer_rosie`, which knows what they mean.
    kbs.iter_mut().for_each(|kb| kb.english = false);
    kbs
}

/// Words that ask about an affordance, and how to say that a category has it.
const AFFORDANCES: &[(&str, &[&str], &str)] = &[
    ("grabbable1", &["grab", "grabbable", "pick", "hold", "carry", "lift"], "grabbable"),
    ("fillable1", &["fill", "fillable", "filled"], "fillable"),
    ("openable1", &["open", "openable", "opened"], "openable"),
    ("activatable1", &["activate", "activatable", "turn", "switch"], "something you can turn on"),
    ("pourable1", &["pour", "pourable", "poured"], "pourable"),
    ("pressable1", &["press", "pressable", "push"], "pressable"),
    ("surface1", &["surface", "on", "onto"], "a surface"),
    ("receptacle1", &["receptacle", "in", "into", "inside", "store"], "a receptacle"),
    ("dispenser1", &["dispense", "dispenser"], "a dispenser"),
    ("drain1", &["drain"], "a drain"),
];

fn article(word: &str) -> &'static str {
    if word.starts_with(['a', 'e', 'i', 'o', 'u']) {
        "an"
    } else {
        "a"
    }
}

fn affordance_phrase(a: &str) -> String {
    AFFORDANCES
        .iter()
        .find(|(h, _, _)| *h == a)
        .map_or_else(|| a.trim_end_matches(char::is_numeric).replace('-', " "), |(_, _, p)| p.to_string())
}

fn key(text: &str) -> String {
    text.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

impl SigmaEngine {
    /// The category a word names (`fork` -> `fork1`), by retrieval.
    pub fn rosie_category(&self, word: &str) -> Option<String> {
        self.recall(CATEGORIES, &[("word", word)], "handle")
    }

    /// The word of a category (`fork1` -> `fork`).
    pub fn rosie_word(&self, handle: &str) -> Option<String> {
        self.recall(CATEGORIES, &[("handle", handle)], "word")
    }

    /// The supercategories of a category, nearest first, retrieved one link at a time as
    /// Rosie does (`fork1` -> utensil1, kitchenware1, object, entity1).
    pub fn rosie_supercategories(&self, handle: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut h = handle.to_string();
        while out.len() < MAX_DEPTH {
            match self.recall(CATEGORIES, &[("handle", &h)], "supercategory") {
                Some(s) if s != ROOT && !out.contains(&s) => {
                    out.push(s.clone());
                    h = s;
                }
                _ => break,
            }
        }
        out
    }

    /// Whether `handle` is `ancestor` or one of its subcategories.
    pub fn rosie_is_a(&self, handle: &str, ancestor: &str) -> bool {
        handle == ancestor || self.rosie_supercategories(handle).iter().any(|s| s == ancestor)
    }

    /// The affordances of a category, each with the category on its chain that gives it
    /// (itself or a supercategory).
    pub fn rosie_affordances(&self, handle: &str) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        for h in std::iter::once(handle.to_string()).chain(self.rosie_supercategories(handle)) {
            let Some(r) = self.ask(CATEGORIES, &[("handle".into(), h.clone())]) else { continue };
            for a in r.get("affordance") {
                if !out.iter().any(|(b, _)| b == a) {
                    out.push((a.clone(), h.clone()));
                }
            }
        }
        out
    }

    /// Every category with an affordance, given for it or inherited: the categories retrieved
    /// with the affordance as cue, then their subcategories, retrieved with each as the
    /// supercategory cue.
    pub fn rosie_with_affordance(&self, affordance: &str) -> Vec<String> {
        let matches = |a: &str, v: &str| -> Vec<String> {
            self.ask(CATEGORIES, &[(a.into(), v.into())])
                .map_or_else(Vec::new, |r| r.matches.into_iter().map(|m| m.0).collect())
        };
        let mut found: BTreeSet<String> = BTreeSet::new();
        let mut queue = matches("affordance", affordance);
        while let Some(h) = queue.pop() {
            if found.insert(h.clone()) {
                queue.extend(matches("supercategory", &h));
            }
        }
        found.into_iter().collect()
    }

    /// The property a word is a value of (`red` -> `color`), by retrieval.
    pub fn rosie_property(&self, word: &str) -> Option<String> {
        self.recall(PROPERTIES, &[("word", word)], "property")
    }

    /// Answers an English question about Rosie's categories and property words, or `None`.
    pub fn answer_rosie(&self, text: &str) -> Option<String> {
        let kb = self.knowledge_bases().into_iter().find(|k| k.name == CATEGORIES)?;
        let words: Vec<String> = crate::language::normalize(text).split(' ').map(String::from).collect();
        let cat_words: Vec<String> =
            kb.attribute("word")?.values.iter().filter(|v| v.as_str() != crate::knowledge::NONE).cloned().collect();
        let props = self.knowledge_bases().into_iter().find(|k| k.name == PROPERTIES);
        let prop_words: Vec<String> =
            props.and_then(|p| p.attribute("word")).map_or_else(Vec::new, |a| a.values.clone());
        let property_names: BTreeSet<String> = props
            .map_or_else(BTreeSet::new, |p| p.graph.values().flat_map(|a| a["property"].iter().cloned()).collect());
        // Mentions of category and property words, longest first, in order; plurals too.
        let lookup = |k: &str, list: &[String]| -> Option<String> {
            let found = |k: &str| list.iter().find(|w| key(w) == k).cloned();
            found(k).or_else(|| k.strip_suffix("es").and_then(found)).or_else(|| k.strip_suffix('s').and_then(found))
        };
        let mut cats: Vec<String> = Vec::new();
        let mut prop: Option<String> = None;
        let mut property_named: Option<String> = None;
        let mut rest: Vec<&str> = Vec::new();
        let mut i = 0;
        while i < words.len() {
            let mut hit = None;
            for j in (i + 1..=words.len().min(i + 3)).rev() {
                let k: String = words[i..j].concat();
                if let Some(w) = lookup(&k, &cat_words) {
                    hit = Some((j, Some(w), None, None));
                } else if let Some(p) = lookup(&k, &property_names.iter().cloned().collect::<Vec<_>>()) {
                    hit = Some((j, None, None, Some(p)));
                } else if let Some(w) = lookup(&k, &prop_words) {
                    hit = Some((j, None, Some(w), None));
                }
                if hit.is_some() {
                    break;
                }
            }
            match hit {
                Some((j, c, p, n)) => {
                    cats.extend(c);
                    prop = prop.or(p);
                    property_named = property_named.or(n);
                    i = j;
                }
                None => {
                    rest.push(&words[i]);
                    i += 1;
                }
            }
        }
        let has = |ws: &[&str]| rest.iter().any(|w| ws.contains(w));
        let asked = AFFORDANCES.iter().find(|(_, ws, _)| {
            ws.iter().any(|w| rest.contains(w))
                && (!["on", "onto", "in", "into", "inside"].iter().any(|p| ws.contains(p))
                    || has(&["put", "place", "set", "store"]))
        });
        let handle = |w: &str| self.rosie_category(w);
        let named = |h: &str| self.rosie_word(h).unwrap_or_else(|| h.to_string());

        // "is red a color?", "what is red?", "what colors do you know?"
        if cats.is_empty() {
            if let Some(w) = prop {
                let p = self.rosie_property(&w)?;
                return Some(match property_named {
                    Some(n) if n != p => format!("No, {w} is not a {n}; it is a {p}."),
                    _ => format!("{} is a {p}.", capitalize(&w)),
                });
            }
            if let Some(n) = property_named {
                let r = self.ask(PROPERTIES, &[("property".into(), n.clone())])?;
                let props = props?;
                let ws: Vec<&str> = r
                    .matches
                    .iter()
                    .filter_map(|(h, _)| props.graph.get(h).and_then(|a| a.get("word")).and_then(|w| w.iter().next()))
                    .map(String::as_str)
                    .collect();
                return Some(format!("The {n} words I know: {}.", ws.join(", ")));
            }
        }
        match (cats.as_slice(), asked) {
            // "what can I fill?", "which things are openable?"
            ([], Some((a, _, phrase))) => {
                let mut ws: Vec<String> = self.rosie_with_affordance(a).iter().map(|h| named(h)).collect();
                ws.sort();
                Some(format!("Things that are {phrase}: {}.", ws.join(", ")))
            }
            // "is a mug a kind of kitchenware?"
            ([x, y, ..], None) => {
                let (hx, hy) = (handle(x)?, handle(y)?);
                Some(if self.rosie_is_a(&hx, &hy) {
                    format!("Yes, {} {x} is a kind of {y}.", article(x))
                } else {
                    format!("No, {} {x} is not a kind of {y}.", article(x))
                })
            }
            // "can I fill a mug?", "is a fork grabbable?", "can I put a cup on the table?"
            ([first, .., last] | [first @ last], Some((a, _, phrase))) => {
                let x = if matches!(*a, "surface1" | "receptacle1") { last } else { first };
                let h = handle(x)?;
                let affs = self.rosie_affordances(&h);
                Some(match affs.iter().find(|(b, _)| b == a) {
                    Some((_, from)) if *from == h => format!("Yes, {} {x} is {phrase}.", article(x)),
                    Some((_, from)) => {
                        let f = named(from);
                        format!("Yes, {} {x} is {phrase}, like all {f}.", article(x))
                    }
                    None => format!("No, {} {x} is not {phrase} that I know of.", article(x)),
                })
            }
            // "what is a fork?"
            ([x], None) => {
                let h = handle(x)?;
                let chain: Vec<String> = self.rosie_supercategories(&h).iter().map(|s| named(s)).collect();
                let mut out = if chain.is_empty() {
                    format!("{} {x} is the most general category I know.", capitalize(article(x)))
                } else {
                    format!("{} {x} is a kind of {}.", capitalize(article(x)), chain.join(", then "))
                };
                let affs: Vec<String> = self.rosie_affordances(&h).iter().map(|(a, _)| affordance_phrase(a)).collect();
                if !affs.is_empty() {
                    let list = match affs.split_last() {
                        Some((last, [])) => last.clone(),
                        Some((last, init)) => format!("{} and {last}", init.join(", ")),
                        None => String::new(),
                    };
                    out.push_str(&format!(" It is {list}."));
                }
                Some(out)
            }
            _ => None,
        }
    }
}

fn capitalize(w: &str) -> String {
    let mut c = w.chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}
