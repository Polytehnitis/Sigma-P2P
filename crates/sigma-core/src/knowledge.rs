//! Starting knowledge: long-term facts imported from Soar semantic memory, answered by
//! message passing.
//!
//! A Soar agent keeps declarative facts in semantic memory (smem) as a graph of long-term
//! identifiers with attributes (`(<BrennaArgall> ^isa CSFaculty ^officeLocation TechRoomM194)`)
//! and retrieves one identifier with a *cue*: some attribute values it must have. Its rules
//! then read other attributes of what was retrieved. [`KnowledgeBase::from_smem`] reads the
//! `smem --add { … }` blocks of a Soar file and turns them into a v2 program, in the way Sigma
//! represents a semantic memory in the graph:
//!
//! * A hidden predicate `<prefix>Retrieved` ranges over the identifiers (Soar's retrieved
//!   long-term identifier).
//! * An attribute that holds at most one value on every identifier becomes a predicate whose
//!   value is the attribute's value (or `none`), as `FacultyOfficeLocation`. One that can hold
//!   several becomes a boolean predicate per value, as `FacultyIsa(CSFaculty)`.
//! * One conditional per attribute links it to `<prefix>Retrieved`, both ways, with a function
//!   that is 1 where the fact holds and [`MISMATCH`] where it does not. So a cue (evidence on
//!   some attributes) selects the identifiers that match it, and the retrieved identifier
//!   predicts its other attributes: Soar's cue-based retrieval, done by message passing.
//!
//! [`SigmaEngine::ask`] runs a retrieval in the live graph without touching working memory,
//! and [`SigmaEngine::answer`] reads the attribute a question wants. The Soar agent's four
//! question types (`answer-what`, `answer-where`, `answer-who`, `retrieve-adj-info`) are
//! [`Question`]s, and [`KnowledgeBase::parse_question`] reads simple English questions.
//!
//! The starting knowledge ([`starting_knowledge`]) is the faculty directory of the
//! [smem-question-answering](https://github.com/SoarGroup/smem-question-answering) Soar agent.
//! Its data comes from NextKB (Qualitative Reasoning Group, Northwestern University), under
//! the Creative Commons Attribution 4.0 license; see `knowledge/smem-question-answering/`.
//!
//! Starting knowledge is compiled as a skill signed by the node itself, at every start. It is
//! not stored, synced or offered to peers, since every node already has it, and its predicates
//! are long-term facts rather than the situation, so no action, transition or structure models
//! are learned for them.

use crate::program::{Conditional, Function, Pattern, PatternKind, PredicateDef, Program, Skill, Term, TypeDef};
use crate::{NodeIdentity, PredicateRole, SigmaEngine};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Function value where a fact does not hold. Small but not zero, so a cue that matches
/// nothing still leaves a (failed) posterior rather than an empty one.
pub const MISMATCH: f32 = 1e-4;
/// The value of a single-valued attribute on an identifier that lacks it.
pub const NONE: &str = "none";
/// The Soar agent behind the starting knowledge.
pub const SOAR_QA_SOURCE: &str = include_str!("../knowledge/smem-question-answering/SMEM.soar");
/// Name of the starting-knowledge base (and of its skill, `knowledge:<name>`).
pub const SOAR_QA_NAME: &str = "smem-question-answering";
/// Predicate prefix of the starting-knowledge base.
pub const SOAR_QA_PREFIX: &str = "Faculty";

/// A Soar semantic memory: identifier -> attribute -> values.
pub type SmemGraph = BTreeMap<String, BTreeMap<String, BTreeSet<String>>>;

/// Reads every `smem --add { … }` block of a Soar source file. Identifiers (`<x>` or `@x`) are
/// named without their brackets, values that are identifiers become their names, and `|…|`
/// strings keep their text (with `(`, `)` and `,`, which atom names cannot hold, made `_`).
pub fn parse_smem(source: &str) -> Result<SmemGraph, String> {
    let tokens = tokenize(source)?;
    let mut graph = SmemGraph::new();
    let mut i = 0;
    while i < tokens.len() {
        // A block starts with `smem --add {` (`smem -a` in older Soar).
        if tokens[i] == "smem" && matches!(tokens.get(i + 1).map(String::as_str), Some("--add" | "-a")) {
            if tokens.get(i + 2).map(String::as_str) != Some("{") {
                return Err("expected `{` after `smem --add`".into());
            }
            i += 3;
            loop {
                match tokens.get(i).map(String::as_str) {
                    Some("}") => break,
                    Some("(") => i = parse_group(&tokens, i + 1, &mut graph)?,
                    Some(t) => return Err(format!("unexpected `{t}` in smem --add")),
                    None => return Err("unterminated smem --add block".into()),
                }
            }
        }
        i += 1;
    }
    Ok(graph)
}

/// One `(<id> ^attr value … ^attr value …)` group, starting after `(`. Returns the index after `)`.
fn parse_group(tokens: &[String], mut i: usize, graph: &mut SmemGraph) -> Result<usize, String> {
    let id = tokens.get(i).and_then(|t| identifier(t)).ok_or("expected an identifier after `(`")?;
    i += 1;
    let node = graph.entry(id).or_default();
    let mut attr: Option<String> = None;
    loop {
        let t = tokens.get(i).ok_or("unterminated smem group")?;
        i += 1;
        match t.as_str() {
            ")" => return Ok(i),
            "(" | "{" | "}" => return Err(format!("unexpected `{t}` in smem group")),
            _ if t.starts_with('^') && t.len() > 1 => attr = Some(t[1..].to_string()),
            _ => {
                let a = attr.as_ref().ok_or("value without an attribute")?;
                let v = identifier(t).unwrap_or_else(|| clean(t));
                if v.is_empty() {
                    return Err("empty value".into());
                }
                node.entry(a.clone()).or_default().insert(v);
            }
        }
    }
}

fn identifier(t: &str) -> Option<String> {
    let name = t.strip_prefix('<').and_then(|s| s.strip_suffix('>')).or_else(|| t.strip_prefix('@'))?;
    (!name.is_empty()).then(|| clean(name))
}

fn clean(t: &str) -> String {
    t.trim_matches('|').replace(['(', ')', ','], "_")
}

/// Soar tokens: brackets, `|strings|`, and runs of anything else; `#` starts a comment.
fn tokenize(source: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut chars = source.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            _ if c.is_whitespace() => _ = chars.next(),
            '#' => while chars.next().is_some_and(|c| c != '\n') {},
            '(' | ')' | '{' | '}' => {
                chars.next();
                out.push(c.to_string());
            }
            '|' => {
                chars.next();
                let mut s = String::from("|");
                loop {
                    match chars.next() {
                        Some('|') => break,
                        Some(c) => s.push(c),
                        None => return Err("unterminated |string|".into()),
                    }
                }
                s.push('|');
                out.push(s);
            }
            _ => {
                let mut s = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_whitespace() || "(){}|".contains(c) {
                        break;
                    }
                    s.push(c);
                    chars.next();
                }
                out.push(s);
            }
        }
    }
    Ok(out)
}

/// One attribute of a knowledge base, as a predicate.
#[derive(Clone, Debug, PartialEq)]
pub struct Attribute {
    /// The Soar attribute (`officeLocation`).
    pub name: String,
    /// The predicate (`FacultyOfficeLocation`).
    pub predicate: String,
    /// Several values per identifier: a boolean predicate per value.
    pub multi: bool,
    /// Every value it takes, in order.
    pub values: Vec<String>,
}

/// A Soar semantic memory as a Sigma program (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct KnowledgeBase {
    pub name: String,
    pub prefix: String,
    pub graph: SmemGraph,
    /// The identifiers, in order: the values of `<prefix>Retrieved`.
    pub nodes: Vec<String>,
    pub attributes: Vec<Attribute>,
}

/// A retrieval: what a cue recalled.
#[derive(Clone, Debug, PartialEq)]
pub struct Retrieval {
    /// The identifier retrieved: the most probable one, the first in order among equals.
    pub node: String,
    /// Its posterior given the cue.
    pub confidence: f32,
    /// Every identifier that matches the cue, most probable first (Soar retrieves one of them).
    pub matches: Vec<(String, f32)>,
    /// The retrieved identifier's attributes as the graph predicts them, by Soar attribute.
    pub attributes: Vec<(String, Vec<String>)>,
}

impl Retrieval {
    /// Values of one attribute of the retrieved identifier.
    pub fn get(&self, attribute: &str) -> &[String] {
        self.attributes.iter().find(|(a, _)| a == attribute).map_or(&[], |(_, v)| v.as_slice())
    }
}

/// A question: a cue, and the attribute of the retrieved identifier that answers it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Question {
    pub cue: Vec<(String, String)>,
    pub want: String,
}

impl Question {
    pub fn new(cue: &[(&str, &str)], want: &str) -> Self {
        Self { cue: cue.iter().map(|(a, v)| (a.to_string(), v.to_string())).collect(), want: want.into() }
    }

    /// The Soar agent's `answer-what`: what kind of thing `node` is.
    pub fn what(node: &str) -> Self {
        Self::new(&[("krfID", node)], "isa")
    }

    /// The Soar agent's `answer-where`: where `node`'s office is.
    pub fn where_is(node: &str) -> Self {
        Self::new(&[("krfID", node)], "officeLocation")
    }

    /// The Soar agent's `answer-who`: the full name of whoever has `attribute = value`.
    pub fn who(attribute: &str, value: &str) -> Self {
        Self::new(&[(attribute, value)], "fullName")
    }

    /// The Soar agent's `retrieve-adj-info`: something stated in microtheory `mt`.
    pub fn adjacent(mt: &str) -> Self {
        Self::new(&[("in-microtheory", mt)], "krfID")
    }
}

/// An answer: the values wanted, and the whole retrieval behind them.
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    pub values: Vec<String>,
    pub retrieval: Retrieval,
}

/// The seven questions the Soar agent asks, in order, as its propose rules state them.
pub fn soar_agent_questions() -> Vec<Question> {
    vec![
        Question::what("InstructorTBA"),
        Question::what("BrennaArgall"),
        Question::where_is("BrennaArgall"),
        Question::where_is("YanChen"),
        Question::who("isa", "CSResearchFaculty"),
        Question::who("phoneNumberOf", "8474671256"),
        Question::adjacent("NUPeopleLanguageInfoMt"),
    ]
}

/// `in-microtheory` -> `InMicrotheory`, `krfID` -> `KrfID`.
fn camel(name: &str) -> String {
    name.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_uppercase().chain(c).collect::<String>()).unwrap_or_default()
        })
        .collect()
}

/// Lowercase letters and digits only, for matching words against names.
fn key(text: &str) -> String {
    text.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// Words that point at an attribute in an English question, if the knowledge base has it.
const ATTRIBUTE_WORDS: &[(&str, &[&str])] = &[
    ("phoneNumberOf", &["phone", "telephone", "call", "number"]),
    ("emailOf", &["email", "mail", "address"]),
    ("personalWebsite", &["website", "site", "homepage", "web", "url"]),
    ("facultyDepartment", &["department", "dept"]),
    ("titleOfPerson", &["title"]),
    ("schooling", &["school", "schools", "study", "studied", "educated", "schooling", "alma", "university"]),
    ("nextArg", &["degree", "degrees"]),
    ("fullName", &["name", "called"]),
    ("in-microtheory", &["microtheory", "mt"]),
    ("genlMt", &["genlmt"]),
    // Last, so "where did she study" asks for schooling.
    ("officeLocation", &["where", "office", "room", "located", "location"]),
];

impl KnowledgeBase {
    /// Builds a knowledge base from a Soar source holding `smem --add` blocks. Predicates are
    /// named `<prefix><Attribute>`.
    pub fn from_smem(name: &str, prefix: &str, source: &str) -> Result<Self, String> {
        let graph = parse_smem(source)?;
        if graph.is_empty() {
            return Err("no smem --add facts".into());
        }
        let nodes: Vec<String> = graph.keys().cloned().collect();
        let mut by_attr: BTreeMap<&str, (BTreeSet<&str>, bool)> = BTreeMap::new();
        for attrs in graph.values() {
            for (a, values) in attrs {
                let e = by_attr.entry(a).or_default();
                e.0.extend(values.iter().map(String::as_str));
                e.1 |= values.len() > 1;
            }
        }
        let attributes = by_attr
            .into_iter()
            .map(|(a, (values, multi))| {
                let mut values: Vec<String> = values.into_iter().map(String::from).collect();
                if !multi {
                    // `none` first, so an identifier without the attribute has a value too.
                    let none = (0..).map(|i| format!("{NONE}{}", "_".repeat(i))).find(|n| !values.contains(n));
                    values.insert(0, none.expect("some name is free"));
                }
                Attribute { name: a.to_string(), predicate: format!("{prefix}{}", camel(a)), multi, values }
            })
            .collect();
        let kb = Self { name: name.into(), prefix: prefix.into(), graph, nodes, attributes };
        kb.program().ground().map_err(|e| format!("{name}: {e}"))?;
        Ok(kb)
    }

    /// The predicate over the identifiers (`<prefix>Retrieved`).
    pub fn retrieved(&self) -> String {
        format!("{}Retrieved", self.prefix)
    }

    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Whether identifier `node` has `attribute = value`.
    pub fn holds(&self, node: &str, attribute: &str, value: &str) -> bool {
        self.graph.get(node).and_then(|a| a.get(attribute)).is_some_and(|v| v.contains(value))
    }

    /// The knowledge as a v2 program: one conditional per attribute (see the module docs).
    pub fn program(&self) -> Program {
        let node_type = format!("{}-node", self.prefix);
        let mut types = vec![TypeDef::symbolic(&node_type, &self.nodes.iter().map(String::as_str).collect::<Vec<_>>())];
        let mut predicates = vec![PredicateDef {
            name: self.retrieved(),
            role: PredicateRole::Context,
            args: vec![],
            value_type: Some(node_type),
        }];
        let mut conditionals = Vec::new();
        let retrieved = Pattern { kind: PatternKind::Condact, predicate: self.retrieved(), args: vec![] };
        for a in &self.attributes {
            let ty = format!("{}-{}", self.prefix, a.name);
            types.push(TypeDef::symbolic(&ty, &a.values.iter().map(String::as_str).collect::<Vec<_>>()));
            let fact = |n: &str, v: &str| if self.holds(n, &a.name, v) { 1.0 } else { MISMATCH };
            let (pattern, function_vars, table) = if a.multi {
                // [value, node, holds]: a boolean atom per value.
                predicates.push(PredicateDef {
                    name: a.predicate.clone(),
                    role: PredicateRole::Context,
                    args: vec![ty],
                    value_type: None,
                });
                let mut t = Vec::with_capacity(a.values.len() * self.nodes.len() * 2);
                for v in &a.values {
                    for n in &self.nodes {
                        let f = fact(n, v);
                        t.extend([if f == 1.0 { MISMATCH } else { 1.0 }, f]);
                    }
                }
                let args = vec![Term::Var("v".into())];
                (Pattern { kind: PatternKind::Condact, predicate: a.predicate.clone(), args }, vec!["v".into()], t)
            } else {
                // [node, value], with `none` where the identifier lacks the attribute.
                predicates.push(PredicateDef {
                    name: a.predicate.clone(),
                    role: PredicateRole::Context,
                    args: vec![],
                    value_type: Some(ty),
                });
                let mut t = Vec::with_capacity(self.nodes.len() * a.values.len());
                for n in &self.nodes {
                    let has = self.graph[n].contains_key(&a.name);
                    t.push(if has { MISMATCH } else { 1.0 });
                    t.extend(a.values[1..].iter().map(|v| fact(n, v)));
                }
                (Pattern { kind: PatternKind::Condact, predicate: a.predicate.clone(), args: vec![] }, vec![], t)
            };
            conditionals.push(Conditional {
                name: format!("{}-{}", self.name, a.name),
                patterns: vec![retrieved.clone(), pattern],
                function_vars,
                function: Function::Table(table),
                learnable: false,
                normal: None,
            });
        }
        Program { types, predicates, conditionals }
    }

    /// The knowledge as a v2 skill signed by `author`. Signing is deterministic, so every
    /// start compiles the same skill.
    pub fn skill(&self, author: &NodeIdentity) -> Skill {
        let mut skill = Skill::program(&format!("knowledge:{}", self.name), 1, self.program());
        author.sign(&mut skill);
        skill
    }

    /// Every ground atom of the knowledge base.
    pub fn atoms(&self) -> Vec<String> {
        let mut out = vec![self.retrieved()];
        for a in &self.attributes {
            if a.multi {
                out.extend(a.values.iter().map(|v| crate::program::atom_name(&a.predicate, &[v])));
            } else {
                out.push(a.predicate.clone());
            }
        }
        out
    }

    /// Reads a simple English question about the knowledge:
    ///
    /// * naming an identifier: "where is Brenna Argall?", "what is InstructorTBA?",
    ///   "what is Larry Birnbaum's email?" (the attribute defaults to `isa`);
    /// * naming a value: "who has the phone number 8474671256?", "who is a CSResearchFaculty?"
    ///   (answered with the full name of whoever has it, or the identifier).
    pub fn parse_question(&self, text: &str) -> Option<Question> {
        let words: Vec<String> = crate::language::normalize(text).split(' ').map(String::from).collect();
        let nodes: HashMap<String, &str> = self.nodes.iter().map(|n| (key(n), n.as_str())).collect();
        let mut values: HashMap<String, (&str, &str)> = HashMap::new();
        for a in self.attributes.iter().filter(|a| a.name != "krfID") {
            for v in a.values.iter().filter(|v| a.multi || *v != &a.values[0]) {
                values.entry(key(v)).or_insert((a.name.as_str(), v.as_str()));
            }
        }
        // The longest run of words that spells an identifier or a value.
        let mut best: Option<(usize, usize, Mention)> = None;
        for i in 0..words.len() {
            for j in (i + 1..=words.len()).rev() {
                let k: String = words[i..j].concat();
                let found = nodes
                    .get(&k)
                    .map(|n| Mention::Node(n.to_string()))
                    .or_else(|| values.get(&k).map(|(a, v)| Mention::Value(a.to_string(), v.to_string())));
                if let Some(m) = found {
                    if best.as_ref().is_none_or(|(bi, bj, _)| j - i > bj - bi) {
                        best = Some((i, j, m));
                    }
                    break;
                }
            }
        }
        let (i, j, mention) = best?;
        // Attribute words outside the mention.
        let rest: Vec<&str> = words[..i].iter().chain(&words[j..]).map(String::as_str).collect();
        let asked = ATTRIBUTE_WORDS
            .iter()
            .filter(|(a, _)| self.attribute(a).is_some())
            .find(|(_, ws)| rest.iter().any(|w| ws.contains(w)))
            .map(|(a, _)| a.to_string())
            .or_else(|| {
                self.attributes.iter().map(|a| a.name.clone()).find(|a| {
                    let ws = crate::language::name_words(a);
                    !ws.is_empty() && ws.iter().all(|w| rest.contains(&w.as_str()))
                })
            });
        match mention {
            Mention::Node(n) => {
                let want = asked.unwrap_or_else(|| "isa".into());
                self.attribute(&want)?;
                Some(Question::new(&[("krfID", &n)], &want).with_node_attribute(self, &n))
            }
            Mention::Value(a, v) => {
                let fallback = if self.attribute("fullName").is_some() { "fullName" } else { "krfID" };
                let want = asked.filter(|w| *w != a).unwrap_or_else(|| fallback.into());
                Some(Question::new(&[(&a, &v)], &want))
            }
        }
    }

    /// An answer as a sentence.
    pub fn say(&self, question: &Question, answer: &Answer) -> String {
        let words = |a: &str| {
            let mut w = crate::language::name_words(a);
            if w.len() > 1 && w.last().is_some_and(|l| l == "of") {
                w.pop();
            }
            w.join(" ")
        };
        let node = &answer.retrieval.node;
        if answer.values.is_empty() {
            return format!("{node} has no {} that I know of.", words(&question.want));
        }
        let values = answer.values.join(" and ");
        if question.cue.iter().any(|(a, _)| a == "krfID" || a == "@node") {
            return format!("{node}: {} {values}.", words(&question.want));
        }
        let who = if values == *node { values } else { format!("{values} ({node})") };
        let cue: Vec<String> = question.cue.iter().map(|(a, v)| format!("{} {v}", words(a))).collect();
        let more = match answer.retrieval.matches.len().saturating_sub(1) {
            0 => String::new(),
            n => format!(" ({n} more match)"),
        };
        format!("{who} has {}{more}.", cue.join(" and "))
    }
}

enum Mention {
    Node(String),
    Value(String, String),
}

impl Question {
    /// A cue on `krfID` only works when the knowledge base has that attribute; otherwise the
    /// identifier is cued directly.
    fn with_node_attribute(mut self, kb: &KnowledgeBase, node: &str) -> Self {
        if kb.attribute("krfID").is_none() {
            self.cue = vec![("@node".into(), node.into())];
        }
        self
    }
}

/// The starting knowledge every node loads: the smem-question-answering Soar agent's faculty
/// directory.
pub fn starting_knowledge() -> Vec<KnowledgeBase> {
    vec![KnowledgeBase::from_smem(SOAR_QA_NAME, SOAR_QA_PREFIX, SOAR_QA_SOURCE).expect("bundled knowledge is valid")]
}

impl SigmaEngine {
    /// Compiles a knowledge base as a skill signed by `author` and keeps it for [`Self::ask`].
    pub fn install_knowledge(&mut self, kb: KnowledgeBase, author: &NodeIdentity) -> Result<(), &'static str> {
        self.compile_skill(kb.skill(author))?;
        self.knowledge_atoms.extend(kb.atoms());
        self.knowledge.retain(|k| k.name != kb.name);
        self.knowledge.push(std::sync::Arc::new(kb));
        Ok(())
    }

    /// Installs [`starting_knowledge`], signed by `author` (the node itself).
    pub fn install_starting_knowledge(&mut self, author: &NodeIdentity) -> Result<(), &'static str> {
        starting_knowledge().into_iter().try_for_each(|kb| self.install_knowledge(kb, author))
    }

    /// The installed knowledge bases.
    pub fn knowledge_bases(&self) -> Vec<&KnowledgeBase> {
        self.knowledge.iter().map(|k| k.as_ref()).collect()
    }

    /// Whether `name` is an atom of installed knowledge (no models are learned for those).
    pub(crate) fn is_knowledge(&self, name: &str) -> bool {
        self.knowledge_atoms.contains(name)
    }

    /// Soar's cue-based retrieval, by message passing in the live graph: the cue is clamped
    /// (working memory is left as it is), the posterior over `<prefix>Retrieved` picks the
    /// identifier, and the identifier, clamped in turn, predicts all its attributes. `None` is
    /// a retrieval failure: an unknown attribute or value, or no identifier with the whole cue.
    pub fn ask(&self, kb: &str, cue: &[(String, String)]) -> Option<Retrieval> {
        let kb = self.knowledge.iter().find(|k| k.name == kb)?.clone();
        let retrieved = *self.index.get(&kb.retrieved())?;
        let mut clamps: HashMap<usize, usize> = HashMap::new();
        for (a, v) in cue {
            if a == "@node" {
                clamps.insert(retrieved, kb.nodes.iter().position(|n| n == v)?);
                continue;
            }
            let attr = kb.attribute(a)?;
            let x = attr.values.iter().position(|w| w == v)?;
            if attr.multi {
                clamps.insert(*self.index.get(&crate::program::atom_name(&attr.predicate, &[v]))?, 1);
            } else {
                clamps.insert(*self.index.get(&attr.predicate)?, x);
            }
        }
        let beliefs = self.infer(|v| clamps.get(&v).copied());
        let post = &beliefs[retrieved];
        let mut ranked: Vec<(usize, f32)> = post.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let &(best, confidence) = ranked.first()?;
        let full = |n: usize| {
            cue.iter().all(|(a, v)| if a == "@node" { kb.nodes[n] == *v } else { kb.holds(&kb.nodes[n], a, v) })
        };
        if !full(best) {
            return None;
        }
        let matches =
            ranked.iter().filter(|&&(n, p)| p >= confidence * 0.5 && full(n)).map(|&(n, p)| (kb.nodes[n].clone(), p));
        let matches = matches.collect();
        // The retrieved identifier's attributes, as the graph predicts them.
        let given = self.infer(|v| (v == retrieved).then_some(best));
        let mut attributes = Vec::new();
        for a in &kb.attributes {
            let values: Vec<String> = if a.multi {
                a.values
                    .iter()
                    .filter(|v| {
                        let atom = crate::program::atom_name(&a.predicate, &[v]);
                        self.index.get(&atom).is_some_and(|&i| given[i].get(1).is_some_and(|&p| p > 0.5))
                    })
                    .cloned()
                    .collect()
            } else {
                let b = &given[self.index[&a.predicate]];
                let (x, p) = b.iter().copied().enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap_or((0, 0.0));
                if x == 0 || p <= 0.5 {
                    continue;
                }
                vec![a.values[x].clone()]
            };
            if !values.is_empty() {
                attributes.push((a.name.clone(), values));
            }
        }
        Some(Retrieval { node: kb.nodes[best].clone(), confidence, matches, attributes })
    }

    /// Answers a question from knowledge base `kb`: retrieves with its cue and reads the
    /// attribute it wants.
    pub fn answer(&self, kb: &str, question: &Question) -> Option<Answer> {
        let retrieval = self.ask(kb, &question.cue)?;
        let values = retrieval.get(&question.want).to_vec();
        Some(Answer { values, retrieval })
    }

    /// Answers an English question from the first installed knowledge base that understands
    /// it, as a sentence.
    pub fn answer_text(&self, text: &str) -> Option<String> {
        self.knowledge.iter().find_map(|kb| {
            let q = kb.parse_question(text)?;
            let a = self.answer(&kb.name, &q)?;
            Some(kb.say(&q, &a))
        })
    }
}
