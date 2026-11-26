//! Skill format v2: typed predicates with arguments and data-described conditionals.
//!
//! A conditional is Sigma's rule form written as data: patterns over predicates whose
//! arguments are constants or variables, plus a function over the patterns' values (and,
//! optionally, over some of the variables). Nothing in it is executed. The engine grounds
//! each conditional over its variables' domains into ordinary factors that share the
//! conditional's function.
//!
//! The v1 table format ([`SkillPackage`]) is unchanged on the wire and keeps its signatures.
//! [`Skill`] wraps both formats; v2 bodies are signed under a separate BLAKE3 key-derivation
//! context, so a signature can never be replayed across formats.

use crate::{NodeIdentity, PredicateRole, PredicateSchema, SkillPackage, MAX_FACTOR_VALUE, MAX_TABLE_LEN};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Most ground factors one v2 skill may expand to.
pub const MAX_GROUNDINGS: usize = 4096;
/// Most values a symbolic type may have.
pub const MAX_TYPE_VALUES: usize = 256;

/// A symbolic type: an ordered list of constants.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TypeDef {
    pub name: String,
    pub values: Vec<String>,
}

/// A predicate with typed arguments. Each ground tuple of arguments becomes one variable,
/// whose domain is `value_type` (or boolean when `None`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredicateDef {
    pub name: String,
    pub role: PredicateRole,
    pub args: Vec<String>,
    pub value_type: Option<String>,
}

/// How messages flow between a pattern and its conditional (Sigma's three pattern kinds).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternKind {
    /// Read only: working memory informs the conditional, never the reverse.
    Condition,
    /// Both ways (every v1 factor behaves like this).
    Condact,
    /// Write only: the conditional informs working memory, never the reverse.
    Action,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Term {
    Var(String),
    Const(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pattern {
    pub kind: PatternKind,
    pub predicate: String,
    pub args: Vec<Term>,
}

/// `function` is row-major over `function_vars` (outermost) then each pattern's value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Conditional {
    pub name: String,
    pub patterns: Vec<Pattern>,
    pub function_vars: Vec<String>,
    pub function: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Program {
    pub types: Vec<TypeDef>,
    pub predicates: Vec<PredicateDef>,
    pub conditionals: Vec<Conditional>,
}

/// One factor of a grounded skill. `offset` locates its table inside the skill's flat weights.
#[derive(Clone, Debug, PartialEq)]
pub struct GroundFactor {
    pub predicates: Vec<PredicateSchema>,
    pub kinds: Vec<PatternKind>,
    pub values: Vec<f32>,
    pub offset: usize,
}

/// Every ground atom a skill declares, and its factors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Grounding {
    pub atoms: Vec<PredicateSchema>,
    pub factors: Vec<GroundFactor>,
}

/// The name of a ground atom: `Pred` for 0-ary predicates, else `Pred(a,b)`.
pub fn atom_name(predicate: &str, args: &[&str]) -> String {
    if args.is_empty() {
        predicate.into()
    } else {
        format!("{predicate}({})", args.join(","))
    }
}

/// Visits every assignment of a mixed-radix counter with digit sizes `dims`.
fn for_each_assignment(dims: &[usize], mut f: impl FnMut(&[usize])) {
    let mut idx = vec![0; dims.len()];
    loop {
        f(&idx);
        let Some(pos) = (0..dims.len()).rev().find(|&i| idx[i] + 1 < dims[i]) else { return };
        idx[pos] += 1;
        idx[pos + 1..].iter_mut().for_each(|x| *x = 0);
    }
}

fn product(dims: impl IntoIterator<Item = usize>, cap: usize) -> Option<usize> {
    dims.into_iter().try_fold(1usize, |a, d| a.checked_mul(d)).filter(|&n| n <= cap)
}

/// Atom names are built from these, so they must not contain `(`, `,` or `)`.
fn plain(name: &str) -> bool {
    !name.is_empty() && !name.contains(['(', ',', ')'])
}

impl Program {
    /// Checks the program and expands it into ground factors. This is the only
    /// interpretation the engine ever applies to a v2 skill.
    pub fn ground(&self) -> Result<Grounding, &'static str> {
        let mut types: HashMap<&str, &[String]> = HashMap::new();
        for t in &self.types {
            let distinct: HashSet<_> = t.values.iter().collect();
            if t.values.is_empty()
                || t.values.len() > MAX_TYPE_VALUES
                || distinct.len() != t.values.len()
                || !t.values.iter().all(|v| plain(v))
            {
                return Err("invalid type values");
            }
            if types.insert(&t.name, &t.values).is_some() {
                return Err("duplicate type");
            }
        }
        let domain = |p: &PredicateDef| -> Result<usize, &'static str> {
            match &p.value_type {
                None => Ok(2),
                Some(t) => types.get(t.as_str()).map(|v| v.len()).filter(|&n| n >= 2).ok_or("invalid value type"),
            }
        };
        let mut preds: HashMap<&str, (&PredicateDef, usize)> = HashMap::new();
        let mut atoms = Vec::new();
        for p in &self.predicates {
            if !plain(&p.name) {
                return Err("invalid predicate name");
            }
            let arg_types: Vec<&[String]> = p
                .args
                .iter()
                .map(|t| types.get(t.as_str()).copied().ok_or("unknown argument type"))
                .collect::<Result<_, _>>()?;
            let d = domain(p)?;
            if preds.insert(&p.name, (p, d)).is_some() {
                return Err("duplicate predicate");
            }
            let dims: Vec<usize> = arg_types.iter().map(|v| v.len()).collect();
            product(dims.iter().copied(), MAX_GROUNDINGS).ok_or("too many ground atoms")?;
            for_each_assignment(&dims, |a| {
                let args: Vec<&str> = a.iter().zip(&arg_types).map(|(&i, v)| v[i].as_str()).collect();
                atoms.push(PredicateSchema { name: atom_name(&p.name, &args), domain_size: d, role: p.role });
            });
            if atoms.len() > MAX_GROUNDINGS {
                return Err("too many ground atoms");
            }
        }

        let mut factors = Vec::new();
        let mut offset = 0;
        for c in &self.conditionals {
            if c.patterns.is_empty() {
                return Err("conditional without patterns");
            }
            // Variable -> type, from every argument position it appears in.
            let mut var_types: Vec<(&str, &[String])> = Vec::new();
            for pat in &c.patterns {
                let (p, _) = preds.get(pat.predicate.as_str()).ok_or("unknown predicate in pattern")?;
                if pat.args.len() != p.args.len() {
                    return Err("pattern arity mismatch");
                }
                for (term, ty) in pat.args.iter().zip(&p.args) {
                    let values = types[ty.as_str()];
                    match term {
                        Term::Const(k) if !values.contains(k) => return Err("constant outside its type"),
                        Term::Const(_) => {}
                        Term::Var(x) => match var_types.iter().find(|(n, _)| n == x) {
                            Some((_, t)) if !std::ptr::eq(*t, values) => return Err("variable used with two types"),
                            Some(_) => {}
                            None => var_types.push((x, values)),
                        },
                    }
                }
            }
            let fv: Vec<usize> = c
                .function_vars
                .iter()
                .map(|x| var_types.iter().position(|(n, _)| n == x).ok_or("function variable not in any pattern"))
                .collect::<Result<_, _>>()?;
            if fv.iter().collect::<HashSet<_>>().len() != fv.len() {
                return Err("duplicate function variable");
            }
            let value_dims: Vec<usize> = c.patterns.iter().map(|p| preds[p.predicate.as_str()].1).collect();
            let slice_len = product(value_dims.iter().copied(), MAX_TABLE_LEN).ok_or("conditional table too large")?;
            let slices = product(fv.iter().map(|&i| var_types[i].1.len()), MAX_TABLE_LEN).ok_or("too many slices")?;
            if c.function.len() != slice_len * slices {
                return Err("function length does not match conditional");
            }
            if !c.function.iter().all(|v| v.is_finite() && (0.0..=MAX_FACTOR_VALUE).contains(v)) {
                return Err("factor value out of bounds");
            }
            let dims: Vec<usize> = var_types.iter().map(|(_, v)| v.len()).collect();
            if factors.len() + product(dims.iter().copied(), MAX_GROUNDINGS).ok_or("too many groundings")?
                > MAX_GROUNDINGS
            {
                return Err("too many groundings");
            }
            let kinds: Vec<PatternKind> = c.patterns.iter().map(|p| p.kind).collect();
            for_each_assignment(&dims, |a| {
                let value = |x: &str| {
                    let i = var_types.iter().position(|(n, _)| *n == x).expect("bound above");
                    var_types[i].1[a[i]].as_str()
                };
                let predicates: Vec<PredicateSchema> = c
                    .patterns
                    .iter()
                    .map(|pat| {
                        let (p, d) = preds[pat.predicate.as_str()];
                        let args: Vec<&str> = pat
                            .args
                            .iter()
                            .map(|t| match t {
                                Term::Var(x) => value(x),
                                Term::Const(k) => k.as_str(),
                            })
                            .collect();
                        PredicateSchema { name: atom_name(&p.name, &args), domain_size: d, role: p.role }
                    })
                    .collect();
                // A binding that maps two patterns onto the same atom is skipped (atoms in one
                // factor are distinct).
                if predicates.iter().map(|p| &p.name).collect::<HashSet<_>>().len() != predicates.len() {
                    return;
                }
                let slice = fv.iter().fold(0, |acc, &i| acc * dims[i] + a[i]);
                let start = slice * slice_len;
                factors.push(GroundFactor {
                    predicates,
                    kinds: kinds.clone(),
                    values: c.function[start..start + slice_len].to_vec(),
                    offset: offset + start,
                });
            });
            offset += c.function.len();
        }
        Ok(Grounding { atoms, factors })
    }

    /// All conditional functions, concatenated: the skill's initial flat weights.
    pub fn weights(&self) -> Vec<f32> {
        self.conditionals.iter().flat_map(|c| c.function.iter().copied()).collect()
    }
}

/// A signed skill in either wire format.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Skill {
    pub skill_name: String,
    pub version: u64,
    pub author_pubkey: [u8; 32],
    pub signature: Vec<u8>,
    pub body: SkillBody,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SkillBody {
    /// Format v1: one factor table over 0-ary predicates (sent as a plain [`SkillPackage`]).
    Table { predicates: Vec<PredicateSchema>, factor_shape: Vec<usize>, factor_values: Vec<f32> },
    /// Format v2: typed predicates and conditionals.
    Program(Program),
}

impl From<SkillPackage> for Skill {
    fn from(s: SkillPackage) -> Self {
        Skill {
            skill_name: s.skill_name,
            version: s.version,
            author_pubkey: s.author_pubkey,
            signature: s.signature,
            body: SkillBody::Table {
                predicates: s.predicates,
                factor_shape: s.factor_shape,
                factor_values: s.factor_values,
            },
        }
    }
}

impl PartialEq<SkillPackage> for Skill {
    fn eq(&self, other: &SkillPackage) -> bool {
        self.as_v1().is_some_and(|s| s == *other)
    }
}

impl Skill {
    /// A new, unsigned v2 skill.
    pub fn program(skill_name: &str, version: u64, program: Program) -> Self {
        Skill {
            skill_name: skill_name.into(),
            version,
            author_pubkey: [0; 32],
            signature: Vec::new(),
            body: SkillBody::Program(program),
        }
    }

    pub fn format_version(&self) -> u8 {
        match self.body {
            SkillBody::Table { .. } => 1,
            SkillBody::Program(_) => 2,
        }
    }

    /// The v1 package, if this skill is in format v1.
    pub fn as_v1(&self) -> Option<SkillPackage> {
        match &self.body {
            SkillBody::Table { predicates, factor_shape, factor_values } => Some(SkillPackage {
                skill_name: self.skill_name.clone(),
                version: self.version,
                author_pubkey: self.author_pubkey,
                signature: self.signature.clone(),
                predicates: predicates.clone(),
                factor_shape: factor_shape.clone(),
                factor_values: factor_values.clone(),
            }),
            SkillBody::Program(_) => None,
        }
    }

    /// Structural, numerical and cryptographic checks. Must pass before compilation.
    pub fn validate(&self) -> Result<(), &'static str> {
        match &self.body {
            SkillBody::Table { .. } => self.as_v1().expect("table body").validate(),
            SkillBody::Program(p) => {
                p.ground()?;
                NodeIdentity::verify_any_skill(self)
            }
        }
    }

    /// Ground atoms and factors of this skill. Assumes `validate` passed.
    pub fn grounding(&self) -> Result<Grounding, &'static str> {
        match &self.body {
            SkillBody::Table { predicates, factor_values, .. } => Ok(Grounding {
                atoms: predicates.clone(),
                factors: vec![GroundFactor {
                    predicates: predicates.clone(),
                    kinds: vec![PatternKind::Condact; predicates.len()],
                    values: factor_values.clone(),
                    offset: 0,
                }],
            }),
            SkillBody::Program(p) => p.ground(),
        }
    }

    /// Initial flat weights (what `SigmaEngine::skill_weights` returns before any learning).
    pub fn weights(&self) -> Vec<f32> {
        match &self.body {
            SkillBody::Table { factor_values, .. } => factor_values.clone(),
            SkillBody::Program(p) => p.weights(),
        }
    }

    /// Dedup key for the mesh's skill index.
    pub fn signature_key(&self) -> String {
        match self.as_v1() {
            Some(v1) => v1.signature_key(),
            None => format!("v2:{}", self.skill_name),
        }
    }

    /// Whether the skill mentions `atom` (a ground atom name such as `Mounted(nas)`).
    pub fn mentions(&self, atom: &str) -> bool {
        let base = atom.split('(').next().unwrap_or(atom);
        match &self.body {
            SkillBody::Table { predicates, .. } => predicates.iter().any(|p| p.name == atom),
            SkillBody::Program(p) => p.conditionals.iter().flat_map(|c| &c.patterns).any(|pat| pat.predicate == base),
        }
    }
}
