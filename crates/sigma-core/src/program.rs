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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TypeDef {
    pub name: String,
    pub kind: TypeKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TypeKind {
    /// An ordered list of constants (usable as arguments and as values).
    Symbolic(Vec<String>),
    /// The real interval `[lo, hi)`, split into `bins` equal regions (usable as a value type,
    /// which makes a hybrid predicate when its arguments are symbolic).
    Continuous { lo: f32, hi: f32, bins: usize },
}

impl TypeDef {
    pub fn symbolic(name: &str, values: &[&str]) -> Self {
        Self { name: name.into(), kind: TypeKind::Symbolic(values.iter().map(|v| v.to_string()).collect()) }
    }

    pub fn continuous(name: &str, lo: f32, hi: f32, bins: usize) -> Self {
        Self { name: name.into(), kind: TypeKind::Continuous { lo, hi, bins } }
    }
}

/// A function over a conditional's dimensions: `function_vars` (outermost), then each
/// pattern's value. A point's coordinate along a dimension is the symbol's index, or the
/// region's midpoint for a continuous value. Every form is tabulated over the region grid
/// when the skill is compiled.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Function {
    /// Explicit row-major values.
    Table(Vec<f32>),
    /// Sigma's piecewise-linear functions: the first piece whose box contains a point gives
    /// `coef[0] + Σ coef[i+1] · x_i`; points outside every piece get `default`.
    PiecewiseLinear { default: f32, pieces: Vec<Piece> },
    /// A normal density over dimension `target`, whose mean is affine in the coordinates
    /// (`mean[0] + Σ mean[i+1] · x_i`, the target's own coefficient ignored), times the
    /// region width: a continuous distribution in the style of PySigma.
    Gaussian { target: usize, mean: Vec<f32>, std: f32 },
}

/// An axis-aligned box `[lo_i, hi_i)` per dimension with an affine function on it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Piece {
    pub lo: Vec<f32>,
    pub hi: Vec<f32>,
    pub coef: Vec<f32>,
}

/// Most pieces a piecewise-linear function may have.
pub const MAX_PIECES: usize = 1024;

impl Function {
    /// Values at every grid point (row-major); `axes[i]` lists dimension i's coordinates and
    /// `widths[i]` its region width (1 for symbols).
    fn tabulate(&self, axes: &[Vec<f32>], widths: &[f32]) -> Result<Vec<f32>, &'static str> {
        let n = axes.len();
        let len = product(axes.iter().map(|a| a.len()), MAX_TABLE_LEN).ok_or("conditional table too large")?;
        let values = match self {
            Function::Table(v) if v.len() == len => v.clone(),
            Function::Table(_) => return Err("function length does not match conditional"),
            Function::PiecewiseLinear { default, pieces } => {
                if pieces.len() > MAX_PIECES
                    || pieces.iter().any(|p| p.lo.len() != n || p.hi.len() != n || p.coef.len() != n + 1)
                {
                    return Err("invalid piecewise-linear function");
                }
                let mut out = Vec::with_capacity(len);
                for_each_assignment(&axes.iter().map(|a| a.len()).collect::<Vec<_>>(), |idx| {
                    let x: Vec<f32> = idx.iter().zip(axes).map(|(&i, a)| a[i]).collect();
                    let inside = |p: &&Piece| x.iter().zip(p.lo.iter().zip(&p.hi)).all(|(v, (l, h))| l <= v && v < h);
                    out.push(
                        pieces.iter().find(inside).map_or(*default, |p| {
                            p.coef[0] + p.coef[1..].iter().zip(&x).map(|(c, v)| c * v).sum::<f32>()
                        }),
                    );
                });
                out
            }
            Function::Gaussian { target, mean, std } => {
                if *target >= n || mean.len() != n + 1 || !(std.is_finite() && *std > 0.0) {
                    return Err("invalid gaussian function");
                }
                let mut out = Vec::with_capacity(len);
                for_each_assignment(&axes.iter().map(|a| a.len()).collect::<Vec<_>>(), |idx| {
                    let x: Vec<f32> = idx.iter().zip(axes).map(|(&i, a)| a[i]).collect();
                    let mu = mean[0]
                        + mean[1..]
                            .iter()
                            .zip(&x)
                            .enumerate()
                            .filter(|(i, _)| i != target)
                            .map(|(_, (c, v))| c * v)
                            .sum::<f32>();
                    let z = (x[*target] - mu) / std;
                    out.push((-0.5 * z * z).exp() / (std * (2.0 * std::f32::consts::PI).sqrt()) * widths[*target]);
                });
                out
            }
        };
        if values.iter().all(|v| v.is_finite() && (0.0..=MAX_FACTOR_VALUE).contains(v)) {
            Ok(values)
        } else {
            Err("factor value out of bounds")
        }
    }
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Conditional {
    pub name: String,
    pub patterns: Vec<Pattern>,
    pub function_vars: Vec<String>,
    pub function: Function,
    /// Whether gradient learning may change the function.
    pub learnable: bool,
    /// Pattern whose values the function is a distribution over, given the others (Sigma's
    /// normalization variable). `None` normalizes the whole table when learning.
    pub normal: Option<usize>,
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
    pub learnable: bool,
    pub normal: Option<usize>,
}

/// Every ground atom a skill declares, its factors and its initial flat weights.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Grounding {
    pub atoms: Vec<PredicateSchema>,
    /// `[lo, hi)` of every ground atom with a continuous value type.
    pub ranges: Vec<(String, f32, f32)>,
    pub factors: Vec<GroundFactor>,
    pub weights: Vec<f32>,
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
        // Symbolic types, and continuous ones as (lo, hi, bins).
        let mut types: HashMap<&str, &[String]> = HashMap::new();
        let mut reals: HashMap<&str, (f32, f32, usize)> = HashMap::new();
        for t in &self.types {
            let fresh = match &t.kind {
                TypeKind::Symbolic(values) => {
                    let distinct: HashSet<_> = values.iter().collect();
                    if values.is_empty()
                        || values.len() > MAX_TYPE_VALUES
                        || distinct.len() != values.len()
                        || !values.iter().all(|v| plain(v))
                    {
                        return Err("invalid type values");
                    }
                    !reals.contains_key(t.name.as_str()) && types.insert(&t.name, values).is_none()
                }
                &TypeKind::Continuous { lo, hi, bins } => {
                    if !(lo.is_finite() && hi.is_finite() && lo < hi) || !(2..=MAX_TYPE_VALUES).contains(&bins) {
                        return Err("invalid continuous type");
                    }
                    !types.contains_key(t.name.as_str()) && reals.insert(&t.name, (lo, hi, bins)).is_none()
                }
            };
            if !fresh {
                return Err("duplicate type");
            }
        }
        let domain = |p: &PredicateDef| -> Result<usize, &'static str> {
            match &p.value_type {
                None => Ok(2),
                Some(t) => types
                    .get(t.as_str())
                    .map(|v| v.len())
                    .or_else(|| reals.get(t.as_str()).map(|r| r.2))
                    .filter(|&n| n >= 2)
                    .ok_or("invalid value type"),
            }
        };
        // Coordinates and region width along a predicate's value dimension.
        let axis = |p: &PredicateDef, d: usize| -> (Vec<f32>, f32) {
            match p.value_type.as_deref().and_then(|t| reals.get(t)) {
                Some(&(lo, hi, bins)) => {
                    let w = (hi - lo) / bins as f32;
                    ((0..bins).map(|i| lo + (i as f32 + 0.5) * w).collect(), w)
                }
                None => ((0..d).map(|i| i as f32).collect(), 1.0),
            }
        };
        let mut ranges = Vec::new();
        let mut weights = Vec::new();
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
            let range = p.value_type.as_deref().and_then(|t| reals.get(t));
            for_each_assignment(&dims, |a| {
                let args: Vec<&str> = a.iter().zip(&arg_types).map(|(&i, v)| v[i].as_str()).collect();
                let name = atom_name(&p.name, &args);
                if let Some(&(lo, hi, _)) = range {
                    ranges.push((name.clone(), lo, hi));
                }
                atoms.push(PredicateSchema { name, domain_size: d, role: p.role });
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
            if c.normal.is_some_and(|n| n >= c.patterns.len()) {
                return Err("normal pattern out of range");
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
            let (mut axes, mut widths): (Vec<Vec<f32>>, Vec<f32>) =
                fv.iter().map(|&i| ((0..var_types[i].1.len()).map(|x| x as f32).collect(), 1.0)).unzip();
            for p in &c.patterns {
                let (def, d) = preds[p.predicate.as_str()];
                let (a, w) = axis(def, d);
                axes.push(a);
                widths.push(w);
            }
            let function = c.function.tabulate(&axes, &widths)?;
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
                    values: function[start..start + slice_len].to_vec(),
                    offset: offset + start,
                    learnable: c.learnable,
                    normal: c.normal,
                });
            });
            offset += function.len();
            weights.extend(function);
        }
        Ok(Grounding { atoms, ranges, factors, weights })
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
                ranges: Vec::new(),
                weights: factor_values.clone(),
                factors: vec![GroundFactor {
                    predicates: predicates.clone(),
                    kinds: vec![PatternKind::Condact; predicates.len()],
                    values: factor_values.clone(),
                    offset: 0,
                    learnable: true,
                    normal: None,
                }],
            }),
            SkillBody::Program(p) => p.ground(),
        }
    }

    /// Initial flat weights (what `SigmaEngine::skill_weights` returns before any learning):
    /// every conditional's tabulated function, concatenated.
    pub fn weights(&self) -> Vec<f32> {
        self.grounding().map(|g| g.weights).unwrap_or_default()
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
