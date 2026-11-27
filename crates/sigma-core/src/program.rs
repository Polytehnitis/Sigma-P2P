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
use std::sync::Arc;

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

    /// The integers `lo..hi` as an ordered symbolic type (`"lo"`, …), for coordinates that
    /// affine transforms shift, mirror or rotate.
    pub fn range(name: &str, lo: i32, hi: i32) -> Self {
        Self { name: name.into(), kind: TypeKind::Symbolic((lo..hi).map(|i| i.to_string()).collect()) }
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
    /// An analytic form read through an affine map of its continuous coordinates: dimension
    /// `i` with `maps[i] = Some(m)` takes the value `x_i = m.coef · u_i + m.offset` where the
    /// inner `function` saw `u_i` (so a piece over `[lo, hi)` covers the mapped interval, and
    /// a Gaussian becomes the density of the mapped value). Lets one function be reused in
    /// other units, mirrored or shifted, as an affine pattern does for symbols.
    Mapped { maps: Vec<Option<ValueMap>>, function: Box<Function> },
}

/// `x = coef · u + offset`, for one dimension of a [`Function::Mapped`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValueMap {
    pub coef: f32,
    pub offset: f32,
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
    /// `function` with dimension `dim` (of `n`) read through `x = coef · u + offset`.
    pub fn mapped(function: Function, n: usize, dim: usize, coef: f32, offset: f32) -> Function {
        let mut maps = vec![None; n];
        if dim < n {
            maps[dim] = Some(ValueMap { coef, offset });
        }
        Function::Mapped { maps, function: Box::new(function) }
    }

    /// The same function without any [`Function::Mapped`] layer: the map is folded into the
    /// pieces' boxes and coefficients, or into the Gaussian's mean and spread, exactly.
    pub fn resolved(&self) -> Result<Function, &'static str> {
        let Function::Mapped { maps, function } = self else { return Ok(self.clone()) };
        let n = maps.len();
        if maps.iter().flatten().any(|m| !(m.coef.is_finite() && m.coef != 0.0 && m.offset.is_finite())) {
            return Err("invalid value map");
        }
        // u_i = (x_i - offset_i) / coef_i = a_i · x_i + b_i
        let a: Vec<f32> = maps.iter().map(|m| m.map_or(1.0, |m| 1.0 / m.coef)).collect();
        let b: Vec<f32> = maps.iter().map(|m| m.map_or(0.0, |m| -m.offset / m.coef)).collect();
        let x_of = |i: usize, u: f32| maps[i].map_or(u, |m| m.coef * u + m.offset);
        let out = match function.as_ref() {
            Function::Table(_) => return Err("a mapped function must be analytic"),
            Function::Mapped { .. } => return Err("nested value maps"),
            Function::PiecewiseLinear { default, pieces } => {
                function.check(n)?;
                let pieces = pieces
                    .iter()
                    .map(|p| {
                        let (mut lo, mut hi) = (Vec::with_capacity(n), Vec::with_capacity(n));
                        for i in 0..n {
                            let (l, h) = (x_of(i, p.lo[i]), x_of(i, p.hi[i]));
                            lo.push(l.min(h));
                            hi.push(l.max(h));
                        }
                        let mut coef = vec![p.coef[0] + (0..n).map(|i| p.coef[i + 1] * b[i]).sum::<f32>()];
                        coef.extend((0..n).map(|i| p.coef[i + 1] * a[i]));
                        Piece { lo, hi, coef }
                    })
                    .collect();
                Function::PiecewiseLinear { default: *default, pieces }
            }
            Function::Gaussian { target, mean, std } => {
                function.check(n)?;
                let t = *target;
                // mean in u: m0 + Σ m_i (a_i x_i + b_i); x_t = (u_t - b_t) / a_t.
                let (at, bt) = (a[t], b[t]);
                let mut m =
                    vec![(mean[0] + (0..n).filter(|&i| i != t).map(|i| mean[i + 1] * b[i]).sum::<f32>() - bt) / at];
                m.extend((0..n).map(|i| if i == t { 0.0 } else { mean[i + 1] * a[i] / at }));
                Function::Gaussian { target: t, mean: m, std: std / at.abs() }
            }
        };
        out.check(n)?;
        Ok(out)
    }

    /// Shape checks for the analytic forms over `n` dimensions.
    fn check(&self, n: usize) -> Result<(), &'static str> {
        match self {
            Function::Table(_) => Ok(()),
            Function::Mapped { maps, .. } if maps.len() != n => Err("value maps do not match the patterns"),
            Function::Mapped { .. } => self.resolved().map(|_| ()),
            Function::PiecewiseLinear { pieces, .. } => {
                if pieces.len() > MAX_PIECES
                    || pieces.iter().any(|p| p.lo.len() != n || p.hi.len() != n || p.coef.len() != n + 1)
                {
                    return Err("invalid piecewise-linear function");
                }
                Ok(())
            }
            Function::Gaussian { target, mean, std } => {
                if *target >= n || mean.len() != n + 1 || !(std.is_finite() && *std > 0.0) {
                    return Err("invalid gaussian function");
                }
                Ok(())
            }
        }
    }

    /// The value of an analytic form at point `x` (a `Table` has none and yields 0). A
    /// Gaussian is a density times `widths[target]` (pass 1s for the pure density).
    pub fn eval(&self, x: &[f32], widths: &[f32]) -> f32 {
        match self {
            Function::Table(_) => 0.0,
            Function::Mapped { .. } => self.resolved().map_or(0.0, |f| f.eval(x, widths)),
            Function::PiecewiseLinear { default, pieces } => {
                let inside = |p: &&Piece| x.iter().zip(p.lo.iter().zip(&p.hi)).all(|(v, (l, h))| l <= v && v < h);
                pieces
                    .iter()
                    .find(inside)
                    .map_or(*default, |p| p.coef[0] + p.coef[1..].iter().zip(x).map(|(c, v)| c * v).sum::<f32>())
            }
            Function::Gaussian { target, mean, std } => {
                let mu = mean[0]
                    + mean[1..]
                        .iter()
                        .zip(x)
                        .enumerate()
                        .filter(|(i, _)| i != target)
                        .map(|(_, (c, v))| c * v)
                        .sum::<f32>();
                let z = (x[*target] - mu) / std;
                (-0.5 * z * z).exp() / (std * (2.0 * std::f32::consts::PI).sqrt()) * widths[*target]
            }
        }
    }

    /// Values at every grid point (row-major); `axes[i]` lists dimension i's coordinates and
    /// `widths[i]` its region width (1 for symbols).
    fn tabulate(&self, axes: &[Vec<f32>], widths: &[f32]) -> Result<Vec<f32>, &'static str> {
        let n = axes.len();
        let len = product(axes.iter().map(|a| a.len()), MAX_TABLE_LEN).ok_or("conditional table too large")?;
        let values = match self {
            Function::Table(v) if v.len() == len => v.clone(),
            Function::Table(_) => return Err("function length does not match conditional"),
            _ => {
                self.check(n)?;
                let resolved = self.resolved()?;
                let mut out = Vec::with_capacity(len);
                for_each_assignment(&axes.iter().map(|a| a.len()).collect::<Vec<_>>(), |idx| {
                    let x: Vec<f32> = idx.iter().zip(axes).map(|(&i, a)| a[i]).collect();
                    out.push(resolved.eval(&x, widths));
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
    /// Sigma's affine transform of a variable: the argument is the value at index
    /// `coef · i + offset` of the argument's (ordered) type, where `i` is the variable's index.
    /// Where that falls outside the type, the pattern is padded: it holds the constant value
    /// `pad` (an index into the predicate's value domain), or with no `pad` the binding is
    /// skipped. This is how a conditional shifts, mirrors or rotates a map or an image.
    Affine {
        var: String,
        coef: i32,
        offset: i32,
        pad: Option<usize>,
    },
}

impl Term {
    /// A shift by `offset` (Sigma's `:offset`), padded with value `pad` outside the type.
    pub fn shift(var: &str, offset: i32, pad: Option<usize>) -> Self {
        Term::Affine { var: var.into(), coef: 1, offset, pad }
    }
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
    /// The analytic function behind a piecewise-linear or Gaussian factor, for evaluation at
    /// exact (particle) points rather than region midpoints.
    pub analytic: Option<Arc<Analytic>>,
    /// Flat weight position of each value, when they are not contiguous from `offset`.
    pub layout: Option<Vec<usize>>,
}

/// An analytic conditional function restricted to one binding of its function variables.
#[derive(Clone, Debug, PartialEq)]
pub struct Analytic {
    pub function: Function,
    /// Coordinates of the bound function variables (the leading dimensions).
    pub prefix: Vec<f32>,
}

impl Analytic {
    /// Value at slot coordinates `x` (a symbol's index, or the real value of a continuous slot);
    /// a Gaussian gives its pure density.
    pub fn eval(&self, x: &[f32]) -> f32 {
        let point: Vec<f32> = self.prefix.iter().chain(x).copied().collect();
        let v = self.function.eval(&point, &vec![1.0; point.len()]);
        if v.is_finite() {
            v.clamp(0.0, MAX_FACTOR_VALUE)
        } else {
            0.0
        }
    }
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
            if crate::graph_memory::internal(&p.name) {
                return Err("reserved predicate name");
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
                        Term::Affine { coef: 0, .. } => return Err("affine coefficient is zero"),
                        Term::Affine { pad: Some(v), .. } if *v >= preds[pat.predicate.as_str()].1 => {
                            return Err("pad outside the predicate's values")
                        }
                        Term::Var(x) | Term::Affine { var: x, .. } => match var_types.iter().find(|(n, _)| n == x) {
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
                let index = |x: &str| var_types.iter().position(|(n, _)| *n == x).expect("bound above");
                // Each pattern's ground atom, or the value it is padded with.
                let mut atoms: Vec<Result<PredicateSchema, usize>> = Vec::with_capacity(c.patterns.len());
                for pat in &c.patterns {
                    let (p, d) = preds[pat.predicate.as_str()];
                    let mut args: Vec<&str> = Vec::with_capacity(pat.args.len());
                    let mut pad = None;
                    for (t, ty) in pat.args.iter().zip(&p.args) {
                        match t {
                            Term::Var(x) => args.push(var_types[index(x)].1[a[index(x)]].as_str()),
                            Term::Const(k) => args.push(k.as_str()),
                            Term::Affine { var, coef, offset, pad: fill } => {
                                let values = types[ty.as_str()];
                                let j = *coef as i64 * a[index(var)] as i64 + *offset as i64;
                                match usize::try_from(j).ok().filter(|&j| j < values.len()) {
                                    Some(j) => args.push(values[j].as_str()),
                                    None => match fill {
                                        Some(v) => pad = pad.or(Some(*v)),
                                        None => return, // outside the type, unpadded: no binding
                                    },
                                }
                            }
                        }
                    }
                    atoms.push(match pad {
                        Some(v) => Err(v),
                        None => Ok(PredicateSchema { name: atom_name(&p.name, &args), domain_size: d, role: p.role }),
                    });
                }
                let predicates: Vec<PredicateSchema> = atoms.iter().filter_map(|x| x.clone().ok()).collect();
                // A binding that maps two patterns onto the same atom is skipped (atoms in one
                // factor are distinct), and so is one with every pattern padded.
                if predicates.is_empty()
                    || predicates.iter().map(|p| &p.name).collect::<HashSet<_>>().len() != predicates.len()
                {
                    return;
                }
                let slice = fv.iter().fold(0, |acc, &i| acc * dims[i] + a[i]);
                let start = slice * slice_len;
                let padded = predicates.len() < c.patterns.len();
                let (values, layout) = if padded {
                    // Keep the entries at the padded patterns' fixed values.
                    let (mut values, mut layout) = (Vec::new(), Vec::new());
                    for_each_assignment(&value_dims, |idx| {
                        if atoms.iter().zip(idx).all(|(x, &i)| x.as_ref().err().is_none_or(|&v| v == i)) {
                            let flat = idx.iter().zip(&value_dims).fold(0, |acc, (&i, &d)| acc * d + i);
                            values.push(function[start + flat]);
                            layout.push(offset + start + flat);
                        }
                    });
                    (values, Some(layout))
                } else {
                    (function[start..start + slice_len].to_vec(), None)
                };
                let kept = |i: usize| atoms[i].is_ok();
                factors.push(GroundFactor {
                    predicates,
                    kinds: kinds.iter().enumerate().filter(|&(i, _)| kept(i)).map(|(_, k)| *k).collect(),
                    values,
                    offset: offset + start,
                    learnable: c.learnable,
                    normal: c.normal.filter(|&n| kept(n)).map(|n| (0..n).filter(|&i| kept(i)).count()),
                    // A padded factor is evaluated through its table (its analytic form spans
                    // every pattern).
                    analytic: (!padded && !matches!(c.function, Function::Table(_))).then(|| {
                        Arc::new(Analytic {
                            // Folded once here, so exact and particle inference see plain forms.
                            function: c.function.resolved().unwrap_or_else(|_| c.function.clone()),
                            prefix: fv.iter().map(|&i| a[i] as f32).collect(),
                        })
                    }),
                    layout,
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
                    analytic: None,
                    layout: None,
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
