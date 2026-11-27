//! Appraisals inside the factor graph, as in Sigma's appraisal work.
//!
//! Each cycle's appraisal measures (desirability, surprise, control, familiarity, and whether
//! the decision is stuck) become soft evidence on internal appraisal predicates, and a
//! conditional maps them to `@Emotion`:
//!
//! * `@Desirable` (bad, neutral, good), `@Surprised`, `@InControl`, `@Stuck` and `@Familiar`
//!   hold the appraisal, as distributions rather than thresholded flags.
//! * One factor over `[@Desirable, @Surprised, @Stuck, @InControl, @Emotion]` says which
//!   emotion each combination of appraisals gives (goal outcomes first, then surprise, then
//!   coping). The emotion label is the max-product (MAP) value of `@Emotion`.
//! * The appraisal predicates stay in the graph through the next cycle, and they are the only
//!   internal predicates a skill may use, so behaviour can depend on affect through ordinary
//!   conditionals (a surprised agent investigates, a stuck one asks for help).

use crate::graph_memory::Builder;
use crate::mind::{Appraisal, Emotion};
use crate::{Inference, PatternKind, PredicateRole, PredicateSchema, SigmaEngine};

pub const DESIRABLE: &str = "@Desirable";
pub const SURPRISED: &str = "@Surprised";
pub const IN_CONTROL: &str = "@InControl";
pub const STUCK: &str = "@Stuck";
pub const FAMILIAR: &str = "@Familiar";
pub const EMOTION: &str = "@Emotion";
pub(crate) const APPRAISAL_SKILL: &str = "@appraisal";

/// The emotions in `@Emotion`'s value order.
pub const EMOTIONS: [Emotion; 6] =
    [Emotion::Neutral, Emotion::Joy, Emotion::Distress, Emotion::Surprise, Emotion::Hope, Emotion::Fear];

/// The appraisal predicates (all `Context`), with their domain sizes.
const PREDICATES: [(&str, usize); 6] =
    [(DESIRABLE, 3), (SURPRISED, 2), (IN_CONTROL, 2), (STUCK, 2), (FAMILIAR, 2), (EMOTION, 6)];

/// The schema of an appraisal predicate, the internal predicates skills may use.
pub fn appraisal_schema(name: &str) -> Option<PredicateSchema> {
    PREDICATES.iter().find(|p| p.0 == name).map(|&(n, d)| PredicateSchema {
        name: n.into(),
        domain_size: d,
        role: PredicateRole::Context,
    })
}

/// The emotion a combination of appraisals gives: an outcome that matters first, then
/// surprise, then coping (stuck, or a confident way forward).
fn emotion_of(desirable: usize, surprised: bool, stuck: bool, in_control: bool) -> Emotion {
    match () {
        _ if desirable == 2 => Emotion::Joy,
        _ if desirable == 0 => Emotion::Distress,
        _ if surprised => Emotion::Surprise,
        _ if stuck => Emotion::Fear,
        _ if in_control => Emotion::Hope,
        _ => Emotion::Neutral,
    }
}

/// Soft evidence for `@Desirable` from desirability in `[-1, 1]`: good above 0.3, bad below
/// -0.3, neutral between.
fn desirable_prior(d: f32) -> Vec<f32> {
    vec![(-d - 0.3).max(0.0), (0.3 - d.abs()).max(0.0) + 1e-9, (d - 0.3).max(0.0)]
}

/// Soft evidence for `@InControl` from control in `[0, 1]`: even odds at 0.7.
fn control_prior(c: f32) -> Vec<f32> {
    let p = if c <= 0.7 { c / 1.4 } else { 0.5 + (c - 0.7) / 0.6 };
    let p = p.clamp(0.0, 1.0);
    vec![1.0 - p, p]
}

fn binary(p: f32) -> Vec<f32> {
    let p = if p.is_finite() { p.clamp(0.0, 1.0) } else { 0.0 };
    vec![1.0 - p, p]
}

impl SigmaEngine {
    fn appraisal_vars(&mut self) -> Option<Vec<usize>> {
        PREDICATES.iter().map(|&(n, _)| self.register_predicate(&appraisal_schema(n)?).ok()).collect()
    }

    /// Puts an appraisal into the graph as soft evidence on the appraisal predicates. Control
    /// and being stuck are set only with `stuck` (after a decision); otherwise they keep the
    /// last decision's.
    pub(crate) fn set_appraisal_priors(&mut self, a: &Appraisal, stuck: Option<bool>) {
        let Some(v) = self.appraisal_vars() else { return };
        self.appraisal_priors.insert(v[0], desirable_prior(a.desirability));
        self.appraisal_priors.insert(v[1], binary(a.surprise));
        self.appraisal_priors.insert(v[4], binary(a.familiarity));
        if let Some(stuck) = stuck {
            self.appraisal_priors.insert(v[2], control_prior(a.control));
            self.appraisal_priors.insert(v[3], binary(f32::from(u8::from(stuck))));
        }
    }

    /// Appraises the cycle in the graph: the measures become soft evidence on the appraisal
    /// predicates (kept for the next cycle), and the emotion is the MAP value of `@Emotion`.
    pub(crate) fn appraise(&mut self, mut a: Appraisal, stuck: bool) -> Appraisal {
        self.set_appraisal_priors(&a, Some(stuck));
        let Some(vars) = self.appraisal_vars() else { return a };
        // The emotion: max-product over the appraisal factor alone (a tree, so exact).
        let mut local = SigmaEngine::new();
        let local_vars: Vec<usize> = PREDICATES
            .iter()
            .map(|&(n, _)| local.register_predicate(&appraisal_schema(n).expect("known")).expect("fresh"))
            .collect();
        local.appraisal_priors = vars
            .iter()
            .zip(&local_vars)
            .filter_map(|(v, &l)| Some((l, self.appraisal_priors.get(v)?.clone())))
            .collect();
        let factor = local.appraisal_factor();
        local.factors.extend(factor.factors);
        local.inference = Inference::MaxProduct;
        let belief = local.infer(|_| None).swap_remove(local_vars[5]);
        let best = belief.iter().copied().fold(0.0f32, f32::max);
        a.emotion = EMOTIONS[belief.iter().position(|&x| x == best).unwrap_or(0)];
        a
    }

    /// The conditional from appraisals to `@Emotion`.
    pub(crate) fn appraisal_factor(&mut self) -> Builder {
        let mut b = Builder::new(APPRAISAL_SKILL);
        let Some(v) = self.appraisal_vars() else { return b };
        let mut values = Vec::with_capacity(3 * 2 * 2 * 2 * 6);
        for d in 0..3 {
            for s in 0..2 {
                for st in 0..2 {
                    for c in 0..2 {
                        let e = emotion_of(d, s == 1, st == 1, c == 1);
                        values.extend(EMOTIONS.iter().map(|&x| f32::from(u8::from(x == e))));
                    }
                }
            }
        }
        let kinds = vec![PatternKind::Condact; 5];
        b.add(vec![v[0], v[1], v[3], v[2], v[5]], kinds, &[3, 2, 2, 2, 6], values, Some(4));
        b
    }
}
