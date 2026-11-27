//! Maximum a posteriori (MAP) estimates by max-product message passing, as Sigma's
//! max-product mode.
//!
//! With [`Inference::MaxProduct`](crate::Inference) every message maximizes over the other
//! variables of its factor instead of summing them out, so each belief is a max-marginal: how
//! probable the best joint assignment is that gives the variable each value. On the grid this
//! is a plain max; on exact piecewise-linear messages it uses `Plm::mul_max_out`. The decision
//! cycle runs on whichever mode is set. [`SigmaEngine::map_estimate`] gives the jointly most
//! probable assignment of the unobserved variables whatever the mode, which can differ from
//! the value each variable is most likely to have on its own.

use crate::graph_memory::internal;
use crate::{Inference, SigmaEngine};

/// Max-marginals this close count as tied when decoding a MAP assignment.
const MAP_TIE: f32 = 1e-4;

/// A MAP estimate: the jointly most probable value of every visible variable, and the
/// max-marginal of that value (before any tie was broken).
#[derive(Clone, Debug, PartialEq)]
pub struct MapEstimate {
    pub assignment: Vec<(String, usize)>,
    pub confidence: Vec<f32>,
}

impl SigmaEngine {
    /// The MAP assignment given working memory, by max-product message passing. Max-marginals
    /// with a single best value give it at once; where several values tie, the first is fixed
    /// and the max-marginals of the rest are recomputed, so the decoded assignment is
    /// consistent.
    pub fn map_estimate(&self) -> MapEstimate {
        let mut engine = self.hypothetical();
        engine.inference = Inference::MaxProduct;
        let mut fixed: Vec<Option<usize>> = vec![None; engine.vars.len()];
        let mut beliefs = engine.infer(|_| None);
        let mut confidence = vec![0.0; engine.vars.len()];
        for v in 0..engine.vars.len() {
            let b = &beliefs[v];
            let best = b.iter().copied().fold(0.0f32, f32::max);
            let argmax = b.iter().position(|&x| x == best).unwrap_or(0);
            confidence[v] = best;
            fixed[v] = Some(argmax);
            if b.iter().filter(|&&x| best - x < MAP_TIE).count() > 1 {
                // A tie: fix this variable and decode the rest given it.
                let clamp = fixed.clone();
                beliefs = engine.infer(|u| clamp[u]);
            }
        }
        let visible = (0..engine.vars.len()).filter(|&v| !internal(&engine.vars[v].schema.name));
        let (assignment, confidence) =
            visible.map(|v| ((engine.vars[v].schema.name.clone(), fixed[v].unwrap_or(0)), confidence[v])).unzip();
        MapEstimate { assignment, confidence }
    }
}
