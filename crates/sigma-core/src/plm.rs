//! Piecewise-linear functions (PLMs), as in Sigma's hybrid mixed models.
//!
//! A [`Plm`] is a function over `n` dimensions. Each dimension is cut by *slices* only where
//! the function changes, and the slices of all dimensions split the space into
//! hyper-rectangular regions, each holding a linear function `c0 + Σ ci·xi`. A discrete
//! dimension of size `d` has one unit slice `[i, i+1)` per value, with no slope along it.
//!
//! Sum-product messages are computed exactly on this representation: multiplying a factor by
//! an incoming one-dimensional message and integrating that dimension out keeps the result
//! linear in the remaining dimensions, so a factor-to-variable message is the exact integral
//! (or, with [`Plm::max_out`], the exact maximum per slice). Only the product of two sloped
//! one-dimensional messages, which is quadratic, is approximated: the region is split until
//! the quadratic part is small, and each piece is replaced by the linear function with the
//! same integral and the same gradient at its centre.

use serde::{Deserialize, Serialize};

/// Most slices along one dimension of a message.
pub const MAX_SLICES: usize = 1024;
/// Largest number of regions in one piecewise-linear function.
pub const MAX_REGIONS: usize = 1 << 18;
/// Relative size of the quadratic part of a product below which a region is not split.
const SPLIT_TOLERANCE: f64 = 1e-3;
const EPS: f64 = 1e-12;

/// An `n`-dimensional piecewise-linear function over a grid of slices.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plm {
    /// Per dimension, the sorted slice boundaries `b0 < b1 < … < bk` (k slices).
    pub bounds: Vec<Vec<f64>>,
    /// Row-major over slice indices: each region's coefficients `[c0, c1, …, cn]`.
    pub coef: Vec<f64>,
}

impl Plm {
    /// The constant function `value` over the given slices.
    pub fn constant(bounds: Vec<Vec<f64>>, value: f64) -> Self {
        let n = bounds.len();
        let regions: usize = bounds.iter().map(|b| b.len() - 1).product();
        let mut coef = vec![0.0; regions * (n + 1)];
        coef.iter_mut().step_by(n + 1).for_each(|c| *c = value);
        Plm { bounds, coef }
    }

    /// A one-dimensional function, constant `values[i]` on slice `[bounds[i], bounds[i+1])`.
    pub fn steps(bounds: Vec<f64>, values: &[f64]) -> Self {
        let coef = values.iter().flat_map(|&v| [v, 0.0]).collect();
        Plm { bounds: vec![bounds], coef }
    }

    /// A one-dimensional function with one linear piece `c0 + c1·x` per slice.
    pub fn linear_pieces(bounds: Vec<f64>, pieces: &[(f64, f64)]) -> Self {
        Plm { bounds: vec![bounds], coef: pieces.iter().flat_map(|&(a, b)| [a, b]).collect() }
    }

    pub fn dims(&self) -> usize {
        self.bounds.len()
    }

    fn slices(&self, d: usize) -> usize {
        self.bounds[d].len() - 1
    }

    fn shape(&self) -> Vec<usize> {
        (0..self.dims()).map(|d| self.slices(d)).collect()
    }

    pub fn regions(&self) -> usize {
        self.coef.len() / (self.dims() + 1)
    }

    fn region(&self, r: usize) -> &[f64] {
        let n = self.dims() + 1;
        &self.coef[r * n..(r + 1) * n]
    }

    /// The slice of dimension `d` that contains `x`, if any.
    pub fn slice_of(&self, d: usize, x: f64) -> Option<usize> {
        let b = &self.bounds[d];
        if !(x >= b[0] && x < b[b.len() - 1]) {
            return None;
        }
        Some(b.partition_point(|&e| e <= x) - 1)
    }

    /// The function's value at `x` (0 outside its domain).
    pub fn eval(&self, x: &[f64]) -> f64 {
        let mut r = 0;
        for (d, &xd) in x.iter().enumerate() {
            match self.slice_of(d, xd) {
                Some(k) => r = r * self.slices(d) + k,
                None => return 0.0,
            }
        }
        let c = self.region(r);
        c[0] + c[1..].iter().zip(x).map(|(a, b)| a * b).sum::<f64>()
    }

    /// Fixes dimension `d` at `x`, leaving a function of the other dimensions (zero when `x`
    /// lies outside the dimension's domain).
    pub fn substitute(&self, d: usize, x: f64) -> Plm {
        let n = self.dims();
        let mut bounds = self.bounds.clone();
        bounds.remove(d);
        let mut out = Plm::constant(bounds, 0.0);
        let Some(k) = self.slice_of(d, x) else { return out };
        let shape = self.shape();
        for_each_index(&out.shape(), |r_out, idx| {
            let mut full = idx.to_vec();
            full.insert(d, k);
            let c = self.region(flat(&full, &shape));
            let o = &mut out.coef[r_out * n..(r_out + 1) * n];
            o[0] = c[0] + c[d + 1] * x;
            let mut j = 1;
            for (i, &ci) in c[1..].iter().enumerate() {
                if i != d {
                    o[j] = ci;
                    j += 1;
                }
            }
        });
        out
    }

    /// Multiplies by `m(x_d)` (a one-dimensional function) and integrates dimension `d` out.
    /// Exact: the result is linear in the remaining dimensions on every region.
    pub fn mul_sum_out(&self, d: usize, m: &Plm) -> Plm {
        assert_eq!(m.dims(), 1, "messages are one-dimensional");
        // Per slice k of dimension d: ∫ m and ∫ x·m over the slice.
        let moments: Vec<(f64, f64)> = self.bounds[d].windows(2).map(|w| m.moments(w[0], w[1])).collect();
        self.reduce(d, |c, k| {
            let (m0, m1) = moments[k];
            (m0, c * m1)
        })
    }

    /// Integrates dimension `d` out (times a constant 1).
    pub fn sum_out(&self, d: usize) -> Plm {
        let moments: Vec<(f64, f64)> =
            self.bounds[d].windows(2).map(|w| (w[1] - w[0], (w[1] * w[1] - w[0] * w[0]) / 2.0)).collect();
        self.reduce(d, |c, k| (moments[k].0, c * moments[k].1))
    }

    /// Removes dimension `d`, summing over its slices. `weigh(c_d, k)` gives, for slice `k`,
    /// the factor multiplying the other coefficients and the extra constant from `c_d·x_d`.
    fn reduce(&self, d: usize, weigh: impl Fn(f64, usize) -> (f64, f64)) -> Plm {
        let n = self.dims();
        let mut bounds = self.bounds.clone();
        bounds.remove(d);
        let mut out = Plm::constant(bounds, 0.0);
        let shape = self.shape();
        for_each_index(&out.shape(), |r_out, idx| {
            let mut full = idx.to_vec();
            full.insert(d, 0);
            for k in 0..shape[d] {
                full[d] = k;
                let c = self.region(flat(&full, &shape));
                let (scale, extra) = weigh(c[d + 1], k);
                let o = &mut out.coef[r_out * n..(r_out + 1) * n];
                o[0] += c[0] * scale + extra;
                let mut j = 1;
                for (i, &ci) in c[1..].iter().enumerate() {
                    if i != d {
                        o[j] += ci * scale;
                        j += 1;
                    }
                }
            }
        });
        out
    }

    /// Multiplies by `m(x_d)` and maximizes dimension `d` out (max-product).
    ///
    /// Within a slice the product of a piece `c0 + c_d·x + c_y·y` with the message
    /// `α + β·x` is quadratic in `x`. Where `c_d·β ≥ 0` its maximum over the slice is at an end,
    /// so the slice gives two candidate linear functions of the remaining dimensions; where
    /// `c_d·β < 0` it may peak inside the slice, at a point that moves with `y`.
    ///
    /// When one dimension remains (every message, and every two-slot factor):
    /// * a region whose slices all peak at their ends gets the exact upper envelope of its
    ///   candidates: it is split where candidates cross, and each part keeps the candidate that
    ///   is largest there;
    /// * a region with an interior peak gets the exact maximum (ends and clipped interior
    ///   stationary point of every slice) evaluated at knots placed by bisection until the
    ///   chord between knots is within [`MAX_PRODUCT_TOLERANCE`] (relative) of it. The maximum
    ///   of quadratics over `x` is convex in `y`, so the chords lie on or just above it.
    ///
    /// With two or more dimensions left the maximum is a convex function of them. Each cell
    /// gets the plane tangent to it at the cell's centre, and cells are cut in half until the
    /// plane matches the maximum at every corner within [`MAX_PRODUCT_TOLERANCE`] or the
    /// budget ([`MAX_REFINED_REGIONS`] cells, [`MAX_REFINE_ROUNDS`] rounds) runs out. The
    /// result never exceeds the true maximum; it is exact in cells that match, which includes
    /// every kink that follows an axis. A kink across the axes cannot be held by axis-aligned
    /// cells, so near one the result is off by about the slopes' difference times a cell's size.
    pub fn mul_max_out(&self, d: usize, m: &Plm) -> Plm {
        if self.dims() == 2 {
            if let Some(out) = self.mul_max_out_quadratic(d, m) {
                return out;
            }
        }
        let p = self.refine(d, &merge_bounds(&self.bounds[d], &m.bounds[0]));
        if p.dims() >= 3 {
            return p.mul_max_out_nd(d, m);
        }
        let n = p.dims();
        let shape = p.shape();
        let mut bounds = p.bounds.clone();
        bounds.remove(d);
        let base = Plm::constant(bounds.clone(), 0.0);
        // Candidate linear functions (`[c0, c1, …]` over the remaining dimensions) per region.
        let mut candidates: Vec<Vec<Vec<f64>>> = vec![Vec::new(); base.regions()];
        for_each_index(&base.shape(), |r_out, idx| {
            let mut full = idx.to_vec();
            full.insert(d, 0);
            for k in 0..shape[d] {
                full[d] = k;
                let c = p.region(flat(&full, &shape));
                let (a, b) = (p.bounds[d][k], p.bounds[d][k + 1]);
                for x in [a, b - (b - a) * 1e-9] {
                    let mx = m.eval(&[x]);
                    let mut lin: Vec<f64> = Vec::with_capacity(n);
                    lin.push((c[0] + c[d + 1] * x) * mx);
                    lin.extend(c[1..].iter().enumerate().filter(|&(i, _)| i != d).map(|(_, &ci)| ci * mx));
                    candidates[r_out].push(lin);
                }
            }
        });
        let value = |lin: &[f64], at: &[f64]| lin[0] + lin[1..].iter().zip(at).map(|(a, b)| a * b).sum::<f64>();
        let best_at = |cands: &[Vec<f64>], at: &[f64]| -> Option<Vec<f64>> {
            cands.iter().max_by(|a, b| value(a, at).total_cmp(&value(b, at))).cloned()
        };
        if bounds.len() == 1 {
            // One dimension left: split each region where two candidates cross.
            let old = &bounds[0];
            let mut cuts: Vec<f64> = old.clone();
            for (r, cands) in candidates.iter().enumerate() {
                let (lo, hi) = (old[r], old[r + 1]);
                for (i, a) in cands.iter().enumerate() {
                    for b in &cands[i + 1..] {
                        let slope = a[1] - b[1];
                        if slope.abs() > 1e-12 {
                            let y = (b[0] - a[0]) / slope;
                            if y > lo + 1e-12 && y < hi - 1e-12 {
                                cuts.push(y);
                            }
                        }
                    }
                }
            }
            cuts.sort_by(f64::total_cmp);
            cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
            let mut out = Plm::constant(vec![cuts.clone()], 0.0);
            for k in 0..cuts.len() - 1 {
                let mid = (cuts[k] + cuts[k + 1]) / 2.0;
                let r = old.partition_point(|&b| b <= mid).saturating_sub(1).min(candidates.len() - 1);
                if let Some(lin) = best_at(&candidates[r], &[mid]) {
                    out.coef[k * n..(k + 1) * n].copy_from_slice(&lin);
                }
            }
            return out;
        }
        // No dimension left: the largest candidate value.
        let mut out = base;
        if let Some(lin) = candidates.first().and_then(|c| best_at(c, &[])) {
            out.coef[..n].copy_from_slice(&lin);
        }
        out
    }

    /// [`Plm::mul_max_out`] with two or more dimensions `y` left, on a function already cut
    /// along `d` wherever the message changes.
    ///
    /// For fixed `x` the product is linear in `y`, so `f(y) = max_x` of it is convex. In each
    /// cell the result is the plane tangent to `f` at the cell's centre (the product at the
    /// maximizing `x` there). A tangent plane never exceeds `f`, and `f` minus it is convex, so
    /// its largest gap is at a corner: a cell whose corners all match within
    /// [`MAX_PRODUCT_TOLERANCE`] (relative to the largest value) is within that tolerance
    /// everywhere. Cells that miss are cut in half along the dimension with the largest gap,
    /// for up to [`MAX_REFINE_ROUNDS`] rounds and [`MAX_REFINED_REGIONS`] cells. Where every
    /// peak lies at a slice end `f` is a maximum of planes, and a matching cell is exact.
    /// Beyond [`MAX_CORNER_DIMS`] dimensions left, cells are not checked or cut.
    fn mul_max_out_nd(&self, d: usize, m: &Plm) -> Plm {
        let n = self.dims();
        let shape = self.shape();
        let others: Vec<usize> = (0..n).filter(|&i| i != d).collect();
        let orig: Vec<Vec<f64>> = others.iter().map(|&i| self.bounds[i].clone()).collect();
        // Per slice of x: (a, b, α, β), the message being α + β·x there.
        let slices: Vec<(f64, f64, f64, f64)> = (0..shape[d])
            .map(|k| {
                let (a, b) = (self.bounds[d][k], self.bounds[d][k + 1]);
                let (xa, xb) = (a + (b - a) * 0.25, a + (b - a) * 0.75);
                let (ma, mb) = (m.eval(&[xa]), m.eval(&[xb]));
                let beta = (mb - ma) / (xb - xa);
                (a, b, ma - beta * xa, beta)
            })
            .collect();
        let coef_at = |idx: &[usize], k: usize| {
            let mut full = idx.to_vec();
            full.insert(d, k);
            self.region(flat(&full, &shape))
        };
        // The exact maximum over x at y in the region `idx` of y, and where it is.
        let best = |idx: &[usize], y: &[f64]| -> (f64, usize, f64) {
            let mut top = (f64::NEG_INFINITY, 0, 0.0);
            for (k, &(a, b, alpha, beta)) in slices.iter().enumerate() {
                let c = coef_at(idx, k);
                let cx = c[d + 1];
                let base = c[0] + others.iter().zip(y).map(|(&i, &yi)| c[i + 1] * yi).sum::<f64>();
                let g = |x: f64| (base + cx * x) * (alpha + beta * x);
                let b_in = b - (b - a) * 1e-9;
                let curvature = cx * beta;
                let peak = if curvature < -1e-12 {
                    (-(cx * alpha + beta * base) / (2.0 * curvature)).clamp(a, b_in)
                } else {
                    a
                };
                for x in [a, b_in, peak] {
                    let v = g(x);
                    if v > top.0 {
                        top = (v, k, x);
                    }
                }
            }
            top
        };
        // The plane tangent to the maximum at y0: the product at the maximizing x.
        let tangent = |idx: &[usize], y0: &[f64]| -> Vec<f64> {
            let (_, k, x) = best(idx, y0);
            let (_, _, alpha, beta) = slices[k];
            let c = coef_at(idx, k);
            let mx = alpha + beta * x;
            let mut lin = Vec::with_capacity(n);
            lin.push((c[0] + c[d + 1] * x) * mx);
            lin.extend(others.iter().map(|&i| c[i + 1] * mx));
            lin
        };
        let value = |lin: &[f64], y: &[f64]| lin[0] + lin[1..].iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
        let rest = others.len();
        let mut cuts = orig.clone();
        for round in 0..=MAX_REFINE_ROUNDS {
            let mut out = Plm::constant(cuts.clone(), 0.0);
            let check = rest <= MAX_CORNER_DIMS && round < MAX_REFINE_ROUNDS;
            let mut scale = 0.0f64;
            // Cells that miss: (gap, dimension to cut, where).
            let mut misses: Vec<(f64, usize, f64)> = Vec::new();
            for_each_index(&out.shape(), |r, cell| {
                let lo: Vec<f64> = cell.iter().enumerate().map(|(j, &k)| cuts[j][k]).collect();
                let hi: Vec<f64> = cell.iter().enumerate().map(|(j, &k)| cuts[j][k + 1]).collect();
                let centre: Vec<f64> = lo.iter().zip(&hi).map(|(a, b)| 0.5 * (a + b)).collect();
                let idx: Vec<usize> = centre
                    .iter()
                    .enumerate()
                    .map(|(j, &y)| orig[j].partition_point(|&e| e <= y).saturating_sub(1).min(orig[j].len() - 2))
                    .collect();
                let lin = tangent(&idx, &centre);
                out.coef[r * n..(r + 1) * n].copy_from_slice(&lin);
                if !check {
                    return;
                }
                let mut gap = 0.0f64;
                for corner in 0..1usize << rest {
                    let y: Vec<f64> = (0..rest).map(|j| if corner >> j & 1 == 1 { hi[j] } else { lo[j] }).collect();
                    let f = best(&idx, &y).0;
                    scale = scale.max(f.abs());
                    gap = gap.max(f - value(&lin, &y));
                }
                if gap <= 0.0 {
                    return;
                }
                // Cut where the gap along an axis is largest (widest cell side on a tie).
                let (mut dim, mut along) = (0, f64::NEG_INFINITY);
                for j in 0..rest {
                    let mut y = centre.clone();
                    let mut g = 0.0f64;
                    for end in [lo[j], hi[j]] {
                        y[j] = end;
                        g = g.max(best(&idx, &y).0 - value(&lin, &y));
                    }
                    let span = orig[j][orig[j].len() - 1] - orig[j][0];
                    let key = g + 1e-15 * (hi[j] - lo[j]) / span.max(EPS);
                    if key > along {
                        (dim, along) = (j, key);
                    }
                }
                misses.push((gap, dim, centre[dim]));
            });
            let tolerance = MAX_PRODUCT_TOLERANCE * scale.max(1e-300);
            misses.retain(|q| q.0 > tolerance);
            if misses.is_empty() || !check {
                return out;
            }
            // Add the worst cells' cuts first while the grid stays within the budget.
            misses.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut added = false;
            for (_, j, at) in misses {
                if cuts[j].iter().any(|&e| (e - at).abs() < 1e-12) {
                    continue;
                }
                let size: usize = cuts.iter().enumerate().map(|(i, c)| c.len() - 1 + usize::from(i == j)).product();
                if size > MAX_REFINED_REGIONS {
                    continue;
                }
                let at_pos = cuts[j].partition_point(|&e| e < at);
                cuts[j].insert(at_pos, at);
                added = true;
            }
            if !added {
                return out;
            }
        }
        unreachable!("the last round returns")
    }

    /// [`Plm::mul_max_out`] for two dimensions when some slice's product with the message peaks
    /// inside the slice. `None` when none can (the end-point envelope is then exact).
    fn mul_max_out_quadratic(&self, d: usize, m: &Plm) -> Option<Plm> {
        let p = self.refine(d, &merge_bounds(&self.bounds[d], &m.bounds[0]));
        let y_dim = 1 - d;
        let shape = p.shape();
        // Per slice of x: (a, b, α, β), the message being α + β·x there.
        let slices: Vec<(f64, f64, f64, f64)> = (0..shape[d])
            .map(|k| {
                let (a, b) = (p.bounds[d][k], p.bounds[d][k + 1]);
                let (xa, xb) = (a + (b - a) * 0.25, a + (b - a) * 0.75);
                let (ma, mb) = (m.eval(&[xa]), m.eval(&[xb]));
                let beta = (mb - ma) / (xb - xa);
                (a, b, ma - beta * xa, beta)
            })
            .collect();
        // The piece of region (x slice k, y slice r): (c0, c_x, c_y) in absolute coordinates.
        let piece = |k: usize, r: usize| {
            let idx = if d == 0 { [k, r] } else { [r, k] };
            let c = p.region(flat(&idx, &shape));
            (c[0], c[d + 1], c[y_dim + 1])
        };
        let interior = (0..shape[d]).any(|k| {
            (0..shape[y_dim]).any(|r| {
                let (_, cx, _) = piece(k, r);
                cx * slices[k].3 < -1e-12
            })
        });
        if !interior {
            return None;
        }
        // The exact maximum over x at y, in y slice r.
        let best = |r: usize, y: f64| -> f64 {
            let mut top = f64::NEG_INFINITY;
            for (k, &(a, b, alpha, beta)) in slices.iter().enumerate() {
                let (c0, cx, cy) = piece(k, r);
                let base = c0 + cy * y;
                let g = |x: f64| (base + cx * x) * (alpha + beta * x);
                let b_in = b - (b - a) * 1e-9;
                top = top.max(g(a)).max(g(b_in));
                let curvature = cx * beta;
                if curvature < -1e-12 {
                    let x = (-(cx * alpha + beta * base) / (2.0 * curvature)).clamp(a, b_in);
                    top = top.max(g(x));
                }
            }
            top
        };
        let ys = &p.bounds[y_dim];
        // Linear pieces `(start, c0, c1)` of the result, region by region.
        let mut pieces: Vec<(f64, f64, f64)> = Vec::new();
        for r in 0..ys.len() - 1 {
            let (lo, hi) = (ys[r], ys[r + 1] - (ys[r + 1] - ys[r]) * 1e-9);
            let mut stack = vec![(lo, best(r, lo), hi, best(r, hi), 0usize)];
            while let Some((a, fa, b, fb, depth)) = stack.pop() {
                let mid = 0.5 * (a + b);
                let fm = best(r, mid);
                let chord = 0.5 * (fa + fb);
                let scale = fa.abs().max(fb.abs()).max(fm.abs()).max(1e-300);
                if depth >= MAX_BISECTIONS || (chord - fm).abs() <= MAX_PRODUCT_TOLERANCE * scale {
                    let slope = (fb - fa) / (b - a);
                    pieces.push((a, fa - slope * a, slope));
                    continue;
                }
                // Right half first on the stack, so pieces come out in order.
                stack.push((mid, fm, b, fb, depth + 1));
                stack.push((a, fa, mid, fm, depth + 1));
            }
        }
        let mut cuts: Vec<f64> = pieces.iter().map(|q| q.0).collect();
        cuts.push(ys[ys.len() - 1]);
        let mut out = Plm::constant(vec![cuts], 0.0);
        for (k, &(_, c0, c1)) in pieces.iter().enumerate() {
            out.coef[k * 2] = c0;
            out.coef[k * 2 + 1] = c1;
        }
        Some(out)
    }

    /// The same function over the slices `bounds` of dimension `d`, each inside one old slice
    /// (zero where it lies outside the old domain).
    fn refine(&self, d: usize, bounds: &[f64]) -> Plm {
        let n = self.dims();
        let shape = self.shape();
        let mut nb = self.bounds.clone();
        nb[d] = bounds.to_vec();
        let mut out = Plm::constant(nb, 0.0);
        let os = out.shape();
        let map: Vec<usize> =
            bounds.windows(2).map(|w| self.slice_of(d, (w[0] + w[1]) / 2.0).unwrap_or(usize::MAX)).collect();
        for_each_index(&os, |r, idx| {
            let mut src = idx.to_vec();
            src[d] = map[idx[d]];
            if src[d] != usize::MAX {
                let c = self.region(flat(&src, &shape)).to_vec();
                out.coef[r * (n + 1)..(r + 1) * (n + 1)].copy_from_slice(&c);
            }
        });
        out
    }

    /// `∫ f` and `∫ x·f` over `[a, b)` of a one-dimensional function.
    pub fn moments(&self, a: f64, b: f64) -> (f64, f64) {
        let bd = &self.bounds[0];
        let (mut m0, mut m1) = (0.0, 0.0);
        for (k, w) in bd.windows(2).enumerate() {
            let (p, q) = (w[0].max(a), w[1].min(b));
            if q <= p {
                continue;
            }
            let (c0, c1) = (self.coef[2 * k], self.coef[2 * k + 1]);
            let (q2, p2, q3, p3) = (q * q, p * p, q * q * q, p * p * p);
            m0 += c0 * (q - p) + c1 * (q2 - p2) / 2.0;
            m1 += c0 * (q2 - p2) / 2.0 + c1 * (q3 - p3) / 3.0;
        }
        (m0, m1)
    }

    /// The integral over the whole domain.
    pub fn integral(&self) -> f64 {
        let mut p = self.clone();
        while p.dims() > 0 {
            p = p.sum_out(p.dims() - 1);
        }
        p.coef[0]
    }

    /// Scales the function to integrate to 1 (uniform when it has no mass).
    pub fn normalized(mut self) -> Plm {
        let z = self.integral();
        if z > EPS && z.is_finite() {
            self.coef.iter_mut().for_each(|c| *c /= z);
            self
        } else {
            let vol: f64 = self.bounds.iter().map(|b| b[b.len() - 1] - b[0]).product();
            Plm::constant(self.bounds, 1.0 / vol.max(EPS))
        }
    }

    /// The product of two one-dimensional functions over the same domain. Where both pieces
    /// slope, the region is split until the quadratic part is small, and each piece becomes the
    /// linear function with the product's integral and its gradient at the centre.
    pub fn mul_1d(&self, other: &Plm) -> Plm {
        let bounds = merge_bounds(&self.bounds[0], &other.bounds[0]);
        let mut out_b = vec![bounds[0]];
        let mut pieces = Vec::new();
        for w in bounds.windows(2) {
            let mid = (w[0] + w[1]) / 2.0;
            let piece = |p: &Plm| match p.slice_of(0, mid) {
                Some(k) => (p.coef[2 * k], p.coef[2 * k + 1]),
                None => (0.0, 0.0),
            };
            let (a0, a1) = piece(self);
            let (b0, b1) = piece(other);
            // Split so that the quadratic term a1·b1·w²/4 stays small against the product.
            let scale = ((a0 + a1 * mid) * (b0 + b1 * mid)).abs().max(EPS);
            let quad = (a1 * b1).abs() * (w[1] - w[0]).powi(2) / 4.0;
            let parts = if quad <= SPLIT_TOLERANCE * scale {
                1
            } else {
                ((quad / (SPLIT_TOLERANCE * scale)).sqrt().ceil() as usize).clamp(1, 16)
            };
            let step = (w[1] - w[0]) / parts as f64;
            for i in 0..parts {
                let (p, q) = (w[0] + i as f64 * step, if i + 1 == parts { w[1] } else { w[0] + (i + 1) as f64 * step });
                let c = (p + q) / 2.0;
                let (fa, fb) = (a0 + a1 * c, b0 + b1 * c);
                // ∫ over [p,q) of (fa + a1 t)(fb + b1 t) dt / (q-p) = fa·fb + a1·b1·(q-p)²/12.
                let mean = fa * fb + a1 * b1 * (q - p).powi(2) / 12.0;
                let slope = a1 * fb + b1 * fa;
                pieces.push((mean - slope * c, slope));
                out_b.push(q);
            }
        }
        Plm::linear_pieces(out_b, &pieces).nonnegative().simplified().capped()
    }

    /// Replaces any region that dips below zero by its (non-negative) mean.
    fn nonnegative(mut self) -> Plm {
        if self.dims() != 1 {
            return self;
        }
        for k in 0..self.slices(0) {
            let (a, b) = (self.bounds[0][k], self.bounds[0][k + 1]);
            let (c0, c1) = (self.coef[2 * k], self.coef[2 * k + 1]);
            if (c0 + c1 * a).min(c0 + c1 * b) < 0.0 {
                self.coef[2 * k] = (c0 + c1 * (a + b) / 2.0).max(0.0);
                self.coef[2 * k + 1] = 0.0;
            }
        }
        self
    }

    /// Merges adjacent slices whose regions hold the same linear function, in every dimension:
    /// slices remain only where the function changes.
    pub fn simplified(mut self) -> Plm {
        let n = self.dims();
        for d in 0..n {
            let shape = self.shape();
            // Flat positions of slice 0 along d; slice k adds k·stride.
            let stride: usize = shape[d + 1..].iter().product();
            let starts: Vec<usize> = (0..self.regions()).filter(|&r| unflat(r, &shape)[d] == 0).collect();
            let same = |a: usize, b: usize| {
                starts.iter().all(|&r0| {
                    let (x, y) = (self.region(r0 + a * stride), self.region(r0 + b * stride));
                    x.iter().zip(y).all(|(x, y)| (x - y).abs() <= 1e-9 * (1.0 + x.abs().max(y.abs())))
                })
            };
            let mut keep = vec![true; shape[d] + 1];
            let mut prev = 0;
            for (k, kept) in keep.iter_mut().enumerate().take(shape[d]).skip(1) {
                if same(prev, k) {
                    *kept = false;
                } else {
                    prev = k;
                }
            }
            if keep.iter().all(|&k| k) {
                continue;
            }
            let bounds: Vec<f64> = self.bounds[d].iter().zip(&keep).filter(|(_, &k)| k).map(|(b, _)| *b).collect();
            let map: Vec<usize> =
                (0..bounds.len() - 1).map(|i| self.slice_of(d, bounds[i]).expect("kept boundary is inside")).collect();
            let mut nb = self.bounds.clone();
            nb[d] = bounds;
            let mut out = Plm::constant(nb, 0.0);
            let os = out.shape();
            for_each_index(&os, |r, idx| {
                let mut src = idx.to_vec();
                src[d] = map[idx[d]];
                let c = self.region(flat(&src, &shape)).to_vec();
                out.coef[r * (n + 1)..(r + 1) * (n + 1)].copy_from_slice(&c);
            });
            self = out;
        }
        self
    }

    /// Keeps a one-dimensional message within [`MAX_SLICES`] by merging the narrowest
    /// neighbouring slices (each merge keeps the pair's integral and first moment).
    fn capped(mut self) -> Plm {
        while self.dims() == 1 && self.slices(0) > MAX_SLICES {
            let b = &self.bounds[0];
            let k = (0..b.len() - 2)
                .min_by(|&i, &j| (b[i + 2] - b[i]).total_cmp(&(b[j + 2] - b[j])))
                .expect("more than one slice");
            let (a, c) = (b[k], b[k + 2]);
            let (m0, m1) = self.moments(a, c);
            // The linear function on [a, c) with these two moments.
            let (w, mid) = (c - a, (a + c) / 2.0);
            let slope = 12.0 * (m1 - mid * m0) / w.powi(3);
            let mean = m0 / w;
            self.bounds[0].remove(k + 1);
            self.coef.splice(2 * k..2 * k + 4, [mean - slope * mid, slope]);
        }
        self
    }

    /// Mass on each of the given one-dimensional slices.
    pub fn masses(&self, bounds: &[f64]) -> Vec<f64> {
        bounds.windows(2).map(|w| self.moments(w[0], w[1]).0.max(0.0)).collect()
    }

    /// The mean of a one-dimensional density.
    pub fn mean(&self) -> f64 {
        let b = &self.bounds[0];
        let (m0, m1) = self.moments(b[0], b[b.len() - 1]);
        if m0 > EPS {
            m1 / m0
        } else {
            (b[0] + b[b.len() - 1]) / 2.0
        }
    }

    /// The mean and variance of a one-dimensional density (exact for its linear pieces).
    pub fn mean_variance(&self) -> (f64, f64) {
        let b = &self.bounds[0];
        let (mut m0, mut m1, mut m2) = (0.0, 0.0, 0.0);
        for (k, w) in b.windows(2).enumerate() {
            let (p, q) = (w[0], w[1]);
            let (c0, c1) = (self.coef[2 * k], self.coef[2 * k + 1]);
            let pw = |e: i32| q.powi(e) - p.powi(e);
            m0 += c0 * pw(1) + c1 * pw(2) / 2.0;
            m1 += c0 * pw(2) / 2.0 + c1 * pw(3) / 3.0;
            m2 += c0 * pw(3) / 3.0 + c1 * pw(4) / 4.0;
        }
        if m0 <= EPS {
            let mid = (b[0] + b[b.len() - 1]) / 2.0;
            return (mid, (b[b.len() - 1] - b[0]).powi(2) / 12.0);
        }
        let mean = m1 / m0;
        (mean, (m2 / m0 - mean * mean).max(0.0))
    }

    /// `∫ |f - g|` of two one-dimensional functions.
    pub fn l1_distance(&self, other: &Plm) -> f64 {
        let bounds = merge_bounds(&self.bounds[0], &other.bounds[0]);
        bounds
            .windows(2)
            .map(|w| {
                let (fa, fb) = (self.eval(&[w[0]]) - other.eval(&[w[0]]), {
                    let x = w[1] - (w[1] - w[0]) * 1e-9;
                    self.eval(&[x]) - other.eval(&[x])
                });
                let len = w[1] - w[0];
                if fa * fb >= 0.0 {
                    (fa.abs() + fb.abs()) / 2.0 * len
                } else {
                    (fa * fa + fb * fb) / (2.0 * (fa.abs() + fb.abs())) * len
                }
            })
            .sum()
    }

    /// Maps dimension `d` through `x' = coef·x + offset`: the result `g` satisfies
    /// `g(…, coef·x + offset, …) = f(…, x, …)`. Exact (slices and slopes are transformed).
    pub fn affine(&self, d: usize, coef: f64, offset: f64) -> Plm {
        assert!(coef != 0.0, "affine coefficient must be non-zero");
        let n = self.dims();
        let shape = self.shape();
        let mut nb = self.bounds.clone();
        let mapped: Vec<f64> = self.bounds[d].iter().map(|b| coef * b + offset).collect();
        let reversed = coef < 0.0;
        nb[d] = if reversed { mapped.iter().rev().copied().collect() } else { mapped };
        let mut out = Plm::constant(nb, 0.0);
        for_each_index(&shape, |r, idx| {
            let mut dst = idx.to_vec();
            if reversed {
                dst[d] = shape[d] - 1 - idx[d];
            }
            let c = self.region(r);
            // f(x) = c0 + cd·x + …, with x = (x' - offset)/coef.
            let o = &mut out.coef[flat(&dst, &shape) * (n + 1)..(flat(&dst, &shape) + 1) * (n + 1)];
            o.copy_from_slice(c);
            o[d + 1] = c[d + 1] / coef;
            o[0] = c[0] - c[d + 1] * offset / coef;
        });
        out
    }

    /// Restricts dimension `d` to the domain `[lo, hi)` (zero outside, slices cut at the ends).
    pub fn clip(&self, d: usize, lo: f64, hi: f64) -> Plm {
        let mut b: Vec<f64> = self.bounds[d].iter().copied().filter(|&x| x > lo && x < hi).collect();
        b.insert(0, lo);
        b.push(hi);
        self.refine(d, &b)
    }
}

/// Relative accuracy of a max-product message where a slice's product peaks inside it.
pub const MAX_PRODUCT_TOLERANCE: f64 = 1e-6;
/// Deepest bisection when placing those knots.
const MAX_BISECTIONS: usize = 24;
/// Most rounds of cutting cells in [`Plm::mul_max_out`] with two or more dimensions left.
pub const MAX_REFINE_ROUNDS: usize = 12;
/// Most cells of that result.
pub const MAX_REFINED_REGIONS: usize = 1 << 12;
/// Most dimensions left for which cells are checked at their corners.
pub const MAX_CORNER_DIMS: usize = 8;

/// The sorted union of two boundary lists (boundaries closer than `1e-12` are merged).
pub fn merge_bounds(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut all: Vec<f64> = a.iter().chain(b).copied().collect();
    all.sort_by(f64::total_cmp);
    all.dedup_by(|x, y| (*x - *y).abs() <= EPS * (1.0 + y.abs()));
    all
}

fn flat(idx: &[usize], shape: &[usize]) -> usize {
    idx.iter().zip(shape).fold(0, |acc, (&i, &d)| acc * d + i)
}

fn unflat(mut r: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];
    for d in (0..shape.len()).rev() {
        idx[d] = r % shape[d];
        r /= shape[d];
    }
    idx
}

/// Visits every index of `shape` in row-major order, with its flat position.
fn for_each_index(shape: &[usize], mut f: impl FnMut(usize, &[usize])) {
    let total: usize = shape.iter().product();
    let mut idx = vec![0; shape.len()];
    for r in 0..total {
        f(r, &idx);
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn integrals_of_linear_regions_are_exact() {
        // f(x, y) = 1 + 2x + 3y on [0,2) x [0,1), one region.
        let f = Plm { bounds: vec![vec![0.0, 2.0], vec![0.0, 1.0]], coef: vec![1.0, 2.0, 3.0] };
        assert!(close(f.integral(), 2.0 + 4.0 + 3.0));
        // ∫ f dx = 2 + 4 + 6y.
        let g = f.sum_out(0);
        assert!(close(g.eval(&[0.5]), 6.0 + 3.0));
        // Times m(x) = x on [0, 2): ∫ (1 + 2x + 3y) x dx = 2 + 16/3 + 6y.
        let m = Plm::linear_pieces(vec![0.0, 2.0], &[(0.0, 1.0)]);
        let h = f.mul_sum_out(0, &m);
        assert!(close(h.eval(&[0.0]), 2.0 + 16.0 / 3.0));
        assert!(close(h.eval(&[1.0 - 1e-12]) - h.eval(&[0.0]), 6.0 * (1.0 - 1e-12)));
        // Fixing y = 0.5: 2.5 + 2x.
        let s = f.substitute(1, 0.5);
        assert!(close(s.eval(&[1.0]), 4.5));
        assert_eq!(f.substitute(1, 7.0).integral(), 0.0, "outside the domain");
    }

    #[test]
    fn slices_appear_only_where_the_function_changes() {
        let f = Plm::steps(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 2.0, 2.0]).simplified();
        assert_eq!(f.bounds[0], vec![0.0, 2.0, 4.0]);
        assert!(close(f.integral(), 6.0));
    }

    #[test]
    fn products_keep_their_integral() {
        // x · x on [0, 1): ∫ = 1/3.
        let x = Plm::linear_pieces(vec![0.0, 1.0], &[(0.0, 1.0)]);
        let p = x.mul_1d(&x);
        assert!((p.integral() - 1.0 / 3.0).abs() < 1e-9, "{}", p.integral());
        assert!(p.bounds[0].len() > 2, "split where the product curves");
        let n = p.clone().normalized();
        assert!(close(n.integral(), 1.0));
        assert!((n.mean() - 0.75).abs() < 1e-3, "mean of 3x^2 is 3/4: {}", n.mean());
    }

    #[test]
    fn affine_maps_are_exact() {
        // f(x) = x on [0, 2); g(x') with x' = 2x + 1 lives on [1, 5) and g(2x+1) = x.
        let f = Plm::linear_pieces(vec![0.0, 2.0], &[(0.0, 1.0)]);
        let g = f.affine(0, 2.0, 1.0);
        assert!(close(g.eval(&[3.0]), 1.0));
        let r = f.affine(0, -1.0, 0.0);
        assert_eq!(r.bounds[0], vec![-2.0, 0.0]);
        assert!(close(r.eval(&[-1.5]), 1.5));
    }

    #[test]
    fn max_out_takes_the_largest_end() {
        // f(x, y) = x + y on [0,1)^2, max over x at x→1: 1 + y.
        let f = Plm { bounds: vec![vec![0.0, 1.0], vec![0.0, 1.0]], coef: vec![0.0, 1.0, 1.0] };
        let one = Plm::constant(vec![vec![0.0, 1.0]], 1.0);
        let g = f.mul_max_out(0, &one);
        assert!((g.eval(&[0.5]) - 1.5).abs() < 1e-6);
    }
}
