//! §5.14 alignment layer — the n-point output warp.
//!
//! This is the geometric stage between the composite and the projector: the
//! operator drags control points until rendered content lands on the physical
//! surface. It is **not scene content** — `scene.json` says what the surface
//! does, `alignment.json` says where the light physically lands — and it is
//! **engine-wide, not per leg**: it describes the install, so it applies after
//! the promote crossfade and after the masters, to the projector swapchain
//! only. Nothing here is ever copied on promote/pull or written by an
//! authoring RPC.
//!
//! The model (the load-bearing decision, see the §3.2 rationale in the plan):
//!
//! ```text
//! W(x) = H⁻¹(x) + R(x)          W: dest uv → source uv
//! ```
//!
//! - `H` is the exact unit-square→quad homography of the four corner handles
//!   (Heckbert). Keeping the base projective is what makes straight lines stay
//!   straight under keystone; fitting a scattered-data interpolator through
//!   four dragged corners would bow the edges instead.
//! - `R` is a **compactly supported** Wendland C² RBF residual over the extra
//!   handles, so a handle bends its own neighbourhood and leaves the rest of
//!   the frame (notably the corners) where the operator put it.
//!
//! Per-handle radii make the collocation matrix asymmetric, so the Wendland
//! positive-definiteness guarantee (uniform σ only) does not cover us here.
//! That is why solve failure keeps the previous coefficients and reports an
//! error instead of half-applying — swap-on-success, like every other
//! mutation in this engine.
//!
//! Nothing in this module talks to the GPU. The runtime representation is the
//! offset LUT baked from [`AlignmentSolution::bake_uniforms`] (see `gpu.rs`),
//! which is also the seam a future camera-driven auto-align uploads a dense
//! field into with no analytic model at all.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::gpu::{WarpBakeUniforms, MAX_WARP_POINTS};

pub const ALIGNMENT_VERSION: u32 = 1;

/// Uniform-array bound on extra handles. Exceeding it is a prescriptive RPC
/// error, never a silent truncation.
pub const MAX_POINTS: usize = MAX_WARP_POINTS;

/// Default support radius in dest-normalized units. Open question in the
/// plan (§7.1): too small feels like a dent, too large like the whole image
/// sliding. Per handle and adjustable, so tuning it is a slider not a
/// redesign — but note the §6 corner-drift caveat: 0.35 reaches a corner from
/// a quarter of the way in.
pub const DEFAULT_RADIUS: f32 = 0.35;

/// Dest positions of the source unit square's corners, in order
/// `(0,0) (1,0) (1,1) (0,1)`.
pub type Corners = [[f32; 2]; 4];

pub fn identity_corners() -> Corners {
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
}

/// One extra (non-corner) handle.
///
/// `anchor` is where the handle grabs the *content* (source uv) and `dest` is
/// where the operator has dragged it (dest uv). Keeping both — rather than a
/// displacement — is what lets corner drags carry the fine corrections along
/// with the content instead of leaving them pinned to the screen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpPoint {
    pub id: String,
    pub anchor: [f32; 2],
    pub dest: [f32; 2],
    pub radius: f32,
}

/// On-disk shape of `alignment.json`. Lenient: unknown fields are ignored and
/// every field has a default, so a file written by a newer build (or a camera
/// script) degrades rather than failing the boot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AlignmentDoc {
    pub version: u32,
    pub enabled: bool,
    /// `#rrggbb`. Painted on dest pixels whose source falls outside the
    /// composite. **Any non-black value floods the physical surface with
    /// light** and breaks the additive thesis — it is an alignment aid, not a
    /// show setting, which is why the UI must warn while it is non-black.
    pub background: String,
    pub corners: Corners,
    pub points: Vec<WarpPoint>,
}

impl Default for AlignmentDoc {
    fn default() -> Self {
        Self {
            version: ALIGNMENT_VERSION,
            enabled: true,
            background: "#000000".to_string(),
            corners: identity_corners(),
            points: Vec::new(),
        }
    }
}

/// §3.6 — a generated pattern substituted for the composite **in source
/// space**, so it warps with the content and reveals misalignment against
/// physical edges. Runtime-only on purpose: a grid left on the wall after a
/// restart would be a nasty surprise, so this is never persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestPattern {
    None,
    Grid,
    Border,
    Corners,
}

impl TestPattern {
    pub fn as_str(self) -> &'static str {
        match self {
            TestPattern::None => "none",
            TestPattern::Grid => "grid",
            TestPattern::Border => "border",
            TestPattern::Corners => "corners",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "none" => Some(TestPattern::None),
            "grid" => Some(TestPattern::Grid),
            "border" => Some(TestPattern::Border),
            "corners" => Some(TestPattern::Corners),
            _ => None,
        }
    }

    /// Shader-side mode id (`pattern.x` in the final-pass uniform).
    pub fn mode(self) -> f32 {
        match self {
            TestPattern::None => 0.0,
            TestPattern::Grid => 1.0,
            TestPattern::Border => 2.0,
            TestPattern::Corners => 3.0,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => TestPattern::Grid,
            2 => TestPattern::Border,
            3 => TestPattern::Corners,
            _ => TestPattern::None,
        }
    }

    fn to_u8(self) -> u8 {
        self.mode() as u8
    }
}

/// A document plus everything derived from it that the renderer needs. Cloned
/// out from behind the mutex; never held across work (§1b render-thread rule).
#[derive(Debug, Clone)]
pub struct AlignmentSolution {
    pub doc: AlignmentDoc,
    /// dest → source. The base layer of `W`.
    pub h_inv: [[f32; 3]; 3],
    /// RBF coefficients, one per `doc.points` entry, same order.
    pub weights: Vec<[f32; 2]>,
    /// Linear-light background (the file stores sRGB hex).
    pub background_linear: [f32; 3],
}

impl AlignmentSolution {
    /// Evaluate `W` on the CPU — used to anchor a newly added handle so that
    /// adding it is a no-op on the rendered image (§3.2), and by the tests.
    pub fn warp(&self, x: [f32; 2]) -> [f32; 2] {
        let mut p = apply_h(&self.h_inv, x);
        for (pt, w) in self.doc.points.iter().zip(self.weights.iter()) {
            let dx = x[0] - pt.dest[0];
            let dy = x[1] - pt.dest[1];
            let t = (dx * dx + dy * dy).sqrt() / pt.radius;
            let f = wendland(t as f64) as f32;
            if f != 0.0 {
                p[0] += w[0] * f;
                p[1] += w[1] * f;
            }
        }
        p
    }

    /// Pack the uniform the bake pass evaluates. Small and fixed-size: the
    /// per-frame cost of the warp is one texel read regardless of how much of
    /// this is populated.
    pub fn bake_uniforms(&self) -> WarpBakeUniforms {
        let mut u = WarpBakeUniforms::zeroed();
        for r in 0..3 {
            for c in 0..3 {
                u.h_inv[r][c] = self.h_inv[r][c];
            }
        }
        let n = self.doc.points.len().min(MAX_WARP_POINTS);
        u.counts[0] = n as u32;
        for i in 0..n {
            let p = &self.doc.points[i];
            u.points[i] = [p.dest[0], p.dest[1], 1.0 / p.radius, 0.0];
            u.weights[i] = [self.weights[i][0], self.weights[i][1], 0.0, 0.0];
        }
        u
    }
}

/// Partial merge accepted by `alignment.set`. Every field is optional; the
/// engine merges onto the current document and re-solves.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AlignmentPatch {
    pub enabled: Option<bool>,
    pub background: Option<String>,
    pub corners: Option<Vec<[f32; 2]>>,
    pub points: Option<Vec<PointPatch>>,
}

/// One handle in an `alignment.set` payload.
///
/// `anchor` omitted means "anchor me wherever the current field already puts
/// this dest position" — the §3.2 no-op-add property, so the UI can drop a
/// handle with nothing but a click position and the image does not twitch.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PointPatch {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub anchor: Option<[f32; 2]>,
    pub dest: [f32; 2],
    #[serde(default)]
    pub radius: Option<f32>,
}

/// Shared alignment state: the solved document plus the dirty stamps the
/// render thread and the persist debounce read.
///
/// Writers (RPC, inline) clone a snapshot out and never hold the lock across
/// work; the render thread only ever calls [`AlignmentState::bake_uniforms`]
/// and [`AlignmentState::final_pass_inputs`], both of which lock, copy, and
/// release.
pub struct AlignmentState {
    inner: Mutex<AlignmentSolution>,
    /// Set on every accepted mutation; cleared by the render thread once the
    /// LUT has been rebaked.
    render_dirty: AtomicBool,
    /// Epoch-ms of the last accepted mutation (0 = clean). Same debounce
    /// mechanism as the §5.3 session sidecar.
    persist_dirty: AtomicU64,
    /// Projector swapchain size, mirrored here so the inline RPC path can
    /// report it (and convert pixel nudges) without a render-thread hop.
    output_w: AtomicU32,
    output_h: AtomicU32,
    /// §3.6 test pattern. Runtime-only — never in the document.
    pattern: AtomicU32,
}

impl AlignmentState {
    /// Build from a document. A document that cannot be solved is rejected
    /// here rather than silently degraded — the caller (boot) falls back to
    /// the default document and logs.
    pub fn new(doc: AlignmentDoc) -> Result<Self, String> {
        let solution = solve(doc)?;
        Ok(Self {
            inner: Mutex::new(solution),
            render_dirty: AtomicBool::new(true),
            persist_dirty: AtomicU64::new(0),
            output_w: AtomicU32::new(0),
            output_h: AtomicU32::new(0),
            pattern: AtomicU32::new(0),
        })
    }

    pub fn identity() -> Self {
        Self::new(AlignmentDoc::default()).expect("identity alignment always solves")
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, AlignmentSolution> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn snapshot(&self) -> AlignmentSolution {
        self.lock().clone()
    }

    pub fn doc(&self) -> AlignmentDoc {
        self.lock().doc.clone()
    }

    pub fn bake_uniforms(&self) -> WarpBakeUniforms {
        self.lock().bake_uniforms()
    }

    /// What the final pass needs each frame: `(warp_enabled, background_rgb)`.
    pub fn final_pass_inputs(&self) -> (bool, [f32; 3]) {
        let s = self.lock();
        (s.doc.enabled, s.background_linear)
    }

    pub fn set_output(&self, width: u32, height: u32) {
        self.output_w.store(width, Ordering::Relaxed);
        self.output_h.store(height, Ordering::Relaxed);
    }

    pub fn output(&self) -> (u32, u32) {
        (
            self.output_w.load(Ordering::Relaxed),
            self.output_h.load(Ordering::Relaxed),
        )
    }

    pub fn test_pattern(&self) -> TestPattern {
        TestPattern::from_u8(self.pattern.load(Ordering::Relaxed) as u8)
    }

    pub fn set_test_pattern(&self, pattern: TestPattern) {
        self.pattern.store(pattern.to_u8() as u32, Ordering::Relaxed);
    }

    /// True once since the last rebake. The render thread consumes this at
    /// the top of a presented frame.
    pub fn take_render_dirty(&self) -> bool {
        self.render_dirty.swap(false, Ordering::Relaxed)
    }

    pub fn mark_render_dirty(&self) {
        self.render_dirty.store(true, Ordering::Relaxed);
    }

    pub fn persist_dirty_ms(&self) -> u64 {
        self.persist_dirty.load(Ordering::Relaxed)
    }

    pub fn clear_persist_dirty(&self) {
        self.persist_dirty.store(0, Ordering::Relaxed);
    }

    fn touch(&self) {
        self.render_dirty.store(true, Ordering::Relaxed);
        self.persist_dirty
            .store(crate::session::now_ms().max(1), Ordering::Relaxed);
    }

    /// Merge a patch, re-solve, and swap on success. On failure the previous
    /// solution keeps rendering and the error is returned verbatim for the
    /// RPC reply.
    pub fn apply_patch(&self, patch: AlignmentPatch) -> Result<AlignmentDoc, String> {
        let current = self.snapshot();
        let next_doc = merge(&current, patch)?;
        let solved = solve(next_doc)?;
        let doc = solved.doc.clone();
        *self.lock() = solved;
        self.touch();
        Ok(doc)
    }

    /// Identity corners, no points, black background. Leaves `enabled` alone —
    /// resetting the geometry should not silently switch the warp off (or on)
    /// under an operator mid-alignment.
    pub fn reset(&self) -> AlignmentDoc {
        let enabled = self.lock().doc.enabled;
        let doc = AlignmentDoc {
            enabled,
            ..AlignmentDoc::default()
        };
        let solved = solve(doc).expect("identity alignment always solves");
        let out = solved.doc.clone();
        *self.lock() = solved;
        self.touch();
        out
    }

    /// The `alignment` telemetry payload / `alignment.get` result.
    ///
    /// Carries the solved RBF **coefficients** alongside the document. They
    /// are derived state, so they never touch `alignment.json` — but a client
    /// that wants to draw the *actual* field (rather than just the corner
    /// quad) would otherwise have to re-implement the solver and drift from
    /// it. The Align canvas's warp grid reads exactly what the LUT was baked
    /// from. Ignored on input.
    pub fn to_json(&self) -> Value {
        let s = self.lock();
        let (w, h) = self.output();
        json!({
            "version": s.doc.version,
            "enabled": s.doc.enabled,
            "background": s.doc.background,
            "corners": s.doc.corners,
            "points": s.doc.points,
            "weights": s.weights,
            "output": [w, h],
            "points_max": MAX_POINTS,
            "test_pattern": self.test_pattern().as_str(),
            "solve_ok": true,
        })
    }
}

// ---------- merge + solve ----------

fn merge(current: &AlignmentSolution, patch: AlignmentPatch) -> Result<AlignmentDoc, String> {
    let mut doc = current.doc.clone();
    doc.version = ALIGNMENT_VERSION;

    if let Some(enabled) = patch.enabled {
        doc.enabled = enabled;
    }
    if let Some(bg) = patch.background {
        parse_hex_rgb(&bg).ok_or_else(|| {
            format!("background {bg:?} is not a #rrggbb colour (e.g. \"#000000\")")
        })?;
        doc.background = bg;
    }

    let corners_moved = patch.corners.is_some();
    if let Some(c) = patch.corners {
        if c.len() != 4 {
            return Err(format!(
                "corners must be exactly 4 dest positions of the source corners \
                 (0,0) (1,0) (1,1) (0,1) — got {}",
                c.len()
            ));
        }
        for (i, p) in c.iter().enumerate() {
            if !p[0].is_finite() || !p[1].is_finite() {
                return Err(format!("corner {i} is not finite"));
            }
        }
        doc.corners = [c[0], c[1], c[2], c[3]];
    }

    match patch.points {
        Some(points) => {
            if points.len() > MAX_POINTS {
                return Err(format!(
                    "{} points exceeds MAX_POINTS = {MAX_POINTS} — remove some before adding more",
                    points.len()
                ));
            }
            let mut next = Vec::with_capacity(points.len());
            let mut next_id = 1usize;
            for (i, p) in points.iter().enumerate() {
                if !p.dest[0].is_finite() || !p.dest[1].is_finite() {
                    return Err(format!("point {i} dest is not finite"));
                }
                let previous = p
                    .id
                    .as_deref()
                    .and_then(|id| current.doc.points.iter().find(|q| q.id == id));
                let radius = p
                    .radius
                    .or_else(|| previous.map(|q| q.radius))
                    .unwrap_or(DEFAULT_RADIUS);
                if !radius.is_finite() || radius <= 0.0 {
                    return Err(format!(
                        "point {i} radius must be finite and > 0 (got {radius})"
                    ));
                }
                // No anchor supplied ⇒ anchor to wherever the *current* field
                // already maps this dest position, which makes adding a
                // handle a no-op on the rendered image (§3.2).
                let anchor = match p.anchor {
                    Some(a) if a[0].is_finite() && a[1].is_finite() => a,
                    Some(_) => return Err(format!("point {i} anchor is not finite")),
                    None => previous
                        .map(|q| q.anchor)
                        .unwrap_or_else(|| current.warp(p.dest)),
                };
                let id = match &p.id {
                    Some(id) if !id.is_empty() => id.clone(),
                    _ => loop {
                        let candidate = format!("p{next_id}");
                        next_id += 1;
                        let taken = next.iter().any(|q: &WarpPoint| q.id == candidate)
                            || points.iter().any(|q| q.id.as_deref() == Some(&candidate));
                        if !taken {
                            break candidate;
                        }
                    },
                };
                next.push(WarpPoint {
                    id,
                    anchor,
                    dest: p.dest,
                    radius,
                });
            }
            let mut seen = std::collections::HashSet::new();
            for p in &next {
                if !seen.insert(p.id.clone()) {
                    return Err(format!("duplicate point id {:?}", p.id));
                }
            }
            doc.points = next;
        }
        None if corners_moved => {
            // §3.2 — corner drags carry the extra handles with them. The
            // dest-space offset `e = d − H(a)` is what stays constant, so the
            // fine corrections stay attached to the content rather than to
            // the screen.
            let h_old = invert3(&current.h_inv)
                .ok_or_else(|| "current corner homography is not invertible".to_string())?;
            let h_new = homography_from_corners(&doc.corners)?;
            for p in doc.points.iter_mut() {
                let anchored_old = apply_h(&h_old, p.anchor);
                let e = [p.dest[0] - anchored_old[0], p.dest[1] - anchored_old[1]];
                let anchored_new = apply_h(&h_new, p.anchor);
                p.dest = [anchored_new[0] + e[0], anchored_new[1] + e[1]];
            }
        }
        None => {}
    }

    Ok(doc)
}

/// Build the derived state for a document: base homography inverse + RBF
/// coefficients. Every failure path here is a rejection, never a partial
/// apply — the caller keeps the previous solution rendering.
pub fn solve(doc: AlignmentDoc) -> Result<AlignmentSolution, String> {
    let h = homography_from_corners(&doc.corners)?;
    let h_inv = invert3(&h).ok_or_else(|| {
        "corner quad is degenerate (collinear or coincident corners) — the base \
         homography has no inverse"
            .to_string()
    })?;

    if doc.points.len() > MAX_POINTS {
        return Err(format!(
            "{} points exceeds MAX_POINTS = {MAX_POINTS}",
            doc.points.len()
        ));
    }
    for (i, p) in doc.points.iter().enumerate() {
        if !p.radius.is_finite() || p.radius <= 0.0 {
            return Err(format!("point {i} ({}) has radius {}", p.id, p.radius));
        }
        if !p.dest[0].is_finite()
            || !p.dest[1].is_finite()
            || !p.anchor[0].is_finite()
            || !p.anchor[1].is_finite()
        {
            return Err(format!("point {i} ({}) has non-finite coordinates", p.id));
        }
    }

    let weights = solve_rbf(&doc.points, &h_inv)?;
    let background_linear = parse_hex_rgb(&doc.background)
        .ok_or_else(|| format!("background {:?} is not a #rrggbb colour", doc.background))?;

    Ok(AlignmentSolution {
        doc,
        h_inv,
        weights,
        background_linear,
    })
}

/// `A w = r` with `A_jk = φ(|d_j − d_k| / σ_k)` and
/// `r_j = a_j − H⁻¹(d_j)`. N ≤ 64, so a dense LU in f64 costs microseconds
/// and the extra precision is free.
fn solve_rbf(points: &[WarpPoint], h_inv: &[[f32; 3]; 3]) -> Result<Vec<[f32; 2]>, String> {
    let n = points.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut a = vec![0.0f64; n * n];
    for j in 0..n {
        for k in 0..n {
            let dx = (points[j].dest[0] - points[k].dest[0]) as f64;
            let dy = (points[j].dest[1] - points[k].dest[1]) as f64;
            let t = (dx * dx + dy * dy).sqrt() / points[k].radius as f64;
            a[j * n + k] = wendland(t);
        }
    }
    let mut rhs = vec![[0.0f64; 2]; n];
    for j in 0..n {
        let base = apply_h(h_inv, points[j].dest);
        rhs[j] = [
            points[j].anchor[0] as f64 - base[0] as f64,
            points[j].anchor[1] as f64 - base[1] as f64,
        ];
    }
    lu_solve(&mut a, n, &mut rhs)?;
    Ok(rhs
        .into_iter()
        .map(|r| [r[0] as f32, r[1] as f32])
        .collect())
}

/// Gaussian elimination with partial pivoting, two right-hand sides.
///
/// A singular system is the documented failure mode (coincident handles,
/// degenerate radii, or a mixed-σ system that happens not to be invertible);
/// the message names the handle so the operator can fix it rather than guess.
fn lu_solve(a: &mut [f64], n: usize, rhs: &mut [[f64; 2]]) -> Result<(), String> {
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in col + 1..n {
            let v = a[row * n + col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if !(best > 1e-12) {
            return Err(format!(
                "handle system is singular at handle {col} — two handles are \
                 probably coincident, or a radius is degenerate; previous \
                 alignment kept"
            ));
        }
        if pivot != col {
            for c in 0..n {
                a.swap(col * n + c, pivot * n + c);
            }
            rhs.swap(col, pivot);
        }
        let d = a[col * n + col];
        for row in col + 1..n {
            let f = a[row * n + col] / d;
            if f == 0.0 {
                continue;
            }
            a[row * n + col] = 0.0;
            for c in col + 1..n {
                a[row * n + c] -= f * a[col * n + c];
            }
            rhs[row][0] -= f * rhs[col][0];
            rhs[row][1] -= f * rhs[col][1];
        }
    }
    for row in (0..n).rev() {
        let mut s = rhs[row];
        for c in row + 1..n {
            s[0] -= a[row * n + c] * rhs[c][0];
            s[1] -= a[row * n + c] * rhs[c][1];
        }
        let d = a[row * n + row];
        rhs[row] = [s[0] / d, s[1] / d];
    }
    Ok(())
}

/// Wendland C² basis, `φ(t) = (1−t)⁴(4t+1)` for `t < 1`, else 0.
///
/// Compact support is the whole point: it is what keeps a mid-frame
/// correction from sliding the corners you already dialled in.
pub fn wendland(t: f64) -> f64 {
    if t >= 1.0 || !t.is_finite() {
        return 0.0;
    }
    let u = 1.0 - t;
    let u2 = u * u;
    u2 * u2 * (4.0 * t + 1.0)
}

// ---------- projective base ----------

/// Heckbert's closed-form unit-square→quad map. `corners` are the dest
/// positions of source `(0,0) (1,0) (1,1) (0,1)`, in that order; the result
/// maps **source → dest** (the shader wants its inverse).
pub fn homography_from_corners(corners: &Corners) -> Result<[[f32; 3]; 3], String> {
    let (x0, y0) = (corners[0][0] as f64, corners[0][1] as f64);
    let (x1, y1) = (corners[1][0] as f64, corners[1][1] as f64);
    let (x2, y2) = (corners[2][0] as f64, corners[2][1] as f64);
    let (x3, y3) = (corners[3][0] as f64, corners[3][1] as f64);

    let sx = x0 - x1 + x2 - x3;
    let sy = y0 - y1 + y2 - y3;

    let (g, h) = if sx.abs() < 1e-12 && sy.abs() < 1e-12 {
        // Parallelogram — the projective terms vanish and the general
        // formulas below reduce to exactly this.
        (0.0, 0.0)
    } else {
        let dx1 = x1 - x2;
        let dx2 = x3 - x2;
        let dy1 = y1 - y2;
        let dy2 = y3 - y2;
        let den = dx1 * dy2 - dx2 * dy1;
        if den.abs() < 1e-12 {
            return Err(
                "corner quad is degenerate (three corners collinear) — drag a corner \
                 back out before continuing"
                    .to_string(),
            );
        }
        ((sx * dy2 - dx2 * sy) / den, (dx1 * sy - sx * dy1) / den)
    };

    let a = x1 - x0 + g * x1;
    let b = x3 - x0 + h * x3;
    let c = x0;
    let d = y1 - y0 + g * y1;
    let e = y3 - y0 + h * y3;
    let f = y0;

    let m = [[a, b, c], [d, e, f], [g, h, 1.0]];
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for cc in 0..3 {
            if !m[r][cc].is_finite() {
                return Err("corner quad produced a non-finite homography".to_string());
            }
            out[r][cc] = m[r][cc] as f32;
        }
    }
    Ok(out)
}

/// Analytic 3×3 inverse via the adjugate. `None` when the matrix is singular.
pub fn invert3(m: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let a: [[f64; 3]; 3] = [
        [m[0][0] as f64, m[0][1] as f64, m[0][2] as f64],
        [m[1][0] as f64, m[1][1] as f64, m[1][2] as f64],
        [m[2][0] as f64, m[2][1] as f64, m[2][2] as f64],
    ];
    let c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    let c01 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
    let c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    let det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let out = [
        [
            c00 * inv_det,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det,
        ],
        [
            c01 * inv_det,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det,
        ],
        [
            c02 * inv_det,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det,
        ],
    ];
    let mut result = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            if !out[r][c].is_finite() {
                return None;
            }
            result[r][c] = out[r][c] as f32;
        }
    }
    Some(result)
}

/// Apply a 3×3 to a 2-vector with the perspective divide.
pub fn apply_h(m: &[[f32; 3]; 3], p: [f32; 2]) -> [f32; 2] {
    let x = m[0][0] * p[0] + m[0][1] * p[1] + m[0][2];
    let y = m[1][0] * p[0] + m[1][1] * p[1] + m[1][2];
    let mut w = m[2][0] * p[0] + m[2][1] * p[1] + m[2][2];
    if w.abs() < 1e-9 {
        w = if w < 0.0 { -1e-9 } else { 1e-9 };
    }
    [x / w, y / w]
}

/// §3.5 migration — derive corner handles from a stored **dest→source**
/// calibration matrix (`src = h · (uv,1)`, the old shader's convention).
/// Corners are *dest positions of source corners*, so this applies the
/// matrix's **inverse** to the unit square. `None` for a non-invertible
/// matrix; the caller falls back to identity with a warning.
pub fn corners_from_dest_to_source(h: &[[f32; 3]; 3]) -> Option<Corners> {
    let inv = invert3(h)?;
    let src = identity_corners();
    let mut out = identity_corners();
    for i in 0..4 {
        let p = apply_h(&inv, src[i]);
        if !p[0].is_finite() || !p[1].is_finite() {
            return None;
        }
        out[i] = p;
    }
    Some(out)
}

// ---------- colour ----------

fn parse_hex_rgb(s: &str) -> Option<[f32; 3]> {
    let h = s.strip_prefix('#')?;
    if h.len() != 6 || !h.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let v = u32::from_str_radix(h, 16).ok()?;
    let srgb = [
        ((v >> 16) & 0xff) as f32 / 255.0,
        ((v >> 8) & 0xff) as f32 / 255.0,
        (v & 0xff) as f32 / 255.0,
    ];
    // The swapchain is sRGB and the final pass writes linear light, so the
    // picked colour has to be linearised or a "#404040" pick lands visibly
    // brighter on the wall than the swatch.
    Some([
        srgb_to_linear(srgb[0]),
        srgb_to_linear(srgb[1]),
        srgb_to_linear(srgb[2]),
    ])
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

// ---------- persistence ----------

/// Per-directory venue state, exactly like the §5.3 session sidecar: all
/// scenes played from one project directory share one physical install.
pub fn alignment_path(scene_path: &Path) -> PathBuf {
    scene_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("alignment.json")
}

pub fn load(path: &Path) -> Result<Option<AlignmentDoc>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let doc: AlignmentDoc =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    if doc.version != ALIGNMENT_VERSION {
        bail!(
            "alignment.json version {} unsupported (this build expects {})",
            doc.version,
            ALIGNMENT_VERSION
        );
    }
    Ok(Some(doc))
}

/// Atomic temp+rename, so a power blink mid-drag can't leave a torn file.
pub fn save(path: &Path, doc: &AlignmentDoc) -> Result<()> {
    let raw = serde_json::to_vec_pretty(doc).context("serializing alignment")?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &raw).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 2], b: [f32; 2], eps: f32) -> bool {
        (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps
    }

    #[test]
    fn identity_corners_round_trip() {
        let s = solve(AlignmentDoc::default()).unwrap();
        for p in [[0.0, 0.0], [1.0, 1.0], [0.37, 0.82], [0.5, 0.5]] {
            assert!(close(s.warp(p), p, 1e-5), "warp({p:?}) = {:?}", s.warp(p));
        }
        // Zero offsets ⇒ a baked LUT of zeros ⇒ identity sampling.
        let u = s.bake_uniforms();
        assert_eq!(u.counts[0], 0);
    }

    #[test]
    fn keystone_maps_corners_exactly() {
        // Deliberately top-vs-bottom asymmetric: a Y-flip mismatch between
        // the bake pass and the final pass cannot hide behind a symmetric
        // test quad (the §Phase-A verify note).
        let corners: Corners = [[0.10, 0.05], [0.90, 0.02], [1.00, 0.98], [0.00, 0.90]];
        let doc = AlignmentDoc {
            corners,
            ..Default::default()
        };
        let s = solve(doc).unwrap();
        let src = identity_corners();
        for i in 0..4 {
            // W maps dest → source, so the dest corner must land on its
            // source corner exactly.
            assert!(
                close(s.warp(corners[i]), src[i], 1e-4),
                "corner {i}: warp({:?}) = {:?}, want {:?}",
                corners[i],
                s.warp(corners[i]),
                src[i]
            );
        }
    }

    #[test]
    fn adding_a_handle_at_w_leaves_the_field_unchanged() {
        // The §3.2 no-op-add property, with one handle already bending the
        // field so this is not the trivial identity case.
        let state = AlignmentState::identity();
        state
            .apply_patch(AlignmentPatch {
                corners: Some(vec![[0.05, 0.0], [0.95, 0.03], [1.0, 1.0], [0.0, 0.97]]),
                ..Default::default()
            })
            .unwrap();
        state
            .apply_patch(AlignmentPatch {
                points: Some(vec![PointPatch {
                    id: None,
                    anchor: Some([0.5, 0.5]),
                    dest: [0.55, 0.48],
                    radius: Some(0.3),
                }]),
                ..Default::default()
            })
            .unwrap();

        let before = state.snapshot();
        let probes = [[0.2, 0.2], [0.5, 0.5], [0.62, 0.55], [0.85, 0.7]];
        let want: Vec<_> = probes.iter().map(|p| before.warp(*p)).collect();

        // Add a second handle with no anchor — the engine anchors it at
        // W_current(dest), so the enlarged solve must reproduce the field.
        let mut points: Vec<PointPatch> = before
            .doc
            .points
            .iter()
            .map(|p| PointPatch {
                id: Some(p.id.clone()),
                anchor: Some(p.anchor),
                dest: p.dest,
                radius: Some(p.radius),
            })
            .collect();
        points.push(PointPatch {
            id: None,
            anchor: None,
            dest: [0.3, 0.7],
            radius: Some(0.25),
        });
        state
            .apply_patch(AlignmentPatch {
                points: Some(points),
                ..Default::default()
            })
            .unwrap();

        let after = state.snapshot();
        assert_eq!(after.doc.points.len(), 2);
        // The new coefficient is exactly zero and nothing else moved.
        assert!(
            after.weights[1][0].abs() < 1e-5 && after.weights[1][1].abs() < 1e-5,
            "new handle coefficient should be ~0, got {:?}",
            after.weights[1]
        );
        for (p, w) in probes.iter().zip(want.iter()) {
            assert!(
                close(after.warp(*p), *w, 1e-4),
                "field moved at {p:?}: {:?} vs {:?}",
                after.warp(*p),
                w
            );
        }
    }

    #[test]
    fn handle_constraint_is_interpolated() {
        let state = AlignmentState::identity();
        state
            .apply_patch(AlignmentPatch {
                points: Some(vec![PointPatch {
                    id: None,
                    anchor: Some([0.5, 0.5]),
                    dest: [0.6, 0.4],
                    radius: Some(0.3),
                }]),
                ..Default::default()
            })
            .unwrap();
        let s = state.snapshot();
        assert!(close(s.warp([0.6, 0.4]), [0.5, 0.5], 1e-4));
        // Compact support: outside the radius the base homography is exact.
        assert!(close(s.warp([0.05, 0.95]), [0.05, 0.95], 1e-5));
    }

    #[test]
    fn degenerate_corners_keep_last_good() {
        let state = AlignmentState::identity();
        let good: Corners = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]];
        state
            .apply_patch(AlignmentPatch {
                corners: Some(good.to_vec()),
                ..Default::default()
            })
            .unwrap();
        // All four corners on one point — no invertible homography.
        let err = state
            .apply_patch(AlignmentPatch {
                corners: Some(vec![[0.5, 0.5]; 4]),
                ..Default::default()
            })
            .unwrap_err();
        assert!(!err.is_empty());
        assert_eq!(state.doc().corners, good);
    }

    #[test]
    fn coincident_handles_are_rejected_not_half_applied() {
        let state = AlignmentState::identity();
        let err = state
            .apply_patch(AlignmentPatch {
                points: Some(vec![
                    PointPatch {
                        id: Some("a".into()),
                        anchor: Some([0.4, 0.4]),
                        dest: [0.5, 0.5],
                        radius: Some(0.3),
                    },
                    PointPatch {
                        id: Some("b".into()),
                        anchor: Some([0.6, 0.6]),
                        dest: [0.5, 0.5],
                        radius: Some(0.3),
                    },
                ]),
                ..Default::default()
            })
            .unwrap_err();
        assert!(err.contains("singular"), "unexpected message: {err}");
        assert!(state.doc().points.is_empty());
    }

    #[test]
    fn corner_drag_carries_extra_handles() {
        let state = AlignmentState::identity();
        state
            .apply_patch(AlignmentPatch {
                points: Some(vec![PointPatch {
                    id: None,
                    anchor: Some([0.5, 0.5]),
                    dest: [0.5, 0.5],
                    radius: Some(0.3),
                }]),
                ..Default::default()
            })
            .unwrap();
        // Translate the whole quad right by 0.2 — the handle must ride along.
        state
            .apply_patch(AlignmentPatch {
                corners: Some(vec![[0.2, 0.0], [1.2, 0.0], [1.2, 1.0], [0.2, 1.0]]),
                ..Default::default()
            })
            .unwrap();
        let d = state.doc().points[0].dest;
        assert!(close(d, [0.7, 0.5], 1e-4), "handle dest = {d:?}");
    }

    #[test]
    fn point_cap_is_a_message_not_a_truncation() {
        let state = AlignmentState::identity();
        let points: Vec<PointPatch> = (0..MAX_POINTS + 1)
            .map(|i| PointPatch {
                id: Some(format!("p{i}")),
                anchor: Some([0.5, 0.5]),
                dest: [i as f32 * 0.001, 0.5],
                radius: Some(0.1),
            })
            .collect();
        let err = state
            .apply_patch(AlignmentPatch {
                points: Some(points),
                ..Default::default()
            })
            .unwrap_err();
        assert!(err.contains("MAX_POINTS"), "unexpected message: {err}");
        assert!(state.doc().points.is_empty());
    }

    #[test]
    fn migration_inverts_the_dest_to_source_matrix() {
        // A stored calibration maps dest uv → source uv; corners are dest
        // positions of source corners, so migration must invert it.
        let corners: Corners = [[0.1, 0.05], [0.9, 0.0], [1.0, 0.95], [0.0, 1.0]];
        let h = homography_from_corners(&corners).unwrap(); // source → dest
        let dest_to_source = invert3(&h).unwrap();
        let recovered = corners_from_dest_to_source(&dest_to_source).unwrap();
        for i in 0..4 {
            assert!(
                close(recovered[i], corners[i], 1e-4),
                "corner {i}: {:?} vs {:?}",
                recovered[i],
                corners[i]
            );
        }
    }

    #[test]
    fn document_round_trips_through_disk() {
        let dir = std::env::temp_dir().join(format!("wzrd-align-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("alignment.json");
        let doc = AlignmentDoc {
            enabled: true,
            background: "#101010".into(),
            corners: [[0.02, 0.01], [0.98, 0.0], [1.0, 0.99], [0.0, 1.0]],
            points: vec![WarpPoint {
                id: "p1".into(),
                anchor: [0.5, 0.5],
                dest: [0.52, 0.5],
                radius: 0.35,
            }],
            ..Default::default()
        };
        save(&path, &doc).unwrap();
        let back = load(&path).unwrap().unwrap();
        assert_eq!(back.points.len(), 1);
        assert_eq!(back.background, "#101010");
        assert!(close(back.corners[1], [0.98, 0.0], 1e-6));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_file_is_none() {
        assert!(load(Path::new("/nonexistent/wzrd/alignment.json"))
            .unwrap()
            .is_none());
    }

    #[test]
    fn background_is_linearised() {
        let s = solve(AlignmentDoc {
            background: "#ffffff".into(),
            ..Default::default()
        })
        .unwrap();
        assert!((s.background_linear[0] - 1.0).abs() < 1e-5);
        let s = solve(AlignmentDoc {
            background: "#000000".into(),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(s.background_linear, [0.0, 0.0, 0.0]);
        assert!(solve(AlignmentDoc {
            background: "black".into(),
            ..Default::default()
        })
        .is_err());
    }
}
