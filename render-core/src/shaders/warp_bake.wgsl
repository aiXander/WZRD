// §5.14 warp bake — evaluates the alignment model into an offset LUT.
//
// Runs *only* when the alignment changes or the swapchain resizes, encoded
// into the same frame encoder as step 0 of the frame (no extra submit). The
// per-frame cost of the warp is therefore one texel read in the final pass,
// independent of handle count: 4 corners and a future camera pass with 500
// correspondences cost exactly the same at show time.
//
// Output is the **offset** `W(x) − x` in source uv, not the absolute source
// position. Zero then means identity, which keeps the disabled/dummy case
// trivial and makes a half-float fallback sub-pixel accurate if the full
// Rg32Float LUT ever gets too big.
//
// Vertex stage (incl. the Y flip) comes from `fullscreen_vs.wgsl`, shared
// verbatim with the final pass — see the note there.

const MAX_POINTS: u32 = 64u;

struct WarpBake {
    // dest → source, the projective base. Rows padded to vec4 for std140.
    h0: vec4<f32>,
    h1: vec4<f32>,
    h2: vec4<f32>,
    // (handle count, _, _, _). Named `counts` because `meta` is a reserved
    // WGSL keyword.
    counts: vec4<u32>,
    // (dest.x, dest.y, 1/radius, _)
    points: array<vec4<f32>, MAX_POINTS>,
    // (w.x, w.y, _, _)
    weights: array<vec4<f32>, MAX_POINTS>,
};

@group(0) @binding(0) var<uniform> bake: WarpBake;

// Wendland C², φ(t) = (1−t)⁴(4t+1) for t < 1 — compactly supported, so a
// handle bends its neighbourhood and leaves the rest of the frame alone.
fn wendland(t: f32) -> f32 {
    if (t >= 1.0) {
        return 0.0;
    }
    let u = 1.0 - t;
    let u2 = u * u;
    return u2 * u2 * (4.0 * t + 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec2<f32> {
    let d = in.uv;

    let h = mat3x3<f32>(
        vec3<f32>(bake.h0.x, bake.h1.x, bake.h2.x),
        vec3<f32>(bake.h0.y, bake.h1.y, bake.h2.y),
        vec3<f32>(bake.h0.z, bake.h1.z, bake.h2.z),
    );
    let p = h * vec3<f32>(d, 1.0);
    var src = p.xy / p.z;

    let n = bake.counts.x;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let pt = bake.points[i];
        let t = length(d - pt.xy) * pt.z;
        let f = wendland(t);
        if (f > 0.0) {
            src = src + bake.weights[i].xy * f;
        }
    }

    return src - d;
}
