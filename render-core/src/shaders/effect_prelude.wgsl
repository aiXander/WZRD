// Shared shader prelude — every effect (built-in or user-authored) is
// compiled as `prelude + effect_body + main`. The structs and bind groups
// here are the contract user WGSL writes against (§3.6, D15).
//
// Output contract: `effect(uv, mask) -> vec4<f32>` returns **premultiplied
// RGBA** (`rgb * intensity`, `intensity`). The compositor blends additively
// (One + One), so naked RGB would dump full-brightness light everywhere the
// shader runs and a dark layer would still contribute. Bake the mask and
// any intensity scalar into the alpha, then multiply rgb by that alpha.

// Audio fields (`audio_low/mid/high`, `onset_low/mid/high`) arrive over
// OSC from the Realtime Audio Feature Server (see audio_refactor_plan.md
// §3.1). `audio_*` floats are already auto-scaled into ~[0, 1]; onset_*
// are decaying envelopes stamped by `/audio/onset/<band>` triggers and
// read with `exp(-dt/τ)` on the engine side. No `audio_rms` in v1 — pick
// a band explicitly if you want "loudness."
struct FrameState {
    time: f32,
    bar_phase: f32,
    beat_phase: f32,
    bpm: f32,
    audio_low: f32,
    audio_mid: f32,
    audio_high: f32,
    onset_low: f32,
    onset_mid: f32,
    onset_high: f32,
    _pad0: f32,
    _pad1: f32,
    resolution: vec4<f32>,   // (width, height, _, _)
};

struct LayerParams {
    slice: u32,
    effect_id: u32,
    // §5.2 per-layer identity — position of this pass within its binding's
    // resolved selection, plus a stable per-layer random and uv geometry.
    layer_index: u32,
    layer_count: u32,
    layer_seed: f32,
    // Operator layer fader (0..1) — the `@opacity`/`@mute`/`@solo` override
    // trio folded into one scalar by the compositor. Applied by `fs_main`
    // after `effect()` returns, so user WGSL never has to honour it.
    opacity: f32,
    centroid_uv: vec2<f32>,
    bbox_uv: vec4<f32>,      // (min_x, min_y, max_x, max_y), uv space
    // 8 scalar slots packed into two vec4s (std140 16-byte alignment).
    params_f: array<vec4<f32>, 2>,
    // 4 colour slots.
    params_c: array<vec4<f32>, 4>,
    // Gig plan C5 image placement: the uv rect the image is stretched into
    // (min_x, min_y, max_x, max_y) — the layer's source box under
    // "region", the whole canvas under "canvas".
    image_box: vec4<f32>,
    // (bound 0|1, encoding 0 = sRGB-decoded RGBA (straight alpha) | 1 =
    // scaled YCoCg (HAP Q), 1 / image_box width, 1 / image_box height).
    image_meta: vec4<f32>,
};

@group(0) @binding(0) var mask_atlas: texture_2d_array<f32>;
@group(0) @binding(1) var mask_sampler: sampler;
@group(0) @binding(2) var<uniform> state: FrameState;
@group(0) @binding(3) var<uniform> layer: LayerParams;
// The binding's `image` input (a HAP clip frame or a still). Passes without
// one bind a 1×1 transparent texture, so `sample_image` is always legal.
@group(0) @binding(4) var image_tex: texture_2d<f32>;
@group(0) @binding(5) var image_sampler: sampler;

// Helper accessors so effect code reads `f_param(0u)` instead of unpacking
// vec4 lanes by hand.
fn f_param(i: u32) -> f32 {
    let v = layer.params_f[i >> 2u];
    let lane = i & 3u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

fn c_param(i: u32) -> vec4<f32> {
    return layer.params_c[i];
}

// §5.2 per-layer identity accessors. One binding over N layers gets a
// distinct value per pass, so a single effect can vary organically across
// its selection instead of animating in lockstep:
//   - `phase += layer_seed()` desynchronizes N copies of a cycle for free
//     (seed is a stable hash of the layer *id* — survives re-segmentation);
//   - `layer_index()` / `layer_count()` support ordered ramps/cascades;
//   - `layer_centroid()` / `layer_bbox()` locate the region for radial
//     blooms, distance falloffs, per-region uv normalization.
fn layer_seed() -> f32 { return layer.layer_seed; }
fn layer_index() -> u32 { return layer.layer_index; }
fn layer_count() -> u32 { return layer.layer_count; }
fn layer_centroid() -> vec2<f32> { return layer.centroid_uv; }
fn layer_bbox() -> vec4<f32> { return layer.bbox_uv; }

// Re-sample this layer's mask at an arbitrary UV. Useful for spatial
// effects (wobble, displacement) that need to test whether a *displaced*
// pixel falls inside the semantic region. Calling it makes the effect
// full-canvas (§2.4 output bounds) unless its descriptor declares
// `"bounds": "region"`. Explicit LOD (the atlas has one mip — identical
// result), so it is legal in non-uniform control flow and an effect may
// early-out on `mask == 0`.
fn sample_mask(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(mask_atlas, mask_sampler, uv, i32(layer.slice), 0.0).r;
}

// Sample the binding's `image` input at canvas uv. Returns premultiplied
// RGBA, transparent outside the image's placement box (and everywhere when
// no image is bound). Uses an explicit LOD so it is legal in non-uniform
// control flow — call it from any branch, at any displaced uv.
fn sample_image(uv: vec2<f32>) -> vec4<f32> {
    let b = layer.image_box;
    let local = (uv - b.xy) * layer.image_meta.zw;
    let inside = all(local >= vec2<f32>(0.0)) && all(local <= vec2<f32>(1.0));
    if (layer.image_meta.x < 0.5 || !inside) {
        return vec4<f32>(0.0);
    }
    let t = textureSampleLevel(image_tex, image_sampler, local, 0.0);
    if (layer.image_meta.y > 0.5) {
        // Scaled YCoCg in DXT5 (HAP Q): rgba = (Co, Cg, scale, Y), values
        // sRGB-encoded — convert, then decode to linear light.
        let scale = t.z * (255.0 / 8.0) + 1.0;
        let co = (t.x - 0.5 * 256.0 / 255.0) / scale;
        let cg = (t.y - 0.5 * 256.0 / 255.0) / scale;
        let enc = clamp(vec3<f32>(t.w + co - cg, t.w + cg, t.w - co - cg), vec3<f32>(0.0), vec3<f32>(1.0));
        let lo = enc / 12.92;
        let hi = pow((enc + 0.055) / 1.055, vec3<f32>(2.4));
        return vec4<f32>(select(hi, lo, enc <= vec3<f32>(0.04045)), 1.0);
    }
    return vec4<f32>(t.rgb * t.a, t.a);
}
