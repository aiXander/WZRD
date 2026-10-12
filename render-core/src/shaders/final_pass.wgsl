// The final compositor pass — everything between the composites and the
// projector, in one pass, one submit:
//
//   1. §5.14 alignment warp (dest → source via the baked offset LUT)
//   2. §5.6 promote crossfade (live × design lerped by adjust.z)
//   3. §5.4 output masters (saturation, then brightness)
//
// Ordering is the point: alignment describes the physical install, so it sits
// *outside* both legs and outside the masters — it is never duplicated per leg
// and never copied on promote/pull.
//
// The native operator preview runs the same math through `fs_preview`
// (bottom of this file). Its target has its own size, so it can never index
// the projector-sized LUT by pixel: when it shows the program view (Align)
// it indexes by uv instead. The projector entry point `fs_main` is untouched
// by that and never sees the preview's extra bindings.
//
// Vertex stage comes from `fullscreen_vs.wgsl`, shared verbatim with
// `warp_bake.wgsl` so the two cannot disagree about the Y flip.

struct FinalPass {
    // (brightness, saturation, mix live→design, warp enable).
    // warp enable lives here rather than in the LUT contents because the
    // projector pass (warp on) and the preview blit (warp off) share a frame,
    // so a zeroed-LUT convention can't serve both — and `enabled: false`
    // becomes a uniform write instead of a rebake-to-zero.
    adjust: vec4<f32>,
    // Linear-light paint for dest pixels whose source falls outside the
    // composite. Non-black floods the physical surface with light and breaks
    // the additive thesis — an alignment aid, not a show setting.
    background: vec4<f32>,
    // §3.6 test pattern: (mode, line thickness in source uv, grid cells, _).
    // mode 0 none / 1 grid / 2 border / 3 corners. Generated in *source*
    // space so it warps with the content and reveals misalignment against
    // physical edges.
    pattern: vec4<f32>,
};

@group(0) @binding(0) var composite_tex: texture_2d<f32>;
@group(0) @binding(1) var composite_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: FinalPass;
@group(0) @binding(3) var design_tex: texture_2d<f32>;
// Rg32Float, sized exactly to this pass's target, holding W(x) − x in source
// uv. Read with textureLoad at the fragment's own pixel — LUT resolution
// equals output resolution, so nearest access is exact and no sampler
// filtering is involved, which is what lets a 32-bit float format stay on
// core wgpu (no `float32-filterable`).
@group(0) @binding(4) var warp_lut: texture_2d<f32>;

fn test_pattern(uv: vec2<f32>, mode: f32, th: f32, cells: f32) -> vec3<f32> {
    // Every mode outlines the source rectangle — that edge is the single most
    // useful thing to line up against a physical surface.
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    var lit = edge < th * 1.5;

    if (mode >= 0.5 && mode < 1.5) {
        let fx = fract(uv.x * cells);
        let fy = fract(uv.y * cells);
        let dx = min(fx, 1.0 - fx) / cells;
        let dy = min(fy, 1.0 - fy) / cells;
        lit = lit || dx < th || dy < th;
    } else if (mode >= 2.5) {
        // Corner L-marks + a centre crosshair: the four points the corner
        // handles actually control, plus the thing that tells you whether the
        // middle is bulging.
        let arm = 0.12;
        let cx = min(uv.x, 1.0 - uv.x);
        let cy = min(uv.y, 1.0 - uv.y);
        lit = lit || (cx < arm && cy < th * 3.0) || (cy < arm && cx < th * 3.0);
        lit = lit
            || (abs(uv.x - 0.5) < th && abs(uv.y - 0.5) < 0.06)
            || (abs(uv.y - 0.5) < th && abs(uv.x - 0.5) < 0.06);
    }

    if (lit) {
        return vec3<f32>(1.0);
    }
    return vec3<f32>(0.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var off = vec2<f32>(0.0);
    if (uniforms.adjust.w > 0.5) {
        // textureLoad takes no implicit derivatives, so this branch is legal
        // regardless of the uniformity of the condition.
        off = textureLoad(warp_lut, vec2<i32>(in.pos.xy), 0).xy;
    }
    return shade(in.uv + off);
}

// Everything after the warp lookup: promote mix, test pattern, masters,
// out-of-source background. Shared by the projector and the preview.
fn shade(uv: vec2<f32>) -> vec4<f32> {
    let inside = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;

    // Sampled unconditionally (the sampler clamps) so the two textureSample
    // calls stay in uniform control flow — the out-of-bounds decision is
    // applied at the very end instead.
    let c_live = textureSample(composite_tex, composite_sampler, uv);
    let c_design = textureSample(design_tex, composite_sampler, uv);
    // §5.6 promote crossfade — 0 outside a fade, ramped 0→1 while promoting.
    var c = mix(c_live, c_design, uniforms.adjust.z);
    if (uniforms.pattern.x > 0.5) {
        c = vec4<f32>(
            test_pattern(uv, uniforms.pattern.x, uniforms.pattern.y, uniforms.pattern.z),
            1.0,
        );
    }

    // Saturation as a lerp from luma, then brightness. Order matters only for
    // readability — both are linear operations on linear-light values.
    let luma = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let rgb = mix(vec3<f32>(luma), c.rgb, uniforms.adjust.y) * uniforms.adjust.x;

    if (!inside) {
        return vec4<f32>(uniforms.background.rgb, 1.0);
    }
    return vec4<f32>(rgb, c.a);
}

// ---------- native preview (docs/TODO/native-preview-underlay.md D2–D4) ----------
//
// Group 1 is bound only by the preview pipeline. Core Animation can only
// alpha-over HTML on top of the native view, so any image blending a room
// needs (Prepare's photo under the light, Align's dimmed underlay) happens
// here, in linear light: light adds to reflected light on the real wall.

struct PreviewExtra {
    // (dim, photo_weight, light_weight, _)
    weights: vec4<f32>,
};

// The pack's surface photo (`pack.json` `surface`), Rgba8UnormSrgb so the
// sample is already linear; a 1×1 black texture when the pack has none.
@group(1) @binding(0) var photo_tex: texture_2d<f32>;
@group(1) @binding(1) var<uniform> preview: PreviewExtra;

@fragment
fn fs_preview(in: VsOut) -> @location(0) vec4<f32> {
    var off = vec2<f32>(0.0);
    if (uniforms.adjust.w > 0.5) {
        // The preview target is not the LUT's size: index by uv. Nearest
        // texel of a projector-res LUT is plenty for a monitor.
        let dims = vec2<f32>(textureDimensions(warp_lut));
        let texel = clamp(floor(in.uv * dims), vec2<f32>(0.0), dims - 1.0);
        off = textureLoad(warp_lut, vec2<i32>(texel), 0).xy;
    }
    let uv = in.uv + off;
    let light = shade(uv);
    // Sampled unconditionally (uniform control flow), masked to the source
    // rect so the clamped edge doesn't smear into the background.
    let inside = all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
    let photo = select(
        vec3<f32>(0.0),
        textureSample(photo_tex, composite_sampler, uv).rgb,
        inside,
    );
    let w = preview.weights;
    return vec4<f32>((light.rgb * w.z + photo * w.y) * w.x, 1.0);
}
