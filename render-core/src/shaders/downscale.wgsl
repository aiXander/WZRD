// Thumbnail downscale for the JPEG `preview` channel (docs/TODO/native-
// preview-underlay.md D5): the linear Rgba16Float composite → a small
// Rgba8UnormSrgb target, so only ~230 KB is read back instead of the full
// composite, and the sRGB encode happens for free on store.
//
// One bilinear tap per output pixel aliases badly at 3–6× reduction, so
// each output pixel averages a 3×3 grid of bilinear taps spread across its
// own footprint (dpdx/dpdy of uv = one output texel) — ~36 source texels at
// 6×, close enough to a box filter for a thumbnail.
//
// Vertex stage: `fullscreen_vs.wgsl`, prepended by `gpu.rs`.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = vec2<f32>(abs(dpdx(in.uv).x), abs(dpdy(in.uv).y));
    var acc = vec3<f32>(0.0);
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let o = vec2<f32>(f32(i), f32(j)) * texel / 3.0;
            acc += textureSampleLevel(src, src_sampler, in.uv + o, 0.0).rgb;
        }
    }
    return vec4<f32>(acc / 9.0, 1.0);
}
