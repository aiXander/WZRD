// Studio file preview (studio plan P1): one media file — a HAP clip or a
// still, opened outside any scene — blitted "contain"-fit onto the native
// preview surface, black around it. Never reads a leg's composite.
//
// Prepended with `fullscreen_vs.wgsl` (dest uv: origin top-left) by `gpu.rs`.

struct FileUniforms {
    // Where the image lands, in dest uv: (x0, y0, x1, y1), contain-fitted.
    image_box: vec4<f32>,
    // (bound, encoding: 0 = sRGB texture / straight alpha, 1 = scaled YCoCg
    // (HAP Q), 1 / box width, 1 / box height).
    image_meta: vec4<f32>,
};

@group(0) @binding(0) var file_tex: texture_2d<f32>;
@group(0) @binding(1) var file_sampler: sampler;
@group(0) @binding(2) var<uniform> file_u: FileUniforms;

@fragment
fn fs_file(in: VsOut) -> @location(0) vec4<f32> {
    let b = file_u.image_box;
    let local = (in.uv - b.xy) * file_u.image_meta.zw;
    let inside = all(local >= vec2<f32>(0.0)) && all(local <= vec2<f32>(1.0));
    // Sample unconditionally (uniform control flow), select after.
    let t = textureSampleLevel(file_tex, file_sampler, clamp(local, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
    if (file_u.image_meta.x < 0.5 || !inside) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    if (file_u.image_meta.y > 0.5) {
        // Scaled YCoCg in DXT5 (HAP Q) — same conversion as the effect
        // prelude's `sample_image`, then sRGB-decode to linear light.
        let scale = t.z * (255.0 / 8.0) + 1.0;
        let co = (t.x - 0.5 * 256.0 / 255.0) / scale;
        let cg = (t.y - 0.5 * 256.0 / 255.0) / scale;
        let enc = clamp(vec3<f32>(t.w + co - cg, t.w + cg, t.w - co - cg), vec3<f32>(0.0), vec3<f32>(1.0));
        let lo = enc / 12.92;
        let hi = pow((enc + 0.055) / 1.055, vec3<f32>(2.4));
        return vec4<f32>(select(hi, lo, enc <= vec3<f32>(0.04045)), 1.0);
    }
    // Straight alpha over black.
    return vec4<f32>(t.rgb * t.a, 1.0);
}
