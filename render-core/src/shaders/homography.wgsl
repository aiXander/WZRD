// Final compositor pass — warps the composite buffer through a 3×3 homography
// into the swapchain (D9) and applies the §5.4 output masters (brightness,
// saturation). Identity matrix = no warp. Calibration UI later edits this
// matrix; the shader stays unchanged.

struct Homography {
    // Row-major 3×3, padded to 4×3 because std140 aligns vec3 to 16 bytes.
    m0: vec4<f32>, // (m00, m01, m02, _)
    m1: vec4<f32>, // (m10, m11, m12, _)
    m2: vec4<f32>, // (m20, m21, m22, _)
    // §5.4 output masters: (brightness, saturation, _, _). Applied here —
    // the last pass before the projector — so the composite (and the
    // operator preview that reads it) stays un-mastered.
    adjust: vec4<f32>,
};

@group(0) @binding(0) var composite_tex: texture_2d<f32>;
@group(0) @binding(1) var composite_sampler: sampler;
@group(0) @binding(2) var<uniform> homography: Homography;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    var uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    var out: VsOut;
    out.pos = pos;
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let h = mat3x3<f32>(
        vec3<f32>(homography.m0.x, homography.m1.x, homography.m2.x),
        vec3<f32>(homography.m0.y, homography.m1.y, homography.m2.y),
        vec3<f32>(homography.m0.z, homography.m1.z, homography.m2.z),
    );
    let src = h * vec3<f32>(in.uv, 1.0);
    let uv = src.xy / src.z;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let c = textureSample(composite_tex, composite_sampler, uv);
    // Saturation as a lerp from luma, then brightness. Order matters only
    // for readability — both are linear operations on linear-light values.
    let luma = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let rgb = mix(vec3<f32>(luma), c.rgb, homography.adjust.y) * homography.adjust.x;
    return vec4<f32>(rgb, c.a);
}
