// Shared vertex + fragment entry points. Sits after the effect body in
// every compiled shader, so `fs_main` calls into `effect()` (which the
// effect body must define).

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
    // Flip Y so authoring-space pixels (y-down) match shader UVs.
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let mask = sample_mask(in.uv);
    return effect(in.uv, mask);
}
