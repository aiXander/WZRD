// Per-layer pass shader for the `tint` built-in (Phase 2 baseline).
// Future effects either fuse into this fragment (fast path, §3.6) or render
// into their own slow-path FBO; the bind-group layout grows accordingly.

struct LayerUniforms {
    color: vec4<f32>,
    slice: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var mask_atlas: texture_2d_array<f32>;
@group(0) @binding(1) var mask_sampler: sampler;
@group(0) @binding(2) var<uniform> layer: LayerUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    // Fullscreen triangle covering NDC [-1,1] with UVs in [0,1].
    var uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    var out: VsOut;
    out.pos = pos;
    // Flip Y so the mask isn't upside-down vs. authoring-space pixels.
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let mask = textureSample(mask_atlas, mask_sampler, in.uv, i32(layer.slice)).r;
    return vec4<f32>(layer.color.rgb, layer.color.a * mask);
}
