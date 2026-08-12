// Shared fullscreen-triangle vertex stage for the two output-side passes:
// `warp_bake.wgsl` (dest → offset LUT) and `final_pass.wgsl` (composite →
// swapchain). Prepended to both by `gpu.rs`.
//
// It lives in its own file for one reason: the bake pass writes a LUT that
// the final pass then indexes by framebuffer pixel, so the two stages must
// agree on the dest-uv convention **including the Y flip**. A drift between
// them renders as a vertically mirrored warp — visually plausible enough to
// waste an evening on a ladder. Sharing the source makes that unrepresentable.

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
    // dest uv: origin top-left, matching `@builtin(position).xy` ordering.
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}
