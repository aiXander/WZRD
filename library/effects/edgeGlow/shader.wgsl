// edgeGlow — the clip's outlines, lit; the flat areas go dark.
// Library clip effect (gig F1, Resolume's Edge Detection). Audio: drive `boost`.
//
// Sobel on gamma-encoded luma over a 3×3 ring `width` output pixels wide.
// Edges take the clip's own colour (normalised so dark edges still glow),
// blended toward `tint` by `colorize`; `fill` lets a little of the plain
// clip show underneath.

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

fn luma(c: vec4<f32>) -> f32 {
    return sqrt(max(dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0));
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let width = max(f_param(0u), 0.25);
    let boost = f_param(1u);
    let fill = f_param(2u);
    let colorize = clamp(f_param(3u), 0.0, 1.0);
    let gain = f_param(4u);
    let tint = c_param(0u);

    let px = width / max(state.resolution.xy, vec2<f32>(1.0));
    let c = fx_in(uv);
    let tl = luma(fx_in(uv + vec2<f32>(-px.x, -px.y)));
    let tc = luma(fx_in(uv + vec2<f32>(0.0, -px.y)));
    let tr = luma(fx_in(uv + vec2<f32>(px.x, -px.y)));
    let ml = luma(fx_in(uv + vec2<f32>(-px.x, 0.0)));
    let mr = luma(fx_in(uv + vec2<f32>(px.x, 0.0)));
    let bl = luma(fx_in(uv + vec2<f32>(-px.x, px.y)));
    let bc = luma(fx_in(uv + vec2<f32>(0.0, px.y)));
    let br = luma(fx_in(uv + vec2<f32>(px.x, px.y)));
    let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    let gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    let edge = clamp(length(vec2<f32>(gx, gy)) * boost, 0.0, 1.0);

    let hue = c.rgb / max(max(c.r, max(c.g, c.b)), 1e-3);
    let ink = mix(hue, tint.rgb, colorize);
    let rgb = ink * edge + c.rgb * fill;
    let a = clamp(max(edge, c.a * fill), 0.0, 1.0);
    return vec4<f32>(rgb, a) * (mask * gain);
}
