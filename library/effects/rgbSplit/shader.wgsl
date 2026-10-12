// rgbSplit — red and blue pulled apart along a slowly spinning direction.
// Library clip effect (gig F1, Resolume's Shift RGB). Audio: drive `amount`.
//
// Red reads at +offset, green in place, blue at −offset; alpha is the max
// of the three so the fringes light up past the clip's edge. Displaced
// reads are clamped into the clip box (no black seams at the box edge);
// pixels outside the box stay dark.

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

const TAU: f32 = 6.2831853;

fn in_box(uv: vec2<f32>) -> bool {
    let b = layer.image_box;
    return all(uv >= b.xy) && all(uv <= b.zw);
}

fn clamp_box(uv: vec2<f32>) -> vec2<f32> {
    let b = layer.image_box;
    return clamp(uv, b.xy, b.zw);
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let amount = f_param(0u);   // uv units of canvas height
    let angle = f_param(1u);    // turns
    let spin = f_param(2u);     // turns per bar
    let gain = f_param(3u);
    if (!in_box(uv)) {
        return vec4<f32>(0.0);
    }

    let bars = state.time * state.bpm / 240.0;
    let a = (angle + bars * spin) * TAU;
    let aspect = state.resolution.x / max(state.resolution.y, 1.0);
    let o = vec2<f32>(cos(a) / aspect, sin(a)) * amount;

    let r = fx_in(clamp_box(uv + o));
    let g = fx_in(uv);
    let bl = fx_in(clamp_box(uv - o));
    let c = vec4<f32>(r.r, g.g, bl.b, max(max(r.a, g.a), bl.a));
    return c * (mask * gain);
}
