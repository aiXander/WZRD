// mirror — fold or flip the clip inside its own box.
// Library clip effect (gig F1). Audio: drive `sway` (or `axis`).
//
// mode 0 = left half mirrored onto the right, 1 = top onto bottom,
// 2 = quad (both), 3 = flip horizontal, 4 = flip vertical (Resolume's Flip).
// In the fold modes the axis sits at `axis` and drifts by ±`sway` once
// every two bars; folded uvs mirror-repeat, so an off-centre axis never
// opens a hole.

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

const TAU: f32 = 6.2831853;

fn tri(x: f32) -> f32 {
    return 1.0 - abs(fract(x * 0.5) * 2.0 - 1.0);
}

fn fold(x: f32, ax: f32) -> f32 {
    return tri(select(2.0 * ax - x, x, x < ax));
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let mode = round(f_param(0u));
    let bars = state.time * state.bpm / 240.0;
    let ax = f_param(1u) + f_param(2u) * sin(bars * 0.5 * TAU);
    let gain = f_param(3u);

    let b = layer.image_box;
    let size = max(b.zw - b.xy, vec2<f32>(1e-6));
    var l = (uv - b.xy) / size;
    if (l.x < 0.0 || l.y < 0.0 || l.x > 1.0 || l.y > 1.0) {
        return vec4<f32>(0.0);
    }
    if (mode < 0.5) {
        l.x = fold(l.x, ax);
    } else if (mode < 1.5) {
        l.y = fold(l.y, ax);
    } else if (mode < 2.5) {
        l = vec2<f32>(fold(l.x, ax), fold(l.y, ax));
    } else if (mode < 3.5) {
        l.x = 1.0 - l.x;
    } else {
        l.y = 1.0 - l.y;
    }
    return fx_in(b.xy + l * size) * (mask * gain);
}
