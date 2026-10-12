// kaleido — mirrored-wedge kaleidoscope around the clip's centre.
// Library clip effect (gig F1). Audio: drive `zoom` or `twist`.
//
// Polar fold in the clip box's own (aspect-correct) space: the angle is
// folded into one mirrored wedge, `twist` bends the wedges into a spiral
// (Resolume's Bendoscope), and anything that lands outside the box
// mirror-repeats back in, so the fold never opens a hole.

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

const TAU: f32 = 6.2831853;

// Mirrored repeat: 0→0, 1→1, 2→0, …
fn tri(x: vec2<f32>) -> vec2<f32> {
    return 1.0 - abs(fract(x * 0.5) * 2.0 - 1.0);
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let segments = max(round(f_param(0u)), 1.0);
    let spin = f_param(1u);     // turns per bar
    let zoom = max(f_param(2u), 0.05);
    let twist = f_param(3u);    // radians per box-height of radius
    let gain = f_param(4u);

    let b = layer.image_box;
    let size = max(b.zw - b.xy, vec2<f32>(1e-6));
    let aspect = size.x * state.resolution.x / max(size.y * state.resolution.y, 1.0);
    let local = (uv - b.xy) / size;

    var p = (local - 0.5) * vec2<f32>(aspect, 1.0);
    let r = length(p) / zoom;
    let bars = state.time * state.bpm / 240.0;
    var a = atan2(p.y, p.x) + bars * spin * TAU + twist * r;
    let wedge = TAU / segments;
    a = abs(fract(a / wedge) * wedge - 0.5 * wedge);
    p = vec2<f32>(cos(a), sin(a)) * r;

    let src = b.xy + tri(p / vec2<f32>(aspect, 1.0) + 0.5) * size;
    return fx_in(src) * (mask * gain);
}
