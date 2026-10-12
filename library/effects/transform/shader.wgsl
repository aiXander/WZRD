// transform — zoom, rotate, spin and pan the clip inside its box.
// Library clip effect (gig F1, Resolume's Transform). Audio: drive `zoom`.
//
// Works in the clip box's aspect-correct space around its centre. With
// `tile` = 1 the clip mirror-repeats past its edges (zooming out fills the
// region); with 0 it stays a single copy and the rest is dark.

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

const TAU: f32 = 6.2831853;

fn tri(x: vec2<f32>) -> vec2<f32> {
    return 1.0 - abs(fract(x * 0.5) * 2.0 - 1.0);
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let zoom = max(f_param(0u), 0.05);
    let rotate = f_param(1u);   // turns
    let spin = f_param(2u);     // turns per bar
    let pan = vec2<f32>(f_param(3u), f_param(4u));
    let tile = f_param(5u) > 0.5;
    let gain = f_param(6u);

    let b = layer.image_box;
    let size = max(b.zw - b.xy, vec2<f32>(1e-6));
    let aspect = size.x * state.resolution.x / max(size.y * state.resolution.y, 1.0);
    let local = (uv - b.xy) / size;

    let bars = state.time * state.bpm / 240.0;
    let a = (rotate + bars * spin) * TAU;
    let p = (local - 0.5) * vec2<f32>(aspect, 1.0) / zoom;
    let r = vec2<f32>(p.x * cos(a) + p.y * sin(a), -p.x * sin(a) + p.y * cos(a));
    var l = r / vec2<f32>(aspect, 1.0) + 0.5 - pan;
    if (tile) {
        l = tri(l);
    }
    return fx_in(b.xy + l * size) * (mask * gain);
}
