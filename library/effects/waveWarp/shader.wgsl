// waveWarp — travelling sine waves ripple the clip sideways.
// Library clip effect (gig F1, Resolume's Wave Warp). Audio: drive `amp`.
//
// Displacement perpendicular to the wave fronts; `angle` turns the fronts
// (0 = horizontal fronts pushing pixels left/right). Displaced reads are
// clamped into the clip box; pixels outside the box stay dark.

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
    let amp = f_param(0u);      // uv units of canvas height
    let freq = f_param(1u);     // waves per canvas height
    let speed = f_param(2u);    // waves per bar
    let angle = f_param(3u) * TAU;
    let gain = f_param(4u);
    if (!in_box(uv)) {
        return vec4<f32>(0.0);
    }

    let aspect = state.resolution.x / max(state.resolution.y, 1.0);
    let p = uv * vec2<f32>(aspect, 1.0);
    let along = vec2<f32>(-sin(angle), cos(angle));   // across the fronts
    let push = vec2<f32>(cos(angle), sin(angle));     // along the fronts
    let bars = state.time * state.bpm / 240.0;
    let w = sin((dot(p, along) * freq - bars * speed) * TAU);
    let o = push * (w * amp) * vec2<f32>(1.0 / aspect, 1.0);
    return fx_in(clamp_box(uv + o)) * (mask * gain);
}
