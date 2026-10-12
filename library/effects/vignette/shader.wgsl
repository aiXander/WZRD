// vignette — the clip fades out toward its own box edges.
// Library clip effect (gig F1, Resolume's Vignette). Audio: drive `radius`.
//
// Distance from the clip-box centre, normalised so 1 = the box edge;
// `roundness` blends a rectangular falloff (0) into an elliptical one (1).
// Softens hard region edges on fabric; with audio on `radius` the light
// breathes out from the centre.

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let radius = f_param(0u);
    let soft = max(f_param(1u), 1e-3);
    let roundness = clamp(f_param(2u), 0.0, 1.0);
    let gain = f_param(3u);

    let b = layer.image_box;
    let size = max(b.zw - b.xy, vec2<f32>(1e-6));
    let p = abs((uv - b.xy) / size - 0.5) * 2.0;
    let d = mix(max(p.x, p.y), length(p), roundness);
    let fade = 1.0 - smoothstep(radius - soft, radius, d);
    return fx_in(uv) * (fade * mask * gain);
}
