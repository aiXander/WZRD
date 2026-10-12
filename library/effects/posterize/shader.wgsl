// posterize — quantize the clip's colour into a few flat bands.
// Library clip effect (gig F1). Audio: drive `levels` (fewer on a hit).
//
// Quantizes straight (un-premultiplied) colour in a gamma-encoded space so
// the bands are perceptually even, then re-premultiplies. Black stays
// black: band 0 of every channel is zero light.

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let levels = max(round(f_param(0u)), 2.0);
    let gamma = max(f_param(1u), 0.05);
    let amount = clamp(f_param(2u), 0.0, 1.0);
    let gain = f_param(3u);

    let c = fx_in(uv);
    let straight = c.rgb / max(c.a, 1e-4);
    let enc = pow(clamp(straight, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / (2.2 * gamma)));
    let steps = levels - 1.0;
    let q = floor(enc * steps + 0.5) / steps;
    let lin = pow(q, vec3<f32>(2.2 * gamma));
    let rgb = mix(straight, lin, amount) * c.a;
    return vec4<f32>(rgb, c.a) * (mask * gain);
}
