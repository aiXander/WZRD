// heat — thermal-camera false colour from the clip's brightness.
// Library clip effect (gig F1, Resolume's Heat). Audio: drive `offset`.
//
// Luma (gamma-encoded) → black · indigo · magenta · orange · yellow ·
// white. `offset` and `cycle` scroll the palette; the result is scaled by
// a ramp on the source luma so dark stays dark whatever the palette does.

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

fn palette(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0) * 5.0;
    let k0 = vec3<f32>(0.0, 0.0, 0.0);
    let k1 = vec3<f32>(0.06, 0.0, 0.35);
    let k2 = vec3<f32>(0.6, 0.0, 0.55);
    let k3 = vec3<f32>(1.0, 0.25, 0.05);
    let k4 = vec3<f32>(1.0, 0.85, 0.1);
    let k5 = vec3<f32>(1.0, 1.0, 1.0);
    if (x < 1.0) { return mix(k0, k1, x); }
    if (x < 2.0) { return mix(k1, k2, x - 1.0); }
    if (x < 3.0) { return mix(k2, k3, x - 2.0); }
    if (x < 4.0) { return mix(k3, k4, x - 3.0); }
    return mix(k4, k5, x - 4.0);
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let contrast = f_param(0u);
    let offset = f_param(1u);
    let cycle = f_param(2u);    // palette scrolls per bar
    let amount = clamp(f_param(3u), 0.0, 1.0);
    let gain = f_param(4u);

    let c = fx_in(uv);
    let straight = c.rgb / max(c.a, 1e-4);
    let l = sqrt(max(dot(straight, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0));
    let bars = state.time * state.bpm / 240.0;
    // Mirrored repeat (0→1→0 over 2) scrolls with no hard wrap seam and is
    // the identity on [0, 1], so offset 0 / cycle 0 is the plain ramp.
    let t = (l - 0.5) * contrast + 0.5 + offset + bars * cycle;
    let hot = palette(1.0 - abs(fract(t * 0.5) * 2.0 - 1.0));
    let lit = smoothstep(0.02, 0.2, l);
    let rgb = mix(straight, hot * lit, amount) * c.a;
    return vec4<f32>(rgb, c.a) * (mask * gain);
}
