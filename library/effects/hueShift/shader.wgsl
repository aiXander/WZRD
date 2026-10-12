// hueShift — rotate the clip's hues (static offset + slow cycle) and set saturation.
// Library clip effect (gig F1, Resolume's Saturation folded in). Audio: drive `shift`.
//
// A rotation around the grey axis in YIQ — linear, so it works on
// premultiplied colour directly and black stays black. Result clamped
// back into [0, alpha].

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

const TAU: f32 = 6.2831853;

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let shift = f_param(0u);    // turns
    let speed = f_param(1u);    // turns per bar
    let sat = max(f_param(2u), 0.0);
    let gain = f_param(3u);

    let c = fx_in(uv);
    let bars = state.time * state.bpm / 240.0;
    let a = (shift + bars * speed) * TAU;

    let y = dot(c.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let i = dot(c.rgb, vec3<f32>(0.596, -0.274, -0.322));
    let q = dot(c.rgb, vec3<f32>(0.211, -0.523, 0.312));
    let ca = cos(a);
    let sa = sin(a);
    let i2 = (i * ca - q * sa) * sat;
    let q2 = (i * sa + q * ca) * sat;
    let rgb = vec3<f32>(
        y + 0.956 * i2 + 0.621 * q2,
        y - 0.272 * i2 - 0.647 * q2,
        y - 1.106 * i2 + 1.703 * q2,
    );
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(c.a)), c.a) * (mask * gain);
}
