// lumaKey — keep only the clip's bright parts (or only its dark parts).
// Library clip effect (gig F1). Audio: drive `threshold`.
//
// Keys on gamma-encoded luma of the straight colour with a soft ramp.
// Keyed-out pixels are zero light, so this also kills a clip's lifted
// dark floor (the haze check_frames warns about). `invert` = 1 keeps the
// dark parts instead.

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let threshold = f_param(0u);
    let soft = max(f_param(1u), 1e-3);
    let invert = f_param(2u) > 0.5;
    let gain = f_param(3u);

    let c = fx_in(uv);
    let straight = c.rgb / max(c.a, 1e-4);
    let l = sqrt(max(dot(straight, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0));
    var key = smoothstep(threshold, threshold + soft, l);
    if (invert) {
        key = 1.0 - key;
    }
    return c * (key * mask * gain);
}
