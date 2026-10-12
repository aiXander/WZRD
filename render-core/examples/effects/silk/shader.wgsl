// Silk: twice domain-warped fbm read as thin glowing ridges — slow liquid
// filaments, like light caught in drifting silk. Big flat surfaces.
// Breathes with the bar; a low-band onset brightens the ridges.

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u.x),
               mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

fn fbm(p0: vec2<f32>) -> f32 {
    var p = p0;
    var amp = 0.5;
    var sum = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        sum = sum + amp * vnoise(p);
        p = mat2x2<f32>(1.6, 1.2, -1.2, 1.6) * p;
        amp = amp * 0.5;
    }
    return sum;
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let scale  = clamp(f_param(0u), 0.5, 8.0);
    let speed  = clamp(f_param(1u), 0.0, 4.0);
    let bright = clamp(f_param(2u), 0.0, 3.0);
    let ca = c_param(0u);
    let cb = c_param(1u);

    let aspect = state.resolution.x / max(state.resolution.y, 1.0);
    let p = vec2<f32>(uv.x * aspect, uv.y) * scale + layer_seed() * 17.0;
    let t = state.time * speed * 0.06;

    let q = vec2<f32>(fbm(p + vec2<f32>(0.0, t)), fbm(p + vec2<f32>(5.2, 1.3) - t));
    let r = vec2<f32>(fbm(p + 3.0 * q + vec2<f32>(1.7, 9.2) + 1.3 * t),
                      fbm(p + 3.0 * q + vec2<f32>(8.3, 2.8) - 0.9 * t));
    let n = fbm(p + 3.5 * r);

    // ridges: distance to the nearest iso-band centre, sharpened
    let band = abs(fract(n * 7.0) - 0.5) * 2.0;
    let ridge = pow(band, 10.0);
    let sheen = smoothstep(0.55, 0.95, n) * 0.25;

    let col = mix(ca.rgb, cb.rgb, smoothstep(0.35, 0.65, q.x + 0.3 * sin(state.time * speed * 0.2)));
    let pulse = 0.8 + 0.2 * cos(state.bar_phase * 6.28318) + 0.8 * state.onset_low;
    let intensity = (ridge + sheen) * bright * pulse;
    let a = clamp(intensity * mask, 0.0, 1.0);
    return vec4<f32>(col * a, a);
}
