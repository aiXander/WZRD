fn hash2(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i + vec2<f32>(0.0, 0.0));
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p0: vec2<f32>) -> f32 {
    var p = p0;
    var amp = 0.5;
    var sum = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        sum = sum + amp * vnoise(p);
        p = p * 2.0;
        amp = amp * 0.5;
    }
    return sum;
}

fn palette(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.00, 0.33, 0.67);
    return a + b * cos(6.28318 * (c * t + d));
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let scale  = clamp(f_param(0u), 0.5, 8.0);
    let speed  = clamp(f_param(1u), 0.0, 4.0);
    let bright = clamp(f_param(2u), 0.0, 3.0);
    let tintc  = c_param(0u);

    let t = state.time * speed * 0.15;
    let p = uv * scale;

    // one level of domain warp for that liquid, flowing look
    let q = vec2<f32>(fbm(p + vec2<f32>(0.0, t)), fbm(p + vec2<f32>(5.2, 1.3) - t));
    let n = fbm(p + 4.0 * q);

    let col = palette(n + t * 0.2) * tintc.rgb;
    let intensity = clamp(n * bright, 0.0, 1.5);
    let a = clamp(intensity * mask, 0.0, 1.0);
    return vec4<f32>(col * a, a);
}