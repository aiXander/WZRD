// noiseDisplace — the clip pushed around by drifting smooth noise.
// Library clip effect (gig F1, Resolume's Displace). Audio: drive `amount`.
//
// Two decorrelated value-noise fields (x, y) over aspect-correct canvas
// space, scrolling in time. Displaced reads are clamped into the clip box
// (no black seams at the box edge); pixels outside the box stay dark.

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

fn in_box(uv: vec2<f32>) -> bool {
    let b = layer.image_box;
    return all(uv >= b.xy) && all(uv <= b.zw);
}

fn clamp_box(uv: vec2<f32>) -> vec2<f32> {
    let b = layer.image_box;
    return clamp(uv, b.xy, b.zw);
}

fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.zyx + 31.32);
    return fract((q.x + q.y) * q.z);
}

// Trilinear value noise in [-1, 1].
fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash3(i);
    let n100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let z0 = mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y);
    let z1 = mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y);
    return mix(z0, z1, u.z) * 2.0 - 1.0;
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let amount = f_param(0u);   // uv units of canvas height
    let scale = f_param(1u);    // noise cells per canvas height
    let speed = f_param(2u);    // noise cells per second
    let gain = f_param(3u);
    if (!in_box(uv)) {
        return vec4<f32>(0.0);
    }

    let aspect = state.resolution.x / max(state.resolution.y, 1.0);
    let p = vec3<f32>(uv * vec2<f32>(aspect, 1.0) * scale, state.time * speed + layer_seed() * 17.0);
    let d = vec2<f32>(vnoise(p), vnoise(p + vec3<f32>(41.7, 13.3, 7.1)));
    let o = d * amount * vec2<f32>(1.0 / aspect, 1.0);
    return fx_in(clamp_box(uv + o)) * (mask * gain);
}
