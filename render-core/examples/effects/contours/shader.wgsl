// Contours: topographic isolines of a slowly morphing height field, drawn
// at constant pixel width (distance = value / gradient), with a sonar sweep
// that lights the lines up once per bar. Reads the architecture as terrain.

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

fn height(p: vec2<f32>, t: f32) -> f32 {
    var q = p;
    var amp = 0.5;
    var sum = 0.0;
    for (var i = 0; i < 3; i = i + 1) {
        sum = sum + amp * vnoise(q + vec2<f32>(t, -0.7 * t));
        q = mat2x2<f32>(1.6, 1.2, -1.2, 1.6) * q;
        amp = amp * 0.5;
    }
    return sum;
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let scale  = clamp(f_param(0u), 0.5, 10.0);
    let lines  = clamp(f_param(1u), 2.0, 40.0);
    let speed  = clamp(f_param(2u), 0.0, 4.0);
    let bright = clamp(f_param(3u), 0.0, 3.0);
    let line_c  = c_param(0u);
    let sweep_c = c_param(1u);

    let aspect = state.resolution.x / max(state.resolution.y, 1.0);
    let p = vec2<f32>(uv.x * aspect, uv.y) * scale + layer_seed() * 9.0;
    let t = state.time * speed * 0.04;

    // value + numeric gradient → distance to the nearest isoline, in uv
    let e = 0.002 * scale;
    let h  = height(p, t) * lines;
    let hx = height(p + vec2<f32>(e, 0.0), t) * lines;
    let hy = height(p + vec2<f32>(0.0, e), t) * lines;
    let g = length(vec2<f32>(hx - h, hy - h)) / (e / scale);
    let d = abs(fract(h + 0.5) - 0.5) / max(g, 1e-3);

    let px = 1.0 / state.resolution.y;                 // one canvas pixel in uv
    let w = 1.1 * px;
    let line = 1.0 - smoothstep(w, w + 1.5 * px, d);
    let halo = exp(-d / (6.0 * px)) * 0.25;

    // sonar: a diagonal front crossing the region once per bar
    let b = layer_bbox();
    let lp = (uv - b.xy) / max(b.zw - b.xy, vec2<f32>(1e-4));
    let front = fract(state.bar_phase + layer_seed()) * 1.6 - 0.3;
    let sweep = exp(-abs((lp.x + lp.y) * 0.5 - front) * 14.0);
    let flare = sweep * (0.6 + 1.2 * state.onset_mid);

    let col = mix(line_c.rgb, sweep_c.rgb, clamp(flare, 0.0, 1.0));
    let intensity = (line * (0.45 + flare) + halo * flare) * bright;
    let a = clamp(intensity * mask, 0.0, 1.0);
    return vec4<f32>(col * a, a);
}
