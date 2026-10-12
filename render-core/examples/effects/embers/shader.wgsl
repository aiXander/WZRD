// Embers: sparks rising up the region in three depth layers, swaying and
// flickering, over a faint heat glow from the base. Tall surfaces.
// Density/rate in region-local space, so a tower and a wall both read
// right; the low band stokes the glow, a high-band onset flares the sparks.

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(hash21(p), hash21(p + vec2<f32>(37.3, 91.7)));
}

// One layer of sparks: one candidate per cell, ~40% of cells lit.
fn sparks(p: vec2<f32>, t: f32, seed: f32) -> f32 {
    let id = floor(p);
    let f = fract(p) - 0.5;
    let h = hash22(id + seed);
    let sway = 0.18 * sin(t * (0.8 + h.y) + h.x * 6.28318 + p.y * 0.7);
    let c = (h - 0.5) * 0.5 + vec2<f32>(sway, 0.0);
    let d = length((f - c) * vec2<f32>(1.0, 0.55)); // stretched: motion streak
    let size = 0.05 + 0.07 * h.y;
    let flick = 0.55 + 0.45 * sin(t * (4.0 + 6.0 * h.x) + h.y * 40.0);
    let lit = step(0.6, hash21(id + seed + 7.1));
    return lit * flick * (1.0 - smoothstep(0.0, size, d));
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let density = clamp(f_param(0u), 2.0, 40.0);
    let rise    = clamp(f_param(1u), 0.0, 4.0);
    let bright  = clamp(f_param(2u), 0.0, 3.0);
    let hot  = c_param(0u);
    let cool = c_param(1u);

    let b = layer_bbox();
    let ext = max(b.zw - b.xy, vec2<f32>(1e-4));
    let lp = (uv - b.xy) / ext;                       // 0..1 inside the region, y down
    let px = ext * state.resolution.xy;
    let rows_per_col = px.y / max(px.x, 1.0);
    let t = state.time;

    var s = 0.0;
    for (var i = 0; i < 3; i = i + 1) {
        let k = 1.0 + f32(i) * 0.75;                   // farther = smaller + slower
        let cols = density * k;
        let p = vec2<f32>(lp.x * cols, lp.y * cols * rows_per_col + t * rise * (1.6 / k) * 2.0);
        s = s + sparks(p, t, f32(i) * 13.7 + layer_seed() * 50.0) / k;
    }
    s = s * (1.0 + 1.5 * state.onset_high);

    // sparks fade as they climb; heat glow hugs the base
    let climb = mix(0.35, 1.0, lp.y);
    let glow = pow(lp.y, 6.0) * (0.1 + 0.35 * state.audio_low);

    let col = mix(cool.rgb, hot.rgb, clamp(s * 1.5, 0.0, 1.0));
    let intensity = (s * climb + glow) * bright;
    let a = clamp(intensity * mask, 0.0, 1.0);
    return vec4<f32>(col * a, a);
}
