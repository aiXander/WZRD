// Scanwave: a stack of oscilloscope traces running along the region —
// interfering sines with a soft phosphor glow and a colour drift along x.
// Built for long horizontal strips; the mid band swells the amplitude.

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let count  = clamp(f_param(0u), 1.0, 12.0);
    let amp    = clamp(f_param(1u), 0.0, 1.0);
    let speed  = clamp(f_param(2u), 0.0, 4.0);
    let bright = clamp(f_param(3u), 0.0, 3.0);
    let ca = c_param(0u);
    let cb = c_param(1u);

    let b = layer_bbox();
    let ext = max(b.zw - b.xy, vec2<f32>(1e-4));
    let lp = (uv - b.xy) / ext;
    let h_px = ext.y * state.resolution.y;              // region height in pixels
    let t = state.time * speed;
    let a_mod = amp * (0.6 + 0.8 * state.audio_mid);

    let n = i32(count);
    var glow = 0.0;
    var core = 0.0;
    for (var i = 0; i < 12; i = i + 1) {
        if (i >= n) { break; }
        let fi = f32(i);
        let base = (fi + 0.5) / count;
        let x = lp.x * 6.28318;
        let y = base + a_mod / count * 1.4 *
            sin(x * (2.0 + fi * 0.37) + t * (0.9 + 0.13 * fi) + fi) *
            sin(x * 0.7 - t * 0.31 + fi * 2.1);
        let d = abs(lp.y - y) * h_px;                  // pixels to this trace
        core = core + 1.0 - smoothstep(0.6, 2.0, d);
        glow = glow + exp(-d / 9.0) * 0.35;
    }

    let hue = 0.5 + 0.5 * sin(lp.x * 3.0 - state.time * 0.4 + state.bar_phase * 6.28318);
    let col = mix(ca.rgb, cb.rgb, hue);
    let pulse = 0.85 + 0.15 * cos(state.beat_phase * 6.28318);
    let intensity = (core + glow) * bright * pulse;
    let a = clamp(intensity * mask, 0.0, 1.0);
    return vec4<f32>(col * a, a);
}
