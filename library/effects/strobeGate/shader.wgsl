// strobeGate — the clip gated on and off in time with the beat.
// Library clip effect (gig F1). Audio: drive `depth`.
//
// PHOTOSENSITIVITY: the audience lies on its back under this. The gate
// rate is capped in the shader at `limit` cycles per second (max 2.5:
// a 3 Hz gate already reads 3.5 flashes/s in check_frames). A `rate`/BPM
// combination above the cap halves until it fits, so it stays on the beat
// grid. Every layer gates in the
// same phase on purpose — desynchronised regions would multiply the
// flashes the room sees. Defaults: one soft-edged gate per two beats
// (1 Hz at 120 BPM) to 15 % light, not to black.

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    var rate = max(f_param(0u), 0.0);              // gate cycles per beat
    let duty = clamp(f_param(1u), 0.05, 0.95);
    let depth = clamp(f_param(2u), 0.0, 1.0);
    let soft = clamp(f_param(3u), 0.0, 0.5);
    let limit = clamp(f_param(4u), 0.1, 2.5);       // hard cap, cycles/s
    let gain = f_param(5u);

    let bps = state.bpm / 60.0;
    for (var i = 0; i < 12; i++) {
        if (rate * bps <= limit) {
            break;
        }
        rate *= 0.5;
    }
    let phase = fract(state.time * bps * rate);
    let s = max(soft * min(duty, 1.0 - duty), 1e-3);
    let open = smoothstep(0.0, s, phase) * (1.0 - smoothstep(duty - s, duty, phase));
    let level = mix(1.0 - depth, 1.0, open);
    return fx_in(uv) * (level * mask * gain);
}
