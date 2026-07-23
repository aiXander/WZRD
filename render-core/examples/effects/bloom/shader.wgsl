fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let rate  = clamp(f_param(0u), 0.0, 4.0);
    let glow  = clamp(f_param(1u), 0.0, 3.0);
    let core  = c_param(0u);
    let edge  = c_param(1u);

    // center of the layer's local uv space
    let c = vec2<f32>(0.5, 0.5);
    let d = uv - c;
    let dist = length(d);
    let angle = atan2(d.y, d.x);

    // blooming pulse: a radius that expands outward, re-seeding each bar
    let phase = fract(state.bar_phase + state.time * rate * 0.15);
    let bloom_r = phase * 0.9;

    // flower-like petal lobes that breathe over time
    let petals = 6.0;
    let lobe = 0.06 * sin(angle * petals + state.time * 0.6);
    let ring = bloom_r + lobe;

    // soft glowing shell around the expanding radius
    let edge_glow = exp(-pow((dist - ring) * 14.0, 2.0));
    // bright core that fades as the bloom opens
    let core_glow = exp(-dist * 6.0) * (1.0 - phase);

    let v = edge_glow * (0.6 + 0.4 * sin(angle * petals)) + core_glow;
    let intensity = clamp(v * glow, 0.0, 1.5);

    // colour ramps from magenta edge into a warm core
    let t = clamp(core_glow + edge_glow * 0.5, 0.0, 1.0);
    let col = mix(edge.rgb, core.rgb, t);

    let a = clamp(intensity * mask, 0.0, 1.0);
    return vec4<f32>(col * a, a);
}