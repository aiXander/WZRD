// Built-in effect catalog. One shared pipeline; `layer.effect_id` switches.
// Each effect_<name> follows the (uv, mask) -> vec4<f32> contract user code
// also uses (D15 — built-ins are reference implementations, not the boundary).
//
// Contract reminder: the compositor blends additively (One + One), so every
// effect returns **premultiplied** RGBA: `vec4<f32>(rgb * intensity,
// intensity)` where `intensity` already includes the mask attenuation.
// Returning naked RGB dumps full-brightness light everywhere the shader
// runs, regardless of the layer's region.
//
// Effect IDs must stay in sync with `effects::builtin_id` in Rust.

fn effect_tint(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let c = c_param(0u);
    let a = clamp(c.a * mask, 0.0, 1.0);
    return vec4<f32>(c.rgb * a, a);
}

// 4-stop palette cycled by `phase` (f_param(0)). Phase typically driven by
// clock.bars(n) — the param value already lives in [0,1).
fn effect_hue_cycle(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let phase = f_param(0u);
    let p = fract(phase) * 4.0;
    let i = u32(floor(p)) % 4u;
    let next = (i + 1u) % 4u;
    let t = fract(p);
    let c0 = c_param(i);
    let c1 = c_param(next);
    let c = mix(c0, c1, t);
    let a = clamp(c.a * mask, 0.0, 1.0);
    return vec4<f32>(c.rgb * a, a);
}

// Additive flash: envelope (f_param(0), typically audio.onset) drives the
// brightness, base (f_param(1)) sets an always-on floor. Under additive
// blending the resulting light sums on top of whatever earlier layers wrote.
fn effect_flash(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let env = clamp(f_param(0u), 0.0, 1.0);
    let base = clamp(f_param(1u), 0.0, 1.0);
    let amt = clamp(base + env, 0.0, 1.0);
    let c = c_param(0u);
    let a = clamp(c.a * amt * mask, 0.0, 1.0);
    return vec4<f32>(c.rgb * a, a);
}

// UV wobble — samples the mask at a sinusoidally displaced UV so the
// region's edge undulates. amp = displacement in UV units, freq = spatial
// frequency, t = phase (typically clock.time or clock.phase).
fn effect_wobble(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let amp = f_param(0u);
    let freq = f_param(1u);
    let t = f_param(2u);
    let dx = sin(uv.y * freq + t * 2.0) * amp;
    let dy = cos(uv.x * freq + t * 2.0) * amp;
    let m = sample_mask(uv + vec2<f32>(dx, dy));
    let c = c_param(0u);
    let a = clamp(c.a * m, 0.0, 1.0);
    return vec4<f32>(c.rgb * a, a);
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    switch (layer.effect_id) {
        case 0u: { return effect_tint(uv, mask); }
        case 1u: { return effect_hue_cycle(uv, mask); }
        case 2u: { return effect_flash(uv, mask); }
        case 3u: { return effect_wobble(uv, mask); }
        default: {
            // Magenta = "unknown effect_id". Premultiplied so additive
            // blending stays sane even on the failure path.
            return vec4<f32>(mask, 0.0, mask, mask);
        }
    }
}
