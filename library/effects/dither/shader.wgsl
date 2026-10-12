// dither — ordered (Bayer 8×8) dither into a few levels, retro-print look.
// Library clip effect (gig F1, Resolume's Dither). Audio: drive `size`.
//
// The clip is read once per `size`-pixel cell (anchored on the clip box)
// and quantized per channel (or on luma with `mono` = 1) against the
// Bayer threshold, in gamma-encoded straight colour. Level 0 is zero
// light, so dark areas go fully black — good for projection.

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

// Bayer 8×8 index (0..63) by bit interleaving, as a threshold in (0, 1).
fn bayer8(p: vec2<u32>) -> f32 {
    let x = p.x & 7u;
    let y = p.y & 7u;
    let z = x ^ y;
    let v = ((z & 1u) << 5u) | ((y & 1u) << 4u) | ((z & 2u) << 2u)
          | ((y & 2u) << 1u) | ((z & 4u) >> 1u) | ((y & 4u) >> 2u);
    return (f32(v) + 0.5) / 64.0;
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let size = max(f_param(0u), 1.0);   // output pixels per dither cell
    let steps = max(round(f_param(1u)), 2.0) - 1.0;
    let mono = f_param(2u) > 0.5;
    let gain = f_param(3u);

    let res = max(state.resolution.xy, vec2<f32>(1.0));
    let origin = layer.image_box.xy;
    let cell = floor((uv - origin) * res / size);
    let c = fx_in(origin + (cell + 0.5) * size / res);
    let thr = bayer8(vec2<u32>(vec2<i32>(cell) & vec2<i32>(7)));

    let straight = c.rgb / max(c.a, 1e-4);
    var enc = sqrt(clamp(straight, vec3<f32>(0.0), vec3<f32>(1.0)));
    if (mono) {
        enc = vec3<f32>(dot(enc, vec3<f32>(0.299, 0.587, 0.114)));
    }
    let q = clamp(floor(enc * steps + thr) / steps, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(q * q * c.a, c.a) * (mask * gain);
}
