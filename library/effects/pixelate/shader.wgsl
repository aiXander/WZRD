// pixelate — big square pixels, optionally rounded into LED dots.
// Library clip effect (gig F1). Audio: drive `size`.
//
// The grid is anchored at the clip box's corner in output pixels, so
// blocks stay put when the box moves. `dot` > 0 shrinks each block to a
// round dot with dark gaps between (reads well on fabric).

// The one place this effect reads its clip. Post-gig chain re-cut: swap
// `sample_image` for `sample_prev` here and nowhere else.
fn fx_in(uv: vec2<f32>) -> vec4<f32> {
    return sample_image(uv);
}

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let size = max(f_param(0u), 1.0);   // output pixels
    let dot_amt = clamp(f_param(1u), 0.0, 1.0);
    let gain = f_param(2u);

    let res = max(state.resolution.xy, vec2<f32>(1.0));
    let origin = layer.image_box.xy;
    let p = (uv - origin) * res / size;
    let cell = floor(p);
    let c = fx_in(origin + (cell + 0.5) * size / res);

    // Dot shape: radius shrinks from the cell's corner (0) to a circle
    // inscribed with a gap (1); soft edge one output pixel wide.
    let d = length(fract(p) - 0.5) * size;
    let radius = mix(size, 0.42 * size, dot_amt);
    let shape = 1.0 - smoothstep(radius - 1.0, radius, d);
    return c * (shape * mask * gain);
}
