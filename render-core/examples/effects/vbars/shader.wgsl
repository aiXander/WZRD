fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let speed = f_param(0u);        // scroll speed (bars/sec-ish)
    let count = clamp(f_param(1u), 1.0, 200.0); // number of bars across
    let duty  = clamp(f_param(2u), 0.02, 0.98); // bar width fraction
    let col   = c_param(0u);

    // scroll the pattern horizontally over time
    let x = uv.x * count - state.time * speed;
    let f = fract(x);
    // hard-edged vertical bar: 1.0 inside the bar, 0.0 in the gap
    let bar = step(f, duty);

    let a = clamp(bar * col.a * mask, 0.0, 1.0);
    return vec4<f32>(col.rgb * a, a);
}