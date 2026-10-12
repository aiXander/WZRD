// edit-churn variant B: same cost class as A, different body, so every
// upsert really recompiles a pipeline.
fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let speed = clamp(f_param(0u), 0.0, 4.0);
    let v = 0.5 + 0.5 * cos(uv.y * 10.0 - state.time * speed);
    let a = clamp(v * mask * 0.6, 0.0, 1.0);
    return vec4<f32>(vec3<f32>(1.0, 0.4, 0.2) * a, a);
}
