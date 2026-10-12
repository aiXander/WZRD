// edit-churn variant A: a cheap animated gradient (the shader body is
// swapped A <-> B by effect.upsert every few frames).
fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let speed = clamp(f_param(0u), 0.0, 4.0);
    let v = 0.5 + 0.5 * sin(uv.x * 12.0 + state.time * speed);
    let a = clamp(v * mask * 0.6, 0.0, 1.0);
    return vec4<f32>(vec3<f32>(0.2, 0.6, 1.0) * a, a);
}
