// Example project-local user effect (D15).
//
// Slowly drifts a colour wash across the layer, modulated by an `amount`
// scalar (typically wired to an audio band). Demonstrates the user-effect
// contract:
//
//   fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32>
//
// `state` (FrameState) and `layer` (LayerParams) are bound from the prelude.
// `f_param(i)` / `c_param(i)` access slots in declaration order from
// `descriptor.json`. Saving this file triggers a hot pipeline rebuild.
//
// Output is premultiplied for additive blending: bake the mask + intensity
// into alpha and multiply rgb by that alpha. See the prelude header.

fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    let amount = clamp(f_param(0u), 0.0, 4.0);
    let base = c_param(0u);

    let n = sin(uv.x * 6.0 + state.time * 0.7)
          + sin(uv.y * 4.0 - state.time * 0.5)
          + sin((uv.x + uv.y) * 8.0 + state.bar_phase * 6.2832);
    let v = 0.5 + 0.25 * n;
    let intensity = clamp(v + amount * 0.4, 0.0, 1.5);

    let a = clamp(base.a * intensity * mask, 0.0, 1.0);
    return vec4<f32>(base.rgb * a, a);
}
