//! Compositor — turns a parsed scene + loaded pack into a linear pass plan
//! and executes it each frame (§3.6 "Internal compile step — pass plan").
//!
//! Phase 2's plan is intentionally tiny: every binding fans out to one pass
//! per selected layer slice, sorted by the layer's z value. The final
//! homography pass copies the composite onto the swapchain (D9). Slow-path
//! FBO routing, layerRefs, and inline-WGSL all arrive in Phase 3.

use anyhow::Result;
use wgpu::util::DeviceExt;

use crate::effects::{EffectInstance, EffectParams};
use crate::gpu::{GpuContext, LayerUniforms};
use crate::pack::LoadedPack;
use crate::scene::{resolve_selector, BindingSpec, EffectRef, SceneFile};

/// A single pass to draw per frame.
pub struct LayerPass {
    /// Binding id — used by hot-reload diff and per-pass telemetry.
    #[allow(dead_code)]
    pub binding_id: String,
    /// Slice index into the mask atlas. Phase 2 leans on the bound uniform
    /// buffer for the actual draw; this is for debug / inspection.
    #[allow(dead_code)]
    pub slice: u32,
    pub z: i32,
    /// Retained so the bind group's reference stays live for the frame.
    #[allow(dead_code)]
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

/// Compiled pass plan. Rebuilt on every scene hot-reload.
pub struct PassPlan {
    pub layer_passes: Vec<LayerPass>,
    pub clear_color: wgpu::Color,
}

impl PassPlan {
    /// Build a fresh plan. `gpu` and `pack` must outlive the plan (the bind
    /// groups reference their resources).
    pub fn build(gpu: &GpuContext, pack: &LoadedPack, scene: &SceneFile) -> Result<Self> {
        let mut passes: Vec<LayerPass> = Vec::new();

        for binding in &scene.bindings {
            let effect = effect_from_binding(binding)?;
            let slices = resolve_selector(&binding.select, pack)?;
            for slice in slices {
                let z = pack.manifest.layers[slice as usize].z;
                let uniforms = build_layer_uniforms(&effect, slice);
                let uniform_buffer =
                    gpu.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!(
                                "layer uniform [{} slice {}]",
                                binding.id, slice
                            )),
                            contents: bytemuck::bytes_of(&uniforms),
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        });
                let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("layer bind group [{} slice {}]", binding.id, slice)),
                    layout: &gpu.layer_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&gpu.mask_atlas_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&gpu.mask_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                    ],
                });
                passes.push(LayerPass {
                    binding_id: binding.id.clone(),
                    slice,
                    z,
                    uniform_buffer,
                    bind_group,
                });
            }
        }

        // Stable z-order; binding order breaks ties so the scene file's order
        // is the visible secondary sort.
        passes.sort_by(|a, b| a.z.cmp(&b.z));

        Ok(Self {
            layer_passes: passes,
            clear_color: wgpu::Color::BLACK,
        })
    }

    pub fn record_and_submit(&self, gpu: &GpuContext) -> Result<(), wgpu::SurfaceError> {
        let frame = gpu.surface.get_current_texture()?;
        let swap_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });

        // 1) Per-layer passes into the composite buffer.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gpu.composite_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.layer_pipeline);
            for layer in &self.layer_passes {
                pass.set_bind_group(0, &layer.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // 2) Final homography pass onto the swapchain.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("homography pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swap_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.homography_pipeline);
            pass.set_bind_group(0, &gpu.homography_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

fn effect_from_binding(binding: &BindingSpec) -> Result<EffectInstance> {
    let name = match &binding.effect {
        EffectRef::Named(s) => s.as_str(),
        EffectRef::Inline(_) => anyhow::bail!(
            "inline WGSL effects are a Phase 3 feature (binding {:?})",
            binding.id
        ),
    };
    EffectInstance::from_spec(name, &binding.params)
}

fn build_layer_uniforms(effect: &EffectInstance, slice: u32) -> LayerUniforms {
    let color = match &effect.params {
        EffectParams::Tint { color } => *color,
    };
    LayerUniforms {
        color,
        slice,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    }
}
