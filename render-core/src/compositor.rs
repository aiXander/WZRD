//! Compositor — turns a parsed scene + loaded pack into a linear pass plan
//! and executes it each frame (§3.6 "Internal compile step — pass plan").
//!
//! Phase 3 widens the Phase 2 plan: per-binding scalar drivers are evaluated
//! each frame, the shared `FrameState` uniform feeds clock + audio into every
//! pass, and project-local / inline WGSL effects each get their own pipeline
//! cached by the gpu layer. The render-thread per-frame work is still tiny —
//! evaluate drivers, write two uniform buffers per pass, draw three verts.
//!
//! Slow-path FBO routing (`layerRef`) is *not* implemented yet — Phase 3 was
//! scoped to "a few effects so the surface works"; deferred per §3.6 once a
//! real scene needs it.

use anyhow::{anyhow, Result};
use wgpu::util::DeviceExt;

use crate::drivers::{ScalarValue, Transport};
use crate::osc::AudioFeatures;
use crate::effects::{
    EffectBinding, EffectDef, EffectKind, EffectRegistry, InlineEffectSpec,
};
use crate::gpu::{
    FrameStateGpu, GpuContext, LayerParamsGpu, BUILTIN_PIPELINE_KEY,
};
use crate::pack::LoadedPack;
use crate::scene::{resolve_selector, BindingSpec, EffectRef, SceneFile};

/// A single pass to draw per frame. Owns its per-binding uniform buffer +
/// bind group; the pipeline lives in the gpu pipeline cache, keyed by
/// `pipeline_key`.
pub struct LayerPass {
    #[allow(dead_code)]
    pub binding_id: String,
    pub pipeline_key: String,
    pub effect_id: u32,
    pub slice: u32,
    pub z: i32,
    /// Resolved per-binding params. Re-evaluated each frame because some
    /// may be driver-bound (clock, audio).
    pub scalars: Vec<ScalarValue>,
    pub colors: Vec<[f32; 4]>,

    pub layer_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

/// Compiled pass plan. Rebuilt on every scene hot-reload.
pub struct PassPlan {
    pub layer_passes: Vec<LayerPass>,
    pub clear_color: wgpu::Color,
}

impl PassPlan {
    /// Build a fresh plan. `gpu` is `&mut` because new user-authored effects
    /// trigger pipeline compilation into the cache.
    ///
    /// On error the caller keeps the previous plan rendering (§3.6
    /// swap-on-success); see `main.rs::rebuild_plan`.
    pub fn build(
        gpu: &mut GpuContext,
        pack: &LoadedPack,
        scene: &SceneFile,
        registry: &EffectRegistry,
    ) -> Result<Self> {
        let mut passes: Vec<LayerPass> = Vec::new();

        for binding in &scene.bindings {
            let def = resolve_effect_def(binding, registry)?;
            let (pipeline_key, effect_id) = ensure_pipeline(gpu, &def)?;
            let resolved = EffectBinding::from_params(def, &binding.params)
                .map_err(|e| anyhow!("binding {:?}: {e:#}", binding.id))?;

            let slices = resolve_selector(&binding.select, pack)?;
            for slice in slices {
                let z = pack.manifest.layers[slice as usize].z;
                let initial = LayerParamsGpu::build(
                    slice,
                    effect_id,
                    &vec![0.0; resolved.scalars.len()],
                    &resolved.colors,
                );
                let layer_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!(
                        "layer params [{} slice {}]",
                        binding.id, slice
                    )),
                    contents: bytemuck::bytes_of(&initial),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("layer bg [{} slice {}]", binding.id, slice)),
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
                            resource: gpu.frame_state_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: layer_buffer.as_entire_binding(),
                        },
                    ],
                });

                passes.push(LayerPass {
                    binding_id: binding.id.clone(),
                    pipeline_key: pipeline_key.clone(),
                    effect_id,
                    slice,
                    z,
                    scalars: resolved.scalars.clone(),
                    colors: resolved.colors.clone(),
                    layer_buffer,
                    bind_group,
                });
            }
        }

        // Stable z-order; binding order breaks ties.
        passes.sort_by(|a, b| a.z.cmp(&b.z));

        Ok(Self {
            layer_passes: passes,
            clear_color: wgpu::Color::BLACK,
        })
    }

    /// Evaluate every driver-bound scalar against the current transport +
    /// audio state and write the result + per-binding colours into each
    /// pass's uniform buffer.
    pub fn tick(&self, gpu: &GpuContext, transport: &Transport, audio: &AudioFeatures) {
        let ctx = transport.frame_context(audio);
        let frame = FrameStateGpu {
            time: ctx.elapsed_sec,
            bar_phase: phase01(ctx.bar_time()),
            beat_phase: phase01(ctx.beat_time()),
            bpm: ctx.bpm,
            audio_low: audio.band(crate::osc::AudioBand::Low),
            audio_mid: audio.band(crate::osc::AudioBand::Mid),
            audio_high: audio.band(crate::osc::AudioBand::High),
            onset_low: audio.onset_envelope(crate::osc::AudioBand::Low, 0.18),
            onset_mid: audio.onset_envelope(crate::osc::AudioBand::Mid, 0.15),
            onset_high: audio.onset_envelope(crate::osc::AudioBand::High, 0.10),
            _pad0: 0.0,
            _pad1: 0.0,
            resolution: [
                gpu.composite_width as f32,
                gpu.composite_height as f32,
                0.0,
                0.0,
            ],
        };
        gpu.write_frame_state(&frame);

        // Per-pass params. Driver evaluation is cheap (a few muls + a few
        // atomic loads); we don't bother with dirty tracking.
        for pass in &self.layer_passes {
            let scalars: Vec<f32> = pass.scalars.iter().map(|s| s.eval(&ctx)).collect();
            let params = LayerParamsGpu::build(pass.slice, pass.effect_id, &scalars, &pass.colors);
            gpu.queue
                .write_buffer(&pass.layer_buffer, 0, bytemuck::bytes_of(&params));
        }
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
            // Avoid redundant pipeline switches when consecutive layers share one.
            let mut bound: Option<&str> = None;
            for layer in &self.layer_passes {
                let pipeline = match gpu.pipeline_cache.get(&layer.pipeline_key) {
                    Some(p) => p,
                    None => {
                        // Pipeline went missing (effect removed). Skip the
                        // pass; the previous frame's output stays on screen
                        // until the next plan rebuild.
                        continue;
                    }
                };
                if bound != Some(layer.pipeline_key.as_str()) {
                    pass.set_pipeline(pipeline);
                    bound = Some(layer.pipeline_key.as_str());
                }
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

fn resolve_effect_def(binding: &BindingSpec, registry: &EffectRegistry) -> Result<EffectDef> {
    match &binding.effect {
        EffectRef::Named(name) => registry.resolve_named(name),
        EffectRef::Inline(spec) => {
            // serde gives us a generic value; convert via the inline spec type.
            let inline_spec: InlineEffectSpec =
                serde_json::from_value(spec.clone()).map_err(|e| {
                    anyhow!(
                        "binding {:?}: invalid inline effect spec ({e})",
                        binding.id
                    )
                })?;
            registry.resolve_inline(&inline_spec)
        }
    }
}

/// Make sure the pipeline for this effect is compiled. Returns the cache
/// key + effect_id the pass should set in its uniform.
fn ensure_pipeline(gpu: &mut GpuContext, def: &EffectDef) -> Result<(String, u32)> {
    match &def.kind {
        EffectKind::BuiltIn { effect_id } => {
            Ok((BUILTIN_PIPELINE_KEY.to_string(), *effect_id))
        }
        EffectKind::User {
            pipeline_key,
            wgsl,
            source_path: _,
        } => {
            if !gpu.pipeline_cache.contains_key(pipeline_key) {
                gpu.upsert_user_pipeline(pipeline_key, wgsl)
                    .map_err(|e| anyhow!("compiling effect {:?}: {e:#}", def.name))?;
                log::info!("compiled user effect pipeline {:?}", pipeline_key);
            }
            // effect_id is irrelevant for user shaders — the user effect()
            // function has its own behaviour. We still pass 0 so the uniform
            // has a defined value.
            Ok((pipeline_key.clone(), 0))
        }
    }
}

fn phase01(t: f32) -> f32 {
    let mut x = t.fract();
    if x < 0.0 {
        x += 1.0;
    }
    x
}
