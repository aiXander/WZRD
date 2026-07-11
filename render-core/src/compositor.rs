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

use crate::drivers::{FrameContext, ParamOverrides, PickRate, ScalarValue};
use crate::telemetry::DriverRow;
use crate::effects::{
    EffectBinding, EffectDef, EffectKind, EffectRegistry, InlineEffectSpec,
};
use crate::gpu::{
    FrameStateGpu, GpuContext, LayerIdentity, LayerParamsGpu, BUILTIN_PIPELINE_KEY,
};
use crate::pack::LoadedPack;
use crate::scene::{resolve_selector, BindingSpec, EffectRef, PickMode, SceneFile};

/// A single pass to draw per frame. Owns its per-binding uniform buffer +
/// bind group; the pipeline lives in the gpu pipeline cache, keyed by
/// `pipeline_key`.
pub struct LayerPass {
    pub binding_id: String,
    pub pipeline_key: String,
    pub effect_id: u32,
    pub slice: u32,
    pub z: i32,
    /// Resolved per-binding params. Re-evaluated each frame because some
    /// may be driver-bound (clock, audio).
    pub scalars: Vec<ScalarValue>,
    /// Declared names for `scalars`, parallel by index. Used by the
    /// `drivers` telemetry snapshot so the UI can label live values.
    pub scalar_names: Vec<String>,
    pub colors: Vec<[f32; 4]>,
    /// How many mask slices this pass's parent binding resolved to (the
    /// "affects" count surfaced in the driver rack).
    pub sibling_count: u32,
    /// §5.2 per-layer identity, baked into the uniform every tick.
    pub identity: LayerIdentity,
    /// Index into `PassPlan::picks` when the parent binding has a `pick`
    /// selector; `None` = always drawn.
    pub pick_group: Option<usize>,
    /// Whether this pass draws this frame. Always `true` for un-picked
    /// passes; toggled by the pick machinery in [`PassPlan::tick`].
    pub active: bool,

    pub layer_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

/// §5.2 pick state for one binding. All member passes stay in the plan;
/// a re-pick only flips `active` flags — no plan rebuild, no GPU work.
/// The choice is `hash(binding_id, cycle) % members`: a pure function of
/// transport time, so runs are deterministic and the §5.6 design leg picks
/// the same layer its promote will.
pub struct PickGroup {
    /// For diagnostics only — the choice hashes `binding_seed`, not this.
    binding_id: String,
    binding_seed: u32,
    /// `None` = `random_static` (cycle pinned to 0, picked on first tick).
    rate: Option<PickRate>,
    /// Indices into `layer_passes` (post z-sort), in resolved-selection order.
    pass_indices: Vec<usize>,
    last_cycle: Option<u64>,
}

/// Compiled pass plan. Rebuilt on every scene hot-reload.
pub struct PassPlan {
    pub layer_passes: Vec<LayerPass>,
    pub picks: Vec<PickGroup>,
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
        let mut picks: Vec<PickGroup> = Vec::new();

        for binding in &scene.bindings {
            let def = resolve_effect_def(binding, registry)?;
            let (pipeline_key, effect_id) = ensure_pipeline(gpu, &def)?;
            let resolved = EffectBinding::from_params(def, &binding.params)
                .map_err(|e| anyhow!("binding {:?}: {e:#}", binding.id))?;

            let scalar_names: Vec<String> = resolved
                .def
                .inputs
                .iter()
                .filter_map(|i| match i {
                    crate::effects::InputSlot::Scalar { name, .. } => Some(name.clone()),
                    crate::effects::InputSlot::Color { .. } => None,
                })
                .collect();

            let slices = resolve_selector(&binding.select, pack)?;
            let sibling_count = slices.len() as u32;

            // §5.2 pick — one group per picked binding. Members start
            // inactive; the first tick() runs the initial pick.
            let pick_group = match &binding.select.pick {
                Some(pick) => {
                    let rate = match pick.mode {
                        PickMode::RandomEach => {
                            // rate presence is enforced at scene parse.
                            let raw = pick.rate.as_ref().ok_or_else(|| {
                                anyhow!("binding {:?}: pick.rate missing", binding.id)
                            })?;
                            Some(PickRate::parse(raw).map_err(|e| {
                                anyhow!("binding {:?}: {e:#}", binding.id)
                            })?)
                        }
                        PickMode::RandomStatic => None,
                    };
                    picks.push(PickGroup {
                        binding_id: binding.id.clone(),
                        binding_seed: fnv1a(&binding.id),
                        rate,
                        pass_indices: Vec::new(),
                        last_cycle: None,
                    });
                    Some(picks.len() - 1)
                }
                None => None,
            };

            for (layer_index, slice) in slices.iter().copied().enumerate() {
                let z = pack.manifest.layers[slice as usize].z;
                let geom = pack.geoms[slice as usize];
                let identity = LayerIdentity {
                    layer_seed: seed01(&pack.manifest.layers[slice as usize].id),
                    layer_index: layer_index as u32,
                    layer_count: sibling_count,
                    centroid_uv: geom.centroid_uv,
                    bbox_uv: geom.bbox_uv,
                };
                let initial = LayerParamsGpu::build(
                    slice,
                    effect_id,
                    &identity,
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
                    scalar_names: scalar_names.clone(),
                    colors: resolved.colors.clone(),
                    sibling_count,
                    identity,
                    pick_group,
                    // Picked passes start dark until the first tick() picks.
                    active: pick_group.is_none(),
                    layer_buffer,
                    bind_group,
                });
            }
        }

        // Stable z-order; binding order breaks ties.
        passes.sort_by(|a, b| a.z.cmp(&b.z));

        // Pick groups address passes by post-sort index. The stable sort
        // keeps member order deterministic, which the pick hash relies on.
        for (i, pass) in passes.iter().enumerate() {
            if let Some(g) = pass.pick_group {
                picks[g].pass_indices.push(i);
            }
        }

        Ok(Self {
            layer_passes: passes,
            picks,
            clear_color: wgpu::Color::BLACK,
        })
    }

    /// Evaluate every driver-bound scalar against the current transport +
    /// audio state and write the result + per-binding colours into each
    /// pass's uniform buffer. `&mut` because §5.2 pick state (active flags,
    /// cycle counters) advances here. The §5.5 override table wins over the
    /// scene-authored value wherever a (binding, param) entry exists.
    pub fn tick(&mut self, gpu: &GpuContext, ctx: &FrameContext, overrides: &ParamOverrides) {
        // §5.2 picks — re-roll when a group's cycle counter changes. The
        // choice is stateless (pure hash of binding id + cycle), so there's
        // no RNG to seed or persist.
        let mut changes: Vec<(usize, usize)> = Vec::new();
        for (gi, group) in self.picks.iter_mut().enumerate() {
            let cycle = group.rate.as_ref().map(|r| r.cycle(ctx)).unwrap_or(0);
            if group.last_cycle != Some(cycle) {
                group.last_cycle = Some(cycle);
                changes.push((gi, pick_choice(group.binding_seed, cycle, group.pass_indices.len())));
            }
        }
        for (gi, choice) in changes {
            let indices = std::mem::take(&mut self.picks[gi].pass_indices);
            for (pos, &pi) in indices.iter().enumerate() {
                self.layer_passes[pi].active = pos == choice;
            }
            if let Some(&pi) = indices.get(choice) {
                log::debug!(
                    "pick[{}]: cycle {:?} -> layer {:?} ({}/{})",
                    self.picks[gi].binding_id,
                    self.picks[gi].last_cycle,
                    self.layer_passes[pi].slice,
                    choice + 1,
                    indices.len()
                );
            }
            self.picks[gi].pass_indices = indices;
        }
        // Audio reads go through `ctx` so the §5.4 audio-listen master scales
        // the FrameState uniform (what user WGSL sees) and the drivers alike.
        let frame = FrameStateGpu {
            time: ctx.elapsed_sec,
            bar_phase: phase01(ctx.bar_time()),
            beat_phase: phase01(ctx.beat_time()),
            bpm: ctx.bpm,
            audio_low: ctx.band(crate::osc::AudioBand::Low),
            audio_mid: ctx.band(crate::osc::AudioBand::Mid),
            audio_high: ctx.band(crate::osc::AudioBand::High),
            onset_low: ctx.onset(crate::osc::AudioBand::Low, 0.18),
            onset_mid: ctx.onset(crate::osc::AudioBand::Mid, 0.15),
            onset_high: ctx.onset(crate::osc::AudioBand::High, 0.10),
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
        // atomic loads); we don't bother with dirty tracking. Inactive
        // (picked-out) passes skip the write — they don't draw this frame,
        // and their uniforms refresh on the tick that reactivates them.
        for pass in &self.layer_passes {
            if !pass.active {
                continue;
            }
            let scalars: Vec<f32> = pass
                .scalars
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    pass.scalar_names
                        .get(i)
                        .and_then(|n| overrides.get(&pass.binding_id, n))
                        .unwrap_or_else(|| s.eval(ctx))
                })
                .collect();
            let params = LayerParamsGpu::build(
                pass.slice,
                pass.effect_id,
                &pass.identity,
                &scalars,
                &pass.colors,
            );
            gpu.queue
                .write_buffer(&pass.layer_buffer, 0, bytemuck::bytes_of(&params));
        }
    }

    /// Encode the per-layer composite pass into `encoder`. Shared by the
    /// presented path and the occluded/offscreen path.
    fn encode_composite(&self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder) {
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
            if !layer.active {
                // Picked out this cycle (§5.2) — the pass stays in the plan
                // so re-picking is a flag flip, but it draws nothing.
                continue;
            }
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

    /// Render the composite buffer only — no swapchain interaction at all.
    /// Used while the projector window is occluded: on macOS an occluded
    /// window's `get_current_texture()` blocks for up to ~1s per frame
    /// (compositor throttling), which used to stall the entire render thread
    /// (IPC, preview, hot-reload). The composite keeps updating so the
    /// operator preview stays live.
    pub fn render_offscreen(&self, gpu: &GpuContext) {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("offscreen frame encoder"),
            });
        self.encode_composite(gpu, &mut encoder);
        gpu.queue.submit(std::iter::once(encoder.finish()));
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
        self.encode_composite(gpu, &mut encoder);

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

    /// Snapshot every scalar param across the plan's bindings for the
    /// `drivers` telemetry channel. Slices belonging to the same binding
    /// share params, so rows are deduped by binding id. §5.5 overrides are
    /// reported as the live value (matching what the shader receives), with
    /// `overridden` flagged so the UI can mark the row.
    pub fn driver_rows(&self, ctx: &FrameContext, overrides: &ParamOverrides) -> Vec<DriverRow> {
        let mut seen = std::collections::HashSet::new();
        let mut rows = Vec::new();
        for pass in &self.layer_passes {
            if !seen.insert(pass.binding_id.as_str()) {
                continue;
            }
            for (i, scalar) in pass.scalars.iter().enumerate() {
                let name = pass
                    .scalar_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("param_{i}"));
                let over = overrides.get(&pass.binding_id, &name);
                rows.push(DriverRow {
                    binding_id: pass.binding_id.clone(),
                    param_name: name,
                    source: scalar.describe(),
                    value: over.unwrap_or_else(|| scalar.eval(ctx)),
                    affects: pass.sibling_count,
                    overridden: over.is_some(),
                });
            }
        }
        rows
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

/// FNV-1a 32-bit. Used instead of `DefaultHasher` because the seed must be
/// stable across runs, builds, and std versions — layer_seed and pick
/// choices are part of the deterministic-replay contract (§5.2).
fn fnv1a(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Stable per-layer random in [0, 1) from a layer id.
fn seed01(id: &str) -> f32 {
    // 24-bit mantissa-safe fraction: exact in f32, uniform enough.
    (fnv1a(id) >> 8) as f32 / (1u32 << 24) as f32
}

/// The §5.2 pick function: member index for (binding, cycle). splitmix64
/// finalizer decorrelates consecutive cycles so "every 4 bars pick a new
/// leaf" doesn't walk a visible pattern.
fn pick_choice(binding_seed: u32, cycle: u64, member_count: usize) -> usize {
    if member_count <= 1 {
        return 0;
    }
    let mut z = ((binding_seed as u64) << 32) ^ cycle;
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z % member_count as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_is_stable() {
        // Pinned value — if this changes, every scene's layer_seed and pick
        // history changes with it. Do not "fix" the constant; fix the code.
        assert_eq!(fnv1a("leaf_01"), 0xd237_6961);
    }

    #[test]
    fn seed01_in_unit_range_and_distinct() {
        let a = seed01("leaf_01");
        let b = seed01("leaf_02");
        assert!((0.0..1.0).contains(&a));
        assert!((0.0..1.0).contains(&b));
        assert_ne!(a, b);
    }

    #[test]
    fn pick_choice_is_deterministic_and_in_range() {
        let seed = fnv1a("bloom");
        for cycle in 0..1000u64 {
            let c = pick_choice(seed, cycle, 20);
            assert!(c < 20);
            assert_eq!(c, pick_choice(seed, cycle, 20));
        }
        // Sanity: over 1000 cycles a 20-member pick should touch most members.
        let mut hit = [false; 20];
        for cycle in 0..1000u64 {
            hit[pick_choice(seed, cycle, 20)] = true;
        }
        assert!(hit.iter().filter(|h| **h).count() >= 18);
    }

    #[test]
    fn pick_choice_degenerate_sets() {
        assert_eq!(pick_choice(fnv1a("x"), 7, 0), 0);
        assert_eq!(pick_choice(fnv1a("x"), 7, 1), 0);
    }
}
