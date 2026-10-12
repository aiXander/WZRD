//! Compositor — turns a parsed scene + loaded pack into a linear pass plan
//! and executes it each frame (§3.6 "Internal compile step — pass plan").
//!
//! Phase 3 widens the Phase 2 plan: per-binding scalar drivers are evaluated
//! each frame, the shared `FrameState` uniform feeds clock + audio into every
//! pass, and project-local / inline WGSL effects each get their own pipeline
//! cached by the gpu layer. The render-thread per-frame work is still tiny —
//! evaluate drivers, fill each drawing pass's slot of the plan's one
//! uniform buffer (one `write_buffer` per leg), draw three verts per pass.
//!
//! Slow-path FBO routing (`layerRef`) is *not* implemented yet — Phase 3 was
//! scoped to "a few effects so the surface works"; deferred per §3.6 once a
//! real scene needs it.

use anyhow::{anyhow, Result};

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::drivers::{FrameContext, ParamOverrides, PickRate, ScalarValue};
use crate::telemetry::DriverRow;
use crate::effects::{
    EffectBinding, EffectDef, EffectKind, EffectRegistry, ImagePlacement, InlineEffectSpec,
    MAX_SCALAR_PARAMS,
};
use crate::gpu::{
    layer_bind_group, FrameStateGpu, GpuContext, ImageEncoding, LayerIdentity, LayerParamsGpu,
    BUILTIN_PIPELINE_KEY,
};
use crate::video::ImageSource;
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
    /// Operator layer fader for this frame, folded from the `@opacity` /
    /// `@mute` / `@solo` overrides in [`PassPlan::tick`]. `0` skips the draw.
    pub opacity: f32,
    /// C5 image for this pass — index into `PassPlan::images`, placement
    /// rect (uv) and texel encoding — when its binding opened one; `None` =
    /// `sample_image` is transparent.
    pub image: Option<(usize, [f32; 4], ImageEncoding)>,

    /// This pass's `LayerParams` slot in the plan's uniform buffer (slot 0
    /// is the FrameState); its bind group points at that offset.
    pub slot: usize,
    pub bind_group: wgpu::BindGroup,
}

/// An opened image input and the binding that reads it.
pub struct BindingImage {
    pub binding_id: String,
    /// Resolved file — with `binding_id`, the key a rebuild matches on.
    path: PathBuf,
    /// Shared only with the plan this one replaced on the same leg
    /// ([`Carry::Replace`]), which is dropped right after the swap — never
    /// with the other leg (playheads are per leg).
    source: Arc<Mutex<ImageSource>>,
    /// C5 playhead rate.
    pub rate: f32,
    /// Playhead = leg clock × `rate` + `offset_sec`. Zero for a freshly
    /// bound clip (the stateless C5 playhead); a rebuild that changes the
    /// rate moves it so the position holds (a rate slider bends playback,
    /// never jumps it).
    offset_sec: f64,
    /// Playhead advance over the last tick — the staleness tolerance.
    step_sec: f64,
    /// The texture shows the playhead frame (set in `tick`): passes bind
    /// it only then, so a stream waking from idle never flashes old frames.
    current: bool,
}

impl BindingImage {
    fn playhead(&self, clock_sec: f64) -> f64 {
        clock_sec * self.rate as f64 + self.offset_sec
    }

    fn source(&self) -> MutexGuard<'_, ImageSource> {
        // A panic mid-upload leaves nothing half-written that matters here.
        self.source.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Perf-monitor PM1 GPU-MB inventory.
    pub fn gpu_bytes(&self) -> u64 {
        self.source().gpu_bytes()
    }
}

/// What a new plan inherits from the one it follows — the clip-continuity
/// rule: no operator or agent edit may restart, blank or jump a playing
/// clip.
#[derive(Clone, Copy)]
pub enum Carry<'a> {
    /// No predecessor (boot): clips open at the stateless playhead.
    Fresh,
    /// The plan this one replaces on the **same leg** (any scene edit,
    /// effect save, identity change, headless hot-reload): a clip it
    /// already plays for the same binding + file is handed on as-is — no
    /// reopen, no black frame, no texture swap — with its playhead held
    /// continuous across a rate change.
    Replace(&'a PassPlan),
    /// The **other** leg's plan (pull; design's rebuild after promote):
    /// streams are never shared across legs, so clips reopen, but at the
    /// same playhead (offsets copied) so the two legs stay frame-identical.
    Mirror(&'a PassPlan),
}

impl<'a> Carry<'a> {
    fn plan(self) -> Option<&'a PassPlan> {
        match self {
            Carry::Fresh => None,
            Carry::Replace(p) | Carry::Mirror(p) => Some(p),
        }
    }
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

/// Compiled pass plan. Rebuilt on every scene hot-reload — seamlessly: the
/// rebuild inherits its predecessor's clips via [`Carry`].
pub struct PassPlan {
    pub layer_passes: Vec<LayerPass>,
    pub picks: Vec<PickGroup>,
    pub clear_color: wgpu::Color,
    /// C5 image inputs, one per binding that opened one. Owned by the plan
    /// so they travel through the promote pointer swap with it.
    pub images: Vec<BindingImage>,
    /// Non-fatal build problems (an image that failed to open: that binding
    /// draws nothing, the rest of the plan plays). Surfaced by `Core` in the
    /// apply reply + `hot_reload`.
    pub warnings: Vec<String>,
    /// This plan's leg transport time at the last tick — the C5 playhead
    /// clock (× each binding's rate).
    clock_sec: f64,
    /// Every clip a drawing pass samples shows its playhead frame (last
    /// tick). Promote waits on this so a TAKE never projects a stale or
    /// unbound clip frame.
    clips_ready: bool,
    /// Per-plan uniform buffer: slot 0 = FrameState, slot `1 + k` = pass
    /// k's LayerParams, each slot `uniform_stride` bytes (the device's
    /// uniform offset alignment). Per plan (§5.6 full-control-switch): the
    /// two legs tick with *different* contexts (own transport speed, own
    /// audioListen), and the buffer travels with the plan through the
    /// promote pointer swap (the bind groups reference it). Filled CPU-side
    /// in `uniform_bytes` and uploaded with **one** `write_buffer` per
    /// rendered tick — wgpu-core 22 allocates a staging buffer per
    /// `write_buffer` call, so per-pass writes cost O(passes) allocations.
    uniforms: wgpu::Buffer,
    uniform_bytes: Vec<u8>,
    uniform_stride: usize,
    /// Per image: does any pass that samples it draw this frame? Computed
    /// once per tick, read by `encode_uploads`.
    drawn: Vec<bool>,
    /// A drawing pass samples a video clip (as of the last tick).
    draws_clips: bool,
}

/// Byte stride between uniform slots: the larger of the two structs,
/// rounded up to the device's `min_uniform_buffer_offset_alignment`.
fn uniform_stride(gpu: &GpuContext) -> usize {
    let align = gpu.device.limits().min_uniform_buffer_offset_alignment.max(16) as usize;
    let size = std::mem::size_of::<LayerParamsGpu>().max(std::mem::size_of::<FrameStateGpu>());
    size.div_ceil(align) * align
}

fn write_slot(bytes: &mut [u8], slot: usize, stride: usize, data: &[u8]) {
    let at = slot * stride;
    bytes[at..at + data.len()].copy_from_slice(data);
}

fn slot_binding(buffer: &wgpu::Buffer, slot: usize, stride: usize, size: usize) -> wgpu::BufferBinding<'_> {
    wgpu::BufferBinding {
        buffer,
        offset: (slot * stride) as u64,
        size: wgpu::BufferSize::new(size as u64),
    }
}

impl PassPlan {
    /// Build a fresh plan. `gpu` is `&mut` because new user-authored effects
    /// trigger pipeline compilation into the cache.
    ///
    /// On error the caller keeps the previous plan rendering (§3.6
    /// swap-on-success); see `main.rs::rebuild_plan`.
    ///
    /// `scene_dir` resolves the relative clip paths of C5 `image` params;
    /// `carry` says what the plan inherits from its predecessor (clip
    /// streams + playhead continuity — see [`Carry`]).
    pub fn build(
        gpu: &mut GpuContext,
        pack: &LoadedPack,
        scene: &SceneFile,
        registry: &EffectRegistry,
        scene_dir: &Path,
        carry: Carry<'_>,
    ) -> Result<Self> {
        let clock_sec = carry.plan().map_or(0.0, |p| p.clock_sec);
        let mut passes: Vec<LayerPass> = Vec::new();
        let mut picks: Vec<PickGroup> = Vec::new();
        let mut images: Vec<BindingImage> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();

        // One uniform buffer for the whole plan (see `uniforms`), sized up
        // front: slot 0 + one slot per resolved layer.
        let uniform_stride = uniform_stride(gpu);
        let mut pass_count = 0usize;
        for binding in &scene.bindings {
            pass_count += resolve_selector(&binding.select, pack)?.len();
        }
        let mut uniform_bytes = vec![0u8; uniform_stride * (pass_count + 1)];
        write_slot(
            &mut uniform_bytes,
            0,
            uniform_stride,
            bytemuck::bytes_of(&FrameStateGpu::zeroed(gpu.composite_width, gpu.composite_height)),
        );
        let uniforms = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plan uniforms"),
            size: uniform_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
                    _ => None,
                })
                .collect();

            let slices = resolve_selector(&binding.select, pack)?;
            let sibling_count = slices.len() as u32;

            // C5 — open the binding's image once; every resolved layer
            // samples the same texture, each placed in its own box. A bad
            // clip is a warning, never a plan failure. A clip the
            // predecessor already plays keeps its position (and, on the same
            // leg, its very stream — see `Carry`).
            let image = resolved.image.as_ref().and_then(|param| {
                let path = scene_dir.join(&param.clip);
                let prev = carry.plan().and_then(|p| {
                    p.images
                        .iter()
                        .find(|i| i.binding_id == binding.id && i.path == path)
                });
                let offset_sec =
                    prev.map_or(0.0, |i| i.playhead(clock_sec) - clock_sec * param.rate as f64);
                let playhead = clock_sec * param.rate as f64 + offset_sec;
                let source = match (carry, prev) {
                    (Carry::Replace(_), Some(i)) => Ok(Arc::clone(&i.source)),
                    _ => ImageSource::open(gpu, &path, playhead).map(|s| Arc::new(Mutex::new(s))),
                };
                match source {
                    Ok(source) => {
                        let (encoding, size, current) = {
                            let s = source.lock().unwrap_or_else(|e| e.into_inner());
                            (s.encoding(), s.size(), s.is_current(playhead, 0.0))
                        };
                        images.push(BindingImage {
                            binding_id: binding.id.clone(),
                            path,
                            source,
                            rate: param.rate,
                            offset_sec,
                            step_sec: prev.map_or(0.0, |i| i.step_sec),
                            current,
                        });
                        Some((images.len() - 1, param.placement, encoding, size))
                    }
                    Err(e) => {
                        warnings.push(format!(
                            "binding {:?}: image {:?} not loaded, binding draws nothing: {e:#}",
                            binding.id, param.clip
                        ));
                        None
                    }
                }
            });

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
                let canvas = (gpu.composite_width, gpu.composite_height);
                let pass_image = image.map(|(i, placement, encoding, size)| {
                    (i, image_box(placement, &geom, size, canvas), encoding)
                });
                let mut initial = LayerParamsGpu::build(
                    slice,
                    effect_id,
                    &identity,
                    &vec![0.0; resolved.scalars.len()],
                    &resolved.colors,
                );
                if let Some((i, b, enc)) = pass_image {
                    if images[i].current {
                        initial = initial.with_image(b, enc);
                    }
                }
                let slot = passes.len() + 1;
                write_slot(&mut uniform_bytes, slot, uniform_stride, bytemuck::bytes_of(&initial));
                let source = image.map(|(i, ..)| images[i].source());
                let bind_group = layer_bind_group(
                    gpu,
                    &format!("layer bg [{} slice {}]", binding.id, slice),
                    slot_binding(&uniforms, 0, uniform_stride, std::mem::size_of::<FrameStateGpu>()),
                    slot_binding(&uniforms, slot, uniform_stride, std::mem::size_of::<LayerParamsGpu>()),
                    source.as_ref().map(|s| s.view()),
                );
                drop(source);

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
                    opacity: 1.0,
                    image: pass_image,
                    slot,
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

        // Build-time upload of every slot's initial contents (one staging
        // buffer); per-frame ticks then rewrite only what draws.
        gpu.queue.write_buffer(&uniforms, 0, &uniform_bytes);
        let drawn = vec![false; images.len()];
        Ok(Self {
            layer_passes: passes,
            picks,
            clear_color: wgpu::Color::BLACK,
            images,
            warnings,
            clock_sec,
            clips_ready: true,
            draws_clips: false,
            drawn,
            uniforms,
            uniform_bytes,
            uniform_stride,
        })
    }

    /// Evaluate every driver-bound scalar against the current transport +
    /// audio state and write the result + per-binding colours into each
    /// drawing pass's uniform slot. `&mut` because §5.2 pick state (active
    /// flags, cycle counters) advances here. The §5.5 override table wins
    /// over the scene-authored value wherever a (binding, param) entry
    /// exists.
    ///
    /// `upload` = this leg renders this frame. Clock, picks, faders and
    /// clip readiness advance either way; the uniform upload (the only GPU
    /// work here) happens only when the composite will be drawn. Every
    /// drawing pass is rewritten on every uploading tick, so a pass that
    /// was skipped (picked out, faded to 0, leg unrendered) is current the
    /// first frame it draws again.
    pub fn tick(&mut self, gpu: &GpuContext, ctx: &FrameContext, overrides: &ParamOverrides, upload: bool) {
        let prev_clock = std::mem::replace(&mut self.clock_sec, ctx.elapsed_sec as f64);
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
        // Operator layer faders. Solo is plan-wide: any soloed binding
        // silences every binding that isn't.
        let any_solo = self
            .layer_passes
            .iter()
            .any(|p| layer_flag(overrides, &p.binding_id, LAYER_SOLO));
        for pass in &mut self.layer_passes {
            pass.opacity = layer_opacity(overrides, &pass.binding_id, any_solo);
        }

        // Per image: does any drawing pass sample it this frame (once per
        // tick; `encode_uploads` reuses it).
        self.drawn.clear();
        self.drawn.resize(self.images.len(), false);
        for p in &self.layer_passes {
            if let Some((i, _, _)) = p.image {
                self.drawn[i] |= p.active && p.opacity > 0.0;
            }
        }

        // Which clips show their playhead frame. A stream that sat idle
        // (muted, picked out, design leg unrendered) holds an old frame; it
        // stays unbound (transparent) for the frame or two the decoder
        // needs, instead of flashing stale content.
        let mut ready = true;
        let mut draws_clips = false;
        for (img, &drawn) in self.images.iter_mut().zip(&self.drawn) {
            img.step_sec = (self.clock_sec - prev_clock) * img.rate as f64;
            let playhead = img.playhead(self.clock_sec);
            let source = img.source.lock().unwrap_or_else(|e| e.into_inner());
            img.current = source.is_current(playhead, img.step_sec);
            draws_clips |= drawn && source.is_clip();
            drop(source);
            ready &= !drawn || img.current;
        }
        self.clips_ready = ready;
        self.draws_clips = draws_clips;

        if !upload {
            return;
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
        let stride = self.uniform_stride;
        write_slot(&mut self.uniform_bytes, 0, stride, bytemuck::bytes_of(&frame));

        // Per-pass params. Driver evaluation is cheap (a few muls + a few
        // atomic loads); we don't bother with dirty tracking. Passes that
        // don't draw this frame (picked out, faded to 0) skip the fill —
        // their slot is refreshed on the tick that brings them back.
        for pass in &self.layer_passes {
            if !pass.active || pass.opacity <= 0.0 {
                continue;
            }
            let mut scalars = [0.0f32; MAX_SCALAR_PARAMS];
            let n = pass.scalars.len().min(MAX_SCALAR_PARAMS);
            for (i, (s, out)) in pass.scalars.iter().zip(&mut scalars).enumerate() {
                *out = pass
                    .scalar_names
                    .get(i)
                    .and_then(|name| overrides.get(&pass.binding_id, name))
                    .unwrap_or_else(|| s.eval(ctx));
            }
            let mut params = LayerParamsGpu::build(
                pass.slice,
                pass.effect_id,
                &pass.identity,
                &scalars[..n],
                &pass.colors,
            );
            params.opacity = pass.opacity;
            // A clip binds only once its first frame has landed (the zeroed
            // texture would read as a colour cast under HAP Q's YCoCg).
            if let Some((i, b, enc)) = pass.image {
                if self.images[i].current {
                    params = params.with_image(b, enc);
                }
            }
            write_slot(&mut self.uniform_bytes, pass.slot, stride, bytemuck::bytes_of(&params));
        }
        // One staging allocation per leg per rendered frame.
        gpu.queue.write_buffer(&self.uniforms, 0, &self.uniform_bytes);
    }

    /// C5 clip uploads for a frame this plan renders: each clip a drawing
    /// pass samples asks its decoder for the playhead frame and encodes at
    /// most one staging→texture copy. Encode before [`Self::encode_composite`]
    /// into the same encoder; call [`Self::after_submit`] after the submit.
    /// Reads the per-image draw flags this frame's [`Self::tick`] computed.
    pub fn encode_uploads(&mut self, encoder: &mut wgpu::CommandEncoder) {
        for (img, &drawn) in self.images.iter().zip(&self.drawn) {
            // Muted / picked-out clips don't decode: no want, no upload.
            if drawn {
                let playhead = img.playhead(self.clock_sec);
                img.source().encode_upload(encoder, playhead);
            }
        }
    }

    /// Every clip a drawing pass samples shows its playhead frame (as of the
    /// last tick). The promote gate: a TAKE starts once this holds.
    pub fn clips_ready(&self) -> bool {
        self.clips_ready
    }

    /// A drawing pass samples a video clip (as of the last tick). Such a
    /// leg can't render sparsely: its decoders prefetch along the per-render
    /// playhead stride, and a ~4-frame stride quantizes into missed
    /// (stale) frames — so `Core` renders it every frame while demanded.
    pub fn draws_clips(&self) -> bool {
        self.draws_clips
    }

    /// Return consumed clip staging slots to their decoders (re-map). Must
    /// follow the submit that carried [`Self::encode_uploads`]' copies.
    /// Returns whether the plan plays any clip — the caller then drives one
    /// `Maintain::Poll` per frame (not one per leg) so the re-map callbacks
    /// fire.
    pub fn after_submit(&mut self) -> bool {
        let mut clips = false;
        for img in &self.images {
            let mut source = img.source();
            source.after_submit();
            clips |= source.is_clip();
        }
        clips
    }

    /// Encode the per-layer composite pass into `encoder`, targeting
    /// `target_view` — the plan's own leg composite (§5.6: live and design
    /// each own a target). Shared by the presented path and the
    /// occluded/offscreen path; `Core` owns pass ordering.
    pub fn encode_composite(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.clear_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes,
        });
        // Avoid redundant pipeline switches when consecutive layers share one.
        let mut bound: Option<&str> = None;
        for layer in &self.layer_passes {
            if !layer.active || layer.opacity <= 0.0 {
                // Picked out this cycle (§5.2) — the pass stays in the plan
                // so re-picking is a flag flip, but it draws nothing. A
                // faded-out/muted layer likewise costs no GPU work.
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

    /// Snapshot the plan's pipeline cache keys — the §5.6 cross-leg eviction
    /// guard consults both legs' sets so a design-side edit can never drop a
    /// pipeline the live plan still draws with.
    pub fn pipeline_keys(&self) -> impl Iterator<Item = &str> {
        self.layer_passes.iter().map(|p| p.pipeline_key.as_str())
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

/// Reserved override params (the `@` prefix can't collide with a WGSL input
/// name) carrying the operator's per-layer mixer state. They ride the §5.5
/// override table, so they are per leg, zero-rebuild, copied by
/// promote/pull and persisted in the session sidecar for free.
pub const LAYER_OPACITY: &str = "@opacity";
pub const LAYER_MUTE: &str = "@mute";
pub const LAYER_SOLO: &str = "@solo";

fn layer_flag(overrides: &ParamOverrides, binding: &str, key: &str) -> bool {
    overrides.get(binding, key).is_some_and(|v| v > 0.5)
}

/// Effective fader for one binding: `@opacity` (default 1, clamped 0..1),
/// zeroed by `@mute`, or by another binding's `@solo` when this one isn't.
pub(crate) fn layer_opacity(overrides: &ParamOverrides, binding: &str, any_solo: bool) -> f32 {
    if layer_flag(overrides, binding, LAYER_MUTE)
        || (any_solo && !layer_flag(overrides, binding, LAYER_SOLO))
    {
        return 0.0;
    }
    overrides
        .get(binding, LAYER_OPACITY)
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(1.0)
}

/// C5 placement rect (uv) for one pass. The target box is the whole canvas,
/// or the layer's own padded `source_box` (C4; `bbox` when the pack has
/// none) so one clip fits whichever region it is bound to. The image is
/// then fit into the box uniformly and centred — "contain", never stretched
/// (the D8 rule: a clip rendered at a model's native aspect is close to, not
/// exactly, its box; contain is the exact inverse of the model's
/// cover-crop). Outside the fitted rect `sample_image` is transparent.
fn image_box(
    placement: ImagePlacement,
    geom: &crate::pack::LayerGeom,
    image_px: (u32, u32),
    canvas_px: (u32, u32),
) -> [f32; 4] {
    let target = match placement {
        ImagePlacement::Canvas => [0.0, 0.0, 1.0, 1.0],
        ImagePlacement::Region => geom.region_box_uv(),
    };
    contain_uv(target, image_px, canvas_px)
}

/// Fit an `image_px`-sized image uniformly into `target` (uv on a
/// `canvas_px` canvas), centred. Aspects compare in pixels, not uv.
fn contain_uv(target: [f32; 4], image_px: (u32, u32), canvas_px: (u32, u32)) -> [f32; 4] {
    let [x0, y0, x1, y1] = target;
    let box_w = (x1 - x0) * canvas_px.0 as f32;
    let box_h = (y1 - y0) * canvas_px.1 as f32;
    let (iw, ih) = (image_px.0.max(1) as f32, image_px.1.max(1) as f32);
    if box_w <= 0.0 || box_h <= 0.0 {
        return target;
    }
    let scale = (box_w / iw).min(box_h / ih);
    let (fw, fh) = (iw * scale / canvas_px.0 as f32, ih * scale / canvas_px.1 as f32);
    let (cx, cy) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
    [cx - fw * 0.5, cy - fh * 0.5, cx + fw * 0.5, cy + fh * 0.5]
}

/// Shared by plan build and the §5.6 probe pre-analysis (which needs to know
/// which pipelines a scene would pull in before building anything).
pub(crate) fn resolve_effect_def(binding: &BindingSpec, registry: &EffectRegistry) -> Result<EffectDef> {
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
pub(crate) mod tests {
    use super::*;
    use crate::drivers::SliderBank;
    use crate::osc::AudioFeatures;

    /// A one-layer 64×32 pack in a fresh temp dir.
    fn test_pack() -> LoadedPack {
        let dir = std::env::temp_dir().join(format!(
            "wzrd-compositor-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        image::GrayImage::from_pixel(64, 32, image::Luma([255]))
            .save(dir.join("a.png"))
            .unwrap();
        std::fs::write(
            dir.join("pack.json"),
            r#"{"version":1,"projector_resolution":[64,32],"layers":[{"id":"a","mask":"a.png"}]}"#,
        )
        .unwrap();
        LoadedPack::load(&dir).unwrap()
    }

    /// Offscreen device for plan/clip tests; `None` (skip) without an
    /// adapter or without BC (HAP) support.
    pub(crate) fn test_gpu() -> Option<GpuContext> {
        let pack = test_pack();
        let gpu = pollster::block_on(GpuContext::new_offscreen(64, 32, None, &pack, false, false)).ok()?;
        gpu.bc_supported.then_some(gpu)
    }

    fn clip_scene(file: &str, rate: f32) -> SceneFile {
        let clip = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(file);
        SceneFile::parse(&format!(
            r#"{{"version":1,"pack":"pack","bindings":[{{"id":"c","select":{{"id":"a"}},
               "effect":"videoClip","params":{{"image":{{"clip":{clip:?},"rate":{rate}}}}}}}]}}"#
        ))
        .unwrap()
    }

    /// The clip-continuity rule (`Carry`): a same-leg rebuild hands the very
    /// stream on, a rate edit holds the position, a cross-leg mirror opens
    /// its own stream at the same position.
    #[test]
    fn rebuilds_carry_clips_without_reopening_or_jumping() {
        let Some(mut gpu) = test_gpu() else {
            eprintln!("no BC-capable adapter — skipping");
            return;
        };
        let pack = test_pack();
        let registry = EffectRegistry::new(None);
        let (audio, sliders, overrides) = (AudioFeatures::new(), SliderBank::new(), ParamOverrides::new());
        let ctx = |t: f32| FrameContext {
            elapsed_sec: t,
            bpm: 120.0,
            beat_offset: 0.0,
            audio: &audio,
            sliders: &sliders,
            audio_listen: 1.0,
        };
        let dir = Path::new("/");

        let mut a = PassPlan::build(&mut gpu, &pack, &clip_scene("hap.mov", 1.0), &registry, dir, Carry::Fresh).unwrap();
        a.tick(&gpu, &ctx(1.0), &overrides, true);
        assert!(a.images[0].current && a.clips_ready());

        // Same leg, rate 1 → 2 at t = 1 s: same stream, playhead still 1 s.
        let b = PassPlan::build(&mut gpu, &pack, &clip_scene("hap.mov", 2.0), &registry, dir, Carry::Replace(&a)).unwrap();
        assert!(Arc::ptr_eq(&a.images[0].source, &b.images[0].source));
        assert!((b.images[0].playhead(1.0) - 1.0).abs() < 1e-9);
        assert!((b.images[0].playhead(1.5) - 2.0).abs() < 1e-9); // then 2× on
        assert!(b.images[0].current, "carried clip binds on the first frame");

        // Other leg: own stream, same position, current from the start.
        let c = PassPlan::build(&mut gpu, &pack, &clip_scene("hap.mov", 2.0), &registry, dir, Carry::Mirror(&b)).unwrap();
        assert!(!Arc::ptr_eq(&b.images[0].source, &c.images[0].source));
        assert!((c.images[0].playhead(1.0) - 1.0).abs() < 1e-9);
        assert!(c.images[0].current);

        // A different file on the binding is a new clip: its own stream on
        // the stateless playhead (clock × rate), still current at once.
        let d = PassPlan::build(&mut gpu, &pack, &clip_scene("hap_q.mov", 2.0), &registry, dir, Carry::Replace(&b)).unwrap();
        assert!(!Arc::ptr_eq(&b.images[0].source, &d.images[0].source));
        assert!((d.images[0].playhead(1.0) - 2.0).abs() < 1e-9);
        assert!(d.images[0].current);
    }

    #[test]
    fn contain_fits_by_pixel_aspect_and_centres() {
        let close = |a: [f32; 4], b: [f32; 4]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5);
        // 16:9 clip on a 16:9 canvas fills it exactly.
        assert!(close(contain_uv([0.0, 0.0, 1.0, 1.0], (1920, 1080), (1920, 1080)), [0.0, 0.0, 1.0, 1.0]));
        // Square clip on a 2:1 canvas → pillarboxed, centred.
        assert!(close(contain_uv([0.0, 0.0, 1.0, 1.0], (100, 100), (200, 100)), [0.25, 0.0, 0.75, 1.0]));
        // 9:16 clip into a 200×400 px box (on a 400×400 canvas): the box
        // (1:2) is narrower than the clip (0.5625) → full width, height
        // 200 / 0.5625 = 355.6 px, centred.
        let h = 200.0 / 0.5625 / 400.0;
        let b = contain_uv([0.25, 0.0, 0.75, 1.0], (9, 16), (400, 400));
        assert!(close(b, [0.25, 0.5 - h / 2.0, 0.75, 0.5 + h / 2.0]), "{b:?}");
        // A 100×400 box (1:4) is narrower still → full width, centred.
        let b = contain_uv([0.0, 0.0, 0.25, 1.0], (9, 16), (400, 400));
        let h = 100.0 / 0.5625 / 400.0;
        assert!(close(b, [0.0, 0.5 - h / 2.0, 0.25, 0.5 + h / 2.0]), "{b:?}");
        // 9:16 clip into a wide 400×200 box → full height, width 112.5 px.
        let b = contain_uv([0.0, 0.25, 1.0, 0.75], (9, 16), (400, 400));
        let w = 200.0 * 0.5625 / 400.0;
        assert!(close(b, [0.5 - w / 2.0, 0.25, 0.5 + w / 2.0, 0.75]), "{b:?}");
        // Wide clip into a tall box → letterboxed vertically.
        let b = contain_uv([0.0, 0.0, 0.5, 1.0], (16, 9), (400, 400));
        assert!(close(b, [0.0, 0.5 - 56.25 / 400.0, 0.5, 0.5 + 56.25 / 400.0]), "{b:?}");
    }

    #[test]
    fn layer_opacity_folds_mute_and_solo() {
        let o = crate::drivers::ParamOverrides::new();
        assert_eq!(layer_opacity(&o, "a", false), 1.0);
        o.set("a", LAYER_OPACITY, 0.4);
        assert!((layer_opacity(&o, "a", false) - 0.4).abs() < 1e-6);
        o.set("a", LAYER_OPACITY, 3.0);
        assert_eq!(layer_opacity(&o, "a", false), 1.0);
        o.set("a", LAYER_MUTE, 1.0);
        assert_eq!(layer_opacity(&o, "a", false), 0.0);
        o.clear("a", LAYER_MUTE);
        // Another binding soloed silences this one; soloing it restores it.
        assert_eq!(layer_opacity(&o, "a", true), 0.0);
        o.set("a", LAYER_SOLO, 1.0);
        assert_eq!(layer_opacity(&o, "a", true), 1.0);
    }

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
