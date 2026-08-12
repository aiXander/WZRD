//! wgpu state: surface, device, queue, mask-atlas texture, pipelines.
//!
//! Phase 3 widens what Phase 2 had:
//!
//! - Two uniforms instead of one — a per-frame `FrameState` (time, audio,
//!   transport phase) shared by every binding, and a per-binding
//!   `LayerParams` (effect_id, slice, scalar / colour slots).
//! - A pipeline cache keyed by effect "pipeline_key" (D15). Built-ins share
//!   one pipeline; each user-authored WGSL effect gets its own.
//! - A WGSL composer that stitches `prelude + effect_body + main` so user
//!   code only writes `fn effect(uv, mask) -> vec4<f32>`.
//!
//! The compositor in `compositor.rs` owns the pass plan and ticks
//! `FrameState` each frame; this file just exposes the bricks.

use std::collections::HashMap;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::effects::{MAX_COLOR_PARAMS, MAX_SCALAR_PARAMS};
use crate::pack::LoadedPack;

/// Format for the offscreen composite buffer.
///
/// `Rgba16Float` is the right home for true additive stacks: contributions
/// can overshoot 1.0 without banding, and the swapchain (sRGB UNORM) clamps
/// + gamma-encodes on the final write — which is exactly what "white soup"
/// looks like on a physical projector (architecture review v1 #1 + #11).
pub const COMPOSITE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const PRELUDE_WGSL: &str = include_str!("shaders/effect_prelude.wgsl");
const MAIN_WGSL: &str = include_str!("shaders/effect_main.wgsl");
const BUILTIN_BODY_WGSL: &str = include_str!("shaders/builtin_effects.wgsl");
/// Shared by the final pass and the warp bake so the two can't drift on the
/// dest-uv convention (a Y-flip mismatch renders as a mirrored warp).
const FULLSCREEN_VS_WGSL: &str = include_str!("shaders/fullscreen_vs.wgsl");
const FINAL_PASS_WGSL: &str = include_str!("shaders/final_pass.wgsl");
const WARP_BAKE_WGSL: &str = include_str!("shaders/warp_bake.wgsl");

/// Uniform-array bound on §5.14 warp handles. Mirrored by `MAX_POINTS` in
/// `alignment.rs` and by the `MAX_POINTS` const in `warp_bake.wgsl` — all
/// three must agree.
pub const MAX_WARP_POINTS: usize = 64;

/// The warp LUT format. `Rg32Float` is **core-renderable** — only *blending*
/// it would need the optional `float32-blendable` feature, and the bake pass
/// doesn't blend — and it is read with `textureLoad` only, so
/// `float32-filterable` isn't needed either. Both matter: CLAUDE.md pins this
/// crate to `Features::empty()`.
pub const WARP_LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;

/// Prepend the shared vertex stage to an output-side fragment shader.
fn compose_output_shader(fragment_src: &str) -> String {
    let mut s = String::with_capacity(FULLSCREEN_VS_WGSL.len() + fragment_src.len() + 1);
    s.push_str(FULLSCREEN_VS_WGSL);
    s.push('\n');
    s.push_str(fragment_src);
    s
}

/// Pipeline cache key for the bundled built-in effects. User effects get
/// content- or path-derived keys (see `effects::EffectKind::User`).
pub const BUILTIN_PIPELINE_KEY: &str = "builtin";

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FrameStateGpu {
    pub time: f32,
    pub bar_phase: f32,
    pub beat_phase: f32,
    pub bpm: f32,
    pub audio_low: f32,
    pub audio_mid: f32,
    pub audio_high: f32,
    pub onset_low: f32,
    pub onset_mid: f32,
    pub onset_high: f32,
    // Two scalars of padding to hit std140 16-byte alignment ahead of
    // `resolution` (vec4). Keep both — the prelude struct mirrors this.
    pub _pad0: f32,
    pub _pad1: f32,
    pub resolution: [f32; 4],
}

impl FrameStateGpu {
    pub fn zeroed(width: u32, height: u32) -> Self {
        Self {
            time: 0.0,
            bar_phase: 0.0,
            beat_phase: 0.0,
            bpm: 120.0,
            audio_low: 0.0,
            audio_mid: 0.0,
            audio_high: 0.0,
            onset_low: 0.0,
            onset_mid: 0.0,
            onset_high: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
            resolution: [width as f32, height as f32, 0.0, 0.0],
        }
    }
}

/// Per-layer identity within a binding's resolved selection (§5.2). Stable
/// inputs for organic variation: `layer_seed` hashes the layer *id* (so it
/// survives re-segmentation, D7), index/count describe the pass's position
/// in the selection, centroid/bbox locate the region in uv space.
#[derive(Debug, Clone, Copy)]
pub struct LayerIdentity {
    /// Stable per-layer random in [0, 1) — FNV-1a of the layer id.
    pub layer_seed: f32,
    pub layer_index: u32,
    pub layer_count: u32,
    pub centroid_uv: [f32; 2],
    /// (min_x, min_y, max_x, max_y), uv space.
    pub bbox_uv: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LayerParamsGpu {
    pub slice: u32,
    pub effect_id: u32,
    pub layer_index: u32,
    pub layer_count: u32,
    pub layer_seed: f32,
    pub _pad0: f32,
    pub centroid_uv: [f32; 2],
    pub bbox_uv: [f32; 4],
    /// 8 scalar slots packed as two vec4 lanes.
    pub params_f: [[f32; 4]; 2],
    pub params_c: [[f32; 4]; MAX_COLOR_PARAMS],
}

impl LayerParamsGpu {
    pub fn build(
        slice: u32,
        effect_id: u32,
        identity: &LayerIdentity,
        scalars: &[f32],
        colors: &[[f32; 4]],
    ) -> Self {
        let mut params_f = [[0.0f32; 4]; 2];
        for (i, v) in scalars.iter().enumerate().take(MAX_SCALAR_PARAMS) {
            params_f[i / 4][i % 4] = *v;
        }
        let mut params_c = [[0.0f32; 4]; MAX_COLOR_PARAMS];
        for (i, c) in colors.iter().enumerate().take(MAX_COLOR_PARAMS) {
            params_c[i] = *c;
        }
        Self {
            slice,
            effect_id,
            layer_index: identity.layer_index,
            layer_count: identity.layer_count,
            layer_seed: identity.layer_seed,
            _pad0: 0.0,
            centroid_uv: identity.centroid_uv,
            bbox_uv: identity.bbox_uv,
            params_f,
            params_c,
        }
    }
}

/// Final-pass uniform. The calibration matrix that used to live here is gone:
/// since §5.14 the warp is consumed at *bake* time out of the offset LUT, not
/// re-derived per frame, so what's left is the per-frame operator state.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FinalPassUniforms {
    /// §5.4 output masters + §5.6 promote mix + §5.14 warp enable:
    /// (brightness, saturation, mix live→design, warp). Applied in the final
    /// pass — the composites (and therefore the operator preview) stay
    /// un-mastered; only the projector output is scaled.
    pub adjust: [f32; 4],
    /// §5.14 out-of-source paint, linear light.
    pub background: [f32; 4],
    /// §3.6 test pattern: (mode, thickness in source uv, grid cells, _).
    pub pattern: [f32; 4],
}

impl FinalPassUniforms {
    pub fn new(brightness: f32, saturation: f32, mix: f32) -> Self {
        Self {
            adjust: [brightness, saturation, mix, 0.0],
            background: [0.0, 0.0, 0.0, 1.0],
            pattern: [0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Neutral masters, no crossfade, no warp — the preview blit's baseline.
    pub fn identity() -> Self {
        Self::new(1.0, 1.0, 0.0)
    }

    pub fn with_warp(mut self, enabled: bool, background: [f32; 3]) -> Self {
        self.adjust[3] = if enabled { 1.0 } else { 0.0 };
        self.background = [background[0], background[1], background[2], 1.0];
        self
    }

    pub fn with_pattern(mut self, mode: f32, thickness: f32, cells: f32) -> Self {
        self.pattern = [mode, thickness, cells, 0.0];
        self
    }
}

/// Bake-pass uniform: the projective base plus every handle's dest position,
/// inverse radius and solved coefficient. ~2 KB, well inside the default
/// 64 KB uniform binding limit, and written only when the alignment changes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WarpBakeUniforms {
    /// dest → source, three rows padded to vec4 for std140.
    pub h_inv: [[f32; 4]; 3],
    /// (handle count, _, _, _)
    pub counts: [u32; 4],
    /// (dest.x, dest.y, 1/radius, _)
    pub points: [[f32; 4]; MAX_WARP_POINTS],
    /// (w.x, w.y, _, _)
    pub weights: [[f32; 4]; MAX_WARP_POINTS],
}

impl WarpBakeUniforms {
    /// Identity base, no handles — a LUT of zeros.
    pub fn zeroed() -> Self {
        let mut u: Self = Zeroable::zeroed();
        u.h_inv = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        u
    }
}

/// §5.6 — which leg a preview/readback consumer samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leg {
    Live,
    Design,
}

/// Native operator-preview swapchain (app-collapse Step 3): a second
/// surface the same device blits a composite onto — the Resolume-style
/// preview. Runs the final-pass pipeline with the warp disabled (§5.6/§5.14:
/// the calibration warp only looks right on the physical surface).
///
/// §5.6 source toggle: one bind group per leg; the LIVE position renders
/// with the *real* brightness/saturation masters (what the crowd sees, sans
/// warp), the DESIGN position stays neutral/un-mastered (§5.4 convention).
pub struct PreviewTarget {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group_live: wgpu::BindGroup,
    bind_group_design: wgpu::BindGroup,
    /// Rewritten per present so the LIVE position tracks the masters and the
    /// DESIGN position stays neutral. `adjust.w` (warp) stays 0 here.
    buffer: wgpu::Buffer,
}

/// §5.14 — the baked offset LUT plus everything needed to refill it.
///
/// Sized exactly to the projector swapchain so the final pass can index it by
/// framebuffer pixel with `textureLoad` (exact, unfiltered, no sampler). A
/// resize recreates it and marks it dirty; the rebake is encoded into the same
/// frame encoder ahead of the final pass, so there is no window in which the
/// final pass could sample a LUT of the wrong size.
pub struct WarpTarget {
    /// Held to keep the wgpu resource alive; the view is what gets bound.
    #[allow(dead_code)]
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    /// Set on creation and on every resize; cleared once a bake is encoded.
    /// Alignment *edits* mark dirty on `AlignmentState` instead — this flag is
    /// only about the texture having been (re)allocated with junk in it.
    ///
    /// A `Cell` so encoding a frame stays a `&self` operation: `redraw` holds
    /// the swapchain frame (borrowed from `GpuContext`) across the whole
    /// encode, and a `&mut self` here would fight it for no benefit.
    needs_bake: std::cell::Cell<bool>,
}

pub struct GpuContext {
    /// Kept so additional surfaces (the preview) can be created against the
    /// same instance/adapter the device came from.
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    /// Attached/detached by the host at runtime (None headless / pre-attach).
    pub preview: Option<PreviewTarget>,

    /// LIVE-leg composite — what the projector pass warps to the swapchain.
    pub composite_texture: wgpu::Texture,
    pub composite_view: wgpu::TextureView,
    /// Bound through the homography bind group; not sampled directly.
    #[allow(dead_code)]
    pub composite_sampler: wgpu::Sampler,
    pub composite_width: u32,
    pub composite_height: u32,

    /// §5.6 DESIGN-leg composite — the scratchpad target, rendered on demand
    /// only. `None` in single-leg (headless) mode, where memory stays what
    /// it was pre-two-deck.
    pub design_texture: Option<wgpu::Texture>,
    pub design_view: Option<wgpu::TextureView>,

    /// Held to keep the wgpu resource alive; not sampled directly.
    #[allow(dead_code)]
    pub mask_atlas: wgpu::Texture,
    pub mask_atlas_view: wgpu::TextureView,
    pub mask_sampler: wgpu::Sampler,

    pub layer_bind_group_layout: wgpu::BindGroupLayout,
    pub layer_pipeline_layout: wgpu::PipelineLayout,

    /// Pipeline cache keyed by `pipeline_key`. Built-ins share
    /// `BUILTIN_PIPELINE_KEY`; user effects each get their own slot.
    pub pipeline_cache: HashMap<String, wgpu::RenderPipeline>,

    pub final_bind_group_layout: wgpu::BindGroupLayout,
    pub final_pipeline: wgpu::RenderPipeline,
    pub final_buffer: wgpu::Buffer,
    pub final_bind_group: wgpu::BindGroup,

    /// §5.14 alignment LUT + bake pipeline.
    pub warp: WarpTarget,
    /// Shared 1×1 zero LUT for passes that must not warp (the preview blit).
    /// Binding the *real* LUT there would index a projector-sized texture with
    /// preview pixel coordinates — out of bounds, and WGSL leaves the
    /// out-of-bounds `textureLoad` result implementation-defined, so it is not
    /// a portable identity fallback.
    warp_dummy_view: wgpu::TextureView,
}

impl GpuContext {
    /// `target` is anything wgpu can hang a surface on — a winit
    /// `Arc<Window>`, a tao window's raw handles, etc. — so this file has no
    /// windowing-crate dependency (app-collapse Step 1). `width`/`height`
    /// are the target's current inner size in physical pixels; a raw handle
    /// can't be queried for its size, so the host passes it in.
    ///
    /// `two_leg` (§5.6): allocate the DESIGN composite alongside the LIVE
    /// one. Headless single-leg passes `false` and stays byte-identical to
    /// the pre-two-deck engine (the homography bind group then binds the
    /// live composite to both texture slots; mix is 0).
    pub async fn new(
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
        pack: &LoadedPack,
        two_leg: bool,
    ) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance
            .create_surface(target)
            .context("creating wgpu surface for window")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("no compatible GPU adapter found")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("render-core device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("requesting wgpu device")?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let composite_width = pack.atlas_width;
        let composite_height = pack.atlas_height;
        let (composite_texture, composite_view, composite_sampler) =
            create_composite(&device, composite_width, composite_height);

        // §5.6 design leg — a second composite of the same shape. Memory
        // doubles in two-leg mode (accepted trade in the roadmap).
        let (design_texture, design_view) = if two_leg {
            let (t, v, _) = create_composite(&device, composite_width, composite_height);
            (Some(t), Some(v))
        } else {
            (None, None)
        };

        let (mask_atlas, mask_atlas_view, mask_sampler) = upload_mask_atlas(&device, &queue, pack);

        let layer_bind_group_layout = create_layer_bind_group_layout(&device);
        let layer_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layer pipeline layout"),
            bind_group_layouts: &[&layer_bind_group_layout],
            push_constant_ranges: &[],
        });

        let mut pipeline_cache = HashMap::new();
        let builtin_pipeline = build_effect_pipeline(
            &device,
            &layer_pipeline_layout,
            "builtin",
            BUILTIN_BODY_WGSL,
        )
        .context("compiling built-in effect pipeline")?;
        pipeline_cache.insert(BUILTIN_PIPELINE_KEY.to_string(), builtin_pipeline);

        let (final_bind_group_layout, final_pipeline) =
            create_final_pipeline(&device, surface_config.format);

        let warp = create_warp_target(&device, surface_config.width, surface_config.height);
        let warp_dummy_view = create_warp_dummy(&device, &queue);

        let final_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("final pass uniform"),
            contents: bytemuck::bytes_of(&FinalPassUniforms::identity()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let final_bind_group = create_final_bind_group(
            &device,
            &final_bind_group_layout,
            "final pass bind group",
            &composite_view,
            &composite_sampler,
            &final_buffer,
            design_view.as_ref().unwrap_or(&composite_view),
            &warp.view,
        );

        Ok(Self {
            instance,
            adapter,
            surface,
            surface_config,
            device,
            queue,
            preview: None,
            composite_texture,
            composite_view,
            composite_sampler,
            composite_width,
            composite_height,
            design_texture,
            design_view,
            mask_atlas,
            mask_atlas_view,
            mask_sampler,
            layer_bind_group_layout,
            layer_pipeline_layout,
            pipeline_cache,
            final_bind_group_layout,
            final_pipeline,
            final_buffer,
            final_bind_group,
            warp,
            warp_dummy_view,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        // §5.14 — the LUT is one texel per output pixel, so it follows the
        // swapchain. Recreating it invalidates the final bind group.
        self.resize_warp(width, height);
    }

    /// Reallocate the warp LUT to `width × height` and rebuild the final-pass
    /// bind group around it. Cheap no-op when the size is unchanged.
    fn resize_warp(&mut self, width: u32, height: u32) {
        if self.warp.width == width && self.warp.height == height {
            return;
        }
        self.warp = create_warp_target(&self.device, width, height);
        self.final_bind_group = create_final_bind_group(
            &self.device,
            &self.final_bind_group_layout,
            "final pass bind group",
            &self.composite_view,
            &self.composite_sampler,
            &self.final_buffer,
            self.design_view.as_ref().unwrap_or(&self.composite_view),
            &self.warp.view,
        );
    }

    pub fn warp_size(&self) -> (u32, u32) {
        (self.warp.width, self.warp.height)
    }

    /// True while the LUT holds undefined contents (freshly allocated or just
    /// resized) and must be rebaked before the final pass reads it.
    pub fn warp_needs_bake(&self) -> bool {
        self.warp.needs_bake.get()
    }

    pub fn write_warp_uniforms(&self, uniforms: &WarpBakeUniforms) {
        self.queue
            .write_buffer(&self.warp.buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Encode the LUT refill as step 0 of the frame — same encoder, same
    /// submit. Edit-triggered, never per frame: if this ever starts running
    /// every frame the design has gone wrong (§6).
    pub fn encode_warp_bake(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("warp bake pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.warp.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.warp.pipeline);
            pass.set_bind_group(0, &self.warp.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.warp.needs_bake.set(false);
    }

    /// Attach the native preview surface (app-collapse Step 3). Idempotent
    /// per target: a second call replaces the previous preview wholesale.
    pub fn attach_preview(
        &mut self,
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let surface = self
            .instance
            .create_surface(target)
            .context("creating preview surface")?;
        let caps = surface.get_capabilities(&self.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&self.device, &config);

        // Same pipeline family as the projector pass, compiled for the
        // preview surface's own format; masters written per present (real on
        // LIVE, neutral on DESIGN — §5.6 source toggle), warp always off.
        let (_, pipeline) = create_final_pipeline(&self.device, format);
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("preview blit uniform"),
            contents: bytemuck::bytes_of(&FinalPassUniforms::identity()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let make_group = |label: &str, view: &wgpu::TextureView| {
            create_final_bind_group(
                &self.device,
                &self.final_bind_group_layout,
                label,
                view,
                &self.composite_sampler,
                &buffer,
                // mix is always 0 on the preview blit; bind the source view
                // again to satisfy the layout.
                view,
                // §5.14 — never the real LUT here (see `warp_dummy_view`).
                &self.warp_dummy_view,
            )
        };
        let bind_group_live = make_group("preview blit bind group (live)", &self.composite_view);
        let bind_group_design = make_group(
            "preview blit bind group (design)",
            self.design_view.as_ref().unwrap_or(&self.composite_view),
        );
        self.preview = Some(PreviewTarget {
            surface,
            config,
            pipeline,
            bind_group_live,
            bind_group_design,
            buffer,
        });
        log::info!("preview surface attached ({width}x{height}, {format:?})");
        Ok(())
    }

    pub fn resize_preview(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if let Some(pv) = self.preview.as_mut() {
            if pv.config.width == width && pv.config.height == height {
                return;
            }
            pv.config.width = width;
            pv.config.height = height;
            pv.surface.configure(&self.device, &pv.config);
        }
    }

    /// Blit a composite onto the preview surface and present. Self-heals
    /// `Lost`/`Outdated` by reconfiguring (frame skipped); other errors are
    /// returned. No-op without an attached preview.
    ///
    /// §5.6 source toggle: `source` picks which leg's composite is sampled;
    /// LIVE applies the real brightness/saturation masters (identity
    /// homography — the calibration warp only reads right on the physical
    /// surface), DESIGN stays neutral/un-mastered.
    ///
    /// §3.1 caveat: the *caller* must ensure the preview window is visible —
    /// `get_current_texture` on an occluded macOS window blocks the render
    /// thread just like the projector swapchain does.
    pub fn render_preview(
        &self,
        source: Leg,
        brightness: f32,
        saturation: f32,
    ) -> Result<(), wgpu::SurfaceError> {
        let Some(pv) = self.preview.as_ref() else {
            return Ok(());
        };
        let uniforms = match source {
            Leg::Live => FinalPassUniforms::new(brightness, saturation, 0.0),
            Leg::Design => FinalPassUniforms::identity(),
        };
        self.queue
            .write_buffer(&pv.buffer, 0, bytemuck::bytes_of(&uniforms));
        let frame = match pv.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                pv.surface.configure(&self.device, &pv.config);
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("preview blit encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("preview blit pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
            pass.set_pipeline(&pv.pipeline);
            pass.set_bind_group(
                0,
                match source {
                    Leg::Live => &pv.bind_group_live,
                    Leg::Design => &pv.bind_group_design,
                },
                &[],
            );
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }

    /// Refresh the final-pass uniform: §5.4 output masters + §5.6 promote mix
    /// + §5.14 warp enable and background. Written once per presented frame
    /// (the buffer is 48 bytes; a dirty flag would cost more complexity).
    pub fn write_final_pass(&self, uniforms: &FinalPassUniforms) {
        self.queue
            .write_buffer(&self.final_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Encode the final pass (live composite × design composite lerped by the
    /// promote mix → masters → alignment warp) onto `target_view`. Pulled out
    /// of `PassPlan` so the §5.6 frame orchestration in `Core` owns pass
    /// ordering: warp bake (when dirty), live composite, design composite (on
    /// demand), then this.
    pub fn encode_final(&self, encoder: &mut wgpu::CommandEncoder, target_view: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("final pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
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
        pass.set_pipeline(&self.final_pipeline);
        pass.set_bind_group(0, &self.final_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Compile (or recompile) a user-authored effect pipeline. Caller is
    /// responsible for keeping the cache key stable across hot-reloads.
    ///
    /// Validates the WGSL with `naga` first so a bad shader surfaces as an
    /// error rather than crashing the device (§3.6 swap-on-success).
    pub fn upsert_user_pipeline(&mut self, pipeline_key: &str, wgsl: &str) -> Result<()> {
        let body = wgsl;
        let pipeline = build_effect_pipeline(
            &self.device,
            &self.layer_pipeline_layout,
            pipeline_key,
            body,
        )?;
        self.pipeline_cache
            .insert(pipeline_key.to_string(), pipeline);
        Ok(())
    }
}

fn create_layer_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("layer bind group layout"),
        entries: &[
            // 0: mask atlas (Texture2DArray<R8>)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            // 1: mask sampler
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // 2: FrameState (shared)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 3: LayerParams (per pass)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_composite(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("composite buffer"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COMPOSITE_FORMAT,
        // COPY_SRC is here pre-emptively for Phase 4's preview-thumbnail
        // readback (architecture review v1 #14) — no readback consumer yet,
        // but adding the flag now means the texture doesn't have to be
        // recreated when one lands.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("composite sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (tex, view, sampler)
}

fn upload_mask_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pack: &LoadedPack,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mask atlas (Texture2DArray<R8>)"),
        size: wgpu::Extent3d {
            width: pack.atlas_width,
            height: pack.atlas_height,
            depth_or_array_layers: pack.layer_count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pack.mask_atlas,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(pack.atlas_width),
            rows_per_image: Some(pack.atlas_height),
        },
        wgpu::Extent3d {
            width: pack.atlas_width,
            height: pack.atlas_height,
            depth_or_array_layers: pack.layer_count,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mask atlas sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (texture, view, sampler)
}

/// Stitch prelude + effect body + main into one shader source, run it
/// through `naga` for early validation (gives nicer errors than wgpu's
/// internal panic path), then create the render pipeline.
fn build_effect_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    label: &str,
    effect_body: &str,
) -> Result<wgpu::RenderPipeline> {
    let source = compose_shader(effect_body);

    // naga pre-validation. Gives a clean error with file:line, without
    // device-level wgpu panic on a malformed module.
    naga::front::wgsl::parse_str(&source).map_err(|e| {
        anyhow::anyhow!("WGSL parse failure in {label}: {}", e.emit_to_string(&source))
    })?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("effect shader [{label}]")),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    Ok(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("effect pipeline [{label}]")),
        layout: Some(pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: COMPOSITE_FORMAT,
                // Additive blending — the load-bearing decision behind the
                // whole "scene-aware additive projection-mapping" thesis
                // (architecture review v1 #1). Effects return premultiplied
                // RGBA — i.e. `vec4(rgb * a, a)` — and the GPU accumulates
                // `dst += src` so dark pixels stay dark on the projector and
                // stacked layers genuinely add light. The Rgba16Float
                // composite (see `COMPOSITE_FORMAT`) makes overdrive safe;
                // the final sRGB swapchain write clamps to [0, 1].
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    }))
}

pub fn compose_shader(effect_body: &str) -> String {
    let mut s = String::with_capacity(PRELUDE_WGSL.len() + effect_body.len() + MAIN_WGSL.len() + 4);
    s.push_str(PRELUDE_WGSL);
    s.push('\n');
    s.push_str(effect_body);
    s.push('\n');
    s.push_str(MAIN_WGSL);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Rust struct must byte-match the WGSL `LayerParams` uniform:
    /// 16 (u32 header) + 8 (seed + pad) + 8 (centroid vec2) + 16 (bbox vec4)
    /// + 32 (params_f) + 64 (params_c) = 144, and 16-aligned throughout.
    #[test]
    fn layer_params_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<LayerParamsGpu>(), 144);
        assert_eq!(std::mem::offset_of!(LayerParamsGpu, centroid_uv), 24);
        assert_eq!(std::mem::offset_of!(LayerParamsGpu, bbox_uv), 32);
        assert_eq!(std::mem::offset_of!(LayerParamsGpu, params_f), 48);
        assert_eq!(std::mem::offset_of!(LayerParamsGpu, params_c), 80);
    }

    /// §5.14 — the bake uniform must byte-match `WarpBake` in
    /// `warp_bake.wgsl`: 3 padded rows (48) + meta vec4 (16) + two
    /// vec4[MAX_POINTS] arrays, 16-aligned throughout.
    #[test]
    fn warp_bake_layout_matches_wgsl() {
        assert_eq!(std::mem::offset_of!(WarpBakeUniforms, counts), 48);
        assert_eq!(std::mem::offset_of!(WarpBakeUniforms, points), 64);
        assert_eq!(
            std::mem::offset_of!(WarpBakeUniforms, weights),
            64 + 16 * MAX_WARP_POINTS
        );
        assert_eq!(
            std::mem::size_of::<WarpBakeUniforms>(),
            64 + 32 * MAX_WARP_POINTS
        );
        // The handle cap is asserted in three places; keep them one number.
        assert_eq!(MAX_WARP_POINTS, crate::alignment::MAX_POINTS);
        assert!(WARP_BAKE_WGSL.contains("const MAX_POINTS: u32 = 64u;"));
    }

    /// The output-side shaders only ever compile on a real device, which is
    /// exactly where a typo is most expensive to discover (mid-show, on a
    /// ladder). naga parses + validates them here for free.
    #[test]
    fn output_shaders_parse_and_validate() {
        for (name, src) in [
            ("final_pass", FINAL_PASS_WGSL),
            ("warp_bake", WARP_BAKE_WGSL),
        ] {
            let composed = compose_output_shader(src);
            let module = naga::front::wgsl::parse_str(&composed)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&composed)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} failed validation: {e:?}"));
        }
    }

    /// The bake pass must agree with `alignment::AlignmentSolution::warp` at
    /// every pixel — same projective base, same kernel, and crucially the
    /// **same dest-uv convention**. A Y-flip mismatch between the two renders
    /// as a vertically mirrored warp, which looks plausible enough on a wall
    /// to cost an evening; this catches it on the desk instead.
    ///
    /// Skips (rather than fails) when no adapter is available, so the suite
    /// still runs on a headless box.
    #[test]
    fn baked_lut_matches_the_cpu_model() {
        use crate::alignment::{AlignmentDoc, WarpPoint};

        const W: u32 = 64;
        const H: u32 = 48;

        let Some((device, queue)) = test_device() else {
            eprintln!("no GPU device — skipping baked_lut_matches_the_cpu_model");
            return;
        };

        // Top-vs-bottom asymmetric keystone plus an off-centre handle, so
        // neither a Y flip nor a dropped residual can pass unnoticed.
        let solution = crate::alignment::solve(AlignmentDoc {
            corners: [[0.12, 0.02], [0.94, 0.10], [0.99, 0.88], [0.03, 0.97]],
            points: vec![WarpPoint {
                id: "p1".into(),
                anchor: [0.5, 0.5],
                dest: [0.55, 0.35],
                radius: 0.3,
            }],
            ..Default::default()
        })
        .expect("test doc solves");

        let warp = create_warp_target(&device, W, H);
        queue.write_buffer(
            &warp.buffer,
            0,
            bytemuck::bytes_of(&solution.bake_uniforms()),
        );

        // Rg32Float = 8 B/px; 64 px → 512 B/row, already 256-aligned.
        let bytes_per_row = W * 8;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warp LUT readback"),
            size: (bytes_per_row * H) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("warp bake test pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &warp.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&warp.pipeline);
            pass.set_bind_group(0, &warp.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &warp.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(H),
                },
            },
            wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map callback").expect("map succeeded");
        let data = slice.get_mapped_range();
        let texels: &[f32] = bytemuck::cast_slice(&data);

        // The fullscreen triangle interpolates dest uv at the pixel centre,
        // with uv.y increasing downward — same direction as the framebuffer
        // row index. That equivalence is the thing under test.
        let mut worst = 0.0f32;
        for y in 0..H {
            for x in 0..W {
                let uv = [
                    (x as f32 + 0.5) / W as f32,
                    (y as f32 + 0.5) / H as f32,
                ];
                let want = solution.warp(uv);
                let i = ((y * W + x) * 2) as usize;
                let got = [texels[i] + uv[0], texels[i + 1] + uv[1]];
                worst = worst
                    .max((got[0] - want[0]).abs())
                    .max((got[1] - want[1]).abs());
            }
        }
        drop(data);
        readback.unmap();
        assert!(
            worst < 2e-4,
            "baked LUT diverges from the CPU model by {worst} in source uv"
        );
    }

    /// Headless adapter+device, or `None` on a machine without one.
    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))?;
        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("render-core test device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()
    }

    /// f32 → f16 bits. Exact for the dyadic values this test uploads.
    fn f16(v: f32) -> u16 {
        let x = v.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
        let mant = ((x >> 13) & 0x03ff) as u16;
        if exp <= 0 {
            sign
        } else if exp >= 31 {
            sign | 0x7c00
        } else {
            sign | ((exp as u16) << 10) | mant
        }
    }

    /// The other half of the Y-flip trap: `baked_lut_matches_the_cpu_model`
    /// proves the LUT holds the right offsets, this proves the **final pass
    /// reads them at the right texel and in the right direction** — and that
    /// out-of-source pixels get the background rather than a wrapped sample.
    ///
    /// The warp under test is a pure +0.25 translation in dest space, so every
    /// output pixel must show the composite from exactly 16 texels to its left
    /// (at 64 px wide) and the left quarter must be pure background. A
    /// vertical flip anywhere in the chain moves the sampled row and fails.
    #[test]
    fn final_pass_samples_through_the_warp() {
        use crate::alignment::AlignmentDoc;

        const N: u32 = 64;
        const SHIFT: u32 = 16; // 0.25 × 64

        let Some((device, queue)) = test_device() else {
            eprintln!("no GPU device — skipping final_pass_samples_through_the_warp");
            return;
        };

        // Composite: texel (x, y) carries its own coordinates as (r, g), so a
        // sample landing on the wrong texel is self-identifying.
        let composite = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test composite"),
            size: wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COMPOSITE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut texels = Vec::with_capacity((N * N * 4) as usize);
        for y in 0..N {
            for x in 0..N {
                texels.push(f16(x as f32 / N as f32));
                texels.push(f16(y as f32 / N as f32));
                texels.push(f16(0.0));
                texels.push(f16(1.0));
            }
        }
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &composite,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texels),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(N * 8),
                rows_per_image: Some(N),
            },
            wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
        );
        let composite_view = composite.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Corners translated +0.25 in x ⇒ W(dest) = dest − (0.25, 0).
        let solution = crate::alignment::solve(AlignmentDoc {
            corners: [[0.25, 0.0], [1.25, 0.0], [1.25, 1.0], [0.25, 1.0]],
            background: "#ffffff".into(),
            ..Default::default()
        })
        .expect("translation solves");

        let warp = create_warp_target(&device, N, N);
        queue.write_buffer(&warp.buffer, 0, bytemuck::bytes_of(&solution.bake_uniforms()));

        // Non-sRGB target so the readback is the shader's own linear values.
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test output"),
            size: wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());

        let (bgl, pipeline) = create_final_pipeline(&device, wgpu::TextureFormat::Rgba8Unorm);
        let uniforms = FinalPassUniforms::new(1.0, 1.0, 0.0)
            .with_warp(true, solution.background_linear);
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test final uniform"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = create_final_bind_group(
            &device,
            &bgl,
            "test final bind group",
            &composite_view,
            &sampler,
            &ubuf,
            &composite_view, // mix is 0
            &warp.view,
        );

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test readback"),
            size: (N * N * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test bake"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &warp.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&warp.pipeline);
            pass.set_bind_group(0, &warp.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test final"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(N * 4),
                    rows_per_image: Some(N),
                },
            },
            wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map callback").expect("map succeeded");
        let data = slice.get_mapped_range();

        let px = |x: u32, y: u32| -> [f32; 3] {
            let i = ((y * N + x) * 4) as usize;
            [
                data[i] as f32 / 255.0,
                data[i + 1] as f32 / 255.0,
                data[i + 2] as f32 / 255.0,
            ]
        };
        // Sampled region: shifted left by exactly SHIFT, same row.
        for y in [1u32, 7, 31, 62] {
            for x in [SHIFT, SHIFT + 5, N - 1] {
                let got = px(x, y);
                let want_r = (x - SHIFT) as f32 / N as f32;
                let want_g = y as f32 / N as f32;
                assert!(
                    (got[0] - want_r).abs() < 0.01,
                    "({x},{y}) column: got r={} want {want_r}",
                    got[0]
                );
                assert!(
                    (got[1] - want_g).abs() < 0.01,
                    "({x},{y}) row (Y flip?): got g={} want {want_g}",
                    got[1]
                );
            }
        }
        // Left quarter maps outside the composite ⇒ background, not a wrap.
        for y in [0u32, 30, 63] {
            for x in [0u32, 5, SHIFT - 2] {
                let got = px(x, y);
                assert!(
                    got[0] > 0.98 && got[1] > 0.98 && got[2] > 0.98,
                    "({x},{y}) should be background white, got {got:?}"
                );
            }
        }
        drop(data);
        readback.unmap();
    }

    #[test]
    fn final_pass_uniform_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<FinalPassUniforms>(), 48);
        assert_eq!(std::mem::offset_of!(FinalPassUniforms, background), 16);
        assert_eq!(std::mem::offset_of!(FinalPassUniforms, pattern), 32);
        // The preview blit must never enable the warp (§3.4) — it binds a
        // 1×1 dummy LUT and would index it out of bounds otherwise.
        assert_eq!(FinalPassUniforms::identity().adjust[3], 0.0);
        assert_eq!(FinalPassUniforms::new(0.5, 1.0, 0.0).adjust[3], 0.0);
    }
}

fn create_final_pipeline(
    device: &wgpu::Device,
    swapchain_format: wgpu::TextureFormat,
) -> (wgpu::BindGroupLayout, wgpu::RenderPipeline) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("final pass bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // §5.6 — the design-leg composite (or the live view again in
            // single-leg mode / preview blits, where mix is 0).
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // §5.14 — the baked offset LUT. Non-filterable on purpose: it is
            // only ever `textureLoad`ed, which is what keeps a 32-bit float
            // format on core wgpu.
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let source = compose_output_shader(FINAL_PASS_WGSL);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("final pass shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("final pass pipeline layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("final pass pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: swapchain_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });
    (bgl, pipeline)
}

#[allow(clippy::too_many_arguments)]
fn create_final_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    live_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniforms: &wgpu::Buffer,
    design_view: &wgpu::TextureView,
    warp_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(live_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(design_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(warp_view),
            },
        ],
    })
}

/// §5.14 — allocate the offset LUT at the output's exact size, plus its bake
/// pipeline and uniform buffer. Contents are undefined until the first bake,
/// which is why `needs_bake` starts true.
fn create_warp_target(device: &wgpu::Device, width: u32, height: u32) -> WarpTarget {
    let width = width.max(1);
    let height = height.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("warp offset LUT"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: WARP_LUT_FORMAT,
        // COPY_SRC so the baked field can be read back — used by the
        // bake-matches-the-CPU-model test, and the obvious debug hook if a
        // camera-driven field ever needs inspecting.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("warp bake bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warp bake uniform"),
        contents: bytemuck::bytes_of(&WarpBakeUniforms::zeroed()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("warp bake bind group"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let source = compose_output_shader(WARP_BAKE_WGSL);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("warp bake shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("warp bake pipeline layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("warp bake pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: WARP_LUT_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    WarpTarget {
        texture,
        view,
        pipeline,
        bind_group,
        buffer,
        width,
        height,
        needs_bake: std::cell::Cell::new(true),
    }
}

/// The 1×1 zero LUT bound wherever the warp must be off (§3.4).
fn create_warp_dummy(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("warp LUT dummy (1x1 zero)"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: WARP_LUT_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0u8; 8],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
