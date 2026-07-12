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
const HOMOGRAPHY_WGSL: &str = include_str!("shaders/homography.wgsl");

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

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct HomographyUniforms {
    /// Three rows of the 3×3, each padded to a vec4 for std140.
    pub rows: [[f32; 4]; 3],
    /// §5.4 output masters: (brightness, saturation, _, _). Applied in the
    /// final pass — the composite (and therefore the operator preview) stays
    /// un-mastered; only the projector output is scaled.
    pub adjust: [f32; 4],
}

impl HomographyUniforms {
    pub fn new(m: Option<[[f32; 3]; 3]>, brightness: f32, saturation: f32, mix: f32) -> Self {
        let rows = match m {
            Some(m) => [
                [m[0][0], m[0][1], m[0][2], 0.0],
                [m[1][0], m[1][1], m[1][2], 0.0],
                [m[2][0], m[2][1], m[2][2], 0.0],
            ],
            None => [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        };
        Self {
            rows,
            adjust: [brightness, saturation, mix, 0.0],
        }
    }

    pub fn identity() -> Self {
        Self::new(None, 1.0, 1.0, 0.0)
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
/// preview. Runs the homography pipeline with an identity matrix (§5.6:
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
    /// Identity-matrix uniforms, rewritten per present so the LIVE position
    /// tracks the masters and the DESIGN position stays neutral.
    buffer: wgpu::Buffer,
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

    /// Held to keep the wgpu resource alive — bind group referencing it stays valid.
    #[allow(dead_code)]
    pub homography_bind_group_layout: wgpu::BindGroupLayout,
    pub homography_pipeline: wgpu::RenderPipeline,
    pub homography_buffer: wgpu::Buffer,
    pub homography_bind_group: wgpu::BindGroup,
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

        let (homography_bind_group_layout, homography_pipeline) =
            create_homography_pipeline(&device, surface_config.format);

        let homography_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("homography uniform"),
            contents: bytemuck::bytes_of(&HomographyUniforms::identity()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let homography_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("homography bind group"),
            layout: &homography_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composite_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&composite_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: homography_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        design_view.as_ref().unwrap_or(&composite_view),
                    ),
                },
            ],
        });

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
            homography_bind_group_layout,
            homography_pipeline,
            homography_buffer,
            homography_bind_group,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
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
        // preview surface's own format; identity matrix, masters written per
        // present (real on LIVE, neutral on DESIGN — §5.6 source toggle).
        let (_, pipeline) = create_homography_pipeline(&self.device, format);
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("preview blit uniform"),
            contents: bytemuck::bytes_of(&HomographyUniforms::identity()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let make_group = |label: &str, view: &wgpu::TextureView| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.homography_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer.as_entire_binding(),
                    },
                    // mix is always 0 on the preview blit; bind the source
                    // view again to satisfy the layout.
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                ],
            })
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
            Leg::Live => HomographyUniforms::new(None, brightness, saturation, 0.0),
            Leg::Design => HomographyUniforms::identity(),
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

    /// Refresh the final-pass uniform: calibration matrix + the §5.4 output
    /// masters + the §5.6 promote mix. Written once per presented frame (the
    /// buffer is 64 bytes; a dirty flag would cost more complexity).
    pub fn write_homography(
        &self,
        m: Option<[[f32; 3]; 3]>,
        brightness: f32,
        saturation: f32,
        mix: f32,
    ) {
        let uniforms = HomographyUniforms::new(m, brightness, saturation, mix);
        self.queue
            .write_buffer(&self.homography_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Encode the final homography pass (live composite × design composite
    /// lerped by the promote mix → masters → warp) onto `target_view`.
    /// Pulled out of `PassPlan` so the §5.6 frame orchestration in `Core`
    /// owns pass ordering: live composite, design composite (on demand),
    /// then this.
    pub fn encode_final(&self, encoder: &mut wgpu::CommandEncoder, target_view: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("homography pass"),
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
        pass.set_pipeline(&self.homography_pipeline);
        pass.set_bind_group(0, &self.homography_bind_group, &[]);
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
}

fn create_homography_pipeline(
    device: &wgpu::Device,
    swapchain_format: wgpu::TextureFormat,
) -> (wgpu::BindGroupLayout, wgpu::RenderPipeline) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("homography bind group layout"),
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
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("homography shader"),
        source: wgpu::ShaderSource::Wgsl(HOMOGRAPHY_WGSL.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("homography pipeline layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("homography pipeline"),
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
