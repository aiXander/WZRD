//! wgpu state: output target (window surface or offscreen), device, queue,
//! mask-atlas texture, pipelines.
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

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
const DOWNSCALE_WGSL: &str = include_str!("shaders/downscale.wgsl");
/// Studio file preview (P1) — one media file, contain-fit, no leg.
const FILE_PREVIEW_WGSL: &str = include_str!("shaders/file_preview.wgsl");

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
    /// Operator layer fader — see `LayerOpacity` in the compositor.
    pub opacity: f32,
    pub centroid_uv: [f32; 2],
    pub bbox_uv: [f32; 4],
    /// 8 scalar slots packed as two vec4 lanes.
    pub params_f: [[f32; 4]; 2],
    pub params_c: [[f32; 4]; MAX_COLOR_PARAMS],
    /// C5 image placement rect, uv (min_x, min_y, max_x, max_y).
    pub image_box: [f32; 4],
    /// (bound, encoding, 1 / box width, 1 / box height) — see
    /// `ImageEncoding`; the reciprocals spare `sample_image` a divide per tap.
    pub image_meta: [f32; 4],
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
            opacity: 1.0,
            centroid_uv: identity.centroid_uv,
            bbox_uv: identity.bbox_uv,
            params_f,
            params_c,
            image_box: [0.0, 0.0, 1.0, 1.0],
            image_meta: [0.0; 4],
        }
    }

    /// Bind this pass's image: placement rect (uv) + texel encoding.
    pub fn with_image(mut self, image_box: [f32; 4], encoding: ImageEncoding) -> Self {
        self.image_box = image_box;
        let [x0, y0, x1, y1] = image_box;
        self.image_meta = [
            1.0,
            encoding.as_f32(),
            1.0 / (x1 - x0).max(1e-6),
            1.0 / (y1 - y0).max(1e-6),
        ];
        self
    }
}

/// How `sample_image` must interpret the texels of a bound image. The
/// texture *format* already handles sRGB decode for stills and plain HAP;
/// HAP Q's scaled YCoCg needs a colour convert in the shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageEncoding {
    /// sRGB-format texture, straight alpha (PNG/JPEG stills, HAP, HAP Alpha).
    Rgba,
    /// Scaled YCoCg in a non-sRGB BC3 texture (HAP Q).
    YCoCg,
}

impl ImageEncoding {
    fn as_f32(self) -> f32 {
        match self {
            ImageEncoding::Rgba => 0.0,
            ImageEncoding::YCoCg => 1.0,
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

/// What a preview blit samples (docs/TODO/native-preview-underlay.md D3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewInput {
    /// One leg's composite, no warp (Perform's monitor, Prepare).
    Leg(Leg),
    /// The projector's own final pass — live × design promote mix, the real
    /// warp LUT (indexed by uv), test pattern (Align).
    Program,
}

/// `fs_preview`'s group-1 uniform: (dim, photo_weight, light_weight, _).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PreviewExtraUniforms {
    pub weights: [f32; 4],
}

impl PreviewExtraUniforms {
    pub fn new(dim: f32, photo_weight: f32, light_weight: f32) -> Self {
        Self {
            weights: [dim, photo_weight, light_weight, 0.0],
        }
    }
}

/// `fs_file`'s uniform (Studio file preview, P1): where the image lands in
/// dest uv (contain-fit) and how to read it — mirrors `FileUniforms` in
/// `file_preview.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FilePreviewUniforms {
    pub image_box: [f32; 4],
    /// (bound, encoding, 1 / box width, 1 / box height).
    pub image_meta: [f32; 4],
}

impl FilePreviewUniforms {
    /// Nothing open: the slot shows black.
    pub fn empty() -> Self {
        Self { image_box: [0.0; 4], image_meta: [0.0; 4] }
    }

    /// An `image` (w×h px) contain-fitted into a `target` (w×h px) drawable.
    pub fn contain(image: (u32, u32), target: (u32, u32), encoding: ImageEncoding) -> Self {
        let ia = image.0.max(1) as f32 / image.1.max(1) as f32;
        let ta = target.0.max(1) as f32 / target.1.max(1) as f32;
        let (bw, bh) = if ia > ta { (1.0, ta / ia) } else { (ia / ta, 1.0) };
        let (x0, y0) = ((1.0 - bw) * 0.5, (1.0 - bh) * 0.5);
        Self {
            image_box: [x0, y0, x0 + bw, y0 + bh],
            image_meta: [1.0, encoding.as_f32(), 1.0 / bw.max(1e-6), 1.0 / bh.max(1e-6)],
        }
    }
}

/// What a preview present draws: a leg / the program view through
/// `fs_preview`, or (Studio P1) the file preview's bind group through
/// `fs_file` (`None` = nothing open → black).
#[derive(Clone, Copy)]
pub enum PreviewDraw<'a> {
    View(PreviewInput),
    File(Option<&'a wgpu::BindGroup>),
}

/// Where the preview blit lands: the underlay's swapchain, or (§2.10 bench)
/// an offscreen texture so a windowless run pays the same blit + decode.
enum PreviewOut {
    Surface(wgpu::Surface<'static>),
    Offscreen {
        texture: wgpu::Texture,
        view: Arc<wgpu::TextureView>,
    },
}

/// One acquired preview image (see [`PreviewOut`]).
enum PreviewFrame {
    Surface {
        frame: wgpu::SurfaceTexture,
        view: wgpu::TextureView,
    },
    Offscreen(Arc<wgpu::TextureView>),
}

impl PreviewFrame {
    fn view(&self) -> &wgpu::TextureView {
        match self {
            PreviewFrame::Surface { view, .. } => view,
            PreviewFrame::Offscreen(v) => v,
        }
    }

    fn present(self) {
        if let PreviewFrame::Surface { frame, .. } = self {
            frame.present();
        }
    }
}

/// Native operator-preview swapchain: a second surface the same device
/// blits onto — the Resolume-style preview. Runs `fs_preview`, the
/// final-pass math plus a preview-only group 1 (surface photo + weights).
///
/// Bind groups: one per leg (dummy 1×1 LUT, warp off — the §5.6 source
/// toggle: LIVE renders with the real masters, DESIGN un-mastered) and one
/// for the program view (the real LUT, which `fs_preview` indexes by uv
/// because this target is never the LUT's size — D4).
pub struct PreviewTarget {
    out: PreviewOut,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    /// Studio file preview (P1): `fs_file` + its group 0 layout + uniform.
    file_pipeline: wgpu::RenderPipeline,
    file_layout: wgpu::BindGroupLayout,
    file_buffer: wgpu::Buffer,
    bind_group_live: wgpu::BindGroup,
    bind_group_design: wgpu::BindGroup,
    /// Rebuilt whenever the warp LUT is reallocated (`resize_warp`).
    bind_group_program: wgpu::BindGroup,
    /// `FinalPassUniforms`, rewritten per present from the caller's view.
    buffer: wgpu::Buffer,
    extra_layout: wgpu::BindGroupLayout,
    extra_buffer: wgpu::Buffer,
    /// Photo + weights; rebuilt when the surface photo changes.
    extra_bind_group: wgpu::BindGroup,
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

/// Offscreen output format: what a Metal/Vulkan swapchain picks (the first
/// sRGB format), so the final pass compiles and costs the same as on a
/// window.
const OFFSCREEN_OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
/// Submits allowed in flight offscreen — the swapchain's
/// `desired_maximum_frame_latency`.
const OFFSCREEN_FRAME_LATENCY: usize = 2;

/// The projector final pass's destination.
enum Output {
    /// A window swapchain (winit / Tauri hosts).
    Window(wgpu::Surface<'static>),
    /// No window (perf-monitor PM3 `--bench`).
    Offscreen(OffscreenOutput),
}

struct OffscreenOutput {
    texture: wgpu::Texture,
    /// Shared with the acquired [`OutputFrame`] (a refcount bump per frame,
    /// no allocation), so a frame in hand borrows nothing from the context
    /// and `Core` can tick its legs *after* `acquire` (§4 #9).
    view: Arc<wgpu::TextureView>,
    /// Submits that drew into it and may still be on the GPU, oldest first.
    /// Without a swapchain nothing else stops the CPU from queueing
    /// unbounded work, and an uncapped bench would measure the queue.
    in_flight: RefCell<VecDeque<wgpu::SubmissionIndex>>,
    /// `--paced`: the virtual vsync period and the next tick.
    vsync: Option<(Duration, Cell<Option<Instant>>)>,
}

impl OffscreenOutput {
    fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, vsync_hz: Option<f64>) -> Self {
        let (texture, view) = create_offscreen_output(device, config);
        Self {
            texture,
            view: Arc::new(view),
            in_flight: RefCell::new(VecDeque::new()),
            vsync: vsync_hz
                .filter(|hz| *hz > 0.0)
                .map(|hz| (Duration::from_secs_f64(1.0 / hz), Cell::new(None))),
        }
    }

    fn resize(&mut self, device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) {
        let (texture, view) = create_offscreen_output(device, config);
        self.texture = texture;
        self.view = Arc::new(view);
    }

    /// Block until this frame may start drawing: the in-flight cap, then
    /// (paced) the next vsync tick on a fixed grid — a late frame snaps to
    /// the next tick, as a real swapchain would.
    ///
    /// `timed` (the perf GPU timer samples this frame): wait for the queue
    /// to drain first, so the frame runs **alone** on the GPU. With two
    /// frames in flight Apple's GPU starts a frame's first pass while the
    /// previous frame is still drawing, and the per-submit span then
    /// counts that wait — measured: smoke 3.9 ms at two in flight, 6.4 ms
    /// at three, 2.4 ms isolated, at an identical 2.6 ms/frame throughput.
    /// Untimed frames keep the queue full, so the GPU stays clocked up
    /// (draining *every* frame lets it downclock: 4.5 ms).
    fn wait_turn(&self, device: &wgpu::Device, timed: bool) {
        let cap = if timed { 1 } else { OFFSCREEN_FRAME_LATENCY };
        let mut q = self.in_flight.borrow_mut();
        while q.len() >= cap {
            if let Some(idx) = q.pop_front() {
                device.poll(wgpu::Maintain::WaitForSubmissionIndex(idx));
            }
        }
        drop(q);
        if let Some((period, next)) = &self.vsync {
            let now = Instant::now();
            let mut tick = next.get().unwrap_or(now);
            while tick < now {
                tick += *period;
            }
            if tick > now {
                std::thread::sleep(tick - now);
            }
            next.set(Some(tick + *period));
        }
    }
}

fn create_offscreen_output(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen output"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn new_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    })
}

/// One acquired output frame (see [`GpuContext::acquire_output`]). Owns
/// what it needs — it borrows nothing from the `GpuContext`, so the
/// caller can mutate the rest of its state between acquire and present.
pub enum OutputFrame {
    Swapchain {
        frame: wgpu::SurfaceTexture,
        view: wgpu::TextureView,
    },
    Offscreen {
        view: Arc<wgpu::TextureView>,
    },
}

impl OutputFrame {
    pub fn view(&self) -> &wgpu::TextureView {
        match self {
            OutputFrame::Swapchain { view, .. } => view,
            OutputFrame::Offscreen { view } => view,
        }
    }
}

pub struct GpuContext {
    /// Kept so additional surfaces (the preview) can be created against the
    /// same instance/adapter the device came from.
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    /// Where the projector final pass lands — the window's swapchain, or
    /// (perf-monitor PM3 `--bench`) an offscreen texture. See [`Output`].
    output: Output,
    /// The output's format + size. For a window it's the live swapchain
    /// configuration; offscreen only `format`/`width`/`height` mean anything.
    pub surface_config: wgpu::SurfaceConfiguration,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Whether the device was created with `TEXTURE_COMPRESSION_BC` (HAP
    /// clips need it; stills don't).
    pub bc_supported: bool,
    /// Whether the device was created with `TIMESTAMP_QUERY` (perf PM2 GPU
    /// columns) — only when asked for *and* offered.
    pub timestamps_supported: bool,

    /// Attached/detached by the host at runtime (None headless / pre-attach).
    pub preview: Option<PreviewTarget>,
    /// The pack's surface photo for the preview's PHOTO/BOTH backdrops —
    /// 1×1 black until `set_surface_photo`.
    photo_texture: wgpu::Texture,
    photo_view: wgpu::TextureView,

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

    /// C5 image input: the shared bilinear clamp sampler, and the 1×1
    /// transparent texture every pass without an image binds in slot 4.
    pub image_sampler: wgpu::Sampler,
    pub dummy_image_view: wgpu::TextureView,

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
    ///
    /// `want_timestamps`: request `TIMESTAMP_QUERY` for the perf monitor's
    /// GPU columns (if the adapter offers it). `false` keeps the feature set
    /// exactly as without PM2.
    pub async fn new(
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
        pack: &LoadedPack,
        two_leg: bool,
        want_timestamps: bool,
    ) -> Result<Self> {
        let instance = new_instance();
        let surface = instance
            .create_surface(target)
            .context("creating wgpu surface for window")?;
        Self::build(instance, Some(surface), width, height, None, pack, two_leg, want_timestamps).await
    }

    /// Perf-monitor PM3 `--bench`: no window and no surface — the adapter is
    /// requested with `compatible_surface: None` and the final pass renders
    /// into an offscreen `width × height` texture in the format a swapchain
    /// would pick. `vsync_hz`: `Some` = `--paced` (acquire waits for a
    /// virtual vsync, like a real swapchain); `None` = uncapped.
    pub async fn new_offscreen(
        width: u32,
        height: u32,
        vsync_hz: Option<f64>,
        pack: &LoadedPack,
        two_leg: bool,
        want_timestamps: bool,
    ) -> Result<Self> {
        Self::build(new_instance(), None, width, height, vsync_hz, pack, two_leg, want_timestamps).await
    }

    #[allow(clippy::too_many_arguments)]
    async fn build(
        instance: wgpu::Instance,
        surface: Option<wgpu::Surface<'static>>,
        width: u32,
        height: u32,
        vsync_hz: Option<f64>,
        pack: &LoadedPack,
        two_leg: bool,
        want_timestamps: bool,
    ) -> Result<Self> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            })
            .await
            .context("no compatible GPU adapter found")?;

        // The one exception to `Features::empty()` (gig plan D2, CLAUDE.md):
        // BC texture compression for HAP clips — requested only when the
        // adapter offers it (Apple Silicon, desktop NVIDIA/AMD/Intel all do).
        // Without it the engine runs as before and refuses HAP clips with a
        // named per-binding warning.
        let bc_supported = adapter
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
        if !bc_supported {
            log::warn!("GPU adapter lacks TEXTURE_COMPRESSION_BC — HAP clips will not play");
        }
        // The second exception (perf-monitor PM2): timestamp queries for the
        // `perf` channel's per-pass GPU columns — requested only when the
        // perf monitor wants them and the adapter offers them (Apple Silicon
        // at pass boundaries; Vulkan/NVIDIA). Without it `perf.gpu` is null.
        let timestamps_supported = want_timestamps
            && adapter.features().contains(crate::gpu_timing::GpuTimer::FEATURES);
        if want_timestamps && !timestamps_supported {
            log::info!("GPU adapter lacks TIMESTAMP_QUERY — perf GPU columns stay null");
        }
        let mut required_features = wgpu::Features::empty();
        if bc_supported {
            required_features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
        }
        if timestamps_supported {
            required_features |= crate::gpu_timing::GpuTimer::FEATURES;
        }
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("render-core device"),
                    required_features,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("requesting wgpu device")?;

        let (output, surface_config) = match surface {
            Some(surface) => {
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
                (Output::Window(surface), surface_config)
            }
            None => {
                let config = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: OFFSCREEN_OUTPUT_FORMAT,
                    width: width.max(1),
                    height: height.max(1),
                    present_mode: wgpu::PresentMode::AutoVsync,
                    alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                };
                let output = Output::Offscreen(OffscreenOutput::new(&device, &config, vsync_hz));
                (output, config)
            }
        };

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

        let image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("image input sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let dummy_image_view = create_dummy_image(&device, &queue);

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

        let (photo_texture, photo_view) = create_photo_texture(&device, &queue, 1, 1, &[0, 0, 0, 255]);

        Ok(Self {
            instance,
            adapter,
            output,
            surface_config,
            device,
            queue,
            bc_supported,
            timestamps_supported,
            preview: None,
            photo_texture,
            photo_view,
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
            image_sampler,
            dummy_image_view,
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
        match &mut self.output {
            Output::Window(surface) => surface.configure(&self.device, &self.surface_config),
            Output::Offscreen(o) => o.resize(&self.device, &self.surface_config),
        }
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
        if let Some(pv) = self.preview.as_mut() {
            pv.bind_group_program = create_final_bind_group(
                &self.device,
                &self.final_bind_group_layout,
                "preview bind group (program)",
                &self.composite_view,
                &self.composite_sampler,
                &pv.buffer,
                self.design_view.as_ref().unwrap_or(&self.composite_view),
                &self.warp.view,
            );
        }
    }

    /// Upload the pack's surface photo (tightly packed RGBA8, sRGB-encoded)
    /// for the preview's photo weight. Sampled as `Rgba8UnormSrgb`, so the
    /// shader sees linear light.
    pub fn set_surface_photo(&mut self, rgba: &[u8], width: u32, height: u32) {
        let (texture, view) = create_photo_texture(&self.device, &self.queue, width, height, rgba);
        self.photo_texture = texture;
        self.photo_view = view;
        if let Some(pv) = self.preview.as_mut() {
            pv.extra_bind_group = create_preview_extra_bind_group(
                &self.device,
                &pv.extra_layout,
                &self.photo_view,
                &pv.extra_buffer,
            );
        }
    }

    pub fn warp_size(&self) -> (u32, u32) {
        (self.warp.width, self.warp.height)
    }

    /// Perf-monitor PM1 — for `Instance::generate_report` (live resource
    /// counts per type, the leak signal).
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    pub fn backend(&self) -> wgpu::Backend {
        self.adapter.get_info().backend
    }

    /// Perf-monitor PM1 — bytes of the long-lived textures/buffers this
    /// context owns, sized from their descriptors. Swapchains are estimated
    /// (format × size × 3 images — the driver owns those allocations).
    pub fn owned_bytes(&self) -> u64 {
        use crate::perf::texture_bytes;
        let swapchain = |c: &wgpu::SurfaceConfiguration| {
            let block = c.format.block_copy_size(None).unwrap_or(4) as u64;
            c.width as u64 * c.height as u64 * block * 3
        };
        let mut total = texture_bytes(&self.composite_texture)
            + self.design_texture.as_ref().map_or(0, texture_bytes)
            + texture_bytes(&self.mask_atlas)
            + texture_bytes(&self.warp.texture)
            + self.warp.buffer.size()
            + self.final_buffer.size()
            + texture_bytes(&self.photo_texture)
            + match &self.output {
                Output::Window(_) => swapchain(&self.surface_config),
                Output::Offscreen(o) => texture_bytes(&o.texture),
            };
        if let Some(p) = self.preview.as_ref() {
            total += match &p.out {
                PreviewOut::Surface(_) => swapchain(&p.config),
                PreviewOut::Offscreen { texture, .. } => texture_bytes(texture),
            } + p.buffer.size()
                + p.extra_buffer.size()
                + p.file_buffer.size();
        }
        total
    }

    /// The frame the final pass draws into: the swapchain's next image, or
    /// the offscreen output. Offscreen, this is where backpressure lives
    /// (charged to `acquire` like a vsync wait): at most two submits stay
    /// in flight (a swapchain's frame latency), a GPU-`timed` frame first
    /// waits for the queue to drain (see `OffscreenOutput::wait_turn`), and
    /// `--paced` waits for the next virtual vsync. `timed` is ignored on a
    /// window.
    pub fn acquire_output(&self, timed: bool) -> Result<OutputFrame, wgpu::SurfaceError> {
        match &self.output {
            Output::Window(surface) => {
                let frame = surface.get_current_texture()?;
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                Ok(OutputFrame::Swapchain { frame, view })
            }
            Output::Offscreen(o) => {
                o.wait_turn(&self.device, timed);
                Ok(OutputFrame::Offscreen { view: Arc::clone(&o.view) })
            }
        }
    }

    /// Window path, a GPU-`timed` frame only: block until every earlier
    /// submit has finished, right before this frame's main submit, so the
    /// sampled frame runs **alone** on the GPU — the swapchain twin of
    /// `OffscreenOutput::wait_turn`'s drain (which already ran offscreen).
    /// Without it a timed frame's first pass begins while the previous
    /// frame's tail still draws and its span counts that wait (perf-monitor
    /// "windowed GPU timing over-reads"). Called after encoding, so the
    /// wait only covers whatever GPU work is still left; untimed frames
    /// (29 in 30 by default) never block here. Returns whether it waited.
    pub fn drain_before_timed_submit(&self, timed: bool) -> bool {
        if timed && matches!(self.output, Output::Window(_)) {
            let _ = self.device.poll(wgpu::Maintain::Wait);
            true
        } else {
            false
        }
    }

    /// Present a frame from [`GpuContext::acquire_output`]; `submitted` is
    /// the submit that drew it (offscreen tracks it for the in-flight cap).
    pub fn present(&self, frame: OutputFrame, submitted: wgpu::SubmissionIndex) {
        match (frame, &self.output) {
            (OutputFrame::Swapchain { frame, .. }, _) => frame.present(),
            (OutputFrame::Offscreen { .. }, Output::Offscreen(o)) => {
                o.in_flight.borrow_mut().push_back(submitted);
            }
            (OutputFrame::Offscreen { .. }, Output::Window(_)) => {}
        }
    }

    /// True when there is no window surface (PM3 bench).
    pub fn is_offscreen(&self) -> bool {
        matches!(self.output, Output::Offscreen(_))
    }

    /// Adapter name + backend for the bench report's machine info.
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
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
    pub fn encode_warp_bake(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
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
                timestamp_writes,
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
        self.install_preview(PreviewOut::Surface(surface), config);
        log::info!("preview surface attached ({width}x{height}, {format:?})");
        Ok(())
    }

    /// §2.10 bench: a preview target with no surface — the blit lands in an
    /// offscreen `Rgba8UnormSrgb` texture, so a windowless run pays the
    /// same preview work (e.g. the Studio file preview's decode + blit).
    pub fn attach_preview_offscreen(&mut self, width: u32, height: u32) {
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        let (texture, view) = create_offscreen_output(&self.device, &config);
        self.install_preview(
            PreviewOut::Offscreen { texture, view: Arc::new(view) },
            config,
        );
        log::info!("offscreen preview target attached ({width}x{height})");
    }

    fn install_preview(&mut self, out: PreviewOut, config: wgpu::SurfaceConfiguration) {
        let format = config.format;
        // Same shader as the projector pass, `fs_preview` entry, compiled
        // for the preview surface's own format; uniforms written per present.
        let (extra_layout, pipeline) =
            create_preview_pipeline(&self.device, format, &self.final_bind_group_layout);
        let (file_layout, file_pipeline) = create_file_preview_pipeline(&self.device, format);
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("preview blit uniform"),
            contents: bytemuck::bytes_of(&FinalPassUniforms::identity()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let file_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("file preview uniform"),
            contents: bytemuck::bytes_of(&FilePreviewUniforms::empty()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let make_group = |label: &str, view: &wgpu::TextureView, mix_view: &wgpu::TextureView| {
            create_final_bind_group(
                &self.device,
                &self.final_bind_group_layout,
                label,
                view,
                &self.composite_sampler,
                &buffer,
                mix_view,
                // §5.14 — never the real LUT here (see `warp_dummy_view`).
                &self.warp_dummy_view,
            )
        };
        let design_view = self.design_view.as_ref().unwrap_or(&self.composite_view);
        // LIVE binds the design composite as its mix partner: during a TAKE
        // the caller passes the promote mix, so the PROGRAM monitor shows the
        // crossfade the wall gets (mix 0 otherwise — live only). DESIGN's
        // mix is always 0; it binds itself again to satisfy the layout.
        let bind_group_live =
            make_group("preview blit bind group (live)", &self.composite_view, design_view);
        let bind_group_design =
            make_group("preview blit bind group (design)", design_view, design_view);
        let bind_group_program = create_final_bind_group(
            &self.device,
            &self.final_bind_group_layout,
            "preview bind group (program)",
            &self.composite_view,
            &self.composite_sampler,
            &buffer,
            self.design_view.as_ref().unwrap_or(&self.composite_view),
            &self.warp.view,
        );
        let extra_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("preview extra uniform"),
            contents: bytemuck::bytes_of(&PreviewExtraUniforms::new(1.0, 0.0, 1.0)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let extra_bind_group =
            create_preview_extra_bind_group(&self.device, &extra_layout, &self.photo_view, &extra_buffer);
        self.preview = Some(PreviewTarget {
            out,
            config,
            pipeline,
            file_pipeline,
            file_layout,
            file_buffer,
            bind_group_live,
            bind_group_design,
            bind_group_program,
            buffer,
            extra_layout,
            extra_buffer,
            extra_bind_group,
        });
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
            match &mut pv.out {
                PreviewOut::Surface(surface) => surface.configure(&self.device, &pv.config),
                PreviewOut::Offscreen { texture, view } => {
                    let (t, v) = create_offscreen_output(&self.device, &pv.config);
                    *texture = t;
                    *view = Arc::new(v);
                }
            }
        }
    }

    /// The preview's next image. `None` = skip this frame (a swapchain that
    /// was `Lost`/`Outdated` got reconfigured).
    fn acquire_preview(pv: &PreviewTarget, device: &wgpu::Device) -> Result<Option<PreviewFrame>, wgpu::SurfaceError> {
        match &pv.out {
            PreviewOut::Surface(surface) => match surface.get_current_texture() {
                Ok(frame) => {
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    Ok(Some(PreviewFrame::Surface { frame, view }))
                }
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    surface.configure(device, &pv.config);
                    Ok(None)
                }
                Err(e) => Err(e),
            },
            PreviewOut::Offscreen { view, .. } => Ok(Some(PreviewFrame::Offscreen(Arc::clone(view)))),
        }
    }

    /// Blit onto the preview surface and present. Self-heals
    /// `Lost`/`Outdated` by reconfiguring (frame skipped); other errors are
    /// returned. No-op without an attached preview.
    ///
    /// The caller builds `uniforms` for `input`: a leg's own masters with
    /// the warp off for `Leg` (the per-leg groups bind a dummy LUT, so a
    /// warp flag there is cleared), the projector's exact uniforms for
    /// `Program`.
    ///
    /// §3.1 caveat: the *caller* must ensure the preview view is visible —
    /// `get_current_texture` on an occluded macOS window blocks the render
    /// thread just like the projector swapchain does.
    ///
    /// `write_uniforms = false` skips re-uploading `uniforms` / `extra`
    /// when the caller knows the buffers already hold them (each
    /// `write_buffer` is a staging allocation in wgpu-core 22).
    pub fn render_preview(
        &self,
        input: PreviewInput,
        mut uniforms: FinalPassUniforms,
        extra: PreviewExtraUniforms,
        write_uniforms: bool,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) -> Result<(), wgpu::SurfaceError> {
        let Some(pv) = self.preview.as_ref() else {
            return Ok(());
        };
        if write_uniforms {
            if input != PreviewInput::Program {
                uniforms.adjust[3] = 0.0;
            }
            self.queue
                .write_buffer(&pv.buffer, 0, bytemuck::bytes_of(&uniforms));
            self.queue
                .write_buffer(&pv.extra_buffer, 0, bytemuck::bytes_of(&extra));
        }
        let Some(frame) = Self::acquire_preview(pv, &self.device)? else {
            return Ok(());
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("preview blit encoder"),
            });
        Self::encode_preview_pass(pv, &mut encoder, frame.view(), PreviewDraw::View(input), timestamp_writes);
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }

    /// Studio file preview (P1): upload `source`'s frame at `playhead_sec`
    /// (a no-op for stills) and blit it contain-fit — never a leg. `file`
    /// `None` clears the slot to black. `uniforms` is written only when
    /// `Some` (the caller tracks changes). Returns whether a frame was
    /// submitted, so the caller knows to recycle the source's staging slots.
    pub fn render_preview_file(
        &self,
        file: Option<(&mut crate::video::ImageSource, &wgpu::BindGroup)>,
        playhead_sec: f64,
        uniforms: Option<FilePreviewUniforms>,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) -> Result<bool, wgpu::SurfaceError> {
        let Some(pv) = self.preview.as_ref() else {
            return Ok(false);
        };
        if let Some(u) = uniforms {
            self.queue.write_buffer(&pv.file_buffer, 0, bytemuck::bytes_of(&u));
        }
        let Some(frame) = Self::acquire_preview(pv, &self.device)? else {
            return Ok(false);
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("file preview encoder"),
            });
        let group = match file {
            Some((source, group)) => {
                source.encode_upload(&mut encoder, playhead_sec);
                Some(group)
            }
            None => None,
        };
        Self::encode_preview_pass(pv, &mut encoder, frame.view(), PreviewDraw::File(group), timestamp_writes);
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(true)
    }

    /// Bind group for the Studio file preview over one opened source's
    /// texture. `None` without an attached preview.
    pub fn file_preview_bind_group(&self, view: &wgpu::TextureView) -> Option<wgpu::BindGroup> {
        let pv = self.preview.as_ref()?;
        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("file preview bind group"),
            layout: &pv.file_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.image_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pv.file_buffer.as_entire_binding(),
                },
            ],
        }))
    }

    /// The preview blit itself, onto any view in the preview surface's
    /// format — the swapchain frame, or `preview.snapshot`'s offscreen
    /// target. Uniforms must already be written (`render_preview` /
    /// `render_preview_file`). `File(None)` just clears to black.
    fn encode_preview_pass(
        pv: &PreviewTarget,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        draw: PreviewDraw<'_>,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("preview blit pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes,
        });
        match draw {
            PreviewDraw::View(input) => {
                pass.set_pipeline(&pv.pipeline);
                pass.set_bind_group(
                    0,
                    match input {
                        PreviewInput::Leg(Leg::Live) => &pv.bind_group_live,
                        PreviewInput::Leg(Leg::Design) => &pv.bind_group_design,
                        PreviewInput::Program => &pv.bind_group_program,
                    },
                    &[],
                );
                pass.set_bind_group(1, &pv.extra_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            PreviewDraw::File(Some(group)) => {
                pass.set_pipeline(&pv.file_pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.draw(0..3, 0..1);
            }
            PreviewDraw::File(None) => {}
        }
    }

    /// `preview.snapshot`: render exactly what the native preview shows
    /// (same draw, uniforms and weights as the last present) into an
    /// offscreen `width × height` texture in the preview surface's format,
    /// and copy it into a fresh readback buffer. Returns the buffer, its
    /// padded row pitch, and whether the bytes are BGRA. `None` without an
    /// attached preview.
    pub fn encode_preview_snapshot(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        draw: PreviewDraw<'_>,
        width: u32,
        height: u32,
    ) -> Option<(wgpu::Buffer, u32, bool)> {
        let pv = self.preview.as_ref()?;
        let format = pv.config.format;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("preview snapshot target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self::encode_preview_pass(pv, encoder, &view, draw, None);
        let (buffer, bytes_per_row) = self.encode_readback(encoder, &texture, width, height);
        let bgra = matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        Some((buffer, bytes_per_row, bgra))
    }

    /// Copy a 4-byte-per-pixel texture into a new `MAP_READ` buffer with
    /// 256-aligned rows. Returns the buffer and its row pitch.
    pub fn encode_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> (wgpu::Buffer, u32) {
        self.encode_readback_reusing(encoder, texture, width, height, None)
    }

    /// [`Self::encode_readback`] into `reuse` (an unmapped readback buffer
    /// from an earlier call) when its size fits exactly; a new buffer
    /// otherwise.
    pub fn encode_readback_reusing(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
        reuse: Option<wgpu::Buffer>,
    ) -> (wgpu::Buffer, u32) {
        let bytes_per_row = (width * 4).div_ceil(256) * 256;
        let size = (bytes_per_row * height) as u64;
        let buffer = reuse.filter(|b| b.size() == size).unwrap_or_else(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("preview readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        (buffer, bytes_per_row)
    }

    /// Physical size of the attached preview surface (the slot's drawable).
    pub fn preview_size(&self) -> Option<(u32, u32)> {
        self.preview.as_ref().map(|p| (p.config.width, p.config.height))
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
    pub fn encode_final(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
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
            timestamp_writes,
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
            // 4: C5 image input (HAP frame / still; 1×1 dummy when unbound)
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 5: image sampler
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// The layer bind group: shared mask atlas + frame state, the pass's own
/// uniform, and its image (or the dummy). The one constructor every pass
/// and probe item goes through, so the slot layout lives in one place.
/// The two uniforms are buffer *ranges*: a plan packs its FrameState and
/// every pass's LayerParams into one buffer at aligned offsets.
pub fn layer_bind_group(
    gpu: &GpuContext,
    label: &str,
    frame_state: wgpu::BufferBinding<'_>,
    layer_params: wgpu::BufferBinding<'_>,
    image: Option<&wgpu::TextureView>,
) -> wgpu::BindGroup {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
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
                resource: wgpu::BindingResource::Buffer(frame_state),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(layer_params),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(
                    image.unwrap_or(&gpu.dummy_image_view),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&gpu.image_sampler),
            },
        ],
    })
}

fn create_dummy_image(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("image input dummy (1×1 transparent)"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &[0, 0, 0, 0],
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
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
    /// + 32 (params_f) + 64 (params_c) + 16 (image_box) + 16 (image_meta)
    /// = 176, and 16-aligned throughout.
    #[test]
    fn layer_params_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<LayerParamsGpu>(), 176);
        assert_eq!(std::mem::offset_of!(LayerParamsGpu, image_box), 144);
        assert_eq!(std::mem::offset_of!(LayerParamsGpu, image_meta), 160);
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
    fn builtin_effect_shader_parses_and_validates() {
        // The prelude (incl. `sample_image`) + built-ins + main, exactly as
        // the device compiles them at boot — and a user body that calls
        // `sample_image` from non-uniform control flow, which only the
        // explicit-LOD sample keeps legal.
        let user = "fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {\n\
                    if (mask > 0.5) { return sample_image(uv * 2.0); }\n\
                    return vec4<f32>(0.0);\n}";
        for (name, body) in [("builtin", BUILTIN_BODY_WGSL), ("user", user)] {
            let composed = compose_shader(body);
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

    #[test]
    fn output_shaders_parse_and_validate() {
        for (name, src) in [
            ("final_pass", FINAL_PASS_WGSL),
            ("warp_bake", WARP_BAKE_WGSL),
            ("downscale", DOWNSCALE_WGSL),
            ("file_preview", FILE_PREVIEW_WGSL),
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
        warp_sample_case(false, 64, 64);
    }

    /// D4: the preview's program view reads the projector-sized LUT **by uv**
    /// from a target of a different size, and lands on the same image.
    #[test]
    fn program_preview_samples_through_the_warp_at_another_size() {
        // 128-px LUT read from a 64-px target (rows stay 256-byte aligned).
        warp_sample_case(true, 128, 64);
    }

    /// Composite and LUT are `n` px square. `preview = false`: the
    /// projector pipeline (`fs_main`, LUT by pixel) at the LUT's own size.
    /// `preview = true`: the preview pipeline (`fs_preview`, LUT by uv) at
    /// `out` px, light only, no dim.
    #[allow(non_snake_case)]
    fn warp_sample_case(preview: bool, n: u32, out: u32) {
        use crate::alignment::AlignmentDoc;

        let N = n;

        let Some((device, queue)) = test_device() else {
            eprintln!("no GPU device — skipping warp sample case (preview={preview})");
            return;
        };
        assert!(preview || out == N, "the projector pass indexes the LUT by pixel");

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
                width: out,
                height: out,
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

        let (bgl, final_pipeline) = create_final_pipeline(&device, wgpu::TextureFormat::Rgba8Unorm);
        let (extra_bgl, preview_pipeline) =
            create_preview_pipeline(&device, wgpu::TextureFormat::Rgba8Unorm, &bgl);
        let (_photo, photo_view) = create_photo_texture(&device, &queue, 1, 1, &[0, 0, 0, 255]);
        let extra_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test preview extra"),
            contents: bytemuck::bytes_of(&PreviewExtraUniforms::new(1.0, 0.0, 1.0)),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let extra_group = create_preview_extra_bind_group(&device, &extra_bgl, &photo_view, &extra_buf);
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
            size: (out * out * 4) as u64,
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
            if preview {
                pass.set_pipeline(&preview_pipeline);
                pass.set_bind_group(1, &extra_group, &[]);
            } else {
                pass.set_pipeline(&final_pipeline);
            }
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
                    bytes_per_row: Some(out * 4),
                    rows_per_image: Some(out),
                },
            },
            wgpu::Extent3d {
                width: out,
                height: out,
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
            let i = ((y * out + x) * 4) as usize;
            [
                data[i] as f32 / 255.0,
                data[i + 1] as f32 / 255.0,
                data[i + 2] as f32 / 255.0,
            ]
        };
        // Texel t of the composite holds t/N at its centre, so a bilinear
        // sample at source u reads u − 0.5/N. Output pixel x sits at dest
        // u = (x+½)/out and samples source u − 0.25 (the translation). At
        // out == N that is exactly 16 texels to the left; a vertical flip
        // anywhere in the chain moves the sampled row and fails.
        let quarter = out / 4;
        for y in [1u32, out / 8, out / 2, out - 2] {
            for x in [quarter + 1, quarter + 5, out - 1] {
                let got = px(x, y);
                let want_r = (x as f32 + 0.5) / out as f32 - 0.25 - 0.5 / N as f32;
                let want_g = (y as f32 + 0.5) / out as f32 - 0.5 / N as f32;
                assert!(
                    (got[0] - want_r).abs() < 0.015,
                    "({x},{y}) column: got r={} want {want_r}",
                    got[0]
                );
                assert!(
                    (got[1] - want_g).abs() < 0.015,
                    "({x},{y}) row (Y flip?): got g={} want {want_g}",
                    got[1]
                );
            }
        }
        // Left quarter maps outside the composite ⇒ background, not a wrap.
        for y in [0u32, out / 2, out - 1] {
            for x in [0u32, quarter / 2, quarter - 2] {
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
        // Defaults never enable the warp: the per-leg preview groups bind a
        // 1×1 dummy LUT (`render_preview` also clears the flag for them).
        assert_eq!(FinalPassUniforms::identity().adjust[3], 0.0);
        assert_eq!(FinalPassUniforms::new(0.5, 1.0, 0.0).adjust[3], 0.0);
        assert_eq!(std::mem::size_of::<PreviewExtraUniforms>(), 16);
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

/// GPU thumbnail for the JPEG `preview` channel (D5): samples a composite
/// into a small `Rgba8UnormSrgb` target with a 3×3 bilinear-tap box filter
/// (`downscale.wgsl`), so the readback is ~230 KB of display-ready sRGB
/// bytes instead of the full linear f16 composite.
pub struct Downscaler {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Reused while the size holds.
    target: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
    /// Bind group over the last source composite, reused while the source
    /// texture is the same object (a resize or leg change re-keys it).
    source_group: Option<(wgpu::Id<wgpu::Texture>, wgpu::BindGroup)>,
}

pub const THUMB_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

impl Downscaler {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("downscale bind group layout"),
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
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("downscale shader"),
            source: wgpu::ShaderSource::Wgsl(compose_output_shader(DOWNSCALE_WGSL).into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("downscale pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("downscale pipeline"),
            layout: Some(&pipeline_layout),
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
                    format: THUMB_FORMAT,
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("downscale sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            layout,
            sampler,
            target: None,
            source_group: None,
        }
    }

    /// Encode `src` → a `width × height` thumbnail. Returns the target
    /// texture (to copy out of).
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> &wgpu::Texture {
        if !matches!(self.target, Some((_, _, w, h)) if w == width && h == height) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("preview thumbnail"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: THUMB_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.target = Some((texture, view, width, height));
        }
        let src_id = src.global_id();
        if !matches!(&self.source_group, Some((id, _)) if *id == src_id) {
            let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("downscale bind group"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.source_group = Some((src_id, group));
        }
        let (texture, view, _, _) = self.target.as_ref().expect("target just ensured");
        let (_, group) = self.source_group.as_ref().expect("source group just ensured");
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("downscale pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        texture
    }

    pub fn gpu_bytes(&self) -> u64 {
        self.target
            .as_ref()
            .map_or(0, |(t, ..)| crate::perf::texture_bytes(t))
    }
}

/// The native preview's pipeline: the final-pass shader's `fs_preview`
/// entry with group 0 = the final-pass layout (shared bind groups) and a
/// preview-only group 1 (surface photo + `PreviewExtraUniforms`).
fn create_preview_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    final_bgl: &wgpu::BindGroupLayout,
) -> (wgpu::BindGroupLayout, wgpu::RenderPipeline) {
    let extra_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("preview extra bind group layout"),
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
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("preview shader"),
        source: wgpu::ShaderSource::Wgsl(compose_output_shader(FINAL_PASS_WGSL).into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("preview pipeline layout"),
        bind_group_layouts: &[final_bgl, &extra_bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("preview pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_preview",
            targets: &[Some(wgpu::ColorTargetState {
                format,
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
    (extra_bgl, pipeline)
}

/// Studio file preview (P1): `fs_file` over one source texture + its
/// contain-fit uniform. Group 0 = (texture, sampler, uniform).
fn create_file_preview_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::BindGroupLayout, wgpu::RenderPipeline) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("file preview bind group layout"),
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
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("file preview shader"),
        source: wgpu::ShaderSource::Wgsl(compose_output_shader(FILE_PREVIEW_WGSL).into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("file preview pipeline layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("file preview pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_file",
            targets: &[Some(wgpu::ColorTargetState {
                format,
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
    (layout, pipeline)
}

fn create_preview_extra_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    photo_view: &wgpu::TextureView,
    extra: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("preview extra bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(photo_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: extra.as_entire_binding(),
            },
        ],
    })
}

/// The surface photo as `Rgba8UnormSrgb` (sampling linearizes it).
/// `rgba` is tightly packed, `width × height × 4` bytes.
fn create_photo_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("surface photo"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
        rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
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
