//! wgpu state: surface, device, queue, mask-atlas texture, pipelines.
//!
//! Phase 2 lives entirely inside this one file — there's only the per-layer
//! `tint` pipeline plus the final homography pass. The compositor in
//! `compositor.rs` consumes this state and issues the per-frame work.

use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::pack::LoadedPack;

/// Format for the offscreen composite buffer.
///
/// `Rgba8UnormSrgb` keeps the output gamma-correct in the projector path.
/// We'll move to 16-bit-float when we add additive stacks (§9 colour banding).
pub const COMPOSITE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LayerUniforms {
    pub color: [f32; 4],
    pub slice: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct HomographyUniforms {
    /// Three rows of the 3×3, each padded to a vec4 for std140.
    pub rows: [[f32; 4]; 3],
}

impl HomographyUniforms {
    pub fn identity() -> Self {
        Self {
            rows: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }
    }

    pub fn from_matrix(m: [[f32; 3]; 3]) -> Self {
        Self {
            rows: [
                [m[0][0], m[0][1], m[0][2], 0.0],
                [m[1][0], m[1][1], m[1][2], 0.0],
                [m[2][0], m[2][1], m[2][2], 0.0],
            ],
        }
    }
}

pub struct GpuContext {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub composite_texture: wgpu::Texture,
    pub composite_view: wgpu::TextureView,
    pub composite_sampler: wgpu::Sampler,
    pub composite_width: u32,
    pub composite_height: u32,

    pub mask_atlas: wgpu::Texture,
    pub mask_atlas_view: wgpu::TextureView,
    pub mask_sampler: wgpu::Sampler,

    pub layer_bind_group_layout: wgpu::BindGroupLayout,
    pub layer_pipeline: wgpu::RenderPipeline,

    pub homography_bind_group_layout: wgpu::BindGroupLayout,
    pub homography_pipeline: wgpu::RenderPipeline,
    pub homography_buffer: wgpu::Buffer,
    pub homography_bind_group: wgpu::BindGroup,
}

impl GpuContext {
    pub async fn new(window: Arc<Window>, pack: &LoadedPack) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
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

        let size = window.inner_size();
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
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Composite buffer sized to the pack's projector_resolution.
        let composite_width = pack.atlas_width;
        let composite_height = pack.atlas_height;
        let (composite_texture, composite_view, composite_sampler) =
            create_composite(&device, composite_width, composite_height);

        let (mask_atlas, mask_atlas_view, mask_sampler) = upload_mask_atlas(&device, &queue, pack);

        let (layer_bind_group_layout, layer_pipeline) =
            create_layer_pipeline(&device, COMPOSITE_FORMAT);
        let (homography_bind_group_layout, homography_pipeline) =
            create_homography_pipeline(&device, surface_config.format);

        // Identity homography by default; calibration UI overwrites later.
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
            ],
        });

        Ok(Self {
            window,
            surface,
            surface_config,
            device,
            queue,
            composite_texture,
            composite_view,
            composite_sampler,
            composite_width,
            composite_height,
            mask_atlas,
            mask_atlas_view,
            mask_sampler,
            layer_bind_group_layout,
            layer_pipeline,
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

    pub fn set_homography(&self, m: Option<[[f32; 3]; 3]>) {
        let uniforms = match m {
            Some(m) => HomographyUniforms::from_matrix(m),
            None => HomographyUniforms::identity(),
        };
        self.queue
            .write_buffer(&self.homography_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Replace the mask atlas after a pack hot-reload. Recreates the texture
    /// because the slice count may change.
    ///
    /// Not wired to the file watcher in Phase 2 — pack changes still require
    /// a process restart. Hook into the watcher in Phase 3 once the slow-path
    /// FBO bookkeeping needs the same swap-on-success protocol.
    #[allow(dead_code)]
    pub fn replace_mask_atlas(&mut self, pack: &LoadedPack) {
        let (atlas, view, sampler) = upload_mask_atlas(&self.device, &self.queue, pack);
        self.mask_atlas = atlas;
        self.mask_atlas_view = view;
        self.mask_sampler = sampler;
        // The composite buffer is also sized to projector_resolution; resize
        // if the pack changed it.
        if pack.atlas_width != self.composite_width || pack.atlas_height != self.composite_height {
            let (tex, view, samp) =
                create_composite(&self.device, pack.atlas_width, pack.atlas_height);
            self.composite_texture = tex;
            self.composite_view = view;
            self.composite_sampler = samp;
            self.composite_width = pack.atlas_width;
            self.composite_height = pack.atlas_height;

            // Recreate the homography bind group since it references composite_view.
            self.homography_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("homography bind group (rebuilt)"),
                    layout: &self.homography_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.composite_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.homography_buffer.as_entire_binding(),
                        },
                    ],
                });
        }
    }
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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

fn create_layer_pipeline(
    device: &wgpu::Device,
    output_format: wgpu::TextureFormat,
) -> (wgpu::BindGroupLayout, wgpu::RenderPipeline) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("layer bind group layout"),
        entries: &[
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
        label: Some("layer shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/layer.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("layer pipeline layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("layer pipeline"),
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
                format: output_format,
                // Standard alpha-over blending. Switches to premultiplied
                // alpha once we have effects that emit pre-multiplied output.
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
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
    });
    (bgl, pipeline)
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
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("homography shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/homography.wgsl").into()),
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
