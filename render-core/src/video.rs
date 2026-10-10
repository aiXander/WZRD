//! Image inputs for effects (gig plan C5, roadmap §5.9): the texture behind
//! a binding's `image` param, sampled in WGSL via `sample_image(uv)`.
//!
//! A source is opened once per (plan, binding) at plan build and owned by
//! the plan, so it travels with the plan through the §2.6 promote pointer
//! swap. Opening never fails the plan: a missing or unreadable file becomes
//! a per-binding warning and that binding samples transparent black.

use std::path::Path;

use anyhow::{bail, Context, Result};
use wgpu::util::DeviceExt;

use crate::gpu::{GpuContext, ImageEncoding};

/// An opened image input. Today: a still (PNG/JPEG) uploaded once.
pub struct ImageSource {
    /// Held to keep the texture alive; the view is what bind groups use.
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    encoding: ImageEncoding,
}

impl ImageSource {
    /// Open `path` by extension. Stills only for now.
    pub fn open(gpu: &GpuContext, path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        match ext.as_str() {
            "png" | "jpg" | "jpeg" => open_still(gpu, path),
            other => bail!(
                "unsupported image input {:?} (extension {other:?}; expected .png / .jpg)",
                path.display().to_string()
            ),
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn encoding(&self) -> ImageEncoding {
        self.encoding
    }
}

fn open_still(gpu: &GpuContext, path: &Path) -> Result<ImageSource> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .into_rgba8();
    let (w, h) = img.dimensions();
    let max = gpu.device.limits().max_texture_dimension_2d;
    if w > max || h > max {
        bail!("{} is {w}×{h}, above the GPU's {max} px texture limit", path.display());
    }
    // sRGB format: PNG/JPEG texels are sRGB-encoded, and the composite is
    // linear light — the format decodes on sample, the swapchain re-encodes.
    let texture = gpu.device.create_texture_with_data(
        &gpu.queue,
        &wgpu::TextureDescriptor {
            label: Some(&format!("image input [{}]", path.display())),
            size: wgpu::Extent3d {
                width: w,
                height: h,
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
        img.as_raw(),
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok(ImageSource {
        _texture: texture,
        view,
        encoding: ImageEncoding::Rgba,
    })
}
