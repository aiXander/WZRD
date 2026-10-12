//! Image inputs for effects (gig plan C5, roadmap §5.9): the texture behind
//! a binding's `image` param, sampled in WGSL via `sample_image(uv)`.
//!
//! Two sources: a **still** (PNG/JPEG, uploaded once) and a **HAP clip**
//! (`.mov`, decoded on its own thread). A source is opened once per (plan,
//! binding) at plan build and owned by the plan, so it travels with the
//! plan through the §2.6 promote pointer swap — a clip never reopens or
//! jumps at TAKE — and a rebuild of the same leg hands it on to the new
//! plan (`compositor::Carry::Replace`), so a scene edit never reopens it
//! either. A clip opens with the frame at its playhead already in the
//! texture, so even a fresh one never shows black or frame 0 first.
//! Opening never fails the plan: a missing, unreadable or non-HAP file
//! becomes a per-binding warning and that binding samples transparent
//! black.
//!
//! **The staging-ring invariant (§5.9).** Each clip stream owns a ring of
//! [`RING_SLOTS`] `MAP_WRITE` staging buffers. Its decode thread does all
//! disk I/O, snappy and row padding, writing straight into a mapped slot,
//! then unmaps it and reports `ready{slot, frame}` over a channel. The
//! render thread's only contact with video is one `copy_buffer_to_texture`
//! per stream per new frame, plus re-mapping consumed slots after submit
//! (the map callback hands them back to the decoder). Decode falling behind
//! means the previous frame stays up — a stutter, never a frame stall.
//!
//! **Playhead (C5):** `(leg transport seconds × rate + offset) rem_euclid
//! duration` — the offset (owned by the plan) only moves when a rate edit
//! needs the position held continuous. HAP is all-intra, so any frame is a
//! free seek.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use wgpu::util::DeviceExt;

use crate::gpu::{GpuContext, ImageEncoding};
use crate::perf::{ClipStats, ThreadCpu};

/// Staging slots per clip stream: up to three prefetched frames plus one in
/// transit back to the decoder.
pub const RING_SLOTS: usize = 4;

/// The slot that carries the frame a clip opens on (see `open_clip`).
const FIRST_SLOT: usize = 0;

/// An opened image input.
pub struct ImageSource {
    kind: SourceKind,
    view: wgpu::TextureView,
    encoding: ImageEncoding,
    /// Pixel size — C5 places the image by its aspect ("contain").
    size: (u32, u32),
}

enum SourceKind {
    /// Held to keep the texture alive; the view is what bind groups use.
    Still { _texture: wgpu::Texture },
    Clip(Box<ClipStream>),
}

impl ImageSource {
    /// Open `path` by extension: `.mov` → HAP clip stream, `.png`/`.jpg` →
    /// still. A clip starts with the frame at `playhead_sec` in its texture.
    pub fn open(gpu: &GpuContext, path: &Path, playhead_sec: f64) -> Result<Self> {
        Self::open_with(OpenGpu::of(gpu), path, playhead_sec)
    }

    /// [`Self::open`] for many inputs at once, in parallel — one result per
    /// `(path, playhead_sec)` job, in job order. Each open is file I/O, a
    /// first-frame decode (or a whole still) and GPU creation: a few ms per
    /// clip, which a plan build that brings in many clips (a clip-set
    /// `scene.load`, the design rebuild after a TAKE) used to pay serially
    /// on the render thread. Workers are capped at the core count.
    pub fn open_many(gpu: &GpuContext, jobs: &[(PathBuf, f64)]) -> Vec<Result<Self>> {
        let ctx = OpenGpu::of(gpu);
        let workers = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .min(jobs.len());
        if workers <= 1 {
            return jobs.iter().map(|(p, t)| Self::open_with(ctx, p, *t)).collect();
        }
        let next = AtomicUsize::new(0);
        let mut out: Vec<Option<Result<Self>>> = jobs.iter().map(|_| None).collect();
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    s.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some((path, t)) = jobs.get(i) else { break };
                            done.push((i, Self::open_with(ctx, path, *t)));
                        }
                        done
                    })
                })
                .collect();
            for h in handles {
                // A panicked worker's jobs stay `None` → an error below.
                if let Ok(done) = h.join() {
                    for (i, r) in done {
                        out[i] = Some(r);
                    }
                }
            }
        });
        out.into_iter()
            .map(|r| r.unwrap_or_else(|| Err(anyhow!("opening the image panicked"))))
            .collect()
    }

    fn open_with(gpu: OpenGpu<'_>, path: &Path, playhead_sec: f64) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        match ext.as_str() {
            "png" | "jpg" | "jpeg" => open_still(gpu, path),
            "mov" => open_clip(gpu, path, playhead_sec),
            other => bail!(
                "unsupported image input {:?} (extension {other:?}; expected a HAP .mov, \
                 or a .png / .jpg still)",
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

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Whether the texture shows (close to) the frame at `playhead_sec`, so
    /// a pass may sample it. Stills always do. A clip whose stream sat idle
    /// (muted, picked out, an unrendered design leg) still holds the frame
    /// from when it stopped; showing that would flash old content before
    /// the decoder catches up, so it reads as not current until it is.
    /// `step_sec` is the playhead's advance per render frame: normal decode
    /// lag (a few frames behind at high rates) stays within tolerance and
    /// never blinks.
    pub fn is_current(&self, playhead_sec: f64, step_sec: f64) -> bool {
        match &self.kind {
            SourceKind::Still { .. } => true,
            SourceKind::Clip(c) => c.current.is_some_and(|cur| {
                near_playhead(
                    cur,
                    c.info.frame_at(playhead_sec),
                    c.info.samples.len() as u32,
                    step_sec * c.info.fps(),
                )
            }),
        }
    }

    /// Render thread, before the composite pass that samples this source:
    /// ask for the frame at `playhead_sec` and encode the upload of the
    /// freshest ready one. No-op for stills.
    pub fn encode_upload(&mut self, encoder: &mut wgpu::CommandEncoder, playhead_sec: f64) {
        if let SourceKind::Clip(c) = &mut self.kind {
            c.encode_upload(encoder, playhead_sec);
        }
    }

    /// Render thread, after the frame's submit: hand consumed staging slots
    /// back to the decoder.
    pub fn after_submit(&mut self) {
        if let SourceKind::Clip(c) = &mut self.kind {
            c.recycle();
        }
    }

    pub fn is_clip(&self) -> bool {
        matches!(self.kind, SourceKind::Clip(_))
    }

    /// Perf-monitor PM1 GPU-MB inventory: the texture, plus a clip's
    /// staging ring.
    pub fn gpu_bytes(&self) -> u64 {
        match &self.kind {
            SourceKind::Still { _texture } => crate::perf::texture_bytes(_texture),
            SourceKind::Clip(c) => {
                crate::perf::texture_bytes(&c.texture)
                    + c.slots.iter().map(|b| b.size()).sum::<u64>()
            }
        }
    }
}

/// What opening an image needs from the GPU: `Copy` + `Sync` (wgpu's
/// `Device`/`Queue` are), so [`ImageSource::open_many`] can open on worker
/// threads.
#[derive(Clone, Copy)]
struct OpenGpu<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    bc_supported: bool,
}

impl<'a> OpenGpu<'a> {
    fn of(gpu: &'a GpuContext) -> Self {
        Self {
            device: &gpu.device,
            queue: &gpu.queue,
            bc_supported: gpu.bc_supported,
        }
    }
}

/// Positional read (`pread` on unix: no shared seek cursor, one syscall).
fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(not(unix))]
    {
        let mut f = file;
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(buf)
    }
}

fn open_still(gpu: OpenGpu<'_>, path: &Path) -> Result<ImageSource> {
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
        gpu.queue,
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
        kind: SourceKind::Still { _texture: texture },
        view,
        encoding: ImageEncoding::Rgba,
        size: (w, h),
    })
}

// ---------------------------------------------------------------------------
// MOV / ISO-BMFF sample table
// ---------------------------------------------------------------------------

/// One video sample (= one HAP frame) in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sample {
    pub offset: u64,
    pub size: u32,
    /// Presentation start, in `timescale` ticks.
    pub start: u64,
}

/// What the engine needs from a clip's container: codec, size and where
/// every frame lives. Edit lists are ignored (a HAP transcode has none that
/// matter; the clip loops over its full sample table).
#[derive(Debug, Clone)]
pub struct ClipInfo {
    pub fourcc: [u8; 4],
    pub width: u32,
    pub height: u32,
    pub timescale: u32,
    /// Sum of all sample durations, in ticks.
    pub duration: u64,
    pub samples: Vec<Sample>,
}

impl ClipInfo {
    pub fn duration_sec(&self) -> f64 {
        self.duration as f64 / self.timescale as f64
    }

    pub fn fps(&self) -> f64 {
        self.samples.len() as f64 / self.duration_sec().max(1e-9)
    }

    /// The stateless C5 playhead: the frame showing at `t` seconds of leg
    /// time, looping. Negative times (reverse rates) wrap the same way.
    pub fn frame_at(&self, t_sec: f64) -> u32 {
        let n = self.samples.len();
        if n <= 1 || self.duration == 0 {
            return 0;
        }
        let ticks = (t_sec * self.timescale as f64).rem_euclid(self.duration as f64) as u64;
        // Last sample whose start ≤ ticks.
        let i = self.samples.partition_point(|s| s.start <= ticks);
        (i.saturating_sub(1)).min(n - 1) as u32
    }
}

/// Read the first video track's sample table. Only `moov` is loaded into
/// memory (a few KB); `mdat` is never read here.
pub fn read_clip_info(file: &mut File) -> Result<ClipInfo> {
    let len = file.metadata()?.len();
    let mut pos = 0u64;
    while pos + 8 <= len {
        file.seek(SeekFrom::Start(pos))?;
        let mut hdr = [0u8; 16];
        file.read_exact(&mut hdr[..8])?;
        let size32 = u32::from_be_bytes(hdr[0..4].try_into().unwrap()) as u64;
        let kind: [u8; 4] = hdr[4..8].try_into().unwrap();
        let (size, header) = match size32 {
            1 => {
                file.read_exact(&mut hdr[8..16])?;
                (u64::from_be_bytes(hdr[8..16].try_into().unwrap()), 16)
            }
            0 => (len - pos, 8),
            s => (s, 8),
        };
        if size < header || pos + size > len {
            bail!("malformed box at byte {pos}");
        }
        if &kind == b"moov" {
            let mut moov = vec![0u8; (size - header) as usize];
            file.read_exact(&mut moov)?;
            return parse_moov(&moov);
        }
        pos += size;
    }
    bail!("no `moov` box — not a QuickTime/MP4 file")
}

/// Iterate the child boxes of a container payload: (type, payload).
fn boxes(mut data: &[u8]) -> impl Iterator<Item = Result<([u8; 4], &[u8])>> {
    std::iter::from_fn(move || {
        if data.len() < 8 {
            return None;
        }
        let size32 = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
        let kind: [u8; 4] = data[4..8].try_into().unwrap();
        let (size, header) = match size32 {
            1 if data.len() >= 16 => (u64::from_be_bytes(data[8..16].try_into().unwrap()) as usize, 16),
            0 => (data.len(), 8),
            s => (s, 8),
        };
        if size < header || size > data.len() {
            data = &[];
            return Some(Err(anyhow!("malformed {:?} box", String::from_utf8_lossy(&kind))));
        }
        let payload = &data[header..size];
        data = &data[size..];
        Some(Ok((kind, payload)))
    })
}

fn child<'a>(data: &'a [u8], kind: &[u8; 4]) -> Result<Option<&'a [u8]>> {
    for b in boxes(data) {
        let (k, p) = b?;
        if &k == kind {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

fn need<'a>(data: &'a [u8], kind: &[u8; 4]) -> Result<&'a [u8]> {
    child(data, kind)?.ok_or_else(|| anyhow!("missing `{}` box", String::from_utf8_lossy(kind)))
}

fn be32(d: &[u8], at: usize) -> Result<u32> {
    d.get(at..at + 4)
        .map(|b| u32::from_be_bytes(b.try_into().unwrap()))
        .ok_or_else(|| anyhow!("truncated box"))
}

fn be64(d: &[u8], at: usize) -> Result<u64> {
    d.get(at..at + 8)
        .map(|b| u64::from_be_bytes(b.try_into().unwrap()))
        .ok_or_else(|| anyhow!("truncated box"))
}

fn parse_moov(moov: &[u8]) -> Result<ClipInfo> {
    for b in boxes(moov) {
        let (kind, trak) = b?;
        if &kind != b"trak" {
            continue;
        }
        let mdia = need(trak, b"mdia")?;
        let hdlr = need(mdia, b"hdlr")?;
        if hdlr.get(8..12) != Some(b"vide") {
            continue;
        }
        return parse_video_track(mdia);
    }
    bail!("no video track")
}

fn parse_video_track(mdia: &[u8]) -> Result<ClipInfo> {
    let mdhd = need(mdia, b"mdhd")?;
    let timescale = match mdhd.first() {
        Some(1) => be32(mdhd, 20)?,
        _ => be32(mdhd, 12)?,
    };
    if timescale == 0 {
        bail!("video track has timescale 0");
    }
    let stbl = need(need(mdia, b"minf")?, b"stbl")?;

    // stsd: the first sample entry carries the codec fourcc and size.
    let stsd = need(stbl, b"stsd")?;
    let entry = stsd.get(8..).ok_or_else(|| anyhow!("truncated stsd"))?;
    if entry.len() < 36 {
        bail!("truncated video sample entry");
    }
    let fourcc: [u8; 4] = entry[4..8].try_into().unwrap();
    let width = u16::from_be_bytes([entry[32], entry[33]]) as u32;
    let height = u16::from_be_bytes([entry[34], entry[35]]) as u32;

    // stsz: per-sample sizes.
    let stsz = need(stbl, b"stsz")?;
    let uniform = be32(stsz, 4)?;
    let count = be32(stsz, 8)? as usize;
    let sizes: Vec<u32> = if uniform != 0 {
        vec![uniform; count]
    } else {
        (0..count).map(|i| be32(stsz, 12 + 4 * i)).collect::<Result<_>>()?
    };

    // stco / co64: chunk offsets.
    let chunk_offsets: Vec<u64> = if let Some(stco) = child(stbl, b"stco")? {
        let n = be32(stco, 4)? as usize;
        (0..n).map(|i| be32(stco, 8 + 4 * i).map(u64::from)).collect::<Result<_>>()?
    } else {
        let co64 = need(stbl, b"co64")?;
        let n = be32(co64, 4)? as usize;
        (0..n).map(|i| be64(co64, 8 + 8 * i)).collect::<Result<_>>()?
    };

    // stsc: run-length (first_chunk, samples_per_chunk) — expand to a
    // per-sample file offset.
    let stsc = need(stbl, b"stsc")?;
    let runs: Vec<(u32, u32)> = (0..be32(stsc, 4)? as usize)
        .map(|i| Ok((be32(stsc, 8 + 12 * i)?, be32(stsc, 12 + 12 * i)?)))
        .collect::<Result<_>>()?;
    let mut offsets: Vec<u64> = Vec::with_capacity(count);
    for (ci, &chunk_off) in chunk_offsets.iter().enumerate() {
        let chunk_no = ci as u32 + 1;
        let per_chunk = runs
            .iter()
            .rev()
            .find(|(first, _)| *first <= chunk_no)
            .map(|r| r.1)
            .ok_or_else(|| anyhow!("stsc has no run covering chunk {chunk_no}"))?;
        let mut off = chunk_off;
        for _ in 0..per_chunk {
            let Some(&size) = sizes.get(offsets.len()) else { break };
            offsets.push(off);
            off += size as u64;
        }
    }
    if offsets.len() != count {
        bail!("sample table mismatch: {} offsets for {count} samples", offsets.len());
    }

    // stts: run-length durations → start times.
    let stts = need(stbl, b"stts")?;
    let mut starts: Vec<u64> = Vec::with_capacity(count);
    let mut t = 0u64;
    for i in 0..be32(stts, 4)? as usize {
        let n = be32(stts, 8 + 8 * i)?;
        let delta = be32(stts, 12 + 8 * i)? as u64;
        for _ in 0..n {
            starts.push(t);
            t += delta;
        }
    }
    if starts.len() < count {
        bail!("stts covers {} of {count} samples", starts.len());
    }
    if count == 0 {
        bail!("video track has no frames");
    }

    let samples = (0..count)
        .map(|i| Sample {
            offset: offsets[i],
            size: sizes[i],
            start: starts[i],
        })
        .collect();
    Ok(ClipInfo {
        fourcc,
        width,
        height,
        timescale,
        duration: t.max(1),
        samples,
    })
}

// ---------------------------------------------------------------------------
// HAP frames
// ---------------------------------------------------------------------------

/// The BC texture layouts HAP frames carry (spec "texture format" nibble).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HapFormat {
    /// `Hap1` — RGB DXT1.
    Bc1,
    /// `Hap5` — RGBA DXT5 (HAP Alpha).
    Bc3,
    /// `HapY` — scaled YCoCg DXT5 (HAP Q).
    Bc3YCoCg,
}

impl HapFormat {
    fn from_fourcc(f: &[u8; 4]) -> Result<Self> {
        Ok(match f {
            b"Hap1" => HapFormat::Bc1,
            b"Hap5" => HapFormat::Bc3,
            b"HapY" => HapFormat::Bc3YCoCg,
            b"HapM" | b"HapA" | b"Hap7" => bail!(
                "HAP variant {:?} is not supported (use plain HAP or HAP Q)",
                String::from_utf8_lossy(f)
            ),
            other => bail!(
                "codec {:?} is not HAP — transcode with ffmpeg `-c:v hap` (the MCP `to_hap` tool)",
                String::from_utf8_lossy(other)
            ),
        })
    }

    fn from_nibble(n: u8) -> Option<Self> {
        match n {
            0xB => Some(HapFormat::Bc1),
            0xE => Some(HapFormat::Bc3),
            0xF => Some(HapFormat::Bc3YCoCg),
            _ => None,
        }
    }

    /// Bytes per 4×4 block.
    pub fn block_bytes(self) -> u32 {
        match self {
            HapFormat::Bc1 => 8,
            HapFormat::Bc3 | HapFormat::Bc3YCoCg => 16,
        }
    }

    fn texture_format(self) -> wgpu::TextureFormat {
        match self {
            // sRGB-encoded colour → the format decodes to linear on sample.
            HapFormat::Bc1 => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
            HapFormat::Bc3 => wgpu::TextureFormat::Bc3RgbaUnormSrgb,
            // YCoCg isn't colour yet — raw channels; `sample_image`
            // converts, then decodes sRGB itself.
            HapFormat::Bc3YCoCg => wgpu::TextureFormat::Bc3RgbaUnorm,
        }
    }

    fn encoding(self) -> ImageEncoding {
        match self {
            HapFormat::Bc3YCoCg => ImageEncoding::YCoCg,
            _ => ImageEncoding::Rgba,
        }
    }
}

/// Parse a HAP section header: (payload offset, payload length, type byte).
fn section(data: &[u8]) -> Result<(usize, usize, u8)> {
    if data.len() < 4 {
        bail!("truncated HAP section header");
    }
    let size = u32::from_le_bytes([data[0], data[1], data[2], 0]) as usize;
    let kind = data[3];
    let (start, size) = if size == 0 {
        if data.len() < 8 {
            bail!("truncated HAP extended section header");
        }
        (8, u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize)
    } else {
        (4, size)
    };
    if start + size > data.len() {
        bail!("HAP section overruns the frame ({} > {})", start + size, data.len());
    }
    Ok((start, size, kind))
}

/// Decode one HAP frame into raw BC blocks (`out` is resized to fit).
/// Returns the frame's texture format; the caller checks it against the
/// clip's and the byte count against the clip size.
pub fn decode_hap_frame(frame: &[u8], out: &mut Vec<u8>) -> Result<HapFormat> {
    let (start, size, kind) = section(frame)?;
    let payload = &frame[start..start + size];
    let format = HapFormat::from_nibble(kind & 0x0F)
        .ok_or_else(|| anyhow!("unsupported HAP texture format 0x{:X}", kind & 0x0F))?;
    match kind >> 4 {
        0xA => {
            out.clear();
            out.extend_from_slice(payload);
        }
        0xB => snappy_into(payload, out, 0)?,
        0xC => decode_chunked(payload, out)?,
        c => bail!("unsupported HAP compressor 0x{c:X}"),
    }
    Ok(format)
}

/// Snappy-decompress `src`, appending at `out[at..]` (resizing `out`).
fn snappy_into(src: &[u8], out: &mut Vec<u8>, at: usize) -> Result<()> {
    let n = snap::raw::decompress_len(src).context("snappy header")?;
    out.resize(at + n, 0);
    snap::raw::Decoder::new()
        .decompress(src, &mut out[at..])
        .context("snappy decompress")?;
    out.truncate(at + n);
    Ok(())
}

/// The "complex" HAP layout: a decode-instructions container (chunk
/// compressor table, chunk size table, optional offset table), then the
/// chunks. Each chunk decompresses into the next run of the output.
fn decode_chunked(payload: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let (start, size, kind) = section(payload)?;
    if kind != 0x01 {
        bail!("HAP chunked frame: expected a decode-instructions container, got 0x{kind:02X}");
    }
    let instructions = &payload[start..start + size];
    let data = &payload[start + size..];

    let mut compressors: &[u8] = &[];
    let mut sizes: Vec<usize> = Vec::new();
    let mut offsets: Option<Vec<usize>> = None;
    let mut rest = instructions;
    while !rest.is_empty() {
        let (s, n, k) = section(rest)?;
        let body = &rest[s..s + n];
        match k {
            0x02 => compressors = body,
            0x03 => {
                sizes = body
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
                    .collect()
            }
            0x04 => {
                offsets = Some(
                    body.chunks_exact(4)
                        .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
                        .collect(),
                )
            }
            _ => {} // unknown instruction sections are skippable by spec
        }
        rest = &rest[s + n..];
    }
    if compressors.len() != sizes.len() || sizes.is_empty() {
        bail!(
            "HAP chunked frame: {} compressors vs {} chunk sizes",
            compressors.len(),
            sizes.len()
        );
    }
    // Size `out` once for the whole frame, then decompress every chunk in
    // place: `out` keeps the previous frame's length, so a same-size frame
    // resizes for free — clearing and growing it chunk by chunk would
    // zero-fill the whole frame before each decode.
    let chunk = |i: usize, cursor: usize| -> Result<&[u8]> {
        let off = offsets.as_ref().map_or(cursor, |o| o[i]);
        data.get(off..off + sizes[i])
            .ok_or_else(|| anyhow!("HAP chunk {i} overruns the frame"))
    };
    let mut total = 0usize;
    let mut cursor = 0usize;
    for (i, &comp) in compressors.iter().enumerate() {
        let c = chunk(i, cursor)?;
        total += match comp {
            0x0A => c.len(),
            0x0B => snap::raw::decompress_len(c).context("snappy header")?,
            c => bail!("unsupported HAP chunk compressor 0x{c:02X}"),
        };
        cursor = offsets.as_ref().map_or(cursor, |o| o[i]) + sizes[i];
    }
    out.resize(total, 0);
    let (mut at, mut cursor) = (0usize, 0usize);
    let mut snappy = snap::raw::Decoder::new();
    for (i, &comp) in compressors.iter().enumerate() {
        let c = chunk(i, cursor)?;
        let n = match comp {
            0x0A => {
                out[at..at + c.len()].copy_from_slice(c);
                c.len()
            }
            _ => {
                let n = snap::raw::decompress_len(c).context("snappy header")?;
                snappy
                    .decompress(c, &mut out[at..at + n])
                    .context("snappy decompress")?
            }
        };
        at += n;
        cursor = offsets.as_ref().map_or(cursor, |o| o[i]) + sizes[i];
    }
    out.truncate(at);
    Ok(())
}

// ---------------------------------------------------------------------------
// Clip stream: decode thread + staging ring
// ---------------------------------------------------------------------------

/// Render thread → decoder: "this frame is wanted now; prefetch along
/// `step`" (signed frames per render frame, |step| ≥ 1).
#[derive(Debug, Clone, Copy)]
struct Want {
    frame: u32,
    step: i32,
}

/// Decoder → render thread: slot `slot` holds frame `frame`, unmapped.
#[derive(Debug, Clone, Copy)]
struct Ready {
    slot: usize,
    frame: u32,
}

/// Geometry of one staged frame: BC block rows padded to the copy pitch.
#[derive(Debug, Clone, Copy)]
struct Layout {
    row_bytes: usize,
    padded_row_bytes: usize,
    block_rows: usize,
}

impl Layout {
    fn slot_bytes(&self) -> u64 {
        (self.padded_row_bytes * self.block_rows) as u64
    }
}

struct ClipStream {
    info: Arc<ClipInfo>,
    texture: wgpu::Texture,
    layout: Layout,
    /// Shared with the decode thread (which writes them while mapped).
    slots: Arc<Vec<wgpu::Buffer>>,
    want_tx: Sender<Want>,
    ready_rx: Receiver<Ready>,
    /// Cloned into each `map_async` callback to hand a slot back.
    free_tx: Sender<usize>,
    /// Ready slots the render thread holds, not yet uploaded.
    ready: Vec<Ready>,
    /// Slots consumed (or gone stale) this frame — re-mapped after submit.
    recycle: Vec<usize>,
    /// Frame currently in the texture (counting `first_upload`, which the
    /// next `encode_upload` puts there before anything samples it).
    current: Option<u32>,
    /// The staging slot holding the frame opened on, not yet copied.
    first_upload: Option<usize>,
    last_want: Option<(u32, i32)>,
    /// Previous call's playhead, to estimate frames per render frame.
    prev_playhead: Option<f64>,
    /// The last [`STEP_HISTORY`] playhead advances (clip frames per render
    /// frame, signed), newest at `step_i - 1`; the prefetch stride is their
    /// median, so one hitch (or a stream waking from idle) doesn't send the
    /// decoder off prefetching at a stride of 2+ for nothing.
    steps: [f64; STEP_HISTORY],
    step_i: usize,
    /// Perf-monitor PM1: decode ms (written by the decoder), stale frames
    /// + ready-queue depth (written here).
    stats: Arc<ClipStats>,
}

fn open_clip(gpu: OpenGpu<'_>, path: &Path, playhead_sec: f64) -> Result<ImageSource> {
    if !gpu.bc_supported {
        bail!(
            "this GPU adapter has no BC texture compression \
             (TEXTURE_COMPRESSION_BC), so HAP clips can't play here"
        );
    }
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let info = read_clip_info(&mut file).with_context(|| format!("reading {}", path.display()))?;
    let format = HapFormat::from_fourcc(&info.fourcc)?;
    let (w, h) = (info.width, info.height);
    if w == 0 || h == 0 || w % 4 != 0 || h % 4 != 0 {
        bail!("HAP clip is {w}×{h}; both dimensions must be non-zero multiples of 4 (BC blocks)");
    }
    let max = gpu.device.limits().max_texture_dimension_2d;
    if w > max || h > max {
        bail!("HAP clip is {w}×{h}, above the GPU's {max} px texture limit");
    }
    let row_bytes = (w / 4 * format.block_bytes()) as usize;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let layout = Layout {
        row_bytes,
        padded_row_bytes: row_bytes.div_ceil(align) * align,
        block_rows: (h / 4) as usize,
    };

    // Decode the playhead frame here, synchronously: a file that isn't
    // really HAP (or is corrupt) fails the open with a named error instead
    // of playing black forever, and the texture starts on the right frame —
    // no black first frame, no flash of frame 0 before the decoder lands.
    let start = info.frame_at(playhead_sec);
    let first = info.samples[start as usize];
    let mut raw = vec![0u8; first.size as usize];
    read_at(&file, first.offset, &mut raw)
        .with_context(|| format!("reading frame {start} of {}", path.display()))?;
    let mut blocks = Vec::new();
    let got = decode_hap_frame(&raw, &mut blocks)
        .with_context(|| format!("decoding frame {start} of {}", path.display()))?;
    if got != format {
        bail!("frame {start} is {got:?} but the track declares {format:?}");
    }
    if blocks.len() != row_bytes * layout.block_rows {
        bail!(
            "frame {start} decodes to {} bytes, expected {} for {w}×{h} {format:?}",
            blocks.len(),
            row_bytes * layout.block_rows
        );
    }

    let label = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("clip frame [{label}]")),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format.texture_format(),
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    // Every slot starts mapped. The first frame goes up through slot 0 —
    // copied by the stream's first `encode_upload`, which every composite
    // that samples it is preceded by — not `queue.write_texture`: that
    // serializes on the queue (≈ 0.3 ms per 1080p clip, with multi-ms
    // spikes), which made parallel opens queue up behind each other. The
    // other slots start free (owned by the decoder).
    let slots: Vec<wgpu::Buffer> = (0..RING_SLOTS)
        .map(|i| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("clip staging [{label} #{i}]")),
                size: layout.slot_bytes(),
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            })
        })
        .collect();
    write_slot(&slots[FIRST_SLOT], &blocks, layout);
    slots[FIRST_SLOT].unmap();
    let slots = Arc::new(slots);

    let (want_tx, want_rx) = crossbeam_channel::bounded::<Want>(8);
    let (ready_tx, ready_rx) = crossbeam_channel::bounded::<Ready>(RING_SLOTS);
    let (free_tx, free_rx) = crossbeam_channel::bounded::<usize>(RING_SLOTS);
    for i in (0..RING_SLOTS).filter(|&i| i != FIRST_SLOT) {
        free_tx.send(i).expect("fresh channel");
    }
    let mut in_flight = [None; RING_SLOTS];
    in_flight[FIRST_SLOT] = Some(start);

    let info = Arc::new(info);
    let stats = ClipStats::register(label.clone());
    let decoder = Decoder {
        info: Arc::clone(&info),
        file,
        format,
        layout,
        slots: Arc::clone(&slots),
        want_rx,
        ready_tx,
        free_rx,
        in_flight,
        // The texture already holds `start`: a first want for it decodes
        // nothing.
        want_frame: Some(start),
        served: Some(start),
        raw,
        blocks,
        label: label.clone(),
        errors: 0,
        stats: Arc::clone(&stats),
        cpu: ThreadCpu::register(format!("clip-decode:{label}")),
    };
    std::thread::Builder::new()
        .name(format!("clip-decode:{label}"))
        .spawn(move || decoder.run())
        .context("spawning clip decode thread")?;

    log::info!(
        "opened HAP clip {label}: {w}×{h} {:?}, {} frames @ {:.2} fps",
        format,
        info.samples.len(),
        info.fps()
    );
    Ok(ImageSource {
        kind: SourceKind::Clip(Box::new(ClipStream {
            info,
            texture,
            layout,
            slots,
            want_tx,
            ready_rx,
            free_tx,
            ready: Vec::with_capacity(RING_SLOTS),
            recycle: Vec::with_capacity(RING_SLOTS),
            current: Some(start),
            first_upload: Some(FIRST_SLOT),
            last_want: None,
            prev_playhead: None,
            steps: [0.0; STEP_HISTORY],
            step_i: 0,
            stats,
        })),
        view,
        encoding: format.encoding(),
        size: (w, h),
    })
}

/// Copy BC block rows into a mapped staging slot at the padded pitch.
fn write_slot(slot: &wgpu::Buffer, blocks: &[u8], layout: Layout) {
    let mut view = slot.slice(..).get_mapped_range_mut();
    if layout.row_bytes == layout.padded_row_bytes {
        view[..blocks.len()].copy_from_slice(blocks);
    } else {
        for (dst, src) in view
            .chunks_exact_mut(layout.padded_row_bytes)
            .zip(blocks.chunks_exact(layout.row_bytes))
        {
            dst[..layout.row_bytes].copy_from_slice(src);
        }
    }
}

/// `shown` is close enough to `wanted` (frames of an `n`-frame loop) to
/// present: within four render frames of playhead advance (`step_frames`
/// per frame), at least four clip frames, either way round the loop — so a
/// decoder a few frames behind, or a rate flipping direction, never reads
/// as stale, while a stream that sat idle for a while does.
fn near_playhead(shown: u32, wanted: u32, n: u32, step_frames: f64) -> bool {
    let d = ahead(shown, wanted, 1, n);
    let tolerance = (4.0 * step_frames.abs()).ceil().max(4.0) as u32;
    d.min(n.max(1) - d) <= tolerance
}

/// Playhead advances the prefetch stride is the median of: one outlier
/// (a hitch, a stream waking from idle) is ignored, a real rate change
/// lands within two render frames.
const STEP_HISTORY: usize = 3;

/// The prefetch stride (signed clip frames per render frame, |step| ≥ 1)
/// from the last advances in `steps` (`filled` written so far, ring order).
fn prefetch_step(steps: &[f64; STEP_HISTORY], filled: usize) -> i32 {
    let s = match filled {
        0 => return 1,
        // Fewer than three: the newest.
        1 | 2 => steps[(filled - 1) % STEP_HISTORY],
        _ => {
            let [a, b, c] = *steps;
            a.min(b).max(a.max(b).min(c))
        }
    };
    match s.round() as i32 {
        0 if s < 0.0 => -1,
        0 => 1,
        r => r,
    }
}

/// Forward distance from `a` to `b` along `dir` (±1), modulo `n`.
fn ahead(a: u32, b: u32, dir: i32, n: u32) -> u32 {
    let n = n.max(1) as i64;
    let d = if dir >= 0 {
        b as i64 - a as i64
    } else {
        a as i64 - b as i64
    };
    d.rem_euclid(n) as u32
}

impl ClipStream {
    /// Encode the copy of staging slot `slot` into the texture.
    fn encode_copy(&self, encoder: &mut wgpu::CommandEncoder, slot: usize) {
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &self.slots[slot],
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.layout.padded_row_bytes as u32),
                    rows_per_image: Some(self.layout.block_rows as u32),
                },
            },
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.info.width,
                height: self.info.height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn encode_upload(&mut self, encoder: &mut wgpu::CommandEncoder, playhead_sec: f64) {
        // The frame the clip opened on, staged at open: always first, so a
        // newer frame chosen below lands on top of it.
        if let Some(slot) = self.first_upload.take() {
            self.encode_copy(encoder, slot);
            self.recycle.push(slot);
        }
        let n = self.info.samples.len() as u32;
        let wanted = self.info.frame_at(playhead_sec);

        // Frames advanced per render frame (signed), for the prefetch stride.
        if let Some(prev) = self.prev_playhead {
            self.steps[self.step_i % STEP_HISTORY] = (playhead_sec - prev) * self.info.fps();
            self.step_i += 1;
        }
        self.prev_playhead = Some(playhead_sec);
        let step = prefetch_step(&self.steps, self.step_i);
        let dir = step.signum();
        if self.last_want != Some((wanted, step)) {
            // Bounded and never awaited: a full queue just means the
            // decoder hasn't drained the older wants yet, and only the
            // newest one matters to it.
            if self.want_tx.try_send(Want { frame: wanted, step }).is_ok() {
                self.last_want = Some((wanted, step));
            }
        }

        while let Ok(r) = self.ready_rx.try_recv() {
            self.ready.push(r);
        }
        if self.ready.is_empty() || self.current == Some(wanted) {
            self.note_frame(wanted);
            return;
        }

        // Window of frames at/after `wanted` worth holding on to.
        let window = (RING_SLOTS as u32) * step.unsigned_abs().max(1);
        let mut best: Option<(usize, u32)> = None; // (index in ready, distance to wanted)
        for (i, r) in self.ready.iter().enumerate() {
            let fwd = ahead(wanted, r.frame, dir, n);
            if fwd == 0 {
                best = Some((i, 0));
                break;
            }
            if fwd < window {
                continue; // prefetched future frame — keep
            }
            // Behind `wanted`: a candidate if newer than what's on screen.
            let behind = ahead(r.frame, wanted, dir, n);
            let newer = match self.current {
                Some(c) => ahead(c, r.frame, dir, n) > 0 && ahead(c, r.frame, dir, n) <= ahead(c, wanted, dir, n),
                None => true,
            };
            if newer && best.is_none_or(|(_, d)| behind < d) {
                best = Some((i, behind));
            }
        }

        let chosen = best.map(|(i, _)| self.ready[i]);
        if let Some(c) = chosen {
            self.encode_copy(encoder, c.slot);
            self.current = Some(c.frame);
        }
        // Recycle the uploaded slot and every slot that is now behind the
        // screen; keep only prefetched future frames.
        let current = self.current;
        let recycle = &mut self.recycle;
        self.ready.retain(|r| {
            let keep = Some(r.frame) != current
                && chosen.map(|c| c.slot) != Some(r.slot)
                && ahead(wanted, r.frame, dir, n) < window;
            if !keep {
                recycle.push(r.slot);
            }
            keep
        });
        self.note_frame(wanted);
    }

    /// Perf: after this frame's upload decision. The playhead wanted
    /// `wanted` and the texture holds something else → stale (the decoder
    /// fell behind). A native-rate repeat — a 30 fps clip on a 60 Hz engine
    /// wanting the same frame twice — has `current == wanted`, so it never
    /// counts. Nothing on screen yet (first frame in flight) isn't stale
    /// either.
    fn note_frame(&self, wanted: u32) {
        self.stats.set_queue(self.ready.len());
        if self.current.is_some_and(|c| c != wanted) {
            self.stats.mark_stale();
        }
    }

    /// Re-map consumed slots; each callback hands its slot back to the
    /// decoder. Must run after the submit that carries their copies.
    fn recycle(&mut self) {
        for slot in self.recycle.drain(..) {
            let tx = self.free_tx.clone();
            self.slots[slot]
                .slice(..)
                .map_async(wgpu::MapMode::Write, move |res| {
                    if res.is_ok() {
                        let _ = tx.try_send(slot);
                    }
                });
        }
    }
}

struct Decoder {
    info: Arc<ClipInfo>,
    file: File,
    format: HapFormat,
    layout: Layout,
    slots: Arc<Vec<wgpu::Buffer>>,
    want_rx: Receiver<Want>,
    ready_tx: Sender<Ready>,
    free_rx: Receiver<usize>,
    /// Frame held by each slot while it is not free (decoded and with the
    /// render thread, or in transit back).
    in_flight: [Option<u32>; RING_SLOTS],
    /// Frame of the newest want.
    want_frame: Option<u32>,
    /// The want frame has gone to the render thread already (a `Ready` for
    /// it was sent, or it was in flight when the want arrived). The render
    /// thread asks again only when its playhead moves to another frame, so
    /// once its slot comes back the frame must not be decoded again — at a
    /// 30 fps clip on a 60 Hz output that re-decode used to double the
    /// decode work, every frame thrown away.
    served: Option<u32>,
    raw: Vec<u8>,
    blocks: Vec<u8>,
    label: String,
    errors: u32,
    stats: Arc<ClipStats>,
    cpu: ThreadCpu,
}

impl Decoder {
    fn run(self) {
        let label = self.label.clone();
        self.run_loop();
        log::debug!("clip {label}: decode thread exited (stream closed)");
    }

    /// A want arrived. A new frame resets `served` — unless that frame is
    /// already in flight, which serves it.
    fn take_want(&mut self, w: Want, want: &mut Option<Want>) {
        if self.want_frame != Some(w.frame) {
            self.want_frame = Some(w.frame);
            self.served = self.in_flight.contains(&Some(w.frame)).then_some(w.frame);
        }
        *want = Some(w);
    }

    fn run_loop(mut self) {
        let mut free: Vec<usize> = Vec::with_capacity(RING_SLOTS);
        let mut returned: Option<usize> = None;
        let mut want: Option<Want> = None;
        let (want_rx, free_rx) = (self.want_rx.clone(), self.free_rx.clone());
        loop {
            // Block until there is something to do: a new want, or a slot
            // coming back. Dropping the stream disconnects `want_rx`, which
            // wakes this and exits — no timeout needed.
            crossbeam_channel::select! {
                recv(want_rx) -> w => match w {
                    Ok(w) => self.take_want(w, &mut want),
                    Err(_) => return, // stream dropped
                },
                recv(free_rx) -> s => returned = s.ok(),
            }
            self.cpu.tick();
            // Wants before returned slots: the render thread sends the want
            // for a frame before it consumes that frame's slot, so a slot
            // coming back must not make a just-consumed frame look unserved.
            loop {
                match want_rx.try_recv() {
                    Ok(w) => self.take_want(w, &mut want),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return,
                }
            }
            let back = returned.take().into_iter().chain(std::iter::from_fn(|| free_rx.try_recv().ok()));
            for s in back {
                self.in_flight[s] = None;
                free.push(s);
            }
            let Some(w) = want else { continue };

            let n = self.info.samples.len() as i64;
            for k in 0..(RING_SLOTS as i64 - 1) {
                if free.is_empty() {
                    break;
                }
                let frame = (w.frame as i64 + k * w.step as i64).rem_euclid(n) as u32;
                if (k == 0 && self.served == Some(frame)) || self.in_flight.contains(&Some(frame)) {
                    continue;
                }
                let slot = free.pop().expect("checked non-empty");
                let started = std::time::Instant::now();
                match self.decode_into(frame, slot) {
                    Ok(()) => {
                        self.stats.record_decode(started.elapsed());
                        self.in_flight[slot] = Some(frame);
                        if k == 0 {
                            self.served = Some(frame);
                        }
                        if self.ready_tx.send(Ready { slot, frame }).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        free.push(slot);
                        self.errors += 1;
                        if self.errors <= 3 {
                            log::warn!("clip {}: frame {frame} skipped: {e:#}", self.label);
                        }
                        break;
                    }
                }
            }
        }
    }

    fn decode_into(&mut self, frame: u32, slot: usize) -> Result<()> {
        let s = self.info.samples[frame as usize];
        self.raw.resize(s.size as usize, 0);
        read_at(&self.file, s.offset, &mut self.raw)?;
        let got = decode_hap_frame(&self.raw, &mut self.blocks)?;
        let expected = self.layout.row_bytes * self.layout.block_rows;
        if got != self.format || self.blocks.len() != expected {
            bail!(
                "decodes to {got:?} / {} bytes, clip is {:?} / {expected}",
                self.blocks.len(),
                self.format
            );
        }
        let buf = &self.slots[slot];
        write_slot(buf, &self.blocks, self.layout);
        buf.unmap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
    }

    /// Fixtures: `ffmpeg -f lavfi -i testsrc2=size=64x32:rate=4:duration=2
    /// -c:v hap -format <hap|hap_q> [-chunks 3] [-compressor none]` —
    /// 8 frames each.
    #[test]
    fn reads_sample_tables_and_decodes_every_variant() {
        for (name, fourcc, format) in [
            ("hap.mov", b"Hap1", HapFormat::Bc1),
            ("hap_q.mov", b"HapY", HapFormat::Bc3YCoCg),
            ("hap_chunks.mov", b"Hap1", HapFormat::Bc1),
            ("hap_raw.mov", b"Hap1", HapFormat::Bc1),
        ] {
            let mut f = File::open(fixture(name)).unwrap();
            let info = read_clip_info(&mut f).unwrap();
            assert_eq!(&info.fourcc, fourcc, "{name}");
            assert_eq!((info.width, info.height), (64, 32), "{name}");
            assert_eq!(info.samples.len(), 8, "{name}");
            assert!((info.duration_sec() - 2.0).abs() < 1e-6, "{name}");
            assert!((info.fps() - 4.0).abs() < 1e-6, "{name}");
            let mut blocks = Vec::new();
            for s in &info.samples {
                let mut raw = vec![0u8; s.size as usize];
                f.seek(SeekFrom::Start(s.offset)).unwrap();
                f.read_exact(&mut raw).unwrap();
                let got = decode_hap_frame(&raw, &mut blocks).unwrap();
                assert_eq!(got, format, "{name}");
                assert_eq!(blocks.len(), (16 * 8 * format.block_bytes()) as usize, "{name}");
            }
        }
    }

    #[test]
    fn chunked_and_plain_decode_to_the_same_blocks() {
        let decode_all = |name: &str| {
            let mut f = File::open(fixture(name)).unwrap();
            let info = read_clip_info(&mut f).unwrap();
            let mut out = Vec::new();
            for s in &info.samples {
                let mut raw = vec![0u8; s.size as usize];
                f.seek(SeekFrom::Start(s.offset)).unwrap();
                f.read_exact(&mut raw).unwrap();
                let mut blocks = Vec::new();
                decode_hap_frame(&raw, &mut blocks).unwrap();
                out.extend(blocks);
            }
            out
        };
        let plain = decode_all("hap.mov");
        assert_eq!(plain, decode_all("hap_chunks.mov"));
        assert_eq!(plain, decode_all("hap_raw.mov"));
    }

    #[test]
    fn non_hap_codec_is_refused_by_name() {
        let mut f = File::open(fixture("h264.mov")).unwrap();
        let info = read_clip_info(&mut f).unwrap();
        let err = HapFormat::from_fourcc(&info.fourcc).unwrap_err();
        assert!(format!("{err}").contains("not HAP"), "{err}");
    }

    #[test]
    fn stateless_playhead_loops_and_reverses() {
        let info = ClipInfo {
            fourcc: *b"Hap1",
            width: 4,
            height: 4,
            timescale: 30,
            duration: 30 * 4, // 4 s, 30 fps
            samples: (0..120)
                .map(|i| Sample {
                    offset: 0,
                    size: 0,
                    start: i,
                })
                .collect(),
        };
        assert_eq!(info.frame_at(0.0), 0);
        assert_eq!(info.frame_at(1.0), 30);
        assert_eq!(info.frame_at(3.999), 119);
        assert_eq!(info.frame_at(4.0), 0); // loops
        assert_eq!(info.frame_at(9.5), 45);
        assert_eq!(info.frame_at(-1.0 / 30.0), 119); // reverse wraps
    }

    #[test]
    fn near_playhead_tolerates_lag_but_not_an_idle_gap() {
        // 1× at 60 Hz on a 30 fps, 300-frame clip: half a frame per tick.
        assert!(near_playhead(100, 100, 300, 0.5));
        assert!(near_playhead(97, 100, 300, 0.5)); // decoder a few behind
        assert!(near_playhead(298, 1, 300, 0.5)); // across the loop seam
        assert!(near_playhead(101, 100, 300, 0.5)); // reversed rate
        assert!(!near_playhead(40, 100, 300, 0.5)); // woke after 2 s idle
        // 8×: four frames of playhead per tick widen the window to 16.
        assert!(near_playhead(86, 100, 300, 4.0));
        assert!(!near_playhead(60, 100, 300, 4.0));
    }

    /// A clip opens with the playhead frame already uploaded — never
    /// black, never frame 0 first.
    #[test]
    fn open_starts_on_the_playhead_frame() {
        let Some(gpu) = crate::compositor::tests::test_gpu() else {
            eprintln!("no BC-capable adapter — skipping");
            return;
        };
        let source = ImageSource::open(&gpu, &fixture("hap.mov"), 1.25).unwrap();
        let SourceKind::Clip(c) = &source.kind else { panic!("clip expected") };
        assert_eq!(c.current, Some(5)); // 4 fps: 1.25 s → frame 5
        assert!(source.is_current(1.25, 0.0));
    }

    /// `open_many` = `open` per job, results in job order, failures per job.
    #[test]
    fn open_many_keeps_job_order() {
        let Some(gpu) = crate::compositor::tests::test_gpu() else {
            eprintln!("no BC-capable adapter — skipping");
            return;
        };
        let jobs = [
            (fixture("hap.mov"), 1.25),
            (fixture("missing.mov"), 0.0),
            (fixture("hap_q.mov"), 0.5),
        ];
        let out = ImageSource::open_many(&gpu, &jobs);
        assert_eq!(out.len(), 3);
        let frame = |r: &Result<ImageSource>| match &r.as_ref().unwrap().kind {
            SourceKind::Clip(c) => c.current,
            _ => panic!("clip expected"),
        };
        assert_eq!(frame(&out[0]), Some(5));
        assert!(format!("{:#}", out[1].as_ref().err().unwrap()).contains("missing.mov"));
        assert_eq!(frame(&out[2]), Some(2));
        assert_eq!(out[2].as_ref().unwrap().encoding(), ImageEncoding::YCoCg);
    }

    #[test]
    fn prefetch_step_ignores_one_outlier() {
        let step = |v: &[f64]| {
            let mut ring = [0.0; STEP_HISTORY];
            for (i, x) in v.iter().enumerate() {
                ring[i % STEP_HISTORY] = *x;
            }
            prefetch_step(&ring, v.len())
        };
        assert_eq!(step(&[]), 1);
        assert_eq!(step(&[0.5, 0.5, 0.5, 0.5]), 1); // 30 fps clip at 60 Hz
        assert_eq!(step(&[0.5, 0.5, 2.0]), 1); // one hitch
        assert_eq!(step(&[0.5, 0.5, 60.0, 0.5]), 1); // woke from idle
        assert_eq!(step(&[0.5, 4.0, 4.0]), 4); // real 8× change, two frames in
        assert_eq!(step(&[-0.5, -0.5, -0.5]), -1); // reverse
        assert_eq!(step(&[0.5, -2.0, -2.0]), -2);
    }

    #[test]
    fn ahead_is_directional_and_modular() {
        assert_eq!(ahead(5, 7, 1, 10), 2);
        assert_eq!(ahead(7, 5, 1, 10), 8);
        assert_eq!(ahead(7, 5, -1, 10), 2);
        assert_eq!(ahead(9, 1, 1, 10), 2);
        assert_eq!(ahead(3, 3, 1, 10), 0);
    }
}
