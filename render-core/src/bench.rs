//! Perf-monitor PM3 — the windowless bench host (`render-core --bench`).
//!
//! A third host next to `WinitHost` (`app.rs`) and the Tauri shell's
//! TauriHost: no window and no surface (`Core::init_gpu_offscreen` — the
//! final pass lands in an offscreen texture at the scenario's size), driven
//! with the same per-frame contract (`poll_inbound` → `pace_frame` →
//! `redraw`). Uncapped by default, so it measures *cost*; `--paced` (or
//! `"paced": true`) runs a virtual 60 Hz vsync for latency/jitter scenarios.
//!
//! A scenario (`render-core/perf/scenarios/*.json`) names a scene, an
//! output size, warm-up + measured frame counts and a **timeline** of RPCs
//! that go through `rpc::dispatch` — the real `promote` / `param.set` /
//! `scene.load` paths, from a worker thread exactly like a WS connection.
//! The report is the Tier-A aggregate over the measured window
//! (`Perf::bench_report`) plus machine info; `scripts/perf_gate.py` runs the
//! suite and compares against a committed baseline.
//!
//! The scene is **staged** into a scratch directory first (scene file +
//! a copy of `effects/` + the scenario's alignment, pack and clip paths made
//! absolute), so a bench never writes `session.json`, design drafts or
//! `effect.upsert` shaders into a real project.

use std::collections::{BTreeMap, HashSet};
use std::net::{SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::core::Core;
use crate::rpc::{self, Actor, EngineCommand, JsonRpcRequest, RpcContext};
use crate::{Cli, PerfMode};

/// `render-core --bench <scenario.json> --bench-out <report.json>`. Parsed
/// by `main.rs` when `--bench` is present — the engine's `Cli` (shared with
/// the Tauri host) stays untouched.
#[derive(Parser, Debug, Clone)]
#[command(name = "render-core --bench", about = "WZRD windowless perf bench (PM3)")]
pub struct BenchArgs {
    /// Scenario file (`render-core/perf/scenarios/*.json`).
    #[arg(long)]
    pub bench: PathBuf,
    /// Where the JSON report goes.
    #[arg(long)]
    pub bench_out: PathBuf,
    /// Force a virtual 60 Hz vsync (overrides the scenario's `paced`).
    #[arg(long)]
    pub paced: bool,
    /// GPU timestamp stride: time one frame in N. A timed frame runs alone
    /// on the GPU (the queue drains first — two frames in flight overlap
    /// and inflate the span), the rest keep the GPU fed so it stays clocked
    /// up. 4 gives ~250 GPU samples per 1000 frames; each timed frame adds
    /// ~0.07 ms of render-thread CPU (reference §2.9), a constant that
    /// baselines compare like with like.
    #[arg(long, default_value_t = DEFAULT_BENCH_GPU_TIMING)]
    pub gpu_timing: u32,
    /// `off` runs without the frame profiler (the report is then empty —
    /// only useful for an on/off wall-time comparison).
    #[arg(long, value_enum, default_value = "on")]
    pub perf: PerfMode,
    /// Scratch directory for the staged scene (default: a fresh temp dir,
    /// removed afterwards).
    #[arg(long)]
    pub work_dir: Option<PathBuf>,
    /// Override the scenario's measured frame count.
    #[arg(long)]
    pub frames: Option<usize>,
    /// Override the scenario's warm-up frame count.
    #[arg(long)]
    pub warmup: Option<usize>,
}

/// `--gpu-timing` default for the bench (see [`BenchArgs::gpu_timing`]).
pub const DEFAULT_BENCH_GPU_TIMING: u32 = 4;

/// How long teardown keeps rendering so in-flight RPCs get their reply.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
/// How long `drop(core)` may take once the report is written before the
/// process exits anyway (exit code 3).
const TEARDOWN_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct Scenario {
    #[serde(default)]
    description: String,
    /// Scene file, relative to the scenario file.
    scene: PathBuf,
    /// Output (final pass) size.
    #[serde(default = "default_size")]
    size: [u32; 2],
    #[serde(default = "default_warmup")]
    warmup_frames: usize,
    frames: usize,
    /// Virtual 60 Hz vsync (latency / jitter scenarios).
    #[serde(default)]
    paced: bool,
    /// §5.6 topology: two legs like the operator app (default), or the
    /// headless single leg.
    #[serde(default = "default_true")]
    two_leg: bool,
    /// `alignment.json` to stage next to the scene (relative to the
    /// scenario). Default: none → identity warp.
    #[serde(default)]
    alignment: Option<PathBuf>,
    /// Feed the engine's own OSC listener (a free port) so `audio.*`
    /// drivers read real values: `{lmh: [l, m, h], onset_hz}`.
    #[serde(default)]
    audio: Option<AudioPin>,
    /// Hold a native preview slot visible for the whole run, on an
    /// offscreen target: `{size: [w, h], params: PreviewParams}` (e.g.
    /// `{"file": true}` + a `preview.file` timeline action = the Studio
    /// monitor playing a clip). Default: no preview.
    #[serde(default)]
    preview: Option<BenchPreview>,
    /// Telemetry channels a bench-side subscriber holds for the whole run
    /// (e.g. `preview` — wakes the JPEG readback + the design composite).
    #[serde(default)]
    subscribe: Vec<String>,
    /// Actor the timeline's RPCs dispatch as (`ui` | `agent` | `system`).
    #[serde(default = "default_actor")]
    actor: String,
    #[serde(default)]
    timeline: Vec<Action>,
    /// Gate hints for `scripts/perf_gate.py` — opaque to the engine.
    #[serde(default)]
    #[allow(dead_code)]
    gate: Value,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct BenchPreview {
    size: [u32; 2],
    #[serde(default)]
    params: crate::core::PreviewParams,
}

fn default_size() -> [u32; 2] {
    [1920, 1080]
}
fn default_warmup() -> usize {
    120
}
fn default_true() -> bool {
    true
}
fn default_actor() -> String {
    "ui".into()
}

#[derive(Deserialize, Debug, Clone)]
struct AudioPin {
    #[serde(default = "default_lmh")]
    lmh: [f32; 3],
    /// Onset messages per second on all three bands (0 = none).
    #[serde(default)]
    onset_hz: f32,
}

fn default_lmh() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

/// One timeline entry. Frames count from the first warm-up frame.
/// `every_frames` + `count` repeats it; `hz` fires it from a background
/// thread at a fixed rate until the run ends (knob storms). Params may use
/// directives: `{"$alt": [a, b, …]}` (picked by repetition index),
/// `{"$scene": "file"}` (a scene file, staged like the main one, as a JSON
/// string — for `scene.load`), `{"$file": "file"}` (raw text, e.g. WGSL),
/// `{"$json": "file"}` (a JSON value, e.g. an `alignment.set` patch),
/// `{"$path": "file"}` (the file's absolute path, e.g. for `preview.file`).
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct Action {
    at_frame: usize,
    #[serde(default)]
    every_frames: Option<usize>,
    #[serde(default)]
    count: Option<usize>,
    #[serde(default)]
    hz: Option<f64>,
    method: String,
    #[serde(default)]
    params: Value,
}

/// Entry point for `render-core --bench`.
pub fn run(args: BenchArgs) -> Result<()> {
    #[cfg(target_os = "macos")]
    crate::hold_latency_critical_assertion();

    let scenario_path = args
        .bench
        .canonicalize()
        .with_context(|| format!("resolving scenario {}", args.bench.display()))?;
    let scenario_dir = scenario_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let raw = std::fs::read_to_string(&scenario_path)
        .with_context(|| format!("reading {}", scenario_path.display()))?;
    let sc: Scenario =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", scenario_path.display()))?;
    let name = scenario_path
        .file_stem()
        .map(|s| s.to_string_lossy().trim_end_matches(".scenario").to_string())
        .unwrap_or_default();
    let warmup = args.warmup.unwrap_or(sc.warmup_frames);
    let frames = args.frames.unwrap_or(sc.frames).max(1);
    let paced = args.paced || sc.paced;
    let actor = Actor::parse(&sc.actor).map_err(|e| anyhow!(e.message))?;

    // --- stage the scene --------------------------------------------------
    let (work_dir, cleanup) = match &args.work_dir {
        Some(d) => (d.clone(), false),
        None => (
            std::env::temp_dir().join(format!("wzrd-bench-{}-{name}", std::process::id())),
            true,
        ),
    };
    let scene_src = scenario_dir.join(&sc.scene);
    let staged_scene = stage_scene(&scene_src, &work_dir, sc.alignment.as_ref().map(|a| scenario_dir.join(a)))?;

    // --- engine -----------------------------------------------------------
    let osc_addr = match &sc.audio {
        Some(_) => free_udp_port()?,
        None => "127.0.0.1:9000".parse().expect("valid addr"),
    };
    let cli = Cli {
        pack: None,
        scene: staged_scene.clone(),
        effects: None,
        display: None,
        windowed: false,
        no_osc: sc.audio.is_none(),
        osc_addr,
        ws_addr: None,
        // Paced: the Tauri host's 240 Hz cap + the virtual vsync, the same
        // pair a windowed engine runs under. Uncapped: neither.
        frame_cap_hz: if paced { 240 } else { 0 },
        perf: args.perf,
        gpu_timing: args.gpu_timing,
    };
    let mut core = Core::with_topology(&cli, sc.two_leg)?;
    core.init_gpu_offscreen(sc.size[0], sc.size[1], paced.then_some(crate::perf::BUDGET_HZ))?;
    if let Some(pv) = &sc.preview {
        core.attach_preview_offscreen(pv.size[0], pv.size[1])?;
        core.set_preview_params(pv.params);
        core.set_preview_visible(true);
    }
    let adapter = core.adapter_info();
    let (ctx, cmd_tx) = core.control_channel();

    // Side load: OSC pin, telemetry subscribers, the RPC worker.
    let stop = Arc::new(AtomicBool::new(false));
    let mut side: Vec<JoinHandle<()>> = Vec::new();
    if let Some(pin) = sc.audio.clone() {
        side.push(spawn_osc_pin(osc_addr, pin, Arc::clone(&stop))?);
    }
    let sub_counts: Arc<Mutex<BTreeMap<String, u64>>> = Arc::default();
    let mut sub_ids = Vec::new();
    if !sc.subscribe.is_empty() {
        let channels: HashSet<String> = sc.subscribe.iter().cloned().collect();
        let (id, rx) = ctx.bus.subscribe(channels);
        sub_ids.push(id);
        let counts = Arc::clone(&sub_counts);
        let stop_c = Arc::clone(&stop);
        side.push(std::thread::Builder::new().name("bench-subscriber".into()).spawn(move || {
            while !stop_c.load(Ordering::Relaxed) {
                if let Ok(frame) = rx.recv_timeout(Duration::from_millis(100)) {
                    if let Ok(mut c) = counts.lock() {
                        *c.entry(frame.channel.clone()).or_default() += 1;
                    }
                }
            }
        })?);
    }
    let mut worker = RpcWorker::spawn(ctx.clone(), cmd_tx.clone(), actor)?;

    // Resolve every action's params up front (file reads off the frame path).
    let mut schedule: Vec<(usize, usize, JsonRpcRequest)> = Vec::new(); // (frame, rep, req)
    let mut background: Vec<(usize, f64, Action)> = Vec::new();
    for action in &sc.timeline {
        if let Some(hz) = action.hz {
            if hz <= 0.0 {
                bail!("timeline {}: hz must be > 0", action.method);
            }
            background.push((action.at_frame, hz, action.clone()));
            continue;
        }
        let count = action.count.unwrap_or(1);
        let every = action.every_frames.unwrap_or(1).max(1);
        for rep in 0..count {
            let params = resolve(&action.params, rep, &scenario_dir, &work_dir)?;
            schedule.push((
                action.at_frame + rep * every,
                rep,
                JsonRpcRequest {
                    jsonrpc: Some("2.0".into()),
                    id: None,
                    method: action.method.clone(),
                    params,
                },
            ));
        }
    }
    schedule.sort_by_key(|(f, rep, _)| (*f, *rep));
    let mut next = 0;

    // --- the run ------------------------------------------------------------
    let started = Instant::now();
    let total = warmup + frames;
    let mut resources_start = Value::Null;
    let mut bg_started = vec![false; background.len()];
    for frame in 0..total {
        if frame == warmup {
            core.perf_mut().bench_begin(frames);
            resources_start = core.gpu_resources().unwrap_or(Value::Null);
        }
        while next < schedule.len() && schedule[next].0 <= frame {
            let req = &schedule[next].2;
            worker.submit(JsonRpcRequest {
                jsonrpc: req.jsonrpc.clone(),
                id: None,
                method: req.method.clone(),
                params: req.params.clone(),
            });
            next += 1;
        }
        for (i, (at, hz, action)) in background.iter().enumerate() {
            if !bg_started[i] && *at <= frame {
                bg_started[i] = true;
                side.push(spawn_rate_action(
                    ctx.clone(),
                    cmd_tx.clone(),
                    actor,
                    *hz,
                    action.clone(),
                    scenario_dir.clone(),
                    work_dir.clone(),
                    Arc::clone(&stop),
                    Arc::clone(&worker.results),
                )?);
            }
        }
        core.poll_inbound();
        if core.exit_requested() {
            bail!("interrupted");
        }
        core.pace_frame();
        if let Err(err) = core.redraw() {
            bail!("frame {frame}: {err:?}");
        }
    }
    let perf = core.perf_mut().bench_report();
    let resources_end = core.gpu_resources().unwrap_or(Value::Null);

    // Teardown. Background senders stop, the unsent rest of the timeline is
    // skipped and the worker's queue closes; keep rendering (unrecorded)
    // until every side thread has *exited* — the worker and an `hz` sender
    // block inside `rpc::dispatch` on a render-thread reply, so stopping the
    // loop first would deadlock them. Exit is judged by the threads
    // themselves (`is_finished`), never by a request counter, and a thread
    // still blocked at the deadline is detached, never joined (an orphaned
    // bench once sat for 1h43m at 0% CPU in `join` on a worker waiting for
    // a reply nobody would send).
    stop.store(true, Ordering::Relaxed);
    worker.cancel();
    worker.close();
    for id in sub_ids {
        ctx.bus.unsubscribe(id);
    }
    let never_sent = schedule.len() - next;
    let drain_started = Instant::now();
    let drain_deadline = drain_started + DRAIN_TIMEOUT;
    while !(worker.is_finished() && side.iter().all(JoinHandle::is_finished))
        && Instant::now() < drain_deadline
    {
        core.poll_inbound();
        core.pace_frame();
        let _ = core.redraw();
    }
    let drain_ms = drain_started.elapsed().as_secs_f64() * 1000.0;
    let unfinished = worker.pending();
    let (actions, worker_stuck) = worker.finish();
    let mut stuck_threads = usize::from(worker_stuck);
    for h in side {
        if h.is_finished() {
            let _ = h.join();
        } else {
            stuck_threads += 1; // detached: dies with the process
        }
    }
    if stuck_threads > 0 {
        log::error!(
            "bench {name}: {stuck_threads} thread(s) still blocked after the {} s drain \
             ({unfinished} timeline RPC(s) without a reply); detached, not joined",
            DRAIN_TIMEOUT.as_secs()
        );
    }
    let subscribers = sub_counts.lock().map(|c| c.clone()).unwrap_or_default();
    let wall_s = started.elapsed().as_secs_f64();

    let report = json!({
        "schema": 1,
        "scenario": name,
        "description": sc.description,
        "config": {
            "size": sc.size,
            "warmup_frames": warmup,
            "frames": frames,
            "paced": paced,
            "two_leg": sc.two_leg,
            "gpu_timing": args.gpu_timing,
            "perf": matches!(args.perf, PerfMode::On),
        },
        "machine": machine_info(adapter.as_ref(), &scenario_dir),
        "perf": perf,
        "resources": { "start": resources_start, "end": resources_end },
        "actions": actions,
        "actions_unfinished": unfinished,
        "actions_never_sent": never_sent,
        "teardown": {
            "drain_ms": (drain_ms * 1000.0).round() / 1000.0,
            "stuck_threads": stuck_threads,
        },
        "subscribers": subscribers,
        "wall_s": (wall_s * 1000.0).round() / 1000.0,
    });
    if let Some(dir) = args.bench_out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(&args.bench_out, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("writing {}", args.bench_out.display()))?;
    log::info!("bench {name}: report → {}", args.bench_out.display());

    // The report is on disk; engine teardown (device, decoder threads) must
    // not be able to hang the process past it. A detached thread still in
    // `dispatch` wakes with "reply channel closed" once `core` drops.
    spawn_teardown_watchdog(&name);
    drop(core);
    if cleanup {
        let _ = std::fs::remove_dir_all(&work_dir);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Staging
// ---------------------------------------------------------------------------

/// Copy `scene` into `work_dir` with absolute pack + clip paths, copy its
/// `effects/` dir (an `effect.upsert` writes there) and the optional
/// alignment. Never copies `session.json` or `.wzrd/` drafts — a bench
/// starts from defaults, not from whatever an operator left behind.
fn stage_scene(scene: &Path, work_dir: &Path, alignment: Option<PathBuf>) -> Result<PathBuf> {
    let scene = scene
        .canonicalize()
        .with_context(|| format!("resolving scene {}", scene.display()))?;
    let src_dir = scene.parent().unwrap_or(Path::new("."));
    if work_dir.exists() {
        std::fs::remove_dir_all(work_dir).with_context(|| format!("clearing {}", work_dir.display()))?;
    }
    std::fs::create_dir_all(work_dir)?;
    let out = work_dir.join(scene.file_name().unwrap_or_else(|| "scene.json".as_ref()));
    std::fs::write(&out, staged_scene_json(&scene)?)?;
    // Always an `effects/` dir, so `effect.upsert` has somewhere to write.
    let fx = src_dir.join("effects");
    if fx.is_dir() {
        copy_dir(&fx, &work_dir.join("effects"))?;
    } else {
        std::fs::create_dir_all(work_dir.join("effects"))?;
    }
    if let Some(a) = alignment {
        std::fs::copy(&a, work_dir.join("alignment.json"))
            .with_context(|| format!("staging alignment {}", a.display()))?;
    }
    Ok(out)
}

/// The scene's JSON with `pack` and every `image.clip` made absolute
/// against the scene's own directory (`$scene` variants use this too).
fn staged_scene_json(scene: &Path) -> Result<String> {
    let src_dir = scene.parent().unwrap_or(Path::new("."));
    let raw = std::fs::read_to_string(scene).with_context(|| format!("reading {}", scene.display()))?;
    let mut v: Value = serde_json::from_str(&raw).with_context(|| format!("parsing {}", scene.display()))?;
    let abs = |p: &str| -> String {
        let path = Path::new(p);
        if path.is_absolute() {
            return p.to_string();
        }
        let joined = src_dir.join(path);
        joined.canonicalize().unwrap_or(joined).to_string_lossy().into_owned()
    };
    if let Some(p) = v.get("pack").and_then(Value::as_str).map(abs) {
        v["pack"] = json!(p);
    }
    if let Some(bindings) = v.get_mut("bindings").and_then(Value::as_array_mut) {
        for b in bindings {
            if let Some(clip) = b.pointer("/params/image/clip").and_then(Value::as_str).map(abs) {
                b["params"]["image"]["clip"] = json!(clip);
            }
        }
    }
    Ok(serde_json::to_string_pretty(&v)?)
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let dest = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &dest)?;
        } else {
            std::fs::copy(entry.path(), &dest)?;
        }
    }
    Ok(())
}

/// Expand `$alt` / `$scene` / `$file` / `$json` directives (see [`Action`]).
fn resolve(v: &Value, rep: usize, scenario_dir: &Path, work_dir: &Path) -> Result<Value> {
    Ok(match v {
        Value::Object(m) if m.len() == 1 && m.contains_key("$alt") => {
            let alts = m["$alt"].as_array().filter(|a| !a.is_empty()).ok_or_else(|| anyhow!("$alt needs a non-empty array"))?;
            resolve(&alts[rep % alts.len()], rep, scenario_dir, work_dir)?
        }
        Value::Object(m) if m.len() == 1 && m.contains_key("$scene") => {
            let rel = m["$scene"].as_str().ok_or_else(|| anyhow!("$scene needs a path"))?;
            let path = scenario_dir.join(rel);
            json!(staged_scene_json(&path.canonicalize().with_context(|| format!("resolving {}", path.display()))?)?)
        }
        Value::Object(m) if m.len() == 1 && m.contains_key("$json") => {
            let rel = m["$json"].as_str().ok_or_else(|| anyhow!("$json needs a path"))?;
            let path = scenario_dir.join(rel);
            let raw = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?
        }
        Value::Object(m) if m.len() == 1 && m.contains_key("$path") => {
            let rel = m["$path"].as_str().ok_or_else(|| anyhow!("$path needs a path"))?;
            let path = scenario_dir.join(rel);
            json!(path.canonicalize().with_context(|| format!("resolving {}", path.display()))?.to_string_lossy())
        }
        Value::Object(m) if m.len() == 1 && m.contains_key("$file") => {
            let rel = m["$file"].as_str().ok_or_else(|| anyhow!("$file needs a path"))?;
            let path = scenario_dir.join(rel);
            json!(std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?)
        }
        Value::Object(m) => {
            let mut out = Map::new();
            for (k, val) in m {
                out.insert(k.clone(), resolve(val, rep, scenario_dir, work_dir)?);
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(|x| resolve(x, rep, scenario_dir, work_dir)).collect::<Result<_>>()?),
        other => other.clone(),
    })
}

// ---------------------------------------------------------------------------
// Side load
// ---------------------------------------------------------------------------

/// Per-method RPC outcome over the run: count, errors, round-trip ms.
#[derive(Default)]
struct MethodStats {
    ms: Vec<f64>,
    errors: Vec<String>,
    /// Still queued when the measured window ended (never sent).
    skipped: usize,
}

type Results = Arc<Mutex<BTreeMap<String, MethodStats>>>;

fn record(results: &Results, method: &str, elapsed: Duration, outcome: Result<Value, rpc::RpcError>) {
    if let Ok(mut r) = results.lock() {
        let s = r.entry(method.to_string()).or_default();
        s.ms.push(elapsed.as_secs_f64() * 1000.0);
        if let Err(e) = outcome {
            s.errors.push(e.message);
        }
    }
}

/// One worker thread dispatching the timeline in order — the shape of a WS
/// connection: queued methods block it until the render thread replies.
struct RpcWorker {
    tx: Option<crossbeam_channel::Sender<JsonRpcRequest>>,
    pending: Arc<AtomicUsize>,
    /// Set when the measured window ends: queued-but-unsent requests are
    /// skipped (counted), only the one in flight still needs its reply.
    cancel: Arc<AtomicBool>,
    results: Results,
    join: Option<JoinHandle<()>>,
}

impl RpcWorker {
    fn spawn(ctx: RpcContext, cmd_tx: crossbeam_channel::Sender<EngineCommand>, actor: Actor) -> Result<Self> {
        let (tx, rx) = crossbeam_channel::unbounded::<JsonRpcRequest>();
        let pending = Arc::new(AtomicUsize::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        let results: Results = Arc::default();
        let (p, r, c) = (Arc::clone(&pending), Arc::clone(&results), Arc::clone(&cancel));
        let join = std::thread::Builder::new().name("bench-rpc".into()).spawn(move || {
            let mut actor = actor;
            for req in rx {
                if c.load(Ordering::Relaxed) {
                    if let Ok(mut r) = r.lock() {
                        r.entry(req.method.clone()).or_default().skipped += 1;
                    }
                    p.fetch_sub(1, Ordering::AcqRel);
                    continue;
                }
                let t0 = Instant::now();
                let out = rpc::dispatch(&ctx, &cmd_tx, &req, &mut actor);
                record(&r, &req.method, t0.elapsed(), out);
                p.fetch_sub(1, Ordering::AcqRel);
            }
        })?;
        Ok(Self { tx: Some(tx), pending, cancel, results, join: Some(join) })
    }

    fn submit(&self, req: JsonRpcRequest) {
        if let Some(tx) = &self.tx {
            self.pending.fetch_add(1, Ordering::AcqRel);
            let _ = tx.send(req);
        }
    }

    fn pending(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }

    fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Close the queue: the worker skips what is still queued (cancelled)
    /// and exits once the request in flight has its reply.
    fn close(&mut self) {
        self.tx = None;
    }

    /// The worker thread has exited (or was never spawned).
    fn is_finished(&self) -> bool {
        self.join.as_ref().map_or(true, JoinHandle::is_finished)
    }

    /// Join the worker if it has exited (a thread still blocked in
    /// `rpc::dispatch` is detached instead: it dies with the process, and
    /// joining it would wait for a render-thread reply that never comes),
    /// then summarise per method. Returns `(summary, worker_stuck)`.
    fn finish(mut self) -> (Value, bool) {
        self.tx = None;
        let mut stuck = false;
        if let Some(j) = self.join.take() {
            if j.is_finished() {
                let _ = j.join();
            } else {
                stuck = true;
            }
        }
        let r = self.results.lock().map(|mut r| std::mem::take(&mut *r)).unwrap_or_default();
        let mut out = Map::new();
        for (method, mut s) in r {
            s.ms.sort_by(|a, b| a.total_cmp(b));
            let pick = |q: f64| {
                let rank = ((q * s.ms.len() as f64).ceil() as usize).clamp(1, s.ms.len().max(1));
                s.ms.get(rank - 1).copied().map(|v| (v * 1000.0).round() / 1000.0)
            };
            out.insert(
                method,
                json!({
                    "n": s.ms.len(),
                    "errors": s.errors.len(),
                    "skipped": s.skipped,
                    "first_errors": s.errors.iter().take(3).collect::<Vec<_>>(),
                    "p50_ms": pick(0.5),
                    "p95_ms": pick(0.95),
                    "max_ms": s.ms.last().map(|v| (v * 1000.0).round() / 1000.0),
                }),
            );
        }
        (Value::Object(out), stuck)
    }
}

/// Exit the process if engine teardown hangs once the report is written.
fn spawn_teardown_watchdog(name: &str) {
    let name = name.to_string();
    let _ = std::thread::Builder::new().name("bench-watchdog".into()).spawn(move || {
        std::thread::sleep(TEARDOWN_TIMEOUT);
        log::error!(
            "bench {name}: engine teardown still running after {} s; exiting (report already written)",
            TEARDOWN_TIMEOUT.as_secs()
        );
        std::process::exit(3);
    });
}

/// A timeline entry with `hz`: fire it at a fixed rate on its own thread
/// until the run ends (a knob storm — inline methods like `param.set`
/// never wait on the render thread).
#[allow(clippy::too_many_arguments)]
fn spawn_rate_action(
    ctx: RpcContext,
    cmd_tx: crossbeam_channel::Sender<EngineCommand>,
    actor: Actor,
    hz: f64,
    action: Action,
    scenario_dir: PathBuf,
    work_dir: PathBuf,
    stop: Arc<AtomicBool>,
    results: Results,
) -> Result<JoinHandle<()>> {
    // Pre-resolve a cycle of params (the `$alt` set) so the loop never reads files.
    let cycle = match action.params.pointer("/value").and_then(|v| v.get("$alt")).and_then(Value::as_array) {
        Some(a) => a.len().max(1),
        None => 2,
    };
    let reqs = (0..cycle)
        .map(|rep| {
            Ok(JsonRpcRequest {
                jsonrpc: Some("2.0".into()),
                id: None,
                method: action.method.clone(),
                params: resolve(&action.params, rep, &scenario_dir, &work_dir)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let period = Duration::from_secs_f64(1.0 / hz);
    Ok(std::thread::Builder::new().name("bench-rate".into()).spawn(move || {
        let mut actor = actor;
        let mut next = Instant::now();
        let mut i = 0usize;
        while !stop.load(Ordering::Relaxed) {
            let req = &reqs[i % reqs.len()];
            let t0 = Instant::now();
            let out = rpc::dispatch(&ctx, &cmd_tx, req, &mut actor);
            record(&results, &req.method, t0.elapsed(), out);
            i += 1;
            next += period;
            let now = Instant::now();
            if next <= now {
                next = now;
            }
            // Sleep in slices so teardown never waits out a slow period.
            while !stop.load(Ordering::Relaxed) {
                let left = next.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                std::thread::sleep(left.min(Duration::from_millis(20)));
            }
        }
    })?)
}

/// Pin `audio.*` by feeding the engine's own OSC listener at 100 Hz (the
/// real ingest path, so `osc_age` reads something too).
fn spawn_osc_pin(addr: SocketAddr, pin: AudioPin, stop: Arc<AtomicBool>) -> Result<JoinHandle<()>> {
    let socket = UdpSocket::bind("127.0.0.1:0").context("binding OSC sender")?;
    let encode = |addr: &str, args: Vec<rosc::OscType>| {
        rosc::encoder::encode(&rosc::OscPacket::Message(rosc::OscMessage { addr: addr.into(), args }))
            .expect("encodable OSC message")
    };
    let lmh = encode("/audio/lmh", pin.lmh.iter().map(|v| rosc::OscType::Float(*v)).collect());
    let onsets: Vec<Vec<u8>> = ["/audio/onset/low", "/audio/onset/mid", "/audio/onset/high"]
        .iter()
        .map(|a| encode(a, vec![rosc::OscType::Float(1.0)]))
        .collect();
    let onset_every = (pin.onset_hz > 0.0).then(|| Duration::from_secs_f64(1.0 / pin.onset_hz as f64));
    Ok(std::thread::Builder::new().name("bench-osc".into()).spawn(move || {
        let mut last_onset = Instant::now();
        while !stop.load(Ordering::Relaxed) {
            let _ = socket.send_to(&lmh, addr);
            if let Some(every) = onset_every {
                if last_onset.elapsed() >= every {
                    for o in &onsets {
                        let _ = socket.send_to(o, addr);
                    }
                    last_onset = Instant::now();
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    })?)
}

fn free_udp_port() -> Result<SocketAddr> {
    let s = UdpSocket::bind("127.0.0.1:0").context("finding a free UDP port")?;
    Ok(s.local_addr()?)
}

// ---------------------------------------------------------------------------
// Machine info
// ---------------------------------------------------------------------------

fn machine_info(adapter: Option<&crate::wgpu::AdapterInfo>, dir: &Path) -> Value {
    let git = |args: &[&str]| -> Option<String> {
        let out = std::process::Command::new("git").args(args).current_dir(dir).output().ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let sha = git(&["rev-parse", "--short=12", "HEAD"]);
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).map(|s| !s.is_empty());
    json!({
        "adapter": adapter.map(|a| a.name.clone()),
        "backend": adapter.map(|a| format!("{:?}", a.backend)),
        "driver": adapter.map(|a| format!("{} {}", a.driver, a.driver_info).trim().to_string()),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "git": { "sha": sha, "dirty": dirty },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Staging is what keeps a bench out of real projects: the copy must
    /// resolve the pack and clips back to the source (a broken clip path is
    /// only a warning in the engine, so the gate would silently measure an
    /// empty binding) and must not carry operator state along.
    #[test]
    fn staging_makes_paths_absolute_and_leaves_operator_state_behind() {
        let root = std::env::temp_dir().join(format!("wzrd-bench-stage-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let src = root.join("project");
        std::fs::create_dir_all(src.join("effects/glow")).unwrap();
        std::fs::create_dir_all(src.join("packs/main")).unwrap();
        std::fs::create_dir_all(src.join("generated")).unwrap();
        std::fs::write(src.join("generated/a.mov"), b"").unwrap();
        std::fs::write(src.join("effects/glow/shader.wgsl"), "// fx").unwrap();
        std::fs::write(src.join("session.json"), "{}").unwrap();
        std::fs::write(
            src.join("scene.json"),
            r#"{"version":1,"pack":"packs/main","bindings":[
                {"id":"c","select":{"all":true},"effect":"videoClip","params":{"image":{"clip":"generated/a.mov"}}}]}"#,
        )
        .unwrap();

        let work = root.join("work");
        let staged = stage_scene(&src.join("scene.json"), &work, None).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&staged).unwrap()).unwrap();
        let src = src.canonicalize().unwrap();
        assert_eq!(v["pack"], json!(src.join("packs/main").to_string_lossy()));
        assert_eq!(
            v.pointer("/bindings/0/params/image/clip").unwrap(),
            &json!(src.join("generated/a.mov").to_string_lossy())
        );
        assert!(work.join("effects/glow/shader.wgsl").exists());
        assert!(!work.join("session.json").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
