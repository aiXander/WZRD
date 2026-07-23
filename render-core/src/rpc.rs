//! JSON-RPC dispatch (§3.11).
//!
//! One logical method set, two consumers: the WS server thread routes
//! requests through `dispatch_inline` for synchronous, side-effect-free
//! methods (`wgsl.validate`, `pack.info`, `scene.getState`), and queues
//! state-mutating methods (`scene.load`, `effect.upsert`, `effect.remove`)
//! as `EngineCommand`s that the render thread drains at frame boundary in
//! `Core::poll_inbound`. Same method names, same params, same error shapes —
//! Phase 7's MCP wrapper proxies the same surface unchanged.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Result};
use crossbeam_channel::Sender;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::core::Core;
use crate::drivers::{Masters, ParamOverrides, SliderBank};
use crate::pack::LoadedPack;
use crate::session;
use crate::telemetry::{Bus, ChangeLog};

/// §5.10 — who a design mutation came from. Declared **per connection**
/// (never per call — per-call tagging is forgettable and accident-spoofable):
/// Tauri direct dispatch passes `Ui`, WS connections default to `Agent` and
/// may re-declare via `hello {actor}`. Engine-internal mutations (file
/// watcher, autosave restore) are `System`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Actor {
    Ui,
    Agent,
    System,
}

impl Actor {
    pub fn as_str(&self) -> &'static str {
        match self {
            Actor::Ui => "ui",
            Actor::Agent => "agent",
            Actor::System => "system",
        }
    }

    pub fn parse(s: &str) -> Result<Self, RpcError> {
        match s {
            "ui" => Ok(Actor::Ui),
            "agent" => Ok(Actor::Agent),
            "system" => Ok(Actor::System),
            other => Err(RpcError::message(format!(
                "unknown actor {other:?} (\"ui\" | \"agent\" | \"system\")"
            ))),
        }
    }
}

/// Swappable current `pack.info` snapshot. Static for the engine's lifetime
/// *except* §5.13 `identity.setGroups`, which refreshes it so every client
/// (and the §5.10 MCP digest) reads the merged groups/labels immediately.
pub struct PackInfoCell {
    inner: RwLock<Arc<PackInfo>>,
}

impl PackInfoCell {
    pub fn new(info: PackInfo) -> Self {
        Self {
            inner: RwLock::new(Arc::new(info)),
        }
    }

    pub fn get(&self) -> Arc<PackInfo> {
        Arc::clone(&self.inner.read().expect("pack info lock"))
    }

    pub fn set(&self, info: PackInfo) {
        *self.inner.write().expect("pack info lock") = Arc::new(info);
    }
}

/// A command that mutates engine state. Issued from the WS server thread;
/// processed on the render thread.
pub enum EngineCommand {
    /// Replace the current scene with the given JSON. Doesn't touch disk.
    /// §5.10: `base_rev` is a compare-and-swap guard — the engine rejects
    /// the apply when the design rev moved since the caller's read.
    SceneLoad {
        json: String,
        base_rev: Option<u64>,
        actor: Actor,
        reply: Sender<Result<Value, String>>,
    },
    /// Reload the scene from `scene_path` (same path the engine was started
    /// with). Useful when the UI saved to disk and wants an explicit
    /// reload trigger without waiting on the file watcher debounce.
    SceneReload {
        actor: Actor,
        reply: Sender<Result<Value, String>>,
    },
    /// Write `wgsl` to the user-effects directory under `name/shader.wgsl`,
    /// optionally write a `descriptor.json`, then re-scan the registry and
    /// re-apply the design scene so the new pipeline is probed — §5.10
    /// verdict-in-reply: the reply defers through the probe session and
    /// carries the verdict + thumbnail, exactly like `scene.load`.
    EffectUpsert {
        name: String,
        wgsl: String,
        descriptor: Option<Value>,
        actor: Actor,
        reply: Sender<Result<Value, String>>,
    },
    /// Remove a project-local effect directory and invalidate its pipeline.
    EffectRemove {
        name: String,
        actor: Actor,
        reply: Sender<Result<Value, String>>,
    },
    /// §5.13 — merge a delta into the pack's identity sidecar (groups +
    /// labels), write it, refresh `pack.info`, re-resolve selectors.
    IdentitySet {
        groups: Option<BTreeMap<String, Option<Vec<String>>>>,
        labels: Option<BTreeMap<String, Option<String>>>,
        actor: Actor,
        reply: Sender<Result<Value, String>>,
    },
    /// §5.5 — describe one effect's inputs (or the whole catalog). Queued
    /// because the registry lives on the render thread.
    EffectDescribe {
        name: Option<String>,
        reply: Sender<Result<Value, String>>,
    },
    /// §5.3 — explicit session sidecar save (masters + knobs + calibration).
    SessionSave {
        reply: Sender<Result<Value, String>>,
    },
    /// §5.6 — crossfade the projector from the live composite to the design
    /// composite, then adopt design's plan into the live slot. `quantize`
    /// "bar" defers the ramp start to the next bar boundary; "now" starts
    /// immediately. Re-entrancy: a *pending* quantized promote is replaced
    /// by a newer one; while a fade is actively ramping, promote and pull
    /// are rejected.
    Promote {
        fade_ms: f32,
        quantize: crate::core::Quantize,
        reply: Sender<Result<Value, String>>,
    },
    /// §5.6 — hard-copy live's scene back into design (the explicit
    /// reverse of promote).
    Pull {
        actor: Actor,
        reply: Sender<Result<Value, String>>,
    },
    /// §5.6 — select which leg's composite the native preview samples.
    PreviewSource {
        source: crate::gpu::Leg,
        reply: Sender<Result<Value, String>>,
    },
}

/// Bound bag of references used to answer the synchronous (read-only)
/// requests. Held by the WS server alongside the command channel.
#[derive(Clone)]
pub struct RpcContext {
    /// Current pack info snapshot. Static except §5.13 `identity.setGroups`,
    /// which swaps in a refreshed merge (see [`PackInfoCell`]).
    pub pack: Arc<PackInfoCell>,
    /// §5.10 — design rev counter + boot epoch + change ring, shared with
    /// the render thread (which records) and every read surface.
    pub changes: Arc<ChangeLog>,
    /// Most recent **design-leg** scene JSON (§5.6 blanket leg rule: reads
    /// follow design). Headless single-leg, this is the one live scene.
    pub scene_state: Arc<parking_lot_lite::SwapValue>,
    /// §5.6 — the **live-leg** scene JSON, served by
    /// `scene.getState { leg: "live" }` so `pull` is verifiable over RPC.
    /// Updated at boot and on every promote completion.
    pub live_scene_state: Arc<parking_lot_lite::SwapValue>,
    pub effects_dir: Option<PathBuf>,
    pub bus: Bus,
    /// **Live-leg** `ui.slider` values. `param.set` writes here directly (no
    /// render-thread hop) — the render thread reads it on its next tick, so
    /// knob latency is bounded by one frame.
    pub sliders: Arc<SliderBank>,
    /// §5.4 **live-leg** masters — written inline by `master.set`, same
    /// latency contract as the slider bank.
    pub masters: Arc<Masters>,
    /// §5.5 **live-leg** per-binding scalar overrides — written inline by
    /// the `param.set {binding, param, value}` form.
    pub overrides: Arc<ParamOverrides>,
    /// §5.6 full-control-switch: the design leg's own control state. On
    /// single-leg (headless) runs these alias the live Arcs, so the `leg`
    /// param is a no-op there. `param.set`/`master.set` default to the
    /// design leg (blanket leg rule); the UI passes `leg` explicitly from
    /// the deck toggle.
    pub design_sliders: Arc<SliderBank>,
    pub design_masters: Arc<Masters>,
    pub design_overrides: Arc<ParamOverrides>,
    /// §5.4 crossfade-time master — the engine-wide default promote fade
    /// (seconds). Not per leg: `master.set {name:"crossfade"}` ignores `leg`.
    pub crossfade: Arc<crate::drivers::Crossfade>,
    /// §5.3 dirty stamp (epoch ms of last operator-state change; 0 = clean).
    /// The render thread debounces session sidecar writes on it.
    pub session_dirty: Arc<AtomicU64>,
    /// §5.6 probe thresholds A < B — written inline by
    /// `probe.setThresholds`, read by the render thread at probe verdicts,
    /// persisted in the session sidecar.
    pub probe_thresholds: Arc<crate::probe::ProbeThresholds>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PackInfo {
    pub pack_dir: String,
    pub width: u32,
    pub height: u32,
    pub layers: Vec<LayerInfo>,
    pub groups: Vec<GroupInfo>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerInfo {
    pub id: String,
    pub slice: u32,
    pub mask_path: String,
    pub label: Option<String>,
    pub tags: Vec<String>,
    pub bbox: Option<[i32; 4]>,
    pub centroid: Option<[f32; 2]>,
    pub z: i32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GroupInfo {
    pub id: String,
    pub members: Vec<String>,
}

impl PackInfo {
    pub fn from_pack(pack: &LoadedPack) -> Self {
        let layers = pack
            .manifest
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| LayerInfo {
                id: l.id.clone(),
                slice: i as u32,
                mask_path: l.mask.clone(),
                // §5.13 — identity-sidecar label override wins.
                label: pack.merged_label(i),
                tags: l.tags.clone(),
                bbox: l.bbox,
                centroid: l.centroid,
                z: l.z,
            })
            .collect();
        let groups = pack
            .merged_groups
            .iter()
            .map(|g| GroupInfo {
                id: g.id.clone(),
                members: g.members.clone(),
            })
            .collect();
        Self {
            pack_dir: pack.pack_dir.display().to_string(),
            width: pack.atlas_width,
            height: pack.atlas_height,
            layers,
            groups,
        }
    }
}

/// §5.6 — the leg a control-surface write targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LegSel {
    Live,
    Design,
}

impl LegSel {
    fn as_str(&self) -> &'static str {
        match self {
            LegSel::Live => "live",
            LegSel::Design => "design",
        }
    }
}

/// Default is design (blanket leg rule — agents/authoring never touch the
/// crowd by accident); the UI passes the deck toggle's leg explicitly.
/// Single-leg runs alias both legs to the same state, so this is a no-op
/// there.
fn parse_leg(leg: Option<&str>) -> Result<LegSel, RpcError> {
    match leg {
        None | Some("design") => Ok(LegSel::Design),
        Some("live") => Ok(LegSel::Live),
        Some(other) => Err(RpcError::message(format!(
            "unknown leg {other:?} (\"design\" | \"live\")"
        ))),
    }
}

/// Both legs' masters — the `masters` telemetry / `master.list` payload.
pub fn masters_state(ctx: &RpcContext) -> crate::telemetry::MastersState {
    crate::telemetry::MastersState {
        live: ctx.masters.snapshot(),
        design: ctx.design_masters.snapshot(),
        crossfade: ctx.crossfade.seconds(),
    }
}

/// Parsed JSON-RPC 2.0 request envelope.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: Option<String>,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// Route a request. For synchronous methods (read-only or strictly local
/// compute) return the result immediately; for mutation methods queue an
/// `EngineCommand` and block on its reply (the caller's WS thread blocks,
/// the render thread never blocks).
///
/// `actor` is the per-connection identity (§5.10): the WS server holds one
/// per connection (default `Agent`, re-declared by `hello`); the Tauri host
/// passes `Ui`. Queued design mutations carry it into the change ring.
pub fn dispatch(
    ctx: &RpcContext,
    cmd_tx: &Sender<EngineCommand>,
    req: &JsonRpcRequest,
    actor: &mut Actor,
) -> Result<Value, RpcError> {
    match req.method.as_str() {
        "pack.info" => {
            Ok(serde_json::to_value(ctx.pack.get().as_ref()).expect("pack info"))
        }

        // §5.10 — declare this connection's actor identity, once per
        // session. Returns the boot epoch + current design rev so a client
        // can seed its `since_rev` cursor in the same round-trip.
        "hello" => {
            #[derive(Deserialize)]
            struct Params {
                actor: String,
            }
            let p: Params = serde_json::from_value(req.params.clone())
                .map_err(|e| RpcError::message(format!("params: {e}")))?;
            *actor = Actor::parse(&p.actor)?;
            Ok(json!({
                "ok": true,
                "actor": actor.as_str(),
                "epoch": ctx.changes.epoch(),
                "rev": ctx.changes.rev(),
            }))
        }

        // §5.10 — change-ring backfill (sticky `changes` replay only carries
        // the last entry). `since_rev` from another epoch or older than the
        // ring's tail returns everything the ring holds with an explicit
        // note — never a silently-partial diff.
        "changes.list" => {
            #[derive(Deserialize, Default)]
            #[serde(default)]
            struct Params {
                since_rev: Option<u64>,
                epoch: Option<u64>,
            }
            let p: Params = if req.params.is_null() {
                Params::default()
            } else {
                serde_json::from_value(req.params.clone())
                    .map_err(|e| RpcError::message(format!("params: {e}")))?
            };
            Ok(ctx.changes.list_json(p.since_rev, p.epoch))
        }

        // §5.13 — identity sidecar delta: groups + labels. Queued (mutates
        // the pack + selector maps owned by the render thread).
        "identity.setGroups" => {
            #[derive(Deserialize, Default)]
            #[serde(default)]
            struct Params {
                groups: Option<BTreeMap<String, Option<Vec<String>>>>,
                labels: Option<BTreeMap<String, Option<String>>>,
            }
            let p: Params = serde_json::from_value(req.params.clone())
                .map_err(|e| RpcError::message(format!("params: {e}")))?;
            if p.groups.is_none() && p.labels.is_none() {
                return Err(RpcError::message(
                    "identity.setGroups takes { groups?: { id: [layerIds] | null }, \
                     labels?: { layerId: label | null } } — provide at least one",
                ));
            }
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            cmd_tx
                .send(EngineCommand::IdentitySet {
                    groups: p.groups,
                    labels: p.labels,
                    actor: *actor,
                    reply: reply_tx,
                })
                .map_err(|_| RpcError::message("engine command channel closed"))?;
            reply_rx
                .recv()
                .map_err(|_| RpcError::message("engine reply channel closed"))?
                .map_err(RpcError::message)
        }

        "scene.getState" => {
            // §5.6 blanket leg rule: reads follow design by default; the
            // optional `leg` param makes `pull` verifiable over RPC.
            #[derive(Deserialize, Default)]
            #[serde(default)]
            struct Params {
                leg: Option<String>,
            }
            let p: Params = if req.params.is_null() {
                Params::default()
            } else {
                serde_json::from_value(req.params.clone())
                    .map_err(|e| RpcError::message(format!("params: {e}")))?
            };
            let state = match p.leg.as_deref() {
                None | Some("design") => &ctx.scene_state,
                Some("live") => &ctx.live_scene_state,
                Some(other) => {
                    return Err(RpcError::message(format!(
                        "unknown leg {other:?} (\"design\" | \"live\")"
                    )))
                }
            };
            let raw = state
                .get()
                .ok_or_else(|| RpcError::message("no scene loaded"))?;
            // §5.10 — the read carries the design rev so a read-modify-write
            // caller can pass it back as `scene.load`'s `base_rev` CAS guard.
            Ok(json!({
                "json": raw,
                "leg": p.leg.unwrap_or_else(|| "design".into()),
                "epoch": ctx.changes.epoch(),
                "rev": ctx.changes.rev(),
            }))
        }

        // §5.6 probe thresholds — inline like the masters (venue state,
        // written from the GUI, persisted in the session sidecar).
        "probe.getThresholds" => {
            Ok(serde_json::to_value(ctx.probe_thresholds.snapshot()).expect("thresholds"))
        }

        "probe.setThresholds" => {
            #[derive(Deserialize)]
            struct Params {
                a_ms: f32,
                b_ms: f32,
            }
            let p: Params = serde_json::from_value(req.params.clone())
                .map_err(|e| RpcError::message(format!("params: {e}")))?;
            ctx.probe_thresholds
                .set(p.a_ms, p.b_ms)
                .map_err(|e| RpcError::message(format!("{e:#}")))?;
            session::touch(&ctx.session_dirty);
            Ok(serde_json::to_value(ctx.probe_thresholds.snapshot()).expect("thresholds"))
        }

        // §5.6 two-deck verbs — queued: they mutate render-thread state and
        // the re-entrancy rules (pending replaced / mid-ramp rejected) must
        // be evaluated on the thread that owns the fade.
        "promote" => {
            #[derive(Deserialize, Default)]
            #[serde(default)]
            struct Params {
                fade_ms: Option<f32>,
                quantize: Option<String>,
            }
            let p: Params = if req.params.is_null() {
                Params::default()
            } else {
                serde_json::from_value(req.params.clone())
                    .map_err(|e| RpcError::message(format!("params: {e}")))?
            };
            // Default to the §5.4 crossfade-time master when the caller omits
            // an explicit fade (headless/MCP promotes, and the UI once it
            // hands the slider off to the master).
            let fade_ms = p.fade_ms.unwrap_or_else(|| ctx.crossfade.ms());
            if !fade_ms.is_finite() || fade_ms < 0.0 {
                return Err(RpcError::message("fade_ms must be finite and >= 0"));
            }
            let quantize = match p.quantize.as_deref() {
                None | Some("bar") => crate::core::Quantize::Bar,
                Some("now") => crate::core::Quantize::Now,
                Some(other) => {
                    return Err(RpcError::message(format!(
                        "unknown quantize {other:?} (\"bar\" | \"now\")"
                    )))
                }
            };
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            cmd_tx
                .send(EngineCommand::Promote {
                    fade_ms,
                    quantize,
                    reply: reply_tx,
                })
                .map_err(|_| RpcError::message("engine command channel closed"))?;
            reply_rx
                .recv()
                .map_err(|_| RpcError::message("engine reply channel closed"))?
                .map_err(RpcError::message)
        }

        "pull" => {
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            cmd_tx
                .send(EngineCommand::Pull {
                    actor: *actor,
                    reply: reply_tx,
                })
                .map_err(|_| RpcError::message("engine command channel closed"))?;
            reply_rx
                .recv()
                .map_err(|_| RpcError::message("engine reply channel closed"))?
                .map_err(RpcError::message)
        }

        "preview.setSource" => {
            #[derive(Deserialize)]
            struct Params {
                source: String,
            }
            let p: Params = serde_json::from_value(req.params.clone())
                .map_err(|e| RpcError::message(format!("params: {e}")))?;
            let source = match p.source.as_str() {
                "live" => crate::gpu::Leg::Live,
                "design" => crate::gpu::Leg::Design,
                other => {
                    return Err(RpcError::message(format!(
                        "unknown preview source {other:?} (\"live\" | \"design\")"
                    )))
                }
            };
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            cmd_tx
                .send(EngineCommand::PreviewSource {
                    source,
                    reply: reply_tx,
                })
                .map_err(|_| RpcError::message("engine command channel closed"))?;
            reply_rx
                .recv()
                .map_err(|_| RpcError::message("engine reply channel closed"))?
                .map_err(RpcError::message)
        }

        "wgsl.validate" => {
            #[derive(Deserialize)]
            struct Params {
                source: String,
            }
            let p: Params = serde_json::from_value(req.params.clone())
                .map_err(|e| RpcError::message(format!("params: {e}")))?;
            let diagnostics = validate_wgsl(&p.source);
            Ok(serde_json::to_value(diagnostics).expect("diagnostics"))
        }

        "telemetry.channels" => {
            Ok(json!({ "channels": crate::telemetry::ALL_CHANNELS }))
        }

        // Live tuning surface (§4 of the design spec — "tune by feel, not by
        // re-prompting"). Two addressing forms, both zero-rebuild:
        //   { name, value }            → named ui.slider (scene-authored
        //                                shared knob, SliderBank)
        //   { binding, param, value }  → §5.5 override of any scalar param
        //                                on a binding — const or driver
        //                                output alike; value: null clears.
        // §5.6 full-control-switch: each leg owns its knobs. Optional
        // `leg: "design"` (default) | "live" picks the target; the UI passes
        // it from the deck toggle. Both forms mark the §5.3 session sidecar
        // dirty so tuning survives a restart.
        "param.set" => {
            #[derive(Deserialize)]
            struct Params {
                #[serde(default)]
                name: Option<String>,
                #[serde(default)]
                binding: Option<String>,
                #[serde(default)]
                param: Option<String>,
                #[serde(default)]
                value: Option<f32>,
                #[serde(default)]
                leg: Option<String>,
            }
            let p: Params = serde_json::from_value(req.params.clone())
                .map_err(|e| RpcError::message(format!("params: {e}")))?;
            if let Some(v) = p.value {
                if !v.is_finite() {
                    return Err(RpcError::message("value must be finite"));
                }
            }
            let leg = parse_leg(p.leg.as_deref())?;
            let (sliders, overrides) = match leg {
                LegSel::Live => (&ctx.sliders, &ctx.overrides),
                LegSel::Design => (&ctx.design_sliders, &ctx.design_overrides),
            };
            match (p.name, p.binding, p.param) {
                (Some(name), None, None) => {
                    let value = p.value.ok_or_else(|| {
                        RpcError::message("slider form requires `value`")
                    })?;
                    sliders.set(&name, value);
                    session::touch(&ctx.session_dirty);
                    Ok(json!({ "ok": true, "name": name, "value": value, "leg": leg.as_str() }))
                }
                (None, Some(binding), Some(param)) => {
                    match p.value {
                        Some(v) => overrides.set(&binding, &param, v),
                        None => {
                            overrides.clear(&binding, &param);
                        }
                    }
                    session::touch(&ctx.session_dirty);
                    Ok(json!({
                        "ok": true,
                        "binding": binding,
                        "param": param,
                        "value": p.value,
                        "leg": leg.as_str(),
                    }))
                }
                _ => Err(RpcError::message(
                    "param.set takes either { name, value } or { binding, param, value } \
                     (value: null clears an override; optional leg: \"design\" | \"live\")",
                )),
            }
        }

        "param.list" => {
            #[derive(Deserialize, Default)]
            #[serde(default)]
            struct Params {
                leg: Option<String>,
            }
            let p: Params = if req.params.is_null() {
                Params::default()
            } else {
                serde_json::from_value(req.params.clone())
                    .map_err(|e| RpcError::message(format!("params: {e}")))?
            };
            let leg = parse_leg(p.leg.as_deref())?;
            let (bank, table) = match leg {
                LegSel::Live => (&ctx.sliders, &ctx.overrides),
                LegSel::Design => (&ctx.design_sliders, &ctx.design_overrides),
            };
            let sliders: serde_json::Map<String, Value> = bank
                .snapshot()
                .into_iter()
                .map(|(k, v)| (k, json!(v)))
                .collect();
            let mut overrides = serde_json::Map::new();
            for (binding, param, value) in table.snapshot() {
                overrides
                    .entry(binding)
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .expect("override entry is object")
                    .insert(param, json!(value));
            }
            Ok(json!({ "sliders": sliders, "overrides": overrides, "leg": leg.as_str() }))
        }

        // §5.4 masters — operator-owned globals, deliberately not reachable
        // through scene.json. Inline write, one-frame latency, sticky
        // `masters` telemetry so every client converges on the same values.
        // §5.6: per-leg (`leg` optional, design default); the telemetry
        // payload carries both legs.
        "master.set" => {
            #[derive(Deserialize)]
            struct Params {
                name: String,
                value: f32,
                #[serde(default)]
                leg: Option<String>,
            }
            let p: Params = serde_json::from_value(req.params.clone())
                .map_err(|e| RpcError::message(format!("params: {e}")))?;
            if !p.value.is_finite() {
                return Err(RpcError::message("value must be finite"));
            }
            // §5.4 crossfade-time master is engine-wide, not per leg — a
            // promote is one global action. `leg` is ignored for it.
            if p.name == "crossfade" {
                let stored = ctx.crossfade.set(p.value);
                session::touch(&ctx.session_dirty);
                ctx.bus.emit_masters(masters_state(ctx));
                return Ok(json!({ "ok": true, "name": p.name, "value": stored }));
            }
            let leg = parse_leg(p.leg.as_deref())?;
            let masters = match leg {
                LegSel::Live => &ctx.masters,
                LegSel::Design => &ctx.design_masters,
            };
            let stored = masters
                .set(&p.name, p.value)
                .map_err(|e| RpcError::message(format!("{e:#}")))?;
            session::touch(&ctx.session_dirty);
            ctx.bus.emit_masters(masters_state(ctx));
            Ok(json!({ "ok": true, "name": p.name, "value": stored, "leg": leg.as_str() }))
        }

        "master.list" => {
            Ok(serde_json::to_value(masters_state(ctx)).expect("masters"))
        }

        "scene.load" => {
            #[derive(Deserialize)]
            struct Params {
                json: String,
                #[serde(default)]
                base_rev: Option<u64>,
            }
            let p: Params = serde_json::from_value(req.params.clone())
                .map_err(|e| RpcError::message(format!("params: {e}")))?;
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            cmd_tx
                .send(EngineCommand::SceneLoad {
                    json: p.json,
                    base_rev: p.base_rev,
                    actor: *actor,
                    reply: reply_tx,
                })
                .map_err(|_| RpcError::message("engine command channel closed"))?;
            reply_rx
                .recv()
                .map_err(|_| RpcError::message("engine reply channel closed"))?
                .map_err(RpcError::message)
        }

        "scene.reload" => {
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            cmd_tx
                .send(EngineCommand::SceneReload {
                    actor: *actor,
                    reply: reply_tx,
                })
                .map_err(|_| RpcError::message("engine command channel closed"))?;
            reply_rx
                .recv()
                .map_err(|_| RpcError::message("engine reply channel closed"))?
                .map_err(RpcError::message)
        }

        "effect.upsert" => {
            #[derive(Deserialize)]
            struct Params {
                name: String,
                wgsl: String,
                #[serde(default)]
                descriptor: Option<Value>,
            }
            let p: Params = serde_json::from_value(req.params.clone())
                .map_err(|e| RpcError::message(format!("params: {e}")))?;
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            cmd_tx
                .send(EngineCommand::EffectUpsert {
                    name: p.name,
                    wgsl: p.wgsl,
                    descriptor: p.descriptor,
                    actor: *actor,
                    reply: reply_tx,
                })
                .map_err(|_| RpcError::message("engine command channel closed"))?;
            reply_rx
                .recv()
                .map_err(|_| RpcError::message("engine reply channel closed"))?
                .map_err(RpcError::message)
        }

        "effect.describe" => {
            #[derive(Deserialize, Default)]
            #[serde(default)]
            struct Params {
                name: Option<String>,
            }
            let p: Params = if req.params.is_null() {
                Params::default()
            } else {
                serde_json::from_value(req.params.clone())
                    .map_err(|e| RpcError::message(format!("params: {e}")))?
            };
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            cmd_tx
                .send(EngineCommand::EffectDescribe {
                    name: p.name,
                    reply: reply_tx,
                })
                .map_err(|_| RpcError::message("engine command channel closed"))?;
            reply_rx
                .recv()
                .map_err(|_| RpcError::message("engine reply channel closed"))?
                .map_err(RpcError::message)
        }

        "session.save" => {
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            cmd_tx
                .send(EngineCommand::SessionSave { reply: reply_tx })
                .map_err(|_| RpcError::message("engine command channel closed"))?;
            reply_rx
                .recv()
                .map_err(|_| RpcError::message("engine reply channel closed"))?
                .map_err(RpcError::message)
        }

        "effect.remove" => {
            #[derive(Deserialize)]
            struct Params {
                name: String,
            }
            let p: Params = serde_json::from_value(req.params.clone())
                .map_err(|e| RpcError::message(format!("params: {e}")))?;
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            cmd_tx
                .send(EngineCommand::EffectRemove {
                    name: p.name,
                    actor: *actor,
                    reply: reply_tx,
                })
                .map_err(|_| RpcError::message("engine command channel closed"))?;
            reply_rx
                .recv()
                .map_err(|_| RpcError::message("engine reply channel closed"))?
                .map_err(RpcError::message)
        }

        other => Err(RpcError {
            code: -32601,
            message: format!("method not found: {other}"),
            data: None,
        }),
    }
}

/// Process an [`EngineCommand`] on the render thread. Sends the reply via
/// the embedded channel so the WS worker can unblock.
pub fn handle(core: &mut Core, cmd: EngineCommand) {
    match cmd {
        EngineCommand::SceneLoad {
            json,
            base_rev,
            actor,
            reply,
        } => {
            // §5.6 — authoring targets the design leg; the reply may be
            // deferred through a probe session (new pipelines are probed
            // before they may enter design). §5.10: base_rev CAS first.
            core.scene_load_rpc(json, base_rev, actor, reply);
        }
        EngineCommand::SceneReload { actor, reply } => {
            match std::fs::read_to_string(core.scene_path()) {
                Ok(raw) => core.scene_load_rpc(raw, None, actor, reply),
                Err(e) => {
                    let _ = reply.send(Err(format!("read scene file: {e}")));
                }
            }
        }
        EngineCommand::EffectUpsert {
            name,
            wgsl,
            descriptor,
            actor,
            reply,
        } => {
            // §5.10 verdict-in-reply: naga-validate before anything touches
            // disk — a broken shader comes back as prescriptive diagnostics
            // in this reply, never as a watcher-side log line.
            let diags = validate_wgsl(&wgsl);
            if !diags.ok {
                let lines: Vec<String> = diags
                    .diagnostics
                    .iter()
                    .map(|d| format!("line {}:{}: {}", d.line, d.column, d.message))
                    .collect();
                let _ = reply.send(Err(format!(
                    "WGSL rejected for effect {name:?} — {}",
                    lines.join("; ")
                )));
                return;
            }
            match write_effect(core, &name, &wgsl, descriptor.as_ref()) {
                Ok(()) => {
                    // Registry rescan here records the new mtimes, so the
                    // watcher's echo of our own write rescans to an empty
                    // change set (§3.5-style self-write dedupe). The design
                    // re-apply routes the new pipeline through the §2.6
                    // probe and defers `reply` until the verdict is in.
                    core.after_effect_upsert(&name, actor, reply);
                }
                Err(e) => {
                    let _ = reply.send(Err(format!("{e:#}")));
                }
            }
        }
        EngineCommand::EffectRemove { name, actor, reply } => {
            match remove_effect(core, &name) {
                Ok(()) => {
                    core.after_effect_remove(&name, actor, reply);
                }
                Err(e) => {
                    let _ = reply.send(Err(format!("{e:#}")));
                }
            }
        }
        EngineCommand::IdentitySet {
            groups,
            labels,
            actor,
            reply,
        } => {
            let _ = reply.send(core.cmd_identity_set(groups, labels, actor));
        }
        EngineCommand::EffectDescribe { name, reply } => {
            let res = core
                .describe_effects(name.as_deref())
                .map_err(|e| format!("{e:#}"));
            let _ = reply.send(res);
        }
        EngineCommand::SessionSave { reply } => {
            let res = core
                .save_session()
                .map(|path| json!({ "ok": true, "path": path.display().to_string() }))
                .map_err(|e| format!("{e:#}"));
            let _ = reply.send(res);
        }
        EngineCommand::Promote {
            fade_ms,
            quantize,
            reply,
        } => {
            let _ = reply.send(core.cmd_promote(fade_ms, quantize));
        }
        EngineCommand::Pull { actor, reply } => {
            let _ = reply.send(core.cmd_pull(actor));
        }
        EngineCommand::PreviewSource { source, reply } => {
            let _ = reply.send(core.cmd_preview_source(source));
        }
    }
}

fn write_effect(core: &mut Core, name: &str, wgsl: &str, descriptor: Option<&Value>) -> Result<()> {
    let dir = core
        .effects_dir()
        .ok_or_else(|| anyhow!("no effects directory bound — re-run with --effects"))?
        .to_path_buf();
    let effect_dir = dir.join(name);
    std::fs::create_dir_all(&effect_dir)?;
    std::fs::write(effect_dir.join("shader.wgsl"), wgsl)?;
    if let Some(desc) = descriptor {
        std::fs::write(
            effect_dir.join("descriptor.json"),
            serde_json::to_vec_pretty(desc)?,
        )?;
    }
    Ok(())
}

fn remove_effect(core: &mut Core, name: &str) -> Result<()> {
    let dir = core
        .effects_dir()
        .ok_or_else(|| anyhow!("no effects directory bound — re-run with --effects"))?
        .to_path_buf();
    let effect_dir = dir.join(name);
    if effect_dir.exists() {
        std::fs::remove_dir_all(&effect_dir)?;
    }
    Ok(())
}

// ---------- WGSL diagnostics ----------

#[derive(Debug, Clone, serde::Serialize)]
pub struct WgslDiagnostic {
    pub severity: String, // "error" | "warning"
    pub line: u32,
    pub column: u32,
    pub end_line: u32,
    pub end_column: u32,
    pub message: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WgslDiagnostics {
    pub ok: bool,
    pub diagnostics: Vec<WgslDiagnostic>,
}

/// Compose the user's body with the engine prelude + main entry, then run
/// the result through `naga`. Diagnostics are returned in *user-source*
/// line/column space (offset back by the prelude length) so Monaco markers
/// point at the right line in the user's editor.
pub fn validate_wgsl(body: &str) -> WgslDiagnostics {
    let composed = crate::gpu::compose_shader(body);
    match naga::front::wgsl::parse_str(&composed) {
        Ok(_) => WgslDiagnostics {
            ok: true,
            diagnostics: Vec::new(),
        },
        Err(e) => {
            let prelude_lines = count_lines(include_str!("shaders/effect_prelude.wgsl"));
            // naga ParseError exposes `location(source)` returning a SourceLocation
            // with 1-based line/col and an offset. Older crate versions instead
            // expose .labels(); use whatever is available via Display fallback.
            let diag = parse_error_to_diag(&e, body, prelude_lines);
            WgslDiagnostics {
                ok: false,
                diagnostics: vec![diag],
            }
        }
    }
}

fn parse_error_to_diag(
    e: &naga::front::wgsl::ParseError,
    user_body: &str,
    prelude_lines: u32,
) -> WgslDiagnostic {
    let labels: Vec<_> = e.labels().collect();
    if let Some((span, _msg)) = labels.first() {
        let composed = crate::gpu::compose_shader(user_body);
        let loc = span.location(&composed);
        // The +1 accounts for the '\n' we add between prelude and body.
        let line_in_user = loc.line_number.saturating_sub(prelude_lines + 1);
        let col = loc.line_position;
        return WgslDiagnostic {
            severity: "error".into(),
            line: line_in_user.max(1),
            column: col,
            end_line: line_in_user.max(1),
            end_column: col + 1,
            message: e.message().to_string(),
        };
    }
    WgslDiagnostic {
        severity: "error".into(),
        line: 1,
        column: 1,
        end_line: 1,
        end_column: 2,
        message: e.message().to_string(),
    }
}

fn count_lines(s: &str) -> u32 {
    s.bytes().filter(|b| *b == b'\n').count() as u32
}

// ---------- error type ----------

#[derive(Debug, Clone)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    pub data: Option<Value>,
}

impl RpcError {
    pub fn message(msg: impl Into<String>) -> Self {
        Self {
            code: -32000,
            message: msg.into(),
            data: None,
        }
    }

    pub fn to_json(&self, id: &Value) -> Value {
        let mut err = json!({
            "code": self.code,
            "message": self.message,
        });
        if let Some(data) = &self.data {
            err["data"] = data.clone();
        }
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": err,
        })
    }
}

// Small lockless current-value holder so the WS thread can read the latest
// scene JSON without grabbing a mutex shared with the render thread.
pub mod parking_lot_lite {
    use std::sync::{Arc, RwLock};

    #[derive(Default)]
    pub struct SwapValue {
        inner: RwLock<Option<Arc<String>>>,
    }

    impl SwapValue {
        pub fn new() -> Self {
            Self::default()
        }
        pub fn set(&self, v: String) {
            *self.inner.write().expect("swap value lock") = Some(Arc::new(v));
        }
        pub fn get(&self) -> Option<String> {
            self.inner
                .read()
                .expect("swap value lock")
                .as_ref()
                .map(|s| (**s).clone())
        }
    }
}
