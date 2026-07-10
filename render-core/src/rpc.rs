//! JSON-RPC dispatch (§3.11).
//!
//! One logical method set, two consumers: the WS server thread routes
//! requests through `dispatch_inline` for synchronous, side-effect-free
//! methods (`wgsl.validate`, `pack.info`, `scene.getState`), and queues
//! state-mutating methods (`scene.load`, `effect.upsert`, `effect.remove`)
//! as `EngineCommand`s that the render thread drains at frame boundary in
//! `Core::poll_inbound`. Same method names, same params, same error shapes —
//! Phase 7's MCP wrapper proxies the same surface unchanged.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use crossbeam_channel::Sender;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::core::Core;
use crate::drivers::SliderBank;
use crate::pack::LoadedPack;
use crate::telemetry::Bus;

/// A command that mutates engine state. Issued from the WS server thread;
/// processed on the render thread.
pub enum EngineCommand {
    /// Replace the current scene with the given JSON. Doesn't touch disk.
    SceneLoad {
        json: String,
        reply: Sender<Result<Value, String>>,
    },
    /// Reload the scene from `scene_path` (same path the engine was started
    /// with). Useful when the UI saved to disk and wants an explicit
    /// reload trigger without waiting on the file watcher debounce.
    SceneReload {
        reply: Sender<Result<Value, String>>,
    },
    /// Write `wgsl` to the user-effects directory under `name/shader.wgsl`,
    /// optionally write a `descriptor.json`, then re-scan the registry. The
    /// file watcher is *also* watching that path so this is mostly a
    /// convenience for IPC consumers that don't have file access.
    EffectUpsert {
        name: String,
        wgsl: String,
        descriptor: Option<Value>,
        reply: Sender<Result<Value, String>>,
    },
    /// Remove a project-local effect directory and invalidate its pipeline.
    EffectRemove {
        name: String,
        reply: Sender<Result<Value, String>>,
    },
}

/// Bound bag of references used to answer the synchronous (read-only)
/// requests. Held by the WS server alongside the command channel.
#[derive(Clone)]
pub struct RpcContext {
    /// Static pack info captured at engine start. The current model has no
    /// in-flight pack swap (Phase 6+), so a snapshot is enough.
    pub pack: Arc<PackInfo>,
    /// Most recent scene JSON as the engine has it on disk + the resolved
    /// scene path. Updated on every successful reload by the render thread
    /// via [`RpcContext::set_scene`].
    pub scene_state: Arc<parking_lot_lite::SwapValue>,
    pub effects_dir: Option<PathBuf>,
    pub bus: Bus,
    /// Live `ui.slider` values. `param.set` writes here directly (no
    /// render-thread hop) — the render thread reads it on its next tick, so
    /// knob latency is bounded by one frame.
    pub sliders: Arc<SliderBank>,
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
                label: l.label.clone(),
                tags: l.tags.clone(),
                bbox: l.bbox,
                centroid: l.centroid,
                z: l.z,
            })
            .collect();
        let groups = pack
            .manifest
            .groups
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
pub fn dispatch(
    ctx: &RpcContext,
    cmd_tx: &Sender<EngineCommand>,
    req: &JsonRpcRequest,
) -> Result<Value, RpcError> {
    match req.method.as_str() {
        "pack.info" => Ok(serde_json::to_value(ctx.pack.as_ref()).expect("pack info")),

        "scene.getState" => {
            let raw = ctx
                .scene_state
                .get()
                .ok_or_else(|| RpcError::message("no scene loaded"))?;
            Ok(json!({ "json": raw }))
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
        // re-prompting"). Sets a named ui.slider value; every param bound to
        // `{"driver": "ui.slider", "name": ...}` picks it up on the next
        // frame. No scene rebuild, no shader recompile.
        "param.set" => {
            #[derive(Deserialize)]
            struct Params {
                name: String,
                value: f32,
            }
            let p: Params = serde_json::from_value(req.params.clone())
                .map_err(|e| RpcError::message(format!("params: {e}")))?;
            if !p.value.is_finite() {
                return Err(RpcError::message("value must be finite"));
            }
            ctx.sliders.set(&p.name, p.value);
            Ok(json!({ "ok": true, "name": p.name, "value": p.value }))
        }

        "param.list" => {
            let values: serde_json::Map<String, Value> = ctx
                .sliders
                .snapshot()
                .into_iter()
                .map(|(k, v)| (k, json!(v)))
                .collect();
            Ok(json!({ "sliders": values }))
        }

        "scene.load" => {
            #[derive(Deserialize)]
            struct Params {
                json: String,
            }
            let p: Params = serde_json::from_value(req.params.clone())
                .map_err(|e| RpcError::message(format!("params: {e}")))?;
            let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
            cmd_tx
                .send(EngineCommand::SceneLoad {
                    json: p.json,
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
                .send(EngineCommand::SceneReload { reply: reply_tx })
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
                    reply: reply_tx,
                })
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
        EngineCommand::SceneLoad { json, reply } => {
            let res = core
                .apply_scene_json(&json)
                .map(|_| json!({ "ok": true }))
                .map_err(|e| format!("{e:#}"));
            let _ = reply.send(res);
        }
        EngineCommand::SceneReload { reply } => {
            match std::fs::read_to_string(core.scene_path()) {
                Ok(raw) => {
                    let res = core
                        .apply_scene_json(&raw)
                        .map(|_| json!({ "ok": true }))
                        .map_err(|e| format!("{e:#}"));
                    let _ = reply.send(res);
                }
                Err(e) => {
                    let _ = reply.send(Err(format!("read scene file: {e}")));
                }
            }
        }
        EngineCommand::EffectUpsert {
            name,
            wgsl,
            descriptor,
            reply,
        } => match write_effect(core, &name, &wgsl, descriptor.as_ref()) {
            Ok(()) => {
                let _ = reply.send(Ok(json!({ "ok": true, "name": name })));
            }
            Err(e) => {
                let _ = reply.send(Err(format!("{e:#}")));
            }
        },
        EngineCommand::EffectRemove { name, reply } => match remove_effect(core, &name) {
            Ok(()) => {
                let _ = reply.send(Ok(json!({ "ok": true, "name": name })));
            }
            Err(e) => {
                let _ = reply.send(Err(format!("{e:#}")));
            }
        },
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
