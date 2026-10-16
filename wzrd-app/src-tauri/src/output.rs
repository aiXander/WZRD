//! The projector output window as a manual switch (Resolume's Output menu).
//!
//! The engine output window is created once, **hidden**, and stays that way
//! until the operator picks a target from the TopBar OUTPUT pill: `off`,
//! `window` (a decorated window — testing without a projector) or a
//! `display` (borderless, covering that display, above its menu bar). A
//! hidden window reads as occluded, so the render thread already renders
//! offscreen and presents nothing (§3.1) — "off" costs no GPU present and
//! "on" is instant, no engine restart.
//!
//! Policy: the target is **never** persisted — output starts off at every
//! launch. What is per-machine (`settings.json` → `output`): the last chosen
//! display, by *name* (indices shuffle on replug), plus `hide_cursor` and
//! `keep_awake`. Flips + fit describe the physical install and live
//! engine-side in `alignment.json` (the webview writes them via
//! `alignment.set`), not here.
//!
//! Env (scripted / AFK launches): `WZRD_OUTPUT=window` or
//! `WZRD_OUTPUT=display:<idx>` turns the output on at boot;
//! `WZRD_DISPLAY=<idx>` is the older spelling of the latter.
//!
//! A watcher thread re-lists displays every 2 s: a vanished target display
//! turns the output off (it never jumps onto the operator's screen) and the
//! webview gets `output:state` with `lost` set.

use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager};

use crate::projects;
use crate::AppState;

/// One attached display. Geometry is in logical points (global, top-left
/// origin — what window placement takes); `width`/`height` are the current
/// mode's pixels (what the engine's swapchain gets).
#[derive(Debug, Clone, Serialize)]
pub struct DisplayInfo {
    pub index: usize,
    /// macOS's own name ("DELL U2720Q", "EPSON PJ") — the persisted key.
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub x: f64,
    pub y: f64,
    pub points_w: f64,
    pub points_h: f64,
    pub scale: f64,
    pub refresh_hz: Option<f64>,
    /// The display's native mode, when it reports one. Differs from
    /// `width`×`height` → macOS is scaling the signal.
    pub native: Option<[u32; 2]>,
    pub builtin: bool,
    /// Holds the operator UI right now.
    pub operator: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Target {
    Off,
    Window,
    Display,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutputState {
    pub target: Target,
    /// The display the output is on (`target == display`).
    pub display: Option<String>,
    /// Last chosen display (persisted) — the popover's default.
    pub last_display: Option<String>,
    pub hide_cursor: bool,
    pub keep_awake: bool,
    /// Set when a display vanished under a live output; cleared on the
    /// next explicit target change.
    pub lost: Option<String>,
}

pub struct Output {
    state: Mutex<OutputState>,
    #[cfg(target_os = "macos")]
    awake: Mutex<Option<mac::Activity>>,
}

impl Output {
    pub fn new(app: &AppHandle) -> Self {
        let saved = projects::read_settings(app).get("output").cloned().unwrap_or(json!({}));
        let flag = |k: &str| saved.get(k).and_then(Value::as_bool).unwrap_or(true);
        Self {
            state: Mutex::new(OutputState {
                target: Target::Off,
                display: None,
                last_display: saved.get("display").and_then(Value::as_str).map(str::to_string),
                hide_cursor: flag("hide_cursor"),
                keep_awake: flag("keep_awake"),
                lost: None,
            }),
            #[cfg(target_os = "macos")]
            awake: Mutex::new(None),
        }
    }

    pub fn snapshot(&self) -> OutputState {
        self.state.lock().expect("output lock").clone()
    }
}

/// Every attached display, in system order.
pub fn displays(app: &AppHandle) -> Vec<DisplayInfo> {
    let operator = operator_center(app);
    #[cfg(target_os = "macos")]
    let mut list = mac::displays();
    #[cfg(not(target_os = "macos"))]
    let mut list = generic_displays(app);
    for d in &mut list {
        d.operator = operator.is_some_and(|(cx, cy)| {
            cx >= d.x && cx < d.x + d.points_w && cy >= d.y && cy < d.y + d.points_h
        });
    }
    list
}

/// Centre of the operator window in global points.
fn operator_center(app: &AppHandle) -> Option<(f64, f64)> {
    let main = app.get_webview_window("main")?;
    let k = main.scale_factor().ok()?;
    let p = main.outer_position().ok()?.to_logical::<f64>(k);
    let s = main.outer_size().ok()?.to_logical::<f64>(k);
    Some((p.x + s.width / 2.0, p.y + s.height / 2.0))
}

#[cfg(not(target_os = "macos"))]
fn generic_displays(app: &AppHandle) -> Vec<DisplayInfo> {
    // Untested sketch (wzrd-app may be Mac-native; CLAUDE.md).
    app.available_monitors()
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let k = m.scale_factor();
            let p = m.position().to_logical::<f64>(k);
            let s = m.size().to_logical::<f64>(k);
            DisplayInfo {
                index: i,
                name: m.name().cloned().unwrap_or_else(|| format!("Display {}", i + 1)),
                width: m.size().width,
                height: m.size().height,
                x: p.x,
                y: p.y,
                points_w: s.width,
                points_h: s.height,
                scale: k,
                refresh_hz: None,
                native: None,
                builtin: false,
                operator: false,
            }
        })
        .collect()
}

/// Point the output somewhere. Main thread only (window placement); sync
/// Tauri commands and the devport dispatcher qualify.
pub fn set_target(app: &AppHandle, target: &str, display: Option<&str>) -> Result<OutputState, String> {
    let out = app.state::<Output>();
    let windows = app.state::<AppState>().windows.clone();
    let window = windows.output_window();
    match target {
        "off" => {
            window.hide().map_err(|e| e.to_string())?;
            let mut s = out.state.lock().expect("output lock");
            s.target = Target::Off;
            s.display = None;
            s.lost = None;
        }
        "window" => {
            let (cw, ch) = windows.canvas_size();
            let near = displays(app).into_iter().find(|d| d.operator);
            place_window(window, cw, ch, near.as_ref())?;
            let mut s = out.state.lock().expect("output lock");
            s.target = Target::Window;
            s.display = None;
            s.lost = None;
        }
        "display" => {
            let name = display.ok_or("output_set: target \"display\" needs a display name")?;
            let d = displays(app)
                .into_iter()
                .find(|d| d.name == name)
                .ok_or_else(|| format!("no display named {name:?} is attached"))?;
            place_fullscreen(window, &d)?;
            let mut s = out.state.lock().expect("output lock");
            s.target = Target::Display;
            s.display = Some(d.name.clone());
            s.last_display = Some(d.name.clone());
            s.lost = None;
        }
        other => return Err(format!("output_set: unknown target {other:?} (off | window | display)")),
    }
    apply_side_effects(app);
    persist(app);
    Ok(emit(app))
}

/// `hide_cursor` / `keep_awake` toggles — applied immediately if live.
pub fn set_options(app: &AppHandle, hide_cursor: Option<bool>, keep_awake: Option<bool>) -> OutputState {
    {
        let out = app.state::<Output>();
        let mut s = out.state.lock().expect("output lock");
        if let Some(v) = hide_cursor {
            s.hide_cursor = v;
        }
        if let Some(v) = keep_awake {
            s.keep_awake = v;
        }
    }
    apply_side_effects(app);
    persist(app);
    emit(app)
}

/// Cursor + display-sleep follow the target and the two options.
fn apply_side_effects(app: &AppHandle) {
    let out = app.state::<Output>();
    let s = out.snapshot();
    let live = s.target != Target::Off;
    let window = app.state::<AppState>().windows.output_window().clone();
    // Only a display target hides the cursor: in window mode you click it.
    let _ = window.set_cursor_visible(!(s.target == Target::Display && s.hide_cursor));
    #[cfg(target_os = "macos")]
    {
        let mut awake = out.awake.lock().expect("awake lock");
        match (live && s.keep_awake, awake.is_some()) {
            (true, false) => *awake = Some(mac::Activity::begin()),
            (false, true) => *awake = None,
            _ => {}
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = live;
}

fn persist(app: &AppHandle) {
    let s = app.state::<Output>().snapshot();
    let mut doc = projects::read_settings(app);
    doc["output"] = json!({
        "display": s.last_display,
        "hide_cursor": s.hide_cursor,
        "keep_awake": s.keep_awake,
    });
    if let Err(e) = projects::write_settings(app, &doc) {
        log::warn!("could not save output settings: {e}");
    }
}

fn emit(app: &AppHandle) -> OutputState {
    let s = app.state::<Output>().snapshot();
    let _ = app.emit("output:state", &s);
    s
}

/// The output window's close button (window mode) means "output off", not
/// "quit the app" — called from the engine window's event handler.
pub fn on_close_requested(app: &AppHandle) {
    let app2 = app.clone();
    let _ = app.run_on_main_thread(move || {
        if let Err(e) = set_target(&app2, "off", None) {
            log::warn!("output off on close: {e}");
        }
    });
}

/// Decorated window at the canvas aspect, at most 1280×800 points, centred
/// on `near` (the operator's display).
fn place_window(window: &tauri::Window, cw: u32, ch: u32, near: Option<&DisplayInfo>) -> Result<(), String> {
    let k = (1280.0 / cw.max(1) as f64).min(800.0 / ch.max(1) as f64).min(1.0);
    let (w, h) = ((cw as f64 * k).max(320.0), (ch as f64 * k).max(180.0));
    #[cfg(target_os = "macos")]
    mac::set_fullscreen_level(window, false);
    let e = |e: tauri::Error| e.to_string();
    window.set_decorations(true).map_err(e)?;
    window.set_size(tauri::LogicalSize::new(w, h)).map_err(e)?;
    if let Some(d) = near {
        let x = d.x + (d.points_w - w) / 2.0;
        let y = d.y + (d.points_h - h) / 2.0;
        window.set_position(tauri::LogicalPosition::new(x, y)).map_err(e)?;
    }
    show_without_focus(window)
}

/// Borderless, exactly covering `d`, above its menu bar and Dock.
fn place_fullscreen(window: &tauri::Window, d: &DisplayInfo) -> Result<(), String> {
    let e = |e: tauri::Error| e.to_string();
    window.set_decorations(false).map_err(e)?;
    window.set_size(tauri::LogicalSize::new(d.points_w, d.points_h)).map_err(e)?;
    window.set_position(tauri::LogicalPosition::new(d.x, d.y)).map_err(e)?;
    #[cfg(target_os = "macos")]
    mac::set_fullscreen_level(window, true);
    show_without_focus(window)
}

/// Bring the output up without taking key focus from the operator window —
/// otherwise every UI shortcut is dead until you click back.
fn show_without_focus(window: &tauri::Window) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        mac::order_front(window);
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    window.show().map_err(|e| e.to_string())
}

/// Boot-time env: `WZRD_OUTPUT=window|display:<idx>`, or `WZRD_DISPLAY=<idx>`.
pub fn apply_env(app: &AppHandle) {
    let spec = std::env::var("WZRD_OUTPUT")
        .ok()
        .or_else(|| std::env::var("WZRD_DISPLAY").ok().map(|i| format!("display:{i}")));
    let Some(spec) = spec else { return };
    let result = match spec.split_once(':') {
        Some(("display", idx)) => match idx.parse::<usize>().ok().and_then(|i| displays(app).into_iter().nth(i)) {
            Some(d) => set_target(app, "display", Some(&d.name)),
            None => Err(format!("{spec}: no display at that index")),
        },
        _ if spec == "window" => set_target(app, "window", None),
        _ => Err(format!("unrecognised output spec {spec:?} (window | display:<idx>)")),
    };
    match result {
        Ok(_) => log::info!("output on at boot ({spec})"),
        Err(e) => log::warn!("WZRD_OUTPUT/WZRD_DISPLAY: {e}"),
    }
}

/// Re-list displays every 2 s: tell the webview when the set changes, and
/// turn a live output off when its display vanishes.
pub fn spawn_watcher(app: AppHandle) {
    thread::Builder::new()
        .name("output-displays".into())
        .spawn(move || {
            let key = |l: &[DisplayInfo]| {
                l.iter()
                    .map(|d| format!("{}@{},{} {}x{}", d.name, d.x, d.y, d.width, d.height))
                    .collect::<Vec<_>>()
            };
            let mut last = Vec::new();
            loop {
                thread::sleep(Duration::from_secs(2));
                let (tx, rx) = crossbeam_channel::bounded(1);
                let a = app.clone();
                if app.run_on_main_thread(move || { let _ = tx.send(displays(&a)); }).is_err() {
                    break;
                }
                let Ok(list) = rx.recv_timeout(Duration::from_secs(5)) else { continue };
                let k = key(&list);
                if k == last {
                    continue;
                }
                last = k;
                let _ = app.emit("output:displays", &list);
                let s = app.state::<Output>().snapshot();
                let gone = s.display.filter(|n| !list.iter().any(|d| &d.name == n));
                if let Some(name) = gone {
                    log::warn!("output display {name:?} disconnected — output off");
                    let a = app.clone();
                    let _ = app.run_on_main_thread(move || {
                        let _ = set_target(&a, "off", None);
                        a.state::<Output>().state.lock().expect("output lock").lost = Some(name);
                        emit(&a);
                    });
                }
            }
        })
        .expect("spawning output display watcher");
}

/// Flash each display's number + name on it for ~2.5 s, so "Display 2" maps
/// to a physical screen before you send light to it.
pub async fn identify(app: AppHandle) -> Result<(), String> {
    let (tx, rx) = crossbeam_channel::bounded(1);
    let a = app.clone();
    app.run_on_main_thread(move || { let _ = tx.send(displays(&a)); })
        .map_err(|e| e.to_string())?;
    let list = rx.recv_timeout(Duration::from_secs(5)).map_err(|e| e.to_string())?;
    let mut opened = Vec::new();
    for d in list {
        let (w, h) = (420.0, 240.0);
        let html = format!(
            "<html><body style=\"margin:0;height:100vh;display:flex;flex-direction:column;\
             align-items:center;justify-content:center;background:#0b0c10;color:#f2f2f2;\
             font:600 20px -apple-system,system-ui;border:3px solid #f5a524;box-sizing:border-box\">\
             <div style=\"font-size:120px;line-height:1\">{}</div><div>{}</div>\
             <div style=\"opacity:.6;font-size:14px;margin-top:6px\">{}×{}</div></body></html>",
            d.index + 1,
            html_escape(&d.name),
            d.width,
            d.height
        );
        let url = format!("data:text/html;charset=utf-8,{}", percent_encode(&html));
        let parsed = url.parse().map_err(|e| format!("{e}"))?;
        let label = format!("identify-{}", d.index);
        let (x, y) = (d.x + (d.points_w - w) / 2.0, d.y + (d.points_h - h) / 2.0);
        // Reused, never destroyed: a destroyed webview window lingers in the
        // window server on macOS, so one per click would leak.
        if let Some(win) = app.get_webview_window(&label) {
            let _ = win.navigate(parsed);
            let _ = win.set_position(tauri::LogicalPosition::new(x, y));
            let _ = win.show();
            opened.push(win);
            continue;
        }
        match tauri::WebviewWindowBuilder::new(&app, &label, tauri::WebviewUrl::External(parsed))
            .title("Identify")
            .decorations(false)
            .resizable(false)
            .always_on_top(true)
            .focused(false)
            .focusable(false)
            .skip_taskbar(true)
            .inner_size(w, h)
            .position(x, y)
            .build()
        {
            Ok(win) => opened.push(win),
            Err(e) => log::warn!("identify on {}: {e}", d.name),
        }
    }
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(2500));
        for w in opened {
            let _ = w.hide();
        }
    });
    Ok(())
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

// ---------- commands ----------

#[tauri::command]
pub fn output_displays(app: AppHandle) -> Vec<DisplayInfo> {
    displays(&app)
}

#[tauri::command]
pub fn output_state(app: AppHandle) -> OutputState {
    app.state::<Output>().snapshot()
}

#[tauri::command]
pub fn output_set(app: AppHandle, target: String, display: Option<String>) -> Result<OutputState, String> {
    set_target(&app, &target, display.as_deref())
}

#[tauri::command]
pub fn output_options(app: AppHandle, hide_cursor: Option<bool>, keep_awake: Option<bool>) -> OutputState {
    set_options(&app, hide_cursor, keep_awake)
}

#[tauri::command]
pub async fn output_identify(app: AppHandle) -> Result<(), String> {
    identify(app).await
}

#[cfg(target_os = "macos")]
mod mac {
    //! CoreGraphics display modes + AppKit names/levels. Mac-only by design
    //! (wzrd-app is exempt from the portability rule).

    use std::ffi::c_void;

    use objc2::rc::Retained;
    use objc2::runtime::AnyObject;
    use objc2::{class, msg_send};
    use objc2_foundation::{NSActivityOptions, NSObject, NSProcessInfo, NSString};

    use super::DisplayInfo;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CGPoint {
        x: f64,
        y: f64,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CGSize {
        width: f64,
        height: f64,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CGRect {
        origin: CGPoint,
        size: CGSize,
    }

    type Mode = *const c_void;

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGGetActiveDisplayList(max: u32, displays: *mut u32, count: *mut u32) -> i32;
        fn CGDisplayBounds(display: u32) -> CGRect;
        fn CGDisplayIsBuiltin(display: u32) -> u32;
        fn CGDisplayCopyDisplayMode(display: u32) -> Mode;
        fn CGDisplayCopyAllDisplayModes(display: u32, options: *const c_void) -> *const c_void;
        fn CGDisplayModeGetPixelWidth(mode: Mode) -> usize;
        fn CGDisplayModeGetPixelHeight(mode: Mode) -> usize;
        fn CGDisplayModeGetRefreshRate(mode: Mode) -> f64;
        fn CGDisplayModeGetIOFlags(mode: Mode) -> u32;
        fn CGDisplayModeRelease(mode: Mode);
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFArrayGetCount(array: *const c_void) -> isize;
        fn CFArrayGetValueAtIndex(array: *const c_void, idx: isize) -> *const c_void;
        fn CFRelease(cf: *const c_void);
    }

    /// IOKit `kDisplayModeNativeFlag`.
    const NATIVE_FLAG: u32 = 0x0200_0000;
    /// `NSMainMenuWindowLevel + 1` — above the menu bar and the Dock.
    const ABOVE_MENU_BAR: isize = 25;
    const CAN_JOIN_ALL_SPACES: usize = 1 << 0;
    const STATIONARY: usize = 1 << 4;

    pub fn displays() -> Vec<DisplayInfo> {
        let mut ids = [0u32; 16];
        let mut n = 0u32;
        if unsafe { CGGetActiveDisplayList(16, ids.as_mut_ptr(), &mut n) } != 0 {
            return Vec::new();
        }
        let names = screen_names();
        let mut seen: Vec<String> = Vec::new();
        ids[..n as usize]
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let b = unsafe { CGDisplayBounds(id) };
                let (width, height, refresh) = unsafe {
                    let m = CGDisplayCopyDisplayMode(id);
                    let r = if m.is_null() {
                        (b.size.width as u32, b.size.height as u32, None)
                    } else {
                        let hz = CGDisplayModeGetRefreshRate(m);
                        let r = (
                            CGDisplayModeGetPixelWidth(m) as u32,
                            CGDisplayModeGetPixelHeight(m) as u32,
                            (hz > 0.0).then_some(hz),
                        );
                        CGDisplayModeRelease(m);
                        r
                    };
                    r
                };
                let base = names
                    .iter()
                    .find(|(sid, _)| *sid == id)
                    .map(|(_, n)| n.clone())
                    .unwrap_or_else(|| format!("Display {}", i + 1));
                // Two identical projectors: keep names unique (they're the key).
                let mut name = base.clone();
                let mut k = 2;
                while seen.contains(&name) {
                    name = format!("{base} ({k})");
                    k += 1;
                }
                seen.push(name.clone());
                DisplayInfo {
                    index: i,
                    name,
                    width,
                    height,
                    x: b.origin.x,
                    y: b.origin.y,
                    points_w: b.size.width,
                    points_h: b.size.height,
                    scale: if b.size.width > 0.0 { width as f64 / b.size.width } else { 1.0 },
                    refresh_hz: refresh,
                    native: native_mode(id),
                    builtin: unsafe { CGDisplayIsBuiltin(id) } != 0,
                    operator: false,
                }
            })
            .collect()
    }

    fn native_mode(id: u32) -> Option<[u32; 2]> {
        unsafe {
            let modes = CGDisplayCopyAllDisplayModes(id, std::ptr::null());
            if modes.is_null() {
                return None;
            }
            let mut found = None;
            for i in 0..CFArrayGetCount(modes) {
                let m = CFArrayGetValueAtIndex(modes, i);
                if CGDisplayModeGetIOFlags(m) & NATIVE_FLAG != 0 {
                    let wh = [CGDisplayModeGetPixelWidth(m) as u32, CGDisplayModeGetPixelHeight(m) as u32];
                    // Several refresh rates share the flag; keep the largest.
                    if found.is_none_or(|f: [u32; 2]| wh[0] * wh[1] > f[0] * f[1]) {
                        found = Some(wh);
                    }
                }
            }
            CFRelease(modes);
            found
        }
    }

    /// `(CGDirectDisplayID, NSScreen.localizedName)` for every screen.
    fn screen_names() -> Vec<(u32, String)> {
        unsafe {
            let screens: *mut AnyObject = msg_send![class!(NSScreen), screens];
            if screens.is_null() {
                return Vec::new();
            }
            let count: usize = msg_send![screens, count];
            let key = NSString::from_str("NSScreenNumber");
            (0..count)
                .filter_map(|i| {
                    let screen: *mut AnyObject = msg_send![screens, objectAtIndex: i];
                    let desc: *mut AnyObject = msg_send![screen, deviceDescription];
                    let num: *mut AnyObject = msg_send![desc, objectForKey: &*key];
                    if num.is_null() {
                        return None;
                    }
                    let id: u32 = msg_send![num, unsignedIntValue];
                    let name: *mut NSString = msg_send![screen, localizedName];
                    let name = if name.is_null() { return None } else { (*name).to_string() };
                    Some((id, name))
                })
                .collect()
        }
    }

    pub fn set_fullscreen_level(window: &tauri::Window, on: bool) {
        let Ok(ns) = window.ns_window() else { return };
        let ns = ns.cast::<AnyObject>();
        unsafe {
            let level: isize = if on { ABOVE_MENU_BAR } else { 0 };
            let behavior: usize = if on { CAN_JOIN_ALL_SPACES | STATIONARY } else { 0 };
            let _: () = msg_send![ns, setLevel: level];
            let _: () = msg_send![ns, setCollectionBehavior: behavior];
        }
    }

    /// `orderFrontRegardless` — visible, but key focus stays where it is.
    pub fn order_front(window: &tauri::Window) {
        let Ok(ns) = window.ns_window() else { return };
        let ns = ns.cast::<AnyObject>();
        unsafe {
            let _: () = msg_send![ns, orderFrontRegardless];
        }
    }

    /// Display-sleep assertion, held while a live output wants it.
    pub struct Activity(Retained<NSObject>);

    // The token is only created/ended on the main thread; the Mutex that
    // holds it lives in managed (Send + Sync) state.
    unsafe impl Send for Activity {}

    impl Activity {
        pub fn begin() -> Self {
            let reason = NSString::from_str("projector output live");
            let options = NSActivityOptions::NSActivityIdleDisplaySleepDisabled
                | NSActivityOptions::NSActivityIdleSystemSleepDisabled;
            let token = unsafe { NSProcessInfo::processInfo().beginActivityWithOptions_reason(options, &reason) };
            log::info!("display sleep disabled while output is live");
            Self(token)
        }
    }

    impl Drop for Activity {
        fn drop(&mut self) {
            unsafe { NSProcessInfo::processInfo().endActivity(&self.0) };
            log::info!("display sleep allowed again");
        }
    }
}
