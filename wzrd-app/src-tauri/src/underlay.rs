//! Native preview underlay (docs/TODO/native-preview-underlay.md, D1): a
//! plain `NSView` inserted into the main window's view tree directly
//! *below* the `WKWebView`, with a transparent webview on top. The engine's
//! preview swapchain renders into it (wgpu turns it into a layer-hosting
//! view with a `CAMetalLayer`), and every bit of HTML — panels, mask
//! outlines, the warp grid, dropdowns — composites over the image.
//!
//! Why a sibling view and not the old child window: it moves with the
//! layout for free, can't detach or fall behind the window, and inside one
//! window `NSView` hit-testing ignores alpha, so the webview (frontmost
//! sibling, full-size) always gets the clicks.
//!
//! View tree (wry 0.55, non-child webview):
//! `NSWindow.contentView` = `WryWebViewParent` → [`PreviewView`, `WKWebView`].
//!
//! Main thread only: every method here is AppKit. The render thread only
//! ever touches the wgpu surface built on top of the view.

use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::{anyhow, Result};
use objc2::msg_send;
use objc2::runtime::{AnyClass, AnyObject};
use objc2_foundation::{NSPoint, NSRect, NSSize};
use raw_window_handle as rwh;

type Id = *mut AnyObject;

pub struct PreviewView {
    /// Our `NSView`; we keep the +1 from `alloc/init` for the app's life.
    view: Id,
    /// Their shared superview (`WryWebViewParent`).
    parent: Id,
    /// `NSWindow*` of the main window — occlusion + backing scale.
    window: Id,
}

// Raw AppKit pointers; only dereferenced on the main thread (Tauri sync
// commands + the setup hook), except `window`, which the render thread
// reads `occlusionState` from — a property read AppKit allows off-main
// (the engine-window probe has done the same since collapse Step 2).
unsafe impl Send for PreviewView {}
unsafe impl Sync for PreviewView {}

impl PreviewView {
    /// Find the main window's `WKWebView`, make it see-through, paint the
    /// window background `bg` (so nothing changes visually where the page
    /// used to paint its own background), and insert a hidden preview view
    /// beneath the webview.
    pub fn install(ns_window: *mut c_void, bg: (f64, f64, f64)) -> Result<Self> {
        let window: Id = ns_window.cast();
        if window.is_null() {
            return Err(anyhow!("main NSWindow is null"));
        }
        unsafe {
            let content: Id = msg_send![window, contentView];
            let webview = find_webview(content)
                .ok_or_else(|| anyhow!("no WKWebView under the main window's contentView"))?;
            let parent: Id = msg_send![webview, superview];
            if parent.is_null() {
                return Err(anyhow!("WKWebView has no superview"));
            }

            // See-through webview, opaque window. `drawsBackground` is a
            // private KVC key (what wry's `transparent` feature sets on the
            // config); setting it on the instance works at runtime too.
            let no: Id = msg_send![class("NSNumber")?, numberWithBool: false];
            let key: Id = msg_send![
                class("NSString")?,
                stringWithUTF8String: c"drawsBackground".as_ptr()
            ];
            let _: () = msg_send![webview, setValue: no, forKey: key];
            let clear: Id = msg_send![class("NSColor")?, clearColor];
            let sel = objc2::sel!(setUnderPageBackgroundColor:);
            let responds: bool = msg_send![webview, respondsToSelector: sel];
            if responds {
                let _: () = msg_send![webview, setUnderPageBackgroundColor: clear];
            }
            let color: Id = msg_send![
                class("NSColor")?,
                colorWithSRGBRed: bg.0,
                green: bg.1,
                blue: bg.2,
                alpha: 1.0f64
            ];
            let _: () = msg_send![window, setBackgroundColor: color];

            let view: Id = msg_send![class("NSView")?, alloc];
            let zero = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(1.0, 1.0));
            let view: Id = msg_send![view, initWithFrame: zero];
            if view.is_null() {
                return Err(anyhow!("NSView init failed"));
            }
            // Our own CAMetalLayer, installed before wgpu sees the view:
            // wgpu reuses a view's existing CAMetalLayer, and its fallback
            // path messages the view's (nil) layer, which panics under
            // objc's debug checks. Layer-hosting order: setLayer, then
            // setWantsLayer. Resize gravity lets the engine render below
            // the slot's pixel size (capped at composite res) and still
            // fill it; contentsScale is refreshed on every `place`.
            let layer: Id = msg_send![class("CAMetalLayer")?, new];
            let resize: Id = msg_send![
                class("NSString")?,
                stringWithUTF8String: c"resize".as_ptr()
            ];
            let _: () = msg_send![layer, setContentsGravity: resize];
            let _: () = msg_send![layer, setOpaque: true];
            let _: () = msg_send![view, setLayer: layer];
            let _: () = msg_send![view, setWantsLayer: true];
            let _: () = msg_send![view, setHidden: true];
            // NSWindowBelow = -1
            let _: () = msg_send![parent, addSubview: view, positioned: -1isize, relativeTo: webview];

            let flipped: bool = msg_send![parent, isFlipped];
            log::info!(
                "preview underlay installed below the WKWebView (superview flipped: {flipped})"
            );
            Ok(Self {
                view,
                parent,
                window,
            })
        }
    }

    /// The wgpu surface target for this view (wgpu picks up our
    /// `CAMetalLayer`). `create_surface` must run on the main thread.
    pub fn surface_target(&self) -> ViewHandle {
        ViewHandle(NonNull::new(self.view.cast()).expect("view is non-null"))
    }

    pub fn ns_window(&self) -> *mut c_void {
        self.window.cast()
    }

    pub fn backing_scale(&self) -> f64 {
        unsafe { msg_send![self.window, backingScaleFactor] }
    }

    /// Place the view over a rect in CSS px (== points), top-left origin,
    /// relative to the webview's viewport — the coordinates
    /// `getBoundingClientRect` reports. Snapped to device pixels so the
    /// drawable maps 1:1 onto the screen.
    ///
    /// The reference is `NSWindow.contentLayoutRect` (the frame minus the
    /// title bar), not the webview's frame: under the full-size-content-view
    /// style the webview's frame spans the title bar too while the page's
    /// y = 0 sits below it — measuring from the frame lands the image one
    /// title bar too high (the child-window path hit the same trap).
    pub fn place(&self, x: f64, y: f64, w: f64, h: f64) {
        let scale = self.backing_scale().max(1.0);
        let snap = |v: f64| (v * scale).round() / scale;
        unsafe {
            let (x, y, w, h) = (snap(x), snap(y), snap(w), snap(h));
            // Window coords, bottom-left origin → the superview's own space
            // (AppKit handles a flipped superview).
            let layout: NSRect = msg_send![self.window, contentLayoutRect];
            let in_window = NSRect::new(
                NSPoint::new(
                    layout.origin.x + x,
                    layout.origin.y + layout.size.height - y - h,
                ),
                NSSize::new(w, h),
            );
            let nil: Id = std::ptr::null_mut();
            let frame: NSRect = msg_send![self.parent, convertRect: in_window, fromView: nil];
            // No implicit layer animation on moves/resizes: the image must
            // land where the HTML says, not glide there.
            let tx = class("CATransaction").ok();
            if let Some(tx) = tx {
                let _: () = msg_send![tx, begin];
                let _: () = msg_send![tx, setDisableActions: true];
            }
            let _: () = msg_send![self.view, setFrame: frame];
            // Follows the window across displays (Retina ↔ 1x).
            let layer: Id = msg_send![self.view, layer];
            if !layer.is_null() {
                let _: () = msg_send![layer, setContentsScale: scale];
            }
            if let Some(tx) = tx {
                let _: () = msg_send![tx, commit];
            }
        }
    }

    pub fn set_hidden(&self, hidden: bool) {
        unsafe {
            let _: () = msg_send![self.view, setHidden: hidden];
        }
    }
}

/// Depth-first search for the first `WKWebView` (or subclass — wry uses
/// `WryWebView`) under `root`.
unsafe fn find_webview(root: Id) -> Option<Id> {
    if root.is_null() {
        return None;
    }
    let wk = AnyClass::get("WKWebView")?;
    let is_wk: bool = msg_send![root, isKindOfClass: wk];
    if is_wk {
        return Some(root);
    }
    let subviews: Id = msg_send![root, subviews];
    let n: usize = msg_send![subviews, count];
    for i in 0..n {
        let child: Id = msg_send![subviews, objectAtIndex: i];
        if let Some(found) = find_webview(child) {
            return Some(found);
        }
    }
    None
}

fn class(name: &str) -> Result<&'static AnyClass> {
    AnyClass::get(name).ok_or_else(|| anyhow!("Objective-C class {name} not loaded"))
}

/// rwh 0.6 handle for an `NSView` — what `wgpu::Instance::create_surface`
/// takes. Pointer only; the view outlives every engine (it lives as long as
/// the main window).
pub struct ViewHandle(NonNull<c_void>);

unsafe impl Send for ViewHandle {}
unsafe impl Sync for ViewHandle {}

impl rwh::HasWindowHandle for ViewHandle {
    fn window_handle(&self) -> Result<rwh::WindowHandle<'_>, rwh::HandleError> {
        let raw = rwh::RawWindowHandle::AppKit(rwh::AppKitWindowHandle::new(self.0));
        Ok(unsafe { rwh::WindowHandle::borrow_raw(raw) })
    }
}

impl rwh::HasDisplayHandle for ViewHandle {
    fn display_handle(&self) -> Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        let raw = rwh::RawDisplayHandle::AppKit(rwh::AppKitDisplayHandle::new());
        Ok(unsafe { rwh::DisplayHandle::borrow_raw(raw) })
    }
}
