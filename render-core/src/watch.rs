//! File watcher — agent + author hot-reload surface (D13, D15, §6.5–6.6).
//!
//! Watches the scene file's parent directory plus (optionally) a project-
//! local effects directory. Pushes a single "something changed" signal up
//! to the main loop, which decides whether to reload the scene, rescan the
//! effects registry, or both. Editor save patterns (write-then-rename, atomic
//! temp swaps) generate event bursts — we debounce by collapsing all events
//! received before the next frame.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    Scene,
    Effects,
}

pub struct SceneWatcher {
    #[allow(dead_code)]
    watcher: RecommendedWatcher,
    rx: Receiver<notify::Result<Event>>,
    scene_target: PathBuf,
    effects_root: Option<PathBuf>,
    last_dispatched: Option<Instant>,
}

impl SceneWatcher {
    pub fn new(scene_path: &Path, effects_root: Option<&Path>) -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = notify::recommended_watcher(move |res| {
            let _ = tx.send(res);
        })
        .context("creating file watcher")?;

        // Watch the scene's parent dir non-recursively (catches editor-rename
        // save patterns that drop the original inode).
        let parent = scene_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        watcher
            .watch(&parent, RecursiveMode::NonRecursive)
            .with_context(|| format!("watching {}", parent.display()))?;

        // Watch the effects dir recursively if it exists — captures both
        // shader.wgsl edits and descriptor.json edits.
        if let Some(dir) = effects_root {
            if dir.exists() {
                watcher
                    .watch(dir, RecursiveMode::Recursive)
                    .with_context(|| format!("watching {}", dir.display()))?;
            }
        }

        Ok(Self {
            watcher,
            rx,
            scene_target: scene_path.to_path_buf(),
            effects_root: effects_root.map(Path::to_path_buf),
            last_dispatched: None,
        })
    }

    /// Drain pending events. Returns the set of change kinds detected since
    /// last call. Empty Vec = no change. Sub-50ms event bursts collapse.
    pub fn poll(&mut self) -> Vec<ChangeKind> {
        let mut hit_scene = false;
        let mut hit_effects = false;
        loop {
            match self.rx.try_recv() {
                Ok(Ok(event)) => {
                    if !event_kind_relevant(&event.kind) {
                        continue;
                    }
                    for p in &event.paths {
                        if path_matches(p, &self.scene_target) {
                            hit_scene = true;
                        }
                        if let Some(root) = &self.effects_root {
                            if path_inside(p, root) {
                                hit_effects = true;
                            }
                        }
                    }
                }
                Ok(Err(_)) => {}
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        if !hit_scene && !hit_effects {
            return Vec::new();
        }
        let now = Instant::now();
        if let Some(prev) = self.last_dispatched {
            if now.duration_since(prev) < Duration::from_millis(50) {
                return Vec::new();
            }
        }
        self.last_dispatched = Some(now);
        let mut out = Vec::with_capacity(2);
        if hit_scene {
            out.push(ChangeKind::Scene);
        }
        if hit_effects {
            out.push(ChangeKind::Effects);
        }
        out
    }
}

fn event_kind_relevant(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
    )
}

fn path_matches(path: &Path, target: &Path) -> bool {
    if path == target {
        return true;
    }
    match (path.file_name(), target.file_name()) {
        (Some(a), Some(b)) => a == b && path.parent() == target.parent(),
        _ => false,
    }
}

fn path_inside(path: &Path, root: &Path) -> bool {
    let canon_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let canon_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    canon_path.starts_with(canon_root)
}
