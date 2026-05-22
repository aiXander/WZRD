//! `scene.json` file watcher — the agent's primary hot-reload surface
//! (D13 + §6.5).
//!
//! Pushes "scene changed" events through a channel; the main loop drains the
//! channel each frame and rebuilds the pass plan if anything fired. Editor
//! save patterns (write-then-rename, atomic temp swaps) generate bursts —
//! we debounce by collapsing all events received before the next frame.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

pub struct SceneWatcher {
    #[allow(dead_code)]
    watcher: RecommendedWatcher,
    rx: Receiver<notify::Result<Event>>,
    target: PathBuf,
    /// Last time we surfaced a reload event upstream. Sub-50ms bursts are
    /// collapsed because most editors fire multiple OS events per save.
    last_dispatched: Option<Instant>,
}

impl SceneWatcher {
    pub fn new(scene_path: &Path) -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = notify::recommended_watcher(move |res| {
            let _ = tx.send(res);
        })
        .context("creating file watcher")?;

        // Watching the parent directory is more robust than watching the
        // single file — editors like `vim` and many save-via-rename flows
        // create a new inode that a watch on the original file misses.
        let parent = scene_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        watcher
            .watch(&parent, RecursiveMode::NonRecursive)
            .with_context(|| format!("watching {}", parent.display()))?;

        Ok(Self {
            watcher,
            rx,
            target: scene_path.to_path_buf(),
            last_dispatched: None,
        })
    }

    /// Drain pending events. Returns `true` if `scene.json` changed since
    /// the last call (debounced).
    pub fn poll(&mut self) -> bool {
        let mut hit = false;
        loop {
            match self.rx.try_recv() {
                Ok(Ok(event)) => {
                    if event_touches(&event, &self.target) {
                        hit = true;
                    }
                }
                Ok(Err(_)) => {}
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        if !hit {
            return false;
        }
        let now = Instant::now();
        if let Some(prev) = self.last_dispatched {
            if now.duration_since(prev) < Duration::from_millis(50) {
                return false;
            }
        }
        self.last_dispatched = Some(now);
        true
    }
}

fn event_touches(event: &Event, target: &Path) -> bool {
    let kind_relevant = matches!(
        event.kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
    );
    if !kind_relevant {
        return false;
    }
    let target_name = target.file_name();
    event.paths.iter().any(|p| {
        p == target
            || (target_name.is_some() && p.file_name() == target_name)
    })
}
