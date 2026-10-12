---
name: afk-session
description: Run an unattended WZRD session while Xander is AFK — the screen-bound test runbook (docs/TODO/afk-test-session.md) and/or a perf hunt with parallel audit + implementer subagents. Use when Xander says he's AFK / away / "the screen is yours", or asks for a systematic engine/UI performance sweep.
---

# AFK session — screen tests + perf hunt

You are the **orchestrator**: you own the screen, the GPU lock, merges and the
summary; subagents own reading and implementing. Your context stays the
overview — delegate any file sweep or implementation to a subagent and keep
only its conclusions. Scratchpad = `$S` below (your session scratchpad).

## Steps

1. **Hold the screen for the whole session.** First command:
   `caffeinate -dimsu -t 10800 &` (one background process, not per-command
   wrapping — on 2026-10-12 the screen locked in a gap between commands and
   every later recording captured the lock screen; nothing unlocks it
   unattended). Check: `pmset -g batt` (note battery vs AC in the results),
   `lsof -i :1420 -i :9123 -i :9124 -i :9177`, and kill **orphans** —
   `pgrep -fl "render-core|wzrd-app"`; a hung `--bench` ignores SIGTERM
   (`sample <pid> 1` it first: the stack is a finding), then `kill -9`.
   `mkdir $S/gpu.lock` — you hold the GPU until the screen blocks are done.
   Done when: caffeinate runs, ports free, no orphans, lock held.

2. **Fan out read-only auditors** (background, in parallel) while you start
   the screen blocks: one each for the engine frame loop (`core.rs`,
   `compositor.rs`, `gpu.rs`, `telemetry.rs`), video decode (`video.rs`),
   WGSL (`render-core/src/shaders`, `library/effects`), and the app
   (`wzrd-app/src`, `src-tauri`). Brief: read-only, no builds, no engine;
   findings ranked impact × confidence with file:line, quoted code, a
   concrete fix, risk, and "already good — don't regress". Done when: each
   report is logged in `$S/findings.md`.

3. **Run the runbook blocks** in `docs/TODO/afk-test-session.md`, in order.
   Pre-build first (`cargo build --release` in render-core; a `cargo build`
   of `wzrd-app/src-tauri` so `pnpm tauri dev` starts in seconds). Look at
   every screenshot and recording frame you produce (a wrong window or the
   lock screen looks like a pass in the numbers). A block failing → record
   it, move on. Done when: every block has a Results row with numbers.

4. **Dispatch implementers** in git worktrees (`isolation: "worktree"`), one
   per file-disjoint package, with [implementer-brief.md](implementer-brief.md)
   pasted in. They may edit from the start; `touch $S/build_ok` when the
   timing-sensitive screen blocks are done, `rmdir $S/gpu.lock` after. Route
   bugs found on screen to the implementer that owns those files
   (SendMessage). When main moves, tell the still-running ones to
   `git merge main` before committing. Done when: every package reported.

5. **Merge and judge.** Merge each branch as it reports (`git merge --no-edit`),
   build, `cargo test --release` (under the lock), `cargo check` the app,
   `tsc --noEmit`, `fallow audit --changed-since <start sha>` (separate
   inherited findings from new ones). Judge with the interleaved A/B in
   [measurement.md](measurement.md), not a single gate run; a regression gets
   a bisect across intermediate builds before anyone "fixes" it. Run the
   full `perf_gate.py` and `--update-baseline --reason` only on a quiet
   machine. Done when: every merged change has a before/after median.

6. **Close out.** Fill the runbook Results and follow each block's "After the
   run" (retire passed blocks; leave only failing steps; append a block for
   anything that still needs eyes on screen — today's work that landed after
   the screen became unavailable included). Doc pass (`doc-pass` skill).
   Restore every fixture you touched (`projects/fixture-a/{session.json,
   alignment.json}`). Commit by path. Then the summary for Xander: what
   passed, what failed (with the number), what was fixed (before → after
   table), what's left and why.

## Reference

- [measurement.md](measurement.md) — A/B method, which bench fields matter,
  on-screen recipes (app CPU, spikes, full-screen recording + crop), driving
  the app with synthetic input, and their traps.
- [implementer-brief.md](implementer-brief.md) — the build gate / GPU lock /
  worktree protocol every implementer follows.
- `scripts/ab.py`, `scripts/appcpu.sh`, `scripts/spikes.py` — the helpers.
- What the 2026-10-12 hunt already fixed (don't re-audit these): region
  scissor per layer pass, per-plan uniform buffer, design leg sampled for a
  JPEG-only consumer, parallel clip opens + one decode per clip frame,
  batched demand-gated app telemetry, sampled cue monitor, opacity-only UI
  animations, bench teardown that never hangs, legs ticking after `acquire`
  (knob→submit ≈ 1 frame), isolated GPU timing on timed window frames. Current state lives in
  `docs/reference/render-engine.md` and `operator-ui.md`.
