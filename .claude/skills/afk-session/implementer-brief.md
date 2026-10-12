# Implementer brief (paste into every implementer subagent's prompt)

Replace `$S` with the session scratchpad (absolute path) before sending.

---

You implement one work package for WZRD (read the repo `CLAUDE.md`; engine
contracts in `docs/reference/render-engine.md`, UI in
`docs/reference/operator-ui.md`). You run in your **own git worktree**; the
orchestrator merges your branch. Other implementers work in parallel on other
packages — stay inside your package's files and keep edits local to the
functions you need, so merges stay clean.

**Machine-sharing protocol** — the screen and GPU may be running
timing-sensitive tests:

1. **Build gate:** run no cargo / pnpm / tsc until `$S/build_ok` exists;
   read and edit meanwhile (`until [ -f $S/build_ok ]; do sleep 10; done`).
2. **Warm cache:** `cp -Rc <main repo>/render-core/target ./render-core/target`
   (APFS clone, instant; same for `wzrd-app/src-tauri/target` and
   `wzrd-app/node_modules`). Builds run under `nice -n 10`.
3. **GPU lock:** anything that renders (`--bench`, `perf_gate.py`, the
   engine, the app, render-core's `cargo test`) runs inside
   `until mkdir $S/gpu.lock 2>/dev/null; do sleep 5; done` … `rmdir $S/gpu.lock`,
   released even on failure, held ≤ 4 min at a time.
4. Windowless only: no windows, screencapture, osascript, or ports
   1420/9123/9124/9125/9177 — the bench (`--bench`) is your engine.

**Engineering rules:** keep render-core Linux/NVIDIA-portable; never block the
render thread; no per-frame allocation in paths you touch. Build the
unmodified binary first and copy it to `$S/<package>-before`, then A/B with
`.claude/skills/afk-session/scripts/ab.py` (interleaved, medians). Add a bench
scenario when no scenario covers your cost path. Leave
`perf_gate.py --update-baseline` to the orchestrator. Update the
`docs/reference/` section whose contract or invariant you changed.

**Before the final commit:** `git merge main` into your branch (same repo),
resolve conflicts keeping both intents, rebuild, re-run `cargo test --release`
and your A/B.

**Report** (≤ 600 words): branch, worktree, sha, what changed (file:line),
before/after numbers, tests run, skipped items with the reason, risks.
Commit message `perf: <package> — <summary>` ending with the repo's
co-author line; stage files by path.
