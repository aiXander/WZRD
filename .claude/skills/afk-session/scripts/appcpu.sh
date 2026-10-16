#!/usr/bin/env bash
# Per-process CPU of the running app in one posture, averaged over ~30 s.
#   appcpu.sh <label> <route: prepare|perform|align> <leg: design|live> [scene]
# Launches `pnpm tauri dev` on the scene (default fixture-a/perf-clips), sets
# the room + deck leg through the debug devport, brings the app to front,
# samples `top` for wzrd-app, every WebKit process and WindowServer, then
# quits and restores fixture-a/session.json. Needs the screen (window in front).
set -u
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$REPO"
SCENE=${4:-projects/fixture-a/perf-clips.scene.json}
B=test_results/appcpu; mkdir -p $B; LOG=$B/app_$1.log
cp projects/fixture-a/session.json $B/session.bak
(cd wzrd-app && WZRD_OUTPUT=window WZRD_SCENE=../$SCENE pnpm tauri dev > ../$LOG 2>&1 &)
n=0; until grep -q "preview underlay installed" $LOG; do sleep 1; n=$((n+1)); [ $n -gt 400 ] && echo TIMEOUT && exit 1; done; sleep 5
PID=$(pgrep -f target/debug/wzrd-app | head -1)
# setRoute, not localStorage + reload: a stored `align` route boots into Perform.
curl -s localhost:9124/cmd/webview_eval -d "{\"js\":\"import('/src/state/store.ts').then(m=>m.useStore.getState().setRoute('$2'))\"}" >/dev/null; sleep 4
curl -s localhost:9124/cmd/webview_eval -d "{\"js\":\"(async()=>{const a=await import('/src/state/actions.ts');a.setDeck('$3')})()\"}" >/dev/null
osascript -e 'tell application "System Events" to set frontmost of (first process whose name is "wzrd-app") to true'; sleep 8
WK=$(ps -axo pid,command | grep -E "WebKit\.(WebContent|GPU)" | grep -v grep | awk '{print $1}')
top -l 7 -s 5 -stats pid,command,cpu -pid $PID $(for p in $WK; do echo -pid $p; done) -pid $(pgrep -x WindowServer) > $B/top_$1.txt
screencapture -x $B/screen_$1.png
kill $PID; sleep 2; pkill -f vite; sleep 1; cp $B/session.bak projects/fixture-a/session.json
awk '/^PID/{s++;next} s>1 && NF>=3 {cpu[$2" "$1]+=$NF; c[$2" "$1]++} END{for(k in cpu) printf "%-30s %6.1f%%\n", k, cpu[k]/c[k]}' $B/top_$1.txt | sort -k3 -rn | head -6
