# Helpers for driving a running `pnpm tauri dev` WZRD app unattended.
#   source .claude/skills/afk-session/scripts/appenv.sh   (repo root; set B=<results dir> first, app log in $B/app.log)
# js '<expr>'   evaluate in the app webview, print the result (devport :9124 + app log)
# room <name>   switch room (prepare|align|perform|debug) via the store's setRoute
# rpc <method> '<json params>'   one engine JSON-RPC call over ws :9123
# snap          perf.snapshot summary (interval, drops, knob latency, GPU passes)
# knobs         100 Hz live brightness wiggle for 10 s (stamps knob_submit)
# sidebyside    main window right (1100×760 pt), engine output top-left (336×189):
#               neither covers the other (a covered engine window drops to ~28 fps)
# shot <name>   screenshot of the main window → $B/<name>.png (needs $B/winbounds:
#               swiftc -O scripts/winbounds.swift -o $B/winbounds)
B=${B:-test_results/afk}; mkdir -p $B
PID=$(pgrep -f target/debug/wzrd-app | head -1)
js() { local t=r$RANDOM; curl -s localhost:9124/cmd/webview_eval -d "$(python3 -c 'import json,sys;print(json.dumps({"js":"(async()=>{let r;try{r=await ("+sys.argv[1]+")}catch(e){r=\"ERR \"+e};await fetch(\"/__shell/cmd/debug_log\",{method:\"POST\",headers:{\"content-type\":\"application/json\"},body:JSON.stringify({msg:\"'$t' \"+(typeof r===\"string\"?r:JSON.stringify(r))})})})()"}))' "$1")" >/dev/null; sleep 1.5; grep -a "\[webview\] $t" $B/app.log | sed "s/.*$t //"; }
# Via the store, not localStorage + reload: a stored route of `align` boots into Perform (2026-10-14).
room() { js "import('/src/state/store.ts').then(m=>(m.useStore.getState().setRoute('$1'),'ok'))" >/dev/null; sleep 2; }
rpc() { uv run -q --with websockets python -c 'import asyncio,json,sys,websockets
async def m():
  async with websockets.connect("ws://127.0.0.1:9123",max_size=None) as w:
    await w.send(json.dumps({"jsonrpc":"2.0","id":1,"method":sys.argv[1],"params":json.loads(sys.argv[2])}))
    while True:
      r=json.loads(await w.recv())
      if r.get("id")==1: print(json.dumps(r.get("result",r.get("error")))[:400]); return
asyncio.run(m())' "$1" "${2:-{\}}"; }
snap() { uv run -q --with websockets python -c 'import asyncio,json,websockets
async def m():
  async with websockets.connect("ws://127.0.0.1:9123",max_size=None) as w:
    await w.send(json.dumps({"jsonrpc":"2.0","id":1,"method":"perf.snapshot","params":{}}))
    while True:
      r=json.loads(await w.recv())
      if r.get("id")==1:
        s=r["result"]; g=s["gpu"] or {}
        print(json.dumps({"presenting":s["presenting"],"work_p95":s["frame"]["work"]["p95"],"interval_p95":s["frame"]["interval"]["p95"],"dropped":s.get("dropped"),"latency":s.get("latency"),"passes":g.get("passes"),"total":g.get("total")})); return
asyncio.run(m())'; }
knobs() { uv run -q --with websockets python -c 'import asyncio,json,websockets
async def m():
  async with websockets.connect("ws://127.0.0.1:9123",max_size=None) as w:
    for i in range(1000):
      await w.send(json.dumps({"jsonrpc":"2.0","id":i,"method":"master.set","params":{"name":"brightness","value":0.95+0.05*(i%2),"leg":"live"}}))
      await asyncio.sleep(0.01)
asyncio.run(m())'; }
sidebyside() { osascript -e 'tell application "System Events" to tell process "wzrd-app"
  repeat with w in windows
    set s to size of w
    if item 1 of s > 900 then
      set position of w to {340, 25}
      set size of w to {1100, 760}
    else
      set position of w to {0, 25}
      set size of w to {336, 189}
    end if
  end repeat
end tell'; }
shot() { screencapture -x -o -l "$($B/winbounds $PID | awk '$4>900{print $1;exit}')" $B/$1.png; }
