// Browser-mode transport (see transport.ts). Engine methods go over the
// engine's own WS JSON-RPC; shell-only file commands go to the Vite dev
// middleware in vite.config.ts (`/__wzrd/<command>`).

type Handler = (payload: any) => void;

const WS_URL =
  new URLSearchParams(window.location.search).get('ws') ?? 'ws://127.0.0.1:9123';

/** Tauri command → engine method + arg reshaping (camelCase → wire names). */
const ENGINE: Record<string, (a: any) => [string, Record<string, unknown>]> = {
  pack_info: () => ['pack.info', {}],
  scene_get_state: (a) => ['scene.getState', { leg: a.leg ?? null }],
  scene_load: (a) => ['scene.load', { json: a.jsonText }],
  scene_reload: () => ['scene.reload', {}],
  param_set: (a) => ['param.set', { name: a.name, value: a.value, leg: a.leg }],
  param_override: (a) => [
    'param.set',
    { binding: a.binding, param: a.param, value: a.value, leg: a.leg },
  ],
  param_list: (a) => ['param.list', { leg: a.leg ?? null }],
  master_set: (a) => ['master.set', { name: a.name, value: a.value, leg: a.leg }],
  effect_describe: (a) => ['effect.describe', { name: a.name ?? null }],
  session_save: () => ['session.save', {}],
  promote: (a) => ['promote', { fade_ms: a.fadeMs, quantize: a.quantize }],
  pull: () => ['pull', {}],
  preview_set_source: (a) => ['preview.setSource', { source: a.source }],
  transport_set: (a) => [
    'transport.set',
    { bpm: a.bpm ?? null, resync: !!a.resync, sync: !!a.sync, leg: a.leg ?? null },
  ],
  probe_get_thresholds: () => ['probe.getThresholds', {}],
  probe_set_thresholds: (a) => ['probe.setThresholds', { a_ms: a.aMs, b_ms: a.bMs }],
  alignment_get: () => ['alignment.get', {}],
  alignment_set: (a) => [
    'alignment.set',
    Object.fromEntries(
      ['enabled', 'background', 'corners', 'points']
        .filter((k) => a[k] != null)
        .map((k) => [k, a[k]])
    ),
  ],
  alignment_reset: () => ['alignment.reset', {}],
  alignment_set_test_pattern: (a) => ['alignment.setTestPattern', { pattern: a.pattern }],
  wgsl_validate: (a) => ['wgsl.validate', { source: a.source }],
  effect_upsert: (a) => [
    'effect.upsert',
    { name: a.name, wgsl: a.wgsl, descriptor: a.descriptor ?? null },
  ],
  effect_remove: (a) => ['effect.remove', { name: a.name }],
};

const FILE_COMMANDS = new Set([
  'list_effects',
  'read_effect',
  'read_scene_file',
  'write_scene_file',
  'read_mask_png',
  'looks_list',
  'look_read',
  'look_write',
  'look_delete',
]);

let ws: WebSocket | null = null;
let connected = false;
let nextId = 1;
const pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();
const listeners = new Map<string, Set<Handler>>();
const sticky = new Map<string, unknown>();
let opening: Promise<void> | null = null;

function emit(event: string, payload: unknown) {
  listeners.get(event)?.forEach((h) => h(payload));
}

function send(method: string, params: Record<string, unknown>): Promise<any> {
  return ensureOpen().then(
    () =>
      new Promise((resolve, reject) => {
        const id = nextId++;
        pending.set(id, { resolve, reject });
        ws!.send(JSON.stringify({ jsonrpc: '2.0', id, method, params }));
      })
  );
}

function ensureOpen(): Promise<void> {
  if (ws && connected) return Promise.resolve();
  if (opening) return opening;
  opening = new Promise<void>((resolve) => {
    const sock = new WebSocket(WS_URL);
    ws = sock;
    sock.onopen = () => {
      connected = true;
      opening = null;
      resolve();
      // Declare ourselves the operator seat, then take every channel.
      send('hello', { actor: 'ui' }).catch(() => {});
      send('telemetry.channels', {})
        .then((r: any) =>
          send('telemetry.subscribe', { channels: r?.channels ?? r ?? [] })
        )
        .catch((e) => console.warn('[bridge] subscribe', e));
      emit('engine:status', { running: true, ws_addr: WS_URL });
    };
    sock.onmessage = (m) => {
      const msg = JSON.parse(m.data);
      if (msg.method === 'telemetry.event') {
        const { channel, payload } = msg.params;
        sticky.set(channel, payload);
        emit('engine:telemetry', { channel, payload });
        return;
      }
      const p = pending.get(msg.id);
      if (!p) return;
      pending.delete(msg.id);
      if (msg.error) p.reject(msg.error.message ?? JSON.stringify(msg.error));
      else p.resolve(msg.result);
    };
    sock.onclose = () => {
      const was = connected;
      connected = false;
      opening = null;
      ws = null;
      pending.forEach((p) => p.reject('engine socket closed'));
      pending.clear();
      if (was) emit('engine:status', { running: false });
      // Reconnect gently; telemetry resumes with a sticky replay.
      window.setTimeout(() => ensureOpen().catch(() => {}), 1000);
      resolve();
    };
  });
  return opening;
}

export function bridgeCall<T>(cmd: string, args: Record<string, any>): Promise<T> {
  if (cmd === 'engine_status') {
    return ensureOpen().then(
      () => ({ running: connected, ws_addr: WS_URL, last_error: null }) as T
    );
  }
  if (cmd === 'last_payload') {
    return ensureOpen().then(() => (sticky.get(args.channel) ?? null) as T);
  }
  if (cmd === 'preview_set_bounds') return Promise.resolve(undefined as T);
  if (FILE_COMMANDS.has(cmd)) {
    return fetch(`/__wzrd/${cmd}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(args),
    }).then(async (r) => {
      const body = await r.json();
      if (!r.ok) throw body.error ?? 'file command failed';
      return body.result as T;
    });
  }
  const map = ENGINE[cmd];
  if (!map) return Promise.reject(`browser bridge: no mapping for ${cmd}`);
  const [method, params] = map(args);
  return send(method, params);
}

export function bridgeListen<T>(event: string, handler: (payload: T) => void) {
  let set = listeners.get(event);
  if (!set) listeners.set(event, (set = new Set()));
  set.add(handler as Handler);
  ensureOpen().catch(() => {});
  return Promise.resolve(() => {
    set!.delete(handler as Handler);
  });
}
