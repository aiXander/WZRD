import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';

/**
 * Dev-only stand-in for the shell's file commands (src-tauri/src/rpc.rs), so
 * the UI opened in a plain browser (src/api/wsBridge.ts) can read/write the
 * same scene, effects, masks and looks the Tauri shell would. Bound to the
 * scene the shell gets: `WZRD_SCENE` (relative to this folder). Never part
 * of a build.
 */
function wzrdFiles(): Plugin {
  const scenePath = path.resolve(
    process.env.WZRD_SCENE ?? '../render-core/examples/phase3_smoke.scene.json'
  );
  const sceneDir = path.dirname(scenePath);
  const effectsDir = path.join(sceneDir, 'effects');
  const looksDir = path.join(sceneDir, 'looks');
  const packDir = () => {
    const scene = JSON.parse(fs.readFileSync(scenePath, 'utf8'));
    return path.resolve(sceneDir, scene.pack ?? '.');
  };
  const stem = (name: string) => {
    if (!/^[A-Za-z0-9 _.-]{1,64}$/.test(name) || name.startsWith('.')) {
      throw new Error(`bad look name ${JSON.stringify(name)}`);
    }
    return name;
  };

  const handlers: Record<string, (a: any) => unknown> = {
    read_scene_file: () => fs.readFileSync(scenePath, 'utf8'),
    write_scene_file: (a) => fs.writeFileSync(scenePath, a.contents),
    list_effects: () =>
      fs.existsSync(effectsDir)
        ? fs
            .readdirSync(effectsDir)
            .filter((n) => fs.existsSync(path.join(effectsDir, n, 'shader.wgsl')))
            .sort()
        : [],
    read_effect: (a) => {
      const dir = path.join(effectsDir, a.name);
      const desc = path.join(dir, 'descriptor.json');
      return {
        name: a.name,
        wgsl: fs.readFileSync(path.join(dir, 'shader.wgsl'), 'utf8'),
        descriptor: fs.existsSync(desc) ? fs.readFileSync(desc, 'utf8') : null,
      };
    },
    read_mask_png: (a) =>
      fs.readFileSync(path.join(packDir(), a.maskPath)).toString('base64'),
    looks_list: () => {
      if (!fs.existsSync(looksDir)) return [];
      return fs
        .readdirSync(looksDir)
        .filter((n) => n.endsWith('.json'))
        .map((n) => {
          const p = path.join(looksDir, n);
          const jpg = p.replace(/\.json$/, '.jpg');
          return {
            name: n.replace(/\.json$/, ''),
            t: fs.statSync(p).birthtimeMs,
            thumb_b64: fs.existsSync(jpg) ? fs.readFileSync(jpg).toString('base64') : null,
          };
        })
        .sort((x, y) => x.t - y.t)
        .map(({ name, thumb_b64 }) => ({ name, thumb_b64 }));
    },
    look_read: (a) => fs.readFileSync(path.join(looksDir, `${stem(a.name)}.json`), 'utf8'),
    look_write: (a) => {
      JSON.parse(a.scene);
      fs.mkdirSync(looksDir, { recursive: true });
      fs.writeFileSync(path.join(looksDir, `${stem(a.name)}.json`), a.scene);
      if (a.thumbB64) {
        fs.writeFileSync(path.join(looksDir, `${stem(a.name)}.jpg`), Buffer.from(a.thumbB64, 'base64'));
      }
    },
    look_delete: (a) => {
      fs.rmSync(path.join(looksDir, `${stem(a.name)}.json`));
      fs.rmSync(path.join(looksDir, `${stem(a.name)}.jpg`), { force: true });
    },
  };

  return {
    name: 'wzrd-dev-files',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__wzrd', (req, res) => {
        let body = '';
        req.on('data', (c) => (body += c));
        req.on('end', () => {
          const cmd = (req.url ?? '').replace(/^\//, '');
          res.setHeader('content-type', 'application/json');
          try {
            const h = handlers[cmd];
            if (!h) throw new Error(`no dev handler for ${cmd}`);
            const result = h(body ? JSON.parse(body) : {});
            res.end(JSON.stringify({ result: result ?? null }));
          } catch (e) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(e) }));
          }
        });
      });
    },
  };
}

// Tauri's expected port matches `tauri.conf.json`'s `devUrl`.
export default defineConfig({
  plugins: [react(), wzrdFiles()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: false,
    // The engine rewrites scene/session/alignment sidecars and the app
    // writes looks — none of that is source; don't full-reload on it.
    watch: { ignored: ['**/looks/**', '**/.wzrd/**', '**/*.scene.json', '**/session.json', '**/alignment.json'] },
  },
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    target: 'esnext',
    minify: false,
    sourcemap: true,
  },
});
