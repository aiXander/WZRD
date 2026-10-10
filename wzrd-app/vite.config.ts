import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Tauri's expected port matches `tauri.conf.json`'s `devUrl`.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: false,
    // The engine rewrites scene/session/alignment sidecars, the app writes
    // looks, the MCP drops clips into generated/ — none of that is source;
    // don't full-reload on it.
    watch: {
      ignored: [
        '**/looks/**',
        '**/.wzrd/**',
        '**/*.scene.json',
        '**/session.json',
        '**/alignment.json',
        '**/projects/**',
      ],
    },
    // Browser bridge (src/api/wsBridge.ts): the shell's own commands go to
    // its debug-only HTTP twin (src-tauri/src/devport.rs). Dev only — this
    // config has no effect on a build.
    proxy: {
      '/__shell': {
        target: `http://127.0.0.1:${process.env.WZRD_DEVPORT ?? 9124}`,
        rewrite: (p) => p.replace(/^\/__shell/, ''),
      },
    },
  },
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    target: 'esnext',
    minify: false,
    sourcemap: true,
  },
});
