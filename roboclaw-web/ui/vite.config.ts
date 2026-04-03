import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const proxy = {
  '/ws': {
    target: 'ws://localhost:8765',
    ws: true,
  },
  '/api/embodied/ws': {
    target: 'ws://localhost:8765',
    ws: true,
  },
  '/api': {
    target: 'http://localhost:8765',
    changeOrigin: true,
  },
}

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy,
  },
  preview: {
    proxy,
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
