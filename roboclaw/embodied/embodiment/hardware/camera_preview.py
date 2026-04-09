"""Temporary local MJPEG camera preview server for identification."""

from __future__ import annotations

import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import cv2


class _PreviewHandler(BaseHTTPRequestHandler):
    """Serves camera previews as MJPEG streams or a tiled HTML page."""

    cameras: dict[int, str] = {}  # idx -> dev path, set by server

    def log_message(self, *_args: Any) -> None:
        pass  # suppress logs

    def do_GET(self) -> None:
        if self.path.startswith("/stream/"):
            self._stream_mjpeg()
        elif self.path == "/snapshot":
            self._snapshot_all()
        else:
            self._index_page()

    def _index_page(self) -> None:
        cams = sorted(self.cameras.items())
        imgs = "\n".join(
            f'<div style="text-align:center">'
            f'<h2>[{idx}] {dev}</h2>'
            f'<img src="/stream/{idx}" style="max-width:100%;border:2px solid #333;border-radius:8px">'
            f'</div>'
            for idx, dev in cams
        )
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Camera Preview</title>
<style>body{{background:#1a1a2e;color:#eee;font-family:system-ui;margin:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px}}</style>
</head><body>
<h1>Camera Preview</h1>
<p>Name your cameras in the terminal, then this page will close automatically.</p>
<div class="grid">{imgs}</div>
</body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _stream_mjpeg(self) -> None:
        try:
            idx = int(self.path.split("/")[-1])
        except (ValueError, IndexError):
            self.send_error(404)
            return
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            self.send_error(404, f"Cannot open camera {idx}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while not getattr(self.server, "_shutdown_flag", False):
                ret, frame = cap.read()
                if not ret:
                    break
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                data = jpg.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                time.sleep(0.1)  # ~10fps
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            cap.release()

    def _snapshot_all(self) -> None:
        """Single-frame snapshot of all cameras (fallback)."""
        self._index_page()


class CameraPreviewServer:
    """Start/stop a temporary MJPEG preview server."""

    def __init__(self, cameras: dict[int, str], port: int = 0):
        self._cameras = cameras
        _PreviewHandler.cameras = cameras
        self._server = HTTPServer(("127.0.0.1", port), _PreviewHandler)
        self._server._shutdown_flag = False
        self.port = self._server.server_address[1]
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> str:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        self._server._shutdown_flag = True
        self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=3)
