"""
Run the kiosk inside a GitHub Codespace, using your laptop's webcam
===================================================================
A Codespace has no webcam and no screen, so kiosk.py can't open a window
there. This file is a TESTING TOOL that fixes that:

  your browser  --(webcam picture)-->  this server (runs the REAL kiosk.py)
  your browser  <--(kiosk screen)----  this server

The browser only turns the webcam on while the kiosk says a session is
running (after START), and turns it off when the session ends.
Nothing is saved. On the Pi, run kiosk.py itself - not this file.

Run:   python3 codespace/web_test.py
Then open the "Ports" tab -> port 8000 -> click the globe icon.
"""
import argparse
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# Use the real kiosk code from the folder above this one
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kiosk as K


# ------------------------------------------------------------ fake camera
class BrowserCamera:
    """Looks like cv2.VideoCapture to kiosk.py, but its pictures come from the browser."""

    def __init__(self):
        self.latest = None

    def read(self):
        if self.latest is None:                      # browser camera still starting
            return True, np.zeros((K.CAM_H, K.CAM_W, 3), np.uint8)
        return True, self.latest

    def release(self):
        self.latest = None


camera = BrowserCamera()
kiosk = K.Kiosk((800, 480), camera_factory=lambda: camera)
lock = threading.Lock()


# ------------------------------------------------------------ one "frame" of the kiosk
def step(frame_bytes, tap, key):
    """Same order as Kiosk.run(): input -> camera -> timers -> draw."""
    with lock:
        if tap:
            x, y = (int(v) for v in tap.split(","))
            kiosk.tap(x, y)
        if key:
            kiosk.handle_key(ord(key[0]))
        if kiosk.state == "STARTING":
            kiosk.begin_session()                    # "opens" the camera
        if frame_bytes and kiosk.cap is not None:
            img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                camera.latest = img
        if kiosk.state == "SESSION":
            kiosk.process_frame()
        kiosk.tick()
        screen = kiosk.render()
        camera_on = kiosk.cap is not None
        return cv2.imencode(".jpg", screen, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), camera_on


# ------------------------------------------------------------ web page (browser side)
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Hallway Kiosk (Codespace test)</title>
<style>body{margin:0;background:#111;color:#aaa;font-family:sans-serif;text-align:center}
img{width:min(100vw,160vh);max-width:1200px;cursor:pointer;margin-top:10px}</style></head><body>
<img id="scr"><p>Click = tap. Press d for the zone debug view. Webcam turns on only during a session.</p>
<canvas id="cv" width="640" height="480" hidden></canvas>
<script>
const scr = document.getElementById('scr'), cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const video = document.createElement('video'); video.playsInline = true; video.muted = true;
let stream = null, tap = null, key = null, url = null;

scr.onclick = e => {               // turn the click into a tap on the 800x480 kiosk screen
  tap = Math.round(e.offsetX * 800 / scr.clientWidth) + ',' + Math.round(e.offsetY * 480 / scr.clientHeight);
};
document.onkeydown = e => { if (e.key === 'd') key = 'd'; };

async function cameraOn()  { stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
                             video.srcObject = stream; await video.play(); }
function cameraOff() { stream.getTracks().forEach(t => t.stop()); stream = null; }

async function loop() {
  let body = null;
  if (stream && video.readyState >= 2) {          // send the webcam picture only during a session
    ctx.drawImage(video, 0, 0, 640, 480);
    body = await new Promise(r => cv.toBlob(r, 'image/jpeg', 0.7));
  }
  const q = new URLSearchParams(); if (tap) q.set('tap', tap); if (key) q.set('key', key); tap = key = null;
  try {
    const r = await fetch('step?' + q, {method: 'POST', body});
    const on = r.headers.get('X-Camera') === '1';
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(await r.blob()); scr.src = url;
    if (on && !stream) await cameraOn().catch(e => alert('Allow the camera: ' + e));
    if (!on && stream) cameraOff();
  } catch (e) { await new Promise(r => setTimeout(r, 1000)); }
  setTimeout(loop, stream ? 0 : 80);
}
loop();
</script></body></html>"""


# ------------------------------------------------------------ tiny web server
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.reply(200, "text/html", PAGE.encode())

    def do_POST(self):
        query = parse_qs(urlparse(self.path).query)
        size = int(self.headers.get("Content-Length") or 0)
        frame = self.rfile.read(size) if size else None
        jpg, camera_on = step(frame, query.get("tap", [None])[0], query.get("key", [None])[0])
        self.reply(200, "image/jpeg", jpg, {"X-Camera": "1" if camera_on else "0", "X-State": kiosk.state})

    def reply(self, code, kind, data, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):                    # keep the terminal quiet
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    port = ap.parse_args().port
    print(f"Kiosk test running on port {port}. Open the Ports tab -> {port} -> globe icon.")
    print("Press Ctrl+C to stop.")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()