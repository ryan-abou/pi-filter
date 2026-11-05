"""
Face mesh helper  -  gives the Crimson Grin 478 face points instead of YuNet's 5.

YuNet (kiosk.py) still decides WHO is shown (zone, settle time, privacy).
This file only adds detail to faces that are already allowed on screen.

Every frame, for a filter with USES_MESH = True:
  1. Make a copy of the camera picture that is BLACK except the boxes around
     faces the kiosk is showing. People outside the zone never even reach
     MediaPipe.
  2. MediaPipe FaceLandmarker (VIDEO mode = it follows faces from frame to
     frame, which is steadier and faster than starting over every frame)
     finds 478 points on each face.
  3. Each mesh is matched to the kiosk face whose box holds its nose tip.
  4. The points are smoothed with a "One Euro" filter: when you hold still
     it smooths a LOT (no shaking), when you move fast it smooths very
     little (no lag). Plain smoothing can only do one or the other.
  5. If MediaPipe misses your face for a moment (fast move, blur, hand in
     front), we keep your last points and slide them along with YuNet's
     box for up to HOLD_SECONDS, so the grin does not blink off.
Result: face["mesh"] = 478 rows of (x, y, z) in screen pixels.

Privacy: the model file is on the Pi, nothing is saved, nothing leaves the
Pi, and everything is forgotten when the session ends (close()).
If MediaPipe is not installed, filters fall back to YuNet's 5 points.
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "face_landmarker.task")

NOSE_TIP = 1
BOX_GROW = 1.5        # how much bigger than YuNet's box the piece we give MediaPipe is
HOLD_SECONDS = 0.4    # keep the last mesh this long if MediaPipe misses a face

# One Euro filter settings (bigger MIN_CUTOFF = less smoothing when still,
# bigger BETA = less lag when moving fast)
MIN_CUTOFF = 1.0      # Hz
BETA = 0.15           # tested: grin stays within ~4% of mouth width while walking fast
D_CUTOFF = 1.0        # Hz, smoothing of the speed estimate


def _alpha(dt, cutoff):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """Smooths all of one face's points together. The whole face uses ONE
    speed, so the points move as a group and the face shape never bends."""

    def __init__(self, points, now):
        self.points = points.copy()
        self.speed = np.zeros_like(points)
        self.time = now

    def update(self, points, now):
        dt = max(now - self.time, 1e-3)
        self.time = now
        speed_now = (points - self.points) / dt
        self.speed += _alpha(dt, D_CUTOFF) * (speed_now - self.speed)
        face_speed = float(np.median(np.linalg.norm(self.speed[:, :2], axis=1)))  # px per second
        cutoff = MIN_CUTOFF + BETA * face_speed
        self.points += _alpha(dt, cutoff) * (points - self.points)
        return self.points.copy()


def _box_center(face):
    x, y, w, h = face["box"]
    return np.array([x + w / 2, y + h / 2], np.float32)


class FaceMesh:
    def __init__(self, max_faces):
        self.max_faces = max_faces
        self.landmarker = None
        self.failed = False       # True = MediaPipe missing, use 5 points
        self.memory = {}          # face id -> {"smooth", "center", "seen"}
        self.last_ms = 0

    def _start(self):
        """Load MediaPipe the first time a mesh filter is used in a session."""
        if self.landmarker is not None or self.failed:
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError:
            print("MediaPipe not installed - Crimson Grin uses 5 points (pip install mediapipe)")
            self.failed = True
            return
        if not os.path.exists(MODEL_PATH):
            print("Missing face_landmarker.task - run: python3 download_model.py")
            self.failed = True
            return
        self.mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        try:
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as error:          # e.g. a missing system library on the Pi
            print("MediaPipe could not start - Crimson Grin uses 5 points:", error)
            self.failed = True

    def close(self):
        """Session over: forget every face."""
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
        self.memory = {}

    def add_points(self, frame, faces, now):
        if not faces:
            self.memory = {}
            return
        self._start()
        if self.landmarker is None:
            return
        h, w = frame.shape[:2]

        # 1. Black picture with only the shown faces cut in
        only_faces = np.zeros_like(frame)
        for face in faces:
            cx, cy = _box_center(face)
            half = max(face["box"][2], face["box"][3]) * BOX_GROW / 2
            x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
            x1, y1 = min(w, int(cx + half)), min(h, int(cy + half))
            only_faces[y0:y1, x0:x1] = frame[y0:y1, x0:x1]

        # 2. MediaPipe (wants RGB and a time that always goes up)
        rgb = np.ascontiguousarray(only_faces[:, :, ::-1])
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self.last_ms = max(self.last_ms + 1, int(now * 1000))
        result = self.landmarker.detect_for_video(image, self.last_ms)
        meshes = [np.array([(p.x * w, p.y * h, p.z * w) for p in lms], np.float32)
                  for lms in result.face_landmarks]

        # 3. Match each mesh to the face whose box holds its nose tip
        found = {}
        for mesh in meshes:
            nose = mesh[NOSE_TIP, :2]
            best, best_d = None, None
            for face in faces:
                x, y, bw, bh = face["box"]
                if face["id"] in found:
                    continue
                if not (x - 0.2 * bw < nose[0] < x + 1.2 * bw and y - 0.2 * bh < nose[1] < y + 1.2 * bh):
                    continue
                d = np.linalg.norm(nose - _box_center(face))
                if best is None or d < best_d:
                    best, best_d = face, d
            if best is not None:
                found[best["id"]] = mesh

        # 4 + 5. Smooth, or hold the last mesh for a moment
        shown_ids = set()
        for face in faces:
            fid = face["id"]
            shown_ids.add(fid)
            mem = self.memory.get(fid)
            if fid in found:
                if mem is None:
                    mem = self.memory[fid] = {"smooth": OneEuro(found[fid], now)}
                    face["mesh"] = found[fid].copy()
                else:
                    face["mesh"] = mem["smooth"].update(found[fid], now)
                mem["center"], mem["seen"] = _box_center(face), now
            elif mem is not None and now - mem["seen"] <= HOLD_SECONDS:
                moved = _box_center(face) - mem["center"]
                held = mem["smooth"].points.copy()
                held[:, :2] += moved
                face["mesh"] = held
        # forget faces that are no longer shown
        self.memory = {k: v for k, v in self.memory.items() if k in shown_ids}