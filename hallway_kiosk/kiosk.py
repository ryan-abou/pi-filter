"""
Hallway Filter Kiosk  -  Prototype 1
====================================
One touchscreen app that runs all four group filters on a Raspberry Pi with
a USB webcam. Run:  python3 kiosk.py

Screens (self.state)
--------------------
MENU      Four filter tiles with sample pictures.              Camera OFF
CONSENT   "In a rush?" short version + full terms,
          an "I agree" box and a START button.                 Camera OFF
STARTING  "Starting camera..." while the camera opens.
SESSION   Camera ON for SESSION_SECONDS. Every face inside the
          taped opt-in zone gets the chosen filter.
          Buttons: SWITCH FILTER (timer restarts) and STOP.
DONE      "Session over". Back to MENU after DONE_SECONDS.      Camera OFF

Privacy promises - the sign and consent screen say these, so keep them TRUE
--------------------------------------------------------------------------
1. The camera is opened ONLY in begin_session() (after START) and released
   in end_session() (timer runs out, STOP, or a camera error).
2. No frame is ever saved to disk and this app does no networking.
3. The screen is black except soft ovals around faces INSIDE the zone.
   Faces outside the zone are detected (so we know to hide them) but are
   never drawn.

Maintainer keys (plug in a keyboard): q or Esc = quit, d = zone debug view.
"""
import math
import os
import sys
import time

import cv2
import numpy as np

import face_mesh
from filters import crimson_grin, gold_crown, robot_circuits, royal_crown

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
MODEL_PATH = os.path.join(HERE, "face_detection_yunet_2023mar.onnx")

# Order of the tiles on the menu
FILTERS = [crimson_grin, royal_crown, robot_circuits, gold_crown]
# ------------------------------------------------------------------ settings
# Session (the same for every filter - only the filter itself is different)
SESSION_SECONDS = 30     # Pavan's stories: "wearing a virtual crown for 30 seconds"
CONSENT_TIMEOUT = 45     # consent screen goes back to the menu if nobody touches it
DONE_SECONDS = 3         # how long "Session over" stays up

# Camera
CAMERA_INDEX = 0
CAM_W, CAM_H = 640, 480

# Opt-in zone  -  CALIBRATE THESE AT THE SPOT (see README, press d)
ZONE_FAR_FACE_W = 40         # face width in pixels when someone stands at the BACK edge (6 ft)
ZONE_CENTER_X = 0.5          # middle of the taped zone in the camera picture (0 = left, 1 = right)
ZONE_HALF_WIDTH_FACES = 3.0  # half the zone width in face widths (a face is ~0.5 ft, so 3 = 1.5 ft)

# Face tracking
SCORE_THRESHOLD = 0.8     # same as the class glasses code
SMOOTH = 0.4              # from Krishna's/Pavan's filters: 1 = no smoothing, smaller = smoother
MATCH_LIMIT = 1.5         # a new detection this close (in eye distances) is the SAME face
SETTLE_SECONDS = 0.3      # a NEW face must be in the zone and facing the camera this long
LOST_HOLD_SECONDS = 0.5   # keep a face this long if the detector misses it (blink, quick turn)
MAX_FACES = 4

SCREEN_SIZE = None        # e.g. (800, 480). None = ask the operating system
WINDOW = "Hallway Filter Kiosk"

# ------------------------------------------------------------- words on screen
QUICK_TERMS = [
    f"Camera is on for {SESSION_SECONDS} seconds, then turns off by itself.",
    "Nothing is saved, recorded or sent anywhere.",
    "Only faces inside the taped zone are shown.",
    "Friends can join by stepping into the zone.",
    "Tap STOP any time to end early.",
]
FULL_TERMS = [
    "What the camera does: after you tap START it looks for faces so it knows where "
    "to put the filter. It finds faces; it does not recognize who you are.",
    "What happens to images: each video frame is used once to draw the screen and is "
    "then replaced by the next one. This app never saves, records or uploads images.",
    "Who is shown: the screen stays black except faces inside the taped zone on the "
    "floor. Anyone who steps into the zone during a session is shown with the filter.",
    "How to stay off screen: stay outside the taped zone. People walking behind, more "
    "than 6 ft from the camera, are not shown.",
    f"Ending: the session stops after {SESSION_SECONDS} seconds or when you tap STOP, "
    "and the camera turns off.",
    "This is a student prototype and face detection can make mistakes. If anything "
    "looks wrong, tap STOP.",
]

# Colors are BGR (OpenCV order)
BG = (34, 28, 24)
PANEL = (60, 50, 42)
EDGE = (95, 85, 75)
WHITE = (255, 255, 255)
GREY = (175, 175, 175)
DARK = (90, 90, 90)
GREEN = (80, 170, 70)
RED = (60, 60, 200)
BLUE = (190, 130, 40)
YELLOW = (60, 210, 245)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOLD = cv2.FONT_HERSHEY_DUPLEX


# ============================================================ face helpers
def make_face(p):
    """Turn one YuNet row into named points the filters use.
    YuNet row: x, y, w, h, eye1 x/y, eye2 x/y, nose x/y, mouth1 x/y, mouth2 x/y, score"""
    eye_a, eye_b = (float(p[4]), float(p[5])), (float(p[6]), float(p[7]))
    mouth_a, mouth_b = (float(p[10]), float(p[11])), (float(p[12]), float(p[13]))
    left_eye, right_eye = sorted([eye_a, eye_b])          # left = smaller x on screen
    mouth_left, mouth_right = sorted([mouth_a, mouth_b])
    return {
        "box": (int(p[0]), int(p[1]), int(p[2]), int(p[3])),
        "left_eye": left_eye, "right_eye": right_eye,
        "nose": (float(p[8]), float(p[9])),
        "mouth_left": mouth_left, "mouth_right": mouth_right,
    }


def eye_distance(p):
    return max(1.0, math.hypot(p[6] - p[4], p[7] - p[5]))


def in_zone(face):
    """Is this face standing inside the taped opt-in zone?

    Distance: faces look smaller the farther away they are. A face narrower
    than ZONE_FAR_FACE_W pixels is behind the back edge of the zone (6 ft).

    Side to side: we measure how far the face is from the zone's center line
    in FACE WIDTHS, not pixels. That keeps the zone the same real width
    whether the person is 2 ft or 6 ft away (a pixel band would get wider
    the farther away you go)."""
    x, y, w, h = face["box"]
    if w < ZONE_FAR_FACE_W:
        return False
    center_x = x + w / 2
    offset_in_faces = abs(center_x - ZONE_CENTER_X * CAM_W) / w
    return offset_in_faces <= ZONE_HALF_WIDTH_FACES


def facing_camera(face):
    """True when the nose is between the eyes (head turned less than ~45 deg).
    Someone walking past with their head turned away fails this."""
    (lx, _), (rx, _) = face["left_eye"], face["right_eye"]
    if rx - lx <= 1:
        return False
    nose_position = (face["nose"][0] - lx) / (rx - lx)   # 0.5 = looking straight
    return 0.2 < nose_position < 0.8


class Track:
    """One person's face, remembered from frame to frame."""
    next_id = 0                   # every new face gets its own number

    def __init__(self, points, now):
        Track.next_id += 1
        self.id = Track.next_id       # lets face_mesh.py keep each person's points apart
        self.points = points          # smoothed YuNet numbers (first 14) - used for drawing
        self.raw = points             # last real detection - used for matching
        self.last_seen = now
        self.seen_this_frame = True
        self.ok_since = None          # when it started being in the zone AND facing us
        self.shown = False            # approved to be drawn on screen


class FaceTracker:
    """Keeps a smoothed position for EACH face and decides who may be shown.

    Rules (these are what make the privacy criteria pass):
      * a NEW face is shown only after SETTLE_SECONDS inside the zone while
        facing the camera (so a passer-by glancing over is never shown)
      * a face already shown stays shown while it is in the zone, even if the
        person turns their head (so the filter does not drop off a user)
      * the moment a face leaves the zone it is hidden
      * if the detector misses a face for a moment it is held for
        LOST_HOLD_SECONDS, then forgotten
    """

    def __init__(self):
        self.tracks = []

    def clear(self):
        self.tracks = []

    def update(self, detections, now):
        found = [] if detections is None else [d[:14].astype(np.float32) for d in detections]
        unused = list(range(len(found)))

        # 1. Match each known face to the nearest new detection
        for track in self.tracks:
            track.seen_this_frame = False
            best, best_dist = None, None
            for i in unused:
                # compare with the last REAL detection, not the smoothed one: the smoothed
                # point lags behind a walking person and would stop matching
                dist = math.hypot(found[i][4] - track.raw[4], found[i][5] - track.raw[5])
                if dist < MATCH_LIMIT * eye_distance(track.raw) and (best is None or dist < best_dist):
                    best, best_dist = i, dist
            if best is not None:
                unused.remove(best)
                # Krishna's/Pavan's smoothing: move part of the way toward the new reading
                track.points = track.points + SMOOTH * (found[best] - track.points)
                track.raw = found[best]
                track.last_seen = now
                track.seen_this_frame = True

        # 2. Detections nobody matched are new people
        for i in unused:
            self.tracks.append(Track(found[i], now))

        # 3. Forget faces that have been missing too long
        self.tracks = [t for t in self.tracks if now - t.last_seen <= LOST_HOLD_SECONDS]

        # 4. Decide who is shown
        for track in self.tracks:
            face = make_face(track.points)
            if not in_zone(face):
                track.shown, track.ok_since = False, None
                continue
            if track.shown:
                continue
            if track.seen_this_frame and facing_camera(face):
                if track.ok_since is None:
                    track.ok_since = now
                if now - track.ok_since >= SETTLE_SECONDS:
                    track.shown = True
            else:
                track.ok_since = None

    def shown_faces(self):
        shown = [t for t in self.tracks if t.shown]
        shown.sort(key=lambda t: -t.points[2])       # biggest (closest) first
        faces = []
        for t in shown[:MAX_FACES]:
            face = make_face(t.points)
            face["id"] = t.id
            faces.append(face)
        return faces

def reveal_faces(camera_frame, output, faces):
    """Copy ONLY soft ovals around the shown faces from the camera onto the
    black output (Abhi's display_participant idea, for several faces)."""
    if not faces:
        return
    h, w = camera_frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for face in faces:
        x, y, bw, bh = face["box"]
        center = (int(x + bw / 2), int(y + bh * 0.40))
        axes = (int(bw * 0.80), int(bh * 0.95))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    alpha = mask[:, :, None].astype(np.float32) / 255.0
    output[:] = (camera_frame * alpha + output * (1 - alpha)).astype(np.uint8)


def draw_debug(output, tracker):
    """Maintainer view (key d): boxes only, never the camera picture.
    green = shown, yellow = in zone but not approved yet, red = outside zone."""
    cx = int(ZONE_CENTER_X * CAM_W)
    cv2.line(output, (cx, 0), (cx, CAM_H), DARK, 1)
    for track in tracker.tracks:
        face = make_face(track.points)
        x, y, w, h = face["box"]
        color = GREEN if track.shown else (YELLOW if in_zone(face) else RED)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        put(output, f"w={w}px", x, max(14, y - 6), 0.5, color)
    put(output, f"ZONE_FAR_FACE_W={ZONE_FAR_FACE_W}  (stand at 6 ft, read w)", 8, 20, 0.45, YELLOW)


# ============================================================ drawing helpers
def text_size(s, scale, thick=1, font=FONT):
    (w, h), _ = cv2.getTextSize(s, font, scale, thick)
    return w, h


def put(img, s, x, y, scale, color=WHITE, thick=1, font=FONT, center=False):
    if center:
        x = x - text_size(s, scale, thick, font)[0] / 2
    cv2.putText(img, s, (int(x), int(y)), font, scale, color, thick, cv2.LINE_AA)


def put_fit(img, s, cx, y, max_w, scale, color=WHITE, thick=1, font=FONT):
    """Centered text that shrinks until it fits max_w."""
    while scale > 0.3 and text_size(s, scale, thick, font)[0] > max_w:
        scale -= 0.05
    put(img, s, cx, y, scale, color, thick, font, center=True)


def wrap(s, scale, thick, max_w, font=FONT):
    lines, line = [], ""
    for word in s.split():
        test = (line + " " + word).strip()
        if text_size(test, scale, thick, font)[0] <= max_w:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def paragraph(img, s, x, y, max_w, scale, color=WHITE, thick=1, gap=1.6, font=FONT):
    """Draw wrapped text, return the y below it."""
    line_h = int(text_size("Hg", scale, thick, font)[1] * gap)
    for line in wrap(s, scale, thick, max_w, font):
        put(img, line, x, y, scale, color, thick, font)
        y += line_h
    return y


def box(img, x1, y1, x2, y2, color, border=None):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    if border:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), border, 2)


def paste(dst, src, x, y):
    h, w = src.shape[:2]
    dst[int(y):int(y) + h, int(x):int(x) + w] = src


def load_png(filt):
    path = os.path.join(ASSETS, filt.PNG)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Missing {path}")
    if img.ndim != 3 or img.shape[2] != 4:
        raise SystemExit(f"{filt.PNG} needs a transparent background (4 channels)")
    return filt.prepare(img)


def make_preview(filt, img, size=300):
    """Sample picture for a menu tile: the REAL filter code drawn on a plain
    grey mannequin, so no real person's photo is used."""
    card = np.full((size, size, 3), (70, 60, 52), np.uint8)
    cv2.ellipse(card, (150, 300), (115, 55), 0, 180, 360, (115, 115, 115), -1)   # shoulders
    cv2.rectangle(card, (128, 225), (172, 260), (140, 140, 140), -1)             # neck
    cv2.ellipse(card, (150, 165), (62, 80), 0, 0, 360, (165, 165, 165), -1)      # head
    for ex in (127, 173):
        cv2.circle(card, (ex, 150), 6, (60, 60, 60), -1)                           # eyes
    cv2.line(card, (150, 160), (146, 180), (120, 120, 120), 2)                    # nose
    cv2.ellipse(card, (150, 202), (20, 8), 0, 0, 180, (90, 90, 90), 2)            # mouth
    points = np.array([100, 105, 100, 125, 173, 150, 127, 150, 150, 180,
                       170, 204, 130, 204], np.float32)
    return filt.draw(card, make_face(points), img)


def open_camera():
    if sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def detect_screen_size():
    if SCREEN_SIZE:
        return SCREEN_SIZE
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return (800, 480)


# ============================================================ the kiosk
class Kiosk:
    def __init__(self, screen_size, camera_factory=open_camera, clock=time.monotonic):
        self.W, self.H = screen_size
        self.S = min(self.W / 800, self.H / 480)        # size everything for this screen
        self.clock = clock
        self.camera_factory = camera_factory
        if not os.path.exists(MODEL_PATH):
            raise SystemExit("Missing face model - run: python3 download_model.py")
        self.detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (CAM_W, CAM_H),
                                                  score_threshold=SCORE_THRESHOLD)
        self.images = [load_png(f) for f in FILTERS]
        self.previews = [make_preview(f, img) for f, img in zip(FILTERS, self.images)]
        self.mesh = face_mesh.FaceMesh(MAX_FACES)     # extra face points for filters that want them        self.cap = None
        self.state = "MENU"
        self.state_since = clock()
        self.choice = 0
        self.agreed = False
        self.session_end = 0.0
        self.show_switch = False
        self.last_output = None
        self.faces_shown = 0
        self.message = ""
        self.hint, self.hint_until = "", 0.0
        self.debug = False
        self.buttons = []
        self.taps = []
        self.last_tap = -1.0
        self.running = True

    def px(self, v):
        return int(v * self.S)

    def go(self, state):
        self.state = state
        self.state_since = self.clock()

    # ---------------------------------------------------------- camera on/off
    def begin_session(self):
        """The ONLY place the camera is opened."""
        self.cap = self.camera_factory()
        if self.cap is None:
            self.message = "Camera problem - please tell a teacher."
            self.go("DONE")
            return
        self.tracker.clear()
        self.faces_shown = 0
        self.last_output = np.zeros((CAM_H, CAM_W, 3), np.uint8)
        self.show_switch = False
        self.message = ""
        self.session_end = self.clock() + SESSION_SECONDS
        self.go("SESSION")

    def end_session(self):
        """Release the camera and forget everything about the session."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.tracker.clear()
        self.mesh.close()                                # forget every face's points too
        self.last_output = None
        self.faces_shown = 0
        self.show_switch = False
        self.go("DONE")

    def process_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.message = "Camera problem - please tell a teacher."
            self.end_session()
            return
        frame = cv2.flip(frame, 1)                       # mirror, like a selfie camera
        if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
            frame = cv2.resize(frame, (CAM_W, CAM_H))
        self.detector.setInputSize((CAM_W, CAM_H))
        _, detections = self.detector.detect(frame)
        self.tracker.update(detections, self.clock())
        faces = self.tracker.shown_faces()

        output = np.zeros_like(frame)                    # start from a BLACK screen
        reveal_faces(frame, output, faces)               # only zone faces show through
        filt, img = FILTERS[self.choice], self.images[self.choice]
        if getattr(filt, "USES_MESH", False):
            self.mesh.add_points(frame, faces, self.clock())   # adds face["mesh"] (478 points)
        for face in faces:
            output = filt.draw(output, face, img)
        if self.debug:
            draw_debug(output, self.tracker)
        self.last_output = output
        self.faces_shown = len(faces)

    # ---------------------------------------------------------- input
    def on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:              # a finger tap is a left click
            self.taps.append((x, y))

    def add_button(self, x1, y1, x2, y2, action):
        self.buttons.append((int(x1), int(y1), int(x2), int(y2), action))

    def button(self, x1, y1, x2, y2, label, action, color, scale=0.8):
        cv2.rectangle(self.screen, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        cv2.rectangle(self.screen, (int(x1), int(y1)), (int(x2), int(y2)), WHITE, 2)
        put_fit(self.screen, label, (x1 + x2) / 2, (y1 + y2) / 2 + self.px(10),
                (x2 - x1) - self.px(12), scale * self.S, WHITE, max(1, self.px(2)), BOLD)
        self.add_button(x1, y1, x2, y2, action)

    def tap(self, x, y):
        now = self.clock()
        if now - self.last_tap < 0.25:                   # ignore accidental double taps
            return
        self.last_tap = now
        if self.state == "CONSENT":
            self.state_since = now                       # still being used: reset timeout
        for x1, y1, x2, y2, action in reversed(self.buttons):   # top-most first
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.do(action)
                return

    def do(self, action):
        name = action[0]
        now = self.clock()
        if name == "pick":
            self.choice, self.agreed = action[1], False
            self.go("CONSENT")
        elif name == "menu":
            self.go("MENU")
        elif name == "agree":
            self.agreed = not self.agreed
        elif name == "start":
            if self.agreed:
                self.go("STARTING")
            else:
                self.hint, self.hint_until = "Tick the box first", now + 2
        elif name == "switch_panel":
            self.show_switch = not self.show_switch
        elif name == "switch":
            if action[1] != self.choice:
                self.choice = action[1]
                self.session_end = now + SESSION_SECONDS     # timer restarts
            self.show_switch = False
        elif name == "stop":
            self.end_session()

    def handle_key(self, key):
        if key in (ord("q"), 27):
            self.running = False
        elif key == ord("d"):
            self.debug = not self.debug

    def tick(self):
        """Timers that move between screens."""
        now = self.clock()
        if self.state == "SESSION" and now >= self.session_end:
            self.end_session()
        elif self.state == "CONSENT" and now - self.state_since > CONSENT_TIMEOUT:
            self.go("MENU")
        elif self.state == "DONE" and now - self.state_since > DONE_SECONDS:
            self.message = ""
            self.go("MENU")

    # ---------------------------------------------------------- screens
    def render(self):
        self.buttons = []
        self.screen = np.full((self.H, self.W, 3), BG, np.uint8)
        getattr(self, "draw_" + self.state.lower())(self.screen)
        return self.screen

    def draw_menu(self, scr):
        W, H, S, px = self.W, self.H, self.S, self.px
        put(scr, "Try a face filter!", W / 2, px(46), 1.1 * S, WHITE, max(1, px(2)), BOLD, center=True)
        put(scr, "Tap a picture to pick one.", W / 2, px(76), 0.6 * S, GREY, 1, center=True)
        margin, strip_h, top = px(16), px(92), px(92)
        n = len(FILTERS)
        tile_w = (W - margin * (n + 1)) // n
        tile_h = H - strip_h - top - margin
        side = max(10, min(tile_w - px(16), tile_h - px(62)))
        for i, filt in enumerate(FILTERS):
            x1 = margin + i * (tile_w + margin)
            x2, y1, y2 = x1 + tile_w, top, top + tile_h
            box(scr, x1, y1, x2, y2, PANEL, EDGE)
            prev = cv2.resize(self.previews[i], (side, side), interpolation=cv2.INTER_AREA)
            paste(scr, prev, x1 + (tile_w - side) // 2, y1 + px(8))
            put_fit(scr, filt.NAME, (x1 + x2) / 2, y1 + px(8) + side + px(28),
                    tile_w - px(10), 0.65 * S, WHITE, max(1, px(2)))
            put_fit(scr, "by " + filt.OWNER, (x1 + x2) / 2, y1 + px(8) + side + px(50),
                    tile_w - px(10), 0.5 * S, GREY)
            self.add_button(x1, y1, x2, y2, ("pick", i))
        y = H - strip_h
        box(scr, 0, y, W, H, (18, 18, 18))
        put(scr, "CAMERA IS OFF", px(16), y + px(28), 0.7 * S, YELLOW, max(1, px(2)), BOLD)
        paragraph(scr, "It only turns on after someone agrees and taps START, for "
                  f"{SESSION_SECONDS} seconds. Nothing is saved or sent anywhere. To stay "
                  "off screen, stay outside the taped zone on the floor.",
                  px(16), y + px(52), W - px(32), 0.5 * S, WHITE)

    def draw_consent(self, scr):
        W, H, S, px = self.W, self.H, self.S, self.px
        filt = FILTERS[self.choice]
        m = px(14)
        put(scr, f"{filt.NAME}  by {filt.OWNER}", m, px(34), 0.8 * S, WHITE, max(1, px(2)), BOLD)
        bar_h = px(82)
        top, bottom = px(48), H - bar_h - px(6)
        split = int(W * 0.40)

        # Left: the short version for people in a rush
        box(scr, m, top, split - m // 2, bottom, (40, 68, 40), EDGE)
        put(scr, "IN A RUSH? SHORT VERSION:", m + px(10), top + px(26), 0.55 * S, YELLOW, max(1, px(2)), BOLD)
        y = top + px(56)
        for item in QUICK_TERMS:
            cv2.circle(scr, (m + px(18), y - px(6)), max(2, px(4)), WHITE, -1)
            y = paragraph(scr, item, m + px(30), y, split - m - px(44), 0.5 * S, WHITE) + px(8)

        # Right: the full terms
        box(scr, split + m // 2, top, W - m, bottom, PANEL, EDGE)
        put(scr, "FULL TERMS", split + m + px(6), top + px(26), 0.55 * S, YELLOW, max(1, px(2)), BOLD)
        y = top + px(50)
        for n, item in enumerate(FULL_TERMS, 1):
            y = paragraph(scr, f"{n}. {item}", split + m + px(6), y,
                          W - split - 2 * m - px(18), 0.4 * S, WHITE, gap=1.55) + px(4)

        # Bottom bar: BACK | [ ] I agree | START   (kept low so seated users can reach)
        by1, by2 = H - bar_h + px(8), H - px(10)
        self.button(m, by1, m + px(120), by2, "BACK", ("menu",), DARK, 0.7)
        cb = by2 - by1 - px(14)
        cx1, cy1 = m + px(140), by1 + px(7)
        cv2.rectangle(scr, (cx1, cy1), (cx1 + cb, cy1 + cb), WHITE, max(2, px(3)))
        if self.agreed:
            cv2.line(scr, (cx1 + cb // 5, cy1 + cb // 2), (cx1 + cb // 2 - px(2), cy1 + cb - cb // 5), GREEN, max(2, px(5)))
            cv2.line(scr, (cx1 + cb // 2 - px(2), cy1 + cb - cb // 5), (cx1 + cb - cb // 6, cy1 + cb // 6), GREEN, max(2, px(5)))
        start_x1 = W - m - px(170)
        label_w = start_x1 - (cx1 + cb + px(12)) - px(10)
        lines = wrap("I agree to be shown on screen with this filter.", 0.55 * S, 1, label_w)
        ly = (by1 + by2) / 2 - (len(lines) - 1) * px(11) + px(6)
        for line in lines:
            put(scr, line, cx1 + cb + px(12), ly, 0.55 * S, WHITE)
            ly += px(22)
        self.add_button(cx1 - px(6), by1, start_x1 - px(10), by2, ("agree",))
        self.button(start_x1, by1, W - m, by2, "START", ("start",), GREEN if self.agreed else DARK, 0.9)
        if self.hint and self.clock() < self.hint_until:
            put(scr, self.hint, start_x1 + px(85), by1 - px(10), 0.55 * S, YELLOW, max(1, px(2)), center=True)

    def draw_starting(self, scr):
        put(scr, "Starting camera...", self.W / 2, self.H / 2, 1.0 * self.S, WHITE, max(1, self.px(2)), BOLD, center=True)

    def draw_session(self, scr):
        W, H, S, px = self.W, self.H, self.S, self.px
        side = max(px(190), int(W * 0.24))
        area_w = W - side
        out = self.last_output if self.last_output is not None else np.zeros((CAM_H, CAM_W, 3), np.uint8)
        scale = min(area_w / CAM_W, H / CAM_H)
        vw, vh = int(CAM_W * scale), int(CAM_H * scale)
        ox, oy = (area_w - vw) // 2, (H - vh) // 2
        paste(scr, cv2.resize(out, (vw, vh)), ox, oy)
        if self.faces_shown == 0 and not self.show_switch:
            put(scr, "Step into the taped zone", ox + vw / 2, oy + vh / 2 - px(14), 0.8 * S, WHITE, max(1, px(2)), center=True)
            put(scr, "and look at the camera", ox + vw / 2, oy + vh / 2 + px(20), 0.8 * S, WHITE, max(1, px(2)), center=True)

        # Sidebar: filter name, countdown, buttons
        sx = area_w
        box(scr, sx, 0, W, H, PANEL)
        filt = FILTERS[self.choice]
        put_fit(scr, filt.NAME, sx + side / 2, px(34), side - px(16), 0.7 * S, WHITE, max(1, px(2)), BOLD)
        remaining = max(0.0, self.session_end - self.clock())
        secs = int(math.ceil(remaining))
        put(scr, f"{secs // 60}:{secs % 60:02d}", sx + side / 2, px(104), 1.9 * S, YELLOW, max(2, px(4)), BOLD, center=True)
        bar_x1, bar_x2, bar_y = sx + px(16), W - px(16), px(120)
        box(scr, bar_x1, bar_y, bar_x2, bar_y + px(12), DARK)
        box(scr, bar_x1, bar_y, bar_x1 + (bar_x2 - bar_x1) * remaining / SESSION_SECONDS, bar_y + px(12), YELLOW)
        put(scr, "seconds left", sx + side / 2, px(152), 0.5 * S, GREY, center=True)
        paragraph(scr, "Friends: step into the taped zone to join.", sx + px(14), px(186), side - px(28), 0.48 * S, WHITE)
        b_h = px(62)
        self.button(sx + px(14), H - 2 * b_h - px(26), W - px(14), H - b_h - px(26),
                    "CLOSE" if self.show_switch else "SWITCH FILTER", ("switch_panel",), BLUE, 0.7)
        self.button(sx + px(14), H - b_h - px(12), W - px(14), H - px(12), "STOP", ("stop",), RED, 0.9)

        if self.show_switch:
            self.draw_switch_panel(scr, ox, oy, vw, vh)

    def draw_switch_panel(self, scr, ox, oy, vw, vh):
        S, px = self.S, self.px
        n = len(FILTERS)
        panel_h = int(vh * 0.62)
        y1 = oy + vh - panel_h
        box(scr, ox, y1, ox + vw, oy + vh, (25, 25, 25), EDGE)
        put(scr, "Pick a different filter (timer restarts)", ox + vw / 2, y1 + px(28), 0.6 * S, YELLOW, max(1, px(2)), center=True)
        gap = px(10)
        tile_w = (vw - gap * (n + 1)) // n
        side = max(10, min(tile_w - px(10), panel_h - px(80)))
        for i, filt in enumerate(FILTERS):
            tx1 = ox + gap + i * (tile_w + gap)
            ty1 = y1 + px(42)
            ty2 = oy + vh - gap
            box(scr, tx1, ty1, tx1 + tile_w, ty2, PANEL, YELLOW if i == self.choice else EDGE)
            prev = cv2.resize(self.previews[i], (side, side), interpolation=cv2.INTER_AREA)
            paste(scr, prev, tx1 + (tile_w - side) // 2, ty1 + px(5))
            put_fit(scr, filt.NAME, tx1 + tile_w / 2, ty2 - px(8), tile_w - px(6), 0.5 * S, WHITE)
            self.add_button(tx1, ty1, tx1 + tile_w, ty2, ("switch", i))

    def draw_done(self, scr):
        W, H, S, px = self.W, self.H, self.S, self.px
        put(scr, "Session over", W / 2, H / 2 - px(50), 1.3 * S, WHITE, max(2, px(3)), BOLD, center=True)
        put(scr, "The camera is off.", W / 2, H / 2, 0.8 * S, YELLOW, max(1, px(2)), center=True)
        if self.message:
            put(scr, self.message, W / 2, H / 2 + px(40), 0.6 * S, RED, max(1, px(2)), center=True)
        put(scr, "Tap anywhere for the menu", W / 2, H - px(40), 0.55 * S, GREY, center=True)
        self.add_button(0, 0, W, H, ("menu",))

    # ---------------------------------------------------------- main loop
    def run(self, fullscreen=True):
        flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
        cv2.namedWindow(WINDOW, flags)
        if fullscreen:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(WINDOW, self.W, self.H)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        try:
            while self.running:
                if self.state == "SESSION":
                    self.process_frame()
                self.tick()
                cv2.imshow(WINDOW, self.render())
                if self.state == "STARTING":
                    cv2.waitKey(1)                   # paint "Starting camera..." first
                    self.begin_session()
                    self.taps.clear()
                    continue
                key = cv2.waitKey(1 if self.state == "SESSION" else 30) & 0xFF
                self.handle_key(key)
                for x, y in self.taps:              # taps made on the screen just shown
                    self.tap(x, y)
                self.taps.clear()
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # python kiosk.py           -> fullscreen (the Pi touchscreen)
    # python kiosk.py --window  -> 800x480 window (testing on a laptop; click = tap)
    if "--window" in sys.argv:
        Kiosk((800, 480)).run(fullscreen=False)
    else:
        Kiosk(detect_screen_size()).run()
