"""
Automated tests for the hallway kiosk (no camera or screen needed).
Run from anywhere:  python3 tests/test_kiosk.py

These do NOT replace the real hallway tests for the acceptance criteria
(distances in feet, real people). They check the LOGIC: the camera only
opens after consent, the timer, switching, and who the tracker shows.

The flow test uses a fake camera that returns tests/pair.png (two news
photos side by side) so the real YuNet detector runs on real faces.
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.dirname(HERE)
sys.path.insert(0, KDIR)
os.chdir(KDIR)

import cv2
import numpy as np

import kiosk as K

ap = argparse.ArgumentParser()
ap.add_argument("--screens", default=None, help="folder to save screenshots")
ap.add_argument("--frame", default=os.path.join(HERE, "pair.png"))
args = ap.parse_args()

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS  " if ok else "FAIL  ") + name + (f"   ({detail})" if detail else ""))


def save(name, img):
    if args.screens:
        os.makedirs(args.screens, exist_ok=True)
        cv2.imencode(".png", img)[1].tofile(os.path.join(args.screens, name))   # test-only file output


# ------------------------------------------------------------ fakes
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def wait(self, s):
        self.t += s


class FakeCam:
    def __init__(self, frame):
        self.frame, self.released, self.reads = frame, False, 0

    def read(self):
        self.reads += 1
        return (not self.released), self.frame.copy()

    def release(self):
        self.released = True


# ------------------------------------------------------------ 1. privacy code check
bad = []
for f in glob.glob(KDIR + "/*.py") + glob.glob(KDIR + "/filters/*.py"):
    if f.endswith("download_model.py"):
        continue
    src = open(f).read()
    for pat in ["imwrite(", "VideoWriter(", "urllib", "socket", "requests", "imencode("]:
        if pat in src:
            bad.append(f"{os.path.basename(f)}: {pat}")
check("App code never saves frames or uses the network", not bad, ", ".join(bad))

# ------------------------------------------------------------ 2. full flow with the real detector
frame = cv2.imread(args.frame)
clock = Clock()
cams = []


def factory():
    cams.append(FakeCam(frame))
    return cams[-1]


k = K.Kiosk((800, 480), camera_factory=factory, clock=clock)


def press(action):
    """Render the screen, find the button for this action, tap its center."""
    k.render()
    for x1, y1, x2, y2, act in k.buttons:
        if act == action:
            clock.wait(0.3)
            k.tap((x1 + x2) // 2, (y1 + y2) // 2)
            return True
    raise AssertionError(f"no button {action} on {k.state}")


def run_frames(seconds, fps=15):
    for _ in range(int(seconds * fps)):
        clock.wait(1 / fps)
        if k.state == "SESSION":
            k.process_frame()
        k.tick()


save("01_menu.png", k.render())
check("Menu: camera is off", not cams)
press(("pick", 3))
check("Tap tile 4 -> consent screen for Gold Crown", k.state == "CONSENT" and k.choice == 3)
press(("start",))
check("START without ticking does nothing and camera stays off", k.state == "CONSENT" and not cams)
save("02_consent_not_ticked.png", k.render())
press(("agree",))
save("03_consent_ticked.png", k.render())
press(("start",))
check("Tick + START -> starting", k.state == "STARTING")
save("04_starting.png", k.render())
k.begin_session()
check("Camera opens only now", len(cams) == 1 and k.state == "SESSION")
start = clock()
run_frames(0.1)
save("05_session_first_frame.png", k.render())
first_seen = None
for _ in range(30):
    run_frames(1 / 15)
    if k.faces_shown == 2 and first_seen is None:
        first_seen = clock() - start
check("Both people in the zone get the filter within 1 s", first_seen is not None and first_seen <= 1.0,
      f"{first_seen:.2f}s" if first_seen else "never")
save("06_session_gold_crown.png", k.render())

for i in range(4):
    if i == k.choice:
        continue
    press(("switch_panel",))
    if i == 0:
        save("07_switch_panel.png", k.render())
    clock.wait(5)                     # pretend 5 s of the session passed
    press(("switch", i))
    left = k.session_end - clock()
    check(f"Switch to {K.FILTERS[i].NAME}: timer back to {K.SESSION_SECONDS}s, panel closed",
          k.choice == i and abs(left - K.SESSION_SECONDS) < 0.01 and not k.show_switch, f"{left:.1f}s left")
    run_frames(0.5)
    check(f"{K.FILTERS[i].NAME} still shown on both faces after switch", k.faces_shown == 2)
    save(f"08_session_{K.FILTERS[i].NAME.lower().replace(' ', '_')}.png", k.render())

press(("switch_panel",))
before = k.session_end
clock.wait(2)
press(("switch", k.choice))
check("Tapping the filter already on changes nothing", k.session_end == before and not k.show_switch)

left = k.session_end - clock()
run_frames(left - 0.2)
still_on = k.state == "SESSION"
run_frames(0.4)
check(f"Session ends by itself when the timer hits 0:00 and camera is released",
      still_on and k.state == "DONE" and cams[0].released and k.cap is None)
save("09_done.png", k.render())
run_frames(K.DONE_SECONDS + 0.5)
check("Done screen returns to menu by itself", k.state == "MENU")

# STOP button
press(("pick", 1)); press(("agree",)); press(("start",)); k.begin_session()
run_frames(1)
press(("stop",))
check("STOP ends early and releases the camera", k.state == "DONE" and cams[-1].released)
clock.wait(4); k.tick()

# consent timeout
press(("pick", 0))
clock.wait(K.CONSENT_TIMEOUT + 1); k.tick()
check("Consent screen goes back to menu if nobody taps", k.state == "MENU")
check("Camera was opened exactly twice (once per START)", len(cams) == 2)

# camera failure
k2 = K.Kiosk((800, 480), camera_factory=lambda: None, clock=clock)
k2.choice, k2.agreed = 0, True
k2.begin_session()
check("No camera -> friendly message, no crash", k2.state == "DONE" and k2.message)

# ------------------------------------------------------------ 3. layouts at other screen sizes
for size in [(800, 480), (1024, 600), (1280, 720), (1920, 1080)]:
    try:
        kk = K.Kiosk(size, camera_factory=factory, clock=clock)
        for st in ["MENU", "CONSENT", "STARTING", "DONE"]:
            kk.state = st
            kk.render()
        kk.state, kk.last_output, kk.session_end = "SESSION", np.zeros((480, 640, 3), np.uint8), clock() + 30
        kk.render(); kk.show_switch = True; kk.render()
        check(f"All screens draw at {size[0]}x{size[1]}", True)
    except Exception as e:
        check(f"All screens draw at {size[0]}x{size[1]}", False, repr(e))


# ------------------------------------------------------------ 4. tracker rules (fake detections)
def det(cx, cy=200, w=100, nose=0.5):
    """One fake YuNet row. nose = where the nose is between the eyes (0.5 = straight)."""
    lx, rx = cx - w * 0.2, cx + w * 0.2
    return [cx - w / 2, cy - w * 0.6, w, w * 1.2, rx, cy - w * 0.1, lx, cy - w * 0.1,
            lx + nose * (rx - lx), cy + w * 0.1, rx - w * 0.05, cy + w * 0.3, lx + w * 0.05, cy + w * 0.3, 0.9]


def feed(tr, rows_fn, seconds, t0, fps=15):
    t = t0
    for _ in range(int(round(seconds * fps))):
        t += 1 / fps
        rows = rows_fn(t)
        tr.update(np.array(rows, np.float32) if rows else None, t)
    return t


C = K.CAM_W / 2
tr = K.FaceTracker(); t = feed(tr, lambda t: [det(C)], 0.2, 0)
a = len(tr.shown_faces()); t = feed(tr, lambda t: [det(C)], 0.2, t)
check("New face waits ~0.3 s before it is shown (passer-by glance)", a == 0 and len(tr.shown_faces()) == 1)

tr = K.FaceTracker(); feed(tr, lambda t: [det(C, w=K.ZONE_FAR_FACE_W - 8)], 3, 0)
check("Face farther than 6 ft (too small) is never shown", len(tr.shown_faces()) == 0)

tr = K.FaceTracker(); feed(tr, lambda t: [det(C, nose=0.05)], 3, 0)
check("New face in zone with head turned away is never shown", len(tr.shown_faces()) == 0)

tr = K.FaceTracker(); t = feed(tr, lambda t: [det(C)], 1, 0); t = feed(tr, lambda t: [det(C, nose=0.05)], 1, t)
check("Face already shown keeps the filter while turning head", len(tr.shown_faces()) == 1)

tr = K.FaceTracker(); t = feed(tr, lambda t: [det(C)], 1, 0)
edge = C + K.ZONE_HALF_WIDTH_FACES * 100
hidden_at = None
for n in range(60):                         # walk sideways ~1.3 m/s at 15 fps
    x = C + n * 25
    t += 1 / 15
    tr.update(np.array([det(x)], np.float32), t)
    if x > edge and hidden_at is None and len(tr.shown_faces()) == 0:
        hidden_at = n
    if x > edge + 150:
        break
first_out = next(n for n in range(60) if C + n * 25 > edge)
check("Walking out of the zone hides the face quickly", hidden_at is not None and (hidden_at - first_out) / 15 < 0.5,
      f"{(hidden_at - first_out) / 15:.2f}s after crossing" if hidden_at is not None else "never hidden")

tr = K.FaceTracker(); t = feed(tr, lambda t: [det(C)], 1, 0)
t = feed(tr, lambda t: [], 0.4, t); held = len(tr.shown_faces())
t = feed(tr, lambda t: [], 0.2, t)
check("Missed detection held 0.5 s, then removed", held == 1 and len(tr.shown_faces()) == 0)

tr = K.FaceTracker()
feed(tr, lambda t: [det(C - 120), det(C + 120), det(C + 40, cy=150, w=K.ZONE_FAR_FACE_W - 10)], 2, 0)
check("Two friends in the zone shown, person far behind hidden", len(tr.shown_faces()) == 2)

tr = K.FaceTracker(); t = feed(tr, lambda t: [det(C - 120), det(C + 120)], 1, 0)
t = feed(tr, lambda t: [det(C - 120)], 1, t)
t = feed(tr, lambda t: [det(C - 120), det(C + 120)], 0.15, t)
ids = len(tr.shown_faces())
check("Friend who leaves and comes back has to settle again (filter never jumps to them)", ids == 1)


passed = sum(ok for _, ok in results)
print(f"\n{passed}/{len(results)} checks passed")
