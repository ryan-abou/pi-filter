# Hallway Filter Kiosk - Prototype 1

One touchscreen app with all four group filters:

| Tile | Filter | Owner | File |
|---|---|---|---|
| 1 | Crimson Grin | Abhi | `filters/crimson_grin.py` |
| 2 | Royal Crown | Ryan | `filters/royal_crown.py` |
| 3 | Robot Circuits | Krishna | `filters/robot_circuits.py` |
| 4 | Gold Crown | Pavan | `filters/gold_crown.py` |

Flow (the same for every filter): **Menu -> Terms (tick box) -> START -> 30 s session -> camera off -> Menu**.
During a session: **SWITCH FILTER** (pick another one, timer restarts at 0:30) or **STOP**.

## 0. Run it on a Windows laptop (for testing)

1. Install Python 3.10+ from python.org (tick **"Add Python to PATH"**).
2. Unzip `hallway_kiosk.zip`, open the `hallway_kiosk` folder, click the address bar, type `cmd`, press Enter.
3. In that window:

```bat
py -m pip install opencv-python numpy
py kiosk.py --window
```

- `--window` = normal 800x480 window; your mouse click works like a finger tap.
- Leave it off (`py kiosk.py`) for fullscreen. Press `q` to quit.
- Run the automatic tests: `py tests\test_kiosk.py`
- Your laptop camera is close to you, so you will always be "in the zone". The zone only
  matters at the real hallway spot after calibration (section 3).
- If the camera doesn't open: close Zoom/Teams/Camera app, and check
  Settings > Privacy > Camera > "Let desktop apps access your camera".

## 1. Set up the Pi (once)

The face model needs **OpenCV 4.8 or newer**. The `apt` version on Pi OS is too old, so use a venv:

```bash
cd ~/hallway_kiosk
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy
python3 download_model.py      # only if face_detection_yunet_2023mar.onnx is missing
python3 kiosk.py              # fullscreen on the touchscreen
```

Next time: `cd ~/hallway_kiosk && source venv/bin/activate && python3 kiosk.py`

Keys (plug in a keyboard): `q` / `Esc` = quit, `d` = calibration view (boxes only, never video).

## 2. Physical setup (this is what makes several criteria pass)

- Put the kiosk at the **side** of the hallway, not the middle (Abhi story 1: people walk past without stopping).
- **Tape a zone on the floor from about 1 ft to 6 ft** in front of the camera, about 3 ft wide.
  It must be deep enough for a wheelchair (about 3.5 ft long) - Ryan story 2.
- Camera at about **4 ft high**, tilted slightly down, so a seated person's face is in the picture.
- The START / STOP buttons are at the bottom of the screen. Mount the screen so they are **no higher than 4 ft**.
- Put the printed privacy sign next to the zone.

## 3. Calibrate the zone (at the real spot, every time the kiosk moves)

1. Start a session, press `d`.
2. Stand on the **back** tape line (6 ft). Read the `w=..px` number above your face.
3. In `kiosk.py`, set `ZONE_FAR_FACE_W` a little **below** that number (e.g. you read 46 -> use 44).
4. Stand at 7-8 ft: your box must turn **red** (not shown).
5. Stand on the left and right tape edges. If the box doesn't turn red just outside the tape,
   change `ZONE_CENTER_X` (middle of the zone) or `ZONE_HALF_WIDTH_FACES` (width).

## 4. How the code is organized

- `kiosk.py`
  - `begin_session()` is the **only** place the camera opens; `end_session()` releases it.
  - `FaceTracker` decides who is shown:
    - a new face must be inside the zone and facing the camera for 0.3 s
    - a face that is already shown keeps its filter while turning the head
    - leaving the zone hides it right away
    - a missed detection is held for 0.5 s
  - `reveal_faces()` keeps the screen black except a soft oval around each shown face.
  - `draw_menu / draw_consent / draw_session / draw_done` draw the screens.
- `filters/*.py` - each has `NAME`, `OWNER`, `PNG`, `prepare()`, `draw()`. The docstring at the top says what changed from the original.
- `tests/test_kiosk.py` - automatic logic tests (fake camera + clock). Run: `python3 tests/test_kiosk.py`

## 5. Hallway test checklist (do these for real before ticking a box on GitHub)

Take a note or a phone video of every test (date, who, distance, pass/fail) and put it in the issue comment.

| Story | Criterion | How to test |
|---|---|---|
| Krishna 1 | No filter on someone outside the zone >6 ft | Session running, one person at 7-8 ft facing camera for 10 s |
| Krishna 1 | No filter on someone in zone with head turned 90 deg | Step in with head already turned, hold 5 s |
| Krishna 1 | Student 8 ft behind never gets filter | User in zone + second person at 8 ft for the whole 30 s |
| Krishna 2 | Filter within 1 s of stepping into zone | Session running, step in, time with phone video |
| Krishna 2 | Stays aligned walking through zone with backpack | Walk slowly across the zone |
| Krishna 2 | Circuits cover forehead to chin at 6 ft | **Needs a taller PNG** - see note |
| Ryan 1 | Filter within 2 s of tapping Start | Stand in zone, tap START, time it |
| Ryan 1 | Stays on during head turn / one step, never jumps | Turn head, take one step; have a friend at 8 ft |
| Ryan 2 | Face detected when seated | Sit in a chair in the zone |
| Ryan 2 | Start no higher than 4 ft | Tape measure to the START button |
| Ryan 2 | Wheelchair fits in zone | Measure the taped zone (>= 3.5 ft deep) |
| Abhi 1 | Filter within 2 s of Start within 5 ft | Stand at 5 ft, tap START, time it |
| Abhi 1 | 5 people pass without stopping | 5 people walk past the kiosk |
| Abhi 1 | Filter gone within 2 s of stepping out | Step out sideways, time it |
| Abhi 2 | Face not shown when >= 5 ft away and outside zone | Stand outside tape, check screen is black there |
| Abhi 2 | No filter outside zone for 5 s | Stand just outside the tape 5 s |
| Abhi 2 | Face gone within 2 s of stepping out | Same as Abhi 1 |
| Pavan 1 | No crown when walking past without tapping | Walk past, camera is off (menu shows CAMERA IS OFF) |
| Pavan 1 | No crown on someone walking behind | Session running, friend walks behind at 8 ft |
| Pavan 2 | Tap -> crown appears | Tap Gold Crown, tick, START |
| Pavan 2 | Crown turns off after 30 s | Stopwatch from START to "Session over" |
| Pavan 2 | Stays off when nobody taps | Leave kiosk on menu, walk in front |

Switching story (new - add to GitHub): new filter on face within 2 s of tapping it; timer shows 0:30 after switching;
tapping the filter already on, or CLOSE, changes nothing.
