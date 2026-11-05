"""
Gold Crown  -  Pavan's filter (kiosk version)

What it does: puts the crown's bottom-center UP_OFFSET eye-distances above the
point between the eyes. "Up" turns with the head, so the crown stays on top
of the head when you tilt. Width = eye distance * WIDTH_SCALE. Hidden past
MAX_TILT degrees.

What changed for the kiosk (so it does what the user stories say):
  * Tap the screen -> crown for 30 seconds -> turns off: the kiosk's timer
    and camera on/off now handle this (SESSION_SECONDS in kiosk.py).
  * "One person at a time" is replaced by the kiosk's opt-in zone, so
    friends standing in the zone each get a crown, but people walking
    behind (outside the zone) never do.
  * smooth() and JUMP_LIMIT moved into the kiosk's face tracker, which keeps
    a separate smoothed position for EACH face.
  overlay_crown() is Pavan's code.
"""
import math

import cv2
import numpy as np

NAME = "Gold Crown"
OWNER = "Pavan"
PNG = "crown.png"

WIDTH_SCALE = 1.8   # crown width = eye distance * this
UP_OFFSET = 1.5     # crown base sits this many eye-distances above the eye midpoint
MAX_TILT = 40       # hide crown if head is tilted more than this many degrees


def prepare(image):
    return image


def draw(frame, face, crown):
    (lx, ly), (rx, ry) = face["left_eye"], face["right_eye"]
    mid = ((lx + rx) / 2, (ly + ry) / 2)
    d = math.hypot(rx - lx, ry - ly)
    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    if abs(angle) < MAX_TILT:
        frame = overlay_crown(frame, crown, mid, d, angle)
    return frame


def overlay_crown(frame, crown, mid, d, angle):
    """Scale + rotate the crown, put its bottom-center above the eyes, alpha blend."""
    c_h, c_w = crown.shape[:2]
    anchor = (c_w / 2, c_h)                                  # bottom-center of the PNG
    M = cv2.getRotationMatrix2D(anchor, -angle, d * WIDTH_SCALE / c_w)
    a = math.radians(angle)                                  # "up" rotates with the head
    M[0, 2] += mid[0] + math.sin(a) * UP_OFFSET * d - anchor[0]
    M[1, 2] += mid[1] - math.cos(a) * UP_OFFSET * d - anchor[1]
    warped = cv2.warpAffine(crown, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    return (frame * (1 - alpha) + warped[:, :, :3] * alpha).astype(np.uint8)
