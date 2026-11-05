"""
Robot Circuits  -  Krishna's filter (kiosk version)

What it does: anchors the circuit PNG on the NOSE, scales it by the distance
between the eyes (200 px in the PNG = 1 eye distance) and tilts it with the
eye line. Hidden if the head tilts past MAX_TILT or the nose is not below the
eyes (a bad detection).

What changed for the kiosk:
  * Krishna's smoothing (SMOOTH) and jump check (JUMP_LIMIT) moved into the
    kiosk's face tracker so every filter gets smooth, steady points.
  * His "one person at a time" check is replaced by the kiosk's opt-in zone,
    so friends inside the zone can join.
  overlay_circuits() and the tilt / nose-below checks are Krishna's code.
"""
import math

import cv2
import numpy as np

NAME = "Robot Circuits"
OWNER = "Krishna"
PNG = "robot_circuits.png"

PNG_D = 200      # in the PNG, 1 eye-distance = 200 px
MAX_TILT = 40    # hide circuits if head is tilted more than this many degrees


def prepare(image):
    return image


def draw(frame, face, img):
    (lx, ly), (rx, ry) = face["left_eye"], face["right_eye"]
    nose = face["nose"]
    d = math.hypot(rx - lx, ry - ly)
    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    a = math.radians(angle)                     # nose must sit below the eyes
    below = (nose[0] - (lx + rx) / 2) * -math.sin(a) + (nose[1] - (ly + ry) / 2) * math.cos(a)
    if abs(angle) < MAX_TILT and below > 0.1 * d:
        frame = overlay_circuits(frame, img, nose, d, angle)
    return frame


def overlay_circuits(frame, img, nose, d, angle):
    """Scale + rotate the circuits around the nose, then alpha blend."""
    h, w = img.shape[:2]
    center = (w / 2, h * 150 / 260)                          # nose point inside the PNG
    M = cv2.getRotationMatrix2D(center, -angle, d / PNG_D)
    M[0, 2] += nose[0] - center[0]
    M[1, 2] += nose[1] - center[1]
    warped = cv2.warpAffine(img, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    return (frame * (1 - alpha) + warped[:, :, :3] * alpha).astype(np.uint8)
