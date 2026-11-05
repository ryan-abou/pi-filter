"""
Batman Mask  -  Abhi's filter

What it does: puts the mask's two eye holes right on top of your two eyes.
The mask grows/shrinks with the distance between your eyes and tilts
with the line between your eyes (same idea as the class glasses code).

How:
  1. EYE_L_PNG / EYE_R_PNG = where the eye holes are in batman_mask.png
     (measured once, in pixels).
  2. scale = (distance between your eyes) / (distance between the holes)
  3. angle = tilt of the line between your eyes
  4. One warp moves + turns + resizes the PNG so the holes land on your eyes.
  5. A soft shadow goes down first (makes the mask look 3D instead of
     "stuck on"), then the mask is blended in using the PNG's transparency.
  6. We only warp/blend the small box around the mask, not the whole
     screen, so it stays fast on the Pi even with 4 people.
"""
import math

import cv2
import numpy as np

NAME = "Batman Mask"
OWNER = "Abhi"
PNG = "batman_mask.png"

EYE_L_PNG = (60, 250)    # center of the left eye hole in the PNG (pixels)
EYE_R_PNG = (179, 251)   # center of the right eye hole in the PNG
SIZE = 1.0               # 1.0 = holes exactly on your eyes; bigger = bigger mask
MAX_TILT = 40            # hide the mask if the head is tilted more than this (degrees)
SHADOW = 0.45            # how dark the shadow is (0 = no shadow, 1 = black)
SHADOW_OFFSET = 0.04     # how far the shadow sits down/right (fraction of mask width)


def prepare(image):
    return image


def draw(frame, face, mask):
    (lx, ly), (rx, ry) = face["left_eye"], face["right_eye"]
    eye_dist = math.hypot(rx - lx, ry - ly)
    angle = math.degrees(math.atan2(ry - ly, rx - lx))       # head tilt
    if eye_dist < 5 or abs(angle) > MAX_TILT:
        return frame

    # Middle of the eye holes in the PNG, and middle of the real eyes
    hole_mid = ((EYE_L_PNG[0] + EYE_R_PNG[0]) / 2, (EYE_L_PNG[1] + EYE_R_PNG[1]) / 2)
    hole_dist = math.hypot(EYE_R_PNG[0] - EYE_L_PNG[0], EYE_R_PNG[1] - EYE_L_PNG[1])
    eye_mid = ((lx + rx) / 2, (ly + ry) / 2)
    scale = SIZE * eye_dist / hole_dist

    # Turn + resize around the hole middle, then slide it onto the eye middle
    M = cv2.getRotationMatrix2D(hole_mid, -angle, scale)
    M[0, 2] += eye_mid[0] - hole_mid[0]
    M[1, 2] += eye_mid[1] - hole_mid[1]

    # Where do the PNG's 4 corners land on screen? That box is all we touch.
    ph, pw = mask.shape[:2]
    corners = np.array([[0, 0, 1], [pw, 0, 1], [0, ph, 1], [pw, ph, 1]], np.float32) @ M.T
    pad = int(pw * scale * SHADOW_OFFSET) + 8
    fh, fw = frame.shape[:2]
    x0 = max(0, int(corners[:, 0].min()) - pad)
    y0 = max(0, int(corners[:, 1].min()) - pad)
    x1 = min(fw, int(corners[:, 0].max()) + pad)
    y1 = min(fh, int(corners[:, 1].max()) + pad)
    if x1 <= x0 or y1 <= y0:
        return frame                                          # mask is fully off screen

    # Warp the PNG into just that box (shift the matrix so the box starts at 0,0)
    M[0, 2] -= x0
    M[1, 2] -= y0
    warped = cv2.warpAffine(mask, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3].astype(np.float32) / 255.0

    # Soft shadow: the mask's shape, blurred and nudged down-right
    shift = int(pw * scale * SHADOW_OFFSET)
    blur = max(3, int(pw * scale * 0.05)) | 1                 # blur size must be odd
    shadow = cv2.GaussianBlur(alpha, (blur, blur), 0)
    shadow = np.roll(shadow, (shift, shift), axis=(0, 1)) * SHADOW

    out = frame.copy()
    roi = out[y0:y1, x0:x1].astype(np.float32)
    roi *= (1 - shadow)[:, :, None]                           # darken under the mask
    a = alpha[:, :, None]
    roi = roi * (1 - a) + warped[:, :, :3] * a                # then the mask on top
    out[y0:y1, x0:x1] = roi.astype(np.uint8)
    return out