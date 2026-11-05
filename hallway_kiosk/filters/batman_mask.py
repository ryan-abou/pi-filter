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
  4. One warp moves + turns + resizes the PNG so the holes land on your
     eyes, then we blend it in using the PNG's transparency.
"""
import math

import cv2
import numpy as np

NAME = "Batman Mask"
OWNER = "Abhi"
PNG = "batman_mask.png"

EYE_L_PNG = (109, 335)   # center of the left eye hole in the PNG (pixels)
EYE_R_PNG = (303, 335)   # center of the right eye hole in the PNG
SIZE = 1.0               # 1.0 = holes exactly on your eyes; bigger = bigger mask
MAX_TILT = 40            # hide the mask if the head is tilted more than this (degrees)


def prepare(image):
    return image


def draw(frame, face, mask):
    (lx, ly), (rx, ry) = face["left_eye"], face["right_eye"]
    eye_dist = math.hypot(rx - lx, ry - ly)
    angle = math.degrees(math.atan2(ry - ly, rx - lx))       # head tilt
    if abs(angle) > MAX_TILT:
        return frame

    # Middle of the eye holes in the PNG, and middle of the real eyes
    hole_mid = ((EYE_L_PNG[0] + EYE_R_PNG[0]) / 2, (EYE_L_PNG[1] + EYE_R_PNG[1]) / 2)
    hole_dist = math.hypot(EYE_R_PNG[0] - EYE_L_PNG[0], EYE_R_PNG[1] - EYE_L_PNG[1])
    eye_mid = ((lx + rx) / 2, (ly + ry) / 2)

    # Turn + resize around the hole middle, then slide it onto the eye middle
    M = cv2.getRotationMatrix2D(hole_mid, -angle, SIZE * eye_dist / hole_dist)
    M[0, 2] += eye_mid[0] - hole_mid[0]
    M[1, 2] += eye_mid[1] - hole_mid[1]

    # Warp the PNG onto the frame and blend it using its transparency
    h, w = frame.shape[:2]
    warped = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    return (frame * (1 - alpha) + warped[:, :, :3] * alpha).astype(np.uint8)