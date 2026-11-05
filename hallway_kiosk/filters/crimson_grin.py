"""
Crimson Grin  -  Abhi's filter (kiosk version 2)

What it does: puts the grin over the mouth and keeps it steady when you move.

Why version 2: version 1 used only the 2 mouth corners. Those 2 points jump
around a little every frame (and squeeze together when you turn your head),
so the grin wobbled and changed size. YuNet gives us 5 points, so now we use
ALL of them:
  1. We say where the 5 points (eyes, nose, mouth corners) belong on the
     grin picture, using a normal face shape (FACE_SHAPE).
  2. cv2.estimateAffinePartial2D finds the ONE move + turn + resize that
     lines those 5 points up best with the 5 points on the real face.
     5 points averaged together wobble much less than 2.
     A point that is way off (bad detection) gets ignored.
  3. Then we slide the grin so its middle sits exactly on the real mouth.
  4. If the head is turned too far sideways we hide the grin, because the
     far mouth corner is hidden and the points are not trustworthy.
"""
import math

import cv2
import numpy as np

NAME = "Crimson Grin"
OWNER = "Abhi"
PNG = "smile_filter.png"

SMILE_SCALE = 1.7   # grin width = mouth width * this (same as Abhi's original)
MOUTH_Y = 0.50      # where the lips meet in the PNG (0 = top, 1 = bottom)
MAX_TURN = 0.30     # hide if the nose is this close to one eye (0.5 = looking straight)

# A normal face's 5 points (the standard face-alignment template, 112x112 px)
FACE_SHAPE = np.float32([
    [38.3, 51.7],   # left eye
    [73.5, 51.5],   # right eye
    [56.0, 71.7],   # nose
    [41.5, 92.4],   # mouth left
    [70.7, 92.2],   # mouth right
])


def prepare(image):
    """Shrink the PNG once so the Pi stays fast."""
    if image.shape[1] > 600:
        new_h = int(image.shape[0] * 600 / image.shape[1])
        image = cv2.resize(image, (600, new_h), interpolation=cv2.INTER_AREA)
    return image


def points_on_png(image):
    """Where the 5 face points belong on the grin picture."""
    h, w = image.shape[:2]
    mouth_mid = (FACE_SHAPE[3] + FACE_SHAPE[4]) / 2
    mouth_w = np.linalg.norm(FACE_SHAPE[4] - FACE_SHAPE[3])
    scale = (w / SMILE_SCALE) / mouth_w            # so the mouth is 1/1.7 of the PNG width
    return (FACE_SHAPE - mouth_mid) * scale + np.float32([w / 2, h * MOUTH_Y])


def draw(frame, face, smile_image):
    (lx, ly), (rx, ry) = face["left_eye"], face["right_eye"]
    eye_dist = math.hypot(rx - lx, ry - ly)

    # 4. Head turned too far? (nose almost in line with one eye)
    nose_pos = (face["nose"][0] - lx) / max(1.0, rx - lx)
    if not (MAX_TURN < nose_pos < 1 - MAX_TURN):
        return frame

    # 1 + 2. Best move/turn/resize from the PNG's 5 points to the face's 5 points
    face_pts = np.float32([face["left_eye"], face["right_eye"], face["nose"],
                           face["mouth_left"], face["mouth_right"]])
    png_pts = points_on_png(smile_image)
    M, _ = cv2.estimateAffinePartial2D(png_pts, face_pts, method=cv2.RANSAC,
                                       ransacReprojThreshold=0.25 * eye_dist)
    if M is None:
        return frame

    # 3. Slide it so the grin's middle lands exactly on the real mouth
    png_mouth = (png_pts[3] + png_pts[4]) / 2
    landed = M[:, :2] @ png_mouth + M[:, 2]
    real_mouth = (face_pts[3] + face_pts[4]) / 2
    M[:, 2] += real_mouth - landed

    # Warp the PNG onto the frame and blend it using its transparency
    h, w = frame.shape[:2]
    warped = cv2.warpAffine(smile_image, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    return (frame * (1 - alpha) + warped[:, :, :3] * alpha).astype(np.uint8)