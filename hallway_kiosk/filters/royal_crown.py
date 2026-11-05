"""
Royal Crown  -  Ryan's filter (kiosk version)

What it does: the crown's bottom-center sits near the top of the face box
(the hairline), 1.18x as wide as the face, tilted with the line between the eyes.

What changed for the kiosk:
  * Ryan's original found faces with a Haar cascade and eyes with a SECOND
    Haar cascade. The eye cascade often missed, so the crown snapped back to
    straight. The kiosk's shared YuNet detector gives a face box AND both eyes
    every frame, so the crown now tilts smoothly every frame.
  * YuNet's face box starts a little lower on the forehead than Haar's, so
    CROWN_BOTTOM_FACE_RATIO was moved from 0.16 to 0.0 (tune it at the spot).
  overlay_crown() and crop_transparent_edges() are Ryan's code.
"""
import math

import cv2
import numpy as np

NAME = "Royal Crown"
OWNER = "Ryan"
PNG = "Blue_Crown_Transparent.png"

CROWN_WIDTH_FACE_RATIO = 1.18
CROWN_BOTTOM_FACE_RATIO = 0.0
CROWN_OPACITY = 1.0


def prepare(image):
    return crop_transparent_edges(image)


def draw(frame, face, crown):
    return overlay_crown(frame, crown, face["box"], face["left_eye"], face["right_eye"])


def crop_transparent_edges(image):
    alpha = image[:, :, 3]
    points = cv2.findNonZero((alpha > 5).astype(np.uint8))
    if points is None:
        return image
    x, y, w, h = cv2.boundingRect(points)
    return image[y:y + h, x:x + w]


def overlay_crown(frame, crown, face_box, left_eye=None, right_eye=None):
    """Anchor the crown's bottom-center at the hairline and alpha-blend it."""
    face_x, face_y, face_w, face_h = face_box
    angle_deg = 0.0
    if left_eye is not None and right_eye is not None:
        angle_deg = math.degrees(math.atan2(
            right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        ))

    target_x = face_x + face_w / 2.0
    target_y = face_y + face_h * CROWN_BOTTOM_FACE_RATIO
    crown_h, crown_w = crown.shape[:2]
    anchor = (crown_w / 2.0, crown_h - 1.0)
    scale = (face_w * CROWN_WIDTH_FACE_RATIO) / crown_w

    transform = cv2.getRotationMatrix2D(anchor, -angle_deg, scale)
    transform[0, 2] += target_x - anchor[0]
    transform[1, 2] += target_y - anchor[1]

    frame_h, frame_w = frame.shape[:2]
    warped = cv2.warpAffine(
        crown, transform, (frame_w, frame_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    alpha *= CROWN_OPACITY
    result = frame.astype(np.float32) * (1.0 - alpha)
    result += warped[:, :, :3].astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)
