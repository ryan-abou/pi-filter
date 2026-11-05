"""
Crimson Grin  -  Abhi's filter (kiosk version)

What it does: puts the grin over the mouth, centered between the two mouth
corners, 1.7x as wide as the mouth, tilted to match the line between the eyes.

What changed for the kiosk:
  * Abhi's original used MediaPipe landmarks 61/291 (mouth corners) and
    33/263 (eyes). The kiosk shares ONE detector (YuNet) for all four filters,
    so this file now gets YuNet's mouth corners and eyes instead.
  * His "one face only / inside the zone / black screen with an oval" privacy
    code moved into kiosk.py so it protects EVERY filter, not just this one.
  * The big PNG (1774 px wide) is shrunk once at start-up so the Pi stays fast.
  apply_smile(), rotate_image() and overlay_image() are Abhi's code.
"""
import math

import cv2
import numpy as np

NAME = "Crimson Grin"
OWNER = "Abhi"
PNG = "smile_filter.png"

SMILE_SCALE = 1.7        # grin width = mouth width * this


def prepare(image):
    """Shrink the PNG once (it is resized again every frame anyway)."""
    if image.shape[1] > 600:
        new_h = int(image.shape[0] * 600 / image.shape[1])
        image = cv2.resize(image, (600, new_h), interpolation=cv2.INTER_AREA)
    return image


def draw(frame, face, smile_image):
    apply_smile(frame, face, smile_image)
    return frame


def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]
    return cv2.warpAffine(image, matrix, (new_width, new_height),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))


def overlay_image(background, overlay, center_x, center_y):
    overlay_height, overlay_width = overlay.shape[:2]
    x1 = int(center_x - overlay_width / 2)
    y1 = int(center_y - overlay_height / 2)
    x2 = x1 + overlay_width
    y2 = y1 + overlay_height
    bg_height, bg_width = background.shape[:2]
    if x2 <= 0 or y2 <= 0 or x1 >= bg_width or y1 >= bg_height:
        return
    clipped_x1, clipped_y1 = max(0, x1), max(0, y1)
    clipped_x2, clipped_y2 = min(bg_width, x2), min(bg_height, y2)
    if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
        return
    cropped_overlay = overlay[clipped_y1 - y1:clipped_y2 - y1,
                              clipped_x1 - x1:clipped_x2 - x1]
    alpha = cropped_overlay[:, :, 3:4].astype(np.float32) / 255.0
    foreground = cropped_overlay[:, :, :3]
    background_region = background[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
    blended = foreground * alpha + background_region * (1 - alpha)
    background_region[:] = blended.astype(np.uint8)


def apply_smile(frame, face, smile_image):
    # YuNet points (the kiosk already sorted them so "left" = smaller x on screen)
    mouth_left = np.array(face["mouth_left"], dtype=np.float32)
    mouth_right = np.array(face["mouth_right"], dtype=np.float32)
    left_eye = np.array(face["left_eye"], dtype=np.float32)
    right_eye = np.array(face["right_eye"], dtype=np.float32)

    mouth_center = (mouth_left + mouth_right) / 2
    mouth_width = np.linalg.norm(mouth_right - mouth_left)
    target_width = max(1, int(mouth_width * SMILE_SCALE))
    original_height, original_width = smile_image.shape[:2]
    target_height = max(1, int(target_width * original_height / original_width))
    resized_smile = cv2.resize(smile_image, (target_width, target_height),
                               interpolation=cv2.INTER_LINEAR)

    eye_difference = right_eye - left_eye
    head_angle = math.degrees(math.atan2(eye_difference[1], eye_difference[0]))
    rotated_smile = rotate_image(resized_smile, -head_angle)
    overlay_image(frame, rotated_smile, mouth_center[0], mouth_center[1])
