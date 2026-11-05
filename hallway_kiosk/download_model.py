"""Run this ONCE (with internet) to get the two face model files
(YuNet for everyone, MediaPipe face mesh for the Crimson Grin).
It only downloads model files. The kiosk itself never uses the network."""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "face_detection_yunet_2023mar.onnx")
URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
       "face_detection_yunet/face_detection_yunet_2023mar.onnx")

if os.path.exists(PATH) and os.path.getsize(PATH) > 100_000:
    print("Model is already here:", PATH)
else:
    print("Downloading YuNet face model...")
    urllib.request.urlretrieve(URL, PATH)
    print("Done:", PATH)

MESH_PATH = os.path.join(HERE, "face_landmarker.task")
MESH_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")
if os.path.exists(MESH_PATH) and os.path.getsize(MESH_PATH) > 1_000_000:
    print("Model is already here:", MESH_PATH)
else:
    print("Downloading MediaPipe face mesh model...")
    urllib.request.urlretrieve(MESH_URL, MESH_PATH)
    print("Done:", MESH_PATH)