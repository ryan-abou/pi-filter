"""Run this ONCE (with internet) if face_detection_yunet_2023mar.onnx is missing.
It only downloads the face model file. The kiosk itself never uses the network."""
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
