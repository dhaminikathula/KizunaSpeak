import csv
from pathlib import Path
import cv2
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

LABEL = "HELLO"

MODEL_PATH = str(
    Path(__file__).parent.parent
    / "assets"
    / "hand_landmarker.task"
)

CSV_PATH = Path(__file__).parent.parent / "data" / "raw" / "hello.csv"

base_options = BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

header_written = CSV_PATH.exists()

with open(CSV_PATH, "a", newline="") as file:

    writer = csv.writer(file)

    while True:

        success, frame = cap.read()

        if not success:
            break

        cv2.imshow("HELLO Dataset Collection", frame)

        key = cv2.waitKey(1)

        if key == ord("s"):

            print("Sample Saved")

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()