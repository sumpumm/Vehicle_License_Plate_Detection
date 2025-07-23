from ultralytics import YOLO
import cv2
from utils import *

class Video:
    def __init__(self, path, model_path):
        self.path = path
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.capture = cv2.VideoCapture(path)
        self.results = None

    def process_video(self):
        self.results = self.model.track(
            source=self.path,
            stream=True,
            save=True,
            conf=0.6,
            project="outputs",
            name="track",
            exist_ok=True
        )
        return self.results

    def get_frame(self, frame_number):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.capture.read()
        if not ret:
            return None
        return frame

    def get_plate_coordinates(self):
        return get_xyxy(self.results)