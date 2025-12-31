from ultralytics import YOLO

class CarDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]
        return results
