from ultralytics import YOLO
import supervision as sv

class CrowdDetector:
    def __init__(self, model_path='yolov8n.pt', conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf
        self.tracker = sv.ByteTrack()

    def get_detections(self, frame):
     
        results = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
       
        detections = self.tracker.update_with_detections(detections)
        return detections