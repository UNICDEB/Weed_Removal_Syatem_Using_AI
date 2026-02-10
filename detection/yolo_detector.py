from ultralytics import YOLO
import cv2

class YOLODetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image, conf=0.5):

        results = self.model(image, conf=conf)[0]

        boxes = []
        annotated = image.copy()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            boxes.append([x1, y1, x2, y2, confidence, cls])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)

        return annotated, boxes
