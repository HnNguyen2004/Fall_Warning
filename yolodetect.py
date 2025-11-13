from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
import numpy as np
from telegram_ultil import send_telegram
import datetime
import threading
from ultralytics import YOLO


def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    print(polygon.contains(centroid))
    return polygon.contains(centroid)


class YoloDetect():
    def __init__(self, detect_class="person", frame_width=1280, frame_height=720):
        # Parameters
        self.model_path = "yolov8n.pt"  # YOLOv8 nano model, bạn có thể thay bằng yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
        self.conf_threshold = 0.5
        self.detect_class = detect_class
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.model = YOLO(self.model_path)
        self.last_alert = None
        self.alert_telegram_each = 15  # seconds

    def draw_prediction(self, img, class_name, x, y, x_plus_w, y_plus_h, confidence, points):
        label = f"{class_name} {confidence:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Tinh toan centroid
        centroid = ((x + x_plus_w) // 2, (y + y_plus_h) // 2)
        cv2.circle(img, centroid, 5, (color), -1)

        if isInside(points, centroid):
            img = self.alert(img)

        return isInside(points, centroid)

    def alert(self, img):
        cv2.putText(img, "ALARM!!!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # New thread to send telegram after 15 seconds
        if (self.last_alert is None) or (
                (datetime.datetime.utcnow() - self.last_alert).total_seconds() > self.alert_telegram_each):
            self.last_alert = datetime.datetime.utcnow()
            cv2.imwrite("alert.png", cv2.resize(img, dsize=None, fx=0.2, fy=0.2))
            thread = threading.Thread(target=self._run_async_telegram)
            thread.start()
        return img
    
    def _run_async_telegram(self):
        import asyncio
        asyncio.run(send_telegram())

    def detect(self, frame, points):
        # Chạy YOLOv8 inference
        results = self.model(frame, conf=self.conf_threshold)
        
        # Lặp qua các kết quả detection
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Lấy thông tin bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Chỉ detect class mong muốn
                    if class_name == self.detect_class and confidence >= self.conf_threshold:
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        self.draw_prediction(frame, class_name, x, y, x + w, y + h, confidence, points)

        return frame
