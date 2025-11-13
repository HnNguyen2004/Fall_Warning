import cv2
import numpy as np
from imutils.video import VideoStream
from yolodetect import YoloDetect
import os
from datetime import datetime
import time

class IntrusionDetectionSystem:
    def __init__(self):
        self.video = None
        self.points = []  # khoi tai vung giam sat
        self.model = YoloDetect()  # goi mo hinh
        self.detect = False  # bien kiem soat tinh nang phat hien xam nhap
        
        # Khởi tạo camera
        self.initialize_camera()
        
        self.last_detection_time = 0
        self.detection_cooldown = 5  # 5 giây giữa các lần lưu
        
        # Đặt callback cho YoloDetect
        try:
            if hasattr(self.model, 'set_intrusion_callback'):
                self.model.set_intrusion_callback(self.on_intrusion_detected)
                print("✓ Intrusion callback set successfully")
            else:
                print("⚠️ YoloDetect doesn't support callback")
        except Exception as e:
            print(f"⚠️ Error setting callback: {e}")

    def initialize_camera(self):
        """Thử các camera index từ 0"""
        for src in [0, 1, 2]:
            try:
                print(f"Trying camera index {src}...")
                video = VideoStream(src=src).start()
                time.sleep(2)  # Đợi camera khởi động
                frame = video.read()
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    print(f"✓ Camera {src} works!")
                    self.video = video
                    return
                else:
                    video.stop()
            except Exception as e:
                print(f"✗ Camera {src} failed: {e}")
        
        print("❌ Không tìm thấy camera nào hoạt động!")
        exit(1)

    def handle_left_click(self, event, x, y, flags, points):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])

    def draw_polygon(self, frame, points):
        for point in points:
            frame = cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)
        if len(points) > 1:
            frame = cv2.polylines(frame, [np.int32(points)], False, (255, 0, 0), thickness=2)
        return frame

    def save_intrusion_image(self, frame, person_name="Unknown"):
        """Lưu ảnh người xâm nhập"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intrusion_{person_name}_{timestamp}.jpg"
        
        os.makedirs("intrusion_images", exist_ok=True)
        image_path = os.path.join("intrusion_images", filename)
        cv2.imwrite(image_path, frame)
        
        print(f"💾 Saved intrusion image: {image_path}")
        return image_path

    def on_intrusion_detected(self, frame, person_name="Person_Detected", confidence=0.8):
        """Callback được gọi khi YoloDetect phát hiện xâm nhập"""
        current_time = datetime.now().timestamp()
        
        # Kiểm tra cooldown để tránh spam
        if (current_time - self.last_detection_time) > self.detection_cooldown:
            print(f"🚨 Intrusion detected: {person_name} (confidence: {confidence})")
            
            # Lưu ảnh
            image_path = self.save_intrusion_image(frame, person_name)
            
            # Cập nhật thời gian detection
            self.last_detection_time = current_time

    def run(self):
        cv2.namedWindow("Intrusion Warning")
        cv2.setMouseCallback("Intrusion Warning", self.handle_left_click, self.points)

        print("📹 Camera started. Instructions:")
        print("- Click to add points for detection area")
        print("- Press 'd' to complete the polygon and start detection")
        print("- Press 'q' to quit")

        while True:
            frame = self.video.read()
            
            # Kiểm tra frame hợp lệ
            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                print("⚠️ Camera không có frame, thử lại...")
                time.sleep(0.1)
                continue
                
            frame = cv2.flip(frame, 1)

            frame = self.draw_polygon(frame, self.points)

            if self.detect:
                frame = self.model.detect(frame=frame, points=self.points)

            cv2.imshow("Intrusion Warning", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                if len(self.points) > 2:
                    self.points.append(self.points[0])
                    self.detect = True
                    print("🔍 Started intrusion detection!")
                else:
                    print("⚠️ Cần ít nhất 3 điểm để tạo vùng detection!")
            elif key == ord('r'):
                # Reset points
                self.points = []
                self.detect = False
                print("🔄 Reset detection area")

    def cleanup(self):
        """Dọn dẹp khi thoát"""
        if self.video:
            self.video.stop()
        cv2.destroyAllWindows()
        print("✓ System shutdown complete")

if __name__ == "__main__":
    system = IntrusionDetectionSystem()
    
    try:
        system.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        system.cleanup()
cv2.destroyAllWindows()