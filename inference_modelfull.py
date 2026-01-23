"""
YOLOv11 Fall Detection - Inference Script
==========================================
Hỗ trợ inference trên:
- Ảnh đơn (jpg, png, bmp, ...)
- Nhiều ảnh (folder)
- Video (mp4, avi, mov, ...)
- Webcam (realtime)

Usage:
    # Inference ảnh
    python inference.py --source path/to/image.jpg
    
    # Inference folder ảnh
    python inference.py --source path/to/folder
    
    # Inference video
    python inference.py --source test.mp4
    
    # Webcam realtime
    python inference.py --source 0
    
    # Custom options
    python inference.py --source test.mp4 --conf 0.5 --save --show
"""

import argparse
import os
import sys
import time
from pathlib import Path
import asyncio
from datetime import datetime
import psycopg2
from teleConnect.telegram_ultil import send_telegram

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest

# ============== DEFAULT CONFIG ==============
DEFAULT_MODEL = "best.pt"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 480
DEFAULT_ALERT_CONF = 0.6

# Colors for visualization (BGR format)
COLORS = {
    0: (0, 0, 255),    # Fall - Red
    1: (0, 255, 0),    # Not Fall - Green
}
CLASS_NAMES = {0: "Fall", 1: "Not Fall"}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLOv11 Fall Detection Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --source image.jpg              # Single image
  python inference.py --source images/                # Folder of images
  python inference.py --source video.mp4             # Video file
  python inference.py --source 0                      # Webcam
  python inference.py --source video.mp4 --save      # Save output video
  python inference.py --source image.jpg --conf 0.5  # Custom confidence
        """
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Path to image, folder, video, or webcam index (0, 1, ...)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to model weights (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=DEFAULT_CONF,
        help=f"Confidence threshold (default: {DEFAULT_CONF})"
    )
    
    parser.add_argument(
        "--alert-conf",
        type=float,
        default=DEFAULT_ALERT_CONF,
        help="Confidence threshold to trigger FALL event/logging (default: 0.6)"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"IoU threshold for NMS (default: {DEFAULT_IOU})"
    )
    
    parser.add_argument(
        "--imgsz", "--img-size",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Inference image size (default: {DEFAULT_IMGSZ})"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use: 'cpu', '0', '0,1', etc. (default: '0')"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in window"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to output folder"
    )
    
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection results as txt files"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="inference_output",
        help="Output folder for saved results (default: inference_output)"
    )
    
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box line width (default: 2)"
    )
    
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide class labels"
    )
    
    parser.add_argument(
        "--hide-conf",
        action="store_true",
        help="Hide confidence scores"
    )
    
    return parser.parse_args()


def is_image(path):
    """Check if file is an image."""
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    return Path(path).suffix.lower() in image_exts


def is_video(path):
    """Check if file is a video."""
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    return Path(path).suffix.lower() in video_exts


def draw_detections(frame, boxes, scores, class_ids, args):
    """Draw bounding boxes on frame."""
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS.get(int(class_id), (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, args.line_width)
        
        # Prepare label
        if not args.hide_labels or not args.hide_conf:
            label_parts = []
            if not args.hide_labels:
                label_parts.append(CLASS_NAMES.get(int(class_id), f"Class {int(class_id)}"))
            if not args.hide_conf:
                label_parts.append(f"{score:.2f}")
            label = " ".join(label_parts)
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 5, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
    
    return frame


def draw_info(frame, fps=None):
    """Draw FPS info overlay on frame."""
    if fps is None:
        return frame
    
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (120, 40), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # FPS text
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame


def save_txt_results(txt_path, boxes, scores, class_ids, img_shape):
    """Save detection results to txt file in YOLO format."""
    h, w = img_shape[:2]
    
    with open(txt_path, 'w') as f:
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            
            # Convert to YOLO format (normalized)
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.4f}\n")


def setup_db():
    """Create PostgreSQL connection and ensure events table exists."""
    try:
        conn = psycopg2.connect(
            host='localhost',
            database='warning_data',
            user='phidinh',
            password='phi01478965',
            port=5432,
        )
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fall_events (
                id SERIAL PRIMARY KEY,
                event_time TIMESTAMP NOT NULL,
                event_type VARCHAR(32) NOT NULL,
                confidence REAL,
                image_path TEXT
            );
            """
        )
        conn.commit()
        cur.close()
        return conn
    except Exception as ex:
        print(f"⚠️ Không thể kết nối Postgres: {ex}")
        return None


def process_image(model, image_path, args, output_dir=None):
    """Process a single image."""
    # Read image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Cannot read image: {image_path}")
        return None
    
    # Run inference
    results = model.predict(
        source=frame,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False
    )[0]
    
    # Get detections
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    
    # Count detections (apply alert threshold for FALL only)
    fall_count = sum(1 for c, s in zip(class_ids, scores) if c == 0 and s >= args.alert_conf)
    not_fall_count = sum(1 for c in class_ids if c == 1)
    
    # Draw detections
    annotated_frame = draw_detections(frame.copy(), boxes, scores, class_ids, args)
    annotated_frame = draw_info(annotated_frame)
    
    # Print results
    print(f"📷 {Path(image_path).name}: Fall={fall_count}, Not Fall={not_fall_count}")
    
    # Save results
    if args.save and output_dir:
        output_path = output_dir / f"result_{Path(image_path).name}"
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"   💾 Saved: {output_path}")
        
        if args.save_txt:
            txt_path = output_dir / f"result_{Path(image_path).stem}.txt"
            save_txt_results(txt_path, boxes, scores, class_ids, frame.shape)
    
    # Show results
    if args.show:
        cv2.imshow("Fall Detection", annotated_frame)
        cv2.waitKey(0)
    
    return annotated_frame


def process_video(model, video_path, args, output_dir=None, db_conn=None, alert_cooldown=10):
    """Process video file or webcam."""
    # Open video source
    if str(video_path).isdigit():
        # Webcam
        source = int(video_path)
        cap = cv2.VideoCapture(source)
        is_webcam = True
        print(f" Opening webcam {source}...")
    else:
        # Video file
        cap = cv2.VideoCapture(str(video_path))
        is_webcam = False
        print(f"🎬 Processing video: {video_path}")
    
    if not cap.isOpened():
        print(f" Cannot open video source: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    
    print(f"   Resolution: {width}x{height}, FPS: {fps:.1f}")
    if total_frames > 0:
        print(f"   Total frames: {total_frames}")
    
    # Setup video writer
    video_writer = None
    if args.save and output_dir and not is_webcam:
        output_path = output_dir / f"result_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"   💾 Output will be saved to: {output_path}")
    
    # Process frames
    frame_count = 0
    fps_history = []
    last_alert_ts = 0
    
    print("\n Starting inference... (Press 'q' to quit)")
    print("-" * 50)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    continue
                else:
                    break
            
            frame_count += 1
            start_time = time.time()
            
            # Run inference
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False
            )[0]
            
            # Get detections
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()
            
            # Count detections (apply alert threshold for FALL only)
            fall_count = sum(1 for c, s in zip(class_ids, scores) if c == 0 and s >= args.alert_conf)
            not_fall_count = sum(1 for c in class_ids if c == 1)
            
            # Calculate FPS
            inference_time = time.time() - start_time
            current_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_history.append(current_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            # Draw detections
            annotated_frame = draw_detections(frame.copy(), boxes, scores, class_ids, args)
            annotated_frame = draw_info(annotated_frame, fps=avg_fps)
            
            # Show progress
            if not is_webcam and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"   Frame {frame_count}/{total_frames} ({progress:.1f}%) - FPS: {avg_fps:.1f}")
            
            # Alert if fall detected
            if fall_count > 0:
                # Draw alert
                cv2.putText(
                    annotated_frame, "!!! FALL DETECTED!",
                    (width // 2 - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA
                )
                
                # Cooldown to prevent spamming
                now_ts = time.time()
                if now_ts - last_alert_ts >= alert_cooldown:
                    last_alert_ts = now_ts
                    # Prepare screenshot path
                    if output_dir is None:
                        output_dir = Path("inference_output")
                        output_dir.mkdir(parents=True, exist_ok=True)
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = output_dir / f"fall_{ts_str}.jpg"
                    cv2.imwrite(str(screenshot_path), annotated_frame)
                    # Telegram notify (async)
                    try:
                        asyncio.run(send_telegram(str(screenshot_path)))
                    except Exception as ex:
                        print(f"Lỗi gửi Telegram: {ex}")
                    # Log to DB
                    try:
                        if db_conn is not None:
                            cur = db_conn.cursor()
                            # Max confidence for fall class (class_id==0) with threshold
                            fall_scores = [float(s) for s, c in zip(scores, class_ids) if int(c) == 0 and float(s) >= args.alert_conf]
                            conf_val = max(fall_scores) if fall_scores else None
                            cur.execute(
                                "INSERT INTO fall_events (event_time, event_type, confidence, image_path) VALUES (%s, %s, %s, %s)",
                                (datetime.now(), "fall", conf_val, str(screenshot_path)),
                            )
                            db_conn.commit()
                            cur.close()
                    except Exception as ex:
                        print(f"Lỗi ghi DB: {ex}")
                    # FCM notify
                    try:
                        send_fcm_fall_alert(0, max(fall_scores) if fall_scores else 0, str(screenshot_path))
                    except Exception as ex:
                        print(f"Lỗi gửi FCM: {ex}")
            
            # Save frame
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Show frame
            if args.show or is_webcam:
                cv2.imshow("Fall Detection", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n Stopped by user")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = output_dir / f"screenshot_{frame_count}.jpg" if output_dir else f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(str(screenshot_path), annotated_frame)
                    print(f"   📸 Screenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n Interrupted by user")
    
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
    
    # Print summary
    print("-" * 50)
    print(f" Processed {frame_count} frames")
    print(f"   Average FPS: {np.mean(fps_history):.1f}")


def send_fcm_fall_alert(event_id: int, confidence: float, image_path: str):
    """Gửi push notification qua Firebase Cloud Messaging HTTP v1 (không dùng Legacy).

    Yêu cầu biến môi trường:
      - FIREBASE_PROJECT_ID: ID dự án Firebase (VD: my-project-id)
      - FCM_SERVICE_ACCOUNT_FILE hoặc GOOGLE_APPLICATION_CREDENTIALS: đường dẫn file service account JSON
      - FCM_TOPIC (tùy chọn, mặc định 'fall_alerts')
    """
    try:
        project_id = os.getenv("FIREBASE_PROJECT_ID")
        sa_file = os.getenv("FCM_SERVICE_ACCOUNT_FILE") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        topic = os.getenv("FCM_TOPIC", "fall_alerts")
        if not project_id:
            print("⚠️ Thiếu FIREBASE_PROJECT_ID trong .env — bỏ qua gửi FCM")
            return False
        if not sa_file or not os.path.exists(sa_file):
            print("⚠️ Thiếu hoặc sai đường dẫn service account JSON (FCM_SERVICE_ACCOUNT_FILE/GOOGLE_APPLICATION_CREDENTIALS)")
            return False

        scopes = ["https://www.googleapis.com/auth/firebase.messaging"]
        credentials = service_account.Credentials.from_service_account_file(sa_file, scopes=scopes)
        credentials.refresh(GoogleAuthRequest())
        access_token = credentials.token
        if not access_token:
            print("⚠️ Không lấy được access token FCM v1")
            return False

        url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        title = "Cảnh báo té ngã"
        body = f"Phát hiện FALL (conf={confidence:.2f})"
        payload = {
            "message": {
                "topic": topic,
                "notification": {
                    "title": title,
                    "body": body,
                },
                "data": {
                    "event_id": str(event_id),
                    "event_type": "fall",
                    "confidence": f"{confidence:.4f}",
                    "image_path": image_path,
                    "timestamp": datetime.now().isoformat(),
                },
            }
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if 200 <= resp.status_code < 300:
            print("📲 Đã gửi push FCM (HTTP v1) thành công")
            return True
        else:
            print(f"⚠️ Gửi FCM v1 thất bại: {resp.status_code} - {resp.text}")
            return False
    except Exception as ex:
        print(f"⚠️ Lỗi gửi FCM v1: {ex}")
        return False


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print(" YOLOv11 Fall Detection - Inference")
    print("=" * 60)
    
    # Check model exists
    expanded_model = os.path.expanduser(args.model)
    if not Path(expanded_model).exists():
        print(f" Model not found: {expanded_model}")
        sys.exit(1)
    
    # Load model
    print(f"\n Loading model: {expanded_model}")
    from ultralytics import YOLO
    model = YOLO(expanded_model)
    print("    Model loaded successfully")
    
    # Create output directory
    output_dir = None
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"    Output directory: {output_dir}")
    
    # Print config
    print(f"\n Configuration:")
    print(f"   Confidence: {args.conf}")
    print(f"   IoU: {args.iou}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Device: {args.device}")
    print(f"   Alert confidence: {args.alert_conf}")
    print()
    
    source = args.source
    
    # Determine source type
    if source.isdigit():
        # Webcam
        # Setup DB for realtime alerts
        db_conn = setup_db()
        try:
            process_video(model, source, args, output_dir, db_conn=db_conn)
        finally:
            if db_conn is not None:
                try:
                    db_conn.close()
                except Exception:
                    pass
    
    elif Path(source).is_file():
        if is_image(source):
            # Single image
            process_image(model, source, args, output_dir)
        elif is_video(source):
            # Video file
            process_video(model, source, args, output_dir)
        else:
            print(f" Unsupported file type: {source}")
    
    elif Path(source).is_dir():
        # Folder of images
        print(f" Processing folder: {source}")
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(source).glob(ext))
        
        if not image_files:
            print(" No images found in folder")
            sys.exit(1)
        
        print(f"   Found {len(image_files)} images")
        
        for img_path in image_files:
            process_image(model, img_path, args, output_dir)
        
        print(f"\n Processed {len(image_files)} images")
    
    else:
        print(f" Source not found: {source}")
        sys.exit(1)
    
    if args.show:
        cv2.destroyAllWindows()
    
    print("\n Done!")


if __name__ == "__main__":
    load_dotenv()
    main()
