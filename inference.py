"""
YOLOv11 Fall Detection - Unified Inference Script
===================================================
Tich hop day du:
  - Smart Filtering: Aspect Ratio, Temporal, Box Area, Smart NMS
  - Da nguon: webcam, video file, anh don, folder anh, RTSP stream
  - Canh bao: Telegram, Firebase FCM, PostgreSQL logging
  - Hien thi: FPS, progress bar, debug overlay
  - Tu dong resize video fit man hinh

Usage:
    # Webcam realtime
    python inference.py --source 0 --show

    # Video file
    python inference.py --source test.mp4 --show --save

    # Anh don
    python inference.py --source image.jpg --show --save

    # Folder anh
    python inference.py --source images/ --save

    # Tuy chinh filtering
    python inference.py --source 0 --show --min-aspect-ratio 1.5 --confirm-frames 8

    # Custom model & confidence
    python inference.py --source video.mp4 --model best.pt --conf 0.3 --alert-conf 0.6 --save
"""

import argparse
import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv

# ============== DEFAULT CONFIG ==============
DEFAULT_MODEL = "best.pt"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 480
DEFAULT_ALERT_CONF = 0.55
DEFAULT_OUTPUT = "result"

# Smart filtering defaults
DEFAULT_MIN_ASPECT_RATIO = 1.2
DEFAULT_CONFIRM_FRAMES = 5
DEFAULT_MIN_BOX_AREA = 0.02
DEFAULT_MAX_BOX_AREA = 0.4
DEFAULT_NMS_MERGE_IOU = 0.5
DEFAULT_ALERT_COOLDOWN = 10

# Colors (BGR)
COLORS = {
    0: (0, 0, 255),   # Fall - Red
    1: (0, 255, 0),   # Not Fall - Green
}
CLASS_NAMES = {0: "Fall", 1: "Not Fall"}

# Display
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720


# =====================================================================
#  ARGUMENT PARSER
# =====================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLOv11 Fall Detection - Unified Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --source 0 --show                     # Webcam
  python inference.py --source video.mp4 --show --save      # Video + save
  python inference.py --source image.jpg --show             # Image
  python inference.py --source images/ --save               # Folder
  python inference.py --source 0 --min-aspect-ratio 1.5     # Strict filter
  python inference.py --source 0 --confirm-frames 8         # 8 frames lien tuc
        """
    )

    # Source & Model
    parser.add_argument("--source", "-s", type=str, required=True,
                        help="Path to image, folder, video, or webcam index (0, 1, ...)")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"Path to model weights (default: {DEFAULT_MODEL})")

    # Detection thresholds
    parser.add_argument("--conf", "-c", type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument("--alert-conf", type=float, default=DEFAULT_ALERT_CONF,
                        help=f"Confidence threshold for FALL alert (default: {DEFAULT_ALERT_CONF})")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU,
                        help=f"IoU threshold for NMS (default: {DEFAULT_IOU})")
    parser.add_argument("--imgsz", "--img-size", type=int, default=DEFAULT_IMGSZ,
                        help=f"Inference image size (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: 'cpu', '0', '0,1', etc. (default: '0')")

    # Smart filtering
    parser.add_argument("--min-aspect-ratio", type=float, default=DEFAULT_MIN_ASPECT_RATIO,
                        help=f"Min aspect ratio (w/h) for Fall (default: {DEFAULT_MIN_ASPECT_RATIO})")
    parser.add_argument("--confirm-frames", type=int, default=DEFAULT_CONFIRM_FRAMES,
                        help=f"Consecutive frames to confirm Fall (default: {DEFAULT_CONFIRM_FRAMES})")
    parser.add_argument("--min-box-area", type=float, default=DEFAULT_MIN_BOX_AREA,
                        help=f"Min box area ratio (default: {DEFAULT_MIN_BOX_AREA})")
    parser.add_argument("--alert-cooldown", type=int, default=DEFAULT_ALERT_COOLDOWN,
                        help=f"Seconds between alerts (default: {DEFAULT_ALERT_COOLDOWN})")

    # Output
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output folder (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--show", action="store_true", help="Display results in window")
    parser.add_argument("--save", action="store_true", help="Save results to output folder")
    parser.add_argument("--save-txt", action="store_true", help="Save detections as YOLO txt")

    # Display
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width")
    parser.add_argument("--hide-labels", action="store_true", help="Hide class labels")
    parser.add_argument("--hide-conf", action="store_true", help="Hide confidence scores")
    parser.add_argument("--debug", action="store_true", help="Show debug overlay")

    # Alerts toggle
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram alerts")
    parser.add_argument("--no-fcm", action="store_true", help="Disable FCM push notifications")
    parser.add_argument("--no-db", action="store_true", help="Disable database logging")

    return parser.parse_args()


# =====================================================================
#  UTILITY: File type checks
# =====================================================================
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}


def is_image(path):
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_video(path):
    return Path(path).suffix.lower() in VIDEO_EXTS


# =====================================================================
#  SMART FILTERING
# =====================================================================
def calculate_iou(box1, box2):
    """Tinh IoU giua 2 boxes [x1, y1, x2, y2]."""
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def smart_nms_with_class_priority(boxes, scores, class_ids, iou_threshold=0.5):
    """
    NMS thong minh: khi 2 box overlap cao, uu tien giu Not Fall (class 1).
    Returns: list cac index duoc giu lai.
    """
    if len(boxes) == 0:
        return []

    indices = list(range(len(boxes)))
    indices.sort(key=lambda i: (-class_ids[i], -scores[i]))

    keep = []
    suppressed = set()

    for i in indices:
        if i in suppressed:
            continue
        keep.append(i)
        for j in indices:
            if j in suppressed or j == i:
                continue
            if calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed.add(j)

    return keep


def check_fall_overlap_with_notfall(boxes, class_ids, iou_threshold=0.3):
    """Kiem tra Fall box co overlap voi Not Fall box khong."""
    result = {}
    for i, cls_i in enumerate(class_ids):
        if int(cls_i) == 0:
            overlaps = False
            for j, cls_j in enumerate(class_ids):
                if int(cls_j) == 1:
                    if calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                        overlaps = True
                        break
            result[i] = overlaps
    return result


def analyze_fall_box(box, frame_shape, args, overlaps_notfall=False, confidence=0.0):
    """Phan tich bounding box de xac dinh Fall that hay false positive."""
    x1, y1, x2, y2 = box
    frame_h, frame_w = frame_shape[:2]

    box_w = x2 - x1
    box_h = y2 - y1
    aspect_ratio = box_w / box_h if box_h > 0 else 0
    box_area_ratio = (box_w * box_h) / (frame_w * frame_h) if (frame_w * frame_h) > 0 else 0
    relative_y = ((y1 + y2) / 2) / frame_h if frame_h > 0 else 0.5

    reasons = []
    is_valid = True

    if overlaps_notfall:
        is_valid = False
        reasons.append("Overlaps NotFall")

    if aspect_ratio < args.min_aspect_ratio:
        is_valid = False
        reasons.append(f"AR={aspect_ratio:.2f}<{args.min_aspect_ratio} (sitting/standing)")

    if box_area_ratio < args.min_box_area:
        is_valid = False
        reasons.append(f"Area={box_area_ratio:.3f}<{args.min_box_area}")

    if box_area_ratio > DEFAULT_MAX_BOX_AREA:
        is_valid = False
        reasons.append(f"Area={box_area_ratio:.3f}>{DEFAULT_MAX_BOX_AREA} (too large)")

    if not is_valid and not overlaps_notfall:
        if args.min_box_area <= box_area_ratio <= DEFAULT_MAX_BOX_AREA:
            if confidence >= 0.80 and aspect_ratio >= 0.8:
                is_valid = True
                reasons = [f"HighConf({confidence:.2f})+AR>={aspect_ratio:.2f}"]
            elif confidence >= 0.70 and aspect_ratio >= 0.95:
                is_valid = True
                reasons = [f"MedConf({confidence:.2f})+AR>={aspect_ratio:.2f}"]

    return {
        'is_valid_fall': is_valid,
        'aspect_ratio': aspect_ratio,
        'box_area_ratio': box_area_ratio,
        'relative_y': relative_y,
        'overlaps_notfall': overlaps_notfall,
        'reason': ', '.join(reasons) if reasons else 'Valid',
    }


def apply_smart_filtering(boxes, scores, class_ids, frame_shape, args):
    """Pipeline filtering hoan chinh: NMS -> overlap check -> analyze."""
    if len(boxes) > 1:
        keep = smart_nms_with_class_priority(boxes, scores, class_ids, DEFAULT_NMS_MERGE_IOU)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

    overlap_status = check_fall_overlap_with_notfall(boxes, class_ids, iou_threshold=0.3)

    fall_analysis = {}
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        if int(cls_id) == 0 and score >= args.alert_conf:
            overlaps = overlap_status.get(i, False)
            analysis = analyze_fall_box(box, frame_shape, args, overlaps, confidence=score)
            fall_analysis[i] = analysis

    return boxes, scores, class_ids, fall_analysis


# =====================================================================
#  DRAWING
# =====================================================================
def draw_detections(frame, boxes, scores, class_ids, fall_analysis, args):
    """Ve bounding boxes len frame."""
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = map(int, box)

        if int(cls_id) == 0:
            analysis = fall_analysis.get(i, {})
            color = (0, 0, 255) if analysis.get('is_valid_fall', False) else (0, 165, 255)
        else:
            color = COLORS.get(int(cls_id), (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, args.line_width)

        if not args.hide_labels or not args.hide_conf:
            parts = []
            if not args.hide_labels:
                parts.append(CLASS_NAMES.get(int(cls_id), f"Class {int(cls_id)}"))
            if not args.hide_conf:
                parts.append(f"{score:.2f}")
            if int(cls_id) == 0 and args.debug:
                ar = fall_analysis.get(i, {}).get('aspect_ratio', 0)
                parts.append(f"AR:{ar:.2f}")

            label = " ".join(parts)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_hud(frame, fps=None, consecutive_falls=0, confirm_frames=5,
             debug_info=None, confirmed=False):
    """Ve HUD overlay: FPS, fall progress bar, debug."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    panel_h = 105 if debug_info else 90
    cv2.rectangle(overlay, (10, 10), (260, panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    y = 28

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    cv2.putText(frame, f"Fall: {consecutive_falls}/{confirm_frames}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y += 18

    progress = min(consecutive_falls / confirm_frames, 1.0) if confirm_frames > 0 else 0
    bar_w = 150
    cv2.rectangle(frame, (20, y), (20 + bar_w, y + 10), (100, 100, 100), -1)
    fill = int(bar_w * progress)
    if progress < 0.6:
        bar_color = (0, 255, 0)
    elif progress < 1.0:
        bar_color = (0, 165, 255)
    else:
        bar_color = (0, 0, 255)
    cv2.rectangle(frame, (20, y), (20 + fill, y + 10), bar_color, -1)

    if debug_info:
        y += 20
        cv2.putText(frame, debug_info, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    if confirmed:
        cv2.putText(frame, "!!! FALL CONFIRMED !!!", (w // 2 - 180, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    return frame


# =====================================================================
#  ALERT SYSTEM
# =====================================================================
class AlertSystem:
    """Quan ly gui canh bao: Telegram, FCM, DB. Credentials lay tu .env."""

    def __init__(self, enable_telegram=True, enable_fcm=True, enable_db=True):
        self.enable_telegram = enable_telegram
        self.enable_fcm = enable_fcm
        self.enable_db = enable_db
        self.db_conn = None

        if self.enable_db:
            self._setup_db()

    def _setup_db(self):
        try:
            import psycopg2  # noqa: F811
            self.db_conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                database=os.getenv("DB_NAME", "warning_data"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
                port=int(os.getenv("DB_PORT", "5432")),
            )
            cur = self.db_conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fall_events (
                    id SERIAL PRIMARY KEY,
                    event_time TIMESTAMP NOT NULL,
                    event_type VARCHAR(32) NOT NULL,
                    confidence REAL,
                    image_path TEXT
                );
            """)
            self.db_conn.commit()
            cur.close()
            print("DB connected")
        except Exception as ex:
            print(f"DB unavailable: {ex}")
            self.db_conn = None

    def _log_to_db(self, confidence, image_path):
        if self.db_conn is None:
            return
        try:
            cur = self.db_conn.cursor()
            cur.execute(
                "INSERT INTO fall_events (event_time, event_type, confidence, image_path) "
                "VALUES (%s, %s, %s, %s)",
                (datetime.now(), "fall", float(confidence), str(image_path)),
            )
            self.db_conn.commit()
            cur.close()
            print("   Logged to database")
        except Exception as ex:
            print(f"   DB log error: {ex}")

    def _send_telegram(self, image_path):
        try:
            from teleConnect.telegram_ultil import send_telegram  # noqa: F811
            import asyncio  # noqa: F811
            asyncio.run(send_telegram(str(image_path)))
        except Exception as ex:
            print(f"   Telegram error: {ex}")

    def _send_fcm(self, confidence, image_path):
        try:
            from google.oauth2 import service_account  # noqa: F811
            from google.auth.transport.requests import Request as GoogleAuthRequest  # noqa: F811
            import requests  # noqa: F811

            project_id = os.getenv("FIREBASE_PROJECT_ID")
            sa_file = os.getenv("FCM_SERVICE_ACCOUNT_FILE") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            topic = os.getenv("FCM_TOPIC", "fall_alerts")

            if not project_id:
                return
            if not sa_file or not os.path.exists(sa_file):
                return

            scopes = ["https://www.googleapis.com/auth/firebase.messaging"]
            credentials = service_account.Credentials.from_service_account_file(sa_file, scopes=scopes)
            credentials.refresh(GoogleAuthRequest())
            access_token = credentials.token
            if not access_token:
                return

            url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
            payload = {
                "message": {
                    "topic": topic,
                    "notification": {
                        "title": "Canh bao te nga",
                        "body": f"Phat hien FALL (conf={confidence:.2f})",
                    },
                    "data": {
                        "event_type": "fall",
                        "confidence": f"{confidence:.4f}",
                        "image_path": str(image_path),
                        "timestamp": datetime.now().isoformat(),
                    },
                }
            }
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                json=payload, timeout=10,
            )
            if 200 <= resp.status_code < 300:
                print("   FCM push sent")
            else:
                print(f"   FCM failed: {resp.status_code}")
        except Exception as ex:
            print(f"   FCM error: {ex}")

    def send_alert(self, confidence, image_path):
        """Gui tat ca alerts trong background thread (non-blocking)."""
        def _do():
            if self.enable_telegram:
                self._send_telegram(image_path)
            if self.enable_fcm:
                self._send_fcm(confidence, image_path)
            if self.enable_db:
                self._log_to_db(confidence, image_path)

        thread = threading.Thread(target=_do, daemon=True)
        thread.start()

    def close(self):
        if self.db_conn:
            try:
                self.db_conn.close()
            except Exception:
                pass


# =====================================================================
#  SAVE TXT (YOLO format)
# =====================================================================
def save_txt_results(txt_path, boxes, scores, class_ids, img_shape):
    h, w = img_shape[:2]
    with open(txt_path, 'w') as f:
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {score:.4f}\n")


# =====================================================================
#  PROCESS IMAGE
# =====================================================================
def process_image(model, image_path, args, output_dir=None):
    """Xu ly 1 anh don."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Cannot read: {image_path}")
        return None

    results = model.predict(
        source=frame, conf=args.conf, iou=args.iou,
        imgsz=args.imgsz, device=args.device, verbose=False,
    )[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()

    boxes, scores, class_ids, fall_analysis = apply_smart_filtering(
        boxes, scores, class_ids, frame.shape, args,
    )

    valid_falls = sum(1 for a in fall_analysis.values() if a['is_valid_fall'])
    not_falls = sum(1 for c in class_ids if int(c) == 1)

    out = draw_detections(frame.copy(), boxes, scores, class_ids, fall_analysis, args)
    out = draw_hud(out, consecutive_falls=valid_falls, confirm_frames=1, confirmed=valid_falls > 0)

    print(f"{Path(image_path).name}: Valid Fall={valid_falls}, Not Fall={not_falls}")

    if args.save and output_dir:
        out_path = output_dir / f"result_{Path(image_path).name}"
        cv2.imwrite(str(out_path), out)
        print(f"   Saved: {out_path}")
        if args.save_txt:
            txt_path = output_dir / f"result_{Path(image_path).stem}.txt"
            save_txt_results(txt_path, boxes, scores, class_ids, frame.shape)

    if args.show:
        cv2.imshow("Fall Detection", out)
        cv2.waitKey(0)

    return out


# =====================================================================
#  PROCESS VIDEO / WEBCAM
# =====================================================================
def process_video(model, video_path, args, output_dir=None, alert_system=None):
    """Xu ly video hoac webcam voi smart filtering + temporal confirmation."""
    if str(video_path).isdigit():
        source = int(video_path)
        cap = cv2.VideoCapture(source)
        is_webcam = True
        print(f"Opening webcam {source}...")
    else:
        cap = cv2.VideoCapture(str(video_path))
        is_webcam = False
        print(f"Processing video: {video_path}")

    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    print(f"   Resolution: {width}x{height}, FPS: {fps:.1f}")
    if total_frames > 0:
        duration = total_frames / fps
        print(f"   Total frames: {total_frames} ({duration:.1f}s)")

    print(f"\nSmart Filtering:")
    print(f"   Min Aspect Ratio: {args.min_aspect_ratio}")
    print(f"   Confirm Frames:   {args.confirm_frames}")
    print(f"   Min Box Area:     {args.min_box_area}")
    print(f"   Alert Cooldown:   {args.alert_cooldown}s")

    video_writer = None
    if args.save and output_dir:
        out_name = f"result_{Path(video_path).stem}.mp4" if not is_webcam else "result_webcam.mp4"
        out_path = output_dir / out_name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"   Output: {out_path}")

    frame_count = 0
    fps_history = []
    last_alert_ts = 0
    consecutive_falls = 0

    print(f"\nStarting inference... (Press 'q' to quit, 's' for screenshot)")
    print("-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    continue
                break

            frame_count += 1
            t0 = time.time()

            results = model.predict(
                source=frame, conf=args.conf, iou=args.iou,
                imgsz=args.imgsz, device=args.device, verbose=False,
            )[0]

            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()

            boxes, scores, class_ids, fall_analysis = apply_smart_filtering(
                boxes, scores, class_ids, frame.shape, args,
            )

            valid_falls = sum(1 for a in fall_analysis.values() if a['is_valid_fall'])
            max_conf = 0.0
            if valid_falls > 0:
                max_conf = max(
                    scores[i] for i in fall_analysis if fall_analysis[i]['is_valid_fall']
                )

            debug_info = None
            if args.debug and fall_analysis:
                first_key = next(iter(fall_analysis))
                a = fall_analysis[first_key]
                debug_info = f"AR:{a['aspect_ratio']:.2f} Area:{a['box_area_ratio']:.3f} Y:{a['relative_y']:.2f}"

            if valid_falls > 0:
                consecutive_falls += 1
            else:
                consecutive_falls = 0

            confirmed = consecutive_falls >= args.confirm_frames

            dt = time.time() - t0
            cur_fps = 1.0 / dt if dt > 0 else 0
            fps_history.append(cur_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)

            annotated = draw_detections(frame.copy(), boxes, scores, class_ids, fall_analysis, args)
            annotated = draw_hud(
                annotated, fps=avg_fps, consecutive_falls=consecutive_falls,
                confirm_frames=args.confirm_frames,
                debug_info=debug_info if args.debug else None,
                confirmed=confirmed,
            )

            if not is_webcam and frame_count % 30 == 0:
                pct = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"   Frame {frame_count}/{total_frames} ({pct:.1f}%) - FPS: {avg_fps:.1f}")

            if confirmed and alert_system:
                now = time.time()
                if now - last_alert_ts >= args.alert_cooldown:
                    last_alert_ts = now
                    consecutive_falls = 0

                    if output_dir is None:
                        output_dir = Path(DEFAULT_OUTPUT)
                        output_dir.mkdir(parents=True, exist_ok=True)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    shot_path = output_dir / f"fall_{ts}.jpg"
                    cv2.imwrite(str(shot_path), annotated)
                    print(f"\nFALL CONFIRMED! Screenshot: {shot_path}")

                    alert_system.send_alert(float(max_conf), str(shot_path))

            if video_writer:
                video_writer.write(annotated)

            if args.show or is_webcam:
                display = annotated
                if width > MAX_DISPLAY_WIDTH or height > MAX_DISPLAY_HEIGHT:
                    scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height)
                    display = cv2.resize(annotated, (int(width * scale), int(height * scale)))
                cv2.imshow("Fall Detection", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nStopped by user")
                    break
                elif key == ord('s'):
                    if output_dir is None:
                        output_dir = Path(DEFAULT_OUTPUT)
                        output_dir.mkdir(parents=True, exist_ok=True)
                    ss = output_dir / f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(str(ss), annotated)
                    print(f"   Screenshot: {ss}")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

    print("-" * 60)
    print(f"Processed {frame_count} frames")
    if fps_history:
        print(f"   Average FPS: {np.mean(fps_history):.1f}")


# =====================================================================
#  MAIN
# =====================================================================
def main():
    args = parse_args()

    print("=" * 60)
    print("YOLOv11 Fall Detection - Unified Inference")
    print("=" * 60)

    model_path = os.path.expanduser(args.model)
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"\nLoading model: {model_path}")
    from ultralytics import YOLO  # noqa: F811
    model = YOLO(model_path)
    print("Model loaded")

    output_dir = None
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")

    print(f"\nConfiguration:")
    print(f"   Model Conf:  {args.conf}")
    print(f"   Alert Conf:  {args.alert_conf}")
    print(f"   IoU:         {args.iou}")
    print(f"   Image Size:  {args.imgsz}")
    print(f"   Device:      {args.device}")

    alert_system = AlertSystem(
        enable_telegram=not args.no_telegram,
        enable_fcm=not args.no_fcm,
        enable_db=not args.no_db,
    )

    source = args.source

    try:
        if source.isdigit():
            if output_dir is None:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
            process_video(model, source, args, output_dir, alert_system)

        elif Path(source).is_file():
            if is_image(source):
                process_image(model, source, args, output_dir)
            elif is_video(source):
                process_video(model, source, args, output_dir, alert_system)
            else:
                print(f"Unsupported file type: {source}")

        elif Path(source).is_dir():
            print(f"Processing folder: {source}")
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(Path(source).glob(ext))
            if not images:
                print("No images found")
                sys.exit(1)
            print(f"   Found {len(images)} images")
            for img in images:
                process_image(model, img, args, output_dir)
            print(f"\nProcessed {len(images)} images")

        else:
            print(f"Source not found: {source}")
            sys.exit(1)
    finally:
        alert_system.close()

    if args.show:
        cv2.destroyAllWindows()

    print("\nDone!")


if __name__ == "__main__":
    load_dotenv()
    main()
