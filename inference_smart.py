"""
YOLOv11 Fall Detection - Smart Inference Script
================================================
Phiên bản cải tiến với:
- Aspect Ratio Filtering: Phân biệt đứng/nằm qua tỷ lệ bounding box
- Temporal Filtering: Chỉ báo khi detect Fall liên tục N frames
- Position Filtering: Lọc theo vị trí trong khung hình
- Tối ưu cho camera góc cao (top-down view)

Usage:
    # Webcam với smart filtering
    python inference_smart.py --source 0 --show
    
    # Video file
    python inference_smart.py --source test.mp4 --show
    
    # Điều chỉnh các tham số filtering
    python inference_smart.py --source 0 --show --min-aspect-ratio 1.2 --confirm-frames 5
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
DEFAULT_CONF = 0.25           # Detect nhiều để không bỏ sót
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 480
DEFAULT_ALERT_CONF = 0.55     # Ngưỡng để tính là Fall alert

# Smart filtering defaults - QUAN TRỌNG cho việc lọc false positive
DEFAULT_MIN_ASPECT_RATIO = 1.2  # width/height >= 1.2 mới là người NẰM (Fall)
DEFAULT_CONFIRM_FRAMES = 5      # Cần detect liên tục N frames mới xác nhận Fall
DEFAULT_MIN_BOX_AREA = 0.02     # Box phải chiếm ít nhất 2% diện tích frame
DEFAULT_NMS_MERGE_IOU = 0.5     # IoU threshold để merge các box trùng

# Colors for visualization (BGR format)
COLORS = {
    0: (0, 0, 255),    # Fall - Red
    1: (0, 255, 0),    # Not Fall - Green
}
CLASS_NAMES = {0: "Fall", 1: "Not Fall"}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLOv11 Fall Detection - Smart Inference với filtering nâng cao",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_smart.py --source 0 --show                    # Webcam
  python inference_smart.py --source video.mp4 --show            # Video file
  python inference_smart.py --source 0 --min-aspect-ratio 1.5    # Strict aspect ratio
  python inference_smart.py --source 0 --confirm-frames 8        # Cần 8 frames liên tục
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
        help=f"Confidence threshold for FALL alert (default: {DEFAULT_ALERT_CONF})"
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
    
    # Smart filtering arguments
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=DEFAULT_MIN_ASPECT_RATIO,
        help=f"Minimum aspect ratio (w/h) to consider as Fall (default: {DEFAULT_MIN_ASPECT_RATIO})"
    )
    
    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=DEFAULT_CONFIRM_FRAMES,
        help=f"Number of consecutive frames to confirm Fall (default: {DEFAULT_CONFIRM_FRAMES})"
    )
    
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=DEFAULT_MIN_BOX_AREA,
        help=f"Minimum box area ratio (default: {DEFAULT_MIN_BOX_AREA})"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug info on screen"
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


def calculate_iou(box1, box2):
    """Tính IoU (Intersection over Union) giữa 2 box."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Tính intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Tính union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def smart_nms_with_class_priority(boxes, scores, class_ids, iou_threshold=0.5):
    """
    NMS thông minh với ưu tiên class:
    - Khi 2 box overlap cao, giữ box có confidence cao hơn
    - Nếu Fall và Not Fall overlap, ưu tiên Not Fall (class 1)
    
    Returns:
        indices: list các index được giữ lại
    """
    if len(boxes) == 0:
        return []
    
    # Sắp xếp theo: class (Not Fall trước), rồi confidence
    # Class 1 (Not Fall) được ưu tiên hơn Class 0 (Fall)
    indices = list(range(len(boxes)))
    
    # Sort: Not Fall (class 1) first, then by confidence descending
    indices.sort(key=lambda i: (-class_ids[i], -scores[i]))
    
    keep = []
    suppressed = set()
    
    for i in indices:
        if i in suppressed:
            continue
        
        keep.append(i)
        
        # Suppress các box overlap với box này
        for j in indices:
            if j in suppressed or j == i:
                continue
            
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                # Nếu cùng class, suppress box có confidence thấp hơn
                # Nếu khác class và overlap cao, suppress Fall (vì Not Fall được ưu tiên)
                suppressed.add(j)
    
    return keep


def filter_overlapping_fall_with_notfall(boxes, scores, class_ids, iou_threshold=0.3):
    """
    Kiểm tra xem Fall box có overlap với Not Fall box không.
    Nếu có, Fall đó có thể là false positive.
    
    Returns:
        dict: {fall_index: overlap_with_notfall (bool)}
    """
    fall_overlap_status = {}
    
    for i, (box_i, class_i) in enumerate(zip(boxes, class_ids)):
        if int(class_i) == 0:  # Fall
            overlaps_notfall = False
            for j, (box_j, class_j) in enumerate(zip(boxes, class_ids)):
                if int(class_j) == 1:  # Not Fall
                    iou = calculate_iou(box_i, box_j)
                    if iou > iou_threshold:
                        overlaps_notfall = True
                        break
            fall_overlap_status[i] = overlaps_notfall
    
    return fall_overlap_status


def analyze_fall_box(box, frame_shape, args, overlaps_notfall=False, confidence=0.0):
    """
    Phân tích bounding box để xác định có phải Fall thật không.
    
    Returns:
        dict: {
            'is_valid_fall': bool,
            'aspect_ratio': float,
            'box_area_ratio': float,
            'reason': str
        }
    """
    x1, y1, x2, y2 = box
    frame_h, frame_w = frame_shape[:2]
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    # Tính aspect ratio (width / height)
    aspect_ratio = box_w / box_h if box_h > 0 else 0
    
    # Tính tỷ lệ diện tích box so với frame
    box_area = box_w * box_h
    frame_area = frame_w * frame_h
    box_area_ratio = box_area / frame_area if frame_area > 0 else 0
    
    # Tính vị trí Y tương đối của box (0 = top, 1 = bottom)
    box_center_y = (y1 + y2) / 2
    relative_y = box_center_y / frame_h if frame_h > 0 else 0.5
    
    # Kiểm tra các điều kiện
    reasons = []
    is_valid = True
    
    # 0. Kiểm tra nếu overlap với Not Fall -> không phải Fall thật
    if overlaps_notfall:
        is_valid = False
        reasons.append("Overlaps NotFall")
    
    # 1. LUÔN kiểm tra aspect ratio - đây là filter quan trọng nhất
    # Người NẰM (Fall thật): box rộng hơn cao → aspect_ratio > 1.2
    # Người NGỒI/ĐỨNG: box cao hơn rộng hoặc gần vuông → aspect_ratio < 1.2
    if aspect_ratio < 1.2:
        is_valid = False
        reasons.append(f"AR={aspect_ratio:.2f}<1.2 (sitting/standing)")
    
    # 2. Kiểm tra kích thước box (quá nhỏ = false positive)
    if box_area_ratio < args.min_box_area:
        is_valid = False
        reasons.append(f"Area={box_area_ratio:.3f}<{args.min_box_area}")
    
    # 3. Kiểm tra box quá lớn (có thể chứa nhiều người/đồ vật)
    if box_area_ratio > 0.4:
        is_valid = False
        reasons.append(f"Area={box_area_ratio:.3f}>0.4 (too large)")
    
    # 4. Nếu confidence rất cao, cho phép relaxed aspect ratio
    # Conf >= 0.80: cho phép AR >= 0.8 (box gần vuông - người cuộn tròn khi té)
    # Conf >= 0.70: cho phép AR >= 0.9
    if not is_valid and not overlaps_notfall:
        if box_area_ratio >= args.min_box_area and box_area_ratio <= 0.4:
            if confidence >= 0.80 and aspect_ratio >= 0.8:
                is_valid = True
                reasons = [f"HighConf({confidence:.2f}) + AR>={aspect_ratio:.2f}"]
            elif confidence >= 0.70 and aspect_ratio >= 0.95:
                is_valid = True
                reasons = [f"MedConf({confidence:.2f}) + AR>={aspect_ratio:.2f}"]
    
    return {
        'is_valid_fall': is_valid,
        'aspect_ratio': aspect_ratio,
        'box_area_ratio': box_area_ratio,
        'relative_y': relative_y,
        'overlaps_notfall': overlaps_notfall,
        'reason': ', '.join(reasons) if reasons else 'Valid'
    }


def draw_detections_smart(frame, boxes, scores, class_ids, fall_analysis, args):
    """Draw bounding boxes với thông tin phân tích."""
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        
        # Xác định màu dựa trên phân tích
        if int(class_id) == 0:  # Fall class
            analysis = fall_analysis.get(i, {})
            if analysis.get('is_valid_fall', False):
                color = (0, 0, 255)  # Red - Valid fall
            else:
                color = (0, 165, 255)  # Orange - Filtered out fall
        else:
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
            
            # Add aspect ratio info for Fall detections
            if int(class_id) == 0 and args.debug:
                analysis = fall_analysis.get(i, {})
                ar = analysis.get('aspect_ratio', 0)
                label_parts.append(f"AR:{ar:.2f}")
            
            label = " ".join(label_parts)
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - 8),
                (x1 + label_w + 4, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
    
    return frame


def draw_info_smart(frame, fps=None, consecutive_falls=0, confirm_frames=5, debug_info=None):
    """Draw FPS và thông tin trạng thái."""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay cho info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    y_offset = 28
    
    # FPS
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 22
    
    # Fall confirmation progress
    progress = min(consecutive_falls / confirm_frames, 1.0)
    bar_width = 150
    cv2.putText(frame, f"Fall: {consecutive_falls}/{confirm_frames}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y_offset += 18
    
    # Progress bar
    cv2.rectangle(frame, (20, y_offset), (20 + bar_width, y_offset + 10), (100, 100, 100), -1)
    fill_width = int(bar_width * progress)
    bar_color = (0, 255, 0) if progress < 0.6 else (0, 165, 255) if progress < 1.0 else (0, 0, 255)
    cv2.rectangle(frame, (20, y_offset), (20 + fill_width, y_offset + 10), bar_color, -1)
    
    # Debug info
    if debug_info:
        y_offset += 20
        cv2.putText(frame, debug_info, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
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
    
    # ========== SMART NMS - Loại bỏ box trùng lặp ==========
    if len(boxes) > 1:
        keep_indices = smart_nms_with_class_priority(boxes, scores, class_ids, iou_threshold=DEFAULT_NMS_MERGE_IOU)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]
    
    # ========== Kiểm tra overlap giữa Fall và Not Fall ==========
    fall_overlap_status = filter_overlapping_fall_with_notfall(boxes, scores, class_ids, iou_threshold=0.3)
    
    # Analyze Fall detections
    fall_analysis = {}
    valid_fall_count = 0
    
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        if int(class_id) == 0 and score >= args.alert_conf:  # Fall class
            overlaps_notfall = fall_overlap_status.get(i, False)
            analysis = analyze_fall_box(box, frame.shape, args, overlaps_notfall, confidence=score)
            fall_analysis[i] = analysis
            if analysis['is_valid_fall']:
                valid_fall_count += 1
    
    not_fall_count = sum(1 for c in class_ids if c == 1)
    
    # Draw detections
    annotated_frame = draw_detections_smart(frame.copy(), boxes, scores, class_ids, fall_analysis, args)
    annotated_frame = draw_info_smart(annotated_frame)
    
    # Print results
    print(f"📷 {Path(image_path).name}: Valid Fall={valid_fall_count}, Not Fall={not_fall_count}")
    
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
        cv2.imshow("Fall Detection - Smart", annotated_frame)
        cv2.waitKey(0)
    
    return annotated_frame


def process_video(model, video_path, args, output_dir=None, db_conn=None, alert_cooldown=10):
    """Process video file or webcam với smart filtering."""
    # Open video source
    if str(video_path).isdigit():
        # Webcam
        source = int(video_path)
        cap = cv2.VideoCapture(source)
        is_webcam = True
        print(f"📹 Opening webcam {source}...")
    else:
        # Video file
        cap = cv2.VideoCapture(str(video_path))
        is_webcam = False
        print(f"🎬 Processing video: {video_path}")
    
    if not cap.isOpened():
        print(f"❌ Cannot open video source: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    
    print(f"   Resolution: {width}x{height}, FPS: {fps:.1f}")
    
    # Kiểm tra video có bị crop/letterbox không
    if width == 0 or height == 0:
        print(f"   ⚠️ Warning: Invalid video resolution!")
    
    # Kiểm tra aspect ratio
    aspect_ratio = width / height if height > 0 else 0
    print(f"   Aspect Ratio: {aspect_ratio:.2f}")
    
    if total_frames > 0:
        print(f"   Total frames: {total_frames}")
    
    print(f"\n📊 Smart Filtering Config:")
    print(f"   Min Aspect Ratio: {args.min_aspect_ratio}")
    print(f"   Confirm Frames: {args.confirm_frames}")
    print(f"   Min Box Area: {args.min_box_area}")
    
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
    
    # Temporal filtering state
    consecutive_fall_frames = 0
    
    print("\n🚀 Starting inference... (Press 'q' to quit, 's' to screenshot)")
    print("-" * 60)
    
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
            
            # ========== SMART NMS - Loại bỏ box trùng lặp ==========
            if len(boxes) > 1:
                keep_indices = smart_nms_with_class_priority(boxes, scores, class_ids, iou_threshold=DEFAULT_NMS_MERGE_IOU)
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                class_ids = class_ids[keep_indices]
            
            # ========== Kiểm tra overlap giữa Fall và Not Fall ==========
            fall_overlap_status = filter_overlapping_fall_with_notfall(boxes, scores, class_ids, iou_threshold=0.3)
            
            # ========== SMART FALL ANALYSIS ==========
            fall_analysis = {}
            valid_fall_count = 0
            max_fall_conf = 0
            debug_info = ""
            
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                if int(class_id) == 0 and score >= args.alert_conf:  # Fall class
                    overlaps_notfall = fall_overlap_status.get(i, False)
                    analysis = analyze_fall_box(box, frame.shape, args, overlaps_notfall, confidence=score)
                    fall_analysis[i] = analysis
                    
                    if analysis['is_valid_fall']:
                        valid_fall_count += 1
                        max_fall_conf = max(max_fall_conf, score)
                    
                    if args.debug:
                        debug_info = f"AR:{analysis['aspect_ratio']:.2f} Area:{analysis['box_area_ratio']:.3f} Y:{analysis.get('relative_y', 0):.2f}"
            
            # ========== TEMPORAL FILTERING ==========
            if valid_fall_count > 0:
                consecutive_fall_frames += 1
            else:
                consecutive_fall_frames = 0
            
            # Chỉ xác nhận Fall khi đủ số frame liên tục
            confirmed_fall = consecutive_fall_frames >= args.confirm_frames
            
            # Calculate FPS
            inference_time = time.time() - start_time
            current_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_history.append(current_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            # Draw detections
            annotated_frame = draw_detections_smart(frame.copy(), boxes, scores, class_ids, fall_analysis, args)
            annotated_frame = draw_info_smart(
                annotated_frame, 
                fps=avg_fps, 
                consecutive_falls=consecutive_fall_frames,
                confirm_frames=args.confirm_frames,
                debug_info=debug_info if args.debug else None
            )
            
            # Show progress for video files
            if not is_webcam and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"   Frame {frame_count}/{total_frames} ({progress:.1f}%) - FPS: {avg_fps:.1f}")
            
            # ========== ALERT IF FALL CONFIRMED ==========
            if confirmed_fall:
                # Draw alert
                cv2.putText(
                    annotated_frame, "!!! FALL CONFIRMED !!!",
                    (width // 2 - 180, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA
                )
                
                # Cooldown to prevent spamming
                now_ts = time.time()
                if now_ts - last_alert_ts >= alert_cooldown:
                    last_alert_ts = now_ts
                    consecutive_fall_frames = 0  # Reset counter after alert
                    
                    # Prepare screenshot path
                    if output_dir is None:
                        output_dir = Path("inference_output")
                        output_dir.mkdir(parents=True, exist_ok=True)
                    
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = output_dir / f"fall_{ts_str}.jpg"
                    cv2.imwrite(str(screenshot_path), annotated_frame)
                    print(f"\n🚨 FALL CONFIRMED! Screenshot saved: {screenshot_path}")
                    
                    # Telegram notify
                    try:
                        asyncio.run(send_telegram(str(screenshot_path)))
                    except Exception as ex:
                        print(f"Lỗi gửi Telegram: {ex}")
                    
                    # Log to DB
                    try:
                        if db_conn is not None:
                            cur = db_conn.cursor()
                            cur.execute(
                                "INSERT INTO fall_events (event_time, event_type, confidence, image_path) VALUES (%s, %s, %s, %s)",
                                (datetime.now(), "fall", float(max_fall_conf), str(screenshot_path)),
                            )
                            db_conn.commit()
                            cur.close()
                    except Exception as ex:
                        print(f"Lỗi ghi DB: {ex}")
                    
                    # FCM notify
                    try:
                        send_fcm_fall_alert(0, float(max_fall_conf), str(screenshot_path))
                    except Exception as ex:
                        print(f"Lỗi gửi FCM: {ex}")
            
            # Save frame
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Show frame
            if args.show or is_webcam:
                # Resize frame to fit screen if too large
                display_frame = annotated_frame
                max_display_width = 1280
                max_display_height = 720
                
                if width > max_display_width or height > max_display_height:
                    scale = min(max_display_width / width, max_display_height / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_frame = cv2.resize(annotated_frame, (new_width, new_height))
                
                cv2.imshow("Fall Detection - Smart", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️ Stopped by user")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    if output_dir is None:
                        output_dir = Path("inference_output")
                        output_dir.mkdir(parents=True, exist_ok=True)
                    screenshot_path = output_dir / f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(str(screenshot_path), annotated_frame)
                    print(f"   📸 Screenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
    
    # Print summary
    print("-" * 60)
    print(f"✅ Processed {frame_count} frames")
    print(f"   Average FPS: {np.mean(fps_history):.1f}")


def send_fcm_fall_alert(event_id: int, confidence: float, image_path: str):
    """Gửi push notification qua Firebase Cloud Messaging HTTP v1."""
    try:
        project_id = os.getenv("FIREBASE_PROJECT_ID")
        sa_file = os.getenv("FCM_SERVICE_ACCOUNT_FILE") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        topic = os.getenv("FCM_TOPIC", "fall_alerts")
        if not project_id:
            print("⚠️ Thiếu FIREBASE_PROJECT_ID trong .env — bỏ qua gửi FCM")
            return False
        if not sa_file or not os.path.exists(sa_file):
            print("⚠️ Thiếu hoặc sai đường dẫn service account JSON")
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
        title = "🚨 Cảnh báo té ngã"
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
            print("📲 Đã gửi push FCM thành công")
            return True
        else:
            print(f"⚠️ Gửi FCM thất bại: {resp.status_code}")
            return False
    except Exception as ex:
        print(f"⚠️ Lỗi gửi FCM: {ex}")
        return False


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("🎯 YOLOv11 Fall Detection - SMART Inference")
    print("=" * 60)
    
    # Check model exists
    expanded_model = os.path.expanduser(args.model)
    if not Path(expanded_model).exists():
        print(f"❌ Model not found: {expanded_model}")
        sys.exit(1)
    
    # Load model
    print(f"\n📦 Loading model: {expanded_model}")
    from ultralytics import YOLO
    model = YOLO(expanded_model)
    print("✅ Model loaded successfully")
    
    # Create output directory
    output_dir = None
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
    
    # Print config
    print(f"\n⚙️ Configuration:")
    print(f"   Model Confidence: {args.conf}")
    print(f"   Alert Confidence: {args.alert_conf}")
    print(f"   IoU: {args.iou}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Device: {args.device}")
    print(f"\n🧠 Smart Filtering:")
    print(f"   Min Aspect Ratio (w/h): {args.min_aspect_ratio}")
    print(f"   Confirm Frames: {args.confirm_frames}")
    print(f"   Min Box Area: {args.min_box_area}")
    print()
    
    source = args.source
    
    # Determine source type
    if source.isdigit():
        # Webcam
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
            process_image(model, source, args, output_dir)
        elif is_video(source):
            process_video(model, source, args, output_dir)
        else:
            print(f"❌ Unsupported file type: {source}")
    
    elif Path(source).is_dir():
        print(f"📁 Processing folder: {source}")
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(source).glob(ext))
        
        if not image_files:
            print("❌ No images found in folder")
            sys.exit(1)
        
        print(f"   Found {len(image_files)} images")
        
        for img_path in image_files:
            process_image(model, img_path, args, output_dir)
        
        print(f"\n✅ Processed {len(image_files)} images")
    
    else:
        print(f"❌ Source not found: {source}")
        sys.exit(1)
    
    if args.show:
        cv2.destroyAllWindows()
    
    print("\n🎉 Done!")


if __name__ == "__main__":
    load_dotenv()
    main()
