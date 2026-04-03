# Fall_Warning

Hệ thống phát hiện té ngã (Fall Detection) sử dụng YOLOv11 với smart filtering, gửi cảnh báo qua Telegram / Firebase FCM và lưu lịch sử vào PostgreSQL.

## Mô tả

Ứng dụng chạy mô hình YOLOv11 (`best.pt`) để phát hiện tình huống té ngã từ nguồn video/camera. Hệ thống bao gồm:
- **Smart Filtering**: Aspect Ratio, Temporal, Box Area, Smart NMS để giảm false positive
- **Đa kênh thông báo**: Telegram Bot, Firebase FCM, PostgreSQL logging
- **Đa nguồn input**: Webcam, video file, ảnh đơn, folder ảnh, RTSP stream
- **Mobile App**: Flutter app hiển thị lịch sử sự kiện

## Cấu trúc chính
- `inference.py` — file inference duy nhất, tích hợp toàn bộ tính năng
- `server/api_server.py` — FastAPI backend server
- `teleConnect/telegram_ultil.py` — gửi tin nhắn Telegram
- `train/train_full_dataset.py` — script training model
- `fall_warning_mobileapp/` — Flutter mobile app
- `best.pt` — model YOLOv11 đã train
- `.env` — cấu hình credentials (Telegram, DB, Firebase)

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Cách chạy

```powershell
# Webcam realtime
python inference.py --source 0 --show

# Video file + lưu kết quả
python inference.py --source testdemo/test.mp4 --show --save

# Ảnh đơn
python inference.py --source image.jpg --show --save

# Folder ảnh
python inference.py --source images/ --save

# Tuỳ chỉnh filtering
python inference.py --source 0 --show --min-aspect-ratio 1.5 --confirm-frames 8

# Tắt thông báo
python inference.py --source 0 --show --no-telegram --no-fcm --no-db
```

Kết quả output mặc định vào thư mục `result/`.

## Cấu hình

Tạo file `.env` tại thư mục gốc với nội dung:

```env
# Telegram
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# PostgreSQL
DB_HOST=localhost
DB_NAME=warning_data
DB_USER=your_user
DB_PASSWORD=your_password
DB_PORT=5432

# Firebase (optional)
FIREBASE_PROJECT_ID=your-project-id
FCM_SERVICE_ACCOUNT_FILE=path/to/service-account.json
```

## Lưu ý
- Đảm bảo `best.pt` tồn tại trong thư mục dự án
- Kiểm tra quyền truy cập camera nếu dùng webcam
- Smart filtering giúp giảm false positive: aspect ratio, temporal confirmation, NMS
