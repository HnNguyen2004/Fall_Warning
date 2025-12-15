                                                                                                                                                               # Fall_Warning

Một project đơn giản phát hiện té ngã (fall detection) sử dụng mô hình YOLOv8 và gửi cảnh báo qua Telegram.

## Mô tả
Ứng dụng này chạy mô hình phát hiện (một file mô hình `yolov8n.pt` có sẵn trong repo) để phát hiện tình huống té ngã từ nguồn video/camera. Khi phát hiện sự kiện quan trọng, project có thể gửi thông báo qua Telegram (cấu hình token/chat id trong `telegram_ultil.py`).

## Yêu cầu
- Python 3.8+ (đã thử trên Windows)
- Các phụ thuộc có trong `requirements.txt` (cài bằng pip)

## Cấu trúc chính
- `main.py` — entrypoint của ứng dụng
- `yolodetect.py` — logic liên quan đến việc dùng YOLO để phát hiện
- `telegram_ultil.py` — hàm/giao tiếp để gửi tin nhắn qua Telegram
- `yolov8n.pt` — file mô hình YOLOv8 (đã có trong repo)
- `requirements.txt` — danh sách package cần cài

## Cài đặt (Windows, PowerShell)
1. Tạo và kích hoạt virt        ual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Cài đặt phụ thuộc:

```powershell
pip install -r requirements.txt
```

Ghi chú: nếu không thể chạy `Activate.ps1` do policy, bạn có thể chạy trực tiếp Python trong venv:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Cách chạy
Sau khi cài đặt, chạy:

```powershell
python main.py
```

Hoặc dùng Python trong venv như ghi ở trên.

## Cấu hình Telegram
Mở file `telegram_ultil.py` để cấu hình `BOT_TOKEN` và `CHAT_ID` hoặc cấu hình dưới dạng biến môi trường (tùy cách implement trong file). Kiểm tra file này để biết nơi cần đặt token/chat id.

## Lưu ý
- Đảm bảo `yolov8n.pt` tồn tại trong thư mục dự án.
- Kiểm tra quyền truy cập camera nếu dùng camera thật.
- Việc phát hiện chính xác phụ thuộc vào mô hình và dữ liệu huấn luyện; cần tinh chỉnh nếu cần.
