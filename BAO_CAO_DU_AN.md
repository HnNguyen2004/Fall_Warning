# 📋 BÁO CÁO DỰ ÁN FALL DETECTION
## Hệ thống phát hiện té ngã sử dụng YOLOv11

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1 Mục tiêu
Xây dựng hệ thống phát hiện người té ngã (Fall Detection) theo thời gian thực, có khả năng:
- Phát hiện té ngã từ camera giám sát
- Gửi cảnh báo qua nhiều kênh (Telegram, Push Notification)
- Lưu trữ lịch sử sự kiện vào database
- Ứng dụng mobile để theo dõi

### 1.2 Công nghệ sử dụng
| Thành phần | Công nghệ |
|------------|-----------|
| AI/ML Model | YOLOv11 (Ultralytics) |
| Backend | Python, FastAPI |
| Database | PostgreSQL |
| Mobile App | Flutter |
| Notification | Telegram Bot, Firebase FCM |

---

## 2. NHỮNG GÌ ĐÃ LÀM ĐƯỢC ✅

### 2.1 Model AI
- [x] Train model YOLOv11 với dataset Fall Detection
- [x] Đạt accuracy cơ bản cho việc phát hiện té ngã
- [x] Hỗ trợ inference trên GPU và CPU

### 2.2 Inference Script (inference_smart.py)
- [x] Xử lý video từ nhiều nguồn: webcam, file video, RTSP stream
- [x] Smart Filtering để giảm false positive:
  - Aspect Ratio Filter (lọc người ngồi/đứng)
  - Box Area Filter (lọc đối tượng quá nhỏ/lớn)
  - Temporal Filter (xác nhận qua N frames liên tục)
  - NMS với ưu tiên class
- [x] Hiển thị FPS, progress bar, thông tin debug
- [x] Tự động resize video để fit màn hình

### 2.3 Hệ thống thông báo
- [x] Gửi cảnh báo qua Telegram kèm ảnh chụp
- [x] Push notification qua Firebase Cloud Messaging (FCM)
- [x] Cooldown để tránh spam thông báo

### 2.4 Database
- [x] Lưu sự kiện té ngã vào PostgreSQL
- [x] Ghi nhận thời gian, confidence, đường dẫn ảnh

### 2.5 Mobile App (Flutter)
- [x] Cấu trúc cơ bản ứng dụng di động
- [x] Kết nối với backend

---

## 3. ƯU ĐIỂM 👍

| Ưu điểm | Mô tả |
|---------|-------|
| **Realtime** | Xử lý 60-80 FPS, phản hồi nhanh |
| **Smart Filtering** | Nhiều lớp filter giảm báo nhầm |
| **Đa kênh thông báo** | Telegram + FCM + Database |
| **Linh hoạt** | Nhiều tham số có thể điều chỉnh qua command line |
| **Đa nguồn input** | Webcam, video file, RTSP stream |
| **Temporal Filtering** | Xác nhận qua nhiều frame, tránh báo nhầm do chuyển động |

---

## 4. NHƯỢC ĐIỂM ❌

### 4.1 Vấn đề Model
| Vấn đề | Nguyên nhân |
|--------|-------------|
| Không detect người ở xa | Dataset thiếu data khoảng cách xa |
| Nhầm người ngồi/cúi = Fall | Thiếu negative samples |
| Nhầm đồ vật (ghế, bàn) = Fall | Dataset không đủ đa dạng |
| Chỉ detect được "đã nằm", không detect "đang té" | Thiếu temporal/sequence data |
| Confidence thấp với góc camera lạ | Training data không đại diện |

### 4.2 Vấn đề Code/System
| Vấn đề | Mô tả |
|--------|-------|
| Model size nhỏ (nano) | Khả năng học hạn chế |
| Không có người tracking | Mỗi frame detect độc lập |
| Chưa phân biệt nhiều người | Khó xác định ai té |

---

## 5. NHỮNG GÌ CHƯA LÀM ĐƯỢC 🔄

- [ ] Detect người ổn định ở mọi khoảng cách
- [ ] Phân biệt chính xác 100% Fall vs Not Fall
- [ ] Tracking từng người qua các frame
- [ ] Detect giai đoạn "đang té" (motion)
- [ ] Hoạt động tốt với camera góc ngang (webcam)
- [ ] API server hoàn chỉnh
- [ ] Mobile app hoàn thiện

---

## 6. HƯỚNG CẢI TIẾN 🚀

### 6.1 Cải tiến Model
| Hướng | Mô tả |
|-------|-------|
| **Retrain với data lớn hơn** | Thêm data đa dạng góc, khoảng cách, lighting |
| **Thêm negative samples** | Ảnh người ngồi, cúi, làm việc, đồ vật |
| **Dùng model lớn hơn** | YOLOv11m hoặc YOLOv11l thay vì nano |
| **Data augmentation** | Rotation, scale, blur, brightness |
| **Sequence training** | Train với video clips, không chỉ ảnh đơn lẻ |

### 6.2 Cải tiến Algorithm
| Hướng | Mô tả |
|-------|-------|
| **Person Tracking** | DeepSORT, ByteTrack để track từng người |
| **Pose Estimation** | Dùng skeleton detection để xác định tư thế |
| **Motion Analysis** | Phân tích chuyển động đột ngột |
| **2-Stage Detection** | Stage 1: Detect person → Stage 2: Classify Fall |
| **Ensemble Model** | Kết hợp nhiều model để tăng accuracy |

### 6.3 Tích hợp thêm
| Hướng | Mô tả |
|-------|-------|
| **Multi-camera** | Hỗ trợ nhiều camera cùng lúc |
| **Cloud deployment** | Deploy lên cloud (AWS, GCP) |
| **Edge computing** | Chạy trên Jetson Nano, Raspberry Pi |
| **Dashboard web** | Giao diện quản lý trên web |
| **Voice alert** | Cảnh báo bằng giọng nói |
| **Auto-call** | Tự động gọi điện khi phát hiện té |

---

## 7. KẾT LUẬN

### Đánh giá tổng thể: **6.5/10**

**Điểm mạnh:**
- Hệ thống hoạt động end-to-end
- Smart filtering giảm đáng kể false positive
- Tích hợp đầy đủ notification

**Điểm yếu chính:**
- Model chưa được train tốt với data đa dạng
- Độ chính xác phụ thuộc nhiều vào góc camera và khoảng cách

**Ưu tiên cải tiến:**
1. 🥇 **Retrain model** với dataset lớn hơn, đa dạng hơn
2. 🥈 Thêm **Person Tracking** để theo dõi từng người
3. 🥉 Tích hợp **Pose Estimation** để xác định tư thế chính xác

---

## 8. THÔNG TIN DỰ ÁN

- **Thư mục:** D:\Fall_Warning
- **File chính:** inference_smart.py
- **Model:** best.pt (YOLOv11)
- **Ngày báo cáo:** 18/01/2026

---

*Báo cáo được tạo tự động từ phân tích dự án*
