# AI_AUDIT_REPORT — mô hình và pipeline webcam

Ngày: 2026-11-13. Bổ sung cho [AUDIT_REPORT.md](AUDIT_REPORT.md) (audit phiên trước: lỗi vòng đời torch, kiến trúc server). Số liệu bên dưới do chạy thật trên máy này (CPU, RAM 7,4 GB); chỉ nêu những gì đã đo.

## 1. Mô hình hiện có (đọc từ trọng số và mã nguồn)

| Khóa | Tệp | Kiến trúc | Lớp (`model.names`) | Có người/điện thoại? |
| --- | --- | --- | --- | --- |
| `training` | `training_yolo.pt` | Ultralytics YOLO (detect) | Cheating, Normal | không (nhãn hành vi) |
| `yolo_duyen` | `yolo11_duyen.pt` | YOLO11 detect | Normal, Cheating | không |
| `gianlan` | `gian_lan.pt` | YOLO detect | cheating-behaviour, Normal, cheating, invigilator, submitting, cell-phones, person | có `person`, `cell-phones` |
| `paper` / `duyen` | `yolo11n-pose.pt` + KNN | YOLO11n-Pose + KNN (sklearn) | person (pose) → Normal/Suspicious | chỉ người |
| `person_coco` (mới) | `yolo11n_coco.pt` | YOLO11n, COCO pretrained (80 lớp) | có `person`, `cell phone` | có |

Chưa có dataset huấn luyện nào trong thư mục dự án; chỉ có `reports/yolo11_duyen/*.json` (số liệu test do tác giả mô hình công bố, mAP50 0,935; không có ảnh đi kèm nên **không kiểm chứng lại được**).

## 2. Phát hiện chính (có bằng chứng)

1. **Không mô hình cũ nào là bộ phát hiện người đáng tin.** Trên coco128 (254 người, 8 điện thoại có ground truth), `gian_lan` **không xuất ra một hộp `person`/`cell-phones` nào** ở conf 0,25 và cả 0,01 (xuất hộp ở lớp khác, ví dụ `Normal`: 65 hộp/40 ảnh). Đây là domain shift (huấn luyện trên ảnh phòng thi), không chứng minh nó tệ trong phòng thi thật. Pose+KNN: recall người nhỏ ≈ 0–3 %. Chi tiết: [MODEL_EVALUATION.md](MODEL_EVALUATION.md).
2. **Người ở xa bị mất vì resize.** Giao diện cũ yêu cầu webcam 640×480 và còn cắt khung gửi AI xuống tối đa 640 px, `imgsz` cố định 640. Người nhỏ (diện tích hộp <32² px, quy ước COCO) có recall 0,32 ở imgsz 640 và 0,45 ở 960 (model COCO). Đã sửa: webcam xin 1280×720, người dùng chọn độ phân giải gửi (640/960/1280) và `imgsz`.
3. **Cảnh báo "nhiều người" mặc định mức cao, không phân biệt chế độ**; ngoài ra chuỗi điều kiện bị ngắt bởi một khung nhấp nháy, nên cảnh báo điện thoại gần như không thể kích hoạt khi mô hình chớp tắt. Đã sửa: chế độ `single`/`room`, dung sai `grace`.
4. **Cảnh báo không mang độ tin cậy.** Đã thêm `confidence`, `frames`, `mode`.
5. **Model có thực sự chạy trên khung webcam không?** Có: `/api/live/frame` → `run_model` → `model.predict` (xác nhận bằng smoke test server thật; ảnh có người cho đếm `person` đúng nhóm). Chưa kiểm chứng với webcam vật lý (xem HANDOVER).
6. **Phần cứng:** GPU NVIDIA GTX 1650 có trên máy nhưng PyTorch cài là `2.15.0+cpu` (`cuda.is_available()` = False) → mọi suy luận/training chạy CPU. Độ trễ đo: 90–250 ms/khung (≈4–10 FPS tối đa phía server, trước khi cộng thời gian mã hóa/mạng của trình duyệt).
7. **Môi trường:** `.venv` hiện là Python 3.14.4 chứa nhiều wheel cp312 còn sót (scikit-learn, pandas, kiwisolver hỏng khi import). Đã cài lại các gói đó đúng bản 3.14 trong `.venv` (scikit-learn 1.9.1, pandas 2.3.3, kiwisolver). `pip check` vẫn báo vài gói "not supported on this platform" (chưa gây lỗi trong các bài test đã chạy). Trọng số KNN được lưu bằng sklearn 1.7.2, hiện nạp bằng 1.9.1 và **chạy được** (smoke test), nhưng độ chính xác sau đổi phiên bản chưa so sánh.
8. Tham số: ngưỡng conf mặc định webcam 0,5 (cao so với 0,25 dùng để đánh giá); IoU NMS mặc định Ultralytics 0,7. Chưa hiệu chỉnh trên dữ liệu phòng thi vì không có.

## 3. Tracking
Không có tracker đa khung cho từng đối tượng; hệ thống chỉ dùng "xác nhận nhiều khung + cooldown" ở mức luật cảnh báo. Chưa thêm tracking theo ID (không cần cho các tín hiệu hiện hỗ trợ; đồng thời tránh theo dõi cá nhân).
