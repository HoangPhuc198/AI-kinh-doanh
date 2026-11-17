# AUDIT_REPORT — ExamVision_Final

Ngày audit: 2026-11-13 · Branch: `main` (sạch, chỉ có `.omc/` chưa theo dõi) · Root: thư mục chứa `.git`.

## 1. Kiến trúc trước khi sửa (đọc từ source, không suy đoán)

| Lớp | Thực tế |
| --- | --- |
| Backend | `http.server.ThreadingHTTPServer` thuần (không Flask/FastAPI), [app/app.py](../../app/app.py) ~117 dòng, cổng 8765 |
| Frontend | HTML + JS thuần, một tệp mỗi trang: `index.html` (kiểm thử), `report.html` (báo cáo) |
| Mô hình | 3 YOLO detect (`training_yolo`, `gian_lan`, `yolo11_duyen`) và 2 mô hình YOLO11-Pose + KNN (`paper_knn.pkl`, `duyen_knn.joblib`) |
| Thư viện CV | ultralytics (pin 8.3.185), torch (không pin), opencv-python, scikit-learn 1.7.2, pandas, joblib |
| Pipeline | Ảnh/video tải lên (base64 JSON) → `/analyze` → `run_model` từng mô hình → JSON (box, nhãn, keypoints); video phát liên tục = trình duyệt cắt khung gửi `/analyze` lặp |
| Lưu trữ | Không có cơ sở dữ liệu, không lưu cảnh báo; chỉ tệp `reports/*.json` tĩnh |
| Webcam | Chưa có |

Lớp nhãn (đọc từ trọng số): `training_yolo` {Cheating, Normal}; `yolo11_duyen` {Normal, Cheating}; `gian_lan` {cheating-behaviour, Normal, cheating, invigilator, submitting, cell-phones, person}; Pose: `person`.
Chỉ `gian_lan` có lớp điện thoại. Không mô hình nào có nhãn hướng đầu/ánh nhìn.

## 2. Môi trường kiểm tra trên máy này

- Python hệ thống 3.12.10: torch 2.15.0+cpu, ultralytics 8.4.193 (lệch pin 8.3.185), opencv 5.0.0 (vi phạm `<5`), **thiếu** scikit-learn/pandas/joblib.
- `import torch` trong process sạch (`python -I`): **thành công**. Không có `importlib.reload`, `del sys.modules` hay circular import nào trong source (đã quét; test hồi quy giữ điều này).
- Chạy `run_model('duyen')` bằng Python hệ thống: `ModuleNotFoundError: joblib` (lỗi môi trường, không phải torch). `yolo_duyen` chạy được (lần đầu ~31 s gồm import torch + nạp mô hình).
- Tạo `.venv` riêng trong dự án theo `requirements.txt` để loại trừ xung đột gói.

## 3. Lỗi torch: "can only be initialized once per process"

Nguồn thông báo: `torch/__init__.py:48-54` (đọc, không sửa). Torch đặt cờ `sys._torch_import_started` ngay khi bắt đầu chạy; nếu `torch/__init__.py` bị thực thi lần thứ hai (lần đầu đã lỗi giữa chừng, hoặc `reload`) thì ném ImportError này.

**Cơ chế trong source (xác minh bằng đọc code + mô phỏng):**

1. `app.py` (cũ) dòng 22 và 29: `import paper_detector` / `from ultralytics import YOLO` được gọi **lười trong mỗi request**; `paper_detector.py` dòng 12 và 25 import `ultralytics`/`torch` (`get_models` lại bọc `lru_cache`).
2. `ThreadingHTTPServer` xử lý mỗi request ở một luồng; hai request đầu tiên có thể import torch đồng thời.
3. `app.py` dòng 109 bắt mọi `Exception` và trả `{'error': str(e)}`. Nếu lần import torch đầu thất bại (ví dụ DLL, thiếu RAM/pagefile — kiểu lỗi thường gặp trên Windows), mỗi request sau thử import lại và chỉ thấy câu "initialized once", **lỗi gốc biến mất**.
4. Phụ: `lru_cache` không an toàn đa luồng (nạp trùng), và `yolo11n-pose.pt` bị nạp hai lần (một lần cho mỗi KNN).

**Giới hạn của kết luận:** trên máy này `import torch` không lỗi nên **không thể tái hiện lỗi gốc ban đầu** (lần import thất bại đầu tiên). Cơ chế "import lười + retry + nuốt lỗi" đã được tái hiện bằng gói `torch` giả (test `test_failed_import_is_never_retried_and_keeps_root_cause`). Nếu lỗi tái diễn trên máy người dùng, thông báo khởi động mới sẽ in **lỗi gốc thật** của lần import đầu — đó là thông tin còn thiếu để xác định nguyên nhân cuối cùng.

## 4. Phương án sửa (đã thực hiện)

- [app/runtime.py](../../app/runtime.py): import torch + ultralytics **đúng một lần**, có khóa, ở luồng chính lúc khởi động; lỗi gốc được lưu và ném lại (`TorchUnavailable.__cause__`), không bao giờ import lại. Registry mô hình có khóa (mỗi trọng số nạp một lần), `inference_lock` tuần tự hóa suy luận.
- `app.py main()`: preflight torch; lỗi → in lỗi gốc, thoát mã 1.
- `paper_detector.py`: bỏ `lru_cache` và import cấp module; dùng registry; pose nạp một lần.
- Không đổi phiên bản torch/ultralytics (chưa có bằng chứng tương thích sai).

## 5. Vấn đề khác phát hiện (chưa đổi hành vi)

- `paper_detector.process_video` không được gọi ở đâu và cần `imageio_ffmpeg` (không có trong requirements) — mã cũ, giữ nguyên.
- Log cũ in `imgsz: 416` trong khi giá trị mặc định thực là 640 (đã bỏ dòng log gây hiểu nhầm).
- `/analyze` vẫn trả lỗi từng mô hình dưới dạng chuỗi (hành vi cũ, giữ để không đổi API).
- `requirements.txt` không pin torch; `opencv-python<5` đang bị vi phạm ở Python hệ thống của máy này.
