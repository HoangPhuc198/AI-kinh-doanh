# HANDOVER — ExamVision_Final

Trạng thái: các thay đổi **chưa commit, chưa push** trên branch `main` (người dùng chưa yêu cầu commit). Xem `git status`.

## Đã làm
- Audit: [AUDIT_REPORT.md](AUDIT_REPORT.md).
- Sửa vòng đời PyTorch: [app/runtime.py](app/runtime.py) (import torch/ultralytics một lần, lưu lỗi gốc, registry mô hình có khóa, `inference_lock`); `app.py main()` preflight; `paper_detector.py` dùng registry.
- Tách [app/inference.py](app/inference.py) (`MODELS`, `run_model`, `warm_model`) khỏi `app.py`.
- Webcam: trang [app/live.html](app/live.html) tại `/giam-sat`; API `/api/live/{status,start,frame,control,review,alerts}` trong `app.py`; luật cảnh báo [app/live/alerts.py](app/live/alerts.py); phiên [app/live/session.py](app/live/session.py).
- Tests: `tests/` (15 unittest) + `tests/smoke_server.py` (thủ công, model thật).

## Nguyên nhân lỗi torch (kết luận có giới hạn)
- Xác minh: `.venv` trong đường dẫn OneDrive dài làm pip cài torch dở dang (MAX_PATH) → `DLL load failed while importing _C`; import lại trong cùng process → "can only be initialized once". Mã cũ import lười trong từng request và nuốt lỗi nên che lỗi gốc.
- Chưa xác minh: lỗi gốc đúng trên máy người dùng (Python hệ thống import torch bình thường). Nếu tái diễn, dòng `KHÔNG import được PyTorch/Ultralytics` lúc khởi động in lỗi gốc.

## Môi trường đang dùng
`.venv` được tạo với `--system-site-packages` (Python 3.12.10) để dùng torch 2.15.0+cpu / ultralytics 8.4.193 của hệ thống; chỉ cài thêm scikit-learn 1.7.2, pandas 2.3.3, joblib, opencv-python 4.14 vào venv. Lệch pin `ultralytics==8.3.185` trong requirements.txt (không phát hiện lỗi). `.venv/` đã gitignore.

## Kết quả kiểm thử (2026-11-13)
- `.venv\Scripts\python.exe -m unittest discover -s tests -t . -v`: 15/15 đạt.
- `tests\smoke_server.py` (server + 5 model thật, ảnh tổng hợp): đạt; mỗi trọng số nạp đúng một lần; chỉ kiểm tra đường ống.
- Trình duyệt (chrome-devtools) với `getUserMedia` **giả lập bằng canvas**: Bật → Bắt đầu → cảnh báo "Không thấy người" → Tạm dừng/Tiếp tục → Dừng đều đúng; track kết thúc, phiên server giải phóng. Đường lỗi camera (NotReadableError) hiện thông báo đúng.
- **Chưa xác minh**: webcam vật lý thật, hiệu năng/độ chính xác với người thật, từ chối quyền camera thật (chỉ đọc mã), ngắt kết nối camera khi đang chạy (có xử lý `track.ended`, chưa thử thật), GPU.
- Luồng cũ `/analyze` (ảnh) chạy với cả 5 model; **luồng video cũ chưa được chạy lại** sau refactor (chỉ logic chung `run_model` đã được dùng).

## Giới hạn / việc tiếp theo
- Nhãn nào là "người"/"điện thoại"/"nghi vấn" là giả định (`SIGNAL_LABELS`); cần kiểm chứng trên video thật và chỉnh ngưỡng (`DEFAULT_RULES`).
- Không có phát hiện hướng đầu/ánh nhìn (không có model kiểm chứng).
- Suy luận lần đầu chậm (10–29 s trong smoke test lúc máy bận) vì khởi động các lớp; có thể thêm warm-up predict trong `warm_model`.
- Suy luận CPU được tuần tự hóa bằng một khóa; FPS thực tế ~2–5 trên máy này.
- `MANIFEST.json` chưa được cập nhật hash cho các file đã đổi/mới.
- Chưa refactor lớn hơn (`app.py` vẫn chứa `report_data` và handler) — cố ý, để tránh đổi cấu trúc không cần thiết. `paper_detector.process_video` là mã cũ không dùng.
- Cân nhắc đưa dự án ra đường dẫn ngắn hoặc bật Long Paths để `pip install -r requirements.txt` đầy đủ vào `.venv`.

## Lệnh (Windows PowerShell, tại thư mục dự án)
```powershell
.\.venv\Scripts\python.exe app\app.py          # http://127.0.0.1:8765 ; webcam: /giam-sat
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
.\.venv\Scripts\python.exe tests\smoke_server.py
```

## Cập nhật phiên 2 (2026-11-13) — người ở xa, huấn luyện, chế độ cảnh báo
Xem [AI_AUDIT_REPORT.md](AI_AUDIT_REPORT.md), [MODEL_EVALUATION.md](MODEL_EVALUATION.md), [DATASET_REPORT.md](DATASET_REPORT.md), [TRAINING_REPORT.md](TRAINING_REPORT.md).

**Đã làm:** thêm mô hình `person_coco` (YOLO11n COCO, `models/yolo11n_coco.pt`; các model cũ giữ nguyên để rollback); `run_model(..., imgsz, iou)` + warm-up + chọn CUDA nếu có; webcam xin 1280×720, chọn độ phân giải gửi và `imgsz` trong UI; chế độ `single` / `room` (room không cảnh báo nhiều người/không thấy người); cảnh báo có `confidence`, `frames`, `mode`; dung sai `grace` chống nhấp nháy; `training/{eval_person,prepare_dataset,train}.py`; test: 20/20 unittest, smoke server thật (6 model), luồng upload video cũ chạy lại đạt (ảnh tĩnh lặp thành video, chỉ kiểm tra không lỗi).

**Chưa làm / không được khẳng định:**
- **Không có training thật** (không dataset phù hợp đã duyệt giấy phép, torch bản CPU dù có GTX 1650). Pipeline train chỉ chạy smoke 1 epoch; `--resume` chưa thử.
- Độ chính xác người xa: recall người nhỏ chỉ ~0,45 (coco128, model COCO, số liệu lạc quan vì rò rỉ train). Không có dữ liệu webcam/phòng thi có nhãn. Điện thoại: gần như không phát hiện được (recall 0–0,25 trên 8 mẫu).
- Webcam vật lý, FPS thật, memory leak khi dừng webcam lâu dài: chưa kiểm thử.
- Không đánh giá mAP (chỉ P/R/F1 ở một ngưỡng). Không tiled/multi-scale inference.
- `.venv` là Python 3.14.4 (không phải 3.12 như ghi trên); đã cài lại scikit-learn/scipy/pandas/kiwisolver/narwhals đúng bản trong `.venv`; `pip check` còn cảnh báo vài gói cp312 sót. Nên tạo lại `.venv` sạch ở đường dẫn ngắn.
- OpenCV `imread` với đường dẫn Unicode (thư mục "Đại học") có thể thất bại; app dùng `imdecode` từ bytes nên không bị ảnh hưởng.
