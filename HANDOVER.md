# HANDOVER — ExamVision_Final

Trạng thái: các thay đổi **chưa commit, chưa push** trên branch `main` (người dùng chưa yêu cầu commit). Xem `git status`.

## Đã làm
- Audit: [AUDIT_REPORT.md](docs/reports/AUDIT_REPORT.md).
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
Xem [AI_AUDIT_REPORT.md](docs/reports/AI_AUDIT_REPORT.md), [MODEL_EVALUATION.md](docs/reports/MODEL_EVALUATION.md), [DATASET_REPORT.md](docs/reports/DATASET_REPORT.md), [TRAINING_REPORT.md](docs/reports/TRAINING_REPORT.md).

**Đã làm:** thêm mô hình `person_coco` (YOLO11n COCO, `models/yolo11n_coco.pt`; các model cũ giữ nguyên để rollback); `run_model(..., imgsz, iou)` + warm-up + chọn CUDA nếu có; webcam xin 1280×720, chọn độ phân giải gửi và `imgsz` trong UI; chế độ `single` / `room` (room không cảnh báo nhiều người/không thấy người); cảnh báo có `confidence`, `frames`, `mode`; dung sai `grace` chống nhấp nháy; `training/{eval_person,prepare_dataset,train}.py`; test: 20/20 unittest, smoke server thật (6 model), luồng upload video cũ chạy lại đạt (ảnh tĩnh lặp thành video, chỉ kiểm tra không lỗi).

**Chưa làm / không được khẳng định:**
- **Không có training thật** (không dataset phù hợp đã duyệt giấy phép, torch bản CPU dù có GTX 1650). Pipeline train chỉ chạy smoke 1 epoch; `--resume` chưa thử.
- Độ chính xác người xa: recall người nhỏ chỉ ~0,45 (coco128, model COCO, số liệu lạc quan vì rò rỉ train). Không có dữ liệu webcam/phòng thi có nhãn. Điện thoại: gần như không phát hiện được (recall 0–0,25 trên 8 mẫu).
- Webcam vật lý, FPS thật, memory leak khi dừng webcam lâu dài: chưa kiểm thử.
- Không đánh giá mAP (chỉ P/R/F1 ở một ngưỡng). Không tiled/multi-scale inference.
- `.venv` là Python 3.14.4 (không phải 3.12 như ghi trên); đã cài lại scikit-learn/scipy/pandas/kiwisolver/narwhals đúng bản trong `.venv`; `pip check` còn cảnh báo vài gói cp312 sót. Nên tạo lại `.venv` sạch ở đường dẫn ngắn.
- OpenCV `imread` với đường dẫn Unicode (thư mục "Đại học") có thể thất bại; app dùng `imdecode` từ bytes nên không bị ảnh hưởng.

## Cập nhật phiên 3 (2026-11-15) — PyTorch CUDA + nâng độ chính xác
- `.venv`: torch 2.14.1+cu130 / torchvision 0.29.1+cu130 (GTX 1650, `cuda.is_available()=True`). Mọi tệp tải về/cache chỉ nằm trong thư mục dự án (`.pipcache/`, `.tmp/` đã gitignore).
- **FP16 tắt mặc định**: trên GTX 1650 FP16 cho kết quả sai (yolo11m recall người 0,013 ở 640 so với 0,63 ở FP32). Bật thử bằng `EXAMVISION_HALF=1`.
- `person_coco` tự dùng `models/yolo11m_coco.pt` (tải từ ultralytics assets v8.3.0) khi có GPU; `EXAMVISION_COCO=n` ép về YOLO11n. Ensemble trên GPU chạy 960 px và cho phép phóng to khung nhỏ.
- Đo trên `datasets/coco_person_phone` test (153 ảnh, conf 0,35, FP32): F1 người YOLO11n@640 0,631 → YOLO11m@960 0,760; recall người nhỏ 0,23 → 0,50; recall điện thoại 0,36 → 0,58. Chi tiết: `reports/gpu_model_sweep_test.json`.
- Ensemble (`training/ensemble_check.py test`): độ trễ trung bình 1129 ms (CPU) → 136 ms; số người phát hiện 721 → 799; cờ "Nghi vấn" 7 → 3 ảnh; "Cần chú ý" 9 → 17 ảnh (ảnh COCO không có gian lận nên cờ là báo nhầm tiềm năng). Test: 51/51 đạt.
- Chưa làm: huấn luyện lại trên GPU (không có dữ liệu thi cử có nhãn mới), kiểm thử webcam thật.

## Cập nhật phiên 4 (2026-11-16) — kết luận Gian lận / Nghi ngờ / Bình thường, ngưỡng chỉnh được
- `app/judge.py`: theo dõi từng người qua khung (ghép IoU), cửa sổ 8 s; **Gian lận** = nhiều nguồn đồng thuận kéo dài ≥ `min_seconds`; **Nghi ngờ** = dấu hiệu chưa đủ; một khung đơn lẻ không bao giờ là Gian lận. Điện thoại/quay đầu đứng một mình tối đa là Nghi ngờ.
- Nguồn mới `pose` (quay/nghiêng đầu từ keypoint YOLO11-Pose) trong `app/ensemble.py`.
- 6 ngưỡng (`judge.Settings`) chỉnh được trên webcam (`/giam-sat`, cả khi đang chạy) và video (`/`), có 3 mẫu Nhạy/Cân bằng/Chặt. Video gửi `stream` + `video_time` để kết luận theo thời gian.
- Cảnh báo mới `cheating` (mức cao) khi có kết luận Gian lận. Test: 69/69.
- **Chưa đo được precision/recall gian lận thật**: dự án không có video thi có nhãn. Chỉ đo được tỉ lệ cờ trên ảnh COCO (không có gian lận): 12/799 người bị "Nghi ngờ", 0 "Gian lận". Ngưỡng mặc định là heuristic.
- Phiên 4 (tiếp): thanh Độ nhạy (nội suy Chặt↔Cân bằng↔Nhạy) + mục Nâng cao; thanh "Ngưỡng" cũ trên webcam = ngưỡng bằng chứng khi chọn Ensemble. Phương án A: bằng chứng 0,55; Nghi ngờ cần ≥2 khung liên tiếp (ảnh đơn: 1 khung); về Bình thường sau 1 s (`recover_seconds`). Test 73/73. Lưu ý khởi động lại server: tiến trình ghi `app/app.py` (dấu `/`); nhiều server cũ có thể cùng chiếm cổng 8765 trên Windows.
- Phiên 4 (video thật 7 s): thêm tín hiệu `pair` (hai người ngồi cạnh nhau cùng quay đầu VỀ PHÍA NHAU, `ensemble.pair_evidence`), `min_seconds` mặc định 3→1,5 (Chặt 3 / Nhạy 1), `SUSPECT_SCORE` 0,30→0,15. Công cụ `training/analyze_video.py` chạy ensemble trên video (có/không nhãn). Chưa có tín hiệu "đưa bài/chép bài". Đã thấy kết quả video này khi thiết kế nên nó KHÔNG còn là tập kiểm tra độc lập. Test 82/82.
- Phiên 4 (2 video thật, video 2 có nhãn tay do người dùng mô tả): bằng chứng chia theo NHÓM ĐỘC LẬP (KNN và quay đầu cùng nhóm keypoint nên không tính là 2 nguồn); quay đầu trừ nhiễu keypoint theo kích thước mặt (KPT_NOISE_PX=1, pose 1280 trên GPU); "Nghi ngờ" = bằng chứng liên tục `suspect_seconds` (1 s). Kết quả: bắt được hai người quay về nhau ở video 1; video 2 (cái liếc ~1 s): precision cao nhưng recall thấp ở mức Cân bằng (chỉ khá hơn ở mức Nhạy). Chỉnh trên 2 video là over-fit nguy cơ cao: cần thêm video + nhãn từng người, và dùng 1 video làm tập kiểm tra độc lập. Test 86/86.
- Phiên 4 (dữ liệu ngoài, 2026-11-16): tải 4 dataset Roboflow CC BY 4.0 vào `datasets/` (+ `datasets/external/Graduation Project...` = dữ liệu huấn luyện `training_yolo`, 6.690 ảnh). Công cụ: `training/eval_external.py` (đo Ensemble trên ảnh có nhãn), `training/compare_models.py` (so từng mô hình; `--exclude` loại ảnh trùng, `--only`). Kết quả chỉ để tham khảo: nhãn ngoài nhiễu/rộng, ảnh đơn (không đo được thời gian). Phát hiện: tập test của Graduation trùng ~86% ảnh train (điểm Roboflow cao giả); 5-21% ảnh valid/test của 3 bộ ngoài gần y hệt ảnh train đó. Sau khi loại ảnh trùng, `training_yolo`@0,5 vẫn bắt nhiều hơn Ensemble (báo nhầm ~8%). **Thử nghiệm**: thêm `training` làm NGUỒN PHỤ (trọng số 0,15, ngưỡng 0,5, chỉ chạy khi có nguồn mạnh nghi vấn, không tạo người, không tự gây Cần chú ý, không nằm trong mẫu số điểm). Đo (ảnh ngoài, đã loại trùng, Cân bằng): malpractice2 báo nhầm 2,5→2,9% / bắt 4,3→6,9%; exam_cheating 1,0→1,0% / 15→33%; video 2 recall Cân bằng 0,12→0,15 (Nhạy không đổi 0,65); video 1 không đổi. Chậm thêm ~40 ms/ảnh. Rủi ro: có thể học đúng kiểu nhãn "cheating" rộng của dữ liệu ngoài; Nhạy trên malpractice2 báo nhầm 4,3%. Test 90/90.
- Phiên 4 (giao diện webcam, để quay demo): `app/live.html` có bảng trạng thái lớn đổi màu (Bình thường xanh / Nghi ngờ vàng / Gian lận đỏ nhấp nháy, viền khung hình đổi màu theo), thẻ từng người kèm lý do bằng tiếng Việt thường và thanh mức nghi ngờ, mục "Diễn biến trong phiên" (mốc mm:ss mỗi lần một người đổi nhãn), tùy chọn "Âm báo" (bíp khi chuyển lên Nghi ngờ/Gian lận), nhãn trên khung lớn hơn. Sửa lỗi: khi không thấy người nào kết luận cũ đứng yên (nay hiện "KHÔNG THẤY NGƯỜI"); lỗi hiển thị trong vòng gửi khung được bắt (`try/catch`) để không làm dừng vòng. Không đổi API/server. Công tắc gỡ lỗi: `EXAMVISION_DEBUG=1` ghi `logs/ensemble_debug.jsonl` (chỉ chỉ số, không ảnh); `EXAMVISION_TRAINING=0` tắt nguồn phụ Training YOLO.
