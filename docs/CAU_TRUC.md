# Bản đồ dự án ExamVision

**Mô hình chính của dự án là ExamVision Ensemble v2** (`app/ensemble_v2.py`, khóa `ensemble_v2`). v2 **tự chứa**: không import `ensemble.py` (v1). Mô tả, lỗi v1 đã sửa và số đo: [ENSEMBLE_V2.md](ENSEMBLE_V2.md). Mọi thứ không thuộc v2 nằm ở mục "Bản cũ" ở cuối và được giữ lại chỉ để quay lại.

## 1. Mô hình chính: Ensemble v2

**Luồng xử lý:** khung hình → `app/ensemble_v2.py` (người + điện thoại COCO, quay đầu YOLO11-Pose, Training YOLO) → `app/judge.py` (kết luận theo thời gian: Bình thường / Nghi ngờ / Gian lận) → `app/live/alerts.py` (cảnh báo) → giao diện.

| File | Vai trò |
|---|---|
| `app/ensemble_v2.py` | **v2**: luật gộp, bộ leo thang, đo quay đầu, chạy mô hình. Tự chứa; các hàm phụ chép từ v1 được đánh dấu "chép từ ensemble.py" |
| `models/yolo11m_coco.pt` (GPU) hoặc `yolo11n_coco.pt` (CPU), `models/yolo11n-pose.pt`, `models/training_yolo.pt` | Trọng số v2 dùng |
| `tests/test_ensemble_v2.py` | Test của v2, gồm `DifferentialTests` (hàm chép từ v1 phải cho kết quả y hệt) và `SelfContainedTests` (v2 không nạp v1) |

**Dùng chung với phần còn lại của ứng dụng** (v2 phụ thuộc vào đây, không phụ thuộc v1):

| File | Vai trò |
|---|---|
| `app/judge.py` | Kết luận theo thời gian + `Settings` (ngưỡng người dùng chỉnh được) |
| `app/runtime.py` | Nạp PyTorch/mô hình một lần, chọn GPU/CPU |
| `app/inference.py` | Đăng ký mô hình (`MODELS`, `ensemble_v2`), `run_model`, `warm_model` |
| `app/app.py`, `app/live/` | Máy chủ web, phiên webcam (`session.py`), luật cảnh báo (`alerts.py`) |
| `app/index.html`, `app/live.html` | Giao diện kiểm thử ảnh/video và giám sát webcam |

**Đánh giá v2** (số liệu làm mốc, ngưỡng chọn trên một phần dữ liệu và kiểm tra trên phần khác):

| File | Vai trò |
|---|---|
| `training/group_eval.py`, `eval_by_group.py`, `pool_by_group.py` | Đánh giá theo nhóm (ảnh gốc + bản gần trùng, hoặc theo video): khoảng tin cậy theo nhóm, rò rỉ giữa split, từng nguồn riêng. `eval_by_group.py` cần `eval_external.py` (bộ ngưỡng Chặt/Cân bằng/Nhạy) |
| `training/pose_branch_study.py`, `pose_branch_report.py` | Đo tín hiệu quay đầu theo nhánh keypoint trên ảnh có nhãn (dùng hàm đo của v1 để so sánh) |
| `tests/test_group_eval.py`, `tests/test_judge.py` | Test công cụ đo và bộ kết luận |
| `reports/eval_by_group_*.json` | Số đo theo nhóm: từng bộ, gộp, v1 và v2 |
| `docs/ENSEMBLE_V2.md` | Lỗi v1 và cách sửa, cách chọn nguồn, kết quả, giới hạn |

## 2. Chung toàn dự án

| Thư mục / file | Chứa gì | Khi nào cần mở |
|---|---|---|
| `README.md` | Giới thiệu, cách cài và chạy | Bắt đầu từ đây |
| `HANDOVER.md` | Nhật ký các phiên làm việc, việc đã/chưa làm | Muốn biết tình trạng hiện tại |
| `requirements.txt` | Thư viện Python | Cài môi trường |
| `models/` | Trọng số mô hình, xem `models/README.md` | Thêm/đổi mô hình |
| `datasets/` | Dữ liệu huấn luyện/đánh giá, xem `datasets/README.md` | Chuẩn bị dữ liệu |
| `reports/` | Số liệu đo (JSON) và kết quả phân tích video (`video_<tên>/`). **Ứng dụng đọc `yolo11_duyen/`, `knn_duyen/`, `snapshots.json` ở đây, đừng đổi tên.** | Xem kết quả đo |
| `docs/` | Tài liệu; `docs/HUONG_DAN_THEM_MODEL.md`, `docs/HUONG_DAN_GITHUB.md` hướng dẫn thêm mô hình, đưa lên GitHub | |
| `docs/reports/` | Báo cáo audit, dữ liệu, huấn luyện, đánh giá mô hình | Hiểu vì sao chọn mô hình, giới hạn |
| `runs/`, `logs/` | Kết quả huấn luyện; log chạy dài (không đưa lên git). Nhật ký debug của v2: `logs/ensemble_v2_debug.jsonl` (bật bằng `EXAMVISION_DEBUG=1`) | Chỉ khi huấn luyện lại / tra lỗi |

## 3. Bản cũ (giữ để quay lại, không thuộc v2)

| File | Vai trò |
|---|---|
| `app/ensemble.py` | Ensemble v1 (khóa `ensemble`): người + YOLO11 Duyên + gianlan + KNN + quay đầu + điện thoại |
| `app/paper_detector.py` | Mô hình KNN theo tư thế (`paper`, `duyen`) |
| `models/gian_lan.pt`, `yolo11_duyen.pt`, `duyen_knn.joblib`, `paper_knn.pkl`, `yolo11_person_phone.pt` | Trọng số của v1 và các mô hình lẻ |
| `tests/test_ensemble.py` | Test của v1 |
| `training/analyze_video.py`, `compare_models.py`, `final_eval.py`, `ensemble_check.py`, `eval_person.py`, `frame_alarm_eval.py` | Công cụ cũ đo/phân tích (một số dùng v1) |

## Lệnh thường dùng (PowerShell, tại thư mục gốc)
```powershell
.\.venv\Scripts\python.exe app\app.py                                  # http://127.0.0.1:8765 ; webcam: /giam-sat ; chọn "ExamVision Ensemble v2"
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v        # chạy toàn bộ test
.\.venv\Scripts\python.exe -m unittest tests.test_ensemble_v2 -v        # chỉ test của v2
.\.venv\Scripts\python.exe training\eval_by_group.py "datasets\<bộ>" --model ensemble_v2   # đo THEO NHÓM
.\.venv\Scripts\python.exe training\pool_by_group.py --model ensemble_v2                     # gộp các bộ đã đo, so sánh từng nguồn
```
