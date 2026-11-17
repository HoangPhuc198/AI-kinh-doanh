# Bản đồ dự án ExamVision

| Thư mục / file | Chứa gì | Khi nào cần mở |
|---|---|---|
| `README.md` | Giới thiệu, cách cài và chạy | Bắt đầu từ đây |
| `HANDOVER.md` | Nhật ký các phiên làm việc, việc đã/chưa làm | Muốn biết tình trạng hiện tại |
| `requirements.txt` | Thư viện Python | Cài môi trường |
| `app/` | **Ứng dụng web** (`app.py`, giao diện `index.html`, `live.html`, ensemble, bộ kết luận `judge.py`) | Sửa tính năng, thuật toán |
| `app/live/` | Phiên webcam và luật cảnh báo | Sửa cảnh báo |
| `models/` | Trọng số mô hình, xem `models/README.md` | Thêm/đổi mô hình |
| `training/` | Script chuẩn bị dữ liệu, huấn luyện, đánh giá, **phân tích video** (`analyze_video.py`) | Huấn luyện, đo chất lượng |
| `app/ensemble_v2.py` | **Ensemble v2**: bản sửa lỗi của v1 (nguồn: người + điện thoại COCO, quay đầu YOLO11-Pose, Training YOLO). v1 (`ensemble.py`) giữ nguyên để quay lại. Danh sách lỗi v1 và cách sửa: đầu file | Sửa luật gộp của v2 |
| `training/group_eval.py`, `eval_by_group.py`, `pool_by_group.py` | Đánh giá theo nhóm (video, hoặc ảnh gốc + bản gần trùng): khoảng tin cậy theo nhóm, rò rỉ giữa split, từng nguồn riêng | Đo trước/sau khi đổi mô hình hay luật |
| `training/pose_branch_study.py`, `pose_branch_report.py` | Đo tín hiệu quay đầu theo nhánh keypoint (hai mắt / một mắt) trên ảnh có nhãn; ngưỡng chọn trên train, đo trên valid+test đã loại ảnh gần trùng | Đổi cách đo quay đầu |
| `tests/` | Test tự động (`unittest`) và smoke test | Kiểm tra sau khi sửa |
| `datasets/` | Dữ liệu huấn luyện/đánh giá, xem `datasets/README.md` | Chuẩn bị dữ liệu |
| `reports/` | Số liệu đo (JSON) và kết quả phân tích video (`video_<tên>/`). **Ứng dụng đọc `yolo11_duyen/`, `knn_duyen/`, `snapshots.json` ở đây, đừng đổi tên.** | Xem kết quả đo |
| `docs/` | Tài liệu | |
| `docs/HUONG_DAN_THEM_MODEL.md`, `docs/HUONG_DAN_GITHUB.md` | Hướng dẫn thêm mô hình, đưa lên GitHub | |
| `docs/reports/` | Báo cáo audit, dữ liệu, huấn luyện, đánh giá mô hình | Hiểu vì sao chọn mô hình, giới hạn |
| `runs/` | Kết quả huấn luyện (trọng số, log của Ultralytics) | Chỉ khi huấn luyện lại |
| `logs/` | Log chạy dài (không đưa lên git) | Tra lỗi |

## Lệnh thường dùng (PowerShell, tại thư mục gốc)
```powershell
.\.venv\Scripts\python.exe app\app.py                                  # http://127.0.0.1:8765 ; webcam: /giam-sat
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v        # chạy test
.\.venv\Scripts\python.exe training\analyze_video.py "video.mp4"        # phân tích video -> reports\video_<tên>\
.\.venv\Scripts\python.exe training\eval_by_group.py "datasets\<bộ>" --model ensemble_v2   # đo THEO NHÓM: khoảng tin cậy theo nhóm, rò rỉ giữa split
.\.venv\Scripts\python.exe training\pool_by_group.py --model ensemble_v2                     # gộp các bộ đã đo, so sánh từng nguồn
```

## Luồng xử lý chính
ảnh/khung hình → `app/ensemble.py` (người + 3 mô hình hành vi + quay đầu + điện thoại) → `app/judge.py` (kết luận theo thời gian: Bình thường / Nghi ngờ / Gian lận) → `app/live/alerts.py` (cảnh báo) → giao diện.
