# models/ — trọng số mô hình

Ứng dụng tải theo **đường dẫn cố định** khai báo trong `app/inference.py` (`MODELS`), nên đừng đổi tên file.

| File | Dùng cho |
|---|---|
| `yolo11n_coco.pt`, `yolo11m_coco.pt`, `yolo11s_coco.pt` | Phát hiện người + điện thoại (COCO). Có GPU thì dùng `yolo11m_coco.pt` |
| `yolo11_duyen.pt` | Hành vi nghi vấn (YOLO11, dataset Duyên) |
| `gian_lan.pt` | Hành vi gian lận (nhãn Cheating) |
| `training_yolo.pt` | Mô hình training YOLO (không dùng trong Ensemble) |
| `yolo11n-pose.pt` | Keypoint để tính quay/nghiêng đầu và cho KNN |
| `duyen_knn.joblib`, `paper_knn.pkl` | Bộ phân loại KNN theo tư thế |
| `yolo11_person_phone.pt` | Fine-tune thử người + điện thoại (kém hơn COCO gốc, xem `docs/reports/TRAINING_REPORT.md`) |

Thêm mô hình mới: xem `docs/HUONG_DAN_THEM_MODEL.md`.
