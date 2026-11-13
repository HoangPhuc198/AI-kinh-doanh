# DATASET_REPORT

## Trong dự án
Không có dataset nào (chỉ có trọng số và JSON số liệu). Nguồn dữ liệu huấn luyện của các model cũ không được ghi trong repo (ngoại trừ [reports/yolo11_duyen/dataset_summary.json](reports/yolo11_duyen/dataset_summary.json), chỉ là thống kê).

## Đã tải (cho đánh giá, không phải huấn luyện chính)
- **coco128** (`ultralytics/assets` v0.0.0, 7 MB, 128 ảnh, 2 ảnh không nhãn). Đặt tại `datasets/coco128` (đã gitignore). Nhãn bắt nguồn từ COCO (chú thích CC BY 4.0); ảnh COCO đến từ Flickr với giấy phép từng ảnh khác nhau — **chưa kiểm tra từng ảnh**, chỉ dùng nội bộ để đánh giá, không tái phân phối. Gói kèm tệp LICENSE GPL‑3.0 của repo Ultralytics.
- Chia thử bằng `prepare_dataset.py` → `datasets/coco128_split` (88/19/19 ảnh), 0 dòng nhãn lỗi, 0 ảnh trùng; **chỉ để kiểm tra pipeline**.
- Trọng số `yolo11n_coco.pt` (Ultralytics, AGPL‑3.0, 5,6 MB).

## Chưa tải / đề xuất cho huấn luyện thật (chưa tải gì; cần bạn xác nhận giấy phép và dung lượng)
Ổ E: còn ~207 GB trống, RAM 7,4 GB, không có GPU dùng được trong PyTorch hiện tại.
| Dataset | Dùng cho | Ghi chú cần xác minh trước khi tải |
| --- | --- | --- |
| COCO 2017 (lọc `person`, `cell phone`) | người, điện thoại | ~19 GB train; ảnh Flickr giấy phép hỗn hợp, nhãn CC BY 4.0 |
| CrowdHuman | nhiều người, che khuất | giấy phép phi thương mại/đăng ký — cần đọc điều khoản |
| WiderPerson / VisDrone / TinyPerson | người nhỏ/xa | mỗi bộ điều khoản riêng, kiểm tra trước |
| Ảnh webcam/phòng thi tự thu | miền đích | cần sự đồng ý của người trong ảnh; nên có để hiệu chỉnh ngưỡng |

Không gán nhãn "gian lận" từ sự hiện diện điện thoại/nhiều người/tư thế đầu; các lớp huấn luyện chỉ là đối tượng (`person`, `cell phone`).

## Công cụ
[training/prepare_dataset.py](training/prepare_dataset.py): kiểm nhãn (lớp ngoài phạm vi, tọa độ ngoài [0,1]), loại trùng md5, gộp ảnh gần giống (dHash), chia theo nhóm (không rò rỉ khung gần giống giữa train/val/test), sinh `data.yaml` và `prepare_report.json`. Đã chạy trên coco128. Chưa thử trên chuỗi khung video thật.
