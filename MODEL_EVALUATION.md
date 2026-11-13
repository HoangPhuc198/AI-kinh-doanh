# MODEL_EVALUATION

Kịch bản: [training/eval_person.py](training/eval_person.py) → [reports/person_eval.json](reports/person_eval.json). Máy: CPU, torch 2.15.0+cpu.

## Tập đánh giá và giới hạn (đọc trước khi dùng số liệu)
- **coco128** (128 ảnh COCO, 254 người: 87 small <32², 82 medium, 85 large; 8 điện thoại). Có ground truth nhưng **không phải ảnh phòng thi/webcam** và **là tập con của COCO train2017**: `yolo11n_coco` đã được huấn luyện trên chính các ảnh này ⇒ số của nó **lạc quan**, không dùng làm bằng chứng tổng quát hóa.
- Chỉ số: ghép hộp tham lam IoU ≥ 0,5, conf 0,25, NMS IoU 0,7. Precision/recall/F1 ở một ngưỡng; **không phải mAP**. Mẫu nhỏ (8 điện thoại) nên số điện thoại gần như vô nghĩa thống kê.
- Chưa đánh giá: webcam độ phân giải thấp, đổi ánh sáng, che khuất, góc ảnh, người ở cuối phòng thi thật (không có dữ liệu có nhãn).

## Phát hiện người (lớp `person`)
| Mô hình | imgsz | Precision | Recall | F1 | Recall small / medium / large | FP | Latency mean / p95 (ms, CPU) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gian_lan (cũ) | 640–1280 | — (0 hộp) | 0 | — | 0 / 0 / 0 | 0 | 173 / 215 / 454 (mean) |
| yolo11n-pose (cũ, nền cho paper/duyen) | 640 | 0,839 | 0,453 | 0,588 | 0,00 / 0,42 / 0,95 | 22 | 134 / 276 |
| | 960 | 0,782 | 0,524 | 0,628 | 0,02 / 0,62 / 0,94 | 37 | 242 / 367 |
| | 1280 | 0,622 | 0,531 | 0,573 | 0,03 / 0,68 / 0,89 | 82 | 411 / 610 |
| yolo11n COCO (mới, *lạc quan*) | 640 | 0,797 | 0,650 | 0,716 | 0,32 / 0,70 / 0,94 | 42 | 88 / 134 |
| | 960 | 0,783 | 0,697 | 0,738 | 0,45 / 0,72 / 0,93 | 49 | 200 / 317 |
| | 1280 | 0,713 | 0,693 | 0,703 | 0,40 / 0,81 / 0,88 | 71 | 724 / 1197 |

Nhận xét có cơ sở: tăng `imgsz` 640→960 nâng recall người nhỏ/vừa mà precision giảm ít; 1280 tốn ~3,6× thời gian và tăng FP, không nâng recall tổng. Mặc định `person_coco` đặt 960. Người nhỏ vẫn bị bỏ sót hơn một nửa (recall 0,45): **không được coi là "phát hiện chính xác người ở xa"**. Không thử tiled inference/multi-scale (chưa có dữ liệu để chứng minh có ích, và nhân chi phí CPU).

## Điện thoại
`yolo11n_coco`: 1280 → recall 0,25, precision 0,5 (2/8 GT, 2 FP); 640/960 → 0 phát hiện đúng. `gian_lan` (`cell-phones`): 0. **Chưa có bằng chứng nào cho thấy hệ thống phát hiện điện thoại đủ tin cậy**; cảnh báo điện thoại cần xem như tín hiệu rất yếu.

## So sánh model cũ/mới và quyết định
Không fine-tune xong (xem [TRAINING_REPORT.md](TRAINING_REPORT.md)), nên **không có "model mới" được chứng minh cải thiện** trên tập kiểm thử riêng và không thay model nào. `person_coco` chỉ được **thêm** vào danh sách (cạnh 5 model cũ, giữ nguyên để rollback) như bộ đếm người/điện thoại tham khảo. Việc nó vượt `gian_lan`/pose trên coco128 chỉ phản ánh domain shift và rò rỉ train/test nêu trên.

## Hiệu năng webcam
Server-side đo qua `tests/smoke_server.py` (CPU): `person_coco` @960 ≈ 190–280 ms/khung; các model YOLO cũ ≈ 85–200 ms. Vòng lặp trình duyệt tuần tự (khung sau chỉ gửi khi khung trước xong) nên không có hàng đợi tăng vô hạn; FPS thực = min(FPS tối đa, 1/độ trễ). **FPS với webcam thật chưa đo.**
