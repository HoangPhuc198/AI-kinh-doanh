# MODEL_EVALUATION

Kịch bản: [training/eval_person.py](../../training/eval_person.py) → [reports/person_eval.json](../../reports/person_eval.json). Máy: CPU, torch 2.15.0+cpu.

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

## Dataset mới `coco_person_phone` — tập test 153 ảnh (891 người, 31 điện thoại), COCO val2017 (model gốc chưa từng thấy)
Số đo bằng `training/eval_person.py --split test`; kết quả: [reports/eval_baseline_cocopp_test.json](../../reports/eval_baseline_cocopp_test.json), [reports/eval_person_phone_v1_test.json](../../reports/eval_person_phone_v1_test.json). Ngưỡng 0,25, IoU ghép 0,5, CPU.

| Mô hình | imgsz | Precision người | Recall người | Recall small / medium / large | Recall điện thoại | Độ trễ TB (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11n COCO (gốc) | 640 | 0,791 | 0,553 | 0,29 / 0,71 / 0,91 | 0,36 | 120–153 |
| yolo11n COCO (gốc) | 960 | 0,729 | 0,668 | 0,48 / 0,80 / 0,89 | 0,45 | 205–339 |
| yolo11n-pose | 640 | 0,900 | 0,345 | 0,00 / 0,48 / 0,92 | — | 97 |
| yolo11n-pose | 960 | 0,798 | 0,407 | 0,02 / 0,65 / 0,92 | — | 236 |
| **person_phone v1 (fine-tune)** | 640 | 0,973 | **0,040** | 0,00 / 0,05 / 0,11 | **0,00** | 204 |
| **person_phone v1 (fine-tune)** | 960 | 1,000 | **0,020** | 0,00 / 0,02 / 0,07 | **0,00** | 339 |

(Độ trễ chênh giữa hai lần đo do CPU bận khi train/đo song song.) **Kết luận:** model fine-tune v1 tệ hơn model gốc ở mọi nhóm kích thước và với điện thoại nên **không thay thế** model nào. Được giữ trong danh sách chỉ để thể hiện đã có dataset/train mới. Mẫu điện thoại chỉ 31 nên số điện thoại không chắc chắn.

## Hiệu chỉnh ngưỡng để giảm báo nhầm (`person_coco`, imgsz 960)
Quét ngưỡng trên **val** (153 ảnh), kiểm tra lại trên **test** (153 ảnh): [reports/conf_sweep_person_coco_val.json](../../reports/conf_sweep_person_coco_val.json), [reports/conf_sweep_person_coco_test.json](../../reports/conf_sweep_person_coco_test.json). Kết quả hai tập nhất quán (đánh đổi precision/recall tương tự).

| conf (người, test) | Precision | Recall | Hộp sai (FP) |
| --- | --- | --- | --- |
| 0,25 | 0,73 | 0,67 | 221 |
| 0,35 | 0,83 | 0,60 | 114 |
| 0,45 | 0,88 | 0,52 | 66 |
| 0,55 | 0,91 | 0,43 | 37 |
| 0,65 | 0,95 | 0,35 | 16 |

Không có một ngưỡng chung tốt cho mọi cảnh báo, nên dùng ngưỡng riêng (`CALIBRATION` trong [app/live/alerts.py](../../app/live/alerts.py)): có người ≥0,25 (ít báo "không thấy người" nhầm), đếm "nhiều người" ≥0,50 (ít báo dư người), điện thoại ≥0,35, hộp hiển thị người ≥0,35. Thanh ngưỡng của giao diện bị khóa với model này.

Tỷ lệ báo nhầm **trên từng khung hình** (test, [reports/frame_alarm_eval.json](../../reports/frame_alarm_eval.json), `training/frame_alarm_eval.py`; một khung, chưa qua xác nhận đa khung):
| Tín hiệu | Ngưỡng chung 0,25 | Ngưỡng hiệu chỉnh |
| --- | --- | --- |
| Báo "nhiều người" nhầm (41 khung có đúng 1 người) | 14,6 % (6 khung) | **7,3 %** (3 khung) |
| Báo "không thấy người" nhầm (149 khung có người) | 5,4 % | 5,4 % |
| Báo điện thoại nhầm (124 khung không có điện thoại) | 0 % | 0 % |
| Phát hiện điện thoại (29 khung có điện thoại) | 48 % | 45 % |

Cỡ mẫu nhỏ (3–8 khung sai) nên chênh lệch mang tính gợi ý, không phải kết luận thống kê. Dữ liệu là ảnh COCO, không phải webcam phòng thi; cảnh báo vật giống điện thoại trong phòng thi thật chưa được đo.

## Bằng chứng về nhóm model "Cheating" (Duyên/KNN)
Từ `reports/knn_duyen/*.json` (số liệu của tác giả, không tái tạo được vì dữ liệu Duyên không có trong dự án): KNN chia ngẫu nhiên: precision/recall/F1 "Cheating" = 0,84/0,75/0,79; **leave-one-video-out: 0,32/0,24/0,27** (3 video, từng fold F1 từ 0,09 đến 0,33). Chênh lệch lớn cho thấy số chia ngẫu nhiên bị thổi phồng do khung gần giống nhau; YOLO11 Duyên (mAP50 0,935, cùng cách chia) nhiều khả năng cũng vậy. Chưa có bằng chứng các nhãn hành vi này hoạt động tốt trên video mới.

## Tối ưu độ trễ ensemble (CPU AMD Ryzen 5 4600H, 6 nhân, không dùng GPU)
Đo trên 24 ảnh test, mô phỏng một phiên webcam. Thay đổi: (1) các mô hình chạy **song song** với khóa riêng từng mô hình và 3 luồng CPU mỗi mô hình (song song 4 mô hình: 6 luồng 1285 ms, 3 luồng 759 ms/khung, so với tuần tự 1054 ms); (2) mô hình hành vi chỉ chạy **mỗi 2 khung** của một phiên, khung còn lại dùng lại kết quả gần nhất (neo người vẫn chạy mọi khung); (3) neo người dùng 800 px thay vì 960 (test: recall người 0,54 so với 0,60, người nhỏ 0,30 so với 0,38, nhanh hơn khoảng 35 %); (4) bỏ `training`.

| | Độ trễ TB mỗi khung |
| --- | --- |
| Trước (5 mô hình tuần tự, 960 px) | khoảng 1.100–2.100 ms (đo ở các lần chạy khác nhau, nhiễu do máy bận) |
| Sau, phiên webcam | **628 ms** (p95 1.281; khung đủ 921, khung dùng lại 335) ≈ 1,6 khung/giây |
| Sau, tải ảnh lẻ (không bỏ qua khung) | 1.152 ms |

Đánh đổi: kết quả hành vi có thể cũ tối đa 1 khung; recall người nhỏ giảm so với 960 px (chọn 960 ở giao diện nếu cần). Hạn chế còn lại là CPU; ước lượng GPU (GTX 1650) sẽ nhanh hơn nhiều lần nhưng **chưa đo**, vì PyTorch hiện là bản CPU. Điểm tin cậy hiển thị theo dạng 0–1 (ví dụ 0.84).

## Tối ưu độ trễ lần 2 (vẫn CPU, không GPU)
Ba thay đổi: (1) **cổng chuyển động**: so ảnh xám 64×48 với khung của lần suy luận thật gần nhất, cảnh không đổi (độ lệch TB ≤3/255 và <0,4 % điểm ảnh đổi >25 mức) thì dùng lại kết quả, nhưng luôn suy luận lại sau tối đa 1 s; (2) **KNN chỉ chạy khi cần**: theo luật `decide`, KNN không đổi nhãn khi chưa có YOLO11 Duyên/Gian lận nghi vấn, nên chỉ chạy sau khi một nguồn mạnh đã nghi vấn (tiết kiệm khoảng 300 ms/lần suy luận); (3) **không phóng to khung** nhỏ hơn `imgsz` (giao diện mặc định gửi 800 px, khớp `imgsz` 800).

Đo mô phỏng phiên webcam (5 khung/giây, ảnh 800×600, nhiễu cảm biến σ=2, `training/` không dùng):
| Kịch bản | Độ trễ TB | p95 | Khung dùng lại kết quả |
| --- | --- | --- | --- |
| Thí sinh ngồi yên | **55 ms** | 307 ms | 24/30 |
| Cảnh đổi liên tục (xấu nhất) | **275 ms** | 513 ms | 0/24 |
| (lần tối ưu 1, đổi liên tục) | 628 ms | 1281 ms | — |

Đánh đổi/giới hạn: ở khung dùng lại, kết quả có thể cũ tối đa 1 s; chuyển động rất nhỏ (dưới ngưỡng) trong cảnh đứng yên có thể bị trễ tới 1 s mới được thấy; "đổi liên tục" dùng ảnh khác nhau mỗi khung (ảnh tĩnh COCO), không phải video thật; chưa đo trên webcam vật lý. Test: `tests/test_ensemble.py` (cổng chuyển động, KNN cascade, không phóng to khung).
