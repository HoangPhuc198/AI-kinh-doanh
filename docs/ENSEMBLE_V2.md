# ExamVision Ensemble v2

Bản sửa lỗi của `ensemble` (v1). v1 giữ nguyên (`app/ensemble.py`, khóa `ensemble`) để quay lại; v2 là khóa `ensemble_v2` (`app/ensemble_v2.py`), chọn được ở cả trang `/` và `/giam-sat`. Mặc định giao diện vẫn là v1. **v2 tự chứa**: không import `ensemble.py`; xem mục "Cấu trúc file và tính tự chứa" bên dưới.

## Lỗi của v1 và cách v2 sửa

| # | Lỗi của v1 (bằng chứng) | v2 |
|---|---|---|
| 1 | **Ngồi im bị báo Gian lận.** KNN + `gianlan` cùng bắn liên tục trên một dáng ngồi yên, v1 coi là "hai nguồn độc lập" và coi việc kéo dài là bằng chứng. Log webcam thật: 60/115 khung Gian lận. KNN phân loại theo tọa độ keypoint **tuyệt đối** trong ảnh (`paper_detector.FEATURES`) nên học cả vị trí ngồi; trên video lạ F1 chỉ 0,27 | Bỏ KNN. Nguồn "hình dáng" (Training YOLO) chỉ củng cố một tín hiệu **đo được**, **không tự gây cờ**. Phát lại log thật: Gian lận 51,3% → 3,4% |
| 2 | **Hành vi rõ ràng không lên Gian lận.** Một nguồn kéo dài chỉ có trọng số 0,3 < ngưỡng 0,45: quay đầu / điện thoại giữ 30 s vẫn "Nghi ngờ" (đã đo) | Tín hiệu đo được giữ liên tục ≥ 3 × `min_seconds` (mặc định 4,5 s) thì **Gian lận**; hai nguồn đồng thuận thì kết luận nhanh như v1 |
| 3 | `assign_turns` lấy detection gán **sau cùng**, `assign_evidence` lấy **lớn nhất**: cùng một người có thể ra hai kết quả khi đứng sát nhau | Một hàm duy nhất (`pose_evidence`: lớn nhất) |
| 4 | Quá nhiều nút ẩn: mức KNN danh nghĩa 0,70, `WEAK/UNNORMALIZED/LATE_SOURCES`, thác KNN/Training, trọng số đặt tay | 3 nguồn, một bảng luật (`decide`), không thác, không mức danh nghĩa. Hằng số mới: `SINGLE_SOURCE_FACTOR`, `PROFILE_EV` |
| 5 | Chưa từng đo trên dữ liệu có nhóm | Có công cụ đo theo nhóm và số đo làm mốc (bên dưới) |
| 6 | **Cúi mặt viết bài bị báo Gian lận.** Nhánh "chỉ thấy một mắt" của `head_turn_evidence` trả hằng số 1,0 / 0,8 mà không đo gì; người cúi mặt hay mất một mắt nên bị đọc thành quay ngang hoàn toàn, lặp lại ở mọi khung rồi leo thang. Ở 2 video phòng thi, 100% lần tín hiệu quay đầu ≥ 0,55 đến từ nhánh này | Tách hai mức: **mạnh** = nhánh hai mắt (đo thật, báo nhầm 3%) vẫn được leo thang; **yếu** = nhánh một mắt (`pose_side`, báo nhầm 31–34%) tự báo tối đa **Nghi ngờ**, chỉ củng cố khi có tín hiệu đo được khác |
| 7 | **Khung bao dính khi tua video.** `draw()` vẽ lại kết quả cuối ở mọi khung hình, không so thời điểm; khi tạm dừng rồi tua thì `analyze()` không chạy nên khung cũ ở nguyên chỗ | Mỗi kết quả mang thời điểm video (`f.t`); chỉ vẽ khi khớp khung đang hiện (trễ ≤ 3 s khi phát); xóa kết quả khi tua; phân tích lại ngay khi tua xong kể cả đang tạm dừng (`app/index.html`) |

Lỗi 6 cũng có ở v1 (cùng hàm). v1 để nguyên theo cam kết quay lại.

## Nguồn của v2 và cách chọn

Mô hình: YOLO11 COCO (người + điện thoại), YOLO11-Pose (quay đầu), Training YOLO (hình dáng). Tiêu chí được chốt **trước** khi xem kết quả: Youden J (= recall − tỉ lệ báo nhầm) gộp ba bộ Roboflow, cận dưới khoảng tin cậy theo nhóm > 0 và J > 0 ở ít nhất 2/3 bộ; chênh J giữa split train và test < 0,15 (dấu hiệu mô hình đã thấy dữ liệu); loại KNN theo thiết kế (lỗi 1); tối đa 3 nguồn.

J gộp, valid + test, ảnh đơn, bản v1 (`reports/eval_by_group_pooled.json`):

| Nguồn | J [khoảng tin cậy 95% theo nhóm] | J từng bộ | Kết quả |
|---|---|---|---|
| training | +0,09 [+0,06, +0,12] | +0,04 / +0,39 / +0,11 | Chọn |
| pose | +0,06 [+0,04, +0,09] | +0,01 / +0,39 / +0,06 | Chọn |
| phone (từ mô hình COCO) | +0,02 [+0,01, +0,03] | +0,02 / +0,03 / +0,02 | Chọn (rất nhỏ) |
| KNN Duyên | +0,04 [+0,01, +0,07] | −0,01 / +0,13 / +0,06 | Loại theo thiết kế |
| gianlan | +0,01 [0,00, +0,02] | −0,01 / +0,05 / +0,01 | Không đạt |
| yolo_duyen | 0,00 [−0,01, +0,01] | 0,00 / +0,08 / 0,00 | Không đạt |

Chênh J train − test của ba nguồn chọn: 0,03 / 0,02 / −0,01 (mẫu train nhỏ: 300–500 ảnh mỗi bộ).

## Nghiên cứu nhánh quay đầu (lỗi 6)

`training/pose_branch_study.py` + `pose_branch_report.py`: ghép từng khung nhãn với người pose, tách theo nhánh keypoint. Số đo chính trên valid + test **loại ảnh có bản gần trùng trong train** (1362 khung nhãn; nhãn "đầu" = head_rotation, peeking, Look_around). Ngưỡng/đặc trưng chỉ chọn trên split train.

| Luật | Báo nhầm (người bình thường) | Recall (lớp đầu) | J [khoảng tin cậy] |
|---|---|---|---|
| v1 hiện tại (nhánh một mắt = 0,8/1,0) | 0,05 | 0,45 | +0,40 [+0,28, +0,53] |
| Nhánh hai mắt riêng | 0,03 | 0,24 | +0,22 [+0,07, +0,39] |
| Nhánh một mắt riêng | **0,31** | 0,81 | +0,51 [+0,30, +0,69] |
| R1: bỏ nhánh một mắt | 0,02 | 0,14 | +0,12 [+0,04, +0,23] (**loại**: mất ~70% recall) |

Các cách "vá" nhánh một mắt đã thử, tiêu chí chốt trước: R2 (mắt và tai cùng bên) không có tác dụng, 42/44 người bình thường cũng khớp; R3 (độ lệch ngang mũi-mắt / chiều cao người ≥ 0,025, AUC train 0,77) cho J +0,34 [+0,23, +0,48], không khác luật hiện tại có ý nghĩa và thêm một ngưỡng nên không dùng; R4 (mũi lệch trục vai / bề rộng vai, AUC train 0,58) không đạt ngưỡng AUC 0,70. Không có phép đo hình học nào tách được "quay ngang" khỏi "cúi mặt mất một mắt" đủ tin cậy ở độ phân giải này. Kết luận được rút **từ các bộ có nhãn**, không từ hai video phòng thi; hai video chỉ dùng kiểm tra hồi quy sau khi đã quyết.

## So với v1 (cùng cách đo, valid + test, gộp 3 bộ, ảnh đơn)

| | Độ chính xác (P) | Recall | Báo nhầm (FAR) | J [khoảng tin cậy] |
|---|---|---|---|---|
| v1 "Nghi ngờ" (≥2 nguồn đồng thuận) | 0,72 | 0,07 | 0,026 | +0,04 [+0,03, +0,07] |
| v2 "Nghi ngờ" (≥2 nguồn, có tín hiệu đo được) | 0,87 | 0,02 | 0,003 | +0,02 [+0,01, +0,03] |
| v1 "Cần chú ý" trở lên | 0,66 | 0,15 | 0,072 | +0,08 [+0,05, +0,11] |
| v2 "Cần chú ý" trở lên | 0,68 | 0,12 | 0,053 | +0,07 [+0,04, +0,10] |

**v2 không phát hiện được nhiều hơn v1.** Ở mức "Cần chú ý" hai bản ngang nhau; ở mức "Nghi ngờ" (≥2 nguồn) v2 ít báo nhầm hơn hẳn (0,3% so với 2,6%) nhưng cũng phát hiện ít hơn (2% so với 7%), vì chỉ còn tín hiệu đo được làm chỗ dựa. Cái được: lỗi 1, 2, 6 và 7, ít nguồn, ít nút ẩn.

## Kết quả kiểm tra hồi quy trên 2 video phòng thi (tham khảo, không dùng để chỉnh)

Qua đường video của giao diện (khung ≤ 960 px, JPEG, trạng thái theo luồng, 4 khung/s): v2 trước khi sửa lỗi 6 báo Gian lận 42% khung của một thí sinh cúi viết (v1 dùng cùng hàm quay đầu; chưa chạy lại v1 trên video ở lần này); v2 cuối: **0% Gian lận ở cả hai video**. Nhưng thí sinh đó vẫn bị **Nghi ngờ 87%** số khung (video `7092083`), và 3% khung ở video `Test-bai-bao`: đây là cái giá của tín hiệu yếu.

## Giới hạn (đọc trước khi tin các con số)

- **Mọi độ chính xác ở đây đều thấp**: recall 2–15% ở báo nhầm 0,3–7%, cả v1 lẫn v2. Ba bộ là ảnh web/stock, nhãn do người ngoài gán, không phải webcam phòng thi. Đừng đọc là độ chính xác trên webcam.
- **Rò rỉ giữa split trong chính các bộ này**: 13–56% ảnh valid/test có bản gần trùng trong train (dHash; cận dưới), cao nhất ở `exam malpractice detection2`. Số dùng cho quyết định lỗi 6 đã loại các ảnh đó.
- **`Cheating Detection`**: chỉ 113/527 khung nhãn ghép được với người (nhiều lớp là vật nhỏ), nên recall của bộ này bị kéo thấp.
- **Ảnh đơn không đo được thời gian.** Lỗi 1, 2 được kiểm chứng bằng unit test (mỗi lỗi bị phá thì test fail), phát lại log webcam thật và phiên giả lập; **chưa** kiểm chứng trên video thi có nhãn thời gian vì dự án không có.
- **Người cúi mặt viết bài vẫn có thể bị "Nghi ngờ"** (tín hiệu yếu `pose_side`, báo nhầm 31–34% ở nhánh này). Muốn tắt hẳn: `WEAK_CUES = ()` trong `ensemble_v2.py`; đổi lại, ca quay ngang (một mắt) sẽ không được báo, recall quay đầu trên dữ liệu có nhãn giảm 0,45 → 0,14.
- `SINGLE_SOURCE_FACTOR` (3) là heuristic. Điện thoại "gần người" chưa phân biệt cầm dùng với để trên bàn: giữ ≥ 4,5 s sẽ lên Gian lận (muốn bỏ: bỏ `'phone'` khỏi `STRONG`).
- Cách duy nhất để cải thiện thật sự mức "cúi viết" so với "quay ngang": video/webcam có nhãn theo đoạn thời gian. Hai video phòng thi chỉ là hai ca, không đủ để chỉnh luật mà không over-fit.
- Chưa thử khung bao khi tua video trong trình duyệt thật; mới kiểm tra logic `fresh()` bằng node với video giả lập.

## Cấu trúc file và tính tự chứa

`app/ensemble_v2.py` không import `ensemble.py` (v1). Nó chỉ dùng chung `judge.py` (kết luận theo thời gian + `Settings`) và `runtime.py` (nạp mô hình); `inference.py` chỉ đăng ký nó và nạp đúng module của mô hình được chọn. Chạy v2 không cần `ensemble.py`, `paper_detector.py` hay trọng số của KNN, `gianlan`, `yolo_duyen`. Trọng số v2 cần: `yolo11m_coco.pt` (GPU) hoặc `yolo11n_coco.pt` (CPU), `yolo11n-pose.pt`, `training_yolo.pt`.

File chia thành tám mục đánh dấu bằng khung `# ====`: (1) nguồn và luật của v2, (2) cấu hình chạy, (3) hình học, (4) vẽ và đóng gói kết quả, (5) cổng chuyển động, (6) quay đầu từ keypoint, (7) gộp bằng chứng, (8) chạy mô hình. Các mục 2–5 và các hàm đo hướng/hai mắt ở mục 6 được **chép nguyên văn từ v1**.

Khác biệt có chủ ý so với bản chép (đều được test):
- `collect_persons`: chỉ neo COCO tạo người. v1 còn thêm người từ các nguồn hành vi, nhưng v2 luôn gọi với danh sách nguồn rỗng nên kết quả giống hệt.
- `assign_evidence`: confidence thiếu (`None`) tính 0,0. v1 dùng mức KNN danh nghĩa 0,70; YOLO luôn có confidence nên không xảy ra.
- `head_turn_evidence`: chỉ nhánh hai mắt. Nhánh một mắt của v1 là `profile_cue` (tín hiệu yếu) ở v2.
- Nhóm luồng riêng (`ensemble_v2`) và nhật ký debug riêng `logs/ensemble_v2_debug.jsonl` (v1: `ensemble_debug.jsonl`).

**Đã kiểm chứng rằng việc tách không đổi hành vi** (so với bản v2 ngay trước khi tách, commit `261b6eb`):
- Đầu ra trên **1038 ảnh** valid + test của ba bộ có nhãn và **121 khung** của hai video phòng thi (id, nhãn, điểm, phiếu từng nguồn, lý do, khung bao): **0 khác biệt**. Chạy bản cũ hai lần cho kết quả giống hệt nhau (đồng hồ treo tường được thay bằng thời gian video để kết quả xác định), nên phép so sánh có ý nghĩa.
- `DifferentialTests`: từng hàm chép phải cho kết quả y hệt hàm gốc của v1 trên dữ liệu ngẫu nhiên (hình học, phiếu, đo quay đầu, cổng chuyển động, kích thước ảnh, vẽ và đóng gói theo từng byte). Cố ý làm sai 8 chỗ khác nhau đều bị bắt.
- `SelfContainedTests`: file v2 không chứa import v1 (phân tích cú pháp), nạp v2 trong tiến trình mới không kéo v1 vào, và đăng ký mô hình của v2 chỉ liệt kê trọng số của v2.
- Công cụ `eval_by_group.py` đo v2 ra báo cáo trùng khít báo cáo đã commit.

## Chạy lại phép đo

```powershell
.\.venv\Scripts\python.exe training\eval_by_group.py "datasets\<bộ>" --model ensemble_v2 --splits valid test
.\.venv\Scripts\python.exe training\pool_by_group.py --model ensemble_v2
.\.venv\Scripts\python.exe training\pose_branch_study.py "datasets\<bộ>"      # rồi:
.\.venv\Scripts\python.exe training\pose_branch_report.py
.\.venv\Scripts\python.exe -m unittest tests.test_ensemble_v2 tests.test_group_eval -v
```

Đổi nguồn hay luật mà không đo lại là cách quay về "chỉnh đi chỉnh lại": mọi thay đổi phải có số trước/sau trên cùng cách đo, ngưỡng chọn trên một phần dữ liệu và kiểm tra trên phần khác. Dữ liệu cần thêm để tiến xa hơn: video/webcam có nhãn theo đoạn thời gian (ngồi im, cúi viết, quay đầu, quay người, điện thoại), chia theo người hoặc video.
