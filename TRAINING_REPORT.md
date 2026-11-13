# TRAINING_REPORT

## Trạng thái: **KHÔNG có training thật nào được thực hiện / hoàn thành.**

Lý do: (1) chưa có dataset phù hợp trong dự án và chưa được xác nhận giấy phép/dung lượng để tải; (2) PyTorch trên máy là bản CPU (`cuda=False`) dù có GTX 1650 (4 GB) — train 960 px trên CPU với 7,4 GB RAM không thực tế.

## Đã làm và đã kiểm chứng
- [training/train.py](training/train.py): fine-tune YOLO11 từ `models/yolo11n_coco.pt`; seed cố định + `deterministic`, early stopping (`--patience`, mặc định 15), trần epoch 300, checkpoint `last.pt`/`best.pt` (+ `--resume`), augmentation có kiểm soát (HSV, scale 0,6, mosaic, random erasing 0,3, không xoay/lật dọc), ghi `training_run.json` (cấu hình, seed, phiên bản torch/ultralytics/python, thiết bị, thời gian, metrics val/test).
- **Smoke test pipeline:** 1 epoch, imgsz 320, 44 ảnh, CPU trên `datasets/coco128_split` chạy trọn: train → val → test → ghi JSON. Metrics của lần chạy này **vô nghĩa** (model đã học các ảnh đó, tập test 19 ảnh) và không dùng ở đâu.
- **Chưa kiểm chứng:** `--resume` (chưa ngắt/tiếp tục thật), chạy trên GPU/Colab.

## Chạy training thật (GPU / Colab)
```bash
pip install ultralytics==8.3.185 "opencv-python<5"
python training/prepare_dataset.py --src <thư_mục_ảnh+nhãn> --dst datasets/examvision --names person "cell phone"
python training/train.py --data datasets/examvision/data.yaml --weights models/yolo11n_coco.pt --name person_phone_v1 --epochs 60 --imgsz 960 --batch 8
# tiếp tục nếu bị ngắt
python training/train.py --resume runs/train/person_phone_v1/weights/last.pt
```
Colab: chọn runtime GPU, `git clone` dự án, chạy các lệnh trên, tải `runs/train/<name>/weights/best.pt` về `models/` rồi đánh giá bằng `training/eval_person.py` (thêm mô hình vào `MODELS`) **trên tập test riêng ảnh webcam/phòng thi**. Chỉ thay model khi cải thiện được chứng minh trên cùng tập test và không giảm recall người gần.

Dùng GPU cục bộ: cài PyTorch bản CUDA vào `.venv` (khoảng vài GB, nên kiểm tra dung lượng; ổ C: chỉ còn 3,6 GB, `.venv` ở ổ E:). Chưa thực hiện.
