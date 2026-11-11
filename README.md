# ExamVision — bản bàn giao

Ứng dụng chạy trên máy để thử ảnh hoặc video với bốn lựa chọn mô hình. Nhận diện là dấu hiệu cần xem xét; không dùng dự đoán làm kết luận gian lận.

## Chuẩn bị

Cần Python 3.10 trở lên, pip và đủ dung lượng cài PyTorch. Nên tạo môi trường riêng. Khi cài cần kết nối Internet; xử lý ảnh và video sau đó diễn ra trên máy.

### macOS / Linux

Mở Terminal trong thư mục này và chạy:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python "Kiem thu mo hinh/app.py"
```

### Windows

Mở PowerShell trong thư mục này và chạy:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe "Kiem thu mo hinh\app.py"
```

Sau đó mở http://127.0.0.1:8765 trong trình duyệt. Để dừng máy chủ, quay lại Terminal và nhấn Ctrl+C. Nếu cổng 8765 đã được sử dụng, dừng phiên ExamVision cũ trước khi chạy lại.

## Sử dụng

1. Chọn ảnh hoặc video tối đa 75 MB. Trình duyệt cần hỗ trợ định dạng video; MP4/H.264 thường phù hợp.
2. Chọn một hoặc nhiều mô hình rồi bấm **Bắt đầu kiểm thử**.
3. Với video, bấm **Phát**; chọn mô hình ở mục **Hiển thị** trên video lớn. Hai mô hình KNN hiển thị thêm các điểm và đường nối tư thế cùng nhãn Normal/Suspicious.
4. Có thể thay đổi ngưỡng phát hiện. Với hai mô hình KNN, ngưỡng áp dụng cho bước phát hiện người của YOLO-Pose, không phải xác suất phân loại hành vi.

Video phát liên tục còn phân tích được thực hiện theo tốc độ của máy; một số khung hình có thể bị bỏ qua, nhãn/box có thể trễ. Hai mô hình KNN chỉ phân loại Normal/Suspicious, không phân loại chi tiết như dùng điện thoại hay quay đầu.

## Các mô hình và tệp cần thiết

| Lựa chọn trên trang | Tệp |
| --- | --- |
| Training YOLO | `Training YOLO/best.pt` |
| GIan lan | `GIan lan/runs/detect/train/weights/best.pt` |
| KNN bài báo | `Kiem thu mo hinh/models/yolo11n-pose.pt` + `Kiem thu mo hinh/models/best_model.pkl` |
| KNN dataset Duyên | `Kiem thu mo hinh/models/yolo11n-pose.pt` + `KNN tư thế/duyen_v1/knn_duyen_v1.joblib` |

KNN cần tọa độ tư thế do YOLO11-Pose trích xuất, không nhận ảnh trực tiếp. Tệp pickle/joblib chỉ nên được nạp từ nguồn tin cậy. Bộ bàn giao sử dụng scikit-learn 1.7.2 để phù hợp mô hình KNN.

## Kết quả và giới hạn

`KNN tư thế/duyen_v1/metrics.json` chứa đánh giá theo cách chia ảnh; `danh_gia_theo_video.json` chứa ba lượt để riêng từng nhóm video; biểu đồ tổng hợp nằm trong `bieu_do/`. Các chỉ số phân loại chỉ tính box nhãn ghép được với tư thế. Train, validation và test theo cách chia ảnh có các khung hình cùng nguồn video; kết quả không phải kiểm thử độc lập theo video. Chỉ có ba nhóm video trong phép đánh giá theo nhóm.

Bộ bàn giao không bao gồm dataset và video gốc. `MANIFEST.json` liệt kê kích thước và SHA-256 của các tệp chính. Mô hình bài báo và mã xử lý liên quan lấy từ [repository tác giả](https://github.com/arifbdu/Suspicious-Activity-Detection-Using-YOLO11-Pose-Estimation). Kiểm tra giấy phép tương ứng của mô hình, dữ liệu và Ultralytics khi phân phối hoặc sử dụng ngoài phạm vi dự án.

## Hướng dẫn cho thành viên

- [Làm việc với GitHub](HUONG_DAN_GITHUB.md): clone, tạo nhánh, commit, push và Pull Request.
- [Thêm mô hình mới](HUONG_DAN_THEM_MODEL.md): đăng ký YOLO Detect hoặc KNN, hiển thị trên web và kiểm thử.
