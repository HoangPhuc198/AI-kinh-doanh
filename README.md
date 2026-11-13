# ExamVision — bản bàn giao

## Cấu trúc thư mục

- `app/`: máy chủ và giao diện web.
  - `app.py`: máy chủ HTTP và các route; `inference.py`: chạy một khung hình qua mô hình; `runtime.py`: khởi tạo PyTorch một lần và registry mô hình; `paper_detector.py`: YOLO11-Pose + KNN.
  - `live/`: luật cảnh báo (`alerts.py`) và quản lý phiên webcam (`session.py`); `live.html`: trang giám sát webcam.
- `tests/`: kiểm thử tự động (`python -m unittest discover -s tests -t .`).
- `models/`: chỉ các trọng số cần cho suy luận.
- `reports/`: metric và biểu đồ đã lưu; không chứa dữ liệu ảnh.
- `docs/`: hướng dẫn GitHub và cách thêm mô hình.
- `requirements.txt`: thư viện để chạy web.


Ứng dụng chạy trên máy để thử ảnh hoặc video với năm lựa chọn mô hình. Nhận diện là dấu hiệu cần xem xét; không dùng dự đoán làm kết luận gian lận.

## Chuẩn bị

Cần Python 3.10 trở lên, pip và đủ dung lượng cài PyTorch. Nên tạo môi trường riêng. Khi cài cần kết nối Internet; xử lý ảnh và video sau đó diễn ra trên máy.

### macOS / Linux

Mở Terminal trong thư mục này và chạy:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app/app.py
```

### Windows

Mở PowerShell trong thư mục này và chạy:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app\app.py
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
| Training YOLO | `models/training_yolo.pt` |
| GIan lan | `models/gian_lan.pt` |
| KNN bài báo | `models/yolo11n-pose.pt` + `models/paper_knn.pkl` |
| KNN dataset Duyên | `models/yolo11n-pose.pt` + `models/duyen_knn.joblib` |
| YOLO11 Duyên | `models/yolo11_duyen.pt` |

KNN cần tọa độ tư thế do YOLO11-Pose trích xuất, không nhận ảnh trực tiếp. Tệp pickle/joblib chỉ nên được nạp từ nguồn tin cậy. Bộ bàn giao sử dụng scikit-learn 1.7.2 để phù hợp mô hình KNN.

## Kết quả và giới hạn

`reports/knn_duyen/metrics.json` chứa đánh giá theo cách chia ảnh; `video_evaluation.json` chứa ba lượt để riêng từng nhóm video; biểu đồ tổng hợp nằm trong `reports/knn_duyen/`. Các chỉ số phân loại chỉ tính box nhãn ghép được với tư thế. Train, validation và test theo cách chia ảnh có các khung hình cùng nguồn video; kết quả không phải kiểm thử độc lập theo video. Chỉ có ba nhóm video trong phép đánh giá theo nhóm.

Bộ bàn giao không bao gồm dataset và video gốc. `MANIFEST.json` liệt kê kích thước và SHA-256 của các tệp chính. Mô hình bài báo và mã xử lý liên quan lấy từ [repository tác giả](https://github.com/arifbdu/Suspicious-Activity-Detection-Using-YOLO11-Pose-Estimation). Kiểm tra giấy phép tương ứng của mô hình, dữ liệu và Ultralytics khi phân phối hoặc sử dụng ngoài phạm vi dự án.

## Hướng dẫn cho thành viên

- [Làm việc với GitHub](docs/HUONG_DAN_GITHUB.md): clone, tạo nhánh, commit, push và Pull Request.
- [Thêm mô hình mới](docs/HUONG_DAN_THEM_MODEL.md): đăng ký YOLO Detect hoặc KNN, hiển thị trên web và kiểm thử.

## Trang báo cáo

Mở ứng dụng và chọn **Báo cáo** trên thanh đầu trang, hoặc truy cập `http://127.0.0.1:8765/bao-cao`. Báo cáo hiển thị số ảnh theo tập, metric đã lưu và các mô hình được cấu hình. Số liệu không tự đồng bộ từ Roboflow; metric Duyên lấy từ các tệp JSON trong `reports/knn_duyen`, còn số liệu Graduation và GIan lan là bản chụp trong `reports/snapshots.json` và cần cập nhật khi dữ liệu hoặc kết quả huấn luyện thay đổi. Trạng thái đủ tệp không xác nhận mô hình chạy thành công.

## YOLO11 Duyên

Chọn **YOLO11 Duyên** trên web để chạy `models/yolo11_duyen.pt`. Đây là mô hình YOLO11n detection 2 lớp (`Normal`, `Cheating`), huấn luyện từ trọng số ngẫu nhiên trên Duyên v1; không dùng bước Pose hoặc KNN. Trên trang Báo cáo, kết quả test của model lấy từ `reports/yolo11_duyen/test_metrics.json`. Các metric là của detection, không so sánh trực tiếp với accuracy/F1 KNN. Để chạy bằng bản final, thư mục `models/` và `reports/yolo11_duyen/` đã chứa trọng số và metadata cần thiết; không cần toàn bộ ảnh huấn luyện.

## Giám sát webcam trực tiếp

Mở `http://127.0.0.1:8765/giam-sat` (hoặc nút **Giám sát webcam** ở đầu trang).

1. Bấm **Bật webcam** và cho phép camera trong trình duyệt. Webcam không tự bật; nếu từ chối quyền, trang hiện thông báo.
2. Chọn mô hình, ngưỡng và FPS tối đa rồi bấm **Bắt đầu giám sát**. Lần đầu mô hình được tải (có thể mất vài chục giây).
3. **Tạm dừng / Tiếp tục** dừng/tiếp tục việc gửi khung cho AI (webcam vẫn hiển thị). **Dừng giám sát** kết thúc phiên và tắt hẳn webcam.
4. Cột bên phải hiển thị trạng thái camera/AI, FPS xử lý, độ trễ suy luận và lịch sử cảnh báo; mỗi cảnh báo có thể đánh dấu *Cần xem xét / Đã xem xét / Bỏ qua (cảnh báo sai)*.

Cách hoạt động: trình duyệt lấy webcam bằng `getUserMedia` (chỉ chạy ở `localhost`/HTTPS), mỗi lần chỉ gửi **một** khung mới nhất tới `/api/live/frame` rồi chờ kết quả, nên không có hàng đợi tích tụ độ trễ. Chỉ một phiên hoạt động tại một thời điểm (phiên thứ hai nhận lỗi 409; phiên không gửi tín hiệu 15 giây tự hết hạn). Không ghi hình và không lưu ảnh; cảnh báo chỉ nằm trong RAM của phiên.

Cảnh báo (đều là *tín hiệu cần xem xét*, không phải kết luận gian lận) chỉ phát khi điều kiện đúng liên tục đủ số khung và thời gian, và có thời gian nghỉ 30 giây giữa hai cảnh báo cùng loại:

| Tín hiệu | Điều kiện | Mô hình hỗ trợ |
| --- | --- | --- |
| Không thấy người | không có đối tượng người ≥ 5 giây | tất cả (giả định mọi nhãn người; xem `live/alerts.py`) |
| Nhiều người | ≥ 2 người ≥ 2 giây | tất cả (có thể đếm trùng nếu mô hình xuất nhiều box cho một người) |
| Điện thoại | nhãn `cell-phones` ≥ 1 giây | chỉ GIan lan |
| Nhãn hành vi nghi vấn | `Suspicious`/`Cheating`/`cheating`/`cheating-behaviour` ≥ 2 giây | tất cả |

Chưa hỗ trợ: hướng đầu/ánh nhìn (không có mô hình được kiểm chứng), nhận diện danh tính, thu âm. Việc gán nhãn nào là "người" hay "nghi vấn" là giả định chưa kiểm chứng trên dữ liệu thật; cần giám thị đánh giá thực tế trước khi dùng.

## Lỗi PyTorch "can only be initialized once per process"

Máy chủ nay import PyTorch/Ultralytics đúng một lần khi khởi động (`app/runtime.py`) và in lỗi gốc nếu lần import đó thất bại, thay vì để từng request import lại. Nếu thấy dòng `KHÔNG import được PyTorch/Ultralytics`, lỗi in kèm là nguyên nhân thật (thường là thiếu Visual C++ Redistributable, thiếu RAM/pagefile hoặc cài torch hỏng); cài lại môi trường `.venv` rồi chạy lại. Chi tiết: [AUDIT_REPORT.md](AUDIT_REPORT.md).

## Windows: đường dẫn quá dài (MAX_PATH)

Thư mục dự án nằm sâu trong OneDrive với tên dài khiến `pip install torch` vào `.venv` **thất bại giữa chừng** (`No such file or directory ... torch\include\ATen\...`), để lại torch hỏng (`DLL load failed while importing _C`) — đây là trạng thái dẫn tới thông báo "can only be initialized once". Cách xử lý: (a) bật Long Paths của Windows (cần quyền quản trị) hoặc (b) clone dự án vào đường dẫn ngắn như `C:\ev\ExamVision_Final`. Kiểm tra môi trường: `.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"`.
