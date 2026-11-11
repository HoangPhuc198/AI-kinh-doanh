# Hướng dẫn thêm mô hình mới vào ExamVision

Hiện tại web **không tự phát hiện mô hình mới**. Khi thêm file trọng số, cần đăng ký cùng một mã định danh ở máy chủ và giao diện. Hướng dẫn dưới đây áp dụng cho cấu trúc repository hiện tại.

## A. Chuẩn bị trước khi tích hợp

1. Xác định loại mô hình: **YOLO object detection** (`.pt`) hay **KNN dùng đặc trưng tư thế** (`.joblib`/`.pkl`). Các loại classification, segmentation, pose hoặc mô hình khác có thể cần viết nhánh xử lý riêng; không mặc định chạy được như object detection.
2. Kiểm tra trọng số, lớp đầu ra, môi trường huấn luyện và giấy phép. Chỉ nạp file từ nguồn tin cậy; `.pt`, `.pkl` và `.joblib` có thể chứa dữ liệu được thực thi lúc nạp.
3. Tạo đường dẫn mới cho trọng số thay vì ghi đè mô hình cũ. Ví dụ tạo thư mục `models/` ở gốc repository và đặt file tại `models/model_moi.pt`. Với KNN có thể dùng `models/knn_moi.joblib`.
4. Xác định mã định danh duy nhất, không dấu, ví dụ `model_moi` hoặc `knn_moi`. Mã này phải giống nhau trong tất cả những vị trí bên dưới.

> Ví dụ dùng tên file minh họa. Bạn cần có file mô hình thật ở đúng đường dẫn; không thể chạy chỉ bằng cách sao chép đoạn mã.

## B. Thêm YOLO object detection `.pt`

### B1. Khai báo mô hình ở máy chủ

Mở `Kiem thu mo hinh/app.py`, tìm từ điển `MODELS` và thêm **một dòng bên trong** từ điển:

```python
'model_moi': ('YOLO mới', PROJECT / 'models' / 'model_moi.pt'),
```

`PROJECT` là thư mục gốc repository. Ví dụ này giả sử `models/model_moi.pt` nằm tại gốc. Nhánh `else` hiện có trong `run_model` sẽ tải `YOLO(...)` và xử lý bounding box cho object detection; nhãn lấy từ `result.names` được lưu trong trọng số. Kiểm tra mô hình thực sự là detection và các lớp có đúng ý nghĩa; đừng suy luận `Cheating` chỉ từ vị trí class ID.

### B2. Thêm ô chọn trên web

Mở `Kiem thu mo hinh/index.html`, tìm các ô `<label class="model">` và thêm ngay sau một ô hiện có:

```html
<label class="model">
  <input type="checkbox" value="model_moi">
  <span>
    <strong>YOLO mới</strong>
    <small>Mô hình detection được huấn luyện mới</small>
  </span>
</label>
```

`value="model_moi"` phải trùng chính xác khóa trong `MODELS`. Thêm `checked` vào thẻ input nếu muốn mô hình được chọn mặc định.

### B3. Khai báo tên trên video

Vẫn trong `index.html`, tìm `let names={...}` trong `startLive` và thêm vào đối tượng:

```javascript
model_moi: 'YOLO mới',
```

Ví dụ, nếu dòng hiện có kết thúc bằng `duyen:'KNN dataset Duyên'`, thêm dấu phẩy rồi thêm mục mới trước dấu `}`. Không xóa các mục cũ. Nếu bỏ qua bước này, tên mô hình sẽ không hiển thị đúng ở danh sách video.

## C. Thêm KNN tư thế `.joblib` hoặc `.pkl`

Mô hình KNN **không nhận ảnh trực tiếp**. Nhánh KNN hiện tại dùng `models/yolo11n-pose.pt` để lấy 17 điểm, tạo 34 cột `x0, y0, …, x16, y16` được chuẩn hóa theo toàn ảnh, rồi gọi classifier. Chỉ dùng trực tiếp nếu KNN mới được huấn luyện với **cùng thứ tự đặc trưng, cách tiền xử lý, định dạng dữ liệu và ý nghĩa nhãn**. Với mô hình cần đặc trưng khác, cần bổ sung bộ trích xuất/adapter và kiểm thử riêng.

### C1. Đăng ký file

Đặt ví dụ `knn_moi.joblib` tại `models/` ở gốc repository. Trong `Kiem thu mo hinh/app.py`, thêm vào `MODELS`:

```python
'knn_moi': ('KNN mới', PROJECT / 'models' / 'knn_moi.joblib'),
```

### C2. Đưa khóa vào nhánh KNN

Tìm dòng hiện tại trong `run_model`:

```python
if key in ('paper', 'duyen'):
```

Đổi thành:

```python
if key in ('paper', 'duyen', 'knn_moi'):
```

Ngay bên dưới, thay dòng gán `classifier_path` hiện tại bằng:

```python
classifier_path = None if key == 'paper' else str(MODELS[key][1])
```

Cách này giữ `paper` dùng KNN gốc của bài báo và cho `duyen`/`knn_moi` lấy file tương ứng từ `MODELS`. Hàm `paper_detector.infer` tiếp tục dùng YOLO11-Pose và bộ tiền xử lý đã lưu trong pipeline.

**Kiểm tra nhãn KNN:** mã hiện tại quy định dự đoán số `1` = `Suspicious`, mọi giá trị còn lại = `Normal`. Nếu mô hình mới trả về chuỗi, dùng ID khác hoặc có hơn hai lớp, cần sửa cách giải mã nhãn ở `app.py` và phần vẽ tương ứng trong `paper_detector.py`; không áp dụng máy móc quy tắc này.

### C3. Thêm điều khiển giao diện

Thêm ô chọn trong `index.html`:

```html
<label class="model">
  <input type="checkbox" value="knn_moi">
  <span><strong>KNN mới</strong><small>KNN trên dữ liệu mới</small></span>
</label>
```

Và thêm vào `let names={...}` trong `startLive`:

```javascript
knn_moi: 'KNN mới',
```

KNN tương thích sẽ hiển thị box, các điểm/đường tư thế và nhãn Normal/Suspicious. Ngưỡng trên giao diện là ngưỡng phát hiện **người của YOLO-Pose**, không phải ngưỡng xác suất KNN.

## D. Kiểm tra sau khi thêm

1. Lưu file; **dừng server cũ bằng Ctrl+C rồi chạy lại** từ gốc repository:

   ```bash
   python "Kiem thu mo hinh/app.py"
   ```

   Trên Windows, nếu dùng môi trường `.venv` như README mà chưa kích hoạt, thay lệnh trên bằng `.\.venv\Scripts\python.exe "Kiem thu mo hinh\app.py"`.

2. Tải lại trang `http://127.0.0.1:8765`. Kiểm tra có ô chọn mô hình mới. Chỉ chọn mô hình đó, thử **một ảnh và một video**; nhấn phát video và chọn đúng tên tại mục **Hiển thị**.
3. So sánh nhãn/box trên web với lệnh inference gốc của mô hình trên cùng ảnh. Với KNN, kiểm tra đủ box, 17 điểm và ánh xạ `0/1`; xem lại điều kiện không phát hiện người.
4. Kiểm tra vẫn chạy được ít nhất một mô hình cũ và xem thông báo lỗi/terminal. Nếu thay môi trường, cập nhật `requirements.txt` phù hợp; `scikit-learn` phải tương thích phiên bản đã lưu KNN. Kiểm tra `pip install` trong môi trường sạch khi bàn giao.
5. Ghi tên mô hình, nguồn, dữ liệu huấn luyện, các lớp, phiên bản thư viện, cách đánh giá, giới hạn và giấy phép. Nếu cập nhật các tệp đã được liệt kê trong `MANIFEST.json`, cập nhật lại kích thước và SHA-256; manifest hiện tại không tự cập nhật.
6. Làm theo [hướng dẫn GitHub](HUONG_DAN_GITHUB.md) để commit, push và tạo Pull Request.

## E. Lỗi thường gặp

- **Không thấy ô chọn hoặc tên ở video:** kiểm tra cùng một khóa ở `MODELS`, `value` của checkbox và `names`.
- **Báo không nhận mô hình, hoặc trang mới nhưng server cũ:** tắt tiến trình cũ, khởi động lại và tải lại trình duyệt.
- **Không thấy box:** xác nhận đang chọn mô hình đúng, ảnh có người/đối tượng, ngưỡng không quá cao và terminal không báo lỗi. Mô hình không phát hiện thì sẽ không có box.
- **Sai nhãn KNN:** kiểm tra thứ tự/chuẩn hóa đặc trưng, phiên bản scikit-learn và ánh xạ `0/1`. YOLO detector có thể có thứ tự hoặc tên lớp khác.
- **Video chậm hoặc nhãn trễ:** các mô hình chạy lần lượt trên CPU, có thể bỏ qua khung hình; chọn ít mô hình hơn để thử. Không xem số khung cảnh báo là metric accuracy/mAP.
