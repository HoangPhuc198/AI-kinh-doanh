# Hướng dẫn làm việc với GitHub

Repository của nhóm: <https://github.com/mduyen035/ExamVision_Final>. Đây là repository riêng tư; người khác cần được chủ repository cấp quyền trước khi clone hoặc mở trên GitHub.

## 1. Tải dự án lần đầu

Cài Git và đăng nhập GitHub trên máy của bạn. Với HTTPS, GitHub có thể yêu cầu xác thực qua trình quản lý thông tin đăng nhập của Git hoặc GitHub CLI. Không nhập mật khẩu tài khoản vào dòng lệnh hay gửi token qua chat.

Mở Terminal hoặc PowerShell tại thư mục muốn chứa dự án:

```bash
git clone https://github.com/mduyen035/ExamVision_Final.git
cd ExamVision_Final
```

Cài môi trường Python và chạy ứng dụng theo [README](../README.md). `git clone` đã tải các mô hình đang có trong repository; không cần tạo lại file `.pt` hoặc `.joblib` để chạy năm lựa chọn hiện tại.

## 2. Tạo nhánh cho thay đổi mới

Trước tiên kiểm tra trạng thái:

```bash
git status
```

Nếu có thay đổi chưa lưu/commit, hãy xem và xử lý chúng trước khi chuyển nhánh hoặc cập nhật. Sau đó lấy bản mới nhất của nhánh `main` và tạo một nhánh riêng, ví dụ khi thêm mô hình:

```bash
git switch main
git pull --ff-only origin main
git switch -c them-model-moi
```

Nếu Git báo có xung đột hoặc không thể cập nhật nhanh bằng `--ff-only`, dừng lại và kiểm tra trạng thái; không dùng `reset --hard` để xóa thay đổi của người khác.

## 3. Sửa, kiểm tra và commit

Đọc [Hướng dẫn thêm mô hình](HUONG_DAN_THEM_MODEL.md), sau đó chạy và kiểm tra ảnh/video trên máy. Xem đúng những gì sắp gửi:

```bash
git status
git diff
```

Ví dụ thêm một YOLO Detect trong `models/model_moi.pt`, sau khi đã sửa hai file web:

```bash
git add "models/model_moi.pt" "app/app.py" "app/index.html"
git diff --cached --stat
git commit -m "Add new detection model"
```

Nếu có sửa tài liệu hoặc file phụ thuộc, thêm chính xác những file đó bằng `git add`. Với mô hình KNN, thay đường dẫn `.pt` bằng file `.joblib`/`.pkl` tương ứng. Không đưa dataset, video thí sinh, môi trường `.venv`, API key hoặc mật khẩu vào commit chỉ để ứng dụng chạy.

## 4. Push và tạo Pull Request

```bash
git push -u origin them-model-moi
```

Mở repository trên GitHub, chọn **Compare & pull request**, mô tả mô hình và cách đã kiểm tra, rồi gửi để nhóm review. Sau khi thay đổi được merge, các thành viên khác có thể cập nhật:

```bash
git switch main
git pull --ff-only origin main
```

Không cần tạo lại repository mỗi lần có mô hình mới. Lần sau tạo nhánh mới từ `main` đã cập nhật; tên nhánh ví dụ ở trên chỉ là tên minh họa.

## 5. File trọng số lớn

File `models/training_yolo.pt` hiện khoảng 50,5 MiB và GitHub đã cảnh báo vì vượt mức khuyến nghị 50 MiB. GitHub chặn file Git thông thường lớn hơn 100 MiB. Nếu mô hình mới lớn, kiểm tra kích thước **trước khi commit**; với file vượt giới hạn cần dùng Git LFS hoặc nơi lưu trọng số phù hợp theo quy định nhóm. Git LFS có hạn mức lưu trữ/băng thông riêng và các thành viên cần cài Git LFS trước khi clone.

Ví dụ cho **một file mới**, sau khi đã cài Git LFS:

```bash
git lfs install
git lfs track "models/model_lon.pt"
git add .gitattributes "models/model_lon.pt"
git status
```

Quy tắc LFS này áp dụng khi file được thêm; không tự chuyển file đã commit trong lịch sử. Nếu đã lỡ commit một file quá lớn, cần xử lý commit/lịch sử cẩn thận trước khi push, tránh tự ý viết lại lịch sử chung.

## 6. Sự cố thường gặp

- **Không truy cập được repo:** kiểm tra đúng tài khoản và quyền truy cập repository riêng tư.
- **Push bị từ chối:** chạy `git status` và đọc thông báo; cập nhật nhánh và giải quyết xung đột trước khi push lại. Tránh `--force` vào `main`.
- **Đã sửa nhưng web chưa đổi:** khởi động lại server Python và tải lại trang; cần sửa cả máy chủ và giao diện như hướng dẫn thêm mô hình.
- **Trọng số không tải được:** kiểm tra đường dẫn, giấy phép, phiên bản thư viện và tính tương thích của loại mô hình. Chỉ nạp `.pt`, `.pkl` hoặc `.joblib` từ nguồn đáng tin cậy.
