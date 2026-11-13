"""Suy luận một khung hình cho mọi mô hình; dùng chung cho tải tệp và webcam."""
import base64
import time
from pathlib import Path

import cv2
import numpy as np

import runtime

PROJECT = Path(__file__).resolve().parent.parent
MODELS = {
    'training': ('Training YOLO', PROJECT/'models/training_yolo.pt'),
    'gianlan': ('GIan lan', PROJECT/'models/gian_lan.pt'),
    'yolo_duyen': ('YOLO11 Duyên', PROJECT/'models/yolo11_duyen.pt'),
    'paper': ('KNN bài báo', PROJECT/'models/yolo11n-pose.pt'),
    'duyen': ('KNN dataset Duyên', PROJECT/'models/duyen_knn.joblib'),
    # Bộ phát hiện đối tượng tổng quát (COCO) có lớp person + cell phone; dùng cho đếm người/điện thoại.
    'person_coco': ('YOLO11n COCO', PROJECT/'models/yolo11n_coco.pt'),
    # Fine-tune từ person_coco trên dataset MỚI coco_person_phone (COCO val2017 lọc, 1014 ảnh); xem TRAINING_REPORT.md.
    # Mô hình tổng hợp: gộp person_coco + yolo_duyen + gianlan + KNN Duyên (xem app/ensemble.py).
    'ensemble': ('ExamVision Ensemble', PROJECT/'models/yolo11n_coco.pt'),
    'person_phone': ('YOLO11n Người+ĐT v1', PROJECT/'models/yolo11_person_phone.pt'),
}
POSE_KEYS = ('paper', 'duyen')
# imgsz mặc định theo mô hình (None = mặc định Ultralytics 640). Giá trị 960 dựa trên
# training/eval_person.py: recall người nhỏ tăng từ 0.32 (640) lên 0.45 (960) với chi phí ~2.3x thời gian.
DEFAULT_IMGSZ = {'person_coco': 960}
ENSEMBLE_IMGSZ = 800  # đo trên test: recall người 0.54 (800) so với 0.60 (960), nhanh hơn ~35-40%
ALLOWED_IMGSZ = (640, 800, 960, 1280)


_device = runtime.device


def _classifier_path(key):
    return str(MODELS['duyen'][1]) if key == 'duyen' else None


def required_files(key):
    """Mọi tệp trọng số cần có để chạy mô hình `key`."""
    if key == 'ensemble':
        import ensemble
        return sorted({p for k in ensemble.components() for p in required_files(k)})
    files = [MODELS[key][1]]
    if key in POSE_KEYS:
        files.append(MODELS['paper'][1])
    return files


def warm_model(key):
    """Nạp trước trọng số (một lần mỗi tiến trình) để khung đầu tiên không bị chậm."""
    if key == 'ensemble':
        import ensemble
        for k in ensemble.components():
            warm_model(k)
        return
    if key in POSE_KEYS:
        import paper_detector
        paper_detector.get_models(_classifier_path(key))
    else:
        model = runtime.get_yolo(MODELS[key][1])
        with runtime.model_lock(key):  # khởi động các lớp bằng một lần suy luận trên ảnh đen
            model.predict(np.zeros((480, 640, 3), np.uint8), verbose=False, device=_device(), imgsz=DEFAULT_IMGSZ.get(key, 640))


def run_model(key, frame, confidence, include_image=True, imgsz=None, iou=None, state=None):
    """imgsz/iou chỉ áp dụng cho mô hình YOLO detect; mô hình pose+KNN giữ cấu hình riêng."""
    if imgsz is not None and imgsz not in ALLOWED_IMGSZ:
        raise ValueError(f'imgsz phải thuộc {ALLOWED_IMGSZ}')
    if key == 'ensemble':
        import ensemble
        return ensemble.run(frame, include_image=include_image, imgsz=imgsz, state=state)
    started = time.perf_counter()
    with runtime.model_lock('pose' if key in POSE_KEYS else key):  # paper và duyen dùng chung một mô hình pose
        if key in POSE_KEYS:
            import paper_detector
            detections = paper_detector.infer(frame, threshold=confidence, classifier_path=_classifier_path(key))
            rendered, counts = (paper_detector.draw(frame.copy(), detections) if include_image else (None, None))
            h, w = frame.shape[:2]
            details = [{'label': 'Suspicious' if pred == 1 else 'Normal', 'confidence': None, 'box': [float(box[0]/w), float(box[1]/h), float(box[2]/w), float(box[3]/h)], 'keypoints': np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0).tolist()} for box, pts, pred in detections]
            if counts is None:
                counts = {}
                for d in details:
                    name = d['label'].lower()
                    counts[name] = counts.get(name, 0) + 1
        else:
            model = runtime.get_yolo(MODELS[key][1])
            kwargs = {'iou': iou} if iou else {}
            result = model.predict(frame, conf=confidence, verbose=False, device=_device(), imgsz=imgsz or DEFAULT_IMGSZ.get(key, 640), **kwargs)[0]
            rendered = result.plot() if include_image else None
            details = [{'label': str(result.names[int(b.cls.item())]), 'confidence': round(float(b.conf.item()), 3), 'box': [float(v) for v in b.xyxyn[0].tolist()]} for b in result.boxes]
            counts = {}
            for d in details:
                counts[d['label']] = counts.get(d['label'], 0) + 1
    elapsed = round((time.perf_counter()-started)*1000)
    output = {'details': details, 'counts': counts, 'ms': elapsed}
    if include_image:
        ok, encoded = cv2.imencode('.jpg', rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError('Không thể tạo ảnh kết quả')
        output['image'] = base64.b64encode(encoded).decode()
    return output
