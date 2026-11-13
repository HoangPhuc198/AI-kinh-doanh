"""Tỷ lệ báo nhầm / bỏ sót TÍN HIỆU trên từng khung (không phải từng hộp), có ground truth.

  python training/frame_alarm_eval.py --data datasets/coco_person_phone --split test --imgsz 960
Khung "một người": đúng 1 người GT (mô phỏng webcam một thí sinh) -> báo 'nhiều người' là NHẦM.
Khung "có người": >=1 người GT -> báo 'không thấy người' là NHẦM.  Khung điện thoại: có/không điện thoại GT.
So sánh ngưỡng chung (đếm mọi hộp >= t) với ngưỡng hiệu chỉnh (xem app/live/alerts.py CALIBRATION). Một khung, chưa qua xác nhận đa khung.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'training'))
import runtime  # noqa: E402
from eval_person import load_gt  # noqa: E402
import cv2  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--data', default='datasets/coco_person_phone'); ap.add_argument('--split', default='test')
ap.add_argument('--imgsz', type=int, default=960); ap.add_argument('--out', default='reports/frame_alarm_eval.json'); a = ap.parse_args()
runtime.load_torch(); model = runtime.get_yolo(ROOT / 'models/yolo11n_coco.pt')
rows = []
for img, gts in load_gt(ROOT / a.data, a.split):
    r = model.predict(cv2.imread(str(img)), conf=0.25, imgsz=a.imgsz, verbose=False, device='cpu')[0]
    p = [float(b.conf) for b in r.boxes if r.names[int(b.cls)] == 'person']
    ph = [float(b.conf) for b in r.boxes if r.names[int(b.cls)] == 'cell phone']
    rows.append((sum(c == 'person' for c, _ in gts), sum(c == 'phone' for c, _ in gts), p, ph))
rate = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
one = [r for r in rows if r[0] == 1]; has = [r for r in rows if r[0] >= 1]; ph_y = [r for r in rows if r[1] >= 1]; ph_n = [r for r in rows if r[1] == 0]
out = {'images': len(rows), 'frames_single_person': len(one), 'frames_with_person': len(has), 'frames_with_phone': len(ph_y), 'frames_without_phone': len(ph_n),
       'false_multiple_people_rate': {f'count>={t}': rate([sum(c >= t for c in r[2]) >= 2 for r in one]) for t in (0.25, 0.35, 0.5, 0.6)},
       'false_no_person_rate': {f'present>={t}': rate([not any(c >= t for c in r[2]) for r in has]) for t in (0.25, 0.35, 0.5)},
       'phone_false_alarm_rate': {f'phone>={t}': rate([any(c >= t for c in r[3]) for r in ph_n]) for t in (0.25, 0.35, 0.45, 0.6)},
       'phone_hit_rate': {f'phone>={t}': rate([any(c >= t for c in r[3]) for r in ph_y]) for t in (0.25, 0.35, 0.45, 0.6)}}
Path(ROOT / a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8'); print(json.dumps(out, ensure_ascii=False, indent=1))
