"""Đánh giá phát hiện người / điện thoại theo nhóm kích thước, có ground truth (định dạng YOLO).

Chạy:  python training/eval_person.py --data datasets/coco128 --out reports/person_eval.json
Kích thước nhóm theo COCO (diện tích hộp GT tính bằng pixel gốc): small <32², medium 32²–96², large >96².
Ghép hộp tham lam theo IoU>=0.5 (không phải mAP đầy đủ; precision/recall ở một ngưỡng conf).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
import runtime  # noqa: E402

# Mô hình -> (trọng số, {nhãn mô hình: nhóm})
MODELS = {
    'gian_lan': ('models/gian_lan.pt', {'person': 'person', 'cell-phones': 'phone'}),
    'yolo11n_pose': ('models/yolo11n-pose.pt', {'person': 'person'}),
    'yolo11n_coco': ('models/yolo11n_coco.pt', {'person': 'person', 'cell phone': 'phone'}),
}
GT_CLASSES = {0: 'person', 67: 'phone'}  # id COCO trong coco128
BUCKETS = (('small', 0, 32 ** 2), ('medium', 32 ** 2, 96 ** 2), ('large', 96 ** 2, float('inf')))


def bucket(area):
    return next(n for n, lo, hi in BUCKETS if lo <= area < hi)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_gt(data):
    items = []
    for img in sorted((data / 'images/train2017').glob('*.jpg')):
        lab = data / 'labels/train2017' / (img.stem + '.txt')
        h, w = cv2.imread(str(img)).shape[:2]
        gts = []
        if lab.is_file():
            for line in lab.read_text().splitlines():
                c, x, y, bw, bh = line.split()[:5]
                if int(c) in GT_CLASSES:
                    x, y, bw, bh = float(x)*w, float(y)*h, float(bw)*w, float(bh)*h
                    gts.append((GT_CLASSES[int(c)], [x-bw/2, y-bh/2, x+bw/2, y+bh/2]))
        items.append((img, gts))
    return items


def evaluate(name, weights, mapping, items, imgsz, conf, iou_nms):
    model = runtime.get_yolo(ROOT / weights)
    stats = {}  # (cls,bucket) -> tp,fn ; (cls,'pred') -> tp,fp
    lat = []
    for img, gts in items:
        frame = cv2.imread(str(img))
        t = time.perf_counter()
        r = model.predict(frame, conf=conf, iou=iou_nms, imgsz=imgsz, verbose=False, device='cpu')[0]
        lat.append((time.perf_counter() - t) * 1000)
        preds = [(mapping[r.names[int(b.cls)]], b.xyxy[0].tolist()) for b in r.boxes if r.names[int(b.cls)] in mapping]
        for cls in set(mapping.values()):
            g = [(box, bucket((box[2]-box[0])*(box[3]-box[1]))) for c, box in gts if c == cls]
            p = [box for c, box in preds if c == cls]
            used = set()
            for box, bk in g:
                best, bi = 0, None
                for i, pb in enumerate(p):
                    if i not in used and iou(box, pb) > best:
                        best, bi = iou(box, pb), i
                hit = best >= 0.5
                if hit:
                    used.add(bi)
                s = stats.setdefault((cls, bk), [0, 0]); s[0 if hit else 1] += 1
            s = stats.setdefault((cls, 'pred'), [0, 0]); s[0] += len(used); s[1] += len(p) - len(used)
    out = {}
    for (cls, bk), (a, b) in stats.items():
        d = out.setdefault(cls, {})
        if bk == 'pred':
            d['precision'] = round(a / (a + b), 3) if a + b else None
            d['pred_total'], d['false_positives'] = a + b, b
        else:
            d[f'recall_{bk}'] = round(a / (a + b), 3) if a + b else None
            d[f'gt_{bk}'] = a + b
    for cls, d in out.items():
        tp = sum(round(d[f'recall_{k}'] * d[f'gt_{k}']) for k, *_ in BUCKETS if d.get(f'gt_{k}'))
        n = sum(d.get(f'gt_{k}', 0) for k, *_ in BUCKETS)
        d['recall_all'] = round(tp / n, 3) if n else None
        if d.get('precision') and d['recall_all']:
            d['f1'] = round(2 * d['precision'] * d['recall_all'] / (d['precision'] + d['recall_all']), 3)
    return {'classes': out, 'latency_ms_mean': round(float(np.mean(lat))), 'latency_ms_p95': round(float(np.percentile(lat, 95)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='datasets/coco128')
    ap.add_argument('--out', default='reports/person_eval.json')
    ap.add_argument('--imgsz', type=int, nargs='+', default=[640, 960, 1280])
    ap.add_argument('--conf', type=float, nargs='+', default=[0.25])
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    a = ap.parse_args()
    runtime.load_torch()
    items = load_gt(ROOT / a.data)
    results = []
    for name in a.models:
        weights, mapping = MODELS[name]
        for sz in a.imgsz:
            for cf in a.conf:
                res = evaluate(name, weights, mapping, items, sz, cf, a.iou)
                results.append({'model': name, 'imgsz': sz, 'conf': cf, 'iou_nms': a.iou, **res})
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    (ROOT / a.out).write_text(json.dumps({'dataset': a.data, 'images': len(items), 'results': results}, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
