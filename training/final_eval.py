"""Đánh giá cuối có kiểm soát overfit / ổn định cho model person + cell phone.

Quy trình (chống rò rỉ): suy luận một lần ở conf thấp -> chọn ngưỡng conf TRÊN TẬP VAL (recall cao nhất mà precision >= mục tiêu)
-> báo cáo TRÊN TẬP TEST với ngưỡng đã khóa, kèm khoảng tin cậy 95% bootstrap theo ảnh. So sánh train/val/test để phát hiện overfit.
Ghép hộp tham lam IoU>=0.5 theo từng lớp. Recall 'medium+large' bỏ người <32 px (không thực tế với webcam phòng thi).

  python training/final_eval.py --data datasets/examvision_v2 --imgsz 800 --out reports/final_eval.json \
      --model n models/yolo11n_coco.pt coco --model v2 runs/train/person_phone_v2b/weights/best.pt two
'coco' = model 80 lớp (person=0, cell phone=67); 'two' = model 2 lớp (0 person, 1 cell phone).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
import runtime  # noqa: E402

CLS = {0: 'person', 1: 'phone'}
MIN_CONF = 0.10


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0


def load_split(data, split):
    items = []
    for img in sorted((data / 'images' / split).glob('*.jpg')):
        frame = cv2.imdecode(np.fromfile(str(img), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gts = []
        lab = data / 'labels' / split / (img.stem + '.txt')
        if lab.is_file():
            for line in lab.read_text().splitlines():
                c, x, y, bw, bh = line.split()[:5]
                x, y, bw, bh = float(x)*w, float(y)*h, float(bw)*w, float(bh)*h
                gts.append((int(c), [x-bw/2, y-bh/2, x+bw/2, y+bh/2]))
        items.append((frame, gts))
    return items


def predict(model, kind, items, imgsz, iou_nms):
    out = []
    for frame, _ in items:
        r = model.predict(frame, conf=MIN_CONF, iou=iou_nms, imgsz=imgsz, verbose=False, device=runtime.device(), half=runtime.use_half())[0]
        preds = []
        for b in r.boxes:
            c = int(b.cls)
            c = {0: 0, 67: 1}.get(c) if kind == 'coco' else c
            if c is not None:
                preds.append((c, float(b.conf), b.xyxy[0].tolist()))
        out.append(preds)
    return out


def match(gts, preds, cls, conf):
    """Trả (tp, fp, [(gt_ok, là_trung_bình_hoặc_lớn)])."""
    g = [b for c, b in gts if c == cls]
    p = sorted([(s, b) for c, s, b in preds if c == cls and s >= conf], reverse=True)
    used, hits = set(), []
    for gb in g:
        area = (gb[2]-gb[0])*(gb[3]-gb[1])
        hits.append([False, area >= 32 ** 2])
    for s, pb in p:
        best, bi = 0.0, None
        for i, gb in enumerate(g):
            if i not in used and iou(gb, pb) > best:
                best, bi = iou(gb, pb), i
        if best >= 0.5:
            used.add(bi); hits[bi][0] = True
    return len(used), len(p) - len(used), hits


def counts(items, preds, cls, conf, idx=None):
    tp = fp = 0; hit = tot = hit_ml = tot_ml = 0
    for k in (range(len(items)) if idx is None else idx):
        t, f, hits = match(items[k][1], preds[k], cls, conf)
        tp += t; fp += f
        for ok, ml in hits:
            tot += 1; hit += ok
            if ml:
                tot_ml += 1; hit_ml += ok
    return tp, fp, hit, tot, hit_ml, tot_ml


def prf(c):
    tp, fp, hit, tot, hit_ml, tot_ml = c
    return {'precision': round(tp / (tp + fp), 3) if tp + fp else None, 'recall': round(hit / tot, 3) if tot else None,
            'recall_medium_large': round(hit_ml / tot_ml, 3) if tot_ml else None, 'gt': tot, 'false_positives': fp}


def bootstrap(items, preds, cls, conf, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ps, rs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(items), len(items))
        tp, fp, hit, tot, *_ = counts(items, preds, cls, conf, idx)
        if tp + fp and tot:
            ps.append(tp / (tp + fp)); rs.append(hit / tot)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return {'precision_ci95': q(ps), 'recall_ci95': q(rs)}


def pick_conf(items, preds, cls, target_p):
    """Conf thấp nhất (recall cao nhất) trên VAL mà precision >= target_p; không có thì conf cho precision lớn nhất."""
    best = None
    for conf in np.round(np.arange(0.15, 0.91, 0.05), 2):
        c = counts(items, preds, cls, float(conf))
        p = c[0] / (c[0] + c[1]) if c[0] + c[1] else 0.0
        r = c[2] / c[3] if c[3] else 0.0
        if p >= target_p:
            return float(conf)
        if best is None or p > best[1]:
            best = (float(conf), p)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--imgsz', type=int, default=800); ap.add_argument('--out', required=True)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--target-person', type=float, default=0.90); ap.add_argument('--target-phone', type=float, default=0.85)
    ap.add_argument('--model', nargs=3, action='append', metavar=('NAME', 'PATH', 'KIND'), required=True)
    a = ap.parse_args()
    runtime.load_torch()
    data = ROOT / a.data
    splits = {s: load_split(data, s) for s in ('train', 'val', 'test')}
    splits['train'] = splits['train'][::3]  # mẫu 1/3 train đủ để so sánh overfit
    targets = {0: a.target_person, 1: a.target_phone}
    result = {'data': a.data, 'imgsz': a.imgsz, 'images': {k: len(v) for k, v in splits.items()}, 'targets': {CLS[k]: v for k, v in targets.items()}, 'models': {}}
    for name, path, kind in a.model:
        model = runtime.get_yolo(ROOT / path)
        preds = {s: predict(model, kind, items, a.imgsz, a.iou) for s, items in splits.items()}
        res = {}
        for cls, cname in CLS.items():
            conf = pick_conf(splits['val'], preds['val'], cls, targets[cls])
            res[cname] = {'conf_chosen_on_val': conf}
            for s in ('train', 'val', 'test'):
                res[cname][s] = prf(counts(splits[s], preds[s], cls, conf))
            res[cname]['test'].update(bootstrap(splits['test'], preds['test'], cls, conf))
        result['models'][name] = res
        print(name, json.dumps(res, ensure_ascii=False), flush=True)
    (ROOT / a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
