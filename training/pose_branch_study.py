"""Đo tín hiệu quay đầu của YOLO11-Pose THEO NHÁNH keypoint trên dataset có nhãn (cùng quy ước lớp với eval_by_group.py).

  python training/pose_branch_study.py "datasets/exam_cheating.v2i.yolov11"

Với mỗi khung nhãn (người), ghép với người pose có IoU cao nhất (>= --iou) và ghi: nhánh (hai mắt / một mắt / ...), mức quay hiện tại
(ensemble.head_turn_evidence), và các đặc trưng hình học của trường hợp MỘT mắt. Cache: .tmp/pose_study/<bộ>.jsonl (chạy lại không tốn thời gian).
Phân tích nằm ở training/pose_branch_report.py. Chỉ đọc dataset.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT / 'training'))
import runtime  # noqa: E402

runtime.load_torch()
import ensemble as E  # noqa: E402
import group_eval as G  # noqa: E402
from eval_by_group import KNOWN, collect, read_img  # noqa: E402
import yaml  # noqa: E402

OUT = ROOT / '.tmp' / 'pose_study'
sys.stdout.reconfigure(encoding='utf-8')


def features(xy, conf, box, scale):
    """Đặc trưng một người (toạ độ ảnh gốc). Chỉ số: 0 mũi, 1-2 mắt trái/phải, 3-4 tai, 5-6 vai."""
    vis = [bool(float(c) >= E.KPT_MIN_CONF) for c in conf[:5]]
    ev = E.head_turn_evidence(xy, conf, scale)
    kind = 'none' if not vis[0] else 'both' if (vis[1] and vis[2]) else 'one' if (vis[1] != vis[2]) else 'nose_only'
    rec = {'kind': kind, 'ev': None if ev is None else round(float(ev), 3), 'conf': [round(float(c), 2) for c in conf[:7]],
           'box_w': round(float(box[2] - box[0]), 1), 'box_h': round(float(box[3] - box[1]), 1), 'scale': round(float(scale), 3),
           # keypoint thô (mũi, 2 mắt, 2 tai, 2 vai: x, y, conf) để thử đặc trưng hình học khác ngoại tuyến mà không chạy lại mô hình
           'kp': [[round(float(xy[i][0]), 1), round(float(xy[i][1]), 1), round(float(conf[i]), 2)] for i in range(7)]}
    if kind == 'one':
        e = 1 if vis[1] else 2
        rec['dx'] = round(float(abs(xy[0][0] - xy[e][0])), 1)
        rec['dy'] = round(float(xy[0][1] - xy[e][1]), 1)             # > 0: mũi thấp hơn mắt (cúi xuống)
        rec['ear'] = [int(vis[3]), int(vis[4])]
        sh = [float(c) >= E.KPT_MIN_CONF for c in conf[5:7]]
        rec['shoulder_w'] = round(float(abs(xy[5][0] - xy[6][0])), 1) if all(sh) else None
        rec['eye_side'] = 'L' if vis[1] else 'R'
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--iou', type=float, default=0.4)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    ds = Path(a.dataset)
    names = yaml.safe_load((ds / 'data.yaml').read_text(encoding='utf-8'))['names']
    cheat_names, normal_names = KNOWN[ds.name]
    cls = dict(enumerate(names))
    ids = {i for i, n in cls.items() if n in cheat_names + normal_names}
    slug = re.sub(r'[^A-Za-z0-9]+', '_', ds.name)[:40]
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f'{slug}.jsonl'
    done = {json.loads(ln)['rel'] for ln in f.read_text(encoding='utf-8').splitlines()} if f.exists() else set()
    items = [it for it in collect(ds) if it['rel'] not in done]
    if a.limit:
        items = items[:a.limit]
    model = runtime.get_yolo(E.POSE_WEIGHTS)
    t0 = time.time()
    with open(f, 'a', encoding='utf-8') as out:
        for n, it in enumerate(items):
            gts = [(cls[c], b) for c, b in G.read_yolo_labels(it['label']) if c in ids] if it['label'].exists() else []
            rec = {'rel': it['rel'], 'split': it['split'], 'units': []}
            im = read_img(it['path'])
            if im is not None and gts:
                h, w = im.shape[:2]
                r = model.predict(im, conf=0.30, verbose=False, device=runtime.device(), half=runtime.use_half(), imgsz=E.POSE_IMGSZ_FN())[0]
                scale = E.POSE_IMGSZ_FN() / max(h, w)
                boxes = [[float(v) for v in b] for b in r.boxes.xyxy.cpu().numpy()] if r.boxes is not None else []
                nb = [[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for b in boxes]
                kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []
                kc = r.keypoints.conf.cpu().numpy() if r.keypoints is not None and r.keypoints.conf is not None else None
                for name, gb in gts:
                    j = G.match_box(gb, nb, a.iou)
                    u = {'cls': name, 'cheat': name in cheat_names, 'matched': j is not None}
                    if j is not None and kc is not None and j < len(kxy):
                        u.update(features(kxy[j], kc[j], boxes[j], scale))
                        u['small'] = bool((boxes[j][3] - boxes[j][1]) < E.POSE_MIN_BOX_PX)
                    rec['units'].append(u)
            out.write(json.dumps(rec, ensure_ascii=False) + '\n'); out.flush()
            if (n + 1) % 200 == 0:
                el = time.time() - t0
                print(f'  {n + 1}/{len(items)} ({el:.0f}s, còn ~{el / (n + 1) * (len(items) - n - 1):.0f}s)', flush=True)
    print('xong', ds.name, '->', f)


if __name__ == '__main__':
    main()
