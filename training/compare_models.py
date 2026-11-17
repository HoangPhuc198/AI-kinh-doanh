"""So sánh từng mô hình của dự án trên cùng dataset ảnh có nhãn ngoài (định dạng YOLO), cùng một cách tính.

  python training/compare_models.py "datasets/<bộ>" --cheat cheating --normal normal

Với mỗi khung nhãn (một người), lấy phát hiện của mô hình có IoU cao nhất (>= --iou). Mô hình "báo" người đó nếu phát hiện ấy mang nhãn gian lận/nghi vấn.
Khung nhãn không ghép được với phát hiện nào = mô hình không báo (tính là bỏ sót nếu nhãn là gian lận, đúng nếu nhãn là bình thường).
Mô hình chỉ phát hiện người/điện thoại (person_coco, person_phone) không có kết luận gian lận nên chỉ báo tỉ lệ phát hiện được người.
Chỉ đọc dataset; chỉ ghi reports/model_compare_<tên>.json. Kết quả chỉ để tham khảo: nhãn do người ngoài gán, mô hình có thể đã thấy ảnh tương tự khi huấn luyện.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT / 'training'))
import runtime  # noqa: E402

runtime.load_torch()
import inference  # noqa: E402
from eval_external import iou, load_gt, sens  # noqa: E402
from judge import LABEL_CHEAT, LABEL_SUSPECT, Settings  # noqa: E402

CHEAT_LABELS = {'cheating', 'cheating-behaviour', 'suspicious'}
YOLO_BEHAVIOR = ('yolo_duyen', 'gianlan', 'training')
POSE_MODELS = ('duyen', 'paper')
PERSON_ONLY = ('person_coco', 'person_phone')


def pct(a, b):
    return round(a / b, 3) if b else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--cheat', nargs='+', required=True)
    ap.add_argument('--normal', nargs='+', required=True)
    ap.add_argument('--splits', nargs='+', default=['valid', 'test'])
    ap.add_argument('--iou', type=float, default=0.4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--exclude', help='file JSON: danh sách tên ảnh bỏ qua (ví dụ ảnh trùng với dữ liệu mô hình đã học)')
    ap.add_argument('--only', nargs='+', help='chỉ chạy các cấu hình có tên chứa một trong các chuỗi này')
    a = ap.parse_args()
    ds = Path(a.dataset)
    names = yaml.safe_load((ds / 'data.yaml').read_text(encoding='utf-8'))['names']
    cheat_ids = {i for i, n in enumerate(names) if n in a.cheat}
    normal_ids = {i for i, n in enumerate(names) if n in a.normal}
    if not cheat_ids or not normal_ids:
        raise SystemExit(f'Không khớp tên lớp. Dataset có: {names}')
    items = []
    for sp in a.splits:
        for img in sorted((ds / sp / 'images').glob('*.jpg')):
            lf = ds / sp / 'labels' / (img.stem + '.txt')
            if lf.exists():
                items.append((img, load_gt(lf, cheat_ids, normal_ids)))
    if a.exclude:
        skip = set(json.loads(Path(a.exclude).read_text(encoding='utf-8')))
        items = [(i, g) for i, g in items if i.name not in skip]
    if a.limit:
        items = items[:a.limit]

    # cấu hình: (tên hiển thị, loại, khóa, ngưỡng)
    configs = [(f'{k} @{c}', 'yolo', k, c) for k in YOLO_BEHAVIOR for c in (0.25, 0.5)]
    configs += [(k, 'pose', k, 0.5) for k in POSE_MODELS]
    configs += [(k, 'person', k, 0.25) for k in PERSON_ONLY]
    configs += [(f'ensemble (độ nhạy {s})', 'ensemble', 'ensemble', s) for s in (0.0, 0.5, 1.0)]
    if a.only:
        configs = [c for c in configs if any(o in c[0] for o in a.only)]
    for _, _, k, _ in configs:
        inference.warm_model(k)
    cfgs = {s: Settings.from_dict(sens(s)) for s in (0.0, 0.5, 1.0)}
    acc = {c[0]: dict(tp=0, fn=0, fp=0, tn=0, matched=0, total=0, ms=0.0) for c in configs}
    done = 0
    for img, gt in items:
        frame = cv2.imdecode(np.fromfile(str(img), np.uint8), cv2.IMREAD_COLOR)
        if frame is None or not gt:
            continue
        done += 1
        cache = {}
        for name, kind, key, p in configs:
            if kind == 'ensemble':
                r = inference.run_model('ensemble', frame, 0.5, include_image=False, state=None, cfg=cfgs[p])
                dets = [(d['box'], d['label'] in (LABEL_SUSPECT, LABEL_CHEAT)) for d in r['details'] if d.get('id') is not None]
            else:
                ck = (key, 0.25 if kind == 'yolo' else p)
                if ck not in cache:
                    cache[ck] = inference.run_model(key, frame, ck[1], include_image=False)
                r = cache[ck]
                if kind == 'person':
                    dets = [(d['box'], False) for d in r['details'] if d['label'] == 'person']
                else:
                    dets = [(d['box'], d['label'].lower() in CHEAT_LABELS) for d in r['details'] if kind == 'pose' or (d.get('confidence') or 0) >= p]
            v = acc[name]
            v['ms'] += r['ms'] if 'ms' in r else 0
            for g in gt:
                best, flag = 0.0, False
                for box, f in dets:
                    o = iou(g['box'], box)
                    if o > best:
                        best, flag = o, f
                ok = best >= a.iou
                v['total'] += 1
                v['matched'] += ok
                pred = ok and flag
                v[('tp' if pred else 'fn') if g['cheat'] else ('fp' if pred else 'tn')] += 1
        if done % 50 == 0:
            print(f'  {done}/{len(items)} ảnh', flush=True)
    out = {'dataset': ds.name, 'images': done, 'iou': a.iou, 'models': {}}
    print(f'\n{ds.name} · {done} ảnh · báo = phát hiện khớp khung nhãn (IoU>={a.iou}) mang nhãn gian lận/nghi vấn')
    print(f"{'mô hình':28} {'báo nhầm':>9} {'bắt được':>9} {'chính xác':>10} {'phát hiện người':>16} {'ms/ảnh':>7}")
    for name, kind, *_ in configs:
        v = acc[name]
        row = {'false_alarm': pct(v['fp'], v['fp'] + v['tn']), 'recall': pct(v['tp'], v['tp'] + v['fn']), 'precision': pct(v['tp'], v['tp'] + v['fp']),
               'person_found': pct(v['matched'], v['total']), 'ms_per_image': round(v['ms'] / max(done, 1)), **{k: v[k] for k in ('tp', 'fn', 'fp', 'tn')}}
        out['models'][name] = row
        f = lambda x: '-' if x is None else f'{x:.3f}'  # noqa: E731
        if kind == 'person':
            print(f"{name:28} {'-':>9} {'-':>9} {'-':>10} {f(row['person_found']):>16} {row['ms_per_image']:>7}")
        else:
            print(f"{name:28} {f(row['false_alarm']):>9} {f(row['recall']):>9} {f(row['precision']):>10} {f(row['person_found']):>16} {row['ms_per_image']:>7}")
    dst = ROOT / 'reports' / f'model_compare_{ds.name}.json'
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('Đã lưu', dst)


if __name__ == '__main__':
    main()
