"""Đo ExamVision Ensemble trên dataset ảnh có nhãn từ ngoài (định dạng YOLO, ví dụ tải từ Roboflow).

  python training/eval_external.py "datasets/exam malpractice detection2.v18i.yolov11" --cheat cheating --normal normal

Chỉ đọc dataset, chỉ ghi kết quả vào reports/external_<tên>.json. Dùng tập valid + test.
Mỗi ảnh chạy riêng (không có thời gian), nên Judge dùng chế độ ảnh đơn (min_frames=1) - chỉ phản ánh từng khung hình.
Khớp theo người: khung nhãn ghép với người Ensemble có IoU cao nhất (>= --iou); không có thì tính là người bị bỏ sót (không phát hiện).
Kết quả chỉ để tham khảo: nhãn do người ngoài gán, chưa được kiểm chứng.
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
import runtime  # noqa: E402

runtime.load_torch()
import inference  # noqa: E402
from judge import LABEL_CHEAT, LABEL_SUSPECT, Settings  # noqa: E402

PRESETS = {
    'strict': dict(person_conf=0.45, phone_conf=0.5, evidence_conf=0.65, suspect_score=0.25, cheat_score=0.6, min_seconds=3, suspect_seconds=1.5, recover_seconds=1),
    'bal': dict(person_conf=0.35, phone_conf=0.35, evidence_conf=0.55, suspect_score=0.15, cheat_score=0.45, min_seconds=1.5, suspect_seconds=1.0, recover_seconds=1),
    'sens': dict(person_conf=0.3, phone_conf=0.25, evidence_conf=0.4, suspect_score=0.1, cheat_score=0.35, min_seconds=1, suspect_seconds=0.6, recover_seconds=1),
}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def sens(s):
    lerp = lambda a, b, t: {k: round(a[k] + (b[k] - a[k]) * t, 2) for k in a}  # noqa: E731
    return lerp(PRESETS['strict'], PRESETS['bal'], s * 2) if s <= 0.5 else lerp(PRESETS['bal'], PRESETS['sens'], (s - 0.5) * 2)


def load_gt(label_file, cheat_ids, normal_ids):
    out = []
    for ln in label_file.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        if c not in cheat_ids and c not in normal_ids:
            continue
        cx, cy, w, h = map(float, p[1:5])
        out.append({'cheat': c in cheat_ids, 'box': [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
    return out


def rate(tp, fp, fn, tn):
    f = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': f(tp, tp + fp), 'recall': f(tp, tp + fn), 'false_alarm_rate': f(fp, fp + tn)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--cheat', nargs='+', required=True, help='tên lớp được coi là gian lận')
    ap.add_argument('--normal', nargs='+', required=True, help='tên lớp được coi là bình thường')
    ap.add_argument('--splits', nargs='+', default=['valid', 'test'])
    ap.add_argument('--iou', type=float, default=0.4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dump', help='thư mục để lưu ảnh các ca báo nhầm / bắt được ở độ nhạy Cân bằng (để xem bằng mắt)')
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
    if a.limit:
        items = items[:a.limit]
    inference.warm_model('ensemble')
    levels = (0.0, 0.5, 1.0)
    cfgs = {s: Settings.from_dict(sens(s)) for s in levels}
    acc = {s: {'person': [0, 0, 0, 0], 'image': [0, 0, 0, 0], 'unmatched_cheat': 0, 'unmatched_normal': 0} for s in levels}
    dumps = {'fp': [], 'tp': []}
    for n, (img, gt) in enumerate(items):
        frame = cv2.imdecode(np.fromfile(str(img), np.uint8), cv2.IMREAD_COLOR)
        if frame is None or not gt:
            continue
        for s in levels:
            r = inference.run_model('ensemble', frame, 0.5, include_image=False, state=None, cfg=cfgs[s])
            people = [d for d in r['details'] if d.get('id') is not None]
            flagged_any = False
            for g in gt:
                best, bi = 0.0, None
                for j, p in enumerate(people):
                    v = iou(g['box'], p['box'])
                    if v > best:
                        best, bi = v, j
                if bi is None or best < a.iou:
                    acc[s]['unmatched_cheat' if g['cheat'] else 'unmatched_normal'] += 1
                    continue
                pred = people[bi]['label'] in (LABEL_SUSPECT, LABEL_CHEAT)
                flagged_any |= pred
                if s == 0.5 and pred:
                    dumps['tp' if g['cheat'] else 'fp'].append((img, g['box'], people[bi]['box'], people[bi]['label'], people[bi].get('evidence') or []))
                k = (0 if g['cheat'] else 2) + (0 if pred else 1)  # tp, fn, fp, tn
                acc[s]['person'][{0: 0, 1: 2, 2: 1, 3: 3}[k]] += 1
            truth_any = any(g['cheat'] for g in gt)
            acc[s]['image'][{(True, True): 0, (True, False): 2, (False, True): 1, (False, False): 3}[(truth_any, flagged_any)]] += 1
        if (n + 1) % 50 == 0:
            print(f'  {n + 1}/{len(items)} ảnh', flush=True)
    out = {'dataset': ds.name, 'classes': names, 'cheat': a.cheat, 'normal': a.normal, 'images': len(items), 'iou': a.iou, 'by_sensitivity': {}}
    for s in levels:
        v = acc[s]
        out['by_sensitivity'][str(s)] = {'theo_nguoi': rate(*v['person']), 'theo_anh': rate(*v['image']),
                                         'khong_ghep_duoc': {'gian_lan': v['unmatched_cheat'], 'binh_thuong': v['unmatched_normal']}}
    from collections import Counter
    for kind, rows in dumps.items():
        print(f'nguồn bằng chứng ở các ca "{kind}" (Cân bằng): ', dict(Counter(e for r in rows for e in r[4]).most_common(6)), f'({len(rows)} ca, {len({r[0] for r in rows})} ảnh khác nhau)')
    if a.dump:
        d = Path(a.dump); d.mkdir(parents=True, exist_ok=True)
        for kind, rows in dumps.items():
            tiles = []
            for img, gb, pb, lb, _ in rows[:24]:
                im = cv2.imdecode(np.fromfile(str(img), np.uint8), cv2.IMREAD_COLOR); h, w = im.shape[:2]
                for b, col in ((gb, (0, 200, 0) if kind == 'fp' else (0, 0, 255)), (pb, (255, 160, 0))):
                    cv2.rectangle(im, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), col, 4)
                tiles.append(cv2.resize(im, (480, 360)))
            while tiles and len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            if tiles:
                cv2.imencode('.jpg', np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))[1].tofile(str(d / f'{kind}_{ds.name[:12].replace(" ", "_")}.jpg'))
    dst = ROOT / 'reports' / f'external_{ds.name}.json'
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    for s, v in out['by_sensitivity'].items():
        print(f"độ nhạy {s}: theo người {v['theo_nguoi']} | theo ảnh {v['theo_anh']} | không ghép được {v['khong_ghep_duoc']}")
    print('Đã lưu', dst)


if __name__ == '__main__':
    main()
