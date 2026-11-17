"""Chạy ExamVision Ensemble trên một video (không cần nhãn) và, nếu có nhãn thời gian, đo recall/precision.

  python training/analyze_video.py VIDEO.mp4                       # dòng thời gian + video đã vẽ khung
  python training/analyze_video.py VIDEO.mp4 --labels nhan.csv     # thêm recall/precision + quét độ nhạy

Kết quả ở reports/video_<tên>/ : timeline.csv (mỗi khung mẫu, mỗi người), segments.txt (các đoạn Nghi ngờ/Gian lận kèm lý do),
annotated.mp4 (khung + nhãn, phát chậm để xem), metrics.json (khi có nhãn).

File nhãn (CSV, có dòng tiêu đề): start,end,label  — giây; label = normal hoặc cheat. Đoạn không ghi nhãn bị bỏ qua khi đo.
  start,end,label
  0,8,normal
  8,14,cheat

Chạy cùng bộ kết luận (app/judge.py) như webcam/video trong ứng dụng, thời gian là thời gian trong video.
Đo theo khung mẫu ở mức VIDEO (có người nào bị kết luận không), không phân biệt từng người; kết quả chỉ phản ánh video của bạn.
"""
import argparse
import base64
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
import runtime  # noqa: E402

runtime.load_torch()
import ensemble  # noqa: E402
import inference  # noqa: E402
from judge import LABEL_CHEAT, LABEL_SUSPECT, Settings  # noqa: E402

PRESETS = {  # giống giao diện (app/live.html): Chặt (0) - Cân bằng (0.5) - Nhạy (1)
    'strict': dict(person_conf=0.45, phone_conf=0.5, evidence_conf=0.65, suspect_score=0.25, cheat_score=0.6, min_seconds=3, suspect_seconds=1.5, recover_seconds=1),
    'bal': dict(person_conf=0.35, phone_conf=0.35, evidence_conf=0.55, suspect_score=0.15, cheat_score=0.45, min_seconds=1.5, suspect_seconds=1.0, recover_seconds=1),
    'sens': dict(person_conf=0.3, phone_conf=0.25, evidence_conf=0.4, suspect_score=0.1, cheat_score=0.35, min_seconds=1, suspect_seconds=0.6, recover_seconds=1),
}


def sensitivity(s):
    lerp = lambda a, b, t: {k: round(a[k] + (b[k] - a[k]) * t, 2) for k in a}  # noqa: E731
    return lerp(PRESETS['strict'], PRESETS['bal'], s * 2) if s <= 0.5 else lerp(PRESETS['bal'], PRESETS['sens'], (s - 0.5) * 2)


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap, None
    tmp = tempfile.mkdtemp()  # OpenCV trên Windows có thể không mở được đường dẫn Unicode
    copy = Path(tmp) / ('v' + Path(path).suffix)
    shutil.copyfile(path, copy)
    cap = cv2.VideoCapture(str(copy))
    if not cap.isOpened():
        raise SystemExit(f'Không mở được video: {path}')
    return cap, tmp


def process(path, cfg, sample_fps, draw_to=None):
    """Trả về danh sách mẫu: {t, people:[{id,label,score,evidence,box}], image?}"""
    cap, tmp = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / sample_fps))
    clock = [0.0]
    ensemble._now = lambda: clock[0]  # cổng chuyển động tính theo thời gian trong video (xử lý offline nhanh hơn thời gian thực)
    state, samples, i = {}, [], 0
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / fps
            clock[0] = t
            r = inference.run_model('ensemble', frame, 0.5, include_image=draw_to is not None, state=state, cfg=cfg, now=t)
            people = [d for d in r['details'] if d.get('id') is not None]
            samples.append({'t': round(t, 2), 'people': [{k: d.get(k) for k in ('id', 'label', 'window_score', 'evidence', 'box', 'frame_tier')} for d in people]})
            if draw_to is not None:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(r['image']), np.uint8), cv2.IMREAD_COLOR)
                cv2.putText(img, f't={t:5.1f}s', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                if writer is None:
                    writer = cv2.VideoWriter(str(draw_to), cv2.VideoWriter_fourcc(*'mp4v'), min(sample_fps, 5), (img.shape[1], img.shape[0]))
                writer.write(img)
        i += 1
    cap.release()
    if writer:
        writer.release()
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return samples


def flags(samples):
    """Mỗi mẫu -> (có Nghi ngờ hoặc Gian lận, có Gian lận) ở mức video."""
    return [(any(p['label'] in (LABEL_SUSPECT, LABEL_CHEAT) for p in s['people']), any(p['label'] == LABEL_CHEAT for p in s['people'])) for s in samples]


def segments(samples, key):
    out, cur = [], None
    for s in samples:
        hit = [p for p in s['people'] if p['label'] == key]
        if hit and cur is None:
            cur = {'start': s['t'], 'end': s['t'], 'ids': set(), 'why': set()}
        if hit:
            cur['end'] = s['t']
            for p in hit:
                cur['ids'].add(p['id']); cur['why'].update(p['evidence'] or [])
        elif cur is not None:
            out.append(cur); cur = None
    if cur is not None:
        out.append(cur)
    return out


def read_labels(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = [(float(r['start']), float(r['end']), r['label'].strip().lower()) for r in csv.DictReader(f)]
    bad = [r for r in rows if r[2] not in ('normal', 'cheat') or r[1] <= r[0]]
    if bad:
        raise SystemExit(f'Nhãn sai (cần start<end, label = normal|cheat): {bad}')
    return rows


def truth_at(t, labels):
    for a, b, lb in labels:
        if a <= t < b:
            return lb == 'cheat'
    return None  # không có nhãn: bỏ qua


def metrics(samples, labels, which):
    tp = fp = fn = tn = 0
    for s, f in zip(samples, flags(samples)):
        truth = truth_at(s['t'], labels)
        if truth is None:
            continue
        pred = f[which]
        tp += truth and pred; fn += truth and not pred; fp += (not truth) and pred; tn += (not truth) and not pred
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': p, 'recall': r,
            'f1': (2 * p * r / (p + r)) if p and r else None, 'frames': tp + fp + fn + tn}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('--labels')
    ap.add_argument('--sensitivity', type=float, default=0.5, help='0 = Chặt ... 1 = Nhạy (mặc định 0.5 = Cân bằng)')
    ap.add_argument('--fps', type=float, default=5, help='số khung mẫu mỗi giây video (mặc định 5, như webcam)')
    ap.add_argument('--out')
    ap.add_argument('--no-video', action='store_true')
    a = ap.parse_args()
    video = Path(a.video)
    out = Path(a.out) if a.out else ROOT / 'reports' / f'video_{video.stem}'
    out.mkdir(parents=True, exist_ok=True)
    inference.warm_model('ensemble')
    cfg = Settings.from_dict(sensitivity(a.sensitivity))
    samples = process(video, cfg, a.fps, None if a.no_video else out / 'annotated.mp4')

    with open(out / 'timeline.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['giay', 'nguoi', 'ket_luan', 'diem_cua_so', 'bang_chung'])
        for s in samples:
            for p in s['people'] or [{'id': '', 'label': '(không thấy người)', 'window_score': '', 'evidence': []}]:
                w.writerow([s['t'], p['id'], p['label'], p['window_score'], '; '.join(p['evidence'] or [])])
    lines = [f'Video: {video.name} · {len(samples)} khung mẫu · độ nhạy {a.sensitivity}', '']
    for key in (LABEL_CHEAT, LABEL_SUSPECT):
        segs = segments(samples, key)
        lines.append(f'{key}: {len(segs)} đoạn')
        for g in segs:
            lines.append(f"  {g['start']:.1f}s - {g['end']:.1f}s · người {sorted(g['ids'])} · {'; '.join(sorted(g['why'])) or '(điểm trung bình cửa sổ)'}")
    (out / 'segments.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))

    if a.labels:
        labels = read_labels(a.labels)
        res = {'labels': labels, 'at_sensitivity': {'Gian lận': metrics(samples, labels, 1), 'Nghi ngờ hoặc Gian lận': metrics(samples, labels, 0)}, 'sweep': {}}
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            sm = process(video, Settings.from_dict(sensitivity(s)), a.fps)
            res['sweep'][str(s)] = {'Gian lận': metrics(sm, labels, 1), 'Nghi ngờ hoặc Gian lận': metrics(sm, labels, 0)}
        (out / 'metrics.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        print('\nĐộ nhạy -> (precision, recall) cho "Nghi ngờ hoặc Gian lận" / "Gian lận":')
        f = lambda m: f"P={m['precision'] if m['precision'] is None else round(m['precision'], 2)} R={m['recall'] if m['recall'] is None else round(m['recall'], 2)}"  # noqa: E731
        for s, r in res['sweep'].items():
            print(f"  {s}: {f(r['Nghi ngờ hoặc Gian lận'])}  |  {f(r['Gian lận'])}")
        print('(P/R = None: video không có khung thuộc nhóm đó, không tính được)')
    print(f'\nĐã lưu vào {out}')


if __name__ == '__main__':
    main()
