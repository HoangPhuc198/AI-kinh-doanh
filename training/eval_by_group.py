"""Đánh giá THEO NHÓM một mô hình tổng hợp (mặc định `ensemble`) trên dataset ảnh có nhãn định dạng YOLO.

  python training/eval_by_group.py "datasets/exam malpractice detection2.v18i.yolov11"
  python training/eval_by_group.py "datasets/exam_cheating.v2i.yolov11" --splits train valid test --model ensemble_v2

Khác eval_external.py: (1) ảnh được gom thành NHÓM (xem training/group_eval.py) và khoảng tin cậy lấy mẫu lại cả nhóm, không coi
từng ảnh là độc lập; (2) báo cáo mức rò rỉ giữa train/valid/test; (3) báo từng nguồn bằng chứng (yolo_duyen, gianlan, pose, ...) riêng,
để biết nguồn nào gây báo nhầm; (4) báo theo split để phát hiện mô hình đã "thấy" ảnh (train tốt hơn hẳn test).

Cách tính: mỗi khung nhãn (một người hoặc vật) ghép với người có IoU cao nhất (>= --iou). Không ghép được = hệ thống không báo
(tính là bỏ sót nếu nhãn gian lận). Mỗi ảnh chạy riêng (không có thời gian) nên chỉ phản ánh từng khung hình; nhãn do người ngoài gán.
Kết quả suy luận được lưu cache (.tmp/eval_by_group/), chạy lại chỉ tính lại chỉ số; cache gắn với mã nguồn + trọng số + ngưỡng nên
đổi mã thì tự chạy lại. Ghi: reports/eval_by_group_<tên dataset>[_<model>].json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT / 'training'))
import runtime  # noqa: E402

runtime.load_torch()
import group_eval as G  # noqa: E402
import inference  # noqa: E402
from eval_external import sens  # noqa: E402
from judge import Settings  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')  # console Windows mặc định cp1252 không in được tiếng Việt
CACHE = ROOT / '.tmp' / 'eval_by_group'
# Quy ước lớp: giống reports/external_*.json (đã dùng ở các lần đo trước) để số liệu so sánh được.
KNOWN = {
    'Cheating Detection.v4i.yolov11': (['earphone', 'hand_gestures', 'head_rotation', 'mobile_phone', 'paper_passing', 'peeking', 'smartwatch'], ['normal']),
    'exam malpractice detection2.v18i.yolov11': (['cheating'], ['normal']),
    'exam_cheating.v2i.yolov11': (['Cheat_paper', 'Look_around', 'Use_phone'], ['Normal']),
}
OPS = {  # điểm vận hành của hệ thống ở khung đơn lẻ
    'suspect_or_above': lambda p: p['label'] != 'Bình thường',    # nhãn Nghi ngờ/Gian lận (ở khung đơn: tức nhiều nguồn đồng thuận)
    'watch_or_above': lambda p: p['frame_tier'] in ('watch', 'suspect'),  # kể cả 'Cần chú ý' (một nguồn mạnh)
}


# Tệp mã quyết định ĐẦU RA của từng mô hình (băm vào dấu vân tay cache). Không gồm inference.py/app.py: thêm mô hình mới ở đó không làm
# đổi kết quả mô hình cũ; thay vào đó băm các hằng số kích thước ảnh của inference (xem fingerprint).
CODE_FILES = {'ensemble': ('ensemble.py', 'judge.py', 'paper_detector.py'), 'ensemble_v2': ('ensemble_v2.py', 'judge.py')}  # v2 tự chứa: không phụ thuộc ensemble.py


def fingerprint(model: str, cfg: Settings, all_sources: bool) -> str:
    gpu_imgsz = inference.ensemble_module(model).GPU_IMGSZ if model in inference.ENSEMBLES else None   # chỉ nạp module của chính mô hình đang đo
    h = hashlib.sha1(json.dumps([model, cfg.to_dict(), all_sources, inference.DEFAULT_IMGSZ, inference.ENSEMBLE_IMGSZ, gpu_imgsz], sort_keys=True).encode())
    for name in CODE_FILES.get(model, sorted(p.name for p in (ROOT / 'app').glob('*.py'))):
        f = ROOT / 'app' / name
        h.update(name.encode()); h.update(f.read_bytes() if f.exists() else b'')
    for f in sorted((ROOT / 'models').glob('*')):
        if f.is_file():
            h.update(f'{f.name}:{f.stat().st_size}'.encode())
    return h.hexdigest()[:12]


def read_img(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def collect(ds: Path):
    items = []
    for sp in G.SPLITS:
        for img in sorted((ds / sp / 'images').glob('*.jpg')):
            items.append({'path': img, 'split': sp, 'rel': f'{sp}/{img.name}', 'label': ds / sp / 'labels' / (img.stem + '.txt')})
    return items


def hashes(ds: Path, items, slug):
    """dHash (thường + lật) của mọi ảnh ở mọi split (cache theo đường dẫn + kích thước tệp)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'hash_{slug}.json'
    cache = json.loads(f.read_text()) if f.exists() else {}
    t0, dirty = time.time(), False
    for n, it in enumerate(items):
        key = f"{it['rel']}|{it['path'].stat().st_size}"
        if key not in cache:
            im = read_img(it['path'])
            cache[key] = [str(G.dhash64(im)), str(G.dhash64(im, True))] if im is not None else None
            dirty = True
        if (n + 1) % 1000 == 0:
            print(f'  dHash {n + 1}/{len(items)} ({time.time() - t0:.0f}s)', flush=True)
    if dirty:
        f.write_text(json.dumps(cache))
    return [cache[f"{it['rel']}|{it['path'].stat().st_size}"] for it in items]


def infer_rows(items, a, cfg, fp, slug):
    """Chạy mô hình trên các ảnh của --splits; trả về {rel: row}. Có cache + tiếp tục khi bị ngắt."""
    f = CACHE / f'pred_{slug}_{a.model}_{fp}.jsonl'
    rows = {}
    if f.exists():
        for ln in f.read_text(encoding='utf-8').splitlines():
            r = json.loads(ln)
            rows[r['rel']] = r
    todo = [it for it in items if it['split'] in a.splits and it['rel'] not in rows]
    if a.limit:
        todo = todo[:a.limit]
    if todo:
        inference.warm_model(a.model)
        t0 = time.time()
        with open(f, 'a', encoding='utf-8') as out:
            for n, it in enumerate(todo):
                frame = read_img(it['path'])
                people = []
                ms = None
                if frame is not None:
                    r = inference.run_model(a.model, frame, 0.5, include_image=False, state=None, cfg=cfg)
                    ms = r['ms']
                    people = [{'box': d['box'], 'label': d['label'], 'frame_tier': d.get('frame_tier'), 'votes': d.get('votes', {}), 'person_conf': d.get('person_conf')}
                              for d in r['details'] if d.get('id') is not None]
                row = {'rel': it['rel'], 'people': people, 'ms': ms}
                rows[it['rel']] = row
                out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush()
                if (n + 1) % 50 == 0:
                    el = time.time() - t0
                    print(f'  suy luận {n + 1}/{len(todo)} ảnh ({el:.0f}s, còn ~{el / (n + 1) * (len(todo) - n - 1):.0f}s)', flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--model', default='ensemble', help="khóa mô hình trong inference.MODELS (vd. ensemble, ensemble_v2)")
    ap.add_argument('--cheat', nargs='+'); ap.add_argument('--normal', nargs='+')
    ap.add_argument('--splits', nargs='+', default=['valid', 'test'], choices=G.SPLITS, help='split được ĐÁNH GIÁ (dHash luôn tính cả 3 split để đo rò rỉ)')
    ap.add_argument('--iou', type=float, default=0.4)
    ap.add_argument('--sensitivity', type=float, default=0.5, help='0 Chặt .. 0.5 Cân bằng .. 1 Nhạy (như thanh Độ nhạy ở trang webcam)')
    ap.add_argument('--group-by', choices=('auto', 'stem', 'regex'), default='auto', help='auto = tên ảnh gốc + dHash gần nhau; regex = mã video trong tên file')
    ap.add_argument('--group-regex', help='regex có 1 nhóm bắt = mã video (dùng với --group-by regex)')
    ap.add_argument('--dhash-threshold', type=int, default=6, help='Hamming tối đa (/64 bit) coi là gần trùng')
    ap.add_argument('--bootstrap', type=int, default=2000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='chỉ chạy suy luận N ảnh mới (thử nhanh)')
    ap.add_argument('--no-all-sources', action='store_true', help='giữ cơ chế chỉ chạy KNN/Training khi cần (mặc định: luôn chạy để đo từng nguồn; nhãn cuối không đổi)')
    a = ap.parse_args()
    ds = Path(a.dataset)
    names = yaml.safe_load((ds / 'data.yaml').read_text(encoding='utf-8'))['names']
    cheat_names, normal_names = (a.cheat, a.normal) if a.cheat and a.normal else KNOWN.get(ds.name, (None, None))
    if not cheat_names:
        raise SystemExit(f'Dataset lạ: cần --cheat và --normal. Lớp có: {names}')
    cls = {i: n for i, n in enumerate(names)}
    cheat_ids = {i for i, n in cls.items() if n in cheat_names}
    normal_ids = {i for i, n in cls.items() if n in normal_names}
    slug = re.sub(r'[^A-Za-z0-9]+', '_', ds.name)[:40]
    cfg = Settings.from_dict(sens(a.sensitivity))
    if not a.no_all_sources and a.model == 'ensemble':   # chỉ v1 có thác KNN/Training; v2 không có nên không nạp v1 khi đo v2
        import ensemble as _ens
        if hasattr(_ens, '_needs_knn'):
            _ens._needs_knn = lambda *x, **k: True  # chỉ cho phép đo: nhãn cuối không đổi vì nguồn yếu không tự gây cờ
    items = collect(ds)
    print(f'{ds.name}: {len(items)} ảnh (3 split); đánh giá {a.splits}; nhóm theo {a.group_by}')
    hs = hashes(ds, items, slug)
    ok = [i for i, h in enumerate(hs) if h]
    items = [items[i] for i in ok]; hs = [hs[i] for i in ok]
    h, hf = [int(x[0]) for x in hs], [int(x[1]) for x in hs]
    pairs = G.near_duplicate_pairs(h, hf, a.dhash_threshold)
    stems = [G.source_stem(it['path'].name) for it in items]
    if a.group_by == 'regex':
        if not a.group_regex:
            raise SystemExit('--group-by regex cần --group-regex')
        groups = G.build_groups([G.video_key(it['path'].name, a.group_regex) for it in items], (), by_key=True)
    else:
        groups = G.build_groups(stems, pairs if a.group_by == 'auto' else (), by_key=True)
    splits = [it['split'] for it in items]
    leak = G.leakage_report(stems, splits, pairs)
    sizes = Counter(Counter(groups).values())
    print(f"  nhóm: {len(set(groups))} (từ {len(items)} ảnh); rò rỉ xác nhận: {leak['fraction_with_confirmed_near_duplicate_in_train']}; "
          f"tên gốc ở nhiều split: {leak['source_names_in_multiple_splits']} (xác nhận cùng ảnh {leak['of_which_confirmed_same_image']})")

    fp = fingerprint(a.model, cfg, not a.no_all_sources)
    rows = infer_rows(items, a, cfg, fp, slug)

    # ---- đơn vị đánh giá: một khung nhãn ----
    units = []
    for it, g in zip(items, groups):  # mọi split đã có kết quả trong cache (số liệu chính chỉ lấy --splits; by_split thì lấy hết)
        if it['rel'] not in rows or not it['label'].exists():
            continue
        people = rows[it['rel']]['people']
        for c, box in G.read_yolo_labels(it['label']):
            if c not in cheat_ids and c not in normal_ids:
                continue
            j = G.match_box(box, [p['box'] for p in people], a.iou)
            p = people[j] if j is not None else None
            u = {'group': g, 'split': it['split'], 'cls': cls[c], 'cheat': c in cheat_ids, 'matched': p is not None, 'ops': {}, 'src': {}}
            for k, fn in OPS.items():
                u['ops'][k] = bool(p and fn(p))
            for s, v in (p['votes'] if p else {}).items():
                u['src'][s] = v >= cfg.evidence_conf
            units.append(u)
    main_units = [u for u in units if u['split'] in a.splits]
    if not main_units:
        raise SystemExit('Không có đơn vị đánh giá nào ở --splits (chưa chạy suy luận? thử bỏ --limit).')
    srcs = sorted({s for u in units for s in u['src']})
    (CACHE / f'units_{slug}_{a.model}.json').write_text(json.dumps({'splits': sorted({u['split'] for u in units}), 'units': units}, ensure_ascii=False), encoding='utf-8')  # cho pool_by_group.py

    def summ(sel, pred):
        sel = list(sel)
        return G.summarise([u['group'] for u in sel], [u['cheat'] for u in sel], [pred(u) for u in sel], a.bootstrap, a.seed)

    def block(pred):
        return {'end_to_end': summ(main_units, pred), 'matched_only': summ([u for u in main_units if u['matched']], pred),
                'by_split': {sp: summ([u for u in units if u['split'] == sp], pred) for sp in G.SPLITS if any(u['split'] == sp for u in units)}}
    out = {
        'meta': {'date': time.strftime('%Y-%m-%d %H:%M'), 'dataset': ds.name, 'model': a.model, 'args': {k: v for k, v in vars(a).items() if k != 'dataset'},
                 'settings': cfg.to_dict(), 'fingerprint': fp, 'seed': a.seed, 'bootstrap': a.bootstrap},
        'classes': {'cheat': cheat_names, 'normal': normal_names},
        'grouping': {'mode': a.group_by, 'dhash_threshold': a.dhash_threshold, 'images_all_splits': len(items), 'groups': len(set(groups)), 'group_size_histogram': dict(sorted(sizes.items()))},
        'leakage': leak,
        'evaluated': {'images': sum(1 for it in items if it['split'] in a.splits and it['rel'] in rows), 'units': len(main_units),
                      'units_cheat': sum(u['cheat'] for u in main_units), 'units_matched': sum(u['matched'] for u in main_units), 'groups': len({u['group'] for u in main_units}),
                      'main_splits': a.splits, 'splits_in_cache': {sp: sum(1 for it in items if it['split'] == sp and it['rel'] in rows) for sp in G.SPLITS}},
        'system': {k: block(lambda u, k=k: u['ops'][k]) for k in OPS},
        'per_source': {s: block(lambda u, s=s: u['src'].get(s, False)) for s in srcs},
        'by_class': {c: {'units': sum(u['cls'] == c for u in main_units), 'matched': sum(u['cls'] == c and u['matched'] for u in main_units),
                         **{k: sum(u['cls'] == c and u['ops'][k] for u in main_units) for k in OPS}} for c in sorted({u['cls'] for u in main_units})},
        'notes': ['Nhãn do người ngoài gán; mô hình có thể đã thấy các ảnh này khi huấn luyện (so sánh theo split để nhận ra).',
                  'Khung đơn: kết luận "Gian lận" không thể đạt (cần kéo dài theo thời gian), nên điểm vận hành là Nghi ngờ/Cần chú ý.',
                  'Khoảng tin cậy: lấy mẫu lại cả nhóm (percentile 95%); ci_if_independent là khoảng nếu coi ảnh độc lập; width_ratio > 1 = khoảng thật rộng hơn.',
                  'Rò rỉ "xác nhận" (dHash) là cận dưới; "cùng tên" là cận trên (gồm trùng tên chung chung).'],
    }
    dst = ROOT / 'reports' / f"eval_by_group_{slug}{'' if a.model == 'ensemble' else '_' + a.model}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

    def fmt(r):
        def ci(k):
            c = r['ci_group_bootstrap'].get(k)
            v = r.get(k)
            return f"{'-' if v is None else f'{v:.2f}'}{'' if not c else f' [{c[0]:.2f}-{c[1]:.2f}]'}"
        wr = r['width_ratio'].get('recall')
        return f"P {ci('precision')} | R {ci('recall')} | FAR {ci('false_alarm_rate')} | J {ci('youden_j')} | tp{r['tp']} fp{r['fp']} fn{r['fn']} tn{r['tn']} | nhóm {r['groups']}" + (f" | độ rộng x{wr}" if wr else '')
    e = out['evaluated']
    print(f"\nĐã đánh giá {e['images']} ảnh, {e['units']} khung nhãn ({e['units_cheat']} gian lận; ghép được người {e['units_matched']}), {e['groups']} nhóm")
    for k in OPS:
        print(f'\n[hệ thống: {k}] end-to-end: ' + fmt(out['system'][k]['end_to_end']))
        for sp, r in out['system'][k]['by_split'].items():
            print(f'    split {sp:<5}: ' + fmt(r))
    print('\n[từng nguồn, end-to-end]')
    for s in srcs:
        print(f'  {s:<11}: ' + fmt(out['per_source'][s]['end_to_end']))
    print('Đã lưu', dst)


if __name__ == '__main__':
    main()
