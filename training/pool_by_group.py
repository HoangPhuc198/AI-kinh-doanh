"""Gộp kết quả eval_by_group.py của NHIỀU dataset để so sánh nguồn/mô hình, theo nhóm (nhóm của mỗi dataset tách biệt).

  python training/pool_by_group.py                       # mọi dataset đã đánh giá, model=ensemble
  python training/pool_by_group.py --model ensemble_v2

In: (1) mỗi nguồn: số liệu gộp trên split được đánh giá + J theo từng dataset + J theo split (train/valid/test) để nhận ra nguồn đã
"thấy" dữ liệu; (2) hệ thống ở hai điểm vận hành. Ghi reports/eval_by_group_pooled[_<model>].json.
Quy tắc chọn nguồn là của người dùng/README, KHÔNG nằm trong script này: script chỉ báo số, không tự chọn.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training'))
import group_eval as G  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')  # console Windows mặc định cp1252 không in được tiếng Việt

CACHE = ROOT / '.tmp' / 'eval_by_group'


def load(model):
    sets = {}
    for f in sorted(CACHE.glob(f'units_*_{model}.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        sets[f.stem[len('units_'):-len(model) - 1]] = d['units']
    return sets


def pooled(sets, pred, boots, seed, sel=lambda u: True):
    groups, truth, p = [], [], []
    for di, (name, units) in enumerate(sets.items()):
        for u in units:
            if sel(u):
                groups.append((di, u['group'])); truth.append(u['cheat']); p.append(pred(u))
    ids = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    return G.summarise([ids[g] for g in groups], truth, p, boots, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='ensemble'); ap.add_argument('--bootstrap', type=int, default=2000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-splits', nargs='+', default=['valid', 'test'], help='split dùng cho số liệu chính')
    a = ap.parse_args()
    sets = load(a.model)
    if not sets:
        raise SystemExit(f'Chưa có units_*_{a.model}.json: chạy eval_by_group.py trước.')
    main_sel = lambda u: u['split'] in a.eval_splits  # noqa: E731
    srcs = sorted({s for us in sets.values() for u in us for s in u['src']})
    jfmt = lambda r: ('  -  ' if r['youden_j'] is None else f"{r['youden_j']:+.2f}")  # noqa: E731

    def row(name, pred):
        allr = pooled(sets, pred, a.bootstrap, a.seed, main_sel)
        per_ds = {d: pooled({d: us}, pred, 0, a.seed, main_sel) for d, us in sets.items()}
        per_split = {sp: pooled(sets, pred, 0, a.seed, lambda u, sp=sp: u['split'] == sp) for sp in G.SPLITS}
        ci = allr['ci_group_bootstrap'].get('youden_j')
        print(f"{name:<18} P {allr['precision'] if allr['precision'] is not None else '-'} R {allr['recall']} FAR {allr['false_alarm_rate']} | "
              f"J {jfmt(allr)}{'' if not ci else f' [{ci[0]:+.2f},{ci[1]:+.2f}]'} | J/dataset " + ' '.join(f'{d[:6]}:{jfmt(r)}' for d, r in per_ds.items())
              + ' | J/split ' + ' '.join(f'{sp}:{jfmt(r)}' for sp, r in per_split.items()))
        return {'pooled': allr, 'per_dataset': per_ds, 'per_split': per_split}
    print(f"mô hình={a.model} | dataset: {list(sets)} | split chính: {a.eval_splits}\n")
    out = {'meta': {'model': a.model, 'datasets': list(sets), 'eval_splits': a.eval_splits, 'bootstrap': a.bootstrap, 'seed': a.seed}, 'system': {}, 'per_source': {}}
    print('[hệ thống]')
    for k in ('suspect_or_above', 'watch_or_above'):
        out['system'][k] = row(k, lambda u, k=k: u['ops'][k])
    print('\n[từng nguồn: bằng chứng >= ngưỡng bằng chứng]')
    for s in srcs:
        out['per_source'][s] = row(s, lambda u, s=s: u['src'].get(s, False))
    dst = ROOT / 'reports' / f"eval_by_group_pooled{'' if a.model == 'ensemble' else '_' + a.model}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\nĐã lưu', dst)


if __name__ == '__main__':
    main()
