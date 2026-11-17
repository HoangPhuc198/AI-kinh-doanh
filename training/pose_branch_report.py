"""Phân tích kết quả của pose_branch_study.py: tín hiệu quay đầu hiện tại ĐÚNG/SAI ở từng nhánh keypoint, và có đặc trưng nào tách được hay không.

  python training/pose_branch_report.py

Quy ước: 'chuẩn' = nhãn bình thường; 'đầu' = nhãn hành vi liên quan đầu (head_rotation, peeking, Look_around); 'gian lận (mọi lớp)' = mọi lớp gian lận
của bộ. Số liệu CHÍNH chỉ lấy valid+test và LOẠI ảnh có bản gần trùng trong train (dHash), vì ảnh gần trùng làm số liệu lạc quan. Đặc trưng/ngưỡng
chỉ được chọn trên split TRAIN rồi đo trên valid+test sạch. Khoảng tin cậy lấy mẫu lại theo NHÓM (training/group_eval.py).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training'))
import group_eval as G  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
STUDY, CACHE = ROOT / '.tmp' / 'pose_study', ROOT / '.tmp' / 'eval_by_group'
HEAD = {'head_rotation', 'peeking', 'Look_around'}
ON = 0.55          # cùng ngưỡng bằng chứng với Settings.evidence_conf
DATASETS = ['exam_cheating.v2i.yolov11', 'exam malpractice detection2.v18i.yolov11', 'Cheating Detection.v4i.yolov11']


def slug(n):
    return re.sub(r'[^A-Za-z0-9]+', '_', n)[:40]


def load(ds_name, dhash_threshold=6):
    """Đơn vị (khung nhãn) kèm nhóm + cờ 'sạch' (ảnh valid/test không có bản gần trùng trong train)."""
    hs = json.loads((CACHE / f'hash_{slug(ds_name)}.json').read_text())
    rows = [json.loads(ln) for ln in (STUDY / f'{slug(ds_name)}.jsonl').read_text(encoding='utf-8').splitlines()]
    rels = [k.rsplit('|', 1)[0] for k in hs]
    vals = [hs[k] for k in hs]
    ok = [i for i, v in enumerate(vals) if v]
    rels = [rels[i] for i in ok]
    h, hf = [int(vals[i][0]) for i in ok], [int(vals[i][1]) for i in ok]
    pairs = G.near_duplicate_pairs(h, hf, dhash_threshold)
    stems = [G.source_stem(r.split('/', 1)[1]) for r in rels]
    groups = G.build_groups(stems, pairs)
    split = [r.split('/', 1)[0] for r in rels]
    leaked = set()
    for i, j, _ in pairs:
        if split[i] != split[j] and 'train' in (split[i], split[j]):
            leaked.add(j if split[i] == 'train' else i)
    idx = {r: i for i, r in enumerate(rels)}
    units = []
    for row in rows:
        i = idx.get(row['rel'])
        if i is None:
            continue
        for u in row['units']:
            u = dict(u)
            u.update(group=(ds_name, groups[i]), split=row['split'], clean=i not in leaked)
            units.append(u)
    return units


def flag_current(u):
    return bool(u.get('matched') and not u.get('small') and (u.get('ev') or 0) >= ON)


def auc(pos, neg):
    """P(x_pos > x_neg) (Mann-Whitney), 0,5 = không tách được."""
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    for v in np.unique(allv):                      # xử lý hòa: lấy hạng trung bình
        m = allv == v
        ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def row(name, units, pred, pos_sel):
    """pos_sel: đơn vị nào tính là dương (gian lận kiểu nào). Âm = nhãn chuẩn."""
    sel = [u for u in units if u['matched'] and (not u['cheat'] or pos_sel(u))]
    g = [hash(u['group']) for u in sel]
    ids = {x: i for i, x in enumerate(dict.fromkeys(g))}
    r = G.summarise([ids[x] for x in g], [u['cheat'] for u in sel], [pred(u) for u in sel], 2000, 0)
    c = r['ci_group_bootstrap'].get('youden_j')
    f = lambda k: '-' if r[k] is None else f'{r[k]:.2f}'  # noqa: E731
    print(f"  {name:<34} chuẩn n={r['fp'] + r['tn']:4d} báo nhầm={f('false_alarm_rate')} | gian lận n={r['tp'] + r['fn']:4d} recall={f('recall')} | "
          f"J {r['youden_j'] if r['youden_j'] is not None else float('nan'):+.2f}{'' if not c else f' [{c[0]:+.2f},{c[1]:+.2f}]'}  nhóm={r['groups']}")
    return r


def ratio(u):
    """Độ lệch ngang mũi-mắt (mắt còn thấy) chia chiều cao hộp người: thước đo quay ngang không phụ thuộc kích thước người."""
    return u['dx'] / u['box_h']


def section4(units, ev):
    """R3: nhánh một mắt chỉ tính khi ratio >= tau. tau chọn CHỈ trên TRAIN (tối đa Youden J), đo trên valid+test sạch + độ nhạy theo tau."""
    tr = [u for u in units if u['split'] == 'train' and u.get('kind') == 'one' and u['matched'] and not u.get('small') and (not u['cheat'] or u['cls'] in HEAD)]
    pos, neg = np.array([ratio(u) for u in tr if u['cheat']]), np.array([ratio(u) for u in tr if not u['cheat']])
    grid = np.round(np.arange(0.01, 0.16, 0.005), 3)
    jt = [(float((pos >= t).mean() - (neg >= t).mean()), float(t)) for t in grid]
    j_best, tau = max(jt)
    print(f'\n4) R3: nhánh một mắt chỉ tính khi (dx/box_h >= tau). Chọn tau TRÊN TRAIN (chuẩn n={len(neg)}, đầu n={len(pos)}):')
    print('   J(train) theo tau:', {t: round(j, 2) for j, t in jt if t in (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12)})
    print(f'   -> tau = {tau} (J train = {j_best:+.2f}); đo trên valid+test SẠCH (chưa dùng để chọn):')

    def r3(t):
        return lambda u: flag_current(u) and (u.get('kind') != 'one' or ratio(u) >= t)
    row(f'R3 tau={tau}', ev, r3(tau), lambda u: u['cls'] in HEAD)
    print('   độ nhạy theo tau (ở valid+test sạch):')
    for t in (round(tau * 0.7, 3), round(tau * 0.85, 3), round(tau * 1.15, 3), round(tau * 1.3, 3)):
        row(f'  R3 tau={t}', ev, r3(t), lambda u: u['cls'] in HEAD)
    allv = [u for u in units if u['split'] in ('valid', 'test')]
    print('   tham khảo: toàn bộ valid+test (kể cả ảnh gần trùng train, lạc quan hơn):')
    row('  R0 hiện tại', allv, flag_current, lambda u: u['cls'] in HEAD)
    row(f'  R3 tau={tau}', allv, r3(tau), lambda u: u['cls'] in HEAD)
    print('   khác nhau theo bộ (valid+test sạch): J của R0 / R3')
    for d in DATASETS:
        sub = [u for u in ev if u['group'][0] == d]
        if sub:
            import io
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                a, b = row('', sub, flag_current, lambda u: u['cls'] in HEAD), row('', sub, r3(tau), lambda u: u['cls'] in HEAD)
            print(f"     {d[:28]:<28} R0 J={a['youden_j'] if a['youden_j'] is not None else float('nan'):+.2f} (n chuẩn={a['fp'] + a['tn']}, đầu={a['tp'] + a['fn']})   R3 J={b['youden_j'] if b['youden_j'] is not None else float('nan'):+.2f}")


def nose_off(u, c_min=0.5):
    """Độ lệch NGANG của mũi so với trung điểm hai vai, chia bề rộng vai (None = không đo được). Quay ngang làm mũi lệch khỏi trục thân;
    cúi mặt chỉ làm mũi tụt xuống nên không lệch ngang. Không phụ thuộc số mắt còn thấy."""
    kp = u.get('kp')
    if not kp or kp[0][2] < c_min or kp[5][2] < c_min or kp[6][2] < c_min:
        return None
    sw = abs(kp[5][0] - kp[6][0])
    return None if sw < 8 else abs(kp[0][0] - (kp[5][0] + kp[6][0]) / 2) / sw


def section5(units, ev):
    """R4: nhánh một mắt chỉ tính khi nose_off >= tau (chọn tau CHỈ trên train). Tiêu chí chấp nhận đã chốt trước khi xem số (xem lời nhắn khi chạy)."""
    def sel(split_ok, clean_only=False):
        return [u for u in units if split_ok(u['split']) and (u['clean'] or not clean_only) and u.get('kind') == 'one' and u['matched'] and not u.get('small')]
    tr = [u for u in sel(lambda s: s == 'train') if (not u['cheat'] or u['cls'] in HEAD)]
    meas = [u for u in tr if nose_off(u) is not None]
    neg, pos = np.array([nose_off(u) for u in meas if not u['cheat']]), np.array([nose_off(u) for u in meas if u['cheat']])
    print(f'\n5) R4: nhánh một mắt chỉ tính khi nose_off >= tau (nose_off = |mũi - trung điểm vai| / bề rộng vai).')
    print(f'   Chốt trước: nhận R4 nếu (a) AUC(train) >= 0,70; (b) valid+test sạch: cận dưới J > J(R1)=+0,12 VÀ báo nhầm của nhánh một mắt giảm >= một nửa; (c) cả 2 video giảm >= 70% khung báo sai.')
    print(f'   đo được nose_off: {len(meas)}/{len(tr)} đơn vị một mắt ở train (chuẩn {len(neg)}, đầu {len(pos)})')
    if not len(pos) or not len(neg):
        return None
    print(f'   AUC(train, chuẩn vs đầu) = {auc(pos, neg):.2f} | phân vị 10/50/90 chuẩn: {[round(float(np.percentile(neg, q)), 2) for q in (10, 50, 90)]}, đầu: {[round(float(np.percentile(pos, q)), 2) for q in (10, 50, 90)]}')
    grid = np.round(np.arange(0.05, 1.01, 0.025), 3)
    jt = [(float((pos >= t).mean() - (neg >= t).mean()), float(t)) for t in grid]
    j_best, tau = max(jt)
    print('   J(train) theo tau:', {t: round(j, 2) for j, t in jt if abs(t * 20 - round(t * 20)) < 1e-6 and 0.1 <= t <= 0.8})
    print(f'   -> tau = {tau} (J train = {j_best:+.2f}); đo trên valid+test SẠCH (chưa dùng để chọn):')
    r4 = lambda t: (lambda u: flag_current(u) and (u.get('kind') != 'one' or ((nose_off(u) or 0) >= t)))  # noqa: E731
    row('R0 hiện tại', ev, flag_current, lambda u: u['cls'] in HEAD)
    row('R1 bỏ nhánh một mắt', ev, lambda u: flag_current(u) and u.get('kind') != 'one', lambda u: u['cls'] in HEAD)
    row(f'R4 tau={tau}', ev, r4(tau), lambda u: u['cls'] in HEAD)
    one = [u for u in ev if u.get('kind') == 'one' and u['matched'] and not u.get('small')]
    nn = [u for u in one if not u['cheat']]
    print(f"   riêng nhánh một mắt (valid+test sạch): chuẩn n={len(nn)}: báo nhầm R0 {np.mean([flag_current(u) for u in nn]):.2f} -> R4 {np.mean([r4(tau)(u) for u in nn]):.2f}; "
          f"đo được nose_off {sum(nose_off(u) is not None for u in one)}/{len(one)}")
    print('   độ nhạy theo tau:')
    for t in (round(tau * 0.8, 3), round(tau * 1.25, 3)):
        row(f'  R4 tau={t}', ev, r4(t), lambda u: u['cls'] in HEAD)
    return tau


def main():
    units = []
    for d in DATASETS:
        if (STUDY / f'{slug(d)}.jsonl').exists():
            units += load(d)
    print(f'Đơn vị: {len(units)} | ghép được người pose: {sum(u["matched"] for u in units)}\n')
    ev = [u for u in units if u['split'] in ('valid', 'test') and u['clean']]
    print(f'Đánh giá = valid+test SẠCH (không có bản gần trùng trong train): {len(ev)} đơn vị\n')

    print('1) Theo nhánh keypoint, tín hiệu quay đầu HIỆN TẠI (ev >= 0,55), valid+test sạch, dương = lớp "đầu":')
    for kind in ('both', 'one', 'nose_only', 'none'):
        sub = [u for u in ev if u['matched'] and u.get('kind') == kind]
        row(f'nhánh {kind}', sub, flag_current, lambda u: u['cls'] in HEAD)
    row('nhánh one, mọi lớp gian lận', [u for u in ev if u.get('kind') == 'one'], flag_current, lambda u: True)
    row('TOÀN BỘ (luật hiện tại)', ev, flag_current, lambda u: u['cls'] in HEAD)
    print('\n2) Các luật thay thế cho nhánh một mắt (không tham số nào chỉnh theo dữ liệu):')
    nodrop = lambda u: flag_current(u) and u.get('kind') != 'one'  # noqa: E731
    row('R1: bỏ nhánh một mắt', ev, nodrop, lambda u: u['cls'] in HEAD)

    def profile_sig(u):
        """Nhìn nghiêng thật: mắt và tai còn thấy cùng MỘT bên mặt (mắt trái + tai trái, hoặc mắt phải + tai phải), tai bên kia bị che."""
        return u.get('kind') == 'one' and u['ear'] == ([1, 0] if u['eye_side'] == 'L' else [0, 1])
    sig = lambda u: flag_current(u) and (u.get('kind') != 'one' or profile_sig(u))  # noqa: E731
    row('R2: một mắt chỉ khi mắt+tai cùng bên', ev, sig, lambda u: u['cls'] in HEAD)
    one = [u for u in ev if u.get('kind') == 'one' and u['matched'] and not u.get('small')]
    print(f"   (nhánh một mắt, valid+test sạch: {len(one)} đơn vị; khớp mẫu nhìn nghiêng: chuẩn {sum(profile_sig(u) for u in one if not u['cheat'])}/{sum(not u['cheat'] for u in one)},"
          f" đầu {sum(profile_sig(u) for u in one if u['cheat'] and u['cls'] in HEAD)}/{sum(u['cheat'] and u['cls'] in HEAD for u in one)})")

    section4(units, ev)
    if any('kp' in u for u in units):
        section5(units, ev)
    print('\n3) Nhánh một mắt: đặc trưng nào tách "chuẩn" khỏi "đầu"? (AUC trên TRAIN; 0,5 = không tách)')
    tr = [u for u in units if u['split'] == 'train' and u.get('kind') == 'one' and u['matched'] and not u.get('small')]
    neg = [u for u in tr if not u['cheat']]
    pos = [u for u in tr if u['cheat'] and u['cls'] in HEAD]
    print(f'   train: chuẩn n={len(neg)}, đầu n={len(pos)}')
    feats = {'dx/box_w': lambda u: u['dx'] / u['box_w'], 'dx/box_h': lambda u: u['dx'] / u['box_h'], 'dy/box_h': lambda u: u['dy'] / u['box_h'],
             'dx/|dy|': lambda u: u['dx'] / max(abs(u['dy']), 1.0), 'dx (px)': lambda u: u['dx'],
             'conf mắt thấy': lambda u: u['conf'][1] if u['eye_side'] == 'L' else u['conf'][2], 'conf mũi': lambda u: u['conf'][0],
             'thấy tai': lambda u: float(sum(u['ear']))}
    for name, fn in feats.items():
        a = auc(np.array([fn(u) for u in pos]), np.array([fn(u) for u in neg]))
        print(f"   {name:<14} AUC={'-' if a is None else f'{a:.2f}'}")
    print('\n   Phân vị (train) của dx/box_w: chuẩn vs đầu:',
          [round(float(np.percentile([u['dx'] / u['box_w'] for u in s], q)), 3) for s in (neg, pos) for q in (25, 50, 75)] if neg and pos else '-')


if __name__ == '__main__':
    main()
