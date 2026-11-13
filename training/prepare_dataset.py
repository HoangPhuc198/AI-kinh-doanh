"""Chuẩn bị dataset YOLO: kiểm tra nhãn, loại ảnh trùng, chia train/val/test KHÔNG rò rỉ.

Đầu vào: thư mục chứa ảnh + nhãn YOLO (`class cx cy w h` chuẩn hóa). Nhãn được tìm tại
`.../labels/<tên>.txt` (thay `images` bằng `labels` trong đường dẫn) hoặc cạnh ảnh.

Chống rò rỉ: ảnh được gom thành "nhóm" (cùng video/chuỗi khung) rồi chia theo NHÓM, không theo ảnh.
Nhóm = tên tệp bỏ hậu tố số khung (video01_000123.jpg -> video01), hoặc tên thư mục cha (`--group-by dir`).
Ảnh trùng byte-hệt-nhau (md5) bị loại; ảnh gần giống (khoảng cách dHash <= --near) bị gộp vào cùng nhóm
để không tách sang hai tập. Dòng nhãn sai (lớp ngoài phạm vi, tọa độ ngoài [0,1]) bị loại và được ghi báo cáo.

Ví dụ:
  python training/prepare_dataset.py --src datasets/raw/mydata --dst datasets/mydata --names person "cell phone"
Kết quả: <dst>/{images,labels}/{train,val,test}, <dst>/data.yaml, <dst>/prepare_report.json
"""
import argparse
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def dhash(img, size=8):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size))
    return int(''.join('1' if b else '0' for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def group_of(path, mode):
    if mode == 'dir':
        return path.parent.name
    return re.sub(r'[_\-]?\d+$', '', path.stem) or path.stem


def check_label(lab, nc):
    """Trả về (các dòng hợp lệ, các dòng lỗi)."""
    ok, bad = [], []
    for i, line in enumerate(lab.read_text().splitlines(), 1):
        p = line.split()
        try:
            assert len(p) == 5
            c, x, y, w, h = int(p[0]), *map(float, p[1:])
            assert 0 <= c < nc and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
            ok.append(line)
        except (AssertionError, ValueError):
            bad.append(f'{lab.name}:{i}: {line!r}')
    return ok, bad


def find_label(img):
    for cand in (Path(str(img).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\')).with_suffix('.txt'), img.with_suffix('.txt')):
        if cand.is_file():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--names', nargs='+', required=True, help='tên lớp theo thứ tự id')
    ap.add_argument('--val', type=float, default=0.15)
    ap.add_argument('--test', type=float, default=0.15)
    ap.add_argument('--group-by', choices=['name', 'dir'], default='name')
    ap.add_argument('--near', type=int, default=4, help='ngưỡng Hamming dHash coi là gần giống')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    images = sorted(p for p in src.rglob('*') if p.suffix.lower() in IMG_EXT)
    report = {'images_found': len(images), 'no_label': 0, 'unreadable': 0, 'bad_label_lines': [], 'exact_duplicates': 0, 'near_duplicate_merges': 0}
    seen, items = set(), []
    for img in images:
        lab = find_label(img)
        if lab is None:
            report['no_label'] += 1
            continue
        md5 = hashlib.md5(img.read_bytes()).hexdigest()
        if md5 in seen:
            report['exact_duplicates'] += 1
            continue
        seen.add(md5)
        frame = cv2.imread(str(img))
        if frame is None:
            report['unreadable'] += 1
            continue
        lines, bad = check_label(lab, len(a.names))
        report['bad_label_lines'] += bad
        items.append({'img': img, 'lines': lines, 'hash': dhash(frame), 'group': group_of(img, a.group_by)})

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, p in enumerate(items):
        for q in items[i + 1:]:
            if bin(p['hash'] ^ q['hash']).count('1') <= a.near and find(p['group']) != find(q['group']):
                parent[find(q['group'])] = find(p['group'])
                report['near_duplicate_merges'] += 1
    groups = defaultdict(list)
    for it in items:
        groups[find(it['group'])].append(it)
    keys = sorted(groups)
    random.Random(a.seed).shuffle(keys)
    n, acc, split = len(items), defaultdict(int), {}
    for k in keys:  # điền test, rồi val, phần còn lại là train; một nhóm không bao giờ bị chia
        name = 'test' if acc['test'] < a.test * n else 'val' if acc['val'] < a.val * n else 'train'
        acc[name] += len(groups[k])
        for it in groups[k]:
            split[id(it)] = name
    for s in ('train', 'val', 'test'):
        (dst/'images'/s).mkdir(parents=True, exist_ok=True)
        (dst/'labels'/s).mkdir(parents=True, exist_ok=True)
    counts, inst = defaultdict(int), defaultdict(lambda: defaultdict(int))
    for it in items:
        s = split[id(it)]
        counts[s] += 1
        shutil.copy2(it['img'], dst/'images'/s/it['img'].name)
        (dst/'labels'/s/(it['img'].stem + '.txt')).write_text('\n'.join(it['lines']) + ('\n' if it['lines'] else ''))
        for ln in it['lines']:
            inst[s][a.names[int(ln.split()[0])]] += 1
    (dst/'data.yaml').write_text(f"path: {dst.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                                 + ''.join(f'  {i}: {n}\n' for i, n in enumerate(a.names)), encoding='utf-8')
    report.update({'seed': a.seed, 'images_kept': len(items), 'groups': len(groups), 'split_images': dict(counts),
                   'split_instances': {k: dict(v) for k, v in inst.items()}, 'bad_label_lines_count': len(report['bad_label_lines'])})
    (dst/'prepare_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'bad_label_lines'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
