"""Open Images V7 (tập validation) -> YOLO, 2 lớp: 0=person, 1=cell phone ("Mobile phone").

Giấy phép: nhãn CC BY 4.0, ảnh CC BY 2.0 (theo Open Images). Chỉ tải ẢNH ĐƯỢC CHỌN vào --out (trong dự án).
Loại ảnh có hộp person/phone kiểu IsGroupOf hoặc IsDepiction (tranh/vẽ) vì sẽ làm nhãn thiếu/sai.
Chọn: tối đa --phone ảnh có điện thoại + --person ảnh chỉ có người (ưu tiên nhiều người nhỏ).

  python training/openimages_to_yolo.py --csv datasets/raw/openimages/validation-annotations-bbox.csv --out datasets/raw/oi_yolo
"""
import argparse
import csv
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IDS = {'/m/01g317': 0, '/m/050k8': 1}
URL = 'https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg'


def fetch(args):
    iid, dest = args
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    try:
        urllib.request.urlretrieve(URL.format(iid), dest)
        return True
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--phone', type=int, default=500); ap.add_argument('--person', type=int, default=700)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    boxes, bad = defaultdict(list), set()
    for r in csv.DictReader(open(a.csv, encoding='utf-8')):
        if r['LabelName'] not in IDS:
            continue
        if r['IsGroupOf'] == '1' or r['IsDepiction'] == '1':
            bad.add(r['ImageID']); continue
        x0, x1, y0, y1 = (float(r[k]) for k in ('XMin', 'XMax', 'YMin', 'YMax'))
        boxes[r['ImageID']].append((IDS[r['LabelName']], x0, x1, y0, y1))
    ids = [i for i in boxes if i not in bad]
    rng = random.Random(a.seed)
    phone = [i for i in ids if any(b[0] == 1 for b in boxes[i])]
    only = [i for i in ids if i not in set(phone)]
    rng.shuffle(phone)
    small = lambda i: sum(1 for b in boxes[i] if b[0] == 0 and (b[2]-b[1])*(b[4]-b[3]) < 0.01)  # noqa: E731
    only.sort(key=lambda i: (-small(i), i))
    half = a.person // 2
    rest = only[half:]; rng.shuffle(rest)
    chosen = sorted(set(phone[:a.phone] + only[:half] + rest[:a.person - half]))
    out = Path(a.out); (out/'images').mkdir(parents=True, exist_ok=True); (out/'labels').mkdir(exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(fetch, [(i, out/'images'/f'{i}.jpg') for i in chosen]))
    n, inst = 0, [0, 0]
    for iid, good in zip(chosen, ok):
        if not good:
            continue
        lines = []
        for c, x0, x1, y0, y1 in boxes[iid]:
            if (x1 - x0) * (y1 - y0) < 1e-5:
                continue
            lines.append(f'{c} {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}'); inst[c] += 1
        (out/'labels'/f'{iid}.txt').write_text('\n'.join(lines) + '\n'); n += 1
    print({'images': n, 'failed': len(chosen) - n, 'with_phone': min(len(phone), a.phone), 'person_boxes': inst[0], 'phone_boxes': inst[1]})


if __name__ == '__main__':
    main()
