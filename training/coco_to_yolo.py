"""Chuyển COCO val2017 -> YOLO với 2 lớp: 0=person, 1=cell phone.

Chỉ giữ ảnh có person hoặc cell phone; bỏ hộp `iscrowd`. Chọn mẫu có kiểm soát để cân bằng:
toàn bộ ảnh có điện thoại + tối đa --max-person-images ảnh chỉ có người (ưu tiên ảnh có người NHỎ để học người ở xa).

  python training/coco_to_yolo.py --images datasets/raw/coco_val/val2017 --ann datasets/raw/coco_val/instances_val2017.json --out datasets/raw/coco_yolo
Sau đó chạy prepare_dataset.py để chia train/val/test.
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

CLASSES = {1: 0, 77: 1}  # id COCO person=1, cell phone=77 -> id mới


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True); ap.add_argument('--ann', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--max-person-images', type=int, default=800); ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    coco = json.loads(Path(a.ann).read_text(encoding='utf-8'))
    meta = {i['id']: i for i in coco['images']}
    per = defaultdict(list)
    for an in coco['annotations']:
        if an['category_id'] in CLASSES and not an.get('iscrowd'):
            per[an['image_id']].append(an)
    phone = [i for i, v in per.items() if any(x['category_id'] == 77 for x in v)]
    only_person = [i for i in per if i not in set(phone)]
    small = lambda i: sum(1 for x in per[i] if x['category_id'] == 1 and x['area'] < 32 * 32)  # noqa: E731
    only_person.sort(key=lambda i: (-small(i), i))
    chosen_p = only_person[:a.max_person_images // 2]  # nửa đầu: nhiều người nhỏ nhất
    rest = only_person[a.max_person_images // 2:]
    random.Random(a.seed).shuffle(rest)
    chosen = sorted(set(phone + chosen_p + rest[:a.max_person_images - len(chosen_p)]))
    out = Path(a.out); (out/'images').mkdir(parents=True, exist_ok=True); (out/'labels').mkdir(exist_ok=True)
    n_inst = defaultdict(int)
    for iid in chosen:
        m = meta[iid]; w, h = m['width'], m['height']
        lines = []
        for an in per[iid]:
            x, y, bw, bh = an['bbox']
            if bw < 2 or bh < 2:
                continue
            lines.append(f"{CLASSES[an['category_id']]} {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}")
            n_inst[CLASSES[an['category_id']]] += 1
        shutil.copy2(Path(a.images)/m['file_name'], out/'images'/m['file_name'])
        (out/'labels'/(Path(m['file_name']).stem + '.txt')).write_text('\n'.join(lines) + '\n')
    print(json.dumps({'images': len(chosen), 'with_phone': len(phone), 'instances': {'person': n_inst[0], 'cell phone': n_inst[1]}}))


if __name__ == '__main__':
    main()
