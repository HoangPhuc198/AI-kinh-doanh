"""Kiểm tra ensemble thật trên ảnh: hộp hợp lệ, lỗi nguồn, tốc độ, tỷ lệ cờ nghi vấn.
Ảnh COCO KHÔNG chứa gian lận => mọi cờ 'Nghi vấn gian lận' là báo nhầm tiềm năng (nhưng khác miền với phòng thi nên chỉ mang tính tham khảo)."""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
import runtime, inference  # noqa: E402
runtime.load_torch()
split = sys.argv[1] if len(sys.argv) > 1 else 'test'
imgs = sorted((ROOT / 'datasets/coco_person_phone/images' / split).glob('*.jpg'))
inference.warm_model('ensemble')
allvotes = []; bad_boxes = bad_scores = 0; ms = []; labels = {}; errs = {}; per_img_flag = {'suspect': 0, 'watch': 0}; npers = 0; srcms = {}
for p in imgs:
    frame = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    r = inference.run_model('ensemble', frame, 0.5, include_image=False)
    ms.append(r['ms']); seen = set()
    for k, v in r['timings'].items(): srcms.setdefault(k, []).append(v)
    for k in r['errors']: errs[k] = errs.get(k, 0) + 1
    for d in r['details']:
        x0, y0, x1, y1 = d['box']
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1): bad_boxes += 1
        if d['confidence'] is None or not 0 <= d['confidence'] <= 1: bad_scores += 1
        labels[d['label']] = labels.get(d['label'], 0) + 1
        if 'votes' in d: allvotes.append(d['votes'])
        if d['label'] != 'cell phone': npers += 1
        seen.add(d['label'])
    per_img_flag['suspect'] += 'Gian lận' in seen; per_img_flag['watch'] += 'Nghi ngờ' in seen  # ảnh đơn: kết luận 1 khung (tối đa Nghi ngờ)
out = {'images': len(imgs), 'persons_total': npers, 'labels': labels, 'invalid_boxes': bad_boxes, 'invalid_scores': bad_scores, 'source_errors': errs,
       'images_with_suspect_flag': per_img_flag['suspect'], 'images_with_watch_flag': per_img_flag['watch'],
       'latency_ms_mean': round(float(np.mean(ms))), 'latency_ms_p95': round(float(np.percentile(ms, 95))), 'per_source_ms_mean': {k: round(float(np.mean(v))) for k, v in srcms.items()}}
(ROOT / f'reports/ensemble_check_{split}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
(ROOT / f'reports/ensemble_votes_{split}.json').write_text(json.dumps(allvotes), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))
