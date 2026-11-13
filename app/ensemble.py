"""ExamVision Ensemble: gộp nhiều mô hình thành kết luận THEO TỪNG NGƯỜI.

Cấu trúc
--------
* Neo người: `person_coco` (YOLO11n COCO, đã hiệu chỉnh ngưỡng, 960 px) cho hộp người ổn định, bổ sung thêm người
  mà các mô hình miền thi cử (yolo_duyen/gianlan/training/KNN) thấy nhưng neo bỏ sót.
* Bằng chứng hành vi cho từng người (mỗi nguồn là một "phiếu"): yolo_duyen, gianlan (nhãn Cheating),
  KNN-pose Duyên (Suspicious, nguồn yếu), và điện thoại nằm gần người (person_coco/gianlan).
* Kết luận theo luật nhiều nguồn đồng thuận (xem `decide`): "Nghi vấn gian lận" chỉ khi >=2 nguồn độc lập cùng nghi
  (có ít nhất 1 nguồn mạnh) VÀ điểm tổng hợp >= SUSPECT_SCORE; 1 nguồn mạnh -> "Cần chú ý"; còn lại "Bình thường".

Giới hạn trung thực
-------------------
Trọng số nguồn và ngưỡng điểm là HEURISTIC đặt theo độ tin cậy ước lượng, KHÔNG được hiệu chỉnh trên video thi cử có nhãn
(không có dữ liệu như vậy trong dự án). Bằng chứng sẵn có: KNN Duyên trên video lạ chỉ đạt precision 0.32/recall 0.24
(reports/knn_duyen/video_evaluation.json), nên đầu ra là "nghi vấn cần giám thị xem xét", KHÔNG phải kết luận gian lận.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

# Các nguồn bỏ phiếu hành vi: khóa mô hình -> (trọng số, ngưỡng conf khi chạy, nhãn coi là "nghi vấn", nhãn coi là "người")
BEHAVIOR_SOURCES = {
    'yolo_duyen': (0.30, 0.30, {'cheating'}, {'normal', 'cheating'}),
    'duyen': (0.20, 0.40, {'suspicious'}, {'normal', 'suspicious'}),
    'gianlan': (0.20, 0.30, {'cheating', 'cheating-behaviour'}, {'normal', 'cheating', 'cheating-behaviour', 'person', 'submitting'}),
}
PHONE_WEIGHT = 0.20
# Nguồn YẾU: chỉ củng cố nghi vấn đã có nguồn mạnh, không tự tạo 'Cần chú ý'. Lý do (đo trên 741 người bình thường, COCO test,
# reports/ensemble_votes_test.json): KNN Duyên bắn cờ 10,4% (precision 0,32 trên video lạ). 'training' (14% ở conf 0,3, chậm nhất) đã bị loại.
WEAK_SOURCES = {'duyen'}
# 'Nghi vấn gian lận' BẮT BUỘC có ít nhất một mô hình hành vi mạnh; điện thoại và KNN chỉ củng cố (điện thoại một mình/cùng KNN chỉ là 'Cần chú ý').
BEHAVIOR_STRONG = {'yolo_duyen', 'gianlan'}
KNN_NOMINAL_CONF = 0.70       # KNN không xuất confidence: dùng mức danh nghĩa cho phiếu "Suspicious"
PERSON_MIN_CONF = 0.35        # xem CALIBRATION trong live/alerts.py
PHONE_MIN_CONF = 0.35
AGREE_MIN_EVIDENCE = 0.50     # một nguồn được tính là "đồng ý nghi vấn" khi bằng chứng >= mức này
SUSPECT_SCORE = 0.30          # điểm tổng hợp tối thiểu cho "Nghi vấn gian lận"
SUSPECT_MIN_SOURCES = 2

LABEL_SUSPECT, LABEL_WATCH, LABEL_NORMAL = 'Nghi vấn gian lận', 'Cần chú ý', 'Bình thường'
SOURCE_NAMES = {'yolo_duyen': 'YOLO11 Duyên', 'duyen': 'KNN Duyên', 'gianlan': 'Gian lận', 'training': 'Training YOLO', 'phone': 'Điện thoại gần người'}


# ---------- hình học (hộp chuẩn hóa [x0,y0,x1,y1] trong [0,1]) ----------
def clamp_box(b):
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in b)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def associated(det_box, person_box):
    """Hộp hành vi thuộc về người nếu chồng lấp đủ (IoU, hoặc phần lớn hộp nằm trong người/ngược lại)."""
    i = inter(det_box, person_box)
    ad, ap = area(det_box), area(person_box)
    if ad <= 0 or ap <= 0:
        return False
    return iou(det_box, person_box) >= 0.30 or i / ad >= 0.60 or i / ap >= 0.60


def center_in(box, container, margin=0.15):
    w, h = container[2] - container[0], container[3] - container[1]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return container[0] - margin * w <= cx <= container[2] + margin * w and container[1] - margin * h <= cy <= container[3] + margin * h


# ---------- hợp nhất (hàm thuần, có unit test) ----------
def _norm(d):
    return {'label': str(d.get('label', '')).lower(), 'confidence': d.get('confidence'), 'box': clamp_box(d['box'])}


def collect_persons(anchor: list[dict], sources: dict[str, list[dict]]):
    """Danh sách người: neo person_coco trước, rồi thêm người từ các nguồn miền thi cử mà không trùng neo."""
    persons = []
    for d in anchor:
        d = _norm(d)
        if d['label'] == 'person' and (d['confidence'] or 0) >= PERSON_MIN_CONF and area(d['box']) > 0:
            persons.append({'box': d['box'], 'person_conf': d['confidence'], 'origin': 'person_coco'})
    for key, (_, _, _, person_labels) in BEHAVIOR_SOURCES.items():
        for d in map(_norm, sources.get(key, [])):
            if d['label'] in person_labels and area(d['box']) > 0 and not any(associated(d['box'], p['box']) for p in persons):
                persons.append({'box': d['box'], 'person_conf': d['confidence'], 'origin': key})
    return persons


def assign_evidence(persons, source_key, dets, cheat_labels):
    """Gán mỗi hộp của một nguồn cho ĐÚNG MỘT người (chồng lấp tốt nhất). Trả về danh sách bằng chứng 0..1 (None = nguồn không thấy)."""
    ev = [None] * len(persons)
    for d in map(_norm, dets):
        best, best_i = 0.0, None
        for i, p in enumerate(persons):
            if associated(d['box'], p['box']):
                ad = area(d['box'])
                fit = max(iou(d['box'], p['box']), inter(d['box'], p['box']) / ad if ad else 0.0)
                if fit > best:
                    best, best_i = fit, i
        if best_i is None:
            continue
        conf = 0.0
        if d['label'] in cheat_labels:
            conf = KNN_NOMINAL_CONF if d['confidence'] is None else float(d['confidence'])
        ev[best_i] = max(ev[best_i] or 0.0, conf)
    return ev


def decide(votes: dict[str, float]):
    """votes: nguồn -> bằng chứng 0..1 (nguồn không thấy/không nghi = 0). Trả về (nhãn, điểm)."""
    weights = {k: v[0] for k, v in BEHAVIOR_SOURCES.items()} | {'phone': PHONE_WEIGHT}
    score = sum(weights[k] * min(1.0, max(0.0, e)) for k, e in votes.items() if k in weights) / sum(weights.values())
    agree = sum(1 for e in votes.values() if e >= AGREE_MIN_EVIDENCE)
    strong = sum(1 for k, e in votes.items() if k not in WEAK_SOURCES and e >= AGREE_MIN_EVIDENCE)
    behavior = any(votes.get(k, 0.0) >= AGREE_MIN_EVIDENCE for k in BEHAVIOR_STRONG)
    if agree >= SUSPECT_MIN_SOURCES and behavior and score >= SUSPECT_SCORE:
        return LABEL_SUSPECT, score
    if strong >= 1:
        return LABEL_WATCH, score
    return LABEL_NORMAL, score


def fuse(anchor: list[dict], sources: dict[str, list[dict]]) -> dict:
    """anchor: đầu ra person_coco (person + cell phone). sources: khóa -> details của từng nguồn hành vi."""
    persons = collect_persons(anchor, sources)
    phones = [_norm(d) for d in anchor if str(d.get('label', '')).lower() == 'cell phone' and (d.get('confidence') or 0) >= PHONE_MIN_CONF]
    phones += [_norm(d) for d in sources.get('gianlan', []) if str(d.get('label', '')).lower() == 'cell-phones' and (d.get('confidence') or 0) >= PHONE_MIN_CONF]
    phones = _dedupe(phones)
    out_people = []
    per_source = {key: assign_evidence(persons, key, sources.get(key, []), cheat) for key, (_, _, cheat, _) in BEHAVIOR_SOURCES.items()}
    for idx, p in enumerate(persons):
        votes, why = {}, []
        for key in BEHAVIOR_SOURCES:
            e = per_source[key][idx]
            votes[key] = e or 0.0
            if e:
                why.append(f'{SOURCE_NAMES[key]} nghi vấn ({e:.0%})' if key != 'duyen' else f'{SOURCE_NAMES[key]}: tư thế nghi vấn')
        near = [ph for ph in phones if center_in(ph['box'], p['box'])]
        votes['phone'] = max((ph['confidence'] for ph in near), default=0.0)
        if near:
            why.append(f"Điện thoại gần người ({votes['phone']:.0%})")
        label, score = decide(votes)
        shown = round(1 - score if label == LABEL_NORMAL else score, 3)  # độ chắc của chính nhãn đang hiển thị
        out_people.append({'label': label, 'confidence': shown, 'score': round(score, 3), 'box': p['box'], 'person_conf': p['person_conf'],
                           'origin': p['origin'], 'votes': {k: round(v, 2) for k, v in votes.items()}, 'evidence': why})
    return {'people': out_people, 'phones': phones}


def _dedupe(boxes):
    kept = []
    for b in sorted(boxes, key=lambda d: -(d['confidence'] or 0)):
        if all(iou(b['box'], k['box']) < 0.5 for k in kept):
            kept.append(b)
    return kept


# ---------- chạy mô hình + vẽ ----------
COLORS = {LABEL_SUSPECT: (60, 60, 235), LABEL_WATCH: (30, 170, 240), LABEL_NORMAL: (90, 200, 90), 'cell phone': (200, 120, 40)}


_FONT_CANDIDATES = ('C:/Windows/Fonts/segoeui.ttf', 'C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/Library/Fonts/Arial.ttf')


def _font(size):
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size), True
        except OSError:
            continue
    return ImageFont.load_default(), False  # không có font Unicode: bỏ dấu để chữ không vỡ


def _ascii(text):
    import unicodedata
    return unicodedata.normalize('NFD', text.replace('đ', 'd').replace('Đ', 'D')).encode('ascii', 'ignore').decode()


def _draw(frame, people, phones):
    """Vẽ hộp + nhãn bằng PIL (cv2.putText không vẽ được chữ có dấu tiếng Việt)."""
    from PIL import Image, ImageDraw
    h, w = frame.shape[:2]
    scale = max(0.6, min(w, h) / 800)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    g = ImageDraw.Draw(img)
    font, unicode_ok = _font(max(12, int(15 * scale)))
    items = [(p['label'], p['confidence'], p['box']) for p in people] + [('cell phone', ph['confidence'], ph['box']) for ph in phones]
    for label, conf, b in items:
        x0, y0, x1, y1 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
        r, gc, bl = COLORS[label][::-1]
        g.rectangle([x0, y0, x1, y1], outline=(r, gc, bl), width=max(2, int(3 * scale)))
        text = f'{label} {conf:.0%}' if conf is not None else label
        text = text if unicode_ok else _ascii(text)
        l, t, rr, bb = g.textbbox((0, 0), text, font=font)
        th = bb - t
        ty = max(0, y0 - th - 8)
        g.rectangle([x0, ty, x0 + (rr - l) + 8, ty + th + 8], fill=(r, gc, bl))
        g.text((x0 + 4, ty + 3 - t), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def components():
    return ['person_coco', *BEHAVIOR_SOURCES]


def run(frame, include_image=True, imgsz=None):
    """Chạy tất cả mô hình thành phần trên một khung rồi hợp nhất. Trả cùng định dạng với `inference.run_model`."""
    import inference  # nhập muộn để tránh vòng import
    started = time.perf_counter()
    timings, errors = {}, {}
    anchor = inference.run_model('person_coco', frame, 0.25, include_image=False, imgsz=imgsz)
    timings['person_coco'] = anchor['ms']
    sources = {}
    for key, (_, conf, _, _) in BEHAVIOR_SOURCES.items():
        try:
            r = inference.run_model(key, frame, conf, include_image=False)
            sources[key], timings[key] = r['details'], r['ms']
        except Exception as exc:  # một nguồn lỗi không làm hỏng cả khung; ghi nhận để hiển thị trung thực
            errors[key] = str(exc)[:200]
    fused = fuse(anchor['details'], sources)
    details = [{'label': p['label'], 'confidence': p['confidence'], 'box': p['box'], 'score': p['score'], 'person_conf': p['person_conf'],
                'votes': p['votes'], 'evidence': p['evidence']} for p in fused['people']]
    details += [{'label': 'cell phone', 'confidence': ph['confidence'], 'box': ph['box']} for ph in fused['phones']]
    counts = {}
    for d in details:
        counts[d['label']] = counts.get(d['label'], 0) + 1
    out = {'details': details, 'counts': counts, 'ms': round((time.perf_counter() - started) * 1000), 'timings': timings, 'errors': errors}
    if include_image:
        ok, enc = cv2.imencode('.jpg', _draw(frame, fused['people'], fused['phones']), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError('Không thể tạo ảnh kết quả')
        import base64
        out['image'] = base64.b64encode(enc).decode()
    return out
