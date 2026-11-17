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

import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from judge import (LABEL_CHEAT, LABEL_NORMAL as V_NORMAL, LABEL_SUSPECT as V_SUSPECT, TIER_NORMAL, TIER_SUSPECT, TIER_WATCH, Judge,
                   Settings)

# Các nguồn bỏ phiếu hành vi: khóa mô hình -> (trọng số, ngưỡng conf khi chạy, nhãn coi là "nghi vấn", nhãn coi là "người")
BEHAVIOR_SOURCES = {
    'yolo_duyen': (0.30, 0.30, {'cheating'}, {'normal', 'cheating'}),
    'duyen': (0.20, 0.40, {'suspicious'}, {'normal', 'suspicious'}),
    'gianlan': (0.20, 0.30, {'cheating', 'cheating-behaviour'}, {'normal', 'cheating', 'cheating-behaviour', 'person', 'submitting'}),
    # Quay/nghiêng đầu từ keypoint YOLO11-Pose (xem head_turn_evidence). Không tự tạo người mới (tập nhãn người rỗng).
    'pose': (0.20, 0.30, {'head_turn'}, set()),
    # Training YOLO (Cheating/Normal, huấn luyện trên dữ liệu Roboflow): NGUỒN PHỤ thử nghiệm. Chỉ chạy khi đã có nguồn mạnh nghi vấn (LATE_SOURCES),
    # không tạo người mới, không tự gây 'Cần chú ý' (WEAK_SOURCES). Ngưỡng 0,5: đo trên ảnh ngoài ở 0,25 báo nhầm ~26%, ở 0,5 ~8%.
    'training': (0.15, 0.50, {'cheating'}, set()),
}
if os.environ.get('EXAMVISION_TRAINING') == '0':  # công tắc thử nghiệm: tắt nguồn phụ Training YOLO
    BEHAVIOR_SOURCES.pop('training', None)
PHONE_WEIGHT = 0.20
PAIR_WEIGHT = 0.20            # 'hai người quay về phía nhau' (xem pair_evidence)
PAIR_TURN_MIN = 0.30          # mỗi người phải quay đầu rõ hơn mức nhiễu nhìn thẳng (~0-0.15); cần CẢ HAI nên đủ chặt
PAIR_MAX_GAP = 2.5            # hai người coi là ngồi cạnh nhau nếu cách nhau <= 2.5 lần bề rộng người
# Nguồn YẾU: chỉ củng cố nghi vấn đã có nguồn mạnh, không tự tạo 'Cần chú ý'. Lý do (đo trên 741 người bình thường, COCO test,
# reports/ensemble_votes_test.json): KNN Duyên bắn cờ 10,4% (precision 0,32 trên video lạ). 'training' từng bị loại (14% ở conf 0,3, chậm nhất);
# nay thử lại ở ngưỡng 0,5 và chỉ làm nguồn phụ (xem BEHAVIOR_SOURCES, HANDOVER.md).
WEAK_SOURCES = {'duyen', 'training'}
UNNORMALIZED = {'training'}  # nguồn phụ không nằm trong mẫu số của điểm: khi nó im lặng, điểm các tín hiệu cũ giữ nguyên
# 'Nghi vấn gian lận' BẮT BUỘC có ít nhất một mô hình hành vi mạnh; điện thoại và KNN chỉ củng cố (điện thoại một mình/cùng KNN chỉ là 'Cần chú ý').
BEHAVIOR_STRONG = {'yolo_duyen', 'gianlan', 'pose', 'pair'}
# Nhóm bằng chứng ĐỘC LẬP: 'đồng thuận' phải đến từ >=2 nhóm khác nhau. KNN Duyên phân loại từ CHÍNH keypoint mà tín hiệu quay đầu dùng
# nên cùng một nhóm (hình học tư thế); đếm cả hai là đếm trùng một bằng chứng.
EVIDENCE_FAMILY = {'pose': 'tu_the', 'duyen': 'tu_the', 'pair': 'tuong_tac', 'yolo_duyen': 'yolo_duyen', 'gianlan': 'gianlan', 'training': 'training', 'phone': 'dien_thoai'}
KNN_NOMINAL_CONF = 0.70       # KNN không xuất confidence: dùng mức danh nghĩa cho phiếu "Suspicious"
PERSON_MIN_CONF = 0.35        # xem CALIBRATION trong live/alerts.py
PHONE_MIN_CONF = 0.35
AGREE_MIN_EVIDENCE = 0.50     # một nguồn được tính là "đồng ý nghi vấn" khi bằng chứng >= mức này
SUSPECT_SCORE = 0.15          # điểm tổng hợp tối thiểu (phụ): điều kiện chính là >=2 nguồn đồng thuận có bằng chứng mạnh
SUSPECT_MIN_SOURCES = 2

LABEL_SUSPECT, LABEL_WATCH, LABEL_NORMAL = 'Nghi vấn gian lận', 'Cần chú ý', 'Bình thường'
SOURCE_NAMES = {'yolo_duyen': 'YOLO11 Duyên', 'duyen': 'KNN Duyên', 'gianlan': 'Gian lận', 'training': 'Training YOLO', 'phone': 'Điện thoại gần người', 'pose': 'Quay đầu', 'pair': 'Hai người quay về phía nhau'}


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


def collect_persons(anchor: list[dict], sources: dict[str, list[dict]], person_min=PERSON_MIN_CONF):
    """Danh sách người: neo person_coco trước, rồi thêm người từ các nguồn miền thi cử mà không trùng neo."""
    persons = []
    for d in anchor:
        d = _norm(d)
        if d['label'] == 'person' and (d['confidence'] or 0) >= person_min and area(d['box']) > 0:
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


def decide(votes: dict[str, float], cfg: Settings | None = None):
    """votes: nguồn -> bằng chứng 0..1 (nguồn không thấy/không nghi = 0). Trả về (nhãn, điểm)."""
    agree_min = cfg.evidence_conf if cfg else AGREE_MIN_EVIDENCE
    weights = {k: v[0] for k, v in BEHAVIOR_SOURCES.items()} | {'phone': PHONE_WEIGHT, 'pair': PAIR_WEIGHT}
    score = min(1.0, sum(weights[k] * min(1.0, max(0.0, e)) for k, e in votes.items() if k in weights) / sum(w for k, w in weights.items() if k not in UNNORMALIZED))
    agree = len({EVIDENCE_FAMILY.get(k, k) for k, e in votes.items() if e >= agree_min})
    strong = sum(1 for k, e in votes.items() if k not in WEAK_SOURCES and e >= agree_min)
    behavior = any(votes.get(k, 0.0) >= agree_min for k in BEHAVIOR_STRONG)
    if agree >= SUSPECT_MIN_SOURCES and behavior and score >= SUSPECT_SCORE:
        return LABEL_SUSPECT, score
    if strong >= 1:
        return LABEL_WATCH, score
    return LABEL_NORMAL, score


def fuse(anchor: list[dict], sources: dict[str, list[dict]], cfg: Settings | None = None) -> dict:
    """anchor: đầu ra person_coco (person + cell phone). sources: khóa -> details của từng nguồn hành vi."""
    phone_min = cfg.phone_conf if cfg else PHONE_MIN_CONF
    persons = collect_persons(anchor, sources, cfg.person_conf if cfg else PERSON_MIN_CONF)
    phones = [_norm(d) for d in anchor if str(d.get('label', '')).lower() == 'cell phone' and (d.get('confidence') or 0) >= phone_min]
    phones += [_norm(d) for d in sources.get('gianlan', []) if str(d.get('label', '')).lower() == 'cell-phones' and (d.get('confidence') or 0) >= phone_min]
    phones = _dedupe(phones)
    out_people = []
    pair = pair_evidence(persons, assign_turns(persons, sources.get('pose', [])))
    per_source = {key: assign_evidence(persons, key, sources.get(key, []), cheat) for key, (_, _, cheat, _) in BEHAVIOR_SOURCES.items()}
    for idx, p in enumerate(persons):
        votes, why = {}, []
        for key in BEHAVIOR_SOURCES:
            e = per_source[key][idx]
            votes[key] = e or 0.0
            if e:
                why.append(f'{SOURCE_NAMES[key]} nghi vấn ({e:.2f})' if key not in ('duyen', 'pose') else f'{SOURCE_NAMES[key]}: tư thế nghi vấn' if key == 'duyen' else f'Quay/nghiêng đầu ({e:.2f})')
        votes['pair'] = pair[idx] or 0.0
        if pair[idx]:
            votes['pose'] = max(votes['pose'], pair[idx])  # quay về phía nhau cũng là quay đầu (đủ để hai nguồn đồng thuận)
            why.append(f"{SOURCE_NAMES['pair']} ({pair[idx]:.2f})")
        near = [ph for ph in phones if center_in(ph['box'], p['box'])]
        votes['phone'] = max((ph['confidence'] for ph in near), default=0.0)
        if near:
            why.append(f"Điện thoại gần người ({votes['phone']:.2f})")
        label, score = decide(votes, cfg)
        shown = round(1 - score if label == LABEL_NORMAL else score, 3)  # độ chắc của chính nhãn đang hiển thị
        out_people.append({'label': label, 'tier': {LABEL_SUSPECT: TIER_SUSPECT, LABEL_WATCH: TIER_WATCH}.get(label, TIER_NORMAL), 'confidence': shown, 'score': round(score, 3), 'box': p['box'], 'person_conf': p['person_conf'],
                           'origin': p['origin'], 'votes': {k: round(v, 2) for k, v in votes.items()}, 'evidence': why})
    return {'people': out_people, 'phones': phones}


def _dedupe(boxes):
    kept = []
    for b in sorted(boxes, key=lambda d: -(d['confidence'] or 0)):
        if all(iou(b['box'], k['box']) < 0.5 for k in kept):
            kept.append(b)
    return kept


# ---------- chạy mô hình + vẽ ----------
COLORS = {LABEL_CHEAT: (60, 60, 235), V_SUSPECT: (30, 170, 240), V_NORMAL: (90, 200, 90), 'cell phone': (200, 120, 40)}


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
        text = (f'{label} {conf:.2f}' if conf is not None else label)
        text = text if unicode_ok else _ascii(text)
        l, t, rr, bb = g.textbbox((0, 0), text, font=font)
        th = bb - t
        ty = max(0, y0 - th - 8)
        g.rectangle([x0, ty, x0 + (rr - l) + 8, ty + th + 8], fill=(r, gc, bl))
        g.text((x0 + 4, ty + 3 - t), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def components():
    return ['person_coco', *BEHAVIOR_SOURCES]


BEHAVIOR_EVERY = 2        # mô hình hành vi chạy mỗi N lần suy luận thật của một phiên; lần còn lại dùng lại kết quả gần nhất
THREADS_PARALLEL = 3      # đo trên Ryzen 5 4600H: 4 mô hình song song nhanh nhất ở 2-3 luồng/mô hình (6 luồng: 1285 ms, 3 luồng: 759 ms)
THREADS_SOLO = 6
LATE_SOURCES = {'duyen', 'training'}  # KNN (~300 ms) và Training YOLO chỉ chạy khi một mô hình hành vi mạnh đã nghi vấn: theo `decide`, nguồn yếu không ảnh hưởng nhãn khi chưa có nguồn mạnh
# Cổng chuyển động: cảnh gần như không đổi thì dùng lại kết quả (thí sinh ngồi yên là trường hợp phổ biến). Luôn suy luận lại sau MAX_STALE_S.
MOTION_MEAN = 3.0         # độ lệch xám trung bình (0-255) trên ảnh 64x48
MOTION_PIXEL, MOTION_AREA = 25, 0.004  # hoặc >0.4% điểm ảnh đổi hơn 25 mức xám (vật nhỏ như điện thoại/bàn tay xuất hiện)
MAX_STALE_S = 1.0
_now = time.monotonic
DEBUG_LOG = Path(__file__).resolve().parent.parent / 'logs' / 'ensemble_debug.jsonl'  # chỉ ghi khi EXAMVISION_DEBUG=1; chỉ có chỉ số, không có ảnh


def _debug(out, now):
    if os.environ.get('EXAMVISION_DEBUG') != '1':
        return out
    try:
        DEBUG_LOG.parent.mkdir(exist_ok=True)
        row = {'t': round(now, 2), 'ms': out['ms'], 'timings': out['timings'], 'errors': out['errors'], 'reused': [out['behavior_reused'], out['result_reused']],
               'people': [{k: d.get(k) for k in ('id', 'label', 'frame_tier', 'window_score', 'votes', 'evidence')} for d in out['details'] if d.get('id') is not None]}
        with open(DEBUG_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    except Exception:
        pass
    return out


_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='ensemble')


def _signature(frame):
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 48), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (3, 3), 0)


def _moved(prev, cur):
    d = cv2.absdiff(prev, cur)
    return float(d.mean()) > MOTION_MEAN or float((d > MOTION_PIXEL).mean()) > MOTION_AREA


GPU_IMGSZ = 960  # đo trên test (training/eval_person.py): YOLO11m 960 px recall người nhỏ 0.50 so với 0.41 ở 640


def _fit_imgsz(frame, requested, allowed=(640, 800, 960, 1280)):
    """CPU: không phóng to khung nhỏ hơn imgsz (phí công). GPU: phóng to vẫn tăng recall người nhỏ và rẻ, nên giữ nguyên."""
    import runtime
    if requested is None:
        requested = GPU_IMGSZ
    if runtime.device() != 'cpu':
        return requested
    longest = max(frame.shape[:2])
    cap = next((a for a in allowed if a >= longest), allowed[-1])
    return min(requested, cap)


def _pack(frame, fused, ms, timings, errors, reused_behavior, reused_result, include_image):
    details = [{'label': p['label'], 'confidence': p['confidence'], 'box': p['box'], 'score': p['score'], 'person_conf': p['person_conf'],
                'votes': p['votes'], 'evidence': p['evidence'], 'id': p.get('id'), 'window_score': p.get('window_score'),
                'observed_s': p.get('observed_s'), 'frame_tier': p.get('frame_tier')} for p in fused['people']]
    details += [{'label': 'cell phone', 'confidence': ph['confidence'], 'box': ph['box']} for ph in fused['phones']]
    counts = {}
    for d in details:
        counts[d['label']] = counts.get(d['label'], 0) + 1
    out = {'details': details, 'counts': counts, 'ms': ms, 'timings': timings, 'errors': errors,
           'behavior_reused': reused_behavior, 'result_reused': reused_result}
    if include_image:
        ok, enc = cv2.imencode('.jpg', _draw(frame, fused['people'], fused['phones']), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError('Không thể tạo ảnh kết quả')
        import base64
        out['image'] = base64.b64encode(enc).decode()
    return out


def _needs_knn(fused, agree_min=AGREE_MIN_EVIDENCE):
    return any(max(p['votes'].get(k, 0.0) for k in BEHAVIOR_STRONG) >= agree_min for p in fused['people'])


# ---------- quay/nghiêng đầu từ keypoint (YOLO11-Pose; 0 mũi, 1-2 mắt, 3-4 tai) ----------
POSE_WEIGHTS = Path(__file__).resolve().parent.parent / 'models' / 'yolo11n-pose.pt'
KPT_MIN_CONF = 0.5
POSE_MIN_BOX_PX = 80      # người quá nhỏ thì keypoint mặt không đáng tin: không kết luận
KPT_NOISE_PX = 1.0        # sai số định vị keypoint (~1 px ở ảnh đưa vào mô hình 1280): lệch mũi do nhiễu = 1/(khoảng cách hai mắt px)
                          # nên mặt càng nhỏ càng phải quay RÕ mới tính (trừ nhiễu thay vì cắt cứng)
PROFILE_MIN_PX = 6        # nhìn nghiêng (một mắt): mũi-mắt dưới mức này (px ảnh mô hình) là nhiễu, không đo


def head_turn_evidence(xy, conf, scale=1.0):
    """0..1 mức quay đầu sang bên (None = không đủ keypoint để kết luận). xy: 17x2 (px), conf: 17.
    Nhìn thẳng: mũi nằm giữa hai mắt (lệch ~0). Quay 45 độ: mũi sát một mắt. Quay nghiêng: chỉ thấy một mắt/tai."""
    vis = [float(c) >= KPT_MIN_CONF for c in conf[:5]]
    if not vis[0]:
        return None
    nose = xy[0][0]
    if vis[1] and vis[2]:
        dist = abs(float(xy[1][0]) - float(xy[2][0]))
        if dist * scale < 3:
            return None  # suy biến
        r = abs(float(nose) - (float(xy[1][0]) + float(xy[2][0])) / 2) / dist
        noise = KPT_NOISE_PX / (dist * scale)  # sai số của tỉ lệ r do keypoint lệch
        return float(min(1.0, max(0.0, (r - 0.20 - noise) / 0.40)))
    if vis[1] != vis[2]:  # chỉ thấy một mắt: mặt nhìn nghiêng
        eye = xy[1][0] if vis[1] else xy[2][0]
        if abs(float(nose) - float(eye)) * scale < PROFILE_MIN_PX:
            return None
        return 1.0 if vis[3] != vis[4] else 0.8
    return None


def POSE_IMGSZ_FN():
    """GPU: 1280 để mặt người ở xa đủ lớn (đo: hai mắt 12-50 px ở video 1080p); CPU: 640 cho nhanh."""
    import runtime
    return 640 if runtime.device() == 'cpu' else 1280


def head_turn_direction(xy, conf):
    """Hướng mặt trong ẢNH: +1 sang phải, -1 sang trái, 0 không rõ (mũi lệch khỏi trung điểm mắt/tai thấy được)."""
    vis = [float(c) >= KPT_MIN_CONF for c in conf[:5]]
    if not vis[0]:
        return 0
    xs = [float(xy[i][0]) for i in (1, 2) if vis[i]] or [float(xy[i][0]) for i in (3, 4) if vis[i]]
    if not xs:
        return 0
    d = float(xy[0][0]) - sum(xs) / len(xs)
    return 1 if d > 0 else -1 if d < 0 else 0


def pair_evidence(persons, turns):
    """Hai người ngồi cạnh nhau cùng quay đầu VỀ PHÍA NHAU -> bằng chứng 0..1 cho cả hai (None = không có).
    turns[i] = (mức quay 0..1, hướng) hoặc None. Quy tắc theo hình học, không phụ thuộc video cụ thể; chỉ là gợi ý tương tác,
    nên vẫn phải kéo dài (app/judge.py) mới thành kết luận."""
    ev = [None] * len(persons)
    for i, a in enumerate(persons):
        for j in range(i + 1, len(persons)):
            ta, tb = turns[i], turns[j]
            if not ta or not tb or ta[0] < PAIR_TURN_MIN or tb[0] < PAIR_TURN_MIN:
                continue
            ba, bb = a['box'], persons[j]['box']
            ca, cb = (ba[0] + ba[2]) / 2, (bb[0] + bb[2]) / 2
            wide, high = max(ba[2] - ba[0], bb[2] - bb[0]), max(ba[3] - ba[1], bb[3] - bb[1])
            if abs(ca - cb) > PAIR_MAX_GAP * wide or abs((ba[1] + ba[3]) / 2 - (bb[1] + bb[3]) / 2) > 0.5 * high:
                continue
            toward_b = 1 if cb > ca else -1
            if ta[1] == toward_b and tb[1] == -toward_b:
                v = 0.5 + 0.5 * min(ta[0], tb[0])
                ev[i], ev[j] = max(ev[i] or 0.0, v), max(ev[j] or 0.0, v)
    return ev


def assign_turns(persons, dets):
    """Gán kết quả quay đầu của mô hình pose cho từng người (hộp chồng lấp tốt nhất)."""
    out = [None] * len(persons)
    for d in dets:
        box, best, bi = clamp_box(d['box']), 0.0, None
        for i, p in enumerate(persons):
            if associated(box, p['box']):
                ad = area(box)
                fit = max(iou(box, p['box']), inter(box, p['box']) / ad if ad else 0.0)
                if fit > best:
                    best, bi = fit, i
        if bi is not None and d.get('turn') is not None:
            out[bi] = (float(d['turn']), int(d.get('dir', 0)))
    return out


def _pose_infer(frame, conf):
    import runtime
    started = time.perf_counter()
    model = runtime.get_yolo(POSE_WEIGHTS)
    with runtime.model_lock('pose'):
        r = model.predict(frame, conf=conf, verbose=False, device=runtime.device(), half=runtime.use_half(), imgsz=POSE_IMGSZ_FN())[0]
    h, w = frame.shape[:2]
    scale = POSE_IMGSZ_FN() / max(h, w)  # tỉ lệ ảnh gốc -> ảnh đưa vào mô hình
    details = []
    if r.boxes is not None and r.keypoints is not None and len(r.boxes):
        boxes, kxy, kc = r.boxes.xyxy.cpu().numpy(), r.keypoints.xy.cpu().numpy(), r.keypoints.conf
        kc = kc.cpu().numpy() if kc is not None else np.ones(kxy.shape[:2])
        for i in range(min(len(boxes), len(kxy))):
            b = boxes[i]
            ev = head_turn_evidence(kxy[i], kc[i], scale) if (b[3] - b[1]) >= POSE_MIN_BOX_PX else None
            details.append({'label': 'head_turn' if ev else 'normal', 'confidence': round(ev or 0.0, 3), 'turn': ev,
                            'dir': head_turn_direction(kxy[i], kc[i]) if ev else 0,
                            'box': [float(b[0] / w), float(b[1] / h), float(b[2] / w), float(b[3] / h)]})
    return {'details': details, 'ms': round((time.perf_counter() - started) * 1000)}


def _judged(fused, state, now, cfg):
    """Áp kết luận theo thời gian lên BẢN SAO của kết quả khung (kết quả thô còn được tái dùng ở khung sau)."""
    import copy
    judge = state.setdefault('judge', Judge()) if state is not None else Judge(min_frames=1)  # ảnh đơn: chỉ có một khung
    out = {'people': copy.deepcopy(fused['people']), 'phones': fused['phones']}
    judge.apply(out['people'], now, cfg)
    return out


def run(frame, include_image=True, imgsz=None, state=None, cfg=None, now=None):
    """Chạy các mô hình thành phần (song song) trên một khung rồi hợp nhất. Cùng định dạng với `inference.run_model`.

    `state` (dict, mỗi phiên một cái) bật 3 tối ưu: cổng chuyển động (cảnh đứng yên -> dùng lại kết quả), mô hình hành vi chỉ
    chạy mỗi BEHAVIOR_EVERY lần suy luận, và KNN chỉ chạy khi cần. Không truyền `state` (tải ảnh/video) thì chạy đủ mọi mô hình
    thành phần cần thiết ở mọi khung.

    `cfg` (judge.Settings) là các ngưỡng người dùng chỉnh; `now` là thời điểm của khung (giây; video dùng thời gian trong video,
    webcam dùng giờ thực) để kết luận theo thời gian. Mỗi `state` giữ một bộ theo dõi người riêng.
    """
    import inference  # nhập muộn để tránh vòng import
    import runtime
    started = time.perf_counter()
    cfg = cfg or Settings()
    now = _now() if now is None else now
    sig = None
    if state is not None:
        sig = _signature(frame)
        last = state.get('fused')
        if last is not None and _now() - state['t'] < MAX_STALE_S and not _moved(state['sig'], sig):
            return _debug(_pack(frame, _judged(last, state, now, cfg), round((time.perf_counter() - started) * 1000), {}, {}, True, True, include_image), now)
        state['runs'] = state.get('runs', -1) + 1
        fresh = state['runs'] % BEHAVIOR_EVERY == 0 or 'sources' not in state
    else:
        fresh = True
    timings, errors = {}, {}
    runtime.set_threads(THREADS_PARALLEL if fresh else THREADS_SOLO)

    def call(key, conf, size):
        if key == 'pose':
            return _pose_infer(frame, min(conf, cfg.evidence_conf))
        return inference.run_model(key, frame, min(conf, cfg.evidence_conf) if key != 'person_coco' else conf, include_image=False, imgsz=size)
    futs = {'person_coco': _POOL.submit(call, 'person_coco', min(0.25, cfg.person_conf, cfg.phone_conf), _fit_imgsz(frame, imgsz or (None if runtime.device() != 'cpu' else inference.ENSEMBLE_IMGSZ)))}
    early = [k for k in BEHAVIOR_SOURCES if k not in LATE_SOURCES]
    if fresh:
        for key in early:
            futs[key] = _POOL.submit(call, key, BEHAVIOR_SOURCES[key][1], None)

    def gather(fs):
        out = {}
        for key, fut in fs.items():
            try:
                r = fut.result()
            except Exception as exc:
                if key == 'person_coco':
                    raise  # neo người là bắt buộc
                errors[key] = str(exc)[:200]  # một nguồn hành vi lỗi không làm hỏng cả khung; ghi nhận để hiển thị trung thực
                continue
            timings[key] = r['ms']
            out[key] = r
        return out
    got = gather(futs)
    anchor = got['person_coco']
    sources = {k: v['details'] for k, v in got.items() if k != 'person_coco'}
    if fresh:
        fused = fuse(anchor['details'], sources, cfg)
        if _needs_knn(fused, cfg.evidence_conf):
            runtime.set_threads(THREADS_SOLO)
            for key in LATE_SOURCES:
                sources.update({k: v['details'] for k, v in gather({key: _POOL.submit(call, key, BEHAVIOR_SOURCES[key][1], None)}).items()})
        if state is not None:
            state['sources'] = sources
    else:
        sources = state['sources']
    fused = fuse(anchor['details'], sources, cfg)
    if state is not None:
        state.update(fused=fused, sig=sig, t=_now())
    return _debug(_pack(frame, _judged(fused, state, now, cfg), round((time.perf_counter() - started) * 1000), timings, errors, not fresh, False, include_image), now)
