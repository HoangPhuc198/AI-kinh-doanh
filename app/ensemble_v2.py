"""ExamVision Ensemble v2 — bản sửa các lỗi của v1 (v1 giữ nguyên trong app/ensemble.py để quay lại). FILE NÀY TỰ CHỨA.

Tự chứa
-------
File này KHÔNG import ensemble.py (v1). Chỉ dùng chung hai mô-đun nền: judge.py (kết luận theo thời gian + Settings) và runtime.py (nạp mô hình).
Các hàm phụ (hình học, vẽ, cổng chuyển động, đo keypoint) được chép NGUYÊN VĂN từ v1 trong các mục đánh dấu "chép từ ensemble.py".
tests/test_ensemble_v2.py kiểm tra: (a) từng hàm chép cho kết quả y hệt bản gốc trên dữ liệu ngẫu nhiên (DifferentialTests), (b) file này không import v1
(SelfContainedTests). Khác biệt CÓ CHỦ Ý so với bản chép:
  - collect_persons: chỉ neo COCO tạo người (v1 còn thêm người từ các nguồn hành vi; v2 luôn gọi với sources rỗng nên kết quả giống hệt);
  - assign_evidence: bằng chứng thiếu (confidence None) tính 0,0 (v1 dùng mức KNN danh nghĩa 0,70; YOLO luôn có confidence nên không xảy ra);
  - head_turn_evidence: chỉ nhánh HAI mắt (nhánh một mắt của v1 nay là profile_cue, tín hiệu yếu);
  - _POOL dùng nhóm luồng riêng ('ensemble_v2'); nhật ký debug ghi vào logs/ensemble_v2_debug.jsonl (v1: ensemble_debug.jsonl).

Lỗi của v1 đã đo/thấy được và cách v2 sửa
------------------------------------------
1. NGỒI IM BỊ BÁO "GIAN LẬN". KNN Duyên + gianlan cùng bắn liên tục trên một dáng ngồi yên, v1 coi là "hai nguồn độc lập" và coi việc kéo dài
   là bằng chứng (log thật: 60/115 khung Gian lận ở người ngồi im). KNN phân loại theo tọa độ keypoint TUYỆT ĐỐI trong ảnh (paper_detector.FEATURES),
   tức học cả VỊ TRÍ ngồi; trên video lạ F1 chỉ 0,27 (docs/reports/MODEL_EVALUATION.md). -> v2 bỏ KNN; nguồn "hình dáng" (APPEARANCE) chỉ được
   củng cố một tín hiệu ĐO ĐƯỢC (quay đầu bằng hình học, điện thoại), không bao giờ tự gây cờ: dáng tĩnh bị mô hình nhìn nhầm thì lặp lại ở mọi khung.
2. HÀNH VI RÕ RÀNG KHÔNG BAO GIỜ LÊN "GIAN LẬN". Một nguồn kéo dài chỉ có trọng số 0,3 < ngưỡng 0,45 nên quay đầu/điện thoại giữ 30 giây vẫn "Nghi ngờ".
   -> v2: tín hiệu đo được (quay đầu / điện thoại) liên tục >= SINGLE_SOURCE_FACTOR x min_seconds thì "Gian lận" (Escalator); hai tín hiệu đồng thuận thì như v1.
3. assign_turns (lấy detection GÁN SAU CÙNG) và assign_evidence (lấy LỚN NHẤT) có thể cho hai kết quả khác nhau cho cùng một người khi đứng sát nhau.
   -> v2 chỉ có một hàm gộp (pose_evidence: lớn nhất).
4. Quá nhiều nút ẩn: mức KNN danh nghĩa 0,70, WEAK/UNNORMALIZED/LATE_SOURCES, thác KNN/Training, trọng số tay -> v2: tối đa 3 nguồn, một bảng luật, không có thác.
5. Chưa từng đo trên dữ liệu có nhóm -> v2 đi kèm số đo theo nhóm (training/eval_by_group.py, reports/eval_by_group_*) làm mốc: đổi gì cũng phải đo lại.
6. CÚI MẶT VIẾT BÀI BỊ BÁO GIAN LẬN. Nhánh "chỉ thấy một mắt" của head_turn_evidence trả hằng số 1,0/0,8 mà không đo, nên người cúi (hay mất một mắt) bị đọc thành quay ngang
   hoàn toàn. Đo trên ảnh có nhãn: nhánh một mắt báo nhầm 31-34% (hai mắt: 3%) và không phép đo hình học nào tách được tốt. -> v2 tách tín hiệu YẾU (pose_side: tối đa
   Nghi ngờ, không leo thang) khỏi tín hiệu MẠNH (hai mắt, đo thật). Chi tiết và số đo: docs/ENSEMBLE_V2.md.

Giới hạn trung thực (KHÔNG được sửa ở v2): quay đầu đo bằng hình học mũi/mắt nên không thấy quay người khi mặt vẫn hướng camera hoặc nhìn xuống;
SINGLE_SOURCE_FACTOR và các ngưỡng thời gian là heuristic, chưa hiệu chỉnh trên video thi có nhãn thời gian; điện thoại "gần người" chưa phân biệt cầm dùng
với để trên bàn. Kết luận là gợi ý cho giám thị, không dùng để xử phạt.
"""
from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from judge import LABEL_CHEAT, LABEL_NORMAL, LABEL_SUSPECT, TIER_NORMAL, TIER_SUSPECT, TIER_WATCH, Judge, Settings

# =====================================================================================================================
# 1. NGUỒN BẰNG CHỨNG VÀ LUẬT CỦA v2
# =====================================================================================================================
STRONG = ('pose', 'phone')      # ĐO ĐƯỢC: quay đầu hai mắt (YOLO11-Pose) / vật thể điện thoại (YOLO COCO, cùng mô hình neo người): được leo thang lên Gian lận
WEAK_CUES = ('pose_side',)      # tín hiệu hình học YẾU (nhìn nghiêng một mắt): tự báo tối đa Nghi ngờ, không leo thang (xem profile_cue)
# Nguồn hình dáng (mô hình huấn luyện riêng): khóa -> (ngưỡng conf khi chạy, nhãn được coi là nghi vấn).
# Chọn bằng đánh giá theo nhóm (training/eval_by_group.py + pool_by_group.py), tiêu chí đặt TRƯỚC khi xem kết quả: Youden J (= recall - tỉ lệ báo nhầm)
# gộp 3 bộ Roboflow có cận dưới khoảng tin cậy theo nhóm > 0 và J > 0 ở >= 2/3 bộ; loại KNN theo thiết kế (lỗi 1); không đạt: gianlan, yolo_duyen.
# Ngưỡng chạy 0,5 giữ như v1 (đo ngoài mẫu ở 0,25 báo nhầm ~26%, ở 0,5 ~8%).
APPEARANCE: dict[str, tuple[float, set]] = {'training': (0.50, {'cheating'})}
SOURCE_NAMES = {'pose': 'Quay đầu', 'pose_side': 'Nhìn nghiêng', 'phone': 'Điện thoại gần người', 'training': 'Training YOLO'}
SINGLE_SOURCE_FACTOR = 3.0      # tín hiệu đo được giữ liên tục >= FACTOR x cfg.min_seconds => Gian lận (mặc định 3 x 1,5 = 4,5 s); heuristic
GAP_S = 1.0                     # mất tín hiệu <= ngần này giây không ngắt chuỗi (mô hình có thể bỏ sót vài khung)
AGREE_MIN_EVIDENCE = 0.55
PROFILE_EV = 0.8                # bằng chứng cố định của tín hiệu YẾU "nhìn nghiêng" (không đo được mức quay)


def components():
    return ['person_coco', 'pose', *APPEARANCE]


# =====================================================================================================================
# 2. CẤU HÌNH CHẠY — chép từ ensemble.py (v1)
# =====================================================================================================================
BEHAVIOR_EVERY = 2        # mô hình hành vi chạy mỗi N lần suy luận thật của một phiên; lần còn lại dùng lại kết quả gần nhất
THREADS_PARALLEL = 3      # đo trên Ryzen 5 4600H: 4 mô hình song song nhanh nhất ở 2-3 luồng/mô hình (6 luồng: 1285 ms, 3 luồng: 759 ms)
THREADS_SOLO = 6
# Cổng chuyển động: cảnh gần như không đổi thì dùng lại kết quả (thí sinh ngồi yên là trường hợp phổ biến). Luôn suy luận lại sau MAX_STALE_S.
MOTION_MEAN = 3.0         # độ lệch xám trung bình (0-255) trên ảnh 64x48
MOTION_PIXEL, MOTION_AREA = 25, 0.004  # hoặc >0.4% điểm ảnh đổi hơn 25 mức xám (vật nhỏ như điện thoại/bàn tay xuất hiện)
MAX_STALE_S = 1.0
_now = time.monotonic
DEBUG_LOG = Path(__file__).resolve().parent.parent / 'logs' / 'ensemble_v2_debug.jsonl'  # chỉ ghi khi EXAMVISION_DEBUG=1; chỉ có chỉ số, không có ảnh
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='ensemble_v2')
GPU_IMGSZ = 960  # đo trên test (training/eval_person.py): YOLO11m 960 px recall người nhỏ 0.50 so với 0.41 ở 640

# quay/nghiêng đầu từ keypoint (YOLO11-Pose; 0 mũi, 1-2 mắt, 3-4 tai)
POSE_WEIGHTS = Path(__file__).resolve().parent.parent / 'models' / 'yolo11n-pose.pt'
KPT_MIN_CONF = 0.5
POSE_MIN_BOX_PX = 80      # người quá nhỏ thì keypoint mặt không đáng tin: không kết luận
KPT_NOISE_PX = 1.0        # sai số định vị keypoint (~1 px ở ảnh đưa vào mô hình 1280): lệch mũi do nhiễu = 1/(khoảng cách hai mắt px)
                          # nên mặt càng nhỏ càng phải quay RÕ mới tính (trừ nhiễu thay vì cắt cứng)
PROFILE_MIN_PX = 6        # nhìn nghiêng (một mắt): mũi-mắt dưới mức này (px ảnh mô hình) là nhiễu, không đo


# =====================================================================================================================
# 3. HÌNH HỌC (hộp chuẩn hóa [x0,y0,x1,y1] trong [0,1]) — chép từ ensemble.py (v1)
# =====================================================================================================================
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


def _norm(d):
    return {'label': str(d.get('label', '')).lower(), 'confidence': d.get('confidence'), 'box': clamp_box(d['box'])}


def _dedupe(boxes):
    kept = []
    for b in sorted(boxes, key=lambda d: -(d['confidence'] or 0)):
        if all(iou(b['box'], k['box']) < 0.5 for k in kept):
            kept.append(b)
    return kept


def collect_persons(anchor: list[dict], person_min: float):
    """Danh sách người: CHỈ neo person_coco tạo người (nguồn hình dáng không tự thêm người)."""
    persons = []
    for d in anchor:
        d = _norm(d)
        if d['label'] == 'person' and (d['confidence'] or 0) >= person_min and area(d['box']) > 0:
            persons.append({'box': d['box'], 'person_conf': d['confidence'], 'origin': 'person_coco'})
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
            conf = float(d['confidence'] or 0.0)   # v1: KNN_NOMINAL_CONF nếu thiếu; YOLO luôn có confidence
        ev[best_i] = max(ev[best_i] or 0.0, conf)
    return ev


# =====================================================================================================================
# 4. VẼ + ĐÓNG GÓI KẾT QUẢ — chép từ ensemble.py (v1)
# =====================================================================================================================
COLORS = {LABEL_CHEAT: (60, 60, 235), LABEL_SUSPECT: (30, 170, 240), LABEL_NORMAL: (90, 200, 90), 'cell phone': (200, 120, 40)}
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


# =====================================================================================================================
# 5. CỔNG CHUYỂN ĐỘNG + KÍCH THƯỚC ẢNH — chép từ ensemble.py (v1)
# =====================================================================================================================
def _signature(frame):
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 48), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (3, 3), 0)


def _moved(prev, cur):
    d = cv2.absdiff(prev, cur)
    return float(d.mean()) > MOTION_MEAN or float((d > MOTION_PIXEL).mean()) > MOTION_AREA


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


# =====================================================================================================================
# 6. QUAY ĐẦU TỪ KEYPOINT: hai mức tín hiệu
# =====================================================================================================================
def POSE_IMGSZ_FN():
    """GPU: 1280 để mặt người ở xa đủ lớn (đo: hai mắt 12-50 px ở video 1080p); CPU: 640 cho nhanh. (chép từ ensemble.py)"""
    import runtime
    return 640 if runtime.device() == 'cpu' else 1280


def head_turn_direction(xy, conf):
    """Hướng mặt trong ẢNH: +1 sang phải, -1 sang trái, 0 không rõ (mũi lệch khỏi trung điểm mắt/tai thấy được). (chép từ ensemble.py)"""
    vis = [float(c) >= KPT_MIN_CONF for c in conf[:5]]
    if not vis[0]:
        return 0
    xs = [float(xy[i][0]) for i in (1, 2) if vis[i]] or [float(xy[i][0]) for i in (3, 4) if vis[i]]
    if not xs:
        return 0
    d = float(xy[0][0]) - sum(xs) / len(xs)
    return 1 if d > 0 else -1 if d < 0 else 0


def head_turn_evidence(xy, conf, scale=1.0):
    """Tín hiệu MẠNH: 0..1 mức quay đầu, chỉ đo khi thấy mũi + HAI mắt (mũi lệch khỏi trung điểm hai mắt); None = không đo được.
    Nhìn thẳng: mũi nằm giữa hai mắt (lệch ~0). Quay 45 độ: mũi sát một mắt. (nhánh hai mắt của ensemble.head_turn_evidence)"""
    vis = [float(c) >= KPT_MIN_CONF for c in conf[:3]]
    if not all(vis):
        return None
    nose = xy[0][0]
    dist = abs(float(xy[1][0]) - float(xy[2][0]))
    if dist * scale < 3:
        return None  # suy biến
    r = abs(float(nose) - (float(xy[1][0]) + float(xy[2][0])) / 2) / dist
    noise = KPT_NOISE_PX / (dist * scale)  # sai số của tỉ lệ r do keypoint lệch
    return float(min(1.0, max(0.0, (r - 0.20 - noise) / 0.40)))


def profile_cue(xy, conf, scale=1.0):
    """Tín hiệu YẾU "nhìn nghiêng": thấy mũi + đúng MỘT mắt (mũi cách mắt >= PROFILE_MIN_PX như v1) -> PROFILE_EV, không thì None.
    v1 coi đây là quay đầu hoàn toàn (1,0 / 0,8) và cho leo thang lên Gian lận (lỗi 6). Đo trên ảnh có nhãn (valid+test, loại ảnh gần trùng train): người
    bình thường rơi vào nhánh này bị báo 31-34% (nhánh hai mắt: 3%) và không đặc trưng hình học nào tách được tốt (AUC <= 0,77; mũi lệch trục vai: 0,58),
    mà người cúi mặt viết bài hay mất một mắt nên cùng một tín hiệu lặp lại ở mọi khung. Vẫn có ích (recall 0,81 ở nhánh này) nên giữ ở mức Nghi ngờ."""
    vis = [float(c) >= KPT_MIN_CONF for c in conf[:3]]
    if not vis[0] or vis[1] == vis[2]:
        return None
    eye = xy[1][0] if vis[1] else xy[2][0]
    return PROFILE_EV if abs(float(xy[0][0]) - float(eye)) * scale >= PROFILE_MIN_PX else None


def _pose_infer(frame, conf):
    """Chạy YOLO11-Pose và trả hai mức: 'turn' (mạnh, hai mắt) và 'side' (yếu, một mắt)."""
    import runtime
    started = time.perf_counter()
    model = runtime.get_yolo(POSE_WEIGHTS)
    with runtime.model_lock('pose'):
        r = model.predict(frame, conf=conf, verbose=False, device=runtime.device(), half=runtime.use_half(), imgsz=POSE_IMGSZ_FN())[0]
    h, w = frame.shape[:2]
    scale = POSE_IMGSZ_FN() / max(h, w)
    details = []
    if r.boxes is not None and r.keypoints is not None and len(r.boxes):
        boxes, kxy, kc = r.boxes.xyxy.cpu().numpy(), r.keypoints.xy.cpu().numpy(), r.keypoints.conf
        kc = kc.cpu().numpy() if kc is not None else np.ones(kxy.shape[:2])
        for i in range(min(len(boxes), len(kxy))):
            b = boxes[i]
            big = (b[3] - b[1]) >= POSE_MIN_BOX_PX
            turn = head_turn_evidence(kxy[i], kc[i], scale) if big else None
            side = profile_cue(kxy[i], kc[i], scale) if big else None
            ev = max(turn or 0.0, side or 0.0)
            details.append({'label': 'head_turn' if ev else 'normal', 'confidence': round(ev, 3), 'turn': turn, 'side': side,
                            'dir': head_turn_direction(kxy[i], kc[i]) if ev else 0,
                            'box': [float(b[0] / w), float(b[1] / h), float(b[2] / w), float(b[3] / h)]})
    return {'details': details, 'ms': round((time.perf_counter() - started) * 1000)}


# =====================================================================================================================
# 7. GỘP BẰNG CHỨNG (hàm thuần, có unit test)
# =====================================================================================================================
def pose_evidence(persons, dets, key='turn'):
    """Mức quay đầu LỚN NHẤT (key='turn': mạnh, 'side': yếu) trong mọi detection pose gán cho mỗi người (None = không có). Một hàm duy nhất cho mọi chỗ dùng (lỗi 3 của v1)."""
    out = [None] * len(persons)
    for d in dets:
        turn = d.get(key)
        if turn is None:
            continue
        box, best, bi = clamp_box(d['box']), 0.0, None
        for i, p in enumerate(persons):
            if associated(box, p['box']):
                ad = area(box)
                fit = max(iou(box, p['box']), inter(box, p['box']) / ad if ad else 0.0)
                if fit > best:
                    best, bi = fit, i
        if bi is not None:
            out[bi] = max(out[bi] or 0.0, float(turn))
    return out


def decide(votes: dict[str, float], cfg: Settings | None = None):
    """votes: nguồn -> bằng chứng 0..1. Trả về (tier, điểm).
    suspect: có tín hiệu ĐO ĐƯỢC (STRONG) và >= 2 nguồn đồng thuận (nguồn yếu chỉ củng cố); watch: đúng một tín hiệu đo được, hoặc riêng tín hiệu yếu
    "nhìn nghiêng" (tối đa Nghi ngờ); nguồn hình dáng đứng một mình = bình thường."""
    on = cfg.evidence_conf if cfg else AGREE_MIN_EVIDENCE
    strong = [k for k in STRONG if votes.get(k, 0.0) >= on]
    weak = [k for k in (*WEAK_CUES, *APPEARANCE) if votes.get(k, 0.0) >= on]
    active = strong + weak
    score = round(max((votes[k] for k in active), default=0.0), 3)
    if strong and len(active) >= 2:
        return TIER_SUSPECT, score
    if strong or any(k in WEAK_CUES for k in weak):
        return TIER_WATCH, score
    return TIER_NORMAL, 0.0


def fuse(anchor: list[dict], sources: dict[str, list[dict]], cfg: Settings | None = None) -> dict:
    """anchor: đầu ra person_coco (person + cell phone). sources: 'pose' + các nguồn APPEARANCE -> details."""
    cfg = cfg or Settings()
    persons = collect_persons(anchor, cfg.person_conf)   # chỉ neo COCO tạo người: nguồn hình dáng không tự thêm người
    phones = _dedupe([_norm(d) for d in anchor if str(d.get('label', '')).lower() == 'cell phone' and (d.get('confidence') or 0) >= cfg.phone_conf])
    turns = pose_evidence(persons, sources.get('pose', []))
    sides = pose_evidence(persons, sources.get('pose', []), 'side')
    app = {k: assign_evidence(persons, k, sources.get(k, []), labels) for k, (_, labels) in APPEARANCE.items()}
    people = []
    for i, p in enumerate(persons):
        near = [ph['confidence'] for ph in phones if center_in(ph['box'], p['box'])]
        votes = {'pose': turns[i] or 0.0, 'pose_side': sides[i] or 0.0, 'phone': max(near, default=0.0), **{k: (app[k][i] or 0.0) for k in APPEARANCE}}
        tier, score = decide(votes, cfg)
        on = cfg.evidence_conf
        why = []
        if votes['pose'] >= on:
            why.append(f"Quay/nghiêng đầu ({votes['pose']:.2f})")
        if votes['pose_side'] >= on:
            why.append(f"Quay/nghiêng đầu (nhìn nghiêng, một mắt) ({votes['pose_side']:.2f})")
        if votes['phone'] >= on:
            why.append(f"Điện thoại gần người ({votes['phone']:.2f})")
        why += [f"{SOURCE_NAMES.get(k, k)} nghi vấn ({votes[k]:.2f})" for k in APPEARANCE if votes[k] >= on and tier != TIER_NORMAL]
        people.append({'label': {TIER_SUSPECT: 'Nghi vấn gian lận', TIER_WATCH: 'Cần chú ý'}.get(tier, 'Bình thường'), 'tier': tier,
                       'confidence': round(1 - score if tier == TIER_NORMAL else score, 3), 'score': score, 'box': p['box'], 'person_conf': p['person_conf'],
                       'origin': p['origin'], 'votes': {k: round(v, 2) for k, v in votes.items()}, 'evidence': why,
                       'strong_on': any(votes[k] >= on for k in STRONG)})  # chỉ tín hiệu ĐO ĐƯỢC (STRONG) mới được leo thang
    return {'people': people, 'phones': phones}


class Escalator:
    """Lỗi 2 của v1: tín hiệu ĐO ĐƯỢC giữ liên tục đủ lâu thì kết luận Gian lận, dù chỉ có một nguồn. Dùng cùng id người với Judge."""

    def __init__(self):
        self._s: dict[int, dict] = {}

    def apply(self, people: list[dict], now: float, cfg: Settings):
        need = SINGLE_SOURCE_FACTOR * cfg.min_seconds
        for p in people:
            s = self._s.setdefault(p['id'], {'since': None, 'last': None, 'seen': now})
            if now < s['seen']:                                    # video tua lại: bỏ chuỗi cũ
                s['since'] = s['last'] = None
            s['seen'] = now
            if s['last'] is not None and now < s['last']:          # video tua lại
                s['since'] = s['last'] = None
            if p.get('strong_on'):
                if s['since'] is None or now - s['last'] > GAP_S:
                    s['since'] = now                                # bắt đầu chuỗi mới
                s['last'] = now
            elif s['last'] is not None and now - s['last'] > cfg.recover_seconds:
                s['since'] = s['last'] = None                       # hết tín hiệu: về như bình thường
            held = (now - s['since']) if s['since'] is not None else 0.0
            p['held_s'] = round(held, 1)
            if s['since'] is not None and now - s['last'] <= cfg.recover_seconds and held >= need:
                p['label'] = LABEL_CHEAT
                p['confidence'] = round(max(p.get('confidence') or 0.0, p.get('score') or 0.0), 3)
        for pid in [k for k, s in self._s.items() if now - s['seen'] > 60]:
            del self._s[pid]                                        # không giữ mãi track đã biến mất (kể cả người chưa từng có tín hiệu)


def _judged(fused, state, now, cfg):
    """Kết luận theo thời gian lên BẢN SAO của kết quả khung (kết quả thô còn được tái dùng ở khung sau)."""
    out = {'people': copy.deepcopy(fused['people']), 'phones': fused['phones']}
    if state is None:                      # ảnh đơn: một khung, không có thời gian để leo thang
        Judge(min_frames=1).apply(out['people'], now, cfg)
        return out
    state.setdefault('judge', Judge()).apply(out['people'], now, cfg)
    state.setdefault('esc', Escalator()).apply(out['people'], now, cfg)
    return out


# =====================================================================================================================
# 8. CHẠY MÔ HÌNH
# =====================================================================================================================
def run(frame, include_image=True, imgsz=None, state=None, cfg=None, now=None):
    """Cùng giao diện với `ensemble.run`; không có thác KNN/Training. Tối ưu: cổng chuyển động (cảnh đứng yên -> dùng lại kết quả) và
    mô hình hành vi chỉ chạy mỗi BEHAVIOR_EVERY lần suy luận thật của một phiên (`state` mỗi phiên một cái; không truyền thì chạy đủ mọi khung)."""
    import inference
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
        return inference.run_model(key, frame, conf if key == 'person_coco' else min(conf, cfg.evidence_conf), include_image=False, imgsz=size)
    futs = {'person_coco': _POOL.submit(call, 'person_coco', min(0.25, cfg.person_conf, cfg.phone_conf), _fit_imgsz(frame, imgsz or (None if runtime.device() != 'cpu' else inference.ENSEMBLE_IMGSZ)))}
    if fresh:
        futs['pose'] = _POOL.submit(call, 'pose', 0.30, None)
        for key, (conf, _) in APPEARANCE.items():
            futs[key] = _POOL.submit(call, key, conf, None)
    got = {}
    for key, fut in futs.items():
        try:
            r = fut.result()
        except Exception as exc:
            if key == 'person_coco':
                raise  # neo người là bắt buộc
            errors[key] = str(exc)[:200]  # một nguồn lỗi không làm hỏng cả khung; ghi nhận để hiển thị trung thực
            continue
        timings[key] = r['ms']
        got[key] = r
    anchor = got['person_coco']
    if fresh:
        sources = {k: v['details'] for k, v in got.items() if k != 'person_coco'}
        if state is not None:
            state['sources'] = sources
    else:
        sources = state['sources']
    fused = fuse(anchor['details'], sources, cfg)
    if state is not None:
        state.update(fused=fused, sig=sig, t=_now())
    return _debug(_pack(frame, _judged(fused, state, now, cfg), round((time.perf_counter() - started) * 1000), timings, errors, not fresh, False, include_image), now)
