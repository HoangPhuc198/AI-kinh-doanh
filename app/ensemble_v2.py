"""ExamVision Ensemble v2 — bản sửa các lỗi của v1 (v1 giữ nguyên trong app/ensemble.py để quay lại).

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
import time

import cv2  # noqa: F401  (giữ cùng phụ thuộc với v1)

import ensemble as v1
from judge import LABEL_CHEAT, LABEL_NORMAL, TIER_NORMAL, TIER_SUSPECT, TIER_WATCH, Judge, Settings

# ---- nguồn bằng chứng ----
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


def components():
    return ['person_coco', 'pose', *APPEARANCE]


# ---------- quay đầu từ keypoint: hai mức tín hiệu ----------
PROFILE_EV = 0.8   # bằng chứng cố định của tín hiệu YẾU "nhìn nghiêng" (không đo được mức quay)


def head_turn_evidence(xy, conf, scale=1.0):
    """Tín hiệu MẠNH: 0..1 mức quay đầu, chỉ đo khi thấy mũi + HAI mắt (mũi lệch khỏi trung điểm hai mắt); None = không đo được."""
    vis = [float(c) >= v1.KPT_MIN_CONF for c in conf[:3]]
    if not all(vis):
        return None
    return v1.head_turn_evidence(xy, conf, scale)


def profile_cue(xy, conf, scale=1.0):
    """Tín hiệu YẾU "nhìn nghiêng": thấy mũi + đúng MỘT mắt (mũi cách mắt >= PROFILE_MIN_PX như v1) -> PROFILE_EV, không thì None.
    v1 coi đây là quay đầu hoàn toàn (1,0 / 0,8) và cho leo thang lên Gian lận (lỗi 6). Đo trên ảnh có nhãn (valid+test, loại ảnh gần trùng train): người
    bình thường rơi vào nhánh này bị báo 31-34% (nhánh hai mắt: 3%) và không đặc trưng hình học nào tách được tốt (AUC <= 0,77; mũi lệch trục vai: 0,58),
    mà người cúi mặt viết bài hay mất một mắt nên cùng một tín hiệu lặp lại ở mọi khung. Vẫn có ích (recall 0,81 ở nhánh này) nên giữ ở mức Nghi ngờ."""
    vis = [float(c) >= v1.KPT_MIN_CONF for c in conf[:3]]
    if not vis[0] or vis[1] == vis[2]:
        return None
    eye = xy[1][0] if vis[1] else xy[2][0]
    return PROFILE_EV if abs(float(xy[0][0]) - float(eye)) * scale >= v1.PROFILE_MIN_PX else None


def _pose_infer(frame, conf):
    """Như ensemble._pose_infer nhưng trả hai mức: 'turn' (mạnh, hai mắt) và 'side' (yếu, một mắt)."""
    import runtime
    import numpy as np
    started = time.perf_counter()
    model = runtime.get_yolo(v1.POSE_WEIGHTS)
    with runtime.model_lock('pose'):
        r = model.predict(frame, conf=conf, verbose=False, device=runtime.device(), half=runtime.use_half(), imgsz=v1.POSE_IMGSZ_FN())[0]
    h, w = frame.shape[:2]
    scale = v1.POSE_IMGSZ_FN() / max(h, w)
    details = []
    if r.boxes is not None and r.keypoints is not None and len(r.boxes):
        boxes, kxy, kc = r.boxes.xyxy.cpu().numpy(), r.keypoints.xy.cpu().numpy(), r.keypoints.conf
        kc = kc.cpu().numpy() if kc is not None else np.ones(kxy.shape[:2])
        for i in range(min(len(boxes), len(kxy))):
            b = boxes[i]
            big = (b[3] - b[1]) >= v1.POSE_MIN_BOX_PX
            turn = head_turn_evidence(kxy[i], kc[i], scale) if big else None
            side = profile_cue(kxy[i], kc[i], scale) if big else None
            ev = max(turn or 0.0, side or 0.0)
            details.append({'label': 'head_turn' if ev else 'normal', 'confidence': round(ev, 3), 'turn': turn, 'side': side,
                            'dir': v1.head_turn_direction(kxy[i], kc[i]) if ev else 0,
                            'box': [float(b[0] / w), float(b[1] / h), float(b[2] / w), float(b[3] / h)]})
    return {'details': details, 'ms': round((time.perf_counter() - started) * 1000)}


# ---------- gộp bằng chứng (hàm thuần, có unit test) ----------
def pose_evidence(persons, dets, key='turn'):
    """Mức quay đầu LỚN NHẤT (key='turn': mạnh, 'side': yếu) trong mọi detection pose gán cho mỗi người (None = không có). Một hàm duy nhất cho mọi chỗ dùng (lỗi 3 của v1)."""
    out = [None] * len(persons)
    for d in dets:
        turn = d.get(key)
        if turn is None:
            continue
        box, best, bi = v1.clamp_box(d['box']), 0.0, None
        for i, p in enumerate(persons):
            if v1.associated(box, p['box']):
                ad = v1.area(box)
                fit = max(v1.iou(box, p['box']), v1.inter(box, p['box']) / ad if ad else 0.0)
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
    persons = v1.collect_persons(anchor, {}, cfg.person_conf)   # chỉ neo COCO tạo người: nguồn hình dáng không tự thêm người
    phones = v1._dedupe([v1._norm(d) for d in anchor if str(d.get('label', '')).lower() == 'cell phone' and (d.get('confidence') or 0) >= cfg.phone_conf])
    turns = pose_evidence(persons, sources.get('pose', []))
    sides = pose_evidence(persons, sources.get('pose', []), 'side')
    app = {k: v1.assign_evidence(persons, k, sources.get(k, []), labels) for k, (_, labels) in APPEARANCE.items()}
    people = []
    for i, p in enumerate(persons):
        near = [ph['confidence'] for ph in phones if v1.center_in(ph['box'], p['box'])]
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


def run(frame, include_image=True, imgsz=None, state=None, cfg=None, now=None):
    """Cùng giao diện với `ensemble.run`; không có thác KNN/Training. Tối ưu (cổng chuyển động, bỏ qua khung) dùng lại của v1."""
    import inference
    import runtime
    started = time.perf_counter()
    cfg = cfg or Settings()
    now = v1._now() if now is None else now
    sig = None
    if state is not None:
        sig = v1._signature(frame)
        last = state.get('fused')
        if last is not None and v1._now() - state['t'] < v1.MAX_STALE_S and not v1._moved(state['sig'], sig):
            return v1._debug(v1._pack(frame, _judged(last, state, now, cfg), round((time.perf_counter() - started) * 1000), {}, {}, True, True, include_image), now)
        state['runs'] = state.get('runs', -1) + 1
        fresh = state['runs'] % v1.BEHAVIOR_EVERY == 0 or 'sources' not in state
    else:
        fresh = True
    timings, errors = {}, {}
    runtime.set_threads(v1.THREADS_PARALLEL if fresh else v1.THREADS_SOLO)

    def call(key, conf, size):
        if key == 'pose':
            return _pose_infer(frame, min(conf, cfg.evidence_conf))
        return inference.run_model(key, frame, conf if key == 'person_coco' else min(conf, cfg.evidence_conf), include_image=False, imgsz=size)
    futs = {'person_coco': v1._POOL.submit(call, 'person_coco', min(0.25, cfg.person_conf, cfg.phone_conf), v1._fit_imgsz(frame, imgsz or (None if runtime.device() != 'cpu' else inference.ENSEMBLE_IMGSZ)))}
    if fresh:
        futs['pose'] = v1._POOL.submit(call, 'pose', 0.30, None)
        for key, (conf, _) in APPEARANCE.items():
            futs[key] = v1._POOL.submit(call, key, conf, None)
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
        state.update(fused=fused, sig=sig, t=v1._now())
    return v1._debug(v1._pack(frame, _judged(fused, state, now, cfg), round((time.perf_counter() - started) * 1000), timings, errors, not fresh, False, include_image), now)
