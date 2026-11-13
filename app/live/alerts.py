"""Luật cảnh báo cho giám sát trực tiếp.

Đây là "tín hiệu cần giám thị xem xét", KHÔNG phải kết luận gian lận. Mỗi tín hiệu chỉ
dựa trên nhãn mà mô hình đang chọn thực sự xuất ra (xem `SIGNAL_LABELS`). Các nhãn được
coi là "người", "điện thoại" hay "hành vi nghi vấn" là GIẢ ĐỊNH về ngữ nghĩa lớp và cần
được kiểm chứng trên dữ liệu thật trước khi dùng nghiêm túc.

Giảm cảnh báo sai: mỗi điều kiện phải đúng liên tục tối thiểu `min_frames` khung VÀ
`min_seconds` giây, và mỗi loại có thời gian nghỉ (cooldown) giữa hai cảnh báo.
Không hỗ trợ (chưa có mô hình kiểm chứng): hướng đầu/ánh nhìn, nhận diện danh tính.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from datetime import datetime

NEEDS_REVIEW, REVIEWED, DISMISSED = 'needs_review', 'reviewed', 'dismissed'
REVIEW_STATUSES = (NEEDS_REVIEW, REVIEWED, DISMISSED)

_BEHAVIOR_FLAGS = {'suspicious', 'cheating', 'cheating-behaviour', 'nghi vấn gian lận'}

# Nhãn nào của mô hình nào được coi là gì. `None` = mọi nhãn đều là một người.
SIGNAL_LABELS = {
    'paper': {'person': None, 'phone': set()},
    'duyen': {'person': None, 'phone': set()},
    'training': {'person': {'normal', 'cheating'}, 'phone': set()},
    'yolo_duyen': {'person': {'normal', 'cheating'}, 'phone': set()},
    # gian_lan có cả lớp 'person' lẫn nhãn hành vi chồng lên cùng một người nên chỉ đếm 'person'
    # cho số người; 'invigilator' (giám thị) không tính là thí sinh.
    'person_coco': {'person': {'person'}, 'phone': {'cell phone'}},
    # ensemble: mỗi hộp người mang nhãn kết luận; điện thoại là hộp riêng. Ngưỡng đã nằm trong app/ensemble.py.
    'ensemble': {'person': {'bình thường', 'cần chú ý', 'nghi vấn gian lận'}, 'phone': {'cell phone'}},
    'person_phone': {'person': {'person'}, 'phone': {'cell phone'}},
    'gianlan': {'person': {'person', 'normal', 'cheating', 'cheating-behaviour', 'submitting'},
                'count_person': {'person'}, 'phone': {'cell-phones'}},
}


# Ngưỡng đã hiệu chỉnh trên tập val coco_person_phone (imgsz 960), kiểm tra trên test; xem MODEL_EVALUATION.md.
# Mỗi tín hiệu dùng ngưỡng riêng vì chi phí sai khác nhau:
#   presence : "có người" (để cảnh báo KHÔNG THẤY NGƯỜI) - ngưỡng thấp => ít bỏ sót => ít báo "không thấy" nhầm
#   count    : đếm người (để cảnh báo NHIỀU NGƯỜI) - ngưỡng cao => ít người dư => ít báo "nhiều người" nhầm
#   phone    : điện thoại - ngưỡng cao vì precision quan trọng hơn recall
#   display  : hộp được vẽ trên giao diện
CALIBRATION = {
    'person_coco': {'infer_conf': 0.25, 'presence': 0.25, 'count': 0.50, 'phone': 0.35, 'display_person': 0.35, 'display_phone': 0.35},
}
CALIBRATION['person_phone'] = CALIBRATION['person_coco']


def display_filter(model_key: str, details: list[dict]) -> list[dict]:
    """Lọc hộp hiển thị theo ngưỡng hiệu chỉnh (mô hình không có hiệu chỉnh: giữ nguyên)."""
    cal = CALIBRATION.get(model_key)
    if not cal:
        return details
    keep = []
    for d in details:
        lb, c = str(d.get('label', '')).lower(), d.get('confidence')
        if lb not in ('person', 'cell phone'):  # bỏ 78 lớp COCO không liên quan
            continue
        need = cal['display_phone'] if lb == 'cell phone' else cal['display_person']
        if c is None or c >= need:
            keep.append(d)
    return keep


@dataclass(frozen=True)
class Rule:
    type: str
    severity: str          # low | medium | high
    min_frames: int
    min_seconds: float
    cooldown: float
    description: str
    grace: float = 1.0     # giây được phép mất tín hiệu mà không ngắt chuỗi (chống nhấp nháy)
    modes: tuple = ('single', 'room')


DEFAULT_RULES = (
    Rule('no_person', 'medium', 3, 5.0, 30.0, 'Không phát hiện người trong khung hình (có thể rời vị trí hoặc bị che).', 1.0, ('single',)),
    Rule('multiple_people', 'high', 3, 2.0, 30.0, 'Phát hiện từ 2 người trở lên trong khung hình (chế độ một thí sinh).', 1.0, ('single',)),
    Rule('phone', 'high', 2, 1.0, 30.0, 'Phát hiện vật thể được mô hình gán là điện thoại.', 1.5),
    Rule('suspicious_label', 'medium', 3, 2.0, 30.0, 'Mô hình gán nhãn hành vi nghi vấn kéo dài.', 1.5),
)


def extract_signals(model_key: str, details: list[dict]) -> dict:
    """Chuyển đầu ra mô hình thành các tín hiệu boolean/đếm."""
    cfg = SIGNAL_LABELS.get(model_key, {'person': None, 'phone': set()})
    cal = CALIBRATION.get(model_key)
    if cal:  # tách ngưỡng theo mục đích tín hiệu (xem CALIBRATION)
        def at_least(lb, d, floor):
            c = d.get('confidence')
            return c is None or c >= floor
        phone_set = cfg['phone']
        present = [d for d in details if str(d.get('label', '')).lower() in cfg['person'] and at_least(0, d, cal['presence'])]
        counted = [d for d in details if str(d.get('label', '')).lower() in cfg['person'] and at_least(0, d, cal['count'])]
        phones = [d for d in details if str(d.get('label', '')).lower() in phone_set and at_least(0, d, cal['phone'])]
        top = lambda ds: round(max(d['confidence'] for d in ds), 2) if ds and all(d.get('confidence') is not None for d in ds) else None  # noqa: E731
        return {'has_person': bool(present), 'person_count': len(counted), 'phone': bool(phones), 'suspicious_label': False,
                'supports_phone': bool(phone_set),
                'conf': {'phone': top(phones), 'suspicious_label': None, 'multiple_people': top(counted), 'no_person': None}}
    labels = [str(d.get('label', '')).lower() for d in details]
    person_set, phone_set = cfg['person'], cfg['phone']
    persons = [lb for lb in labels if person_set is None or lb in person_set]
    count_set = cfg.get('count_person', person_set)
    counted = [lb for lb in labels if count_set is None or lb in count_set]
    def best(pred):
        confs = [d.get('confidence') for d in details if pred(str(d.get('label', '')).lower()) and d.get('confidence') is not None]
        return round(max(confs), 2) if confs else None
    return {
        'has_person': bool(persons),
        'person_count': len(counted) if counted else len(persons),
        'phone': any(lb in phone_set for lb in labels),
        'suspicious_label': any(lb in _BEHAVIOR_FLAGS for lb in labels),
        'supports_phone': bool(phone_set),
        'evidence': next((d.get('evidence') for d in details if str(d.get('label', '')).lower() in _BEHAVIOR_FLAGS and d.get('evidence')), None),
        # độ tin cậy cao nhất của các hộp chứng minh cho tín hiệu (None nếu mô hình không có confidence)
        'conf': {'phone': best(phone_set.__contains__), 'suspicious_label': best(_BEHAVIOR_FLAGS.__contains__),
                 'multiple_people': best(lambda lb: person_set is None or lb in person_set), 'no_person': None},
    }


@dataclass
class _Track:
    frames: int = 0
    since: float | None = None
    last_alert: float = float('-inf')
    last_active: float | None = None


@dataclass
class AlertEngine:
    rules: tuple = DEFAULT_RULES
    clock: callable = time.monotonic
    mode: str = 'single'   # 'single' = webcam một thí sinh; 'room' = camera toàn phòng (nhiều người là bình thường)
    _tracks: dict = field(default_factory=dict)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def reset_streaks(self):
        """Gọi khi tạm dừng/tiếp tục để khoảng trống không bị tính là liên tục."""
        for t in self._tracks.values():
            t.frames, t.since, t.last_active = 0, None, None

    @staticmethod
    def _active(rule_type: str, s: dict) -> bool:
        if rule_type == 'no_person':
            return not s['has_person']
        if rule_type == 'multiple_people':
            return s['person_count'] >= 2
        if rule_type == 'phone':
            return s['phone']
        return s['suspicious_label']

    def update(self, signals: dict, model_name: str = '') -> list[dict]:
        now = self.clock()
        fired = []
        for rule in self.rules:
            if self.mode not in rule.modes:
                continue
            track = self._tracks.setdefault(rule.type, _Track())
            if not self._active(rule.type, signals):
                if track.last_active is not None and now - track.last_active > rule.grace:
                    track.frames, track.since = 0, None
                continue
            if track.last_active is not None and now - track.last_active > rule.grace:
                track.frames, track.since = 0, None  # khoảng trống quá dài: bắt đầu chuỗi mới
            track.last_active = now
            if track.since is None:
                track.since = now
            track.frames += 1
            if (track.frames >= rule.min_frames and now - track.since >= rule.min_seconds
                    and now - track.last_alert >= rule.cooldown):
                track.last_alert = now
                fired.append({
                    'id': next(self._ids),
                    'time': datetime.now().astimezone().isoformat(timespec='seconds'),
                    'type': rule.type,
                    'severity': rule.severity,
                    'description': rule.description,
                    'model': model_name,
                    'mode': self.mode,
                    'confidence': signals.get('conf', {}).get(rule.type),
                    'frames': track.frames,
                    'evidence': signals.get('evidence') if rule.type == 'suspicious_label' else None,
                    'duration_s': round(now - track.since, 1),
                    'status': NEEDS_REVIEW,
                    'note': 'Tín hiệu cần giám thị xem xét; không phải kết luận gian lận.',
                })
        return fired

