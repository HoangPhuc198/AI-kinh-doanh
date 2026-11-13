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

_BEHAVIOR_FLAGS = {'suspicious', 'cheating', 'cheating-behaviour'}

# Nhãn nào của mô hình nào được coi là gì. `None` = mọi nhãn đều là một người.
SIGNAL_LABELS = {
    'paper': {'person': None, 'phone': set()},
    'duyen': {'person': None, 'phone': set()},
    'training': {'person': {'normal', 'cheating'}, 'phone': set()},
    'yolo_duyen': {'person': {'normal', 'cheating'}, 'phone': set()},
    # gian_lan có cả lớp 'person' lẫn nhãn hành vi chồng lên cùng một người nên chỉ đếm 'person'
    # cho số người; 'invigilator' (giám thị) không tính là thí sinh.
    'person_coco': {'person': {'person'}, 'phone': {'cell phone'}},
    'gianlan': {'person': {'person', 'normal', 'cheating', 'cheating-behaviour', 'submitting'},
                'count_person': {'person'}, 'phone': {'cell-phones'}},
}


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
                    'duration_s': round(now - track.since, 1),
                    'status': NEEDS_REVIEW,
                    'note': 'Tín hiệu cần giám thị xem xét; không phải kết luận gian lận.',
                })
        return fired

