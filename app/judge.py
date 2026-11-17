"""Kết luận theo THỜI GIAN cho từng người: Bình thường / Nghi ngờ / Gian lận.

Một khung hình không đủ để kết luận gian lận (người quay đầu 1 giây chưa là gian lận). `Judge` theo dõi từng người qua các khung
(ghép hộp theo IoU), lưu "mức bằng chứng" của từng khung trong một cửa sổ trượt rồi kết luận:

* Gian lận  : điểm cửa sổ >= ngưỡng gian lận VÀ tín hiệu kéo dài >= thời gian tối thiểu (mặc định 3 giây).
* Nghi ngờ  : điểm cửa sổ >= ngưỡng nghi ngờ, hoặc khung hiện tại có nhiều nguồn cùng nghi nhưng chưa đủ thời gian.
* Bình thường: còn lại.

Điểm khung: nhiều nguồn đồng thuận = 1.0; một nguồn mạnh (một mô hình hoặc điện thoại) = WATCH_WEIGHT; không có = 0.
Vì vậy "chỉ điện thoại" hoặc "chỉ quay đầu" kéo dài cùng lắm là Nghi ngờ; Gian lận cần nhiều nguồn đồng thuận kéo dài.

Giới hạn trung thực: các ngưỡng mặc định là heuristic, CHƯA được hiệu chỉnh trên video thi cử có nhãn (dự án không có dữ liệu đó).
Kết luận là gợi ý tự động cho giám thị, không dùng để xử phạt.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

LABEL_CHEAT, LABEL_SUSPECT, LABEL_NORMAL = 'Gian lận', 'Nghi ngờ', 'Bình thường'
TIER_SUSPECT, TIER_WATCH, TIER_NORMAL = 'suspect', 'watch', 'normal'  # kết quả của ensemble.decide (một khung)
WATCH_WEIGHT = 0.3
WINDOW_S = 8.0       # độ dài cửa sổ trượt
TRACK_TTL_S = 2.0    # mất dấu quá lâu thì bỏ track
HOLD_S = 1.0         # đã kết luận Gian lận thì giữ nhãn tối thiểu ngần này giây (tránh nhấp nháy)
MATCH_IOU = 0.25
GAP_S = 1.0
MIN_SPAN_S = 1.0     # cần quan sát ít nhất ngần này giây mới dựa vào điểm cửa sổ để nghi ngờ


@dataclass
class Settings:
    """Mọi ngưỡng người dùng chỉnh được (webcam và video dùng chung)."""
    person_conf: float = 0.35     # ngưỡng phát hiện người
    phone_conf: float = 0.35      # ngưỡng điện thoại
    evidence_conf: float = 0.55   # một nguồn hành vi chỉ được tính là "nghi" khi bằng chứng >= mức này
    suspect_score: float = 0.15   # điểm cửa sổ tối thiểu để "Nghi ngờ"
    cheat_score: float = 0.45     # điểm cửa sổ tối thiểu để "Gian lận"
    min_seconds: float = 1.5      # tín hiệu phải kéo dài ngần này giây mới kết luận "Gian lận"
    suspect_seconds: float = 1.0  # bằng chứng liên tục ngần này giây thì "Nghi ngờ" (cái liếc ~1 s vẫn được ghi nhận)
    recover_seconds: float = 1.0  # hết bằng chứng ngần này giây thì về "Bình thường" ngay (xóa lịch sử cũ)

    RANGES = {'person_conf': (0.10, 0.90), 'phone_conf': (0.10, 0.90), 'evidence_conf': (0.20, 0.95),
              'suspect_score': (0.02, 0.90), 'cheat_score': (0.10, 1.00), 'min_seconds': (0.0, 30.0), 'recover_seconds': (0.5, 10.0), 'suspect_seconds': (0.3, 10.0)}

    @classmethod
    def from_dict(cls, raw) -> 'Settings':
        """Đọc từ JSON của client; giá trị thiếu dùng mặc định, ngoài khoảng thì bị kẹp, sai kiểu thì báo lỗi."""
        s = cls()
        for f in fields(cls):
            if raw and raw.get(f.name) is not None:
                try:
                    v = float(raw[f.name])
                except (TypeError, ValueError):
                    raise ValueError(f'Ngưỡng "{f.name}" không hợp lệ.')
                lo, hi = cls.RANGES[f.name]
                setattr(s, f.name, min(hi, max(lo, v)))
        if s.cheat_score < s.suspect_score:  # ngưỡng gian lận không được thấp hơn ngưỡng nghi ngờ
            s.cheat_score = s.suspect_score
        return s

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


class _Track:
    def __init__(self, tid, box, t):
        self.id, self.box, self.last, self.first = tid, box, t, t
        self.hist: list[tuple[float, float]] = []   # (thời điểm, điểm khung)
        self.cheat_until = float('-inf')
        self.streak = 0   # số khung liên tiếp nhiều nguồn cùng nghi


class Judge:
    """Một đối tượng cho mỗi phiên webcam / mỗi video. `now` là giờ thực (webcam) hoặc thời điểm trong video (video)."""

    def __init__(self, min_frames=2):
        # min_frames: số khung liên tiếp nhiều nguồn cùng nghi mới tính 'Nghi ngờ' (chống báo nhầm do một khung nhiễu).
        # Ảnh đơn lẻ (không có khung sau) dùng min_frames=1.
        self.min_frames = min_frames
        self._tracks: list[_Track] = []
        self._next = 1

    def _match(self, people, t):
        self._tracks = [k for k in self._tracks if t - k.last <= TRACK_TTL_S or t < k.last]  # t < last: video tua lại
        pairs = sorted(((_iou(p['box'], k.box), i, j) for i, p in enumerate(people) for j, k in enumerate(self._tracks)), reverse=True)
        used_p, used_t, assign = set(), set(), {}
        for score, i, j in pairs:
            if score < MATCH_IOU:
                break
            if i in used_p or j in used_t:
                continue
            used_p.add(i); used_t.add(j); assign[i] = self._tracks[j]
        for i, p in enumerate(people):
            if i not in assign:
                k = _Track(self._next, p['box'], t)
                self._next += 1
                self._tracks.append(k)
                assign[i] = k
        return assign

    def apply(self, people: list[dict], now: float, cfg: Settings) -> list[dict]:
        """Ghi đè `label` bằng kết luận; giữ nhãn khung ở `frame_tier`. Thêm id, window_score, observed_s."""
        assign = self._match(people, now)
        for i, p in enumerate(people):
            k = assign[i]
            if now < k.last:  # video bị tua lại: bỏ lịch sử cũ
                k.hist.clear(); k.first = now; k.cheat_until = float('-inf'); k.streak = 0
            k.box, k.last = p['box'], now
            tier = p.get('tier', TIER_NORMAL)
            w = 1.0 if tier == TIER_SUSPECT else WATCH_WEIGHT if tier == TIER_WATCH else 0.0
            k.streak = k.streak + 1 if tier == TIER_SUSPECT else 0
            k.hist.append((now, w))
            k.hist = [(ts, v) for ts, v in k.hist if now - ts <= WINDOW_S]
            span = now - k.hist[0][0]
            mean = sum(v for _, v in k.hist) / len(k.hist)
            # thời gian tín hiệu "đang bật" liên tục gần nhất (khung có bằng chứng): từ khung bằng chứng đầu tiên của chuỗi hiện tại
            # (khoảng trống <= GAP_S không ngắt chuỗi vì mô hình có thể bỏ sót vài khung)
            active_since = last_on = None
            for ts, v in k.hist:
                if v > 0:
                    if active_since is None or ts - last_on > GAP_S:
                        active_since = ts
                    last_on = ts
            sustained = (now - active_since) if active_since is not None and now - last_on <= GAP_S else 0.0
            quiet = now - last_on if last_on is not None else float('inf')
            if quiet > cfg.recover_seconds:
                # Không còn bằng chứng: về Bình thường NGAY (không đợi điểm trung bình của cửa sổ tụt dần) và quên lịch sử cũ
                k.hist = [(now, w)]; k.cheat_until = float('-inf')
                mean, span, label = 0.0, 0.0, LABEL_NORMAL
            elif mean >= cfg.cheat_score and sustained >= cfg.min_seconds and span >= cfg.min_seconds:
                label = LABEL_CHEAT
                k.cheat_until = now + HOLD_S
            elif now < k.cheat_until:
                label = LABEL_CHEAT  # giữ nhãn ngắn để không nhấp nháy
            elif (sustained >= cfg.suspect_seconds and sum(1 for _, v in k.hist if v > 0) >= max(2, self.min_frames)) or (mean >= cfg.suspect_score and span >= MIN_SPAN_S and sum(1 for _, v in k.hist if v > 0) >= self.min_frames) or (tier == TIER_SUSPECT and k.streak >= self.min_frames):  # cần nhiều nguồn đồng thuận trong >= min_frames khung liên tiếp
                label = LABEL_SUSPECT
            else:
                label = LABEL_NORMAL
            p['frame_tier'] = tier
            p['id'] = k.id
            p['window_score'] = round(mean, 3)
            p['observed_s'] = round(span, 1)
            p['label'] = label
            p['confidence'] = round(1 - mean if label == LABEL_NORMAL else max(mean, p.get('score', 0.0)), 3)
        return people
