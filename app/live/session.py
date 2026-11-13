"""Phiên giám sát trực tiếp: tối đa một phiên hoạt động, lưu cảnh báo trong RAM.

Không lưu hình ảnh/video nào: khung hình chỉ tồn tại trong bộ nhớ khi suy luận.
"""
from __future__ import annotations

import secrets
import threading
import time

from .alerts import AlertEngine, REVIEW_STATUSES


class SessionBusy(RuntimeError):
    pass


class SessionNotFound(KeyError):
    pass


class LiveSession:
    def __init__(self, model_key, model_name, confidence, clock=time.monotonic, mode='single', imgsz=None):
        self.id = secrets.token_urlsafe(12)
        self.model_key, self.model_name, self.confidence = model_key, model_name, confidence
        self.mode, self.imgsz = mode, imgsz
        self.engine = AlertEngine(clock=clock, mode=mode)
        self.alerts: list[dict] = []
        self.frames = 0
        self.state = 'running'
        self.clock = clock
        self.last_seen = clock()
        self.lock = threading.Lock()

    def record(self, new_alerts):
        self.frames += 1
        self.alerts.extend(new_alerts)

    def review(self, alert_id, status):
        if status not in REVIEW_STATUSES:
            raise ValueError('Trạng thái không hợp lệ')
        for a in self.alerts:
            if a['id'] == alert_id:
                a['status'] = status
                return a
        raise KeyError(alert_id)


class SessionManager:
    def __init__(self, idle_timeout=15.0, clock=time.monotonic):
        self.idle_timeout, self.clock = idle_timeout, clock
        self._current: LiveSession | None = None
        self._lock = threading.Lock()

    def _expire(self):
        s = self._current
        if s and self.clock() - s.last_seen > self.idle_timeout:
            self._current = None

    def start(self, model_key, model_name, confidence, mode='single', imgsz=None) -> LiveSession:
        with self._lock:
            self._expire()
            if self._current is not None:
                raise SessionBusy('Đã có phiên giám sát khác đang hoạt động.')
            self._current = LiveSession(model_key, model_name, confidence, self.clock, mode, imgsz)
            return self._current

    def get(self, session_id) -> LiveSession:
        with self._lock:
            self._expire()
            s = self._current
            if s is None or s.id != session_id:
                raise SessionNotFound(session_id)
            s.last_seen = self.clock()
            return s

    def stop(self, session_id):
        with self._lock:
            if self._current is not None and self._current.id == session_id:
                self._current = None

    def active(self):
        with self._lock:
            self._expire()
            s = self._current
            return None if s is None else {'id': s.id, 'model': s.model_key, 'state': s.state, 'frames': s.frames}

