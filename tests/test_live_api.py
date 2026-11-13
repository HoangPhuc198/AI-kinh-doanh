import base64
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

from tests import _paths  # noqa: F401
import app as server
from live.session import SessionManager


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


JPEG = base64.b64encode(cv2.imencode('.jpg', np.zeros((48, 64, 3), np.uint8))[1]).decode()


class LiveApi(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.orig = (server.sessions, server.run_model, server.warm_model)
        server.sessions = SessionManager(idle_timeout=15, clock=self.clock)
        self.calls = {'warm': 0, 'infer': 0}

        def warm(key):
            self.calls['warm'] += 1

        def infer(key, frame, conf, include_image=True, imgsz=None, iou=None):
            self.calls['infer'] += 1
            self.calls['imgsz'] = imgsz
            assert include_image is False
            return {'details': [], 'counts': {}, 'ms': 5}

        server.warm_model, server.run_model = warm, infer
        self.srv = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f'http://127.0.0.1:{self.srv.server_address[1]}'

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        server.sessions, server.run_model, server.warm_model = self.orig

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_room_mode_and_imgsz_validation(self):
        self.assertEqual(self.call('/api/live/start', {'model': 'person_coco', 'mode': 'bogus'})[0], 400)
        self.assertEqual(self.call('/api/live/start', {'model': 'person_coco', 'imgsz': 123})[0], 400)
        st, s = self.call('/api/live/start', {'model': 'person_coco', 'mode': 'room', 'imgsz': 1280})
        self.assertEqual((st, s['mode']), (200, 'room'))
        for _ in range(10):  # khung rỗng ở chế độ phòng không được sinh cảnh báo "không thấy người"
            st, r = self.call('/api/live/frame', {'session': s['session'], 'data': JPEG})
            self.assertEqual((st, r['new_alerts']), (200, []))
            self.clock.t += 1
        self.assertEqual(self.calls['imgsz'], 1280)

    def test_full_session_flow(self):
        st, s = self.call('/api/live/start', {'model': 'yolo_duyen'})
        self.assertEqual(st, 200)
        sid = s['session']
        self.assertEqual(self.call('/api/live/start', {'model': 'yolo_duyen'})[0], 409)  # một phiên duy nhất
        alerts = []
        for _ in range(8):
            st, r = self.call('/api/live/frame', {'session': sid, 'data': JPEG})
            self.assertEqual(st, 200)
            alerts += r['new_alerts']
            self.clock.t += 1
        self.assertEqual([a['type'] for a in alerts], ['no_person'])
        self.assertEqual(self.calls['warm'], 1)
        st, hist = self.call(f'/api/live/alerts?session={sid}')
        self.assertEqual(len(hist['alerts']), 1)
        st, a = self.call('/api/live/review', {'session': sid, 'alert': alerts[0]['id'], 'status': 'dismissed'})
        self.assertEqual(a['status'], 'dismissed')
        self.call('/api/live/control', {'session': sid, 'action': 'pause'})
        self.assertEqual(self.call('/api/live/frame', {'session': sid, 'data': JPEG})[0], 409)
        self.call('/api/live/control', {'session': sid, 'action': 'resume'})
        self.call('/api/live/control', {'session': sid, 'action': 'stop'})
        self.assertEqual(self.call('/api/live/frame', {'session': sid, 'data': JPEG})[0], 404)
        self.assertEqual(self.call('/api/live/start', {'model': 'yolo_duyen'})[0], 200)  # đã giải phóng

    def test_stale_session_expires_so_new_tab_can_start(self):
        self.call('/api/live/start', {'model': 'yolo_duyen'})
        self.clock.t += 16
        self.assertEqual(self.call('/api/live/start', {'model': 'yolo_duyen'})[0], 200)

    def test_bad_inputs(self):
        self.assertEqual(self.call('/api/live/start', {'model': 'nope'})[0], 400)
        self.assertEqual(self.call('/api/live/frame', {'session': 'x', 'data': JPEG})[0], 404)
        st, s = self.call('/api/live/start', {'model': 'yolo_duyen'})
        self.assertEqual(self.call('/api/live/frame', {'session': s['session'], 'data': 'AAAA'})[0], 400)

    def test_live_page_served(self):
        with urllib.request.urlopen(self.base + '/giam-sat') as r:
            self.assertIn('Giám sát webcam', r.read().decode())


if __name__ == '__main__':
    unittest.main()
