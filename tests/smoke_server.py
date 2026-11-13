"""Smoke test thủ công với máy chủ và mô hình THẬT (không nằm trong unittest discover).

Chạy:  .venv\\Scripts\\python.exe tests\\smoke_server.py
Khung hình là ảnh tổng hợp (không có người) nên chỉ kiểm tra đường ống, không đo độ chính xác.
"""
import base64
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
PORT = 8791
BASE = f'http://127.0.0.1:{PORT}'


def call(path, body=None, timeout=300):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    data = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    env = dict(os.environ, EXAMVISION_PORT=str(PORT), PYTHONIOENCODING='utf-8')
    proc = subprocess.Popen([sys.executable, str(PROJECT / 'app' / 'app.py')], cwd=str(PROJECT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    log = []
    threading.Thread(target=lambda: [log.append(l) for l in proc.stdout], daemon=True).start()
    try:
        for _ in range(120):
            try:
                st, status = call('/api/live/status', timeout=2); break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit('Máy chủ không khởi động:\n' + ''.join(log))
        print('status:', status['torch'])
        st, r = call('/analyze', {'data': data, 'models': ['yolo_duyen', 'duyen', 'paper', 'training', 'gianlan', 'ensemble'],
                                  'confidence': 0.25, 'video': False, 'no_image': True})
        for k, v in r['results'].items():
            print('analyze', k, v.get('error') or f"{len(v['frames'][0]['details'])} đối tượng, {v['frames'][0]['ms']} ms")
        # Ảnh THẬT có người (coco128) để chắc chắn mô hình thực sự phát hiện trên khung hình, không chỉ chạy được.
        real = sorted((PROJECT / 'datasets/coco128/images/train2017').glob('*.jpg'))
        if real:
            people = next(p for p in real if p.name == '000000000009.jpg') if any(p.name == '000000000009.jpg' for p in real) else real[0]
            rdata = base64.b64encode(people.read_bytes()).decode()
            st, s = call('/api/live/start', {'model': 'person_coco', 'confidence': 0.25, 'mode': 'room', 'imgsz': 960}); assert st == 200, s
            st, f = call('/api/live/frame', {'session': s['session'], 'data': rdata}); assert st == 200, f
            print('real image', people.name, 'counts', f['counts'], 'ms', f['ms'])
            call('/api/live/control', {'session': s['session'], 'action': 'stop'})
        for model in ('yolo_duyen', 'gianlan', 'duyen', 'person_coco', 'ensemble'):
            st, s = call('/api/live/start', {'model': model, 'confidence': 0.5}); assert st == 200, s
            sid = s['session']
            assert call('/api/live/start', {'model': model})[0] == 409
            ms = []
            for _ in range(5):
                st, f = call('/api/live/frame', {'session': sid, 'data': data}); assert st == 200, f
                ms.append(f['ms'])
            print('live', model, 'ms/frame', ms)
            call('/api/live/control', {'session': sid, 'action': 'stop'})
        print('loaded:', call('/api/live/status')[1]['loaded_models'])
    finally:
        proc.terminate()
        proc.wait(timeout=20)
    loads = [l for l in log if 'Đã nạp' in l]
    print('\n'.join(l.rstrip() for l in loads))
    names = [l.split('Đã nạp', 1)[1].rsplit(' trong ', 1)[0] for l in loads]
    assert len(names) == len(set(names)), f'Mô hình bị nạp trùng: {names}'
    print('SMOKE OK: mỗi trọng số nạp đúng một lần')


if __name__ == '__main__':
    main()
