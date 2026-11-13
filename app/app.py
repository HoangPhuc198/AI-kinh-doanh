import base64, json, tempfile, os, sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import cv2
import numpy as np
import runtime
from inference import ALLOWED_IMGSZ, MODELS, PROJECT, required_files, run_model, warm_model
from live.alerts import CALIBRATION, display_filter, extract_signals
from live.session import SessionManager, SessionBusy, SessionNotFound

ROOT = Path(__file__).resolve().parent
sessions = SessionManager()
MAX_LIVE_FRAME = 4_000_000

def report_data():
    def read_json(path):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
    metrics = read_json(PROJECT/'reports/knn_duyen/metrics.json')
    videos = read_json(PROJECT/'reports/knn_duyen/video_evaluation.json')
    snapshots = read_json(PROJECT/'reports/snapshots.json') or {}
    models = []
    for key, (name, path) in MODELS.items():
        dependencies = [path]
        if key in ('paper', 'duyen'):
            dependencies.append(PROJECT/'models/yolo11n-pose.pt')
        if key == 'paper':
            dependencies.append(PROJECT/'models/paper_knn.pkl')
        models.append({'key': key, 'name': name, 'available': all(p.is_file() for p in dependencies),
                       'kind': 'YOLO11-Pose + KNN' if key in ('paper', 'duyen') else 'YOLO detection',
                       'path': str(path.relative_to(PROJECT))})
    return {'models': models, 'duyen': metrics, 'video_evaluation': videos,
            'yolo_duyen_test': read_json(PROJECT/'reports/yolo11_duyen/test_metrics.json'),
            'yolo_duyen_run': read_json(PROJECT/'reports/yolo11_duyen/run_config.json'),
            'yolo_duyen_dataset': read_json(PROJECT/'reports/yolo11_duyen/dataset_summary.json'),
            'snapshots': snapshots}

LIVE_MODELS = ('ensemble', 'person_phone', 'person_coco', 'yolo_duyen', 'training', 'gianlan', 'paper', 'duyen')

def live_status():
    return {'torch': runtime.torch_status(), 'loaded_models': runtime.loaded_models(),
            'models': {k: MODELS[k][0] for k in LIVE_MODELS}, 'calibrated': sorted(set(CALIBRATION) | {'ensemble'}), 'session': sessions.active()}

def live_start(req):
    key = req.get('model')
    if key not in LIVE_MODELS: raise ValueError('Mô hình không hợp lệ.')
    if not all(p.is_file() for p in required_files(key)):
        raise FileNotFoundError('Thiếu tệp trọng số của mô hình này.')
    confidence = max(0.01, min(1.0, float(req.get("confidence", .5))))
    mode = req.get('mode', 'single')
    if mode not in ('single', 'room'): raise ValueError('Chế độ không hợp lệ.')
    imgsz = int(req['imgsz']) if req.get('imgsz') else None
    if imgsz is not None and imgsz not in ALLOWED_IMGSZ: raise ValueError('imgsz không hợp lệ.')
    session = sessions.start(key, MODELS[key][0], confidence, mode, imgsz)  # 409 nếu đã có phiên khác
    try:
        warm_model(key)
    except BaseException:
        sessions.stop(session.id); raise
    return 200, {'session': session.id, 'model': MODELS[key][0], 'supports_phone': extract_signals(key, [])['supports_phone'], 'mode': mode, 'calibrated': key in CALIBRATION or key == 'ensemble'}

def live_frame(req):
    session = sessions.get(req.get('session'))
    if session.state != 'running': return 409, {'error': 'Phiên đang tạm dừng.'}
    raw = base64.b64decode(req['data'], validate=True)
    if len(raw) > MAX_LIVE_FRAME: raise ValueError('Khung hình quá lớn.')
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None: raise ValueError('Không giải mã được khung hình.')
    cal = CALIBRATION.get(session.model_key)
    result = run_model(session.model_key, frame, cal['infer_conf'] if cal else session.confidence, include_image=False, imgsz=session.imgsz, **({'state': session.ens_state} if session.model_key == 'ensemble' else {}))
    with session.lock:
        new = session.engine.update(extract_signals(session.model_key, result['details']), session.model_name)
        session.record(new)
        total = len(session.alerts)
    shown = display_filter(session.model_key, result['details'])
    shown_counts = {}
    for d in shown: shown_counts[d['label']] = shown_counts.get(d['label'], 0) + 1
    return 200, {'details': shown, 'counts': shown_counts, 'ms': result['ms'],
                 'new_alerts': new, 'alert_total': total, 'frames': session.frames}

def live_control(req):
    action = req.get('action')
    if action == 'stop':
        sessions.stop(req.get('session')); return 200, {'ok': True}
    session = sessions.get(req.get('session'))
    if action in ('pause', 'resume'):
        with session.lock:
            session.state = 'paused' if action == 'pause' else 'running'
            session.engine.reset_streaks()
    elif action != 'ping':
        raise ValueError('Hành động không hợp lệ.')
    return 200, {'state': session.state}

def live_review(req):
    session = sessions.get(req.get('session'))
    with session.lock:
        return 200, session.review(int(req['alert']), req.get('status'))

class Handler(BaseHTTPRequestHandler):
    def respond(self, status, body, typ='application/json; charset=utf-8'):
        if isinstance(body, dict): body=json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status); self.send_header('Content-Type',typ); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    def live_post(self, fn):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 8_000_000: raise ValueError('Nội dung không hợp lệ.')
            status, body = fn(json.loads(self.rfile.read(length)))
            self.respond(status, body)
        except SessionBusy as e: self.respond(409, {'error': str(e)})
        except SessionNotFound: self.respond(404, {'error': 'Phiên không tồn tại hoặc đã hết hạn.'})
        except runtime.TorchUnavailable as e: self.respond(503, {'error': str(e)})
        except Exception as e: self.respond(400, {'error': str(e)})
    def live_alerts(self):
        try:
            session = sessions.get(parse_qs(urlparse(self.path).query).get('session', [''])[0])
            with session.lock: self.respond(200, {'alerts': list(session.alerts), 'frames': session.frames})
        except SessionNotFound: self.respond(404, {'error': 'Phiên không tồn tại hoặc đã hết hạn.'})
    def do_GET(self):
        if urlparse(self.path).path == '/': self.respond(200,(ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
        elif urlparse(self.path).path == '/bao-cao': self.respond(200,(ROOT/'report.html').read_bytes(),'text/html; charset=utf-8')
        elif urlparse(self.path).path == '/giam-sat': self.respond(200,(ROOT/'live.html').read_bytes(),'text/html; charset=utf-8')
        elif urlparse(self.path).path == '/api/report': self.respond(200,report_data())
        elif urlparse(self.path).path == '/api/live/status': self.respond(200,live_status())
        elif urlparse(self.path).path == '/api/live/alerts': self.live_alerts()
        else: self.respond(404,{'error':'Không tìm thấy'})
    def do_POST(self):
        live_routes={'/api/live/start':live_start,'/api/live/frame':live_frame,'/api/live/control':live_control,'/api/live/review':live_review}
        if self.path in live_routes: return self.live_post(live_routes[self.path])
        if self.path != '/analyze': return self.respond(404,{'error':'Không tìm thấy'})
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0 < length <= 110_000_000: raise ValueError('Tệp quá lớn (tối đa khoảng 75 MB).')
            req=json.loads(self.rfile.read(length)); raw=base64.b64decode(req['data'],validate=True)
            keys=[k for k in req['models'] if k in MODELS]
            if not keys: raise ValueError('Chọn ít nhất một mô hình.')
            threshold=max(0.01,min(1.0,float(req.get("confidence",.25))))
            frames=[]
            if req.get('video'):
                with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
                    f.write(raw); f.flush(); cap=cv2.VideoCapture(f.name)
                    if not cap.isOpened(): raise ValueError('Không mở được video.')
                    fps=cap.get(cv2.CAP_PROP_FPS) or 25; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); n=min(12,max(1,int(req.get('samples',6))))
                    indices=np.linspace(0,max(0,total-1),n,dtype=int) if total else [0]
                    for i in sorted(set(indices)):
                        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,frame=cap.read()
                        if ok: frames.append((round(i/fps,2),frame))
                    cap.release()
            else:
                frame=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
                if frame is not None: frames=[(0,frame)]
            if not frames: raise ValueError('Không đọc được ảnh hoặc khung hình video.')
            output={}
            for key in keys:
                try:
                    items=[]
                    for timestamp, frame in frames:
                        item=run_model(key,frame,threshold,not req.get('no_image', False)); item['time']=timestamp; items.append(item)
                    output[key]={'name':MODELS[key][0], 'frames':items}
                except Exception as e:
                    import traceback; traceback.print_exc()
                    output[key]={'name':MODELS[key][0], 'error':str(e)}
            self.respond(200,{'results':output, 'video':bool(req.get('video'))})
        except Exception as e: self.respond(400,{'error':str(e)})

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr): stream.reconfigure(encoding='utf-8', errors='replace')
    # Import torch/ultralytics đúng một lần, ở luồng chính, trước khi nhận request.
    # Nếu lỗi: in lỗi gốc thật và dừng, thay vì để từng request import lại.
    try:
        print('[ExamVision] Đang khởi tạo PyTorch...', flush=True)
        print('[ExamVision] torch', runtime.load_torch().__version__, flush=True)
    except BaseException as exc:
        print(f'[ExamVision] KHÔNG import được PyTorch/Ultralytics: {exc!r}', file=sys.stderr, flush=True)
        raise SystemExit(1)
    port = int(os.environ.get('EXAMVISION_PORT', '8765'))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Mở http://127.0.0.1:{port}', flush=True)
    server.serve_forever()

