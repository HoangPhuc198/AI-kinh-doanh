import base64, json, time, tempfile, os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MODELS = {
    'training': ('Training YOLO', PROJECT/'models/training_yolo.pt'),
    'gianlan': ('GIan lan', PROJECT/'models/gian_lan.pt'),
    'yolo_duyen': ('YOLO11 Duyên', PROJECT/'models/yolo11_duyen.pt'),
    'paper': ('KNN bài báo', PROJECT/'models/yolo11n-pose.pt'),
    'duyen': ('KNN dataset Duyên', PROJECT/'models/duyen_knn.joblib'),
}
cache = {}

def run_model(key, frame, confidence, include_image=True):
    started = time.perf_counter()
    if key in ('paper', 'duyen'):
        import paper_detector
        classifier_path = str(MODELS['duyen'][1]) if key == 'duyen' else None
        detections = paper_detector.infer(frame, threshold=confidence, classifier_path=classifier_path)
        rendered, counts = paper_detector.draw(frame.copy(), detections)
        h, w = frame.shape[:2]
        details = [{'label': 'Suspicious' if pred == 1 else 'Normal', 'confidence': None, 'box': [float(box[0]/w), float(box[1]/h), float(box[2]/w), float(box[3]/h)], 'keypoints': np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0).tolist()} for box, pts, pred in detections]
    else:
        from ultralytics import YOLO
        if key not in cache: cache[key] = YOLO(str(MODELS[key][1]))
        result = cache[key].predict(frame, conf=confidence, verbose=False, device='cpu')[0]
        rendered = result.plot()
        details = [{'label': str(result.names[int(b.cls.item())]), 'confidence': round(float(b.conf.item()), 3), 'box': [float(v) for v in b.xyxyn[0].tolist()]} for b in result.boxes]
        counts = {}
        for d in details: counts[d['label']] = counts.get(d['label'], 0) + 1
    elapsed = round((time.perf_counter()-started)*1000)
    output = {'details': details, 'counts': counts, 'ms': elapsed}
    if include_image:
        ok, encoded = cv2.imencode('.jpg', rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok: raise ValueError('Không thể tạo ảnh kết quả')
        output['image'] = base64.b64encode(encoded).decode()
    return output

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

class Handler(BaseHTTPRequestHandler):
    def respond(self, status, body, typ='application/json; charset=utf-8'):
        if isinstance(body, dict): body=json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status); self.send_header('Content-Type',typ); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if urlparse(self.path).path == '/': self.respond(200,(ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
        elif urlparse(self.path).path == '/bao-cao': self.respond(200,(ROOT/'report.html').read_bytes(),'text/html; charset=utf-8')
        elif urlparse(self.path).path == '/api/report': self.respond(200,report_data())
        else: self.respond(404,{'error':'Không tìm thấy'})
    def do_POST(self):
        if self.path != '/analyze': return self.respond(404,{'error':'Không tìm thấy'})
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0 < length <= 110_000_000: raise ValueError('Tệp quá lớn (tối đa khoảng 75 MB).')
            req=json.loads(self.rfile.read(length)); raw=base64.b64decode(req['data'],validate=True)
            keys=[k for k in req['models'] if k in MODELS]
            if not keys: raise ValueError('Chọn ít nhất một mô hình.')
            threshold=max(0.05,min(0.95,float(req.get('confidence',.25))))
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
                except Exception as e: output[key]={'name':MODELS[key][0], 'error':str(e)}
            self.respond(200,{'results':output, 'video':bool(req.get('video'))})
        except Exception as e: self.respond(400,{'error':str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('EXAMVISION_PORT', '8765'))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Mở http://127.0.0.1:{port}', flush=True)
    server.serve_forever()
