"""Fine-tune YOLO11 phát hiện person / cell phone, có thể chạy lại và resume.

Ví dụ (GPU / Colab):
  python training/train.py --data datasets/mydata/data.yaml --weights models/yolo11n_coco.pt --name person_phone_v1 --epochs 60 --imgsz 960 --batch 8
Resume sau khi bị ngắt:
  python training/train.py --resume runs/train/person_phone_v1/weights/last.pt
Chỉ kiểm tra đường ống (CPU, vài phút):  --epochs 1 --imgsz 320 --batch 4 --workers 0 --fraction 0.25

Mặc định có: seed cố định, early stopping (--patience), checkpoint mỗi epoch (last.pt) + best.pt,
augmentation có kiểm soát (ánh sáng HSV, scale mạnh để học người nhỏ/xa, mosaic, random-erasing mô phỏng che khuất),
và ghi `training_run.json` (cấu hình, seed, phiên bản, thiết bị, thời gian, metrics val/test). Không train vô hạn: epochs có trần.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))
import runtime  # noqa: E402  (import torch/ultralytics đúng một lần)

MAX_EPOCHS = 300


def metrics_of(m):
    return {'precision': round(float(m.box.mp), 4), 'recall': round(float(m.box.mr), 4), 'mAP50': round(float(m.box.map50), 4),
            'mAP50_95': round(float(m.box.map), 4), 'per_class_mAP50': {m.names[int(i)]: round(float(v), 4) for i, v in zip(m.box.ap_class_index, m.box.ap50)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--weights', default='models/yolo11n_coco.pt')
    ap.add_argument('--name', default='examvision_run'); ap.add_argument('--project', default='runs/train')
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--batch', type=int, default=8); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--patience', type=int, default=15); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fraction', type=float, default=1.0); ap.add_argument('--device', default=None)
    ap.add_argument('--resume', default=None, help='đường dẫn last.pt để tiếp tục')
    ap.add_argument('--freeze', type=int, default=None, help='đóng băng N lớp đầu (backbone) khi dữ liệu nhỏ')
    ap.add_argument('--lr0', type=float, default=None); ap.add_argument('--optimizer', default='auto')
    ap.add_argument('--weight-decay', type=float, default=0.0005)
    a = ap.parse_args()
    torch = runtime.load_torch()
    from ultralytics import YOLO, __version__ as uv
    device = a.device if a.device is not None else (0 if torch.cuda.is_available() else 'cpu')
    if device == 'cpu' and a.epochs > 3 and a.fraction == 1.0:
        print('CẢNH BÁO: đang train trên CPU; chỉ nên dùng cho kiểm tra đường ống (--epochs 1). Dùng GPU/Colab cho train thật.', file=sys.stderr)
    if a.resume:
        YOLO(a.resume).train(resume=True)
        return
    if not a.data or a.epochs > MAX_EPOCHS:
        sys.exit(f'Cần --data và epochs <= {MAX_EPOCHS}')
    weights = ROOT / a.weights if not Path(a.weights).is_absolute() else Path(a.weights)
    started = time.time()
    model = YOLO(str(weights))
    cfg = dict(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, patience=a.patience, seed=a.seed, deterministic=True,
               project=str(ROOT / a.project), name=a.name, exist_ok=False, device=device, fraction=a.fraction, save_period=-1, plots=False,
               # augmentation có kiểm soát: ánh sáng, scale (người nhỏ/xa), che khuất; không lật dọc/xoay để giữ ngữ cảnh phòng thi
               hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, scale=0.6, translate=0.1, degrees=0.0, fliplr=0.5, flipud=0.0, mosaic=1.0, erasing=0.3, close_mosaic=5)
    cfg.update(optimizer=a.optimizer, weight_decay=a.weight_decay)
    if a.freeze:
        cfg['freeze'] = a.freeze
    if a.lr0:
        cfg['lr0'] = a.lr0
    model.train(**cfg)
    best = Path(model.trainer.best)
    final = YOLO(str(best))
    out = {'config': {k: v for k, v in cfg.items() if k != 'project'}, 'base_weights': a.weights, 'best_weights': str(best),
           'versions': {'torch': torch.__version__, 'ultralytics': uv, 'python': platform.python_version()}, 'device': str(device),
           'completed_epochs': int(getattr(model.trainer, 'epoch', -1)) + 1, 'wall_seconds': round(time.time() - started),
           'val': metrics_of(final.val(data=a.data, split='val', imgsz=a.imgsz, device=device, plots=False))}
    try:
        out['test'] = metrics_of(final.val(data=a.data, split='test', imgsz=a.imgsz, device=device, plots=False))
    except Exception as exc:  # dataset không có tập test
        out['test'] = f'không đánh giá được: {exc}'
    (best.parent.parent / 'training_run.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
