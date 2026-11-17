"""Vòng đời PyTorch/Ultralytics và registry mô hình dùng chung cho toàn tiến trình.

Vì sao có module này
--------------------
`torch/__init__.py` đặt cờ `sys._torch_import_started` ngay khi bắt đầu chạy. Nếu lần
`import torch` đầu tiên thất bại (thiếu DLL, thiếu RAM...) hoặc hai luồng cùng import,
lần import tiếp theo sẽ ném "torch can only be initialized once per process" và che mất
lỗi gốc. Máy chủ HTTP đa luồng trước đây import torch/ultralytics lười trong từng
request nên dễ rơi vào tình huống đó.

Quy tắc ở đây:
* torch và ultralytics được import đúng một lần, có khóa, ở luồng chính khi khởi động.
* Nếu lần import đó lỗi, lỗi gốc được lưu lại và ném lại nguyên vẹn; không bao giờ
  import lại, không reload, không xóa `sys.modules`.
* Mỗi trọng số YOLO chỉ nạp một lần cho mỗi tiến trình.
* Suy luận được tuần tự hóa theo từng mô hình (khóa riêng) vì Ultralytics predictor không an toàn đa luồng.
"""
from __future__ import annotations

import os
import threading
import time

_import_lock = threading.Lock()
_torch = None
_YOLO = None
_import_error: BaseException | None = None

_models: dict[str, object] = {}
_models_lock = threading.Lock()

# Predictor của Ultralytics không thread-safe TRÊN CÙNG MỘT mô hình, nhưng các mô hình khác nhau độc lập:
# mỗi mô hình có một khóa riêng để ensemble chạy chúng song song.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def model_lock(name: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(name, threading.Lock())


def set_threads(n: int):
    """Số luồng CPU cho PyTorch (toàn tiến trình). Chạy nhiều mô hình song song thì mỗi mô hình nên dùng ít luồng hơn."""
    torch = load_torch()
    if torch.get_num_threads() != n:
        torch.set_num_threads(n)


class TorchUnavailable(RuntimeError):
    """PyTorch/Ultralytics không import được; `__cause__` giữ lỗi gốc của lần thử đầu."""


def load_torch():
    """Import torch + ultralytics đúng một lần; trả về module torch."""
    global _torch, _YOLO, _import_error
    with _import_lock:
        if _torch is not None:
            return _torch
        if _import_error is not None:
            raise TorchUnavailable(
                f'PyTorch không khả dụng. Lỗi gốc ở lần import đầu: {_import_error!r}'
            ) from _import_error
        try:
            import torch
            from ultralytics import YOLO
        except BaseException as exc:  # ghi lại lỗi gốc rồi ném lại, không nuốt
            _import_error = exc
            raise
        _torch, _YOLO = torch, YOLO
        return _torch


def torch_status() -> dict:
    """Trạng thái không kích hoạt import."""
    return {
        'ready': _torch is not None,
        'error': None if _import_error is None else repr(_import_error),
        'version': getattr(_torch, '__version__', None),
        'cuda': bool(_torch.cuda.is_available()) if _torch is not None else None,
    }


def device():
    """GPU CUDA đầu tiên nếu có, ngược lại CPU."""
    return 0 if load_torch().cuda.is_available() else 'cpu'


def use_half() -> bool:
    """FP16 mặc định TẮT. Đo trên GTX 1650: FP16 cho kết quả SAI (yolo11m recall người 0.013 so với 0.63 ở FP32, precision
    dao động giữa các lần chạy). Dòng GTX 16xx nổi tiếng lỗi FP16. Chỉ bật bằng EXAMVISION_HALF=1 trên GPU đã kiểm chứng."""
    return device() != 'cpu' and os.getenv('EXAMVISION_HALF', '0') == '1'


def get_yolo(path) -> object:
    """Nạp (một lần) và trả về mô hình YOLO cho `path`."""
    load_torch()
    key = str(path)
    with _models_lock:
        model = _models.get(key)
        if model is None:
            started = time.perf_counter()
            model = _YOLO(key)
            _models[key] = model
            print(f'[ExamVision] Đã nạp {key} trong {time.perf_counter() - started:.1f}s', flush=True)
        return model


def loaded_models() -> list[str]:
    with _models_lock:
        return sorted(_models)


def reset_for_tests():
    """Chỉ dùng trong test: xóa cache mô hình (không đụng tới module torch)."""
    with _models_lock:
        _models.clear()
