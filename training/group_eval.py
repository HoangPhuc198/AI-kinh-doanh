"""Đánh giá THEO NHÓM: phần logic thuần (không import torch, có unit test trong tests/test_group_eval.py).

Vì sao cần: các khung/ảnh cùng một video (hoặc cùng một ảnh gốc đã tăng cường) giống nhau nên KHÔNG độc lập. Tính khoảng tin cậy
như thể từng ảnh độc lập sẽ cho khoảng hẹp giả tạo, còn chia train/test theo ảnh thì rò rỉ (KNN Duyên: F1 0,79 khi chia ngẫu nhiên,
0,27 khi bỏ riêng từng video - docs/reports/MODEL_EVALUATION.md). Ở đây:

* gom ảnh thành NHÓM (video, hoặc ảnh gốc + bản gần trùng); khoảng tin cậy lấy mẫu lại CẢ NHÓM (cluster bootstrap);
* báo cáo rò rỉ giữa các split.

Ba bộ Roboflow của dự án KHÔNG phải video: tên file là `<tên ảnh gốc>_jpg.rf.<hash>.jpg` (ảnh stock/web/điện thoại + bản tăng cường).
Nên nhóm = hợp nhất khi (cùng tên ảnh gốc) HOẶC (dHash gần nhau, kể cả lật ngang). Gom thừa chỉ làm khoảng tin cậy rộng hơn (an toàn);
gom thiếu mới làm hẹp giả tạo. Với dữ liệu quay thật có mã video trong tên file, dùng `group_by='regex'`.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import cv2
import numpy as np

RF_SUFFIX = re.compile(r'\.rf\.[0-9a-fA-F]{32}$')
SPLITS = ('train', 'valid', 'test')


# ---------- tên file / nhóm ----------
def source_stem(filename: str) -> str:
    """'abc_jpg.rf.<32 hex>.jpg' -> 'abc_jpg' (bỏ đuôi Roboflow và phần mở rộng)."""
    stem = filename.rsplit('.', 1)[0] if '.' in filename else filename
    return RF_SUFFIX.sub('', stem)


def dhash64(img, flip: bool = False) -> int:
    """dHash 64 bit (ảnh thu về 9x8 xám, so sánh điểm kề nhau theo hàng). flip=True: băm ảnh đã lật ngang."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    if flip:
        s = s[:, ::-1]
    bits = (s[:, 1:] > s[:, :-1]).astype(np.uint8).ravel()
    return int(np.frombuffer(np.packbits(bits).tobytes(), dtype='>u8')[0])


_POP8 = np.array([bin(i).count('1') for i in range(256)], np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, 'bitwise_count'):  # numpy >= 2.0
        return np.bitwise_count(x)
    return _POP8[np.ascontiguousarray(x).view(np.uint8).reshape(x.shape + (8,))].sum(-1)


def near_duplicate_pairs(h, hf, threshold: int):
    """Mọi cặp (i<j, khoảng cách) có Hamming <= threshold, tính cả bản lật ngang. h/hf: dHash thường/lật của từng ảnh."""
    h, hf = np.asarray(h, np.uint64), np.asarray(hf, np.uint64)
    out = []
    for s in range(0, len(h), 512):
        a, af = h[s:s + 512, None], hf[s:s + 512, None]
        d = np.minimum(_popcount(a ^ h[None, :]), np.minimum(_popcount(a ^ hf[None, :]), _popcount(af ^ h[None, :])))
        for i, j in zip(*np.nonzero(d <= threshold)):
            if i + s < j:
                out.append((int(i + s), int(j), int(d[i, j])))
    return out


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_groups(keys, pairs=(), by_key: bool = True):
    """keys[i]: khóa nhóm sơ bộ của ảnh i (tên ảnh gốc, hoặc mã video). pairs: cặp (i, j, _) gần trùng. Trả về nhóm (0..G-1) của từng ảnh."""
    dsu, first = _DSU(len(keys)), {}
    if by_key:
        for i, k in enumerate(keys):
            dsu.union(first.setdefault(k, i), i)
    for i, j, _ in pairs:
        dsu.union(i, j)
    ids = {}
    return [ids.setdefault(dsu.find(i), len(ids)) for i in range(len(keys))]


def video_key(filename: str, pattern: str) -> str:
    """Mã video lấy từ tên file bằng regex (nhóm bắt đầu tiên); không khớp thì dùng tên ảnh gốc."""
    m = re.search(pattern, filename)
    return m.group(1) if m and m.groups() else source_stem(filename)


def leakage_report(stems, splits, pairs, train: str = 'train'):
    """Mức rò rỉ giữa các split. 'xác nhận' = cặp ảnh khác split có dHash gần nhau (cận dưới: bản tăng cường mạnh có thể không bị bắt).
    'cùng tên' = ảnh gốc cùng tên xuất hiện ở nhiều split (cận trên: gồm cả trùng tên chung chung như '11_JPG' của ảnh khác nhau)."""
    n_split = Counter(splits)
    leaked = defaultdict(set)
    confirmed_stems = set()
    for i, j, _ in pairs:
        if splits[i] != splits[j]:
            if stems[i] == stems[j]:
                confirmed_stems.add(stems[i])
            if train in (splits[i], splits[j]):
                other = j if splits[i] == train else i
                leaked[splits[other]].add(other)
    stem_splits = defaultdict(set)
    for s, sp in zip(stems, splits):
        stem_splits[s].add(sp)
    cross = {s for s, v in stem_splits.items() if len(v) > 1}
    return {
        'images_per_split': dict(n_split),
        'images_with_confirmed_near_duplicate_in_train': {sp: len(v) for sp, v in leaked.items()},
        'fraction_with_confirmed_near_duplicate_in_train': {sp: round(len(v) / n_split[sp], 3) for sp, v in leaked.items()},
        'source_names_in_multiple_splits': len(cross),
        'of_which_confirmed_same_image': len(cross & confirmed_stems),
        'of_which_name_only_probably_different_images': len(cross - confirmed_stems),
        'total_source_names': len(stem_splits),
    }


# ---------- nhãn YOLO / ghép khung ----------
def read_yolo_labels(path):
    out = []
    for ln in path.read_text(encoding='utf-8').splitlines():
        p = ln.split()
        if len(p) >= 5:
            cx, cy, w, h = map(float, p[1:5])
            out.append((int(p[0]), [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return out


def iou(a, b):
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match_box(gt_box, boxes, iou_min: float):
    """Chỉ số hộp có IoU cao nhất với gt_box và >= iou_min; None nếu không có."""
    best, bi = 0.0, None
    for j, b in enumerate(boxes):
        v = iou(gt_box, b)
        if v > best:
            best, bi = v, j
    return bi if bi is not None and best >= iou_min else None


# ---------- chỉ số + khoảng tin cậy theo nhóm ----------
def metrics_from_counts(c):
    """c[..., 4] = (tp, fp, fn, tn) -> dict mảng: precision, recall, false_alarm_rate, f1 (nan nếu mẫu số 0)."""
    c = np.asarray(c, float)
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]

    def div(a, b):
        return np.divide(a, b, out=np.full(np.shape(a), np.nan), where=np.asarray(b) > 0)
    p, r, far = div(tp, tp + fp), div(tp, tp + fn), div(fp, fp + tn)
    # youden_j = recall - false_alarm_rate: 0 = ngang ngẫu nhiên, 1 = hoàn hảo; không phụ thuộc tỉ lệ ca gian lận trong tập
    return {'precision': p, 'recall': r, 'false_alarm_rate': far, 'f1': div(2 * tp, 2 * tp + fp + fn), 'youden_j': r - far}


def wilson(k, n, z: float = 1.96):
    """Khoảng Wilson cho tỉ lệ k/n nếu các mẫu ĐỘC LẬP (để so với khoảng theo nhóm)."""
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0.0, mid - half)), 3), round(float(min(1.0, mid + half)), 3)]


def group_counts(groups, truth, pred):
    """Đếm (tp, fp, fn, tn) theo từng nhóm CÓ đơn vị đánh giá. Trả về mảng (G, 4)."""
    groups, truth, pred = np.asarray(groups), np.asarray(truth, bool), np.asarray(pred, bool)
    _, inv = np.unique(groups, return_inverse=True)
    g = inv.max() + 1 if len(inv) else 0
    cols = [(truth & pred), (~truth & pred), (truth & ~pred), (~truth & ~pred)]
    return np.stack([np.bincount(inv, weights=c, minlength=g) for c in cols], axis=1)


def summarise(groups, truth, pred, boots: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Chỉ số gộp + khoảng tin cậy lấy mẫu lại cả nhóm (percentile) + khoảng 'giả sử độc lập' và tỉ lệ độ rộng (design effect)."""
    counts = group_counts(groups, truth, pred)
    total = counts.sum(0) if len(counts) else np.zeros(4)
    tp, fp, fn, tn = (int(x) for x in total)
    point = {k: (None if np.isnan(v) else round(float(v), 3)) for k, v in metrics_from_counts(total).items()}
    out = {'units': tp + fp + fn + tn, 'groups': int(len(counts)), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, **point, 'ci_group_bootstrap': {}, 'ci_if_independent': {}, 'width_ratio': {}}
    if len(counts) < 2 or boots <= 0:  # một nhóm: không lấy mẫu lại được; boots=0: chỉ cần số điểm (bảng phụ)
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(boots, len(counts)))
    m = metrics_from_counts(counts[idx].sum(1))
    naive = {'precision': wilson(tp, tp + fp), 'recall': wilson(tp, tp + fn), 'false_alarm_rate': wilson(fp, fp + tn)}
    for k, v in m.items():
        ok = v[~np.isnan(v)]
        if len(ok) < boots * 0.5:
            continue  # quá nhiều lần lấy mẫu không xác định (vd. không có ca dương): không báo khoảng
        lo, hi = np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out['ci_group_bootstrap'][k] = [round(float(lo), 3), round(float(hi), 3)]
        if naive.get(k):
            out['ci_if_independent'][k] = naive[k]
            w = naive[k][1] - naive[k][0]
            if w > 0:
                out['width_ratio'][k] = round(float(hi - lo) / w, 2)
    return out
