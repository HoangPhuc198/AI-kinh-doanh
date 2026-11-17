import sys
import unittest
from pathlib import Path

import numpy as np

from tests import _paths  # noqa: F401

sys.path.insert(0, str(_paths.PROJECT / 'training'))
import group_eval as G  # noqa: E402


class GroupingTests(unittest.TestCase):
    def test_source_stem_strips_roboflow_suffix(self):
        self.assertEqual(G.source_stem('abc_jpg.rf.' + 'a' * 32 + '.jpg'), 'abc_jpg')
        self.assertEqual(G.source_stem('plain.jpg'), 'plain')

    def test_same_stem_is_one_group_and_groups_are_transitive(self):
        g = G.build_groups(['a', 'a', 'b', 'c', 'd'], pairs=[(2, 3, 2)])  # b~c gần trùng
        self.assertEqual(g[0], g[1])
        self.assertEqual(g[2], g[3])
        self.assertEqual(len(set(g)), 3)
        self.assertEqual(sorted(set(g)), [0, 1, 2])  # nhóm đánh số liên tục

    def test_regex_video_key_falls_back_to_stem(self):
        self.assertEqual(G.video_key('IMG_3485_MOV-0001_jpg.rf.' + 'b' * 32 + '.jpg', r'^(IMG_\d+_MOV)'), 'IMG_3485_MOV')
        self.assertEqual(G.video_key('zzz.jpg', r'^(IMG_\d+_MOV)'), 'zzz')

    def test_near_duplicate_detects_flip_and_rejects_different_image(self):
        rng = np.random.default_rng(0)
        a = (rng.random((120, 160)) * 255).astype(np.uint8)
        a = np.kron(a[::8, ::8], np.ones((8, 8), np.uint8))  # ảnh có cấu trúc thô để dHash ổn định
        b = rng.integers(0, 255, a.shape, dtype=np.uint8)
        imgs = [a, a[:, ::-1].copy(), b]
        h = [G.dhash64(i) for i in imgs]
        hf = [G.dhash64(i, True) for i in imgs]
        pairs = {(i, j) for i, j, _ in G.near_duplicate_pairs(h, hf, threshold=6)}
        self.assertIn((0, 1), pairs)       # bản lật ngang là gần trùng
        self.assertNotIn((0, 2), pairs)    # ảnh khác thì không

    def test_leakage_separates_confirmed_from_name_only(self):
        stems = ['x', 'x', 'y', 'y', 'z']
        splits = ['train', 'test', 'train', 'valid', 'test']
        rep = G.leakage_report(stems, splits, pairs=[(0, 1, 1)])  # chỉ x được xác nhận là cùng ảnh
        self.assertEqual(rep['images_with_confirmed_near_duplicate_in_train'], {'test': 1})
        self.assertEqual(rep['source_names_in_multiple_splits'], 2)
        self.assertEqual(rep['of_which_confirmed_same_image'], 1)
        self.assertEqual(rep['of_which_name_only_probably_different_images'], 1)


class MetricTests(unittest.TestCase):
    def test_match_box_respects_iou_threshold(self):
        gt = [0.1, 0.1, 0.5, 0.5]
        self.assertEqual(G.match_box(gt, [[0.6, 0.6, 0.9, 0.9], [0.12, 0.1, 0.5, 0.52]], 0.4), 1)
        self.assertIsNone(G.match_box(gt, [[0.6, 0.6, 0.9, 0.9]], 0.4))

    def test_read_yolo_labels(self):
        p = Path(self.id() + '.txt')
        p.write_text('2 0.5 0.5 0.2 0.4\nbad\n', encoding='utf-8')
        try:
            (c, box), = G.read_yolo_labels(p)
        finally:
            p.unlink()
        self.assertEqual(c, 2)
        self.assertEqual([round(v, 2) for v in box], [0.4, 0.3, 0.6, 0.7])

    def test_group_counts_and_rates(self):
        c = G.group_counts([0, 0, 1, 1], [True, False, True, False], [True, True, False, False])
        self.assertEqual(c.tolist(), [[1, 1, 0, 0], [0, 0, 1, 1]])
        m = G.metrics_from_counts(c.sum(0))
        self.assertAlmostEqual(float(m['precision']), 0.5)
        self.assertAlmostEqual(float(m['recall']), 0.5)
        self.assertAlmostEqual(float(m['false_alarm_rate']), 0.5)

    def test_undefined_rate_is_nan_not_error(self):
        m = G.metrics_from_counts([0, 0, 0, 5])
        self.assertTrue(np.isnan(m['precision']))
        self.assertTrue(np.isnan(m['recall']))

    def test_bootstrap_is_deterministic_and_contains_point_estimate(self):
        rng = np.random.default_rng(1)
        groups = np.repeat(np.arange(40), 5)
        truth = rng.random(200) < 0.4
        pred = np.where(rng.random(200) < 0.8, truth, ~truth)
        a, b = G.summarise(groups, truth, pred, boots=500, seed=3), G.summarise(groups, truth, pred, boots=500, seed=3)
        self.assertEqual(a, b)
        lo, hi = a['ci_group_bootstrap']['recall']
        self.assertLessEqual(lo, a['recall'])
        self.assertGreaterEqual(hi, a['recall'])

    def test_correlated_groups_give_wider_interval_than_independent_assumption(self):
        # 10 nhóm, mỗi nhóm 30 ảnh GIỐNG HỆT nhau (đúng/sai cả nhóm): 300 ảnh nhưng thực chất chỉ 10 mẫu
        groups = np.repeat(np.arange(10), 30)
        truth = np.ones(300, bool)
        pred = np.repeat(np.arange(10) % 2 == 0, 30)
        r = G.summarise(groups, truth, pred, boots=1000, seed=0)
        self.assertGreater(r['width_ratio']['recall'], 2.0)

    def test_zero_bootstrap_returns_point_estimate_only(self):
        r = G.summarise([0, 1, 2, 3], [True, True, False, False], [True, False, False, True], boots=0)
        self.assertEqual(r['ci_group_bootstrap'], {})
        self.assertEqual((r['tp'], r['fp'], r['fn'], r['tn']), (1, 1, 1, 1))

    def test_single_group_returns_point_estimate_without_ci(self):
        r = G.summarise([0, 0, 0], [True, True, False], [True, False, False], boots=100)
        self.assertEqual(r['ci_group_bootstrap'], {})
        self.assertEqual((r['tp'], r['fn'], r['tn']), (1, 1, 1))


if __name__ == '__main__':
    unittest.main()
