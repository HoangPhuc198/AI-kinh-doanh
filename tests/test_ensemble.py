import unittest
from unittest import mock

import numpy as np

from tests import _paths  # noqa: F401
import ensemble
from ensemble import LABEL_NORMAL, LABEL_SUSPECT, LABEL_WATCH, fuse

_pose_patch = mock.patch.object(ensemble, '_pose_infer', lambda frame, conf: {'details': [], 'ms': 1})  # không nạp mô hình pose thật


def setUpModule():
    _pose_patch.start()


def tearDownModule():
    _pose_patch.stop()


P1 = [0.10, 0.20, 0.40, 0.90]
P2 = [0.60, 0.20, 0.90, 0.90]


def person(box, conf=0.9):
    return {'label': 'person', 'confidence': conf, 'box': box}


class FuseTests(unittest.TestCase):
    def test_clean_person_is_normal(self):
        r = fuse([person(P1)], {'yolo_duyen': [{'label': 'Normal', 'confidence': 0.9, 'box': P1}]})
        self.assertEqual([p['label'] for p in r['people']], [LABEL_NORMAL])
        self.assertGreater(r['people'][0]['confidence'], 0.9)  # độ chắc của nhãn "bình thường"

    def test_two_sources_agree_is_suspect(self):
        src = {'yolo_duyen': [{'label': 'Cheating', 'confidence': 0.9, 'box': P1}], 'duyen': [{'label': 'Suspicious', 'confidence': None, 'box': P1}]}
        p = fuse([person(P1)], src)['people'][0]
        self.assertEqual(p['label'], LABEL_SUSPECT)
        self.assertEqual(len(p['evidence']), 2)

    def test_single_source_is_only_watch(self):
        src = {'yolo_duyen': [{'label': 'Cheating', 'confidence': 0.95, 'box': P1}]}
        self.assertEqual(fuse([person(P1)], src)['people'][0]['label'], LABEL_WATCH)

    def test_weak_evidence_is_not_counted_as_agreement(self):
        src = {'yolo_duyen': [{'label': 'Cheating', 'confidence': 0.4, 'box': P1}], 'gianlan': [{'label': 'cheating', 'confidence': 0.4, 'box': P1}]}
        self.assertNotEqual(fuse([person(P1)], src)['people'][0]['label'], LABEL_SUSPECT)

    def test_only_the_matching_person_is_flagged(self):
        src = {'yolo_duyen': [{'label': 'Cheating', 'confidence': 0.9, 'box': P1}], 'duyen': [{'label': 'Suspicious', 'confidence': None, 'box': P1}]}
        labels = {tuple(p['box']): p['label'] for p in fuse([person(P1), person(P2)], src)['people']}
        self.assertEqual(labels[tuple(P1)], LABEL_SUSPECT)
        self.assertEqual(labels[tuple(P2)], LABEL_NORMAL)

    def test_one_big_box_is_assigned_to_a_single_person(self):
        big = [0.05, 0.15, 0.95, 0.95]
        src = {'yolo_duyen': [{'label': 'Cheating', 'confidence': 0.9, 'box': big}]}
        labels = [p['label'] for p in fuse([person(P1), person(P2)], src)['people']]
        self.assertEqual(sorted(labels), sorted([LABEL_WATCH, LABEL_NORMAL]))

    def test_phone_near_person_adds_evidence_far_phone_does_not(self):
        near = {'label': 'cell phone', 'confidence': 0.8, 'box': [0.2, 0.5, 0.25, 0.6]}
        far = {'label': 'cell phone', 'confidence': 0.8, 'box': [0.45, 0.5, 0.5, 0.6]}
        r = fuse([person(P1), person(P2), near, far], {})
        by = {tuple(p['box']): p for p in r['people']}
        self.assertTrue(any('Điện thoại' in e for e in by[tuple(P1)]['evidence']))
        self.assertEqual(by[tuple(P2)]['evidence'], [])
        self.assertEqual(len(r['phones']), 2)  # cả hai điện thoại vẫn được vẽ

    def test_low_confidence_anchor_person_is_dropped(self):
        self.assertEqual(fuse([person(P1, 0.2)], {})['people'], [])

    def test_domain_model_adds_person_anchor_missed_and_does_not_duplicate(self):
        src = {'yolo_duyen': [{'label': 'Normal', 'confidence': 0.8, 'box': P1}, {'label': 'Normal', 'confidence': 0.8, 'box': P2}]}
        r = fuse([person(P1)], src)
        self.assertEqual(len(r['people']), 2)
        self.assertEqual(sorted(p['origin'] for p in r['people']), ['person_coco', 'yolo_duyen'])

    def test_invigilator_label_is_not_a_candidate(self):
        src = {'gianlan': [{'label': 'invigilator', 'confidence': 0.9, 'box': P2}]}
        self.assertEqual(fuse([person(P1)], src)['people'][0]['box'], P1)
        self.assertEqual(len(fuse([person(P1)], src)['people']), 1)

    def test_boxes_are_always_valid(self):
        weird = person([-0.2, 1.4, 0.5, -0.1])
        for p in fuse([weird], {'yolo_duyen': [{'label': 'Cheating', 'confidence': 1.7, 'box': [1.2, -1, 0.3, 0.3]}]})['people']:
            x0, y0, x1, y1 = p['box']
            self.assertTrue(0 <= x0 <= x1 <= 1 and 0 <= y0 <= y1 <= 1)
            self.assertTrue(0 <= p['confidence'] <= 1 and 0 <= p['score'] <= 1)

    def test_empty_input(self):
        self.assertEqual(fuse([], {}), {'people': [], 'phones': []})


class RunTests(unittest.TestCase):
    def fake_runner(self, fail=()):
        def run_model(key, frame, conf, include_image=True, imgsz=None, iou=None):
            if key in fail:
                raise RuntimeError('mô hình lỗi')
            det = {'person_coco': [person(P1)], 'yolo_duyen': [{'label': 'Cheating', 'confidence': 0.9, 'box': P1}],
                   'duyen': [{'label': 'Suspicious', 'confidence': None, 'box': P1}]}.get(key, [])
            return {'details': det, 'counts': {}, 'ms': 10}
        return run_model

    def test_run_returns_valid_structure_with_image(self):
        import inference
        with mock.patch.object(inference, 'run_model', self.fake_runner()):
            out = ensemble.run(np.zeros((120, 160, 3), np.uint8))
        self.assertEqual(out['counts'], {ensemble.V_SUSPECT: 1})
        self.assertIn('image', out)
        self.assertEqual(out['errors'], {})
        self.assertEqual(set(out['timings']), set(ensemble.components()))

    def test_one_failing_source_is_reported_not_fatal(self):
        import inference
        with mock.patch.object(inference, 'run_model', self.fake_runner(fail=('gianlan',))):
            out = ensemble.run(np.zeros((120, 160, 3), np.uint8), include_image=False)
        self.assertIn('gianlan', out['errors'])
        self.assertEqual(out['counts'], {ensemble.V_SUSPECT: 1})


if __name__ == '__main__':
    unittest.main()


class WeakSourceTests(unittest.TestCase):
    def test_knn_alone_never_flags(self):
        src = {'duyen': [{'label': 'Suspicious', 'confidence': None, 'box': P1}]}
        self.assertEqual(fuse([person(P1)], src)['people'][0]['label'], LABEL_NORMAL)

    def test_knn_cannot_make_suspect_with_only_phone_weak_evidence(self):
        src = {'duyen': [{'label': 'Suspicious', 'confidence': None, 'box': P1}]}
        weak_phone = {'label': 'cell phone', 'confidence': 0.4, 'box': [0.2, 0.5, 0.25, 0.6]}  # < 0.5: không tính đồng thuận
        self.assertNotEqual(fuse([person(P1), weak_phone], src)['people'][0]['label'], LABEL_SUSPECT)

    def test_phone_plus_knn_without_behavior_model_is_only_watch(self):
        phone = {'label': 'cell phone', 'confidence': 0.9, 'box': [0.2, 0.5, 0.25, 0.6]}
        src = {'duyen': [{'label': 'Suspicious', 'confidence': None, 'box': P1}]}
        self.assertEqual(fuse([person(P1), phone], src)['people'][0]['label'], LABEL_WATCH)

    def test_two_strong_sources_are_suspect_without_knn(self):
        phone = {'label': 'cell phone', 'confidence': 0.9, 'box': [0.2, 0.5, 0.25, 0.6]}
        src = {'yolo_duyen': [{'label': 'Cheating', 'confidence': 0.9, 'box': P1}]}
        self.assertEqual(fuse([person(P1), phone], src)['people'][0]['label'], LABEL_SUSPECT)


class TrainingSupportTests(unittest.TestCase):
    """Training YOLO là nguồn phụ: một mình không bao giờ báo, không tạo người, không làm loãng điểm khi im lặng."""

    def test_training_alone_never_flags(self):
        src = {'training': [{'label': 'Cheating', 'confidence': 0.95, 'box': P1}]}
        self.assertEqual(fuse([person(P1)], src)['people'][0]['label'], LABEL_NORMAL)

    def test_training_does_not_create_people(self):
        src = {'training': [{'label': 'Cheating', 'confidence': 0.95, 'box': [0.6, 0.1, 0.9, 0.9]}]}
        self.assertEqual(len(fuse([person(P1)], src)['people']), 1)

    def test_training_supports_a_strong_source(self):
        src = {'yolo_duyen': [{'label': 'Cheating', 'confidence': 0.9, 'box': P1}], 'training': [{'label': 'Cheating', 'confidence': 0.9, 'box': P1}]}
        self.assertEqual(fuse([person(P1)], src)['people'][0]['label'], LABEL_SUSPECT)

    def test_silent_training_does_not_dilute_score(self):
        from ensemble import decide
        votes = {'yolo_duyen': 0.9, 'pose': 0.9}
        self.assertAlmostEqual(decide(votes)[1], decide({**votes, 'training': 0.0})[1])


class DrawTests(unittest.TestCase):
    def test_vietnamese_labels_render_and_boxes_stay_inside_image(self):
        frame = np.full((200, 300, 3), 40, np.uint8)
        out = ensemble._draw(frame, [{'label': ensemble.V_SUSPECT, 'confidence': 0.6, 'box': [0.0, 0.0, 1.0, 1.0]}], [])
        self.assertEqual(out.shape, frame.shape)
        self.assertFalse(np.array_equal(out, frame))

    def test_ascii_fallback_removes_diacritics(self):
        self.assertEqual(ensemble._ascii('Nghi vấn gian lận Đ'), 'Nghi van gian lan D')


class SchedulingTests(unittest.TestCase):
    def test_behavior_models_run_every_other_frame_and_are_reused(self):
        import inference
        calls = []

        def run_model(key, frame, conf, include_image=True, imgsz=None, iou=None, state=None):
            calls.append(key)
            det = {'person_coco': [person(P1)], 'yolo_duyen': [{'label': 'Cheating', 'confidence': 0.9, 'box': P1}],
                   'duyen': [{'label': 'Suspicious', 'confidence': None, 'box': P1}]}.get(key, [])
            return {'details': det, 'counts': {}, 'ms': 1}
        state, outs = {}, []
        with mock.patch.object(inference, 'run_model', run_model):
            for i in range(4):  # khung khác nhau mỗi lần để cổng chuyển động không dùng lại kết quả
                frame = np.random.default_rng(i).integers(0, 255, (60, 80, 3), dtype=np.uint8)
                outs.append(ensemble.run(frame, include_image=False, state=state))
        self.assertEqual([o['behavior_reused'] for o in outs], [False, True, False, True])
        self.assertEqual(calls.count('person_coco'), 4)          # neo người chạy mọi khung
        self.assertEqual(calls.count('yolo_duyen'), 2)           # mô hình hành vi chỉ 2/4 khung
        self.assertEqual(outs[0]['counts'], {ensemble.V_NORMAL: 1})  # khung đầu: chưa đủ 2 khung liên tiếp
        self.assertTrue(all(o['counts'] == {ensemble.V_SUSPECT: 1} for o in outs[1:]))  # kết luận ổn định cả khung dùng lại

    def test_no_state_runs_everything_each_call(self):
        import inference
        seen = []

        def run_model(key, frame, conf, include_image=True, imgsz=None, iou=None, state=None):
            seen.append(key)
            return {'details': [], 'counts': {}, 'ms': 1}
        with mock.patch.object(inference, 'run_model', run_model):
            ensemble.run(np.zeros((60, 80, 3), np.uint8), include_image=False)
        self.assertEqual(sorted(seen), sorted(set(ensemble.components()) - ensemble.LATE_SOURCES - {'pose'}))  # không có nghi vấn mạnh -> không cần KNN


def _runner(calls, cheat=True):
    def run_model(key, frame, conf, include_image=True, imgsz=None, iou=None, state=None):
        calls.append((key, imgsz))
        det = {'person_coco': [person(P1)], 'yolo_duyen': [{'label': 'Cheating' if cheat else 'Normal', 'confidence': 0.9, 'box': P1}],
               'duyen': [{'label': 'Suspicious', 'confidence': None, 'box': P1}]}.get(key, [])
        return {'details': det, 'counts': {}, 'ms': 1}
    return run_model


class CascadeAndGateTests(unittest.TestCase):
    def test_knn_runs_only_when_a_strong_behavior_model_is_suspicious(self):
        import inference
        for cheat, expect_knn in ((True, True), (False, False)):
            calls = []
            with mock.patch.object(inference, 'run_model', _runner(calls, cheat)):
                ensemble.run(np.zeros((60, 80, 3), np.uint8), include_image=False)
            self.assertEqual('duyen' in [c[0] for c in calls], expect_knn)

    def test_static_scene_reuses_result_until_stale_or_motion(self):
        import inference
        calls, still = [], np.full((60, 80, 3), 90, np.uint8)
        clock = [0.0]
        state = {}
        with mock.patch.object(inference, 'run_model', _runner(calls)), mock.patch.object(ensemble, '_now', lambda: clock[0]):
            first = ensemble.run(still, include_image=False, state=state)
            n = len(calls)
            second = ensemble.run(still.copy(), include_image=False, state=state)
            self.assertEqual((first['result_reused'], second['result_reused'], len(calls)), (False, True, n))
            self.assertEqual(second['counts'], {ensemble.V_SUSPECT: 1})  # khung thứ 2 liên tiếp (dùng lại kết quả) đủ điều kiện Nghi ngờ
            moved = still.copy(); moved[10:40, 10:40] = 255          # có vật thể/chuyển động
            clock[0] = 0.3
            self.assertFalse(ensemble.run(moved, include_image=False, state=state)['result_reused'])
            clock[0] = 5.0                                              # quá hạn MAX_STALE_S: bắt buộc suy luận lại
            self.assertFalse(ensemble.run(moved.copy(), include_image=False, state=state)['result_reused'])

    def test_gpu_keeps_requested_imgsz_and_defaults_to_960(self):
        import runtime
        with mock.patch.object(runtime, 'device', return_value=0):
            small = np.zeros((480, 640, 3), np.uint8)
            self.assertEqual(ensemble._fit_imgsz(small, 800), 800)
            self.assertEqual(ensemble._fit_imgsz(small, None), ensemble.GPU_IMGSZ)

    @mock.patch('runtime.device', return_value='cpu')
    def test_small_frames_are_not_upscaled(self, _):
        self.assertEqual(ensemble._fit_imgsz(np.zeros((480, 640, 3), np.uint8), 800), 640)
        self.assertEqual(ensemble._fit_imgsz(np.zeros((720, 1280, 3), np.uint8), 800), 800)
        self.assertEqual(ensemble._fit_imgsz(np.zeros((500, 700, 3), np.uint8), 960), 800)
