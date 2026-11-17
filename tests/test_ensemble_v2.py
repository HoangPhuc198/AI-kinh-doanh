import unittest
from unittest import mock

import numpy as np

from tests import _paths  # noqa: F401
import ensemble
import ensemble_v2 as v2
from judge import LABEL_CHEAT, LABEL_NORMAL, LABEL_SUSPECT, Settings

P1 = [0.10, 0.20, 0.40, 0.90]
APP = {'training': (0.5, {'cheating'})}   # nguồn hình dáng giả: test không phụ thuộc việc chọn nguồn thật
PHONE_BOX = [0.2, 0.5, 0.25, 0.6]


def setUpModule():
    patch.start()


def tearDownModule():
    patch.stop()


patch = mock.patch.dict(v2.APPEARANCE, APP)


def frame(turn=None, app=None, phone=None, box=P1, side=None):
    """Một khung: (anchor, sources) theo đúng định dạng đầu ra các mô hình."""
    anchor = [{'label': 'person', 'confidence': 0.9, 'box': box}]
    if phone is not None:
        anchor.append({'label': 'cell phone', 'confidence': phone, 'box': PHONE_BOX})
    sources = {'pose': [{'label': 'head_turn', 'confidence': max(turn or 0.0, side or 0.0), 'turn': turn, 'side': side, 'dir': 1, 'box': box}] if (turn is not None or side is not None) else [],
               'training': [{'label': 'Cheating', 'confidence': app, 'box': box}] if app is not None else []}
    return anchor, sources


def simulate(feed, seconds, fps=5, cfg=None):
    """feed(t) -> kwargs của frame(); trả về [(t, nhãn người #1)] sau Judge + Escalator như phiên webcam."""
    cfg, state, out = cfg or Settings(), {}, []
    for i in range(int(seconds * fps) + 1):
        t = i / fps
        fused = v2.fuse(*frame(**feed(t)), cfg)
        out.append((t, v2._judged(fused, state, t, cfg)['people'][0]['label']))
    return out


def first(labels, name):
    return next((t for t, lb in labels if lb == name), None)


class DecideTests(unittest.TestCase):
    def test_appearance_alone_never_flags_even_when_very_confident(self):
        self.assertEqual(v2.decide({'training': 0.99})[0], 'normal')

    def test_single_measured_signal_is_watch(self):
        self.assertEqual(v2.decide({'pose': 0.9})[0], 'watch')
        self.assertEqual(v2.decide({'phone': 0.9})[0], 'watch')

    def test_two_sources_with_a_measured_one_are_suspect(self):
        self.assertEqual(v2.decide({'pose': 0.9, 'phone': 0.9})[0], 'suspect')
        self.assertEqual(v2.decide({'pose': 0.9, 'training': 0.8})[0], 'suspect')
        self.assertEqual(v2.decide({'phone': 0.9, 'training': 0.8})[0], 'suspect')

    def test_side_look_cue_alone_is_watch_never_suspect(self):
        self.assertEqual(v2.decide({'pose_side': 0.8})[0], 'watch')
        self.assertEqual(v2.decide({'pose_side': 0.8, 'training': 0.9})[0], 'watch')   # hai nguồn yếu không tự thành nghi vấn

    def test_side_look_cue_corroborates_a_measured_signal(self):
        self.assertEqual(v2.decide({'pose_side': 0.8, 'phone': 0.9})[0], 'suspect')

    def test_weak_evidence_below_threshold_is_ignored(self):
        self.assertEqual(v2.decide({'pose': 0.4, 'training': 0.4})[0], 'normal')
        self.assertEqual(v2.decide({'pose': 0.4, 'training': 0.4}, Settings(evidence_conf=0.3))[0], 'suspect')


def keypoints(nose_x=120.0, eye_conf=(0.9, 0.9), ear_conf=(0.9, 0.9)):
    """17 keypoint giả: mũi ở nose_x, mắt ở x=100 và x=140 (cách nhau 40 px), tọa độ chỉ cần 5 điểm đầu."""
    xy = np.zeros((17, 2))
    xy[0] = (nose_x, 120)
    xy[1], xy[2] = (100, 100), (140, 100)
    xy[3], xy[4] = (80, 105), (160, 105)
    conf = np.zeros(17)
    conf[0], conf[1], conf[2], conf[3], conf[4] = 0.95, eye_conf[0], eye_conf[1], ear_conf[0], ear_conf[1]
    return xy, conf


class HeadTurnTests(unittest.TestCase):
    """Lỗi 6 của v1: chỉ thấy MỘT mắt (người cúi mặt viết bài hay mất một mắt) bị đọc thành quay nghiêng với bằng chứng hằng số 0,8/1,0."""

    def test_facing_forward_is_measured_as_no_turn(self):
        self.assertEqual(v2.head_turn_evidence(*keypoints(nose_x=120.0)), 0.0)

    def test_a_clear_two_eye_turn_is_measured(self):
        self.assertGreater(v2.head_turn_evidence(*keypoints(nose_x=100.0)), 0.55)

    def test_one_visible_eye_is_not_measured_in_v2_but_is_a_full_turn_in_v1(self):
        xy, conf = keypoints(eye_conf=(0.9, 0.1))
        self.assertIsNone(v2.head_turn_evidence(xy, conf))
        self.assertGreaterEqual(ensemble.head_turn_evidence(xy, conf), 0.8)   # v1 giữ nguyên hành vi cũ (để quay lại)

    def test_no_nose_or_no_eyes_is_not_measured(self):
        xy, conf = keypoints()
        conf[0] = 0.1
        self.assertIsNone(v2.head_turn_evidence(xy, conf))
        xy, conf = keypoints(eye_conf=(0.1, 0.1))
        self.assertIsNone(v2.head_turn_evidence(xy, conf))

    def test_profile_cue_needs_exactly_one_eye_and_a_visible_nose_offset(self):
        xy, conf = keypoints(eye_conf=(0.9, 0.1))
        self.assertEqual(v2.profile_cue(xy, conf), v2.PROFILE_EV)
        self.assertIsNone(v2.profile_cue(*keypoints()))                                  # hai mắt: là nhánh đo được, không phải tín hiệu yếu
        self.assertIsNone(v2.profile_cue(*keypoints(eye_conf=(0.1, 0.1))))               # không mắt nào
        xy, conf = keypoints(nose_x=101.0, eye_conf=(0.9, 0.1))                         # mũi sát mắt: dưới nhiễu định vị
        self.assertIsNone(v2.profile_cue(xy, conf))

    def test_head_down_writing_with_a_lost_eye_can_be_suspect_but_never_gian_lan(self):
        # lỗi 6 của v1: người cúi viết 30 s, pose liên tục chỉ thấy một mắt. Nay: tối đa Nghi ngờ (tín hiệu yếu không leo thang).
        side = v2.profile_cue(*keypoints(eye_conf=(0.9, 0.1)))
        labels = simulate(lambda t: dict(side=side), 30)
        self.assertNotIn(LABEL_CHEAT, {lb for _, lb in labels})
        self.assertEqual(labels[-1][1], LABEL_SUSPECT)

    def test_a_measured_two_eye_turn_still_reaches_gian_lan(self):
        turn = v2.head_turn_evidence(*keypoints(nose_x=100.0))
        self.assertIsNotNone(first(simulate(lambda t: dict(turn=turn), 30), LABEL_CHEAT))

    def test_side_cue_plus_phone_concludes_quickly(self):
        side = v2.profile_cue(*keypoints(eye_conf=(0.9, 0.1)))
        self.assertLessEqual(first(simulate(lambda t: dict(side=side, phone=0.9), 10), LABEL_CHEAT), 2.0)


class FuseTests(unittest.TestCase):
    def test_overlapping_pose_detections_use_the_maximum_in_any_order(self):  # lỗi 3 của v1
        persons = [{'box': P1}]
        turned = {'turn': 0.9, 'box': P1}
        straight = {'turn': 0.05, 'box': P1}
        self.assertEqual(v2.pose_evidence(persons, [turned, straight]), [0.9])
        self.assertEqual(v2.pose_evidence(persons, [straight, turned]), [0.9])

    def test_unmeasured_pose_is_none_not_zero(self):
        self.assertEqual(v2.pose_evidence([{'box': P1}], [{'turn': None, 'box': P1}]), [None])

    def test_appearance_flag_alone_leaves_person_normal_with_no_reason(self):
        p = v2.fuse(*frame(app=0.9))['people'][0]
        self.assertEqual((p['label'], p['evidence'], p['frame_tier'] if 'frame_tier' in p else p['tier']), ('Bình thường', [], 'normal'))

    def test_reasons_use_the_texts_the_webcam_page_already_understands(self):
        p = v2.fuse(*frame(turn=0.9, phone=0.9, app=0.8))['people'][0]
        self.assertEqual(p['evidence'][:2], ['Quay/nghiêng đầu (0.90)', 'Điện thoại gần người (0.90)'])
        self.assertEqual(p['tier'], 'suspect')

    def test_only_the_anchor_creates_people(self):
        anchor, sources = frame(app=0.9)
        sources['training'].append({'label': 'Cheating', 'confidence': 0.9, 'box': [0.6, 0.1, 0.9, 0.9]})
        self.assertEqual(len(v2.fuse(anchor, sources)['people']), 1)

    def test_far_phone_is_not_attributed(self):
        anchor, sources = frame(phone=0.9)
        anchor[1]['box'] = [0.8, 0.1, 0.9, 0.2]
        self.assertEqual(v2.fuse(anchor, sources)['people'][0]['votes']['phone'], 0.0)


class TemporalTests(unittest.TestCase):
    def test_sitting_still_with_constant_appearance_flags_is_never_flagged(self):
        # lỗi 1 của v1: dáng ngồi yên bị mô hình hình dáng gắn cờ lặp lại ở MỌI khung, kéo dài 60 s
        labels = simulate(lambda t: dict(app=0.8), 60)
        self.assertEqual({lb for _, lb in labels}, {LABEL_NORMAL})

    def test_sitting_still_with_flickering_borderline_flags_is_never_flagged(self):
        labels = simulate(lambda t: dict(app=0.6 if int(t * 5) % 3 else 0.45), 60)
        self.assertEqual({lb for _, lb in labels}, {LABEL_NORMAL})

    def test_a_held_head_turn_reaches_gian_lan_after_three_times_min_seconds(self):  # lỗi 2 của v1
        labels = simulate(lambda t: dict(turn=0.95), 30)
        self.assertIsNotNone(first(labels, LABEL_SUSPECT))
        self.assertLess(first(labels, LABEL_SUSPECT), 2.0)
        self.assertGreaterEqual(first(labels, LABEL_CHEAT), 4.5)
        self.assertLessEqual(first(labels, LABEL_CHEAT), 5.0)
        self.assertEqual(labels[-1][1], LABEL_CHEAT)

    def test_escalation_time_follows_the_sensitivity_setting(self):
        slow = first(simulate(lambda t: dict(turn=0.95), 30, cfg=Settings(min_seconds=3.0)), LABEL_CHEAT)
        fast = first(simulate(lambda t: dict(turn=0.95), 30, cfg=Settings(min_seconds=1.0)), LABEL_CHEAT)
        self.assertGreaterEqual(slow, 9.0)
        self.assertLessEqual(fast, 3.5)

    def test_short_glances_never_reach_gian_lan(self):
        labels = simulate(lambda t: dict(turn=0.95 if (t % 4.0) < 1.0 else None), 60)  # liếc 1 s, nhìn thẳng 3 s
        self.assertNotIn(LABEL_CHEAT, {lb for _, lb in labels})

    def test_two_agreeing_sources_conclude_quickly(self):
        labels = simulate(lambda t: dict(turn=0.95, phone=0.9), 10)
        self.assertLessEqual(first(labels, LABEL_CHEAT), 2.0)

    def test_phone_held_is_also_measured_evidence(self):
        self.assertIsNotNone(first(simulate(lambda t: dict(phone=0.9), 30), LABEL_CHEAT))

    def test_label_returns_to_normal_after_the_signal_stops(self):
        labels = simulate(lambda t: dict(turn=0.95) if t < 8 else {}, 14)
        self.assertEqual(first([x for x in labels if x[0] > 8], LABEL_NORMAL) is not None, True)
        self.assertEqual(labels[-1][1], LABEL_NORMAL)

    def test_vanished_tracks_are_forgotten_even_without_any_signal(self):
        esc, cfg = v2.Escalator(), Settings()
        for pid in range(1, 6):
            esc.apply([{'id': pid, 'strong_on': False}], float(pid), cfg)
        esc.apply([{'id': 99, 'strong_on': False}], 100.0, cfg)
        self.assertEqual(list(esc._s), [99])

    def test_video_rewind_drops_the_old_streak(self):
        esc, cfg = v2.Escalator(), Settings()
        for t in (10.0, 11.0, 12.0, 13.0, 14.0, 15.0):
            p = [{'id': 1, 'strong_on': True, 'label': 'Nghi ngờ', 'confidence': 0.5, 'score': 0.9}]
            esc.apply(p, t, cfg)
        self.assertEqual(p[0]['label'], LABEL_CHEAT)
        p = [{'id': 1, 'strong_on': True, 'label': 'Nghi ngờ', 'confidence': 0.5, 'score': 0.9}]
        esc.apply(p, 2.0, cfg)   # tua về giây 2
        self.assertEqual(p[0]['label'], 'Nghi ngờ')
        self.assertEqual(p[0]['held_s'], 0.0)

    def test_single_image_never_concludes_gian_lan(self):
        out = v2._judged(v2.fuse(*frame(turn=0.95, phone=0.9)), None, 0.0, Settings())
        self.assertEqual(out['people'][0]['label'], LABEL_SUSPECT)


class RunTests(unittest.TestCase):
    def runner(self, fail=()):
        def run_model(key, frame_, conf, include_image=True, imgsz=None, iou=None, state=None):
            if key in fail:
                raise RuntimeError('mô hình lỗi')
            det = {'person_coco': [{'label': 'person', 'confidence': 0.9, 'box': P1}], 'training': [{'label': 'Cheating', 'confidence': 0.9, 'box': P1}]}.get(key, [])
            return {'details': det, 'counts': {}, 'ms': 5}
        return run_model

    def pose(self, turn=0.95):
        return lambda frame_, conf: {'details': [{'label': 'head_turn', 'confidence': turn, 'turn': turn, 'dir': 1, 'box': P1}], 'ms': 3}

    def test_run_returns_valid_structure_and_flags_two_sources(self):
        import inference
        with mock.patch.object(inference, 'run_model', self.runner()), mock.patch.object(v2, '_pose_infer', self.pose()):
            out = v2.run(np.zeros((120, 160, 3), np.uint8))
        self.assertEqual(out['counts'], {ensemble.V_SUSPECT: 1})
        self.assertIn('image', out)
        self.assertEqual(out['errors'], {})
        self.assertEqual(set(out['timings']), set(v2.components()))
        d = out['details'][0]
        self.assertTrue(all(k in d for k in ('label', 'confidence', 'box', 'votes', 'evidence', 'id', 'window_score', 'frame_tier')))

    def test_failing_pose_is_reported_not_fatal(self):
        import inference

        def boom(frame_, conf):
            raise RuntimeError('pose lỗi')
        with mock.patch.object(inference, 'run_model', self.runner()), mock.patch.object(v2, '_pose_infer', boom):
            out = v2.run(np.zeros((120, 160, 3), np.uint8), include_image=False)
        self.assertIn('pose', out['errors'])
        self.assertEqual(out['counts'], {ensemble.V_NORMAL: 1})   # chỉ còn nguồn hình dáng: không đủ để báo

    def test_failing_anchor_is_fatal(self):
        import inference
        with mock.patch.object(inference, 'run_model', self.runner(fail=('person_coco',))), mock.patch.object(v2, '_pose_infer', self.pose()):
            with self.assertRaises(RuntimeError):
                v2.run(np.zeros((120, 160, 3), np.uint8), include_image=False)

    def test_webcam_session_reaches_gian_lan_and_reuses_sources(self):
        import inference
        calls = []

        def run_model(key, frame_, conf, include_image=True, imgsz=None, iou=None, state=None):
            calls.append(key)
            return self.runner()(key, frame_, conf)
        state, outs = {}, []
        with mock.patch.object(inference, 'run_model', run_model), mock.patch.object(v2, '_pose_infer', self.pose()):
            for i in range(12):
                f = np.random.default_rng(i).integers(0, 255, (60, 80, 3), dtype=np.uint8)  # cảnh luôn đổi: không bị cổng chuyển động dùng lại
                outs.append(v2.run(f, include_image=False, state=state, now=i * 0.5))
        self.assertEqual(calls.count('person_coco'), 12)
        self.assertEqual(calls.count('training'), 6)             # mô hình hành vi chỉ 1/2 số khung
        self.assertEqual(outs[-1]['counts'], {ensemble.LABEL_CHEAT: 1})


if __name__ == '__main__':
    unittest.main()
