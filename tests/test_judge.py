import unittest

from tests import _paths  # noqa: F401
import ensemble
from judge import LABEL_CHEAT, LABEL_NORMAL, LABEL_SUSPECT, Judge, Settings

BOX = [0.1, 0.2, 0.4, 0.9]


def feed(judge, tiers, cfg=None, dt=0.5, box=BOX, t0=0.0):
    cfg, out = cfg or Settings(), []
    for i, tier in enumerate(tiers):
        out.append(judge.apply([{'box': list(box), 'tier': tier, 'score': 0.5}], t0 + i * dt, cfg)[0])
    return out


class JudgeTests(unittest.TestCase):
    def test_clean_person_stays_normal(self):
        self.assertEqual({p['label'] for p in feed(Judge(), ['normal'] * 20)}, {LABEL_NORMAL})

    def test_one_agreeing_frame_is_suspect_not_cheating(self):
        labels = [p['label'] for p in feed(Judge(), ['normal'] * 6 + ['suspect'] * 2 + ['normal'] * 6)]
        self.assertNotIn(LABEL_CHEAT, labels)
        self.assertEqual(labels[7], LABEL_SUSPECT)

    def test_one_noisy_frame_is_ignored_in_a_stream(self):
        labels = [p['label'] for p in feed(Judge(), ['normal'] * 6 + ['suspect'] + ['normal'] * 6)]
        self.assertEqual(set(labels), {LABEL_NORMAL})

    def test_single_image_mode_flags_one_frame(self):
        self.assertEqual(feed(Judge(min_frames=1), ['suspect'])[0]['label'], LABEL_SUSPECT)

    def test_default_needs_only_a_short_streak_for_cheating(self):
        labels = [p['label'] for p in feed(Judge(), ['suspect'] * 6)]  # 2.5 s
        self.assertEqual(labels[-1], LABEL_CHEAT)
        self.assertNotEqual(labels[2], LABEL_CHEAT)         # ~1 s: chưa đủ 1.5 s

    def test_sustained_multi_source_agreement_is_cheating(self):
        labels = [p['label'] for p in feed(Judge(), ['suspect'] * 12)]  # 6 giây
        self.assertEqual(labels[0], LABEL_NORMAL)          # một khung chưa đủ
        self.assertEqual(labels[1], LABEL_SUSPECT)         # 2 khung liên tiếp nhưng chưa đủ thời gian
        self.assertEqual(labels[-1], LABEL_CHEAT)

    def test_sustained_single_source_only_reaches_suspect(self):
        labels = [p['label'] for p in feed(Judge(), ['watch'] * 30)]  # điện thoại/quay đầu một mình
        self.assertNotIn(LABEL_CHEAT, labels)
        self.assertEqual(labels[-1], LABEL_SUSPECT)

    def test_one_second_glance_is_suspect_not_cheating(self):
        labels = [p['label'] for p in feed(Judge(), ['normal'] * 6 + ['watch'] * 4 + ['normal'] * 8)]   # ~2 s liếc nhẹ
        self.assertIn(LABEL_SUSPECT, labels)
        self.assertNotIn(LABEL_CHEAT, labels)
        self.assertEqual(labels[-1], LABEL_NORMAL)

    def test_single_weak_frame_is_normal(self):
        self.assertEqual(feed(Judge(), ['watch'])[0]['label'], LABEL_NORMAL)

    def test_brief_gap_does_not_break_the_streak(self):
        tiers = ['suspect'] * 4 + ['normal'] + ['suspect'] * 7
        self.assertEqual(feed(Judge(), tiers)[-1]['label'], LABEL_CHEAT)

    def test_cheating_label_is_held_briefly_then_clears(self):
        labels = [p['label'] for p in feed(Judge(), ['suspect'] * 12 + ['normal'] * 40)]
        self.assertEqual(labels[11], LABEL_CHEAT)
        self.assertEqual(labels[12], LABEL_CHEAT)           # giữ nhãn ngắn
        self.assertEqual(labels[-1], LABEL_NORMAL)

    def test_returns_to_normal_quickly_after_long_cheating(self):
        labels = [p['label'] for p in feed(Judge(), ['suspect'] * 40 + ['normal'] * 8)]  # 20 s gian lận rồi 4 s bình thường
        self.assertEqual(labels[39], LABEL_CHEAT)
        self.assertEqual(labels[-1], LABEL_NORMAL)
        self.assertEqual(labels[-5], LABEL_NORMAL)         # đã bình thường sau ~2 s (recover 1.5 s), không đợi cửa sổ 8 s

    def test_recover_seconds_is_adjustable(self):
        slow = Settings.from_dict({'recover_seconds': 6})
        labels = [p['label'] for p in feed(Judge(), ['suspect'] * 20 + ['normal'] * 8, slow)]
        self.assertNotEqual(labels[-1], LABEL_NORMAL)      # 4 s chưa đủ 6 s

    def test_thresholds_change_the_outcome(self):
        strict = Settings.from_dict({'cheat_score': 1.0, 'min_seconds': 30})
        self.assertNotIn(LABEL_CHEAT, [p['label'] for p in feed(Judge(), ['suspect'] * 12, strict)])
        eager = Settings.from_dict({'min_seconds': 0, 'cheat_score': 0.1})
        self.assertEqual(feed(Judge(), ['suspect'] * 2, eager)[-1]['label'], LABEL_CHEAT)

    def test_two_people_are_tracked_separately(self):
        j, cfg = Judge(), Settings()
        a, b = [0.05, 0.2, 0.3, 0.9], [0.6, 0.2, 0.9, 0.9]
        for i in range(12):
            out = j.apply([{'box': a, 'tier': 'suspect', 'score': 0.6}, {'box': b, 'tier': 'normal', 'score': 0.0}], i * 0.5, cfg)
        self.assertEqual([p['label'] for p in out], [LABEL_CHEAT, LABEL_NORMAL])
        self.assertNotEqual(out[0]['id'], out[1]['id'])

    def test_video_rewind_resets_history(self):
        j = Judge()
        feed(j, ['suspect'] * 12)
        self.assertNotEqual(feed(j, ['normal'], t0=0.0)[0]['label'], LABEL_CHEAT)


class SettingsTests(unittest.TestCase):
    def test_clamps_defaults_and_rejects_garbage(self):
        s = Settings.from_dict({'person_conf': 5, 'min_seconds': -3})
        self.assertEqual((s.person_conf, s.min_seconds), (0.9, 0.0))
        self.assertEqual(Settings.from_dict(None), Settings())
        with self.assertRaises(ValueError):
            Settings.from_dict({'phone_conf': 'abc'})

    def test_cheat_threshold_never_below_suspect_threshold(self):
        s = Settings.from_dict({'suspect_score': 0.6, 'cheat_score': 0.2})
        self.assertGreaterEqual(s.cheat_score, s.suspect_score)


class HeadTurnTests(unittest.TestCase):
    def kp(self, nose_x, eye_l=100.0, eye_r=140.0, conf=(0.9,) * 5):
        xy = [[nose_x, 50], [eye_l, 40], [eye_r, 40], [90, 45], [150, 45]]
        return xy, list(conf)

    def test_frontal_face_is_not_a_turn(self):
        self.assertEqual(ensemble.head_turn_evidence(*self.kp(120.0)), 0.0)

    def test_nose_near_one_eye_is_a_turn(self):
        self.assertGreater(ensemble.head_turn_evidence(*self.kp(98.0)), 0.7)

    def test_profile_with_one_eye_is_a_turn(self):
        self.assertGreaterEqual(ensemble.head_turn_evidence(*self.kp(110.0, conf=(0.9, 0.9, 0.1, 0.9, 0.1))), 0.8)

    def test_small_face_needs_a_clearer_turn(self):
        big = ensemble.head_turn_evidence(*self.kp(104.0))                 # mặt 40 px: lệch vừa phải đã có bằng chứng
        small = ensemble.head_turn_evidence(*self.kp(104.0), scale=0.2)    # cùng tư thế nhưng mặt chỉ 8 px: nhiễu che mất
        self.assertGreater(big, small)   # cùng tư thế: mặt nhỏ -> nhiễu lớn -> bằng chứng thấp hơn
        self.assertLess(small, 0.5)

    def test_no_nose_means_no_verdict(self):
        self.assertIsNone(ensemble.head_turn_evidence(*self.kp(120.0, conf=(0.1, 0.9, 0.9, 0.9, 0.9))))


class PairTests(unittest.TestCase):
    A, B, C = [0.10, 0.2, 0.25, 0.9], [0.30, 0.2, 0.45, 0.9], [0.80, 0.2, 0.95, 0.9]   # A,B cạnh nhau; C ở xa

    def persons(self, *boxes):
        return [{'box': b} for b in boxes]

    def test_two_neighbours_turning_toward_each_other(self):
        ev = ensemble.pair_evidence(self.persons(self.A, self.B), [(0.8, 1), (0.8, -1)])
        self.assertTrue(all(e and e >= 0.7 for e in ev))

    def test_turning_the_same_way_is_not_a_pair(self):
        self.assertEqual(ensemble.pair_evidence(self.persons(self.A, self.B), [(0.8, 1), (0.8, 1)]), [None, None])

    def test_turning_away_from_each_other_is_not_a_pair(self):
        self.assertEqual(ensemble.pair_evidence(self.persons(self.A, self.B), [(0.8, -1), (0.8, 1)]), [None, None])

    def test_far_apart_people_are_not_a_pair(self):
        self.assertEqual(ensemble.pair_evidence(self.persons(self.A, self.C), [(0.8, 1), (0.8, -1)]), [None, None])

    def test_both_must_turn_clearly(self):
        self.assertEqual(ensemble.pair_evidence(self.persons(self.A, self.B), [(0.8, 1), (0.1, -1)]), [None, None])

    def test_pair_makes_both_people_suspect_tier(self):
        anchor = [{'label': 'person', 'confidence': 0.9, 'box': self.A}, {'label': 'person', 'confidence': 0.9, 'box': self.B}]
        pose = [{'label': 'head_turn', 'confidence': 0.4, 'turn': 0.4, 'dir': 1, 'box': self.A}, {'label': 'head_turn', 'confidence': 0.4, 'turn': 0.4, 'dir': -1, 'box': self.B}]
        r = ensemble.fuse(anchor, {'pose': pose})
        self.assertEqual([p['tier'] for p in r['people']], ['suspect', 'suspect'])
        self.assertTrue(any('quay về phía nhau' in e for e in r['people'][0]['evidence']))

    def test_knn_plus_head_turn_is_one_evidence_family(self):
        anchor = [{'label': 'person', 'confidence': 0.9, 'box': self.A}]
        src = {'pose': [{'label': 'head_turn', 'confidence': 0.9, 'turn': 0.9, 'dir': 1, 'box': self.A}],
               'duyen': [{'label': 'Suspicious', 'confidence': None, 'box': self.A}]}
        self.assertEqual(ensemble.fuse(anchor, src)['people'][0]['tier'], 'watch')   # cùng keypoint: không phải 2 nguồn độc lập

    def test_head_turn_plus_phone_still_counts_as_two_families(self):
        anchor = [{'label': 'person', 'confidence': 0.9, 'box': self.A}, {'label': 'cell phone', 'confidence': 0.9, 'box': [0.12, 0.5, 0.17, 0.6]}]
        src = {'pose': [{'label': 'head_turn', 'confidence': 0.9, 'turn': 0.9, 'dir': 1, 'box': self.A}]}
        self.assertEqual(ensemble.fuse(anchor, src)['people'][0]['tier'], 'suspect')

    def test_single_person_turning_stays_weak(self):
        anchor = [{'label': 'person', 'confidence': 0.9, 'box': self.A}, {'label': 'person', 'confidence': 0.9, 'box': self.B}]
        pose = [{'label': 'head_turn', 'confidence': 0.9, 'turn': 0.9, 'dir': 1, 'box': self.A}, {'label': 'normal', 'confidence': 0.0, 'turn': 0.0, 'dir': 0, 'box': self.B}]
        self.assertEqual(ensemble.fuse(anchor, {'pose': pose})['people'][0]['tier'], 'watch')

    def test_direction_follows_nose_offset(self):
        xy = [[130, 50], [100, 40], [140, 40], [90, 45], [150, 45]]
        self.assertEqual(ensemble.head_turn_direction(xy, [0.9] * 5), 1)
        xy[0][0] = 110
        self.assertEqual(ensemble.head_turn_direction(xy, [0.9] * 5), -1)


class FuseWithSettingsTests(unittest.TestCase):
    def test_person_threshold_slider_filters_people(self):
        anchor = [{'label': 'person', 'confidence': 0.5, 'box': BOX}]
        self.assertEqual(len(ensemble.fuse(anchor, {}, Settings.from_dict({'person_conf': 0.4}))['people']), 1)
        self.assertEqual(len(ensemble.fuse(anchor, {}, Settings.from_dict({'person_conf': 0.6}))['people']), 0)

    def test_head_turn_plus_phone_is_suspect_tier(self):
        anchor = [{'label': 'person', 'confidence': 0.9, 'box': BOX}, {'label': 'cell phone', 'confidence': 0.9, 'box': [0.2, 0.5, 0.25, 0.6]}]
        r = ensemble.fuse(anchor, {'pose': [{'label': 'head_turn', 'confidence': 0.9, 'box': BOX}]})
        self.assertEqual(r['people'][0]['tier'], 'suspect')


if __name__ == '__main__':
    unittest.main()
