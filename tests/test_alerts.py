import unittest
from tests import _paths  # noqa: F401
from live.alerts import AlertEngine, extract_signals


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def sig(model, *labels):
    return extract_signals(model, [{'label': l} for l in labels])


class AlertTests(unittest.TestCase):
    def setUp(self):
        self.c = Clock()
        self.e = AlertEngine(clock=self.c)

    def feed(self, s, seconds, step=1.0):
        out = []
        for _ in range(int(seconds / step)):
            out += self.e.update(s)
            self.c.t += step
        return out

    def test_no_person_needs_duration_and_frames(self):
        self.assertEqual(self.feed(sig('yolo_duyen'), 4), [])  # < 5s
        out = self.feed(sig('yolo_duyen'), 3)
        self.assertEqual([a['type'] for a in out], ['no_person'])
        self.assertEqual(out[0]['status'], 'needs_review')

    def test_single_frame_blip_is_ignored(self):
        self.e.update(sig('yolo_duyen'))
        self.c.t += 10
        self.assertEqual(self.e.update(sig('yolo_duyen', 'Normal')), [])

    def test_cooldown(self):
        first = self.feed(sig('yolo_duyen'), 8)
        again = self.feed(sig('yolo_duyen'), 20)  # still inside 30s cooldown
        self.assertEqual((len(first), len(again)), (1, 0))
        self.assertEqual(len(self.feed(sig('yolo_duyen'), 20)), 1)

    def test_multiple_people_and_flag(self):
        out = self.feed(sig('yolo_duyen', 'Normal', 'Cheating'), 4)
        self.assertEqual({a['type'] for a in out}, {'multiple_people', 'suspicious_label'})

    def test_phone_only_from_model_with_phone_class(self):
        self.assertEqual(self.feed(sig('yolo_duyen', 'Normal'), 5), [])
        out = self.feed(sig('gianlan', 'person', 'cell-phones'), 4)
        self.assertEqual([a['type'] for a in out], ['phone'])

    def test_gianlan_overlapping_labels_do_not_double_count(self):
        self.assertEqual(sig('gianlan', 'person', 'Normal')['person_count'], 1)

    def test_room_mode_does_not_flag_many_people_or_empty_frame(self):
        e = AlertEngine(clock=self.c, mode='room')
        out = []
        for _ in range(10):
            out += e.update(sig('person_coco', 'person', 'person', 'person'))
            self.c.t += 1
        for _ in range(10):
            out += e.update(sig('person_coco'))
            self.c.t += 1
        self.assertEqual(out, [])

    def test_room_mode_still_reports_phone_with_confidence(self):
        e = AlertEngine(clock=self.c, mode='room')
        s = extract_signals('person_coco', [{'label': 'cell phone', 'confidence': 0.71}, {'label': 'person', 'confidence': 0.9}])
        out = []
        for _ in range(3):
            out += e.update(s)
            self.c.t += 1
        self.assertEqual([(a['type'], a['confidence'], a['mode']) for a in out], [('phone', 0.71, 'room')])

    def test_brief_dropout_does_not_break_phone_streak(self):
        e = AlertEngine(clock=self.c)
        phone = sig('gianlan', 'person', 'cell-phones')
        out = e.update(phone); self.c.t += 0.5
        out += e.update(sig('gianlan', 'person')); self.c.t += 0.5   # nhấp nháy 1 khung
        out += e.update(phone); self.c.t += 0.6
        out += e.update(phone)
        self.assertEqual([a['type'] for a in out], ['phone'])

    def test_long_gap_restarts_streak(self):
        e = AlertEngine(clock=self.c)
        phone = sig('gianlan', 'person', 'cell-phones')
        e.update(phone); self.c.t += 5
        self.assertEqual(e.update(phone), [])  # khoảng trống > grace: chuỗi mới, chưa đủ khung

    def test_reset_streaks_clears_pause_gap(self):
        self.feed(sig('yolo_duyen'), 4)
        self.e.reset_streaks()
        self.c.t += 100
        self.assertEqual(self.e.update(sig('yolo_duyen')), [])


if __name__ == '__main__':
    unittest.main()
