import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_screening_agreement import decision, image_decision, counts, classification, grouped_metrics
from inspection_data import matched_intervals


class ScreeningTests(unittest.TestCase):
    def test_signed_inclusive_bounds(self):
        rule = dict(design_spacing_mm=100, lower_tolerance_mm=-5, upper_tolerance_mm=10)
        for value in (95, 100, 110):
            self.assertEqual(decision(value, rule), 'PASS')
        for value in (94.9, 110.1):
            self.assertEqual(decision(value, rule), 'FAIL')
        self.assertEqual(image_decision([], [rule]), 'N/A')
        self.assertEqual(image_decision([100, 111], [rule]), 'FAIL')

    def test_missed_bar_does_not_shift_interval_labels(self):
        _, gaps = matched_intervals([0, 100, 200, 300], [0, 200, 300], 1, 20)
        self.assertEqual(gaps, [(2, 1)])

    def test_abstentions_and_confusion_orientation(self):
        states = [('FAIL', 'FAIL'), ('FAIL', 'PASS'), ('FAIL', 'N/A'),
                  ('PASS', 'PASS'), ('PASS', 'FAIL'), ('PASS', 'N/A')]
        rows = [dict(reference_status=a, prediction_status=b, sample_key='site-1/1') for a, b in states]
        m = classification(counts(rows))
        self.assertEqual(m['agreement'], 0.5)
        self.assertEqual(m['agreement_over_all'], 1 / 3)
        self.assertEqual(m['false_pass_rate'], 0.5)
        self.assertEqual(m['unavailable_n'], 2)
        boot = grouped_metrics(rows, ['site-1/1'], 20, 42)
        self.assertEqual(boot['image_cluster_bootstrap_95ci']['agreement']['bounds'], [0.5, 0.5])

    def test_all_fail_degeneracy(self):
        m = classification([10, 0, 0, 0, 0, 0])
        self.assertEqual(m['agreement'], 1)
        self.assertIsNone(m['balanced_accuracy'])
        self.assertIsNone(m['kappa'])


if __name__ == '__main__':
    unittest.main()
