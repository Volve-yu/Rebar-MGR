import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_traditional_measurement import hough_centers, score, summarize


class TraditionalTests(unittest.TestCase):
    def test_straight_and_tilted_bars(self):
        for tilt in (0, 15):
            image = np.zeros((360, 640), np.uint8)
            for center in (100, 250, 400):
                cv2.fillConvexPoly(image, np.array([[center-8, 0], [center+8, 0],
                    [center+8+tilt, 359], [center-8+tilt, 359]]), 255)
            for mask in (False, True):
                result = hough_centers(image, dict(length=.35, low=50), mask)
                self.assertEqual(len(result), 3)
                np.testing.assert_allclose(result, np.array([100, 250, 400])+tilt/2, atol=3)

    def test_blank_and_horizontal(self):
        image = np.zeros((360, 640), np.uint8)
        self.assertEqual(hough_centers(image, dict(length=.35, low=50)), [])
        image[100:120, :] = 255
        self.assertEqual(hough_centers(image, dict(length=.35, low=50), True), [])

    def test_cached_candidates_match_uncached(self):
        image = np.zeros((360, 640), np.uint8)
        image[:, 90:110] = 255
        image[60:300, 240:260] = 255
        cache = {}
        for length in (.2, .35, .5):
            for low in (50, 100):
                cfg = dict(length=length, low=low)
                cv2.setRNGSeed(42)
                expected = hough_centers(image, cfg, True)
                cv2.setRNGSeed(42)
                self.assertEqual(hough_centers(image, cfg, True, cache), expected)

    def test_partial_occlusion_and_horizontal_crossing(self):
        image = np.zeros((360, 640), np.uint8)
        for center in (100, 250, 400):
            image[:, center-8:center+8] = 255
        image[150:190, :] = 0
        image[280:290, :] = 255
        result = hough_centers(image, dict(length=.2, low=50), True)
        self.assertEqual(len(result), 3)
        np.testing.assert_allclose(result, [100, 250, 400], atol=3)

    def test_missed_bar_does_not_create_false_interval(self):
        row = score('test', '1', [10, 110], [10, 60, 110], [50, 50], 1, 0)
        self.assertEqual(row['matched_intervals'], 0)
        self.assertIsNone(summarize([row])['spacing_mae_mm'])
        self.assertEqual(summarize([row])['interval_coverage'], 0)


if __name__ == '__main__':
    unittest.main()
