"""Algorithm tests using synthetic edges, never experimental paper results."""
import unittest
import cv2
import numpy as np
from geometry import fit_axis, unique_centers


class GeometryTests(unittest.TestCase):
    def test_two_sides_of_vertical_bar(self):
        image = np.zeros((240, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (80, 5), (100, 234), (255, 255, 255), -1)
        for method in ("ht", "sht"):
            x = fit_axis(image, [65, 0, 115, 240], method)
            self.assertIsNotNone(x)
            self.assertAlmostEqual(x, 90, delta=1.5)

    def test_tilted_bar(self):
        image = np.zeros((240, 200, 3), dtype=np.uint8)
        cv2.fillPoly(image, [np.array([[65, 0], [85, 0], [115, 239], [95, 239]])], (255, 255, 255))
        self.assertAlmostEqual(fit_axis(image, [50, 0, 130, 240]), 90, delta=2)

    def test_empty_is_failure(self):
        image = np.zeros((240, 200, 3), dtype=np.uint8)
        self.assertIsNone(fit_axis(image, [0, 0, 200, 240]))
        self.assertIsNone(fit_axis(image, [20, 20, 20, 30]))

    def test_center_merge(self):
        self.assertEqual(unique_centers([20, 1, 2, 40]), [1., 20., 40.])


if __name__ == "__main__":
    unittest.main()
