"""Checks for reconstruction metrics."""

import unittest

import numpy as np

from src.metrics import mse, psnr


class TestMetrics(unittest.TestCase):
    def test_known_error(self):
        reference = np.array([[0.0, 1.0]])
        estimate = np.array([[0.1, 0.9]])

        self.assertAlmostEqual(mse(reference, estimate), 0.01)
        self.assertAlmostEqual(psnr(reference, estimate), 20.0)

    def test_exact_reconstruction(self):
        image = np.array([[0.2, 0.8]])

        self.assertEqual(mse(image, image), 0.0)
        self.assertEqual(psnr(image, image), float("inf"))

    def test_invalid_images(self):
        for reference, estimate in [
            ([], []),
            ([[0.0]], [[0.0, 1.0]]),
            ([[np.nan]], [[0.0]]),
        ]:
            with self.subTest(reference=reference, estimate=estimate):
                with self.assertRaises(ValueError):
                    mse(reference, estimate)


if __name__ == "__main__":
    unittest.main()