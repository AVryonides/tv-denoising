"""Checks for reproducible synthetic image generation."""

import unittest

import numpy as np

from src.data import make_image


class TestData(unittest.TestCase):
    def test_same_seed_reproduces_both_images(self):
        clean1, noisy1 = make_image(seed=0)
        clean2, noisy2 = make_image(seed=0)

        np.testing.assert_array_equal(clean1, clean2)
        np.testing.assert_array_equal(noisy1, noisy2)

    def test_zero_noise_returns_clean_image(self):
        clean, noisy = make_image(seed=1, sigma=0.0)

        np.testing.assert_array_equal(clean, noisy)

    def test_noise_level_does_not_change_clean_image(self):
        clean1, noisy1 = make_image(seed=2, sigma=0.05)
        clean2, noisy2 = make_image(seed=2, sigma=0.20)

        np.testing.assert_array_equal(clean1, clean2)

        # With the same seed, increasing sigma fourfold should
        # multiply the generated noise by four.
        np.testing.assert_allclose(
            noisy2 - clean2,
            4.0 * (noisy1 - clean1),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_shapes_ranges_and_finite_values(self):
        for size in [32, 96, 128]:
            with self.subTest(size=size):
                clean, noisy = make_image(seed=3, size=size)

                self.assertEqual(clean.shape, (size, size))
                self.assertEqual(noisy.shape, (size, size))
                self.assertTrue(np.isfinite(clean).all())
                self.assertTrue(np.isfinite(noisy).all())
                self.assertTrue((clean >= 0.0).all())
                self.assertTrue((clean <= 1.0).all())


if __name__ == "__main__":
    unittest.main()