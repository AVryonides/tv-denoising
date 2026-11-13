"""Verification checks for Chambolle's baseline solver."""

import unittest

import numpy as np

from src.baseline import chambolle_denoise


class TestBaseline(unittest.TestCase):
    def test_one_iteration(self):
        result = chambolle_denoise(
            [[0.0, 1.0]], lam=0.5, iterations=1
        )
        np.testing.assert_allclose(result, [[0.1, 0.9]])

    def test_constant_image_is_preserved(self):
        for shape in [(4, 5), (1, 5), (4, 1), (1, 1)]:
            with self.subTest(shape=shape):
                noisy = np.full(shape, 0.7)
                result = chambolle_denoise(noisy)
                np.testing.assert_allclose(
                    result, noisy, rtol=0, atol=1e-12
                )

    def test_mean_shape_finiteness_and_input_preservation(self):
        rng = np.random.default_rng(42)
        noisy = rng.normal(size=(12, 15))
        original = noisy.copy()

        result = chambolle_denoise(noisy, lam=0.1, iterations=200)

        self.assertEqual(result.shape, noisy.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(
            float(result.mean()), float(noisy.mean()), places=12
        )
        np.testing.assert_array_equal(noisy, original)

    def test_two_pixel_solution_after_many_iterations(self):
        # For f = [0, 1] and lambda = 0.2, the exact minimizer
        # is [lambda, 1 - lambda] = [0.2, 0.8].
        result = chambolle_denoise(
            [[0.0, 1.0]], lam=0.2, iterations=1000
        )
        np.testing.assert_allclose(
            result, [[0.2, 0.8]], rtol=0, atol=1e-8
        )

    def test_integer_input(self):
        noisy = np.array([[0, 1]], dtype=int)

        result = chambolle_denoise(
            noisy, lam=0.5, iterations=1
        )

        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        np.testing.assert_allclose(result, [[0.1, 0.9]])

    def test_invalid_inputs(self):
        invalid_images = [
            [],
            [0.0, 1.0],
            np.zeros((2, 2, 2)),
            [[np.nan]],
            [[np.inf]],
        ]
        for noisy in invalid_images:
            with self.subTest(noisy=noisy):
                with self.assertRaises(ValueError):
                    chambolle_denoise(noisy)

        for lam in [0.0, -0.1, np.nan, np.inf]:
            with self.subTest(lam=lam):
                with self.assertRaises(ValueError):
                    chambolle_denoise([[0.0]], lam=lam)

        for iterations in [0, -1, 1.5, True]:
            with self.subTest(iterations=iterations):
                with self.assertRaises(ValueError):
                    chambolle_denoise(
                        [[0.0]], iterations=iterations
                    )


if __name__ == "__main__":
    unittest.main()