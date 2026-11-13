"""Checks for the discrete gradient and divergence operators."""

import unittest

import numpy as np

from src.operators import div, grad


class TestOperators(unittest.TestCase):
    def test_hand_computable_example(self):
        u = np.array([
            [1.0, 3.0, 6.0],
            [2.0, 5.0, 9.0],
        ])
        original = u.copy()

        vertical, horizontal = grad(u)

        np.testing.assert_array_equal(
            vertical,
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
        )
        np.testing.assert_array_equal(
            horizontal,
            [[2.0, 3.0, 0.0], [3.0, 4.0, 0.0]],
        )
        np.testing.assert_array_equal(
            div(vertical, horizontal),
            [[3.0, 3.0, 0.0], [2.0, -1.0, -7.0]],
        )
        np.testing.assert_array_equal(u, original)

    def test_constant_image_has_zero_gradient(self):
        u = np.full((4, 5), 0.7)

        vertical, horizontal = grad(u)

        np.testing.assert_array_equal(vertical, np.zeros_like(u))
        np.testing.assert_array_equal(horizontal, np.zeros_like(u))

    def test_negative_adjoint_and_zero_sum(self):
        rng = np.random.default_rng(42)

        for shape in [(4, 5), (1, 5), (4, 1), (1, 1)]:
            with self.subTest(shape=shape):
                u = rng.normal(size=shape)
                vertical = rng.normal(size=shape)
                horizontal = rng.normal(size=shape)
                vertical_original = vertical.copy()
                horizontal_original = horizontal.copy()

                grad_vertical, grad_horizontal = grad(u)
                divergence = div(vertical, horizontal)

                left = np.sum(
                    grad_vertical * vertical
                    + grad_horizontal * horizontal
                )
                right = -np.sum(u * divergence)

                np.testing.assert_allclose(
                    left, right, rtol=1e-12, atol=1e-12
                )
                np.testing.assert_allclose(
                    divergence.sum(), 0.0, atol=1e-12
                )
                self.assertEqual(divergence.shape, shape)
                self.assertTrue(np.isfinite(divergence).all())

                np.testing.assert_array_equal(
                    vertical, vertical_original
                )
                np.testing.assert_array_equal(
                    horizontal, horizontal_original
                )


if __name__ == "__main__":
    unittest.main()