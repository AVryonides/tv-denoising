"""Checks for the modified dual solver and its projection."""

import unittest

import numpy as np

from src.modified import project_unit_disk, projected_dual_denoise

class TestProjection(unittest.TestCase):
    def test_known_vectors(self):
        a1 = np.array([[3.0, 0.3, 0.0]])
        a2 = np.array([[4.0, 0.4, 0.0]])

        p1, p2 = project_unit_disk(a1, a2)

        np.testing.assert_allclose(p1, [[0.6, 0.3, 0.0]])
        np.testing.assert_allclose(p2, [[0.8, 0.4, 0.0]])

    def test_feasibility_idempotence_and_input_preservation(self):
        rng = np.random.default_rng(42)
        a1 = rng.normal(size=(5, 7))
        a2 = rng.normal(size=(5, 7))
        original1 = a1.copy()
        original2 = a2.copy()

        p1, p2 = project_unit_disk(a1, a2)

        norms = np.sqrt(p1**2 + p2**2)
        self.assertLessEqual(float(norms.max()), 1.0 + 1e-12)

        # Projecting an already projected field changes nothing.
        again1, again2 = project_unit_disk(p1, p2)
        np.testing.assert_allclose(again1, p1, atol=1e-12)
        np.testing.assert_allclose(again2, p2, atol=1e-12)

        np.testing.assert_array_equal(a1, original1)
        np.testing.assert_array_equal(a2, original2)

class TestModifiedSolver(unittest.TestCase):
    def test_first_iteration(self):
        for accelerated in [False, True]:
            with self.subTest(accelerated=accelerated):
                result = projected_dual_denoise(
                    [[0.0, 1.0]],
                    lam=0.5,
                    iterations=1,
                    accelerated=accelerated,
                )
                np.testing.assert_allclose(result, [[0.125, 0.875]])

    def test_known_solution(self):
        for accelerated in [False, True]:
            with self.subTest(accelerated=accelerated):
                result = projected_dual_denoise(
                    [[0.0, 1.0]],
                    lam=0.2,
                    iterations=1000,
                    accelerated=accelerated,
                )
                np.testing.assert_allclose(
                    result, [[0.2, 0.8]], rtol=0, atol=1e-8
                )

    def test_constant_images_and_boundary_shapes(self):
        for accelerated in [False, True]:
            for shape in [(4, 5), (1, 5), (4, 1), (1, 1)]:
                with self.subTest(
                    accelerated=accelerated, shape=shape
                ):
                    noisy = np.full(shape, 0.7)
                    result = projected_dual_denoise(
                        noisy, accelerated=accelerated
                    )
                    np.testing.assert_allclose(
                        result, noisy, rtol=0, atol=1e-12
                    )

    def test_output_properties_and_feasibility(self):
        rng = np.random.default_rng(42)
        noisy = rng.normal(size=(8, 11))
        original = noisy.copy()

        for accelerated in [False, True]:
            with self.subTest(accelerated=accelerated):
                result, info = projected_dual_denoise(
                    noisy,
                    iterations=100,
                    return_info=True,
                    accelerated=accelerated,
                )

                self.assertEqual(result.shape, noisy.shape)
                self.assertTrue(np.isfinite(result).all())
                self.assertAlmostEqual(
                    float(result.mean()),
                    float(noisy.mean()),
                    places=12,
                )

                norms = np.sqrt(info["p1"]**2 + info["p2"]**2)
                self.assertLessEqual(
                    float(norms.max()), 1.0 + 1e-12
                )
                self.assertGreaterEqual(
                    info["normalized_gap"], -1e-10
                )
                np.testing.assert_array_equal(noisy, original)

    def test_stopping_and_iteration_cap(self):
        for accelerated in [False, True]:
            with self.subTest(accelerated=accelerated):
                _, info = projected_dual_denoise(
                    [[0.0, 1.0]],
                    lam=0.2,
                    iterations=1000,
                    tolerance=1e-6,
                    check_every=5,
                    return_info=True,
                    accelerated=accelerated,
                )
                self.assertTrue(info["converged"])
                self.assertLessEqual(
                    info["normalized_gap"], 1e-6
                )

                _, capped = projected_dual_denoise(
                    [[0.0, 1.0]],
                    lam=0.2,
                    iterations=1,
                    tolerance=1e-12,
                    return_info=True,
                    accelerated=accelerated,
                )
                self.assertFalse(capped["converged"])
                self.assertEqual(capped["iterations"], 1)


if __name__ == "__main__":
    unittest.main()