"""Verification of TV objectives and gap-based stopping."""

import unittest

import numpy as np

from src.baseline import chambolle_denoise
from src.diagnostics import (
    total_variation,
    primal_objective,
    dual_objective,
    primal_dual_gap,
)


class TestDiagnostics(unittest.TestCase):
    def test_known_optimum_has_zero_gap(self):
        noisy = np.array([[0.0, 1.0]])
        image = np.array([[0.2, 0.8]])
        p1 = np.zeros_like(noisy)
        p2 = np.array([[-1.0, 0.0]])

        self.assertAlmostEqual(total_variation(image), 0.6)
        self.assertAlmostEqual(
            primal_objective(image, noisy, 0.2), 0.16
        )
        self.assertAlmostEqual(
            dual_objective(noisy, p1, p2, 0.2), 0.16
        )
        self.assertAlmostEqual(
            primal_dual_gap(image, noisy, p1, p2, 0.2), 0.0
        )

    def test_feasible_field_gives_lower_bound(self):
        rng = np.random.default_rng(10)
        noisy = rng.normal(size=(5, 7))
        image = rng.normal(size=(5, 7))

        p1 = rng.normal(size=noisy.shape)
        p2 = rng.normal(size=noisy.shape)
        scale = np.maximum(1.0, np.sqrt(p1**2 + p2**2))
        p1 = p1 / scale
        p2 = p2 / scale

        gap = primal_dual_gap(image, noisy, p1, p2, lam=0.2)

        self.assertGreaterEqual(gap, -1e-10)

    def test_tolerance_stops_at_verified_accuracy(self):
        noisy = np.array([[0.0, 1.0]])

        image, info = chambolle_denoise(
            noisy,
            lam=0.2,
            iterations=2000,
            tolerance=1e-6,
            check_every=5,
            return_info=True,
        )

        self.assertTrue(info["converged"])
        self.assertLess(info["iterations"], 2000)
        self.assertLessEqual(info["normalized_gap"], 1e-6)
        self.assertGreaterEqual(info["normalized_gap"], -1e-10)

        field_norm = np.sqrt(info["p1"]**2 + info["p2"]**2)
        self.assertLessEqual(float(field_norm.max()), 1.0 + 1e-12)

        # Re-running for exactly the reported count must produce
        # the same image: diagnostics must not alter the updates.
        fixed = chambolle_denoise(
            noisy,
            lam=0.2,
            iterations=info["iterations"],
        )
        np.testing.assert_allclose(image, fixed, rtol=0, atol=1e-14)

    def test_iteration_cap_reports_failure_to_reach_tolerance(self):
        _, info = chambolle_denoise(
            [[0.0, 1.0]],
            lam=0.2,
            iterations=1,
            tolerance=1e-12,
            check_every=10,
            return_info=True,
        )

        self.assertFalse(info["converged"])
        self.assertEqual(info["iterations"], 1)
        self.assertGreater(info["normalized_gap"], 1e-12)

    def test_fixed_iteration_interface_is_preserved(self):
        noisy = np.array([[0.0, 1.0]])

        image = chambolle_denoise(noisy, iterations=7)
        image_with_info, info = chambolle_denoise(
            noisy, iterations=7, return_info=True
        )

        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(info["iterations"], 7)
        np.testing.assert_array_equal(image, image_with_info)


if __name__ == "__main__":
    unittest.main()