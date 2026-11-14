"""Reconstruction metrics for grayscale images."""

import numpy as np


def mse(reference, estimate):
    """Return mean squared error between finite, matching arrays."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    if reference.shape != estimate.shape or reference.size == 0:
        raise ValueError("Images must have matching, non-empty shapes.")

    if not (
        np.isfinite(reference).all()
        and np.isfinite(estimate).all()
    ):
        raise ValueError("Images must contain finite values.")

    return float(np.mean((reference - estimate)**2))


def psnr(reference, estimate, data_range=1.0):
    """Return PSNR in decibels; higher is better."""
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive.")

    error = mse(reference, estimate)

    if error == 0.0:
        return float("inf")

    return float(10.0 * np.log10(data_range**2 / error))