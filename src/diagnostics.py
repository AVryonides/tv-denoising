"""Objective and convergence measurements for TV denoising."""

import numpy as np

from src.operators import div, grad


def total_variation(image):
    """Return the sum of isotropic gradient magnitudes."""
    g1, g2 = grad(image)
    magnitude = np.sqrt(g1**2 + g2**2)
    return float(np.sum(magnitude))


def primal_objective(image, noisy, lam):
    """Return the TV denoising objective."""
    data_term = 0.5 * np.sum((image - noisy)**2)
    regularization_term = lam * total_variation(image)
    return float(data_term + regularization_term)


def dual_objective(noisy, p1, p2, lam):
    """Return the dual objective; requires a feasible auxiliary field.

    The field must satisfy sqrt(p1**2 + p2**2) <= 1 at every pixel.
    Inputs are floating-point arrays with matching image shapes.
    """
    q = lam * div(p1, p2)
    return float(np.sum(noisy * q) - 0.5 * np.sum(q**2))


def primal_dual_gap(image, noisy, p1, p2, lam):
    """Return the primal objective minus the dual objective."""
    primal = primal_objective(image, noisy, lam)
    dual = dual_objective(noisy, p1, p2, lam)
    return primal - dual