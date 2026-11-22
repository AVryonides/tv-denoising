"""Projected dual methods for isotropic TV denoising.

Acceleration follows the fast gradient projection approach of
Beck and Teboulle (2009), adapted to our operator sign convention.
"""

import numpy as np

from src.operators import div, grad
from src.solver_utils import (
    build_info,
    prepare_noisy,
    recover_image,
    should_stop_on_gap,
    validate_solver_args,
)


def project_unit_disk(a1, a2):
    """Project each two-component pixel vector onto the unit disk.

    Parameters
    ----------
    a1, a2 : numpy.ndarray, shape (H, W)
        Floating-point vector-field components with matching shapes.

    Returns
    -------
    p1, p2 : numpy.ndarray, shape (H, W)
        Projected components whose per-pixel vector norms are <= 1.
    """
    magnitude = np.sqrt(a1**2 + a2**2)
    denominator = np.maximum(1.0, magnitude)

    return a1 / denominator, a2 / denominator


def projected_dual_denoise(
    noisy,
    lam=0.1,
    iterations=200,
    return_info=False,
    tolerance=None,
    check_every=10,
    accelerated=True,
):
    """Denoise using projected dual gradients with optional acceleration.

    Parameters
    ----------
    noisy : array-like, shape (H, W)
        Finite, non-empty grayscale image.
    lam : float
        Finite positive regularization strength.
    iterations : int
        Fixed iteration count, or maximum count if tolerance is supplied.
    return_info : bool
        Return (image, diagnostics) when True.
    tolerance : float or None
        Positive normalized primal-dual gap tolerance, or None
        for a fixed iteration count.
    check_every : int
        Evaluate the stopping criterion at this iteration interval
        and at the iteration cap.
    accelerated : bool
        Enable momentum. False gives the projected-gradient ablation.

    Returns
    -------
    image : numpy.ndarray, shape (H, W)
        Reconstruction. If return_info=True, returns (image, info).
        The info dictionary contains the feasible auxiliary field,
        objectives, gap, iteration count and stopping status.
    """
    f = prepare_noisy(noisy)
    validate_solver_args(lam, iterations, tolerance, check_every)

    if not isinstance(accelerated, (bool, np.bool_)):
        raise ValueError("accelerated must be a boolean.")

    tau = 1.0 / 8.0

    # p is feasible; y is the extrapolated field.
    p1 = np.zeros_like(f)
    p2 = np.zeros_like(f)
    y1 = p1.copy()
    y2 = p2.copy()

    t = 1.0
    converged = False
    completed_iterations = 0

    for step in range(1, iterations + 1):
        v = div(y1, y2) - f / lam
        g1, g2 = grad(v)

        a1 = y1 + tau * g1
        a2 = y2 + tau * g2
        p1_new, p2_new = project_unit_disk(a1, a2)

        if accelerated:
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            beta = (t - 1.0) / t_new
        else:
            t_new = 1.0
            beta = 0.0

        # Form momentum before replacing the old feasible field.
        y1 = p1_new + beta * (p1_new - p1)
        y2 = p2_new + beta * (p2_new - p2)

        p1, p2 = p1_new, p2_new
        t = t_new
        completed_iterations = step

        if tolerance is not None and (
            step % check_every == 0 or step == iterations
        ):
            # Diagnostics use feasible p, not extrapolated y.
            if should_stop_on_gap(f, p1, p2, lam, tolerance):
                converged = True
                break

    if return_info:
        return build_info(
            f,
            p1,
            p2,
            lam,
            completed_iterations,
            converged,
            accelerated=accelerated,
        )

    return recover_image(f, lam, p1, p2)
