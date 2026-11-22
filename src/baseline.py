"""Chambolle fixed-point TV denoising.

Reference: Chambolle (2004), using the discretization and
sign conventions in the course mathematical guide.
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


def chambolle_denoise(
    noisy,
    lam=0.1,
    iterations=200,
    return_info=False,
    tolerance=None,
    check_every=10,
):
    """Denoise a grayscale image using Chambolle's fixed-point method.

    Parameters
    ----------
    noisy : array-like, shape (H, W)
        Finite, two-dimensional noisy grayscale image.
    lam : float
        Positive, finite regularization strength.
    iterations : int
        Positive number of fixed-point iterations.
    return_info : bool
        If True, return the image and a dictionary containing the
        final auxiliary field and iteration count.

    Returns
    -------
    numpy.ndarray, shape (H, W)
        Reconstructed floating-point image.
    """
    f = prepare_noisy(noisy)
    validate_solver_args(lam, iterations, tolerance, check_every)

    tau = 1.0 / 8.0
    p1 = np.zeros_like(f)
    p2 = np.zeros_like(f)

    converged = False
    completed_iterations = 0
    for step in range(1, iterations + 1):
        v = div(p1, p2) - f / lam
        g1, g2 = grad(v)

        magnitude = np.sqrt(g1**2 + g2**2)
        denominator = 1.0 + tau * magnitude

        p1 = (p1 + tau * g1) / denominator
        p2 = (p2 + tau * g2) / denominator

        completed_iterations = step

        if tolerance is not None and (
            step % check_every == 0 or step == iterations
        ):
            if should_stop_on_gap(f, p1, p2, lam, tolerance):
                converged = True
                break

    if return_info:
        return build_info(
            f, p1, p2, lam, completed_iterations, converged
        )

    return recover_image(f, lam, p1, p2)
