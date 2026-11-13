"""Chambolle fixed-point TV denoising.

Reference: Chambolle (2004), using the discretization and
sign conventions in the course mathematical guide.
"""

import numpy as np
from src.operators import div, grad
from src.diagnostics import primal_objective, dual_objective

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
    # Copy the input into a floating-point array.
    f = np.array(noisy, dtype=float, copy=True)

    if f.ndim != 2 or f.size == 0:
        raise ValueError("noisy must be a non-empty 2D image.")

    if not np.isfinite(f).all():
        raise ValueError("noisy must contain only finite values.")

    if not np.isscalar(lam) or not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be a finite positive scalar.")

    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (int, np.integer))
        or iterations <= 0
    ):
        raise ValueError("iterations must be a positive integer.")

    if tolerance is not None:
        if (
            not np.isscalar(tolerance)
            or not np.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise ValueError("tolerance must be finite and positive.")

    if (
        isinstance(check_every, (bool, np.bool_))
        or not isinstance(check_every, (int, np.integer))
        or check_every <= 0
    ):
        raise ValueError("check_every must be a positive integer.")
    # Initialize the step size and auxiliary vector field.
    tau = 1.0 / 8.0
    p1 = np.zeros_like(f)
    p2 = np.zeros_like(f)

    converged = False
    completed_iterations = 0
    for step in range(1, iterations + 1):        # Gradient of the current divergence minus scaled observations.
        v = div(p1, p2) - f / lam
        g1, g2 = grad(v)

        # Calculate one gradient magnitude and denominator per pixel.
        magnitude = np.sqrt(g1**2 + g2**2)
        denominator = 1.0 + tau * magnitude

        # Update both components using the same previous vector field.
        p1_new = (p1 + tau * g1) / denominator
        p2_new = (p2 + tau * g2) / denominator

        p1, p2 = p1_new, p2_new

        completed_iterations = step

        if tolerance is not None and (
            step % check_every == 0 or step == iterations
        ):
            current_image = f - lam * div(p1, p2)

            primal = primal_objective(current_image, f, lam)
            dual = dual_objective(f, p1, p2, lam)

            scale = max(1.0, abs(primal), abs(dual))
            normalized_gap = (primal - dual) / scale

            if not np.isfinite(normalized_gap):
                raise FloatingPointError("Non-finite primal-dual gap.")

            if normalized_gap < -1e-10:
                raise FloatingPointError(
                    "Primal-dual gap is significantly negative."
                )

            if normalized_gap <= tolerance:
                converged = True
                break

    # Recover the image from the final auxiliary vector field.
    u = f - lam * div(p1, p2)

    if return_info:
        primal = primal_objective(u, f, lam)
        dual = dual_objective(f, p1, p2, lam)
        gap = primal - dual
        scale = max(1.0, abs(primal), abs(dual))

        info = {
            "p1": p1,
            "p2": p2,
            "iterations": completed_iterations,
            "primal": primal,
            "dual": dual,
            "gap": gap,
            "normalized_gap": gap / scale,
            "converged": converged,
        }
        return u, info

    return u