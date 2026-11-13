"""Chambolle fixed-point TV denoising.

Reference: Chambolle (2004), using the discretization and
sign conventions in the course mathematical guide.
"""

import numpy as np
from src.operators import div, grad

def chambolle_denoise(noisy, lam=0.1, iterations=200):
    """Denoise a grayscale image using Chambolle's fixed-point method.

    Parameters
    ----------
    noisy : array-like, shape (H, W)
        Finite, two-dimensional noisy grayscale image.
    lam : float
        Positive, finite regularization strength.
    iterations : int
        Positive number of fixed-point iterations.

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

    # Initialize the step size and auxiliary vector field.
    tau = 1.0 / 8.0
    p1 = np.zeros_like(f)
    p2 = np.zeros_like(f)

    for _ in range(iterations):
        # Gradient of the current divergence minus scaled observations.
        v = div(p1, p2) - f / lam
        g1, g2 = grad(v)

        # Calculate one gradient magnitude and denominator per pixel.
        magnitude = np.sqrt(g1**2 + g2**2)
        denominator = 1.0 + tau * magnitude

        # Update both components using the same previous vector field.
        p1_new = (p1 + tau * g1) / denominator
        p2_new = (p2 + tau * g2) / denominator

        p1, p2 = p1_new, p2_new

    # Recover the image from the final auxiliary vector field.
    u = f - lam * div(p1, p2)
    return u