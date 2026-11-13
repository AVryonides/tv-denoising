"""Synthetic image generation based on the course mathematical guide."""

import numpy as np


def make_image(seed, size=96, sigma=0.1):
    """Return a clean synthetic image and its noisy observation.

    Parameters
    ----------
    seed : int
        Seed for reproducible image geometry and Gaussian noise.
    size : int
        Image height and width in pixels; must be at least 32.
    sigma : float
        Finite, non-negative Gaussian noise standard deviation.

    Returns
    -------
    clean, noisy : numpy.ndarray, shape (size, size)
        Clean intensities lie in [0, 1]. Noisy values are not clipped.
    """
    if (
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size < 32
    ):
        raise ValueError("size must be an integer of at least 32.")

    if not np.isscalar(sigma) or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative.")

    rng = np.random.default_rng(seed)

    clean = np.full((size, size), 0.15)
    yy, xx = np.mgrid[:size, :size]

    for _ in range(4):
        row, col = rng.integers(0, size - 20, size=2)
        height, width = rng.integers(10, 22, size=2)

        clean[row:row + height, col:col + width] = rng.uniform(
            0.3, 0.9
        )

    for _ in range(2):
        cy, cx = rng.integers(15, size - 15, size=2)
        radius = rng.integers(6, 14)

        mask = (yy - cy)**2 + (xx - cx)**2 < radius**2
        clean[mask] = rng.uniform(0.3, 0.9)

    noisy = clean + sigma * rng.normal(size=clean.shape)

    return clean, noisy