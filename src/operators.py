"""Discrete image operators following the project mathematical guide."""

import numpy as np


def grad(u):
    """Return vertical and horizontal forward differences.

    Parameters
    ----------
    u : numpy.ndarray, shape (H, W)
        Two-dimensional floating-point grayscale image.

    Returns
    -------
    vertical, horizontal : numpy.ndarray
        Arrays of shape (H, W). The final vertical row and
        final horizontal column are zero.
    """
    vertical = np.zeros_like(u)
    horizontal = np.zeros_like(u)

    vertical[:-1] = u[1:] - u[:-1]
    horizontal[:, :-1] = u[:, 1:] - u[:, :-1]

    return vertical, horizontal

def div(vertical, horizontal):
    """Return the divergence of a two-component vector field.

    Parameters
    ----------
    vertical, horizontal : numpy.ndarray, shape (H, W)
        Floating-point arrays with matching shapes.

    Returns
    -------
    numpy.ndarray, shape (H, W)
        Negative adjoint of the forward-difference gradient.
        The final vertical row and final horizontal column
        do not contribute.
    """
    out = np.zeros_like(vertical)

    out[:-1] += vertical[:-1]
    out[1:] -= vertical[:-1]

    out[:, :-1] += horizontal[:, :-1]
    out[:, 1:] -= horizontal[:, :-1]

    return out