"""Shared validation, stopping and info helpers for dual TV solvers."""

import numpy as np

from src.diagnostics import dual_objective, primal_objective
from src.operators import div


def prepare_noisy(noisy):
    """Copy noisy input to a finite non-empty 2D float array."""
    f = np.array(noisy, dtype=float, copy=True)

    if f.ndim != 2 or f.size == 0:
        raise ValueError("noisy must be a non-empty 2D image.")

    if not np.isfinite(f).all():
        raise ValueError("noisy must contain only finite values.")

    return f


def validate_solver_args(lam, iterations, tolerance, check_every):
    """Validate common dual-solver keyword arguments."""
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


def recover_image(f, lam, p1, p2):
    """Recover the primal image from the dual vector field."""
    return f - lam * div(p1, p2)


def normalized_gap(primal, dual):
    """Return the gap scaled by max(1, |primal|, |dual|)."""
    scale = max(1.0, abs(primal), abs(dual))
    return (primal - dual) / scale


def check_gap_or_raise(normalized):
    """Raise if a normalized gap is non-finite or significantly negative."""
    if not np.isfinite(normalized):
        raise FloatingPointError("Non-finite primal-dual gap.")

    if normalized < -1e-10:
        raise FloatingPointError(
            "Primal-dual gap is significantly negative."
        )


def should_stop_on_gap(f, p1, p2, lam, tolerance):
    """Evaluate the normalized gap stopping rule for a feasible field."""
    current_image = recover_image(f, lam, p1, p2)
    primal = primal_objective(current_image, f, lam)
    dual = dual_objective(f, p1, p2, lam)
    normalized = normalized_gap(primal, dual)
    check_gap_or_raise(normalized)
    return normalized <= tolerance


def build_info(f, p1, p2, lam, iterations, converged, **extra):
    """Build the standard return_info dictionary for a dual solver."""
    image = recover_image(f, lam, p1, p2)
    primal = primal_objective(image, f, lam)
    dual = dual_objective(f, p1, p2, lam)
    gap = primal - dual

    info = {
        "p1": p1,
        "p2": p2,
        "iterations": iterations,
        "primal": primal,
        "dual": dual,
        "gap": gap,
        "normalized_gap": normalized_gap(primal, dual),
        "converged": converged,
    }
    info.update(extra)
    return image, info
