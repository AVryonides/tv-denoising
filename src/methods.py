"""Shared registry of denoising methods used by experiment and plot scripts."""

from src.baseline import chambolle_denoise
from src.modified import projected_dual_denoise

METHODS = [
    ("chambolle", "Chambolle", chambolle_denoise, {}),
    (
        "projected",
        "Projected",
        projected_dual_denoise,
        {"accelerated": False},
    ),
    (
        "accelerated",
        "Accelerated",
        projected_dual_denoise,
        {"accelerated": True},
    ),
]
