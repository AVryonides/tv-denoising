"""Shared figure-saving helpers."""

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "results" / "figures"


def save_figure(fig, name, output_dir=None, dpi=200):
    """Save a figure as PNG and PDF, then close it."""
    directory = Path(output_dir) if output_dir is not None else FIGURES
    directory.mkdir(parents=True, exist_ok=True)

    for extension in ["png", "pdf"]:
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=dpi)
        print(f"Saved: {path}")

    plt.close(fig)
