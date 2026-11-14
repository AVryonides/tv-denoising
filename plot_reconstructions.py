"""Save fixed-case reconstructions and aggregate quality comparisons."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.baseline import chambolle_denoise
from src.data import make_image, make_textured_image
from src.metrics import mse
from src.modified import projected_dual_denoise


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "results" / "figures"

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


def save_figure(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)

    for extension in ["png", "pdf"]:
        path = FIGURES / f"{name}.{extension}"
        fig.savefig(path, dpi=200)
        print(f"Saved: {path}")

    plt.close(fig)


def load_results(folder):
    directory = ROOT / "results" / folder

    with (directory / "cases.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    with (directory / "metadata.json").open() as handle:
        metadata = json.load(handle)

    return rows, metadata["config"]


def plot_examples():
    fig, axes = plt.subplots(
        2, 5, figsize=(13, 6), constrained_layout=True
    )

    for row_index, (folder, generator, label) in enumerate([
        ("final", make_image, "Shapes"),
        ("stress_final", make_textured_image, "Texture"),
    ]):
        rows, config = load_results(folder)
        seed = 1000
        sigma = 0.10

        clean, noisy = generator(
            seed=seed, size=config["size"], sigma=sigma
        )

        settings = {
            "lam": config["lam"],
            "iterations": config["iterations"],
            "tolerance": config["tolerance"],
            "check_every": config["check_every"],
            "return_info": True,
        }

        panels = [(clean, "Clean"), (noisy, "Noisy")]

        for name, title, solver, options in METHODS:
            image, info = solver(noisy, **settings, **options)

            if not info["converged"]:
                raise RuntimeError(f"{label}: {name} did not converge.")

            saved = next(
                r for r in rows
                if int(r["seed"]) == seed
                and float(r["sigma"]) == sigma
                and r["method"] == name
            )

            # Verify this regeneration agrees with the saved result.
            np.testing.assert_allclose(
                mse(clean, image),
                float(saved["mse"]),
                rtol=1e-8,
                atol=1e-12,
            )

            panels.append((image, title))

        for column, (image, title) in enumerate(panels):
            ax = axes[row_index, column]
            ax.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_title(
                f"{title}\nMSE={mse(clean, image):.5f}",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        axes[row_index, 0].set_ylabel(label, fontsize=12)

    fig.suptitle(
        "Fixed test example: seed=1000, σ=0.10, λ=0.175\n"
        "Shared display scale [0, 1]; reconstructed arrays are not clipped"
    )
    save_figure(fig, "reconstruction_examples")


def plot_quality():
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4),
        sharey=True,
        constrained_layout=True,
    )

    summaries = []

    for ax, folder, label in zip(
        axes,
        ["final", "stress_final"],
        ["Shapes", "Texture stress cases"],
    ):
        rows, _ = load_results(folder)
        sigmas = sorted({float(r["sigma"]) for r in rows})

        for method, title in [
            ("noisy", "Noisy input"),
            ("chambolle", "Chambolle"),
            ("projected", "Projected"),
            ("accelerated", "Accelerated"),
        ]:
            means = []

            for sigma in sigmas:
                # Noisy MSE is repeated across methods: count it once.
                selected = [
                    r for r in rows
                    if float(r["sigma"]) == sigma
                    and r["method"] == (
                        "chambolle" if method == "noisy" else method
                    )
                ]

                if len(selected) != 20:
                    raise ValueError("Expected 20 cases per condition.")

                key = "noisy_mse" if method == "noisy" else "mse"
                mean_mse = float(np.mean([
                    float(r[key]) for r in selected
                ]))
                means.append(mean_mse)

                summaries.append({
                    "image_type": selected[0]["image_type"],
                    "sigma": sigma,
                    "method": method,
                    "cases": len(selected),
                    "mean_mse": mean_mse,
                })

                styles = {
                "noisy": {
                    "color": "tab:blue",
                    "linestyle": "-",
                    "marker": "o",
                    "markersize": 6,
                },
                "chambolle": {
                    "color": "tab:orange",
                    "linestyle": "-",
                    "marker": "o",
                    "markersize": 11,
                    "markerfacecolor": "none",
                },
                "projected": {
                    "color": "tab:green",
                    "linestyle": "--",
                    "marker": "s",
                    "markersize": 8,
                    "markerfacecolor": "none",
                },
                "accelerated": {
                    "color": "tab:red",
                    "linestyle": ":",
                    "marker": "x",
                    "markersize": 6,
                },
            }

            ax.plot(
                sigmas,
                means,
                label=title,
                **styles[method],
            )

        ax.set_title(label)
        ax.set_xlabel("Noise standard deviation σ")
        ax.set_xticks(sigmas)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Mean reconstruction MSE — log scale")
    fig.suptitle(
        "Reconstruction quality across 20 test cases per condition\n"
        "Descriptive means; solver curves may overlap"
    )
    save_figure(fig, "reconstruction_quality")

    path = ROOT / "results" / "quality_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summaries[0])
        )
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_examples()
    plot_quality()