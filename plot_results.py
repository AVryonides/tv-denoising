"""Plot paired speedups from saved experiment summaries."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def main():
    summary_path = ROOT / "results" / "paired_summary.csv"

    with summary_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4), sharey=True,
        constrained_layout=True,
    )

    comparisons = [
        ("chambolle", "Versus Chambolle", -0.003),
        ("projected", "Versus projected, no momentum", 0.003),
    ]

    for ax, image_type in zip(axes, ["shapes", "texture"]):
        for reference, label, offset in comparisons:
            selected = sorted(
                [
                    row for row in rows
                    if row["image_type"] == image_type
                    and row["reference"] == reference
                    and row["candidate"] == "accelerated"
                ],
                key=lambda row: float(row["sigma"]),
            )

            sigma = np.array([
                float(row["sigma"]) for row in selected
            ])
            mean = np.array([
                float(row["mean_paired_speedup"])
                for row in selected
            ])
            lower = np.array([
                float(row["speedup_ci95_low"])
                for row in selected
            ])
            upper = np.array([
                float(row["speedup_ci95_high"])
                for row in selected
            ])

            ax.errorbar(
                sigma + offset,
                mean,
                yerr=np.vstack((mean - lower, upper - mean)),
                marker="o",
                capsize=4,
                linestyle="none",
                label=label,
            )

        ax.axhline(
            1.0, color="gray", linestyle="--", linewidth=1
        )
        ax.set_title(
            "Shapes" if image_type == "shapes"
            else "Texture stress cases"
        )
        ax.set_xlabel("Noise standard deviation σ")
        ax.set_xticks([0.05, 0.10, 0.20])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Mean paired runtime speedup (×)")
    fig.suptitle(
        "Accelerated dual solver: speedup at matched gap tolerance\n"
        "20 paired cases per condition; percentile bootstrap 95% CIs"
    )

    output_dir = ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for extension in ["png", "pdf"]:
        output_path = output_dir / f"runtime_speedup.{extension}"
        fig.savefig(output_path, dpi=200)
        print(f"Saved: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()