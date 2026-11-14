"""Summarize final paired comparisons with bootstrap confidence intervals."""

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent

COMPARISONS = [
    ("chambolle", "accelerated"),
    ("projected", "accelerated"),
    ("chambolle", "projected"),
]


def bootstrap_mean_ci(values, rng, draws=10000):
    """Return a mean and percentile bootstrap 95% confidence interval."""
    values = np.asarray(values, dtype=float)
    indices = rng.integers(
        0, len(values), size=(draws, len(values))
    )
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def main():
    rng = np.random.default_rng(2026)
    summaries = []

    for folder in ["final", "stress_final"]:
        path = ROOT / "results" / folder / "cases.csv"

        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        if len(rows) != 180:
            raise ValueError(f"{path}: expected 180 rows.")

        indexed = {}
        for row in rows:
            key = (
                row["image_type"],
                float(row["sigma"]),
                int(row["seed"]),
                row["method"],
            )
            if key in indexed:
                raise ValueError(f"Duplicate result: {key}")
            indexed[key] = row

        conditions = sorted({
            (row["image_type"], float(row["sigma"]))
            for row in rows
        })

        for image_type, sigma in conditions:
            seeds = sorted({
                int(row["seed"])
                for row in rows
                if row["image_type"] == image_type
                and float(row["sigma"]) == sigma
            })

            if seeds != list(range(1000, 1020)):
                raise ValueError("Unexpected final-test seed set.")

            for reference, candidate in COMPARISONS:
                pairs = [
                    (
                        indexed[(image_type, sigma, seed, reference)],
                        indexed[(image_type, sigma, seed, candidate)],
                    )
                    for seed in seeds
                ]

                valid = [
                    (a, b) for a, b in pairs
                    if a["converged"] == "True"
                    and b["converged"] == "True"
                ]

                if not valid:
                    raise ValueError("No converged pairs to compare.")

                # Positive savings means the candidate is faster.
                savings_ms = [
                    1000.0 * (
                        float(a["runtime_median_s"])
                        - float(b["runtime_median_s"])
                    )
                    for a, b in valid
                ]

                speedups = [
                    float(a["runtime_median_s"])
                    / float(b["runtime_median_s"])
                    for a, b in valid
                ]

                # Negative difference means lower candidate MSE.
                mse_differences = [
                    float(b["mse"]) - float(a["mse"])
                    for a, b in valid
                ]

                mean_saving, saving_low, saving_high = (
                    bootstrap_mean_ci(savings_ms, rng)
                )
                mean_speedup, speedup_low, speedup_high = (
                    bootstrap_mean_ci(speedups, rng)
                )
                mean_mse_diff, mse_low, mse_high = (
                    bootstrap_mean_ci(mse_differences, rng)
                )

                summaries.append({
                    "image_type": image_type,
                    "sigma": sigma,
                    "reference": reference,
                    "candidate": candidate,
                    "paired_cases": len(pairs),
                    "converged_pairs": len(valid),
                    "mean_runtime_saving_ms": mean_saving,
                    "saving_ci95_low": saving_low,
                    "saving_ci95_high": saving_high,
                    "mean_paired_speedup": mean_speedup,
                    "speedup_ci95_low": speedup_low,
                    "speedup_ci95_high": speedup_high,
                    "mean_mse_difference": mean_mse_diff,
                    "mse_difference_ci95_low": mse_low,
                    "mse_difference_ci95_high": mse_high,
                })

                print(
                    f"{image_type:7s} sigma={sigma:.2f} "
                    f"{reference} -> {candidate}: "
                    f"saving={mean_saving:.2f} ms "
                    f"[{saving_low:.2f}, {saving_high:.2f}], "
                    f"speedup={mean_speedup:.2f}x "
                    f"[{speedup_low:.2f}, {speedup_high:.2f}]"
                )

    output_path = ROOT / "results" / "paired_summary.csv"

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summaries[0])
        )
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()