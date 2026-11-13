"""Select a shared lambda using validation images only."""

import csv
import json
from pathlib import Path

import numpy as np

from src.data import make_image
from src.metrics import mse
from src.modified import projected_dual_denoise


ROOT = Path(__file__).resolve().parent


def main():
    settings = {
        "split": "validation",
        "seeds": list(range(100, 110)),
        "sigmas": [0.05, 0.10, 0.20],
        "size": 96,
        "candidates": [
            0.02, 0.04, 0.06, 0.08, 0.10,
            0.15, 0.175, 0.20, 0.225, 0.25, 0.30, 0.40,],
        "iterations": 20000,
        "tolerance": 1e-5,
        "check_every": 10,
    }

    output_dir = ROOT / "outputs" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "settings.json").open("w") as handle:
        json.dump(settings, handle, indent=2)
        handle.write("\n")

    scores = []

    with (output_dir / "lambda_cases.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lambda", "seed", "sigma", "mse",
                "iterations", "normalized_gap", "converged",
            ],
        )
        writer.writeheader()

        for lam in settings["candidates"]:
            errors = []
            all_converged = True

            for sigma in settings["sigmas"]:
                for seed in settings["seeds"]:
                    clean, noisy = make_image(
                        seed=seed,
                        size=settings["size"],
                        sigma=sigma,
                    )

                    image, info = projected_dual_denoise(
                        noisy,
                        lam=lam,
                        iterations=settings["iterations"],
                        tolerance=settings["tolerance"],
                        check_every=settings["check_every"],
                        accelerated=True,
                        return_info=True,
                    )

                    error = mse(clean, image)
                    errors.append(error)
                    all_converged &= bool(info["converged"])

                    writer.writerow({
                        "lambda": lam,
                        "seed": seed,
                        "sigma": sigma,
                        "mse": error,
                        "iterations": info["iterations"],
                        "normalized_gap": info["normalized_gap"],
                        "converged": info["converged"],
                    })

            mean_mse = float(np.mean(errors))
            scores.append({
                "lambda": lam,
                "mean_mse": mean_mse,
                "all_converged": all_converged,
            })
            handle.flush()

            print(
                f"lambda={lam:.3f} "
                f"mean_MSE={mean_mse:.8f} "
                f"all_converged={all_converged}",
                flush=True,
            )

    with (output_dir / "lambda_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["lambda", "mean_mse", "all_converged"],
        )
        writer.writeheader()
        writer.writerows(scores)

    # Do not silently select a parameter from incomplete solves.
    if not all(row["all_converged"] for row in scores):
        raise RuntimeError(
            "Some validation solves did not reach tolerance. "
            "Inspect the saved results before selecting lambda."
        )

    best = min(
        scores,
        key=lambda row: (row["mean_mse"], row["lambda"]),
    )

    with (output_dir / "selection.json").open("w") as handle:
        json.dump({
            "selected_lambda": best["lambda"],
            "validation_mean_mse": best["mean_mse"],
            "criterion": "Minimum equally weighted mean validation MSE",
            "settings": settings,
        }, handle, indent=2)
        handle.write("\n")

    print(f"\nSelected shared lambda: {best['lambda']}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()