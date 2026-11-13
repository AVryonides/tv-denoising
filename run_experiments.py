"""Run paired TV-denoising experiments and save case-level results."""

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

from src.baseline import chambolle_denoise
from src.data import make_image
from src.metrics import mse, psnr
from src.modified import projected_dual_denoise


ROOT = Path(__file__).resolve().parent

METHODS = [
    ("chambolle", chambolle_denoise, {}),
    ("projected", projected_dual_denoise, {"accelerated": False}),
    ("accelerated", projected_dual_denoise, {"accelerated": True}),
]


def git_output(*args):
    """Read Git metadata without modifying the repository."""
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with config_path.open() as handle:
        config = json.load(handle)

    repeats = config["repeats"]
    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats < 1
    ):
        raise ValueError("repeats must be a positive integer.")

    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "lam": config["lam"],
        "iterations": config["iterations"],
        "tolerance": config["tolerance"],
        "check_every": config["check_every"],
        "return_info": True,
    }

    status = git_output("status", "--porcelain")
    metadata = {
        "config": config,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timing": (
            "perf_counter; one full warm-up per method and case; "
            "median of repeated solver calls; data generation, "
            "quality metrics and file writing excluded"
        ),
    }

    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    fieldnames = [
        "split", "seed", "size", "sigma", "method", "lam",
        "tolerance", "check_every", "iteration_cap",
        "iterations", "converged", "normalized_gap",
        "noisy_mse", "mse", "psnr_db",
        "runtime_median_s", "runtime_min_s", "runtime_max_s",
        "runtime_samples_s",
    ]

    csv_path = output_dir / "cases.csv"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for sigma in config["sigmas"]:
            for seed in config["seeds"]:
                clean, noisy = make_image(
                    seed=seed,
                    size=config["size"],
                    sigma=sigma,
                )

                # Warm up every method on this exact case.
                for _, solver, options in METHODS:
                    solver(noisy, **settings, **options)

                timings = {name: [] for name, _, _ in METHODS}
                results = {}

                # Rotate method order to reduce systematic order effects.
                for repeat in range(repeats):
                    offset = repeat % len(METHODS)
                    order = METHODS[offset:] + METHODS[:offset]

                    for name, solver, options in order:
                        start = perf_counter()
                        image, info = solver(
                            noisy, **settings, **options
                        )
                        elapsed = perf_counter() - start

                        timings[name].append(elapsed)
                        results[name] = (image, info)

                for name, _, _ in METHODS:
                    image, info = results[name]
                    samples = timings[name]

                    if not np.isfinite(info["normalized_gap"]):
                        raise FloatingPointError(
                            f"Non-finite gap for {name}, seed={seed}."
                        )

                    row = {
                        "split": config["split"],
                        "seed": seed,
                        "size": config["size"],
                        "sigma": sigma,
                        "method": name,
                        "lam": config["lam"],
                        "tolerance": config["tolerance"],
                        "check_every": config["check_every"],
                        "iteration_cap": config["iterations"],
                        "iterations": info["iterations"],
                        "converged": info["converged"],
                        "normalized_gap": info["normalized_gap"],
                        "noisy_mse": mse(clean, noisy),
                        "mse": mse(clean, image),
                        "psnr_db": psnr(clean, image),
                        "runtime_median_s": float(np.median(samples)),
                        "runtime_min_s": min(samples),
                        "runtime_max_s": max(samples),
                        "runtime_samples_s": json.dumps(samples),
                    }
                    writer.writerow(row)

                    print(
                        f"sigma={sigma:.2f} seed={seed} "
                        f"{name:11s} "
                        f"iterations={info['iterations']:4d} "
                        f"time={row['runtime_median_s']:.4f}s "
                        f"converged={info['converged']}"
                    )

                handle.flush()

    print(f"\nResults saved to: {csv_path}")
    if metadata["git_dirty"]:
        print("Development run: repository has uncommitted changes.")


if __name__ == "__main__":
    main()