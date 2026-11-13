"""Display and save a reproducible baseline denoising example."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.baseline import chambolle_denoise
from src.data import make_image


def main():
    seed = 0
    size = 96
    sigma = 0.1
    lam = 0.1
    iterations = 200

    clean, noisy = make_image(seed, size=size, sigma=sigma)

    denoised = chambolle_denoise(
        noisy,
        lam=lam,
        iterations=iterations,
    )

    noisy_mse = float(np.mean((noisy - clean)**2))
    denoised_mse = float(np.mean((denoised - clean)**2))

    print(f"Noisy MSE:    {noisy_mse:.6f}")
    print(f"Denoised MSE: {denoised_mse:.6f}")

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4), constrained_layout=True
    )

    panels = [
        (clean, "Clean"),
        (noisy, f"Noisy\nMSE = {noisy_mse:.5f}"),
        (denoised, f"Denoised\nMSE = {denoised_mse:.5f}"),
    ]

    for ax, (image, title) in zip(axes, panels):
        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_axis_off()

    fig.suptitle(
        f"Baseline demonstration | seed={seed}, size={size}, "
        f"sigma={sigma}, lambda={lam}, iterations={iterations}"
    )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "baseline_demo.png"
    fig.savefig(output_path, dpi=180)
    print(f"Figure saved to: {output_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()