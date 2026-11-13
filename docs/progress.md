# Progress log 13/11/26 - 10:06am

## Image operators

- Created the Python environment and recorded dependency versions.
- Implemented the course-provided forward gradient and divergence.
- Used zero forward differences at the image boundaries.
- Verified a hand-computable example, constant images, the negative-adjoint
  identity, zero-sum divergence, and single-row/column cases.
- Checked divergence output shapes, finite values and input preservation.
- Next: implement and verify Chambolle's baseline fixed-point iteration.

## Assistance

AI assisted with explanations, operator documentation and verification tests.
The gradient and divergence formulas/code originate from the course
mathematical guide.

## Synthetic image demonstration

- Added the synthetic rectangles-and-disks generator from the course guide.
- Added unmodified additive Gaussian noise without clipping.
- Created demo.py to display and save clean, noisy and denoised images.
- Development example: seed=0, size=96, sigma=0.1, lambda=0.1,
  iterations=200.
- Observed noisy MSE=0.010024 and denoised MSE=0.000454.
- This is a development example, not final-test evidence.
- Added checks for reproducibility, zero noise, matched clean images
  across noise levels, shapes, clean intensity range and finite outputs.
- AI assistance supplied the demonstration script, generator adaptation
  and verification tests.

  ## Run checks

```bash
python -m unittest discover -s tests -v
```

## Run the baseline demonstration

```bash
python demo.py
```

This displays clean, noisy and denoised synthetic images, prints their
reconstruction MSE, and saves outputs/baseline_demo.png.
Close the figure window to return to the terminal.