# Progress log 13/11/26

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

## Convergence diagnostics

- Added isotropic TV, primal and dual objectives, and the primal-dual gap.
- Added optional baseline stopping using:
  (primal - dual) / max(1, abs(primal), abs(dual)).
- Retained fixed-iteration behaviour when no tolerance is supplied.
- Check the gap every check_every iterations and at the iteration cap.
- A development image with seed=0, size=96, sigma=0.1 and lambda=0.1
  reached tolerance=0.001 after 820 iterations, checking every 10.
- This tolerance is provisional pending validation.
- AI assistance supplied diagnostic calculations, stopping-rule edits
  and verification tests.

## Accelerated dual solver

- Implemented projected dual gradients with optional FISTA-style momentum,
  following Beck and Teboulle (2009) with our operator sign convention.
- Retained accelerated=False as the no-momentum ablation.
- Both modes optimize the same objective as the Chambolle baseline.
- Verified projection, first iteration, known solution, constant images,
  boundary shapes, mean preservation, input preservation, feasibility,
  stopping tolerance and iteration-cap behaviour.
- All 25 tests passed.
- Development case: seed=0, size=96, sigma=0.1, lambda=0.1,
  normalized-gap tolerance=0.001, checked every 10 iterations.
- Iterations: Chambolle=820, projected=590, accelerated=100.
- These are development observations; runtime improvement and
  generalization have not yet been established.
- AI assistance supplied solver code, explanations and verification tests.

## Run the development smoke comparison

```bash
python run_experiments.py --config configs/smoke.json
```

Runs all three methods on two development seeds at three noise levels.
Writes 18 case-level rows to outputs/smoke/cases.csv and records the
configuration, software environment and Git state in metadata.json.
Rerunning overwrites these files.

Runtime includes solver initialization and convergence diagnostics.
Data generation, reconstruction metrics and file writing are excluded.
Each method receives one warm-up per case, followed by three timed runs.
The CSV retains individual timings and their median.

This smoke comparison uses provisional settings and is not the final
scientific evaluation.

## Experiment runner

- Added MSE and PSNR metrics and a configurable CSV-producing runner.
- Smoke run covered two development seeds and three noise levels.
- All 18 method-case combinations reached normalized-gap tolerance=0.001.
- Recorded three timings after one warm-up per method and case.
- Acceleration reduced runtime on all six development cases.
- Results remain preliminary; lambda has not yet been validation-selected.
- AI assistance supplied the runner, metrics and metric tests.