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

## Iteration-cap development check

- With the validation-selected lambda=0.175, Chambolle failed to reach
  normalized-gap tolerance=0.001 within 5000 iterations on development
  seeds 0 and 1 at sigma=0.05.
- All other smoke cases reached tolerance.
- Increased the shared iteration cap to 20000 for a development recheck.
- Lambda and stopping tolerance remain unchanged.
- Capped runs are not treated as successful time-to-tolerance results.
- Rechecking with the shared cap of 20000 allowed all 18 smoke runs
  to reach tolerance.
- The largest observed count was 5700 iterations for Chambolle.
- Retained lambda=0.175, tolerance=0.001 and check_every=10.

## Final experiment protocol

- Both shapes and texture smoke runs completed: all 36 method-case
  combinations reached the fixed tolerance.
- Added a reproducible checkerboard stress generator and regional MSE.
- Final configurations use seeds 1000–1019 and three noise levels,
  with 20 paired cases per noise level and image type.
- Frozen settings: lambda=0.175, normalized-gap tolerance=0.001,
  check_every=10, iteration cap=20000, three timing repeats after warm-up.
- Primary outcome: paired runtime difference at matched gap tolerance.
- Secondary outcomes: convergence rate, iterations, MSE and PSNR;
  texture-region MSE is also reported for stress images.
- Report capped runs explicitly and exclude them from successful
  time-to-tolerance comparisons.
- Related conditions share seeds; uncertainty calculations must preserve
  that pairing and dependence.

  ## Final paired analysis

- Both final runs used clean commit
  e374205d3ba0f34b63ecd81081cfabc211cff5b3.
- All 360 method-case combinations reached tolerance.
- Calculated paired runtime savings, speedups and MSE differences.
- Used 10000 bootstrap resamples of 20 paired seed cases separately
  within each image-type/noise condition, with analysis seed 2026.
- Accelerated-versus-baseline mean paired speedups ranged from
  6.41x to 15.38x; all corresponding 95% intervals exceeded one.
- The no-momentum ablation was slower than acceleration in every
  condition on average.
- Texture loss remains a shared model limitation.
- AI assistance supplied the analysis and plotting scripts.