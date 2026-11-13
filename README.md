# Total-Variation Image Denoising
## Aim

Implement and verify Chambolle's 2004 fixed-point algorithm for
isotropic total-variation denoising.

Investigate an accelerated dual solver as an alternative numerical
method for the same denoising objective.

## Planned comparison

Compare runtime and iterations at matched optimization accuracy.
Also report reconstruction MSE/PSNR and representative images.

Use synthetic grayscale images with additive Gaussian noise,
separate development, validation and test cases, and paired inputs.

## Status

Initial project setup. Algorithms and experiments are not implemented.

## References

- Chambolle, A. (2004). An Algorithm for Total Variation Minimization
  and Applications.
- Beck, A. and Teboulle, M. (2009). Fast Gradient-Based Algorithms
  for Constrained Total Variation Image Denoising and Deblurring Problems.

## Assistance

AI assistance is being used for explanations, project planning,
setup, verification and debugging. A detailed declaration will
be maintained as the project develops.

## Chambolle baseline

- Implemented the fixed-point iteration with tau = 1/8.
- Used the existing gradient and divergence boundary conventions.
- Verified a hand-computable first iteration and an exact two-pixel solution.
- Checked constant-image preservation, mean preservation, finite outputs,
  output shape, integer inputs, input preservation and invalid inputs.
- All nine operator and baseline tests passed.
- The solver currently uses a fixed iteration count; convergence diagnostics
  will be added before the final experimental comparison.

### Assistance

I drafted the baseline implementation. AI assistance identified errors and
provided a corrected version, explanations and verification tests.