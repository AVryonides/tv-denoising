# Development review: initial results and corrections

### Baseline implementation

The first baseline draft had errors in array conversion, zero-array
initialization, loop indexing, gradient magnitude calculation, update
parentheses and assignment of the updated auxiliary field.

These were corrected by using a floating-point image copy, zeros_like,
a fixed-count loop, per-pixel gradient magnitudes, parenthesized numerators
and replacement of both auxiliary arrays after each update.

Verification progressed from a one-iteration example to an exact
two-pixel optimum and checks of boundary behaviour and mean preservation.

### Initial denoising demonstration

The first shapes demonstration used seed=0, size=96, sigma=0.1,
lambda=0.1 and 200 iterations.

Noisy MSE was 0.010024 and denoised MSE was 0.000454.
This established a working demonstration, not final evaluation evidence.

### Git branch integration

Missing-module errors occurred after switching to branches that did not
yet contain the baseline or synthetic-data commits.

The existing committed work was merged into the dependent branch.
The files were not recreated, and the issue did not require installing
an additional Python package.

### Diagnostic implementation

The initial diagnostics file contained NotImplementedError placeholders.
The functions were completed from the primal and dual formulas.

The known two-pixel optimum gave primal=dual=0.16 and approximately zero gap.
On a development image at lambda=0.1, the baseline reached normalized-gap
tolerance=0.001 after 820 iterations.

### Initial solver comparison

At seed=0, sigma=0.1, lambda=0.1 and tolerance=0.001:

- Chambolle required 820 iterations.
- Projected gradients required 590 iterations.
- Acceleration required 100 iterations.

These observations motivated the larger experiment but were not treated
as final performance estimates.

### Validation and iteration cap

The initial validation grid selected its largest candidate, lambda=0.20.
The grid was expanded using the same validation split, selecting an
interior candidate, lambda=0.175.

With that lambda, the original cap of 5000 iterations was insufficient
for two low-noise Chambolle development cases.

The shared cap was increased to 20000 without changing lambda or tolerance.
All subsequent smoke cases converged; the slowest shapes smoke case used
5700 iterations.

### Experiment-runner corrections

Generator selection was initially placed outside main(), before config
existed, and image_type was misspelled. The block was moved after
configuration loading and the name was corrected.

Incorrect indentation also placed CSV writing outside the per-method loop
and inside the texture-only condition. It was corrected so every method
writes one row for each case, while only regional MSE is conditional.

Both shapes and texture smoke runs then produced 18 method-case results
and all reached tolerance.

### Figure presentation

The quality curves initially overlapped so closely that the last plotted
solver covered the others. Different line styles and hollow markers made
all methods identifiable without changing the numerical values.

### Final interpretation

All 360 final method-case combinations converged using frozen settings.
Acceleration reduced measured runtime while retaining similar MSE.

The texture stress case showed that minimizing the TV objective can erase
genuine detail. In low-noise texture cases, denoising increased mean MSE
relative to the noisy observation.