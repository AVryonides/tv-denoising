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