# Changelog

## 0.1.3

- Declare Python 3.8–3.15 compatibility in package metadata.
- Add automatic wheel and source builds, package checks, and installed-wheel
  tests on Python 3.8–3.15. The oldest Python job also tests the minimum
  supported PyTorch version, 2.2.1; newer jobs use compatible current versions.
- Publish tested distributions to PyPI through Trusted Publishing when a
  GitHub release is published and its tag matches the package version.

## 0.1.2

- Add external inputs through sampled tensors (`u`) or a callback (`u_fn`), with
  the right-hand side called as `f(t, y, u)` when inputs are provided.
- Preserve the existing `from FDEint import FDEint` import and the `f(t, y)`
  interface for equations without inputs.
- Correct the time and input associated with stored right-hand-side evaluations
  and support time grids with a nonzero initial time.
- Validate scalar fractional orders, strictly increasing time grids, and tensor
  shapes; support shared and batched observation grids.
- Support differentiation through initial states, the fractional order,
  sampled inputs, input callbacks, and right-hand-side parameters.
- Add an external-input example, API documentation, and a reproducible package
  build and release workflow.
- Correct the learned-order example's final solve to use its trained fractional
  order and document optional example dependencies.

## 0.1.1

- Previously published PyPI release of the predictor-corrector solver.
